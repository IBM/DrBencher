"""
Shared Harmony-format generation logic for gpt-oss-120b.

`HarmonyGeneratorBase` owns everything that is IDENTICAL between the in-process
vLLM backend and the vLLM-serve (HTTP) backend:

  * a dedicated asyncio event-loop thread + sync wrappers,
  * `generate_response` (single-turn, returns the final-channel text),
  * `generate_agentic_response` (the tool-calling loop used by V2 verification),
  * shutdown / __del__.

Subclasses implement exactly ONE primitive:

    async def _generate_with_retry(self, tokens, stop_tokens, temperature,
                                   max_tokens) -> list[Message]

and set `self.encoding` (a HARMONY_GPT_OSS encoding). Everything above renders
the conversation to Harmony tokens and parses assistant messages back, so the
in-process and served paths produce identical tool-dispatch behaviour (browser
search, python, and per-domain tools all flow through the rendered
`SystemContent`, not through any OpenAI `tools=` translation).
"""
import asyncio
import datetime
import os
import signal
import sys
import threading
import warnings
from typing import Awaitable, Callable, Dict, List

# Silence the cosmetic ResourceWarning aiohttp emits for the gpt-oss browser
# tool's unclosed sessions (the loop exception handler below covers the printed
# variant). Narrowly scoped to this exact message.
warnings.filterwarnings("ignore", message="Unclosed client session")

# Agentic loops can span many slow tool calls (browser retries, API backoff);
# the per-loop sync wait must be generous. Override with HARMONY_AGENTIC_TIMEOUT.
_AGENTIC_SYNC_TIMEOUT = int(os.environ.get("HARMONY_AGENTIC_TIMEOUT", "3600"))

from openai_harmony import (
    Conversation,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
)


def _extract_text(msg) -> str:
    """Best-effort text extraction from a Harmony Message's content."""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if hasattr(content, "text"):
        return content.text
    if isinstance(content, str):
        return content
    return str(content)


class HarmonyGeneratorBase:
    """Backend-agnostic Harmony generator. Subclass must implement
    `_generate_with_retry` and set `self.encoding`."""

    def __init__(self):
        self._closed = False
        # Dedicated event loop in a background thread so the sync wrappers can
        # drive async generation from anywhere.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        # Subclasses MUST assign self.encoding before generation is called.
        self.encoding = None

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        # The gpt-oss browser tool creates aiohttp sessions on this loop and does
        # not always close them; their __del__ posts a cosmetic "Unclosed client
        # session" to the loop's exception handler at GC/shutdown. Swallow just
        # that message so it doesn't clutter the run output.
        self._loop.set_exception_handler(self._loop_exception_handler)
        self._loop.run_forever()

    @staticmethod
    def _loop_exception_handler(loop, context):
        if context.get("message") == "Unclosed client session":
            return
        loop.default_exception_handler(context)

    # ------------------------------------------------------------------
    # Primitive implemented by each backend
    # ------------------------------------------------------------------
    async def _generate_with_retry(
        self,
        tokens: List[int],
        stop_tokens: List[int],
        temperature: float = 1.0,
        max_tokens: int = 16384,
        max_retries: int = 3,
    ) -> List[Message]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Single-turn generation (V1 / question generation / evaluation)
    # ------------------------------------------------------------------
    async def generate_response(
        self,
        developer_content: str,
        user_content: str,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        reasoning_effort: str = "low",
    ) -> str:
        effort_map = {
            "high": ReasoningEffort.HIGH,
            "medium": ReasoningEffort.MEDIUM,
            "low": ReasoningEffort.LOW,
        }
        system_content = (
            SystemContent.new()
            .with_reasoning_effort(effort_map.get(reasoning_effort, ReasoningEffort.LOW))
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
        )
        messages = [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.DEVELOPER, developer_content),
            Message.from_role_and_content(Role.USER, user_content),
        ]
        conversation = Conversation.from_messages(messages)
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        stop_tokens = self.encoding.stop_tokens_for_assistant_actions()

        new_messages = await self._generate_with_retry(
            tokens, stop_tokens, temperature, max_tokens
        )

        # Only the final-channel assistant message is a valid response; analysis
        # / thinking content must never leak out.
        if new_messages:
            for msg in reversed(new_messages):
                if (
                    msg.author.role == Role.ASSISTANT
                    and getattr(msg, "channel", None) == "final"
                ):
                    return _extract_text(msg)
        return ""

    def generate_response_sync(
        self,
        developer_content: str,
        user_content: str,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        reasoning_effort: str = "low",
    ) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self.generate_response(
                developer_content, user_content, temperature, max_tokens, reasoning_effort
            ),
            self._loop,
        )
        # High reasoning_effort generations emit many tokens and legitimately run
        # long (minutes) — especially in eager mode — so the per-call cap must be
        # generous. Default 900s; override with HARMONY_GEN_TIMEOUT. (The old
        # hard 300s surfaced as an empty-message TimeoutError -> "Phase 1 failed".)
        return future.result(timeout=int(os.environ.get("HARMONY_GEN_TIMEOUT", "900")))

    # ------------------------------------------------------------------
    # Agentic tool-calling loop (V2 verification)
    # ------------------------------------------------------------------
    async def generate_agentic_response(
        self,
        messages: List[Message],
        tool_handler: Callable[[Message], Awaitable[List[Message]]],
        tool_prefix: "str | tuple[str, ...]" = "browser.",
        tool_configs=None,  # accepted for interface compat; tools come from SystemContent
        max_iterations: int = 200,
        max_consecutive_failures: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ) -> List[Message]:
        """Run the Harmony agentic loop: generate an assistant turn, dispatch a
        tool call when the assistant's ``recipient`` starts with *tool_prefix*,
        and stop when it emits a ``final``-channel message or safety limits hit.

        Identical for in-process and served backends: both render the same
        conversation to Harmony tokens and parse assistant messages back.
        """
        consecutive_failures = 0
        namespace_failures: Dict[str, int] = {}
        disabled_namespaces: set = set()
        iteration_count = 0

        while True:
            iteration_count += 1

            # --- safety: force a final answer when limits are hit ----------
            if iteration_count > max_iterations or consecutive_failures > max_consecutive_failures:
                messages.append(
                    Message.from_role_and_content(
                        Role.USER,
                        "You have reached the maximum number of reasoning iterations. "
                        "Please provide your final answer now.",
                    )
                )
                conversation = Conversation.from_messages(messages)
                tokens = self.encoding.render_conversation_for_completion(
                    conversation, Role.ASSISTANT
                )
                stop_tokens = self.encoding.stop_tokens_for_assistant_actions()
                try:
                    final_messages = await self._generate_with_retry(
                        tokens, stop_tokens, temperature, max_tokens
                    )
                    for msg in final_messages:
                        if msg is not None and hasattr(msg, "content"):
                            messages.append(msg)
                    last = messages[-1]
                    if last.author.role == Role.ASSISTANT:
                        last.channel = "final"
                except Exception as e:
                    print(f"[Agentic] Error generating final message: {e}", flush=True)
                    fallback = Message.from_role_and_content(
                        Role.ASSISTANT,
                        "Maximum iterations reached. Unable to generate final answer.",
                    )
                    fallback.channel = "final"
                    messages.append(fallback)
                break

            last_message = messages[-1]
            recipient = getattr(last_message, "recipient", None)

            # --- tool call dispatch ----------------------------------------
            if recipient and str(recipient).startswith(tool_prefix):
                recipient_str = str(recipient)
                tool_ns = recipient_str.split(".")[0] if "." in recipient_str else recipient_str

                if tool_ns in disabled_namespaces:
                    messages.append(
                        Message.from_role_and_content(
                            Role.SYSTEM,
                            f"Tool '{tool_ns}' disabled due to {max_consecutive_failures} "
                            f"consecutive failures. Use a different tool.",
                        )
                    )
                    continue
                if consecutive_failures >= max_consecutive_failures * 3:
                    messages.append(
                        Message.from_role_and_content(
                            Role.SYSTEM,
                            f"Too many tool failures ({consecutive_failures}). "
                            f"Please provide your final answer now.",
                        )
                    )
                    continue

                try:
                    result_msgs = await tool_handler(last_message)
                    if result_msgs and any(
                        "error" in str(getattr(m, "content", "")).lower() for m in result_msgs
                    ):
                        consecutive_failures += 1
                        namespace_failures[tool_ns] = namespace_failures.get(tool_ns, 0) + 1
                        if namespace_failures[tool_ns] >= max_consecutive_failures:
                            disabled_namespaces.add(tool_ns)
                    else:
                        consecutive_failures = max(0, consecutive_failures - 1)
                        namespace_failures[tool_ns] = 0
                    messages.extend(result_msgs)
                except Exception as e:
                    consecutive_failures += 1
                    namespace_failures[tool_ns] = namespace_failures.get(tool_ns, 0) + 1
                    if namespace_failures[tool_ns] >= max_consecutive_failures:
                        disabled_namespaces.add(tool_ns)
                    print(f"[Agentic] Tool call exception ns={tool_ns}: {e}", flush=True)
                    messages.append(
                        Message.from_role_and_content(Role.SYSTEM, f"Tool call failed: {e}")
                    )
                continue

            # --- check for final channel -----------------------------------
            if (
                last_message.author.role == Role.ASSISTANT
                and getattr(last_message, "channel", None) == "final"
            ):
                break

            # --- generate next assistant turn ------------------------------
            conversation = Conversation.from_messages(messages)
            tokens = self.encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT
            )
            stop_tokens = self.encoding.stop_tokens_for_assistant_actions()
            new_messages = await self._generate_with_retry(
                tokens, stop_tokens, temperature, max_tokens
            )
            if new_messages:
                messages.extend(new_messages)
                last = messages[-1]
                if (
                    last.author.role == Role.ASSISTANT
                    and getattr(last, "channel", None) == "final"
                ):
                    break
            else:
                print("[Agentic] No new messages generated — breaking.", flush=True)
                break

        return messages

    def generate_agentic_response_sync(
        self,
        messages: List[Message],
        tool_handler: Callable[[Message], Awaitable[List[Message]]],
        tool_prefix: "str | tuple[str, ...]" = "browser.",
        tool_configs=None,
        max_iterations: int = 200,
        max_consecutive_failures: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ) -> List[Message]:
        future = asyncio.run_coroutine_threadsafe(
            self.generate_agentic_response(
                messages,
                tool_handler,
                tool_prefix=tool_prefix,
                tool_configs=tool_configs,
                max_iterations=max_iterations,
                max_consecutive_failures=max_consecutive_failures,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            self._loop,
        )
        return future.result(timeout=_AGENTIC_SYNC_TIMEOUT)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _shutdown_backend(self) -> None:
        """Hook for subclasses to release backend resources (engine/client)."""

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            self._shutdown_backend()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        finally:
            self._closed = True

    def __del__(self):
        try:
            if not getattr(self, "_closed", True):
                self.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level generator registry + convenience wrapper
#
# Canonical home for the harmony-generator singleton and the
# ``gen_from_prompt_harmony`` helper, shared by every domain.  These were
# previously duplicated — with SEPARATE module globals — in
# math_harmony_drbencher.py and wikidata_harmony.py, which forced callers to
# register the generator twice (once per registry).  They now live here (a leaf
# module imported by both backends) and are re-exported from those modules for
# backwards compatibility, so there is a single shared generator.
# ---------------------------------------------------------------------------

_harmony_generator = None


def get_harmony_generator():
    """Return the process-wide harmony generator, or None if unset."""
    return _harmony_generator


def set_harmony_generator(generator):
    """Register the process-wide harmony generator (in-process or served)."""
    global _harmony_generator
    _harmony_generator = generator


def _iter_descendant_pids(root_pid: int) -> List[int]:
    """PIDs of all descendants of ``root_pid`` (best-effort, Linux /proc).

    Returns [] on platforms without /proc (e.g. macOS) — the bench only runs
    on the Linux cluster, so this is a no-op elsewhere.
    """
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []
    children: Dict[int, List[int]] = {}
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                data = fh.read()
            # Format: "<pid> (<comm>) <state> <ppid> ...". comm may contain
            # spaces/parens, so parse the fields after the final ')'.
            ppid = int(data[data.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)
    out: List[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def _kill_descendants(sig: int) -> None:
    """Send ``sig`` to every descendant of this process (not the group)."""
    for pid in _iter_descendant_pids(os.getpid()):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def shutdown_and_exit(exit_code: int = 0, grace_seconds: int = 60) -> None:
    """Tear down the harmony generator, kill its workers, and exit this process.

    The in-process vLLM backend runs the model in tensor-parallel worker
    *subprocesses*. Calling ``os._exit()`` alone kills only the main process
    and orphans those workers; they then linger in CUDA/NCCL teardown and keep
    the batch job (e.g. LSF) in RUN state until it is killed by hand
    (``TERM_OWNER``). So on completion we:

      1. shut the generator down (``AsyncLLMEngine.shutdown()`` reaps the
         workers and releases the GPUs), then
      2. SIGKILL any worker subprocess that survived, then exit.

    We target only *descendants* of this process — never the process group —
    because the full-pipeline launcher invokes this module once per category in
    a single shell process group (a killpg would take out the parent loop and
    skip the merge step). A SIGALRM watchdog guarantees we still exit if
    graceful shutdown hangs. Output files are already written and closed by the
    bench; we only flush the console streams. This function does not return.
    """
    def _force_exit(signum=None, frame=None):
        _kill_descendants(signal.SIGKILL)
        os._exit(exit_code)

    # Arm the watchdog before touching the engine — shutdown is the step most
    # likely to hang. (signal/alarm only work on the main thread.)
    try:
        signal.signal(signal.SIGALRM, _force_exit)
        signal.alarm(grace_seconds)
    except (ValueError, OSError):
        pass

    try:
        generator = get_harmony_generator()
        if generator is not None:
            generator.shutdown()
    except Exception:
        pass

    # Belt-and-suspenders: SIGKILL any worker the engine shutdown missed, so
    # nothing orphaned keeps the job alive. Descendants only — the parent shell
    # survives so a multi-category pipeline loop continues to the next step.
    _kill_descendants(signal.SIGKILL)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def gen_from_prompt_harmony(prompt, temperature=0.7, max_tokens=1024,
                            developer_content="You are a helpful AI assistant.",
                            reasoning_effort="low"):
    """Generate text via the registered harmony generator.

    Drop-in replacement for ``gen_from_prompt`` when running in Harmony mode.
    ``reasoning_effort`` ("low" | "medium" | "high") is threaded to the
    generator so callers can raise it for the steps that need real reasoning
    (fact derivation, question composition).
    """
    generator = get_harmony_generator()
    if generator is None:
        raise RuntimeError(
            "Harmony generator not initialized but gen_from_prompt_harmony was called"
        )
    if isinstance(prompt, list):
        prompt = prompt[0]
    return generator.generate_response_sync(
        developer_content=developer_content,
        user_content=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
