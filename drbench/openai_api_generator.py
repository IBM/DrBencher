# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI-API-compatible generator for vLLM serve (or any OpenAI-compatible endpoint).

Drop-in replacement for HarmonyVLLMGenerator — same interface, but talks to an
external server via HTTP instead of loading the model in-process.

Usage:
    gen = OpenAIAPIGenerator(
        base_url="http://localhost:8000/v1",
        model_name="openai/gpt-oss-120b",
    )
    set_harmony_generator(gen)
"""

import asyncio
import json
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
import openai
from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)


class OpenAIAPIGenerator:
    """
    Generator that uses the OpenAI chat completions API.

    Compatible with vllm serve, TGI, or any OpenAI-compatible endpoint.
    Implements the same public interface as HarmonyVLLMGenerator so that
    all existing benchmark code works without modification.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
    ):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(300.0, connect=30.0),  # 5 min total, 30s connect
        )
        self.model = model_name

        # Domain benchmarks call:
        #   asyncio.run_coroutine_threadsafe(backend.close(), generator._loop)
        # We need a running event loop to satisfy that contract.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        print(f"[OpenAIAPI] Initialized: base_url={base_url}, model={model_name}", flush=True)

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ------------------------------------------------------------------
    # generate_response_sync  (V1 / question generation / evaluation)
    # ------------------------------------------------------------------

    def generate_response_sync(
        self,
        developer_content: str,
        user_content: str,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        reasoning_effort: str = "low",
        _max_retries: int = 2,
    ) -> str:
        """Non-agentic single-turn generation."""
        messages = [
            {"role": "system", "content": developer_content},
            {"role": "user", "content": user_content},
        ]
        for attempt in range(_max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=max(temperature, 0.01),
                    max_tokens=max_tokens,
                    timeout=300.0,
                )
                return resp.choices[0].message.content or ""
            except (httpx.TimeoutException, openai.APITimeoutError) as e:
                print(f"[OpenAIAPI] Timeout (attempt {attempt+1}/{_max_retries+1}): {e}",
                      flush=True)
                if attempt == _max_retries:
                    return ""
            except Exception as e:
                print(f"[OpenAIAPI] generate_response_sync error: {e}", flush=True)
                return ""
        return ""

    # ------------------------------------------------------------------
    # generate_agentic_response_sync  (V2 agentic verification)
    # ------------------------------------------------------------------

    def generate_agentic_response_sync(
        self,
        messages: List[Message],
        tool_handler: Callable[[Message], Awaitable[List[Message]]],
        tool_prefix: "str | tuple[str, ...]" = "browser.",
        tool_configs: Optional[List[ToolNamespaceConfig]] = None,
        max_iterations: int = 200,
        max_consecutive_failures: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ) -> List[Message]:
        """
        Agentic tool-calling loop via the OpenAI chat completions API.

        Args:
            messages: Initial Harmony Message list (system, developer, user).
            tool_handler: Async callable that takes a Harmony tool-call Message
                and returns a list of result Messages.
            tool_prefix: Recipient prefix(es) for tool dispatch (same as Harmony path).
            tool_configs: List of ToolNamespaceConfig — required for OpenAI API path.
                The Harmony path reads tool schemas from the SystemContent; this path
                needs them passed explicitly.
            max_iterations: Hard cap on loop iterations.
            max_consecutive_failures: Failures before force-terminate.
            temperature: Sampling temperature.
            max_tokens: Max tokens per generation call.

        Returns:
            Full message list (Harmony Message format) including all assistant
            and tool messages.
        """
        if tool_configs is None:
            raise ValueError(
                "tool_configs is required for OpenAIAPIGenerator. "
                "Pass tool_configs=[browser_tool.tool_config, python_tool.tool_config, ...]"
            )

        # Convert initial Harmony messages → OpenAI format
        oai_messages = _harmony_msgs_to_openai(messages)
        # Convert tool configs → OpenAI tools list
        oai_tools = _tool_configs_to_openai(tool_configs)

        consecutive_failures = 0
        namespace_failures: Dict[str, int] = {}
        disabled_namespaces: set = set()
        iteration_count = 0

        print(f"[OpenAIAPI Agentic] Starting loop: tool_prefix={tool_prefix}, "
              f"max_iter={max_iterations}, {len(oai_tools)} tools", flush=True)

        while True:
            iteration_count += 1

            # --- safety: force final answer when limits hit ---
            if iteration_count > max_iterations or consecutive_failures > max_consecutive_failures:
                oai_messages.append({
                    "role": "user",
                    "content": "You have reached the maximum number of reasoning iterations. "
                               "Please provide your final answer now.",
                })
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=oai_messages,
                        temperature=max(temperature, 0.01),
                        max_tokens=max_tokens,
                    timeout=300.0,
                    )
                    final_text = resp.choices[0].message.content or ""
                except Exception as e:
                    print(f"[OpenAIAPI Agentic] Error generating final: {e}", flush=True)
                    final_text = "Maximum iterations reached. Unable to generate final answer."

                final_msg = _make_harmony_assistant_message(final_text, channel="final")
                messages.append(final_msg)
                break

            # --- generate next assistant turn ---
            print(f"[OpenAIAPI Agentic] Iteration {iteration_count}/{max_iterations}", flush=True)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=oai_messages,
                    tools=oai_tools if oai_tools else openai.NOT_GIVEN,
                    temperature=max(temperature, 0.01),
                    max_tokens=max_tokens,
                    timeout=300.0,
                )
            except Exception as e:
                print(f"[OpenAIAPI Agentic] API error: {e}", flush=True)
                consecutive_failures += 1
                continue

            choice = resp.choices[0]
            assistant_msg = choice.message

            # --- Collect structured tool calls from the vLLM parser ---
            structured_tool_calls = []  # list of (id, name, arguments_json)

            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    structured_tool_calls.append(
                        (tc.id, tc.function.name, tc.function.arguments)
                    )

            # --- no tool calls: final answer ---
            if not structured_tool_calls:
                content_text = assistant_msg.content or ""
                print(f"[OpenAIAPI Agentic] Final answer at iteration {iteration_count} "
                      f"({len(content_text)} chars)", flush=True)

                # Append to OpenAI history
                oai_messages.append({"role": "assistant", "content": content_text})

                # Append Harmony final message
                final_msg = _make_harmony_assistant_message(content_text, channel="final")
                messages.append(final_msg)
                break

            # --- tool calls present: dispatch each one ---
            # Build the assistant message for OpenAI history (with tool_calls)
            oai_assistant = {"role": "assistant", "content": assistant_msg.content or ""}
            oai_assistant["tool_calls"] = [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": tc_args,
                    },
                }
                for tc_id, tc_name, tc_args in structured_tool_calls
            ]
            oai_messages.append(oai_assistant)

            # Also add assistant content to Harmony message list if present
            if assistant_msg.content:
                messages.append(_make_harmony_assistant_message(assistant_msg.content))

            for tool_call_id, func_name, func_args in structured_tool_calls:

                # Convert OpenAI function name to Harmony recipient
                recipient = _openai_tool_name_to_recipient(func_name)

                # Check tool_prefix match
                if isinstance(tool_prefix, tuple):
                    matches = any(recipient.startswith(p) for p in tool_prefix)
                else:
                    matches = recipient.startswith(tool_prefix)

                if not matches:
                    print(f"[OpenAIAPI Agentic] Recipient '{recipient}' doesn't match "
                          f"prefix '{tool_prefix}', skipping", flush=True)
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": f"Unknown tool: {recipient}"}),
                    })
                    continue

                # Check namespace disabling
                tool_ns = recipient.split(".")[0] if "." in recipient else recipient
                if tool_ns in disabled_namespaces:
                    err_text = f"Tool '{tool_ns}' disabled due to repeated failures."
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": err_text}),
                    })
                    continue

                # Create Harmony tool-call message and dispatch
                harmony_tc_msg = _make_harmony_tool_call_message(recipient, func_args)
                messages.append(harmony_tc_msg)

                try:
                    # tool_handler is async — run in our event loop
                    future = asyncio.run_coroutine_threadsafe(
                        tool_handler(harmony_tc_msg), self._loop
                    )
                    result_msgs = future.result(timeout=120)

                    print(f"[OpenAIAPI Agentic] Tool '{recipient}' returned "
                          f"{len(result_msgs)} messages", flush=True)

                    # Check for errors
                    has_error = any(
                        "error" in str(getattr(m, "content", "")).lower()
                        for m in result_msgs
                    )
                    if has_error:
                        consecutive_failures += 1
                        namespace_failures[tool_ns] = namespace_failures.get(tool_ns, 0) + 1
                        if namespace_failures[tool_ns] >= max_consecutive_failures:
                            disabled_namespaces.add(tool_ns)
                    else:
                        consecutive_failures = max(0, consecutive_failures - 1)
                        namespace_failures[tool_ns] = 0

                    # Collect tool result text
                    result_text = _extract_text_from_messages(result_msgs)

                    # Append to Harmony message list
                    messages.extend(result_msgs)

                except Exception as e:
                    consecutive_failures += 1
                    namespace_failures[tool_ns] = namespace_failures.get(tool_ns, 0) + 1
                    if namespace_failures[tool_ns] >= max_consecutive_failures:
                        disabled_namespaces.add(tool_ns)
                    print(f"[OpenAIAPI Agentic] Tool exception '{recipient}': {e}", flush=True)
                    result_text = json.dumps({"error": str(e)})
                    err_msg = Message.from_role_and_content(
                        Role.SYSTEM, f"Tool call failed: {e}"
                    )
                    messages.append(err_msg)

                # Append tool result to OpenAI history
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text,
                })

        print(f"[OpenAIAPI Agentic] Loop ended: {iteration_count} iterations, "
              f"{len(messages)} total messages", flush=True)
        return messages

    # ------------------------------------------------------------------
    # Compatibility stubs
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """No-op for API-based generator."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ======================================================================
# Helper functions
# ======================================================================

def _harmony_msgs_to_openai(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert a list of Harmony Messages to OpenAI chat format."""
    oai = []
    for msg in messages:
        role = msg.author.role if hasattr(msg, "author") else None

        # Extract text content
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, str):
                    parts.append(item)
            text = "".join(parts)
        elif hasattr(content, "text"):
            text = content.text
        elif isinstance(content, str):
            text = content
        else:
            # SystemContent or complex objects — convert to string
            text = str(content)

        if role == Role.SYSTEM:
            # Skip system messages — they contain Harmony metadata (tool configs, etc.)
            # The developer message carries the actual system prompt.
            continue
        elif role == Role.DEVELOPER:
            oai.append({"role": "system", "content": text})
        elif role == Role.USER:
            oai.append({"role": "user", "content": text})
        elif role == Role.ASSISTANT:
            oai.append({"role": "assistant", "content": text})
        elif role == Role.TOOL:
            # Tool results from initial context (shouldn't normally appear)
            oai.append({"role": "user", "content": f"[Tool result]: {text}"})
        else:
            # Unknown role — include as user message
            oai.append({"role": "user", "content": text})

    return oai


def _tool_configs_to_openai(
    tool_configs: List[ToolNamespaceConfig],
) -> List[Dict[str, Any]]:
    """
    Convert Harmony ToolNamespaceConfig list to OpenAI tools format.

    Each ToolDescription within a namespace becomes a separate function.
    Names are formatted as "namespace__funcname" (double underscore) to
    comply with OpenAI's regex: ^[a-zA-Z0-9_-]{1,64}$
    """
    oai_tools = []
    for ns_config in tool_configs:
        ns_name = ns_config.name  # e.g. "bio", "browser", "python"
        for tool_desc in ns_config.tools:
            func_name = f"{ns_name}__{tool_desc.name}"  # e.g. "bio__search_protein"
            # Ensure name complies with OpenAI's constraints
            func_name = func_name.replace(".", "_").replace("-", "_")
            if len(func_name) > 64:
                func_name = func_name[:64]

            oai_tools.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": tool_desc.description or "",
                    "parameters": tool_desc.parameters or {"type": "object", "properties": {}},
                },
            })
    return oai_tools


def _openai_tool_name_to_recipient(name: str) -> str:
    """
    Convert OpenAI function name back to Harmony recipient.

    "bio__search_protein" → "bio.search_protein"
    "browser__search"     → "browser.search"
    "python__execute"     → "python.execute"

    Uses first double-underscore as separator; remaining __ are kept as-is.
    """
    parts = name.split("__", 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return name


def _make_harmony_tool_call_message(recipient: str, arguments_json: str) -> Message:
    """Create a Harmony Message that looks like an assistant tool call."""
    msg = Message.from_role_and_content(Role.ASSISTANT, arguments_json)
    msg.recipient = recipient
    return msg


def _make_harmony_assistant_message(
    content_text: str, channel: Optional[str] = None
) -> Message:
    """Create a Harmony Message for an assistant response."""
    msg = Message.from_role_and_content(Role.ASSISTANT, content_text)
    if channel:
        msg.channel = channel
    return msg


def _extract_text_from_messages(messages: List[Message]) -> str:
    """Extract text content from a list of Harmony Messages."""
    parts = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            for item in content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, str):
                    parts.append(item)
        elif hasattr(content, "text"):
            parts.append(content.text)
        elif isinstance(content, str):
            parts.append(content)
        else:
            parts.append(str(content))
    return "\n".join(parts)
