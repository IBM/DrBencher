"""
In-process Harmony-format vLLM generator for gpt-oss-120b.

Loads the model directly via vLLM's AsyncLLMEngine. Shares all Harmony
rendering / agentic-loop logic with the served backend through
`HarmonyGeneratorBase`; this class only implements the token-generation
primitive (`_generate_with_retry`) by streaming from the in-process engine.
"""
import asyncio
import os
import uuid
from typing import List, Optional

# NOTE: vLLM (and its torch/CUDA stack) is imported lazily inside the methods
# below, NOT at module top. This lets the serve client import
# HarmonyVLLMGenerator without loading torch — only in-process instantiation
# pulls vLLM in. (A login node may kill any torch-importing process.)
from openai_harmony import (
    Message, Role, StreamableParser, load_harmony_encoding, HarmonyEncodingName
)

from .harmony_base import HarmonyGeneratorBase


class HarmonyVLLMGenerator(HarmonyGeneratorBase):
    """vLLM in-process generator using OpenAI Harmony message format."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = None,
        enable_prefix_caching: bool = True,
    ):
        super().__init__()  # starts the dedicated event-loop thread
        self.model_path = model_path

        from vllm import AsyncEngineArgs  # lazy: pulls torch/CUDA only in-process

        # enforce_eager skips CUDA-graph capture + torch.compile (~10 min for
        # gpt-oss-120b on 4 GPUs). Off by default (full compilation → faster
        # inference); set HARMONY_ENFORCE_EAGER=1 for fast startup (smoke tests).
        enforce_eager = os.environ.get("HARMONY_ENFORCE_EAGER", "0") == "1"

        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )

        print(f"[HarmonyVLLM] Loading model: {model_path}", flush=True)
        print(f"  - Tensor parallel size: {tensor_parallel_size}", flush=True)
        print(f"  - GPU memory utilization: {gpu_memory_utilization}", flush=True)
        print(f"  - Enforce eager: {enforce_eager}", flush=True)

        # Load timeout must exceed full engine init (compilation + CUDA-graph
        # capture) — measured ~915s for gpt-oss-120b/4-GPU in non-eager mode, so
        # the old hard-coded 600s killed the main thread mid-load. Default 1800s;
        # override with HARMONY_LOAD_TIMEOUT.
        load_timeout = int(os.environ.get("HARMONY_LOAD_TIMEOUT", "1800"))
        future = asyncio.run_coroutine_threadsafe(
            self._init_engine(engine_args), self._loop
        )
        self.engine = future.result(timeout=load_timeout)

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        print(f"[HarmonyVLLM] Model loaded successfully", flush=True)

    async def _init_engine(self, engine_args):
        from vllm import AsyncLLMEngine
        return AsyncLLMEngine.from_engine_args(engine_args)

    async def _generate_tokens(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ):
        """Stream generated token ids from the in-process vLLM engine."""
        from vllm import SamplingParams, TokensPrompt

        if temperature <= 0:
            temperature = 0.01

        sp = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=stop_tokens,
            temperature=temperature,
        )
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
        rid = uuid.uuid4().hex

        seen = 0
        async for req_out in self.engine.generate(
            prompt=prompt,
            sampling_params=sp,
            request_id=rid,
        ):
            out = req_out.outputs[0]
            token_ids = out.token_ids
            for tid in token_ids[seen:]:
                yield tid
            seen = len(token_ids)
            if getattr(req_out, "finished", False):
                break

    async def _generate_with_retry(
        self,
        tokens: List[int],
        stop_tokens: List[int],
        temperature: float = 1.0,
        max_tokens: int = 16384,
        max_retries: int = 3,
    ) -> List[Message]:
        """Generate and incrementally parse the stream into Harmony Messages."""
        last_exception = None

        for attempt in range(1, max_retries + 1):
            parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
            parse_error = None
            draining = False
            token_count = 0

            stream = self._generate_tokens(tokens, stop_tokens, temperature, max_tokens)
            try:
                async for token_id in stream:
                    token_count += 1
                    if not draining:
                        try:
                            parser.process(token_id)
                        except Exception as pe:
                            parse_error = pe
                            draining = True

                if parse_error is not None:
                    last_exception = parse_error
                    print(f"  [HarmonyVLLM] Parse error on attempt {attempt}/{max_retries} "
                          f"(generated {token_count} tokens): {parse_error}", flush=True)
                    continue

                msg_count = len(parser.messages)
                if msg_count == 0:
                    print(f"  [HarmonyVLLM] WARNING: Generated {token_count} tokens but "
                          f"parser produced 0 messages (attempt {attempt}/{max_retries})", flush=True)
                else:
                    print(f"  [HarmonyVLLM] Generated {token_count} tokens → "
                          f"{msg_count} message(s)", flush=True)

                return parser.messages

            except Exception as e:
                last_exception = e
                print(f"  [HarmonyVLLM] Generation error on attempt {attempt}/{max_retries}: {e}", flush=True)
            finally:
                try:
                    await stream.aclose()
                except Exception:
                    pass

        if last_exception:
            raise last_exception
        return []

    # -- lifecycle ------------------------------------------------------
    def _shutdown_backend(self) -> None:
        if getattr(self, "engine", None) is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._shutdown_engine(), self._loop
                )
                future.result(timeout=30)
            except Exception:
                pass
            self.engine = None

    async def _shutdown_engine(self):
        if self.engine is not None and hasattr(self.engine, "shutdown"):
            self.engine.shutdown()
