# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-serve (HTTP) Harmony generator for gpt-oss-120b.

Drop-in replacement for `HarmonyVLLMGenerator` that talks to an external
`vllm serve` (OpenAI-compatible) endpoint instead of loading the model
in-process. It shares ALL rendering / agentic-loop logic with the in-process
backend via `HarmonyGeneratorBase`; the only difference is the token-generation
primitive, which renders the conversation to Harmony token ids and posts them
to ``/v1/completions`` (NOT ``/v1/chat/completions``).

Why raw ``/v1/completions`` with token ids: vLLM's ``GptOssReasoningParser``
rejects custom function tool calls on ``/v1/chat/completions``. Posting the
Harmony-rendered token ids and parsing the returned token ids back reproduces
the exact tool-call behaviour of the in-process path — browser/Wikipedia
search, python, and per-domain tools all flow through the rendered
``SystemContent`` tool namespaces, so no OpenAI ``tools=`` translation is needed.
"""
import asyncio
import re
from typing import List, Optional

import httpx
from openai_harmony import Message, Role, load_harmony_encoding, HarmonyEncodingName

from .harmony_base import HarmonyGeneratorBase

_CONTEXT_OVERFLOW_MARKERS = ("maximum context length", "reduce the length")


class HarmonyServeGenerator(HarmonyGeneratorBase):
    """Harmony generator backed by a remote vLLM serve endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        seed: Optional[int] = None,
        request_timeout: float = 300.0,
    ):
        super().__init__()  # starts the dedicated event-loop thread
        # Normalize: accept base_url with or without a trailing /v1.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._completions_url = root + "/v1/completions"
        self.model = model_name
        self._seed = seed
        self._request_timeout = request_timeout
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        # tiktoken is only needed as a fallback if the server omits token_ids.
        try:
            import tiktoken
            self._tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
        except Exception:
            self._tiktoken_enc = None

        print(f"[HarmonyServe] endpoint={self._completions_url} model={model_name}", flush=True)

    async def _post_completions(self, client: httpx.AsyncClient, payload: dict) -> httpx.Response:
        return await client.post(self._completions_url, json=payload, headers=self._headers)

    async def _generate_with_retry(
        self,
        tokens: List[int],
        stop_tokens: List[int],
        temperature: float = 1.0,
        max_tokens: int = 16384,
        max_retries: int = 3,
    ) -> List[Message]:
        """Render is done by the base; here we POST token ids and parse back."""
        if temperature <= 0:
            temperature = 0.01

        payload = {
            "model": self.model,
            "prompt": tokens,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "skip_special_tokens": False,
            "return_tokens_as_token_ids": True,
            "stop_token_ids": stop_tokens,
        }
        if self._seed is not None:
            payload["seed"] = self._seed

        last_exception = None
        timeout = httpx.Timeout(self._request_timeout, connect=30.0)

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await self._post_completions(client, payload)

                    # Context-overflow fallback: retry once with max_tokens set to
                    # the EXACT remaining space. Omitting max_tokens would make vLLM
                    # default to 16 tokens and silently truncate the final answer.
                    if resp.status_code == 400 and any(
                        m in resp.text.lower() for m in _CONTEXT_OVERFLOW_MARKERS
                    ):
                        m = re.search(r"maximum context length is (\d+)", resp.text)
                        remaining = (int(m.group(1)) - len(tokens)) if m else 0
                        if remaining > 0:
                            retry_payload = dict(payload, max_tokens=remaining)
                            resp = await self._post_completions(client, retry_payload)

                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"/v1/completions returned {resp.status_code}: {resp.text[:1000]}"
                        )
                    result = resp.json()

                choice = result["choices"][0]
                response_text = choice.get("text", "") or ""
                response_token_ids = choice.get("token_ids") or []

                # Fallback: re-encode text if the server didn't return token ids.
                if not response_token_ids and response_text and self._tiktoken_enc is not None:
                    response_token_ids = self._tiktoken_enc.encode(
                        response_text, allowed_special="all", disallowed_special=()
                    )

                if not response_token_ids:
                    print(f"  [HarmonyServe] WARNING: empty completion "
                          f"(attempt {attempt}/{max_retries})", flush=True)
                    continue

                messages = self.encoding.parse_messages_from_completion_tokens(
                    response_token_ids, Role.ASSISTANT, strict=False
                )
                print(f"  [HarmonyServe] {len(response_token_ids)} tokens → "
                      f"{len(messages)} message(s)", flush=True)
                return messages

            except Exception as e:
                last_exception = e
                print(f"  [HarmonyServe] error on attempt {attempt}/{max_retries}: {e}", flush=True)

        if last_exception:
            raise last_exception
        return []
