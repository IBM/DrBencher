#!/usr/bin/env python3
"""Direct V2-agentic smoke test for the vLLM-serve Harmony backend.

Exercises the exact code path that the biochem SMOKE run never reached (no QA
survived Phase 3): the agentic tool-calling loop over /v1/completions, with the
real biochem tools (browser/Wikipedia search, python sandbox, bio API). Prints
every tool call the model makes and the final answer.

Run IN the GPU job (imports torch via the tools):
    python scripts/benchmark/_agentic_serve_test.py --url http://localhost:8000/v1 \
        --model openai/gpt-oss-120b
"""
import argparse
import asyncio
import datetime
import sys
from pathlib import Path

# Run as a plain script (python scripts/benchmark/_agentic_serve_test.py): put the
# repo root on sys.path so `drbench` / `tools` import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openai_harmony import Message, Role, SystemContent, ReasoningEffort

from drbench.harmony_serve import HarmonyServeGenerator
from tools.kg_browser import MultiSourceKnowledgeBackend, MultiSourceKnowledgeBrowserTool
from tools.hybrid_exec_qid import HybridPythonTool
from tools.bio_tool import BiochemTool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--question", default=(
        "Use the available tools to determine the molecular formula of dopamine "
        "(look it up in PubChem or Wikipedia). Finish with a line 'Exact Answer: <formula>'."
    ))
    ap.add_argument("--max_iterations", type=int, default=20)
    args = ap.parse_args()

    gen = HarmonyServeGenerator(base_url=args.url, model_name=args.model)

    backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
    browser_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
    python_tool = HybridPythonTool(timeout=60)
    python_tool.set_qid("agentic_serve_test")
    bio_tool = BiochemTool()

    system_content = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
        .with_tools(browser_tool.tool_config)
        .with_tools(python_tool.tool_config)
        .with_tools(bio_tool.tool_config)
    )
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(
            Role.DEVELOPER,
            "You are a biochemistry research assistant. Use the tools to look up "
            "facts before answering. End with a line 'Exact Answer: <answer>'.",
        ),
        Message.from_role_and_content(Role.USER, f"Question: {args.question}"),
    ]

    tool_calls = []

    async def _tool_handler(msg, _browser=browser_tool, _python=python_tool, _bio=bio_tool):
        recipient = str(getattr(msg, "recipient", ""))
        tool_calls.append(recipient)
        print(f"  >>> TOOL CALL: {recipient}", flush=True)
        results = []
        if recipient.startswith("bio"):
            async for m in _bio.process(msg):
                results.append(m)
        elif recipient.startswith("browser."):
            async for m in _browser.process(msg):
                results.append(m)
        elif recipient.startswith("python"):
            async for m in _python.process(msg):
                results.append(m)
        else:
            results.append(Message.from_role_and_content(Role.SYSTEM, f"Unknown tool: {recipient}"))
        return results

    print(f"[test] Running agentic loop against {args.url} ...", flush=True)
    result_messages = gen.generate_agentic_response_sync(
        messages,
        _tool_handler,
        tool_prefix=("bio", "browser.", "python"),
        tool_configs=[browser_tool.tool_config, python_tool.tool_config, bio_tool.tool_config],
        max_iterations=args.max_iterations,
        temperature=1.0,
    )

    # Extract the final-channel answer.
    final_text = ""
    for m in reversed(result_messages):
        if m.author.role == Role.ASSISTANT and getattr(m, "channel", None) == "final":
            c = m.content
            final_text = c[0].text if isinstance(c, list) and hasattr(c[0], "text") else str(c)
            break

    print("\n" + "=" * 60, flush=True)
    print(f"[test] tool calls made ({len(tool_calls)}): {tool_calls}", flush=True)
    print(f"[test] total messages: {len(result_messages)}", flush=True)
    print(f"[test] FINAL ANSWER:\n{final_text}", flush=True)
    print("=" * 60, flush=True)
    if not tool_calls:
        print("[test] WARNING: no tool calls dispatched — the agentic serve path did NOT "
              "exercise tools (model answered directly).", flush=True)
    gen.shutdown()


if __name__ == "__main__":
    main()
