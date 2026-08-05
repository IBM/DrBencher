"""Benchmark answerability verification (V1 closed-book, V2 agentic).

Extracted from the former math_harmony_drbencher.py — only the live verification
path (verify_bench_v1/v2 + their math-answer-checking helpers) is retained here.
The generator/exp_mode machinery that surrounded them was dead and was removed.
"""

import datetime
import copy
import re
import os, json, tqdm
import numpy as np
from openai_harmony import Message, SystemContent, Role, ReasoningEffort
from .util import gen_from_prompt
from fractions import Fraction
try:
    from tools.hybrid_exec_qid import HybridPythonTool
    PYTHON_TOOL_AVAILABLE = True
except ImportError:
    PYTHON_TOOL_AVAILABLE = False
from .harmony_base import get_harmony_generator, gen_from_prompt_harmony
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Concurrent per-sample execution
#
# V1/V2 verification generates `num_samples` independent solve attempts per QA
# pair. Each attempt is self-contained (its own tools + unique qid) and reaches
# the model through the shared background asyncio loop
# (asyncio.run_coroutine_threadsafe). Running the samples on a small thread pool
# lets their generate calls batch on the vLLM AsyncLLMEngine and hides one
# sample's tool-call latency behind another's generation.
# ---------------------------------------------------------------------------
def _sample_concurrency(num_samples, env_key, default=None):
    """Resolve sample concurrency: env override (env_key) else `default` else
    `num_samples`, clamped to [1, num_samples]."""
    val = os.environ.get(env_key)
    if val and val.isdigit() and int(val) > 0:
        n = int(val)
    else:
        n = default if default else num_samples
    return max(1, min(n, num_samples))


def run_samples_concurrently(sample_fn, num_samples, max_concurrency):
    """Run sample_fn(i) for i in 0..num_samples-1 concurrently, returning results
    in index order.

    sample_fn must be self-contained (build its own tools/qid) and catch its own
    exceptions, returning a per-sample record — mirroring today's per-sample
    try/except/finally so a failing sample never breaks the batch.
    """
    if num_samples <= 0:
        return []
    if max_concurrency <= 1 or num_samples == 1:
        return [sample_fn(i) for i in range(num_samples)]
    results = [None] * num_samples
    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = {ex.submit(sample_fn, i): i for i in range(num_samples)}
        for fut in futs:
            results[futs[fut]] = fut.result()
    return results


MATH_VERIFICATION_PROMPT = """You are answering a mathematics question. Provide ONLY the direct answer.

Rules:
- For numerical answers, give the simplified number or fraction
- For formulas, give the formula in standard notation
- For theorem/concept names, give the exact name
- Do NOT include explanations, proofs, or reasoning

Question: {question}

Answer (just the answer, nothing else):"""

MATH_V2_DEVELOPER_CONTENT = """You are a math verification agent. Answer the given question by writing and executing Python code using the provided Python tool. You have access to sympy, numpy, scipy, and mpmath libraries.

Approach:
1. Break the problem into computational steps
2. Write Python code to compute or verify each step
3. Use symbolic computation (sympy) for exact results when possible
4. Use numerical methods (numpy/scipy) as fallback

Your final response MUST be in this exact format:
Explanation: {{your explanation for your final answer.}}
Exact Answer: \\boxed{{{{your succinct, final answer}}}}
Confidence: {{your confidence score between 0% and 100% for your answer}}""".strip()


def _normalize_math_text(text):
    """Normalize math text by stripping whitespace, markdown, and LaTeX markup."""
    text = text.strip()
    # Strip markdown formatting
    text = re.sub(r'[*_]{1,2}', '', text)
    # Strip LaTeX delimiters
    text = text.replace('$', '').replace('\\(', '').replace('\\)', '')
    text = text.replace('\\[', '').replace('\\]', '')
    # Strip \text{...}, \textbf{...}, \mathrm{...} etc.
    text = re.sub(r'\\(?:text|textbf|textit|textrm|mathrm|mathbf)\{([^}]*)\}', r'\1', text)
    return text.strip()


def _normalize_latex_expr(text):
    """Normalize LaTeX math expressions for comparison."""
    text = _normalize_math_text(text)
    # \frac{a}{b} -> a/b
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)
    # \cdot -> *
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    # \sqrt{x} -> sqrt(x)
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
    # \pi -> pi, \infty -> infinity
    text = text.replace('\\pi', 'pi').replace('\\infty', 'infinity')
    # Remove remaining backslashes from LaTeX commands
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def _try_parse_fraction(text):
    """Try to parse a fraction from text. Returns float or None."""
    text = _normalize_math_text(text).strip()
    # Handle LaTeX \frac{a}{b}
    frac_match = re.search(r'\\frac\{([^}]*)\}\{([^}]*)\}', text)
    if frac_match:
        try:
            num = Fraction(frac_match.group(1).strip())
            den = Fraction(frac_match.group(2).strip())
            return float(num / den)
        except (ValueError, ZeroDivisionError, TypeError):
            pass
    # Handle plain fractions like "3/7"
    frac_match2 = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', text)
    if frac_match2:
        try:
            num = Fraction(frac_match2.group(1))
            den = Fraction(frac_match2.group(2))
            return float(num / den)
        except (ValueError, ZeroDivisionError, TypeError):
            pass
    # Try parsing as a plain number
    try:
        return float(text.replace(',', ''))
    except ValueError:
        pass
    return None


def is_math_answer_correct(predicted, gold):
    """
    Check if predicted math answer matches gold answer.

    Uses layered comparison:
    1. Normalize and exact match
    2. Substring match (both directions, len > 3 guard)
    3. LaTeX equivalence
    4. Numeric comparison with 5% tolerance
    5. Fraction normalization
    """
    if not predicted or not predicted.strip():
        return False
    pred_norm = _normalize_math_text(predicted).lower()
    gold_norm = _normalize_math_text(gold).lower()

    # 1. Exact match
    if pred_norm == gold_norm:
        return True

    # 2. Substring match (both directions)
    if gold_norm in pred_norm and len(gold_norm) > 3:
        return True
    if pred_norm in gold_norm and len(pred_norm) > 3:
        return True

    # Handle common variations
    pred_simple = pred_norm.replace('-', ' ').replace('_', ' ')
    gold_simple = gold_norm.replace('-', ' ').replace('_', ' ')
    if pred_simple == gold_simple:
        return True

    # 3. LaTeX equivalence
    pred_latex = _normalize_latex_expr(predicted)
    gold_latex = _normalize_latex_expr(gold)
    if pred_latex == gold_latex and len(pred_latex) > 1:
        return True

    # 4. Numeric comparison with 5% tolerance
    pred_num = _try_parse_fraction(predicted)
    gold_num = _try_parse_fraction(gold)
    if pred_num is not None and gold_num is not None:
        if gold_num == 0 and pred_num == 0:
            return True
        if gold_num != 0 and abs(pred_num - gold_num) / max(abs(gold_num), abs(pred_num)) <= 0.05:
            return True

    # 5. Fraction normalization via fractions.Fraction
    try:
        pred_frac = Fraction(pred_norm).limit_denominator(10000)
        gold_frac = Fraction(gold_norm).limit_denominator(10000)
        if pred_frac == gold_frac:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    return False


def _llm_judge_math_answer(predicted, gold, question):
    """Use LLM as judge for math answer comparison when heuristics fail."""
    if not predicted or not predicted.strip():
        return False
    generator = get_harmony_generator()
    if generator is None:
        return False

    prompt = (
        f"Question: {question}\n"
        f"Predicted: {predicted}\n"
        f"Gold: {gold}\n"
    )
    developer_content = (
        "Compare the predicted answer with the gold answer for the given math question. "
        "The prediction is correct if it is mathematically equivalent to the gold answer — "
        "different notation, simplified forms, or equivalent expressions should not affect "
        'correctness (e.g. "1/2" = "0.5", "x^2" = "x²", "pi/4" = "π/4"). '
        "Output brief reasoning, then \"## true\" or \"## false\"."
    )

    try:
        response = gen_from_prompt_harmony(
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            developer_content=developer_content,
        )
        parts = response.lower().split("##")
        if len(parts) >= 2:
            verdict = parts[-1].strip()
            result = verdict.startswith("true")
            print(f"    LLM judge: pred={predicted!r} gold={gold!r} -> {result}", flush=True)
            return result
        return False
    except Exception as e:
        print(f"    LLM judge error: {e}", flush=True)
        return False


def verify_math_qa_pair(question, gold_answer, agent_info, num_samples=10, temperature=0.7,
                        answer_checker=None, llm_judge=None, verification_prompt=None,
                        developer_content=None):
    """
    Verify a math QA pair by sampling multiple answers and computing accuracy (V1).

    Parallel to wiki_harmony's verify_qa_pair().
    """
    if answer_checker is None:
        answer_checker = is_math_answer_correct
    if llm_judge is None:
        llm_judge = _llm_judge_math_answer
    if verification_prompt is None:
        verification_prompt = MATH_VERIFICATION_PROMPT
    if developer_content is None:
        developer_content = "You are answering a mathematics question. Provide ONLY the direct answer."

    agent_lm, agent_tokenizer, agent_client = agent_info
    prompt = verification_prompt.replace("{question}", question)

    def _run_sample(i):
        """One closed-book sample; returns {answer, correct}. Self-contained and
        exception-safe so it can run concurrently across samples."""
        try:
            if get_harmony_generator() is not None:
                raw_answer = gen_from_prompt_harmony(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=1024,
                    developer_content=developer_content,
                ).strip()
            else:
                request_result = gen_from_prompt(
                    model=agent_lm, tokenizer=agent_tokenizer, prompt=[prompt],
                    echo_prompt=False, temperature=temperature, max_tokens=1024,
                    process_func=None, service=agent_client,
                    terminate_by_linebreak='no', verbose=False
                )
                raw_answer = request_result.completions[0].text.strip()

            # Clean up artifacts
            raw_answer = raw_answer.replace('\xa0', ' ').replace('\u202f', ' ')

            # Extract answer from first meaningful line
            sampled_answer = ""
            for line in raw_answer.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if any(p in line for p in ['**?**', '[\xa0', 'assistantfinal']):
                    continue
                # Skip reasoning lines
                line_lower = line.lower()
                skip_patterns = ['the user asks', 'we need to', 'let me', 'i need to',
                                 'so the answer', 'actually', 'maybe']
                if any(pat in line_lower for pat in skip_patterns):
                    for pat in ['the correct answer is', 'answer:', 'is:']:
                        if pat in line_lower:
                            idx = line_lower.find(pat) + len(pat)
                            candidate = line[idx:].strip().strip('.,')
                            if candidate and len(candidate) < 100:
                                sampled_answer = candidate
                                break
                    if sampled_answer:
                        break
                    continue
                sampled_answer = line
                break

            # Clean up
            for artifact in ['assistantfinal', 'assistant', '\xa0', '\u202f']:
                sampled_answer = sampled_answer.replace(artifact, ' ')
            sampled_answer = ' '.join(sampled_answer.split())
            sampled_answer = sampled_answer.strip('"\'')
            for prefix in ['The answer is ', 'Answer: ', 'It is ', 'Answer is ']:
                if sampled_answer.lower().startswith(prefix.lower()):
                    sampled_answer = sampled_answer[len(prefix):].strip()

            correct = bool(answer_checker(sampled_answer, gold_answer)
                           or llm_judge(sampled_answer, gold_answer, question))
            return {"answer": sampled_answer, "correct": correct}

        except Exception as e:
            print(f"Error sampling answer {i+1}: {e}", flush=True)
            return {"answer": "", "correct": False}

    conc = _sample_concurrency(num_samples, "DRBENCH_V1_CONCURRENCY")
    records = run_samples_concurrently(_run_sample, num_samples, conc)
    sampled_answers = [r["answer"] for r in records]
    correct_count = sum(1 for r in records if r["correct"])

    accuracy = correct_count / num_samples if num_samples > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'num_samples': num_samples,
        'sampled_answers': sampled_answers,
        'gold_answer': gold_answer
    }


def _extract_text_from_message(msg):
    """Extract plain text from a Harmony Message's content."""
    content = msg.content
    if isinstance(content, list):
        return ''.join(
            item.text if hasattr(item, 'text') else str(item) for item in content
        )
    elif hasattr(content, 'text'):
        return content.text
    elif isinstance(content, str):
        return content
    else:
        return str(content)


def _extract_boxed_content(text):
    """Extract content from \\boxed{...} handling nested braces correctly.

    A simple non-greedy regex like r'\\\\boxed\\{(.+?)\\}' fails on nested
    braces, e.g. \\boxed{\\dfrac{2}{5}} extracts only '\\dfrac{2'.
    This function counts brace depth to find the matching closing brace.

    Returns the inner content string, or None if no \\boxed{ is found.
    """
    idx = text.find('\\boxed{')
    if idx == -1:
        return None
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1]
    # Unbalanced — fall back to everything after \boxed{
    return text[start:].rstrip('}').strip()


def _try_parse_math_answer(text):
    """Parse answer from 'Exact Answer: \\boxed{...}' or fallback patterns."""
    # Try Exact Answer: \boxed{...}
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text)
    if match:
        answer = match.group(1).strip()
        boxed = _extract_boxed_content(answer)
        if boxed is not None:
            answer = boxed.strip()
        return _normalize_math_text(answer)

    # Fallback: bare \boxed{...}
    boxed = _extract_boxed_content(text)
    if boxed is not None:
        return _normalize_math_text(boxed.strip())

    return None


def _extract_math_answer_from_agentic_response(messages):
    """
    Extract the predicted answer from an agentic response message list.

    Only considers "final"-channel assistant messages to prevent
    analysis/thinking content from leaking into answers.

    Returns:
        Extracted answer string, or empty string if not found.
    """
    for msg in reversed(messages):
        if msg.author.role != Role.ASSISTANT:
            continue
        if getattr(msg, 'channel', None) != 'final':
            continue
        text = _extract_text_from_message(msg)
        if not text.strip():
            continue
        answer = _try_parse_math_answer(text)
        if answer is not None:
            return answer

    return ""


def verify_bench_v1(qa_pairs, agent_info, num_samples=10, temperature=0.7,
                    threshold=0.5, outfile_prefix=None, subarea="",
                    bench_label="se_bench",
                    answer_checker=None, llm_judge=None,
                    verification_prompt=None):
    """
    Generic V1 verification for bench modes: direct LLM answer sampling, no tools.

    For each QA pair, sends the question to the model num_samples times,
    scores each with answer_checker() + llm_judge() fallback,
    and records accuracy.

    Args:
        qa_pairs: List of QA dicts with 'question' and 'gold_answer'.
        agent_info: (agent_lm, agent_tokenizer, agent_client) tuple.
        num_samples: Number of answer samples per question.
        temperature: Sampling temperature.
        threshold: Keep questions with accuracy < threshold.
        outfile_prefix: Cache prefix for output files.
        subarea: Sub-area name for logging.
        bench_label: Label for cache files (e.g., "se_bench", "cs_bench").
        answer_checker: Function(predicted, gold) -> bool. Defaults to is_math_answer_correct.
        llm_judge: Function(predicted, gold, question) -> bool. Defaults to _llm_judge_math_answer.
        verification_prompt: Prompt template with {question} placeholder. Defaults to MATH_VERIFICATION_PROMPT.

    Returns:
        (filtered_pairs, all_pairs) — filtered has accuracy < threshold.
    """
    if answer_checker is None:
        answer_checker = is_math_answer_correct
    if llm_judge is None:
        llm_judge = _llm_judge_math_answer
    if verification_prompt is None:
        verification_prompt = MATH_VERIFICATION_PROMPT

    all_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v1.json" if outfile_prefix else None
    filtered_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v1_filtered.json" if outfile_prefix else None

    # Check cache
    if all_cache and os.path.exists(all_cache) and filtered_cache and os.path.exists(filtered_cache):
        print(f"Found cached V1 results: {all_cache}", flush=True)
        with open(all_cache, "r") as f:
            all_pairs = json.load(f)
        with open(filtered_cache, "r") as f:
            filtered_pairs = json.load(f)
        return filtered_pairs, all_pairs

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== {bench_label.upper()} V1 VERIFICATION ({subarea}) ===", flush=True)
    print(f"Verifying {len(qa_pairs)} QA pairs with {num_samples} samples each...", flush=True)

    for idx, qa in enumerate(tqdm.tqdm(qa_pairs, desc=f"V1 {subarea}")):
        if 'question' not in qa or 'gold_answer' not in qa:
            continue

        q_text = qa['question'].strip()
        if not q_text or q_text in ('**', '*'):
            print(f"  [{idx+1}] SKIP - Empty question", flush=True)
            qa_annotated = copy.deepcopy(qa)
            qa_annotated['verification_accuracy'] = 0.0
            qa_annotated['verification_correct'] = 0
            qa_annotated['verification_samples'] = num_samples
            qa_annotated['sampled_answers'] = []
            all_pairs.append(qa_annotated)
            continue

        v1_result = verify_math_qa_pair(
            qa['question'], qa['gold_answer'], agent_info,
            num_samples=num_samples, temperature=temperature,
            answer_checker=answer_checker, llm_judge=llm_judge,
            verification_prompt=verification_prompt,
        )

        qa_annotated = copy.deepcopy(qa)
        qa_annotated['verification_accuracy'] = v1_result['accuracy']
        qa_annotated['verification_correct'] = v1_result['correct_count']
        qa_annotated['verification_samples'] = v1_result['num_samples']
        qa_annotated['sampled_answers'] = v1_result['sampled_answers']
        all_pairs.append(qa_annotated)

        if v1_result['accuracy'] < threshold:
            filtered_pairs.append(qa_annotated)
            print(f"  [{idx+1}] KEEP - V1 Accuracy: {v1_result['accuracy']:.1%} - Q: {q_text[:60]}...", flush=True)
        else:
            print(f"  [{idx+1}] EASY - V1 Accuracy: {v1_result['accuracy']:.1%} - Q: {q_text[:60]}...", flush=True)

    print(f"\nV1 complete: {len(filtered_pairs)}/{len(all_pairs)} passed filter (accuracy < {threshold:.0%})", flush=True)

    # Save caches
    if all_cache:
        output_dir = os.path.dirname(all_cache)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(all_cache, "w") as f:
            json.dump(all_pairs, f, indent=2)
    if filtered_cache:
        with open(filtered_cache, "w") as f:
            json.dump(filtered_pairs, f, indent=2)

    return filtered_pairs, all_pairs
