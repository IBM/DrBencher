# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Economics Benchmark: World Bank WDI data as ground truth.

Creates questions in a 2-level difficulty space:
  Level 1 (Analyst): 2-3 steps — identify country from clues → fetch indicator → verify
  Level 2 (Senior Analyst): 3-5 steps — multi-year data, composite metrics, cross-metric

Data source: World Bank WDI API (free, no auth)
Clue source: Wikipedia country articles (non-economic descriptive facts)

Run via:  python -m drbench.economics_drbencher --exp_mode economics_bench ...
"""

import argparse
import copy
import datetime
import json
import math
import os
import random
import re
import sys
import time

import requests
import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

# OpenAI Harmony imports
from openai_harmony import (
    Message, SystemContent, Role, ReasoningEffort,
)

from .util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from .tool_util import (
    _generate_lm_answers, _generate_lm_answers_harmony, extract_json_v2,
    search_wikidata_entities, fetch_multihop_triples, fetch_wikipedia_for_entities,
)
from .wikidata_harmony import (
    validate_triple_in_chains,
    verify_facts_against_grounding,
    WIKIDATA_FACT_EXTRACTION_DEVELOPER,
    WIKIDATA_FACT_EXTRACTION_PROMPT,
)
from .harmony_vllm import HarmonyVLLMGenerator

from .multiskill_drbencher import (
    build_multiskill_grounding_articles,
)
from .multiskill_utils import (
    run_phase3_validation,
    check_kg_uniqueness_comparative,
    strip_side_prefix,
    extract_chain_clue_facts,
    verify_facts_grounding,
    filter_poisoned_facts,
    poison_used_facts,
    compute_cci_fields,
)

from .econ_template import (
    ECONOMICS_TEMPLATES,
    select_economics_template,
    select_comparative_template,
    select_temporal_template,
)

# Reuse verification infrastructure (V1/V2) from bench_verify
from .bench_verify import (
    verify_bench_v1,
    _sample_concurrency,
    run_samples_concurrently,
)
from .harmony_base import (
    gen_from_prompt_harmony,
    get_harmony_generator,
    set_harmony_generator,
    shutdown_and_exit,
)

# Economics utilities
from .econ_util import (
    fetch_country_indicators,
    get_country_name,
    get_country_info,
    get_indicator_value,
    get_indicator_history,
    compare_countries_indicator,
    fetch_country_clues,
    get_region_countries,
    resolve_country_wikidata_id,
    find_best_year,
    ECONOMICS_INDICATORS,
    COUNTRY_UNIVERSE_REGIONS,
    ECONOMICS_THEMES,
)

# Browser tool for V2 verification
try:
    from tools.kg_browser import MultiSourceKnowledgeBackend, MultiSourceKnowledgeBrowserTool
    _HAS_BROWSER = True
except ImportError:
    _HAS_BROWSER = False

# Python tool for V2 verification
try:
    from tools.hybrid_exec_qid import HybridPythonTool
    _HAS_PYTHON_TOOL = True
except ImportError:
    _HAS_PYTHON_TOOL = False

# World Bank tool for V2 verification
try:
    from tools.econ_tool import WorldBankEconomicsTool
    _HAS_ECON_TOOL = True
except ImportError:
    _HAS_ECON_TOOL = False

# Diversity filter (optional)
try:
    from .diversity import diversity_filter, diversity_report
    _HAS_DIVERSITY = True
except ImportError:
    _HAS_DIVERSITY = False


# ===========================================================================
# Prompts
# ===========================================================================

ECONOMICS_QA_DEVELOPER = (
    "You are an economist creating research questions that test "
    "the ability to identify countries from geographic/cultural descriptions and "
    "compute economic indicators from World Bank data."
)

ECONOMICS_QA_PROMPT = """Compose an economics research question that:
1. Uses clue facts to describe an unnamed country (readers must figure out which country)
2. Then asks for a specific economic indicator computation requiring the solver to look up World Bank data

CLUE FACTS (about the unnamed country — do NOT name it or reveal its ISO code):
{facts_text}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- STYLE GUIDANCE ---
- Write 2-4 sentences total.
- First 1-2 sentences: describe the country using 3+ clue facts from different topics, without naming it or its ISO code.
- Last 1-2 sentences: pose the economic indicator question.
- CRITICAL: Do NOT reveal ANY economic figures in the question — not GDP, population, or any computed values. The solver must look up ALL data themselves from World Bank sources.
- CRITICAL: Do NOT name the country, its ISO code, or its capital city anywhere in the question.
- You MUST specify the exact data year (e.g., "for 2022" or "using 2022 data"). NEVER use vague time references.
- Sound natural and conversational, like a real economist's research question.
- Do NOT mention "World Bank", "WDI", or specific indicator codes directly.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your economics research question>
Used_Facts: <comma-separated fact_ids used, e.g. W1, W2, W3>
Reasoning: <brief chain: clues identify country → look up economic data → computation gives answer>
"""

ECONOMICS_COMPARATIVE_QA_PROMPT = """Compose a comparative economics research question that:
1. Describes TWO unnamed countries using clue facts about each
2. Asks for a specific economic indicator comparison requiring the solver to look up World Bank data for both

CLUE FACTS FOR COUNTRY A (do NOT name this country, its ISO code, or its capital city):
{facts_text_a}

CLUE FACTS FOR COUNTRY B (do NOT name this country, its ISO code, or its capital city):
{facts_text_b}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- STYLE GUIDANCE ---
- Write 4-6 sentences total.
- First 1-2 sentences: describe the first country using 3+ clue facts from different topics, without naming it or its ISO code.
- Next 1-2 sentences: describe the second country using 3+ clue facts from different topics, without naming it or its ISO code.
- Last 1-2 sentences: pose the comparative economic indicator question.
- Refer to them as "the first country" and "the second country" (or similar distinct references).
- CRITICAL: Do NOT reveal ANY economic figures in the question — not GDP, population, or any computed values. The solver must look up ALL data themselves from World Bank sources.
- CRITICAL: Do NOT name either country, its ISO code, or its capital city anywhere in the question.
- You MUST reference the {data_year} data year in your question (e.g., "for {data_year}" or "using {data_year} data"). NEVER use vague time references.
- Sound natural and conversational, like a real economist's research question.
- Do NOT mention "World Bank", "WDI", or specific indicator codes directly.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your comparative economics research question>
Used_Facts_A: <comma-separated fact_ids for country A, e.g. A_C1_F1, A_C2_F1>
Used_Facts_B: <comma-separated fact_ids for country B, e.g. B_C1_F1, B_C2_F1>
Reasoning: <brief chain: clues identify both countries -> look up economic data -> comparison gives answer>

REQUIREMENTS:
- Use 3+ clue facts per country from different chains
- Do NOT name either country in the question
- Do NOT include economic indicator values in the question
- Every claim must come VERBATIM from the listed facts
- The question MUST have exactly ONE unambiguous answer"""

ECONOMICS_V1_PROMPT = """Solve this economics research question. It requires:
1. Identifying a country from its geographic/cultural description
2. Looking up its economic data from World Bank sources
3. Computing the requested economic indicator

Provide ONLY the final numerical answer (with % or units if applicable).
Do not show your work.

Question: {question}

Answer:"""

ECONOMICS_V2_DEVELOPER = """You are an expert economist.
You have THREE tools available:
- **World Bank tool**: Look up country economic data from World Development Indicators (search_country, get_indicators, get_indicator, compare_countries, get_indicator_history)
- **Browser tool**: Search Wikipedia/web to identify countries from descriptions
- **Python tool**: Execute Python code for calculations

Recommended approach:
1. Use the browser to search for and identify the country from the description clues
2. Use the World Bank tool to fetch economic indicators (GDP, population, etc.)
3. Use Python to perform the computation and get the final answer

Give your final answer as a single number (with units if appropriate) on the last line."""


# ===========================================================================
# Answer Checker
# ===========================================================================

def is_economics_answer_correct(predicted, gold, tolerance=0.05):
    """Economics answer comparison with 5% tolerance.

    Handles: percentages, ratios, dollar amounts (with B/M suffixes),
    index values, and population figures.

    Args:
        predicted: Predicted answer string.
        gold: Gold answer string.
        tolerance: Relative tolerance (default 5%).

    Returns:
        True if answer is correct within tolerance.
    """
    if not predicted or not predicted.strip():
        return False

    pred_clean = _normalize_econ_text(predicted)
    gold_clean = _normalize_econ_text(gold)

    # 1. Exact match
    if pred_clean == gold_clean:
        return True

    # 2. Numeric comparison
    pred_num = _try_parse_econ_number(pred_clean)
    gold_num = _try_parse_econ_number(gold_clean)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0 and pred_num == 0:
            return True
        if gold_num == 0:
            return abs(pred_num) < 0.01
        rel_error = abs(pred_num - gold_num) / max(abs(gold_num), abs(pred_num))
        if rel_error <= tolerance:
            return True

    return False


def _normalize_econ_text(text):
    """Normalize text for economic comparison."""
    text = text.strip()
    for suffix in ["%", "percentage points", "pp", "times", "ratio", "x",
                   "US$", "USD", "$", "per capita", "people", "km²",
                   "metric tons", "years", "index"]:
        text = text.replace(suffix, "").strip()
    text = text.replace(",", "")
    text = text.strip().strip("'\"")
    return text


def _try_parse_econ_number(text):
    """Parse a number from economics text. Returns float or None."""
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]

    multipliers = {
        "trillion": 1e12, "billion": 1e9, "million": 1e6,
        "thousand": 1e3, "T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3,
    }
    for word, mult in multipliers.items():
        if word in text:
            text = text.replace(word, "").strip()
            try:
                return float(text) * mult
            except ValueError:
                continue

    try:
        return float(text)
    except ValueError:
        match = re.search(r'-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None


def _llm_judge_econ_answer(predicted, gold, question):
    """Use LLM to judge if predicted answer matches gold for economics questions."""
    prompt = f"""Compare these two answers to the same economics question.
The gold answer is computed from verified World Bank WDI data.
The predicted answer may use different units, rounding, or phrasing.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Are these answers equivalent (same value within ~5% tolerance, possibly different formatting)?
Respond with exactly YES or NO."""

    try:
        response = gen_from_prompt_harmony(prompt, temperature=0.0, max_tokens=10)
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


# ===========================================================================
# QA Generation Pipeline
# ===========================================================================

def compose_economics_question(clues, reasoning_ctx, iso3, country_name,
                               prev_questions):
    """Compose an economics QA question using LLM.

    Args:
        clues: List of clue fact dicts.
        reasoning_ctx: Template context from select_*_template().
        iso3: ISO-3 code (for validation only).
        country_name: Country name (for validation only).
        prev_questions: List of previously generated question strings.

    Returns:
        Dict with "question", "used_facts" keys, or None on failure.
    """
    facts_text = "\n".join(
        f"[{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) {c['fact']}"
        for c in clues
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    prompt = ECONOMICS_QA_PROMPT.format(
        facts_text=facts_text,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx.get("question_hint", ""),
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=ECONOMICS_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception as e:
        print(f"  Composition LLM error: {e}", flush=True)
        return None

    # Parse response
    question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    facts_match = re.search(r'Used_Facts:\s*(.+?)(?:\n|$)', response)

    if not question_match:
        return None

    question = question_match.group(1).strip()

    # Strip fact ID annotations
    question = re.sub(r'\s*[\(\[]\s*[AB]?_?C\d+_F\d+\s*[\)\]]', '', question)

    used_facts = []
    if facts_match:
        used_facts = [f.strip() for f in facts_match.group(1).split(",") if f.strip()]

    return {"question": question, "used_facts": used_facts}


def compose_economics_comparative_question(clues_a, clues_b, reasoning_ctx,
                                            iso3_a, iso3_b, country_name_a,
                                            country_name_b, prev_questions,
                                            data_year=2022):
    """Compose a comparative economics QA question with per-entity clue blocks.

    Unlike compose_economics_question (which merges clues), this keeps A/B clues
    separate so the LLM is forced to describe both countries with 3+ clues each.

    Args:
        clues_a: List of clue fact dicts for country A.
        clues_b: List of clue fact dicts for country B.
        reasoning_ctx: Template context from select_comparative_template().
        iso3_a: ISO-3 code for country A (for validation only).
        iso3_b: ISO-3 code for country B (for validation only).
        country_name_a: Country A name (for validation only).
        country_name_b: Country B name (for validation only).
        prev_questions: List of previously generated question strings.
        data_year: Data year for economic indicators (default: 2022).

    Returns:
        Dict with "question", "used_facts", "used_facts_a", "used_facts_b"
        keys, or None on failure.
    """
    facts_text_a = "\n".join(
        f"[A_{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) "
        f"[Country A] {c['fact']}"
        for c in clues_a
    )
    facts_text_b = "\n".join(
        f"[B_{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) "
        f"[Country B] {c['fact']}"
        for c in clues_b
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    prompt = ECONOMICS_COMPARATIVE_QA_PROMPT.format(
        facts_text_a=facts_text_a,
        facts_text_b=facts_text_b,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx.get("question_hint", ""),
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        data_year=data_year,
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=ECONOMICS_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception as e:
        print(f"  Comparative composition LLM error: {e}", flush=True)
        return None

    # Parse response
    q_match = re.search(
        r'Question:\s*(.+?)(?:\nUsed_Facts|\nReasoning|\Z)', response, re.DOTALL
    )
    if not q_match:
        return None

    question = q_match.group(1).strip()

    # Strip fact-ID annotations the LLM may copy from the prompt
    question = re.sub(r'\s*[\(\[]\s*(?:[AB]_)?(?:C\d+_F\d+|W\d+|F\d+)\s*[\)\]]', '', question)

    # Parse Used_Facts_A and Used_Facts_B separately
    used_facts_a = []
    used_facts_b = []
    m_a = re.search(r'Used_Facts_A:\s*(.+?)(?:\n|$)', response)
    m_b = re.search(r'Used_Facts_B:\s*(.+?)(?:\n|$)', response)
    if m_a:
        used_facts_a = [f.strip() for f in m_a.group(1).split(",") if f.strip()]
    if m_b:
        used_facts_b = [f.strip() for f in m_b.group(1).split(",") if f.strip()]

    # Also try fallback: plain Used_Facts line (in case LLM ignores per-entity format)
    all_used = used_facts_a + used_facts_b
    if not all_used:
        m_plain = re.search(r'Used_Facts:\s*(.+?)(?:\n|$)', response)
        if m_plain:
            all_used = [f.strip() for f in m_plain.group(1).split(",") if f.strip()]
            used_facts_a = [f for f in all_used if f.startswith("A_")]
            used_facts_b = [f for f in all_used if f.startswith("B_")]

    return {
        "question": question,
        "used_facts": used_facts_a + used_facts_b,
        "used_facts_a": used_facts_a,
        "used_facts_b": used_facts_b,
    }


def gen_economics_qa_from_entity(entity_data, agent_info, num_questions=3,
                                  prev_questions=None, used_templates=None,
                                  articles=None, region=None):
    """Generate economics QA pairs from a single entity using chain-based clues.

    Parallels gen_financial_qa_from_entity with Phases 0/1/1.5/2/3.

    Args:
        entity_data: Dict with 'iso3', 'wikidata_id', 'label', 'indicators',
                     'chains', 'articles'.
        agent_info: (lm, tokenizer, client) tuple.
        num_questions: Questions to generate per entity.
        prev_questions: Previously generated question strings.
        used_templates: Set of template IDs already used.
        articles: Wikipedia article dicts (overrides entity_data['articles']).
        region: Region key.

    Returns:
        List of QA pair dicts.
    """
    if prev_questions is None:
        prev_questions = []
    if used_templates is None:
        used_templates = set()

    iso3 = entity_data["iso3"]
    indicators = entity_data["indicators"]
    country_name = indicators.get("name", iso3)
    data_year = indicators.get("data_year", 2022)
    capital = indicators.get("capital", "")
    chains = entity_data.get("chains", [])
    wikidata_id = entity_data.get("wikidata_id", "")
    if articles is None:
        articles = entity_data.get("articles", [])

    if not chains or len(chains) < 3:
        print(f"  No KG chains for {country_name} ({iso3})", flush=True)
        return []

    qa_pairs = []
    max_attempts = num_questions * 3
    poisoned_props = set()

    for attempt in range(max_attempts):
        if len(qa_pairs) >= num_questions:
            break

        print(f"\n  [{country_name}] Attempt {attempt+1}/{max_attempts} "
              f"(generated {len(qa_pairs)}/{num_questions})", flush=True)

        # Phase 0: Select template and compute gold answer
        reasoning_ctx = select_economics_template(indicators, used_templates, region)
        is_temporal = False
        if reasoning_ctx is None:
            reasoning_ctx = select_temporal_template(iso3, indicators, used_templates)
            is_temporal = True
        if reasoning_ctx is None:
            print(f"  No valid template remaining for {iso3}", flush=True)
            single_count = len([t for t in ECONOMICS_TEMPLATES
                                if ECONOMICS_TEMPLATES[t]["type"] in ("single", "composite")])
            temp_count = len([t for t in ECONOMICS_TEMPLATES
                              if ECONOMICS_TEMPLATES[t]["type"] == "temporal"])
            if len(used_templates) >= single_count + temp_count:
                used_templates.clear()
                continue
            break

        print(f"  Phase 0: {reasoning_ctx['template_id']} → {reasoning_ctx['gold_answer']}",
              flush=True)

        # Phase 1: Extract clue facts from KG chains
        extracted_facts, _ = extract_chain_clue_facts(chains, country_name, agent_info)
        if extracted_facts is None:
            continue

        # Phase 1.5: Verify facts against Wikipedia grounding documents
        grounded_facts = verify_facts_grounding(extracted_facts, chains, articles, agent_info)
        if grounded_facts is None:
            continue
        extracted_facts = grounded_facts

        # Filter out poisoned facts from previously rejected QAs
        extracted_facts = filter_poisoned_facts(extracted_facts, poisoned_props)
        if len(extracted_facts) < 3:
            print(f"  Too few facts after poisoned-fact filtering", flush=True)
            continue

        # Phase 2: Compose question from grounded clues
        clues_for_composition = [
            {
                "fact_id": f.get("fact_id", f"C{f.get('chain_num', 0)}_F1"),
                "topic": f.get("property", "description"),
                "fact": f.get("fact", ""),
            }
            for f in extracted_facts
        ]
        composition = compose_economics_question(
            clues_for_composition, reasoning_ctx, iso3, country_name, prev_questions,
        )
        if composition is None:
            print(f"  Phase 2: Composition failed", flush=True)
            continue

        question = composition["question"]
        used_facts = composition["used_facts"]
        print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

        # Phase 3: Validation
        # 3-pre: Must specify data year (domain-specific)
        if str(data_year) not in question:
            if is_temporal:
                start_year = reasoning_ctx.get("start_year")
                end_year = reasoning_ctx.get("end_year")
                if (start_year and str(start_year) not in question) or \
                   (end_year and str(end_year) not in question):
                    print(f"  Phase 3: Data year not specified in question", flush=True)
                    continue
            else:
                print(f"  Phase 3: Data year not specified in question", flush=True)
                continue

        # Reject truncated questions (LLM output cut off)
        if not question.rstrip().endswith(("?", ".")):
            print(f"  Phase 3: Question appears truncated", flush=True)
            continue

        # Build entity_names for leak check: country name, ISO codes, capital, common aliases
        entity_names = [country_name, iso3]
        if capital:
            entity_names.append(capital)
        # Add ISO-2 if derivable
        info = get_country_info(iso3)
        if info:
            iso2 = info.get("iso2Code", "")
            if iso2 and len(iso2) == 2:
                entity_names.append(iso2)

        # Collect indicator values for value-leak check
        ind_values = []
        for ind_key in reasoning_ctx["template"]["required_indicators"]:
            val = indicators.get(ind_key)
            if val is not None and isinstance(val, (int, float)):
                ind_values.append(val)

        passed, reason = run_phase3_validation(
            question=question,
            gold_answer=reasoning_ctx["gold_answer"],
            computation_code=reasoning_ctx["code"],
            entity_label=country_name,
            entity_names=entity_names,
            entity_values=ind_values,
            used_facts=used_facts,
            extracted_facts=extracted_facts,
            gen_fn=gen_from_prompt_harmony,
            data_year=data_year,
            entity_type="country",
            chains=chains,
            entity_qid=wikidata_id,
        )
        if not passed:
            # Poison the used facts only when the rejection is intrinsic to the
            # pair (3d3b); other reasons are phrasing/combination, so retry
            # instead of banning innocent facts and starving sparse entities.
            poison_used_facts(reason, used_facts, extracted_facts, poisoned_props)
            print(f"  Phase 3: {reason}", flush=True)
            continue
        print(f"  Phase 3: PASSED", flush=True)

        # Build output fields
        source_triples = [c["path_description"] for c in chains[:5]]
        _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
        facts_for_output = [
            {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
            for f in extracted_facts
        ]

        # Store grounding indicator data
        grounding_indicators = {
            k: indicators[k]
            for k in reasoning_ctx["template"]["required_indicators"]
            if k in indicators
        }

        # Build full-text grounding articles from chain intermediates
        full_grounding_articles = build_multiskill_grounding_articles(
            country_name, reasoning_ctx["gold_answer"],
            reasoning_ctx["template"]["answer_unit"],
            extracted_facts, chains, articles or [],
        )

        qa_pair = {
            "question": question,
            "gold_answer": reasoning_ctx["gold_answer"],
            "computation_code": reasoning_ctx["code"],
            "template_id": reasoning_ctx["template_id"],
            "template_label": reasoning_ctx["template"]["label"],
            "template_type": reasoning_ctx["template"]["type"],
            "template_level": reasoning_ctx["template"]["level"],
            "answer_unit": reasoning_ctx["template"]["answer_unit"],
            "iso3": iso3,
            "country_name": country_name,
            "entity_label": country_name,
            "data_year": data_year,
            "used_facts": used_facts,
            "extracted_facts": facts_for_output,
            "source_triples": source_triples,
            "grounding_articles": full_grounding_articles,
            "wikidata_entity": wikidata_id,
            "data_source": "world_bank_wdi",
            "region": region,
            "grounding_clues": clues_for_composition,
            "grounding_indicators": grounding_indicators,
        }
        qa_pair.update(compute_cci_fields(reasoning_ctx["template"], extracted_facts, is_comparative=False))

        if is_temporal:
            qa_pair["start_year"] = reasoning_ctx.get("start_year")
            qa_pair["end_year"] = reasoning_ctx.get("end_year")

        qa_pairs.append(qa_pair)
        prev_questions.append(question)
        used_templates.add(reasoning_ctx["template_id"])

    return qa_pairs


def gen_economics_qa_comparative(iso3_a, indicators_a, clues_a,
                                  iso3_b, indicators_b, clues_b,
                                  used_templates, prev_questions, region,
                                  chains_a=None, articles_a=None,
                                  chains_b=None, articles_b=None,
                                  agent_info=None):
    """Generate a comparative economics QA pair (two countries)."""
    name_a = indicators_a.get("name", iso3_a)
    name_b = indicators_b.get("name", iso3_b)
    capital_a = indicators_a.get("capital", "")
    capital_b = indicators_b.get("capital", "")
    data_year_a = indicators_a.get("data_year", 2022)
    data_year_b = indicators_b.get("data_year", 2022)
    if data_year_a != data_year_b:
        print(f"  Skipping comparative: {iso3_a} year={data_year_a} vs {iso3_b} year={data_year_b}", flush=True)
        return None
    data_year = data_year_a

    # Phase 0: Select comparative template
    reasoning_ctx = select_comparative_template(indicators_a, indicators_b, used_templates)
    if reasoning_ctx is None:
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} → {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1 + 1.5: Extract and ground chain-based clues for each country
    facts_a = None
    facts_b = None
    if chains_a and chains_b:
        facts_a, _ = extract_chain_clue_facts(chains_a, name_a, agent_info)
        if facts_a is not None and articles_a:
            facts_a = verify_facts_grounding(facts_a, chains_a, articles_a, agent_info)
        facts_b, _ = extract_chain_clue_facts(chains_b, name_b, agent_info)
        if facts_b is not None and articles_b:
            facts_b = verify_facts_grounding(facts_b, chains_b, articles_b, agent_info)

        if facts_a and facts_b:
            clues_a = [
                {"fact_id": f.get("fact_id", ""), "topic": f.get("property", "description"),
                 "fact": f.get("fact", "")}
                for f in facts_a
            ]
            clues_b = [
                {"fact_id": f.get("fact_id", ""), "topic": f.get("property", "description"),
                 "fact": f.get("fact", "")}
                for f in facts_b
            ]

    # BOTH countries must have grounded clues (need 3+ each for per-side validation)
    if len(clues_a) < 3:
        print(f"  Comparative: insufficient clues for country A ({name_a}, have {len(clues_a)}, need 3+)", flush=True)
        return None
    if len(clues_b) < 3:
        print(f"  Comparative: insufficient clues for country B ({name_b}, have {len(clues_b)}, need 3+)", flush=True)
        return None

    # Phase 2: Compose question with per-entity clue blocks
    composition = compose_economics_comparative_question(
        clues_a, clues_b, reasoning_ctx, iso3_a, iso3_b,
        name_a, name_b, prev_questions, data_year=data_year,
    )
    if composition is None:
        return None

    question = composition["question"]
    used_facts = composition.get("used_facts", [])

    # Validate per-side used_facts count (need 3+ identifying clues per country)
    a_used = [f for f in used_facts if str(f).startswith("A_")]
    b_used = [f for f in used_facts if str(f).startswith("B_")]
    if len(a_used) < 3 or len(b_used) < 3:
        print(f"  Phase 3: Insufficient per-side clues (A={len(a_used)}, B={len(b_used)}, need 3+ each)", flush=True)
        return None

    # Reject truncated questions (LLM output cut off)
    if not question.rstrip().endswith(("?", ".")):
        print(f"  Phase 3: Comparative question appears truncated", flush=True)
        return None

    # Phase 3: Validation
    all_values = []
    for indicators_src in [indicators_a, indicators_b]:
        for ind in reasoning_ctx["template"]["required_indicators"]:
            val = indicators_src.get(ind)
            if val is not None and isinstance(val, (int, float)):
                all_values.append(val)

    entity_names = [name_a, iso3_a, name_b, iso3_b]
    if capital_a:
        entity_names.append(capital_a)
    if capital_b:
        entity_names.append(capital_b)

    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=f"{name_a} / {name_b}",
        entity_names=entity_names,
        entity_values=all_values,
        used_facts=used_facts,
        extracted_facts=(facts_a or []) + (facts_b or []),
        data_year=data_year,
        entity_type="country",
    )
    if passed:
        # 3h (comparative): each side must be uniquely identifiable from its clues
        passed, reason = check_kg_uniqueness_comparative(
            strip_side_prefix(used_facts, "A_"), facts_a or [], chains_a,
            strip_side_prefix(used_facts, "B_"), facts_b or [], chains_b,
        )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    grounding_indicators_a = {
        k: indicators_a[k]
        for k in reasoning_ctx["template"]["required_indicators"]
        if k in indicators_a
    }
    grounding_indicators_b = {
        k: indicators_b[k]
        for k in reasoning_ctx["template"]["required_indicators"]
        if k in indicators_b
    }

    source_triples_a = [c["path_description"] for c in (chains_a or [])[:3]]
    source_triples_b = [c["path_description"] for c in (chains_b or [])[:3]]

    grounding_articles_a = build_multiskill_grounding_articles(
        name_a, reasoning_ctx["gold_answer"],
        reasoning_ctx["template"]["answer_unit"],
        facts_a or [], chains_a or [], articles_a or [],
    ) if chains_a else []
    grounding_articles_b = build_multiskill_grounding_articles(
        name_b, reasoning_ctx["gold_answer"],
        reasoning_ctx["template"]["answer_unit"],
        facts_b or [], chains_b or [], articles_b or [],
    ) if chains_b else []

    result = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": reasoning_ctx["template"]["label"],
        "template_type": "comparative",
        "template_level": reasoning_ctx["template"]["level"],
        "answer_unit": reasoning_ctx["template"]["answer_unit"],
        "iso3_a": iso3_a,
        "country_name_a": name_a,
        "iso3_b": iso3_b,
        "country_name_b": name_b,
        "data_year": data_year,
        "used_facts": used_facts,
        "data_source": "world_bank_wdi",
        "source_triples": source_triples_a + source_triples_b,
        "grounding_clues_a": clues_a,
        "grounding_clues_b": clues_b,
        "grounding_indicators_a": grounding_indicators_a,
        "grounding_indicators_b": grounding_indicators_b,
        "grounding_articles": grounding_articles_a + grounding_articles_b,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=True))
    return result


# ===========================================================================
# V2 Verification: World Bank + Browser + Python Tools
# ===========================================================================

def verify_economics_v2(qa_pairs, num_samples=10, temperature=1.0,
                         max_iterations=200, threshold=0.5,
                         outfile_prefix=None, subarea="",
                         bench_label="economics_bench"):
    """V2 verification using World Bank + Browser + Python tools."""
    all_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v2.json" if outfile_prefix else None
    filtered_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v2_filtered.json" if outfile_prefix else None

    # Check cache
    if all_cache and os.path.exists(all_cache) and filtered_cache and os.path.exists(filtered_cache):
        print(f"Found cached V2 results: {all_cache}", flush=True)
        with open(all_cache, "r") as f:
            all_pairs = json.load(f)
        with open(filtered_cache, "r") as f:
            filtered_pairs = json.load(f)
        return filtered_pairs, all_pairs

    generator = get_harmony_generator()
    if generator is None:
        raise RuntimeError("Harmony generator not initialized for V2 verification")
    if not _HAS_BROWSER:
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available for economics V2")
    if not _HAS_PYTHON_TOOL:
        raise RuntimeError("HybridPythonTool not available for economics V2")
    if not _HAS_ECON_TOOL:
        raise RuntimeError("WorldBankEconomicsTool not available for economics V2")

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== ECONOMICS V2 VERIFICATION ({subarea}) ===", flush=True)
    print(f"Tools: World Bank + Browser + Python", flush=True)
    print(f"Verifying {len(qa_pairs)} QA pairs with {num_samples} samples each...", flush=True)

    for idx, qa in enumerate(tqdm.tqdm(qa_pairs, desc=f"V2 {subarea}")):
        if 'question' not in qa or 'gold_answer' not in qa:
            continue

        q_text = qa['question'].strip()
        if not q_text or q_text in ('**', '*'):
            qa_annotated = copy.deepcopy(qa)
            qa_annotated['v2_accuracy'] = 0.0
            qa_annotated['v2_correct_count'] = 0
            qa_annotated['v2_num_samples'] = num_samples
            qa_annotated['v2_sampled_answers'] = []
            qa_annotated['v2_tool_call_count'] = []
            qa_annotated['v2_iteration_count'] = []
            all_pairs.append(qa_annotated)
            continue

        sampled_answers = []
        correct_count = 0
        per_sample_tool_calls = []
        per_sample_iterations = []
        per_sample_errors = []

        def _run_sample(i):
            """One agentic V2 sample; self-contained + exception-safe so samples
            can run concurrently. Returns an ordered per-sample record."""
            backend = None
            python_tool = None
            econ_tool = None
            qid_str = f"{bench_label}_v2_{subarea}_{idx}_{i}"
            try:
                backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
                browser_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
                python_tool = HybridPythonTool(timeout=60)
                python_tool.set_qid(qid_str)
                econ_tool = WorldBankEconomicsTool()

                system_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.HIGH)
                    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                    .with_tools(browser_tool.tool_config)
                    .with_tools(python_tool.tool_config)
                    .with_tools(econ_tool.tool_config)
                )

                messages = [
                    Message.from_role_and_content(Role.SYSTEM, system_content),
                    Message.from_role_and_content(Role.DEVELOPER, ECONOMICS_V2_DEVELOPER),
                    Message.from_role_and_content(Role.USER, f"Question: {qa['question']}"),
                ]

                tool_call_counter = [0]

                async def _tool_handler(msg, _browser=browser_tool, _python=python_tool,
                                        _econ=econ_tool, _counter=tool_call_counter):
                    _counter[0] += 1
                    recipient = str(getattr(msg, 'recipient', ''))
                    results = []
                    if recipient.startswith("worldbank"):
                        async for m in _econ.process(msg):
                            results.append(m)
                    elif recipient.startswith("browser."):
                        async for m in _browser.process(msg):
                            results.append(m)
                    elif recipient.startswith("python"):
                        async for m in _python.process(msg):
                            results.append(m)
                    else:
                        error = Message.from_role_and_content(
                            Role.SYSTEM, f"Unknown tool: {recipient}"
                        )
                        results.append(error)
                    return results

                result_messages = generator.generate_agentic_response_sync(
                    messages,
                    _tool_handler,
                    tool_prefix=("worldbank", "browser.", "python"),
                    tool_configs=[browser_tool.tool_config, python_tool.tool_config, econ_tool.tool_config],
                    max_iterations=max_iterations,
                    temperature=temperature,
                )

                iteration_count = len(result_messages) - len(messages)

                answer = _extract_econ_answer(result_messages)
                answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
                answer = ' '.join(answer.split())
                correct = bool(is_economics_answer_correct(answer, qa['gold_answer']) or
                               _llm_judge_econ_answer(answer, qa['gold_answer'], qa['question']))
                log = (f"    V2 sample {i+1}/{num_samples}: '{answer}' "
                       f"(gold: '{qa['gold_answer']}') tools={tool_call_counter[0]} "
                       f"iters={iteration_count}")
                return {"answer": answer, "tool_calls": tool_call_counter[0],
                        "iterations": iteration_count, "error": None,
                        "correct": correct, "log": log}

            except Exception as e:
                import traceback
                error_str = f"{type(e).__name__}: {e}"
                tb_str = traceback.format_exc()
                log = (f"    V2 sample {i+1}/{num_samples} error: {error_str}\n{tb_str}")
                return {"answer": "", "tool_calls": 0, "iterations": 0,
                        "error": error_str, "correct": False, "log": log}

            finally:
                if backend is not None:
                    try:
                        import asyncio
                        asyncio.run_coroutine_threadsafe(
                            backend.close(), generator._loop
                        ).result(timeout=5)
                    except Exception:
                        pass
                if python_tool is not None:
                    try:
                        python_tool._terminate_worker(qid_str)
                    except Exception:
                        pass

        conc = _sample_concurrency(num_samples, "DRBENCH_V2_CONCURRENCY", default=4)
        records = run_samples_concurrently(_run_sample, num_samples, conc)
        for r in records:
            sampled_answers.append(r["answer"])
            per_sample_tool_calls.append(r["tool_calls"])
            per_sample_iterations.append(r["iterations"])
            per_sample_errors.append(r["error"])
            if r["correct"]:
                correct_count += 1
            print(r["log"], flush=True)

        accuracy = correct_count / num_samples if num_samples > 0 else 0.0
        import numpy as np
        avg_tool_calls = float(np.mean(per_sample_tool_calls)) if per_sample_tool_calls else 0.0
        avg_iterations = float(np.mean(per_sample_iterations)) if per_sample_iterations else 0.0

        qa_annotated = copy.deepcopy(qa)
        qa_annotated['v2_accuracy'] = accuracy
        qa_annotated['v2_correct_count'] = correct_count
        qa_annotated['v2_num_samples'] = num_samples
        qa_annotated['v2_sampled_answers'] = sampled_answers
        qa_annotated['v2_tool_call_count'] = per_sample_tool_calls
        qa_annotated['v2_iteration_count'] = per_sample_iterations
        qa_annotated['v2_avg_tool_calls'] = avg_tool_calls
        qa_annotated['v2_avg_iterations'] = avg_iterations
        errors_only = [e for e in per_sample_errors if e is not None]
        if errors_only:
            qa_annotated['v2_errors'] = errors_only
        all_pairs.append(qa_annotated)

        if accuracy < threshold:
            filtered_pairs.append(qa_annotated)
            print(f"  [{idx+1}] KEEP - V2 Accuracy: {accuracy:.1%} "
                  f"avg_tools={avg_tool_calls:.1f} avg_iters={avg_iterations:.1f} "
                  f"- Q: {q_text[:60]}...", flush=True)
        else:
            print(f"  [{idx+1}] SOLVED - V2 Accuracy: {accuracy:.1%} "
                  f"avg_tools={avg_tool_calls:.1f} avg_iters={avg_iterations:.1f} "
                  f"- Q: {q_text[:60]}...", flush=True)

    print(f"\nV2 complete: {len(filtered_pairs)}/{len(all_pairs)} passed filter "
          f"(accuracy < {threshold:.0%})", flush=True)

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


def _extract_econ_answer(messages):
    """Extract the final answer from an agentic response."""
    from .multiskill_utils import unwrap_content
    for msg in reversed(messages):
        if hasattr(msg, 'author') and msg.author.role == Role.ASSISTANT:
            content = unwrap_content(getattr(msg, 'content', ''))
            if not content.strip():
                continue
            lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
            if lines:
                last = lines[-1]
                for prefix in ["Answer:", "Final answer:", "The answer is", "Result:",
                               "Final Answer:"]:
                    if last.lower().startswith(prefix.lower()):
                        last = last[len(prefix):].strip()
                return last
    return ""


# ===========================================================================
# Orchestrator
# ===========================================================================

def _generate_region_qa_pairs(region, per_region, data_year, num_chains,
                               questions_per_entity, agent_info, prefix, bench_label):
    """Generate QA pairs for a single region.

    Uses chains-cache pattern (parallel to financial):
    1. Fetch countries → resolve Wikidata QIDs → fetch KG chains + Wikipedia articles
    2. Cache as {prefix}__{region}.economics_chains.json with backfill
    3. Generate QA pairs per entity via gen_economics_qa_from_entity
    4. Pair entities for comparative QAs
    """
    chains_cache = f"{prefix}__{region}.economics_chains.json"
    if os.path.exists(chains_cache):
        print(f"Loading cached chains for {region}...", flush=True)
        with open(chains_cache) as f:
            country_data_list = json.load(f)

        # Backfill Wikipedia articles
        needs_resave = False
        for cd in country_data_list:
            if "articles" not in cd or not cd["articles"]:
                print(f"  Fetching Wikipedia articles for cached entity {cd.get('label', cd['iso3'])}...",
                      flush=True)
                cd["articles"] = fetch_wikipedia_for_entities(cd.get("chains", []))
                print(f"  {cd.get('label', cd['iso3'])}: fetched {len(cd['articles'])} Wikipedia articles",
                      flush=True)
                needs_resave = True
        if needs_resave:
            with open(chains_cache, "w") as f:
                json.dump(country_data_list, f, indent=2)
            print(f"Re-saved chains cache with Wikipedia articles for {region}", flush=True)
    else:
        iso3_codes = get_region_countries(region)
        if not iso3_codes:
            print(f"No countries for region {region}", flush=True)
            return []

        country_data_list = []
        for iso3 in tqdm.tqdm(iso3_codes, desc=f"Fetching {region} countries"):
            time.sleep(2)  # Rate-limit API calls

            # Find best year with data
            # For economics, check a few key indicators
            best_year = find_best_year(iso3, ["gdp", "population"], data_year)
            if best_year is None:
                print(f"  {iso3}: no data for required indicators, skipping", flush=True)
                continue

            indicators = fetch_country_indicators(iso3, best_year)
            if indicators is None:
                continue

            # Check minimum data availability
            has_gdp = indicators.get("gdp") is not None
            has_pop = indicators.get("population") is not None
            if not has_gdp and not has_pop:
                continue

            # Resolve Wikidata QID
            wikidata_id = resolve_country_wikidata_id(iso3)
            if wikidata_id is None:
                print(f"  {iso3}: no Wikidata QID found, skipping", flush=True)
                continue

            # Fetch multi-hop KG chains
            chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains)
            if not chains or len(chains) < 3:
                print(f"  {iso3}: insufficient KG chains ({len(chains) if chains else 0}), skipping",
                      flush=True)
                continue

            # Fetch Wikipedia articles for chain entities
            articles = fetch_wikipedia_for_entities(chains)
            print(f"  {iso3}: fetched {len(articles)} Wikipedia articles "
                  f"for {len(chains)} chains", flush=True)

            country_data_list.append({
                "iso3": iso3,
                "wikidata_id": wikidata_id,
                "label": indicators.get("name", iso3),
                "indicators": indicators,
                "chains": chains,
                "articles": articles,
            })

            if len(country_data_list) >= per_region:
                break

        # Save chains cache
        if country_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(country_data_list, f, indent=2)
            print(f"Cached {len(country_data_list)} country data for {region}", flush=True)

    print(f"Region {region}: {len(country_data_list)} countries available", flush=True)

    # Generate QA pairs from entities
    all_qa_pairs = []
    prev_questions = []
    used_templates = set()

    for cd in country_data_list:
        if len(all_qa_pairs) >= per_region:
            break

        qa_pairs = gen_economics_qa_from_entity(
            cd, agent_info,
            num_questions=questions_per_entity,
            prev_questions=prev_questions,
            used_templates=used_templates,
            articles=cd.get("articles", []),
            region=region,
        )

        for qa in qa_pairs:
            qa["region"] = region
        all_qa_pairs.extend(qa_pairs)
        print(f"  {cd['iso3']}: generated {len(qa_pairs)} QA pairs "
              f"(total: {len(all_qa_pairs)}/{per_region})", flush=True)

    # Comparative questions
    target_comp = max(per_region // 4, 2)
    comp_used = set()
    for i in range(0, len(country_data_list) - 1, 2):
        if len([q for q in all_qa_pairs if q.get("template_type") == "comparative"]) >= target_comp:
            break

        cd_a = country_data_list[i]
        cd_b = country_data_list[i + 1]

        clues_a = cd_a.get("clues", [])
        clues_b = cd_b.get("clues", [])
        if not clues_a:
            clues_a = []
        if not clues_b:
            clues_b = []

        # Retry a rejected pair a few times: comparative composition is
        # one-shot per call, so a single stochastic Phase-3 rejection would
        # otherwise discard a viable pair (single-entity gen gets
        # num_questions*3 attempts; give each comparative pair a few too).
        qa = None
        for _ in range(3):
            qa = gen_economics_qa_comparative(
                cd_a["iso3"], cd_a["indicators"], clues_a,
                cd_b["iso3"], cd_b["indicators"], clues_b,
                comp_used, prev_questions, region,
                chains_a=cd_a.get("chains", []),
                articles_a=cd_a.get("articles", []),
                chains_b=cd_b.get("chains", []),
                articles_b=cd_b.get("articles", []),
                agent_info=agent_info,
            )
            if qa:
                break
        if qa:
            qa["region"] = region
            all_qa_pairs.append(qa)
            prev_questions.append(qa["question"])
            print(f"  {cd_a['iso3']} vs {cd_b['iso3']}: comparative QA generated "
                  f"(total: {len(all_qa_pairs)})", flush=True)

    return all_qa_pairs


def run_economics_bench(args, agent_info):
    """Run the economics benchmark pipeline.

    Per region:
    1. Fetch countries + economic data + Wikipedia clues
    2. Generate QA pairs by template type
    3. V1 verification (closed-book)
    4. V2 verification (World Bank + Browser + Python tools)
    5. Merge all regions into final output
    """
    prefix = args.outfile_prefix1
    run_id = getattr(args, "run_id", None) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{prefix}__{run_id}"
    regions_arg = args.economics_regions
    per_region = args.economics_per_region
    data_year = args.economics_data_year
    num_chains = getattr(args, 'economics_num_chains', 20)
    questions_per_entity = getattr(args, 'economics_questions_per_entity', 5)
    v1_samples = args.economics_v1_samples
    v2_samples = args.economics_v2_samples
    v1_threshold = args.economics_v1_threshold
    v2_threshold = args.economics_v2_threshold

    bench_label = "economics_bench"

    # Parse regions
    if regions_arg:
        regions_to_process = [r.strip() for r in regions_arg.split(",")]
    else:
        regions_to_process = list(COUNTRY_UNIVERSE_REGIONS.keys())

    is_subset = regions_arg is not None and len(regions_to_process) < len(COUNTRY_UNIVERSE_REGIONS)

    print(f"=== ECONOMICS BENCH (run_id={run_id}) ===", flush=True)
    print(f"Regions: {regions_to_process}", flush=True)
    print(f"Target QAs per region: {per_region}", flush=True)
    print(f"Data year: {data_year}", flush=True)
    print(f"KG chains per country: {num_chains}", flush=True)
    print(f"Questions per entity: {questions_per_entity}", flush=True)
    print(f"V1: {v1_samples} samples, threshold < {v1_threshold:.0%}", flush=True)
    print(f"V2: {v2_samples} samples, threshold < {v2_threshold:.0%}", flush=True)
    print("=" * 50, flush=True)

    all_final = []
    summary = {}

    for region in regions_to_process:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Region: {region.upper()} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        output_dir = os.path.dirname(run_prefix) if "/" in run_prefix else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check full cache (V2)
        v2_cache = f"{run_prefix}__{region}.{bench_label}_v2.json"
        v2_filtered_cache = f"{run_prefix}__{region}.{bench_label}_v2_filtered.json"
        if os.path.exists(v2_cache) and os.path.exists(v2_filtered_cache):
            print(f"Found cached V2 results for {region}, loading...", flush=True)
            with open(v2_cache) as f:
                v2_all = json.load(f)
            with open(v2_filtered_cache) as f:
                v2_filtered = json.load(f)
            all_final.extend(v2_filtered)
            summary[region] = {
                "generated": "cached", "v1_filtered": "cached",
                "v2_total": len(v2_all), "v2_filtered": len(v2_filtered),
            }
            continue
        else:
            # Check V1 filtered cache
            v1_filtered_cache = f"{run_prefix}__{region}.{bench_label}_v1_filtered.json"
            if os.path.exists(v1_filtered_cache):
                print(f"Found cached V1 filtered for {region}, skipping to V2...", flush=True)
                with open(v1_filtered_cache) as f:
                    v1_filtered = json.load(f)
            else:
                # Check raw bench cache
                bench_cache = f"{run_prefix}__{region}.{bench_label}.json"
                if os.path.exists(bench_cache):
                    print(f"Found cached bench problems for {region}...", flush=True)
                    with open(bench_cache) as f:
                        qa_pairs = json.load(f)
                else:
                    # Step 1: Generate QA pairs
                    qa_pairs = _generate_region_qa_pairs(
                        region, per_region, data_year, num_chains,
                        questions_per_entity, agent_info, prefix, bench_label,
                    )

                    if qa_pairs:
                        with open(bench_cache, "w") as f:
                            json.dump(qa_pairs, f, indent=2)
                        print(f"Saved {len(qa_pairs)} raw QA pairs to {bench_cache}", flush=True)

                if not qa_pairs:
                    print(f"No QA pairs for region {region}", flush=True)
                    summary[region] = {"generated": 0, "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                    continue

                # Step 2: V1 verification
                v1_filtered, v1_all = verify_bench_v1(
                    qa_pairs, agent_info,
                    num_samples=v1_samples,
                    temperature=0.7,
                    threshold=v1_threshold,
                    outfile_prefix=run_prefix,
                    subarea=region,
                    bench_label=bench_label,
                    answer_checker=is_economics_answer_correct,
                    llm_judge=_llm_judge_econ_answer,
                    verification_prompt=ECONOMICS_V1_PROMPT,
                )

            if not v1_filtered:
                print(f"No V1-filtered QA pairs for {region}", flush=True)
                summary[region] = {"generated": len(qa_pairs) if 'qa_pairs' in dir() else "cached",
                                   "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                continue

            # Step 3: V2 verification
            v2_filtered, v2_all = verify_economics_v2(
                v1_filtered,
                num_samples=v2_samples,
                temperature=1.0,
                max_iterations=200,
                threshold=v2_threshold,
                outfile_prefix=run_prefix,
                subarea=region,
                bench_label=bench_label,
            )

        all_final.extend(v2_filtered)
        summary[region] = {
            "generated": len(v1_all) if 'v1_all' in dir() else "cached",
            "v1_filtered": len(v1_filtered) if 'v1_filtered' in dir() else "cached",
            "v2_total": len(v2_all),
            "v2_filtered": len(v2_filtered),
        }

    # Skip final merge when processing a subset (worker job)
    if is_subset:
        print(f"\nSubset mode: processed {regions_to_process}. "
              f"Merge will happen in the merge job.", flush=True)
        _print_summary(summary)
        return all_final

    # Final merge
    final_path = f"{run_prefix}.{bench_label}_final.json"
    if all_final:
        # Deduplicate
        seen_keys = set()
        deduped = []
        for qa in all_final:
            key = (qa.get("template_id", ""), qa.get("country_name_a", ""),
                   qa.get("country_name_b", ""), qa.get("gold_answer", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(qa)
        if len(deduped) < len(all_final):
            print(f"\nDeduplication: {len(all_final)} → {len(deduped)} "
                  f"({len(all_final) - len(deduped)} duplicates removed)", flush=True)
        all_final = deduped

        # Optional diversity filter
        if _HAS_DIVERSITY and len(all_final) > 10:
            print(f"\nApplying diversity filter to {len(all_final)} questions...", flush=True)
            all_final = diversity_filter(all_final, target_count=len(all_final), min_distance=0.03)

        with open(final_path, "w") as f:
            json.dump(all_final, f, indent=2)
        print(f"\nSaved {len(all_final)} final QA pairs to {final_path}", flush=True)

        if _HAS_DIVERSITY:
            diversity_report(all_final, text_keys=["question"])

    _print_summary(summary)
    return all_final


def _print_summary(summary):
    """Print pipeline summary table."""
    print(f"\n{'='*60}", flush=True)
    print("ECONOMICS BENCH SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for region, stats in summary.items():
        print(f"  {region:28s}: generated={stats.get('generated','?'):>6} "
              f"v1_kept={stats.get('v1_filtered',0):>4} "
              f"v2_kept={stats.get('v2_filtered',0):>4}", flush=True)
    total = sum(s.get("v2_filtered", 0) for s in summary.values())
    print(f"  {'TOTAL':28s}: {total:>4} final questions", flush=True)
    print(f"{'='*60}\n", flush=True)


# ===========================================================================
# CLI __main__
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="economics_drbencher",
        description="Economics Benchmark: World Bank WDI + Wikipedia entity ID",
    )

    # Model args
    parser.add_argument("--agent_modelname", default="openai/gpt-oss-120b")
    parser.add_argument("--use_harmony", type=str, default="no")
    parser.add_argument("--harmony_model_path", type=str, default=None)
    parser.add_argument("--use_vllm_serve", type=str, default="no",
                        help="Use external vllm serve endpoint (yes/no)")
    parser.add_argument("--vllm_serve_url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vllm serve endpoint")
    parser.add_argument("--vllm_serve_model", type=str, default=None,
                        help="Model name for vllm serve (required when use_vllm_serve=yes)")
    parser.add_argument("--use_helm", type=str, default="yes")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Output
    parser.add_argument("--outfile_prefix1", type=str, default="output/economics/bench")

    # Exp mode
    parser.add_argument("--exp_mode", type=str, default="economics_bench")

    # Run ID
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: auto YYYYMMDD_HHMMSS)")

    # Economics-specific
    parser.add_argument("--economics_regions", type=str, default=None,
                        help="Comma-separated regions (default: all). E.g. 'north_america,europe_central_asia'")
    parser.add_argument("--economics_per_region", type=int, default=40,
                        help="Target QA pairs per region")
    parser.add_argument("--economics_data_year", type=int, default=2022,
                        help="Data year for economic indicators (default: 2022)")
    parser.add_argument("--economics_v1_samples", type=int, default=10,
                        help="V1 sampling attempts per question")
    parser.add_argument("--economics_v2_samples", type=int, default=10,
                        help="V2 (WorldBank+Browser+Python) sampling attempts per question")
    parser.add_argument("--economics_v1_threshold", type=float, default=0.5,
                        help="V1 accuracy ceiling — keep below this")
    parser.add_argument("--economics_v2_threshold", type=float, default=0.5,
                        help="V2 accuracy ceiling — keep below this")
    parser.add_argument("--economics_num_chains", type=int, default=20,
                        help="KG chains per country for entity grounding (default: 20)")
    parser.add_argument("--economics_questions_per_entity", type=int, default=5,
                        help="Questions to generate per country (default: 5)")

    args = parser.parse_args()

    # Initialize Harmony generator
    if args.use_harmony.lower() == "yes":
        import torch
        model_path = args.harmony_model_path or args.agent_modelname
        tp_size = args.tensor_parallel_size or torch.cuda.device_count()
        harmony_gen = HarmonyVLLMGenerator(
            model_path=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        set_harmony_generator(harmony_gen)
    elif args.use_vllm_serve.lower() == "yes":
        from .openai_api_generator import OpenAIAPIGenerator
        if not args.vllm_serve_model:
            raise ValueError("--vllm_serve_model is required when --use_vllm_serve=yes")
        gen = OpenAIAPIGenerator(
            base_url=args.vllm_serve_url,
            model_name=args.vllm_serve_model,
        )
        set_harmony_generator(gen)

    if args.use_harmony.lower() == "yes" or args.use_vllm_serve.lower() == "yes":
        print("Generator mode: using generator for all inference", flush=True)
        agent_info = (None, None, None)
    elif args.use_helm == "yes":
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        agent_info = (agent_lm, agent_tokenizer, agent_client)
    else:
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        agent_info = (agent_lm, agent_tokenizer, agent_client)

    # Run benchmark
    if args.exp_mode == "economics_bench":
        run_economics_bench(args, agent_info)
        # The in-process vLLM engine spawns worker subprocesses that outlive the
        # bench; shut the engine down and terminate the process group so the job
        # exits cleanly instead of hanging until it is killed by hand.
        shutdown_and_exit(0)
    else:
        print(f"Unknown exp_mode: {args.exp_mode}", flush=True)
        sys.exit(1)
