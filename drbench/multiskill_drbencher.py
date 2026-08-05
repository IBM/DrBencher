# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Skill Benchmark: Combines hard multi-hop entity identification (Wikidata KG)
with hard domain-specific reasoning (scientific, mathematical).

Creates questions in a 2D difficulty space:
  - Retrieval difficulty: multi-hop clues to identify entity from KG chains
  - Reasoning difficulty: domain-specific computation on entity's quantitative properties

Prototype: 2 reasoning domains
  1. Quantitative Modeling — apply math formulas to KG numerical data
  2. Scientific Inference — apply scientific principles to KG entity data

Run via:  python -m drbench.multiskill_drbencher --exp_mode multiskill_bench ...
"""

import argparse
import copy
import datetime
import json
import math
import os
import random
import re
import subprocess
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
    search_wikidata_entities, get_entity_data, get_entity_label,
    sparql_query, fetch_multihop_triples, fetch_wikipedia_for_entities,
)
from .harmony_vllm import HarmonyVLLMGenerator

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

from .multiskill_utils import run_phase3_validation, filter_poisoned_facts, compute_cci_fields

from .multiskill_template import (
    _UNIT_NORMALIZATION,
    REASONING_DOMAINS,
    PROPERTY_DOMAIN_COMPATIBILITY,
    REFERENCE_CITIES, REFERENCE_AREAS, REFERENCE_RIVERS,
    REFERENCE_HEIGHTS, REFERENCE_ELEVATIONS, REFERENCE_MASSES,
    MULTISKILL_THEMES,
    select_reasoning_context,
    _fill_template_parameters,
    _execute_computation_code,
    _build_gold_chain,
)

# Reuse fact extraction / grounding from wikidata pipeline
from .wikidata_harmony import (
    validate_triple_in_chains,
    validate_fact_references,
    validate_answer_in_facts,
    format_extracted_facts,
    parse_used_facts,
    filter_chains_by_grounding,
    verify_facts_against_grounding,
    _markup_sentences,
    _paragraph_mentions_entity,
    _sentence_mentions_patterns,
    set_harmony_generator as wikidata_set_harmony_generator,
    WIKIDATA_FACT_EXTRACTION_DEVELOPER,
    WIKIDATA_FACT_EXTRACTION_PROMPT,
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

# Diversity filter (optional)
try:
    from .diversity import diversity_filter, diversity_report
    _HAS_DIVERSITY = True
except ImportError:
    _HAS_DIVERSITY = False


# ===========================================================================
# Quantitative Wikidata Properties
# ===========================================================================

QUANTITATIVE_WIKIDATA_PROPERTIES = {
    "P1082": "population",
    "P2046": "area",
    "P2131": "nominal GDP",
    "P2132": "nominal GDP per capita",
    "P2044": "elevation above sea level",
    "P2067": "mass",
    "P625":  "coordinate location",
    "P2120": "radius",
    "P2054": "density",
    "P1114": "quantity",
    "P2043": "length",
    "P2048": "height",
    "P1101": "floors above ground",
    "P2660": "topographic prominence",
    "P4511": "vertical depth",
}

# Property labels for prompts
PROPERTY_LABELS = {
    "P1082": "population",
    "P2046": "area (km²)",
    "P2131": "nominal GDP (USD)",
    "P2132": "nominal GDP per capita (USD)",
    "P2044": "elevation (m)",
    "P2067": "mass (kg)",
    "P625":  "coordinates (lat, lon)",
    "P2120": "radius (km)",
    "P2054": "density (kg/m³)",
    "P1114": "quantity",
    "P2043": "length (km)",
    "P2048": "height (m)",
    "P1101": "floors above ground",
    "P2660": "prominence (m)",
    "P4511": "depth (m)",
}



# ===========================================================================
# Prompts
# ===========================================================================

MULTISKILL_QA_DEVELOPER = (
    "You are a quiz-show writer who creates multi-skill research questions. "
    "Each question requires first identifying an entity from clues, then applying "
    "domain-specific reasoning to its quantitative properties."
)

MULTISKILL_QA_PROMPT = """Compose a multi-skill research question that:
1. Uses clue facts to describe an unnamed entity (readers must figure out what it is)
2. Then asks a quantitative reasoning question requiring the solver to look up the entity's properties

CLUE FACTS (about the unnamed seed entity — do NOT name it):
{facts_text}

REASONING TASK:
Domain: {domain_label}
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

Entity's quantitative properties (for your reference — do NOT reveal these values in the question):
{quant_data_text}

{external_params_text}

--- STYLE GUIDANCE ---
- Write 2–4 sentences total.
- First 1–2 sentences: describe the entity using 3 clue facts from different chains, without naming it.
- Last 1–2 sentences: pose the reasoning question.
- CRITICAL: Do NOT reveal ANY quantitative values in the question — not the entity's properties (population, area, elevation, GDP, mass, radius, etc.) and not the coordinates of any location (including reference cities). The solver must look up ALL numerical data themselves.
- You MAY mention the name of a reference city (e.g., "New York City") but do NOT include its coordinates.
- You MAY include external/hypothetical parameters (e.g., growth rate, time period) that are not real-world properties of any entity.
- CRITICAL — no free answers: The question must force the solver to DERIVE every entity needed for the computation. Do NOT name any specific city, place, landmark, or entity from the clue facts anywhere in the question if the solver could use that name to directly look up computation inputs (coordinates, population, etc.). Instead, reference them indirectly so the solver must identify the seed entity first, then look up the needed value themselves.
  BAD:  "…its most populous city is Doha. What is the distance between Doha and São Paulo?" (Doha is given — solver skips identification)
  BAD:  "…its most populous city is Doha. What is the distance between this city and São Paulo?" (Doha is still given — solver already knows it)
  GOOD: "…it borders Saudi Arabia. What is the distance between this nation's most populous city and São Paulo?" (solver must identify the nation, THEN look up its most populous city)
  External reference cities (e.g., São Paulo) that are NOT derived from the clue chain MAY be named.
- Use clear, unambiguous language. Avoid obscure words like "namesake" — prefer plain phrasing.
- CRITICAL — no paraphrasing: Use the EXACT terminology from the source facts. Do NOT substitute synonyms or approximate terms. For example, if a fact says "designed by", do NOT write "created by". Stick to the source wording.
- When referring to demographic or statistical data, ALWAYS specify the exact year (e.g., "its 2023 population" or "according to its 2020 census data"). NEVER use vague phrases like "latest", "current", "most recent", or "up-to-date".
- Sound natural and conversational, not formulaic.
- Do NOT use awkward meta-phrases like "is described", "is the subject", "is being discussed", or "is considered".
- Do NOT mention "Wikidata", "KG", "entity", or "property".
- Use natural references like "this country", "there", "its", "the same lake".

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your multi-skill question>
Used_Facts: <comma-separated fact_ids used, e.g. C1_F1, C2_F1, C4_F1>
Reasoning: <brief chain: clues identify entity → look up properties → computation gives answer>

REQUIREMENTS:
- Select 3 or more clue facts from 3 or more different chains. Each fact must convey distinct information — do NOT repeat the same relationship (e.g., do not use three facts that all mention the same river).
- Do NOT name the seed entity in the question
- Do NOT name any specific city, place, or landmark from the clue facts that the solver could use to directly look up computation inputs — reference them indirectly (e.g., "this nation's capital", "its most populous city")
- Do NOT include the entity's quantitative property values in the question
- Every claim must come VERBATIM from the listed facts — do NOT add background knowledge, etymologies, or inferred relationships (e.g., do NOT add "named after X" unless a source fact explicitly states it)
- The question MUST have exactly ONE unambiguous answer
- If the question involves population, area, GDP, or other time-varying data, you MUST specify the exact year (e.g., "its 2023 population"). NEVER write "latest", "most recent", "current", or similar vague time references."""

MULTISKILL_V1_PROMPT = """Solve this multi-skill research question. It requires:
1. Identifying an entity from descriptive clues
2. Applying quantitative reasoning to compute the answer

Provide ONLY the final numerical answer (with appropriate units if needed).
Do not show your work.

Question: {question}

Answer:"""

MULTISKILL_V2_DEVELOPER = """You are an expert research assistant solving multi-skill questions.
These questions require:
1. Identifying a real-world entity from descriptive clues
2. Looking up quantitative properties of that entity
3. Applying mathematical or scientific formulas to reach the answer

You have TWO tools available:
- **Browser tool**: Search and browse Wikidata and Wikipedia to identify entities and look up their properties (population, area, elevation, coordinates, GDP, etc.)
- **Python tool**: Execute Python code for calculations (math, numpy, scipy available)

Recommended approach:
1. Use the browser to search for the entity matching the clues
2. Use the browser to look up the entity's quantitative properties
3. Use Python to perform the computation and get the final answer

Give your final answer as a single number (with units if appropriate) on the last line."""

# ===========================================================================
# Data Helpers
# ===========================================================================

def fetch_quantitative_properties(entity_id, properties=None):
    """Fetch quantitative (numeric) properties from Wikidata for an entity.

    Wraps get_entity_data() and extracts quantity-type claims as floats.
    Also extracts point-in-time qualifiers (P585) when available.

    Args:
        entity_id: Wikidata entity ID (e.g., "Q142" for France)
        properties: Optional list of property IDs to fetch. Defaults to
                    QUANTITATIVE_WIKIDATA_PROPERTIES keys.

    Returns:
        Dict mapping property ID to {'amount': float, 'unit': str, 'label': str,
        'year': int or None}
        Only includes properties that have quantity-type values.
    """
    if properties is None:
        properties = list(QUANTITATIVE_WIKIDATA_PROPERTIES.keys())

    # Fetch raw entity data with qualifiers
    raw_entity = _fetch_entity_raw(entity_id)
    if not raw_entity:
        # Fallback to get_entity_data (no qualifier support)
        return _fetch_quant_props_simple(entity_id, properties)

    result = {}
    raw_claims = raw_entity.get("claims", {})

    for prop_id in properties:
        if prop_id not in raw_claims:
            continue

        if prop_id == "P625":
            # Coordinates: take the first valid one
            for claim in raw_claims[prop_id]:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                val = datavalue.get("value", {})
                if datavalue.get("type") == "globecoordinate":
                    try:
                        lat = float(val.get("latitude", 0))
                        lon = float(val.get("longitude", 0))
                        result[prop_id] = {
                            "amount": (lat, lon),
                            "unit": "degrees",
                            "label": "coordinate location",
                            "year": None,
                        }
                        break
                    except (ValueError, TypeError):
                        continue
            continue

        # For quantity properties: prefer claims with year qualifiers,
        # and among those pick the most recent year
        best = None
        for claim in raw_claims[prop_id]:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            dtype = datavalue.get("type", "")
            val = datavalue.get("value", {})

            if dtype != "quantity":
                continue
            try:
                amount_str = val.get("amount", "0")
                amount = float(amount_str.lstrip("+"))
                unit_uri = val.get("unit", "1")
                if "entity/" in unit_uri:
                    unit_id = unit_uri.split("entity/")[-1]
                    unit_label = get_entity_label(unit_id)
                else:
                    unit_label = "dimensionless"

                year = _extract_year_qualifier(claim)

                candidate = {
                    "amount": amount,
                    "unit": unit_label,
                    "label": QUANTITATIVE_WIKIDATA_PROPERTIES.get(prop_id, prop_id),
                    "year": year,
                }
                # Prefer dated claims; among dated, prefer most recent
                if best is None:
                    best = candidate
                elif year and (not best["year"] or year > best["year"]):
                    best = candidate
            except (ValueError, TypeError):
                continue

        if best:
            result[prop_id] = best

    return result


def _fetch_entity_raw(entity_id):
    """Fetch raw entity JSON from Wikidata API (with qualifiers intact)."""
    url = (
        f"https://www.wikidata.org/w/api.php?action=wbgetentities"
        f"&ids={entity_id}&props=claims&format=json"
    )
    try:
        response = requests.get(url, headers={"User-Agent": "DrBencher/1.0"}, timeout=30)
        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("entities", {}).get(entity_id, {})
    except Exception:
        return None


def _extract_year_qualifier(claim):
    """Extract year from P585 (point in time) qualifier of a claim."""
    qualifiers = claim.get("qualifiers", {})
    for q in qualifiers.get("P585", []):
        datavalue = q.get("datavalue", {})
        if datavalue.get("type") == "time":
            time_str = datavalue.get("value", {}).get("time", "")
            # Format: +2023-01-01T00:00:00Z
            match = re.match(r'[+-]?(\d{4})', time_str)
            if match:
                return int(match.group(1))
    return None


def _fetch_quant_props_simple(entity_id, properties):
    """Fallback: fetch quantitative properties without qualifier support."""
    entity_data = get_entity_data(entity_id)
    if not entity_data:
        return {}

    result = {}
    claims = entity_data.get("claims", {})

    for prop_id in properties:
        if prop_id not in claims:
            continue

        for val in claims[prop_id]:
            if val.get("type") == "quantity":
                try:
                    amount_str = val.get("amount", "0")
                    amount = float(amount_str.lstrip("+"))
                    unit_uri = val.get("unit", "1")
                    if "entity/" in unit_uri:
                        unit_id = unit_uri.split("entity/")[-1]
                        unit_label = get_entity_label(unit_id)
                    else:
                        unit_label = "dimensionless"

                    result[prop_id] = {
                        "amount": amount,
                        "unit": unit_label,
                        "label": QUANTITATIVE_WIKIDATA_PROPERTIES.get(prop_id, prop_id),
                        "year": None,
                    }
                    break
                except (ValueError, TypeError):
                    continue

            elif val.get("type") == "globecoordinate" and prop_id == "P625":
                raw = val.get("raw", {})
                try:
                    lat = float(raw.get("latitude", 0))
                    lon = float(raw.get("longitude", 0))
                    result[prop_id] = {
                        "amount": (lat, lon),
                        "unit": "degrees",
                        "label": "coordinate location",
                        "year": None,
                    }
                    break
                except (ValueError, TypeError):
                    continue

    return result


def _parse_coordinate_from_claims(claims):
    """Extract coordinate location from claims if present."""
    if "P625" not in claims:
        return None
    for val in claims["P625"]:
        if val.get("type") == "globecoordinate":
            raw = val.get("raw", {})
            try:
                return (float(raw.get("latitude", 0)), float(raw.get("longitude", 0)))
            except (ValueError, TypeError):
                continue
    return None



# ===========================================================================
# Phase 2: Question Composition (LLM)
# ===========================================================================

def compose_multiskill_question(clue_facts, reasoning_context, seed_label, prev_questions):
    """Compose a multi-skill question from clue facts + reasoning context.

    Args:
        clue_facts: List of fact dicts from Phase 1 (fact extraction)
        reasoning_context: Dict from select_reasoning_context()
        seed_label: Label of the seed entity (for internal use, not revealed)
        prev_questions: List of previously generated question strings

    Returns:
        Dict with question, used_facts, reasoning, or None.
    """
    facts_text = format_extracted_facts(clue_facts)

    prev_q_text = "\n".join(f"- {q}" for q in prev_questions) if prev_questions else "None"

    # Build external params text (hypothetical parameters the question MAY include)
    external_params = reasoning_context.get("external_params", {})
    if external_params:
        ext_lines = ["External/hypothetical parameters (you MAY include these in the question):"]
        for k, v in external_params.items():
            ext_lines.append(f"- {k}: {v}")
        external_params_text = "\n".join(ext_lines)
    else:
        external_params_text = ""

    prompt = MULTISKILL_QA_PROMPT.format(
        facts_text=facts_text,
        domain_label=reasoning_context["domain_label"],
        template_label=reasoning_context["template_label"],
        question_hint=reasoning_context["question_hint"],
        answer_unit=reasoning_context["answer_unit"],
        gold_answer=reasoning_context["gold_answer"],
        quant_data_text=reasoning_context["quant_data_text"],
        external_params_text=external_params_text,
        previous_questions=prev_q_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt,
            temperature=0.7,
            max_tokens=8196,
            developer_content=MULTISKILL_QA_DEVELOPER,
        )
    except Exception as e:
        print(f"  Question composition failed: {e}", flush=True)
        return None

    if not response:
        return None

    # Parse response
    question_match = re.search(r'Question:\s*(.+?)(?:\nUsed_Facts:|\nReasoning:|\Z)', response, re.DOTALL)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\Z)', response, re.DOTALL)
    used_facts = parse_used_facts(response)

    if not question_match:
        return None

    question = question_match.group(1).strip()

    # Strip fact ID annotations that the LLM sometimes copies from the prompt
    # e.g. "(C8_F1)", "[C3_F2]", "(A_C1_F1)", "(B_C2_F1)"
    question = re.sub(r'\s*[\(\[]\s*[AB]?_?C\d+_F\d+\s*[\)\]]', '', question)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    return {
        "question": question,
        "used_facts": used_facts,
        "reasoning": reasoning,
    }


# ===========================================================================
# Answer Verification
# ===========================================================================

def is_multiskill_answer_correct(predicted, gold, tolerance=0.05):
    """Check if predicted multi-skill answer matches gold answer.

    Layered comparison:
    1. Exact match (after normalization)
    2. Numeric comparison with relative tolerance

    Args:
        predicted: Predicted answer string
        gold: Gold answer string
        tolerance: Relative tolerance for numeric comparison (default 5%)

    Returns:
        True if answer is correct within tolerance.
    """
    if not predicted or not predicted.strip():
        return False

    # Normalize
    pred_clean = _normalize_numeric_text(predicted)
    gold_clean = _normalize_numeric_text(gold)

    # 1. Exact match
    if pred_clean == gold_clean:
        return True

    # 2. Numeric comparison
    pred_num = _try_parse_number(pred_clean)
    gold_num = _try_parse_number(gold_clean)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0 and pred_num == 0:
            return True
        if gold_num != 0:
            rel_error = abs(pred_num - gold_num) / max(abs(gold_num), abs(pred_num))
            if rel_error <= tolerance:
                return True

    return False


def _normalize_numeric_text(text):
    """Normalize text for numeric comparison."""
    text = text.strip()
    # Normalize Unicode whitespace (narrow no-break space, non-breaking space, etc.)
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Remove common units and suffixes (word-boundary aware to avoid stripping
    # 'm' from 'million' etc.)
    # Process multi-char suffixes first, then single-char with boundary check
    for suffix in ["°C", "°F", "kPa", "Pa", "km²", "kg/m³", "km", "m/s²", "m/s",
                    "people/km²", "people", "USD", "$", "years", "hours",
                    "seconds", "joules", "metres", "times", "ratio",
                    "milliseconds", "N"]:
        if len(suffix) > 1:
            text = text.replace(suffix, "").strip()
        else:
            # Single-char: only strip if at end or followed by whitespace
            text = re.sub(rf'\s*{re.escape(suffix)}\s*$', '', text).strip()
    # Remove commas in numbers
    text = text.replace(",", "")
    # Remove leading/trailing whitespace and quotes
    text = text.strip().strip("'\"")
    return text


def _try_parse_number(text):
    """Try to parse a number from text. Returns float or None."""
    text = text.strip()
    # Normalize Unicode whitespace before parsing
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Handle scientific notation variants: "2.4 x 10^-4", "2.4×10^-4", etc.
    text = re.sub(r'\s*[×x]\s*10\^', 'e', text)
    # Handle common multipliers
    multipliers = {
        "million": 1e6, "billion": 1e9, "trillion": 1e12,
        "thousand": 1e3, "hundred": 1e2,
        "M": 1e6, "B": 1e9, "T": 1e12, "K": 1e3,
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
        # Try extracting first number
        match = re.search(r'-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None


def _llm_judge_multiskill_answer(predicted, gold, question):
    """Use LLM to judge if predicted answer matches gold for multi-skill questions."""
    prompt = f"""Compare these two answers to the same quantitative question.
The gold answer is the ground truth computed from verified data.
The predicted answer may use different units, rounding, or phrasing.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Are these answers equivalent (same value within ~5% tolerance, possibly different units or rounding)?
Respond with exactly YES or NO."""

    try:
        response = gen_from_prompt_harmony(prompt, temperature=0.0, max_tokens=10)
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


def is_entity_match(predicted, gold, **kwargs):
    """Check if predicted entity matches gold entity (case-insensitive substring).

    Args:
        predicted: Predicted entity name string.
        gold: Gold entity name string.

    Returns:
        True if one is a substring of the other (case-insensitive).
    """
    if not predicted or not predicted.strip():
        return False
    pred = predicted.strip().strip('"').strip("'").strip(".").lower()
    gold_lower = gold.strip().lower()
    return gold_lower in pred or pred in gold_lower


def _llm_judge_entity_match(predicted, gold, question):
    """Use LLM to judge if predicted entity matches gold entity."""
    prompt = f"""Do these two names refer to the same real-world entity?

Name A: {predicted}
Name B: {gold}

Context (the question that was asked):
{question}

Respond with exactly YES or NO."""

    try:
        response = gen_from_prompt_harmony(prompt, temperature=0.0, max_tokens=10)
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


# ===========================================================================
# V2 Verification: Browser + Python Tools
# ===========================================================================

def verify_multiskill_v2(qa_pairs, num_samples=10, temperature=1.0,
                           max_iterations=200, threshold=0.5,
                           outfile_prefix=None, subarea="",
                           bench_label="multiskill_bench"):
    """V2 verification using both Wikimedia browser and Python tools.

    For each QA pair, runs an agentic loop where the model can:
    - Use the browser tool to search Wikidata/Wikipedia and look up entity properties
    - Use the Python tool to execute code for computations

    Args:
        qa_pairs: List of V1-filtered QA dicts.
        num_samples: Agentic samples per question.
        temperature: Sampling temperature.
        max_iterations: Max agentic loop iterations per sample.
        threshold: Keep questions with v2 accuracy < threshold.
        outfile_prefix: Cache prefix for output files.
        subarea: Sub-area name for logging.
        bench_label: Label for cache files.

    Returns:
        (filtered_pairs, all_pairs) — filtered has v2_accuracy < threshold.
    """
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
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available for multiskill V2")
    if not _HAS_PYTHON_TOOL:
        raise RuntimeError("HybridPythonTool not available for multiskill V2")

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== MULTISKILL V2 VERIFICATION ({subarea}) ===", flush=True)
    print(f"Tools: Wikimedia browser + Python", flush=True)
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
            qid_str = f"{bench_label}_v2_{subarea}_{idx}_{i}"
            try:
                # Initialize both tools
                backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
                browser_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
                python_tool = HybridPythonTool(timeout=60)
                python_tool.set_qid(qid_str)

                system_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.HIGH)
                    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                    .with_tools(browser_tool.tool_config)
                    .with_tools(python_tool.tool_config)
                )

                messages = [
                    Message.from_role_and_content(Role.SYSTEM, system_content),
                    Message.from_role_and_content(Role.DEVELOPER, MULTISKILL_V2_DEVELOPER),
                    Message.from_role_and_content(Role.USER, f"Question: {qa['question']}"),
                ]

                # Multi-tool handler: route by recipient prefix
                tool_call_counter = [0]

                async def _tool_handler(msg, _browser=browser_tool, _python=python_tool,
                                        _counter=tool_call_counter):
                    _counter[0] += 1
                    recipient = str(getattr(msg, 'recipient', ''))
                    results = []
                    if recipient.startswith("browser."):
                        async for m in _browser.process(msg):
                            results.append(m)
                    elif recipient.startswith("python"):
                        async for m in _python.process(msg):
                            results.append(m)
                    else:
                        # Unknown tool — return error
                        error = Message.from_role_and_content(
                            Role.SYSTEM, f"Unknown tool: {recipient}"
                        )
                        results.append(error)
                    return results

                result_messages = generator.generate_agentic_response_sync(
                    messages,
                    _tool_handler,
                    tool_prefix=("browser.", "python"),
                    max_iterations=max_iterations,
                    temperature=temperature,
                )

                iteration_count = len(result_messages) - len(messages)

                answer = _extract_multiskill_answer(result_messages)
                answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
                answer = ' '.join(answer.split())
                correct = bool(is_multiskill_answer_correct(answer, qa['gold_answer']) or
                               _llm_judge_multiskill_answer(answer, qa['gold_answer'], qa['question']))
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
        # Preserve errors so they're visible in the output JSON (not just stdout)
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


def _extract_multiskill_answer(messages):
    """Extract the final answer from an agentic response with browser + Python tools."""
    # Look for the last assistant message
    for msg in reversed(messages):
        if hasattr(msg, 'author') and msg.author.role == Role.ASSISTANT:
            from .multiskill_utils import unwrap_content
            content = unwrap_content(getattr(msg, 'content', ''))
            if not content.strip():
                continue
            # Try to extract the last line as the answer
            lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
            if lines:
                last = lines[-1]
                # Strip common prefixes
                for prefix in ["Answer:", "Final answer:", "The answer is", "Result:"]:
                    if last.lower().startswith(prefix.lower()):
                        last = last[len(prefix):].strip()
                return last
    return ""


# ===========================================================================
# Entity Fetching for Multi-Skill
# ===========================================================================

def fetch_multiskill_entities(theme, target_count=20):
    """Fetch entities suitable for multi-skill questions from Wikidata.

    Searches for entities of a given type that have quantitative properties.

    Args:
        theme: Theme key from MULTISKILL_THEMES
        target_count: Number of entities to return

    Returns:
        List of entity dicts with id, label, quant_props, chains
    """
    theme_types = MULTISKILL_THEMES.get(theme, ["Q6256"])

    # SPARQL query for entities of this type with quantitative properties
    type_filter = " ".join(f"{{ ?item wdt:P31/wdt:P279? wd:{t} }}" for t in theme_types)
    if len(theme_types) > 1:
        type_filter = "{ " + " UNION ".join(
            f"{{ ?item wdt:P31/wdt:P279? wd:{t} }}" for t in theme_types
        ) + " }"

    # Fetch entities that have at least one quantitative property
    quant_props_filter = " UNION ".join(
        f"{{ ?item wdt:{p} ?v_{p} }}" for p in [
            "P1082", "P2046", "P2131", "P2044", "P2067",
            "P2048", "P2043", "P625", "P2120",
        ]
    )

    query = f"""
    SELECT DISTINCT ?item ?itemLabel WHERE {{
      {type_filter}
      {{ {quant_props_filter} }}
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 20)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY MD5(STR(?item))
    LIMIT {target_count * 3}
    """

    results = sparql_query(query)
    if not results:
        print(f"No entities found for theme '{theme}'", flush=True)
        return []

    entities = []
    for row in results:
        eid = row.get("item", {}).get("value", "").split("/")[-1]
        label = row.get("itemLabel", {}).get("value", eid)
        if not eid.startswith("Q"):
            continue
        # Skip entities where Wikidata returned the Q-ID as the label
        # (happens when no English label exists)
        if re.match(r'^Q\d+$', label):
            label = get_entity_label(eid)
            if re.match(r'^Q\d+$', label):
                print(f"  Skipping {eid}: no human-readable label available", flush=True)
                continue
        entities.append({"id": eid, "label": label})

    random.shuffle(entities)
    print(f"Found {len(entities)} candidate entities for theme '{theme}'", flush=True)
    return entities[:target_count]


# ===========================================================================
# Grounding Articles for Human Verification
# ===========================================================================

def build_multiskill_grounding_articles(entity_label, gold_answer, answer_unit,
                                          extracted_facts, chains, articles):
    """Build trimmed, marked-up grounding articles for a multiskill QA pair.

    Single target entity (unlike logic bench's multi-entity). Collects
    Wikipedia articles, scores paragraphs by entity/property relevance,
    and applies <<Q_RELEVANT>>, <<A_RELEVANT>>, <<QA_RELEVANT>> markup.

    Args:
        entity_label: The bridging entity name (answer entity).
        gold_answer: The computed gold answer string.
        answer_unit: Unit string for the answer (e.g. "km", "kg").
        extracted_facts: List of fact dicts with entity/property/value fields.
        chains: List of KG chain dicts.
        articles: List of Wikipedia article dicts.

    Returns:
        List of article dicts with 'qid', 'title', 'paragraph',
        'chain_indices', 'original_length', 'trimmed_length'.
    """
    MAX_CHARS_PER_ARTICLE = 5000

    # -- Build answer patterns (green): bridging entity + gold answer --------
    answer_raw = set()
    if entity_label:
        answer_raw.add(entity_label)
    if gold_answer:
        answer_raw.add(str(gold_answer))
        try:
            numeric = float(str(gold_answer).replace(",", ""))
            if numeric == int(numeric):
                answer_raw.add(f"{int(numeric):,}")
        except (ValueError, OverflowError):
            pass
    if answer_unit:
        answer_raw.add(answer_unit)

    # -- Build question patterns (blue): other entities + values + keywords --
    question_raw = set()
    relationship_keywords = set()
    for fact in extracted_facts:
        if not isinstance(fact, dict):
            continue
        # Entity names (excluding the bridging entity)
        for key in ('entity', 'value'):
            val = fact.get(key, '')
            if val and val != entity_label:
                question_raw.add(val)
        # Property keywords
        prop = fact.get('property', '')
        if prop:
            prop_lower = prop.lower().strip()
            question_raw.add(prop_lower)
            relationship_keywords.add(prop_lower)
            for word in prop_lower.split():
                if len(word) > 3:
                    question_raw.add(word)
                    relationship_keywords.add(word)

    def _make_patterns(raw_set):
        patterns = set()
        for e in raw_set:
            lowered = str(e).lower().strip()
            if lowered:
                patterns.add(lowered)
                normalized = re.sub(r'[.,\-]', ' ', lowered).strip()
                normalized = ' '.join(normalized.split())
                if normalized and normalized != lowered:
                    patterns.add(normalized)
        return patterns

    answer_patterns = _make_patterns(answer_raw)
    question_patterns = _make_patterns(question_raw) - answer_patterns
    entity_patterns = question_patterns | answer_patterns

    if not entity_patterns:
        return []

    # -- Filter to articles about named entities in the question -------------
    relevant_labels = {entity_label.lower()} if entity_label else set()
    for fact in extracted_facts:
        if not isinstance(fact, dict):
            continue
        for key in ('entity', 'value'):
            val = fact.get(key, '')
            if val and len(val) > 2 and not val.replace(",", "").replace(".", "").isdigit():
                relevant_labels.add(val.lower())

    # -- Score, markup, trim each article ------------------------------------
    result = []
    seen_qids = set()

    for article in articles:
        qid = article.get('qid', '')
        title = article.get('title', '')
        key = qid or title
        if key and key in seen_qids:
            continue
        # Keep only if article title matches a named entity in the question
        if not title or title.lower() not in relevant_labels:
            continue
        if key:
            seen_qids.add(key)

        original_text = article.get('paragraph', '')
        original_length = len(original_text)
        title = article.get('title', 'Untitled')

        paragraphs = original_text.split('\n')

        scored_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if _paragraph_mentions_entity(para, entity_patterns):
                para_lower = para.lower()
                mention_count = sum(
                    1 for pat in entity_patterns
                    if (re.search(r'\b' + re.escape(pat) + r'\b', para_lower)
                        if len(pat) <= 3 else pat in para_lower)
                )
                if relationship_keywords:
                    mention_count += sum(
                        3 for kw in relationship_keywords if kw in para_lower
                    )
                scored_paragraphs.append((mention_count, para))

        # Fallback: keep lead paragraph
        if not scored_paragraphs and paragraphs:
            lead = paragraphs[0].strip()
            if lead:
                scored_paragraphs.append((0, lead))

        if not scored_paragraphs:
            continue

        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        kept_paragraphs = []
        total_chars = 0
        for _count, para in scored_paragraphs:
            if total_chars + len(para) > MAX_CHARS_PER_ARTICLE and kept_paragraphs:
                break
            marked_para = _markup_sentences(para, question_patterns, answer_patterns)
            kept_paragraphs.append(marked_para)
            total_chars += len(para)

        trimmed_text = "\n\n".join(kept_paragraphs)

        result.append({
            'qid': qid,
            'title': title,
            'paragraph': trimmed_text,
            'chain_indices': article.get('chain_indices', []),
            'original_length': original_length,
            'trimmed_length': len(trimmed_text),
        })

    return result


def _build_component_evidence(used_facts, extracted_facts, chains, articles,
                               quant_props, reasoning_ctx, entity_label):
    """Map each used clue fact to 1-2 supporting sentences from grounding articles.

    Also appends quantitative-property entries for each parameter used in the
    reasoning context so the annotator can see the Wikidata source values.

    Args:
        used_facts: List of fact_id strings (e.g. ["C1_F1", "C3_F1"]).
        extracted_facts: List of fact dicts with fact_id, chain_num, entity,
            property, value, fact keys.
        chains: List of KG chain dicts from fetch_multihop_triples.
        articles: List of Wikipedia article dicts (with paragraph, title, chain_indices).
        quant_props: Dict of PID -> {label, amount, unit, ...} for the entity.
        reasoning_ctx: Reasoning context dict with parameters, domain, etc.
        entity_label: The bridging entity label string.

    Returns:
        List of component dicts — clue components and quantitative property components.
    """
    _TAG_RE = re.compile(r'<</?(?:QA?_?)?RELEVANT>>')
    _CITE_RE = re.compile(r'\[\d+\]')  # Strip Wikipedia citation markers [1], [2], etc.

    # Build a lookup: fact_id -> fact dict
    fact_lookup = {}
    for f in extracted_facts:
        fid = f.get("fact_id")
        if fid:
            fact_lookup[fid] = f

    # Build article index: chain_num -> list of articles
    chain_to_articles = {}
    for art in articles:
        for ci in art.get("chain_indices", []):
            chain_to_articles.setdefault(ci, []).append(art)

    components = []

    for fid in used_facts:
        fact = fact_lookup.get(fid)
        if not fact:
            continue

        chain_num = fact.get("chain_num", 0)
        entity_val = fact.get("entity", "")
        prop_val = fact.get("property", "")
        value_val = fact.get("value", "")

        entity_low = str(entity_val).lower().strip()
        value_low = str(value_val).lower().strip()

        # Search ALL grounding articles.
        # Tiered scoring — higher = better evidence:
        #   10: sentence mentions BOTH entity and value explicitly
        #    5: sentence from entity-titled article, mentions value
        #    4: sentence from value-titled article, mentions entity
        scored_sentences = []
        for art in articles:
            title_low = art.get("title", "").lower()
            is_entity_article = (entity_low and (
                entity_low in title_low or title_low in entity_low))
            is_value_article = (value_low and (
                value_low in title_low or title_low in value_low))

            raw_para = _TAG_RE.sub('', art.get("paragraph", ""))
            clean_para = _CITE_RE.sub('', raw_para)
            for sent in re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_para):
                sent_stripped = sent.strip()
                if not sent_stripped or len(sent_stripped) > 400:
                    continue
                sent_lower = sent_stripped.lower()
                has_entity = entity_low and entity_low in sent_lower
                has_value = value_low and value_low in sent_lower

                if has_entity and has_value:
                    scored_sentences.append(
                        (10, sent_stripped, art.get("title", "")))
                elif has_value and is_entity_article:
                    scored_sentences.append(
                        (5, sent_stripped, art.get("title", "")))
                elif has_entity and is_value_article:
                    scored_sentences.append(
                        (4, sent_stripped, art.get("title", "")))
                elif not value_low and has_entity:
                    scored_sentences.append(
                        (1, sent_stripped, art.get("title", "")))

        scored_sentences.sort(key=lambda x: (-x[0], len(x[1])))
        evidence = [
            {"sentence": s, "article": a}
            for _sc, s, a in scored_sentences[:2]
        ]

        clue_triple = f'"{entity_val}" -[{prop_val}]-> "{value_val}"'
        components.append({
            "fact_id": fid,
            "clue": clue_triple,
            "clue_text": fact.get("fact", ""),
            "evidence": evidence,
        })

    # Append quantitative property components
    _param_to_pid = {
        "population": "P1082", "area": "P2046", "gdp": "P2131",
        "gdp_per_capita": "P2132", "elevation": "P2044",
        "mass_kg": "P2067", "radius_m": "P2120", "density": "P2054",
        "length": "P2043", "height": "P2048", "floors": "P1101",
        "prominence": "P2660", "depth": "P4511",
    }
    seen_pids: set = set()
    for pname in reasoning_ctx.get("parameters", {}):
        pid = _param_to_pid.get(pname)
        if pid and pid in quant_props and pid not in seen_pids:
            seen_pids.add(pid)
            qp = quant_props[pid]
            components.append({
                "type": "quantitative_property",
                "property": qp["label"],
                "value": qp["amount"],
                "unit": qp.get("unit", ""),
                "source": f"Wikidata {pid}",
            })

    return components


# ===========================================================================
# Generation Pipeline: gen_multiskill_qa_from_entity
# ===========================================================================

def gen_multiskill_qa_from_entity(entity, chains, agent_info, num_questions=3,
                                    prev_questions=None, used_templates=None,
                                    articles=None, theme=None):
    """Generate multi-skill QA pairs from a single entity.

    Phases:
        0: Select reasoning context (domain + template + compute answer)
        1: Extract clue facts from KG chains + validate triples against chain data
        1.5: Verify extracted facts against Wikipedia grounding documents
        2: Compose question from grounded clues + reasoning task
        3: Validate (computation re-execution, entity leak, value leak, etc.)

    Args:
        entity: Dict with 'id', 'label', 'quant_props'
        chains: List of KG chain dicts from fetch_multihop_triples
        agent_info: (lm, tokenizer, client) tuple
        num_questions: Questions to generate per entity
        prev_questions: Previously generated question strings
        used_templates: Set of (domain, template_id) already used
        articles: List of Wikipedia article dicts for grounding
        theme: Theme key from MULTISKILL_THEMES (e.g. 'rivers', 'dams')

    Returns:
        List of QA pair dicts
    """
    if prev_questions is None:
        prev_questions = []
    if used_templates is None:
        used_templates = set()
    if articles is None:
        articles = []

    entity_id = entity["id"]
    entity_label = entity["label"]
    quant_props = entity.get("quant_props", {})

    if not quant_props:
        print(f"  No quantitative properties for {entity_label} ({entity_id})", flush=True)
        return []

    if not chains:
        print(f"  No KG chains for {entity_label} ({entity_id})", flush=True)
        return []

    # Format chains for fact extraction
    formatted_chains = "\n".join(
        f"Chain {i+1}: {c['path_description']}" for i, c in enumerate(chains)
    )

    qa_pairs = []
    max_attempts = num_questions * 3
    poisoned_props = set()

    for attempt in range(max_attempts):
        if len(qa_pairs) >= num_questions:
            break

        print(f"\n  [{entity_label}] Attempt {attempt+1}/{max_attempts} "
              f"(generated {len(qa_pairs)}/{num_questions})", flush=True)

        # Phase 0: Select reasoning context
        reasoning_ctx = select_reasoning_context(
            entity_id, entity_label, quant_props, chains, used_templates,
            theme=theme,
        )
        if reasoning_ctx is None:
            print(f"  No valid reasoning template remaining", flush=True)
            break

        print(f"  Phase 0: {reasoning_ctx['domain']}/{reasoning_ctx['template_id']} "
              f"→ {reasoning_ctx['gold_answer']}", flush=True)

        # Phase 1: Extract clue facts
        fact_prompt = WIKIDATA_FACT_EXTRACTION_PROMPT.format(
            seed_entity=entity_label,
            chains_text=formatted_chains,
        )
        try:
            fact_response = gen_from_prompt_harmony(
                fact_prompt,
                temperature=0.7,
                max_tokens=2048,
                developer_content=WIKIDATA_FACT_EXTRACTION_DEVELOPER,
            )
        except Exception as e:
            print(f"  Phase 1 failed: {e}", flush=True)
            continue

        # Parse facts
        try:
            fact_json = extract_json_v2(fact_response, None)
            if isinstance(fact_json, dict):
                raw_facts = fact_json.get("facts", [])
            elif isinstance(fact_json, list):
                raw_facts = fact_json
            else:
                print(f"  Phase 1: unexpected format", flush=True)
                continue
        except Exception as e:
            print(f"  Phase 1: JSON parse failed: {e}", flush=True)
            continue

        # Ensure every fact has a fact_id (LLM may omit it)
        for fi, f in enumerate(raw_facts):
            if "fact_id" not in f:
                cn = f.get("chain_num", fi + 1)
                f["fact_id"] = f"C{cn}_F1"

        # Validate each fact triple against chain data (same as wikidata_bench)
        extracted_facts = []
        for fact in raw_facts:
            chain_num = fact.get('chain_num', 0)
            is_valid, hop_idx = validate_triple_in_chains(
                chain_num, fact.get('entity', ''), fact.get('property', ''),
                fact.get('value', ''), chains,
            )
            if is_valid:
                extracted_facts.append(fact)
            else:
                print(f"      Rejected {fact.get('fact_id', '?')}: triple not found in chain {chain_num} "
                      f"({fact.get('entity', '')} -[{fact.get('property', '')}]-> {fact.get('value', '')})",
                      flush=True)

        if len(extracted_facts) < 3:
            print(f"  Phase 1: Only {len(extracted_facts)} validated facts (need 3+)", flush=True)
            continue

        fact_chains = {f['chain_num'] for f in extracted_facts}
        print(f"  Phase 1: {len(extracted_facts)} validated clue facts across "
              f"{len(fact_chains)} chains", flush=True)

        if len(fact_chains) < 3:
            print(f"  Phase 1: Facts span only {len(fact_chains)} chains (need 3+)", flush=True)
            continue

        # Phase 1.5: Verify facts against Wikipedia grounding documents
        use_harmony = get_harmony_generator() is not None
        if articles:
            print(f"  Phase 1.5: Verifying facts against grounding documents...", flush=True)
            grounded_facts, rejected_grounding_facts = verify_facts_against_grounding(
                extracted_facts, chains, articles, use_harmony,
                agent_info=agent_info if not use_harmony else None,
            )
            grounded_fact_chains = {f['chain_num'] for f in grounded_facts}
            print(f"  Phase 1.5: {len(grounded_facts)}/{len(extracted_facts)} facts grounded "
                  f"across {len(grounded_fact_chains)} chains", flush=True)

            if len(grounded_facts) < 3 or len(grounded_fact_chains) < 3:
                print(f"  Phase 1.5: Insufficient grounded facts — "
                      f"{len(grounded_facts)} facts across {len(grounded_fact_chains)} chains",
                      flush=True)
                continue

            extracted_facts = grounded_facts
        else:
            print(f"  Phase 1.5: No Wikipedia articles available, skipping grounding check",
                  flush=True)

        # Filter out poisoned facts from previously rejected QAs
        extracted_facts = filter_poisoned_facts(extracted_facts, poisoned_props)
        if len(extracted_facts) < 3:
            print(f"  Too few facts after poisoned-fact filtering", flush=True)
            continue

        # Phase 2: Compose question
        composition = compose_multiskill_question(
            extracted_facts, reasoning_ctx, entity_label, prev_questions
        )
        if composition is None:
            print(f"  Phase 2: Composition failed", flush=True)
            continue

        question = composition["question"]
        used_facts = composition["used_facts"]
        print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

        # Phase 3: Validation (shared utility)
        passed, reason = run_phase3_validation(
            question=question,
            gold_answer=reasoning_ctx["gold_answer"],
            computation_code=reasoning_ctx["computation_code"],
            entity_label=entity_label,
            entity_names=[entity_label],
            entity_values=reasoning_ctx.get("entity_values", []),
            used_facts=used_facts,
            extracted_facts=extracted_facts,
            gen_fn=gen_from_prompt_harmony,
            reference_params=reasoning_ctx.get("parameters"),
            data_year=reasoning_ctx.get("data_year"),
            entity_type="entity",
            check_facts=True,
            min_chains=3,
        )
        if not passed:
            # Poison facts used in this rejected QA
            facts_by_id = {f["fact_id"]: f for f in extracted_facts if "fact_id" in f}
            for fid in used_facts:
                f = facts_by_id.get(fid)
                if f:
                    poisoned_props.add((f.get("property", ""), f.get("value", "")))
            print(f"  Phase 3: {reason}", flush=True)
            continue
        print(f"  Phase 3: PASSED", flush=True)

        # Build gold_chain: all intermediate entities/values needed to reach the answer
        gold_chain = _build_gold_chain(
            entity_id, entity_label, quant_props, reasoning_ctx
        )

        # Build QA pair
        source_triples = [c["path_description"] for c in chains[:5]]
        # Append quantitative property summaries with units
        _param_to_pid = {
            "population": "P1082", "area": "P2046", "gdp": "P2131",
            "gdp_per_capita": "P2132", "elevation": "P2044",
            "mass_kg": "P2067", "radius_m": "P2120", "density": "P2054",
            "length": "P2043", "height": "P2048", "floors": "P1101",
        }
        _seen_pids: set = set()
        for pname in reasoning_ctx.get("parameters", {}):
            pid = _param_to_pid.get(pname)
            if pid and pid in quant_props and pid not in _seen_pids:
                _seen_pids.add(pid)
                qp = quant_props[pid]
                source_triples.append(
                    f"[{qp['label']}] {entity_label}: {qp['amount']} {qp.get('unit', '')} (Wikidata {pid})"
                )
        grounding_articles = build_multiskill_grounding_articles(
            entity_label, reasoning_ctx["gold_answer"],
            reasoning_ctx["answer_unit"], extracted_facts, chains, articles,
        )
        # Build component-level evidence mapping
        component_evidence = _build_component_evidence(
            used_facts, extracted_facts, chains, articles,
            quant_props, reasoning_ctx, entity_label,
        )

        # Filter extracted_facts to essential keys for JSON output
        _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
        facts_for_output = [
            {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
            for f in extracted_facts
        ]

        # Look up raw template for CCI computation
        _tmpl = REASONING_DOMAINS[reasoning_ctx["domain"]]["templates"][reasoning_ctx["template_id"]]

        qa_pair = {
            "question": question,
            "gold_answer": reasoning_ctx["gold_answer"],
            "gold_entity": gold_chain,
            "reasoning_domain": reasoning_ctx["domain"],
            "template_id": reasoning_ctx["template_id"],
            "computation_code": reasoning_ctx["computation_code"],
            "reasoning_chain": reasoning_ctx["reasoning_chain"],
            "used_facts": used_facts,
            "extracted_facts": facts_for_output,
            "component_evidence": component_evidence,
            "source_triples": source_triples,
            "grounding_articles": grounding_articles,
            "wikidata_entity": entity_id,
            "entity_label": entity_label,
            "data_source": "multiskill",
            "domain_label": reasoning_ctx["domain_label"],
            "template_label": reasoning_ctx["template_label"],
            "answer_unit": reasoning_ctx["answer_unit"],
        }
        qa_pair.update(compute_cci_fields(_tmpl, extracted_facts, is_comparative=False))

        qa_pairs.append(qa_pair)
        prev_questions.append(question)
        used_templates.add((reasoning_ctx["domain"], reasoning_ctx["template_id"]))

    return qa_pairs


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_multiskill_bench(args, agent_info):
    """Run the multi-skill benchmark pipeline.

    Per theme:
    1. Fetch entities with quantitative properties
    2. For each entity: fetch KG chains + quantitative data
    3. Generate multi-skill QA pairs (Phases 0-3)
    4. V1 verification (closed-book)
    5. V2 verification (Python tool)
    6. Merge all themes into final output

    Args:
        args: Parsed CLI arguments
        agent_info: (lm, tokenizer, client) tuple
    """
    prefix = args.outfile_prefix1
    run_id = getattr(args, "run_id", None) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{prefix}__{run_id}"
    themes_arg = args.multiskill_themes
    per_theme = args.multiskill_per_theme
    v1_samples = args.multiskill_v1_samples
    v2_samples = args.multiskill_v2_samples
    v1_threshold = args.multiskill_v1_threshold
    v2_threshold = args.multiskill_v2_threshold
    num_chains = args.multiskill_num_chains
    questions_per_entity = args.multiskill_questions_per_entity

    bench_label = "multiskill_bench"

    # Parse themes
    if themes_arg:
        themes_to_process = [t.strip() for t in themes_arg.split(",")]
    else:
        themes_to_process = list(MULTISKILL_THEMES.keys())

    is_subset = themes_arg is not None and len(themes_to_process) < len(MULTISKILL_THEMES)

    print(f"=== MULTI-SKILL BENCH (run_id={run_id}) ===", flush=True)
    print(f"Themes: {themes_to_process}", flush=True)
    print(f"Target QAs per theme: {per_theme}", flush=True)
    print(f"V1: {v1_samples} samples, threshold < {v1_threshold:.0%}", flush=True)
    print(f"V2: {v2_samples} samples, threshold < {v2_threshold:.0%}", flush=True)
    print("=" * 50, flush=True)

    all_final = []
    summary = {}

    for theme in themes_to_process:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Theme: {theme.upper()} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Ensure output directory exists
        output_dir = os.path.dirname(run_prefix) if "/" in run_prefix else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check full cache (V2)
        v2_cache = f"{run_prefix}__{theme}.{bench_label}_v2.json"
        v2_filtered_cache = f"{run_prefix}__{theme}.{bench_label}_v2_filtered.json"
        if os.path.exists(v2_cache) and os.path.exists(v2_filtered_cache):
            print(f"Found cached V2 results for {theme}, loading...", flush=True)
            with open(v2_cache) as f:
                v2_all = json.load(f)
            with open(v2_filtered_cache) as f:
                v2_filtered = json.load(f)
            all_final.extend(v2_filtered)
            summary[theme] = {
                "generated": "cached", "v1_filtered": "cached",
                "v2_total": len(v2_all), "v2_filtered": len(v2_filtered),
            }
            continue
        else:
            # Check V1 filtered cache
            v1_filtered_cache = f"{run_prefix}__{theme}.{bench_label}_v1_filtered.json"
            if os.path.exists(v1_filtered_cache):
                print(f"Found cached V1 filtered for {theme}, skipping to V2...", flush=True)
                with open(v1_filtered_cache) as f:
                    v1_filtered = json.load(f)
            else:
                # Check raw bench cache
                bench_cache = f"{run_prefix}__{theme}.{bench_label}.json"
                if os.path.exists(bench_cache):
                    print(f"Found cached bench problems for {theme}...", flush=True)
                    with open(bench_cache) as f:
                        qa_pairs = json.load(f)
                else:
                    # Step 1: Fetch entities and generate QA pairs
                    qa_pairs = _generate_theme_qa_pairs(
                        theme, per_theme, num_chains, questions_per_entity,
                        agent_info, prefix, bench_label,
                    )

                    # Save raw bench cache
                    if qa_pairs:
                        with open(bench_cache, "w") as f:
                            json.dump(qa_pairs, f, indent=2)
                        print(f"Saved {len(qa_pairs)} raw QA pairs to {bench_cache}", flush=True)

                if not qa_pairs:
                    print(f"No QA pairs for theme {theme}", flush=True)
                    summary[theme] = {"generated": 0, "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                    continue

                # Step 2: V1 verification
                v1_filtered, v1_all = verify_bench_v1(
                    qa_pairs, agent_info,
                    num_samples=v1_samples,
                    temperature=0.7,
                    threshold=v1_threshold,
                    outfile_prefix=run_prefix,
                    subarea=theme,
                    bench_label=bench_label,
                    answer_checker=is_multiskill_answer_correct,
                    llm_judge=_llm_judge_multiskill_answer,
                    verification_prompt=MULTISKILL_V1_PROMPT,
                )

            if not v1_filtered:
                print(f"No V1-filtered QA pairs for {theme}", flush=True)
                summary[theme] = {"generated": len(qa_pairs) if 'qa_pairs' in dir() else "cached",
                                  "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                continue

            # Step 3: V2 verification (browser + Python tools)
            v2_filtered, v2_all = verify_multiskill_v2(
                v1_filtered,
                num_samples=v2_samples,
                temperature=1.0,
                max_iterations=200,
                threshold=v2_threshold,
                outfile_prefix=run_prefix,
                subarea=theme,
                bench_label=bench_label,
            )

        all_final.extend(v2_filtered)
        summary[theme] = {
            "generated": len(v1_all) if 'v1_all' in dir() else "cached",
            "v1_filtered": len(v1_filtered) if 'v1_filtered' in dir() else "cached",
            "v2_total": len(v2_all),
            "v2_filtered": len(v2_filtered),
        }

    # Skip final merge when processing a subset (worker job)
    if is_subset:
        print(f"\nSubset mode: processed {themes_to_process}. "
              f"Merge will happen in the merge job.", flush=True)
        _print_summary(summary)
        return all_final

    # Final merge
    final_path = f"{run_prefix}.{bench_label}_final.json"
    if all_final:
        # Optional diversity filter
        if _HAS_DIVERSITY and len(all_final) > 10:
            print(f"\nApplying diversity filter to {len(all_final)} questions...", flush=True)
            all_final = diversity_filter(all_final, target_count=len(all_final), min_distance=0.03)

        with open(final_path, "w") as f:
            json.dump(all_final, f, indent=2)
        print(f"\nSaved {len(all_final)} final QA pairs to {final_path}", flush=True)

        # Diversity report
        if _HAS_DIVERSITY:
            diversity_report(all_final, text_keys=["question"])

    _print_summary(summary)
    return all_final


def _generate_theme_qa_pairs(theme, per_theme, num_chains, questions_per_entity,
                              agent_info, prefix, bench_label):
    """Generate QA pairs for a single theme.

    Fetches entities, their quantitative properties and KG chains,
    then generates multi-skill questions.
    """
    # Check chains cache
    chains_cache = f"{prefix}__{theme}.multiskill_chains.json"
    if os.path.exists(chains_cache):
        print(f"Loading cached chains for {theme}...", flush=True)
        with open(chains_cache) as f:
            entity_data_list = json.load(f)

        # Backfill Wikipedia articles for cached entities that lack them
        needs_resave = False
        for ed in entity_data_list:
            if "articles" not in ed or not ed["articles"]:
                print(f"  Fetching Wikipedia articles for cached entity {ed['label']}...",
                      flush=True)
                ed["articles"] = fetch_wikipedia_for_entities(ed["chains"])
                print(f"  {ed['label']}: fetched {len(ed['articles'])} Wikipedia articles",
                      flush=True)
                needs_resave = True
        if needs_resave:
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Re-saved chains cache with Wikipedia articles for {theme}", flush=True)
    else:
        entities = fetch_multiskill_entities(theme, target_count=per_theme * 2)
        if not entities:
            return []

        entity_data_list = []
        for ent in tqdm.tqdm(entities, desc=f"Fetching {theme} entity data"):
            quant_props = fetch_quantitative_properties(ent["id"])
            if not quant_props:
                continue

            chains = fetch_multihop_triples(ent["id"], num_hops=2, limit=num_chains)
            if not chains or len(chains) < 3:
                continue

            # Fetch Wikipedia articles for chain entities (grounding documents)
            articles = fetch_wikipedia_for_entities(chains)
            print(f"  {ent['label']}: fetched {len(articles)} Wikipedia articles "
                  f"for {len(chains)} chains", flush=True)

            entity_data_list.append({
                "id": ent["id"],
                "label": ent["label"],
                "quant_props": _serialize_quant_props(quant_props),
                "chains": chains,
                "articles": articles,
            })

            if len(entity_data_list) >= per_theme:
                break

        # Save chains cache
        if entity_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Cached {len(entity_data_list)} entity data for {theme}", flush=True)

    # Generate QA pairs from entities
    all_qa_pairs = []
    prev_questions = []
    used_templates = set()

    for ed in entity_data_list:
        if len(all_qa_pairs) >= per_theme:
            break

        # Deserialize quant_props
        quant_props = _deserialize_quant_props(ed["quant_props"])
        entity = {"id": ed["id"], "label": ed["label"], "quant_props": quant_props}

        qa_pairs = gen_multiskill_qa_from_entity(
            entity, ed["chains"], agent_info,
            num_questions=questions_per_entity,
            prev_questions=prev_questions,
            used_templates=used_templates,
            articles=ed.get("articles", []),
            theme=theme,
        )

        for qa in qa_pairs:
            qa["theme"] = theme
        all_qa_pairs.extend(qa_pairs)
        print(f"  {ed['label']}: generated {len(qa_pairs)} QA pairs "
              f"(total: {len(all_qa_pairs)}/{per_theme})", flush=True)

    return all_qa_pairs


def _serialize_quant_props(quant_props):
    """Serialize quant_props for JSON storage (handles tuples)."""
    result = {}
    for k, v in quant_props.items():
        entry = dict(v)
        if isinstance(entry.get("amount"), tuple):
            entry["amount"] = list(entry["amount"])
            entry["_is_coordinate"] = True
        result[k] = entry
    return result


def _deserialize_quant_props(serialized):
    """Deserialize quant_props from JSON (restore tuples)."""
    result = {}
    for k, v in serialized.items():
        entry = dict(v)
        if entry.pop("_is_coordinate", False):
            entry["amount"] = tuple(entry["amount"])
        result[k] = entry
    return result


def _print_summary(summary):
    """Print pipeline summary table."""
    print(f"\n{'='*60}", flush=True)
    print("MULTI-SKILL BENCH SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for theme, stats in summary.items():
        print(f"  {theme:20s}: generated={stats.get('generated','?'):>6} "
              f"v1_kept={stats.get('v1_filtered',0):>4} "
              f"v2_kept={stats.get('v2_filtered',0):>4}", flush=True)
    total = sum(s.get("v2_filtered", 0) for s in summary.values())
    print(f"  {'TOTAL':20s}: {total:>4} final questions", flush=True)
    print(f"{'='*60}\n", flush=True)


# ===========================================================================
# CLI __main__
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="multiskill_drbencher",
        description="Multi-Skill Benchmark: multi-hop entity ID + quantitative reasoning",
    )

    # Model args
    parser.add_argument("--agent_modelname", default="openai/gpt-oss-120b")
    parser.add_argument("--use_harmony", type=str, default="no")
    parser.add_argument("--harmony_model_path", type=str, default=None)
    parser.add_argument("--use_helm", type=str, default="no")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Output
    parser.add_argument("--outfile_prefix1", type=str, default="multiskill/bench")

    # Exp mode (for compatibility with other bench scripts)
    parser.add_argument("--exp_mode", type=str, default="multiskill_bench")

    # Run ID for multi-run support (auto-generated if not provided)
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: auto YYYYMMDD_HHMMSS)")

    # Multi-skill specific
    parser.add_argument("--multiskill_themes", type=str, default=None,
                        help="Comma-separated themes (default: all). E.g. 'countries,mountains'")
    parser.add_argument("--multiskill_per_theme", type=int, default=50,
                        help="Target QA pairs per theme")
    parser.add_argument("--multiskill_v1_samples", type=int, default=10,
                        help="V1 sampling attempts per question")
    parser.add_argument("--multiskill_v2_samples", type=int, default=10,
                        help="V2 (Python tool) sampling attempts per question")
    parser.add_argument("--multiskill_v1_threshold", type=float, default=0.5,
                        help="V1 accuracy ceiling — keep below this")
    parser.add_argument("--multiskill_v2_threshold", type=float, default=0.5,
                        help="V2 accuracy ceiling — keep below this")
    parser.add_argument("--multiskill_num_chains", type=int, default=50,
                        help="KG chains per entity")
    parser.add_argument("--multiskill_questions_per_entity", type=int, default=3,
                        help="Questions to generate per entity")

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
        wikidata_set_harmony_generator(harmony_gen)

    # When Harmony is enabled, skip loading separate models
    if args.use_harmony.lower() == "yes":
        print("Harmony mode: using Harmony generator for all inference", flush=True)
        agent_info = (None, None, None)
    elif args.use_helm == "yes":
        from .util import helm_process_args
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        agent_info = (agent_lm, agent_tokenizer, agent_client)
    else:
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        agent_info = (agent_lm, agent_tokenizer, agent_client)

    # Run benchmark
    if args.exp_mode == "multiskill_bench":
        run_multiskill_bench(args, agent_info)
        # The in-process vLLM engine spawns worker subprocesses that outlive the
        # bench; shut the engine down and terminate the process group so the job
        # exits cleanly instead of hanging until it is killed by hand.
        shutdown_and_exit(0)
    else:
        print(f"Unknown exp_mode: {args.exp_mode}", flush=True)
        sys.exit(1)
