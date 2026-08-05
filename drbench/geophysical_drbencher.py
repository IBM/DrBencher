# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Geophysical Benchmark: multi-step physics reasoning with Wikidata entity data.

Creates questions requiring 3-5 reasoning steps: the solver must identify
an entity from multi-hop KG clues, then apply the correct physics/math model
(without it being named), chain intermediate computations, and produce a
final numerical answer.

Data sources:
  Entities:   Wikidata SPARQL (mountains P2044, cities P625, structures P2048+P625)
  Clues:      Wikipedia articles via multi-hop KG chains
  Properties: Wikidata quantitative claims (elevation, coordinates, height)

Run via:  python -m drbench.geophysical_drbencher --exp_mode geophysical_bench ...
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

# Reuse fact extraction / grounding from wikidata pipeline
from .wikidata_harmony import (
    validate_triple_in_chains,
    verify_facts_against_grounding,
    format_extracted_facts,
    parse_used_facts,
    set_harmony_generator as wikidata_set_harmony_generator,
    WIKIDATA_FACT_EXTRACTION_DEVELOPER,
    WIKIDATA_FACT_EXTRACTION_PROMPT,
)

# Reuse quantitative property fetching from multiskill
from .multiskill_drbencher import (
    fetch_quantitative_properties,
    build_multiskill_grounding_articles,
    _serialize_quant_props,
    _deserialize_quant_props,
    is_multiskill_answer_correct,
    _llm_judge_multiskill_answer,
    _try_parse_number,
    _normalize_numeric_text,
)

# Geophysical templates
from .geophysical_template import (
    GEOPHYSICAL_TEMPLATES,
    GEOPHYSICAL_THEMES,
    category_to_entity_type,
    select_geophysical_template,
    select_comparative_geophysical_template,
    generate_template_params,
    compute_gold_answer,
    _execute_computation_code,
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
# Prompts
# ===========================================================================

GEOPHYSICAL_QA_DEVELOPER = (
    "You are a geophysics expert creating research questions that test "
    "multi-step scientific reasoning. Each question requires first identifying "
    "an entity from clues, then applying the correct physics model to compute "
    "a numerical answer."
)

GEOPHYSICAL_QA_PROMPT = """Compose a geophysical reasoning question that:
1. Uses clue facts to describe an unnamed entity — readers must figure out which entity
2. Then poses a scenario requiring multi-step physics/math computation

CLUE FACTS (about the unnamed entity — do NOT name it directly):
{facts_text}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}
Reasoning steps: {description}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

{external_params_text}

--- STYLE GUIDANCE ---
- Write 2-4 sentences total.
- First 1-2 sentences: describe the entity using 3+ clue facts from different chains, without naming it directly.
- Last 1-2 sentences: describe the SCENARIO (e.g., "A pendulum clock calibrated at sea level...") and ask for a specific numerical result.
- CRITICAL: Do NOT name the physics formula or model — describe the scenario and let the solver figure out which approach to use.
- CRITICAL: Do NOT reveal ANY quantitative values from the entity's data (elevation, coordinates, height, etc.) in the question — the solver must look up ALL entity data themselves.
- You MAY include external/hypothetical parameters (e.g., heater power, wind speed, skydiver mass) that are not real-world properties of the entity.
- CRITICAL: Do NOT name the entity directly anywhere in the question. Use descriptive clues only.
- Sound natural and conversational, like a real geophysics problem set question.
- Use clear, unambiguous language.
- Do NOT mention "Wikidata", "KG", "entity", or "property".

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your geophysical reasoning question>
Used_Facts: <comma-separated fact_ids used, e.g. C1_F1, C2_F1, C4_F1>
Reasoning: <brief chain: clues identify entity -> look up data -> multi-step computation gives answer>
"""

GEOPHYSICAL_COMPARATIVE_QA_PROMPT = """Compose a comparative geophysical reasoning question that:
1. Describes TWO unnamed entities using separate clue facts for each
2. Poses a scenario requiring multi-step physics/math computation involving both entities

CLUE FACTS FOR ENTITY A (do NOT name this entity directly):
{facts_text_a}

CLUE FACTS FOR ENTITY B (do NOT name this entity directly):
{facts_text_b}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}
Reasoning steps: {description}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

{external_params_text}

--- STYLE GUIDANCE ---
- Write 4-6 sentences total.
- First 1-2 sentences: describe Entity A using 3+ clue facts from different chains, without naming it directly.
- Next 1-2 sentences: describe Entity B using 3+ clue facts from different chains, without naming it directly.
- Last 1-2 sentences: describe the SCENARIO (e.g., "Compare the gravitational potential energy...") and ask for a specific numerical result.
- Refer to them as "the first location" and "the second location" (or similar distinct references).
- CRITICAL: Do NOT name the physics formula or model — describe the scenario and let the solver figure out which approach to use.
- CRITICAL: Do NOT reveal ANY quantitative values from either entity's data (elevation, coordinates, height, etc.) in the question — the solver must look up ALL entity data themselves.
- You MAY include external/hypothetical parameters (e.g., heater power, wind speed, skydiver mass) that are not real-world properties of the entities.
- CRITICAL: Do NOT name either entity directly anywhere in the question. Use descriptive clues only.
- Sound natural and conversational, like a real geophysics problem set question.
- Use clear, unambiguous language.
- Do NOT mention "Wikidata", "KG", "entity", or "property".

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your comparative geophysical reasoning question>
Used_Facts_A: <comma-separated fact_ids for entity A, e.g. A_C1_F1, A_C2_F1>
Used_Facts_B: <comma-separated fact_ids for entity B, e.g. B_C1_F1, B_C2_F1>
Reasoning: <brief chain: clues identify both entities -> look up data -> multi-step computation gives answer>

REQUIREMENTS:
- Use 3+ clue facts per entity from different chains
- Do NOT name either entity in the question
- Do NOT include quantitative entity values in the question
- Every claim must come VERBATIM from the listed facts
- The question MUST have exactly ONE unambiguous answer"""

GEOPHYSICAL_V1_PROMPT = """Solve this geophysical reasoning question. It requires:
1. Identifying a real-world entity from descriptive clues
2. Looking up its quantitative properties (elevation, coordinates, height, etc.)
3. Applying the correct physics/math model (potentially multi-step)
4. Computing the final numerical answer

Provide ONLY the final numerical answer (with units if applicable).
Do not show your work.

Question: {question}

Answer:"""

GEOPHYSICAL_V2_DEVELOPER = """You are an expert geophysics research assistant.
You have TWO tools available:
- **Browser tool**: Search and browse Wikidata and Wikipedia to identify entities and look up their properties (elevation, coordinates, height, etc.)
- **Python tool**: Execute Python code for calculations (math, numpy, scipy available)

Recommended approach:
1. Use the browser to identify the entity matching the clues
2. Use the browser to look up the entity's quantitative properties
3. Use Python to perform the multi-step computation and get the final answer

Give your final answer as a single number (with units if appropriate) on the last line."""


# ===========================================================================
# Answer Checker
# ===========================================================================

def is_geophysical_answer_correct(predicted, gold, tolerance=0.05):
    """Geophysical answer comparison with 5% tolerance."""
    if not predicted or not predicted.strip():
        return False

    pred_clean = _normalize_geophysical_text(predicted)
    gold_clean = _normalize_geophysical_text(gold)

    # 1. Exact match
    if pred_clean == gold_clean:
        return True

    # 2. Numeric comparison
    pred_num = _try_parse_number(pred_clean)
    gold_num = _try_parse_number(gold_clean)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0 and pred_num == 0:
            return True
        if gold_num == 0:
            return abs(pred_num) < 0.01
        rel_error = abs(pred_num - gold_num) / max(abs(gold_num), abs(pred_num))
        if rel_error <= tolerance:
            return True

    return False


def _normalize_geophysical_text(text):
    """Normalize text for geophysical comparison."""
    text = text.strip()
    for suffix in ["°C", "°F", "kPa", "Pa", "km²", "km", "m/s²", "m/s",
                    "mGal", "hours", "seconds", "minutes", "metres", "m",
                    "minutes after midnight", "km/s", "kg/m³"]:
        if len(suffix) > 1:
            text = text.replace(suffix, "").strip()
        else:
            text = re.sub(rf'\s*{re.escape(suffix)}\s*$', '', text).strip()
    text = text.replace(",", "")
    text = text.strip().strip("'\"")
    return text


def _llm_judge_geophysical_answer(predicted, gold, question):
    """Use LLM to judge if predicted answer matches gold."""
    prompt = f"""Compare these two answers to the same geophysical question.
The gold answer is computed from verified Wikidata properties.
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
# Entity Discovery
# ===========================================================================

def fetch_geophysical_entities(category, target_count=20):
    """Fetch entities suitable for geophysical questions from Wikidata.

    Args:
        category: Category key from GEOPHYSICAL_THEMES
        target_count: Number of entities to return

    Returns:
        List of entity dicts with id, label
    """
    cat_info = GEOPHYSICAL_THEMES.get(category)
    if not cat_info:
        print(f"Unknown geophysical category: {category}", flush=True)
        return []

    wikidata_types = cat_info["wikidata_types"]
    type_filter = " UNION ".join(
        f"{{ ?item wdt:P31/wdt:P279? wd:{t} }}" for t in wikidata_types
    )
    if len(wikidata_types) > 1:
        type_filter = "{ " + type_filter + " }"

    # Build property requirements from theme config
    prop_filters = cat_info["sparql_filters"]
    prop_filter_str = "\n      ".join(prop_filters)

    query = f"""
    SELECT DISTINCT ?item ?itemLabel WHERE {{
      {type_filter}
      {prop_filter_str}
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 20)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY MD5(STR(?item))
    LIMIT {target_count * 3}
    """

    results = sparql_query(query)
    if not results:
        print(f"No entities found for category '{category}'", flush=True)
        return []

    entities = []
    for row in results:
        eid = row.get("item", {}).get("value", "").split("/")[-1]
        label = row.get("itemLabel", {}).get("value", eid)
        if not eid.startswith("Q"):
            continue
        if re.match(r'^Q\d+$', label):
            label = get_entity_label(eid)
            if re.match(r'^Q\d+$', label):
                print(f"  Skipping {eid}: no human-readable label", flush=True)
                continue
        entities.append({"id": eid, "label": label})

    random.shuffle(entities)
    print(f"Found {len(entities)} candidate entities for category '{category}'", flush=True)
    return entities[:target_count]


# ===========================================================================
# QA Generation Pipeline
# ===========================================================================

def gen_geophysical_qa_from_entity(entity, chains, agent_info, num_questions=5,
                                    prev_questions=None, used_templates=None,
                                    articles=None, category=None):
    """Generate geophysical QA pairs from a single entity.

    Phases:
        0: Select template + compute gold answer from Wikidata properties
        1: Extract clue facts from KG chains + validate triples
        1.5: Verify facts against Wikipedia grounding documents
        2: Compose question from grounded clues + reasoning scenario
        3: Validate (recomputation, name/value leak, etc.)

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
    entity_type = GEOPHYSICAL_THEMES.get(category, {}).get("entity_type", "entity")

    if not quant_props:
        print(f"  No quantitative properties for {entity_label} ({entity_id})", flush=True)
        return []

    if not chains:
        print(f"  No KG chains for {entity_label} ({entity_id})", flush=True)
        return []

    qa_pairs = []
    max_attempts = num_questions * 3
    poisoned_props = set()

    for attempt in range(max_attempts):
        if len(qa_pairs) >= num_questions:
            break

        print(f"\n  [{entity_label}] Attempt {attempt+1}/{max_attempts} "
              f"(generated {len(qa_pairs)}/{num_questions})", flush=True)

        # Phase 0: Select template and compute gold answer
        tmpl_result = select_geophysical_template(entity_type, quant_props, used_templates)
        if tmpl_result is None:
            print(f"  No valid geophysical template remaining", flush=True)
            break

        tmpl_id, tmpl = tmpl_result
        params = generate_template_params(tmpl_id, tmpl, quant_props)
        if params is None:
            print(f"  Template {tmpl_id}: parameter generation failed", flush=True)
            continue

        gold_answer = compute_gold_answer(tmpl_id, tmpl, params)
        if gold_answer is None:
            print(f"  Template {tmpl_id}: computation failed", flush=True)
            continue

        # Build question hint with filled parameters
        try:
            question_hint = tmpl["question_hint"].format(**params)
        except KeyError:
            question_hint = tmpl["question_hint"]

        print(f"  Phase 0: {tmpl_id} -> {gold_answer}", flush=True)

        # Phase 1: Extract clue facts from KG chains
        extracted_facts, _ = extract_chain_clue_facts(chains, entity_label, agent_info)
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

        # Phase 2: Compose question
        facts_text = format_extracted_facts(extracted_facts)
        prev_q_text = "\n".join(f"- {q}" for q in prev_questions[-10:]) if prev_questions else "None"

        # Build external params text
        entity_prop_names = {"latitude", "longitude", "elevation", "height",
                             "latitude_a", "longitude_a", "latitude_b", "longitude_b",
                             "elevation_a", "elevation_b", "height_a", "height_b"}
        external_params = {k: v for k, v in params.items() if k not in entity_prop_names}
        if external_params:
            ext_lines = ["External/hypothetical parameters (you MAY include these in the question):"]
            for k, v in external_params.items():
                ext_lines.append(f"- {k}: {v}")
            external_params_text = "\n".join(ext_lines)
        else:
            external_params_text = ""

        compose_prompt = GEOPHYSICAL_QA_PROMPT.format(
            facts_text=facts_text,
            template_label=tmpl["label"],
            question_hint=question_hint,
            answer_unit=tmpl["answer_unit"],
            gold_answer=gold_answer,
            description=tmpl["description"],
            external_params_text=external_params_text,
            previous_questions=prev_q_text,
        )

        try:
            response = gen_from_prompt_harmony(
                compose_prompt, temperature=1.0, max_tokens=8196,
                developer_content=GEOPHYSICAL_QA_DEVELOPER,
                reasoning_effort="high",
            )
        except Exception as e:
            print(f"  Phase 2 failed: {e}", flush=True)
            continue

        if not response:
            continue

        question_match = re.search(
            r'Question:\s*(.+?)(?:\nUsed_Facts:|\nReasoning:|\Z)', response, re.DOTALL
        )
        if not question_match:
            print(f"  Phase 2: Could not parse question from response", flush=True)
            continue

        question = question_match.group(1).strip()
        used_facts = parse_used_facts(response)
        print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

        # Phase 3: Validation
        # Collect entity property values for leak detection
        entity_values = []
        for prop_name in entity_prop_names:
            if prop_name in params:
                try:
                    entity_values.append(float(params[prop_name]))
                except (ValueError, TypeError):
                    pass

        code = tmpl["code_template"].format(**params)

        passed, reason = run_phase3_validation(
            question=question,
            gold_answer=gold_answer,
            computation_code=code,
            entity_label=entity_label,
            entity_names=[entity_label],
            entity_values=entity_values,
            used_facts=used_facts,
            extracted_facts=extracted_facts,
            gen_fn=gen_from_prompt_harmony,
            entity_type=entity_type,
            check_facts=True,
            min_chains=3,
            chains=chains,
            entity_qid=entity_id,
        )
        if not passed:
            # Poison the used facts only when the rejection is intrinsic to the
            # pair (3d3b); other reasons are phrasing/combination, so retry
            # instead of banning innocent facts and starving sparse entities.
            poison_used_facts(reason, used_facts, extracted_facts, poisoned_props)
            print(f"  Phase 3: {reason}", flush=True)
            continue
        print(f"  Phase 3: PASSED", flush=True)

        # Build grounding articles
        grounding_articles = build_multiskill_grounding_articles(
            entity_label, gold_answer, tmpl["answer_unit"],
            extracted_facts, chains, articles,
        )

        # Build source triples
        source_triples = [c["path_description"] for c in chains[:5]]
        # Append quantitative property summaries
        _prop_ids = {"P2044", "P625", "P2048"}
        for pid in _prop_ids:
            if pid in quant_props:
                qp = quant_props[pid]
                source_triples.append(
                    f"[{qp['label']}] {entity_label}: {qp['amount']} "
                    f"{qp.get('unit', '')} (Wikidata {pid})"
                )

        # Filter facts to essential keys
        _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
        facts_for_output = [
            {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
            for f in extracted_facts
        ]

        qa_pair = {
            "question": question,
            "gold_answer": gold_answer,
            "template_id": tmpl_id,
            "template_label": tmpl["label"],
            "template_category": tmpl["category"],
            "template_type": tmpl["type"],
            "template_steps": tmpl["steps"],
            "computation_code": code,
            "answer_unit": tmpl["answer_unit"],
            "used_facts": used_facts,
            "extracted_facts": facts_for_output,
            "source_triples": source_triples,
            "grounding_articles": grounding_articles,
            "wikidata_entity": entity_id,
            "entity_label": entity_label,
            "entity_type": category_to_entity_type(category),
            "category": category,
            "data_source": "geophysical",
        }
        qa_pair.update(compute_cci_fields(tmpl, extracted_facts, is_comparative=False))

        qa_pairs.append(qa_pair)
        prev_questions.append(question)
        used_templates.add(tmpl_id)

    return qa_pairs


def gen_geophysical_qa_comparative(entity_a, entity_b,
                                     chains_a, chains_b,
                                     articles_a, articles_b,
                                     agent_info, used_templates,
                                     prev_questions, category):
    """Generate a comparative geophysical QA pair (two entities).

    Uses templates like gravity_train_period, line_of_sight_range, seismic_p_wave_time.
    """
    quant_props_a = entity_a.get("quant_props", {})
    quant_props_b = entity_b.get("quant_props", {})
    entity_type = GEOPHYSICAL_THEMES.get(category, {}).get("entity_type", "entity")

    tmpl_result = select_comparative_geophysical_template(
        entity_type, quant_props_a, quant_props_b, used_templates,
    )
    if tmpl_result is None:
        return None

    tmpl_id, tmpl = tmpl_result
    params = generate_template_params(tmpl_id, tmpl, quant_props_a, quant_props_b)
    if params is None:
        return None

    gold_answer = compute_gold_answer(tmpl_id, tmpl, params)
    if gold_answer is None:
        return None

    print(f"  Phase 0 (comparative): {tmpl_id} -> {gold_answer}", flush=True)

    try:
        question_hint = tmpl["question_hint"].format(**params)
    except KeyError:
        question_hint = tmpl["question_hint"]

    # Phase 1 + 1.5: Extract and ground chain-based clues for both entities
    facts_a, _ = extract_chain_clue_facts(chains_a, entity_a["label"], agent_info)
    if facts_a is not None and articles_a:
        facts_a = verify_facts_grounding(facts_a, chains_a, articles_a, agent_info)
    facts_b, _ = extract_chain_clue_facts(chains_b, entity_b["label"], agent_info)
    if facts_b is not None and articles_b:
        facts_b = verify_facts_grounding(facts_b, chains_b, articles_b, agent_info)

    if not facts_a or len(facts_a) < 2 or not facts_b or len(facts_b) < 2:
        print(f"  Comparative: insufficient facts (A={len(facts_a) if facts_a else 0}, "
              f"B={len(facts_b) if facts_b else 0})", flush=True)
        return None

    # Merge clues for composition
    combined_clues = []
    for c in facts_a[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"A_{c['fact_id']}"
        c_copy["fact"] = f"[Entity A] {c.get('fact', c.get('value', ''))}"
        combined_clues.append(c_copy)
    for c in facts_b[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"B_{c['fact_id']}"
        c_copy["fact"] = f"[Entity B] {c.get('fact', c.get('value', ''))}"
        combined_clues.append(c_copy)

    # Phase 2: Compose question
    facts_text = format_extracted_facts(combined_clues)
    prev_q_text = "\n".join(f"- {q}" for q in prev_questions[-10:]) if prev_questions else "None"

    entity_prop_names = {"latitude", "longitude", "elevation", "height",
                         "latitude_a", "longitude_a", "latitude_b", "longitude_b",
                         "elevation_a", "elevation_b"}
    external_params = {k: v for k, v in params.items() if k not in entity_prop_names}
    if external_params:
        ext_lines = ["External/hypothetical parameters (you MAY include these):"]
        for k, v in external_params.items():
            ext_lines.append(f"- {k}: {v}")
        external_params_text = "\n".join(ext_lines)
    else:
        external_params_text = ""

    compose_prompt = GEOPHYSICAL_QA_PROMPT.format(
        facts_text=facts_text,
        template_label=tmpl["label"],
        question_hint=question_hint,
        answer_unit=tmpl["answer_unit"],
        gold_answer=gold_answer,
        description=tmpl["description"],
        external_params_text=external_params_text,
        previous_questions=prev_q_text,
    )

    try:
        response = gen_from_prompt_harmony(
            compose_prompt, temperature=1.0, max_tokens=8196,
            developer_content=GEOPHYSICAL_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception:
        return None

    question_match = re.search(
        r'Question:\s*(.+?)(?:\nUsed_Facts:|\nReasoning:|\Z)', response, re.DOTALL
    )
    if not question_match:
        return None

    question = question_match.group(1).strip()
    used_facts = parse_used_facts(response)

    # Phase 3: Validation
    entity_values = []
    for prop_name in entity_prop_names:
        if prop_name in params:
            try:
                entity_values.append(float(params[prop_name]))
            except (ValueError, TypeError):
                pass

    code = tmpl["code_template"].format(**params)

    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=gold_answer,
        computation_code=code,
        entity_label=f"{entity_a['label']} / {entity_b['label']}",
        entity_names=[entity_a["label"], entity_b["label"]],
        entity_values=entity_values,
        entity_type=GEOPHYSICAL_THEMES.get(category, {}).get("entity_type", "entity"),
    )
    if passed:
        # 3h (comparative): each side must be uniquely identifiable from its clues
        passed, reason = check_kg_uniqueness_comparative(
            strip_side_prefix(used_facts, "A_"), facts_a or [], chains_a,
            strip_side_prefix(used_facts, "B_"), facts_b or [], chains_b,
        )
    if not passed:
        print(f"  Phase 3 (comparative): {reason}", flush=True)
        return None

    used_templates.add(tmpl_id)

    result = {
        "question": question,
        "gold_answer": gold_answer,
        "template_id": tmpl_id,
        "template_label": tmpl["label"],
        "template_category": tmpl["category"],
        "template_type": "comparative",
        "template_steps": tmpl["steps"],
        "computation_code": code,
        "answer_unit": tmpl["answer_unit"],
        "used_facts": used_facts,
        "entity_name_a": entity_a["label"],
        "entity_name_b": entity_b["label"],
        "entity_type": category_to_entity_type(category),
        "category": category,
        "data_source": "geophysical",
        "grounding_clues_a": facts_a,
        "grounding_clues_b": facts_b,
        "grounding_articles": (articles_a or [])[:3] + (articles_b or [])[:3],
    }
    result.update(compute_cci_fields(tmpl, is_comparative=True))
    return result


# ===========================================================================
# V2 Verification (Browser + Python Tools)
# ===========================================================================

def verify_geophysical_v2(qa_pairs, num_samples=10, temperature=1.0,
                            max_iterations=200, threshold=0.5,
                            outfile_prefix=None, subarea="",
                            bench_label="geophysical_bench"):
    """V2 verification using Browser + Python tools."""
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
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available for geophysical V2")
    if not _HAS_PYTHON_TOOL:
        raise RuntimeError("HybridPythonTool not available for geophysical V2")

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== GEOPHYSICAL V2 VERIFICATION ({subarea}) ===", flush=True)
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
                    Message.from_role_and_content(Role.DEVELOPER, GEOPHYSICAL_V2_DEVELOPER),
                    Message.from_role_and_content(Role.USER, f"Question: {qa['question']}"),
                ]

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
                        error = Message.from_role_and_content(
                            Role.SYSTEM, f"Unknown tool: {recipient}"
                        )
                        results.append(error)
                    return results

                result_messages = generator.generate_agentic_response_sync(
                    messages, _tool_handler,
                    tool_prefix=("browser.", "python"),
                    tool_configs=[browser_tool.tool_config, python_tool.tool_config],
                    max_iterations=max_iterations,
                    temperature=temperature,
                )

                iteration_count = len(result_messages) - len(messages)
                answer = _extract_geophysical_answer(result_messages)
                answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
                answer = ' '.join(answer.split())
                correct = bool(is_geophysical_answer_correct(answer, qa['gold_answer']) or
                               _llm_judge_geophysical_answer(answer, qa['gold_answer'], qa['question']))
                log = (f"    V2 sample {i+1}/{num_samples}: '{answer}' "
                       f"(gold: '{qa['gold_answer']}') tools={tool_call_counter[0]} "
                       f"iters={iteration_count}")
                return {"answer": answer, "tool_calls": tool_call_counter[0],
                        "iterations": iteration_count, "error": None,
                        "correct": correct, "log": log}

            except Exception as e:
                import traceback
                error_str = f"{type(e).__name__}: {e}"
                log = (f"    V2 sample {i+1}/{num_samples} error: {error_str}\n"
                       + traceback.format_exc())
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


def _extract_geophysical_answer(messages):
    """Extract the final answer from an agentic response."""
    for msg in reversed(messages):
        if hasattr(msg, 'author') and msg.author.role == Role.ASSISTANT:
            from .multiskill_utils import unwrap_content
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
# Per-Category Orchestrator
# ===========================================================================

def _generate_category_qa_pairs(category, per_category, num_chains,
                                  questions_per_entity, agent_info,
                                  prefix, bench_label):
    """Generate QA pairs for a single entity category.

    Fetches entities, their quantitative properties and KG chains,
    then generates single-entity + comparative geophysical questions.
    """
    # Check chains cache
    chains_cache = f"{prefix}__{category}.geophysical_chains.json"
    if os.path.exists(chains_cache):
        print(f"Loading cached chains for {category}...", flush=True)
        with open(chains_cache) as f:
            entity_data_list = json.load(f)

        # Backfill Wikipedia articles
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
            print(f"Re-saved chains cache with Wikipedia articles for {category}", flush=True)
    else:
        entities = fetch_geophysical_entities(category, target_count=per_category * 2)
        if not entities:
            return []

        # Determine which properties to fetch for this category
        required_props = GEOPHYSICAL_THEMES[category]["required_wikidata_props"]
        # Always fetch P2044, P625, P2048 for maximal template compatibility
        all_props = list(set(required_props + ["P2044", "P625", "P2048"]))

        entity_data_list = []
        for ent in tqdm.tqdm(entities, desc=f"Fetching {category} entity data"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            quant_props = fetch_quantitative_properties(ent["id"], properties=all_props)
            if not quant_props:
                continue

            # Check minimum required properties
            if not all(p in quant_props for p in required_props):
                continue

            chains = fetch_multihop_triples(ent["id"], num_hops=2, limit=num_chains)
            if not chains or len(chains) < 3:
                continue

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

            if len(entity_data_list) >= per_category:
                break

        # Save chains cache
        if entity_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Cached {len(entity_data_list)} entity data for {category}", flush=True)

    print(f"Category {category}: {len(entity_data_list)} entities available", flush=True)

    # Generate QA pairs
    all_qa_pairs = []
    prev_questions = []
    used_templates = set()

    # Single-entity questions
    target_single = per_category * 2 // 3
    for ed in entity_data_list:
        if len([q for q in all_qa_pairs if q.get("template_type") != "comparative"]) >= target_single:
            break

        quant_props = _deserialize_quant_props(ed["quant_props"])
        entity = {"id": ed["id"], "label": ed["label"], "quant_props": quant_props}

        qa_pairs = gen_geophysical_qa_from_entity(
            entity, ed["chains"], agent_info,
            num_questions=questions_per_entity,
            prev_questions=prev_questions,
            used_templates=used_templates,
            articles=ed.get("articles", []),
            category=category,
        )

        for qa in qa_pairs:
            qa["category"] = category
        all_qa_pairs.extend(qa_pairs)
        print(f"  {ed['label']}: generated {len(qa_pairs)} QA pairs "
              f"(total: {len(all_qa_pairs)}/{per_category})", flush=True)

    # Comparative questions (for templates that require 2 entities)
    target_comp = per_category // 3
    comp_used = set()
    for i in range(0, len(entity_data_list) - 1, 2):
        if len([q for q in all_qa_pairs if q.get("template_type") == "comparative"]) >= target_comp:
            break

        ed_a = entity_data_list[i]
        ed_b = entity_data_list[i + 1]

        qp_a = _deserialize_quant_props(ed_a["quant_props"])
        qp_b = _deserialize_quant_props(ed_b["quant_props"])

        entity_a = {"id": ed_a["id"], "label": ed_a["label"], "quant_props": qp_a}
        entity_b = {"id": ed_b["id"], "label": ed_b["label"], "quant_props": qp_b}

        # Retry a rejected pair a few times: comparative composition is
        # one-shot per call, so a single stochastic Phase-3 rejection would
        # otherwise discard a viable pair (single-entity gen gets
        # num_questions*3 attempts; give each comparative pair a few too).
        qa = None
        for _ in range(3):
            qa = gen_geophysical_qa_comparative(
                entity_a, entity_b,
                ed_a["chains"], ed_b["chains"],
                ed_a.get("articles", []), ed_b.get("articles", []),
                agent_info, comp_used, prev_questions, category,
            )
            if qa:
                break
        if qa:
            all_qa_pairs.append(qa)
            prev_questions.append(qa["question"])
            print(f"  {ed_a['label']} vs {ed_b['label']}: comparative QA generated "
                  f"(total: {len(all_qa_pairs)})", flush=True)

    return all_qa_pairs


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_geophysical_bench(args, agent_info):
    """Run the geophysical benchmark pipeline.

    Per category:
    1. Fetch entities with quantitative properties + KG chains
    2. Generate multi-step geophysical QA pairs (Phases 0-3)
    3. V1 verification (closed-book)
    4. V2 verification (Browser + Python tools)
    5. Merge all categories into final output (with diversity filter)
    """
    prefix = args.outfile_prefix1
    run_id = getattr(args, "run_id", None) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{prefix}__{run_id}"
    categories_arg = args.geophysical_categories
    per_category = args.geophysical_per_category
    v1_samples = args.geophysical_v1_samples
    v2_samples = args.geophysical_v2_samples
    v1_threshold = args.geophysical_v1_threshold
    v2_threshold = args.geophysical_v2_threshold
    num_chains = args.geophysical_num_chains
    questions_per_entity = args.geophysical_questions_per_entity

    bench_label = "geophysical_bench"

    # Parse categories
    if categories_arg:
        categories_to_process = [s.strip() for s in categories_arg.split(",")]
    else:
        categories_to_process = list(GEOPHYSICAL_THEMES.keys())

    is_subset = categories_arg is not None and len(categories_to_process) < len(GEOPHYSICAL_THEMES)

    print(f"=== GEOPHYSICAL BENCH (run_id={run_id}) ===", flush=True)
    print(f"Categories: {categories_to_process}", flush=True)
    print(f"Target QAs per category: {per_category}", flush=True)
    print(f"V1: {v1_samples} samples, threshold < {v1_threshold:.0%}", flush=True)
    print(f"V2: {v2_samples} samples, threshold < {v2_threshold:.0%}", flush=True)
    print("=" * 50, flush=True)

    all_final = []
    summary = {}

    for category in categories_to_process:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Category: {category.upper()} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Ensure output directory exists
        output_dir = os.path.dirname(run_prefix) if "/" in run_prefix else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check full cache (V2 — terminal filter stage)
        v2_cache = f"{run_prefix}__{category}.{bench_label}_v2.json"
        v2_filtered_cache = f"{run_prefix}__{category}.{bench_label}_v2_filtered.json"
        if os.path.exists(v2_cache) and os.path.exists(v2_filtered_cache):
            print(f"Found cached V2 results for {category}, loading...", flush=True)
            with open(v2_cache) as f:
                v2_all = json.load(f)
            with open(v2_filtered_cache) as f:
                v2_filtered = json.load(f)
            all_final.extend(v2_filtered)
            summary[category] = {
                "generated": "cached", "v1_filtered": "cached",
                "v2_total": len(v2_all), "v2_filtered": len(v2_filtered),
            }
            continue

        # Check V1 filtered cache
        v1_filtered_cache = f"{run_prefix}__{category}.{bench_label}_v1_filtered.json"
        if os.path.exists(v1_filtered_cache):
            print(f"Found cached V1 filtered for {category}, skipping to V2...", flush=True)
            with open(v1_filtered_cache) as f:
                v1_filtered = json.load(f)
        else:
            # Check raw bench cache
            bench_cache = f"{run_prefix}__{category}.{bench_label}.json"
            if os.path.exists(bench_cache):
                print(f"Found cached bench problems for {category}...", flush=True)
                with open(bench_cache) as f:
                    qa_pairs = json.load(f)
            else:
                # Step 1: Generate QA pairs
                qa_pairs = _generate_category_qa_pairs(
                    category, per_category, num_chains,
                    questions_per_entity, agent_info, prefix, bench_label,
                )

                # Save raw bench cache
                if qa_pairs:
                    with open(bench_cache, "w") as f:
                        json.dump(qa_pairs, f, indent=2)
                    print(f"Saved {len(qa_pairs)} raw QA pairs to {bench_cache}", flush=True)

            if not qa_pairs:
                print(f"No QA pairs for category {category}", flush=True)
                summary[category] = {"generated": 0, "v1_filtered": 0,
                                     "v2_total": 0, "v2_filtered": 0}
                continue

            # Step 2: V1 verification
            v1_filtered, v1_all = verify_bench_v1(
                qa_pairs, agent_info,
                num_samples=v1_samples,
                temperature=0.7,
                threshold=v1_threshold,
                outfile_prefix=run_prefix,
                subarea=category,
                bench_label=bench_label,
                answer_checker=is_geophysical_answer_correct,
                llm_judge=_llm_judge_geophysical_answer,
                verification_prompt=GEOPHYSICAL_V1_PROMPT,
            )

        if not v1_filtered:
            print(f"No V1-filtered QA pairs for {category}", flush=True)
            summary[category] = {
                "generated": len(qa_pairs) if 'qa_pairs' in dir() else "cached",
                "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0,
            }
            continue

        # Step 3: V2 verification (browser + Python tools)
        v2_filtered, v2_all = verify_geophysical_v2(
            v1_filtered,
            num_samples=v2_samples,
            temperature=1.0,
            max_iterations=200,
            threshold=v2_threshold,
            outfile_prefix=run_prefix,
            subarea=category,
            bench_label=bench_label,
        )

        all_final.extend(v2_filtered)
        summary[category] = {
            "generated": len(v1_all) if 'v1_all' in dir() else "cached",
            "v1_filtered": len(v1_filtered) if 'v1_filtered' in dir() else "cached",
            "v2_total": len(v2_all),
            "v2_filtered": len(v2_filtered),
        }

    # Skip final merge when processing a subset (worker job)
    if is_subset:
        print(f"\nSubset mode: processed {categories_to_process}. "
              f"Merge will happen in the merge job.", flush=True)
        _print_summary(summary)
        return all_final

    # Final merge
    final_path = f"{run_prefix}.{bench_label}_final.json"
    if all_final:
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
    print("GEOPHYSICAL BENCH SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for category, stats in summary.items():
        print(f"  {category:20s}: generated={stats.get('generated','?'):>6} "
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
        prog="geophysical_drbencher",
        description="Geophysical Benchmark: multi-step physics reasoning with Wikidata entity data",
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
    parser.add_argument("--use_helm", type=str, default="no")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Output
    parser.add_argument("--outfile_prefix1", type=str,
                        default="output/geophysical/bench")

    # Exp mode
    parser.add_argument("--exp_mode", type=str, default="geophysical_bench")

    # Run ID for multi-run support (auto-generated if not provided)
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: auto YYYYMMDD_HHMMSS)")

    # Geophysical-specific
    parser.add_argument("--geophysical_categories", type=str, default=None,
                        help="Comma-separated themes (default: all 14). "
                             "E.g. 'mountains,volcanoes,cities,countries,islands,"
                             "rivers,lakes,deserts,glaciers,buildings,towers,"
                             "bridges,dams,waterfalls'")
    parser.add_argument("--geophysical_per_category", type=int, default=50,
                        help="Target QA pairs per category")
    parser.add_argument("--geophysical_v1_samples", type=int, default=10,
                        help="V1 sampling attempts per question")
    parser.add_argument("--geophysical_v2_samples", type=int, default=10,
                        help="V2 (Browser+Python) sampling attempts per question")
    parser.add_argument("--geophysical_v1_threshold", type=float, default=0.5,
                        help="V1 accuracy ceiling -- keep below this")
    parser.add_argument("--geophysical_v2_threshold", type=float, default=0.5,
                        help="V2 accuracy ceiling -- keep below this")
    parser.add_argument("--geophysical_num_chains", type=int, default=20,
                        help="KG chains per entity")
    parser.add_argument("--geophysical_questions_per_entity", type=int, default=5,
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
    elif args.use_vllm_serve.lower() == "yes":
        from .openai_api_generator import OpenAIAPIGenerator
        if not args.vllm_serve_model:
            raise ValueError("--vllm_serve_model is required when --use_vllm_serve=yes")
        gen = OpenAIAPIGenerator(
            base_url=args.vllm_serve_url,
            model_name=args.vllm_serve_model,
        )
        set_harmony_generator(gen)
        wikidata_set_harmony_generator(gen)

    # Model loading
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
    if args.exp_mode == "geophysical_bench":
        run_geophysical_bench(args, agent_info)
        # The in-process vLLM engine spawns worker subprocesses that outlive the
        # bench; shut the engine down and terminate the process group so the job
        # exits cleanly instead of hanging until it is killed by hand.
        shutdown_and_exit(0)
    else:
        print(f"Unknown exp_mode: {args.exp_mode}", flush=True)
        sys.exit(1)
