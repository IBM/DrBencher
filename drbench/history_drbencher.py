# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Historical/Causal Benchmark: Wikidata temporal data as ground truth.

Creates questions in a 2-level difficulty space:
  Level 1 (Single-Entity): identify person/event/org from clues -> fetch dates -> compute
  Level 1 (Comparative):   compare two entities' temporal data
  Level 2 (Multi-Step):    multi-step temporal calculations, cross-category pairing
  Level 2 (Superlative):   rank N entities (N=3 or 5) by temporal extremes/aggregates

Data source: Wikidata REST API (birth/death, start/end, inception, point-in-time)
Entity clues: Wikipedia article extracts

Run via:  python -m drbench.history_drbencher --exp_mode history_bench ...
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
    fetch_multihop_triples, fetch_wikipedia_for_entities,
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
from .wikidata_harmony import format_extracted_facts, parse_used_facts
from .multiskill_drbencher import (
    build_multiskill_grounding_articles,
)

# History utilities
from .history_util import (
    get_entity_temporal_data,
    normalize_entity_dates,
    fetch_entity_clues,
    get_category_entities,
    get_all_entity_data,
    parse_wikidata_time,
    verify_temporal_grounding,
    ENTITY_UNIVERSE,
    HISTORY_THEMES,
    TEMPORAL_PROPERTIES,
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

# History tool for V2 verification
try:
    from tools.history_tool import HistoryTool
    _HAS_HISTORY_TOOL = True
except ImportError:
    _HAS_HISTORY_TOOL = False

# Diversity filter (optional)
try:
    from .diversity import diversity_filter, diversity_report
    _HAS_DIVERSITY = True
except ImportError:
    _HAS_DIVERSITY = False


# ===========================================================================
# History Templates (33 single/comparative/cross-category templates)
# ===========================================================================

HISTORY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # --- Level 1: Single-Entity ---
    "lifespan_years": {
        "level": 1,
        "type": "single",
        "entity_type": ["figure"],
        "label": "Lifespan in Years",
        "required_data": ["birth_year", "death_year"],
        "code_template": "print({death_year} - {birth_year})",
        "question_hint": "How many years did this person live?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "duration_years": {
        "level": 1,
        "type": "single",
        "entity_type": ["organization"],
        "label": "Duration of Existence",
        "required_data": ["inception_year", "dissolution_year"],
        "code_template": "print({dissolution_year} - {inception_year})",
        "question_hint": "How many years did this entity exist?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "conflict_duration": {
        "level": 1,
        "type": "single",
        "entity_type": ["conflict"],
        "label": "Conflict Duration",
        "required_data": ["start_year", "end_year"],
        "code_template": "print({end_year} - {start_year})",
        "question_hint": "How many years did this conflict last?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "years_since_founding": {
        "level": 1,
        "type": "single",
        "entity_type": ["organization"],
        "label": "Age at Reference Year",
        "required_data": ["inception_year"],
        "code_template": "print({reference_year} - {inception_year})",
        "question_hint": "How old was this organization in the year {reference_year}?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },

    # --- Level 1: Comparative ---
    "founding_gap": {
        "level": 1,
        "type": "comparative",
        "entity_type": ["organization"],
        "label": "Gap Between Founding Dates",
        "required_data": ["inception_year"],
        "code_template": "print(year_gap({inception_year_a}, {inception_year_b}))",
        "question_hint": "What is the difference in years between the founding dates of these two organizations?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "lifespan_difference": {
        "level": 1,
        "type": "comparative",
        "entity_type": ["figure"],
        "label": "Lifespan Difference",
        "required_data": ["birth_year", "death_year"],
        "code_template": (
            "lifespan_a = {death_year_a} - {birth_year_a}\n"
            "lifespan_b = {death_year_b} - {birth_year_b}\n"
            "print(abs(lifespan_a - lifespan_b))"
        ),
        "question_hint": "What is the difference in lifespan (in years) between these two historical figures?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "duration_comparison": {
        "level": 1,
        "type": "comparative",
        "entity_type": ["conflict"],
        "label": "Duration Difference",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "dur_a = {end_year_a} - {start_year_a}\n"
            "dur_b = {end_year_b} - {start_year_b}\n"
            "print(abs(dur_a - dur_b))"
        ),
        "question_hint": "What is the difference in duration (in years) between these two conflicts?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "contemporaneity_overlap": {
        "level": 1,
        "type": "comparative",
        "entity_type": ["figure", "conflict"],
        "label": "Temporal Overlap",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "overlap = max(0, min({end_year_a}, {end_year_b}) - max({start_year_a}, {start_year_b}))\n"
            "print(overlap)"
        ),
        "question_hint": "How many years of temporal overlap existed between these two entities?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },

    # --- Level 2: Multi-Step ---
    "age_at_event": {
        "level": 2,
        "type": "cross_category",
        "entity_type": ["figure"],
        "cross_type": ["conflict", "milestone"],
        "label": "Age at Event",
        "required_data": ["birth_year"],
        "cross_required_data": ["event_year"],
        "code_template": "print({event_year} - {birth_year})",
        "question_hint": "How old was this historical figure when this event occurred?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "midpoint_year": {
        "level": 2,
        "type": "single",
        "entity_type": ["conflict"],
        "label": "Midpoint Year",
        "required_data": ["start_year", "end_year"],
        "code_template": "print(({start_year} + {end_year}) // 2)",
        "question_hint": "What year is the midpoint of this conflict?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "era_span_decades": {
        "level": 2,
        "type": "single",
        "entity_type": ["conflict"],
        "label": "Full Decades Spanned",
        "required_data": ["start_year", "end_year"],
        "code_template": "print(({end_year} - {start_year}) // 10)",
        "question_hint": "How many full decades did this conflict span?",
        "answer_unit": "decades",
        "reasoning_depth": 3,
    },
    "founding_ratio": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["organization"],
        "label": "Age Ratio at Reference Year",
        "required_data": ["inception_year"],
        "code_template": (
            "age_a = {reference_year} - {inception_year_a}\n"
            "age_b = {reference_year} - {inception_year_b}\n"
            "print(round(age_a / age_b, 2))"
        ),
        "question_hint": "What is the ratio of the ages of these two organizations in the year {reference_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 3,
    },
    "combined_duration": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["conflict"],
        "label": "Combined Duration",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "dur_a = {end_year_a} - {start_year_a}\n"
            "dur_b = {end_year_b} - {start_year_b}\n"
            "print(dur_a + dur_b)"
        ),
        "question_hint": "What is the combined duration (in years) of these two conflicts?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "years_between_milestones": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["milestone"],
        "label": "Years Between Milestones",
        "required_data": ["event_year"],
        "code_template": "print(year_gap({event_year_a}, {event_year_b}))",
        "question_hint": "How many years elapsed between these two historical milestones?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "generation_count": {
        "level": 2,
        "type": "single",
        "entity_type": ["conflict", "organization"],
        "label": "Generation Count",
        "required_data": ["start_year", "end_year"],
        "code_template": "print(({end_year} - {start_year}) // {generation_years})",
        "question_hint": "Assuming a generation is {generation_years} years, how many full generations spanned the existence of this entity?",
        "answer_unit": "generations",
        "reasoning_depth": 3,
    },
    # --- Additional Level 2 ---
    "succession_gap": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["organization"],
        "label": "Succession Gap Between Empires/Organizations",
        "required_data": ["inception_year", "dissolution_year"],
        "code_template": (
            "end_a = {dissolution_year_a}\n"
            "start_b = {inception_year_b}\n"
            "gap = year_gap(end_a, start_b)\n"
            "print(gap)"
        ),
        "question_hint": "How many years elapsed between the end of the first entity and the founding of the second?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "reign_length": {
        "level": 1,
        "type": "single",
        "entity_type": ["organization"],
        "label": "Duration of Existence in Decades",
        "required_data": ["inception_year", "dissolution_year"],
        "code_template": (
            "duration = {dissolution_year} - {inception_year}\n"
            "decades = duration / 10\n"
            "print(round(decades, 1))"
        ),
        "question_hint": "How many decades did this entity exist?",
        "answer_unit": "decades",
        "reasoning_depth": 2,
    },
    "battle_interval": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["conflict"],
        "label": "Gap Between Conflicts",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "end_a = {end_year_a}\n"
            "start_b = {start_year_b}\n"
            "gap = year_gap(end_a, start_b)\n"
            "print(gap)"
        ),
        "question_hint": "How many years of peace separated the end of the first conflict from the start of the second?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "diplomatic_gap": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["milestone"],
        "label": "Time Between Milestones",
        "required_data": ["event_year"],
        "code_template": (
            "gap = year_gap({event_year_a}, {event_year_b})\n"
            "decades = gap / 10\n"
            "print(round(decades, 1))"
        ),
        "question_hint": "How many decades separate these two historical milestones?",
        "answer_unit": "decades",
        "reasoning_depth": 3,
    },
    "empire_peak_duration": {
        "level": 2,
        "type": "single",
        "entity_type": ["organization"],
        "label": "Duration in Centuries",
        "required_data": ["inception_year", "dissolution_year"],
        "code_template": (
            "duration = {dissolution_year} - {inception_year}\n"
            "centuries = duration / 100\n"
            "print(round(centuries, 2))"
        ),
        "question_hint": "How many centuries did this entity endure?",
        "answer_unit": "centuries",
        "reasoning_depth": 2,
    },
    "revolution_to_constitution_gap": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["conflict"],
        "cross_type": ["milestone"],
        "label": "Years from Revolution to Constitutional Event",
        "required_data": ["start_year"],
        "cross_required_data": ["event_year"],
        "code_template": "print(year_gap({start_year}, {event_year}))",
        "question_hint": "How many years elapsed between the start of this revolution and this constitutional milestone?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "territorial_change_years": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["organization"],
        "label": "Overlap Duration Between Two Empires",
        "required_data": ["inception_year", "dissolution_year"],
        "code_template": (
            "start_a = {inception_year_a}\nend_a = {dissolution_year_a}\n"
            "start_b = {inception_year_b}\nend_b = {dissolution_year_b}\n"
            "overlap = max(0, min(end_a, end_b) - max(start_a, start_b))\n"
            "print(overlap)"
        ),
        "question_hint": "How many years did these two entities coexist?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "age_at_founding": {
        "level": 2,
        "type": "cross_category",
        "entity_type": ["figure"],
        "cross_type": ["organization"],
        "label": "Age When Organization Was Founded",
        "required_data": ["birth_year"],
        "cross_required_data": ["inception_year"],
        "code_template": "print({inception_year} - {birth_year})",
        "question_hint": "How old was this person when this organization was founded?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "conflict_as_fraction_of_life": {
        "level": 2,
        "type": "cross_category",
        "entity_type": ["figure"],
        "cross_type": ["conflict"],
        "label": "Conflict Duration as Fraction of Lifespan",
        "required_data": ["birth_year", "death_year"],
        "cross_required_data": ["start_year", "end_year"],
        "code_template": (
            "lifespan = {death_year} - {birth_year}\n"
            "conflict_dur = {end_year} - {start_year}\n"
            "fraction = conflict_dur / lifespan * 100\n"
            "print(round(fraction, 2))"
        ),
        "question_hint": "What percentage of this person's lifespan was occupied by this conflict?",
        "answer_unit": "%",
        "reasoning_depth": 4,
    },
    "century_of_event": {
        "level": 1,
        "type": "single",
        "entity_type": ["milestone", "conflict", "organization"],
        "label": "Century of Event",
        "required_data": ["start_year"],
        "code_template": (
            "import math\n"
            "year = {start_year}\n"
            "century = math.ceil(year / 100) if year > 0 else math.ceil(abs(year) / 100)\n"
            "print(century)"
        ),
        "question_hint": "In which century (as an ordinal number) did this entity originate?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    # --- New templates for yield improvement ---
    "years_to_next_century": {
        "level": 2,
        "type": "single",
        "entity_type": ["milestone", "conflict"],
        "label": "Years to Next Century Boundary",
        "required_data": ["start_year"],
        "code_template": (
            "year = {start_year}\n"
            "next_cent = (year // 100 + 1) * 100\n"
            "print(next_cent - year)"
        ),
        "question_hint": "How many years separated this event from the start of the next century?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "age_at_death_decade": {
        "level": 2,
        "type": "single",
        "entity_type": ["figure"],
        "label": "Age at Death in Full Decades",
        "required_data": ["birth_year", "death_year"],
        "code_template": (
            "age = {death_year} - {birth_year}\n"
            "print(age // 10)"
        ),
        "question_hint": "How many full decades did this person live?",
        "answer_unit": "decades",
        "reasoning_depth": 3,
    },
    "years_active_before_event": {
        "level": 2,
        "type": "cross_category",
        "entity_type": ["organization"],
        "cross_type": ["milestone", "conflict"],
        "label": "Organization Age at Event",
        "required_data": ["inception_year"],
        "cross_required_data": ["event_year"],
        "code_template": "print({event_year} - {inception_year})",
        "question_hint": "How many years had this organization existed when this event occurred?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "fraction_of_century": {
        "level": 2,
        "type": "single",
        "entity_type": ["conflict"],
        "label": "Duration as Percentage of a Century",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "dur = {end_year} - {start_year}\n"
            "print(round(dur / 100 * 100, 1))"
        ),
        "question_hint": "What percentage of a century did this conflict span?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "lifetime_overlap_with_century": {
        "level": 2,
        "type": "single",
        "entity_type": ["figure"],
        "label": "Lifetime Overlap with Birth Century",
        "required_data": ["birth_year", "death_year"],
        "code_template": (
            "import math\n"
            "cent_start = ({birth_year} // 100) * 100 + 1\n"
            "cent_end = cent_start + 99\n"
            "overlap = min({death_year}, cent_end) - max({birth_year}, cent_start)\n"
            "print(max(0, overlap))"
        ),
        "question_hint": "How many years of this person's life fell within the century in which they were born?",
        "answer_unit": "years",
        "reasoning_depth": 4,
    },
    "avg_duration_pair": {
        "level": 2,
        "type": "comparative",
        "entity_type": ["conflict"],
        "label": "Average Duration of Two Conflicts",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "d1 = {end_year_a} - {start_year_a}\n"
            "d2 = {end_year_b} - {start_year_b}\n"
            "print(round((d1 + d2) / 2, 1))"
        ),
        "question_hint": "What is the average duration (in years) of these two conflicts?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "birth_gap": {
        "level": 1,
        "type": "comparative",
        "entity_type": ["figure"],
        "label": "Birth Year Gap",
        "required_data": ["birth_year"],
        "code_template": "print(year_gap({birth_year_a}, {birth_year_b}))",
        "question_hint": "How many years apart were these two historical figures born?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "event_to_dissolution": {
        "level": 2,
        "type": "cross_category",
        "entity_type": ["organization"],
        "cross_type": ["milestone"],
        "label": "Years from Milestone to Dissolution",
        "required_data": ["dissolution_year"],
        "cross_required_data": ["event_year"],
        "code_template": "print(year_gap({event_year}, {dissolution_year}))",
        "question_hint": "How many years separated this milestone from the dissolution of this organization?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    # --- Level 3: Multi-step compound templates ---
    "lifespan_per_century_fraction": {
        "level": 3,
        "type": "single",
        "entity_type": ["figure"],
        "label": "Lifespan as Fraction of Birth-Century Remaining",
        "required_data": ["birth_year", "death_year"],
        "code_template": (
            "import math\n"
            "birth = {birth_year}\n"
            "death = {death_year}\n"
            "lifespan = death - birth\n"
            "century_end = (birth // 100 + 1) * 100\n"
            "remaining = century_end - birth\n"
            "print(round(lifespan / remaining, 2))"
        ),
        "question_hint": "What is the ratio of this person's lifespan to the number of years remaining in their birth century?",
        "answer_unit": "ratio",
        "reasoning_depth": 4,
    },
    "conflict_midpoint_to_century_end": {
        "level": 3,
        "type": "single",
        "entity_type": ["conflict"],
        "label": "Years from Conflict Midpoint to Century End",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "start = {start_year}\n"
            "end = {end_year}\n"
            "midpoint = (start + end) / 2\n"
            "century_end = (int(midpoint) // 100 + 1) * 100\n"
            "print(round(century_end - midpoint, 1))"
        ),
        "question_hint": "How many years separated the midpoint of this conflict from the end of that century?",
        "answer_unit": "years",
        "reasoning_depth": 4,
    },
    "founding_age_ratio_at_event": {
        "level": 3,
        "type": "cross_category",
        "entity_type": ["organization"],
        "cross_type": ["milestone", "conflict"],
        "label": "Organization-Age to Event-Century Ratio",
        "required_data": ["inception_year"],
        "cross_required_data": ["event_year"],
        "code_template": (
            "import math\n"
            "age = {event_year} - {inception_year}\n"
            "century = math.ceil({event_year} / 100)\n"
            "print(round(age / century, 2))"
        ),
        "question_hint": "What is the ratio of this organization's age at the time of this event to the century number in which the event occurred?",
        "answer_unit": "ratio",
        "reasoning_depth": 4,
    },
    "lifespan_geometric_mean": {
        "level": 3,
        "type": "comparative",
        "entity_type": ["figure"],
        "label": "Geometric Mean of Two Lifespans",
        "required_data": ["birth_year", "death_year"],
        "code_template": (
            "import math\n"
            "ls_a = {death_year_a} - {birth_year_a}\n"
            "ls_b = {death_year_b} - {birth_year_b}\n"
            "print(round(math.sqrt(ls_a * ls_b), 2))"
        ),
        "question_hint": "What is the geometric mean of the lifespans of these two historical figures?",
        "answer_unit": "years",
        "reasoning_depth": 4,
    },
    "conflict_duration_ratio": {
        "level": 3,
        "type": "comparative",
        "entity_type": ["conflict"],
        "label": "Duration Ratio of Two Conflicts",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "d_a = {end_year_a} - {start_year_a}\n"
            "d_b = {end_year_b} - {start_year_b}\n"
            "ratio = max(d_a, d_b) / min(d_a, d_b)\n"
            "print(round(ratio, 2))"
        ),
        "question_hint": "What is the ratio of the longer conflict's duration to the shorter one?",
        "answer_unit": "ratio",
        "reasoning_depth": 4,
    },
    "age_at_event_as_percent_of_lifespan": {
        "level": 3,
        "type": "cross_category",
        "entity_type": ["figure"],
        "cross_type": ["conflict", "milestone"],
        "label": "Age at Event as Percentage of Lifespan",
        "required_data": ["birth_year", "death_year"],
        "cross_required_data": ["event_year"],
        "code_template": (
            "age = {event_year} - {birth_year}\n"
            "lifespan = {death_year} - {birth_year}\n"
            "print(round(age / lifespan * 100, 2))"
        ),
        "question_hint": "What percentage of this person's life had elapsed when this event occurred?",
        "answer_unit": "%",
        "reasoning_depth": 4,
    },
    "combined_overlap_fraction": {
        "level": 3,
        "type": "comparative",
        "entity_type": ["organization", "conflict"],
        "label": "Overlap as Fraction of Combined Duration",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "s_a, e_a = {start_year_a}, {end_year_a}\n"
            "s_b, e_b = {start_year_b}, {end_year_b}\n"
            "overlap = max(0, min(e_a, e_b) - max(s_a, s_b))\n"
            "total = (e_a - s_a) + (e_b - s_b)\n"
            "print(round(overlap / total * 100, 2))"
        ),
        "question_hint": "What percentage of their combined duration did these two entities overlap?",
        "answer_unit": "%",
        "reasoning_depth": 5,
    },
    "harmonic_mean_durations": {
        "level": 3,
        "type": "comparative",
        "entity_type": ["conflict", "organization"],
        "label": "Harmonic Mean of Two Durations",
        "required_data": ["start_year", "end_year"],
        "code_template": (
            "d_a = {end_year_a} - {start_year_a}\n"
            "d_b = {end_year_b} - {start_year_b}\n"
            "hm = 2 * d_a * d_b / (d_a + d_b)\n"
            "print(round(hm, 2))"
        ),
        "question_hint": "What is the harmonic mean of the durations of these two entities?",
        "answer_unit": "years",
        "reasoning_depth": 4,
    },
}


# ===========================================================================
# Superlative (N-Entity Ranking) Templates (12 templates)
# ===========================================================================

HISTORY_SUPERLATIVE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "earliest_founding": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization"],
        "label": "Earliest Founding Year",
        "required_data": ["inception_year"],
        "value_key": "inception_year",
        "code_template": "print(min(vals))",
        "question_hint": "Which of these organizations was founded earliest, and in what year?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "latest_founding": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization"],
        "label": "Latest Founding Year",
        "required_data": ["inception_year"],
        "value_key": "inception_year",
        "code_template": "print(max(vals))",
        "question_hint": "Which of these organizations was founded most recently, and in what year?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "earliest_event": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["milestone"],
        "label": "Earliest Event Year",
        "required_data": ["event_year"],
        "value_key": "event_year",
        "code_template": "print(min(vals))",
        "question_hint": "Which of these events occurred earliest, and in what year?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "latest_event": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["milestone"],
        "label": "Latest Event Year",
        "required_data": ["event_year"],
        "value_key": "event_year",
        "code_template": "print(max(vals))",
        "question_hint": "Which of these events occurred most recently, and in what year?",
        "answer_unit": "",
        "reasoning_depth": 2,
    },
    "longest_duration": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization", "conflict"],
        "label": "Longest Duration",
        "required_data": ["start_year", "end_year"],
        "value_key": "_duration",
        "code_template": "print(max(vals))",
        "question_hint": "Which of these entities lasted the longest, and for how many years?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "shortest_duration": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization", "conflict"],
        "label": "Shortest Duration",
        "required_data": ["start_year", "end_year"],
        "value_key": "_duration",
        "code_template": "print(min(vals))",
        "question_hint": "Which of these entities lasted the shortest, and for how many years?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "longest_lifespan": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["figure"],
        "label": "Longest Lifespan",
        "required_data": ["birth_year", "death_year"],
        "value_key": "_lifespan",
        "code_template": "print(max(vals))",
        "question_hint": "Which of these figures lived the longest, and for how many years?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "shortest_lifespan": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["figure"],
        "label": "Shortest Lifespan",
        "required_data": ["birth_year", "death_year"],
        "value_key": "_lifespan",
        "code_template": "print(min(vals))",
        "question_hint": "Which of these figures lived the shortest, and for how many years?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "duration_spread": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization", "conflict"],
        "label": "Duration Spread (Max - Min)",
        "required_data": ["start_year", "end_year"],
        "value_key": "_duration",
        "code_template": "print(max(vals) - min(vals))",
        "question_hint": "What is the difference in duration between the longest and shortest of these entities?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "avg_duration_n": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization", "conflict"],
        "label": "Average Duration of N Entities",
        "required_data": ["start_year", "end_year"],
        "value_key": "_duration",
        "code_template": "print(round(sum(vals) / len(vals), 1))",
        "question_hint": "What is the average duration (in years) of these entities?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "median_duration_n": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization", "conflict"],
        "label": "Median Duration of N Entities",
        "required_data": ["start_year", "end_year"],
        "value_key": "_duration",
        "code_template": "print(sorted(vals)[len(vals) // 2])",
        "question_hint": "What is the median duration (in years) of these entities?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
    "founding_spread": {
        "level": 2,
        "type": "superlative",
        "n_entities": [3, 5],
        "entity_type": ["organization"],
        "label": "Founding Year Spread",
        "required_data": ["inception_year"],
        "value_key": "inception_year",
        "code_template": "print(max(vals) - min(vals))",
        "question_hint": "How many years separate the earliest and latest founding dates among these organizations?",
        "answer_unit": "years",
        "reasoning_depth": 3,
    },
}


# ===========================================================================
# Prompts
# ===========================================================================

HISTORY_QA_DEVELOPER = (
    "You are a historian creating research questions that test "
    "the ability to identify historical entities (people, conflicts, "
    "organizations, events) from descriptions and perform temporal "
    "calculations with their dates."
)

HISTORY_QA_PROMPT = """Compose a historical research question that:
1. Uses clue facts to describe an unnamed historical entity — readers must figure out which one
2. Then asks for a specific temporal computation requiring the solver to look up the entity's dates

CLUE FACTS (about the unnamed entity — do NOT name it directly):
{facts_text}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- VERIFIED TEMPORAL DATA (from Wikidata — use as ground truth for consistency) ---
{temporal_data_text}
The computation uses EXACTLY these date values. Your clue facts must be consistent with
these dates. Do NOT introduce temporal claims that contradict these verified dates.
For example, if the verified start year is 1337, do NOT describe the entity as "beginning
in the 14th century's first decade" or "starting in the 1200s". Your clues should be
about non-temporal attributes (geography, participants, outcomes, etc.).

--- STYLE GUIDANCE ---
- Write 2-4 sentences total.
- First 1-2 sentences: describe the entity using 3+ clue facts from different topics, without naming it directly. Do NOT use entity names, Wikidata IDs, or specific year numbers.
- Last 1-2 sentences: pose the temporal computation question.
- CRITICAL: Do NOT reveal ANY year values, dates, or temporal data in the question — the solver must look up ALL dates themselves.
- CRITICAL: Do NOT name the entity directly anywhere in the question. Use descriptive clues only.
- Sound natural and conversational, like a real historical research question.
- Include all necessary parameters for the computation (e.g., reference year, generation length) — these are NOT entity-specific data.
- CRITICAL: The question MUST ask EXACTLY what the hint describes. Do NOT rephrase or add extra arithmetic steps (e.g., if the hint says "In which century?", ask "In which century?" — do NOT ask "subtract X from Y, then find the century" or "how many years between X and Y").
- CRITICAL: The temporal question must be about the TARGET ENTITY itself, not about a location it is named after. If the entity is a battle or conflict, ask about when the battle/conflict occurred — do NOT ask about when the associated city was founded or established as a municipality.
- CRITICAL: Use ONLY the provided clue facts for entity description. Do NOT add historical claims from your own knowledge that might contradict the verified temporal data.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your historical question>
Used_Facts: <comma-separated fact_ids used, e.g. W1, W2, W3>
Reasoning: <brief chain: clues identify entity -> look up dates -> computation gives answer>
"""

HISTORY_COMPARATIVE_QA_PROMPT = """Compose a comparative historical research question that:
1. Describes TWO unnamed historical entities (people, conflicts, organizations, or events) using clue facts about each
2. Asks for a specific temporal comparison requiring the solver to look up dates for both

CLUE FACTS FOR ENTITY A (do NOT name this entity directly):
{facts_text_a}

CLUE FACTS FOR ENTITY B (do NOT name this entity directly):
{facts_text_b}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- VERIFIED TEMPORAL DATA (from Wikidata — use as ground truth for consistency) ---
Entity A: {temporal_data_text_a}
Entity B: {temporal_data_text_b}
The computation uses EXACTLY these date values. Your clue facts must be consistent with
these dates. Do NOT introduce temporal claims that contradict these verified dates.
Use ONLY the provided clue facts for entity descriptions — do NOT add historical claims
from your own knowledge.

--- STYLE GUIDANCE ---
- Write 4-6 sentences total.
- First 1-2 sentences: describe the first entity using 3+ clue facts from different topics, without naming it directly. Do NOT use entity names, Wikidata IDs, or specific year numbers.
- Next 1-2 sentences: describe the second entity using 3+ clue facts from different topics, without naming it directly. Do NOT use entity names, Wikidata IDs, or specific year numbers.
- Last 1-2 sentences: pose the temporal comparison question.
- Refer to them as "the first entity" and "the second entity" (or similar distinct references).
- CRITICAL: Do NOT reveal ANY year values, dates, or temporal data in the question — the solver must look up ALL dates themselves.
- CRITICAL: Do NOT name either entity directly anywhere in the question. Use descriptive clues only.
- Sound natural and conversational, like a real historical research question.
- Include all necessary parameters for the computation (e.g., reference year, generation length) — these are NOT entity-specific data.
- CRITICAL: The question MUST ask EXACTLY what the hint describes. Do NOT rephrase or add extra arithmetic steps (e.g., if the hint says "difference in birth years", ask exactly that — do NOT ask "subtract X from Y, then find the century" or "how many years between X and Y").
- CRITICAL: Use ONLY the provided clue facts for entity description. Do NOT add historical claims from your own knowledge that might contradict the verified temporal data.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your comparative historical question>
Used_Facts_A: <comma-separated fact_ids for entity A, e.g. A_C1_F1, A_C2_F1>
Used_Facts_B: <comma-separated fact_ids for entity B, e.g. B_C1_F1, B_C2_F1>
Reasoning: <brief chain: clues identify both entities -> look up dates -> comparison gives answer>

REQUIREMENTS:
- Use 3+ clue facts per entity from different chains
- Do NOT name either entity in the question
- Do NOT include year values or dates in the question
- Every claim must come VERBATIM from the listed facts
- The question MUST ask EXACTLY what the hint describes.
- The question MUST have exactly ONE unambiguous answer"""

HISTORY_V1_PROMPT = """Solve this historical research question. It requires:
1. Identifying a historical entity (person, conflict, organization, or event) from its description
2. Looking up its temporal data (dates, years)
3. Performing the requested computation

Provide ONLY the final numerical answer (with units if applicable).
Do not show your work.

Question: {question}

Answer:"""

HISTORY_V2_DEVELOPER = """You are an expert historical research assistant.
You have THREE tools available:
- **History tool**: Query Wikidata for temporal data (dates, years)
  - history.search_entity(query) — search for entities by name
  - history.get_entity_dates(qid) — get birth/death/start/end dates
  - history.get_entity_info(qid) — get label, description, and dates
  - history.compare_entities(qid_a, qid_b) — compare two entities' dates
- **Browser tool**: Search Wikipedia for entity identification
- **Python tool**: Execute calculations (math, numpy available)

Recommended approach:
1. Use the browser to search for and identify the historical entity from the description clues
2. Use the history tool to look up temporal data (dates, years)
3. Use Python for computation
Give your final answer as a single number (with units if applicable) on the last line."""

HISTORY_SUPERLATIVE_QA_DEVELOPER = (
    "You are a historian creating research questions that test "
    "the ability to identify MULTIPLE historical entities from descriptions "
    "and compare their temporal data to find extremes or aggregates."
)

HISTORY_SUPERLATIVE_QA_PROMPT = """Compose a historical ranking question that:
1. Describes {n_entities} unnamed historical entities using clue facts
2. Asks a ranking or aggregate question about their temporal properties

{entity_clue_blocks}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- VERIFIED TEMPORAL DATA (from Wikidata — use as ground truth for consistency) ---
{temporal_data_text}
The computation uses EXACTLY these date values. Your clue facts must be consistent with
these dates. Do NOT introduce temporal claims that contradict these verified dates.
Use ONLY the provided clue facts for entity descriptions — do NOT add historical claims
from your own knowledge.

--- STYLE GUIDANCE ---
- Write {sentence_range} sentences total.
- Describe each entity in 1-2 sentences using clue facts from different topics, without naming it directly.
- Refer to them as "the first [type]", "the second [type]", "the third [type]", etc.
- Last 1-2 sentences: pose the ranking/aggregate question.
- CRITICAL: Do NOT reveal ANY year values, dates, or temporal data in the question — the solver must look up ALL dates themselves.
- CRITICAL: Do NOT name ANY entity directly anywhere in the question. Use descriptive clues only.
- Sound natural and conversational, like a real historical research question.
- Use EXACT terminology from source facts.
- Do NOT mention "Wikidata", "KG", "entity", or "property".
- CRITICAL: Use ONLY the provided clue facts for entity description. Do NOT add historical claims from your own knowledge that might contradict the verified temporal data.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your historical ranking question>
{used_facts_format}
Reasoning: <brief chain: clues identify each entity -> look up dates -> computation gives answer>

REQUIREMENTS:
- Select 2-3 clue facts from different chains per entity
- Do NOT name any entity in the question
- Do NOT include year values or temporal data in the question
- Every claim must come VERBATIM from the listed facts
- The question MUST have exactly ONE unambiguous answer"""


# ===========================================================================
# Answer Checker
# ===========================================================================

def is_history_answer_correct(predicted, gold, tolerance=0):
    """History answer comparison — exact match only.

    Historical answers must match exactly (after normalization and numeric
    parsing).  No +/-1 year or percentage tolerance.

    Args:
        predicted: Predicted answer string.
        gold: Gold answer string.
        tolerance: Not used (kept for API compat).

    Returns:
        True if answer is correct.
    """
    if not predicted or not predicted.strip():
        return False

    pred_clean = _normalize_history_text(predicted)
    gold_clean = _normalize_history_text(gold)

    # 1. Exact string match
    if pred_clean == gold_clean:
        return True

    # 2. Numeric exact match (handles formatting differences like "1,776" vs "1776")
    pred_num = _try_parse_history_number(pred_clean)
    gold_num = _try_parse_history_number(gold_clean)

    if pred_num is not None and gold_num is not None:
        if pred_num == gold_num:
            return True

    return False


def _normalize_history_text(text):
    """Normalize text for history comparison."""
    text = text.strip()
    for suffix in ["years", "year", "decades", "decade", "centuries",
                   "century", "generations", "generation", "ratio",
                   "BCE", "BC", "CE", "AD", "times"]:
        text = text.replace(suffix, "").strip()
    text = text.replace(",", "")
    text = text.strip().strip("'\"")
    return text


def _try_parse_history_number(text):
    """Parse a number from history text. Returns float or None."""
    text = text.strip()
    # Handle negative (BCE) years
    text = text.replace("−", "-")  # Unicode minus to ASCII minus
    # Normalize Unicode whitespace
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')

    try:
        return float(text)
    except ValueError:
        pass

    # Handle fractions like "137/56"
    frac_match = re.search(r'(-?[\d]+\.?[\d]*)\s*/\s*([\d]+\.?[\d]*)', text)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass

    match = re.search(r'-?[\d]+\.?[\d]*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _llm_judge_history_answer(predicted, gold, question):
    """Use LLM to judge if predicted answer matches gold for history questions."""
    prompt = f"""Compare these two answers to the same historical question.
The gold answer is computed from verified Wikidata temporal data.
The predicted answer may use different formatting.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Are these answers equivalent (same value within +/- 1 year)?
Respond with exactly YES or NO."""

    try:
        response = gen_from_prompt_harmony(prompt, temperature=0.0, max_tokens=10)
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


# ===========================================================================
# QA Generation Pipeline
# ===========================================================================

def _year_gap(year_a: int, year_b: int) -> int:
    """Compute the gap in years between two years, accounting for no year 0.

    BC years are negative (e.g., 1184 BC = -1184).
    """
    diff = abs(year_b - year_a)
    # If years straddle the BC/AD boundary (one negative, one positive),
    # subtract 1 because there is no year 0.
    if (year_a < 0) != (year_b < 0) and year_a != 0 and year_b != 0:
        diff -= 1
    return diff


def _execute_computation_code(code: str) -> Optional[str]:
    """Execute Python computation code and return printed output."""
    import io
    import contextlib
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, {"__builtins__": __builtins__, "math": math, "round": round,
                         "abs": abs, "min": min, "max": max,
                         "year_gap": _year_gap})
        return f.getvalue().strip()
    except Exception:
        return None


def _generate_template_params(template_id: str, tmpl: dict,
                              entity_data: dict) -> Optional[Dict[str, Any]]:
    """Generate random parameters for templates that need extra params."""
    params = {}

    if template_id == "years_since_founding":
        inception = entity_data.get("inception_year")
        if inception is None:
            return None
        # Pick a reference year after founding (but not the current year to avoid trivial answers)
        min_ref = max(inception + 10, 1950)
        max_ref = 2020
        if min_ref >= max_ref:
            min_ref = inception + 5
            max_ref = inception + 100
        params["reference_year"] = random.randint(min_ref, max_ref)

    elif template_id == "founding_ratio":
        # Will be handled in the comparative path
        min_ref = 1990
        max_ref = 2020
        params["reference_year"] = random.randint(min_ref, max_ref)

    elif template_id == "generation_count":
        params["generation_years"] = random.choice([25, 30, 33])

    return params


def _entity_type_from_category(category: str) -> str:
    """Map category name to entity type."""
    mapping = {
        "conflicts": "conflict",
        "organizations": "organization",
        "figures": "figure",
        "milestones": "milestone",
        "treaties": "milestone",
        "revolutions": "conflict",
        "dynasties": "organization",
        "explorations": "milestone",
        "inventions": "milestone",
        "pandemics": "conflict",
        "space_missions": "milestone",
        "natural_disasters": "conflict",
        "scientific_discoveries": "milestone",
        "cultural_movements": "conflict",
        "colonial_events": "conflict",
        "archaeological_discoveries": "milestone",
        # ---- NEW TOPICS (2026-03-16) ----
        "empires": "organization",
        "assassinations": "milestone",
        "sieges": "conflict",
        "religious_events": "milestone",
        "coups": "milestone",
        "migrations": "milestone",
        "constitutions": "milestone",
        "independence_movements": "conflict",
        "civil_wars": "conflict",
        "genocides_atrocities": "conflict",
        "economic_crises": "milestone",
        "naval_battles": "conflict",
        "peace_accords": "milestone",
        "scientific_institutions": "organization",
        "technological_milestones": "milestone",
        "famines": "conflict",
        "liberation_leaders": "figure",
        "cold_war_events": "milestone",
        "world_fairs_olympics": "milestone",
        "trade_routes": "organization",
    }
    return mapping.get(category, category.rstrip("s"))


def select_history_template(entity_data: dict, entity_type: str,
                             used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a template that the entity's data can support.

    Args:
        entity_data: Normalized entity data dict.
        entity_type: "conflict", "organization", "figure", or "milestone".
        used_templates: Set of template IDs already used.

    Returns:
        Dict with template info and computed gold answer, or None.
    """
    template_ids = list(HISTORY_TEMPLATES.keys())
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = HISTORY_TEMPLATES[tmpl_id]

        # Skip comparatives and cross-category (handled separately)
        if tmpl["type"] in ("comparative", "cross_category"):
            continue

        # Check entity type compatibility
        etype_list = tmpl["entity_type"]
        if entity_type not in etype_list:
            continue

        # Check required data availability
        required = tmpl["required_data"]
        all_available = True
        base_params = {}
        for key in required:
            val = entity_data.get(key)
            if val is None:
                all_available = False
                break
            base_params[key] = val

        if not all_available:
            continue

        # Generate extra parameters
        extra_params = _generate_template_params(tmpl_id, tmpl, entity_data)
        if extra_params is None:
            continue

        params = {**base_params, **extra_params}

        # Reject conflicts with trivial duration (end - start <= 1)
        _DURATION_EXEMPT = {"midpoint_year", "century_of_event",
                            "years_to_next_century", "conflict_midpoint_to_century_end"}
        if entity_type == "conflict" and tmpl_id not in _DURATION_EXEMPT:
            dur = params.get("end_year", 0) - params.get("start_year", 0)
            if dur <= 1:
                continue

        # Fill code template and compute gold answer
        try:
            code = tmpl["code_template"].format(**params)
        except KeyError:
            continue

        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        # Validate gold answer
        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            if abs(gold_float) < 1e-4:
                continue  # Near-zero answers are degenerate
            # Reject trivially degenerate ratios
            if tmpl_id in ("conflict_duration_ratio", "founding_ratio") and gold_float == 1.0:
                continue
        except ValueError:
            continue

        # Fill question hint
        try:
            question_hint = tmpl["question_hint"].format(**params)
        except KeyError:
            question_hint = tmpl["question_hint"]

        # Entity-type-specific hint overrides to prevent place/event confusion
        if tmpl_id == "century_of_event":
            _CENTURY_HINTS = {
                "conflict": "In which century (as an ordinal number) did this conflict/battle first take place?",
                "milestone": "In which century (as an ordinal number) did this event occur?",
                "organization": "In which century (as an ordinal number) was this organization founded?",
            }
            question_hint = _CENTURY_HINTS.get(entity_type, question_hint)

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "question_hint": question_hint,
        }

    return None


def select_comparative_template(data_a: dict, data_b: dict, entity_type: str,
                                 used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a comparative template for two entities."""
    template_ids = [tid for tid, t in HISTORY_TEMPLATES.items()
                    if t["type"] == "comparative" and entity_type in t["entity_type"]]
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = HISTORY_TEMPLATES[tmpl_id]
        required = tmpl["required_data"]

        params = {}
        all_available = True
        for key in required:
            val_a = data_a.get(key)
            val_b = data_b.get(key)
            if val_a is None or val_b is None:
                all_available = False
                break
            params[f"{key}_a"] = val_a
            params[f"{key}_b"] = val_b

        if not all_available:
            continue

        # Generate extra parameters
        extra_params = _generate_template_params(tmpl_id, tmpl, data_a)
        if extra_params is None:
            continue
        params.update(extra_params)

        # For founding_ratio, ensure both ages are positive
        if tmpl_id == "founding_ratio":
            ref = params.get("reference_year", 2000)
            age_a = ref - params.get("inception_year_a", ref)
            age_b = ref - params.get("inception_year_b", ref)
            if age_a <= 0 or age_b <= 0:
                continue

        # Reject conflicts with trivial duration (end - start <= 1)
        _COMP_DURATION_EXEMPT = {"battle_interval"}
        if entity_type == "conflict" and tmpl_id not in _COMP_DURATION_EXEMPT:
            dur_a = params.get("end_year_a", 0) - params.get("start_year_a", 0)
            dur_b = params.get("end_year_b", 0) - params.get("start_year_b", 0)
            if dur_a <= 1 or dur_b <= 1:
                continue

        try:
            code = tmpl["code_template"].format(**params)
        except KeyError:
            continue

        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            if abs(gold_float) < 1e-4:
                continue  # Near-zero answers are degenerate
            # Reject trivially degenerate ratios
            if tmpl_id in ("conflict_duration_ratio", "founding_ratio") and gold_float == 1.0:
                continue
        except ValueError:
            continue

        try:
            question_hint = tmpl["question_hint"].format(**params)
        except KeyError:
            question_hint = tmpl["question_hint"]

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "question_hint": question_hint,
        }

    return None


def select_cross_category_template(figure_data: dict, event_data: dict,
                                    event_type: str,
                                    used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a cross-category template pairing a figure with an event/conflict."""
    template_ids = [tid for tid, t in HISTORY_TEMPLATES.items()
                    if t["type"] == "cross_category"]
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = HISTORY_TEMPLATES[tmpl_id]

        # Check figure has required data
        fig_required = tmpl["required_data"]
        params = {}
        all_available = True
        for key in fig_required:
            val = figure_data.get(key)
            if val is None:
                all_available = False
                break
            params[key] = val

        if not all_available:
            continue

        # Check event has required cross data
        cross_required = tmpl.get("cross_required_data", [])
        for key in cross_required:
            val = event_data.get(key)
            if val is None:
                # Try start_year as fallback for event_year
                if key == "event_year":
                    val = event_data.get("start_year")
                if val is None:
                    all_available = False
                    break
            params[key] = val

        if not all_available:
            continue

        # Validate: person must be alive during event
        if tmpl_id == "age_at_event":
            birth = params.get("birth_year")
            event = params.get("event_year")
            death = figure_data.get("death_year")
            if birth is None or event is None:
                continue
            age = event - birth
            if age < 0 or age > 120:
                continue
            if death is not None and event > death:
                continue

        try:
            code = tmpl["code_template"].format(**params)
        except KeyError:
            continue

        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            if gold_float < 0:
                continue  # Negative age doesn't make sense
            if abs(gold_float) < 1e-4:
                continue  # Near-zero answers are degenerate
        except ValueError:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "question_hint": tmpl["question_hint"],
        }

    return None


# ===========================================================================
# Temporal data formatting for composition prompts
# ===========================================================================

_HISTORY_TEMPORAL_DATA_LABELS = {
    "birth_year": "Birth year",
    "death_year": "Death year",
    "start_year": "Start year",
    "end_year": "End year",
    "event_year": "Event year",
    "inception_year": "Inception year",
    "dissolution_year": "Dissolution year",
    "year": "Key year",
    "reference_year": "Reference year",
    "generation_years": "Generation length",
}


def _format_history_temporal_data_text(entity_data: dict, params: dict) -> str:
    """Format verified Wikidata temporal data for inclusion in composition prompt.

    Includes both the raw entity_data fields and the template params so the LLM
    knows exactly which dates the computation uses.
    """
    lines = []

    # Show the template params (these are the values used in the computation)
    for key in ("birth_year", "death_year", "start_year", "end_year",
                "event_year", "inception_year", "dissolution_year",
                "year", "reference_year", "generation_years"):
        val = params.get(key)
        if val is not None:
            label = _HISTORY_TEMPORAL_DATA_LABELS.get(key, key)
            lines.append(f"  {label}: {val}")

    # Also include entity_data fields not already covered by params
    for key in ("birth_year", "death_year", "start_year", "end_year",
                "event_year", "inception_year", "dissolution_year"):
        if key not in params:
            val = entity_data.get(key)
            if val is not None:
                label = _HISTORY_TEMPORAL_DATA_LABELS.get(key, key)
                lines.append(f"  {label}: {val}")

    return "\n".join(lines) if lines else "(no temporal data available)"


def compose_history_question(clues, reasoning_ctx, entity_name,
                              prev_questions, entity_data=None):
    """Compose a history QA question using LLM."""
    facts_text = "\n".join(
        f"[{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) {c['fact']}"
        for c in clues
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    # Format verified temporal data for the prompt
    temporal_data_text = _format_history_temporal_data_text(
        entity_data or {}, reasoning_ctx.get("params", {}),
    )

    prompt = HISTORY_QA_PROMPT.format(
        facts_text=facts_text,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        temporal_data_text=temporal_data_text,
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=HISTORY_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception as e:
        print(f"  Composition LLM error: {e}", flush=True)
        return None

    question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    facts_match = re.search(r'Used_Facts:\s*(.+?)(?:\n|$)', response)

    if not question_match:
        return None

    question = question_match.group(1).strip()

    # Strip fact ID annotations that the LLM sometimes copies from the prompt
    # e.g. "(C8_F1)", "[C3_F2]", "(A_C1_F1)", "(B_C2_F1)", "(W1)", "[W2]"
    question = re.sub(r'\s*[\(\[]\s*(?:[AB]?_?C\d+_F\d+|W\d+)\s*[\)\]]', '', question)

    used_facts = []
    if facts_match:
        used_facts = [f.strip() for f in facts_match.group(1).split(",") if f.strip()]

    return {"question": question, "used_facts": used_facts}


def compose_history_comparative_question(clues_a, clues_b, reasoning_ctx,
                                          entity_name_a, entity_name_b,
                                          prev_questions,
                                          entity_data_a=None,
                                          entity_data_b=None):
    """Compose a comparative history QA question with per-entity clue blocks.

    Args:
        clues_a: List of clue fact dicts for entity A.
        clues_b: List of clue fact dicts for entity B.
        reasoning_ctx: Template context from select_comparative_template().
        entity_name_a, entity_name_b: Entity names (for validation only).
        prev_questions: List of previously generated question strings.
        entity_data_a, entity_data_b: Entity temporal data dicts.

    Returns:
        Dict with "question", "used_facts", "used_facts_a", "used_facts_b"
        keys, or None on failure.
    """
    facts_text_a = "\n".join(
        f"[A_{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) "
        f"[Entity A] {c['fact']}"
        for c in clues_a
    )
    facts_text_b = "\n".join(
        f"[B_{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) "
        f"[Entity B] {c['fact']}"
        for c in clues_b
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    # Format verified temporal data for both entities
    params = reasoning_ctx.get("params", {})
    # Split params into per-entity dicts for formatting
    params_a = {}
    params_b = {}
    for key in ("birth_year", "death_year", "start_year", "end_year",
                "event_year", "inception_year", "dissolution_year"):
        if f"{key}_a" in params:
            params_a[key] = params[f"{key}_a"]
        if f"{key}_b" in params:
            params_b[key] = params[f"{key}_b"]
    # reference_year is shared
    if "reference_year" in params:
        params_a["reference_year"] = params["reference_year"]
        params_b["reference_year"] = params["reference_year"]

    temporal_data_text_a = _format_history_temporal_data_text(entity_data_a or {}, params_a)
    temporal_data_text_b = _format_history_temporal_data_text(entity_data_b or {}, params_b)

    prompt = HISTORY_COMPARATIVE_QA_PROMPT.format(
        facts_text_a=facts_text_a,
        facts_text_b=facts_text_b,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        temporal_data_text_a=temporal_data_text_a,
        temporal_data_text_b=temporal_data_text_b,
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=HISTORY_QA_DEVELOPER,
            reasoning_effort="high",
        )
    except Exception as e:
        print(f"  Comparative composition LLM error: {e}", flush=True)
        return None

    q_match = re.search(
        r'Question:\s*(.+?)(?:\nUsed_Facts|\nReasoning|\Z)', response, re.DOTALL
    )
    if not q_match:
        return None

    question = q_match.group(1).strip()
    question = re.sub(r'\s*[\(\[]\s*(?:[AB]_)?(?:C\d+_F\d+|W\d+|F\d+)\s*[\)\]]', '', question)

    used_facts_a = []
    used_facts_b = []
    m_a = re.search(r'Used_Facts_A:\s*(.+?)(?:\n|$)', response)
    m_b = re.search(r'Used_Facts_B:\s*(.+?)(?:\n|$)', response)
    if m_a:
        used_facts_a = [f.strip() for f in m_a.group(1).split(",") if f.strip()]
    if m_b:
        used_facts_b = [f.strip() for f in m_b.group(1).split(",") if f.strip()]

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


def _check_history_question_consistency(question: str, template_id: str,
                                         gold_answer: str,
                                         entity_type: str = "") -> Tuple[bool, str]:
    """History-specific Phase 3 check: verify question text matches code logic.

    Returns (passed, reason).
    """
    q_lower = question.lower()

    # For century_of_event: the question must ask for a century, not a year
    # difference, subtraction, or "how many years".
    if template_id == "century_of_event":
        asks_century = ("century" in q_lower
                        and ("which century" in q_lower
                             or "what century" in q_lower
                             or "in which century" in q_lower
                             or "ordinal" in q_lower))
        asks_years = ("how many years" in q_lower
                      or "subtract" in q_lower
                      or "difference" in q_lower
                      or "years passed" in q_lower
                      or "years separate" in q_lower)
        if asks_years and not asks_century:
            return False, (
                "century_of_event: question asks for years/difference "
                "but code computes century number"
            )
        if not asks_century:
            return False, (
                "century_of_event: question does not clearly ask "
                "'in which century'"
            )

    if template_id == "decade_of_start":
        if "decade" not in q_lower and "which decade" not in q_lower:
            return False, "decade_of_start: question does not ask about a decade"

    if template_id in ("lifespan_years",):
        if ("how many years" not in q_lower and "how long" not in q_lower
                and "lifespan" not in q_lower and "live" not in q_lower):
            return False, "lifespan_years: question does not ask about lifespan"

    if template_id in ("duration_years", "conflict_duration"):
        if ("how many years" not in q_lower and "duration" not in q_lower
                and "how long" not in q_lower and "last" not in q_lower
                and "exist" not in q_lower):
            return False, f"{template_id}: question does not ask about duration"

    if template_id == "midpoint_year":
        if "midpoint" not in q_lower and "middle" not in q_lower and "halfway" not in q_lower:
            return False, "midpoint_year: question does not ask about a midpoint"

    if template_id == "era_span_decades":
        if "decade" not in q_lower:
            return False, "era_span_decades: question does not mention decades"

    if template_id == "generation_count":
        if "generation" not in q_lower:
            return False, "generation_count: question does not mention generations"

    if template_id == "contemporaneity_overlap":
        if "overlap" not in q_lower and "concurrent" not in q_lower and "simultaneous" not in q_lower:
            return False, "contemporaneity_overlap: question does not ask about temporal overlap"

    if template_id == "duration_comparison":
        if "difference" not in q_lower and "longer" not in q_lower and "shorter" not in q_lower:
            return False, "duration_comparison: question does not ask about duration difference"

    if template_id == "founding_gap":
        if "apart" not in q_lower and "gap" not in q_lower and "between" not in q_lower and "difference" not in q_lower:
            return False, "founding_gap: question does not ask about gap between founding dates"

    if template_id == "lifespan_difference":
        if "difference" not in q_lower and "longer" not in q_lower and "shorter" not in q_lower:
            return False, "lifespan_difference: question does not ask about lifespan difference"

    # NOTE: General place/location vs event confusion guard is now in the
    # shared run_phase3_validation() (check 3f2) in multiskill_utils.py.

    return True, ""


def _check_history_question_computation_match(question: str, question_hint: str,
                                                template_id: str) -> Tuple[bool, str]:
    """LLM-based check: does the question ask what the template computes?

    This catches cases where the LLM rephrases the computation incorrectly,
    e.g., template computes duration but question asks about start year difference.
    """
    prompt = f"""A historical research question was generated from a template. Verify that the
question asks for EXACTLY the computation described in the hint.

Template hint (what the question SHOULD ask): {question_hint}
Template ID: {template_id}

Question: {question}

Does the question ask for EXACTLY the same computation as the hint? Consider:
- "Duration" vs "difference in start dates" are DIFFERENT computations
- "Century of event" vs "how many years" are DIFFERENT computations
- "Lifespan" vs "founding gap" are DIFFERENT computations
- Minor wording changes (synonyms, passive voice) are acceptable

Respond with exactly YES or NO, followed by a brief reason."""

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=0.0, max_tokens=128,
            developer_content="You are a question-validation expert. Be strict.",
        ).strip()
        if response.upper().startswith("NO"):
            reason = response[2:].strip().lstrip(":").lstrip(",").strip()
            return False, f"question-computation mismatch: {reason[:120]}"
        return True, ""
    except Exception as e:
        print(f"  _check_history_question_computation_match error: {e}", flush=True)
        # On error, don't block — return True
        return True, ""


def gen_history_qa_from_entity(entity_data, agent_info, num_questions=5,
                                prev_questions=None, used_templates=None,
                                articles=None, category=None):
    """Generate history QA pairs from a single entity using chain-based clues.

    Parallels gen_financial_qa_from_entity with Phases 0/1/1.5/2/3.

    Args:
        entity_data: Dict with 'name', 'qid', 'entity_type', 'data',
                     'chains', 'articles'.
        agent_info: (lm, tokenizer, client) tuple.
        num_questions: Questions to generate per entity.
        prev_questions: Previously generated question strings.
        used_templates: Set of template IDs already used.
        articles: Wikipedia article dicts (overrides entity_data['articles']).
        category: Category key.

    Returns:
        List of QA pair dicts.
    """
    if prev_questions is None:
        prev_questions = []
    if used_templates is None:
        used_templates = set()

    entity_name = entity_data["name"]
    entity_type_val = entity_data.get("entity_type", "figure")
    data = entity_data.get("data", {})
    chains = entity_data.get("chains", [])
    qid = entity_data.get("qid", "")
    if articles is None:
        articles = entity_data.get("articles", [])

    if not chains or len(chains) < 2:
        print(f"  No KG chains for {entity_name}", flush=True)
        return []

    qa_pairs = []
    max_attempts = num_questions * 3
    poisoned_props = set()

    for attempt in range(max_attempts):
        if len(qa_pairs) >= num_questions:
            break

        print(f"\n  [{entity_name}] Attempt {attempt+1}/{max_attempts} "
              f"(generated {len(qa_pairs)}/{num_questions})", flush=True)

        # Phase 0: Select template and compute gold answer
        reasoning_ctx = select_history_template(data, entity_type_val, used_templates)
        if reasoning_ctx is None:
            print(f"  No valid template remaining for {entity_name}", flush=True)
            single_templates = [t for t, d in HISTORY_TEMPLATES.items()
                                if d["type"] == "single" and entity_type_val in d["entity_type"]]
            if len(used_templates & set(single_templates)) >= len(single_templates):
                used_templates -= set(single_templates)
                continue
            break

        print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

        # Phase 1: Extract clue facts from KG chains
        extracted_facts, _ = extract_chain_clue_facts(chains, entity_name, agent_info)
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
        composition = compose_history_question(
            clues_for_composition, reasoning_ctx, entity_name, prev_questions,
            entity_data=data,
        )
        if composition is None:
            print(f"  Phase 2: Composition failed", flush=True)
            continue

        question = composition["question"]
        used_facts = composition["used_facts"]
        print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

        # Phase 3: Validation
        all_years = data.get("_all_years", [])
        passed, reason = run_phase3_validation(
            question=question,
            gold_answer=reasoning_ctx["gold_answer"],
            computation_code=reasoning_ctx["code"],
            entity_label=entity_name,
            entity_names=[entity_name, qid],
            entity_values=all_years,
            used_facts=used_facts,
            extracted_facts=extracted_facts,
            gen_fn=gen_from_prompt_harmony,
            entity_type=entity_type_val,
            check_facts=True,
            min_chains=2,
            chains=chains,
            entity_qid=qid,
        )
        if not passed:
            # Poison the used facts only when the rejection is intrinsic to the
            # pair (3d3b); other reasons are phrasing/combination, so retry
            # instead of banning innocent facts and starving sparse entities.
            poison_used_facts(reason, used_facts, extracted_facts, poisoned_props)
            print(f"  Phase 3: {reason}", flush=True)
            continue

        # 3-extra: History-specific question-code consistency
        passed, reason = _check_history_question_consistency(
            question, reasoning_ctx["template_id"], reasoning_ctx["gold_answer"],
            entity_type=entity_type_val,
        )
        if not passed:
            # Poison the used facts only when the rejection is intrinsic to the
            # pair (3d3b); other reasons are phrasing/combination, so retry
            # instead of banning innocent facts and starving sparse entities.
            poison_used_facts(reason, used_facts, extracted_facts, poisoned_props)
            print(f"  Phase 3 (history): {reason}", flush=True)
            continue

        # 3-extra2: LLM-based question-computation match
        passed, reason = _check_history_question_computation_match(
            question, reasoning_ctx["question_hint"], reasoning_ctx["template_id"],
        )
        if not passed:
            # Poison the used facts only when the rejection is intrinsic to the
            # pair (3d3b); other reasons are phrasing/combination, so retry
            # instead of banning innocent facts and starving sparse entities.
            poison_used_facts(reason, used_facts, extracted_facts, poisoned_props)
            print(f"  Phase 3 (computation match): {reason}", flush=True)
            continue
        print(f"  Phase 3: PASSED", flush=True)

        # Build output fields
        source_triples = [c["path_description"] for c in chains[:5]]
        _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
        facts_for_output = [
            {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
            for f in extracted_facts
        ]

        # Build full-text grounding articles
        grounding_articles = build_multiskill_grounding_articles(
            entity_name, reasoning_ctx["gold_answer"],
            reasoning_ctx["template"]["answer_unit"],
            extracted_facts, chains, articles or [],
        )

        # Store grounding data for verification
        _grounding_keys = {"birth_date", "death_date", "start_date", "end_date",
                           "duration_years", "casualties", "population",
                           "year_founded", "year_dissolved", "_all_years"}
        grounding_entity_data = {k: data[k] for k in _grounding_keys if k in data}

        qa_pair = {
            "question": question,
            "gold_answer": reasoning_ctx["gold_answer"],
            "computation_code": reasoning_ctx["code"],
            "template_id": reasoning_ctx["template_id"],
            "template_label": reasoning_ctx["template"]["label"],
            "template_type": reasoning_ctx["template"]["type"],
            "template_level": reasoning_ctx["template"]["level"],
            "answer_unit": reasoning_ctx["template"]["answer_unit"],
            "entity_name": entity_name,
            "entity_type": entity_type_val,
            "category": category,
            "used_facts": used_facts,
            "extracted_facts": facts_for_output,
            "source_triples": source_triples,
            "grounding_articles": grounding_articles,
            "data_source": "wikidata_temporal",
            "grounding_clues": clues_for_composition,
            "grounding_entity_data": grounding_entity_data,
        }
        qa_pair.update(compute_cci_fields(reasoning_ctx["template"], extracted_facts, is_comparative=False))

        if qid:
            qa_pair["qid"] = qid

        qa_pairs.append(qa_pair)
        prev_questions.append(question)
        used_templates.add(reasoning_ctx["template_id"])

    return qa_pairs


def gen_history_qa_single(entity_name, entity_data, clues, entity_type,
                           used_templates, prev_questions, category):
    """Generate a single-entity history QA pair (legacy clue-based fallback)."""
    # Phase 0: Select template
    reasoning_ctx = select_history_template(entity_data, entity_type, used_templates)
    if reasoning_ctx is None:
        print(f"  select_history_template returned None for entity_type={entity_type}, "
              f"used_templates={len(used_templates)}, data_keys={list(entity_data.keys())}",
              flush=True)
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1: Clue facts ready
    if len(clues) < 3:
        print(f"  Phase 1: Insufficient clues ({len(clues)})", flush=True)
        return None

    # Phase 2: Compose question
    composition = compose_history_question(
        clues, reasoning_ctx, entity_name, prev_questions,
        entity_data=entity_data,
    )
    if composition is None:
        print(f"  Phase 2: Composition failed", flush=True)
        return None

    question = composition["question"]
    used_facts = composition["used_facts"]
    print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

    # Phase 3: Validation (shared utility)
    qid = entity_data.get("qid", "")
    all_years = entity_data.get("_all_years", [])
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=entity_name,
        entity_names=[entity_name, qid],
        entity_values=all_years,
        used_facts=used_facts,
        extracted_facts=[],
        gen_fn=gen_from_prompt_harmony,
        entity_type=entity_type,
    )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    # Phase 3-extra: History-specific question-code consistency
    passed, reason = _check_history_question_consistency(
        question, reasoning_ctx["template_id"], reasoning_ctx["gold_answer"],
        entity_type=entity_type,
    )
    if not passed:
        print(f"  Phase 3 (history): {reason}", flush=True)
        return None

    # Phase 3-extra2: LLM-based question-computation match
    passed, reason = _check_history_question_computation_match(
        question, reasoning_ctx["question_hint"], reasoning_ctx["template_id"],
    )
    if not passed:
        print(f"  Phase 3 (computation match): {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Store grounding data for verification
    _grounding_keys = {"birth_date", "death_date", "start_date", "end_date",
                       "duration_years", "casualties", "population",
                       "year_founded", "year_dissolved", "_all_years"}
    grounding_entity_data = {k: entity_data[k] for k in _grounding_keys if k in entity_data}

    result = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": reasoning_ctx["template"]["label"],
        "template_type": reasoning_ctx["template"]["type"],
        "template_level": reasoning_ctx["template"]["level"],
        "answer_unit": reasoning_ctx["template"]["answer_unit"],
        "entity_name": entity_name,
        "entity_type": entity_type,
        "category": category,
        "used_facts": used_facts,
        "data_source": "wikidata_temporal",
        "grounding_clues": clues,
        "grounding_entity_data": grounding_entity_data,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=False))

    if qid:
        result["qid"] = qid

    return result


def gen_history_qa_comparative(name_a, data_a, clues_a,
                                name_b, data_b, clues_b,
                                entity_type, used_templates,
                                prev_questions, category,
                                chains_a=None, articles_a=None,
                                chains_b=None, articles_b=None,
                                agent_info=None):
    """Generate a comparative history QA pair (two entities).

    When chains_a/chains_b are provided, runs Phase 1/1.5 (chain-based fact
    extraction + grounding) for each entity. Falls back to simple clues
    otherwise.
    """
    reasoning_ctx = select_comparative_template(data_a, data_b, entity_type, used_templates)
    if reasoning_ctx is None:
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1 + 1.5: Extract and ground chain-based clues for each entity
    if chains_a and chains_b and agent_info:
        facts_a, _ = extract_chain_clue_facts(chains_a, name_a, agent_info)
        if facts_a is not None and articles_a:
            facts_a = verify_facts_grounding(facts_a, chains_a, articles_a, agent_info)
        facts_b, _ = extract_chain_clue_facts(chains_b, name_b, agent_info)
        if facts_b is not None and articles_b:
            facts_b = verify_facts_grounding(facts_b, chains_b, articles_b, agent_info)

        if facts_a and facts_b:
            # Use chain-based clues
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

    combined_clues = []
    for c in clues_a[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"A_{c.get('fact_id', '')}"
        c_copy["fact"] = f"[Entity A] {c.get('fact', '')}"
        combined_clues.append(c_copy)
    for c in clues_b[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"B_{c.get('fact_id', '')}"
        c_copy["fact"] = f"[Entity B] {c.get('fact', '')}"
        combined_clues.append(c_copy)

    if len(combined_clues) < 4:
        return None

    # Merge entity data for temporal data formatting
    merged_entity_data = {}
    for k, v in data_a.items():
        merged_entity_data[f"{k}_a" if k != "_all_years" else k] = v
    for k, v in data_b.items():
        if k == "_all_years":
            merged_entity_data.setdefault(k, []).extend(v if isinstance(v, list) else [])
        else:
            merged_entity_data[f"{k}_b"] = v

    composition = compose_history_question(
        combined_clues, reasoning_ctx, f"{name_a}/{name_b}", prev_questions,
        entity_data=merged_entity_data,
    )
    if composition is None:
        return None

    question = composition["question"]

    # Phase 3: Validation (shared utility)
    all_years = list(data_a.get("_all_years", [])) + list(data_b.get("_all_years", []))
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=f"{name_a} / {name_b}",
        entity_names=[name_a, name_b],
        entity_values=all_years,
        entity_type=entity_type,
    )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    # Phase 3-extra: Verify question uses clues from BOTH entities
    used_facts = composition.get("used_facts", [])
    has_a_clue = any(str(f).startswith("A_") for f in used_facts)
    has_b_clue = any(str(f).startswith("B_") for f in used_facts)
    if not (has_a_clue and has_b_clue):
        print(f"  Phase 3 (history): comparative question only uses clues from "
              f"{'Entity A' if has_a_clue else 'Entity B' if has_b_clue else 'neither'}",
              flush=True)
        return None

    # Phase 3-extra: 3h (comparative) — each side must be uniquely identifiable
    # from its clues.  Only when chain-based facts were derived (facts_a/facts_b
    # exist iff this branch ran); the fallback-clue path has no KG structure.
    if chains_a and chains_b and agent_info:
        ok, reason = check_kg_uniqueness_comparative(
            strip_side_prefix(used_facts, "A_"), facts_a or [], chains_a,
            strip_side_prefix(used_facts, "B_"), facts_b or [], chains_b,
        )
        if not ok:
            print(f"  Phase 3 (history): {reason}", flush=True)
            return None

    # Phase 3-extra: History-specific question-code consistency
    passed, reason = _check_history_question_consistency(
        question, reasoning_ctx["template_id"], reasoning_ctx["gold_answer"],
        entity_type=entity_type,
    )
    if not passed:
        print(f"  Phase 3 (history): {reason}", flush=True)
        return None

    # Phase 3-extra: LLM-based question-computation match
    passed, reason = _check_history_question_computation_match(
        question, reasoning_ctx["question_hint"], reasoning_ctx["template_id"],
    )
    if not passed:
        print(f"  Phase 3 (history-comp-match): {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Store grounding data for verification
    _grounding_keys = {"birth_date", "death_date", "start_date", "end_date",
                       "duration_years", "casualties", "population",
                       "year_founded", "year_dissolved", "_all_years"}
    grounding_entity_data_a = {k: data_a[k] for k in _grounding_keys if k in data_a}
    grounding_entity_data_b = {k: data_b[k] for k in _grounding_keys if k in data_b}

    result = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": reasoning_ctx["template"]["label"],
        "template_type": "comparative",
        "template_level": reasoning_ctx["template"]["level"],
        "answer_unit": reasoning_ctx["template"]["answer_unit"],
        "entity_name_a": name_a,
        "entity_name_b": name_b,
        "entity_type": entity_type,
        "category": category,
        "used_facts": composition.get("used_facts", []),
        "data_source": "wikidata_temporal",
        "grounding_clues_a": clues_a,
        "grounding_clues_b": clues_b,
        "grounding_entity_data_a": grounding_entity_data_a,
        "grounding_entity_data_b": grounding_entity_data_b,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=True))

    qid_a = data_a.get("qid", "")
    qid_b = data_b.get("qid", "")
    if qid_a:
        result["qid_a"] = qid_a
    if qid_b:
        result["qid_b"] = qid_b

    return result


def gen_history_qa_cross_category(figure_name, figure_data, figure_clues,
                                   event_name, event_data, event_clues,
                                   event_type, used_templates,
                                   prev_questions, category):
    """Generate a cross-category QA pair (figure + event/conflict)."""
    reasoning_ctx = select_cross_category_template(
        figure_data, event_data, event_type, used_templates,
    )
    if reasoning_ctx is None:
        return None

    print(f"  Phase 0 (cross): {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    combined_clues = []
    for c in figure_clues[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"FIG_{c['fact_id']}"
        c_copy["fact"] = f"[Person] {c['fact']}"
        combined_clues.append(c_copy)
    for c in event_clues[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"EVT_{c['fact_id']}"
        c_copy["fact"] = f"[Event] {c['fact']}"
        combined_clues.append(c_copy)

    if len(combined_clues) < 4:
        return None

    # Merge entity data for temporal data formatting
    merged_cross_data = {}
    for k, v in figure_data.items():
        merged_cross_data[k] = v
    for k, v in event_data.items():
        if k not in merged_cross_data:
            merged_cross_data[k] = v

    composition = compose_history_question(
        combined_clues, reasoning_ctx, f"{figure_name}/{event_name}", prev_questions,
        entity_data=merged_cross_data,
    )
    if composition is None:
        return None

    question = composition["question"]

    # Phase 3: Validation (shared utility)
    all_years = list(figure_data.get("_all_years", [])) + list(event_data.get("_all_years", []))
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=f"{figure_name} / {event_name}",
        entity_names=[figure_name, event_name],
        entity_values=all_years,
        entity_type="figure",
    )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    # Phase 3-extra: History-specific question-code consistency
    passed, reason = _check_history_question_consistency(
        question, reasoning_ctx["template_id"], reasoning_ctx["gold_answer"],
        entity_type="figure",
    )
    if not passed:
        print(f"  Phase 3 (history-cross): {reason}", flush=True)
        return None

    # Phase 3-extra: LLM-based question-computation match
    passed, reason = _check_history_question_computation_match(
        question, reasoning_ctx["question_hint"], reasoning_ctx["template_id"],
    )
    if not passed:
        print(f"  Phase 3 (history-cross-comp-match): {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Store grounding data for verification
    _grounding_keys = {"birth_date", "death_date", "start_date", "end_date",
                       "duration_years", "casualties", "population",
                       "year_founded", "year_dissolved", "_all_years"}
    grounding_entity_data_a = {k: figure_data[k] for k in _grounding_keys if k in figure_data}
    grounding_entity_data_b = {k: event_data[k] for k in _grounding_keys if k in event_data}

    result = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": reasoning_ctx["template"]["label"],
        "template_type": "cross_category",
        "template_level": reasoning_ctx["template"]["level"],
        "answer_unit": reasoning_ctx["template"]["answer_unit"],
        "entity_name_a": figure_name,
        "entity_name_b": event_name,
        "entity_type": f"figure+{event_type}",
        "category": category,
        "used_facts": composition.get("used_facts", []),
        "data_source": "wikidata_temporal",
        "grounding_clues_a": figure_clues,
        "grounding_clues_b": event_clues,
        "grounding_entity_data_a": grounding_entity_data_a,
        "grounding_entity_data_b": grounding_entity_data_b,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=True))

    return result


# ===========================================================================
# Superlative (N-Entity Ranking) QA Generation
# ===========================================================================

_ENTITY_LABELS = ["A", "B", "C", "D", "E"]


def _extract_superlative_value(data: dict, value_key: str) -> Optional[int]:
    """Extract a single value for a superlative template.

    value_key starting with '_' means computed:
      _duration  = end_year - start_year  (or dissolution_year - inception_year)
      _lifespan  = death_year - birth_year
    Otherwise, it is a direct key into data.
    """
    if value_key == "_duration":
        sy = data.get("start_year") or data.get("inception_year")
        ey = data.get("end_year") or data.get("dissolution_year")
        if sy is not None and ey is not None and ey > sy:
            return ey - sy
        return None
    elif value_key == "_lifespan":
        by = data.get("birth_year")
        dy = data.get("death_year")
        if by is not None and dy is not None and dy > by:
            return dy - by
        return None
    else:
        val = data.get(value_key)
        return int(val) if val is not None else None


def select_superlative_template(
    entities_data: List[dict],
    entity_type: str,
    n_entities: int,
    used_templates: set,
) -> Optional[Dict[str, Any]]:
    """Select a superlative template that N entities can support.

    Args:
        entities_data: List of N entity data dicts (each with temporal fields).
        entity_type: Entity type string (e.g. "organization", "conflict").
        n_entities: Number of entities (3 or 5).
        used_templates: Set of template IDs already used.

    Returns:
        Dict with template info, vals list, code, gold_answer, or None.
    """
    template_ids = list(HISTORY_SUPERLATIVE_TEMPLATES.keys())
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = HISTORY_SUPERLATIVE_TEMPLATES[tmpl_id]

        # Check entity type
        if entity_type not in tmpl["entity_type"]:
            continue

        # Check n_entities
        if n_entities not in tmpl["n_entities"]:
            continue

        # Extract value per entity
        value_key = tmpl["value_key"]
        vals = []
        all_ok = True
        for ed in entities_data:
            v = _extract_superlative_value(ed, value_key)
            if v is None:
                all_ok = False
                break
            vals.append(v)

        if not all_ok:
            continue

        # For min/max templates, require all distinct values
        code_str = tmpl["code_template"]
        is_minmax = "min(vals)" in code_str or "max(vals)" in code_str
        if is_minmax and len(set(vals)) < n_entities:
            continue

        # Build code with actual vals prepended
        full_code = f"vals = {vals}\n{code_str}"
        gold_answer = _execute_computation_code(full_code)
        if gold_answer is None:
            continue

        # Validate gold answer
        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            if abs(gold_float) < 1e-4:
                continue
        except ValueError:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "vals": vals,
            "code": full_code,
            "gold_answer": gold_answer,
            "question_hint": tmpl["question_hint"],
        }

    return None


def compose_history_superlative_question(
    clues_per_entity: Dict[str, List[dict]],
    reasoning_ctx: dict,
    entity_names: List[str],
    prev_questions: List[str],
    n_entities: int,
    entities_data: Optional[List[dict]] = None,
) -> Optional[dict]:
    """Compose a superlative history question from N entities' clues.

    Args:
        clues_per_entity: {"A": [...], "B": [...], ...} — clue dicts per label.
        reasoning_ctx: Output from select_superlative_template().
        entity_names: List of N entity name strings.
        prev_questions: Previously generated question strings.
        n_entities: Number of entities (3 or 5).
        entities_data: List of N entity data dicts (for temporal data formatting).

    Returns:
        {"question": ..., "used_facts_per_entity": {"A": [...], ...}} or None.
    """
    labels = _ENTITY_LABELS[:n_entities]

    # Build entity clue blocks
    blocks = []
    for label in labels:
        clues = clues_per_entity.get(label, [])
        facts_text = "\n".join(
            f"[{label}_{c.get('fact_id', f'F{i}')}] ({c.get('topic', 'description')}) {c['fact']}"
            for i, c in enumerate(clues)
        )
        blocks.append(
            f"CLUE FACTS FOR ENTITY {label} (do NOT name it):\n{facts_text}"
        )
    entity_clue_blocks = "\n\n".join(blocks)

    # Build Used_Facts output format
    used_facts_lines = "\n".join(
        f"Used_Facts_{label}: <fact_ids for Entity {label}>"
        for label in labels
    )

    sentence_range = "4-8" if n_entities == 3 else "6-12"
    prev_text = "\n".join(f"- {q}" for q in prev_questions[-10:]) if prev_questions else "(none)"

    # Format temporal data for all entities
    temporal_lines = []
    labels = _ENTITY_LABELS[:n_entities]
    if entities_data:
        for i, label in enumerate(labels):
            if i < len(entities_data):
                ed = entities_data[i]
                entity_lines = _format_history_temporal_data_text(ed, reasoning_ctx.get("params", {}))
                if entity_lines:
                    temporal_lines.append(f"Entity {label}:\n{entity_lines}")
    temporal_data_text = "\n".join(temporal_lines) if temporal_lines else "(no temporal data available)"

    prompt = HISTORY_SUPERLATIVE_QA_PROMPT.format(
        n_entities=n_entities,
        entity_clue_blocks=entity_clue_blocks,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        sentence_range=sentence_range,
        previous_questions=prev_text,
        used_facts_format=used_facts_lines,
        temporal_data_text=temporal_data_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=0.7, max_tokens=1536,
            developer_content=HISTORY_SUPERLATIVE_QA_DEVELOPER,
        )
    except Exception as e:
        print(f"  Superlative composition LLM error: {e}", flush=True)
        return None

    # Parse question
    q_match = re.search(
        r'Question:\s*(.+?)(?:\nUsed_Facts|\nReasoning|\Z)', response, re.DOTALL
    )
    if not q_match:
        return None

    question = q_match.group(1).strip()

    # Strip fact-ID annotations the LLM may copy from the prompt
    question = re.sub(r'\s*[\(\[]\s*(?:[A-E]_)?(?:C\d+_F\d+|W\d+|F\d+)\s*[\)\]]', '', question)

    # Parse Used_Facts per entity
    used_facts_per_entity: Dict[str, List[str]] = {}
    for label in labels:
        pat = rf'Used_Facts_{label}:\s*(.+?)(?:\n|$)'
        m = re.search(pat, response)
        if m:
            used_facts_per_entity[label] = [
                f.strip() for f in m.group(1).split(",") if f.strip()
            ]
        else:
            used_facts_per_entity[label] = []

    return {"question": question, "used_facts_per_entity": used_facts_per_entity}


def _check_history_multi_entity_uniqueness(
    question: str, entity_labels: List[str], entity_refs: List[str],
) -> bool:
    """LLM check that ALL entities in a multi-entity history question are identifiable."""
    refs_text = ", ".join(f'"{r}"' for r in entity_refs)
    lines_text = "\n".join(f"{r}: <name>" for r in entity_refs)
    prompt = f"""Read the following question. It describes multiple real-world historical entities using clue facts.
Identify EACH entity. The entities are referred to as {refs_text}.

Question: {question}

For each entity, give the name on its own line in the format:
{lines_text}"""
    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=0.0, max_tokens=4096,
            developer_content="You are a historical entity identification expert.",
        ).strip()

        all_found = True
        for label in entity_labels:
            if label.lower() not in response.lower():
                print(f"  Phase 3h-multi: Failed to identify '{label}'", flush=True)
                all_found = False
        if all_found:
            print(f"  Phase 3h-multi: All {len(entity_labels)} entities identified", flush=True)
        return all_found
    except Exception as e:
        print(f"  Phase 3h-multi: Check error: {e}", flush=True)
        return False


def gen_history_qa_superlative(
    entity_data_list: List[dict],
    agent_info,
    n_entities: int = 3,
    prev_questions: Optional[List[str]] = None,
    used_templates: Optional[set] = None,
    category: str = "",
) -> List[dict]:
    """Generate superlative (N-entity ranking) history QA pairs.

    Pipeline: Phase 0 (template) → Phase 1 (facts) → Phase 1.5 (grounding) →
              Phase 2 (compose) → Phase 3 (validation).

    Args:
        entity_data_list: List of N entity dicts with 'name', 'qid',
                         'entity_type', 'data', 'chains', 'articles'.
        agent_info: (lm, tokenizer, client) tuple.
        n_entities: Number of entities (3 or 5).
        prev_questions: Previously generated question strings.
        used_templates: Set of template IDs already used.
        category: Category key.

    Returns:
        List of QA pair dicts (0 or 1).
    """
    if prev_questions is None:
        prev_questions = []
    if used_templates is None:
        used_templates = set()

    if len(entity_data_list) < n_entities:
        return []

    labels = _ENTITY_LABELS[:n_entities]
    entity_names = [ed["name"] for ed in entity_data_list[:n_entities]]
    entity_type = entity_data_list[0].get("entity_type", "organization")
    entities_data = [ed.get("data", {}) for ed in entity_data_list[:n_entities]]

    # Phase 0: Select template and compute gold answer
    reasoning_ctx = select_superlative_template(
        entities_data, entity_type, n_entities, used_templates,
    )
    if reasoning_ctx is None:
        print(f"  Superlative: No valid template for {n_entities}x {entity_type}", flush=True)
        return []

    print(f"  Superlative Phase 0: {reasoning_ctx['template_id']} "
          f"n={n_entities} -> {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1 + 1.5: Extract and ground clue facts for each entity
    clues_per_entity: Dict[str, List[dict]] = {}
    extracted_facts_per_entity: Dict[str, List[dict]] = {}
    grounding_articles_per_entity: Dict[str, List[dict]] = {}

    for i, label in enumerate(labels):
        ed = entity_data_list[i]
        chains = ed.get("chains", [])
        articles = ed.get("articles", [])
        name = ed["name"]

        if not chains or len(chains) < 2:
            print(f"  Superlative: Insufficient chains for {name} ({label})", flush=True)
            return []

        # Phase 1: Extract clue facts
        facts, _ = extract_chain_clue_facts(chains, name, agent_info)
        if facts is None or len(facts) < 2:
            print(f"  Superlative Phase 1: Insufficient facts for {name} ({label})", flush=True)
            return []

        # Phase 1.5: Verify facts against Wikipedia
        grounded = verify_facts_grounding(facts, chains, articles, agent_info)
        if grounded is None or len(grounded) < 2:
            print(f"  Superlative Phase 1.5: Grounding failed for {name} ({label})", flush=True)
            return []
        facts = grounded

        extracted_facts_per_entity[label] = facts
        clues_per_entity[label] = [
            {
                "fact_id": f.get("fact_id", f"C{f.get('chain_num', 0)}_F1"),
                "topic": f.get("property", "description"),
                "fact": f.get("fact", ""),
            }
            for f in facts
        ]

        # Build grounding articles per entity
        grounding_articles_per_entity[label] = build_multiskill_grounding_articles(
            name, reasoning_ctx["gold_answer"],
            reasoning_ctx["template"]["answer_unit"],
            facts, chains, articles,
        )

    # Phase 2: Compose question
    composition = compose_history_superlative_question(
        clues_per_entity, reasoning_ctx, entity_names, prev_questions, n_entities,
        entities_data=entities_data,
    )
    if composition is None:
        print(f"  Superlative Phase 2: Composition failed", flush=True)
        return []

    question = composition["question"]
    used_facts_per_entity = composition["used_facts_per_entity"]
    print(f"  Superlative Phase 2: \"{question[:80]}...\"", flush=True)

    # Phase 3: Validation
    # 3a: Recompute gold answer
    recomputed = _execute_computation_code(reasoning_ctx["code"])
    if recomputed != reasoning_ctx["gold_answer"]:
        print(f"  Superlative Phase 3: Recomputation mismatch "
              f"({recomputed} != {reasoning_ctx['gold_answer']})", flush=True)
        return []

    # 3b: Name leak — all N names + QIDs must not appear in question
    all_names = []
    for ed in entity_data_list[:n_entities]:
        all_names.append(ed["name"])
        if ed.get("qid"):
            all_names.append(ed["qid"])
    for nm in all_names:
        if nm.lower() in question.lower():
            print(f"  Superlative Phase 3: Name leak '{nm}'", flush=True)
            return []

    # 3c: Value leak — all years from all N entities checked
    all_years = []
    for ed in entity_data_list[:n_entities]:
        all_years.extend(ed.get("data", {}).get("_all_years", []))
    q_lower = question.lower()
    for yr in all_years:
        yr_str = str(yr)
        if len(yr_str) >= 4 and yr_str in q_lower:
            print(f"  Superlative Phase 3: Year value leak '{yr_str}'", flush=True)
            return []

    # 3d: Every entity must have >= 1 fact referenced
    for label in labels:
        if not used_facts_per_entity.get(label):
            print(f"  Superlative Phase 3: No used facts for entity {label}", flush=True)
            return []

    # 3e: History-specific question-code consistency
    passed, reason = _check_history_question_consistency(
        question, reasoning_ctx["template_id"], reasoning_ctx["gold_answer"],
        entity_type=entity_type,
    )
    if not passed:
        print(f"  Superlative Phase 3 (history): {reason}", flush=True)
        return []

    # 3e2: LLM-based question-computation match
    passed, reason = _check_history_question_computation_match(
        question, reasoning_ctx["question_hint"], reasoning_ctx["template_id"],
    )
    if not passed:
        print(f"  Superlative Phase 3 (history-comp-match): {reason}", flush=True)
        return []

    # 3f: Multi-entity uniqueness (LLM-based)
    entity_refs = [f"Entity {label}" for label in labels]
    id_ok = _check_history_multi_entity_uniqueness(
        question, entity_names, entity_refs,
    )
    if not id_ok:
        print(f"  Superlative Phase 3: Multi-entity uniqueness check failed", flush=True)
        return []

    print(f"  Superlative Phase 3: PASSED", flush=True)

    # Build output
    _ESSENTIAL_FACT_KEYS = ("fact_id", "chain_num", "entity", "property", "value", "fact")
    _grounding_keys = {"birth_date", "death_date", "start_date", "end_date",
                       "duration_years", "casualties", "population",
                       "year_founded", "year_dissolved", "_all_years"}

    grounding_clues_per_entity = {}
    grounding_entity_data_per_entity = {}
    facts_for_output_per_entity = {}
    source_triples_per_entity = {}

    for i, label in enumerate(labels):
        ed = entity_data_list[i]
        data = ed.get("data", {})
        chains = ed.get("chains", [])

        grounding_clues_per_entity[label] = clues_per_entity[label]
        grounding_entity_data_per_entity[label] = {
            k: data[k] for k in _grounding_keys if k in data
        }
        facts_for_output_per_entity[label] = [
            {k: f[k] for k in _ESSENTIAL_FACT_KEYS if k in f}
            for f in extracted_facts_per_entity.get(label, [])
        ]
        source_triples_per_entity[label] = [
            c["path_description"] for c in chains[:5]
        ]

    # CCI fields computed inline (N > 2 entities)
    tmpl = reasoning_ctx["template"]
    total_facts = sum(len(v) for v in extracted_facts_per_entity.values())
    unique_chains = set()
    for label_facts in extracted_facts_per_entity.values():
        for f in label_facts:
            unique_chains.add(f.get("chain_num", 0))

    qa_pair = {
        "question": question,
        "gold_answer": reasoning_ctx["gold_answer"],
        "computation_code": reasoning_ctx["code"],
        "template_id": reasoning_ctx["template_id"],
        "template_label": tmpl["label"],
        "template_type": "superlative",
        "template_level": tmpl["level"],
        "answer_unit": tmpl["answer_unit"],
        "entity_names": entity_names,
        "n_entities": n_entities,
        "entity_type": entity_type,
        "category": category,
        "used_facts_per_entity": used_facts_per_entity,
        "extracted_facts_per_entity": facts_for_output_per_entity,
        "source_triples_per_entity": source_triples_per_entity,
        "grounding_articles_per_entity": grounding_articles_per_entity,
        "grounding_clues_per_entity": grounding_clues_per_entity,
        "grounding_entity_data_per_entity": grounding_entity_data_per_entity,
        "data_source": "wikidata_temporal",
        # CCI fields
        "num_entities": n_entities,
        "reasoning_depth": tmpl.get("reasoning_depth", 3),
        "num_clue_facts": total_facts,
        "num_clue_chains": len(unique_chains),
    }

    for i, label in enumerate(labels):
        ed = entity_data_list[i]
        if ed.get("qid"):
            qa_pair[f"qid_{label.lower()}"] = ed["qid"]

    used_templates.add(reasoning_ctx["template_id"])
    return [qa_pair]


# ===========================================================================
# V2 Verification: History + Browser + Python Tools
# ===========================================================================

def verify_history_v2(qa_pairs, num_samples=10, temperature=1.0,
                       max_iterations=200, threshold=0.5,
                       outfile_prefix=None, subarea="",
                       bench_label="history_bench"):
    """V2 verification using History + Browser + Python tools."""
    all_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v2.json" if outfile_prefix else None
    filtered_cache = f"{outfile_prefix}__{subarea}.{bench_label}_v2_filtered.json" if outfile_prefix else None

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
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available for history V2")
    if not _HAS_PYTHON_TOOL:
        raise RuntimeError("HybridPythonTool not available for history V2")
    if not _HAS_HISTORY_TOOL:
        raise RuntimeError("HistoryTool not available for history V2")

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== HISTORY V2 VERIFICATION ({subarea}) ===", flush=True)
    print(f"Tools: History + Browser + Python", flush=True)
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
            history_tool = None
            qid_str = f"{bench_label}_v2_{subarea}_{idx}_{i}"
            try:
                backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
                browser_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
                python_tool = HybridPythonTool(timeout=60)
                python_tool.set_qid(qid_str)
                history_tool = HistoryTool()

                system_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.HIGH)
                    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                    .with_tools(browser_tool.tool_config)
                    .with_tools(python_tool.tool_config)
                    .with_tools(history_tool.tool_config)
                )

                messages = [
                    Message.from_role_and_content(Role.SYSTEM, system_content),
                    Message.from_role_and_content(Role.DEVELOPER, HISTORY_V2_DEVELOPER),
                    Message.from_role_and_content(Role.USER, f"Question: {qa['question']}"),
                ]

                tool_call_counter = [0]

                async def _tool_handler(msg, _browser=browser_tool, _python=python_tool,
                                        _history=history_tool, _counter=tool_call_counter):
                    _counter[0] += 1
                    recipient = str(getattr(msg, 'recipient', ''))
                    results = []
                    if recipient.startswith("history"):
                        async for m in _history.process(msg):
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
                    tool_prefix=("history", "browser.", "python"),
                    tool_configs=[browser_tool.tool_config, python_tool.tool_config, history_tool.tool_config],
                    max_iterations=max_iterations,
                    temperature=temperature,
                )

                iteration_count = len(result_messages) - len(messages)

                answer = _extract_history_answer(result_messages)
                answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
                answer = ' '.join(answer.split())
                correct = bool(is_history_answer_correct(answer, qa['gold_answer']) or
                               _llm_judge_history_answer(answer, qa['gold_answer'], qa['question']))
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


def _extract_history_answer(messages):
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

def _generate_category_qa_pairs(category, per_category, agent_info,
                                 prefix, bench_label,
                                 num_chains=20, questions_per_entity=5):
    """Generate QA pairs for a single entity category.

    Uses the unified chains pipeline:
    1. Fetch entities → temporal data + KG chains + Wikipedia articles
    2. Cache as {prefix}__{category}.history_chains.json
    3. Generate QA pairs per entity via chain-based Phase 1/1.5/2/3
    """
    entity_type = _entity_type_from_category(category)

    # Check chains cache (new pattern, replaces history_entity_data.json)
    chains_cache = f"{prefix}__{category}.history_chains.json"
    old_data_cache = f"{prefix}__{category}.history_entity_data.json"

    entity_data_list = []

    if os.path.exists(chains_cache):
        print(f"Loading cached chains for {category}...", flush=True)
        with open(chains_cache) as f:
            entity_data_list = json.load(f)

        if not entity_data_list:
            # Empty cache — delete and fall through to fresh-fetch
            print(f"  Chains cache for {category} is empty, deleting and regenerating...",
                  flush=True)
            os.remove(chains_cache)
        else:
            # Backfill Wikipedia articles for cached entities that lack them
            needs_resave = False
            for ed in entity_data_list:
                if "articles" not in ed or not ed["articles"]:
                    print(f"  Fetching Wikipedia articles for cached entity {ed.get('name', '')}...",
                          flush=True)
                    ed["articles"] = fetch_wikipedia_for_entities(ed.get("chains", []))
                    print(f"  {ed.get('name', '')}: fetched {len(ed['articles'])} Wikipedia articles",
                          flush=True)
                    needs_resave = True

            # Backfill temporal grounding for cached entities that lack it
            verified_list = []
            for ed in entity_data_list:
                if ed.get("temporal_grounded"):
                    verified_list.append(ed)
                    continue
                # Run temporal grounding verification
                data = ed.get("data", {})
                name = ed.get("name", "")
                qid = ed.get("qid", "")
                articles = ed.get("articles", [])
                grounded = verify_temporal_grounding(data, name, qid, articles, agent_info)
                if grounded is None:
                    print(f"  {name} ({qid}): temporal data not grounded by Wikipedia, "
                          f"removing from cache", flush=True)
                    needs_resave = True
                    continue
                ed["temporal_grounded"] = True
                verified_list.append(ed)
                needs_resave = True
            if len(verified_list) < len(entity_data_list):
                print(f"  Temporal grounding removed {len(entity_data_list) - len(verified_list)} "
                      f"entities from {category} cache", flush=True)
            entity_data_list = verified_list

            # Append new entities from ENTITY_UNIVERSE that aren't in the cache yet
            cached_names = {ed.get("name", "") for ed in entity_data_list}
            cached_qids = {ed.get("qid", "") for ed in entity_data_list}
            universe_entities = get_category_entities(category)
            new_entities = [
                e for e in universe_entities
                if e.get("name", "") not in cached_names and e.get("qid", "") not in cached_qids
            ]
            if new_entities:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"  Found {len(new_entities)} new entities in ENTITY_UNIVERSE "
                      f"not yet cached for {category}, fetching...", flush=True)
                for entity in tqdm.tqdm(new_entities, desc=f"Appending new {category} entities"):
                    time.sleep(2)
                    name = entity.get("name", "")
                    qid = entity.get("qid", "")
                    wiki = entity.get("wiki", name)

                    data = get_all_entity_data(qid, entity_type)
                    if data is None:
                        print(f"  {name} ({qid}): get_all_entity_data returned None, skipping",
                              flush=True)
                        continue

                    has_useful_data = False
                    if entity_type == "figure":
                        has_useful_data = data.get("birth_year") is not None
                    elif entity_type == "conflict":
                        has_useful_data = data.get("start_year") is not None
                    elif entity_type == "organization":
                        has_useful_data = data.get("inception_year") is not None
                    elif entity_type == "milestone":
                        has_useful_data = data.get("event_year") is not None

                    if not has_useful_data:
                        print(f"  {name} ({qid}): no useful temporal data, skipping", flush=True)
                        continue

                    chains = fetch_multihop_triples(qid, num_hops=2, limit=num_chains)
                    if chains and len(chains) >= 2:
                        articles = fetch_wikipedia_for_entities(chains)
                    else:
                        chains = chains or []
                        articles = []

                    # Verify temporal data against Wikipedia articles
                    data = verify_temporal_grounding(data, name, qid, articles, agent_info)
                    if data is None:
                        print(f"  {name} ({qid}): temporal data not grounded by Wikipedia, skipping",
                              flush=True)
                        continue

                    clues = fetch_entity_clues(wiki, entity_type)
                    entity_data_list.append({
                        "name": name,
                        "qid": qid,
                        "entity_type": entity_type,
                        "data": data,
                        "clues": clues,
                        "chains": chains,
                        "articles": articles,
                        "obscurity": entity.get("obscurity", 1),
                        "temporal_grounded": True,
                        "cached_at": now_str,
                    })
                    print(f"  {name}: appended ({len(chains)} chains, {len(articles)} articles)",
                          flush=True)
                needs_resave = True

            if needs_resave:
                with open(chains_cache, "w") as f:
                    json.dump(entity_data_list, f, indent=2)
                print(f"Re-saved chains cache for {category} ({len(entity_data_list)} entities)",
                      flush=True)

    if not entity_data_list and os.path.exists(old_data_cache):
        # Backfill: upgrade old entity_data cache to chains cache
        print(f"Upgrading old entity data cache for {category} to chains format...", flush=True)
        with open(old_data_cache) as f:
            old_list = json.load(f)

        entity_data_list = []
        for ed in tqdm.tqdm(old_list, desc=f"Upgrading {category} cache"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            qid = ed.get("qid", "")
            name = ed.get("name", "")
            if not qid:
                continue

            chains = fetch_multihop_triples(qid, num_hops=2, limit=num_chains)
            if not chains or len(chains) < 2:
                print(f"  {name}: insufficient KG chains ({len(chains) if chains else 0}), skipping",
                      flush=True)
                # Keep entity with empty chains — can still use legacy clues
                ed["chains"] = chains or []
                ed["articles"] = []
                entity_data_list.append(ed)
                continue

            articles = fetch_wikipedia_for_entities(chains)
            print(f"  {name}: fetched {len(articles)} Wikipedia articles "
                  f"for {len(chains)} chains", flush=True)

            # Verify temporal data against Wikipedia articles
            data = ed.get("data", {})
            grounded = verify_temporal_grounding(data, name, qid, articles, agent_info)
            if grounded is None:
                print(f"  {name} ({qid}): temporal data not grounded by Wikipedia, skipping",
                      flush=True)
                continue

            ed["chains"] = chains
            ed["articles"] = articles
            ed["temporal_grounded"] = True
            entity_data_list.append(ed)

        if entity_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Upgraded and cached {len(entity_data_list)} entities for {category}", flush=True)

    if not entity_data_list:
        entities = get_category_entities(category)
        if not entities:
            print(f"No entities for category {category}", flush=True)
            return []

        for entity in tqdm.tqdm(entities, desc=f"Fetching {category} data"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            name = entity.get("name", "")
            qid = entity.get("qid", "")
            wiki = entity.get("wiki", name)

            # Fetch temporal data from Wikidata
            data = get_all_entity_data(qid, entity_type)
            if data is None:
                print(f"  {name} ({qid}): get_all_entity_data returned None, skipping",
                      flush=True)
                continue

            # Check minimum data availability
            has_useful_data = False
            if entity_type == "figure":
                if data.get("birth_year") is not None:
                    has_useful_data = True
            elif entity_type == "conflict":
                if data.get("start_year") is not None:
                    has_useful_data = True
            elif entity_type == "organization":
                if data.get("inception_year") is not None:
                    has_useful_data = True
            elif entity_type == "milestone":
                if data.get("event_year") is not None:
                    has_useful_data = True

            if not has_useful_data:
                print(f"  {name} ({qid}): no useful temporal data "
                      f"(type={entity_type}, keys={list(data.keys())}), skipping",
                      flush=True)
                continue

            # Fetch multi-hop KG chains
            chains = fetch_multihop_triples(qid, num_hops=2, limit=num_chains)
            if chains and len(chains) >= 2:
                # Fetch Wikipedia articles for chain entities (grounding documents)
                articles = fetch_wikipedia_for_entities(chains)
                print(f"  {name}: fetched {len(articles)} Wikipedia articles "
                      f"for {len(chains)} chains", flush=True)
            else:
                # Keep entity with empty chains — can still use legacy clues
                chains = chains or []
                articles = []
                print(f"  {name}: insufficient KG chains ({len(chains)}), will use legacy clues",
                      flush=True)

            # Verify temporal data against Wikipedia articles
            data = verify_temporal_grounding(data, name, qid, articles, agent_info)
            if data is None:
                print(f"  {name} ({qid}): temporal data not grounded by Wikipedia, skipping",
                      flush=True)
                continue

            # Also fetch legacy clues as fallback
            clues = fetch_entity_clues(wiki, entity_type)

            entity_data_list.append({
                "name": name,
                "qid": qid,
                "entity_type": entity_type,
                "data": data,
                "clues": clues,
                "chains": chains,
                "articles": articles,
                "obscurity": entity.get("obscurity", 1),
                "temporal_grounded": True,
                "cached_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            if len(entity_data_list) >= per_category * 2:
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

    # Sort entities by obscurity (tier 1 = less famous, processed first)
    # Entities without an obscurity field default to tier 1 (obscure).
    entity_data_list.sort(key=lambda ed: ed.get("obscurity", 1))

    # Generate QA pairs
    all_qa_pairs = []
    prev_questions = []
    used_templates = set()

    # Single-entity questions (per-entity loop)
    # Cap per entity: max 2 questions per entity to ensure diversity
    MAX_QA_PER_ENTITY = 2
    target_single = per_category * 2 // 3
    entity_qa_counts: Dict[str, int] = {}  # track per-entity QA count

    for ed in entity_data_list:
        if len([q for q in all_qa_pairs if q.get("template_type") == "single"]) >= target_single:
            break

        ent_name = ed.get("name", "")
        if entity_qa_counts.get(ent_name, 0) >= MAX_QA_PER_ENTITY:
            continue  # skip entities that already contributed enough

        chains = ed.get("chains", [])
        articles = ed.get("articles", [])
        clues = ed.get("clues", [])

        remaining = MAX_QA_PER_ENTITY - entity_qa_counts.get(ent_name, 0)

        # Use chain-based pipeline if chains available, else legacy clues
        if chains and len(chains) >= 2:
            entity_qas = gen_history_qa_from_entity(
                ed, agent_info,
                num_questions=min(remaining, questions_per_entity),
                prev_questions=prev_questions,
                used_templates=used_templates,
                articles=articles,
                category=category,
            )
            for qa in entity_qas:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
            entity_qa_counts[ent_name] = entity_qa_counts.get(ent_name, 0) + len(entity_qas)
            if entity_qas:
                print(f"  {ed['name']}: generated {len(entity_qas)} chain-based QA pairs "
                      f"(total: {len(all_qa_pairs)})", flush=True)
        else:
            # Legacy fallback
            qa = gen_history_qa_single(
                ed["name"], ed.get("data", {}), clues, ed.get("entity_type", entity_type),
                used_templates, prev_questions, category,
            )
            if qa:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
                entity_qa_counts[ent_name] = entity_qa_counts.get(ent_name, 0) + 1
                print(f"  {ed['name']}: single QA generated (legacy) "
                      f"(total: {len(all_qa_pairs)})", flush=True)

        # Reset used_templates after cycling through all single templates for this entity type
        et = ed.get("entity_type", entity_type)
        single_templates = [t for t, d in HISTORY_TEMPLATES.items()
                           if d["type"] == "single" and et in d["entity_type"]]
        if len(used_templates & set(single_templates)) >= len(single_templates):
            used_templates -= set(single_templates)

    # Comparative questions
    target_comp = per_category // 3
    comp_used = set()
    for i in range(0, len(entity_data_list) - 1, 2):
        if len([q for q in all_qa_pairs if q.get("template_type") == "comparative"]) >= target_comp:
            break

        ed_a = entity_data_list[i]
        ed_b = entity_data_list[i + 1]

        et_a = ed_a.get("entity_type", _entity_type_from_category(category))
        et_b = ed_b.get("entity_type", _entity_type_from_category(category))
        if et_a != et_b:
            continue

        # Retry a rejected pair a few times: comparative composition is
        # one-shot per call, so a single stochastic Phase-3 rejection would
        # otherwise discard a viable pair (single-entity gen gets
        # num_questions*3 attempts; give each comparative pair a few too).
        qa = None
        for _ in range(3):
            qa = gen_history_qa_comparative(
                ed_a["name"], ed_a.get("data", {}), ed_a.get("clues", []),
                ed_b["name"], ed_b.get("data", {}), ed_b.get("clues", []),
                et_a, comp_used, prev_questions, category,
                chains_a=ed_a.get("chains", []),
                articles_a=ed_a.get("articles", []),
                chains_b=ed_b.get("chains", []),
                articles_b=ed_b.get("articles", []),
                agent_info=agent_info,
            )
            if qa:
                break
        if qa:
            all_qa_pairs.append(qa)
            prev_questions.append(qa["question"])
            print(f"  {ed_a['name']} vs {ed_b['name']}: comparative QA generated "
                  f"(total: {len(all_qa_pairs)})", flush=True)

    # Superlative questions (N-entity ranking)
    target_sup = per_category // 4
    sup_used = set()
    sup_count = 0

    # Filter entities with >=2 chains and useful temporal data
    sup_eligible = [
        ed for ed in entity_data_list
        if len(ed.get("chains", [])) >= 2 and ed.get("data")
    ]
    random.shuffle(sup_eligible)

    # Try N=5 first, then N=3
    for n_ent in [5, 3]:
        if sup_count >= target_sup:
            break
        if len(sup_eligible) < n_ent:
            continue

        # Form non-overlapping groups of n_ent
        used_names = set()
        groups = []
        for ed in sup_eligible:
            if ed["name"] in used_names:
                continue
            # Start a group
            group = [ed]
            used_names.add(ed["name"])
            for ed2 in sup_eligible:
                if len(group) >= n_ent:
                    break
                if ed2["name"] not in used_names:
                    group.append(ed2)
                    used_names.add(ed2["name"])
            if len(group) == n_ent:
                groups.append(group)
            if sup_count + len(groups) >= target_sup:
                break

        for group in groups:
            if sup_count >= target_sup:
                break
            qas = gen_history_qa_superlative(
                group, agent_info,
                n_entities=n_ent,
                prev_questions=prev_questions,
                used_templates=sup_used,
                category=category,
            )
            for qa in qas:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
                sup_count += 1
                names_str = ", ".join(ed["name"] for ed in group)
                print(f"  Superlative ({n_ent}): {names_str} -> "
                      f"template={qa['template_id']} (total: {len(all_qa_pairs)})",
                      flush=True)

    return all_qa_pairs


def _generate_cross_category_qa_pairs(entity_data_by_category, per_category,
                                       prefix, bench_label, prev_questions):
    """Generate cross-category QA pairs (figure + event/conflict)."""
    figures = entity_data_by_category.get("figures", [])
    conflicts = entity_data_by_category.get("conflicts", [])
    milestones = entity_data_by_category.get("milestones", [])

    if not figures:
        return []

    events = []
    for ed in conflicts:
        events.append(("conflict", ed))
    for ed in milestones:
        events.append(("milestone", ed))

    if not events:
        return []

    all_qa_pairs = []
    used_templates = set()
    target = per_category // 4  # About 25% cross-category

    random.shuffle(figures)
    random.shuffle(events)

    for fig in figures:
        if len(all_qa_pairs) >= target:
            break
        for event_type, evt in events:
            if len(all_qa_pairs) >= target:
                break

            qa = gen_history_qa_cross_category(
                fig["name"], fig["data"], fig["clues"],
                evt["name"], evt["data"], evt["clues"],
                event_type, used_templates, prev_questions, "cross_category",
            )
            if qa:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
                print(f"  {fig['name']} + {evt['name']}: cross-category QA generated "
                      f"(total: {len(all_qa_pairs)})", flush=True)
                break  # One cross-category per figure

    return all_qa_pairs


def run_history_bench(args, agent_info):
    """Run the history benchmark pipeline.

    Per category:
    1. Fetch entities + temporal data + Wikipedia clues
    2. Generate QA pairs by template type
    3. V1 verification (closed-book)
    4. V2 verification (History + Browser + Python tools)
    Merge all categories into final output
    """
    prefix = args.outfile_prefix1
    run_id = getattr(args, "run_id", None) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{prefix}__{run_id}"
    categories_arg = args.history_categories
    per_category = args.history_per_category
    v1_samples = args.history_v1_samples
    v2_samples = args.history_v2_samples
    v1_threshold = args.history_v1_threshold
    v2_threshold = args.history_v2_threshold
    num_chains = getattr(args, "history_num_chains", 20)
    questions_per_entity = getattr(args, "history_questions_per_entity", 5)

    bench_label = "history_bench"

    if categories_arg:
        categories_to_process = [s.strip() for s in categories_arg.split(",")]
    else:
        categories_to_process = list(ENTITY_UNIVERSE.keys())

    is_subset = categories_arg is not None and len(categories_to_process) < len(ENTITY_UNIVERSE)

    print(f"=== HISTORICAL/CAUSAL BENCH (run_id={run_id}) ===", flush=True)
    print(f"Categories: {categories_to_process}", flush=True)
    print(f"Target QAs per category: {per_category}", flush=True)
    print(f"V1: {v1_samples} samples, threshold < {v1_threshold:.0%}", flush=True)
    print(f"V2: {v2_samples} samples, threshold < {v2_threshold:.0%}", flush=True)
    print("=" * 50, flush=True)

    all_final = []
    summary = {}
    entity_data_by_category = {}

    for category in categories_to_process:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Category: {category.upper()} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        output_dir = os.path.dirname(run_prefix) if "/" in run_prefix else None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check full cache (V2)
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
        else:
            # Check V1 filtered cache
            v1_filtered_cache = f"{run_prefix}__{category}.{bench_label}_v1_filtered.json"
            if os.path.exists(v1_filtered_cache):
                print(f"Found cached V1 filtered for {category}, skipping to V2...", flush=True)
                with open(v1_filtered_cache) as f:
                    v1_filtered = json.load(f)
            else:
                bench_cache = f"{run_prefix}__{category}.{bench_label}.json"
                if os.path.exists(bench_cache):
                    print(f"Found cached bench problems for {category}...", flush=True)
                    with open(bench_cache) as f:
                        qa_pairs = json.load(f)
                else:
                    qa_pairs = _generate_category_qa_pairs(
                        category, per_category, agent_info, prefix, bench_label,
                        num_chains=num_chains, questions_per_entity=questions_per_entity,
                    )

                    if qa_pairs:
                        with open(bench_cache, "w") as f:
                            json.dump(qa_pairs, f, indent=2)
                        print(f"Saved {len(qa_pairs)} raw QA pairs to {bench_cache}", flush=True)

                if not qa_pairs:
                    print(f"No QA pairs for category {category}", flush=True)
                    summary[category] = {"generated": 0, "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                    continue

                v1_filtered, v1_all = verify_bench_v1(
                    qa_pairs, agent_info,
                    num_samples=v1_samples,
                    temperature=0.7,
                    threshold=v1_threshold,
                    outfile_prefix=run_prefix,
                    subarea=category,
                    bench_label=bench_label,
                    answer_checker=is_history_answer_correct,
                    llm_judge=_llm_judge_history_answer,
                    verification_prompt=HISTORY_V1_PROMPT,
                )

            if not v1_filtered:
                print(f"No V1-filtered QA pairs for {category}", flush=True)
                summary[category] = {"generated": len(qa_pairs) if 'qa_pairs' in dir() else "cached",
                                     "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                continue

            v2_filtered, v2_all = verify_history_v2(
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

    if is_subset:
        print(f"\nSubset mode: processed {categories_to_process}. "
              f"Merge will happen in the merge job.", flush=True)
        _print_summary(summary)
        return all_final

    # Final merge
    final_path = f"{run_prefix}.{bench_label}_final.json"
    if all_final:
        # Deduplicate across categories
        seen_keys = set()
        deduped = []
        for qa in all_final:
            if qa.get("template_type") == "superlative":
                key = (qa.get("template_id", ""),
                       tuple(sorted(qa.get("entity_names", []))),
                       qa.get("gold_answer", ""))
            else:
                key = (qa.get("template_id", ""),
                       qa.get("entity_name", qa.get("entity_name_a", "")),
                       qa.get("entity_name_b", ""),
                       qa.get("gold_answer", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(qa)
        if len(deduped) < len(all_final):
            print(f"\nDeduplication: {len(all_final)} -> {len(deduped)} "
                  f"({len(all_final) - len(deduped)} duplicates removed)", flush=True)
        all_final = deduped

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
    print("HISTORY BENCH SUMMARY", flush=True)
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
        prog="history_drbencher",
        description="Historical/Causal Benchmark: Wikidata temporal + Wikipedia entity ID",
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
    parser.add_argument("--outfile_prefix1", type=str, default="output/history/bench")

    # Exp mode
    parser.add_argument("--exp_mode", type=str, default="history_bench")

    # Run ID for multi-run support (auto-generated if not provided)
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: auto YYYYMMDD_HHMMSS)")

    # History-specific
    parser.add_argument("--history_categories", type=str, default=None,
                        help="Comma-separated categories (default: all). E.g. 'conflicts,figures'")
    parser.add_argument("--history_per_category", type=int, default=50,
                        help="Target QA pairs per category")
    parser.add_argument("--history_v1_samples", type=int, default=10,
                        help="V1 sampling attempts per question")
    parser.add_argument("--history_v2_samples", type=int, default=10,
                        help="V2 (History+Browser+Python) sampling attempts per question")
    parser.add_argument("--history_v1_threshold", type=float, default=0.5,
                        help="V1 accuracy ceiling -- keep below this")
    parser.add_argument("--history_v2_threshold", type=float, default=0.5,
                        help="V2 accuracy ceiling -- keep below this")
    parser.add_argument("--history_num_chains", type=int, default=20,
                        help="Number of KG chains to fetch per entity")
    parser.add_argument("--history_questions_per_entity", type=int, default=5,
                        help="Questions to generate per entity (chain-based pipeline)")

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

    if args.exp_mode == "history_bench":
        run_history_bench(args, agent_info)
        # The in-process vLLM engine spawns worker subprocesses that outlive the
        # bench; shut the engine down and terminate the process group so the job
        # exits cleanly instead of hanging until it is killed by hand.
        shutdown_and_exit(0)
    else:
        print(f"Unknown exp_mode: {args.exp_mode}", flush=True)
        sys.exit(1)
