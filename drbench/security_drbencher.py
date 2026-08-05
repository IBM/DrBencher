# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Security/Cyber Benchmark: Cryptographic Standards + MITRE ATT&CK.

Creates questions in a 2-level difficulty space:
  Level 1 (Single-Entity): identify algorithm/group/technique from clues -> fetch data -> compute
  Level 2 (Multi-Step):    multi-step crypto/ATT&CK computations with known parameters

Data sources:
  Cryptographic algorithms: NIST/ISO/IETF standards (key sizes, block sizes, rounds, security strength)
  MITRE ATT&CK:  threat groups, techniques, software (technique counts, tactic coverage, etc.)
  Wikipedia:      entity identification clues via multi-hop KG chains

Run via:  python -m drbench.security_drbencher --exp_mode security_bench ...
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

# Security utilities (crypto + ATT&CK reference data lookups)
from .security_util import (
    get_crypto_algorithm_data,
    get_attack_entity_data,
    search_crypto_algorithms,
    search_attack_entities,
    fetch_entity_clues,
    get_category_entities,
    resolve_security_wikidata_id,
    ENTITY_UNIVERSE,
    SECURITY_THEMES,
    CRYPTO_REFERENCE_DATA,
    ATTACK_REFERENCE_DATA,
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

# Security tool for V2 verification
try:
    from tools.security_tool import SecurityTool
    _HAS_SECURITY_TOOL = True
except ImportError:
    _HAS_SECURITY_TOOL = False

# Diversity filter (optional)
try:
    from .diversity import diversity_filter, diversity_report
    _HAS_DIVERSITY = True
except ImportError:
    _HAS_DIVERSITY = False


# ===========================================================================
# Security Templates (Cryptographic Standards + MITRE ATT&CK)
# ===========================================================================

SECURITY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # --- Crypto Single (12) ---
    "key_size_lookup": {
        "level": 1, "type": "single", "entity_type": "algorithm",
        "label": "Key Size Lookup",
        "required_data": ["key_size"],
        "code_template": "print({key_size})",
        "question_hint": "What is the key size (in bits) of this cryptographic algorithm?",
        "answer_unit": "bits", "reasoning_depth": 1,
    },
    "block_size_lookup": {
        "level": 1, "type": "single", "entity_type": "algorithm",
        "label": "Block Size Lookup",
        "required_data": ["block_size"],
        "code_template": "print({block_size})",
        "question_hint": "What is the block size (in bits) of this cipher?",
        "answer_unit": "bits", "reasoning_depth": 1,
    },
    "output_size_lookup": {
        "level": 1, "type": "single", "entity_type": "algorithm",
        "label": "Output/Digest Size Lookup",
        "required_data": ["output_size"],
        "code_template": "print({output_size})",
        "question_hint": "What is the output (digest) size in bits of this hash function?",
        "answer_unit": "bits", "reasoning_depth": 1,
    },
    "rounds_count": {
        "level": 1, "type": "single", "entity_type": "algorithm",
        "label": "Round Count Lookup",
        "required_data": ["rounds"],
        "code_template": "print({rounds})",
        "question_hint": "How many rounds does this cryptographic algorithm use?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    "security_strength": {
        "level": 1, "type": "single", "entity_type": "algorithm",
        "label": "Security Strength",
        "required_data": ["security_strength"],
        "code_template": "print({security_strength})",
        "question_hint": "What is the security strength (in bits) of this algorithm?",
        "answer_unit": "bits", "reasoning_depth": 1,
    },
    "key_to_block_ratio": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Key-to-Block Size Ratio",
        "required_data": ["key_size", "block_size"],
        "code_template": "print(round({key_size} / {block_size}, 2))",
        "question_hint": "What is the ratio of key size to block size for this cipher?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "security_margin": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Security Margin (Key - Strength)",
        "required_data": ["key_size", "security_strength"],
        "code_template": "print({key_size} - {security_strength})",
        "question_hint": "What is the security margin (key size minus security strength in bits) for this algorithm?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    "rounds_per_block_bit": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Rounds per Block Bit",
        "required_data": ["rounds", "block_size"],
        "code_template": "print(round({rounds} / {block_size}, 4))",
        "question_hint": "What is the number of rounds per block bit for this cipher?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "birthday_bound": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Birthday Bound",
        "required_data": ["output_size"],
        "code_template": "print({output_size} // 2)",
        "question_hint": "What is the birthday bound (in bits) for collision resistance of this hash function?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    "grover_effective": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Post-Quantum Effective Strength (Grover)",
        "required_data": ["key_size"],
        "code_template": "print({key_size} // 2)",
        "question_hint": "Under Grover's algorithm, what is the effective post-quantum security strength (in bits) of this algorithm?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    "aead_overhead_bits": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "AEAD Authentication Overhead",
        "required_data": ["nonce_size", "tag_size"],
        "code_template": "print({nonce_size} + {tag_size})",
        "question_hint": "What is the total authentication overhead (nonce + tag, in bits) for this AEAD scheme?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    "brute_force_log2": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Brute-Force Work (log2)",
        "required_data": ["key_size"],
        "code_template": "print({key_size})",
        "question_hint": "How many bits of brute-force work (log2 of key space) are needed to break this algorithm?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    # --- Crypto Comparative (5) ---
    "key_size_difference": {
        "level": 1, "type": "comparative", "entity_type": "algorithm",
        "label": "Key Size Difference",
        "required_data": ["key_size"],
        "code_template": "print(abs({key_size_b} - {key_size_a}))",
        "question_hint": "What is the difference in key sizes (in bits) between these two algorithms?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    "security_strength_ratio": {
        "level": 2, "type": "comparative", "entity_type": "algorithm",
        "label": "Security Strength Ratio",
        "required_data": ["security_strength"],
        "code_template": "print(round({security_strength_a} / {security_strength_b}, 2))",
        "question_hint": "What is the ratio of security strengths (first / second) between these two algorithms?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "rounds_difference": {
        "level": 1, "type": "comparative", "entity_type": "algorithm",
        "label": "Round Count Difference",
        "required_data": ["rounds"],
        "code_template": "print(abs({rounds_b} - {rounds_a}))",
        "question_hint": "What is the difference in round counts between these two algorithms?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "output_size_ratio": {
        "level": 2, "type": "comparative", "entity_type": "algorithm",
        "label": "Output Size Ratio",
        "required_data": ["output_size"],
        "code_template": "print(round({output_size_a} / {output_size_b}, 2))",
        "question_hint": "What is the ratio of output sizes (first / second) between these two hash functions?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "year_gap": {
        "level": 1, "type": "comparative", "entity_type": "algorithm",
        "label": "Publication Year Gap",
        "required_data": ["year"],
        "code_template": "print(abs({year_b} - {year_a}))",
        "question_hint": "How many years apart were these two algorithms published?",
        "answer_unit": "years", "reasoning_depth": 1,
    },
    # --- Crypto Multi-Step (8) ---
    "key_space_exponent_diff": {
        "level": 2, "type": "comparative", "entity_type": "algorithm",
        "label": "Key Space Exponent Difference",
        "required_data": ["key_size"],
        "code_template": "print({key_size_a} - {key_size_b})",
        "question_hint": "What is the difference in log2 key space between these two algorithms?",
        "answer_unit": "bits", "reasoning_depth": 3,
    },
    "security_strength_per_round": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Security Strength per Round",
        "required_data": ["security_strength", "rounds"],
        "code_template": "print(round({security_strength} / {rounds}, 2))",
        "question_hint": "What is the security strength per round (strength/rounds) of this algorithm?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "block_throughput_ratio": {
        "level": 2, "type": "comparative", "entity_type": "algorithm",
        "label": "Block Throughput Ratio",
        "required_data": ["block_size", "rounds"],
        "code_template": "print(round(({block_size_a} / {rounds_a}) / ({block_size_b} / {rounds_b}), 2))",
        "question_hint": "What is the ratio of block throughput (block_size/rounds) between these two ciphers?",
        "answer_unit": "", "reasoning_depth": 3,
    },
    "encryption_work_factor": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Encryption Work Factor",
        "required_data": ["rounds", "block_size"],
        "code_template": "print(round({rounds} * {block_size} / 8, 0))",
        "question_hint": "How many bytes are processed per block encryption (rounds x block_size / 8) for this cipher?",
        "answer_unit": "bytes", "reasoning_depth": 2,
    },
    "nist_security_category": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "NIST Security Category",
        "required_data": ["security_strength"],
        "code_template": (
            "s = {security_strength}\n"
            "cat = 5 if s >= 256 else (3 if s >= 192 else (1 if s >= 128 else 0))\n"
            "print(cat)"
        ),
        "question_hint": "What NIST security category does this algorithm fall into? (1 if >=128 bits, 3 if >=192, 5 if >=256, 0 otherwise)",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "quantum_security_gap": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "Quantum Security Gap",
        "required_data": ["key_size"],
        "code_template": "print({key_size} - {key_size} // 2)",
        "question_hint": "What is the gap between classical and post-quantum (Grover) security strength (in bits) for this algorithm?",
        "answer_unit": "bits", "reasoning_depth": 2,
    },
    "state_to_output_ratio": {
        "level": 2, "type": "single", "entity_type": "algorithm",
        "label": "State-to-Output Ratio",
        "required_data": ["state_size", "security_strength"],
        "code_template": "print(round({state_size} / {security_strength}, 2))",
        "question_hint": "What is the ratio of internal state size to security strength for this stream cipher?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "combined_key_block_bits": {
        "level": 1, "type": "single", "entity_type": "algorithm",
        "label": "Combined Key + Block Bits",
        "required_data": ["key_size", "block_size"],
        "code_template": "print({key_size} + {block_size})",
        "question_hint": "What is the combined key size plus block size (in bits) for this cipher?",
        "answer_unit": "bits", "reasoning_depth": 1,
    },
    # --- ATT&CK Single (8) ---
    "group_technique_count": {
        "level": 1, "type": "single", "entity_type": "attack_group",
        "label": "Threat Group Technique Count",
        "required_data": ["technique_count"],
        "code_template": "print({technique_count})",
        "question_hint": "How many ATT&CK techniques are associated with this threat group?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    "group_tactic_coverage": {
        "level": 2, "type": "single", "entity_type": "attack_group",
        "label": "Tactic Coverage Percentage",
        "required_data": ["tactic_count"],
        "code_template": "print(round({tactic_count} / 14 * 100, 2))",
        "question_hint": "What percentage of the 14 ATT&CK tactics does this threat group cover?",
        "answer_unit": "%", "reasoning_depth": 2,
    },
    "group_software_count": {
        "level": 1, "type": "single", "entity_type": "attack_group",
        "label": "Threat Group Software Arsenal",
        "required_data": ["software_count"],
        "code_template": "print({software_count})",
        "question_hint": "How many software tools are associated with this threat group in ATT&CK?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    "technique_sub_count": {
        "level": 1, "type": "single", "entity_type": "attack_technique",
        "label": "Sub-Technique Count",
        "required_data": ["sub_technique_count"],
        "code_template": "print({sub_technique_count})",
        "question_hint": "How many sub-techniques does this ATT&CK technique have?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    "technique_group_usage": {
        "level": 1, "type": "single", "entity_type": "attack_technique",
        "label": "Technique Group Usage Count",
        "required_data": ["group_count"],
        "code_template": "print({group_count})",
        "question_hint": "How many threat groups are documented as using this ATT&CK technique?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    "technique_mitigation_count": {
        "level": 1, "type": "single", "entity_type": "attack_technique",
        "label": "Technique Mitigation Count",
        "required_data": ["mitigation_count"],
        "code_template": "print({mitigation_count})",
        "question_hint": "How many mitigations are documented for this ATT&CK technique?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    "group_years_active": {
        "level": 2, "type": "single", "entity_type": "attack_group",
        "label": "Years Active",
        "required_data": ["first_seen"],
        "code_template": "print(2025 - {first_seen})",
        "question_hint": "As of 2025, how many years has this threat group been active since first observation?",
        "answer_unit": "years", "reasoning_depth": 2,
    },
    "technique_detection_sources": {
        "level": 1, "type": "single", "entity_type": "attack_technique",
        "label": "Detection Data Source Count",
        "required_data": ["data_source_count"],
        "code_template": "print({data_source_count})",
        "question_hint": "How many data sources can be used to detect this ATT&CK technique?",
        "answer_unit": "", "reasoning_depth": 1,
    },
    # --- ATT&CK Comparative (4) ---
    "technique_count_difference": {
        "level": 2, "type": "comparative", "entity_type": "attack_group",
        "label": "Technique Count Difference",
        "required_data": ["technique_count"],
        "code_template": "print(abs({technique_count_b} - {technique_count_a}))",
        "question_hint": "What is the difference in technique counts between these two threat groups?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "tactic_coverage_ratio": {
        "level": 2, "type": "comparative", "entity_type": "attack_group",
        "label": "Tactic Coverage Ratio",
        "required_data": ["tactic_count"],
        "code_template": "print(round({tactic_count_a} / {tactic_count_b}, 2))",
        "question_hint": "What is the ratio of tactic coverage (first / second) between these two threat groups?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "software_count_difference": {
        "level": 2, "type": "comparative", "entity_type": "attack_group",
        "label": "Software Arsenal Difference",
        "required_data": ["software_count"],
        "code_template": "print(abs({software_count_b} - {software_count_a}))",
        "question_hint": "What is the difference in software tool counts between these two threat groups?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "activity_span_difference": {
        "level": 2, "type": "comparative", "entity_type": "attack_group",
        "label": "Activity Span Difference",
        "required_data": ["first_seen"],
        "code_template": "print(abs((2025 - {first_seen_a}) - (2025 - {first_seen_b})))",
        "question_hint": "What is the difference in years active (as of 2025) between these two threat groups?",
        "answer_unit": "years", "reasoning_depth": 2,
    },
    # --- ATT&CK Multi-Step (5) ---
    "technique_density": {
        "level": 2, "type": "single", "entity_type": "attack_group",
        "label": "Technique Acquisition Density",
        "required_data": ["technique_count", "first_seen"],
        "code_template": "print(round({technique_count} / (2025 - {first_seen}), 2))",
        "question_hint": "What is the technique density (techniques per year active, as of 2025) for this threat group?",
        "answer_unit": "techniques/year", "reasoning_depth": 3,
    },
    "sub_technique_ratio": {
        "level": 2, "type": "single", "entity_type": "attack_group",
        "label": "Sub-Technique Ratio",
        "required_data": ["sub_technique_count", "technique_count"],
        "code_template": "print(round({sub_technique_count} / {technique_count} * 100, 2))",
        "question_hint": "What percentage of this group's techniques have associated sub-techniques?",
        "answer_unit": "%", "reasoning_depth": 2,
    },
    "group_arsenal_index": {
        "level": 2, "type": "single", "entity_type": "attack_group",
        "label": "Group Arsenal Index",
        "required_data": ["technique_count", "software_count"],
        "code_template": "print(round({technique_count} * {software_count} / 100, 2))",
        "question_hint": "What is the arsenal index (technique_count x software_count / 100) for this threat group?",
        "answer_unit": "", "reasoning_depth": 3,
    },
    "detection_per_sub": {
        "level": 2, "type": "single", "entity_type": "attack_technique",
        "label": "Detection Sources per Sub-Technique",
        "required_data": ["data_source_count", "sub_technique_count"],
        "code_template": "print(round({data_source_count} / max({sub_technique_count}, 1), 2))",
        "question_hint": "What is the ratio of detection data sources to sub-techniques for this ATT&CK technique?",
        "answer_unit": "", "reasoning_depth": 2,
    },
    "software_technique_count": {
        "level": 1, "type": "single", "entity_type": "attack_software",
        "label": "Software Technique Count",
        "required_data": ["technique_count"],
        "code_template": "print({technique_count})",
        "question_hint": "How many ATT&CK techniques are implemented by this malware/tool?",
        "answer_unit": "", "reasoning_depth": 1,
    },
}


# ===========================================================================
# Prompts
# ===========================================================================

SECURITY_QA_DEVELOPER = (
    "You are a cybersecurity expert creating research questions that test "
    "the ability to identify cryptographic algorithms, threat groups, or attack techniques "
    "from descriptions and perform quantitative computations with their parameters."
)

SECURITY_QA_PROMPT = """Compose a cybersecurity question that:
1. Uses clue facts to describe an unnamed cryptographic algorithm, threat group, or attack technique — readers must figure out which one
2. Then asks for a specific quantitative computation requiring the solver to look up the entity's cryptographic parameters or ATT&CK metrics

CLUE FACTS (about the unnamed entity — do NOT name it directly):
{facts_text}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- STYLE GUIDANCE ---
- Write 2-4 sentences total.
- First 1-2 sentences: describe the entity using 3+ clue facts from different topics, without naming it directly. Do NOT use algorithm names, ATT&CK IDs, or entity names.
- Last 1-2 sentences: pose the quantitative question.
- CRITICAL: Do NOT reveal ANY quantitative values from the entity's data (key sizes, round counts, technique counts, etc.) in the question — the solver must look up ALL data themselves.
- CRITICAL: Do NOT name the entity directly anywhere in the question. Do NOT use algorithm names or ATT&CK IDs. Use descriptive clues only.
- CRITICAL: Use ONLY facts from the CLUE FACTS above. Do NOT add any claims, descriptions, or context not directly supported by the provided clue facts.
- CRITICAL: Do NOT use vague temporal language like "latest", "current", "recent", "most recent", "up-to-date", or "present-day". Instead, use "publicly available" or omit time references entirely.
- Sound natural and conversational, like a real cybersecurity risk assessment question.
- Include all necessary parameters for the computation (e.g., SLE, multiplier, days) — these are NOT entity-specific data.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your cybersecurity question>
Used_Facts: <comma-separated fact_ids used, e.g. W1, W2, W3>
Reasoning: <brief chain: clues identify entity -> look up data -> computation gives answer>
"""

SECURITY_COMPARATIVE_QA_PROMPT = """Compose a comparative cybersecurity question that:
1. Describes TWO unnamed entities (cryptographic algorithms, threat groups, or attack techniques) using clue facts about each
2. Asks for a specific quantitative comparison requiring the solver to look up parameters or metrics for both

CLUE FACTS FOR ENTITY A (do NOT name this entity, its algorithm name, or ATT&CK ID):
{facts_text_a}

CLUE FACTS FOR ENTITY B (do NOT name this entity, its algorithm name, or ATT&CK ID):
{facts_text_b}

REASONING TASK:
Template: {template_label}
Hint: {question_hint}
The answer involves: {answer_unit}

Gold answer (for your reference, do NOT reveal in the question): {gold_answer}

--- STYLE GUIDANCE ---
- Write 4-6 sentences total.
- First 1-2 sentences: describe the first entity using 3+ clue facts from different topics, without naming it directly. Do NOT use algorithm names, ATT&CK IDs, or entity names.
- Next 1-2 sentences: describe the second entity using 3+ clue facts from different topics, without naming it directly. Do NOT use algorithm names, ATT&CK IDs, or entity names.
- Last 1-2 sentences: pose the comparative quantitative question.
- Refer to them as "the first algorithm" and "the second algorithm" (or "the first group" / "the second group", etc.).
- CRITICAL: Do NOT reveal ANY quantitative values from either entity's data (key sizes, round counts, technique counts, etc.) in the question — the solver must look up ALL data themselves.
- CRITICAL: Do NOT name either entity directly anywhere in the question. Do NOT use algorithm names or ATT&CK IDs. Use descriptive clues only.
- CRITICAL: Use ONLY facts from the CLUE FACTS above. Do NOT add any claims, descriptions, or context not directly supported by the provided clue facts.
- CRITICAL: Do NOT use vague temporal language like "latest", "current", "recent", "most recent", "up-to-date", or "present-day". Instead, use "publicly available" or omit time references entirely.
- Sound natural and conversational, like a real cybersecurity risk assessment question.
- Include all necessary parameters for the computation (e.g., SLE, multiplier, days) — these are NOT entity-specific data.

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {gold_answer}
Question: <your comparative cybersecurity question>
Used_Facts_A: <comma-separated fact_ids for entity A, e.g. A_C1_F1, A_C2_F1>
Used_Facts_B: <comma-separated fact_ids for entity B, e.g. B_C1_F1, B_C2_F1>
Reasoning: <brief chain: clues identify both entities -> look up data -> comparison gives answer>

REQUIREMENTS:
- Use 3+ clue facts per entity from different chains
- Do NOT name either entity in the question
- Do NOT include security metric values in the question
- Every claim must come VERBATIM from the listed facts
- The question MUST have exactly ONE unambiguous answer"""

SECURITY_V1_PROMPT = """Solve this cybersecurity question. It requires:
1. Identifying a cryptographic algorithm, threat group, or attack technique from its description
2. Looking up its quantitative parameters (key sizes, round counts, ATT&CK metrics, etc.)
3. Performing the requested computation

Provide ONLY the final numerical answer (with units if applicable).
Do not show your work.

Question: {question}

Answer:"""

SECURITY_V2_DEVELOPER = """You are an expert cybersecurity research assistant.
You have THREE tools available:
- **Security tool**: Query cryptographic algorithm parameters and MITRE ATT&CK metrics
  - security.get_algorithm(name) — get algorithm parameters (key_size, block_size, rounds, etc.)
  - security.search_algorithms(query, algo_type) — search crypto algorithms
  - security.get_attack_entity(name) — get ATT&CK group/technique/software metrics
  - security.search_attack(query, entity_type) — search ATT&CK entities
  - security.compare_algorithms(names) — compare crypto algorithm parameters
- **Browser tool**: Search Wikipedia for entity identification
- **Python tool**: Execute calculations (math, numpy available)

Recommended approach:
1. Use the browser to search for and identify the algorithm/group/technique from the description clues
2. Use the security tool to look up quantitative parameters
3. Use Python for computation
Give your final answer as a single number (with units if applicable) on the last line."""


# ===========================================================================
# Answer Checker
# ===========================================================================

def is_security_answer_correct(predicted, gold, tolerance=0.05):
    """Security answer comparison with 5% tolerance.

    Handles: CVSS scores, percentages, ratios, dollar amounts, days.

    Args:
        predicted: Predicted answer string.
        gold: Gold answer string.
        tolerance: Relative tolerance (default 5%).

    Returns:
        True if answer is correct within tolerance.
    """
    if not predicted or not predicted.strip():
        return False

    pred_clean = _normalize_security_text(predicted)
    gold_clean = _normalize_security_text(gold)

    # 1. Exact match
    if pred_clean == gold_clean:
        return True

    # 2. Numeric comparison
    pred_num = _try_parse_security_number(pred_clean)
    gold_num = _try_parse_security_number(gold_clean)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0 and pred_num == 0:
            return True
        if gold_num == 0:
            return abs(pred_num) < 0.01
        rel_error = abs(pred_num - gold_num) / max(abs(gold_num), abs(pred_num))
        if rel_error <= tolerance:
            return True

        # 3. Order of magnitude for large numbers
        if abs(gold_num) > 1e3 and abs(pred_num) > 1e3:
            if gold_num > 0 and pred_num > 0:
                log_ratio = abs(math.log10(pred_num) - math.log10(gold_num))
                if log_ratio <= 0.05:
                    return True

    return False


def _normalize_security_text(text):
    """Normalize text for security comparison."""
    text = text.strip()
    # Normalize Unicode whitespace (narrow no-break space, non-breaking space, etc.)
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    for suffix in ["%", "bits", "bytes", "rounds", "techniques", "tactics",
                   "years", "techniques/year"]:
        text = text.replace(suffix, "").strip()
    text = text.replace(",", "")
    text = text.strip().strip("'\"")
    return text


def _try_parse_security_number(text):
    """Parse a number from security text. Returns float or None."""
    text = text.strip()
    # Normalize Unicode whitespace before parsing
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Handle scientific notation variants: "2.4 x 10^-4", "2.4×10^-4", etc.
    text = re.sub(r'\s*[×x]\s*10\^', 'e', text)

    multipliers = {
        "million": 1e6, "billion": 1e9, "thousand": 1e3,
        "M": 1e6, "B": 1e9, "K": 1e3,
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


def _llm_judge_security_answer(predicted, gold, question):
    """Use LLM to judge if predicted answer matches gold for security questions."""
    prompt = f"""Compare these two answers to the same cybersecurity question.
The gold answer is computed from verified reference data (NIST/ATT&CK).
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

def _execute_computation_code(code: str) -> Optional[str]:
    """Execute Python computation code and return printed output."""
    import io
    import contextlib
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, {"__builtins__": __builtins__, "math": math, "round": round,
                         "abs": abs, "min": min, "max": max})
        return f.getvalue().strip()
    except Exception:
        return None


def _generate_template_params(template_id: str, tmpl: dict,
                              entity_data: dict) -> Optional[Dict[str, Any]]:
    """Generate random parameters for templates that need extra params.

    Crypto + ATT&CK templates get all parameters directly from reference data.
    No random parameters needed for any current template.
    """
    return {}


def select_security_template(entity_data: dict, entity_type: str,
                              used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a template that the entity's data can support.

    Args:
        entity_data: Entity data dict (from CRYPTO_REFERENCE_DATA or ATTACK_REFERENCE_DATA).
        entity_type: "algorithm", "attack_group", "attack_technique", or "attack_software".
        used_templates: Set of template IDs already used.

    Returns:
        Dict with template info and computed gold answer, or None.
    """
    template_ids = list(SECURITY_TEMPLATES.keys())
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = SECURITY_TEMPLATES[tmpl_id]

        # Skip comparatives (handled separately)
        if tmpl["type"] == "comparative":
            continue

        # Check entity type compatibility — attack_software can also use attack_technique templates
        # because they share some metrics (technique_count, group_count)
        if tmpl["entity_type"] != entity_type:
            if not (entity_type == "attack_software" and tmpl["entity_type"] == "attack_technique"):
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
            if isinstance(val, (int, float)) and val == 0 and key not in ("security_strength", "nonce_size"):
                all_available = False
                break
            if isinstance(val, dict):
                continue
            base_params[key] = val

        if not all_available:
            continue

        # Generate extra parameters
        extra_params = _generate_template_params(tmpl_id, tmpl, entity_data)
        if extra_params is None:
            continue

        params = {**base_params, **extra_params}

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
        except ValueError:
            continue

        # Reject trivially saturated percentage answers (e.g. 100.0% or 0.0%)
        if tmpl.get("answer_unit") == "%" and gold_float in (0.0, 100.0):
            continue

        # Fill question hint
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


def select_comparative_template(data_a: dict, data_b: dict, entity_type: str,
                                 used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a comparative template for two entities."""
    template_ids = [tid for tid, t in SECURITY_TEMPLATES.items()
                    if t["type"] == "comparative" and t["entity_type"] == entity_type]
    random.shuffle(template_ids)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = SECURITY_TEMPLATES[tmpl_id]
        required = tmpl["required_data"]

        params = {}
        all_available = True
        for key in required:
            val_a = data_a.get(key)
            val_b = data_b.get(key)
            if val_a is None or val_b is None:
                all_available = False
                break
            if isinstance(val_a, (int, float)) and val_a == 0:
                all_available = False
                break
            if isinstance(val_b, (int, float)) and val_b == 0:
                all_available = False
                break
            params[f"{key}_a"] = val_a
            params[f"{key}_b"] = val_b

        if not all_available:
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


def compose_security_question(clues, reasoning_ctx, entity_name,
                               prev_questions):
    """Compose a security QA question using LLM."""
    facts_text = "\n".join(
        f"[{c['fact_id']}] ({c.get('topic', c.get('property', 'description'))}) {c['fact']}"
        for c in clues
    )

    prev_text = "\n".join(prev_questions[-10:]) if prev_questions else "(none)"

    prompt = SECURITY_QA_PROMPT.format(
        facts_text=facts_text,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=SECURITY_QA_DEVELOPER,
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
    # e.g. "(C8_F1)", "[C3_F2]", "(A_C1_F1)", "(B_C2_F1)"
    question = re.sub(r'\s*[\(\[]\s*[AB]?_?C\d+_F\d+\s*[\)\]]', '', question)

    used_facts = []
    if facts_match:
        used_facts = [f.strip() for f in facts_match.group(1).split(",") if f.strip()]

    return {"question": question, "used_facts": used_facts}


def compose_security_comparative_question(clues_a, clues_b, reasoning_ctx,
                                           entity_name_a, entity_name_b,
                                           prev_questions):
    """Compose a comparative security QA question using LLM (separate A/B clue blocks)."""
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

    prompt = SECURITY_COMPARATIVE_QA_PROMPT.format(
        facts_text_a=facts_text_a,
        facts_text_b=facts_text_b,
        template_label=reasoning_ctx["template"]["label"],
        question_hint=reasoning_ctx["question_hint"],
        answer_unit=reasoning_ctx["template"]["answer_unit"],
        gold_answer=reasoning_ctx["gold_answer"],
        previous_questions=prev_text,
    )

    try:
        response = gen_from_prompt_harmony(
            prompt, temperature=1.0, max_tokens=8196,
            developer_content=SECURITY_QA_DEVELOPER,
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
    # Strip fact ID annotations that the LLM sometimes copies from the prompt
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


def gen_security_qa_from_entity(entity_data, agent_info, num_questions=5,
                                 prev_questions=None, used_templates=None,
                                 articles=None, category=None):
    """Generate security QA pairs from a single entity using chain-based clues.

    Parallels gen_financial_qa_from_entity with Phases 0/1/1.5/2/3.

    Args:
        entity_data: Dict with 'name', 'entity_id', 'entity_type', 'data',
                     'chains', 'articles', optionally 'wikidata_id'.
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
    entity_type_val = entity_data.get("entity_type", "algorithm")
    data = entity_data.get("data", {})
    chains = entity_data.get("chains", [])
    wikidata_id = entity_data.get("wikidata_id", "")
    entity_id = entity_data.get("entity_id", "")
    if articles is None:
        articles = entity_data.get("articles", [])

    if not chains or len(chains) < 2:
        print(f"  No KG chains for {entity_name}", flush=True)
        return []

    qa_pairs = []
    max_attempts = num_questions * 3

    # Cache Phase 1+1.5 results — chains are static per entity, so grounded
    # facts only need to be extracted once.
    cached_grounded_facts = None
    p1_fail_count = 0
    P1_MAX_RETRIES = 3  # allow a few retries for stochastic LLM extraction
    poisoned_props = set()

    for attempt in range(max_attempts):
        if len(qa_pairs) >= num_questions:
            break

        print(f"\n  [{entity_name}] Attempt {attempt+1}/{max_attempts} "
              f"(generated {len(qa_pairs)}/{num_questions})", flush=True)

        # Phase 0: Select template and compute gold answer
        reasoning_ctx = select_security_template(data, entity_type_val, used_templates)
        if reasoning_ctx is None:
            print(f"  No valid template remaining for {entity_name}", flush=True)
            # Include both own-type and attack_technique templates for attack_software
            eligible_types = {entity_type_val}
            if entity_type_val == "attack_software":
                eligible_types.add("attack_technique")
            single_templates = [t for t, d in SECURITY_TEMPLATES.items()
                                if d["type"] == "single" and d["entity_type"] in eligible_types]
            if len(used_templates & set(single_templates)) >= len(single_templates):
                used_templates -= set(single_templates)
                continue
            break

        print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

        # Phase 1+1.5: Extract and ground clue facts (cached per entity)
        if cached_grounded_facts is not None:
            extracted_facts = cached_grounded_facts
        else:
            if p1_fail_count >= P1_MAX_RETRIES:
                # Chains can't produce usable facts — give up on entity
                print(f"    Phase 1: giving up (failed {p1_fail_count} times)", flush=True)
                break

            _min_props = 2 if entity_type_val == "algorithm" else 3
            extracted_facts, _ = extract_chain_clue_facts(
                chains, entity_name, agent_info,
                min_distinct_properties=_min_props)
            if extracted_facts is None:
                p1_fail_count += 1
                continue

            # Phase 1.5: Verify facts against Wikipedia grounding documents
            grounded_facts = verify_facts_grounding(extracted_facts, chains, articles, agent_info)
            if grounded_facts is None:
                p1_fail_count += 1
                continue
            extracted_facts = grounded_facts
            cached_grounded_facts = extracted_facts

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
        composition = compose_security_question(
            clues_for_composition, reasoning_ctx, entity_name, prev_questions,
        )
        if composition is None:
            print(f"  Phase 2: Composition failed", flush=True)
            continue

        question = composition["question"]
        used_facts = composition["used_facts"]
        print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

        # Phase 3: Validation
        sec_values = [
            data[k] for k in reasoning_ctx["template"]["required_data"]
            if k in data and isinstance(data[k], (int, float))
        ]
        # Build acceptable alternative identifiers for uniqueness check
        acceptable_ids = [entity_id]
        if entity_data.get("chain_wiki"):
            acceptable_ids.append(entity_data["chain_wiki"])
        if entity_data.get("wiki"):
            acceptable_ids.append(entity_data["wiki"])
        passed, reason = run_phase3_validation(
            question=question,
            gold_answer=reasoning_ctx["gold_answer"],
            computation_code=reasoning_ctx["code"],
            entity_label=entity_name,
            entity_names=[entity_name, entity_id],
            entity_values=sec_values,
            used_facts=used_facts,
            extracted_facts=extracted_facts,
            gen_fn=gen_from_prompt_harmony,
            entity_type=entity_type_val,
            check_facts=True,
            min_chains=2,
            acceptable_identifiers=acceptable_ids,
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

        grounding_articles = build_multiskill_grounding_articles(
            entity_name, reasoning_ctx["gold_answer"],
            reasoning_ctx["template"]["answer_unit"],
            extracted_facts, chains, articles or [],
        )

        _grounding_keys = {"type", "family", "key_size", "block_size", "rounds",
                           "security_strength", "output_size", "nonce_size",
                           "tag_size", "state_size", "year",
                           "attack_id", "technique_count", "sub_technique_count",
                           "tactic_count", "software_count", "first_seen",
                           "group_count", "mitigation_count", "data_source_count",
                           "tactic", "country", "notable_campaign",
                           "target_sectors", "platform", "permission_required",
                           "software_type"}
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
            "wikidata_entity": wikidata_id,
            "data_source": "crypto_attack",
            "grounding_clues": clues_for_composition,
            "grounding_entity_data": grounding_entity_data,
        }
        qa_pair.update(compute_cci_fields(reasoning_ctx["template"], extracted_facts, is_comparative=False))

        if entity_id:
            qa_pair["entity_id"] = entity_id

        qa_pairs.append(qa_pair)
        prev_questions.append(question)
        used_templates.add(reasoning_ctx["template_id"])

    return qa_pairs


def _build_attack_attribute_clues(entity_name, entity_data, entity_type):
    """Build supplementary clue facts from ATT&CK distinctive attributes.

    For ATT&CK entities (groups, techniques, software), creates additional
    clue facts from the enriched reference data fields (country,
    notable_campaign, target_sectors, platform, permission_required,
    software_type) to improve entity uniqueness identification.

    Returns:
        List of clue fact dicts compatible with the composition pipeline.
    """
    clues = []
    clue_idx = 0

    if entity_type == "attack_group":
        if "country" in entity_data:
            clue_idx += 1
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "country of origin",
                "fact": f"This threat group is attributed to {entity_data['country']}.",
            })
        if "notable_campaign" in entity_data:
            clue_idx += 1
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "notable campaign",
                "fact": f"This group is known for: {entity_data['notable_campaign']}.",
            })
        if "target_sectors" in entity_data and entity_data["target_sectors"]:
            clue_idx += 1
            sectors = ", ".join(entity_data["target_sectors"])
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "target sectors",
                "fact": f"This group primarily targets the {sectors} sectors.",
            })

    elif entity_type == "attack_technique":
        if "platform" in entity_data and entity_data["platform"]:
            clue_idx += 1
            platforms = ", ".join(entity_data["platform"][:3])
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "platform",
                "fact": f"This technique applies to: {platforms}.",
            })
        if "permission_required" in entity_data:
            clue_idx += 1
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "permission required",
                "fact": f"This technique requires {entity_data['permission_required']}-level permissions.",
            })

    elif entity_type == "attack_software":
        if "software_type" in entity_data:
            clue_idx += 1
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "software type",
                "fact": f"This software is classified as a {entity_data['software_type']}.",
            })
        if "platform" in entity_data and entity_data["platform"]:
            clue_idx += 1
            platforms = ", ".join(entity_data["platform"][:3])
            clues.append({
                "fact_id": f"ATTR_{clue_idx}",
                "topic": "platform",
                "fact": f"This software targets: {platforms}.",
            })

    return clues


def gen_security_qa_single(entity_name, entity_data, clues, entity_type,
                             used_templates, prev_questions, category,
                             acceptable_identifiers=None):
    """Generate a single-entity security QA pair (legacy clue-based fallback)."""
    # Phase 0: Select template
    reasoning_ctx = select_security_template(entity_data, entity_type, used_templates)
    if reasoning_ctx is None:
        print(f"  {entity_name}: no valid template (type={entity_type}, "
              f"used={len(used_templates)}, data_keys={list(entity_data.keys())[:5]})", flush=True)
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    # Phase 1: Clue facts ready — supplement with ATT&CK attribute clues
    if entity_type in ("attack_group", "attack_technique", "attack_software"):
        attr_clues = _build_attack_attribute_clues(entity_name, entity_data, entity_type)
        if attr_clues:
            # Avoid duplicates: only add attribute clues with new topics
            existing_topics = {c.get("topic", "") for c in clues}
            for ac in attr_clues:
                if ac["topic"] not in existing_topics:
                    clues.append(ac)
    if len(clues) < 3:
        print(f"  Phase 1: Insufficient clues ({len(clues)})", flush=True)
        return None

    # Phase 2: Compose question
    composition = compose_security_question(
        clues, reasoning_ctx, entity_name, prev_questions,
    )
    if composition is None:
        print(f"  Phase 2: Composition failed", flush=True)
        return None

    question = composition["question"]
    used_facts = composition["used_facts"]
    print(f"  Phase 2: \"{question[:80]}...\"", flush=True)

    # Phase 3: Validation (shared utility)
    entity_id = entity_data.get("entity_id", entity_data.get("attack_id", ""))
    sec_values = [
        entity_data[k] for k in reasoning_ctx["template"]["required_data"]
        if k in entity_data and isinstance(entity_data[k], (int, float))
    ]
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=entity_name,
        entity_names=[entity_name, entity_id],
        entity_values=sec_values,
        used_facts=used_facts,
        extracted_facts=[],
        gen_fn=gen_from_prompt_harmony,
        entity_type=entity_type,
        acceptable_identifiers=acceptable_identifiers,
    )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Store grounding data for verification
    _grounding_keys = {"type", "family", "key_size", "block_size", "rounds",
                       "security_strength", "output_size", "nonce_size",
                       "tag_size", "state_size", "year",
                       "attack_id", "technique_count", "sub_technique_count",
                       "tactic_count", "software_count", "first_seen",
                       "group_count", "mitigation_count", "data_source_count",
                       "tactic", "country", "notable_campaign",
                       "target_sectors", "platform", "permission_required",
                       "software_type"}
    grounding_entity_data = {k: entity_data[k] for k in _grounding_keys if k in entity_data}

    # Synthesize chain-style fields from legacy clues for display consistency
    extracted_facts = [
        {"fact_id": c.get("fact_id", f"W{i}"), "entity": entity_name,
         "property": c.get("topic", ""), "value": "", "fact": c.get("fact", "")}
        for i, c in enumerate(clues)
    ]
    source_triples = [c.get("fact", "") for c in clues[:5]]
    # Group clues by source article
    _article_map = {}
    for c in clues:
        src = c.get("source", "unknown")
        _article_map.setdefault(src, []).append(c.get("fact", ""))
    grounding_articles = [
        {"title": title, "paragraph": " ".join(facts)}
        for title, facts in _article_map.items()
    ]

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
        "extracted_facts": extracted_facts,
        "source_triples": source_triples,
        "grounding_articles": grounding_articles,
        "data_source": "crypto_attack",
        "grounding_clues": clues,
        "grounding_entity_data": grounding_entity_data,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=False))

    # Include entity ID
    if entity_id:
        result["entity_id"] = entity_id

    return result


def gen_security_qa_comparative(name_a, data_a, clues_a,
                                  name_b, data_b, clues_b,
                                  entity_type, used_templates,
                                  prev_questions, category):
    """Generate a comparative security QA pair (two entities)."""
    reasoning_ctx = select_comparative_template(data_a, data_b, entity_type, used_templates)
    if reasoning_ctx is None:
        return None

    print(f"  Phase 0: {reasoning_ctx['template_id']} -> {reasoning_ctx['gold_answer']}", flush=True)

    combined_clues = []
    for c in clues_a[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"A_{c['fact_id']}"
        c_copy["fact"] = f"[Entity A] {c['fact']}"
        combined_clues.append(c_copy)
    for c in clues_b[:5]:
        c_copy = dict(c)
        c_copy["fact_id"] = f"B_{c['fact_id']}"
        c_copy["fact"] = f"[Entity B] {c['fact']}"
        combined_clues.append(c_copy)

    if len(combined_clues) < 4:
        return None

    composition = compose_security_question(
        combined_clues, reasoning_ctx, f"{name_a}/{name_b}", prev_questions,
    )
    if composition is None:
        return None

    question = composition["question"]

    # NOTE: KG-grounded uniqueness (check 3h) is intentionally NOT wired here.
    # Security comparative clues come from fetch_entity_clues() (Wikipedia-text
    # facts: {fact_id, fact, topic, source}) which carry no chain_num / hop QIDs,
    # so check_kg_uniqueness_comparative has no KG structure to query. V2 covers
    # uniqueness for this path. (The security single path DOES use chain-derived
    # facts and is wired.)

    # Phase 3: Validation (shared utility)
    id_a = data_a.get("attack_id", data_a.get("entity_id", ""))
    id_b = data_b.get("attack_id", data_b.get("entity_id", ""))
    all_values = []
    for data_src in [data_a, data_b]:
        for k in reasoning_ctx["template"]["required_data"]:
            val = data_src.get(k)
            if val is not None and isinstance(val, (int, float)):
                all_values.append(val)
    passed, reason = run_phase3_validation(
        question=question,
        gold_answer=reasoning_ctx["gold_answer"],
        computation_code=reasoning_ctx["code"],
        entity_label=f"{name_a} / {name_b}",
        entity_names=[name_a, name_b, id_a, id_b],
        entity_values=all_values,
        entity_type=entity_type,
    )
    if not passed:
        print(f"  Phase 3: {reason}", flush=True)
        return None

    used_templates.add(reasoning_ctx["template_id"])

    # Store grounding data for verification
    _grounding_keys = {"type", "family", "key_size", "block_size", "rounds",
                       "security_strength", "output_size", "nonce_size",
                       "tag_size", "state_size", "year",
                       "attack_id", "technique_count", "sub_technique_count",
                       "tactic_count", "software_count", "first_seen",
                       "group_count", "mitigation_count", "data_source_count",
                       "tactic", "country", "notable_campaign",
                       "target_sectors", "platform", "permission_required",
                       "software_type"}
    grounding_entity_data_a = {k: data_a[k] for k in _grounding_keys if k in data_a}
    grounding_entity_data_b = {k: data_b[k] for k in _grounding_keys if k in data_b}

    # Synthesize chain-style fields from legacy clues for display consistency
    facts_a = [
        {"fact_id": f"A_{c.get('fact_id', f'W{i}')}", "entity": name_a,
         "property": c.get("topic", ""), "value": "", "fact": c.get("fact", "")}
        for i, c in enumerate(clues_a)
    ]
    facts_b = [
        {"fact_id": f"B_{c.get('fact_id', f'W{i}')}", "entity": name_b,
         "property": c.get("topic", ""), "value": "", "fact": c.get("fact", "")}
        for i, c in enumerate(clues_b)
    ]
    triples_a = [c.get("fact", "") for c in clues_a[:5]]
    triples_b = [c.get("fact", "") for c in clues_b[:5]]
    _article_map_a = {}
    for c in clues_a:
        src = c.get("source", "unknown")
        _article_map_a.setdefault(src, []).append(c.get("fact", ""))
    _article_map_b = {}
    for c in clues_b:
        src = c.get("source", "unknown")
        _article_map_b.setdefault(src, []).append(c.get("fact", ""))
    articles_a = [{"title": t, "paragraph": " ".join(fs)} for t, fs in _article_map_a.items()]
    articles_b = [{"title": t, "paragraph": " ".join(fs)} for t, fs in _article_map_b.items()]

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
        "extracted_facts": facts_a + facts_b,
        "source_triples": triples_a + triples_b,
        "grounding_articles": articles_a + articles_b,
        "data_source": "crypto_attack",
        "grounding_clues_a": clues_a,
        "grounding_clues_b": clues_b,
        "grounding_entity_data_a": grounding_entity_data_a,
        "grounding_entity_data_b": grounding_entity_data_b,
    }
    result.update(compute_cci_fields(reasoning_ctx["template"], is_comparative=True))

    # Include entity IDs
    if id_a:
        result["entity_id_a"] = id_a
    if id_b:
        result["entity_id_b"] = id_b

    return result


# ===========================================================================
# V2 Verification: Security + Browser + Python Tools
# ===========================================================================

def verify_security_v2(qa_pairs, num_samples=10, temperature=1.0,
                        max_iterations=200, threshold=0.5,
                        outfile_prefix=None, subarea="",
                        bench_label="security_bench"):
    """V2 verification using Security + Browser + Python tools."""
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
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available for security V2")
    if not _HAS_PYTHON_TOOL:
        raise RuntimeError("HybridPythonTool not available for security V2")
    if not _HAS_SECURITY_TOOL:
        raise RuntimeError("SecurityTool not available for security V2")

    all_pairs = []
    filtered_pairs = []

    print(f"\n=== SECURITY V2 VERIFICATION ({subarea}) ===", flush=True)
    print(f"Tools: Security + Browser + Python", flush=True)
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
            security_tool = None
            qid_str = f"{bench_label}_v2_{subarea}_{idx}_{i}"
            try:
                backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
                browser_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
                python_tool = HybridPythonTool(timeout=60)
                python_tool.set_qid(qid_str)
                security_tool = SecurityTool()

                system_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.HIGH)
                    .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                    .with_tools(browser_tool.tool_config)
                    .with_tools(python_tool.tool_config)
                    .with_tools(security_tool.tool_config)
                )

                messages = [
                    Message.from_role_and_content(Role.SYSTEM, system_content),
                    Message.from_role_and_content(Role.DEVELOPER, SECURITY_V2_DEVELOPER),
                    Message.from_role_and_content(Role.USER, f"Question: {qa['question']}"),
                ]

                tool_call_counter = [0]

                async def _tool_handler(msg, _browser=browser_tool, _python=python_tool,
                                        _security=security_tool, _counter=tool_call_counter):
                    _counter[0] += 1
                    recipient = str(getattr(msg, 'recipient', ''))
                    results = []
                    if recipient.startswith("security"):
                        async for m in _security.process(msg):
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
                    tool_prefix=("security", "browser.", "python"),
                    tool_configs=[browser_tool.tool_config, python_tool.tool_config, security_tool.tool_config],
                    max_iterations=max_iterations,
                    temperature=temperature,
                )

                iteration_count = len(result_messages) - len(messages)

                answer = _extract_security_answer(result_messages)
                answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
                answer = ' '.join(answer.split())
                correct = bool(is_security_answer_correct(answer, qa['gold_answer']) or
                               _llm_judge_security_answer(answer, qa['gold_answer'], qa['question']))
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


def _extract_security_answer(messages):
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

    Uses the unified chains pipeline with fallback:
    1. Fetch entities → quantitative data + resolve Wikidata QID + KG chains + Wikipedia articles
    2. Cache as {prefix}__{category}.security_chains.json
    3. Generate QA pairs per entity via chain-based Phase 1/1.5/2/3
    4. Fallback to legacy clues if QID resolution or chains fail
    """
    # Map category to entity_type using SECURITY_THEMES
    theme_info = SECURITY_THEMES.get(category, {})
    entity_type = theme_info.get("entity_type", "algorithm")

    # Check chains cache (new pattern, replaces security_entity_data.json)
    chains_cache = f"{prefix}__{category}.security_chains.json"
    old_data_cache = f"{prefix}__{category}.security_entity_data.json"

    if os.path.exists(chains_cache):
        print(f"Loading cached chains for {category}...", flush=True)
        with open(chains_cache) as f:
            entity_data_list = json.load(f)

        # Backfill Wikipedia articles for cached entities that lack them
        needs_resave = False
        for ed in entity_data_list:
            if ed.get("chains") and ("articles" not in ed or not ed["articles"]):
                print(f"  Fetching Wikipedia articles for cached entity {ed.get('name', '')}...",
                      flush=True)
                ed["articles"] = fetch_wikipedia_for_entities(ed.get("chains", []))
                print(f"  {ed.get('name', '')}: fetched {len(ed['articles'])} Wikipedia articles",
                      flush=True)
                needs_resave = True

        # Backfill new ATT&CK reference data fields into cached entities
        _ATTACK_NEW_FIELDS = {"country", "notable_campaign", "target_sectors",
                              "platform", "permission_required", "software_type"}
        for ed in entity_data_list:
            data = ed.get("data", {})
            etype = ed.get("entity_type", "")
            if etype in ("attack_group", "attack_technique", "attack_software"):
                ref = ATTACK_REFERENCE_DATA.get(ed.get("name", ""), {})
                for field in _ATTACK_NEW_FIELDS:
                    if field in ref and field not in data:
                        data[field] = ref[field]
                        needs_resave = True

        # Append new entities from ENTITY_UNIVERSE that aren't in the cache yet
        cached_names = {ed.get("name", "") for ed in entity_data_list}
        cached_ids = {ed.get("entity_id", "") for ed in entity_data_list}
        universe_entities = get_category_entities(category)
        new_entities = [
            e for e in universe_entities
            if e.get("name", "") not in cached_names and e.get("id", "") not in cached_ids
        ]
        if new_entities:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Found {len(new_entities)} new entities in ENTITY_UNIVERSE "
                  f"not yet cached for {category}, fetching...", flush=True)
            for entity in tqdm.tqdm(new_entities, desc=f"Appending new {category} entities"):
                time.sleep(2)
                name = entity.get("name", "")
                entity_id = entity.get("id", "")
                wiki = entity.get("wiki", name)

                # Look up reference data based on entity type
                if entity_type == "algorithm":
                    data = get_crypto_algorithm_data(name)
                elif entity_type in ("attack_group", "attack_technique", "attack_software"):
                    data = get_attack_entity_data(name)
                else:
                    data = None

                if data is None:
                    print(f"  {name}: no reference data found, skipping", flush=True)
                    continue

                wikidata_id = resolve_security_wikidata_id(name, wiki)
                chains = []
                articles = []
                if wikidata_id:
                    chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains,
                                                    min_sitelinks=2)
                    if chains and len(chains) >= 2:
                        articles = fetch_wikipedia_for_entities(chains)
                    else:
                        chains = chains or []

                if (not chains or len(chains) < 2) and entity.get("chain_wiki"):
                    alt_qid = resolve_security_wikidata_id(entity["chain_wiki"], entity["chain_wiki"])
                    if alt_qid and alt_qid != wikidata_id:
                        alt_chains = fetch_multihop_triples(
                            alt_qid, num_hops=2, limit=num_chains,
                            min_sitelinks=2)
                        if alt_chains and len(alt_chains) >= 2:
                            chains = alt_chains
                            wikidata_id = alt_qid
                            articles = fetch_wikipedia_for_entities(chains)

                clues = fetch_entity_clues(wiki, entity_type)
                entity_data_list.append({
                    "name": name,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "wikidata_id": wikidata_id or "",
                    "data": data,
                    "clues": clues,
                    "chains": chains,
                    "articles": articles,
                    "chain_wiki": entity.get("chain_wiki", ""),
                    "wiki": entity.get("wiki", ""),
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

    elif os.path.exists(old_data_cache):
        # Backfill: upgrade old entity_data cache to chains cache
        print(f"Upgrading old entity data cache for {category} to chains format...", flush=True)
        with open(old_data_cache) as f:
            old_list = json.load(f)

        entity_data_list = []
        for ed in tqdm.tqdm(old_list, desc=f"Upgrading {category} cache"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            name = ed.get("name", "")
            wiki = ed.get("wiki", name)

            # Try to resolve Wikidata QID
            wikidata_id = resolve_security_wikidata_id(name, wiki)
            chains = []
            if wikidata_id:
                chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains,
                                                min_sitelinks=2)

            # Fallback: use chain_wiki if direct QID yields < 2 chains
            chain_wiki = ed.get("chain_wiki", "")
            if (not chains or len(chains) < 2) and chain_wiki:
                alt_qid = resolve_security_wikidata_id(chain_wiki, chain_wiki)
                if alt_qid and alt_qid != wikidata_id:
                    alt_chains = fetch_multihop_triples(
                        alt_qid, num_hops=2, limit=num_chains,
                        min_sitelinks=2)
                    if alt_chains and len(alt_chains) >= 2:
                        chains = alt_chains
                        wikidata_id = alt_qid
                        print(f"  {name}: chain_wiki fallback -> {chain_wiki} ({alt_qid})",
                              flush=True)

            if chains and len(chains) >= 2:
                articles = fetch_wikipedia_for_entities(chains)
                print(f"  {name}: fetched {len(articles)} Wikipedia articles "
                      f"for {len(chains)} chains", flush=True)
                ed["wikidata_id"] = wikidata_id or ""
                ed["chains"] = chains
                ed["articles"] = articles
            else:
                ed["wikidata_id"] = wikidata_id or ""
                ed["chains"] = chains or []
                ed["articles"] = []

            entity_data_list.append(ed)

        if entity_data_list:
            output_dir = os.path.dirname(chains_cache) if "/" in chains_cache else None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(chains_cache, "w") as f:
                json.dump(entity_data_list, f, indent=2)
            print(f"Upgraded and cached {len(entity_data_list)} entities for {category}", flush=True)

    else:
        entities = get_category_entities(category)
        if not entities:
            print(f"No entities for category {category}", flush=True)
            return []

        entity_data_list = []
        for entity in tqdm.tqdm(entities, desc=f"Fetching {category} data"):
            time.sleep(2)  # Rate-limit Wikidata API calls
            name = entity.get("name", "")
            entity_id = entity.get("id", "")
            wiki = entity.get("wiki", name)

            # Look up reference data based on entity type
            if entity_type == "algorithm":
                data = get_crypto_algorithm_data(name)
            elif entity_type in ("attack_group", "attack_technique", "attack_software"):
                data = get_attack_entity_data(name)
            else:
                data = None

            if data is None:
                print(f"  {name}: no reference data found, skipping", flush=True)
                continue

            # Resolve Wikidata QID
            wikidata_id = resolve_security_wikidata_id(name, wiki)
            chains = []
            articles = []

            if wikidata_id:
                chains = fetch_multihop_triples(wikidata_id, num_hops=2, limit=num_chains,
                                                min_sitelinks=2)
                if chains and len(chains) >= 2:
                    articles = fetch_wikipedia_for_entities(chains)
                    print(f"  {name}: fetched {len(articles)} Wikipedia articles "
                          f"for {len(chains)} chains", flush=True)
                else:
                    chains = chains or []
                    print(f"  {name}: insufficient KG chains ({len(chains)}) from direct QID",
                          flush=True)

            # Fallback: use chain_wiki if direct QID yields < 2 chains
            if (not chains or len(chains) < 2) and entity.get("chain_wiki"):
                alt_qid = resolve_security_wikidata_id(entity["chain_wiki"], entity["chain_wiki"])
                if alt_qid and alt_qid != wikidata_id:
                    alt_chains = fetch_multihop_triples(
                        alt_qid, num_hops=2, limit=num_chains,
                        min_sitelinks=2)
                    if alt_chains and len(alt_chains) >= 2:
                        chains = alt_chains
                        wikidata_id = alt_qid
                        articles = fetch_wikipedia_for_entities(chains)
                        print(f"  {name}: chain_wiki fallback -> {entity['chain_wiki']} ({alt_qid}): "
                              f"{len(chains)} chains, {len(articles)} articles", flush=True)

            if not chains or len(chains) < 2:
                if not wikidata_id:
                    print(f"  {name}: no Wikidata QID found, will use legacy clues", flush=True)
                else:
                    print(f"  {name}: insufficient KG chains ({len(chains)}), will use legacy clues",
                          flush=True)

            # Always fetch legacy clues as fallback
            clues = fetch_entity_clues(wiki, entity_type)

            entity_data_list.append({
                "name": name,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "wikidata_id": wikidata_id or "",
                "data": data,
                "clues": clues,
                "chains": chains,
                "articles": articles,
                "chain_wiki": entity.get("chain_wiki", ""),
                "wiki": entity.get("wiki", ""),
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

    # Generate QA pairs
    all_qa_pairs = []
    prev_questions = []
    used_templates = set()

    # Single-entity questions (per-entity loop)
    target_single = per_category * 2 // 3
    for ed in entity_data_list:
        if len([q for q in all_qa_pairs if q.get("template_type") == "single"]) >= target_single:
            break

        chains = ed.get("chains", [])
        articles = ed.get("articles", [])
        clues = ed.get("clues", [])

        # Use chain-based pipeline if chains available, else legacy clues
        if chains and len(chains) >= 2:
            entity_qas = gen_security_qa_from_entity(
                ed, agent_info,
                num_questions=questions_per_entity,
                prev_questions=prev_questions,
                used_templates=used_templates,
                articles=articles,
                category=category,
            )
            for qa in entity_qas:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
            if entity_qas:
                print(f"  {ed['name']}: generated {len(entity_qas)} chain-based QA pairs "
                      f"(total: {len(all_qa_pairs)})", flush=True)
        else:
            # Legacy fallback for entities without Wikidata entries
            legacy_acceptable_ids = [ed.get("entity_id", "")]
            if ed.get("chain_wiki"):
                legacy_acceptable_ids.append(ed["chain_wiki"])
            if ed.get("wiki"):
                legacy_acceptable_ids.append(ed["wiki"])
            qa = gen_security_qa_single(
                ed["name"], ed.get("data", {}), clues, ed.get("entity_type", entity_type),
                used_templates, prev_questions, category,
                acceptable_identifiers=legacy_acceptable_ids,
            )
            if qa:
                all_qa_pairs.append(qa)
                prev_questions.append(qa["question"])
                print(f"  {ed['name']}: single QA generated (legacy) "
                      f"(total: {len(all_qa_pairs)})", flush=True)

        # Reset used_templates after cycling — include attack_technique templates for attack_software
        et = ed.get("entity_type", entity_type)
        eligible_types = {et}
        if et == "attack_software":
            eligible_types.add("attack_technique")
        single_templates = [t for t, d in SECURITY_TEMPLATES.items()
                           if d["type"] == "single" and d["entity_type"] in eligible_types]
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

        if ed_a.get("entity_type", entity_type) != ed_b.get("entity_type", entity_type):
            continue

        # Retry a rejected pair a few times: comparative composition is
        # one-shot per call, so a single stochastic Phase-3 rejection would
        # otherwise discard a viable pair (single-entity gen gets
        # num_questions*3 attempts; give each comparative pair a few too).
        qa = None
        for _ in range(3):
            qa = gen_security_qa_comparative(
                ed_a["name"], ed_a.get("data", {}), ed_a.get("clues", []),
                ed_b["name"], ed_b.get("data", {}), ed_b.get("clues", []),
                ed_a.get("entity_type", entity_type), comp_used, prev_questions, category,
            )
            if qa:
                break
        if qa:
            all_qa_pairs.append(qa)
            prev_questions.append(qa["question"])
            print(f"  {ed_a['name']} vs {ed_b['name']}: comparative QA generated "
                  f"(total: {len(all_qa_pairs)})", flush=True)

    return all_qa_pairs


def run_security_bench(args, agent_info):
    """Run the security benchmark pipeline.

    Per category:
    1. Fetch entities + quantitative data + Wikipedia clues
    2. Generate QA pairs by template type
    3. V1 verification (closed-book)
    4. V2 verification (Security + Browser + Python tools)
    5. Merge all categories into final output
    """
    prefix = args.outfile_prefix1
    run_id = getattr(args, "run_id", None) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{prefix}__{run_id}"
    categories_arg = args.security_categories
    per_category = args.security_per_category
    v1_samples = args.security_v1_samples
    v2_samples = args.security_v2_samples
    v1_threshold = args.security_v1_threshold
    v2_threshold = args.security_v2_threshold
    num_chains = getattr(args, "security_num_chains", 20)
    questions_per_entity = getattr(args, "security_questions_per_entity", 5)

    bench_label = "security_bench"

    if categories_arg:
        categories_to_process = [s.strip() for s in categories_arg.split(",")]
    else:
        categories_to_process = list(ENTITY_UNIVERSE.keys())

    is_subset = categories_arg is not None and len(categories_to_process) < len(ENTITY_UNIVERSE)

    print(f"=== SECURITY/CYBER BENCH (run_id={run_id}) ===", flush=True)
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
                    answer_checker=is_security_answer_correct,
                    llm_judge=_llm_judge_security_answer,
                    verification_prompt=SECURITY_V1_PROMPT,
                )

            if not v1_filtered:
                print(f"No V1-filtered QA pairs for {category}", flush=True)
                summary[category] = {"generated": len(qa_pairs) if 'qa_pairs' in dir() else "cached",
                                     "v1_filtered": 0, "v2_total": 0, "v2_filtered": 0}
                continue

            v2_filtered, v2_all = verify_security_v2(
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
    print("SECURITY BENCH SUMMARY", flush=True)
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
        prog="security_drbencher",
        description="Security/Cyber Benchmark: Cryptographic Standards + MITRE ATT&CK + Wikipedia entity ID",
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
    parser.add_argument("--outfile_prefix1", type=str, default="output/security/bench")

    # Exp mode
    parser.add_argument("--exp_mode", type=str, default="security_bench")

    # Run ID for multi-run support (auto-generated if not provided)
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: auto YYYYMMDD_HHMMSS)")

    # Security-specific
    parser.add_argument("--security_categories", type=str, default=None,
                        help="Comma-separated categories (default: all). E.g. 'block_ciphers,attack_groups'")
    parser.add_argument("--security_per_category", type=int, default=50,
                        help="Target QA pairs per category")
    parser.add_argument("--security_v1_samples", type=int, default=10,
                        help="V1 sampling attempts per question")
    parser.add_argument("--security_v2_samples", type=int, default=10,
                        help="V2 (Security+Browser+Python) sampling attempts per question")
    parser.add_argument("--security_v1_threshold", type=float, default=0.5,
                        help="V1 accuracy ceiling -- keep below this")
    parser.add_argument("--security_v2_threshold", type=float, default=0.5,
                        help="V2 accuracy ceiling -- keep below this")
    parser.add_argument("--security_num_chains", type=int, default=20,
                        help="Number of KG chains to fetch per entity")
    parser.add_argument("--security_questions_per_entity", type=int, default=5,
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

    if args.exp_mode == "security_bench":
        run_security_bench(args, agent_info)
        # The in-process vLLM engine spawns worker subprocesses that outlive the
        # bench; shut the engine down and terminate the process group so the job
        # exits cleanly instead of hanging until it is killed by hand.
        shutdown_and_exit(0)
    else:
        print(f"Unknown exp_mode: {args.exp_mode}", flush=True)
        sys.exit(1)
