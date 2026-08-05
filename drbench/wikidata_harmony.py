import re, time, os, argparse, ast, json, tqdm
import glob
import random
import copy
import asyncio
import datetime
import difflib
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
import numpy as np

# OpenAI Harmony imports for gpt-oss-120b
from openai_harmony import (
    Message, Conversation, SystemContent, Role, ReasoningEffort,
    StreamableParser, load_harmony_encoding, HarmonyEncodingName
)

from .util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from .tool_util import (
    _generate_lm_answers, _generate_lm_answers_harmony, extract_json_v2,
    search_related_pages, search_step, get_pageviews,
    search_wikidata_entities, get_entity_data, get_entity_label,
    sparql_query, fetch_multihop_triples, fetch_wikipedia_for_entities
)
from .harmony_vllm import HarmonyVLLMGenerator

# KG Browser import for V2 browser-based verification
try:
    from tools.kg_browser import MultiSourceKnowledgeBrowserTool, MultiSourceKnowledgeBackend
    BROWSER_TOOL_AVAILABLE = True
except ImportError:
    BROWSER_TOOL_AVAILABLE = False

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

# Supported answer entity types for multi-hop QA
ANSWER_ENTITY_TYPES = [
    "person",           # e.g., "Albert Einstein", "Napoleon"
    "location",         # e.g., "Paris", "Mount Everest"
    "date",             # e.g., "1945", "July 4, 1776"
    "event",            # e.g., "Battle of Waterloo", "Apollo 11 landing"
    "organization",     # e.g., "United Nations", "NASA"
    "number/quantity",  # e.g., "42 million", "3.14"
    "work",             # e.g., "Mona Lisa", "War and Peace" (artistic/literary)
    "concept/term",     # e.g., "democracy", "photosynthesis"
    "time period",      # e.g., "Renaissance", "Jurassic period"
]

# Wikidata multi-hop QA generation prompt (single question at a time)
WIKIDATA_MULTIHOP_QA_PROMPT_SINGLE = """You are generating a multi-hop question from Wikidata knowledge graph triples.
Each chain shows: Entity1 -[property]-> Entity2 -[property]-> Entity3

Answer: {{specific entity from end of a chain}}
Chain Used: {{chain number(s)}}
Question: {{natural multi-hop question requiring 2+ hops}}
Reasoning: {{Step-by-step reasoning through the chain}}

TRIPLE CHAINS:
{triple_chains}

REQUIREMENTS:
- Answer MUST be a specific entity from the chains
- Question must require 2+ relationship hops
- Question should sound natural (no mention of "Wikidata" or "triples")
- Answerable by a knowledgeable person without the triples

Generate one multi-hop question now:
"""

# Developer content for Harmony format (simpler, more direct)
HARMONY_WIKIDATA_DEVELOPER_CONTENT = """You generate multi-hop questions from Wikidata knowledge graph triples.
Response format: Answer: / Chain Used: / Question: / Reasoning:"""

# --- Answer-First Grounded QA Generation Prompts (Wikidata Triples) ---

WIKIDATA_ANSWER_SELECTION_DEVELOPER = "You select a quiz-worthy answer entity from Wikidata triple chains radiating from a seed entity."

WIKIDATA_ANSWER_SELECTION_PROMPT = """The following Wikidata triple chains all radiate from the seed entity "{seed_entity}".
Pick one non-seed entity or value from any chain that would make a good quiz answer (e.g., a city, person, date, number, organization, or notable concept).

{chains_text}
{avoid_clause}
Respond with EXACTLY this format (nothing else):
Answer: <an entity or value label from the chains that is NOT "{seed_entity}">
Source_Chain: <the single chain number the answer comes from, e.g. 3>"""

WIKIDATA_FACT_EXTRACTION_DEVELOPER = "You extract clue facts about a seed entity from Wikidata triple chains, citing specific triple relationships."

WIKIDATA_FACT_EXTRACTION_PROMPT = """These Wikidata triple chains all start from the seed entity "{seed_entity}".
Extract one descriptive fact per chain about "{seed_entity}" that could serve as a clue to identify it.

For each fact, cite the specific triple relationship from the chain (e.g., "Japan" -[capital]-> "Tokyo").

{chains_text}

Respond in this JSON format:
{{
  "facts": [
    {{"fact_id": "C1_F1", "chain_num": 1, "entity": "...", "property": "...", "value": "...", "fact": "short factual clue about {seed_entity}"}},
    {{"fact_id": "C2_F1", "chain_num": 2, "entity": "...", "property": "...", "value": "...", "fact": "another clue about {seed_entity}"}},
    {{"fact_id": "C3_F1", "chain_num": 3, "entity": "...", "property": "...", "value": "...", "fact": "..."}}
  ]
}}

REQUIREMENTS:
- Extract exactly 1 fact per chain that describes the seed entity "{seed_entity}"
- Each fact must cite the entity, property, and value from a specific triple in that chain
- Include the chain number for each fact
- Facts should be useful as identifying clues (e.g., capital city, continent, official language)
- Prefer facts based on SPECIFIC, low-frequency predicates (e.g., "founded by", "signed the Treaty of X", "hosted the 1964 Olympics") over generic, high-frequency predicates (e.g., "has effect", "participant", "has part", "influenced by") which tend to produce ambiguous questions
- Prefer PERSISTENT properties (geography, founding date, capital, official language) over TRANSIENT ones (temporary political groupings, short-lived alliances, passing influence)
- PROPERTY LABEL FIDELITY: Use the EXACT Wikidata property label from the chain for the "property" JSON field (e.g., "country of citizenship", "educated at"), NOT generalized synonyms like "founded by" or "associated with"
- PROPERTY DIVERSITY: Maximize variety of distinct property labels across facts — do NOT merge different properties into one umbrella term. If the chains contain "country of citizenship", "educated at", and "occupation", use all three as separate properties
- MULTI-HOP PROPERTY LABELS: For 2-hop chains (A -[p1]-> B -[p2]-> C), use the hop-2 property label (p2) in the "property" field, not the hop-1 label
- SEMANTIC FIDELITY: The fact text MUST faithfully represent the Wikidata property label without overstating or embellishing the relationship. Use the property's literal meaning. For example: "found in taxon" means "detected in the organism" — do NOT rephrase as "produced by", "sourced from", or "originates from"; "habitat type" is a generic category — do NOT map it to specific named locations
- NO ENTITY-LEVEL EMBELLISHMENT: State ONLY the specific triple relationship for each fact. Do NOT add descriptors of what the seed entity "{seed_entity}" *is*, its category, or what it *is used for* (e.g. "which is used as a medication", "a drug used to treat X", "a mountain in the Andes") unless that exact claim is the property/value of THAT triple. Even if OTHER chains establish such context (e.g. one chain gives "has use → pharmacotherapy"), do NOT carry it into unrelated facts (e.g. "effective dose", "side effect") — each fact must stand on its own single triple
- MULTI-HOP ATTRIBUTION: For 2-hop chains (A -[p1]-> B -[p2]-> C), if you use a hop-2 property, the fact text must explicitly mention the intermediate entity B. Write "A is related to B, which has property C" — do NOT flatten to "A has property C"
- Use fact_id format: C{{chain_num}}_F1"""

# --- Chain Grounding Check Prompts (Pre-filter) ---

CHAIN_GROUNDING_CHECK_DEVELOPER = "You are a fact-checking assistant that determines whether a Wikidata triple chain represents a genuinely connected relationship or just independent facts sharing an entity."

CHAIN_GROUNDING_CHECK_PROMPT = """Examine this 2-hop Wikidata triple chain:

{chain_description}

And this excerpt from the Wikipedia article for the intermediate entity "{intermediate_entity}":

---
{article_excerpt}
---

Does this chain represent a CONNECTED relationship — i.e., does the Wikipedia article describe or imply a meaningful link between the first hop's relationship and the second hop's relationship through the intermediate entity?

Or are the two hops merely INDEPENDENT facts that happen to share the intermediate entity (e.g., "X has effect Y" and "Y has diplomatic relation Z" are unrelated properties of Y)?

Respond with exactly one word: SUPPORTED or UNSUPPORTED"""

# --- Fact Grounding Verification Prompts (Phase 1.5) ---

FACT_GROUNDING_VERIFICATION_DEVELOPER = "You are a fact-verification assistant that checks whether specific factual claims are supported by Wikipedia article text."

FACT_GROUNDING_VERIFICATION_PROMPT = """Determine whether the following factual claim is supported by the Wikipedia article excerpt below.

CLAIM: {claim_text}

WIKIPEDIA ARTICLE: "{article_title}"
---
{article_excerpt}
---

Is this claim supported by the article text?
- SUPPORTED: The article explicitly states or directly implies this fact.
- PARTIALLY_SUPPORTED: The article mentions related information that is consistent with the claim but does not explicitly confirm it.
- UNSUPPORTED: The article does not contain information supporting this claim, or contradicts it.

Respond with exactly one word: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED"""

WIKIDATA_GROUNDED_QA_DEVELOPER = "You are a quiz-show writer who crafts engaging, conversational trivia questions. You weave factual clues into short narratives that feel natural to read aloud, never sounding like a database query or fill-in-the-blank template."

WIKIDATA_GROUNDED_QA_PROMPT = """Compose a multi-hop question using these clue facts that describe an unnamed entity, then ask for the answer "{answer}".

CLUE FACTS (about the unnamed seed entity — do NOT name it in the question):
{facts_text}

ANSWER FACT (the triple that leads to the answer):
{answer_fact}

--- STYLE EXAMPLES ---

BAD (formulaic, one run-on sentence):
"What is the official currency of the country whose capital is Nairobi, whose official language is Swahili, and whose continent is Africa?"

BAD (nested relative clauses):
"What is the head of state of the entity that is located in South America, that has a population of over 200 million, and that gained independence in 1822?"

GOOD (natural, multi-sentence narrative):
"A certain African country has its capital in Nairobi and recognizes Swahili as an official language. What currency is used there?"

GOOD (varied structure, conversational):
"This South American nation gained independence in 1822 and is home to over 200 million people. Who currently serves as its head of state?"

--- STYLE GUIDANCE ---
- Write 2–3 sentences, not one run-on sentence.
- Weave facts into a narrative: describe the entity, don't enumerate its properties.
- Vary sentence openings and structure across questions.
- Use natural references like "this country", "there", "its", "the same city" instead of nested relative clauses.
- Use simple past or present tense. Do NOT use present participles (-ing forms) to describe the entity (e.g., write "sits on the Danube" not "sitting on the Danube"; write "operated in Europe" not "operating in Europe").
- Ground clues in PERSISTENT properties (geography, founding date, official language, capital) rather than TRANSIENT ones (temporary political blocs, passing influence, short-lived alliances).
- Prefer clues from SPECIFIC, low-frequency predicates (e.g., "signed the Treaty of Trianon", "hosted the 1936 Olympics") over GENERIC, high-frequency predicates (e.g., "has effect", "participant", "influenced by") — specific predicates naturally narrow to one answer.

--- ANTI-PATTERNS TO AVOID ---
- "What is X of the entity/country/person that..."
- Comma-separated chains of "whose" or "which" clauses
- Starting with "What is [answer-type] of..."
- Packing all clues into a single sentence
- Present participles as main descriptors ("a country sitting on...", "a nation operating in...")
- Clues based on transient or era-specific groupings (e.g., "Eastern Bloc member", "Cold War ally") without anchoring to a persistent, distinguishing property

--- PREVIOUSLY GENERATED QUESTIONS (do NOT repeat these) ---
{previous_questions}

--- OUTPUT FORMAT ---

Respond with EXACTLY this format:
Answer: {answer}
Question: <a question that uses exactly 3 of the most discriminating clue facts to describe the seed entity WITHOUT naming it, then asks for the answer>
Used_Facts: <comma-separated fact_ids of the clue facts used, e.g. C1_F1, C2_F1, C4_F1>
Reasoning: <brief chain: clues identify the seed, then the answer follows from its relationship to the seed>

REQUIREMENTS:
- Select exactly 3 clue facts from 3 different chains — the most discriminating subset that, together with the answer relationship, uniquely identifies the answer. Do NOT use all available facts; fewer specific clues make harder, better questions
- Do NOT name the seed entity in the question — the reader must figure it out from the clues
- Every claim in the question must come from one of the listed Used_Facts or the Answer Fact
- Do NOT add any information not present in the provided facts
- The question should sound natural (no mention of "Wikidata" or "triples")
- Used_Facts must list exactly the 3 clue fact_ids used in the question (no more, no less)
- The question MUST have exactly ONE unambiguous answer — the clues must narrow it down so no other entity or value could plausibly be the answer
- Ask for the SPECIFIC answer entity; avoid vague phrasing (e.g. "which empire?") when a more precise question is possible (e.g. "what is the name of the colonial empire?")
- If the answer could be confused with a closely related entity (e.g. "German Empire" vs "German colonial empire"), add enough distinguishing detail to eliminate the ambiguity
- CRITICAL: Uniqueness applies to the ANSWER, not just the seed. Even if clues perfectly identify the seed entity, the question is ambiguous if multiple entities satisfy the answer relationship. For example, "which socialist republic was affected by the Eastern Front" is ambiguous because many were. Add distinguishing constraints (dates, specific events, unique attributes) to narrow to exactly one answer."""

UNIQUENESS_CHECK_PROMPT = """Given this question, determine whether it has exactly one unambiguous answer.

Question: {question}
Intended answer: {answer}

Think step-by-step:
1. What entity or concept do the clues describe?
2. What relationship leads to the intended answer?
3. Could any OTHER entity or value also correctly answer this question via the same or a similar relationship?

Respond with EXACTLY this format:
Alternatives: <comma-separated list of other plausible answers, or "none">
Verdict: UNIQUE or AMBIGUOUS
Reason: <one sentence>"""


def _is_metadata_entity(label):
    """Check if an entity label looks like Wikipedia/Wikimedia metadata rather than real content."""
    label_lower = label.lower().strip()
    METADATA_PREFIXES = [
        "wikipedia:", "category:", "wikiproject", "wikimedia",
        "template:", "portal:", "module:", "list of ",
    ]
    for prefix in METADATA_PREFIXES:
        if label_lower.startswith(prefix):
            return True
    METADATA_SUBSTRINGS = [
        "vital articles", "wikimedia", "wikiproject",
    ]
    for sub in METADATA_SUBSTRINGS:
        if sub in label_lower:
            return True
    return False


def answer_is_chain_entity(answer, chains):
    """Check if the answer matches any non-metadata entity or value label in the chain data.

    Rejects Wikipedia/Wikimedia metadata entities (e.g., "Wikipedia:Vital articles/Level/4",
    "Category:Films shot in Africa", "WikiProject Africa").

    Args:
        answer: Candidate answer string
        chains: List of chain dicts

    Returns:
        True if the answer matches a real (non-metadata) entity/value label (case-insensitive)
    """
    answer_lower = answer.lower().strip()
    if _is_metadata_entity(answer):
        return False
    for chain in chains:
        for hop in chain.get('chain', []):
            if hop.get('entity', {}).get('label', '').lower().strip() == answer_lower:
                return True
            if hop.get('value', {}).get('label', '').lower().strip() == answer_lower:
                return True
    return False


def _answer_has_sibling_values(answer, chains):
    """Reject answers where the same (entity, property) pair leads to multiple values.

    Example: if the answer "6th arrondissement of Paris" comes from the hop
    Paris -[contains administrative territorial entity]-> 6th arrondissement,
    and other chains also have Paris -[contains administrative territorial entity]-> 7th/8th/...,
    then the answer is ambiguous and should be rejected.
    """
    answer_lower = answer.lower().strip()

    # Find all (entity, property) contexts where this answer appears as a value
    answer_contexts = set()
    for chain in chains:
        for hop in chain.get('chain', []):
            v = hop.get('value', {}).get('label', '').lower().strip()
            if v == answer_lower:
                e = hop.get('entity', {}).get('label', '').lower().strip()
                p = hop.get('property', {}).get('label', '').lower().strip()
                answer_contexts.add((e, p))

    if not answer_contexts:
        return False

    # For each context, check if any other chain has the same (entity, property) with a different value
    for e_ctx, p_ctx in answer_contexts:
        for chain in chains:
            for hop in chain.get('chain', []):
                e = hop.get('entity', {}).get('label', '').lower().strip()
                p = hop.get('property', {}).get('label', '').lower().strip()
                v = hop.get('value', {}).get('label', '').lower().strip()
                if e == e_ctx and p == p_ctx and v != answer_lower:
                    return True

    return False


def _check_answer_uniqueness(question, answer, use_harmony, _gen=None):
    """Ask the LLM whether the question has a unique answer or is ambiguous.

    Args:
        question: The generated question text
        answer: The intended gold answer
        use_harmony: Whether to use Harmony generator
        _gen: Fallback generation function (used when not in Harmony mode)

    Returns:
        (is_unique: bool, alternatives: str) tuple
    """
    prompt_text = UNIQUENESS_CHECK_PROMPT.format(question=question, answer=answer)

    try:
        if use_harmony:
            gen = get_harmony_generator()
            if gen is None:
                print(f"    Warning: Harmony generator unavailable, accepting question", flush=True)
                return True, "generator_unavailable"
            response = gen.generate_response_sync(
                developer_content="You are a careful evaluator of question uniqueness.",
                user_content=prompt_text,
                temperature=0.3,
                max_tokens=512,
                reasoning_effort="low",
            )
        else:
            if _gen is None:
                raise RuntimeError("_gen must be provided when not using Harmony mode")
            response = _gen(prompt_text, temperature=0.3, max_tokens=512)

        # Parse the Verdict line
        verdict_match = re.search(r'Verdict:\s*(UNIQUE|AMBIGUOUS)', response, re.IGNORECASE)
        alternatives_match = re.search(r'Alternatives:\s*(.+?)(?:\n|$)', response)

        alternatives = alternatives_match.group(1).strip() if alternatives_match else "unknown"
        if verdict_match:
            is_unique = verdict_match.group(1).strip().upper() == "UNIQUE"
            return is_unique, alternatives
        else:
            # If we can't parse the verdict, be conservative and accept
            print(f"    Warning: Could not parse uniqueness verdict, accepting question", flush=True)
            return True, "unparseable"
    except Exception as e:
        print(f"    Warning: Uniqueness check failed ({e}), accepting question", flush=True)
        return True, "error"


def validate_triple_in_chains(chain_num, entity_label, property_label, value_label,
                              chains, target_entity=None):
    """Check if a claimed triple actually exists in the chain data.

    Args:
        chain_num: 1-indexed chain number
        entity_label: Claimed entity label (e.g., "Einstein")
        property_label: Claimed property label (e.g., "birthplace")
        value_label: Claimed value label (e.g., "Ulm")
        chains: List of chain dicts from chains_data['chains']
        target_entity: Optional alias for the chain root entity.  When chains
            come from a ``chain_wiki`` source (e.g. EternalBlue for NotPetya),
            the LLM may write the *target* entity name instead of the chain
            root.  This parameter allows hop-0 to accept the target name.

    Returns:
        (is_valid, matched_hop_index) tuple, or (False, None) if not found
    """
    if chain_num < 1 or chain_num > len(chains):
        return False, None

    # Guard against None labels from malformed LLM extractions
    if not entity_label or not property_label or not value_label:
        return False, None

    chain = chains[chain_num - 1]
    hops = chain.get('chain', [])

    entity_lower = entity_label.lower().strip()
    property_lower = property_label.lower().strip()
    value_lower = value_label.lower().strip()

    for hop_idx, hop in enumerate(hops):
        hop_entity = hop.get('entity', {}).get('label', '').lower().strip()
        hop_property = hop.get('property', {}).get('label', '').lower().strip()
        hop_value = hop.get('value', {}).get('label', '').lower().strip()

        if hop_entity == entity_lower and hop_property == property_lower and hop_value == value_lower:
            return True, hop_idx

        # Relaxed: hop 0 allows target_entity as alias for chain root
        if target_entity and hop_idx == 0:
            target_lower = target_entity.lower().strip()
            if entity_lower == target_lower and hop_property == property_lower and hop_value == value_lower:
                return True, hop_idx

    # --- Collapsed hop-1 matching ---
    # LLMs often write entity=A, property=p2, value=C for a 2-hop chain
    # "A -[p1]-> B -[p2]-> C", following the MULTI-HOP PROPERTY LABELS
    # instruction.  Accept this as a valid hop-1 match when the entity
    # matches hop-0's entity (the seed) and property/value match hop-1.
    if len(hops) >= 2:
        h0_entity = hops[0].get('entity', {}).get('label', '').lower().strip()
        h1_prop = hops[1].get('property', {}).get('label', '').lower().strip()
        h1_value = hops[1].get('value', {}).get('label', '').lower().strip()

        seed_ok = (entity_lower == h0_entity)
        if target_entity and not seed_ok:
            seed_ok = (entity_lower == target_entity.lower().strip())

        if seed_ok and property_lower == h1_prop and value_lower == h1_value:
            return True, 1  # matched as collapsed hop-1

    # --- Compound-property matching for collapsed 2-hop triples ---
    # LLMs often collapse "A -[p1]-> B -[p2]-> C" into a single fact with
    # entity=A, property="p1 → p2", value=C.  Detect this and validate
    # against the full 2-hop chain.
    compound_parts = re.split(r'\s*(?:→|->)\s*', property_lower)
    if len(compound_parts) == 2 and len(hops) >= 2:
        p0, p1 = compound_parts
        hop0 = hops[0]
        hop1 = hops[1]
        h0_entity = hop0.get('entity', {}).get('label', '').lower().strip()
        h0_prop = hop0.get('property', {}).get('label', '').lower().strip()
        h1_prop = hop1.get('property', {}).get('label', '').lower().strip()
        h1_value = hop1.get('value', {}).get('label', '').lower().strip()

        entity_ok = (entity_lower == h0_entity)
        if target_entity and not entity_ok:
            entity_ok = (entity_lower == target_entity.lower().strip())

        if entity_ok and h0_prop == p0 and h1_prop == p1 and h1_value == value_lower:
            return True, 0  # matched as compound path starting at hop 0

    return False, None


def validate_fact_references(used_fact_ids, extracted_facts, min_chains=3):
    """Check that all used fact IDs exist and span the minimum number of chains.

    Args:
        used_fact_ids: List of fact_id strings (e.g., ["C1_F1", "C3_F1"])
        extracted_facts: List of fact dicts with 'fact_id' and 'chain_num'
        min_chains: Minimum number of distinct chains required

    Returns:
        (is_valid, reason) tuple
    """
    fact_id_set = {f['fact_id'] for f in extracted_facts}

    # Check all referenced IDs exist
    missing = [fid for fid in used_fact_ids if fid not in fact_id_set]
    if missing:
        return False, f"Missing fact_ids: {missing}"

    # Check chain span
    fact_lookup = {f['fact_id']: f for f in extracted_facts}
    chains_used = {fact_lookup[fid]['chain_num'] for fid in used_fact_ids}
    if len(chains_used) < min_chains:
        return False, f"Only {len(chains_used)} chains used, need {min_chains}+"

    return True, "OK"


def validate_answer_in_facts(answer, used_fact_ids, extracted_facts):
    """Check that the answer appears in at least one of the referenced facts.

    Args:
        answer: The answer string
        used_fact_ids: List of fact_id strings
        extracted_facts: List of fact dicts

    Returns:
        (is_valid, reason) tuple
    """
    fact_lookup = {f['fact_id']: f for f in extracted_facts}
    answer_lower = answer.lower().strip()

    for fid in used_fact_ids:
        if fid in fact_lookup:
            fact = fact_lookup[fid]
            # Check in fact text, entity, property, and value
            if (answer_lower in fact.get('fact', '').lower()
                    or answer_lower in fact.get('entity', '').lower()
                    or answer_lower in fact.get('value', '').lower()):
                return True, "OK"

    return False, f"Answer '{answer}' not found in any referenced facts"


def _check_chain_structural_diversity(sampled_chains, min_unique_intermediates=2, min_unique_property_pairs=2, pool_chains=None):
    """Check that sampled chains have sufficient structural diversity.

    Rejects degenerate samples where all chains share the same intermediate entity
    and/or the same (property1, property2) pair (e.g., all chains are
    "X -[has effect]-> Japan -[diplomatic relation]-> Z").

    When pool_chains is provided, thresholds are clamped to the pool's actual
    diversity so that categories with naturally low diversity (e.g., only 1 unique
    intermediate across all chains) can still pass.

    Args:
        sampled_chains: List of chain dicts
        min_unique_intermediates: Minimum number of distinct intermediate entities
        min_unique_property_pairs: Minimum number of distinct (p1, p2) property pairs
        pool_chains: Optional full pool of chains to compute achievable diversity

    Returns:
        (is_diverse, reason_string) tuple
    """
    # Compute pool-level diversity to clamp thresholds
    if pool_chains is not None:
        pool_intermediates = set()
        pool_property_pairs = set()
        for chain in pool_chains:
            hops = chain.get('chain', [])
            if len(hops) >= 2:
                inter = hops[0].get('value', {}).get('label', '').lower().strip()
                if inter:
                    pool_intermediates.add(inter)
                p1 = hops[0].get('property', {}).get('label', '').lower().strip()
                p2 = hops[1].get('property', {}).get('label', '').lower().strip()
                if p1 and p2:
                    pool_property_pairs.add((p1, p2))
        min_unique_intermediates = max(1, min(min_unique_intermediates, len(pool_intermediates)))
        min_unique_property_pairs = max(1, min(min_unique_property_pairs, len(pool_property_pairs)))

    intermediates = set()
    property_pairs = set()

    for chain in sampled_chains:
        hops = chain.get('chain', [])
        if len(hops) >= 2:
            # Intermediate entity is the value of hop 0 (= entity of hop 1)
            intermediate = hops[0].get('value', {}).get('label', '').lower().strip()
            if intermediate:
                intermediates.add(intermediate)
            p1 = hops[0].get('property', {}).get('label', '').lower().strip()
            p2 = hops[1].get('property', {}).get('label', '').lower().strip()
            if p1 and p2:
                property_pairs.add((p1, p2))

    if len(intermediates) < min_unique_intermediates:
        return False, (f"Only {len(intermediates)} unique intermediate(s) "
                       f"(need {min_unique_intermediates}+): {intermediates}")

    if len(property_pairs) < min_unique_property_pairs:
        return False, (f"Only {len(property_pairs)} unique property pair(s) "
                       f"(need {min_unique_property_pairs}+): {property_pairs}")

    return True, "OK"


# Module-level caches for grounding checks (persist across retries within a run)
_chain_grounding_cache = {}
_fact_grounding_cache = {}


def _find_article_for_entity(entity_label, entity_qid, articles):
    """Find a Wikipedia article matching an entity by QID first, then title fallback.

    Args:
        entity_label: Entity label string
        entity_qid: Entity QID string (e.g., 'Q17')
        articles: List of article dicts with 'qid', 'title', 'paragraph'

    Returns:
        Article dict or None
    """
    if entity_qid:
        for art in articles:
            if art.get('qid', '') == entity_qid:
                return art
    # Title fallback (case-insensitive)
    label_lower = entity_label.lower().strip()
    for art in articles:
        if art.get('title', '').lower().strip() == label_lower:
            return art
    return None


def filter_chains_by_grounding(sampled_chains, articles, use_harmony, agent_info=None, max_checks_per_chain=2):
    """Filter chains by checking if their 2-hop relationship is grounded in Wikipedia.

    For each chain, finds Wikipedia articles for the intermediate entity and asks
    the LLM whether the chain represents a connected relationship.

    Args:
        sampled_chains: List of chain dicts
        articles: List of article dicts from chains_data['articles']
        use_harmony: Whether to use Harmony generator
        agent_info: Tuple of (model, tokenizer, client) for non-Harmony mode
        max_checks_per_chain: Max articles to check per chain before giving up

    Returns:
        (passed_chains, passed_indices, rejected_indices) tuple
    """
    global _chain_grounding_cache

    passed_chains = []
    passed_indices = []
    rejected_indices = []

    for idx, chain in enumerate(sampled_chains):
        path_desc = chain.get('path_description', '')

        # Check cache first
        if path_desc in _chain_grounding_cache:
            if _chain_grounding_cache[path_desc]:
                passed_chains.append(chain)
                passed_indices.append(idx)
            else:
                rejected_indices.append(idx)
            continue

        hops = chain.get('chain', [])
        if len(hops) < 2:
            # Single-hop chain: pass by default
            _chain_grounding_cache[path_desc] = True
            passed_chains.append(chain)
            passed_indices.append(idx)
            continue

        # Get intermediate entity (value of first hop)
        intermediate_label = hops[0].get('value', {}).get('label', '')
        intermediate_qid = hops[0].get('value', {}).get('id', '')

        # Find articles for intermediate entity
        article = _find_article_for_entity(intermediate_label, intermediate_qid, articles)

        if not article or not article.get('paragraph', '').strip():
            # No article found — be lenient, pass the chain
            _chain_grounding_cache[path_desc] = True
            passed_chains.append(chain)
            passed_indices.append(idx)
            continue

        # Truncate article text
        article_text = article.get('paragraph', '')[:4000]

        prompt_text = CHAIN_GROUNDING_CHECK_PROMPT.format(
            chain_description=path_desc,
            intermediate_entity=intermediate_label,
            article_excerpt=article_text,
        )

        try:
            if use_harmony:
                gen = get_harmony_generator()
                if gen is None:
                    # Generator unavailable — lenient accept
                    _chain_grounding_cache[path_desc] = True
                    passed_chains.append(chain)
                    passed_indices.append(idx)
                    continue
                response = gen.generate_response_sync(
                    developer_content=CHAIN_GROUNDING_CHECK_DEVELOPER,
                    user_content=prompt_text,
                    temperature=0.0,
                    max_tokens=256,
                    reasoning_effort="low",
                )
            else:
                agent_lm, agent_tokenizer, agent_client = agent_info
                full_prompt = CHAIN_GROUNDING_CHECK_DEVELOPER + "\n\n" + prompt_text
                result = gen_from_prompt(
                    model=agent_lm, tokenizer=agent_tokenizer, prompt=[full_prompt],
                    echo_prompt=False, temperature=0.0, max_tokens=32,
                    process_func=None, service=agent_client,
                    terminate_by_linebreak='no', verbose=False,
                )
                response = result.completions[0].text

            verdict = response.strip().upper()
            is_supported = 'SUPPORTED' in verdict and 'UNSUPPORTED' not in verdict

            _chain_grounding_cache[path_desc] = is_supported
            if is_supported:
                passed_chains.append(chain)
                passed_indices.append(idx)
            else:
                rejected_indices.append(idx)
                print(f"      Chain grounding REJECTED: {path_desc[:80]}...", flush=True)

        except Exception as e:
            # On error, be lenient
            print(f"      Chain grounding check error: {e}", flush=True)
            _chain_grounding_cache[path_desc] = True
            passed_chains.append(chain)
            passed_indices.append(idx)

    return passed_chains, passed_indices, rejected_indices


def verify_facts_against_grounding(extracted_facts, sampled_chains, articles, use_harmony,
                                    agent_info=None, min_surviving_facts=3):
    """Verify extracted facts against Wikipedia grounding documents.

    For each fact, finds the relevant Wikipedia article and asks the LLM whether
    the fact is supported by the article text.

    Args:
        extracted_facts: List of fact dicts with 'fact', 'entity', 'value', 'chain_num', etc.
        sampled_chains: List of chain dicts (for finding entity QIDs)
        articles: List of article dicts from chains_data['articles']
        use_harmony: Whether to use Harmony generator
        agent_info: Tuple of (model, tokenizer, client) for non-Harmony mode
        min_surviving_facts: Minimum grounded facts required

    Returns:
        (grounded_facts, rejected_facts) tuple
    """
    global _fact_grounding_cache

    grounded_facts = []
    rejected_facts = []

    for fact in extracted_facts:
        fact_text = fact.get('fact', '')
        entity_label = fact.get('entity', '')
        value_label = fact.get('value', '')
        chain_num = fact.get('chain_num', 0)

        # Try to find QID from the chain data
        entity_qid = ''
        value_qid = ''
        if 1 <= chain_num <= len(sampled_chains):
            chain = sampled_chains[chain_num - 1]
            for hop in chain.get('chain', []):
                if hop.get('entity', {}).get('label', '').lower().strip() == entity_label.lower().strip():
                    entity_qid = hop.get('entity', {}).get('id', '')
                if hop.get('value', {}).get('label', '').lower().strip() == value_label.lower().strip():
                    value_qid = hop.get('value', {}).get('id', '')

        # Try entity article first, then value article
        article = _find_article_for_entity(entity_label, entity_qid, articles)
        article_qid = entity_qid
        if not article or not article.get('paragraph', '').strip():
            article = _find_article_for_entity(value_label, value_qid, articles)
            article_qid = value_qid

        if not article or not article.get('paragraph', '').strip():
            # No article found — be lenient, accept the fact
            grounded_facts.append(fact)
            continue

        cache_key = (fact_text, article_qid or article.get('title', ''))
        if cache_key in _fact_grounding_cache:
            if _fact_grounding_cache[cache_key]:
                grounded_facts.append(fact)
            else:
                rejected_facts.append(fact)
            continue

        # Select relevant paragraphs: prioritize those mentioning fact entities
        full_text = article.get('paragraph', '')
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
        entity_lower = entity_label.lower()
        value_lower = value_label.lower()

        # Score paragraphs by relevance to this fact
        scored = []
        for para in paragraphs:
            pl = para.lower()
            score = 0
            if entity_lower and entity_lower in pl:
                score += 2
            if value_lower and value_lower in pl:
                score += 2
            scored.append((score, para))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Cap at 3000 chars, prioritizing highest-scoring paragraphs
        excerpt_parts = []
        total_chars = 0
        for _score, para in scored:
            if total_chars + len(para) > 3000 and excerpt_parts:
                break
            excerpt_parts.append(para)
            total_chars += len(para)

        article_excerpt = "\n\n".join(excerpt_parts)
        claim_text = f"{entity_label} -[{fact.get('property', '')}]-> {value_label}: {fact_text}"

        prompt_text = FACT_GROUNDING_VERIFICATION_PROMPT.format(
            claim_text=claim_text,
            article_title=article.get('title', 'Unknown'),
            article_excerpt=article_excerpt,
        )

        try:
            if use_harmony:
                gen = get_harmony_generator()
                if gen is None:
                    # Generator unavailable — lenient accept
                    _fact_grounding_cache[cache_key] = True
                    grounded_facts.append(fact)
                    continue
                response = gen.generate_response_sync(
                    developer_content=FACT_GROUNDING_VERIFICATION_DEVELOPER,
                    user_content=prompt_text,
                    temperature=0.0,
                    max_tokens=256,
                    reasoning_effort="low",
                )
            else:
                agent_lm, agent_tokenizer, agent_client = agent_info
                full_prompt = FACT_GROUNDING_VERIFICATION_DEVELOPER + "\n\n" + prompt_text
                result = gen_from_prompt(
                    model=agent_lm, tokenizer=agent_tokenizer, prompt=[full_prompt],
                    echo_prompt=False, temperature=0.0, max_tokens=32,
                    process_func=None, service=agent_client,
                    terminate_by_linebreak='no', verbose=False,
                )
                response = result.completions[0].text

            verdict = response.strip().upper()
            is_grounded = 'SUPPORTED' in verdict  # Matches both SUPPORTED and PARTIALLY_SUPPORTED

            _fact_grounding_cache[cache_key] = is_grounded
            if is_grounded:
                grounded_facts.append(fact)
            else:
                rejected_facts.append(fact)
                print(f"      Fact grounding REJECTED: {fact.get('fact_id', '?')} - {fact_text[:60]}...", flush=True)

        except Exception as e:
            # On error, be lenient
            print(f"      Fact grounding check error: {e}", flush=True)
            _fact_grounding_cache[cache_key] = True
            grounded_facts.append(fact)

    return grounded_facts, rejected_facts


def format_extracted_facts(extracted_facts):
    """Format extracted facts for the question composition prompt.

    Args:
        extracted_facts: List of fact dicts with fact_id, chain_num, entity, property, value, fact

    Returns:
        Formatted string
    """
    lines = []
    for f in extracted_facts:
        fact_id = f.get("fact_id", f"C{f.get('chain_num', '?')}_F1")
        chain_num = f.get("chain_num", "?")
        triple_str = f'"{f.get("entity", "")}" -[{f.get("property", "")}]-> "{f.get("value", "")}"'
        lines.append(f"[{fact_id}] (Chain {chain_num}: {triple_str}): {f.get('fact', '')}")
    return "\n".join(lines)


def parse_used_facts(response_text):
    """Parse the Used_Facts field from the question composition response.

    Args:
        response_text: The LLM response text

    Returns:
        List of fact_id strings, or empty list if not found
    """
    match = re.search(r'Used_Facts:\s*(.+?)(?:\n|$)', response_text)
    if not match:
        return []
    raw = match.group(1).strip()
    return [fid.strip() for fid in raw.split(',') if fid.strip()]


# The harmony generator registry + gen_from_prompt_harmony now live in the
# shared leaf module harmony_base.py. Previously this module kept its OWN
# separate _harmony_generator global, so callers had to register the generator
# twice (once here, once in math_harmony_drbencher). Re-exporting the shared
# versions unifies them into a single generator instance.
from .harmony_base import (  # noqa: E402,F401
    get_harmony_generator,
    set_harmony_generator,
    gen_from_prompt_harmony,
)



def get_summary_of_results(json_dict, gold_key="python_answer", verbose=False):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = 'In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n'
    for line in json_dict:
        # Build category2 string with available fields
        if 'additional_requirement' in line and 'wiki_entity' in line:
            line['category2'] = f"{line['category']} || {line['wiki_entity']} [{line['additional_requirement']}]"
        elif 'additional_requirement' in line:
            line['category2'] = f"{line['category']} [{line['additional_requirement']}]"
        else:
            line['category2'] = line['category']
        category2correct_count[line['category2']].append(line['is_correct'])
        category2question[(line['category2'], line['is_correct'])].append(line)
    for category in category2correct_count:
        acc_temp = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        str_summary += f"category: {category}, accuracy: {round(acc_temp, 3)} " \
                       f"|| {sum([1 if x == 'true' else 0 for x in category2correct_count[category]])} out of {len(category2correct_count[category])}" + "\n"
        if verbose:
            str_summary += "# Questions answered correctly:\n"
            for qq in category2question[(category, 'true')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"

            str_summary += "# Questions answered incorrectly:\n"
            for qq in category2question[(category, 'false')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "\n + ------------------------------------ + \n"
    return str_summary

def summarize_over_history(history_json_dict, gold_key="python_answer", verbose=True):
    '''
    :param history: a list of dictionaries. Each dictionary corresponds to a run.
    :return: a summary of the results.
    '''
    # augment each line of the dictionary with the iteration number.
    for idx, json_dict in enumerate(history_json_dict):
        for line in json_dict:
            line['iteration'] = idx
    # concatenate the dictionaries.
    json_dict = [line for json_dict in history_json_dict for line in json_dict]
    # a summary of the results.
    str_summary = get_summary_of_results(json_dict, gold_key=gold_key, verbose=verbose)
    return str_summary


def get_acc_lst(json_dict, gold_key="python_answer"):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    for line in json_dict:
        category2correct_count[line['category']].append(line['is_correct'])
    acc_lst = []
    for category in category2correct_count:
        acc = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        acc_lst.append(acc)
    return acc_lst



def solve_and_compare_questions(test_taker_info, agent_info, question_json, gold_answer, outfile_prefix, gold_ans_key='gold_answer'):
    # Use Harmony version if generator is available
    if get_harmony_generator() is not None:
        test_taker_output = _generate_lm_answers_harmony(
            question_json,
            get_harmony_generator(),
            outfile_prefix=outfile_prefix
        )
    else:
        test_taker_output = _generate_lm_answers(question_json,
                             test_taker_info,
                             agent_info,
                             outfile_prefix=outfile_prefix)

    # Handle empty outputs
    if len(test_taker_output) == 0 or len(gold_answer) == 0:
        print("No questions/answers to compare, skipping...")
        return []

    summary_prev_iteration, history_json = fast_compare_answers(gold_answer, test_taker_output,
                                                                agent_info, outfile_prefix=outfile_prefix,
                                                                gold_ans_key=gold_ans_key)

    return history_json


def fast_compare_answers(gold_output, test_taker_output, agent_model_info, outfile_prefix='att1', gold_ans_key='gold_answer'):
    if os.path.exists(f"{outfile_prefix}.compare_answers.json"):
        print('FOUND compare_answers.json')
        json_dict = json.load(open(f"{outfile_prefix}.compare_answers.json", "r"))
        str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
        return str_summary, json_dict

    print("Comparing the answers generated by the python code and the test taker...")
    agent_lm, agent_tokenizer, agent_client = agent_model_info
    print(len(gold_output), len(test_taker_output))
    assert len(gold_output) == len(test_taker_output)
    context_str = """Your goal is to compare the prediction with the gold answer, and judge the correctness of the prediction.
We'd still consider the prediction to be correct if
1. the prediction is semantically the same as the gold answer: formating or different way of reference shouldn't affect correctness. For example, if the gold answer is Jan 21, and the test taker output is 01/21, we would still consider the prediction to be correct. For example, United States and USA refer to the same entity.
2. the prediction refers a broader entity that contains the gold answer. For example, if the gold answer is Beijing, and the test taker output is Asia, we will then consider correctness based on the question.
3. If the question is slightly ambiguous, such that there are multiple correct answers: For example, if the question asks for reasons why something happens, and it could be caused by multiple reasons, we will consider the prediction to be correct if the prediction contains one of the correct answers.

You should output a short and succinct reasoning for the your correctness prediction. Then, you should output delimiter "##" and output "true" if the prediction is correct, and "false" if the prediction is incorrect.
Example Format:
Question: What is 1+1?
pred=2 || gold=2.0
reason: identical numbers ## true
"""
    out_handle = open(f"{outfile_prefix}.compare_answers.jsonl", 'w')
    final_lst = []
    correct_count2 = 0
    for idx, (line_gold, line_pred) in tqdm.tqdm(enumerate(zip(gold_output, test_taker_output))):
        line = {'id': str(idx + 1), 'question': line_gold['question'], 'gold_answer': line_gold[gold_ans_key],
                "test_taker_answer": line_pred['test_taker_response']}
        # add other fields in line_gold to line.
        for k, v in line_gold.items():
            if k not in line:
                line[k] = v
        pred = line_pred['test_taker_response'].strip()
        gold = line_gold[gold_ans_key].strip()
        q_str = f"Question {idx+1}: {line_gold['question']}\npred={pred} || gold={gold}\nreason:"
        context = context_str + q_str
        if get_harmony_generator() is not None:
            response = gen_from_prompt_harmony(
                prompt=q_str,
                temperature=0.0,
                max_tokens=3000,
                developer_content="You are comparing a prediction with a gold answer. Output reasoning then ## and true/false."
            )
        else:
            request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                             echo_prompt=False, temperature=0.0, max_tokens=3000,
                                             process_func=None, service=agent_client,
                                             terminate_by_linebreak='no', verbose=False)
            response = request_result.completions[0].text
        line['reasons'] = response.strip()
        line['is_correct'] = response.strip().split('##')[-1].strip()
        test_taker_line = test_taker_output[idx]
        line['question'] = test_taker_line['question']
        if 'category' in test_taker_line:
            line['category'] = test_taker_line['category']
        else:
            line['category'] = 'None'
        if 'difficulty' in test_taker_line:
            line['difficulty'] = test_taker_line['difficulty']
        if line["is_correct"] == 'true':
            correct_count2 += 1

        print(json.dumps(line), file=out_handle)

        final_lst.append(line)
    json_dict = final_lst
    if len(json_dict) == 0:
        print("No questions to compare, accuracy: N/A")
        accuracy = 0.0
    else:
        accuracy = correct_count2 / len(json_dict)
        print("accuracy: ", accuracy)
        assert len(json_dict) == len(test_taker_output)
    out_handle.close()


    with open(f"{outfile_prefix}.compare_answers.json", 'w') as out_handle:
        json.dump(json_dict, out_handle, indent=2)

    str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
    return str_summary, json_dict


VERIFICATION_PROMPT = """You are answering a knowledge question. Provide ONLY the direct answer in a few words.

Rules:
- Answer with the specific answer requested (person, location, date, event, organization, number, work, concept, or time period)
- Do NOT include explanations, reasoning, or additional context
- If the answer is a person, give their name
- If the answer is a date, give the date
- If the answer is a place, give the place name

Question: {question}

Answer (just the answer, nothing else):"""


def verify_qa_pair(question, gold_answer, agent_info, num_samples=10, temperature=0.7):
    """
    Verify a QA pair by sampling multiple answers and computing accuracy.

    Args:
        question: The question to verify
        gold_answer: The expected correct answer
        agent_info: Tuple of (model, tokenizer, client)
        num_samples: Number of answer samples to generate
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        Dict with verification results including accuracy and sampled answers
    """
    agent_lm, agent_tokenizer, agent_client = agent_info

    prompt = VERIFICATION_PROMPT.format(question=question)

    sampled_answers = []
    correct_count = 0

    for i in range(num_samples):
        try:
            if get_harmony_generator() is not None:
                raw_answer = gen_from_prompt_harmony(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=1024,
                    developer_content="You are answering a knowledge question. Provide ONLY the direct answer in a few words."
                ).strip()
            else:
                request_result = gen_from_prompt(
                    model=agent_lm,
                    tokenizer=agent_tokenizer,
                    prompt=[prompt],
                    echo_prompt=False,
                    temperature=temperature,
                    max_tokens=1024,
                    process_func=None,
                    service=agent_client,
                    terminate_by_linebreak='no',
                    verbose=False
                )
                raw_answer = request_result.completions[0].text.strip()
            # Clean up non-breaking spaces and other unicode artifacts
            raw_answer = raw_answer.replace('\xa0', ' ').replace('\u202f', ' ')

            # Find first non-empty, non-reasoning, non-garbage line
            sampled_answer = ""
            for line in raw_answer.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Skip garbage patterns (model confusion artifacts)
                if any(p in line for p in ['**?**', '[\xa0', '[ \xa0', '[  ]', '...assistant', 'assistantfinal']):
                    continue
                # Skip lines that are just "The" or "The ..." with nothing meaningful
                if line.lower().startswith('the') and len(line) < 10:
                    continue
                # Skip lines that are mostly punctuation/special chars
                alnum_chars = sum(1 for c in line if c.isalnum())
                if len(line) > 0 and alnum_chars / len(line) < 0.3:
                    continue
                # Skip lines that are reasoning/meta-commentary
                skip_patterns = [
                    'the user asks', 'we need to', 'the question',
                    'let me', 'i need to', 'so they want', 'so the answer',
                    'the correct answer is', 'we need answer', 'that sounds like',
                    'actually', 'maybe the', 'but question'
                ]
                line_lower = line.lower()
                if any(pat in line_lower for pat in skip_patterns):
                    # Check if line contains answer after the pattern
                    for pat in ['the correct answer is', 'answer:', 'is:', 'so answer:']:
                        if pat in line_lower:
                            idx = line_lower.find(pat) + len(pat)
                            candidate = line[idx:].strip().strip('.,')
                            # Clean up model artifacts from candidate
                            for artifact in ['assistantfinal', 'assistant', '\xa0']:
                                candidate = candidate.replace(artifact, ' ').strip()
                            if candidate and len(candidate) < 100 and len(candidate) > 1:
                                sampled_answer = candidate
                                break
                    if sampled_answer:
                        break
                    continue
                # Use this line as the answer
                sampled_answer = line
                break
            # Clean up model artifacts
            for artifact in ['assistantfinal', 'assistant', '\xa0', '\u202f']:
                sampled_answer = sampled_answer.replace(artifact, ' ')
            sampled_answer = ' '.join(sampled_answer.split())  # Normalize whitespace
            sampled_answer = sampled_answer.strip('"\'')
            # Remove common answer prefixes like "The answer is..."
            for prefix in ['The answer is ', 'Answer: ', 'It is ', 'It was ', 'Answer is ', 'The answer:']:
                if sampled_answer.lower().startswith(prefix.lower()):
                    sampled_answer = sampled_answer[len(prefix):].strip()
            sampled_answers.append(sampled_answer)

            # Check if answer is correct (heuristics + LLM judge fallback)
            if is_answer_correct(sampled_answer, gold_answer) or _llm_judge_answer(sampled_answer, gold_answer, question):
                correct_count += 1

        except Exception as e:
            print(f"Error sampling answer {i+1}: {e}", flush=True)
            sampled_answers.append("")

    accuracy = correct_count / num_samples if num_samples > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'num_samples': num_samples,
        'sampled_answers': sampled_answers,
        'gold_answer': gold_answer
    }


def _normalize_whitespace(text):
    """Collapse all Unicode whitespace (including \\u202f, \\u00a0, etc.) into regular spaces."""
    import re
    return re.sub(r'\s+', ' ', text).strip()


def is_answer_correct(predicted, gold):
    """
    Check if predicted answer matches gold answer.
    Uses flexible matching: case-insensitive, partial match allowed.
    """
    pred_lower = _normalize_whitespace(predicted).lower()
    gold_lower = _normalize_whitespace(gold).lower()

    # Exact match
    if pred_lower == gold_lower:
        return True

    # Gold answer contained in prediction
    if gold_lower in pred_lower:
        return True

    # Prediction contained in gold answer (for longer gold answers)
    if pred_lower in gold_lower and len(pred_lower) > 3:
        return True

    # Handle common variations
    pred_normalized = pred_lower.replace('-', ' ').replace('_', ' ')
    gold_normalized = gold_lower.replace('-', ' ').replace('_', ' ')

    if pred_normalized == gold_normalized:
        return True
    if gold_normalized in pred_normalized:
        return True

    return False


V2_DEVELOPER_CONTENT = """You are a deep research agent. Answer the given question by interacting with the Wikipedia browser tool. Perform reasoning and use the tool step by step, in an interleaved manner. Call the browser tool if and only if you think it will help generating the correct answer. Your response should be in the following format:
Explanation: {{your explanation for your final answer.}}
Exact Answer: \\boxed{{{{your succinct, final answer}}}}
Confidence: {{your confidence score between 0% and 100% for your answer}}""".strip()

V3_VERIFICATION_PROMPT = """You are answering a knowledge question using the provided reference documents. Provide ONLY the direct answer in a few words.

Reference Documents:
{documents}

Rules:
- Answer ONLY based on information in the reference documents above
- Answer with the specific answer requested (person, location, date, event, organization, number, work, concept, or time period)
- Do NOT include explanations, reasoning, or additional context
- If the answer is a person, give their name
- If the answer is a date, give the date
- If the answer is a place, give the place name

Question: {question}

Answer (just the answer, nothing else):"""


def _extract_text_from_message(msg):
    """Extract plain text from a Message's content."""
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


def _strip_latex_markup(text):
    r"""Strip LaTeX markup like \text{...}, \textbf{...}, \mathrm{...} etc."""
    return re.sub(r'\\(?:text|textbf|textit|textrm|mathrm|mathbf)\{([^}]*)\}', r'\1', text)


def _extract_boxed_content(text):
    r"""Extract content from \boxed{...} handling nested braces correctly."""
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
    return text[start:].rstrip('}').strip()


def _try_parse_answer(text):
    """Try to parse an answer from text using Exact Answer / boxed patterns."""
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text)
    if match:
        answer = match.group(1).strip()
        boxed = _extract_boxed_content(answer)
        if boxed is not None:
            answer = boxed.strip()
        return _strip_latex_markup(answer)

    boxed = _extract_boxed_content(text)
    if boxed is not None:
        return _strip_latex_markup(boxed.strip())

    return None


def _extract_answer_from_agentic_response(messages):
    """
    Extract the predicted answer from an agentic response message list.

    Only considers "final"-channel assistant messages.  If none exists the
    response is discarded (returns empty string).
    """
    for msg in reversed(messages):
        if msg.author.role != Role.ASSISTANT:
            continue
        if getattr(msg, 'channel', None) != 'final':
            continue
        text = _extract_text_from_message(msg)
        if not text.strip():
            continue
        answer = _try_parse_answer(text)
        if answer is not None:
            return answer

    return ""


def _llm_judge_answer(predicted, gold, question):
    """Use LLM as judge to check semantic equivalence when heuristics fail."""
    generator = get_harmony_generator()
    if generator is None:
        return False

    prompt = (
        f"Question: {question}\n"
        f"Predicted: {predicted}\n"
        f"Gold: {gold}\n"
    )
    developer_content = (
        "Compare the predicted answer with the gold answer for the given question. "
        "The prediction is correct if it is semantically equivalent to the gold answer — "
        "formatting differences, abbreviations, or different phrasing should not affect "
        'correctness (e.g. "Jan 21" = "January 21", "USA" = "United States"). '
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
            print(f"    LLM judge: pred={predicted!r} gold={gold!r} -> {result} (verdict={verdict!r})", flush=True)
            return result
        print(f"    LLM judge: no '##' delimiter in response: {response!r}", flush=True)
        return False
    except Exception as e:
        print(f"    LLM judge error: {e}", flush=True)
        return False


def _load_grounding_docs(outfile_prefix):
    """Load grounding documents for V3 verification.

    Prefers Wikipedia articles from articles.json; falls back to triple descriptions.
    """
    # Try Wikipedia articles first
    articles_path = f"{outfile_prefix}.articles.json"
    if os.path.exists(articles_path):
        with open(articles_path) as f:
            articles = json.load(f)
        if articles and any(len(a.get('paragraph', '')) > 200 for a in articles):
            return articles

    # Fall back to triple descriptions
    path = f"{outfile_prefix}.triples.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        chains = data.get('chains', [])
        docs = []
        for i, chain in enumerate(chains, 1):
            docs.append({
                'title': chain.get('path_description', 'Triple chain'),
                'paragraph': chain.get('path_description', ''),
                'chain_indices': [i],
            })
        return docs
    return None


def _collect_relevant_entities(qa_dict, chains_data):
    """Collect all entity name strings relevant to a question from chains data."""
    entities = set()

    # Seed entity label
    seed_label = chains_data.get('seed_entity', {}).get('label', '')
    if seed_label:
        entities.add(seed_label)

    # Gold answer
    gold = qa_dict.get('gold_answer', '')
    if gold:
        entities.add(gold)

    # Entities from chains referenced in chain_used
    chain_used_str = qa_dict.get('chain_used', '')
    if chain_used_str:
        try:
            used_indices = {int(x.strip()) for x in chain_used_str.split(',')}
        except (ValueError, AttributeError):
            used_indices = set()

        for chain in chains_data.get('chains', []):
            if chain.get('chain_index') in used_indices:
                for step in chain.get('steps', []):
                    for key in ('entity_label', 'value_label', 'entity', 'value',
                                'property_label'):
                        val = step.get(key, '')
                        if val and len(val) > 1:
                            entities.add(val)

    # Entities from extracted_facts
    for fact in qa_dict.get('extracted_facts', []):
        if isinstance(fact, dict):
            for key in ('entity', 'value', 'entity_label', 'value_label'):
                val = fact.get(key, '')
                if val and len(val) > 1:
                    entities.add(val)
        elif isinstance(fact, str) and len(fact) > 1:
            entities.add(fact)

    return entities


def _paragraph_mentions_entity(paragraph, entity_patterns):
    """Check if a paragraph mentions any entity (case-insensitive).

    entity_patterns: set of lowered entity strings.
    Short entities (<=3 chars) use word-boundary regex to avoid false matches.
    """
    para_lower = paragraph.lower()
    for pat in entity_patterns:
        if len(pat) <= 3:
            if re.search(r'\b' + re.escape(pat) + r'\b', para_lower):
                return True
        else:
            if pat in para_lower:
                return True
    return False


def _sentence_mentions_patterns(sent_lower, patterns):
    """Check if a lowered sentence mentions any pattern from the set."""
    for pat in patterns:
        if len(pat) <= 3:
            if re.search(r'\b' + re.escape(pat) + r'\b', sent_lower):
                return True
        else:
            if pat in sent_lower:
                return True
    return False


def _markup_sentences(paragraph, question_patterns, answer_patterns):
    """Wrap sentences with Q/A/QA RELEVANT tags based on which entity patterns match.

    - <<QA_RELEVANT>>: sentence mentions both question and answer entities
    - <<Q_RELEVANT>>: sentence mentions only question entities
    - <<A_RELEVANT>>: sentence mentions only answer entities
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
    marked = []
    for sent in sentences:
        sent_lower = sent.lower()
        has_q = _sentence_mentions_patterns(sent_lower, question_patterns)
        has_a = _sentence_mentions_patterns(sent_lower, answer_patterns)
        if has_q and has_a:
            marked.append(f"<<QA_RELEVANT>>{sent}<</QA_RELEVANT>>")
        elif has_q:
            marked.append(f"<<Q_RELEVANT>>{sent}<</Q_RELEVANT>>")
        elif has_a:
            marked.append(f"<<A_RELEVANT>>{sent}<</A_RELEVANT>>")
        else:
            marked.append(sent)
    return " ".join(marked)


def build_grounding_articles_for_qa(qa_dict, chains_data):
    """Build trimmed, marked-up grounding articles for a single QA pair.

    Filters articles to those relevant to the question's chains, keeps only
    paragraphs mentioning relevant entities, and adds semantic markup:
    <<Q_RELEVANT>> (question entities), <<A_RELEVANT>> (answer entities),
    <<QA_RELEVANT>> (both).
    Returns list of dicts with 'qid', 'title', 'paragraph', 'chain_indices',
    'original_length', 'trimmed_length'.
    """
    MAX_CHARS_PER_ARTICLE = 5000

    entities = _collect_relevant_entities(qa_dict, chains_data)

    # Collect answer entity strings (gold_answer + normalized variants)
    answer_entities = set()
    gold = qa_dict.get('gold_answer', '')
    if gold:
        answer_entities.add(gold)

    def _make_patterns(raw_entities):
        patterns = set()
        for e in raw_entities:
            lowered = e.lower().strip()
            if lowered:
                patterns.add(lowered)
                normalized = re.sub(r'[.,\-]', ' ', lowered).strip()
                normalized = ' '.join(normalized.split())
                if normalized and normalized != lowered:
                    patterns.add(normalized)
        return patterns

    answer_patterns = _make_patterns(answer_entities)
    question_patterns = _make_patterns(entities - answer_entities)
    # Union for paragraph-level filtering (same behaviour as before)
    entity_patterns = question_patterns | answer_patterns

    if not entity_patterns:
        return []

    # Filter articles to those whose chain_indices overlap with chain_used
    chain_used_str = qa_dict.get('chain_used', '')
    used_indices = set()
    if chain_used_str:
        try:
            used_indices = {int(x.strip()) for x in chain_used_str.split(',')}
        except (ValueError, AttributeError):
            pass

    articles = chains_data.get('articles', [])
    result = []

    for article in articles:
        # Filter by chain_indices if available
        art_chains = set(article.get('chain_indices', []))
        if used_indices and art_chains and not art_chains.intersection(used_indices):
            continue

        original_text = article.get('paragraph', '')
        original_length = len(original_text)
        title = article.get('title', 'Untitled')
        qid = article.get('qid', '')

        # Split into paragraphs
        paragraphs = original_text.split('\n')

        # Extract relationship/property keywords from extracted_facts for bonus scoring
        relationship_keywords = set()
        for fact in qa_dict.get('extracted_facts', []):
            if isinstance(fact, dict):
                prop = fact.get('property', '')
                if prop:
                    # Add the property label and its individual words (e.g., "official language" -> {"official language", "official", "language"})
                    prop_lower = prop.lower().strip()
                    relationship_keywords.add(prop_lower)
                    for word in prop_lower.split():
                        if len(word) > 3:  # Skip short words like "of", "in", "has"
                            relationship_keywords.add(word)

        # Score and filter paragraphs by entity mention count + relationship keyword bonus
        scored_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if _paragraph_mentions_entity(para, entity_patterns):
                # Count mentions for ranking
                para_lower = para.lower()
                mention_count = sum(
                    1 for pat in entity_patterns
                    if (re.search(r'\b' + re.escape(pat) + r'\b', para_lower)
                        if len(pat) <= 3 else pat in para_lower)
                )
                # Bonus: +3 per relationship keyword found in paragraphs that mention entities
                if relationship_keywords:
                    rel_bonus = sum(
                        3 for kw in relationship_keywords
                        if kw in para_lower
                    )
                    mention_count += rel_bonus
                scored_paragraphs.append((mention_count, para))

        # Fallback: keep lead paragraph if nothing matched
        if not scored_paragraphs and paragraphs:
            lead = paragraphs[0].strip()
            if lead:
                scored_paragraphs.append((0, lead))

        if not scored_paragraphs:
            continue

        # Sort by mention count descending, cap at MAX_CHARS_PER_ARTICLE
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        kept_paragraphs = []
        total_chars = 0
        for _count, para in scored_paragraphs:
            if total_chars + len(para) > MAX_CHARS_PER_ARTICLE and kept_paragraphs:
                break
            # Apply markup
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


def verify_qa_batch(qa_pairs, agent_info, num_samples=10, temperature=0.7, min_accuracy=0.0, max_accuracy=0.5):
    """
    Verify a batch of QA pairs and filter by accuracy threshold.

    For challenging benchmarks, we want questions the model struggles with,
    so we keep questions with accuracy <= max_accuracy (hard questions).

    Args:
        qa_pairs: List of QA dicts with 'question' and 'gold_answer' keys
        agent_info: Tuple of (model, tokenizer, client)
        num_samples: Number of samples per question
        temperature: Sampling temperature
        min_accuracy: Minimum accuracy threshold (default 0.0, no minimum)
        max_accuracy: Maximum accuracy threshold to keep hard questions (default 0.5)

    Returns:
        Tuple of (verified_qa_pairs, all_qa_with_verification)
    """
    verified_pairs = []
    all_pairs_with_verification = []

    print(f"Verifying {len(qa_pairs)} QA pairs with {num_samples} samples each...", flush=True)
    print(f"Keeping questions with accuracy between {min_accuracy:.0%} and {max_accuracy:.0%}", flush=True)

    for idx, qa in enumerate(tqdm.tqdm(qa_pairs, desc="Verifying QA pairs")):
        if 'question' not in qa or 'gold_answer' not in qa:
            continue

        verification = verify_qa_pair(
            qa['question'],
            qa['gold_answer'],
            agent_info,
            num_samples=num_samples,
            temperature=temperature
        )

        # Add verification results to QA pair
        qa_with_verification = copy.deepcopy(qa)
        qa_with_verification['verification_accuracy'] = verification['accuracy']
        qa_with_verification['verification_correct'] = verification['correct_count']
        qa_with_verification['verification_samples'] = verification['num_samples']
        qa_with_verification['sampled_answers'] = verification['sampled_answers']

        all_pairs_with_verification.append(qa_with_verification)

        # Keep if within accuracy range (for hard questions: min_accuracy <= acc <= max_accuracy)
        if min_accuracy <= verification['accuracy'] <= max_accuracy:
            verified_pairs.append(qa_with_verification)
            print(f"  [{idx+1}] KEEP (hard) - Accuracy: {verification['accuracy']:.1%} - Q: {qa['question'][:50]}...", flush=True)
        else:
            print(f"  [{idx+1}] SKIP (too easy) - Accuracy: {verification['accuracy']:.1%} - Q: {qa['question'][:50]}...", flush=True)

    print(f"\nVerification complete: {len(verified_pairs)}/{len(qa_pairs)} kept as challenging (accuracy <= {max_accuracy:.0%})", flush=True)

    return verified_pairs, all_pairs_with_verification


def verify_qa_pair_with_grounding(question, gold_answer, grounding_docs, agent_info,
                                   num_samples=5, temperature=0.7):
    """
    Verify a QA pair by providing grounding documents in the prompt (V3).

    Tests whether the question is answerable given the source material.
    """
    agent_lm, agent_tokenizer, agent_client = agent_info

    # Format grounding documents with adaptive truncation
    total_chars = sum(len(doc.get('paragraph', doc.get('content', doc.get('text', '')))) for doc in grounding_docs)
    max_per_doc = 3000 if total_chars > 20000 else 8000

    doc_texts = []
    for doc in grounding_docs:
        title = doc.get('title', 'Untitled')
        paragraph = doc.get('paragraph', doc.get('content', doc.get('text', '')))
        doc_texts.append(f"### {title}\n{paragraph[:max_per_doc]}")
    documents_str = "\n\n".join(doc_texts)

    prompt = V3_VERIFICATION_PROMPT.replace("{documents}", documents_str).replace("{question}", question)

    sampled_answers = []
    correct_count = 0

    for i in range(num_samples):
        try:
            if get_harmony_generator() is not None:
                raw_answer = gen_from_prompt_harmony(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=1024,
                    developer_content="You are answering a knowledge question using provided reference documents. Provide ONLY the direct answer in a few words."
                ).strip()
            else:
                request_result = gen_from_prompt(
                    model=agent_lm,
                    tokenizer=agent_tokenizer,
                    prompt=[prompt],
                    echo_prompt=False,
                    temperature=temperature,
                    max_tokens=1024,
                    process_func=None,
                    service=agent_client,
                    terminate_by_linebreak='no',
                    verbose=False
                )
                raw_answer = request_result.completions[0].text.strip()

            # Clean up non-breaking spaces and other unicode artifacts
            raw_answer = raw_answer.replace('\xa0', ' ').replace('\u202f', ' ')

            # Find first non-empty, non-reasoning, non-garbage line
            sampled_answer = ""
            for line in raw_answer.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if any(p in line for p in ['**?**', '[\xa0', '[ \xa0', '[  ]', '...assistant', 'assistantfinal']):
                    continue
                if line.lower().startswith('the') and len(line) < 10:
                    continue
                alnum_chars = sum(1 for c in line if c.isalnum())
                if len(line) > 0 and alnum_chars / len(line) < 0.3:
                    continue
                skip_patterns = [
                    'the user asks', 'we need to', 'the question',
                    'let me', 'i need to', 'so they want', 'so the answer',
                    'the correct answer is', 'we need answer', 'that sounds like',
                    'actually', 'maybe the', 'but question'
                ]
                line_lower = line.lower()
                if any(pat in line_lower for pat in skip_patterns):
                    for pat in ['the correct answer is', 'answer:', 'is:', 'so answer:']:
                        if pat in line_lower:
                            idx_pat = line_lower.find(pat) + len(pat)
                            candidate = line[idx_pat:].strip().strip('.,')
                            for artifact in ['assistantfinal', 'assistant', '\xa0']:
                                candidate = candidate.replace(artifact, ' ').strip()
                            if candidate and len(candidate) < 100 and len(candidate) > 1:
                                sampled_answer = candidate
                                break
                    if sampled_answer:
                        break
                    continue
                sampled_answer = line
                break

            # Clean up model artifacts
            for artifact in ['assistantfinal', 'assistant', '\xa0', '\u202f']:
                sampled_answer = sampled_answer.replace(artifact, ' ')
            sampled_answer = ' '.join(sampled_answer.split())
            sampled_answer = sampled_answer.strip('"\'')
            for prefix in ['The answer is ', 'Answer: ', 'It is ', 'It was ', 'Answer is ', 'The answer:']:
                if sampled_answer.lower().startswith(prefix.lower()):
                    sampled_answer = sampled_answer[len(prefix):].strip()
            sampled_answers.append(sampled_answer)

            if is_answer_correct(sampled_answer, gold_answer) or _llm_judge_answer(sampled_answer, gold_answer, question):
                correct_count += 1

        except Exception as e:
            print(f"Error sampling V3 answer {i+1}: {e}", flush=True)
            sampled_answers.append("")

    accuracy = correct_count / num_samples if num_samples > 0 else 0.0

    return {
        'v3_accuracy': accuracy,
        'v3_correct_count': correct_count,
        'v3_num_samples': num_samples,
        'v3_sampled_answers': sampled_answers,
        'gold_answer': gold_answer
    }


def verify_qa_batch_with_grounding(qa_pairs, grounding_docs, agent_info,
                                    num_samples=5, temperature=0.7,
                                    min_accuracy=0.0, max_accuracy=1.0,
                                    correctness_threshold=0.7,
                                    v1_num_samples=10, v1_temperature=0.7,
                                    v2_num_samples=3, v2_max_iterations=200):
    """
    Verify a batch of QA pairs using grounding documents (V3).

    Annotates every QA pair with V3 scores and a v3_correct field ("yes"/"no").
    Also computes V1 and V2 scores if not already present on each QA dict.
    """
    filtered_pairs = []
    all_pairs = []

    print(f"V3 Grounding Verification: {len(qa_pairs)} QA pairs with {num_samples} samples each...", flush=True)
    print(f"Keeping questions with V3 accuracy between {min_accuracy:.0%} and {max_accuracy:.0%}", flush=True)

    for idx, qa in enumerate(tqdm.tqdm(qa_pairs, desc="V3 Grounding Verification")):
        if 'question' not in qa or 'gold_answer' not in qa:
            continue

        qa_annotated = copy.deepcopy(qa)

        # --- V1: run if scores not already present ---
        if 'verification_accuracy' not in qa_annotated:
            v1_result = verify_qa_pair(
                qa_annotated['question'],
                qa_annotated['gold_answer'],
                agent_info,
                num_samples=v1_num_samples,
                temperature=v1_temperature,
            )
            qa_annotated['verification_accuracy'] = v1_result['accuracy']
            qa_annotated['verification_correct'] = v1_result['correct_count']
            qa_annotated['verification_samples'] = v1_result['num_samples']
            qa_annotated['sampled_answers'] = v1_result['sampled_answers']

        # --- V2: run if scores not already present and browser available ---
        if 'v2_accuracy' not in qa_annotated and BROWSER_TOOL_AVAILABLE:
            try:
                v2_result = verify_qa_pair_with_browser(
                    qa_annotated['question'],
                    qa_annotated['gold_answer'],
                    num_samples=v2_num_samples,
                    max_iterations=v2_max_iterations,
                )
                qa_annotated['v2_accuracy'] = v2_result['v2_accuracy']
                qa_annotated['v2_correct_count'] = v2_result['v2_correct_count']
                qa_annotated['v2_num_samples'] = v2_result['v2_num_samples']
                qa_annotated['v2_sampled_answers'] = v2_result['v2_sampled_answers']
            except Exception as e:
                print(f"  [{idx+1}] V2 skipped due to error: {e}", flush=True)

        # --- V3: use pre-built grounding articles if available, else filter by chain ---
        if qa_annotated.get('grounding_articles'):
            qa_docs = qa_annotated['grounding_articles']
        else:
            # Fallback: filter grounding docs to chains used by this question
            qa_chain_used = qa_annotated.get('chain_used', '')
            if qa_chain_used:
                try:
                    used_indices = {int(x.strip()) for x in qa_chain_used.split(',')}
                    relevant_docs = [
                        doc for doc in grounding_docs
                        if any(ci in used_indices for ci in doc.get('chain_indices', []))
                    ]
                    qa_docs = relevant_docs if relevant_docs else grounding_docs
                except (ValueError, AttributeError):
                    qa_docs = grounding_docs
            else:
                qa_docs = grounding_docs

        # --- V3: grounding-document verification ---
        verification = verify_qa_pair_with_grounding(
            qa_annotated['question'],
            qa_annotated['gold_answer'],
            qa_docs,
            agent_info,
            num_samples=num_samples,
            temperature=temperature
        )

        qa_annotated['v3_accuracy'] = verification['v3_accuracy']
        qa_annotated['v3_correct_count'] = verification['v3_correct_count']
        qa_annotated['v3_num_samples'] = verification['v3_num_samples']
        qa_annotated['v3_sampled_answers'] = verification['v3_sampled_answers']
        qa_annotated['v3_correct'] = "yes" if verification['v3_accuracy'] >= correctness_threshold else "no"

        all_pairs.append(qa_annotated)

        if min_accuracy <= verification['v3_accuracy'] <= max_accuracy:
            filtered_pairs.append(qa_annotated)
            print(f"  [{idx+1}] KEEP - V3 Accuracy: {verification['v3_accuracy']:.1%} v3_correct={qa_annotated['v3_correct']} - Q: {qa['question'][:50]}...", flush=True)
        else:
            print(f"  [{idx+1}] SKIP - V3 Accuracy: {verification['v3_accuracy']:.1%} v3_correct={qa_annotated['v3_correct']} - Q: {qa['question'][:50]}...", flush=True)

    print(f"\nV3 Verification complete: {len(filtered_pairs)}/{len(all_pairs)} passed filter [{min_accuracy:.0%}, {max_accuracy:.0%}]", flush=True)

    return filtered_pairs, all_pairs


def verify_qa_pair_with_browser(question, gold_answer, num_samples=3,
                                 temperature=1.0, max_iterations=200):
    """
    Verify a QA pair using the agentic Wikipedia browser tool (V2).

    For each sample, creates a fresh Wikipedia browser tool, builds messages with tool
    config, runs the agentic loop, and checks the extracted answer against gold.
    """
    generator = get_harmony_generator()
    if generator is None:
        raise RuntimeError("Harmony generator not initialized for V2 verification")
    if not BROWSER_TOOL_AVAILABLE:
        raise RuntimeError("MultiSourceKnowledgeBrowserTool not available")

    sampled_answers = []
    correct_count = 0

    for i in range(num_samples):
        backend = None
        try:
            # Fresh tool per sample
            backend = MultiSourceKnowledgeBackend("en", primary_source="wikimedia")
            wiki_tool = MultiSourceKnowledgeBrowserTool(backend=backend)
            tool_config = wiki_tool.tool_config

            # Build system content with tool config
            system_content = (
                SystemContent.new()
                .with_reasoning_effort(ReasoningEffort.LOW)
                .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                .with_tools(tool_config)
            )

            messages = [
                Message.from_role_and_content(Role.SYSTEM, system_content),
                Message.from_role_and_content(Role.DEVELOPER, V2_DEVELOPER_CONTENT),
                Message.from_role_and_content(Role.USER, f"Question: {question}"),
            ]

            # Define async tool handler
            async def _tool_handler(msg, _tool=wiki_tool):
                results = []
                async for m in _tool.process(msg):
                    results.append(m)
                return results

            # Run agentic loop
            result_messages = generator.generate_agentic_response_sync(
                messages,
                _tool_handler,
                tool_prefix="browser.",
                max_iterations=max_iterations,
                temperature=temperature,
            )

            # Extract answer
            answer = _extract_answer_from_agentic_response(result_messages)
            answer = answer.replace('\xa0', ' ').replace('\u202f', ' ')
            answer = ' '.join(answer.split())
            sampled_answers.append(answer)

            if is_answer_correct(answer, gold_answer) or _llm_judge_answer(answer, gold_answer, question):
                correct_count += 1

            print(f"    V2 sample {i+1}/{num_samples}: '{answer}' (gold: '{gold_answer}')", flush=True)

        except Exception as e:
            print(f"    V2 sample {i+1}/{num_samples} error: {e}", flush=True)
            sampled_answers.append("")

        finally:
            if backend is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        backend.close(), generator._loop
                    ).result(timeout=5)
                except Exception:
                    pass

    accuracy = correct_count / num_samples if num_samples > 0 else 0.0

    return {
        'v2_accuracy': accuracy,
        'v2_correct_count': correct_count,
        'v2_num_samples': num_samples,
        'v2_sampled_answers': sampled_answers,
        'gold_answer': gold_answer,
    }


def verify_qa_batch_with_browser(qa_pairs, mode="keep_unanswerable",
                                  num_samples=3, temperature=1.0,
                                  min_accuracy=0.0, max_accuracy=0.3,
                                  max_iterations=200):
    """
    Run V2 browser-based verification on a batch of QA pairs and filter.
    """
    filtered_pairs = []
    all_pairs_with_v2 = []

    print(f"\n=== VERIFICATION 2 (KG Browser Tool) ===", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Verifying {len(qa_pairs)} QA pairs with {num_samples} browser samples each...", flush=True)
    print(f"Keeping questions with V2 accuracy in [{min_accuracy:.0%}, {max_accuracy:.0%}]", flush=True)

    for idx, qa in enumerate(tqdm.tqdm(qa_pairs, desc="V2 Browser Verification")):
        if 'question' not in qa or 'gold_answer' not in qa:
            continue

        v2_result = verify_qa_pair_with_browser(
            qa['question'],
            qa['gold_answer'],
            num_samples=num_samples,
            temperature=temperature,
            max_iterations=max_iterations,
        )

        qa_with_v2 = copy.deepcopy(qa)
        qa_with_v2['v2_accuracy'] = v2_result['v2_accuracy']
        qa_with_v2['v2_correct_count'] = v2_result['v2_correct_count']
        qa_with_v2['v2_num_samples'] = v2_result['v2_num_samples']
        qa_with_v2['v2_sampled_answers'] = v2_result['v2_sampled_answers']

        all_pairs_with_v2.append(qa_with_v2)

        in_range = min_accuracy <= v2_result['v2_accuracy'] <= max_accuracy

        if in_range:
            filtered_pairs.append(qa_with_v2)
            label = "KEEP"
        else:
            label = "SKIP"

        print(
            f"  [{idx+1}] {label} - V2 Accuracy: {v2_result['v2_accuracy']:.1%} "
            f"- Q: {qa['question'][:50]}...",
            flush=True,
        )

    print(
        f"\nV2 complete: {len(filtered_pairs)}/{len(qa_pairs)} kept "
        f"(mode={mode}, range=[{min_accuracy:.0%}, {max_accuracy:.0%}])",
        flush=True,
    )

    return filtered_pairs, all_pairs_with_v2


# ===== Wikidata-Specific Functions =====

def fetch_related_entities(seed_category, num_chains=20):
    """
    Fetch multi-hop triple chains from Wikidata for a seed category.
    Replaces fetch_related_articles() from the wiki pipeline.

    Args:
        seed_category: Starting category/topic to search Wikidata
        num_chains: Number of relationship chains to fetch

    Returns:
        Dict with 'seed_entity', 'chains', and 'formatted_text'
    """
    # Search for the seed entity on Wikidata
    search_results = search_wikidata_entities(seed_category)

    if not search_results:
        print(f"No Wikidata entities found for '{seed_category}'", flush=True)
        return {'seed_entity': None, 'chains': [], 'formatted_text': ''}

    # Use the top result
    seed_entity = search_results[0]
    entity_id = seed_entity['id']
    print(f"Found Wikidata entity: {seed_entity['label']} ({entity_id}) - {seed_entity.get('description', '')}", flush=True)

    # Fetch multi-hop triples
    chains = fetch_multihop_triples(entity_id, num_hops=2, limit=num_chains)

    if not chains:
        print(f"No multi-hop chains found for {seed_entity['label']} ({entity_id})", flush=True)
        return {'seed_entity': seed_entity, 'chains': [], 'formatted_text': ''}

    # Format chains as human-readable text for LLM
    formatted_lines = []
    for i, chain in enumerate(chains, 1):
        formatted_lines.append(f"Chain {i}: {chain['path_description']}")

    formatted_text = "\n".join(formatted_lines)

    print(f"Fetched {len(chains)} triple chains for '{seed_entity['label']}'", flush=True)

    # Fetch Wikipedia articles for chain entities
    articles = fetch_wikipedia_for_entities(chains)
    print(f"Fetched {len(articles)} Wikipedia articles for chain entities", flush=True)

    return {
        'seed_entity': seed_entity,
        'chains': chains,
        'formatted_text': formatted_text,
        'articles': articles,
    }


def gen_multihop_qa_from_triples(chains_data, agent_info, num_questions=10):
    """
    Generate multi-hop QA pairs from Wikidata triple chains using seed-centric approach.

    All chains radiate from a single seed entity. The question describes the seed
    using clue facts from 3+ chains (without naming it), then asks for a non-seed
    answer entity from one specific chain.

    Three-phase flow per question:
      Phase 0: Pick a non-seed answer entity from the chains
      Phase 1: Extract clue facts about the seed entity (one per chain)
      Phase 2: Compose question using clues to describe seed, asking for the answer
      Phase 3: Programmatic validation (fact IDs exist, span 3+ chains, answer in facts)

    Args:
        chains_data: Dict from fetch_related_entities() with 'seed_entity', 'chains', 'formatted_text'
        agent_info: Tuple of (model, tokenizer, client)
        num_questions: Number of QA pairs to generate

    Returns:
        List of QA pairs with multi-hop reasoning chains
    """
    formatted_text = chains_data.get('formatted_text', '')
    chains = chains_data.get('chains', [])
    seed_entity_data = chains_data.get('seed_entity', {})
    seed_label = seed_entity_data.get('label', '') if seed_entity_data else ''

    if not chains or not formatted_text:
        print("No triple chains available for QA generation", flush=True)
        return []

    if len(chains) < 3:
        print(f"Warning: Need at least 3 chains for multi-hop QA, got {len(chains)}", flush=True)
        return []

    if not seed_label:
        print("Warning: No seed entity label available", flush=True)
        return []

    # Check if Harmony generator is available
    use_harmony = get_harmony_generator() is not None

    if not use_harmony:
        agent_lm, agent_tokenizer, agent_client = agent_info

        def _gen(prompt_text, temperature=0.7, max_tokens=4096):
            """Helper to call gen_from_prompt and return text."""
            result = gen_from_prompt(
                model=agent_lm, tokenizer=agent_tokenizer, prompt=[prompt_text],
                echo_prompt=False, temperature=temperature, max_tokens=max_tokens,
                process_func=None, service=agent_client,
                terminate_by_linebreak='no', verbose=False,
            )
            return result.completions[0].text

    # Number of chains to sample per question attempt
    chains_per_question = min(7, len(chains))

    qa_pairs = []
    used_answers = set()  # Track previously selected answers for diversity
    used_chain_sets = []  # Track chain index sets to avoid high-overlap samples
    chain_use_counts = [0] * len(chains)  # Track per-chain usage for weighted sampling
    generated_questions = []  # Track generated question texts for diversity

    def _chain_overlap_too_high(candidate_set, used_sets, threshold=0.5):
        for prev in used_sets:
            overlap = len(candidate_set & prev) / max(len(candidate_set), 1)
            if overlap > threshold:
                return True
        return False

    for q_idx in range(num_questions):
        print(f"  Generating question {q_idx + 1}/{num_questions}...", flush=True)

        max_retries = 8
        tried_this_question = set()  # Track answers tried for this question to avoid repeats
        for retry in range(max_retries):
            try:
                # Sample chains with weights favoring less-used chains
                weights = [1.0 / (1 + chain_use_counts[i]) for i in range(len(chains))]
                sampled_indices = sorted(random.choices(
                    range(len(chains)), weights=weights, k=chains_per_question
                ))
                # Deduplicate (random.choices allows repeats)
                sampled_indices = sorted(set(sampled_indices))
                while len(sampled_indices) < chains_per_question:
                    remaining = [i for i in range(len(chains)) if i not in sampled_indices]
                    if not remaining:
                        break
                    sampled_indices.append(random.choice(remaining))
                    sampled_indices = sorted(set(sampled_indices))

                # Check overlap with previously used chain sets (up to 3 resample attempts)
                for _resample in range(3):
                    if not _chain_overlap_too_high(set(sampled_indices), used_chain_sets):
                        break
                    weights = [1.0 / (1 + chain_use_counts[i]) for i in range(len(chains))]
                    sampled_indices = sorted(random.choices(
                        range(len(chains)), weights=weights, k=chains_per_question
                    ))
                    sampled_indices = sorted(set(sampled_indices))
                    while len(sampled_indices) < chains_per_question:
                        remaining = [i for i in range(len(chains)) if i not in sampled_indices]
                        if not remaining:
                            break
                        sampled_indices.append(random.choice(remaining))
                        sampled_indices = sorted(set(sampled_indices))
                sampled_chains = [chains[i] for i in sampled_indices]

                # --- Pre-filter: Structural diversity check (zero LLM cost) ---
                is_diverse, diversity_reason = _check_chain_structural_diversity(sampled_chains, pool_chains=chains)
                if not is_diverse:
                    print(f"    Retry {retry + 1}/{max_retries}: Structural diversity failed - {diversity_reason}", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(0.5)
                    continue

                # --- Pre-filter: Chain grounding coherence (LLM-based, cached) ---
                chain_articles = chains_data.get('articles', [])
                grounded_chains, grounded_indices, rejected_chain_indices = filter_chains_by_grounding(
                    sampled_chains, chain_articles, use_harmony,
                    agent_info=agent_info if not use_harmony else None,
                )
                if len(grounded_chains) < 3:
                    print(f"    Retry {retry + 1}/{max_retries}: Chain grounding failed - "
                          f"only {len(grounded_chains)}/{len(sampled_chains)} chains passed", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(0.5)
                    continue

                # Update sampled_chains and sampled_indices to the grounded subset
                sampled_indices = [sampled_indices[gi] for gi in grounded_indices]
                sampled_chains = grounded_chains
                print(f"    Chain grounding: {len(sampled_chains)} chains passed "
                      f"({len(rejected_chain_indices)} rejected)", flush=True)

                # Format sampled chains text (re-number from 1)
                sampled_lines = []
                for i, chain in enumerate(sampled_chains, 1):
                    sampled_lines.append(f"Chain {i}: {chain['path_description']}")
                sampled_chains_text = "\n".join(sampled_lines)

                # Build avoid clause: previously successful answers + answers tried this question
                all_avoid = used_answers | tried_this_question
                if all_avoid:
                    avoid_list = ", ".join(sorted(all_avoid))
                    avoid_clause = f"\nDo NOT select any of these previously used answers: {avoid_list}\n"
                else:
                    avoid_clause = ""

                # Phase 0: Pick a non-seed answer entity from the chains
                print(f"    Phase 0: Selecting candidate answer (seed='{seed_label}')...", flush=True)

                p0_user_content = WIKIDATA_ANSWER_SELECTION_PROMPT.format(
                    chains_text=sampled_chains_text,
                    seed_entity=seed_label,
                    avoid_clause=avoid_clause,
                )

                if use_harmony:
                    p0_response = get_harmony_generator().generate_response_sync(
                        developer_content=WIKIDATA_ANSWER_SELECTION_DEVELOPER,
                        user_content=p0_user_content,
                        temperature=1.0,
                        max_tokens=1024,
                        reasoning_effort="low",
                    )
                else:
                    p0_prompt = WIKIDATA_ANSWER_SELECTION_DEVELOPER + "\n\n" + p0_user_content
                    p0_response = _gen(p0_prompt, temperature=1.0, max_tokens=1024)

                answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', p0_response)
                if not answer_match:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 0 failed - no answer selected", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                answer = answer_match.group(1).strip()
                # Strip format artifacts the model often leaks into the answer
                answer = re.sub(r'Source_Chain:.*', '', answer).strip()
                answer = re.sub(r'\(no valid\).*', '', answer).strip()
                if answer.lower() in ('none', ''):
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 0 failed - empty/none answer", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue
                tried_this_question.add(answer)

                # Validate answer is NOT the seed entity
                if answer.lower().strip() == seed_label.lower().strip():
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 0 failed - answer IS the seed entity", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Validate the answer is a real entity from the chains (not a format artifact)
                if not answer_is_chain_entity(answer, sampled_chains):
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 0 failed - '{answer}' is not a chain entity", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Reject ambiguous answers where the same (entity, property) hop has multiple values
                if _answer_has_sibling_values(answer, chains):
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 0 failed - "
                          f"'{answer}' is ambiguous (sibling values exist for same property)", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Skip if answer was already used (LLM ignored the avoid clause)
                if answer.lower() in {a.lower() for a in used_answers}:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 0 failed - '{answer}' already used", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Parse source chain number
                source_chain_match = re.search(r'Source_Chain:\s*(\d+)', p0_response)
                source_chain_num = int(source_chain_match.group(1)) if source_chain_match else None

                print(f"    Phase 0: Answer='{answer}', source chain={source_chain_num}", flush=True)

                # Find the answer fact triple from the source chain (for Phase 2 context)
                answer_fact_str = ""
                if source_chain_num and 1 <= source_chain_num <= len(sampled_chains):
                    src_chain = sampled_chains[source_chain_num - 1]
                    answer_lower = answer.lower().strip()
                    for hop in src_chain.get('chain', []):
                        e_label = hop.get('entity', {}).get('label', '')
                        p_label = hop.get('property', {}).get('label', '')
                        v_label = hop.get('value', {}).get('label', '')
                        if e_label.lower().strip() == answer_lower or v_label.lower().strip() == answer_lower:
                            answer_fact_str = f'"{e_label}" -[{p_label}]-> "{v_label}"'
                            break
                if not answer_fact_str:
                    # Fallback: search all sampled chains for the answer
                    answer_lower = answer.lower().strip()
                    for chain in sampled_chains:
                        for hop in chain.get('chain', []):
                            e_label = hop.get('entity', {}).get('label', '')
                            p_label = hop.get('property', {}).get('label', '')
                            v_label = hop.get('value', {}).get('label', '')
                            if e_label.lower().strip() == answer_lower or v_label.lower().strip() == answer_lower:
                                answer_fact_str = f'"{e_label}" -[{p_label}]-> "{v_label}"'
                                break
                        if answer_fact_str:
                            break

                # Phase 1: Extract clue facts about the SEED ENTITY from all sampled chains
                print(f"    Phase 1: Extracting clue facts about seed '{seed_label}'...", flush=True)

                if use_harmony:
                    p1_response = get_harmony_generator().generate_response_sync(
                        developer_content=WIKIDATA_FACT_EXTRACTION_DEVELOPER,
                        user_content=WIKIDATA_FACT_EXTRACTION_PROMPT.format(
                            seed_entity=seed_label, chains_text=sampled_chains_text,
                        ),
                        temperature=0.3,
                        max_tokens=4096,
                        reasoning_effort="low",
                    )
                else:
                    p1_prompt = WIKIDATA_FACT_EXTRACTION_DEVELOPER + "\n\n" + WIKIDATA_FACT_EXTRACTION_PROMPT.format(
                        seed_entity=seed_label, chains_text=sampled_chains_text,
                    )
                    p1_response = _gen(p1_prompt, temperature=0.3, max_tokens=4096)

                # Parse JSON from response
                try:
                    json_match = re.search(r'\{[\s\S]*\}', p1_response)
                    if not json_match:
                        print(f"    Retry {retry + 1}/{max_retries}: Phase 1 failed - no JSON in response", flush=True)
                        if retry < max_retries - 1:
                            time.sleep(1)
                        continue
                    parsed = json.loads(json_match.group(0))
                    raw_facts = parsed.get('facts', [])
                except (json.JSONDecodeError, KeyError):
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 1 failed - JSON parse error", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Validate each fact by checking the triple exists in sampled chain data
                extracted_facts = []
                for fact in raw_facts:
                    chain_num = fact.get('chain_num', 0)
                    entity_label = fact.get('entity', '')
                    property_label = fact.get('property', '')
                    value_label = fact.get('value', '')

                    is_valid, hop_idx = validate_triple_in_chains(
                        chain_num, entity_label, property_label, value_label, sampled_chains
                    )

                    if is_valid:
                        extracted_facts.append(fact)
                    else:
                        print(f"      Rejected {fact.get('fact_id', '?')}: triple not found in chain {chain_num} "
                              f"({entity_label} -[{property_label}]-> {value_label})", flush=True)

                if len(extracted_facts) < 3:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 1 failed - only {len(extracted_facts)} validated facts", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Check facts span enough chains
                fact_chains = {f['chain_num'] for f in extracted_facts}
                print(f"    Phase 1: {len(extracted_facts)} validated clue facts across {len(fact_chains)} chains", flush=True)
                if len(fact_chains) < 3:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 1 failed - facts from only {len(fact_chains)} chains", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Phase 1.5: Fact grounding verification against Wikipedia
                print(f"    Phase 1.5: Verifying facts against grounding documents...", flush=True)
                grounded_facts, rejected_grounding_facts = verify_facts_against_grounding(
                    extracted_facts, sampled_chains, chain_articles, use_harmony,
                    agent_info=agent_info if not use_harmony else None,
                )
                grounded_fact_chains = {f['chain_num'] for f in grounded_facts}
                print(f"    Phase 1.5: {len(grounded_facts)}/{len(extracted_facts)} facts grounded "
                      f"across {len(grounded_fact_chains)} chains", flush=True)

                if len(grounded_facts) < 3 or len(grounded_fact_chains) < 3:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 1.5 failed - "
                          f"{len(grounded_facts)} grounded facts across {len(grounded_fact_chains)} chains", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Replace extracted_facts with grounded subset
                extracted_facts = grounded_facts

                # Phase 2: Compose question — describe seed with clues, ask for the answer
                print(f"    Phase 2: Composing question from clue facts...", flush=True)
                facts_text = format_extracted_facts(extracted_facts)

                # Build the answer fact text for the prompt
                answer_fact_for_prompt = answer_fact_str if answer_fact_str else f'(answer "{answer}" from the chains)'

                # Build previous questions text for diversity
                if generated_questions:
                    prev_q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(generated_questions))
                else:
                    prev_q_text = "(none yet)"

                if use_harmony:
                    p2_response = get_harmony_generator().generate_response_sync(
                        developer_content=WIKIDATA_GROUNDED_QA_DEVELOPER,
                        user_content=WIKIDATA_GROUNDED_QA_PROMPT.format(
                            answer=answer, facts_text=facts_text,
                            answer_fact=answer_fact_for_prompt,
                            previous_questions=prev_q_text,
                        ),
                        temperature=1.0,
                        max_tokens=2048,
                        reasoning_effort="low",
                    )
                else:
                    p2_prompt = WIKIDATA_GROUNDED_QA_DEVELOPER + "\n\n" + WIKIDATA_GROUNDED_QA_PROMPT.format(
                        answer=answer, facts_text=facts_text,
                        answer_fact=answer_fact_for_prompt,
                        previous_questions=prev_q_text,
                    )
                    p2_response = _gen(p2_prompt, temperature=1.0, max_tokens=2048)

                question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', p2_response)
                used_fact_ids = parse_used_facts(p2_response)
                reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|$)', p2_response, re.DOTALL)

                if not question_match or not used_fact_ids:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 2 failed - no question/facts", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                question = question_match.group(1).strip()
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ''

                # Phase 3: Programmatic validation
                print(f"    Phase 3: Validating...", flush=True)

                # Check fact references exist and span 3+ chains
                refs_valid, refs_reason = validate_fact_references(used_fact_ids, extracted_facts, min_chains=3)
                if not refs_valid:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 3 failed - {refs_reason}", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # Check answer appears in at least one referenced fact or the answer fact
                ans_valid, ans_reason = validate_answer_in_facts(answer, used_fact_ids, extracted_facts)
                if not ans_valid:
                    # In seed-centric mode, the answer may not appear in clue facts (they're about the seed).
                    # Accept if we have a valid answer_fact_str linking the answer to the seed.
                    if not answer_fact_str:
                        print(f"    Retry {retry + 1}/{max_retries}: Phase 3 failed - {ans_reason}", flush=True)
                        if retry < max_retries - 1:
                            time.sleep(1)
                        continue

                # Phase 3b: Answer uniqueness check
                print(f"    Phase 3b: Checking answer uniqueness...", flush=True)
                is_unique, alternatives = _check_answer_uniqueness(
                    question, answer, use_harmony,
                    _gen=_gen if not use_harmony else None,
                )
                if not is_unique:
                    print(f"    Retry {retry + 1}/{max_retries}: Phase 3b failed — "
                          f"ambiguous answer, alternatives: {alternatives}", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue

                # All phases passed — build QA pair
                fact_lookup = {f['fact_id']: f for f in extracted_facts}
                # Map sampled chain numbers back to original chain indices
                sampled_chain_nums = {fact_lookup[fid]['chain_num'] for fid in used_fact_ids if fid in fact_lookup}
                original_chain_indices = [sampled_indices[cn - 1] + 1 for cn in sampled_chain_nums if cn - 1 < len(sampled_indices)]

                qa_pair = {
                    'answer': answer,
                    'seed_entity': seed_label,
                    'chain_used': ', '.join(str(c) for c in sorted(original_chain_indices)),
                    'question': question,
                    'reasoning_chain': reasoning,
                    'used_facts': used_fact_ids,
                    'extracted_facts': extracted_facts,
                    'answer_fact': answer_fact_str,
                }
                qa_pairs.append(qa_pair)
                used_answers.add(answer)
                used_chain_sets.append(set(sampled_indices))
                for idx in sampled_indices:
                    chain_use_counts[idx] += 1
                generated_questions.append(question)
                print(f"    Generated: {question[:60]}...", flush=True)
                break

            except Exception as e:
                print(f"    Retry {retry + 1}/{max_retries}: Error - {e}", flush=True)
                if retry < max_retries - 1:
                    time.sleep(1)

    print(f"Generated {len(qa_pairs)}/{num_questions} questions successfully", flush=True)
    return qa_pairs


def generate_multihop_questions(line_, agent_info, outfile_prefix, num_chains=20, num_questions=10,
                                 verify=True, num_verification_samples=10,
                                 min_verification_accuracy=0.0, max_verification_accuracy=0.5,
                                 verification_temperature=0.7,
                                 verify2=False, num_verification2_samples=3,
                                 verification2_mode="keep_unanswerable",
                                 min_verification2_accuracy=0.0, max_verification2_accuracy=0.3,
                                 verification2_max_iterations=200,
                                 verify3=False, num_verification3_samples=5,
                                 verification3_correctness_threshold=0.7,
                                 verification3_temperature=0.7,
                                 min_verification3_accuracy=0.0,
                                 max_verification3_accuracy=1.0):
    """
    Generate multi-hop questions for a given category by fetching Wikidata triple chains.

    Args:
        line_: Dict with category info (category, additional_requirement, etc.)
        agent_info: Tuple of (model, tokenizer, client)
        outfile_prefix: Prefix for output files
        num_chains: Number of triple chains to fetch
        num_questions: Number of QA pairs to generate
        verify: Whether to run V1 verification step
        num_verification_samples: Number of samples for V1 verification
        min_verification_accuracy: Minimum V1 accuracy to keep a QA pair
        max_verification_accuracy: Maximum V1 accuracy to keep a QA pair
        verification_temperature: Sampling temperature for V1 verification
        verify2: Whether to run V2 browser-based verification
        num_verification2_samples: Number of agentic samples for V2
        verification2_mode: 'keep_unanswerable' or 'keep_answerable'
        min_verification2_accuracy: Min V2 accuracy threshold
        max_verification2_accuracy: Max V2 accuracy threshold
        verification2_max_iterations: Max agentic loop iterations for V2
        verify3: Whether to run V3 grounding-document verification
        num_verification3_samples: Number of samples for V3 verification
        verification3_correctness_threshold: V3 accuracy threshold for v3_correct
        verification3_temperature: Sampling temperature for V3 verification
        min_verification3_accuracy: Min V3 accuracy threshold
        max_verification3_accuracy: Max V3 accuracy threshold

    Returns:
        List of annotated multi-hop QA pairs (all pairs with V1/V2/V3 scores)
    """
    # Check for V3-verified questions first
    if verify3 and os.path.exists(f"{outfile_prefix}.all_with_verification3.json"):
        print("found ", f"{outfile_prefix}.all_with_verification3.json", flush=True)
        v3_lst = []
        with open(f"{outfile_prefix}.all_with_verification3.json", "r") as f:
            for line in f:
                if line.strip():
                    v3_lst.append(json.loads(line))
        return v3_lst

    # Check for V2-scored questions (all_with_verification2 has all questions with V1+V2 scores)
    if verify2 and os.path.exists(f"{outfile_prefix}.all_with_verification2.json"):
        print("found ", f"{outfile_prefix}.all_with_verification2.json", flush=True)
        v2_all_lst = []
        with open(f"{outfile_prefix}.all_with_verification2.json", "r") as f:
            for line in f:
                if line.strip():
                    v2_all_lst.append(json.loads(line))
        # Filter to only V2-passing questions before feeding to V3
        v2_filtered_lst = [q for q in v2_all_lst
                           if 'v2_accuracy' in q and min_verification2_accuracy <= q['v2_accuracy'] <= max_verification2_accuracy]
        if verify3 and len(v2_filtered_lst) > 0:
            grounding_docs = _load_grounding_docs(outfile_prefix)
            if grounding_docs is not None:
                print(f"\n=== VERIFICATION STAGE (V3 - Grounding Documents) ===", flush=True)
                print(f"V2 filter (cached): {len(v2_filtered_lst)}/{len(v2_all_lst)} passed V2 range [{min_verification2_accuracy:.0%}, {max_verification2_accuracy:.0%}]", flush=True)
                v3_filtered, v3_all = verify_qa_batch_with_grounding(
                    v2_filtered_lst, grounding_docs, agent_info,
                    num_samples=num_verification3_samples,
                    temperature=verification3_temperature,
                    correctness_threshold=verification3_correctness_threshold,
                    min_accuracy=min_verification3_accuracy,
                    max_accuracy=max_verification3_accuracy,
                    v1_num_samples=num_verification_samples,
                    v1_temperature=verification_temperature,
                    v2_num_samples=num_verification2_samples,
                    v2_max_iterations=verification2_max_iterations,
                )
                with open(f"{outfile_prefix}.all_with_verification3.json", "w") as f:
                    for line in v3_all:
                        print(json.dumps(line), file=f)

                with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
                    for line in v3_filtered:
                        print(json.dumps(line), file=f)

                return v3_all
            else:
                print("WARNING: V3 requested but no triples.json found. Skipping V3.", flush=True)
        return v2_all_lst

    # Check for V1-scored questions (all_with_verification has all questions with V1 scores)
    if os.path.exists(f"{outfile_prefix}.all_with_verification.json"):
        print("found ", f"{outfile_prefix}.all_with_verification.json", flush=True)
        v1_all_lst = []
        with open(f"{outfile_prefix}.all_with_verification.json", "r") as f:
            for line in f:
                if line.strip():
                    v1_all_lst.append(json.loads(line))
        # V2 runs only on V1-filtered (hard) questions
        v1_filtered_lst = [q for q in v1_all_lst
                           if min_verification_accuracy <= q.get('verification_accuracy', 1.0) <= max_verification_accuracy]
        v1_rejected_lst = [q for q in v1_all_lst if q not in v1_filtered_lst]
        print(f"V1 filter (cached): {len(v1_filtered_lst)}/{len(v1_all_lst)} passed, {len(v1_rejected_lst)} skipped for V2", flush=True)
        if verify2 and BROWSER_TOOL_AVAILABLE and len(v1_filtered_lst) > 0:
            v2_filtered, v2_verified = verify_qa_batch_with_browser(
                v1_filtered_lst,
                mode=verification2_mode,
                num_samples=num_verification2_samples,
                min_accuracy=min_verification2_accuracy,
                max_accuracy=max_verification2_accuracy,
                max_iterations=verification2_max_iterations,
            )
            # Merge V2-verified questions with V1-rejected (skipped) questions
            v2_all = v2_verified + v1_rejected_lst
            with open(f"{outfile_prefix}.all_with_verification2.json", "w") as f:
                for line in v2_all:
                    print(json.dumps(line), file=f)
            with open(f"{outfile_prefix}.verified2_questions.json", "w") as f:
                for line in v2_filtered:
                    print(json.dumps(line), file=f)
            print(f"V2 Verification: {len(v2_filtered)}/{len(v2_verified)} passed filter ({len(v1_rejected_lst)} skipped from V1)", flush=True)
            # V3 runs only on V2-filtered questions (those that passed V2 accuracy range)
            if verify3 and len(v2_filtered) > 0:
                grounding_docs = _load_grounding_docs(outfile_prefix)
                if grounding_docs is not None:
                    print(f"\n=== VERIFICATION STAGE (V3 - Grounding Documents) ===", flush=True)
                    v3_filtered, v3_all = verify_qa_batch_with_grounding(
                        v2_filtered, grounding_docs, agent_info,
                        num_samples=num_verification3_samples,
                        temperature=verification3_temperature,
                        correctness_threshold=verification3_correctness_threshold,
                        min_accuracy=min_verification3_accuracy,
                        max_accuracy=max_verification3_accuracy,
                        v1_num_samples=num_verification_samples,
                        v1_temperature=verification_temperature,
                        v2_num_samples=num_verification2_samples,
                        v2_max_iterations=verification2_max_iterations,
                    )
                    with open(f"{outfile_prefix}.all_with_verification3.json", "w") as f:
                        for line in v3_all:
                            print(json.dumps(line), file=f)

                    with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
                        for line in v3_filtered:
                            print(json.dumps(line), file=f)

                    return v3_all
                else:
                    print("WARNING: V3 requested but no triples.json found. Skipping V3.", flush=True)
            return v2_all
        # V3 after V1 (no V2) — runs on ALL questions
        if verify3 and len(v1_all_lst) > 0:
            grounding_docs = _load_grounding_docs(outfile_prefix)
            if grounding_docs is not None:
                print(f"\n=== VERIFICATION STAGE (V3 - Grounding Documents) ===", flush=True)
                v3_filtered, v3_all = verify_qa_batch_with_grounding(
                    v1_all_lst, grounding_docs, agent_info,
                    num_samples=num_verification3_samples,
                    temperature=verification3_temperature,
                    correctness_threshold=verification3_correctness_threshold,
                    min_accuracy=min_verification3_accuracy,
                    max_accuracy=max_verification3_accuracy,
                    v1_num_samples=num_verification_samples,
                    v1_temperature=verification_temperature,
                    v2_num_samples=num_verification2_samples,
                    v2_max_iterations=verification2_max_iterations,
                )
                with open(f"{outfile_prefix}.all_with_verification3.json", "w") as f:
                    for line in v3_all:
                        print(json.dumps(line), file=f)

                with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
                    for line in v3_filtered:
                        print(json.dumps(line), file=f)

                return v3_all
            else:
                print("WARNING: V3 requested but no triples.json found. Skipping V3.", flush=True)
        return v1_all_lst

    # Fall back to unverified if verification was disabled
    if os.path.exists(f"{outfile_prefix}.multihop_questions.json") and not verify:
        print("found ", f"{outfile_prefix}.multihop_questions.json", flush=True)
        full_lst = []
        with open(f"{outfile_prefix}.multihop_questions.json", "r") as f:
            for line in f:
                if line.strip():
                    full_lst.append(json.loads(line))
        return full_lst

    # Fetch related entities and triple chains from Wikidata
    print(f"Fetching Wikidata triples for: {line_['category']}", flush=True)
    chains_data = fetch_related_entities(line_['category'], num_chains=num_chains)

    if not chains_data['chains']:
        print(f"No triple chains found, skipping multi-hop generation", flush=True)
        return []

    # Save fetched triples for reference
    triples_to_save = {
        'seed_entity': chains_data['seed_entity'],
        'num_chains': len(chains_data['chains']),
        'chains': chains_data['chains'],
    }
    with open(f"{outfile_prefix}.triples.json", "w") as f:
        json.dump(triples_to_save, f, indent=2)

    # Generate articles.json with Wikipedia content
    articles = chains_data.get('articles', [])
    if not articles:
        # Fallback: use triple descriptions (legacy behavior)
        seed_label = chains_data.get('seed_entity', {}).get('label', 'Unknown Entity')
        descriptions = [c.get('path_description', '') for c in chains_data['chains'] if c.get('path_description')]
        paragraph = "\n".join(f"- {d}" for d in descriptions) if descriptions else "(no triple chains)"
        articles = [{"title": seed_label, "paragraph": paragraph}]
    with open(f"{outfile_prefix}.articles.json", "w") as f:
        json.dump(articles, f, indent=2)

    # Generate multi-hop QA pairs from triples
    print(f"Generating multi-hop QA with {len(chains_data['chains'])} triple chains", flush=True)
    qa_pairs = gen_multihop_qa_from_triples(chains_data, agent_info, num_questions=num_questions)

    # Augment QA pairs with metadata
    full_lst = []
    seed_entity = chains_data.get('seed_entity', {})

    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue
        if 'question' not in qa or 'answer' not in qa:
            continue

        line = copy.deepcopy(line_)
        line['question'] = qa['question']
        line['gold_answer'] = qa['answer']
        line['chain_used'] = qa.get('chain_used', '')
        line['reasoning_chain'] = qa.get('reasoning_chain', '')
        line['difficulty'] = qa.get('difficulty', 'hard')
        line['wikidata_entity'] = seed_entity.get('id', '')
        line['wiki_entity'] = seed_entity.get('label', '')  # compat with wiki pipeline
        line['source_triples'] = [c['path_description'] for c in chains_data['chains']]
        line['source_articles'] = [a['title'] for a in chains_data.get('articles', [])] or [seed_entity.get('label', '')]
        line['data_source'] = 'wikidata'
        line['num_hops'] = 2
        if 'used_facts' in qa:
            line['used_facts'] = qa['used_facts']
        if 'extracted_facts' in qa:
            line['extracted_facts'] = qa['extracted_facts']

        # Build per-question trimmed & marked-up grounding articles
        line['grounding_articles'] = build_grounding_articles_for_qa(line, chains_data)

        full_lst.append(line)

    # Save unverified questions
    with open(f"{outfile_prefix}.multihop_questions.json", "w") as f:
        for line in full_lst:
            print(json.dumps(line), file=f)

    print(f"Generated {len(full_lst)} multi-hop questions", flush=True)

    # V1 Verification step
    if verify and len(full_lst) > 0:
        print(f"\n=== VERIFICATION STAGE (V1) ===", flush=True)
        print(f"Verifying {len(full_lst)} questions with {num_verification_samples} samples each...", flush=True)

        verified_lst, all_with_verification = verify_qa_batch(
            full_lst,
            agent_info,
            num_samples=num_verification_samples,
            temperature=verification_temperature,
            min_accuracy=min_verification_accuracy,
            max_accuracy=max_verification_accuracy
        )

        # Save all questions with V1 verification scores
        with open(f"{outfile_prefix}.all_with_verification.json", "w") as f:
            for line in all_with_verification:
                print(json.dumps(line), file=f)

        # Save V1-filtered questions
        with open(f"{outfile_prefix}.verified_questions.json", "w") as f:
            for line in verified_lst:
                print(json.dumps(line), file=f)

        print(f"\nV1 Verification: {len(verified_lst)}/{len(full_lst)} passed filter", flush=True)

        # V2 Verification step (browser-based) — runs only on V1-filtered (hard) questions
        v1_rejected = [q for q in all_with_verification if q not in verified_lst]
        print(f"V2 will run on {len(verified_lst)} V1-passed questions ({len(v1_rejected)} skipped as too easy)", flush=True)
        if verify2 and BROWSER_TOOL_AVAILABLE and len(verified_lst) > 0:
            v2_filtered, v2_verified = verify_qa_batch_with_browser(
                verified_lst,
                mode=verification2_mode,
                num_samples=num_verification2_samples,
                min_accuracy=min_verification2_accuracy,
                max_accuracy=max_verification2_accuracy,
                max_iterations=verification2_max_iterations,
            )

            # Merge V2-verified questions with V1-rejected (skipped) questions
            v2_all = v2_verified + v1_rejected
            with open(f"{outfile_prefix}.all_with_verification2.json", "w") as f:
                for line in v2_all:
                    print(json.dumps(line), file=f)

            with open(f"{outfile_prefix}.verified2_questions.json", "w") as f:
                for line in v2_filtered:
                    print(json.dumps(line), file=f)

            print(f"V2 Verification: {len(v2_filtered)}/{len(v2_verified)} passed filter ({len(v1_rejected)} skipped from V1)", flush=True)

            # V3 runs only on V2-filtered questions (those that passed V2 accuracy range)
            if verify3 and len(v2_filtered) > 0:
                grounding_docs = _load_grounding_docs(outfile_prefix)
                if grounding_docs is not None:
                    print(f"\n=== VERIFICATION STAGE (V3 - Grounding Documents) ===", flush=True)
                    v3_filtered, v3_all = verify_qa_batch_with_grounding(
                        v2_filtered, grounding_docs, agent_info,
                        num_samples=num_verification3_samples,
                        temperature=verification3_temperature,
                        correctness_threshold=verification3_correctness_threshold,
                        min_accuracy=min_verification3_accuracy,
                        max_accuracy=max_verification3_accuracy,
                        v1_num_samples=num_verification_samples,
                        v1_temperature=verification_temperature,
                        v2_num_samples=num_verification2_samples,
                        v2_max_iterations=verification2_max_iterations,
                    )
                    with open(f"{outfile_prefix}.all_with_verification3.json", "w") as f:
                        for line in v3_all:
                            print(json.dumps(line), file=f)

                    with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
                        for line in v3_filtered:
                            print(json.dumps(line), file=f)

                    return v3_all
                else:
                    print("WARNING: V3 requested but no triples.json found. Skipping V3.", flush=True)

            return v2_all

        # V3 after V1 (no V2) — runs on ALL questions with V1 scores
        if verify3 and len(all_with_verification) > 0:
            grounding_docs = _load_grounding_docs(outfile_prefix)
            if grounding_docs is not None:
                print(f"\n=== VERIFICATION STAGE (V3 - Grounding Documents) ===", flush=True)
                v3_filtered, v3_all = verify_qa_batch_with_grounding(
                    all_with_verification, grounding_docs, agent_info,
                    num_samples=num_verification3_samples,
                    temperature=verification3_temperature,
                    correctness_threshold=verification3_correctness_threshold,
                    min_accuracy=min_verification3_accuracy,
                    max_accuracy=max_verification3_accuracy,
                    v1_num_samples=num_verification_samples,
                    v1_temperature=verification_temperature,
                    v2_num_samples=num_verification2_samples,
                    v2_max_iterations=verification2_max_iterations,
                )
                with open(f"{outfile_prefix}.all_with_verification3.json", "w") as f:
                    for line in v3_all:
                        print(json.dumps(line), file=f)

                with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
                    for line in v3_filtered:
                        print(json.dumps(line), file=f)

                return v3_all
            else:
                print("WARNING: V3 requested but no triples.json found. Skipping V3.", flush=True)

        return all_with_verification

    # V3 on unverified (no V1/V2)
    if verify3 and len(full_lst) > 0:
        grounding_docs = _load_grounding_docs(outfile_prefix)
        if grounding_docs is not None:
            print(f"\n=== VERIFICATION STAGE (V3 - Grounding Documents) ===", flush=True)
            v3_filtered, v3_all = verify_qa_batch_with_grounding(
                full_lst, grounding_docs, agent_info,
                num_samples=num_verification3_samples,
                temperature=verification3_temperature,
                correctness_threshold=verification3_correctness_threshold,
                min_accuracy=min_verification3_accuracy,
                max_accuracy=max_verification3_accuracy,
                v1_num_samples=num_verification_samples,
                v1_temperature=verification_temperature,
                v2_num_samples=num_verification2_samples,
                v2_max_iterations=verification2_max_iterations,
            )
            with open(f"{outfile_prefix}.all_with_verification3.json", "w") as f:
                for line in v3_all:
                    print(json.dumps(line), file=f)
            with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
                for line in v3_filtered:
                    print(json.dumps(line), file=f)
            return v3_all
        else:
            print("WARNING: V3 requested but no triples.json found. Skipping V3.", flush=True)

    return full_lst


def generate_full_multihop_qa(theme, agent_info, history, iters, outfile_prefix='att1',
                              historical_psg=None,
                              category_gen_func=None,
                              acc_target=None,
                              num_chains=20,
                              num_questions=10,
                              verify=True,
                              num_verification_samples=10,
                              min_verification_accuracy=0.0,
                              max_verification_accuracy=0.5,
                              verification_temperature=0.7,
                              verify2=False,
                              num_verification2_samples=3,
                              verification2_mode="keep_unanswerable",
                              min_verification2_accuracy=0.0,
                              max_verification2_accuracy=0.3,
                              verification2_max_iterations=200,
                              verify3=False,
                              num_verification3_samples=5,
                              verification3_correctness_threshold=0.7,
                              verification3_temperature=0.7,
                              min_verification3_accuracy=0.0,
                              max_verification3_accuracy=1.0):
    """
    Generate full multi-hop QA dataset for a theme using Wikidata triples.

    This is the main entry point for Wikidata multi-hop benchmark generation.

    Args:
        theme: Topic theme (e.g., 'history')
        agent_info: Tuple of (model, tokenizer, client)
        history: Previous iteration history
        iters: Current iteration number
        outfile_prefix: Prefix for output files
        historical_psg: Previously used categories to avoid repetition
        category_gen_func: Function to generate categories
        acc_target: Target accuracy range
        num_chains: Number of triple chains to fetch per category
        num_questions: Number of questions to generate per category
        verify: Whether to run V1 verification step
        num_verification_samples: Number of samples for V1 verification
        min_verification_accuracy: Minimum V1 accuracy threshold
        max_verification_accuracy: Maximum V1 accuracy threshold
        verification_temperature: Sampling temperature for V1 verification
        verify2: Whether to run V2 browser-based verification
        num_verification2_samples: Number of agentic samples for V2
        verification2_mode: 'keep_unanswerable' or 'keep_answerable'
        min_verification2_accuracy: Min V2 accuracy threshold
        max_verification2_accuracy: Max V2 accuracy threshold
        verification2_max_iterations: Max agentic loop iterations for V2
        verify3: Whether to run V3 grounding-document verification
        num_verification3_samples: Number of samples for V3 verification
        verification3_correctness_threshold: V3 accuracy threshold for v3_correct
        verification3_temperature: Sampling temperature for V3 verification
        min_verification3_accuracy: Min V3 accuracy threshold
        max_verification3_accuracy: Max V3 accuracy threshold
    """
    # Check for V3-verified questions first
    if verify3 and os.path.exists(f"{outfile_prefix}.verified3_questions.json"):
        print("FOUND verified3_questions.json", flush=True)
        return historical_psg

    # Check for V2-verified questions
    if verify2 and os.path.exists(f"{outfile_prefix}.verified2_questions.json"):
        print("FOUND verified2_questions.json", flush=True)
        return historical_psg

    # Check for V1-verified questions
    if os.path.exists(f"{outfile_prefix}.verified_questions.json"):
        print("FOUND verified_questions.json", flush=True)
        return historical_psg

    if os.path.exists(f"{outfile_prefix}.multihop_questions.json") and not verify:
        print("FOUND multihop_questions.json", flush=True)
        return historical_psg

    # Use default category generation if not specified
    if category_gen_func is None:
        category_gen_func = _refine_categories_targetacc_augmented

    # Generate categories
    if acc_target is not None:
        json_category = category_gen_func(theme, agent_info, history, iters,
                                          outfile_prefix=outfile_prefix,
                                          acc_target=acc_target)
    else:
        json_category = category_gen_func(theme, agent_info, history, iters,
                                          outfile_prefix=outfile_prefix)

    # Apply saliency reranking
    json_category = saliency_rerank(json_category, 5)

    full_lst = []
    historical_psg = historical_psg or []

    for line_ in json_category:
        if line_['category'] in historical_psg:
            print(f"Skipping repetitive category: {line_['category']}", flush=True)
            continue

        historical_psg.append(line_['category'])

        if 'additional_requirement' not in line_:
            continue

        page_title = line_['category'].replace(' ', '_')
        pageviews = get_pageviews(page_title)
        line_['salience'] = pageviews if pageviews is not None else 0
        print(f"Salience of {page_title}: {round(line_['salience'] / 1000000, 2)}M", flush=True)

        try:
            qa_pairs = generate_multihop_questions(
                line_, agent_info,
                outfile_prefix + f'__{page_title}',
                num_chains=num_chains,
                num_questions=num_questions,
                verify=verify,
                num_verification_samples=num_verification_samples,
                min_verification_accuracy=min_verification_accuracy,
                max_verification_accuracy=max_verification_accuracy,
                verification_temperature=verification_temperature,
                verify2=verify2,
                num_verification2_samples=num_verification2_samples,
                verification2_mode=verification2_mode,
                min_verification2_accuracy=min_verification2_accuracy,
                max_verification2_accuracy=max_verification2_accuracy,
                verification2_max_iterations=verification2_max_iterations,
                verify3=verify3,
                num_verification3_samples=num_verification3_samples,
                verification3_correctness_threshold=verification3_correctness_threshold,
                verification3_temperature=verification3_temperature,
                min_verification3_accuracy=min_verification3_accuracy,
                max_verification3_accuracy=max_verification3_accuracy,
            )
            full_lst.extend(qa_pairs)
        except Exception as e:
            print(f"Error generating multi-hop questions for {page_title}: {e}", flush=True)
            continue

    # Save combined results
    if verify3:
        with open(f"{outfile_prefix}.verified3_questions.json", "w") as f:
            json.dump(full_lst, f, indent=2)
    elif verify2:
        with open(f"{outfile_prefix}.verified2_questions.json", "w") as f:
            json.dump(full_lst, f, indent=2)
    elif verify:
        with open(f"{outfile_prefix}.verified_questions.json", "w") as f:
            json.dump(full_lst, f, indent=2)
    else:
        with open(f"{outfile_prefix}.multihop_questions.json", "w") as f:
            json.dump(full_lst, f, indent=2)

    with open(f"{outfile_prefix}.categories_augmented.json", "w") as f:
        json.dump(json_category, f, indent=2)

    # Print summary
    print(f"\n=== GENERATION SUMMARY ===", flush=True)
    print(f"Total verified questions: {len(full_lst)}", flush=True)
    if full_lst:
        avg_accuracy = np.mean([q.get('verification_accuracy', 0) for q in full_lst])
        print(f"Average V1 verification accuracy: {avg_accuracy:.1%}", flush=True)
        if verify2:
            avg_v2 = np.mean([q.get('v2_accuracy', 0) for q in full_lst])
            print(f"Average V2 verification accuracy: {avg_v2:.1%}", flush=True)
        if verify3:
            avg_v3 = np.mean([q.get('v3_accuracy', 0) for q in full_lst])
            print(f"Average V3 verification accuracy: {avg_v3:.1%}", flush=True)

    return historical_psg


# ===== Category Generation Functions (duplicated from wiki_harmony_drbencher) =====

def _refine_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    category_json = _generate_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        full_cat_lst.extend(cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are not repetitive from prior selections and covers a wide range of topics.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) medium-to-low frequency — prefer specific subtopics and lesser-known events over the most famous, universally-known topics, while still being notable enough to have a Wikipedia article.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting:
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement.
Make sure the categories are similar to wikipedia categories.
The categories should be exactly in the following format (a list of dictionaries):
```json
[
{"id": "1", "category": "Neoplatonism", "parent_category": "Philosophy", "additional_requirement": "focus on key figures and their departures from classical Platonism"},
{"id": "2", "category": "War of the Pacific", "parent_category": "Military History", "additional_requirement": "key battles and territorial outcomes"},
...
]
```
Do not use python code block.
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration:
The goal is to find a set of categories that have broad coverage of topics and are not repetitive from prior selections.

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list.
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')

def _refine_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    category_json = _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        full_cat_lst.extend(cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are likely to achieve the target accuracy of {ACC_TARGET}.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) likely to obtain the target accuracy of {ACC_TARGET}, you can judge this based on the accuracy statistics from previous iterations, and (3) medium-to-low frequency — prefer specific subtopics and lesser-known events over the most famous, universally-known topics, while still being notable enough to have a Wikipedia article.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting:
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement.
Make sure the categories are similar to wikipedia categories.
The categories should be exactly in the following format (a list of dictionaries):
```json
[
{"id": "1", "category": "Neoplatonism", "parent_category": "Philosophy", "additional_requirement": "focus on key figures and their departures from classical Platonism"},
{"id": "2", "category": "War of the Pacific", "parent_category": "Military History", "additional_requirement": "key battles and territorial outcomes"},
...
]
```
Do not use python code block.
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration:
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}.

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list, that are most likely to achieve the target accuracy level, while still being relevant and salient.
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')



def _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that achieve the target accuracy of {ACC_TARGET}.
The categories should be diverse, under the theme of THEME. Prefer medium-to-low frequency topics — specific subtopics, lesser-known events, or niche areas rather than the most famous, universally-known topics. Categories should still be notable enough to have a dedicated Wikipedia article with substantive content.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting:
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement.
Make sure the categories are similar to wikipedia categories.
The categories should be exactly in the following format (a list of dictionaries):
```json
[
{"id": "1", "category": "Neoplatonism", "parent_category": "Philosophy", "additional_requirement": "focus on key figures and their departures from classical Platonism"},
{"id": "2", "category": "War of the Pacific", "parent_category": "Military History", "additional_requirement": "key battles and territorial outcomes"},
...
]
```
Do not use python code block.
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration:
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}.

For iteration 1, you can start with a wide variety of categories for us to build upon later.
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage. Coming up with new categories that can are likely to achieve the target accuracy level.
2. For example, If you find the model now lacks categories of 0.3 -- 0.5 accuracy, you should come up with more categories that would yield accuracy in that range, by either reducing the difficulty of questions that achieve lower accuracy (via subcategory or via additional requirement), or increasing the difficulty of questions that achieve higher accuracy.
3. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix)

def _generate_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that target medium-to-low frequency topics — specific enough to challenge a knowledgeable reader, but still notable enough to have a dedicated Wikipedia article with substantive content.
The categories should be diverse, under the theme of THEME. Avoid the most famous, universally-known topics (e.g., "World War II", "Cold War") and instead prefer more specific subtopics, lesser-known events, or niche areas (e.g., "War of the Pacific", "Congress of Vienna", "Taiping Rebellion").
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting:
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement.
Make sure the categories are similar to wikipedia categories.
The categories should be exactly in the following format (a list of dictionaries):
```json
[
{"id": "1", "category": "Neoplatonism", "parent_category": "Philosophy", "additional_requirement": "focus on key figures and their departures from classical Platonism"},
{"id": "2", "category": "War of the Pacific", "parent_category": "Military History", "additional_requirement": "key battles and territorial outcomes"},
...
]
```
Do not use python code block.
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration:
The goal is to find a set of categories that target medium-to-low frequency topics with broad coverage.

For iteration 1, you can start with a wide variety of categories for us to build upon later.
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage.
2. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix)

def _refine_categories(theme, context, agent_info, history, iters, candidate_lst, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1." + "Here are the category candidates to select from (delimited by ||): " + " || ".join(candidate_lst) + "\n"
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters) + "Here are the category candidates to select from (delimited by ||): " + "||".join(candidate_lst) + "\n"
    full_context = DEFAULT_JSON_MESSAGE + context
    if get_harmony_generator() is not None:
        response = gen_from_prompt_harmony(
            prompt=context,
            temperature=0.0,
            max_tokens=2000,
            developer_content="You are a helpful AI assistant. Output valid JSON."
        )
    else:
        request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[full_context],
                                         echo_prompt=False, temperature=0.0, max_tokens=2000,
                                         process_func=None, service=agent_client,
                                         terminate_by_linebreak='no',)
        response = request_result.completions[0].text

    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)
        out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json

def _generate_categories(theme, context, agent_info, history, iters, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters)
    full_context = DEFAULT_JSON_MESSAGE + context
    if get_harmony_generator() is not None:
        response = gen_from_prompt_harmony(
            prompt=context,
            temperature=0.0,
            max_tokens=2000,
            developer_content="You are a helpful AI assistant. Output valid JSON."
        )
    else:
        request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[full_context],
                                         echo_prompt=False, temperature=0.0, max_tokens=2000,
                                         process_func=None, service=agent_client,
                                         terminate_by_linebreak='no', )
        response = request_result.completions[0].text

    # Ensure output directory exists
    output_dir = os.path.dirname(f"{outfile_prefix}.full_thoughts.txt")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)
        out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json


def saliency_rerank(json_lst, num_keep=5):
    for line_ in json_lst:
        page_title = line_['category'].replace(' ', '_')
        pageviews = get_pageviews(page_title)
        line_['salience'] = pageviews if pageviews is not None else 0
    json_lst = sorted(json_lst, key=lambda x: x['salience'], reverse=True)
    for line in json_lst:
        print(f'salience of {line["category"]}: ', round(line['salience'] / 1000000, 2), 'M')
    return json_lst[:num_keep]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='WikidataHarmonyDRBencher',
        description='Generate multi-hop QA benchmarks from Wikidata knowledge graph triples with Harmony support',
        epilog='Wikidata multi-hop QA generation pipeline')

    parser.add_argument('--agent_modelname', default='openai/gpt-oss-120b')
    parser.add_argument('--temperature', type=float, default=0.001)
    parser.add_argument('--theme', type=str, default='history')  # comma-separated list of themes
    parser.add_argument('--use_helm', type=str, default='yes')
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--acc_target', type=str, default="0.1--0.3")
    parser.add_argument('--num_iters', type=int, default=8)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)

    # Wikidata multi-hop parameters
    parser.add_argument('--num_chains', type=int, default=20)  # number of triple chains to fetch
    parser.add_argument('--num_questions', type=int, default=10)  # questions per category

    # V1 Verification parameters
    parser.add_argument('--verify', type=str, default='yes')
    parser.add_argument('--num_verification_samples', type=int, default=10)
    parser.add_argument('--min_verification_accuracy', type=float, default=0.0)
    parser.add_argument('--max_verification_accuracy', type=float, default=0.5)
    parser.add_argument('--verification_temperature', type=float, default=0.7)

    # V2 browser-based verification parameters
    parser.add_argument('--verify2', type=str, default='no')
    parser.add_argument('--num_verification2_samples', type=int, default=3)
    parser.add_argument('--verification2_mode', type=str, default='keep_unanswerable')
    parser.add_argument('--min_verification2_accuracy', type=float, default=0.0)
    parser.add_argument('--max_verification2_accuracy', type=float, default=0.3)

    # V3 grounding-document verification parameters
    parser.add_argument('--verify3', type=str, default='no')
    parser.add_argument('--num_verification3_samples', type=int, default=5)
    parser.add_argument('--verification3_correctness_threshold', type=float, default=0.7)
    parser.add_argument('--verification3_temperature', type=float, default=0.7)
    parser.add_argument('--min_verification3_accuracy', type=float, default=0.0)
    parser.add_argument('--max_verification3_accuracy', type=float, default=1.0)

    # Harmony format parameters (for gpt-oss-120b)
    parser.add_argument('--use_harmony', type=str, default='no')
    parser.add_argument('--harmony_model_path', type=str, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)

    parser.add_argument('--outfile_prefix1', type=str, default='att1')

    args2 = parser.parse_args()
    args = copy.deepcopy(args2)

    # Initialize Harmony generator if requested
    if args.use_harmony.lower() == 'yes':
        print("=== Initializing Harmony vLLM Generator ===", flush=True)
        model_path = args.harmony_model_path or args.agent_modelname

        import torch
        tp_size = args.tensor_parallel_size or torch.cuda.device_count()

        harmony_gen = HarmonyVLLMGenerator(
            model_path=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        set_harmony_generator(harmony_gen)
        print("Harmony generator initialized successfully", flush=True)
        print("=" * 40, flush=True)

    if args.use_helm == 'yes':
        test_taker_info = helm_process_args(args.agent_modelname)
        print('loaded helm models')
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
    else:
        agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
            args.agent_modelname, tensor_parallel_size=args.tensor_parallel_size)
        test_taker_info = (agent_lm, agent_tokenizer, agent_client)

    evaluator_info = (agent_lm, agent_tokenizer, agent_client)
    agent_info = (agent_lm, agent_tokenizer, agent_client)

    # Only multihop mode for Wikidata pipeline
    do_verify = args.verify.lower() == 'yes'
    do_verify2 = args.verify2.lower() == 'yes'
    do_verify3 = args.verify3.lower() == 'yes'

    if do_verify2 and not BROWSER_TOOL_AVAILABLE:
        print("WARNING: --verify2 yes but MultiSourceKnowledgeBrowserTool not available. Disabling V2.", flush=True)
        do_verify2 = False

    # Parse comma-separated themes into list
    themes = [t.strip() for t in args.theme.split(',')]

    print(f"=== WIKIDATA MULTI-HOP QA GENERATION ===", flush=True)
    print(f"Themes: {themes}", flush=True)
    print(f"V1 Verification: {'Enabled' if do_verify else 'Disabled'}", flush=True)
    if do_verify:
        print(f"  - Samples per question: {args.num_verification_samples}", flush=True)
        print(f"  - Keep questions with accuracy <= {args.max_verification_accuracy:.0%} (hard questions)", flush=True)
        print(f"  - Sampling temperature: {args.verification_temperature}", flush=True)
    print(f"V2 Browser Verification: {'Enabled' if do_verify2 else 'Disabled'}", flush=True)
    if do_verify2:
        print(f"  - Agentic samples per question: {args.num_verification2_samples}", flush=True)
        print(f"  - Mode: {args.verification2_mode}", flush=True)
        print(f"  - Accuracy range: [{args.min_verification2_accuracy:.0%}, {args.max_verification2_accuracy:.0%}]", flush=True)
    print(f"V3 Grounding-Doc Verification: {'Enabled' if do_verify3 else 'Disabled'}", flush=True)
    if do_verify3:
        print(f"  - Samples per question: {args.num_verification3_samples}", flush=True)
        print(f"  - Correctness threshold: {args.verification3_correctness_threshold:.0%}", flush=True)
        print(f"  - Accuracy range: [{args.min_verification3_accuracy:.0%}, {args.max_verification3_accuracy:.0%}]", flush=True)
        print(f"  - Sampling temperature: {args.verification3_temperature}", flush=True)
    print(f"Triple chains per category: {args.num_chains}", flush=True)
    print(f"Questions per category: {args.num_questions}", flush=True)
    print("=" * 40, flush=True)

    # Loop over each theme
    for theme in themes:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Processing Theme: {theme.upper()} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        history_dict = []
        historical_psg = []

        for iters in range(args.num_iters):
            # Include theme as prefix of output file name
            if '/' in args.outfile_prefix1:
                base_dir, model_prefix = args.outfile_prefix1.rsplit('/', 1)
                args.outfile_prefix = f"{base_dir}/{theme}.{model_prefix}{iters + 1}"
            else:
                args.outfile_prefix = f"{theme}.{args.outfile_prefix1}{iters + 1}"
            summarized_content = summarize_over_history(history_dict, gold_key='gold_answer', verbose=False)
            history = [summarized_content]

            historical_psg = generate_full_multihop_qa(
                theme, agent_info, history, iters + 1,
                outfile_prefix=args.outfile_prefix,
                historical_psg=historical_psg,
                category_gen_func=_refine_categories_targetacc_augmented,
                acc_target=args.acc_target,
                num_chains=args.num_chains,
                num_questions=args.num_questions,
                verify=do_verify,
                num_verification_samples=args.num_verification_samples,
                min_verification_accuracy=args.min_verification_accuracy,
                max_verification_accuracy=args.max_verification_accuracy,
                verification_temperature=args.verification_temperature,
                verify2=do_verify2,
                num_verification2_samples=args.num_verification2_samples,
                verification2_mode=args.verification2_mode,
                min_verification2_accuracy=args.min_verification2_accuracy,
                max_verification2_accuracy=args.max_verification2_accuracy,
                verify3=do_verify3,
                num_verification3_samples=args.num_verification3_samples,
                verification3_correctness_threshold=args.verification3_correctness_threshold,
                verification3_temperature=args.verification3_temperature,
                min_verification3_accuracy=args.min_verification3_accuracy,
                max_verification3_accuracy=args.max_verification3_accuracy,
            )

            # Load generated questions — prefer V3 > V2 > V1 output
            if do_verify3:
                questions_file = f"{args.outfile_prefix}.verified3_questions.json"
                if not os.path.exists(questions_file):
                    questions_file = f"{args.outfile_prefix}.verified2_questions.json"
                if not os.path.exists(questions_file):
                    questions_file = f"{args.outfile_prefix}.verified_questions.json"
            elif do_verify2:
                questions_file = f"{args.outfile_prefix}.verified2_questions.json"
                if not os.path.exists(questions_file):
                    questions_file = f"{args.outfile_prefix}.verified_questions.json"
            elif do_verify:
                questions_file = f"{args.outfile_prefix}.verified_questions.json"
            else:
                questions_file = f"{args.outfile_prefix}.multihop_questions.json"

            with open(questions_file, "r") as f:
                json_category = json.load(f)
            if isinstance(json_category, list) and len(json_category) == 1:
                json_category = json_category[0]

            if len(json_category) == 0:
                print(f"No questions generated for {theme} iteration {iters + 1}, skipping evaluation", flush=True)
                continue

            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(
                test_taker_info, evaluator_info, json_category, gold_answer_json,
                args.outfile_prefix, 'gold_answer'
            )
            history_dict.append(json_dict)

            verbose_description = get_summary_of_results(json_dict, verbose=False)
            print(verbose_description)

        print(f"\n=== Completed Theme: {theme.upper()} ===\n", flush=True)
