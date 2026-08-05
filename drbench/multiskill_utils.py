"""Shared validation utilities for all sub-benchmarks.

Phase 1/1.5 helpers (chain-based fact extraction and grounding verification)
and Phase 3 validation checks.

Each Phase 3 check returns ``(passed: bool, reason: str)``.  ``reason`` is
empty when *passed* is True.  The :func:`run_phase3_validation` orchestrator
runs all checks in order and short-circuits on the first failure.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .multiskill_template import _execute_computation_code  # noqa: F401
from .tool_util import extract_json_v2, sparql_query, get_entity_labels_batch
from .wikidata_harmony import (
    validate_triple_in_chains,
    verify_facts_against_grounding,
    WIKIDATA_FACT_EXTRACTION_DEVELOPER,
    WIKIDATA_FACT_EXTRACTION_PROMPT,
)
from .harmony_base import (
    gen_from_prompt_harmony,
    get_harmony_generator,
)


# ---------------------------------------------------------------------------
# Content unwrapping (handles TextContent objects from agentic responses)
# ---------------------------------------------------------------------------

def unwrap_content(content):
    """Unwrap message content to plain text.

    Handles TextContent objects, lists of TextContent, and plain strings.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Single TextContent-like object with .text attribute
    if hasattr(content, 'text'):
        return content.text
    # List of TextContent-like objects
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if hasattr(item, 'text'):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Non-Latin text filter (Fix 2: reject Arabic, Cyrillic, CJK, etc.)
# ---------------------------------------------------------------------------

_NON_LATIN_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F'   # Arabic
    r'\u0400-\u04FF'                   # Cyrillic
    r'\u4E00-\u9FFF'                   # CJK Unified Ideographs
    r'\u3040-\u309F\u30A0-\u30FF'      # Japanese Hiragana + Katakana
    r'\uAC00-\uD7AF]'                  # Korean Hangul
)


# ---------------------------------------------------------------------------
# Known factual error blocklist (Fix 3: e.g. Nasdaq/Sweden from Wikidata)
# ---------------------------------------------------------------------------

_KNOWN_FACTUAL_ERRORS = [
    ("nasdaq", "sweden"),    # Nasdaq HQ is NYC, not Sweden (Wikidata P17 error from Stockholm merger)
    ("nasdaq", "schweden"),  # German variant
]


def _has_known_error(fact_text: str) -> bool:
    """Return True if fact_text contains a known incorrect KG fact pair."""
    text_lower = fact_text.lower()
    return any(a in text_lower and b in text_lower for a, b in _KNOWN_FACTUAL_ERRORS)


# ---------------------------------------------------------------------------
# CCI (Cognitive Complexity Index) computation
# ---------------------------------------------------------------------------

def compute_cci_fields(
    template: Dict[str, Any],
    extracted_facts: Optional[List[Dict]] = None,
    is_comparative: bool = False,
) -> Dict[str, Any]:
    """Compute CCI fields for a QA pair.

    Args:
        template: Template dict (must have ``reasoning_depth`` or ``steps``).
        extracted_facts: List of extracted fact dicts (with ``chain_num`` key).
        is_comparative: Whether the template is comparative (2-entity).

    Returns:
        Dict with ``num_entities``, ``num_properties``, ``reasoning_depth``,
        ``knowledge_span``, ``cci``, and ``num_chains_used``.
    """
    # --- reasoning depth (d) ---
    d = template.get("reasoning_depth") or template.get("steps") or 1

    # --- entities (E) and properties (P) ---
    req = (
        template.get("required_properties")
        or template.get("required_metrics")
        or template.get("required_data")
        or template.get("required_indicators")
        or []
    )
    # Date-based templates (e.g. artist_age_at_creation) have required_dates
    # but empty required_metrics; count dates as property lookups too.
    if not req:
        req = template.get("required_dates") or []
    E = 2 if is_comparative else 1
    P = max(len(req), 1)  # every question needs at least 1 property lookup

    # --- CCI = E + P ---
    k = E + P
    cci = E + P

    # --- chains used ---
    num_chains = 0
    if extracted_facts:
        num_chains = len({f.get("chain_num") for f in extracted_facts if f.get("chain_num") is not None})

    return {
        "num_entities": E,
        "num_properties": P,
        "reasoning_depth": d,
        "knowledge_span": k,
        "cci": cci,
        "num_chains_used": num_chains,
    }


# ---------------------------------------------------------------------------
# Phase 1: Chain-based clue fact extraction
# ---------------------------------------------------------------------------

def extract_chain_clue_facts(chains, entity_label, agent_info,
                             min_distinct_properties=3):
    """Extract and validate clue facts from KG chains (Phase 1).

    Runs LLM-based fact extraction from multi-hop chains, then validates
    each extracted triple against the chain data.

    Args:
        chains: KG chain dicts from fetch_multihop_triples.
        entity_label: Human-readable entity name.
        agent_info: (lm, tokenizer, client) tuple.
        min_distinct_properties: Minimum number of distinct property labels
            required across extracted facts (default 3).

    Returns:
        (extracted_facts, None) tuple, or (None, None) on failure.
        extracted_facts is a list of validated fact dicts.
    """
    if not chains or len(chains) < 3:
        return None, None

    formatted_chains = "\n".join(
        f"Chain {i+1}: {c['path_description']}" for i, c in enumerate(chains)
    )

    # Phase 1: Extract facts from KG chains via LLM
    fact_prompt = WIKIDATA_FACT_EXTRACTION_PROMPT.format(
        seed_entity=entity_label,
        chains_text=formatted_chains,
    )
    try:
        fact_response = gen_from_prompt_harmony(
            fact_prompt,
            temperature=1.0,
            max_tokens=8196,
            developer_content=WIKIDATA_FACT_EXTRACTION_DEVELOPER,
            reasoning_effort="medium",
        )
    except Exception as e:
        print(f"    Phase 1 failed: {e}", flush=True)
        return None, None

    # Parse facts
    try:
        fact_json = extract_json_v2(fact_response, None)
        if isinstance(fact_json, dict):
            raw_facts = fact_json.get("facts", [])
        elif isinstance(fact_json, list):
            raw_facts = fact_json
        else:
            print(f"    Phase 1: unexpected format", flush=True)
            return None, None
    except Exception as e:
        print(f"    Phase 1: JSON parse failed: {e}", flush=True)
        return None, None

    # Ensure every fact has a fact_id
    for fi, f in enumerate(raw_facts):
        if "fact_id" not in f:
            cn = f.get("chain_num", fi + 1)
            f["fact_id"] = f"C{cn}_F1"

    # Validate each fact triple against chain data
    extracted_facts = []
    for fact in raw_facts:
        chain_num = fact.get('chain_num', 0)
        fact_text = fact.get('fact', '')

        # Fix 2: Reject facts containing non-Latin text
        if _NON_LATIN_RE.search(fact_text):
            print(f"      Rejected {fact.get('fact_id', '?')}: non-Latin text in fact",
                  flush=True)
            continue

        # Fix 3: Reject known factual errors from KG
        if _has_known_error(fact_text):
            print(f"      Rejected {fact.get('fact_id', '?')}: known factual error in fact",
                  flush=True)
            continue

        is_valid, hop_idx = validate_triple_in_chains(
            chain_num, fact.get('entity', ''), fact.get('property', ''),
            fact.get('value', ''), chains, target_entity=entity_label,
        )
        if is_valid:
            fact["hop_idx"] = hop_idx
            # For hop >= 1 facts, record the intermediate entity so downstream
            # checks can verify it is preserved in the question text.
            if hop_idx is not None and hop_idx >= 1 and 1 <= chain_num <= len(chains):
                chain_hops = chains[chain_num - 1].get("chain", [])
                if hop_idx < len(chain_hops):
                    ie_label = chain_hops[hop_idx].get("entity", {}).get("label", "")
                    fact["intermediate_entity"] = ie_label
                    # If the fact text omits the intermediate entity (common
                    # with collapsed hop-1 extractions), augment it so that
                    # the question composer preserves the multi-hop
                    # attribution and passes the 3d4 flattening check.
                    fact_text = fact.get("fact", "")
                    if ie_label and ie_label.lower() not in fact_text.lower():
                        h0_prop = chain_hops[0].get("property", {}).get("label", "")
                        if h0_prop:
                            fact["fact"] = (
                                f"Its {h0_prop}, {ie_label}, {fact_text}"
                            )
            extracted_facts.append(fact)
        else:
            print(f"      Rejected {fact.get('fact_id', '?')}: triple not found in chain {chain_num} "
                  f"({fact.get('entity', '')} -[{fact.get('property', '')}]-> {fact.get('value', '')})",
                  flush=True)

    if len(extracted_facts) < 3:
        print(f"    Phase 1: Only {len(extracted_facts)} validated facts (need 3+)", flush=True)
        return None, None

    fact_chains = {f['chain_num'] for f in extracted_facts}
    print(f"    Phase 1: {len(extracted_facts)} validated clue facts across "
          f"{len(fact_chains)} chains", flush=True)

    if len(fact_chains) < 3:
        print(f"    Phase 1: Facts span only {len(fact_chains)} chains (need 3+)", flush=True)
        return None, None

    # Fix 4: Enforce clue property diversity
    distinct_props = {f.get('property', '') for f in extracted_facts}
    distinct_props.discard('')  # ignore empty property strings
    if len(distinct_props) < min_distinct_properties:
        print(f"    Phase 1: Only {len(distinct_props)} distinct properties "
              f"(need {min_distinct_properties}+): {distinct_props}",
              flush=True)
        return None, None

    return extracted_facts, None


# ---------------------------------------------------------------------------
# Phase 1.5: Wikipedia grounding verification
# ---------------------------------------------------------------------------

def verify_facts_grounding(extracted_facts, chains, articles, agent_info):
    """Phase 1.5: Verify extracted facts against Wikipedia grounding documents.

    Args:
        extracted_facts: Validated facts from Phase 1.
        chains: KG chain dicts.
        articles: Wikipedia article dicts.
        agent_info: (lm, tokenizer, client) tuple.

    Returns:
        Grounded facts list, or None if insufficient.
    """
    if not articles:
        print(f"    Phase 1.5: No Wikipedia articles available, skipping grounding check",
              flush=True)
        return extracted_facts

    use_harmony = get_harmony_generator() is not None
    print(f"    Phase 1.5: Verifying facts against grounding documents...", flush=True)
    grounded_facts, rejected_grounding_facts = verify_facts_against_grounding(
        extracted_facts, chains, articles, use_harmony,
        agent_info=agent_info if not use_harmony else None,
    )
    grounded_fact_chains = {f['chain_num'] for f in grounded_facts}
    print(f"    Phase 1.5: {len(grounded_facts)}/{len(extracted_facts)} facts grounded "
          f"across {len(grounded_fact_chains)} chains", flush=True)

    if len(grounded_facts) < 3 or len(grounded_fact_chains) < 3:
        print(f"    Phase 1.5: Insufficient grounded facts — "
              f"{len(grounded_facts)} facts across {len(grounded_fact_chains)} chains",
              flush=True)
        return None

    return grounded_facts


# ---------------------------------------------------------------------------
# Poisoned fact filtering
# ---------------------------------------------------------------------------

def filter_poisoned_facts(
    extracted_facts: List[Dict[str, Any]],
    poisoned_props: set,
) -> List[Dict[str, Any]]:
    """Remove facts whose ``(property, value)`` pair has been poisoned.

    *poisoned_props* is a set of ``(property_label, value)`` tuples
    collected from facts used in previously rejected QAs.
    """
    if not poisoned_props:
        return extracted_facts
    return [
        f for f in extracted_facts
        if (f.get("property", ""), f.get("value", "")) not in poisoned_props
    ]


# Phase-3 rejection reasons for which the (property, value) pair itself is the
# problem, so excluding it on later attempts is the corrective action.  Only
# 3d3b (one property supplies >50% of the used facts) qualifies: dropping that
# property's pairs forces the next attempt to draw clues from other properties.
# Every other reason (name/value leaks, flattening, answer recomputation, KG
# ambiguity, chain-spanning, vague time, …) is about this question's phrasing or
# the specific combination of facts — the facts are innocent, so poisoning them
# only starves sparse entities.  Those simply retry (the loop's next attempt,
# with temperature=1.0 fact extraction + composition, explores a fresh question).
_POISON_WORTHY_PREFIXES = ("3d3b:",)


def should_poison_on_rejection(reason: str) -> bool:
    """Return True iff a Phase-3 rejection reason warrants excluding the used
    ``(property, value)`` pairs from later attempts (see _POISON_WORTHY_PREFIXES).
    """
    return (reason or "").strip().startswith(_POISON_WORTHY_PREFIXES)


def poison_used_facts(reason, used_facts, extracted_facts, poisoned_props):
    """Add the ``(property, value)`` pairs of *used_facts* to *poisoned_props* —
    but only when *reason* is poison-worthy (see :func:`should_poison_on_rejection`).

    A no-op for phrasing / answer / combination rejections, so those don't ban
    innocent facts and starve sparse entities.  Mutates *poisoned_props* in place.
    """
    if not should_poison_on_rejection(reason):
        return
    facts_by_id = {f["fact_id"]: f for f in extracted_facts if "fact_id" in f}
    for fid in used_facts:
        f = facts_by_id.get(fid)
        if f:
            poisoned_props.add((f.get("property", ""), f.get("value", "")))


# ---------------------------------------------------------------------------
# Cross-entity fact filtering (for comparative QAs)
# ---------------------------------------------------------------------------

def filter_cross_entity_facts(
    facts: List[Dict[str, Any]],
    other_entity_names: List[str],
) -> List[Dict[str, Any]]:
    """Remove facts that mention the *other* entity in a comparative pair.

    This prevents cross-contamination where entity A's clue facts reference
    entity B (e.g. from shared KG chain traversals), which would leak
    the other entity's identity.

    Only considers name strings >= 3 characters to avoid false matches
    on short common substrings.
    """
    if not other_entity_names:
        return facts

    other_lower = [n.lower() for n in other_entity_names if len(n) >= 3]
    if not other_lower:
        return facts

    clean: List[Dict[str, Any]] = []
    for f in facts:
        text = " ".join(
            str(f.get(k, "")) for k in ("fact", "value", "entity")
        ).lower()
        if any(name in text for name in other_lower):
            continue
        clean.append(f)
    return clean


# ---------------------------------------------------------------------------
# 1. Answer recomputation
# ---------------------------------------------------------------------------


def check_answer_recomputation(
    computation_code: str,
    gold_answer: str,
) -> Tuple[bool, str]:
    """Re-execute *computation_code* and verify it reproduces *gold_answer*."""
    recomputed = _execute_computation_code(computation_code)
    if recomputed is None or recomputed != gold_answer:
        return False, f"Recomputation mismatch ({recomputed} != {gold_answer})"
    return True, ""


# ---------------------------------------------------------------------------
# 2. Entity-name leak
# ---------------------------------------------------------------------------

_CORPORATE_SUFFIXES = re.compile(
    r',?\s*\b(?:Inc\.?|Corp\.?|Co\.?|Ltd\.?|LLC|L\.?P\.?|PLC|S\.A\.?|N\.V\.?'
    r'|Group|Holdings|Company|Incorporated|Corporation|Limited)\s*$',
    re.IGNORECASE,
)

_COMMON_NAME_WORDS = {
    # Corporate / geographic (financial domain)
    "american", "united", "national", "general", "first", "international",
    "global", "western", "eastern", "northern", "southern", "central",
    "pacific", "atlantic", "royal", "imperial", "standard", "applied",
    "advanced", "digital", "energy", "power", "health", "service",
    "services", "capital", "financial", "technology", "systems", "industries",
    "resources", "solutions", "partners", "group", "holdings",
    # Security / cyber domain — generic terms that appear in vuln/incident names
    "vulnerability", "overflow", "bypass", "exploitation", "execution",
    "injection", "buffer", "remote", "privilege", "credential", "backdoor",
    "breach", "attacks", "attack", "supply", "chain",
    # Generic incident/attack descriptor terms
    "campaign", "theft", "rapid", "escalation", "editor", "equation",
    "reset", "exchange", "proxy", "utils", "pipe", "dirty",
    # Common tech / vendor names — too generic to identify a single entity
    "apache", "microsoft", "windows", "google", "linux", "oracle",
    "framework", "kernel", "server", "protocol", "network", "security",
    "software", "desktop",
    "cisco", "citrix", "fortinet", "forti", "zimbra", "winrar",
    "confluence", "outlook", "weblogic", "moveit", "solarwinds",
    "log4j", "mshtml",
    # History domain — generic event/entity type words
    "treaty", "battle", "siege", "conquest", "revolution", "revolt",
    "dynasty", "empire", "kingdom", "republic", "confederation",
    "expedition", "exploration", "voyage",
    "plague", "pandemic", "famine",
    "great", "grand", "world",
    # History domain — common English words that appear in entity names
    # (e.g., "years" from "Thirty Years' War", "berlin" from "Fall of the
    #  Berlin Wall", "space" from "Space Shuttle Challenger disaster")
    "years", "civil", "independence", "declaration", "peace", "spring",
    "movement", "berlin", "paris", "london", "moscow", "rome", "vienna",
    "russian", "french", "english", "spanish", "chinese", "indian",
    "german", "portuguese", "mexican", "egyptian", "ottoman", "roman",
    "african", "european", "asian", "ocean", "olympic", "nuclear",
    "atomic", "space", "shuttle", "canal", "disaster", "earthquake",
    "hurricane", "eruption", "flood", "structure", "discovery",
    "project", "program", "conference", "accords", "pact", "march",
    "summer", "winter", "sinking", "printing", "press", "telegraph",
    "industrial", "scientific", "cultural", "colonial", "ancient",
    "modern", "medieval", "classical", "third", "second", "first",
    "thirty", "hundred", "seven", "young",
    "death", "flight", "south", "north", "east", "west",
    "mount", "saint", "island", "river", "point", "bible",
    # Financial domain — industry-generic words in company names
    "surgical", "sportswear", "instruments", "entertainment",
    "pharmaceutical", "therapeutics", "diagnostics", "automotive",
    "semiconductor", "wireless", "renewable", "manufacturing",
    "logistics", "freight", "hospitality", "restaurant", "petroleum",
    "communications", "biotech", "aerospace", "insurance",
    "bancorp", "bancorporation", "properties", "realty", "brands",
    "foods", "beverages", "utilities", "materials", "metals",
    "chemical", "gaming", "motors", "pharma", "credit",
    "electric", "electronics",
}


def check_entity_name_leak(
    question: str,
    entity_names: List[str],
    clue_terms: Optional[set] = None,
) -> Tuple[bool, str]:
    """Check whether any identifier in *entity_names* appears in *question*.

    Also checks base names with corporate suffixes stripped (e.g.
    ``"S&P Global Inc."`` → also checks ``"S&P Global"``).

    *clue_terms* is an optional set of lower-cased words that originate from
    extracted-fact entities/values (e.g. founder "Huntsman", intermediate
    entity "Olin").  If a partial-name word appears in *clue_terms*, it is
    considered a legitimate clue reference rather than a name leak.

    Each domain passes its own identifiers:
    - Multiskill: ``[entity_label]``
    - Financial: ``[company_name, ticker]``
    - Security: ``[entity_name, cve_id]``
    - History: ``[entity_name, qid]``
    - Biochem: ``[entity_name]``
    """
    clue_terms = clue_terms or set()
    q_lower = question.lower()
    for name in entity_names:
        if not name:
            continue
        names_to_check = {name.lower()}
        base = _CORPORATE_SUFFIXES.sub("", name).strip()
        if base and len(base) > 2 and base.lower() != name.lower():
            names_to_check.add(base.lower())
        for n in names_to_check:
            if len(n) < 5:
                # Short names (tickers like "S", "FMC") — use word-boundary match
                # to avoid false positives from substring hits
                if re.search(r'\b' + re.escape(n) + r'\b', q_lower):
                    return False, f"Entity identifier '{n}' leaked in question"
            else:
                if n in q_lower:
                    return False, f"Entity identifier '{n}' leaked in question"

        # Per-word check: detect partial name leaks (e.g. "Berkshire" from
        # "Berkshire Hathaway Inc.")
        if base and len(base) > 2:
            for word in base.lower().split():
                # Strip trailing/leading punctuation (e.g. "Congo," → "Congo")
                word = word.strip(".,;:!?'\"()")
                if len(word) < 5:
                    continue
                if word in _COMMON_NAME_WORDS:
                    continue
                # Skip words that originate from extracted-fact clues (e.g.
                # founder "Huntsman" when entity is "Huntsman Corp")
                if word in clue_terms:
                    continue
                if re.search(r'\b' + re.escape(word) + r'\b', q_lower):
                    return False, f"Partial name word '{word}' leaked in question"
    return True, ""


# ---------------------------------------------------------------------------
# 3. Quantitative-value leak
# ---------------------------------------------------------------------------

def check_value_leak(
    question: str,
    values: Sequence[Union[int, float]],
    min_len: int = 3,
) -> Tuple[bool, str]:
    """Check whether any quantitative *values* appear in *question*.

    Values are checked in multiple string formats: int, comma-separated, and
    float with 1–2 decimal places.
    """
    for val in values:
        if val is None or not isinstance(val, (int, float)):
            continue
        val_strs: set[str] = set()
        try:
            if val == int(val):
                val_strs.add(str(int(val)))
                val_strs.add(f"{val:,.0f}")
            else:
                val_strs.add(f"{val:.1f}")
                val_strs.add(f"{val:.2f}")
        except (ValueError, OverflowError):
            pass
        val_strs.add(str(val))
        for vs in val_strs:
            if len(vs) > min_len and vs in question:
                return False, f"Entity property value {vs} leaked in question"
    return True, ""


# ---------------------------------------------------------------------------
# 4. Clue-bypass detection  (critical — the missing check in sub-benchmarks)
# ---------------------------------------------------------------------------

_QUANT_KW_PATTERNS = [
    r'\busing\s+the\b', r'\bwhat\s+(?:is|would|will|was)\b',
    r'\bcalculate\b', r'\bcompute\b', r'\bestimate\b',
    r'\bhow\s+(?:far|much|many)\b', r'\bdetermine\b',
]


# ---------------------------------------------------------------------------
# 4b. Property semantics guard  (check 3d3)
# ---------------------------------------------------------------------------

PROPERTY_SEMANTIC_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "found in taxon": {
        "disallowed": [
            "produced by", "synthesized by", "made by", "created by",
            "manufactured by", "sourced from", "originates from",
            "derives from", "obtained from", "secreted by",
            "reported from", "reported in", "associated with",
            "linked to", "attributed to",
        ],
        "reason": "P703 'found in taxon' means detected in organism, not produced/synthesized by it",
    },
}


def check_property_semantics(
    question: str,
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Check that the question does not misrepresent the semantics of Wikidata properties.

    For each used fact whose ``property`` label is in
    :data:`PROPERTY_SEMANTIC_CONSTRAINTS`, verify that none of the
    disallowed phrasings appear in the question alongside the fact's value.
    """
    facts_by_id: Dict[str, Dict[str, Any]] = {
        f["fact_id"]: f for f in extracted_facts if "fact_id" in f
    }
    q_lower = question.lower()

    for fid in used_facts:
        fact = facts_by_id.get(fid)
        if not fact:
            continue
        prop_label = fact.get("property", "").strip().lower()
        constraint = PROPERTY_SEMANTIC_CONSTRAINTS.get(prop_label)
        if not constraint:
            continue
        value = fact.get("value", "").strip().lower()
        if not value:
            continue
        for phrase in constraint["disallowed"]:
            if phrase in q_lower and value in q_lower:
                return (
                    False,
                    f"Property '{prop_label}' value '{value}' misrepresented "
                    f"as '{phrase}' — {constraint['reason']}",
                )
    return True, ""


# ---------------------------------------------------------------------------
# 4b1b. Property concentration guard  (check 3d3b)
# ---------------------------------------------------------------------------

def check_property_concentration(
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
    max_fraction: float = 0.5,
) -> Tuple[bool, str]:
    """Reject if a single Wikidata property contributes >*max_fraction* of used facts.

    When all identifying clues come from one property (e.g. P703 "found in taxon"),
    the entity is rarely uniquely identifiable — many entities share the same
    set of taxa.
    """
    if len(used_facts) < 2:
        return True, ""

    facts_by_id: Dict[str, Dict[str, Any]] = {
        f["fact_id"]: f for f in extracted_facts if "fact_id" in f
    }

    prop_counts: Dict[str, int] = {}
    total = 0
    for fid in used_facts:
        fact = facts_by_id.get(fid)
        if not fact:
            continue
        prop = fact.get("property", "").strip().lower()
        if not prop:
            continue
        prop_counts[prop] = prop_counts.get(prop, 0) + 1
        total += 1

    if total < 2:
        return True, ""

    for prop, count in prop_counts.items():
        if count / total > max_fraction:
            return (
                False,
                f"Property concentration: '{prop}' contributes {count}/{total} "
                f"({count/total:.0%}) of used facts (max {max_fraction:.0%})",
            )
    return True, ""


# ---------------------------------------------------------------------------
# 4c. Multi-hop property flattening guard  (check 3d4)
# ---------------------------------------------------------------------------

_FLATTENING_DEVELOPER = (
    "You are a knowledge-graph multi-hop reasoning auditor. You verify that a "
    "question keeps the intermediate entity in a two-hop relationship rather "
    "than collapsing (flattening) a far-hop property directly onto the subject."
)

_FLATTENING_PROMPT = """A benchmark question is ABOUT a subject entity. Some clues are two-hop: the subject links to an INTERMEDIATE entity, and that INTERMEDIATE has a far VALUE via some property. Such a clue is valid only if the question keeps the intermediate in the reasoning path.

Subject entity (what the question is about): "{target}"

Two-hop clues to audit — each is  SUBJECT -> [INTERMEDIATE] -[property]-> VALUE :
{items_text}

QUESTION:
{question}

For each item, choose a verdict:
- PRESERVED: the question keeps the intermediate in the reasoning path — it either names the INTERMEDIATE, or refers to it unambiguously (e.g. by its type plus a defining descriptor, such as "the continent whose discoverer is X"), so a solver must still resolve the intermediate to use VALUE. ALSO PRESERVED when the property is genuinely transitive, so attributing VALUE to the subject is factually correct (e.g. "located in <region/country>" inherited down a containment chain, or a mountain's range inherited from the park that contains it).
- FLATTENED: the question attributes VALUE (or its property) directly to the SUBJECT as if it were the subject's own property, the INTERMEDIATE is absent from the question, AND the property is not transitive — so the attribution is not warranted (e.g. calling the subject itself "a World Heritage Site" when it is the containing park that holds that designation).

Respond in JSON — one entry per item:
{{
  "results": [
    {{"fact_id": "C1_F1", "verdict": "PRESERVED"}},
    {{"fact_id": "C2_F1", "verdict": "FLATTENED", "reason": "..."}}
  ]
}}
Only use PRESERVED or FLATTENED as verdicts."""


def check_multihop_flattening(
    question: str,
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
    gen_fn: Optional[Callable[..., str]] = None,
    entity_label: str = "",
) -> Tuple[bool, str]:
    """Detect hop>=1 far-hop properties flattened directly onto the target entity.

    A two-hop clue (target -> intermediate -[p]-> value) is valid only if the
    question keeps the intermediate in the reasoning path.  The hard case — the
    question uses the far *value* but does not literally name the *intermediate*
    — cannot be adjudicated by string matching: it conflates genuine flattening
    ("this monolith is a World Heritage Site", non-transitive, wrong) with valid
    paraphrase ("the continent whose discoverer is Columbus") and transitive
    inheritance ("located in Nevada").  We therefore hand exactly those
    ambiguous facts to an LLM judge.

    Cheap by construction: a fact is a candidate only when its far value appears
    in the question AND the intermediate is not literally named.  A question that
    names its intermediates produces no candidates and makes no LLM call.  Fails
    open (returns ``(True, "")``) when *gen_fn* is absent or the LLM/JSON fails —
    the KG-uniqueness gate and V2 remain as downstream backstops.
    """
    facts_by_id: Dict[str, Dict[str, Any]] = {
        f["fact_id"]: f for f in extracted_facts if "fact_id" in f
    }
    q_lower = question.lower()

    candidates: List[Dict[str, Any]] = []
    for fid in used_facts:
        fact = facts_by_id.get(fid)
        if not fact:
            continue
        hop_idx = fact.get("hop_idx")
        if hop_idx is None or hop_idx < 1:
            continue
        intermediate = fact.get("intermediate_entity", "").strip()
        if not intermediate:
            continue
        value = fact.get("value", "").strip()
        if not value or value.lower() not in q_lower:
            continue
        # Intermediate literally named → unambiguously preserved, no LLM needed.
        if intermediate.lower() in q_lower:
            continue
        candidates.append(fact)

    if not candidates:
        return True, ""

    # Ambiguous facts remain — only an LLM can separate flattening from valid
    # paraphrase / transitive inheritance.  Fail open if no generator.
    if gen_fn is None:
        return True, ""

    items = []
    for f in candidates:
        items.append(
            f'[{f.get("fact_id", "?")}] INTERMEDIATE: "{f.get("intermediate_entity", "")}"'
            f'  property: "{f.get("property", "")}"  VALUE: "{f.get("value", "")}"'
        )
    prompt = _FLATTENING_PROMPT.format(
        target=entity_label or "(the question's subject)",
        items_text="\n".join(items),
        question=question,
    )
    try:
        response = gen_fn(
            prompt, temperature=0.0, max_tokens=2048,
            developer_content=_FLATTENING_DEVELOPER,
        )
        parsed = extract_json_v2(response, None)
    except Exception as e:
        print(f"    3d4 flattening judge failed (LLM error): {e} — passing", flush=True)
        return True, ""  # fail-open

    if isinstance(parsed, dict):
        results = parsed.get("results", [])
    elif isinstance(parsed, list):
        results = parsed
    else:
        return True, ""  # unparseable — fail open

    cand_by_id = {f.get("fact_id", ""): f for f in candidates}
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("verdict", "").upper() == "FLATTENED":
            fid = r.get("fact_id", "")
            fact = cand_by_id.get(fid, {})
            return (
                False,
                f"Fact {fid}: far value '{fact.get('value', '')}' "
                f"(via intermediate '{fact.get('intermediate_entity', '')}', "
                f"property '{fact.get('property', '')}') is flattened onto "
                f"'{entity_label}' — {r.get('reason', 'intermediate omitted')}",
            )
    return True, ""


def check_clue_bypass(
    question: str,
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
    entity_label: str,
    reference_params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Detect fact-derived named entities leaking into the quantitative portion
    of the question (enabling solvers to bypass entity identification).

    *used_facts* is a list of ``fact_id`` strings.  *extracted_facts* is the
    full list of fact dicts (with ``fact_id``, ``value``, ``entity`` keys).
    *reference_params* is an optional dict of external/reference values that are
    expected in the quantitative portion (e.g. reference cities).
    """
    # Locate the quantitative portion of the question
    quant_start = len(question)
    for pat in _QUANT_KW_PATTERNS:
        m = re.search(pat, question, re.IGNORECASE)
        if m and m.start() < quant_start:
            quant_start = m.start()

    if quant_start >= len(question):
        return True, ""  # No quantitative portion detected — nothing to check

    quant_portion = question[quant_start:].lower()

    # Collect fact-derived named values
    used_fact_values: set[str] = set()
    for fid in used_facts:
        for f in extracted_facts:
            if f.get("fact_id") == fid:
                val = str(f.get("value", "")).strip()
                if val and len(val) > 2 and not val.replace(",", "").replace(".", "").isdigit():
                    used_fact_values.add(val)
                ent = str(f.get("entity", "")).strip()
                if ent and len(ent) > 2 and ent.lower() != entity_label.lower():
                    used_fact_values.add(ent)

    # Reference parameters are external — OK to name in the quantitative portion
    if reference_params:
        for ref_key in ("reference_city", "reference_name", "reference_river",
                        "reference_structure", "reference_location", "reference_body"):
            ref_val = reference_params.get(ref_key, "")
            if ref_val:
                used_fact_values.discard(ref_val)

    for val in used_fact_values:
        if val.lower() in quant_portion:
            return (
                False,
                f"Clue value '{val}' appears in quantitative portion "
                f"— solver can bypass entity identification",
            )
    return True, ""


# ---------------------------------------------------------------------------
# 5. Vague time references
# ---------------------------------------------------------------------------

_VAGUE_TIME_PATTERNS = [
    r'\blatest\b', r'\bmost recent\b', r'\bcurrent\b',
    r'\bup[- ]to[- ]date\b', r'\brecent\b', r'\bpresent[- ]day\b',
]


def check_vague_time_references(question: str) -> Tuple[bool, str]:
    """Reject questions containing vague temporal language."""
    q_lower = question.lower()
    for pattern in _VAGUE_TIME_PATTERNS:
        if re.search(pattern, q_lower):
            return False, f"Vague time reference found: '{pattern}'"
    return True, ""


# ---------------------------------------------------------------------------
# 6. Year-mismatch check
# ---------------------------------------------------------------------------

def check_year_mismatch(
    question: str,
    data_year: Optional[int],
) -> Tuple[bool, str]:
    """If *data_year* is set and the question mentions years, ensure the data
    year is among them."""
    if not data_year:
        return True, ""
    years_in_question = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', question)]
    if years_in_question and data_year not in years_in_question:
        return (
            False,
            f"Year mismatch — question mentions {years_in_question} but data is from {data_year}",
        )
    return True, ""


# ---------------------------------------------------------------------------
# 7. Fact grounding (LLM-based)
# ---------------------------------------------------------------------------

def check_fact_grounding(
    question: str,
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
    gen_fn: Callable[..., str],
) -> Tuple[bool, str]:
    """LLM-based: verify all question clues come from source facts."""
    used_fact_texts: list[str] = []
    for fid in used_facts:
        for f in extracted_facts:
            if f.get("fact_id") == fid:
                used_fact_texts.append(f.get("fact", ""))
                break

    if not used_fact_texts:
        return True, ""  # Nothing to check

    facts_summary = "\n".join(f"- {ft}" for ft in used_fact_texts)
    grounding_prompt = f"""You are checking whether a question introduces FABRICATED information — i.e., specific factual claims that are NOT supported by (and cannot be reasonably inferred from) the source facts below.

Source facts:
{facts_summary}

Question: {question}

IMPORTANT distinctions:
- ALLOWED: paraphrasing, synonyms, reasonable inferences, generic descriptions (e.g., "a Belgian researcher" from "born in Belgium"), category labels (e.g., "symmetric cipher", "hash function"), and standard domain terminology.
- NOT ALLOWED: specific names, dates, places, numbers, relationships, or events that appear in the question but have NO basis in any source fact.

First, list any specific claims in the question that are clearly fabricated (not supported by or inferable from the source facts). Then conclude with your verdict.

If you found fabricated claims, end with: VERDICT: FAIL
If all specific claims are grounded or reasonably inferable, end with: VERDICT: PASS"""
    try:
        response = gen_fn(
            grounding_prompt, temperature=0.0, max_tokens=4096,
            developer_content="You are a fact-grounding verifier. Focus on catching genuine fabrications, not paraphrasing.",
        ).strip()
        # Check for explicit FAIL verdict
        response_upper = response.upper()
        if "VERDICT: FAIL" in response_upper or "VERDICT:FAIL" in response_upper:
            # Extract the fabricated claims for the failure reason
            lines = response.strip().split("\n")
            fabricated = [l.strip() for l in lines if l.strip() and not l.strip().upper().startswith("VERDICT")]
            summary = fabricated[-1] if fabricated else "fabricated claims detected"
            return False, f"Fact grounding FAILED — {summary}"
        return True, ""
    except Exception as e:
        return True, ""  # Fail-open on error to avoid blocking on transient issues


# ---------------------------------------------------------------------------
# 10. Fact-reference validation (multiskill 3a — chain spanning)
# ---------------------------------------------------------------------------

def check_fact_references(
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
    min_chains: int = 3,
) -> Tuple[bool, str]:
    """Check that *used_facts* all reference valid extracted facts and span at
    least *min_chains* distinct chains."""
    if not used_facts:
        return False, "No used facts provided"

    for fid in used_facts:
        found = any(f.get("fact_id") == fid for f in extracted_facts)
        if not found:
            return False, "Invalid fact references"

    chain_nums: set = set()
    for fid in used_facts:
        for f in extracted_facts:
            if f.get("fact_id") == fid:
                chain_nums.add(f.get("chain_num"))
    if len(chain_nums) < min_chains:
        return False, f"Facts span only {len(chain_nums)} chains (need {min_chains})"

    return True, ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def check_kg_uniqueness(
    used_facts: List[str],
    extracted_facts: List[Dict[str, Any]],
    chains: List[Dict[str, Any]],
    entity_qid: Optional[str] = None,
    max_report: int = 5,
) -> Tuple[bool, str]:
    """Data-grounded uniqueness check (replaces the closed-book 3h/3i gates).

    The clues assert direct facts about the answer entity: each used chain's
    hop-0 triple is ``answer -[p1]-> mid``, and hop-1 clues restate that same
    hop-0 link in their text.  So every clue reduces to a direct constraint
    ``?x wdt:<p1> wd:<mid>`` on the answer.  We ask Wikidata how many entities
    satisfy ALL those constraints at once — if more than one does, the clue set
    is genuinely ambiguous (independent of whether any model can *solve* it).

    Unlike the old ``check_uniqueness`` (which asked a closed-book LLM to name
    the entity — hopeless for search-required multi-hop questions), this uses
    ground-truth KG structure and makes no LLM call.

    Fails open — returns ``(True, "")`` — whenever it cannot build enough
    structural constraints or the SPARQL query errors; V2 still filters those.
    """
    if not used_facts or not extracted_facts or not chains:
        return True, ""

    used_ids = set(used_facts)
    constraints: Dict[Tuple[str, str], bool] = {}
    for f in extracted_facts:
        if f.get("fact_id") not in used_ids:
            continue
        cn = f.get("chain_num", 0)
        if not (1 <= cn <= len(chains)):
            continue
        hops = chains[cn - 1].get("chain") or []
        if not hops:
            continue
        hop0 = hops[0]
        p1 = hop0.get("property", {}).get("id", "")
        mid = hop0.get("value", {}).get("id", "")
        if re.match(r"^P\d+$", p1) and re.match(r"^Q\d+$", mid):
            constraints[(p1, mid)] = True

    # Need at least two independent constraints to say anything about
    # uniqueness; a single common property matches many entities.
    if len(constraints) < 2:
        return True, ""

    triples = " . ".join(f"?x wdt:{p} wd:{m}" for (p, m) in constraints)
    query = f"SELECT DISTINCT ?x WHERE {{ {triples} }} LIMIT 20"
    try:
        rows = sparql_query(query)
    except Exception as e:
        print(f"    KG-uniqueness query error: {e} — skipping", flush=True)
        return True, ""
    if not rows:
        # No matches at all (query hiccup / stale data) — don't reject.
        return True, ""

    matches = [r.get("x", {}).get("value", "").split("/")[-1] for r in rows]
    matches = [m for m in matches if re.match(r"^Q\d+$", m)]
    if len(matches) <= 1:
        return True, ""

    labels = get_entity_labels_batch(matches[:max_report])
    named = ", ".join(f"{labels.get(m, m)} ({m})" for m in matches[:max_report])
    return False, (
        f"KG-ambiguous — {len(matches)} entities match all "
        f"{len(constraints)} clue triples: {named}"
    )


def strip_side_prefix(used_facts: List[str], prefix: str) -> List[str]:
    """Return base fact-ids for one side of a comparative question.

    Comparative composers tag combined clue ids with ``"A_"`` / ``"B_"``
    prefixes so the LLM can attribute each clue to a side; the base id
    (e.g. ``"C1_F1"``) matches that side's ``extracted_facts`` fact_ids.
    Returns the ids that start with *prefix*, with the prefix removed.
    """
    return [u[len(prefix):] for u in used_facts if str(u).startswith(prefix)]


def check_kg_uniqueness_comparative(
    used_a: List[str],
    facts_a: List[Dict[str, Any]],
    chains_a: List[Dict[str, Any]],
    used_b: List[str],
    facts_b: List[Dict[str, Any]],
    chains_b: List[Dict[str, Any]],
    entity_qid_a: Optional[str] = None,
    entity_qid_b: Optional[str] = None,
) -> Tuple[bool, str]:
    """Per-side KG-grounded uniqueness for comparative questions (check 3h).

    A comparative question ("which of A or B has higher X?") is well-posed only
    if BOTH compared entities are uniquely identifiable from their own side's
    clues.  Runs :func:`check_kg_uniqueness` independently on each side and fails
    if EITHER side over-matches.  Each side fails open exactly as the
    single-entity check does (needs >=2 KG constraints, else passes).

    ``used_a`` / ``used_b`` must be BASE fact-ids matching that side's
    ``facts_a`` / ``facts_b`` fact_ids — strip any ``"A_"`` / ``"B_"`` prefix
    with :func:`strip_side_prefix` before calling.
    """
    for label, used, facts, chains, qid in (
        ("A", used_a, facts_a, chains_a, entity_qid_a),
        ("B", used_b, facts_b, chains_b, entity_qid_b),
    ):
        if not (used and facts and chains):
            continue
        ok, reason = check_kg_uniqueness(used, facts, chains, qid)
        if not ok:
            return False, f"comparative side {label}: {reason}"
    return True, ""


def run_phase3_validation(
    *,
    question: str,
    gold_answer: str,
    computation_code: str,
    entity_label: str,
    entity_names: List[str],
    entity_values: Sequence[Union[int, float]] = (),
    used_facts: Optional[List[str]] = None,
    extracted_facts: Optional[List[Dict[str, Any]]] = None,
    gen_fn: Optional[Callable[..., str]] = None,
    reference_params: Optional[Dict[str, Any]] = None,
    data_year: Optional[int] = None,
    entity_type: str = "entity",
    check_facts: bool = False,
    min_chains: int = 3,
    acceptable_identifiers: Optional[List[str]] = None,
    skip_fact_grounding: bool = True,
    chains: Optional[List[Dict[str, Any]]] = None,
    entity_qid: Optional[str] = None,
) -> Tuple[bool, str]:
    """Run all Phase 3 checks in order.  Returns ``(passed, reason)``.

    Parameters
    ----------
    question : str
        The composed question text.
    gold_answer : str
        Expected answer string.
    computation_code : str
        Python code that must reproduce *gold_answer*.
    entity_label : str
        Human-readable label for LLM prompts.
    entity_names : list[str]
        All identifiers to check for name leaks (name, ticker, QID, CVE-ID…).
    entity_values : sequence of int/float
        Quantitative values to check for leaks.
    used_facts : list[str] | None
        Fact-ID strings used in the question.
    extracted_facts : list[dict] | None
        Full extracted-fact dicts with ``fact_id``, ``value``, ``entity``.
    gen_fn : callable | None
        LLM generation function for grounding/uniqueness/ambiguity checks.
        If *None*, the LLM-based checks are skipped.
    reference_params : dict | None
        External/reference parameters exempt from clue-bypass detection.
    data_year : int | None
        Year the data comes from, for year-mismatch validation.
    entity_type : str
        For LLM prompt phrasing (``"entity"``, ``"company"``, etc.).
    check_facts : bool
        Whether to run fact-reference validation (chain-spanning check).
    min_chains : int
        Minimum distinct chains that used_facts must span (when *check_facts*).
    acceptable_identifiers : list[str] | None
        Alternative names that also count as correct for the uniqueness check
        (e.g. underlying CVE-ID, chain_wiki product name).
    skip_fact_grounding : bool
        If True, skip LLM-based fact grounding check (3g) while keeping
        uniqueness (3h) and ambiguity (3i).  Useful for domains where
        questions are already grounded by construction via Phase 1/1.5.
    """
    used_facts = used_facts or []
    extracted_facts = extracted_facts or []

    # 3a: Fact reference validation (optional — multiskill only)
    if check_facts:
        ok, reason = check_fact_references(used_facts, extracted_facts, min_chains)
        if not ok:
            return False, f"3a: {reason}"

    # 3b: Recompute gold answer
    ok, reason = check_answer_recomputation(computation_code, gold_answer)
    if not ok:
        return False, f"3b: {reason}"

    # 3c: Entity name/identifier leak
    # Build clue_terms from extracted-fact entities/values so that legitimate
    # clue-derived words (e.g. founder "Huntsman" when entity is "Huntsman
    # Corp") are not flagged as name leaks.
    clue_terms: set = set()
    for f in extracted_facts:
        for key in ("entity", "value", "intermediate_entity"):
            val = f.get(key, "")
            if val:
                for w in val.lower().split():
                    w = w.strip(".,;:!?'\"()")
                    if len(w) >= 5:
                        clue_terms.add(w)
    ok, reason = check_entity_name_leak(question, entity_names, clue_terms)
    if not ok:
        return False, f"3c: {reason}"

    # 3d: Quantitative value leak
    ok, reason = check_value_leak(question, entity_values)
    if not ok:
        return False, f"3d: {reason}"

    # 3d2: Clue-bypass detection
    if used_facts and extracted_facts:
        ok, reason = check_clue_bypass(
            question, used_facts, extracted_facts, entity_label, reference_params,
        )
        if not ok:
            return False, f"3d2: {reason}"

    # 3d3: Property semantics (keyword-based, kept as fast pre-filter)
    if used_facts and extracted_facts:
        ok, reason = check_property_semantics(question, used_facts, extracted_facts)
        if not ok:
            return False, f"3d3: {reason}"

    # 3d3b: Property concentration (single property dominates clues)
    if used_facts and extracted_facts:
        ok, reason = check_property_concentration(used_facts, extracted_facts)
        if not ok:
            return False, f"3d3b: {reason}"

    # 3d4: Multi-hop property flattening (LLM-judged for the ambiguous cases)
    if used_facts and extracted_facts:
        ok, reason = check_multihop_flattening(
            question, used_facts, extracted_facts,
            gen_fn=gen_fn, entity_label=entity_label,
        )
        if not ok:
            return False, f"3d4: {reason}"

    # 3e: Vague time references
    ok, reason = check_vague_time_references(question)
    if not ok:
        return False, f"3e: {reason}"

    # 3f: Year mismatch
    ok, reason = check_year_mismatch(question, data_year)
    if not ok:
        return False, f"3f: {reason}"

    # 3f2: Entity-type / question-target mismatch
    # When multi-hop KG chains traverse a named-after location (e.g.,
    # "Battle of Verdun" → Verdun city), the LLM composer may phrase the
    # question about the *location* instead of the actual entity.  Reject
    # questions that refer to "this place/location/city" when the entity is
    # not a geographic type.
    _GEOGRAPHIC_ENTITY_TYPES = {
        "mountain", "city", "structure", "country", "location", "island",
        "river", "lake", "desert", "glacier", "peninsula", "plateau",
        "geographic_feature",
    }
    if entity_type not in _GEOGRAPHIC_ENTITY_TYPES:
        q_lower = question.lower()
        # Phrases that indicate the question's *subject* is the traversed
        # place rather than the actual entity.  Anchored deliberately: bare
        # tokens like "municipal", "settlement", "incorporated as" collide with
        # ordinary financial/administrative vocabulary ("municipal bond", "cash
        # settlement", "incorporated as a holding company", "born in the
        # settlement of X") and caused false rejections.  Each phrase below
        # names a place noun or a demonstrative place reference, so it matches
        # only when the place itself is the topic.
        _PLACE_PHRASES = (
            # Demonstrative reference to the place as the question's subject.
            "this place", "this location", "this city", "this town",
            "this village", "this municipality", "this settlement",
            "this commune", "this site",
            # The named-after place described as being founded/incorporated.
            "incorporated as a city", "incorporated as a town",
            "incorporated as a village", "incorporated as a municipality",
            "incorporated as a commune", "incorporated as a settlement",
            "established as a city", "established as a town",
            "established as a village", "established as a municipality",
            "established as a commune", "established as a settlement",
            "become a commune", "became a commune",
        )
        if any(p in q_lower for p in _PLACE_PHRASES):
            return False, (
                f"3f2: question asks about a place/location but entity_type "
                f"is '{entity_type}' — likely named-after confusion"
            )

    # LLM-based checks require gen_fn
    if gen_fn is not None:
        # 3g: Fact grounding (skippable — known ~88% false rejection rate)
        if not skip_fact_grounding and used_facts and extracted_facts:
            ok, reason = check_fact_grounding(
                question, used_facts, extracted_facts, gen_fn,
            )
            if not ok:
                return False, f"3g: {reason}"

    # 3h: KG-grounded uniqueness (data-based). Replaces the old closed-book
    # uniqueness (3h) + ambiguity (3i) LLM gates, which asked a closed-book
    # model to identify the entity — hopeless for search-required multi-hop
    # questions, so they rejected exactly the hard items the benchmark wants.
    # Answerability/uniqueness-via-search is enforced by the V2 filter; here we
    # only reject genuinely over-matching clue sets, using ground-truth KG
    # structure. Runs only when chains are provided (else V2 alone covers it).
    if chains is not None and used_facts and extracted_facts:
        ok, reason = check_kg_uniqueness(used_facts, extracted_facts, chains, entity_qid)
        if not ok:
            return False, f"3h: {reason}"

    return True, ""
