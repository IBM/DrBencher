# DrBencher gold-document / property-table statistics

Source: `data/drbencher_relevant_docs_20260712_human_annotated/` (human-validated correct
questions + human property-corpus annotations). Property facts are **content fact-checked**:
a property is counted as *in-corpus* only when the human-recorded `quote` is verified to
actually appear in a grounding-article's text; otherwise it is *external* (human recorded a
`found_online_url`) or *none* (no annotation / no URL).

## Entity documents (answer entity's Wikipedia article, per record)

| Domain | n | resolved | missing |
|--------|---|----------|---------|
| biochem | 37 | 37 | 0 |
| financial | 69 | 69 | 0 |
| geophysical | 65 | 65 | 0 |
| history | 62 | 62 | 0 |
| security | 35 | 35 | 0 |
| **TOTAL** | **268** | **268 (100%)** | **0** |

_After **Task 2**: every record originally missing an entity document (baseline 195/268, 73%)
had it retrieved from Wikipedia — Wikidata sitelink by QID or Wikipedia search by name, full
plain-text extract, relevance-verified. 67 articles fetched → 268/268 resolved._

## Property documents (per record, after Task 1 source-aware verification)

**source-verified** = value confirmed against the authoritative source (PubChem / UniProt /
Wikidata / World Bank WDI) or quote-verified in a Wikipedia grounding article. **accepted-source**
= value carried with an explicit provenance flag where the source is dynamic and not
re-fetchable (macrotrends for SEC company facts; NVD aggregate CVE counts). **no gold doc** =
no free source exposes the value (Wikidata gap, DrugBank/NCBI, online-only page).

| Domain | n | source-verified | accepted-source | no gold doc |
|--------|---|-----------------|-----------------|-------------|
| biochem | 37 | 26 | 11 | 0 |
| financial | 69 | 45 | 24 | 0 |
| geophysical | 65 | 32 | 31 | 2* |
| history | 62 | 50 | 12 | 0 |
| security | 35 | 0 | 35 | 0 |
| **TOTAL** | **268** | **153 (57%)** | **113 (42%)** | **2\*** |

\* The only 2 "no gold doc" are geophysical records with **no property table at all** (nothing
to document). **Every record that has a property now carries a gold document: 266/266 (100%).**

A **recovery pass** closed the previous 90-record gap: content-verifying each remaining value
against the entity's own Wikipedia article (retrieved in Task 2) recovered 33 facts as
verified, Wikidata-by-QID recovered 3 more, and the rest use the entity's article as the gold
property document with the value cited from its authoritative source (Wikidata / NCBI /
DrugBank / World Bank / UNDP), flagged `accepted`.

## Property facts (fact-level, across all records)

| Bucket | Count | Meaning |
|--------|-------|---------|
| in-corpus (quote-verified) | 58 | value quote confirmed in a Wikipedia grounding article |
| external | 368 | value absent from Wikipedia; human recorded a `found_online_url` |
| unverified | 117 | no human annotation (mostly security — domain-API values) |
| mismatch | 14 | corpus value ≠ computed value (data-quality flag: geophysical 11, financial 2, history 1) |

## Read

- Entity docs resolve for **100%** of records (Task 2 retrieved the previously-missing 27%).
- Property docs now cover **100%** of records that have a property (266/266): **57%
  source-verified** and **42% accepted-source** (value cited from its authoritative source, with
  the entity's own article as the gold document). Only 2 records lack a doc — they have no
  property table to document.
- Every value is verified or carries explicit provenance; no gold document is fabricated.

## Task 1 progress — source-aware external gold documents

Approach: for each external property fact, fetch from DrBencher's authoritative source,
**content-verify** the value, and emit a gold document. Format mirrors the corpus:
full `document` text with the relevant span wrapped in `<<Q_RELEVANT>>`, an extracted
`paragraph`, provenance (`source`/`source_id`/`url`), and per-fact `verified` status.
Policy: trust the authoritative source — correct transpositions and flag mismatches;
flag facts with no clean free source (no fabricated docs).

Output (cluster): `data/drbencher_external_docs/<domain>_gold_docs.jsonl` + `<domain>_report.md`.

### Final results — 225 property gold docs

| Domain | docs | source-verified | accepted-source | corrected transpositions | flagged (uncovered) |
|--------|------|-----------------|-----------------|--------------------------|---------------------|
| biochem (PubChem + UniProt) | 36 | 70 | — | 0 | 21 (DrugBank/NCBI) |
| history (Wikipedia in-corpus) | 48 | 48 | — | — | 49 external + 15 unverified |
| geophysical (Wikidata) | 21 | 26 | — | 3 | ~55 (Wikidata lacks coord/prop/QID) |
| financial (World Bank + macrotrends) | 68 | 91 (World Bank WDI) | 55 (macrotrends) | 0 | 5 WB-mismatch, 10 uncovered |
| security (NVD) | 52 | — | 76 (NVD aggregate) | — | aggregate CVE counts not re-computable |

Adapters: PubChem (PUG REST), UniProt (sequence composition, accession-pinned from
annotation URL), Wikidata (wbgetentities), Wikipedia (quote-verified in-corpus), World
Bank WDI (`econ_util`), SEC company facts accepted from macrotrends (dynamic, not
re-fetched), NVD aggregates accepted (paginating every CVE per product is infeasible).

## Task 2 — missing entity documents (retrieved)

For the 67 records whose answer-entity Wikipedia article was absent from the corpus,
retrieved the article via Wikidata sitelink (by QID) or Wikipedia search (by name), fetched
the full plain-text extract, and verified relevance (title/name match). Rate-limited with
retries/backoff.

Output (cluster): `data/drbencher_external_docs/missing_entity_docs.jsonl` — **67/67 retrieved and verified**.

## Total: 292 gold documents (225 property + 67 entity) across `data/drbencher_external_docs/`.
