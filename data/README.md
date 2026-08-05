# DrBencher — Datasets

This directory contains the DrBencher deep-research benchmark: multi-hop questions that
require **identifying a hidden entity** from grounding articles, **retrieving its
quantitative properties**, and **performing a computation** to reach a numeric answer.

Each question is self-contained: the answer is derivable from the supporting material
shipped with the record — the grounding article, the property documents, and the
computation code — without any external lookup.

## Purpose: a self-sufficient release for reproducible evaluation

This release is deliberately packaged to be **self-sufficient**: every value a question
needs to be answered is present in the supporting documents shipped with the record. The
goal is to let users **consistently reproduce results using retrieval tools** — a
retrieval-augmented / tool-using agent can index the packaged documents, retrieve the
relevant evidence, and compute the answer, and get the same result every time.

Why this matters: the underlying facts come from live sources (Wikidata, World Bank, SEC
EDGAR, PubChem, NCBI, NVD, …) whose values **drift over time** — GDP figures are revised,
knowledge-graph entries are edited, filings are restated. An evaluation that queries those
sources live is not reproducible: the same question can yield different answers on
different days, and results are not comparable across systems or over time. By freezing the
exact evidence needed for each answer into the release, we decouple the benchmark from
source drift, so scores are **reproducible, comparable, and independent of external
availability**. Every shipped value was verified against its authoritative source at
release time (see the citation in each property document).

## License

Released under the **Community Data License Agreement – Permissive, Version 2.0
(CDLA-Permissive-2.0)**. See <https://cdla.dev/permissive-2-0/>.

Underlying facts are sourced from public providers cited in each record
(Wikipedia/Wikidata, NCBI, UniProt, PubChem, SEC EDGAR, World Bank, NVD, and the
Metropolitan Museum of Art); those sources retain their own terms.

## Contents

Question counts per domain, for the two evaluation sets: `drbencher/` (the main set)
and `postcolm2026_drbencher/` (an additional set).

| Domain | `drbencher/` | `postcolm2026_drbencher/` |
|---|---:|---:|
| Biochemistry | 34 | 36 |
| Financial / economic | 68 | 17 |
| Geophysical | 59 | 30 |
| History | 59 | 40 |
| Cybersecurity | 35 | 18 |
| **Total** | **255** | **141** |

Files are named `{domain}_eval_questions.jsonl` in `drbencher/` and
`{domain}_postcolm2026_questions.jsonl` in `postcolm2026_drbencher/` (where the
geophysical file uses the stem `geo` in the latter).

## Record format

Each file is [JSON Lines](https://jsonlines.org/) — one JSON object per question with the
following fields:

Each field is either **Gold** — benchmark-authored, verified ground truth — or
**Support** — the evidence corpus provided for solving.

| Field | Role | Type | Description |
|---|---|---|---|
| `qid` | — | string | Unique question id, e.g. `biochem/0`. |
| `question` | **Gold** | string | The natural-language question (describes the entity via oblique clues). |
| `answer` | **Gold** | string | Gold answer (numeric). |
| `answer_unit` | **Gold** | string | Unit of the gold answer (may be empty). |
| `property_lookup` | **Gold** | list | Structured `{entity, property, value, unit, source}` — the gold property values used. |
| `property_chain` | **Gold** | object | The multi-hop knowledge-graph chain of `entity –[property]→ value` hops (`source_triples`, plus `used_facts`/`facts`) from which the question's identifying clues were composed — how the hidden entity is pinned down (distinct from the computation's quantitative values). |
| `computation_code` | **Gold** | string | Reference Python that derives `answer` from the property values (the gold derivation). |
| `entity_document` | **Support** | object | Grounding article used to **identify the entity** (`qid`, `title`, `text`, `url`). |
| `property_documents` | **Support** | list | Documents holding the **quantitative property values**, each with a citation `url`/`source`. |

So the **gold** (benchmark-authored) fields are `question`, `answer` (+ `answer_unit`),
`property_lookup`, `property_chain`, and `computation_code`; the **support** fields are
`entity_document` and `property_documents`.

Evaluation protocol (a separate axis from Gold/Support): a system is given the `question`
together with the support documents and must derive the answer from them. The gold
**answer-key** fields — `answer`, `answer_unit`, `property_lookup`, `property_chain`,
`computation_code` — are withheld from the system and used only for scoring and provenance.
(`computation_code` reproduces `answer` exactly and is provided so results are reproducible
— but it is not a solver input.)

## Scoring

Answers are scored with a **2% relative tolerance**, except **history**, which uses
**exact match** (answers there are years/counts).
