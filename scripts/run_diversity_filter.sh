#!/bin/bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#
# Apply the diversity filter to already-generated *_v2_filtered.json files.
# For each domain under output/<domain>/, merges the per-category v2_filtered
# files (if not already merged), then runs the diversity filter in place.
#
# Diversity filtering is a CPU job (sentence-transformer embeddings), so run it
# on a compute node via bsub — NOT on the login node (torch import is killed
# there):
#   bsub -J diversity_filter -n 4 -M 64G -W 2:00 -G <your_lsf_group> \
#        -o ./LOGS/diversity_filter.%J.out -e ./LOGS/diversity_filter.%J.err \
#        bash scripts/run_diversity_filter.sh
#
# Or directly (on a node with the env):
#   bash scripts/run_diversity_filter.sh [--min-distance 0.3] [--method graph|greedy] \
#        [--dir ./output] [domain ...]
#
# Env overrides: PYTHON, MIN_DISTANCE, METHOD, OUTPUT_DIR.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p ./LOGS

PYTHON=${PYTHON:-.venv/bin/python}
MIN_DISTANCE=${MIN_DISTANCE:-0.3}
METHOD=${METHOD:-graph}
OUTPUT_DIR=${OUTPUT_DIR:-./output}

# Parse flags; anything else is treated as a domain name.
DOMAINS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --min-distance) MIN_DISTANCE="$2"; shift 2 ;;
        --method)       METHOD="$2";       shift 2 ;;
        --dir)          OUTPUT_DIR="$2";   shift 2 ;;
        *)              DOMAINS+=("$1");   shift ;;
    esac
done

if [ ${#DOMAINS[@]} -eq 0 ]; then
    DOMAINS=(biochem economics financial geophysical history security)
fi

for domain in "${DOMAINS[@]}"; do
    # Support both "geophysical" and "some/path/geophysical" as arguments.
    if [[ "$domain" == */* ]]; then
        dir="$domain"
        domain_name="$(basename "$domain")"
    else
        dir="${OUTPUT_DIR}/${domain}"
        domain_name="$domain"
    fi
    merged="${dir}/${domain_name}_bench_v2_filtered_merged.json"

    if [ ! -d "$dir" ]; then
        echo "SKIP: ${dir} does not exist"
        continue
    fi

    # ── Step 1: merge per-category v2_filtered files if the merged file is absent ──
    # Per-category files look like <prefix>__<run_id>__<category>.<domain>_bench_v2_filtered.json
    per_cat_files=( "${dir}"/*."${domain_name}_bench_v2_filtered.json" )
    if [ ! -f "$merged" ] && [ ${#per_cat_files[@]} -gt 0 ] && [ -f "${per_cat_files[0]}" ]; then
        echo "Merging ${#per_cat_files[@]} per-category files → ${merged}"
        ${PYTHON} -c "
import json, glob
files = sorted(glob.glob('${dir}/*.${domain_name}_bench_v2_filtered.json'))
merged = []
for f in files:
    merged.extend(json.load(open(f)))
print(f'  Merged {len(merged)} questions from {len(files)} files')
json.dump(merged, open('${merged}', 'w'), indent=2)
"
    fi

    if [ ! -f "$merged" ]; then
        echo "SKIP: ${merged} not found and no per-category files to merge"
        continue
    fi

    n_before=$(${PYTHON} -c "import json; print(len(json.load(open('${merged}'))))")
    echo "=== ${domain_name}: ${n_before} questions before filter ==="

    # ── Step 2: run the diversity filter in place ──
    bad_file="${merged%.json}_bad.json"

    ${PYTHON} -m drbench.diversity \
        --benchmark_file "${merged}" \
        --filter \
        --method "${METHOD}" \
        --min_distance "${MIN_DISTANCE}" \
        --text_keys question \
        --output "${merged}"

    n_good=$(${PYTHON} -c "import json; print(len(json.load(open('${merged}'))))")
    n_bad=0
    [ -f "${bad_file}" ] && n_bad=$(${PYTHON} -c "import json; print(len(json.load(open('${bad_file}'))))")
    echo "=== ${domain_name}: ${n_good} good + ${n_bad} bad (from ${n_before}) ==="
    echo ""
done

echo "Done. Filtered files written under ${OUTPUT_DIR}/."
