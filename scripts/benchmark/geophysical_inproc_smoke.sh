#!/bin/bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#
# Geophysical SMOKE test, IN-PROCESS vLLM (the original --use_harmony backend).
#
#   ./scripts/benchmark/geophysical_inproc_smoke.sh
#
# Purpose: establish a known-good baseline on a fresh domain using the ORIGINAL
# in-process backend (model loaded on the GPUs, no `vllm serve`). If this
# generates QA cleanly, the `vllm serve` conversion is the thing that broke QA
# generation, and we can convert this path to serve step by step.
#
# In-process needs GPUs, so this submits a GPU bsub job on a compute node.
# NOTE: compute nodes on this cluster are Wikipedia-403/429-blocked, so the
# article-fetch step may fail here. If it does, that is the egress issue (not
# the QA bug) and we fall back to the serve-split for fetching.
#
# Tunables (env vars, optional):
#   NGPU (4)  GROUP (your LSF group; "" to omit)  QUEUE (cluster default)
#   WALLTIME (4:00)  CATEGORY (hills)  PER_CATEGORY (3)  SAMPLES (2)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

NGPU="${NGPU:-4}"
# Set GROUP to your own LSF fairshare group id for `bsub -G`
# (e.g. `export GROUP=grp_myteam`); leave it unset to submit without -G.
# GROUP="${GROUP-grp_alignment}"   # <-- replace grp_alignment with your group
WALLTIME="${WALLTIME:-4:00}"
CATEGORY="${CATEGORY:-hills}"
PER_CATEGORY="${PER_CATEGORY:-3}"
SAMPLES="${SAMPLES:-2}"
# run_id is the resume key: outputs are written under
# output/geophysical/smoketest__${RUN_ID}__<cat>.*. Re-launching with the SAME
# RUN_ID resumes — the per-category cache checks load any finished bench/v1/v2
# stage and continue (e.g. skip QA-gen+V1, go straight to V2). A fresh RUN_ID
# (the default) starts over (though the timestamp-less chains cache is still
# reused, so fetching is skipped either way).
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOBNAME="geo_inproc_smoke_$$"
mkdir -p LOGS

bsub_args=(-J "$JOBNAME" -gpu "num=${NGPU}" -n "$((NGPU * 2))" -W "$WALLTIME"
           -o "LOGS/${JOBNAME}.%J.out" -e "LOGS/${JOBNAME}.%J.err")
[ -n "${GROUP:-}" ] && bsub_args+=(-G "$GROUP")
[ -n "${QUEUE:-}" ] && bsub_args+=(-q "$QUEUE")

echo "[geo-smoke] Submitting in-process geophysical smoke:"
echo "  GPUs: ${NGPU} (tensor_parallel_size=${NGPU})   category: ${CATEGORY}"
echo "  per_category: ${PER_CATEGORY}   v1/v2 samples: ${SAMPLES}"
echo "  run_id: ${RUN_ID}  (re-launch with RUN_ID=${RUN_ID} to resume)"
echo "  logs: LOGS/${JOBNAME}.<jobid>.{out,err}"

exec bsub "${bsub_args[@]}" \
  bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && \
    python -m drbench.geophysical_drbencher \
      --exp_mode geophysical_bench \
      --use_harmony yes --tensor_parallel_size ${NGPU} \
      --agent_modelname openai/gpt-oss-120b --use_helm no \
      --geophysical_categories ${CATEGORY} \
      --geophysical_per_category ${PER_CATEGORY} \
      --geophysical_v1_samples ${SAMPLES} --geophysical_v2_samples ${SAMPLES} \
      --run_id ${RUN_ID} \
      --outfile_prefix1 output/geophysical/smoketest"
