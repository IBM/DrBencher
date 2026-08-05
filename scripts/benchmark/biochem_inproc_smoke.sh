#!/bin/bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#
# Biochem SMOKE test, IN-PROCESS vLLM (the original --use_harmony backend).
#
#   ./scripts/benchmark/biochem_inproc_smoke.sh
#
# Purpose: run biochem with the in-process backend (model loaded on the GPUs, no
# `vllm serve`) to isolate whether biochem's stall is specific to the vllm-serve
# path. Analogous to geophysical_inproc_smoke.sh.
#
# In-process needs GPUs, so this submits a GPU bsub job on a compute node.
# NOTE: compute nodes on this cluster are Wikipedia-403/429-blocked, so the
# article-fetch step may fail here. If it does, that is the egress issue (not
# the QA bug) and we fall back to the serve-split for fetching.
#
# Tunables (env vars, optional):
#   NGPU (4)  GROUP (grp_alignment; "" to omit)  QUEUE (cluster default)
#   WALLTIME (4:00)  CATEGORY (neurotransmitters)  PER_CATEGORY (3)  SAMPLES (2)  RUN_ID
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

NGPU="${NGPU:-4}"
GROUP="${GROUP-grp_alignment}"
WALLTIME="${WALLTIME:-4:00}"
CATEGORY="${CATEGORY:-neurotransmitters}"
PER_CATEGORY="${PER_CATEGORY:-3}"
SAMPLES="${SAMPLES:-2}"
# run_id is the resume key: outputs are written under
# output/biochem/smoketest__${RUN_ID}__<cat>.*. Re-launching with the SAME
# RUN_ID resumes — the per-category cache checks load any finished bench/v1/v2
# stage and continue (e.g. skip QA-gen+V1, go straight to V2). A fresh RUN_ID
# (the default) starts over (though the timestamp-less chains cache is still
# reused, so fetching is skipped either way).
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOBNAME="biochem_inproc_smoke_$$"
mkdir -p LOGS

bsub_args=(-J "$JOBNAME" -gpu "num=${NGPU}" -n "$((NGPU * 2))" -W "$WALLTIME"
           -o "LOGS/${JOBNAME}.%J.out" -e "LOGS/${JOBNAME}.%J.err")
[ -n "${GROUP:-}" ] && bsub_args+=(-G "$GROUP")
[ -n "${QUEUE:-}" ] && bsub_args+=(-q "$QUEUE")

echo "[biochem-smoke] Submitting in-process biochem smoke:"
echo "  GPUs: ${NGPU} (tensor_parallel_size=${NGPU})   category: ${CATEGORY}"
echo "  per_category: ${PER_CATEGORY}   v1/v2 samples: ${SAMPLES}"
echo "  run_id: ${RUN_ID}  (re-launch with RUN_ID=${RUN_ID} to resume)"
echo "  logs: LOGS/${JOBNAME}.<jobid>.{out,err}"

exec bsub "${bsub_args[@]}" \
  bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && \
    python -m drbench.biochem_drbencher \
      --exp_mode biochem_bench \
      --use_harmony yes --tensor_parallel_size ${NGPU} \
      --agent_modelname openai/gpt-oss-120b --use_helm no \
      --biochem_categories ${CATEGORY} \
      --biochem_per_category ${PER_CATEGORY} \
      --biochem_v1_samples ${SAMPLES} --biochem_v2_samples ${SAMPLES} \
      --run_id ${RUN_ID} \
      --outfile_prefix1 output/biochem/smoketest"
