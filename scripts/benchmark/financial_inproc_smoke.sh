#!/bin/bash
#
# Financial SMOKE test, IN-PROCESS vLLM (the original --use_harmony backend).
#
#   ./scripts/benchmark/financial_inproc_smoke.sh
#
# Purpose: quick end-to-end check on one financial sector using the in-process
# backend (model loaded on the GPUs, no `vllm serve`). Analogous to
# geophysical_inproc_smoke.sh. NOTE: financial is organized by SECTOR
# (--financial_sectors / --financial_per_sector), not category.
#
# In-process needs GPUs, so this submits a GPU bsub job on a compute node.
# NOTE: compute nodes on this cluster are Wikipedia-403/429-blocked, so the
# article-fetch step may fail here. If it does, that is the egress issue (not
# the QA bug) and we fall back to the serve-split for fetching.
#
# Tunables (env vars, optional):
#   NGPU (4)  GROUP (grp_alignment; "" to omit)  QUEUE (cluster default)
#   WALLTIME (4:00)  SECTOR (construction)  PER_SECTOR (3)  SAMPLES (2)  RUN_ID
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

NGPU="${NGPU:-4}"
GROUP="${GROUP-grp_alignment}"
WALLTIME="${WALLTIME:-4:00}"
SECTOR="${SECTOR:-construction}"
PER_SECTOR="${PER_SECTOR:-3}"
SAMPLES="${SAMPLES:-2}"
# run_id is the resume key: outputs are written under
# output/financial/smoketest__${RUN_ID}__<sector>.*. Re-launching with the SAME
# RUN_ID resumes — the per-sector cache checks load any finished bench/v1/v2
# stage and continue (e.g. skip QA-gen+V1, go straight to V2). A fresh RUN_ID
# (the default) starts over (though the timestamp-less chains cache is still
# reused, so fetching is skipped either way).
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOBNAME="fin_inproc_smoke_$$"
mkdir -p LOGS

bsub_args=(-J "$JOBNAME" -gpu "num=${NGPU}" -n "$((NGPU * 2))" -W "$WALLTIME"
           -o "LOGS/${JOBNAME}.%J.out" -e "LOGS/${JOBNAME}.%J.err")
[ -n "${GROUP:-}" ] && bsub_args+=(-G "$GROUP")
[ -n "${QUEUE:-}" ] && bsub_args+=(-q "$QUEUE")

echo "[fin-smoke] Submitting in-process financial smoke:"
echo "  GPUs: ${NGPU} (tensor_parallel_size=${NGPU})   sector: ${SECTOR}"
echo "  per_sector: ${PER_SECTOR}   v1/v2 samples: ${SAMPLES}"
echo "  run_id: ${RUN_ID}  (re-launch with RUN_ID=${RUN_ID} to resume)"
echo "  logs: LOGS/${JOBNAME}.<jobid>.{out,err}"

exec bsub "${bsub_args[@]}" \
  bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && \
    python -m drbench.financial_drbencher \
      --exp_mode financial_bench \
      --use_harmony yes --tensor_parallel_size ${NGPU} \
      --agent_modelname openai/gpt-oss-120b --use_helm no \
      --financial_sectors ${SECTOR} \
      --financial_per_sector ${PER_SECTOR} \
      --financial_v1_samples ${SAMPLES} --financial_v2_samples ${SAMPLES} \
      --run_id ${RUN_ID} \
      --outfile_prefix1 output/financial/smoketest"
