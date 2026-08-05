#!/bin/bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#
# Submit a DrBencher domain generation run to LSF (bsub) on a GPU node.
# gpt-oss-120b under vLLM needs GPUs, so it cannot run on the login node — this
# wrapper submits the per-domain run_<domain>_bench.sh to the batch scheduler.
#
# Usage:
#   ./scripts/benchmark/bsub_bench.sh biochem
#   NGPU=4 QUEUE=x86_6h ./scripts/benchmark/bsub_bench.sh financial
#
# Tunables (environment variables, all optional):
#   NGPU       number of GPUs        (default 4; also sets TENSOR_PARALLEL_SIZE)
#   GROUP      LSF fairshare group   (-G, default grp_alignment; "" to omit)
#   QUEUE      LSF queue (-q)        (default: LSF's default queue if unset)
#   PROJECT    LSF project (-P)      (default: none)
#   WALLTIME   wall-clock limit (-W) (default 168:00, i.e. HH:MM)
#   CORES_PER_GPU  CPU slots per GPU (default 2; used to derive NCORES)
#   NCORES     CPU slots (-n)        (default NGPU*CORES_PER_GPU, e.g. 8 for 4 GPUs)
#   GPU_OPTS   full -gpu spec        (default "num=${NGPU}"; e.g. "num=8:mode=exclusive_process")
#   JOBNAME    LSF job name (-J)     (default drbench_<domain>)
#
# Logs are written to LOGS/<jobname>.<jobid>.{out,err}.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DOMAIN="${1:-}"
if [ -z "$DOMAIN" ]; then
  echo "usage: ./scripts/benchmark/bsub_bench.sh <domain>" >&2
  echo "  domains: biochem | financial | economics | geophysical | history | security" >&2
  exit 1
fi

RUN_SCRIPT="scripts/benchmark/run_${DOMAIN}_bench.sh"
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "ERROR: no run script for domain '${DOMAIN}' (expected ${RUN_SCRIPT})." >&2
  echo "  domains: biochem | financial | economics | geophysical | history | security" >&2
  exit 1
fi

NGPU="${NGPU:-4}"
GROUP="${GROUP-grp_alignment}"
WALLTIME="${WALLTIME:-168:00}"
# CPU slots scale with GPUs: vLLM tensor-parallel workers are GPU-bound, so a
# few cores per GPU (driver + workers + tokenization) is plenty. Default 2/GPU
# (=> 8 for a 4-GPU run); override CORES_PER_GPU or set NCORES directly.
CORES_PER_GPU="${CORES_PER_GPU:-2}"
NCORES="${NCORES:-$((NGPU * CORES_PER_GPU))}"
GPU_OPTS="${GPU_OPTS:-num=${NGPU}}"
JOBNAME="${JOBNAME:-drbench_${DOMAIN}}"
mkdir -p LOGS

# Assemble bsub args; -q / -P are only added when the caller set them, so we
# otherwise fall through to the cluster's defaults.
bsub_args=(
  -J "$JOBNAME"
  -gpu "$GPU_OPTS"
  -n "$NCORES"
  -W "$WALLTIME"
  -o "LOGS/${JOBNAME}.%J.out"
  -e "LOGS/${JOBNAME}.%J.err"
)
[ -n "${GROUP:-}" ]   && bsub_args+=(-G "$GROUP")
[ -n "${QUEUE:-}" ]   && bsub_args+=(-q "$QUEUE")
[ -n "${PROJECT:-}" ] && bsub_args+=(-P "$PROJECT")

echo "Submitting ${DOMAIN} run to LSF:"
echo "  GPUs (-gpu):     ${GPU_OPTS}   (TENSOR_PARALLEL_SIZE=${NGPU})"
echo "  Group (-G):      ${GROUP:-<none>}"
echo "  Queue (-q):      ${QUEUE:-<cluster default>}"
echo "  Project (-P):    ${PROJECT:-<none>}"
echo "  Walltime (-W):   ${WALLTIME}"
echo "  Cores (-n):      ${NCORES}"
echo "  Logs:            LOGS/${JOBNAME}.<jobid>.{out,err}"
echo ""

# The job activates the uv-managed env and runs the domain script. Tensor-parallel
# size is matched to the GPU count so vLLM shards across exactly what LSF granted.
exec bsub "${bsub_args[@]}" \
  bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && TENSOR_PARALLEL_SIZE='${NGPU}' bash '${RUN_SCRIPT}'"
