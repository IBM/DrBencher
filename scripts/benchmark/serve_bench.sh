#!/bin/bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#
# Run a DrBencher domain against a SHARED gpt-oss-120b vLLM server.
#
# Instead of loading the model in-process (one 4-GPU allocation per domain), one
# `vllm serve` job answers all domains. The first call launches the shared server;
# later calls reuse it — so running all six domains costs one GPU allocation, not six.
#
# Usage (run from a LOGIN NODE — compute nodes are Wikipedia-blocked):
#   ./scripts/benchmark/serve_bench.sh biochem
#   ./scripts/benchmark/serve_bench.sh financial      # reuses the same server
#   ./scripts/benchmark/serve_bench.sh stop           # bkill the shared server
#
# How it works:
#   1. Find a RUN/PEND LSF job named `gptoss_serve`. Reuse it if healthy; wait for
#      it if still starting; submit one if none.
#   2. Wait until the server answers /v1/models.
#   3. Run run_<domain>_bench.sh HERE on the login node (torch-free) pointed at it.
#   4. Leave the server running for the next domain.
#
# Tunables (env vars, optional):
#   NGPU (4)  PORT (8000)  MODEL (openai/gpt-oss-120b)  GPU_MEM (0.9)
#   GROUP (your LSF group; "" to omit)  QUEUE (cluster default)  WALLTIME (168:00)
#   STARTUP_TIMEOUT (2700)  JOBNAME (gptoss_serve)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

NGPU="${NGPU:-4}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
GPU_MEM="${GPU_MEM:-0.9}"
# Set GROUP to your own LSF fairshare group id for `bsub -G`
# (e.g. `export GROUP=grp_myteam`); leave it unset to submit without -G.
# GROUP="${GROUP-grp_alignment}"   # <-- replace grp_alignment with your group
WALLTIME="${WALLTIME:-168:00}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2700}"
JOBNAME="${JOBNAME:-gptoss_serve}"
mkdir -p LOGS

DOMAIN="${1:-}"
DOMAINS="biochem | financial | economics | geophysical | history | security"

# --- serve_bench.sh stop : tear the shared server down ----------------------
if [ "$DOMAIN" = "stop" ]; then
  IDS="$(bjobs -noheader -o 'id job_name' 2>/dev/null | awk -v n="$JOBNAME" '$2==n{print $1}')"
  if [ -z "$IDS" ]; then echo "No '${JOBNAME}' job running."; exit 0; fi
  echo "Stopping ${JOBNAME} job(s): ${IDS}"
  # shellcheck disable=SC2086
  bkill $IDS
  exit 0
fi

if [ -z "$DOMAIN" ]; then
  echo "usage: ./scripts/benchmark/serve_bench.sh <domain>|stop" >&2
  echo "  domains: ${DOMAINS}" >&2
  exit 1
fi

RUN_SCRIPT="scripts/benchmark/run_${DOMAIN}_bench.sh"
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "ERROR: no run script for domain '${DOMAIN}' (expected ${RUN_SCRIPT})." >&2
  echo "  domains: ${DOMAINS}" >&2
  exit 1
fi

# --- must run on a login node (clean Wikipedia egress) ----------------------
CLIENT_HOST="$(hostname)"
case "$CLIENT_HOST" in
  login*|*login*) : ;;
  *) echo "" >&2
     echo "[serve_bench] ##########################################################" >&2
     echo "[serve_bench] WARNING: '${CLIENT_HOST}' is NOT a login node. Wikipedia" >&2
     echo "[serve_bench] browser.search is 403/429-blocked on compute nodes. Run" >&2
     echo "[serve_bench] this FROM A LOGIN NODE (not inside a bsub/interactive job)." >&2
     echo "[serve_bench] ##########################################################" >&2
     echo "" >&2 ;;
esac

# --- helpers ----------------------------------------------------------------
# Host of a RUN job, cleaned to a bare hostname.
job_host() {
  bjobs -noheader -o 'exec_host' "$1" 2>/dev/null \
    | tr ' ' '\n' | head -1 | sed 's/^[0-9]*\*//' | cut -d: -f1
}

# --- 1. discover-or-launch the shared server --------------------------------
# Existing job named $JOBNAME (RUN takes priority, else PEND) so back-to-back
# invocations share one server instead of each submitting their own.
JOBID="$(bjobs -noheader -o 'id stat job_name' 2>/dev/null \
         | awk -v n="$JOBNAME" '$3==n && $2=="RUN"{print $1; exit}')"
[ -z "$JOBID" ] && JOBID="$(bjobs -noheader -o 'id stat job_name' 2>/dev/null \
         | awk -v n="$JOBNAME" '$3==n && $2=="PEND"{print $1; exit}')"

if [ -n "$JOBID" ]; then
  echo "[serve_bench] Reusing existing ${JOBNAME} job ${JOBID}."
else
  # --return-tokens-as-token-ids: HarmonyServeGenerator reads output token ids
  # back from /v1/completions (required by the Harmony token round-trip).
  serve_args=(-J "$JOBNAME" -gpu "num=${NGPU}" -n "$((NGPU * 2))" -W "$WALLTIME"
              -o "LOGS/${JOBNAME}.%J.out" -e "LOGS/${JOBNAME}.%J.err")
  [ -n "${GROUP:-}" ] && serve_args+=(-G "$GROUP")
  [ -n "${QUEUE:-}" ] && serve_args+=(-q "$QUEUE")
  echo "[serve_bench] No ${JOBNAME} job — submitting (${NGPU} GPUs, port ${PORT})..."
  SUBMIT_OUT="$(bsub "${serve_args[@]}" \
    bash -c "cd '$REPO_ROOT' && source .venv/bin/activate && \
      vllm serve ${MODEL} --tensor-parallel-size ${NGPU} --port ${PORT} \
      --gpu-memory-utilization ${GPU_MEM} --return-tokens-as-token-ids")"
  echo "$SUBMIT_OUT"
  JOBID="$(echo "$SUBMIT_OUT" | grep -oE '[0-9]+' | head -1)"
  [ -n "$JOBID" ] || { echo "[serve_bench] ERROR: could not parse job id." >&2; exit 1; }
fi

# --- 2. wait for RUN + discover the server host -----------------------------
echo "[serve_bench] Waiting for job ${JOBID} to start on a GPU node..."
HOST=""
for _ in $(seq 1 360); do
  STAT="$(bjobs -noheader -o 'stat' "$JOBID" 2>/dev/null || true)"
  if [ "$STAT" = "RUN" ]; then
    HOST="$(job_host "$JOBID")"
    [ -n "$HOST" ] && break
  fi
  if [ "$STAT" = "EXIT" ] || [ "$STAT" = "DONE" ]; then
    echo "[serve_bench] ERROR: server job ended early (STAT=$STAT). See LOGS/${JOBNAME}.${JOBID}.err" >&2
    exit 1
  fi
  sleep 5
done
[ -n "$HOST" ] || { echo "[serve_bench] ERROR: job never reached RUN with a host." >&2; exit 1; }
BASE_URL="http://${HOST}:${PORT}"
echo "[serve_bench] Server on ${HOST}; base url ${BASE_URL}"

# --- 3. wait until the server is healthy ------------------------------------
echo "[serve_bench] Waiting for vLLM to answer /v1/models (up to ${STARTUP_TIMEOUT}s)..."
READY=0
for _ in $(seq 1 "$((STARTUP_TIMEOUT / 5))"); do
  if curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
[ "$READY" = "1" ] || { echo "[serve_bench] ERROR: server not healthy in ${STARTUP_TIMEOUT}s. See LOGS/${JOBNAME}.${JOBID}.out" >&2; exit 1; }
echo "[serve_bench] Server healthy."

# --- 4. run the torch-free domain client HERE on the login node -------------
# DRBENCH_ON_NODE=1 stops run_<domain>_bench.sh from self-submitting to a GPU node;
# DRBENCHER_NO_TORCH=1 keeps the client torch-free; VLLM_SERVE_URL selects the
# serve backend inside the run script.
echo "[serve_bench] Running ${DOMAIN} client on ${CLIENT_HOST} against ${BASE_URL}/v1 ..."
export DRBENCH_ON_NODE=1
export DRBENCHER_NO_TORCH=1
export VLLM_SERVE_URL="${BASE_URL}/v1"
export VLLM_SERVE_MODEL="${MODEL}"
bash "$RUN_SCRIPT"

echo ""
echo "[serve_bench] Done — output under ./output/${DOMAIN}/"
echo "[serve_bench] Shared server (job ${JOBID}) left RUNNING for the next domain."
echo "[serve_bench] Stop it with: ./scripts/benchmark/serve_bench.sh stop   (or: bkill ${JOBID})"
