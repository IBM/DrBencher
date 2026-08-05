#!/usr/bin/env bash
#
# One-push installer for DrBencher — sets up everything with uv.
#
#   ./scripts/install.sh
#
# It will:
#   1. install uv itself if it isn't already on your PATH,
#   2. create a project virtualenv (.venv/) and install the full generation +
#      verification stack pinned by pyproject.toml / uv.lock (`uv sync`),
#   3. drop a `set_environment.sh` helper that activates the env.
#
# Afterwards, activate the environment with either of:
#   source .venv/bin/activate
#   source set_environment.sh
set -o errexit

# Always operate from the repo root, regardless of where the script is called from.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- 1. ensure uv is installed --------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found — installing it from https://astral.sh/uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv in ~/.local/bin (or ~/.cargo/bin); make it usable now.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is still not on PATH after install. Open a new shell (or add" >&2
  echo "       ~/.local/bin to your PATH) and re-run ./scripts/install.sh." >&2
  exit 1
fi
echo "Using uv $(uv --version) at $(command -v uv)"

# ---- 2. create the environment + install locked dependencies --------------
# The vLLM/torch stack pulls very large CUDA wheels (hundreds of MB each), which
# routinely blow past uv's default 30s per-download timeout on slower links. Give
# them plenty of headroom; honor an existing override if the caller set one.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

echo "Creating .venv/ and installing dependencies (this can take a while the"
echo "first time — it pulls the vLLM/torch stack; UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}s)..."
# Retry a few times so a single stalled CUDA-wheel download doesn't abort the whole
# install — uv resumes from its cache, so retries only fetch what's still missing.
attempt=1
max_attempts=3
until uv sync; do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "ERROR: 'uv sync' failed after ${max_attempts} attempts." >&2
    echo "       Re-run ./scripts/install.sh to resume (cached wheels are kept), or" >&2
    echo "       raise the timeout further, e.g. UV_HTTP_TIMEOUT=1200 ./scripts/install.sh" >&2
    exit 1
  fi
  echo "uv sync attempt ${attempt} failed (likely a slow download) — retrying..." >&2
  attempt=$((attempt + 1))
done

# ---- 3. leave a convenience activation helper -----------------------------
printf 'source .venv/bin/activate\n' > set_environment.sh

echo ""
echo "Done. Activate the environment with:"
echo "    source .venv/bin/activate      # or: source set_environment.sh"
echo ""
echo "Then generate a benchmark, e.g.:"
echo "    ./scripts/benchmark/run_biochem_bench.sh"
