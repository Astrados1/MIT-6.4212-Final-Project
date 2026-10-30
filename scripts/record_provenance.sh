#!/usr/bin/env bash
# Record code, dependency, checkpoint and asset provenance after installation.
#
# Usage:  TIPTOP_DIR=/path/to/workspace bash scripts/record_provenance.sh [output_file]
#         Optional: ASSETS_ZIP=/path/to/assets.zip  ASSETS_URL=<url it came from>
#
# Run again whenever anything is reinstalled or changed. Each pilot run stores the
# sha256 of the provenance file it was given, so keep old files rather than editing them.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../provenance/pins.env
source "$REPO_ROOT/provenance/pins.env"
: "${TIPTOP_DIR:?Set TIPTOP_DIR to the parent directory of the upstream repositories}"
OUT="${1:-$REPO_ROOT/provenance/provenance_$(date -u +%Y%m%dT%H%M%SZ).txt}"

T="$TIPTOP_DIR/tiptop"
M="$TIPTOP_DIR/M2T2"
S="$TIPTOP_DIR/droid-sim-evals"

repo_state() {
    local label="$1" dir="$2" pin="${3:-}"
    if [ ! -d "$dir/.git" ]; then echo "$label: MISSING ($dir)"; return; fi
    local head dirty
    head="$(git -C "$dir" rev-parse HEAD)"
    dirty="$(git -C "$dir" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
    echo "$label: $head  describe=$(git -C "$dir" describe --tags --always 2>/dev/null)  modified_tracked_files=$dirty"
    if [ -n "$pin" ] && [ "$head" != "$pin" ]; then echo "  WARNING: differs from pin $pin"; fi
    if [ "$dirty" != "0" ]; then git -C "$dir" diff --stat | sed 's/^/  /'; fi
}

hash_files() {
    # hash_files <label> <find args...>
    local label="$1"; shift
    echo "$label:"
    find "$@" -type f 2>/dev/null | sort | while read -r f; do
        printf '  %s  %s  (%s bytes)\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f" "$(stat -c %s "$f")"
    done
}

{
    echo "# Provenance record"
    echo "collected_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname: $(hostname)"
    echo "project_repo: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo 'no commits')"

    echo; echo "## Source revisions"
    repo_state tiptop "$T" "$TIPTOP_SHA"
    repo_state droid-sim-evals "$S" "$SIM_SHA"
    repo_state M2T2 "$M" "$M2T2_SHA"
    repo_state cuRobo "$T/curobo"
    repo_state cuTAMP "$T/cutamp"
    echo "droid-sim-evals submodules:"
    git -C "$S" submodule status --recursive 2>&1 | sed 's/^/  /'

    echo; echo "## Installed planner versions (tiptop pixi env)"
    (cd "$T" && pixi run python - <<'PY'
import importlib.metadata as md
for name in ("cutamp", "nvidia-curobo", "curobo", "torch", "sam2", "google-genai"):
    try:
        print(f"  {name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name}: not installed under this name")
try:
    import torch
    print(f"  torch.cuda: {torch.version.cuda}  available={torch.cuda.is_available()}")
except Exception as e:
    print(f"  torch import failed: {e}")
PY
    ) 2>&1
    echo "expected cuTAMP: $EXPECTED_CUTAMP_VERSION"

    echo; echo "## Configuration"
    hash_files "tiptop config" "$T/tiptop/config/tiptop.yml"
    echo "robot.type: $(grep -m1 -E '^\s+type:' "$T/tiptop/config/tiptop.yml" | awk '{print $2}')"
    echo "gemini default model(s) in code:"
    grep -n 'model_id: str = ' "$T/tiptop/perception/gemini.py" | sed 's/^/  /'
    echo "expected gemini model: $EXPECTED_GEMINI_MODEL"
    echo "GOOGLE_API_KEY: $( [ -n "${GOOGLE_API_KEY:-}" ] && echo set || echo unset ) (value never recorded)"

    echo; echo "## Lockfiles"
    hash_files "lockfiles" "$T/pixi.lock" "$M/pixi.lock" "$S/uv.lock"

    echo; echo "## Model checkpoints"
    hash_files "M2T2 weights" "$M" \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' \) -not -path '*/.pixi/*'
    hash_files "SAM2 checkpoints" "$HOME/.cache" "$T" -name 'sam2*.pt' -not -path '*/.pixi/*'

    echo; echo "## Simulation assets"
    if [ -n "${ASSETS_ZIP:-}" ]; then
        echo "assets.zip source URL: ${ASSETS_URL:-<not given>}"
        hash_files "assets archive" "$ASSETS_ZIP"
    else
        echo "assets.zip: not given (set ASSETS_ZIP and ASSETS_URL to record the archive)"
    fi
    hash_files "scene USDs" "$S/assets" -maxdepth 1 -name '*.usd'
} > "$OUT"

echo "Wrote $OUT"
