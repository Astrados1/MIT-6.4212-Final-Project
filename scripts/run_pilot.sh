#!/usr/bin/env bash
# Run the week-one pilot: one fresh TiPToP plan per trial, each in its own process.
#
# Prerequisites: M2T2 server and TiPToP server already running (docs/setup.md, step D),
# and the target object name confirmed with observed_eval.py --inspect-only.
#
# Usage:
#   TIPTOP_DIR=/path/to/workspace TARGET_OBJECT=<rigid object name> bash scripts/run_pilot.sh
# Optional:
#   SCENE=1  VARIANTS="0 1 2 3 4"  TRIALS=3  OUT_NAME=pilot_scene1
#   INSTRUCTION="Put the Rubik's cube in the bowl."  PROVENANCE_FILE=provenance/provenance_<...>.txt
#   SEED_BASE=1000   WS_HOST=localhost  WS_PORT=8765
#
# Every scheduled trial is written to schedule.jsonl before it starts, so crashes
# that never produce a record still appear in the summary as failures.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TIPTOP_DIR:?Set TIPTOP_DIR}"
: "${TARGET_OBJECT:?Set TARGET_OBJECT (see observed_eval.py --inspect-only)}"
SCENE="${SCENE:-1}"
VARIANTS="${VARIANTS:-0 1 2 3 4}"
TRIALS="${TRIALS:-3}"
OUT_NAME="${OUT_NAME:-pilot_scene${SCENE}}"
DEFAULT_INSTRUCTION="Put the Rubik's cube in the bowl."
INSTRUCTION="${INSTRUCTION:-$DEFAULT_INSTRUCTION}"
SEED_BASE="${SEED_BASE:-1000}"
WS_HOST="${WS_HOST:-localhost}"
WS_PORT="${WS_PORT:-8765}"
SIM="$TIPTOP_DIR/droid-sim-evals"
OUT="$REPO_ROOT/results/$OUT_NAME"

if [ -e "$OUT/schedule.jsonl" ]; then
    echo "ERROR: $OUT already has a schedule. Choose a new OUT_NAME rather than mixing pilots." >&2
    exit 1
fi
mkdir -p "$OUT/runs"

PROV_ARGS=()
if [ -n "${PROVENANCE_FILE:-}" ]; then
    PROV_ARGS=(--provenance-file "$(cd "$(dirname "$PROVENANCE_FILE")" && pwd)/$(basename "$PROVENANCE_FILE")")
else
    echo "WARNING: PROVENANCE_FILE not set; records will not link to a provenance file." >&2
fi

# Trial-major order spreads any drift over time (server warm-up, API load) across variants.
for trial in $(seq 0 $((TRIALS - 1))); do
    for variant in $VARIANTS; do
        run_id="s${SCENE}v${variant}t${trial}_$(date +%Y%m%d-%H%M%S)"
        seed=$((SEED_BASE + 100 * variant + trial))
        mkdir -p "$OUT/runs/$run_id"

        RUN_ID="$run_id" SCENE="$SCENE" VARIANT="$variant" TRIAL="$trial" SEED="$seed" \
        INSTRUCTION="$INSTRUCTION" TARGET_OBJECT="$TARGET_OBJECT" python3 - >>"$OUT/schedule.jsonl" <<'PY'
import json, os
from datetime import datetime, timezone
print(json.dumps({
    "run_id": os.environ["RUN_ID"], "scene": int(os.environ["SCENE"]),
    "variant": int(os.environ["VARIANT"]), "trial": int(os.environ["TRIAL"]),
    "seed": int(os.environ["SEED"]), "instruction": os.environ["INSTRUCTION"],
    "target_object": os.environ["TARGET_OBJECT"],
    "scheduled_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, sort_keys=True))
PY

        echo "==> $run_id (scene $SCENE, variant $variant, trial $trial, seed $seed)"
        (cd "$SIM" && uv run python "$REPO_ROOT/grasp_eval/observed_eval.py" \
            --sim-root "$SIM" --tiptop-dir "$TIPTOP_DIR/tiptop" \
            --scene "$SCENE" --variant "$variant" --trial "$trial" --seed "$seed" \
            --instruction "$INSTRUCTION" --target-object "$TARGET_OBJECT" \
            --run-id "$run_id" --out-dir "$OUT" --ws-host "$WS_HOST" --ws-port "$WS_PORT" \
            "${PROV_ARGS[@]}") >"$OUT/runs/$run_id/console.log" 2>&1
        code=$?
        printf '%s\t%s\n' "$run_id" "$code" >>"$OUT/exit_codes.tsv"
        echo "    exit code $code"
    done
done

echo
(cd "$REPO_ROOT" && python3 -m grasp_eval.summarize "$OUT")
