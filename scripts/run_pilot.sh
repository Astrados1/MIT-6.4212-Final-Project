#!/usr/bin/env bash
# Run the week-one pilot: one fresh TiPToP plan per trial, each in its own process.
# Thin wrapper around `python3 -m grasp_eval.run_suite pilot` (see docs/probes.md for other suites).
#
# Prerequisites: M2T2 server and TiPToP server already running (docs/setup.md, step D),
# and the target object name confirmed with observed_eval.py --inspect-only.
#
# Usage:
#   TIPTOP_DIR=/path/to/workspace TARGET_OBJECT=<rigid object name> bash scripts/run_pilot.sh
# Optional:
#   SCENE=1  VARIANTS="0 1 2 3 4"  TRIALS=3  OUT_NAME=pilot_scene1
#   INSTRUCTION="Put the Rubik's cube in the bowl."  PROVENANCE_FILE=provenance/provenance_<...>.txt
#   SEED_BASE=1000   WS_HOST=localhost  WS_PORT=8765   DRY_RUN=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TARGET_OBJECT:?Set TARGET_OBJECT (see observed_eval.py --inspect-only)}"
SCENE="${SCENE:-1}"
DEFAULT_INSTRUCTION="Put the Rubik's cube in the bowl."

ARGS=(pilot --scene "$SCENE" --variants ${VARIANTS:-0 1 2 3 4} --trials "${TRIALS:-3}"
      --instruction "${INSTRUCTION:-$DEFAULT_INSTRUCTION}" --target-object "$TARGET_OBJECT"
      --out-name "${OUT_NAME:-pilot_scene${SCENE}}" --seed-base "${SEED_BASE:-1000}"
      --ws-host "${WS_HOST:-localhost}" --ws-port "${WS_PORT:-8765}")
if [ -n "${DRY_RUN:-}" ]; then
    ARGS+=(--dry-run)
else
    : "${TIPTOP_DIR:?Set TIPTOP_DIR}"
    ARGS+=(--tiptop-dir "$TIPTOP_DIR")
fi
if [ -n "${PROVENANCE_FILE:-}" ]; then
    ARGS+=(--provenance-file "$PROVENANCE_FILE")
else
    echo "WARNING: PROVENANCE_FILE not set; records will not link to a provenance file." >&2
fi

cd "$REPO_ROOT"
exec python3 -m grasp_eval.run_suite "${ARGS[@]}"
