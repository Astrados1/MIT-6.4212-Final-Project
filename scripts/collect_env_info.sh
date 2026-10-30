#!/usr/bin/env bash
# Record the execution machine: OS, GPU, driver, disk, toolchain versions.
#
# Usage:  bash scripts/collect_env_info.sh [output_file]
#         (default: provenance/machine_<hostname>.txt)
#
# Prints no credential values. For GOOGLE_API_KEY it reports only set/unset.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO_ROOT/provenance/machine_$(hostname -s).txt}"

section() { printf '\n## %s\n' "$1"; }
run() { printf '$ %s\n' "$*"; "$@" 2>&1 || printf '(command failed or unavailable: exit %s)\n' "$?"; }

{
    echo "# Execution machine record"
    echo "collected_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname: $(hostname)"

    section "OS"
    run uname -a
    run cat /etc/os-release

    section "GPU and driver"
    run nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
    run nvidia-smi
    run nvcc --version

    section "CPU, memory, disk"
    run nproc
    run free -g
    run df -h "$HOME"
    [ -n "${TIPTOP_DIR:-}" ] && run df -h "$TIPTOP_DIR"

    section "Toolchain"
    run git --version
    run python3 --version
    run pixi --version
    run uv --version

    section "Rendering / headless"
    echo "DISPLAY=${DISPLAY:-<unset>}"
    run vulkaninfo --summary
    echo "(Isaac Sim headless camera rendering is verified by the replay smoke test, not here.)"

    section "Credentials (presence only)"
    if [ -n "${GOOGLE_API_KEY:-}" ]; then echo "GOOGLE_API_KEY: set"; else echo "GOOGLE_API_KEY: unset"; fi
} > "$OUT"

echo "Wrote $OUT"
