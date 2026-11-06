#!/usr/bin/env bash
# Copy new results, videos, logs and planner outputs from the GPU pod to this computer, repeatedly.
#
#   POD_HOST=193.183.22.62 POD_PORT=1490 bash scripts/backup_from_pod.sh [interval_s]
#
# Incremental: each pass copies only files changed since the previous pass (a marker file on the
# pod), so it stays cheap as the run grows. Copies, never deletes, in either direction. Destination
# is <project>/results/pod_backup/ (gitignored): the pod's results/ tree, /workspace/logs, and from
# each TiPToP output directory the plan, run log, config and git.diff (not the large point clouds).
# Runs until killed; one pass with interval 0.
set -uo pipefail

: "${POD_HOST:?set POD_HOST}"
: "${POD_PORT:?set POD_PORT}"
KEY="${POD_KEY:-$HOME/.ssh/runpod_ed25519}"
KNOWN="${POD_KNOWN_HOSTS:-$HOME/.ssh/runpod_known_hosts}"
INTERVAL="${1:-1200}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/results/pod_backup"
mkdir -p "$DEST"
SSH=(ssh -n -i "$KEY" -o UserKnownHostsFile="$KNOWN" -o StrictHostKeyChecking=yes -o BatchMode=yes
     -o ConnectTimeout=20 -o ServerAliveInterval=15 -p "$POD_PORT" "root@$POD_HOST")

# On the pod: list files newer than the marker (all files on the first pass), tar them to stdout.
REMOTE='set -e
M=/workspace/.backup_marker; NEXT=/workspace/.backup_marker.next
touch "$NEXT"
cd /workspace
NEWER=""; [ -e "$M" ] && NEWER="-newer $M"
{
  find MIT-6.4212-Final-Project/results -type f $NEWER 2>/dev/null
  find logs -type f $NEWER 2>/dev/null
  find tiptop-project/tiptop/tiptop_server_outputs -maxdepth 2 -type f $NEWER \
       \( -name tiptop_plan.json -o -name tiptop_run.log -o -name tiptop.yml -o -name git.diff -o -name metadata.json \) 2>/dev/null
} > /tmp/backup_list.txt
tar cf - --no-same-owner -T /tmp/backup_list.txt 2>/dev/null
mv "$NEXT" "$M"'

while true; do
  stamp=$(date +"%Y-%m-%d %H:%M:%S")
  if "${SSH[@]}" "$REMOTE" | tar xf - --no-same-owner -C "$DEST" 2>/dev/null; then
    n=$(find "$DEST" -type f | wc -l); size=$(du -sh "$DEST" | cut -f1)
    echo "[$stamp] backup ok: $n files, $size in $DEST"
  else
    echo "[$stamp] backup FAILED (pod unreachable?); will retry"
  fi
  [ "$INTERVAL" -gt 0 ] || break
  sleep "$INTERVAL"
done
