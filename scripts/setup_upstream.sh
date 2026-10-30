#!/usr/bin/env bash
# Clone tiptop, M2T2 and droid-sim-evals into $TIPTOP_DIR at the pinned revisions.
#
# Usage:  TIPTOP_DIR=/path/to/workspace bash scripts/setup_upstream.sh
#
# Existing checkouts are never deleted or reset. If a repo already exists, the
# script only reports its current revision and whether it matches the pin.
# Dependency installation (pixi/uv) is a separate, manual step: see docs/setup.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../provenance/pins.env
source "$REPO_ROOT/provenance/pins.env"

: "${TIPTOP_DIR:?Set TIPTOP_DIR to the parent directory for the upstream repositories}"
mkdir -p "$TIPTOP_DIR"

clone_pinned() {
    local name="$1" url="$2" sha="$3" recurse="$4"
    local dest="$TIPTOP_DIR/$name"

    if [ -d "$dest/.git" ]; then
        local head
        head="$(git -C "$dest" rev-parse HEAD)"
        if [ -n "$sha" ] && [ "$head" != "$sha" ]; then
            echo "WARNING: $dest is at $head, pin is $sha. Leaving it untouched." >&2
        else
            echo "$name already present at $head"
        fi
        return
    fi
    if [ -e "$dest" ]; then
        echo "ERROR: $dest exists but is not a git checkout. Refusing to overwrite." >&2
        exit 1
    fi

    echo "==> Cloning $name"
    git clone "$url" "$dest"
    if [ -n "$sha" ]; then
        git -C "$dest" checkout --detach "$sha"
    else
        echo "NOTE: no pin for $name; using $(git -C "$dest" rev-parse HEAD). Record it in provenance/pins.env."
    fi
    if [ "$recurse" = "yes" ]; then
        git -C "$dest" submodule update --init --recursive
    fi
}

clone_pinned tiptop "$TIPTOP_URL" "$TIPTOP_SHA" no
clone_pinned M2T2 "$M2T2_URL" "$M2T2_SHA" no
clone_pinned droid-sim-evals "$SIM_URL" "$SIM_SHA" yes

echo
echo "Upstream checkouts in $TIPTOP_DIR:"
for r in tiptop M2T2 droid-sim-evals; do
    printf '  %-16s %s\n' "$r" "$(git -C "$TIPTOP_DIR/$r" rev-parse HEAD)"
done
echo "Next: install dependencies as described in docs/setup.md, then run scripts/record_provenance.sh."
