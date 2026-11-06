"""Repair the three scene-4 variants whose Rubik's cube is authored under the table.

    cd $TIPTOP_DIR/droid-sim-evals
    uv run --no-project --with usd-core python $PROJECT/scripts/fix_scene_assets.py --assets assets [--check]

In the released asset archive (SHA-256 b791aef3..., docs/runpod_reproduction.md), each scene-4
variant places /World/rubiks_cube at z = 0.0907 m and moves its mesh child RubikCube by a
per-variant offset. Seven variants use a z offset of +0.2006 m (the cube starts 20 cm above the
table and drops onto it). Variants 1, 2 and 3 use -0.1986 m: the cube starts 11 cm below the
tabletop (z = 0.045 m), falls through the void and reaches z = -218 m during settling
(baseline paper run, Nov 5; stage 2, Nov 6). The value looks like a sign error.

This sets those three z offsets to the value the other seven variants use and changes nothing
else: x/y offsets (each variant's layout), orientation, scale and every other object stay as
authored. Originals are copied to assets/originals_pre_fix/ once, the fix is idempotent, and any
file whose current value is not one of the two known values is refused rather than edited.
Prints SHA-256 before and after for provenance. Every repaired variant must pass
`python3 -m grasp_eval.stages scenes --scenes 4 --variants 1 2 3` before use.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

CUBE_MESH = "/World/rubiks_cube/RubikCube"
FIXES = {f"scene4_{v}.usd" for v in (1, 2, 3)}
BAD_Z = -0.19863615566511325  # as released
GOOD_Z = 0.20057768036734341  # every other scene-4 variant
TOL = 1e-6


def plan_fix(z: float) -> str:
    """'fix' (released, broken), 'ok' (already repaired) or 'unknown' (do not touch)."""
    if abs(z - BAD_Z) < TOL:
        return "fix"
    if abs(z - GOOD_Z) < TOL:
        return "ok"
    return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--assets", type=Path, required=True, help="droid-sim-evals assets/ directory")
    p.add_argument("--check", action="store_true", help="report only; change nothing")
    a = p.parse_args(argv)
    from pxr import Gf, Sdf  # usd-core; see the usage line above

    status = 0
    backup = a.assets / "originals_pre_fix"
    for name in sorted(FIXES):
        path = a.assets / name
        layer = Sdf.Layer.FindOrOpen(str(path))
        spec = layer.GetPrimAtPath(CUBE_MESH) if layer else None
        attr = spec.attributes.get("xformOp:transform") if spec else None
        if attr is None:
            print(f"{name}: no {CUBE_MESH} xformOp:transform; not the released file, left untouched")
            status = 1
            continue
        m = Gf.Matrix4d(attr.default)
        state = plan_fix(m[3][2])
        before = sha256(path)
        if state == "ok":
            print(f"{name}: already repaired (z offset {m[3][2]:+.4f}), sha256 {before}")
        elif state == "unknown":
            print(f"{name}: unexpected z offset {m[3][2]:+.6f}; refusing to edit")
            status = 1
        elif a.check:
            print(f"{name}: NEEDS FIX (z offset {m[3][2]:+.4f} puts the cube under the table), sha256 {before}")
            status = 1
        else:
            backup.mkdir(exist_ok=True)
            if not (backup / name).exists():
                shutil.copy2(path, backup / name)
            m.SetRow(3, Gf.Vec4d(m[3][0], m[3][1], GOOD_Z, m[3][3]))
            attr.default = m
            layer.Save()
            print(f"{name}: z offset {BAD_Z:+.4f} -> {GOOD_Z:+.4f}; sha256 {before} -> {sha256(path)} "
                  f"(original kept in {backup / name})")
    return status


if __name__ == "__main__":
    sys.exit(main())
