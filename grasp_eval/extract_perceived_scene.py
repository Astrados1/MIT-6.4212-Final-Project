"""Export the scene TiPToP perceived (its pickled cuTAMP environment) as plain JSON.

Must run inside the tiptop pixi environment, because unpickling needs cuTAMP and cuRobo:

    cd $TIPTOP_DIR/tiptop
    pixi run python $PROJECT/grasp_eval/extract_perceived_scene.py \
        --save-dir tiptop_server_outputs/<timestamp> --out perceived_scene.json

For each perceived object it writes the name, cuTAMP type (Movable/Surface/...), the
stored pose, and the world-frame axis-aligned bounding box of its geometry. It also
writes how many M2T2 grasps were associated with each object (perception/grasps.pt),
Gemini's grounded goal atoms and cuTAMP's failure reason (metadata.json). Coordinates
are in TiPToP's world frame, which includes the simulator client's 1.5 cm camera
z-offset (see docs/upstream_notes.md); attribution.py corrects for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _world_vertices(obj) -> np.ndarray:
    mesh = obj.get_trimesh_mesh(process=False, process_color=False)
    verts = np.asarray(mesh.vertices, dtype=float)
    T = np.asarray(obj.get_transform_matrix(), dtype=float)
    return verts @ T[:3, :3].T + T[:3, 3]


def describe(env) -> dict:
    types = {}
    for type_name, objs in env.type_to_objects.items():
        for obj in objs:
            types.setdefault(obj.name, []).append(type_name)
    out = []
    for role, objs in (("movable", env.movables), ("static", env.statics)):
        for obj in objs:
            entry = {"name": obj.name, "role": role, "types": types.get(obj.name, []),
                     "pose": [float(v) for v in obj.pose] if obj.pose is not None else None}
            try:
                v = _world_vertices(obj)
                entry["aabb_min"] = v.min(axis=0).tolist()
                entry["aabb_max"] = v.max(axis=0).tolist()
                entry["aabb_center"] = ((v.min(axis=0) + v.max(axis=0)) / 2).tolist()
            except Exception as e:  # keep going; one odd shape should not lose the rest
                entry["geometry_error"] = f"{type(e).__name__}: {e}"
            out.append(entry)
    return {"env_name": getattr(env, "name", None), "objects": out}


def m2t2_grasp_counts(path: Path) -> dict[str, int]:
    """Number of M2T2 grasps TiPToP associated with each object label (perception/grasps.pt).

    cuTAMP 0.0.6 uses these for an object when there is at least one; otherwise it samples
    heuristic grasps (cutamp/particle_initialization.py), so a zero means "heuristic".
    """
    import torch  # tiptop environment only

    try:
        grasps = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        grasps = torch.load(path, map_location="cpu")
    return {label: int(len(entry["poses"])) for label, entry in grasps.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--save-dir", type=Path, required=True, help="tiptop_server_outputs/<timestamp>")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    import dill  # tiptop environment only

    pkl = args.save_dir / "perception" / "cutamp_env.pkl"
    result = {"save_dir": str(args.save_dir.resolve()), "source": str(pkl)}
    if not pkl.exists():
        result["error"] = "cutamp_env.pkl not found (perception may have failed before scene construction)"
    else:
        with open(pkl, "rb") as f:
            env = dill.load(f)
        result.update(describe(env))
    grasps_pt = args.save_dir / "perception" / "grasps.pt"
    if grasps_pt.exists():
        try:
            result["m2t2_grasp_counts"] = m2t2_grasp_counts(grasps_pt)
        except Exception as e:  # the counts are optional; keep the rest of the export
            result["m2t2_grasp_counts_error"] = f"{type(e).__name__}: {e}"
    meta = args.save_dir / "metadata.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        result["grounded_atoms"] = m.get("perception", {}).get("grounded_atoms")
        result["perception_duration_s"] = m.get("perception", {}).get("duration")
        result["planning_duration_s"] = m.get("planning", {}).get("duration")
        result["planning_failure_reason"] = m.get("planning", {}).get("failure_reason")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
