"""Pass/fail rules for the staged GPU checks (docs/staged_testing.md).

Pure functions on plain dicts and lists, standard library only, so they run in the system
python3 on the pod and in CPU tests here. The simulator-side measurements come from
grasp_eval/stage_worker.py; grasp_eval/stages.py applies these rules and writes the reports.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_LIMITS_FILE = PROJECT_ROOT / "provenance" / "simulator-joint-limits_20261105.json"
PLANNER_LIMITS_FILE = PROJECT_ROOT / "provenance" / "planner-joint-limits_20261105.json"

# TiPToP's DROID home and capture poses (configs/tiptop_panda_sim.yml, same in the upstream FR3 config).
Q_HOME = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]
Q_CAPTURE = [-0.034, 0.090, 0.080, -1.319, -0.003, 1.253, 0.030]

# Joint commands the FR3 planner can produce but the simulated Panda cannot reach (0-based joint
# index -> command). 4.219 rad on joint 6 is what baseline pilot run 002 commanded before release.
LIMIT_PROBES = {5: 4.219, 6: 3.0}

# Isaac Sim 5.0 crashes at startup on R595 (github.com/isaac-sim/IsaacSim/issues/619).
# 590.48.01 is the driver the Nov-5 baseline actually ran on; NVIDIA recommended R580.
KNOWN_BAD_DRIVER_MAJORS = {595: "Isaac Sim 5.0 crashes at startup on R595 (IsaacSim issue 619)"}
NEWEST_TESTED_DRIVER_MAJOR = 590

# Libraries whose absence crashed or blinded Isaac Sim in the Nov-4 containers -> apt package.
REQUIRED_GRAPHICS_LIBS = {"libEGL.so.1": "libegl1", "libGL.so.1": "libgl1",
                          "libXt.so.6": "libxt6", "libGLU.so.1": "libglu1-mesa"}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


# --------------------------------------------------------------------------- stage 0: pod

def driver_verdict(version: str) -> tuple[str, str]:
    """(status, why) for an NVIDIA driver version string such as '590.48.01'."""
    m = re.match(r"\s*(\d+)", version or "")
    if not m:
        return FAIL, f"could not read the driver version from {version!r}"
    major = int(m.group(1))
    if major in KNOWN_BAD_DRIVER_MAJORS:
        return FAIL, f"driver {version}: {KNOWN_BAD_DRIVER_MAJORS[major]}. Stop this pod and pick another host"
    if major > NEWEST_TESTED_DRIVER_MAJOR:
        return WARN, (f"driver {version} is newer than any tested with Isaac Sim 5.0 here "
                      f"(newest working: {NEWEST_TESTED_DRIVER_MAJOR}); stage 1 will tell")
    return PASS, f"driver {version}"


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Rows of `nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader`."""
    gpus = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        try:
            cc = float(parts[3])
        except ValueError:
            cc = float("nan")
        gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2], "compute_cap": cc})
    return gpus


def vulkan_verdict(summary_text: str | None) -> tuple[str, str]:
    """Whether `vulkaninfo --summary` sees the NVIDIA GPU rather than only a CPU renderer."""
    if summary_text is None:
        return WARN, "vulkaninfo not installed (apt install vulkan-tools) - Vulkan not checked"
    names = re.findall(r"deviceName\s*=\s*(.+)", summary_text)
    if any("nvidia" in n.lower() for n in names) or ("NVIDIA" in summary_text and not names):
        return PASS, "Vulkan sees the NVIDIA GPU"
    if names:
        seen = sorted(set(n.strip() for n in names))
        cpu = any("llvmpipe" in n.lower() or "swiftshader" in n.lower() for n in seen)
        return FAIL, (f"Vulkan does not see an NVIDIA GPU, only {seen}{' (a CPU renderer)' if cpu else ''}. In the "
                      "container, install libegl1 and libgl1 and set NVIDIA_DRIVER_CAPABILITIES=all")
    return FAIL, "vulkaninfo found no device"


def missing_graphics_libs(ldconfig_text: str) -> dict[str, str]:
    """Required libraries missing from `ldconfig -p` output -> apt package that provides each."""
    return {lib: pkg for lib, pkg in REQUIRED_GRAPHICS_LIBS.items() if lib not in (ldconfig_text or "")}


# --------------------------------------------------------------------------- stage 1: launch + cameras

def frame_problems(stats: dict, min_std: float = 2.0, min_lit_frac: float = 0.05) -> list[str]:
    """Problems in one camera frame summary {shape, mean, std, lit_frac} from the worker.

    A missing renderer gives a uniform (often black) image; real scenes have texture.
    """
    if not stats or not stats.get("shape"):
        return ["no frame"]
    problems = []
    shape = stats["shape"]
    if len(shape) != 3 or shape[-1] not in (3, 4) or min(shape[:2]) < 16:
        problems.append(f"unexpected image shape {shape}")
    if stats.get("std", 0.0) < min_std:
        problems.append(f"image is nearly uniform (std {stats.get('std', 0.0):.2f}): renderer not drawing")
    if stats.get("lit_frac", 0.0) < min_lit_frac:
        problems.append(f"image is almost all black ({100 * stats.get('lit_frac', 0.0):.1f}% lit pixels)")
    return problems


# --------------------------------------------------------------------------- stage 2: scenes

def scene_problems(check: dict, *, z_range=(0.0, 0.6), max_abs_xy: float = 1.2, max_settle_motion_m: float = 0.005,
                   reference_objects: list[str] | None = None) -> tuple[list[str], list[str]]:
    """(failures, warnings) for one scene/variant from the worker's scene check.

    Run preflight.inspection_problems on the same dict for gripper bodies, boxes and goal names.
    Here: every object must start on the table (the Nov-5 paper run found scene-4 cubes at
    z ~ -218 m), inside the workspace, and stay still while the robot holds its pose.
    """
    fails, warns = [], []
    centers = check.get("geometry_centers_after_settle") or check.get("rigid_objects_after_settle") or {}
    if not centers:
        fails.append("no rigid objects in the scene")
    for name, c in sorted(centers.items()):
        if len(c) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in c):
            fails.append(f"{name}: position is not finite ({c})")
            continue
        x, y, z = c
        if not z_range[0] <= z <= z_range[1]:
            where = "below the table" if z < z_range[0] else "far above the table"
            fails.append(f"{name}: center z = {z:.3f} m, {where} (expected {z_range[0]}-{z_range[1]} m)")
        if abs(x) > max_abs_xy or abs(y) > max_abs_xy:
            fails.append(f"{name}: center ({x:.2f}, {y:.2f}) m is outside the workspace")
    for name, d in sorted((check.get("settle_motion_m") or {}).items()):
        if d is not None and d > max_settle_motion_m:
            warns.append(f"{name} still moved {100 * d:.1f} cm while the robot held still (not settled)")
    if reference_objects is not None and centers:
        missing = sorted(set(reference_objects) - set(centers))
        extra = sorted(set(centers) - set(reference_objects))
        if missing or extra:
            warns.append(f"objects differ from this scene's first variant: missing {missing}, extra {extra}")
    for cam, stats in sorted((check.get("frames") or {}).items()):
        for p in frame_problems(stats):
            warns.append(f"{cam} camera: {p}")
    return fails, warns


# --------------------------------------------------------------------------- stage 3: arm

def load_limits(path: Path = SIM_LIMITS_FILE, n: int = 7) -> list[tuple[float, float]]:
    """[(lower, upper)] for the first n arm joints from a provenance joint-limit file.

    Reads the simulator format (a list of {name, lower_rad, upper_rad}) and the planner format
    ({model: {joint_name: {lower, upper}}}, first model).
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [(float(r["lower_rad"]), float(r["upper_rad"])) for r in data[:n]]
    joints = next(iter(data.values()))
    names = sorted((k for k in joints if re.fullmatch(r"panda_joint\d", k)), key=lambda k: int(k[-1]))
    return [(float(joints[k]["lower"]), float(joints[k]["upper"])) for k in names[:n]]


def arm_test_plan(limits: list[tuple[float, float]], *, q_home=Q_HOME, q_capture=Q_CAPTURE, excursion: float = 0.25,
                  margin: float = 0.05, limit_probes: dict[int, float] | None = None) -> list[dict]:
    """Ordered joint targets for the planner-free arm check.

    Small one-joint moves around home (free space above the table), the capture pose, a gripper
    close/open with nothing in the hand, and optional probes that command a joint past the
    simulator's limit (as the FR3 planner did in pilot run 002) to measure where it stops.
    """
    probes = LIMIT_PROBES if limit_probes is None else limit_probes
    home = [float(v) for v in q_home]

    def clip(j, v):
        lo, hi = limits[j]
        return min(max(v, lo + margin), hi - margin)

    plan = [{"label": "home", "q": home, "gripper": 0.0, "kind": "track"}]
    for j in range(7):
        for sign in (+1, -1):
            q = list(home)
            q[j] = clip(j, home[j] + sign * excursion)
            plan.append({"label": f"joint{j + 1}_{'plus' if sign > 0 else 'minus'}", "q": q, "gripper": 0.0,
                         "kind": "track", "joint": j})
        plan.append({"label": f"joint{j + 1}_home", "q": list(home), "gripper": 0.0, "kind": "track"})
    plan.append({"label": "capture", "q": [float(v) for v in q_capture], "gripper": 0.0, "kind": "track"})
    plan.append({"label": "home_again", "q": list(home), "gripper": 0.0, "kind": "track"})
    plan.append({"label": "gripper_close", "q": list(home), "gripper": 1.0, "kind": "gripper_closed"})
    plan.append({"label": "gripper_open", "q": list(home), "gripper": 0.0, "kind": "gripper_open"})
    for j, cmd in sorted(probes.items()):
        q = list(home)
        q[j] = float(cmd)
        plan.append({"label": f"probe_joint{j + 1}_to_{cmd:g}", "q": q, "gripper": 0.0, "kind": "limit_probe",
                     "joint": j})
        plan.append({"label": f"probe_joint{j + 1}_return", "q": list(home), "gripper": 0.0, "kind": "track"})
    return plan


def arm_problems(summary: dict, expected_limits: list[tuple[float, float]], planner_limits=None, *,
                 track_tol_rad: float = 0.02, limit_tol_rad: float = 0.05, live_limit_tol_rad: float = 1e-3,
                 max_object_motion_m: float = 0.01, min_gripper_change: float = 0.3) -> dict:
    """{fails, warns, notes} for the worker's arm summary.

    summary: {"steps": [{label, kind, q_target, q_meas_end, joint?, gripper_obs_end, finger_closure_end}],
              "sim_joint_limits_live": [[lo, hi], ...] | None, "object_motion_m": {name: m},
              "limit_violations": [{step, joint, value}]}
    """
    fails, warns, notes = [], [], []
    steps = summary.get("steps") or []
    if not steps:
        return {"fails": ["the arm check recorded no steps"], "warns": [], "notes": []}

    live = summary.get("sim_joint_limits_live")
    if live:
        for j, ((lo, hi), (elo, ehi)) in enumerate(zip(live, expected_limits)):
            if abs(lo - elo) > live_limit_tol_rad or abs(hi - ehi) > live_limit_tol_rad:
                warns.append(f"joint {j + 1} live limits [{lo:.4f}, {hi:.4f}] differ from the recorded "
                             f"simulator limits [{elo:.4f}, {ehi:.4f}]")
    else:
        warns.append("simulator did not report live joint limits; compared against the recorded ones")
    limits = [tuple(x) for x in live] if live else expected_limits

    worst = 0.0
    for s in steps:
        q_t, q_m = s.get("q_target"), s.get("q_meas_end")
        if q_t is None or q_m is None:
            fails.append(f"{s.get('label')}: no measurement")
            continue
        err = [abs(a - b) for a, b in zip(q_t, q_m)]
        if s["kind"] == "limit_probe":
            j = s["joint"]
            lo, hi = limits[j]
            stop_at = hi if q_t[j] > hi else lo
            reached = q_m[j]
            fr3 = f"; the planner's FR3 model allows {planner_limits[j][1]:.4f}" if planner_limits else ""
            if abs(reached - stop_at) <= limit_tol_rad:
                notes.append(f"joint {j + 1} commanded {q_t[j]:.3f} rad stopped at {reached:.4f} rad, the simulator "
                             f"limit {stop_at:.4f}{fr3}. This is the arm mismatch: FR3 plans that used this range "
                             f"were not executed as planned. Planning for panda_robotiq keeps plans inside it")
            elif (q_t[j] > hi and reached > hi + limit_tol_rad) or (q_t[j] < lo and reached < lo - limit_tol_rad):
                warns.append(f"joint {j + 1} went past the simulator limit {stop_at:.4f} to {reached:.4f} rad: "
                             f"limits are not enforced as recorded")
            else:
                warns.append(f"joint {j + 1} commanded {q_t[j]:.3f} rad stopped at {reached:.4f} rad, short of the "
                             f"limit {stop_at:.4f} (collision or controller limit?) - check the video")
            others = max((e for k, e in enumerate(err) if k != j), default=0.0)
            if others > track_tol_rad:
                warns.append(f"{s['label']}: other joints off by up to {others:.3f} rad while joint {j + 1} was held "
                             f"at its limit")
            continue
        e = max(err)
        worst = max(worst, e)
        if e > track_tol_rad:
            k = err.index(e)
            fails.append(f"{s['label']}: joint {k + 1} ended {e:.3f} rad from its target (tolerance {track_tol_rad})")

    closed = [s for s in steps if s["kind"] == "gripper_closed"]
    opened = [s for s in steps if s["kind"] == "gripper_open"]
    if closed and opened:
        g_c, g_o = closed[-1].get("gripper_obs_end"), opened[-1].get("gripper_obs_end")
        if g_c is None or g_o is None:
            fails.append("gripper state was not observed")
        elif abs(g_c - g_o) < min_gripper_change:
            fails.append(f"gripper did not move: observed {g_o:.3f} open vs {g_c:.3f} closed")
        else:
            notes.append(f"gripper observation {g_o:.3f} open, {g_c:.3f} closed")
        f_c, f_o = closed[-1].get("finger_closure_end"), opened[-1].get("finger_closure_end")
        if f_c is not None and f_o is not None:
            notes.append(f"finger joint closure {f_o:.2f} open, {f_c:.2f} closed (1.0 = pi/4 rad, unvalidated scale)")
    else:
        fails.append("the gripper close/open steps did not run")

    for v in summary.get("limit_violations") or []:
        fails.append(f"{v['step']}: joint {v['joint'] + 1} measured {v['value']:.4f} rad, outside its limits")
    for name, d in sorted((summary.get("object_motion_m") or {}).items()):
        if d is not None and d > max_object_motion_m:
            warns.append(f"{name} moved {100 * d:.1f} cm during the arm check: the arm may have hit it (see video)")
    notes.insert(0, f"largest end-of-move tracking error {worst:.4f} rad over "
                    f"{sum(1 for s in steps if s['kind'] != 'limit_probe')} moves")
    return {"fails": fails, "warns": warns, "notes": notes}
