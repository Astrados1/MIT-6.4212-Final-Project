"""Staged GPU checks: prove each layer works before paying for the next (docs/staged_testing.md).

    python3 -m grasp_eval.stages pod                                   # 0. the pod itself (incl. a real CUDA test), no installs
    python3 -m grasp_eval.stages sim    --tiptop-dir /opt/tiptop-project  # 1. Isaac Sim starts and renders
    python3 -m grasp_eval.stages scenes --tiptop-dir /opt/tiptop-project  # 2. every scene/variant loads sanely
    python3 -m grasp_eval.stages arm    --tiptop-dir /opt/tiptop-project  # 3. the arm and gripper obey commands

None of these calls TiPToP, M2T2 or Gemini, so they use no API quota. Stage 4 (trials) is the
existing run_suite / recovery_suite. Each stage writes results/stages/<stage>_<utc>/report.json
and report.md, prints one [PASS|WARN|FAIL] line per check, and exits 1 if anything FAILed.
Standard library only, like preflight.py; the simulator work runs in subprocesses through
grasp_eval/stage_worker.py inside the droid-sim-evals environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.preflight import FAIL, PASS, WARN, Report, _sh, gpu_roles, inspection_problems, parse_os_release  # noqa: E402
from grasp_eval.stage_checks import (  # noqa: E402
    PLANNER_LIMITS_FILE, SIM_LIMITS_FILE, arm_problems, container_env, cuda_driver_probe,
    driver_capabilities_verdict, driver_verdict, frame_problems, load_limits, missing_graphics_libs,
    parse_nvidia_smi_csv, scene_problems, vulkan_verdict,
)

WORKER = PROJECT_ROOT / "grasp_eval" / "stage_worker.py"
STAGES = ("pod", "sim", "scenes", "arm")
CRASH_HINT = ("native crash (exit 139). On a driver-595 host this is the Isaac Sim 5.0 startup crash: "
              "run `python3 -m grasp_eval.stages pod`, and use a host with driver 590 or older")
CUDA_PROBE = ("import torch\nprint('torch', torch.__version__, 'CUDA', torch.version.cuda)\n"
              "assert torch.cuda.device_count() > 0, 'no CUDA device'\n"
              "x = torch.ones((32, 32), device='cuda')\nassert (x @ x).sum().item() == 32768\n"
              "print('CUDA_MATMUL_PASSED', torch.cuda.get_device_name(0))")


# --------------------------------------------------------------------------- helpers

def worker_env() -> dict:
    """The caller's environment plus UV_NO_SYNC=1: a plain `uv run` can re-sync the lockfile and undo
    the CUDA 12.8 Torch override that Blackwell GPUs need (docs/runpod_reproduction.md)."""
    return {**os.environ, "UV_NO_SYNC": "1"}


def sh_sim(cmd: list[str], cwd: Path | None = None, timeout: float = 60) -> tuple[int, str]:
    """preflight._sh, but in the simulator's environment (UV_NO_SYNC=1)."""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=timeout,
                           env=worker_env())
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"


def run_worker(mode: str, a, out: Path, extra: list[str] | None = None, run=subprocess.run) -> tuple:
    """Run stage_worker.py in the simulator environment. Returns (exit code, result dict or None, log path)."""
    sim = a.tiptop_dir.resolve() / "droid-sim-evals"
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{mode}.log"
    cmd = ["uv", "run", "python", str(WORKER), mode, "--out-dir", str(out), *(extra or [])]
    print(f"    running {mode} (log: {log})", flush=True)
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            code = run(cmd, cwd=sim, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout_s,
                       env=worker_env()).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    except OSError as e:
        log.write_text(f"{type(e).__name__}: {e}\n")
        code = 127
    path = out / f"{mode}_result.json"
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        result = None
    return code, result, log


def exit_detail(code, result, log: Path) -> str:
    """Why a worker did not finish, in one line."""
    if code == 139 or code == -11:
        why = CRASH_HINT
    elif code == "timeout":
        why = "timed out (raise --timeout-s; a cold shader cache can take minutes)"
    elif result and result.get("error"):
        why = result["error"].strip().splitlines()[-1]
    else:
        why = f"exit {code}"
    reached = ", ".join(k for k in (result or {}).get("timings_s", {}) if k != "total")
    return f"{why}; reached: {reached or 'nothing'}; log: {log}"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_report(rep: Report, out: Path, stage: str, a, extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = {"stage": stage, "utc": stamp(), "args": {k: str(v) for k, v in vars(a).items()},
            "checks": rep.checks, **(extra or {})}
    (out / "report.json").write_text(json.dumps(data, indent=2))
    lines = [f"# Stage report: {stage}", "", f"Written {data['utc']}. "
             f"Result: **{'FAILED' if rep.failed else 'OK'}**.", "", "| check | status | detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {c['status']} | {str(c['detail']).replace('|', '/')} |" for c in rep.checks]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'FAILED' if rep.failed else 'OK'}: {out / 'report.md'}")


# --------------------------------------------------------------------------- stage 0: pod

def stage_pod(a, rep: Report, out: Path, sh=_sh, cuda_probe=cuda_driver_probe, env=container_env) -> dict:
    code, text = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                     "--format=csv,noheader"])
    gpus = parse_nvidia_smi_csv(text) if code == 0 else []
    if not gpus:
        rep.add("gpu", FAIL, f"nvidia-smi found no GPU (exit {code}): {text[-200:]}")
    for i, g in enumerate(gpus):
        status, why = driver_verdict(g["driver"])
        rep.add(f"driver:gpu{i}", status, f"{g['name']}, {why}")
        sim_why = gpu_roles(g["name"], g["compute_cap"]).get("simulator")
        if sim_why and "no RT cores" in sim_why:
            rep.add(f"rt-cores:gpu{i}", FAIL, f"{g['name']}: {sim_why}")
        elif sim_why:
            rep.add(f"rt-cores:gpu{i}", WARN, f"{g['name']}: {sim_why}. Needs the CUDA 12.8 Torch override "
                                              "(docs/runpod_reproduction.md)")
        else:
            rep.add(f"rt-cores:gpu{i}", PASS, f"{g['name']}, compute capability {g['compute_cap']:g}, {g['memory']}")

    if gpus:
        status, why = cuda_probe()
        rep.add("cuda", status, why)

    vk = shutil.which("vulkaninfo")
    status, why = vulkan_verdict(sh([vk, "--summary"], timeout=60)[1] if vk else None)
    rep.add("vulkan", status, why)

    code, text = sh(["ldconfig", "-p"])
    missing = missing_graphics_libs(text) if code == 0 else {}
    if code != 0:
        rep.add("graphics-libs", WARN, "ldconfig -p failed; libraries not checked")
    elif missing:
        rep.add("graphics-libs", FAIL, f"missing {sorted(missing)}: apt-get install -y {' '.join(sorted(set(missing.values())))}")
    else:
        rep.add("graphics-libs", PASS, "libEGL, libGL, libXt and libGLU present")
    ffmpeg = shutil.which("ffmpeg")
    rep.add("ffmpeg", PASS if ffmpeg else WARN,
            ffmpeg or "not installed: runs and the arm check save no video (apt-get install -y ffmpeg)")

    try:
        os_id, ver = parse_os_release(Path("/etc/os-release").read_text())
        ok = os_id == "ubuntu" and tuple(int(x) for x in ver.split(".")[:2]) >= (22, 4)
        rep.add("os", PASS if ok else WARN, f"{os_id} {ver}" + ("" if ok else " (Isaac Sim supports Ubuntu 22.04+)"))
    except (OSError, ValueError):
        rep.add("os", WARN, "could not read /etc/os-release")

    free_gb = shutil.disk_usage(a.disk_path).free / 1e9
    rep.add("disk", PASS if free_gb >= 60 else WARN,
            f"{free_gb:.0f} GB free at {a.disk_path}" + ("" if free_gb >= 60 else
                                                         " (the simulator stack and assets need about 60 GB)"))
    eula = env("OMNI_KIT_ACCEPT_EULA")
    rep.add("env:OMNI_KIT_ACCEPT_EULA", PASS if eula else WARN,
            eula or "not set: Isaac Sim stops at its license prompt without it (set it yourself once you accept)")
    status, why = driver_capabilities_verdict(env("NVIDIA_DRIVER_CAPABILITIES"))
    rep.add("env:NVIDIA_DRIVER_CAPABILITIES", status, why)
    return {"gpus": gpus}


# --------------------------------------------------------------------------- stage 1: sim

def stage_sim(a, rep: Report, out: Path, sh=sh_sim, run=subprocess.run) -> dict:
    sim = a.tiptop_dir.resolve() / "droid-sim-evals"
    if not sim.is_dir():
        rep.add("layout", FAIL, f"{sim} not found; run scripts/setup_upstream.sh first")
        return {}
    code, text = sh(["uv", "run", "python", "-c", CUDA_PROBE], cwd=sim, timeout=180)
    ok = code == 0 and "CUDA_MATMUL_PASSED" in text
    rep.add("cuda", PASS if ok else FAIL, text.strip().splitlines()[-1] if text.strip() else f"exit {code}")
    if not ok:
        return {}

    code, res, log = run_worker("launch", a, out, run=run)
    if code == 0 and res and res.get("app_started"):
        rep.add("launch", PASS, f"Isaac Sim started in {res['timings_s'].get('app_started', '?')} s")
    else:
        rep.add("launch", FAIL, exit_detail(code, res, log))
        return {"launch": res}

    code, res, log = run_worker("render", a, out, ["--scene", str(a.scene), "--variant", str(a.variant)], run=run)
    if code != 0 or not res or not res.get("ok"):
        rep.add("render", FAIL, exit_detail(code, res, log))
        return {"render": res}
    t = res.get("timings_s", {})
    rep.add("env", PASS, f"scene {a.scene} variant {a.variant} built in {t.get('env_created', '?')} s, settled at "
                         f"{t.get('settled', '?')} s; objects {res.get('object_names')}")
    for cam, stats in sorted((res.get("frames") or {}).items()):
        problems = frame_problems(stats)
        rep.add(f"camera:{cam}", FAIL if problems else PASS,
                "; ".join(problems) if problems else f"{stats['shape']}, mean {stats['mean']}, std {stats['std']} "
                                                     f"-> {stats.get('png', '(no png)')}")
    return {"render": res}


# --------------------------------------------------------------------------- stage 2: scenes

def scene_variants(scenes_cfg: dict, scenes: list[int], variants: list[int] | None, first_n: int | None) -> list:
    out = []
    for s in scenes:
        configured = scenes_cfg[str(s)]["variants"]
        chosen = variants if variants is not None else (configured[:first_n] if first_n else configured)
        out += [(s, v) for v in chosen]
    return out


def stage_scenes(a, rep: Report, out: Path, run=subprocess.run) -> dict:
    scenes_cfg = json.loads(Path(a.scenes_config).read_text())
    todo = scene_variants(scenes_cfg, a.scenes, a.variants, None if a.all_variants else a.first_variants)
    rows, reference = [], {}
    for i, (s, v) in enumerate(todo, 1):
        sub = out / f"s{s}v{v}"
        res = None
        if a.resume and (sub / "scene_result.json").exists():
            res = json.loads((sub / "scene_result.json").read_text())
            code, log = (0 if res.get("ok") else "earlier failure"), sub / "scene.log"
            print(f"==> [{i}/{len(todo)}] scene {s} variant {v} (kept from earlier run)", flush=True)
        else:
            print(f"==> [{i}/{len(todo)}] scene {s} variant {v}", flush=True)
            code, res, log = run_worker("scene", a, sub, ["--scene", str(s), "--variant", str(v)], run=run)
        name = f"scene{s}:variant{v}"
        if code != 0 or not res or not res.get("ok"):
            rep.add(name, FAIL, exit_detail(code, res, log))
            rows.append({"scene": s, "variant": v, "status": FAIL})
            continue
        fails = inspection_problems(res, scenes_cfg[str(s)], a.gripper_bodies)
        f2, warns = scene_problems(res, reference_objects=reference.get(s))
        fails += f2
        reference.setdefault(s, sorted(res.get("geometry_centers_after_settle", {})))
        status = FAIL if fails else WARN if warns else PASS
        detail = "; ".join(fails + warns) or (f"{len(res.get('object_names', []))} objects on the table and still; "
                                              f"images in {sub}")
        rep.add(name, status, detail)
        rows.append({"scene": s, "variant": v, "status": status})
    bad = [f"s{r['scene']}v{r['variant']}" for r in rows if r["status"] == FAIL]
    if bad:
        print(f"\nDo not run trials on: {', '.join(bad)}", flush=True)
    return {"variants": rows, "do_not_use": bad}


# --------------------------------------------------------------------------- stage 3: arm

def stage_arm(a, rep: Report, out: Path, run=subprocess.run) -> dict:
    extra = ["--scene", str(a.scene), "--variant", str(a.variant)] + (["--no-limit-probe"] if a.no_limit_probe else [])
    code, res, log = run_worker("arm", a, out, extra, run=run)
    if code != 0 or not res or not res.get("steps"):
        rep.add("arm", FAIL, exit_detail(code, res, log))
        return {"arm": res}
    if not res.get("ok"):
        rep.add("arm:complete", FAIL, f"stopped early at {res.get('episode_ended_early_at') or '?'} "
                                      f"after {len(res['steps'])}/{len(res.get('plan', []))} moves; log: {log}")
    verdict = arm_problems(res, load_limits(SIM_LIMITS_FILE), load_limits(PLANNER_LIMITS_FILE))
    for msg in verdict["fails"]:
        rep.add("arm", FAIL, msg)
    for msg in verdict["warns"]:
        rep.add("arm", WARN, msg)
    for msg in verdict["notes"]:
        rep.add("arm:info", PASS, msg)
    if not res.get("video"):
        rep.add("arm:video", WARN, f"no video saved ({res.get('video_error') or 'video disabled'}); "
                                   "the numbers above still hold. apt-get install -y ffmpeg if ffmpeg is missing")
    if not verdict["fails"]:
        rep.add("arm", PASS, f"{len(res['steps'])} moves in {res.get('sim_duration_s')} s simulated; "
                             f"watch {res.get('video', '(no video)')}")
    return {"arm": {k: res.get(k) for k in ("steps", "sim_joint_limits_live", "object_motion_m", "video", "trace")}}


# --------------------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--tiptop-dir", type=Path, default=os.environ.get("TIPTOP_DIR"),
                   help="parent of droid-sim-evals/ (stages sim, scenes, arm); default $TIPTOP_DIR")
    p.add_argument("--out-dir", type=Path, default=None, help="default results/stages/<stage>_<utc>")
    p.add_argument("--timeout-s", type=float, default=1200, help="per simulator process")
    p.add_argument("--scene", type=int, default=1, help="sim, arm: scene to load")
    p.add_argument("--variant", type=int, default=0, help="sim, arm: variant to load")
    p.add_argument("--disk-path", default="/workspace" if Path("/workspace").exists() else str(PROJECT_ROOT))
    g = p.add_argument_group("scenes")
    g.add_argument("--scenes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    g.add_argument("--variants", type=int, nargs="+", default=None, help="these variants of every scene")
    g.add_argument("--first-variants", type=int, default=5,
                   help="default: the first N configured variants (5 covers the pilot and paper suites)")
    g.add_argument("--all-variants", action="store_true", help="every configured variant (51 launches)")
    g.add_argument("--resume", action="store_true", help="with --out-dir: keep variants already checked there")
    g.add_argument("--scenes-config", default=str(PROJECT_ROOT / "configs" / "scenes.json"))
    g.add_argument("--gripper-bodies", nargs="+", default=["left_inner_finger", "right_inner_finger"])
    p.add_argument("--no-limit-probe", action="store_true", help="arm: skip commanding joints 6/7 past their limits")
    a = p.parse_args(argv)
    if a.stage != "pod" and not a.tiptop_dir:
        p.error(f"stage {a.stage} needs --tiptop-dir (or $TIPTOP_DIR)")
    if a.resume and not a.out_dir:
        p.error("--resume needs the --out-dir of the run to continue")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    out = (a.out_dir or PROJECT_ROOT / "results" / "stages" / f"{a.stage}_{stamp()}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    print(f"Stage {a.stage} -> {out}", flush=True)
    fn = {"pod": stage_pod, "sim": stage_sim, "scenes": stage_scenes, "arm": stage_arm}[a.stage]
    extra = fn(a, rep, out)
    write_report(rep, out, a.stage, a, extra)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
