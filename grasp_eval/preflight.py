"""Preflight checks on the GPU machine before spending hours on a suite.

    python3 -m grasp_eval.preflight --tiptop-dir $TIPTOP_DIR [--scenes 1 2 3 4 5] [--sim-check] [--smoke-run]

Checks, cheapest first:
  gpu          every visible GPU (CUDA_VISIBLE_DEVICES): which components it can run. The simulator needs RT
               cores; the simulator and M2T2 environments stop at compute capability 9.0 (no Blackwell)
  os           Ubuntu 22.04 or later (Isaac Sim's supported OS)
  network      the node can reach Gemini, the SAM2 download host and PyPI
  tools        uv, pixi and git on PATH
  layout       the three upstream checkouts and every scene/variant USD the suite will load
  pins         tiptop and droid-sim-evals at the pinned SHAs; the Gemini ER 2 patch applied
  servers      TiPToP (ws host:port/health) and M2T2 (/health) answer
  config       configs/scenes.json goals have every object/container name filled in
  credentials  if GOOGLE_API_KEY is set in this shell, ask the Gemini API whether the key can see the pinned
               model (key sent as a header, never printed); warning only if unset
  disk         free space for results and server outputs
  sim-check    (--sim-check) observed_eval.py --inspect-only per scene: gripper bodies exist, every object
               has a plausible geometry box, and goal/target names exist in the simulator
  smoke-run    (--smoke-run) one full pilot run plus attribution; checks the record, trace, task check
               and paper category all came out

Writes results/preflight_<utc>.json and exits 1 if any check FAILs. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MIN_FREE_GB = 20
PLAUSIBLE_EXTENT_M = (0.01, 1.0)  # any object box side outside this is suspicious

# GPUs without RT cores. Isaac Sim does not support them, and it needs RT cores to render cameras.
NO_RT_CORE_GPUS = ("A100", "A800", "A30", "H100", "H800", "H200", "H20", "B100", "B200", "GB200", "V100", "P100")
# Newest compute capability each pinned environment's PyTorch/CUDA build supports (docs/upstream_notes.md):
# simulator: torch 2.7.0 built for CUDA 12.6; M2T2: torch 2.4.1 / nvcc 12.1 (both stop at sm_90, so no
# Blackwell); TiPToP: torch 2.7.1 / nvcc 12.9 (expected to cover Blackwell, sm_120).
ENV_MAX_COMPUTE_CAP = {"simulator": 9.0, "m2t2": 9.0, "tiptop": 12.0}
# Hosts the GPU node must reach: Gemini (every run), SAM2 checkpoint (first run), packages (install).
REQUIRED_HOSTS = {"gemini": "https://generativelanguage.googleapis.com/",
                  "sam2_download": "https://dl.fbaipublicfiles.com/",
                  "pypi": "https://pypi.org/simple/"}


def gpu_roles(name: str, compute_cap: float) -> dict[str, str | None]:
    """For one GPU: component -> None if it should run there, else why it can't."""
    no_rt = any(re.search(rf"\b{m}\b", name) for m in NO_RT_CORE_GPUS)
    out: dict[str, str | None] = {}
    for comp, max_cc in ENV_MAX_COMPUTE_CAP.items():
        if comp == "simulator" and no_rt:
            out[comp] = "no RT cores, so Isaac Sim can't render the cameras"
        elif compute_cap > max_cc:
            out[comp] = (f"compute capability {compute_cap:g} is newer than its pinned PyTorch/CUDA build "
                         f"supports (up to {max_cc:g})")
        else:
            out[comp] = None
    return out


def parse_os_release(text: str) -> tuple[str, str]:
    vals = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text, flags=re.M))
    return vals.get("ID", ""), vals.get("VERSION_ID", "")


def reachable(url: str, timeout: float = 8.0) -> tuple[bool, str]:
    """Any HTTP answer (even 403/404) means the host is reachable from this node."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code}"
    except Exception as e:  # DNS failure, connection refused, proxy block, timeout
        return False, f"{type(e).__name__}: {e}"


def gemini_model_check(model: str, key: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Ask the Gemini API whether this key can see `model`. The key goes in a header, never the URL or logs."""
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}",
                                 headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200, f"HTTP {r.status}: {model} is available to this key"
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        return False, f"HTTP {e.code}: {msg[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _sh(cmd: list[str], cwd: Path | None = None, timeout: float = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"


def http_ok(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200, f"HTTP {r.status}"
    except Exception as e:  # connection refused, timeout, HTTP error
        return False, f"{type(e).__name__}: {e}"


def read_pins(path: Path = PROJECT_ROOT / "provenance" / "pins.env") -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def m2t2_url_from_config(tiptop: Path) -> str:
    """perception.m2t2.url from tiptop.yml without needing a YAML parser."""
    cfg = tiptop / "tiptop" / "config" / "tiptop.yml"
    if cfg.exists():
        m = re.search(r"m2t2:\s*\n\s+url:\s*\"?([^\"\n]+)\"?", cfg.read_text())
        if m:
            return m.group(1).strip()
    return "http://localhost:8123"


def scene_variants(scenes_cfg: dict, scenes: list[int], variants: list[int] | None) -> list[tuple[int, int]]:
    out = []
    for s in scenes:
        for v in (variants if variants is not None else scenes_cfg[str(s)]["variants"]):
            out.append((s, v))
    return out


def goal_problems(scenes_cfg: dict, scenes: list[int]) -> list[str]:
    """Unset names in the goal specs: each one leaves task success unchecked."""
    problems = []
    for s in scenes:
        sc = scenes_cfg.get(str(s), {})
        goal = sc.get("goal")
        if not goal:
            problems.append(f"scene {s}: no goal")
            continue
        for i, clause in enumerate(goal):
            if not clause.get("container"):
                problems.append(f"scene {s} goal[{i}]: container not set")
            if not clause.get("objects") or any(not o for o in clause["objects"]):
                problems.append(f"scene {s} goal[{i}]: object name(s) not set")
            if "$target" in clause.get("objects", []) and not sc.get("target_object") and not sc.get("object_aliases"):
                problems.append(f"scene {s}: goal uses $target but no target_object or object_aliases")
    return problems


def inspection_problems(info: dict, scene_cfg: dict, gripper_bodies: list[str]) -> list[str]:
    """Problems in one scene_inspection.json written by observed_eval.py --inspect-only."""
    problems = []
    bodies = set(info.get("robot_body_names", []))
    missing = [b for b in gripper_bodies if b not in bodies]
    if missing:
        problems.append(f"gripper bodies {missing} not in robot bodies; pass --gripper-bodies (see robot_body_names)")
    objects = info.get("rigid_objects_after_settle", {})
    bounds = info.get("rigid_object_local_bounds") or {}
    for name in objects:
        if name not in bounds:
            problems.append(f"no geometry box for {name}: {(info.get('rigid_object_bounds_notes') or {}).get(name)}")
            continue
        lo, hi = bounds[name]
        ext = [h - l for l, h in zip(lo, hi)]
        if not all(PLAUSIBLE_EXTENT_M[0] <= e <= PLAUSIBLE_EXTENT_M[1] for e in ext):
            problems.append(f"implausible box for {name}: extents {[round(e, 3) for e in ext]} m")
    if (info.get("rigid_object_bounds_notes") or {}).get("_error"):
        problems.append(f"reading boxes failed: {info['rigid_object_bounds_notes']['_error']}")
    names = set(objects)
    for clause in scene_cfg.get("goal") or []:
        for n in [clause.get("container"), *clause.get("objects", [])]:
            if n and n != "$target" and n not in names:
                problems.append(f"goal name {n!r} is not a simulator object {sorted(names)}")
    t = scene_cfg.get("target_object")
    if t and t not in names:
        problems.append(f"target_object {t!r} is not a simulator object")
    return problems


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str = "", **data) -> None:
        self.checks.append({"check": name, "status": status, "detail": detail, **data})
        print(f"[{status}] {name}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == FAIL for c in self.checks)


def run_checks(a) -> Report:
    rep = Report()
    root = a.tiptop_dir.resolve()
    tiptop, sim, m2t2 = root / "tiptop", root / "droid-sim-evals", root / "M2T2"
    scenes_cfg = json.loads(Path(a.scenes_config).read_text())

    code, out = _sh(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version",
                     "--format=csv,noheader"])
    if code != 0 or not out:
        rep.add("gpu", FAIL, out or "nvidia-smi printed nothing")
    else:
        sim_ok = False
        for line in out.splitlines():
            idx, name, cc, total, used, driver = [f.strip() for f in line.split(",")]
            try:
                roles = gpu_roles(name, float(cc))
            except ValueError:
                rep.add(f"gpu{idx}", WARN, f"{name}: compute capability {cc!r} unreadable")
                continue
            ok = [c for c, why in roles.items() if why is None]
            sim_ok |= "simulator" in ok
            detail = f"{name}, sm {cc}, {used} / {total} used, driver {driver}; can run: {', '.join(ok) or 'nothing'}"
            blocked = "; ".join(f"{c}: {why}" for c, why in roles.items() if why)
            status = PASS if {"simulator", "m2t2"} <= set(ok) else (WARN if ok else FAIL)
            rep.add(f"gpu{idx}", status, detail + (f" (not {blocked})" if blocked else ""), roles=roles)
        if not sim_ok:
            rep.add("gpu:simulator", FAIL, "no visible GPU can run the simulator; use an RTX card with compute "
                    "capability up to 9.0 (e.g. RTX A6000), or set CUDA_VISIBLE_DEVICES to one")

    os_text = Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else ""
    os_id, os_ver = parse_os_release(os_text)
    try:
        ubuntu_ok = os_id == "ubuntu" and tuple(int(x) for x in os_ver.split(".")) >= (22, 4)
    except ValueError:
        ubuntu_ok = False
    rep.add("os", PASS if ubuntu_ok else WARN, f"{os_id or 'unknown'} {os_ver}"
            + ("" if ubuntu_ok else ": Isaac Sim officially needs Ubuntu 22.04 or later; expect install problems"))

    for name, url in REQUIRED_HOSTS.items():
        ok, detail = reachable(url)
        rep.add(f"network:{name}", PASS if ok else (FAIL if name == "gemini" else WARN), f"{url} -> {detail}")

    missing_tools = [t for t in ("uv", "pixi", "git") if shutil.which(t) is None]
    rep.add("tools", FAIL if missing_tools else PASS, f"missing: {missing_tools}" if missing_tools else "uv, pixi, git")

    missing = [str(p) for p in (tiptop, sim, m2t2) if not p.is_dir()]
    usds = [sim / "assets" / f"scene{s}_{v}.usd" for s, v in scene_variants(scenes_cfg, a.scenes, a.variants)]
    missing += [str(u) for u in usds if not u.exists()]
    rep.add("layout", FAIL if missing else PASS,
            f"{len(missing)} missing, e.g. {missing[:3]}" if missing else f"checkouts and {len(usds)} scene USDs present")

    pins = read_pins()
    for label, path, key in (("tiptop", tiptop, "TIPTOP_SHA"), ("droid-sim-evals", sim, "SIM_SHA")):
        code, head = _sh(["git", "-C", str(path), "rev-parse", "HEAD"])
        ok = code == 0 and head == pins.get(key)
        rep.add(f"pin:{label}", PASS if ok else WARN, head if ok else f"HEAD {head} != pin {pins.get(key)}")
    for patch in sorted((PROJECT_ROOT / "patches" / "tiptop").glob("*.patch")):
        code, out = _sh(["git", "-C", str(tiptop), "apply", "--reverse", "--check", str(patch)])
        rep.add(f"patch:{patch.name}", PASS if code == 0 else FAIL,
                "applied" if code == 0 else f"not applied: git -C {tiptop} apply {patch}")

    if a.no_servers:
        rep.add("servers", WARN, "skipped (--no-servers)")
    else:
        for name, url in (("tiptop", f"http://{a.ws_host}:{a.ws_port}/health"),
                          ("m2t2", m2t2_url_from_config(tiptop).rstrip("/") + "/health")):
            ok, detail = http_ok(url)
            rep.add(f"server:{name}", PASS if ok else FAIL, f"{url} -> {detail}")

    problems = goal_problems(scenes_cfg, a.scenes)
    rep.add("config", (FAIL if a.require_goals else WARN) if problems else PASS,
            "; ".join(problems) if problems else "every goal name set", problems=problems)

    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        model = pins.get("EXPECTED_GEMINI_MODEL", "gemini-robotics-er-2-preview")
        ok, detail = gemini_model_check(model, key)
        rep.add("credentials", PASS if ok else FAIL, detail)
    else:
        rep.add("credentials", WARN, "GOOGLE_API_KEY not set here; export it in this shell to test it, and it must "
                "be set in the TiPToP server's shell")

    for label, path in (("results", PROJECT_ROOT), ("tiptop", root)):
        free = shutil.disk_usage(path).free / 1e9 if path.exists() else 0.0
        rep.add(f"disk:{label}", PASS if free >= MIN_FREE_GB else WARN, f"{free:.0f} GB free at {path}")

    if a.sim_check:
        sim_check(a, rep, sim, scenes_cfg)
    if a.smoke_run and not rep.failed:
        smoke_run(a, rep, scenes_cfg)
    elif a.smoke_run:
        rep.add("smoke-run", FAIL, "skipped: fix the failures above first")
    return rep


def sim_check(a, rep: Report, sim: Path, scenes_cfg: dict) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "results" / f"preflight_inspect_{stamp}"
    for s in a.scenes:
        v = (a.variants or scenes_cfg[str(s)]["variants"])[0]
        run_id = f"inspect_s{s}v{v}"
        code, log = _sh(["uv", "run", "python", str(PROJECT_ROOT / "grasp_eval" / "observed_eval.py"),
                         "--inspect-only", "--scene", str(s), "--variant", str(v), "--run-id", run_id,
                         "--out-dir", str(out_dir), "--gripper-bodies", *a.gripper_bodies],
                        cwd=sim, timeout=a.sim_timeout_s)
        path = out_dir / "runs" / run_id / "scene_inspection.json"
        if not path.exists():
            rep.add(f"sim-check:scene{s}", FAIL, f"no inspection written (exit {code}); last output: {log[-300:]}")
            continue
        info = json.loads(path.read_text())
        problems = inspection_problems(info, scenes_cfg[str(s)], a.gripper_bodies)
        rep.add(f"sim-check:scene{s}", FAIL if problems else PASS,
                "; ".join(problems) if problems else f"{len(info.get('rigid_objects_after_settle', {}))} objects ok",
                inspection=str(path))


def smoke_problems(record: dict | None, attr: dict | None, trace_keys: set[str]) -> list[str]:
    if record is None:
        return ["no record written"]
    problems = []
    if record.get("harness_exception"):
        problems.append("harness exception: " + record["harness_exception"].strip().splitlines()[-1])
    if not record.get("server_success"):
        problems.append(f"no plan: {record.get('server_error')}")
    if not record.get("executed_steps"):
        problems.append("no steps executed")
    for k in ("objects_quat", "gripper_quat", "objects_pos", "gripper_pos"):
        if record.get("executed_steps") and k not in trace_keys:
            problems.append(f"trace missing {k}")
    if not record.get("object_local_bounds"):
        problems.append("record has no object geometry boxes")
    if attr is None:
        problems.append("no attribution written")
    else:
        if attr.get("paper_category") is None:
            problems.append("attribution has no paper_category")
        if (attr.get("task_check") or {}).get("task_success") is None:
            problems.append(f"task success unchecked: {(attr.get('task_check') or {}).get('reason')}")
        if attr.get("perceived_scene_available") is False:
            problems.append(f"perceived scene not extracted: {attr.get('perceived_scene_error')}")
        if record.get("server_success") and attr.get("grasp_source") is None:
            problems.append("grasp source unknown (grasps.pt not read)")
    return problems


def smoke_run(a, rep: Report, scenes_cfg: dict) -> None:
    from grasp_eval import run_suite

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"preflight_smoke_{stamp}"
    s = a.scenes[0]
    sc = scenes_cfg[str(s)]
    args = ["paper", "--tiptop-dir", str(a.tiptop_dir), "--out-name", name, "--scenes", str(s),
            "--variants", str((a.variants or sc["variants"])[0]), "--trials", "1",
            "--ws-host", a.ws_host, "--ws-port", str(a.ws_port), "--no-require-goals",
            "--eval-args", "--gripper-bodies", *a.gripper_bodies]
    run_suite.main(args)
    out = PROJECT_ROOT / "results" / name
    recs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()] if (out / "runs.jsonl").exists() else []
    attrs = [json.loads(line) for line in (out / "attribution.jsonl").read_text().splitlines()] \
        if (out / "attribution.jsonl").exists() else []
    record, attr = (recs or [None])[0], (attrs or [None])[0]
    trace_keys = set()
    if record and record.get("artifacts", {}).get("trace"):
        try:
            import zipfile
            with zipfile.ZipFile(record["artifacts"]["trace"]) as z:  # .npz is a zip of .npy files
                trace_keys = {n[:-4] for n in z.namelist()}
        except Exception:
            pass
    problems = smoke_problems(record, attr, trace_keys)
    rep.add("smoke-run", FAIL if problems else PASS,
            "; ".join(problems) if problems else
            f"stable_lift={record.get('stable_lift')} paper_category={attr.get('paper_category')}",
            results=str(out))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tiptop-dir", type=Path, required=True, help="parent of tiptop/, M2T2/, droid-sim-evals/")
    p.add_argument("--scenes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--variants", type=int, nargs="+", default=None, help="default: every variant in the config")
    p.add_argument("--scenes-config", default=str(PROJECT_ROOT / "configs" / "scenes.json"))
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--gripper-bodies", nargs="+", default=["left_inner_finger", "right_inner_finger"])
    p.add_argument("--no-servers", action="store_true", help="skip server health checks")
    p.add_argument("--require-goals", action=argparse.BooleanOptionalAction, default=True,
                   help="FAIL (default) or only WARN when goal names are unset")
    p.add_argument("--sim-check", action="store_true", help="run --inspect-only for each scene (needs the GPU)")
    p.add_argument("--sim-timeout-s", type=float, default=900)
    p.add_argument("--smoke-run", action="store_true", help="one full run plus attribution (needs servers)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    rep = run_checks(a)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = PROJECT_ROOT / "results" / f"preflight_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"utc": stamp, "args": {k: str(v) for k, v in vars(a).items()},
                               "checks": rep.checks}, indent=2))
    print(f"\n{'FAILED' if rep.failed else 'OK'}: report written to {out}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
