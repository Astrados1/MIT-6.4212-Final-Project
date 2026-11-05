"""Simulator-side measurements for the staged GPU checks (docs/staged_testing.md).

Normally started by grasp_eval/stages.py. To run one by hand, from the droid-sim-evals checkout
inside its uv environment (UV_NO_SYNC=1 keeps a CUDA-wheel override in place):

    uv run python $PROJECT/grasp_eval/stage_worker.py launch --out-dir $OUT
    uv run python $PROJECT/grasp_eval/stage_worker.py render --scene 1 --variant 0 --out-dir $OUT
    uv run python $PROJECT/grasp_eval/stage_worker.py scene  --scene 4 --variant 2 --out-dir $OUT
    uv run python $PROJECT/grasp_eval/stage_worker.py arm    --scene 1 --variant 0 --out-dir $OUT

Modes, each in a fresh process (one simulator per process, as in observed_eval.py):
  launch  start Isaac Sim headless with cameras enabled, then exit. The Nov-4 R595 crash
          happened here, before any scene existed.
  render  build the DROID environment for one scene, reset and settle as upstream does, and
          save both camera images with simple statistics (a broken renderer gives blank images).
  scene   as render, plus the scene_inspection.json fields (objects, geometry boxes and centers,
          robot bodies/joints) and how far each object still moves while the robot holds still.
  arm     planner-free arm check: one-joint moves around TiPToP's home pose, the capture pose,
          gripper close/open, and commands past the simulator's joint-6/7 limits (the FR3 vs
          Panda mismatch). Records tracking error, gripper state, object motion, a trace and a video.

Nothing here talks to TiPToP, M2T2 or Gemini. The result JSON is written before the app is
closed, because closing Isaac Sim can end the process. Pass/fail rules: grasp_eval/stage_checks.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.stage_checks import LIMIT_PROBES, Q_CAPTURE, Q_HOME, SIM_LIMITS_FILE, arm_test_plan, load_limits  # noqa: E402

MODES = ("launch", "render", "scene", "arm")
CAMERAS = ("external_cam", "wrist_cam")
FINGER_JOINT = "finger_joint"
FINGER_CLOSED_RAD = np.pi / 4  # as recovery_eval: simulator's closed target for the Robotiq finger joint
VIDEO_FPS = 15


def frame_stats(img) -> dict:
    """Shape, mean, std and fraction of lit pixels of an HxWxC uint8-like image."""
    a = np.asarray(img)
    if a.size == 0:
        return {"shape": list(a.shape)}
    f = a.astype(np.float32)
    rgb = f[..., :3] if f.ndim == 3 else f
    lit = (rgb.max(axis=-1) if rgb.ndim == 3 else rgb) > 8
    return {"shape": list(a.shape), "dtype": str(a.dtype), "mean": round(float(rgb.mean()), 3),
            "std": round(float(rgb.std()), 3), "min": float(rgb.min()), "max": float(rgb.max()),
            "lit_frac": round(float(lit.mean()), 4)}


def end_mean(samples: list, n: int = 5) -> list[float] | None:
    """Mean of the last n samples (vectors), or None."""
    if not samples:
        return None
    return np.mean(np.asarray(samples[-n:], dtype=float), axis=0).round(5).tolist()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Simulator-side staged GPU check")
    p.add_argument("mode", choices=MODES)
    p.add_argument("--sim-root", type=Path, default=Path.cwd(), help="droid-sim-evals checkout (default: cwd)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--variant", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--hold-s", type=float, default=1.0, help="scene: how long the robot holds still after settling")
    a = p.add_argument_group("arm mode")
    a.add_argument("--max-joint-speed", type=float, default=0.5, help="rad/s peak along each move")
    a.add_argument("--settle-s", type=float, default=1.5, help="hold at each target before measuring")
    a.add_argument("--gripper-settle-s", type=float, default=2.0)
    a.add_argument("--excursion", type=float, default=0.25, help="rad, one-joint moves around home")
    a.add_argument("--no-limit-probe", action="store_true", help="skip commanding joints 6/7 past their limits")
    a.add_argument("--limits-json", type=Path, default=SIM_LIMITS_FILE,
                   help="recorded simulator limits, used if the simulator does not report them")
    a.add_argument("--no-video", action="store_true")
    args, _ = p.parse_known_args(argv)  # leave AppLauncher flags for Isaac Lab
    return args


def live_joint_limits(robot, n: int = 7) -> list[list[float]] | None:
    """Arm joint position limits as the articulation reports them (Isaac Lab attribute names vary)."""
    for attr in ("joint_pos_limits", "soft_joint_pos_limits", "joint_limits"):
        try:
            lim = getattr(robot.data, attr)
        except Exception:
            continue
        if lim is None:
            continue
        try:
            return [[round(float(lo), 6), round(float(hi), 6)] for lo, hi in lim[0, :n].tolist()]
        except Exception:
            continue
    return None


def main(argv=None) -> int:  # pragma: no cover - needs Isaac Sim
    args = parse_args(argv)
    sim_root = args.sim_root.resolve()
    sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"mode": args.mode, "scene": args.scene, "variant": args.variant, "seed": args.seed,
                    "ok": False, "timings_s": {}, "pid": os.getpid()}
    t0 = time.monotonic()
    result_path = out / f"{args.mode}_result.json"

    def save():
        result["timings_s"]["total"] = round(time.monotonic() - t0, 2)
        result_path.write_text(json.dumps(result, indent=2))

    def mark(name):
        result["timings_s"][name] = round(time.monotonic() - t0, 2)
        save()  # a later native crash still leaves how far the process got

    save()
    simulation_app = env = None
    try:
        import random

        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                           "cuda_available": bool(torch.cuda.is_available()),
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args_cli, _ = parser.parse_known_args(argv)
        args_cli.enable_cameras = True
        args_cli.headless = args.headless
        mark("before_app_start")
        simulation_app = AppLauncher(args_cli).app
        result["app_started"] = True
        mark("app_started")
        if args.mode == "launch":
            result["ok"] = True
            return 0

        import gymnasium as gym

        import src.sim_evals.environments  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from src.sim_evals.sim_utils import settle_sim

        from grasp_eval.observed_eval import usd_local_bounds
        from grasp_eval.task_goal import geometry_center_positions

        env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
        env_cfg.set_scene(str(args.scene), args.variant)
        env_cfg.episode_length_s = 600.0 if args.mode == "arm" else 90.0
        env_cfg.seed = args.seed
        env = gym.make("DROID", cfg=env_cfg)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        step_dt = float(env.unwrapped.step_dt)
        names = sorted(scene.rigid_objects.keys())
        result.update({"step_dt": step_dt, "object_names": names})
        try:  # before the first physics step, as observed_eval
            bounds, notes = usd_local_bounds(scene, names)
        except Exception as e:
            bounds, notes = {}, {"_error": f"{type(e).__name__}: {e}"}
        mark("env_created")

        def positions():
            return {n: scene.rigid_objects[n].data.root_pos_w[0].tolist() for n in names}

        def quats():
            return {n: scene.rigid_objects[n].data.root_quat_w[0].tolist() for n in names}

        def centers():
            p, q = positions(), quats()
            return {n: (geometry_center_positions(p[n], q[n], bounds[n]).tolist() if n in bounds else p[n])
                    for n in names}

        obs, _ = env.reset()
        obs, _ = env.reset()  # upstream: second render cycle for materials
        obs, _ = env.reset()
        obs = settle_sim(env, obs, reset_episode_buf=True)
        mark("settled")

        import mediapy

        frames = {}
        for cam in CAMERAS:
            img = obs["policy"][cam][0].cpu().numpy()
            frames[cam] = frame_stats(img)
            try:
                path = out / f"{args.mode}_s{args.scene}v{args.variant}_{cam}.png"
                mediapy.write_image(str(path), img)
                frames[cam]["png"] = str(path)
            except Exception as e:
                frames[cam]["png_error"] = f"{type(e).__name__}: {e}"
        result["frames"] = frames
        mark("frames_captured")
        if args.mode == "render":
            result["ok"] = True
            return 0

        def act(q, g):
            o, _, term, trunc, _ = env.step(torch.tensor(np.r_[np.asarray(q, float), float(g)],
                                                         dtype=torch.float32)[None])
            return o, bool(term), bool(trunc)

        q_start = obs["policy"]["arm_joint_pos"][0].cpu().numpy().astype(float)
        result["start_joint_pos"] = q_start.round(5).tolist()
        c0 = centers()

        if args.mode == "scene":
            result.update({
                "rigid_objects_after_settle": positions(),
                "rigid_object_quats_after_settle": quats(),
                "geometry_centers_after_settle": c0,
                "rigid_object_local_bounds": bounds,
                "rigid_object_bounds_notes": notes,
                "robot_body_names": list(robot.data.body_names),
                "robot_joint_names": list(robot.data.joint_names),
            })
            for _ in range(max(1, int(round(args.hold_s / step_dt)))):
                obs, term, trunc = act(q_start, 0.0)
                if term or trunc:
                    result["episode_ended_during_hold"] = True
                    break
            c1 = centers()
            result["geometry_centers_after_hold"] = c1
            result["settle_motion_m"] = {n: round(float(np.linalg.norm(np.subtract(c1[n], c0[n]))), 5)
                                         for n in names}
            (out / "scene_inspection.json").write_text(json.dumps(
                {k: result[k] for k in ("rigid_objects_after_settle", "rigid_object_quats_after_settle",
                                        "geometry_centers_after_settle", "rigid_object_local_bounds",
                                        "rigid_object_bounds_notes", "robot_body_names", "robot_joint_names")}
                | {"scene": args.scene, "variant": args.variant, "step_dt": step_dt}, indent=2))
            result["ok"] = True
            return 0

        # ------------------------------------------------------------------ arm
        from grasp_eval.recovery_eval import _StreamingVideo, joint_interpolation
        from grasp_eval.video_overlay import add_caption

        recorded = load_limits(args.limits_json)
        live = live_joint_limits(robot)
        result["sim_joint_limits_live"] = live
        limits = [tuple(x) for x in live] if live else recorded
        plan = arm_test_plan(limits, q_home=Q_HOME, q_capture=Q_CAPTURE, excursion=args.excursion,
                             limit_probes={} if args.no_limit_probe else LIMIT_PROBES)
        result["plan"] = plan
        finger_ids, _ = robot.find_joints([FINGER_JOINT])
        video = None if args.no_video else _StreamingVideo(out / "arm_check.mp4", VIDEO_FPS)
        trace = {k: [] for k in ("t", "q_cmd", "q_meas", "g_cmd", "g_obs", "finger_closure", "step")}
        steps, violations, seen = [], [], set()
        max_motion = {n: 0.0 for n in names}
        q_cmd, n_steps, ended = q_start.copy(), 0, False

        def record(o, q, g, i, label, kind):
            nonlocal n_steps
            n_steps += 1
            qm = o["policy"]["arm_joint_pos"][0].cpu().numpy().astype(float)
            fc = (float(robot.data.joint_pos[0, finger_ids[0]]) / FINGER_CLOSED_RAD) if finger_ids else float("nan")
            trace["t"].append(n_steps * step_dt)
            trace["q_cmd"].append(np.asarray(q, float))
            trace["q_meas"].append(qm)
            trace["g_cmd"].append(float(g))
            trace["g_obs"].append(float(o["policy"]["gripper_pos"][0, 0]))
            trace["finger_closure"].append(fc)
            trace["step"].append(i)
            if kind != "limit_probe":
                for j, (lo, hi) in enumerate(limits):
                    if (qm[j] < lo - 0.01 or qm[j] > hi + 0.01) and (label, j) not in seen:
                        seen.add((label, j))
                        violations.append({"step": label, "joint": j, "value": round(float(qm[j]), 5)})
            if video is not None:
                try:
                    viz = np.concatenate([o["policy"]["external_cam"][0].cpu().numpy(),
                                          o["policy"]["wrist_cam"][0].cpu().numpy()], axis=1)
                    video.add(add_caption(viz, f"arm check {i + 1}/{len(plan)}: {label} | gripper "
                                               f"{'CLOSE' if g > 0.5 else 'open'}"))
                except Exception:
                    pass
            return qm

        try:
            for i, st in enumerate(plan):
                tgt, g, kind = np.asarray(st["q"], float), float(st["gripper"]), st["kind"]
                t_start = n_steps
                for q in joint_interpolation(q_cmd, tgt, args.max_joint_speed, step_dt):
                    o, term, trunc = act(q, g)
                    record(o, q, g, i, st["label"], kind)
                    if term or trunc:
                        ended = True
                        break
                q_cmd = tgt
                hold = args.gripper_settle_s if kind.startswith("gripper") else args.settle_s
                lo_idx = len(trace["t"])
                for _ in range(max(1, int(round(hold / step_dt)))):
                    if ended:
                        break
                    o, term, trunc = act(tgt, g)
                    record(o, tgt, g, i, st["label"], kind)
                    ended = term or trunc
                held = slice(lo_idx, len(trace["t"]))
                c = centers()
                for n in names:
                    max_motion[n] = max(max_motion[n], float(np.linalg.norm(np.subtract(c[n], c0[n]))))
                steps.append({
                    "label": st["label"], "kind": kind, "joint": st.get("joint"), "gripper_cmd": g,
                    "q_target": tgt.round(5).tolist(),
                    "q_meas_end": end_mean(trace["q_meas"][held]),
                    "gripper_obs_end": (end_mean([[v] for v in trace["g_obs"][held]]) or [None])[0],
                    "finger_closure_end": (end_mean([[v] for v in trace["finger_closure"][held]]) or [None])[0],
                    "sim_steps": n_steps - t_start,
                })
                result["steps"] = steps
                if ended:
                    result["episode_ended_early_at"] = st["label"]
                    break
        finally:
            if video is not None:
                video.close()
                result["video"] = str(out / "arm_check.mp4")
            if trace["t"]:
                np.savez_compressed(out / "arm_trace.npz", **{k: np.asarray(v) for k, v in trace.items()},
                                    step_labels=np.asarray([s["label"] for s in plan]))
                result["trace"] = str(out / "arm_trace.npz")
        result.update({"steps": steps, "limit_violations": violations,
                       "object_motion_m": {n: round(v, 5) for n, v in max_motion.items()},
                       "sim_duration_s": round(n_steps * step_dt, 2)})
        result["ok"] = not ended and len(steps) == len(plan)
        return 0
    except Exception:
        result["error"] = traceback.format_exc()
        return 1
    finally:
        save()
        try:
            if env is not None:
                env.close()
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
