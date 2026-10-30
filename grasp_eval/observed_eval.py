"""One observed TiPToP episode in droid-sim-evals, with observation-only grasp scoring.

Run from the droid-sim-evals checkout, inside its uv environment:

    cd $TIPTOP_DIR/droid-sim-evals
    uv run python $PROJECT/grasp_eval/observed_eval.py --inspect-only --scene 1 --variant 0 \
        --out-dir $PROJECT/results/inspect
    uv run python $PROJECT/grasp_eval/observed_eval.py --scene 1 --variant 0 --trial 0 \
        --instruction "Put the Rubik's cube in the bowl." --target-object <name from inspect> \
        --out-dir $PROJECT/results/pilot_scene1

The control loop mirrors upstream tiptop_eval.py at SIM_SHA (provenance/pins.env):
same app launch, env config, 90 s episode limit, double reset, settle_sim,
client, plan execution and break conditions. What this file adds only reads
state: target/gripper poses after each step, the server's raw JSON response,
a trace file, a video, and a run record. Differences that remain:
  * one episode per process (upstream loops episodes in one process), so each
    trial gets a fresh connection and a fresh plan;
  * client.reset() is not called after the episode, because the process exits;
  * a video is saved for every run that executed any steps;
  * Python/numpy/torch seeds are set and recorded (upstream sets none).
Keep this list current if anything else changes, and never alter actions here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from grasp_eval.records import RunRecord, append_jsonl, auto_label  # noqa: E402
from grasp_eval.scoring import LiftCriteria, score_first_grasp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("observed_eval")

# Robotiq 2F-85 finger links; their midpoint is the gripper reference point.
# Unverified against the flattened USD: confirm with --inspect-only.
DEFAULT_GRIPPER_BODIES = ["left_inner_finger", "right_inner_finger"]
EPISODE_LENGTH_S = 90.0  # upstream tiptop_eval.py override
VIDEO_FPS = 15


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout
        return bool(out.strip())
    except Exception:
        return None


def _git_diff_sha256(path: Path) -> str | None:
    """Hash of tracked-file changes, so each run shows exactly which local patch state it used."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "diff", "HEAD"], capture_output=True, check=True
        ).stdout
        return sha256(out).hexdigest() if out else None
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Observed TiPToP episode (observation-only scoring)")
    p.add_argument("--sim-root", type=Path, default=Path.cwd(), help="droid-sim-evals checkout (default: cwd)")
    p.add_argument("--tiptop-dir", type=Path, default=None, help="tiptop checkout, for recording its SHA")
    p.add_argument("--scene", type=int, required=True)
    p.add_argument("--variant", type=int, required=True)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--instruction", default=None)
    p.add_argument("--target-object", default=None, help="rigid-object name as listed by --inspect-only")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--headless", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--provenance-file", type=Path, default=None)
    p.add_argument("--gripper-bodies", nargs="+", default=DEFAULT_GRIPPER_BODIES)
    p.add_argument("--inspect-only", action="store_true",
                   help="list rigid objects and robot bodies after settling, then exit without planning")
    c = LiftCriteria()
    p.add_argument("--min-rise-m", type=float, default=c.min_rise_m)
    p.add_argument("--min-hold-s", type=float, default=c.min_hold_s)
    p.add_argument("--max-obj-to-gripper-m", type=float, default=c.max_obj_to_gripper_m)
    args, _ = p.parse_known_args(argv)  # leave AppLauncher flags for Isaac Lab
    if not args.inspect_only and (not args.instruction or not args.target_object):
        p.error("--instruction and --target-object are required unless --inspect-only")
    return args


class _RecordingSocket:
    """Pass-through websocket wrapper that remembers text frames (the plan JSON)."""

    def __init__(self, ws, on_text):
        self._ws = ws
        self._on_text = on_text

    def recv(self, *a, **kw):
        msg = self._ws.recv(*a, **kw)
        if isinstance(msg, str):
            self._on_text(msg)
        return msg

    def __getattr__(self, name):
        return getattr(self._ws, name)


def make_observed_client_class(base):
    class ObservedTiptopClient(base):
        """Upstream client plus a copy of the server's raw response. Plan execution is untouched."""

        last_response_text: str | None = None

        def _connect(self, max_retries: int = 12) -> None:
            super()._connect(max_retries)
            self._ws = _RecordingSocket(self._ws, self._remember)

        def _remember(self, text: str) -> None:
            self.last_response_text = text

    return ObservedTiptopClient


def main(argv=None) -> int:
    args = parse_args(argv)
    sim_root = args.sim_root.resolve()
    sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)

    run_id = args.run_id or (
        f"s{args.scene}v{args.variant}t{args.trial}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = args.out_dir.resolve() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    file_log = logging.FileHandler(run_dir / "observed_eval.log")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_log)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Imports below need the droid-sim-evals environment.
    import gymnasium as gym
    import mediapy
    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(argv)
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.inference.tiptop_websocket import TiptopWebsocketClient
    from src.sim_evals.sim_utils import settle_sim
    from src.visual_utils import add_top_padding, overlay_timer_ms

    criteria = LiftCriteria(
        min_rise_m=args.min_rise_m,
        min_hold_s=args.min_hold_s,
        max_obj_to_gripper_m=args.max_obj_to_gripper_m,
    )
    rec = RunRecord(
        run_id=run_id,
        scene=args.scene,
        variant=args.variant,
        trial=args.trial,
        instruction=args.instruction or "",
        target_object=args.target_object or "",
        started_utc=_utc_now(),
        seed=args.seed,
        criteria=criteria.to_dict(),
        versions={
            "project_sha": _git_sha(PROJECT_ROOT),
            "project_dirty": _git_dirty(PROJECT_ROOT),
            "sim_sha": _git_sha(sim_root),
            "sim_dirty": _git_dirty(sim_root),
            "tiptop_sha": _git_sha(args.tiptop_dir) if args.tiptop_dir else None,
            # Non-null when tiptop has local changes (expected: the Gemini ER 2 patch).
            "tiptop_diff_sha256": _git_diff_sha256(args.tiptop_dir) if args.tiptop_dir else None,
            "provenance_file": str(args.provenance_file) if args.provenance_file else None,
            "provenance_sha256": _file_sha256(args.provenance_file) if args.provenance_file else None,
        },
    )
    wall_start = time.time()
    env = client = None
    video = []
    trace = {k: [] for k in ("t", "obj_pos", "gripper_pos", "gripper_cmd", "gripper_obs", "joint_pos")}

    try:
        env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
        env_cfg.set_scene(str(args.scene), args.variant)
        env_cfg.episode_length_s = EPISODE_LENGTH_S
        if args.seed is not None:
            env_cfg.seed = args.seed
        env = gym.make("DROID", cfg=env_cfg)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        step_dt = float(env.unwrapped.step_dt)

        obs, _ = env.reset()
        obs, _ = env.reset()  # upstream: second render cycle for materials

        def object_positions() -> dict[str, list[float]]:
            return {name: obj.data.root_pos_w[0].tolist() for name, obj in scene.rigid_objects.items()}

        if args.inspect_only:
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            info = {
                "scene": args.scene,
                "variant": args.variant,
                "rigid_objects_after_settle": object_positions(),
                "robot_body_names": list(robot.data.body_names),
                "robot_joint_names": list(robot.data.joint_names),
                "step_dt": step_dt,
            }
            path = run_dir / "scene_inspection.json"
            path.write_text(json.dumps(info, indent=2))
            logger.info("Scene inspection written to %s\n%s", path, json.dumps(info, indent=2))
            return 0

        if args.target_object not in scene.rigid_objects:
            raise KeyError(
                f"target object {args.target_object!r} not in scene rigid objects "
                f"{sorted(scene.rigid_objects)}; run --inspect-only"
            )
        target = scene[args.target_object]
        body_ids, body_names = robot.find_bodies(args.gripper_bodies, preserve_order=True)
        if len(body_ids) != len(args.gripper_bodies):
            raise KeyError(f"gripper bodies {args.gripper_bodies} matched {body_names}; run --inspect-only")

        def gripper_ref() -> list[float]:
            return robot.data.body_pos_w[0, body_ids].mean(dim=0).tolist()

        ClientCls = make_observed_client_class(TiptopWebsocketClient)
        try:
            client = ClientCls(host=args.ws_host, port=args.ws_port)
        except Exception as e:
            rec.server_error = f"could not connect to TiPToP server: {e}"
            raise

        max_steps = env.env.max_episode_length
        with torch.no_grad():
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            rec.initial_object_positions = object_positions()
            baseline_z = rec.initial_object_positions[args.target_object][2]

            frame_idx = 0
            for _ in range(max_steps):
                try:
                    ret = client.infer(obs, rec.instruction)
                except Exception as e:  # upstream: log, skip the episode
                    logger.error("Planning failed: %s", e)
                    rec.server_error = rec.server_error or str(e)
                    break

                if client.plan_done:
                    rec.plan_done = True
                    logger.info("Plan fully executed at step %d", frame_idx)
                    break

                viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                viz = add_top_padding(viz, pad_px=40)
                overlay_timer_ms(viz, int(frame_idx * 1000 / VIDEO_FPS))
                video.append(viz)
                frame_idx += 1

                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                # Observation only, after the step.
                trace["t"].append(frame_idx * step_dt)
                trace["obj_pos"].append(target.data.root_pos_w[0].tolist())
                trace["gripper_pos"].append(gripper_ref())
                trace["gripper_cmd"].append(float(ret["action"][-1]))
                trace["gripper_obs"].append(float(obs["policy"]["gripper_pos"][0, 0]))
                trace["joint_pos"].append(obs["policy"]["arm_joint_pos"][0].tolist())

                if term or trunc:
                    rec.terminated = bool(term)
                    rec.truncated = bool(trunc)
                    break

        rec.executed_steps = frame_idx
        rec.sim_duration_s = frame_idx * step_dt
        rec.final_object_positions = object_positions()

        text = client.last_response_text
        if text is not None:
            rec.server_response_received = True
            try:
                resp = json.loads(text)
            except json.JSONDecodeError:
                rec.server_success = False
                rec.server_error = f"non-JSON server reply (server-side exception): {text[:500]}"
            else:
                rec.server_success = bool(resp.get("success"))
                rec.server_error = resp.get("error") or (None if rec.server_success else rec.server_error)
                rec.server_save_dir = resp.get("save_dir")
                steps = ((resp.get("plan") or {}).get("steps")) or []
                steps = [s for s in steps if s.get("type") != "metadata"]
                rec.plan_num_steps = len(steps) if rec.server_success else None
                rec.plan_num_close_steps = (
                    sum(1 for s in steps if s.get("type") == "gripper" and s.get("action") == "close")
                    if rec.server_success else None
                )
        rec.planning_time_s = client.last_planning_time

        if trace["t"]:
            score = score_first_grasp(
                np.array(trace["t"]), np.array(trace["obj_pos"]), np.array(trace["gripper_pos"]),
                np.array(trace["gripper_cmd"]), baseline_z, criteria,
            )
            rec.score = score.to_dict()
        if rec.server_success and rec.executed_steps > 0:
            rec.stable_lift = bool(rec.score and rec.score["lifted"])

    except Exception:
        rec.harness_exception = traceback.format_exc()
        logger.error("Harness exception:\n%s", rec.harness_exception)
    finally:
        if args.inspect_only:
            if env is not None:
                env.close()
            simulation_app.close()
        else:
            try:
                if trace["t"]:
                    trace_path = run_dir / "trace.npz"
                    np.savez_compressed(trace_path, **{k: np.asarray(v) for k, v in trace.items()})
                    rec.artifacts["trace"] = str(trace_path)
                if video:
                    video_path = run_dir / "video.mp4"
                    mediapy.write_video(video_path, video, fps=VIDEO_FPS)
                    rec.artifacts["video"] = str(video_path)
            except Exception:
                # Still write the record; a lost artifact must not lose the run.
                rec.artifacts["artifact_error"] = traceback.format_exc()
            rec.artifacts["log"] = str(run_dir / "observed_eval.log")
            if rec.server_save_dir:
                rec.artifacts["server_outputs"] = rec.server_save_dir
                rec.artifacts["server_plan"] = str(Path(rec.server_save_dir) / "tiptop_plan.json")
                rec.artifacts["server_log"] = str(Path(rec.server_save_dir) / "tiptop_run.log")

            rec.failure_category, rec.failure_detail = auto_label(rec)
            rec.finished_utc = _utc_now()
            rec.wall_duration_s = time.time() - wall_start
            (run_dir / "record.json").write_text(json.dumps(rec.to_dict(), indent=2, sort_keys=True))
            append_jsonl(args.out_dir.resolve() / "runs.jsonl", rec.to_dict())
            logger.info(
                "Run %s: plan=%s executed_steps=%d stable_lift=%s failure=%s",
                run_id, rec.server_success, rec.executed_steps, rec.stable_lift, rec.failure_category,
            )
            if client is not None:
                client.close()
            if env is not None:
                env.close()
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
