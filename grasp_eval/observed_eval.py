"""One observed TiPToP episode in droid-sim-evals, with observation-only grasp scoring and optional probes.

Run from the droid-sim-evals checkout, inside its uv environment:

    cd $TIPTOP_DIR/droid-sim-evals
    uv run python $PROJECT/grasp_eval/observed_eval.py --inspect-only --scene 1 --variant 0 \
        --out-dir $PROJECT/results/inspect
    uv run python $PROJECT/grasp_eval/observed_eval.py --scene 1 --variant 0 --trial 0 \
        --instruction "Put the Rubik's cube in the bowl." --target-object <name from inspect> \
        --out-dir $PROJECT/results/pilot_scene1

Without probe flags, the control loop mirrors upstream tiptop_eval.py at SIM_SHA
(provenance/pins.env): same app launch, env config, 90 s episode limit, double reset,
settle_sim, client, plan execution and break conditions. What this file adds only reads
state: all rigid-object poses, the gripper reference point and commanded/measured joints
after each step, the server's raw JSON response, a trace, a video, and a run record.
Differences that remain in that baseline mode:
  * one episode per process (upstream loops episodes in one process), so each
    trial gets a fresh connection and a fresh plan;
  * client.reset() is not called after the episode, because the process exits;
  * a video is saved for every run that executed any steps;
  * Python/numpy/torch seeds are set and recorded (upstream sets none).

Probe flags (docs/probes.md) deliberately change the experiment and are recorded in
record.probe. Results with probes must never be mixed with baseline results:
  * --plan-json: execute a saved TiPToP plan instead of asking the server;
  * --displace-*: move the target object after the plan is obtained, before motion;
  * --depth-noise-std-m / --depth-dropout-frac / --cam-trans-mm / --cam-rot-deg:
    corrupt the observation sent to TiPToP (the simulator itself is unchanged).
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

from grasp_eval.attribution import normalize_name, pick_objects  # noqa: E402
from grasp_eval.probe_math import corrupt_depth, displaced_pose, perturb_camera_pose  # noqa: E402
from grasp_eval.records import RunRecord, append_jsonl, auto_label  # noqa: E402
from grasp_eval.scoring import LiftCriteria, score_first_grasp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("observed_eval")

# Robotiq 2F-85 finger links; their midpoint is the gripper reference point.
# Unverified against the flattened USD: confirm with --inspect-only.
DEFAULT_GRIPPER_BODIES = ["left_inner_finger", "right_inner_finger"]
EPISODE_LENGTH_S = 90.0  # upstream tiptop_eval.py override
VIDEO_FPS = 15
FROM_PLAN = "from-plan"


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
    p.add_argument("--target-object", default=None,
                   help=f"rigid-object name as listed by --inspect-only, or '{FROM_PLAN}' to use the plan's first Pick")
    p.add_argument("--object-aliases", type=Path, default=None,
                   help="JSON {perceived label: sim object name} used with --target-object from-plan")
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

    g = p.add_argument_group("probes (change the experiment; see docs/probes.md)")
    g.add_argument("--probe-label", default=None, help="condition name recorded in record.probe")
    g.add_argument("--probe-seed", type=int, default=None, help="RNG seed for sensor corruption (default: --seed)")
    g.add_argument("--plan-json", type=Path, default=None, help="replay this saved tiptop_plan.json")
    g.add_argument("--displace-dx-m", type=float, default=0.0)
    g.add_argument("--displace-dy-m", type=float, default=0.0)
    g.add_argument("--displace-yaw-deg", type=float, default=0.0)
    g.add_argument("--depth-noise-std-m", type=float, default=0.0)
    g.add_argument("--depth-dropout-frac", type=float, default=0.0)
    g.add_argument("--cam-trans-mm", type=float, default=0.0)
    g.add_argument("--cam-rot-deg", type=float, default=0.0)

    args, _ = p.parse_known_args(argv)  # leave AppLauncher flags for Isaac Lab
    if not args.inspect_only and (not args.instruction or not args.target_object):
        p.error("--instruction and --target-object are required unless --inspect-only")
    args.displace = any(v != 0 for v in (args.displace_dx_m, args.displace_dy_m, args.displace_yaw_deg))
    args.corrupt = any(v != 0 for v in (args.depth_noise_std_m, args.depth_dropout_frac,
                                        args.cam_trans_mm, args.cam_rot_deg))
    if args.displace and args.target_object == FROM_PLAN:
        p.error("displacement needs an explicit --target-object (the object to move)")
    if args.corrupt and args.plan_json:
        p.error("sensor corruption only affects fresh plans; it cannot be combined with --plan-json")
    return args


def probe_description(args) -> dict:
    if not (args.displace or args.corrupt or args.plan_json):
        return {"type": "baseline", **({"label": args.probe_label} if args.probe_label else {})}
    kinds = [k for k, on in (("replay", args.plan_json), ("displacement", args.displace),
                             ("sensor", args.corrupt)) if on]
    desc = {"type": "+".join(kinds), "label": args.probe_label}
    if args.displace:
        desc["displacement"] = {"dx_m": args.displace_dx_m, "dy_m": args.displace_dy_m,
                                "yaw_deg": args.displace_yaw_deg, "when": "after_plan_before_motion"}
    if args.corrupt:
        desc["sensor"] = {"depth_noise_std_m": args.depth_noise_std_m, "depth_dropout_frac": args.depth_dropout_frac,
                          "cam_trans_mm": args.cam_trans_mm, "cam_rot_deg": args.cam_rot_deg}
    return desc


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
        """Upstream client plus a copy of the server's raw response.

        Plan execution (_step_plan) is never overridden. Probe hooks: a saved plan can
        replace the server query, and the outgoing observation can be corrupted.
        """

        last_response_text: str | None = None
        replay_plan_path: Path | None = None
        corruption: dict | None = None
        corruption_applied: dict | None = None
        replay_q_init: np.ndarray | None = None

        def _connect(self, max_retries: int = 12) -> None:
            if self.replay_plan_path is not None:
                self._ws = None  # replay needs no server
                return
            super()._connect(max_retries)
            self._ws = _RecordingSocket(self._ws, self._remember)

        def _remember(self, text: str) -> None:
            self.last_response_text = text

        def _build_request(self, raw_obs, curr_obs, instruction):
            request = super()._build_request(raw_obs, curr_obs, instruction)
            c = self.corruption
            if c:
                rng = np.random.default_rng(c["seed"])
                applied = {"seed": c["seed"]}
                if c["depth_noise_std_m"] or c["depth_dropout_frac"]:
                    request["depth"] = corrupt_depth(request["depth"], c["depth_noise_std_m"],
                                                     c["depth_dropout_frac"], rng)
                    applied["depth_noise_std_m"] = c["depth_noise_std_m"]
                    applied["depth_dropout_frac"] = c["depth_dropout_frac"]
                if c["cam_trans_mm"] or c["cam_rot_deg"]:
                    request["world_from_cam"], applied["camera"] = perturb_camera_pose(
                        request["world_from_cam"], c["cam_trans_mm"] / 1000.0, c["cam_rot_deg"], rng)
                self.corruption_applied = applied
            return request

        def _query_server(self, raw_obs, curr_obs, instruction):
            if self.replay_plan_path is None:
                return super()._query_server(raw_obs, curr_obs, instruction)
            text = self.replay_plan_path.read_text()
            plan = json.loads(text)
            self.last_response_text = json.dumps({
                "success": True, "plan": plan, "error": None,
                "save_dir": str(self.replay_plan_path.resolve().parent),
            })
            # Same post-processing as upstream _query_server for a successful response.
            steps = plan["steps"]
            for step in steps:
                if step["type"] == "trajectory":
                    step["positions"] = np.array(step["positions"], dtype=np.float32)
                    if "velocities" in step:
                        step["velocities"] = np.array(step["velocities"], dtype=np.float32)
            self._plan = [s for s in steps if s["type"] != "metadata"]
            self._last_planning_time = None
            if plan.get("q_init") is not None:
                self.replay_q_init = np.asarray(plan["q_init"], dtype=float)

    return ObservedTiptopClient


def resolve_target(requested: str, plan: dict | None, names: list[str], aliases: dict,
                   first_close_idx: int | None, objects_pos: np.ndarray | None,
                   gripper_pos: np.ndarray | None) -> tuple[str | None, str]:
    """Explicit name, else the plan's first Pick mapped to a sim name, else nearest object at first close."""
    if requested != FROM_PLAN:
        return requested, "explicit"
    picks = pick_objects(plan)
    if picks:
        label = picks[0]
        if label in aliases and aliases[label] in names:
            return aliases[label], "plan_label"
        exact = [n for n in names if normalize_name(n) == normalize_name(label)]
        partial = [n for n in names if normalize_name(label) in normalize_name(n) or normalize_name(n) in normalize_name(label)]
        if len(exact) == 1:
            return exact[0], "plan_label"
        if len(partial) == 1:
            return partial[0], "plan_label"
    if first_close_idx is not None and objects_pos is not None and gripper_pos is not None:
        d = np.linalg.norm(objects_pos[first_close_idx] - gripper_pos[first_close_idx], axis=1)
        return names[int(np.argmin(d))], "nearest_at_close"
    return None, "unresolved"


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
        probe=probe_description(args),
        plan_source="replay" if args.plan_json else "server",
        replayed_plan=str(args.plan_json.resolve()) if args.plan_json else None,
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
    aliases = json.loads(args.object_aliases.read_text()) if args.object_aliases else {}
    wall_start = time.time()
    env = client = None
    video = []
    trace = {k: [] for k in ("t", "objects_pos", "gripper_pos", "gripper_cmd", "gripper_obs",
                             "joint_pos", "joint_cmd")}
    object_names: list[str] = []

    try:
        env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
        env_cfg.set_scene(str(args.scene), args.variant)
        env_cfg.episode_length_s = EPISODE_LENGTH_S
        if args.seed is not None:
            env_cfg.seed = args.seed
        env = gym.make("DROID", cfg=env_cfg)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        device = env.unwrapped.device
        step_dt = float(env.unwrapped.step_dt)
        object_names = sorted(scene.rigid_objects.keys())

        obs, _ = env.reset()
        obs, _ = env.reset()  # upstream: second render cycle for materials

        def object_positions() -> dict[str, list[float]]:
            return {name: scene.rigid_objects[name].data.root_pos_w[0].tolist() for name in object_names}

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

        if args.target_object != FROM_PLAN and args.target_object not in scene.rigid_objects:
            raise KeyError(
                f"target object {args.target_object!r} not in scene rigid objects "
                f"{object_names}; run --inspect-only"
            )
        body_ids, body_names = robot.find_bodies(args.gripper_bodies, preserve_order=True)
        if len(body_ids) != len(args.gripper_bodies):
            raise KeyError(f"gripper bodies {args.gripper_bodies} matched {body_names}; run --inspect-only")

        def gripper_ref() -> list[float]:
            return robot.data.body_pos_w[0, body_ids].mean(dim=0).tolist()

        ClientCls = make_observed_client_class(TiptopWebsocketClient)
        if args.plan_json:
            ClientCls.replay_plan_path = args.plan_json.resolve()
        if args.corrupt:
            ClientCls.corruption = {
                "seed": args.probe_seed if args.probe_seed is not None else (args.seed or 0),
                "depth_noise_std_m": args.depth_noise_std_m, "depth_dropout_frac": args.depth_dropout_frac,
                "cam_trans_mm": args.cam_trans_mm, "cam_rot_deg": args.cam_rot_deg,
            }
        try:
            client = ClientCls(host=args.ws_host, port=args.ws_port)
        except Exception as e:
            rec.server_error = f"could not connect to TiPToP server: {e}"
            raise

        def apply_displacement() -> None:
            target = scene[args.target_object]
            pos = target.data.root_pos_w[0].cpu().numpy()
            quat = target.data.root_quat_w[0].cpu().numpy()
            new_pos, new_quat = displaced_pose(pos, quat, args.displace_dx_m, args.displace_dy_m,
                                               args.displace_yaw_deg)
            pose = torch.tensor([[*new_pos, *new_quat]], dtype=torch.float32, device=device)
            target.write_root_pose_to_sim(pose)
            target.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))
            others = {n: p for n, p in object_positions().items() if n != args.target_object}
            nearest = min((float(np.linalg.norm(np.asarray(p[:2]) - new_pos[:2])), n) for n, p in others.items()) \
                if others else (None, None)
            rec.probe["displacement"].update({
                "applied": True, "pos_before": pos.tolist(), "pos_after": new_pos.tolist(),
                "quat_before": quat.tolist(), "quat_after": new_quat.tolist(),
                "nearest_other_object": nearest[1], "nearest_other_xy_m": nearest[0],
            })

        max_steps = env.env.max_episode_length
        displaced = False
        with torch.no_grad():
            obs, _ = env.reset()
            obs = settle_sim(env, obs, reset_episode_buf=True)
            rec.initial_object_positions = object_positions()
            q_now = obs["policy"]["arm_joint_pos"][0].cpu().numpy()

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

                if args.displace and not displaced:
                    apply_displacement()  # probe: after the plan exists, before the first motion step
                    displaced = True

                viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                viz = add_top_padding(viz, pad_px=40)
                overlay_timer_ms(viz, int(frame_idx * 1000 / VIDEO_FPS))
                video.append(viz)
                frame_idx += 1

                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                # Observation only, after the step.
                trace["t"].append(frame_idx * step_dt)
                trace["objects_pos"].append(
                    [scene.rigid_objects[n].data.root_pos_w[0].tolist() for n in object_names])
                trace["gripper_pos"].append(gripper_ref())
                trace["gripper_cmd"].append(float(ret["action"][-1]))
                trace["gripper_obs"].append(float(obs["policy"]["gripper_pos"][0, 0]))
                trace["joint_pos"].append(obs["policy"]["arm_joint_pos"][0].tolist())
                trace["joint_cmd"].append([float(v) for v in ret["action"][:7]])

                if term or trunc:
                    rec.terminated = bool(term)
                    rec.truncated = bool(trunc)
                    break

        rec.executed_steps = frame_idx
        rec.sim_duration_s = frame_idx * step_dt
        rec.final_object_positions = object_positions()
        if client.replay_q_init is not None:
            rec.plan_q_init_max_abs_diff_rad = float(np.max(np.abs(client.replay_q_init - q_now)))
        if client.corruption_applied:
            rec.probe["sensor_applied"] = client.corruption_applied

        plan = None
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
                plan = resp.get("plan") or None
                steps = [s for s in ((plan or {}).get("steps") or []) if s.get("type") != "metadata"]
                rec.plan_num_steps = len(steps) if rec.server_success else None
                rec.plan_num_close_steps = (
                    sum(1 for s in steps if s.get("type") == "gripper" and s.get("action") == "close")
                    if rec.server_success else None
                )
        rec.planning_time_s = client.last_planning_time

        if trace["t"]:
            objects_pos = np.asarray(trace["objects_pos"])
            gripper_pos = np.asarray(trace["gripper_pos"])
            closes = np.flatnonzero(np.asarray(trace["gripper_cmd"]) > criteria.close_threshold)
            target, rec.target_source = resolve_target(
                args.target_object, plan, object_names, aliases,
                int(closes[0]) if closes.size else None, objects_pos, gripper_pos)
            if target is not None:
                rec.target_object = target
                k = object_names.index(target)
                score = score_first_grasp(
                    np.array(trace["t"]), objects_pos[:, k, :], gripper_pos,
                    np.array(trace["gripper_cmd"]), rec.initial_object_positions[target][2], criteria,
                )
                rec.score = score.to_dict()
        elif args.target_object == FROM_PLAN:
            rec.target_source = "unresolved"
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
                    arrays = {k: np.asarray(v) for k, v in trace.items()}
                    np.savez_compressed(trace_path, object_names=np.asarray(object_names), **arrays)
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
                rec.artifacts["server_plan"] = str(
                    args.plan_json.resolve() if args.plan_json else Path(rec.server_save_dir) / "tiptop_plan.json")
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
