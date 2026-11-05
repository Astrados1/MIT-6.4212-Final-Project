"""CPU stand-ins for week-two tests: an upstream-faithful TiPToP client and a toy world.

FakeTiptopClient copies the plan-execution bookkeeping of TiptopWebsocketClient at
droid-sim-evals 69a914c (src/sim_evals/inference/tiptop_websocket.py): waypoint stride
round(50/15)=3 keeping the last waypoint, 20-tick gripper dwell holding measured joints,
`_current_plan_step` advanced when a trajectory starts, `plan_done` as upstream. The server
is replaced by a callable that returns a plan dict or raises.
"""

from __future__ import annotations

import json

import numpy as np


class PlanningError(Exception):
    """Same name as the upstream exception (recovery_eval classifies by name)."""


class FakeTiptopClient:
    replay_plan_path = None

    def __init__(self, server, gripper_action_steps: int = 20, stride: int = 3):
        self._server = server  # callable(request) -> plan dict, or raises
        self._gripper_action_steps = gripper_action_steps
        self._waypoint_stride = stride
        self.connections = 0
        self.requests: list[dict] = []
        self.last_response_text = None
        self.last_request_summary = None
        self._plan = None
        self._current_plan_step = 0
        self._current_trajectory = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0
        self._action_chunk_done = False
        self._last_planning_time = None
        self._connect()

    def _connect(self):
        self.connections += 1

    @property
    def last_planning_time(self):
        return self._last_planning_time

    @property
    def plan_done(self) -> bool:
        if self._plan is None:
            return False
        if self._gripper_action_pending is not None:
            return False
        if self._current_trajectory is not None and self._current_waypoint_idx < len(self._current_trajectory):
            return False
        return self._current_plan_step >= len(self._plan)

    def reset(self):
        self._plan = None
        self._current_plan_step = 0
        self._current_trajectory = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0
        self._action_chunk_done = False
        self._last_planning_time = None
        self._connect()

    def infer(self, obs, instruction):
        curr = {"joint_position": np.asarray(obs["q"], dtype=float)[None], "gripper_position": np.array([obs["g"]])}
        if self._plan is None:
            self._query_server(obs, curr, instruction)
        return self._step_plan(curr)

    def _query_server(self, obs, curr, instruction):
        request = {"q_init": np.asarray(obs["q"], dtype=np.float32), "rgb": np.array([obs["frame"]]),
                   "depth": np.array([obs["frame"]], dtype=np.float32), "world_from_cam": np.eye(4),
                   "task": instruction}
        self.requests.append({"q_init": request["q_init"].tolist(), "frame": obs["frame"], "task": instruction})
        self.last_request_summary = {"q_init": request["q_init"].tolist(), "frame": obs["frame"], "task": instruction}
        self._last_planning_time = 1.5
        try:
            plan = self._server(request)
        except PlanningError as e:
            self.last_response_text = json.dumps({"success": False, "error": str(e), "plan": None,
                                                  "save_dir": None})
            self._plan = []
            raise
        steps = plan["steps"]
        self.last_response_text = json.dumps({"success": True, "error": None, "save_dir": plan.get("_save_dir"),
                                              "plan": {"steps": [{k: (v.tolist() if hasattr(v, "tolist") else v)
                                                                  for k, v in s.items()} for s in steps]}})
        for s in steps:
            if s["type"] == "trajectory":
                s["positions"] = np.asarray(s["positions"], dtype=np.float32)
        self._plan = [s for s in steps if s["type"] != "metadata"]

    def _subsample(self, traj):
        if self._waypoint_stride <= 1 or len(traj) == 0:
            return traj
        idx = np.arange(0, len(traj), self._waypoint_stride)
        if idx[-1] != len(traj) - 1:
            idx = np.append(idx, len(traj) - 1)
        return traj[idx]

    def _step_plan(self, curr):  # verbatim logic of the upstream _step_plan
        self._action_chunk_done = False
        if self._gripper_action_pending is not None:
            if self._gripper_action_steps_remaining > 0:
                self._gripper_action_steps_remaining -= 1
                g = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = g
                return {"action": np.concatenate([curr["joint_position"].flatten(), [g]])}
            self._last_gripper_state = 1.0 if self._gripper_action_pending == "close" else 0.0
            self._gripper_action_pending = None
            self._current_plan_step += 1
            self._action_chunk_done = True
        if self._current_trajectory is None or self._current_waypoint_idx >= len(self._current_trajectory):
            if self._plan is None or self._current_plan_step >= len(self._plan):
                g = curr["gripper_position"][0]
                return {"action": np.concatenate([curr["joint_position"].flatten(), [g]])}
            step = self._plan[self._current_plan_step]
            if step["type"] == "gripper":
                self._gripper_action_pending = step["action"]
                self._gripper_action_steps_remaining = self._gripper_action_steps
                g = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = g
                return {"action": np.concatenate([curr["joint_position"].flatten(), [g]])}
            self._current_trajectory = self._subsample(step["positions"])
            self._current_waypoint_idx = 0
            self._current_plan_step += 1
        wp = self._current_trajectory[self._current_waypoint_idx]
        self._current_waypoint_idx += 1
        if self._current_waypoint_idx >= len(self._current_trajectory):
            self._action_chunk_done = True
        return {"action": np.concatenate([wp, [self._last_gripper_state]])}

    def close(self):
        pass


# --------------------------------------------------------------------------- toy world

TABLE_Z = 0.03  # resting height of the cube centre


def gripper_of(q) -> np.ndarray:
    """Toy kinematics: the first three joints are the finger-midpoint position offsets."""
    q = np.asarray(q, dtype=float)
    return np.array([0.5 + q[0], q[1], 0.3 + q[2]])


def q_for(pos) -> np.ndarray:
    q = np.zeros(7)
    q[:3] = np.asarray(pos, dtype=float) - np.array([0.5, 0.0, 0.3])
    return q


def line(q0, q1, n=9) -> list[list[float]]:
    return [list(np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * i / (n - 1)) for i in range(n)]


def scene_plan(obj_pos, q_start, target="rubiks_cube", save_dir="/srv/out/1", pick_only=False) -> dict:
    """A plan with the cuTAMP 0.0.6 structure of tiptop_scene1_plan.json, in toy joint space."""
    obj_pos = np.asarray(obj_pos, dtype=float)
    q_app = q_for(obj_pos + [0, 0, 0.12])
    q_grasp = q_for(obj_pos)
    q_ret = q_for(obj_pos + [0, 0, 0.05])  # approach_offset: 5 cm back along the grasp axis
    q_bowl_app = q_for([0.7, 0.2, 0.15])
    q_bowl = q_for([0.7, 0.2, 0.08])
    pick = f"Pick({target}, grasp1, q1)"
    place = f"Place({target}, grasp1, pose1, red_bowl, q2)"
    steps = [
        {"type": "trajectory", "label": pick, "dt": 0.02, "positions": line(q_start, q_app)},
        {"type": "trajectory", "label": pick, "dt": 0.02, "positions": line(q_app, q_grasp)},
        {"type": "gripper", "label": pick, "action": "close"},
    ]
    if pick_only:
        steps += [{"type": "trajectory", "label": "GoToInitial(q0)", "dt": 0.02, "positions": line(q_grasp, q_ret)},
                  {"type": "trajectory", "label": "GoToInitial(q0)", "dt": 0.02, "positions": line(q_ret, q_start)}]
    else:
        steps += [
            {"type": "trajectory", "label": place, "dt": 0.02, "positions": line(q_grasp, q_ret)},
            {"type": "trajectory", "label": place, "dt": 0.02, "positions": line(q_ret, q_bowl_app)},
            {"type": "trajectory", "label": place, "dt": 0.02, "positions": line(q_bowl_app, q_bowl)},
            {"type": "gripper", "label": place, "action": "open"},
            {"type": "trajectory", "label": "GoToInitial(q0)", "dt": 0.02, "positions": line(q_bowl, q_start)},
        ]
    return {"version": "1.0.0", "q_init": list(q_start), "steps": [{"type": "metadata"}] + steps,
            "_save_dir": save_dir}


class ToyWorld:
    """Joints track commands exactly; the cube attaches if the close happens at it and grasp_ok."""

    step_dt = 1 / 15

    def __init__(self, grasp_outcomes=(True,), episode_steps=10_000):
        self.object_names = ["bowl", "cube"]
        self.pos = {"cube": np.array([0.5, 0.0, TABLE_Z]), "bowl": np.array([0.7, 0.2, 0.02])}
        self.q = np.zeros(7)
        self.g = 0.0
        self.attached = False
        self.offset = None
        self.outcomes = list(grasp_outcomes)
        self.closes_seen = 0
        self.steps = 0
        self.episode_steps = episode_steps
        self.resets = 0  # never incremented by the backend: there is no reset method to call
        self.bump_bowl_during_return = False
        self.drop_after_steps_held = None  # simulate a slip: detach after N held steps
        self._held_steps = 0

    def obs(self):
        return {"q": self.q.copy(), "g": self.g, "frame": self.steps}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        was_closed = self.g > 0.5
        self.q = action[:7].copy()
        self.g = float(action[7])
        grip = gripper_of(self.q)
        if self.g > 0.5 and not was_closed:
            ok = self.outcomes[min(self.closes_seen, len(self.outcomes) - 1)]
            self.closes_seen += 1
            if ok and np.linalg.norm(grip - self.pos["cube"]) < 0.02:
                self.attached, self.offset, self._held_steps = True, self.pos["cube"] - grip, 0
        if self.g <= 0.5 and self.attached:
            self.attached = False
            self.pos["cube"] = np.array([*self.pos["cube"][:2], TABLE_Z])
        if self.attached:
            self._held_steps += 1
            if self.drop_after_steps_held is not None and self._held_steps > self.drop_after_steps_held:
                self.attached = False
                self.pos["cube"] = np.array([*self.pos["cube"][:2], TABLE_Z])
            else:
                self.pos["cube"] = grip + self.offset
        if self.bump_bowl_during_return and self.g <= 0.5 and self.closes_seen > 0:
            self.pos["bowl"] = self.pos["bowl"] + np.array([0.002, 0, 0])
        self.steps += 1
        return self.obs(), False, self.steps >= self.episode_steps

    def object_positions(self):
        return {n: p.tolist() for n, p in self.pos.items()}

    def object_quats(self):
        return {n: [1.0, 0.0, 0.0, 0.0] for n in self.pos}

    def gripper_ref(self):
        return gripper_of(self.q).tolist()

    def gripper_quat(self):
        return [1.0, 0.0, 0.0, 0.0]

    def finger_closure(self):
        return None

    def joint_pos(self, obs):
        return np.asarray(obs["q"], dtype=float)

    def gripper_obs(self, obs):
        return float(obs["g"])

    def set_object_pose(self, name, pos, quat):
        self.pos[name] = np.asarray(pos, dtype=float)
