"""CPU stand-ins for the task-level checkpoint tests: a multi-object toy world and a planning server.

Toy kinematics as in recovery_fakes: joints 0-2 offset the finger-midpoint position, so a joint path
is a Cartesian path and "vertical" is joint 2. Behaviours, per close (in order of closes):
  "ok"      the object attaches after the hand has risen `lag_m` (grasp settling), then follows;
  "miss"    nothing attaches;
  "slip"    attaches like "ok", then detaches after `slip_after` held steps and falls to the table;
  "drop"    attaches like "ok", then detaches during the carry once the hand is `drop_at_xy_m` away;
  "lost"    the object is knocked off the table at the close.
On open, an attached object is released straight down onto the table (or into the container).
"""

from __future__ import annotations

import numpy as np

from tests.recovery_fakes import FakeTiptopClient, PlanningError, gripper_of, line, q_for  # noqa: F401

TABLE_TOP = 0.0
HALF = 0.02  # cube half-size
REST_Z = TABLE_TOP + HALF
BOWL = np.array([0.75, 0.25, 0.0])  # container center on the table
BOWL_HALF = np.array([0.08, 0.08, 0.03])


class TaskToyWorld:
    step_dt = 1 / 15

    def __init__(self, cubes=None, behaviours=("ok",), lag_m=0.02, slip_after=12, drop_at_xy_m=0.08,
                 episode_steps=100_000):
        cubes = cubes or {"cube": [0.5, 0.0]}
        self.pos = {n: np.array([x, y, REST_Z]) for n, (x, y) in cubes.items()}
        self.pos["bowl"] = BOWL + np.array([0, 0, BOWL_HALF[2]])
        self.object_names = sorted(self.pos)
        self.object_local_bounds = {n: [[-HALF] * 3, [HALF] * 3] for n in cubes}
        self.object_local_bounds["bowl"] = [(-BOWL_HALF).tolist(), BOWL_HALF.tolist()]
        self.q = np.zeros(7)
        self.g = 0.0
        self.behaviours = list(behaviours)
        self.lag_m, self.slip_after, self.drop_at_xy_m = lag_m, slip_after, drop_at_xy_m
        self.closes = 0
        self.steps = 0
        self.episode_steps = episode_steps
        self.held = None  # name of the object in the hand
        self.pending = None  # (name, behaviour, close_z, close_xy) waiting for the lag to pass
        self.offset = None
        self.held_steps = 0
        self.behaviour = None
        self.close_xy = None
        self.open_events: list[dict] = []  # (object rise above rest at the moment of opening)

    def obs(self):
        return {"q": self.q.copy(), "g": self.g, "frame": self.steps}

    def _nearest(self, grip):
        names = [n for n in self.pos if n != "bowl"]
        d = {n: float(np.linalg.norm(self.pos[n] - grip)) for n in names}
        n = min(d, key=d.get)
        return n if d[n] < 0.03 else None

    def _release(self, name):
        p = self.pos[name]
        in_bowl = np.all(np.abs(p[:2] - BOWL[:2]) < BOWL_HALF[:2] - HALF)
        self.pos[name] = np.array([p[0], p[1], (BOWL[2] + HALF + 0.005) if in_bowl else REST_Z])

    def step(self, action):
        action = np.asarray(action, dtype=float)
        was_closed = self.g > 0.5
        self.q = action[:7].copy()
        self.g = float(action[7])
        grip = gripper_of(self.q)
        if self.g > 0.5 and not was_closed:
            b = self.behaviours[min(self.closes, len(self.behaviours) - 1)]
            self.closes += 1
            n = self._nearest(grip)
            if n is not None and b == "lost":
                self.pos[n] = np.array([self.pos[n][0], self.pos[n][1], -1.0])
            elif n is not None and b != "miss":
                self.pending = (n, b, grip[2], grip[:2].copy())
        if self.g <= 0.5 and (self.held or self.pending):
            if self.held:
                self.open_events.append({"object": self.held, "rise": float(self.pos[self.held][2] - REST_Z)})
                self._release(self.held)
            self.held = self.pending = None
        if self.pending is not None:
            n, b, z0, xy0 = self.pending
            if grip[2] - z0 >= self.lag_m:
                self.held, self.behaviour, self.close_xy = n, b, xy0
                self.offset = self.pos[n] - grip
                self.held_steps = 0
                self.pending = None
        if self.held is not None:
            self.held_steps += 1
            n = self.held
            detach = ((self.behaviour == "slip" and self.held_steps > self.slip_after)
                      or (self.behaviour == "drop" and np.linalg.norm(grip[:2] - self.close_xy) > self.drop_at_xy_m))
            if detach:
                self.held = None
                self._release(n)
            else:
                self.pos[n] = grip + self.offset
        self.steps += 1
        return self.obs(), False, self.steps >= self.episode_steps

    # TaskWorld interface
    def object_positions(self):
        return {n: p.tolist() for n, p in self.pos.items()}

    def object_quats(self):
        return {n: [1.0, 0.0, 0.0, 0.0] for n in self.pos}

    def gripper_ref(self):
        return gripper_of(self.q).tolist()

    def joint_pos(self, obs):
        return np.asarray(obs["q"], dtype=float)

    def gripper_obs(self, obs):
        return float(obs["g"])

    def vertical_lift_step(self, q_cmd, dz):
        q = np.asarray(q_cmd, dtype=float).copy()
        q[2] += dz
        return q

    def set_object_pose(self, name, pos, quat):
        self.pos[name] = np.asarray(pos, dtype=float)

    def in_bowl(self, name) -> bool:
        p = self.pos[name]
        return bool(np.all(np.abs(p[:2] - BOWL[:2]) < BOWL_HALF[:2]) and p[2] < 0.1)


def task_plan(world: TaskToyWorld, q_start, targets, save_dir="/srv/out/x") -> dict:
    """cuTAMP-0.0.6-shaped plan picking `targets` in order and placing each in the bowl."""
    steps, q = [], np.asarray(q_start, dtype=float)
    for i, name in enumerate(targets):
        obj = np.asarray(world.pos[name], dtype=float)
        q_above = q_for(obj + [0, 0, 0.12])
        q_grasp = q_for(obj)
        q_ret = q_for(obj + [0, 0, 0.05])
        drop_xy = BOWL[:2] + np.array([0.03 * (i - 0.5), 0.0])
        q_bowl_app = q_for([drop_xy[0], drop_xy[1], 0.18])
        q_bowl = q_for([drop_xy[0], drop_xy[1], 0.08])
        pick = f"Pick({name}, grasp{i + 1}, q{2 * i + 1})"
        place = f"Place({name}, grasp{i + 1}, pose{i + 1}, bowl, q{2 * i + 2})"
        steps += [
            {"type": "trajectory", "label": pick, "dt": 0.02, "positions": line(q, q_above, 40)},
            {"type": "trajectory", "label": pick, "dt": 0.02, "positions": line(q_above, q_grasp, 60)},  # 2 mm apart
            {"type": "gripper", "label": pick, "action": "close"},
            {"type": "trajectory", "label": place, "dt": 0.02, "positions": line(q_grasp, q_ret, 26)},
            {"type": "trajectory", "label": place, "dt": 0.02, "positions": line(q_ret, q_bowl_app, 12)},
            {"type": "trajectory", "label": place, "dt": 0.02, "positions": line(q_bowl_app, q_bowl, 9)},
            {"type": "gripper", "label": place, "action": "open"},
        ]
        q = q_bowl
    steps += [{"type": "trajectory", "label": "GoToInitial(q0)", "dt": 0.02, "positions": line(q, q_start, 12)}]
    return {"version": "1.0.0", "q_init": list(map(float, q_start)), "steps": [{"type": "metadata"}] + steps,
            "_save_dir": save_dir}


class TaskServer:
    """Plans every goal object not yet in the bowl, from the current world state (like cuTAMP, it does not
    re-pick an object whose goal already holds). `script[i]`: None to plan, or an exception to raise."""

    def __init__(self, world: TaskToyWorld, goal_objects, script=None):
        self.world, self.goal_objects, self.script = world, list(goal_objects), list(script or [])
        self.calls = 0
        self.planned: list[list[str]] = []

    def __call__(self, request):
        i = self.calls
        self.calls += 1
        item = self.script[i] if i < len(self.script) else None
        if isinstance(item, BaseException):
            raise item
        todo = [n for n in self.goal_objects if not self.world.in_bowl(n)]
        self.planned.append(todo)
        return task_plan(self.world, request["q_init"], todo, save_dir=f"/srv/out/{i}")
