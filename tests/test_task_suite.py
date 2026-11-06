"""CPU tests for the paired task suite, its summary, and the checkpoint calibration tool."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from grasp_eval import task_suite
from grasp_eval.checkpoint_calibration import analyse_grasp, grasp_windows
from grasp_eval.task_checkpoint import TaskCheckCriteria
from grasp_eval.task_summary import paired_rows, sign_test_p, summarize

GOAL = [{"objects": ["$target"], "relation": "in", "container": "_24_bowl"}]
REAL_SCENES = Path(__file__).resolve().parents[1] / "configs" / "scenes.json"


def make_baseline(d: Path, runs):
    """runs: (run_id, scene, plan?, task_success)."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "runs.jsonl", "w") as f, open(d / "attribution.jsonl", "w") as g:
        for rid, scene, has_plan, ok in runs:
            f.write(json.dumps({"run_id": rid, "scene": scene, "variant": 0, "trial": 0, "seed": 2000,
                                "instruction": "Put the Rubik's cube in the bowl.", "target_object": "rubiks_cube",
                                "goal": GOAL, "server_success": has_plan,
                                "server_save_dir": f"/srv/{rid}" if has_plan else None}) + "\n")
            if has_plan:
                g.write(json.dumps({"run_id": rid, "task_check": {"task_success": ok}}) + "\n")


class TaskSuiteTest(unittest.TestCase):
    def setUp(self):
        self._o = contextlib.redirect_stdout(io.StringIO())
        self._o.__enter__()
        self._e = contextlib.redirect_stderr(io.StringIO())
        self._e.__enter__()

    def tearDown(self):
        self._o.__exit__(None, None, None)
        self._e.__exit__(None, None, None)

    def test_trials_replay_the_baseline_plan_and_skip_planless_runs(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"
            make_baseline(base, [("000_a", 1, True, False), ("001_b", 4, False, None)])
            ts = task_suite.baseline_trials(base, {"1": {"goal": GOAL}})
            self.assertEqual(ts[0]["plan"], str(Path("/srv/000_a") / "tiptop_plan.json"))
            self.assertIsNone(ts[1]["plan"])
            a = task_suite.parse_args(["--baseline", str(base), "--tiptop-dir", d, "--lift-mode", "vertical_ik",
                                       "--eval-args", "--hold-s", "0.5"])
            cmd = task_suite.command(ts[0], a, Path(d) / "out", Path(d) / "sim", Path(d) / "tt", base)
            self.assertIn("--plan-json", cmd)
            self.assertEqual(cmd[cmd.index("--paired-baseline-run") + 1], "000_a")
            self.assertEqual(cmd[cmd.index("--seed") + 1], "2000")
            self.assertEqual(json.loads(cmd[cmd.index("--goal-json") + 1]), GOAL)
            self.assertEqual(cmd[-2:], ["--hold-s", "0.5"])  # eval args last
            self.assertEqual(cmd[cmd.index("--lift-mode") + 1], "vertical_ik")

    def test_suite_runs_pairs_records_skips_and_stops_on_quota(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"
            make_baseline(base, [("000_a", 1, True, False), ("001_b", 1, False, None), ("002_c", 1, True, True),
                                 ("003_d", 1, True, True)])
            calls = []

            def fake_run(cmd, cwd, log, timeout):
                tid = cmd[cmd.index("--trial-id") + 1]
                calls.append(tid)
                out = Path(cmd[cmd.index("--out-dir") + 1])
                quota = tid.startswith("002_c")
                rec = {"trial_id": tid, "scene": 1, "paired_baseline_run": cmd[cmd.index("--paired-baseline-run") + 1],
                       "task_success": not quota, "task_success_at_baseline_limit": not quota, "num_retries": 1,
                       "stop_reason": "plan_executed", "first_checkpoint_verdict": "empty",
                       "checkpoints": [{"verdict": "empty"}, {"verdict": "held"}], "events": [],
                       "plans": [{"source": "replay"}, {"source": "server", "server_success": not quota,
                                                        "server_error": "429 RESOURCE_EXHAUSTED" if quota else None}]}
                (out / "trials" / tid / "record.json").write_text(json.dumps(rec))
                with open(out / "task_trials.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return 0

            with mock.patch.object(task_suite, "PROJECT_ROOT", Path(d)), \
                    mock.patch.object(task_suite, "run_process", fake_run):
                rc = task_suite.main(["--baseline", str(base), "--tiptop-dir", d, "--no-health-check",
                                      "--out-name", "t", "--scenes-config", str(REAL_SCENES)])
            out = Path(d) / "results" / "t"
            self.assertEqual(rc, 3)  # stopped on the quota error
            self.assertEqual([c.split("_rec")[0] for c in calls], ["000_a", "002_c"])
            skipped = [json.loads(x) for x in (out / "skipped.jsonl").read_text().splitlines()]
            self.assertEqual([s["run_id"] for s in skipped], ["001_b"])
            self.assertTrue((out / "summary_task.md").exists())


class TaskSummaryTest(unittest.TestCase):
    def test_sign_test(self):
        self.assertAlmostEqual(sign_test_p(6, 0), 2 / 64)
        self.assertEqual(sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(sign_test_p(3, 3), 1.0)

    def test_paired_counts(self):
        with tempfile.TemporaryDirectory() as d:
            base, suite = Path(d) / "base", Path(d) / "suite"
            make_baseline(base, [("r1", 1, True, False), ("r2", 1, True, True), ("r3", 2, True, True),
                                 ("r4", 2, False, None)])
            suite.mkdir()
            recs = [("r1", True, False, 1), ("r2", True, True, 0), ("r3", False, False, 0)]
            with open(suite / "task_trials.jsonl", "w") as f:
                for rid, ok, ok_lim, retries in recs:
                    f.write(json.dumps({"paired_baseline_run": rid, "scene": 1 if rid != "r3" else 2,
                                        "task_success": ok, "task_success_at_baseline_limit": ok_lim,
                                        "num_retries": retries, "stop_reason": "plan_executed",
                                        "first_checkpoint_verdict": "empty" if retries else "held",
                                        "checkpoints": [], "events": []}) + "\n")
            (suite / "skipped.jsonl").write_text(json.dumps({"run_id": "r4"}) + "\n")
            s = summarize(paired_rows(suite, base))
            o = s["overall"]
            self.assertEqual((o["pairs"], o["baseline_success"], o["recovery_success"], o["rescued"], o["harmed"]),
                             (4, 2, 2, 1, 1))
            self.assertEqual(s["overall_at_baseline_limit"]["rescued"], 0)  # the rescue took longer than 90 s
            self.assertEqual(s["recovered"], 1)
            self.assertEqual(s["skipped_baseline_no_plan"], 1)
            self.assertEqual(s["replay_fidelity"], {"trials_without_retry": 2, "same_outcome_as_baseline": 1})


def synthetic_trace(held: bool, n=120):
    """Close at sample 20, slow vertical lift; the object follows after 2 cm of lag if held."""
    t = np.arange(n) / 15
    cmd = np.r_[np.zeros(20), np.ones(n - 20)]
    hand = np.tile([0.5, 0.0, 0.25], (n, 1)).astype(float)
    hand[40:, 2] += np.minimum(np.arange(n - 40) * 0.002, 0.12)
    obj = np.tile([0.5, 0.0, 0.08], (n, 1)).astype(float)
    if held:
        obj[:, 2] += np.maximum(0.0, hand[:, 2] - 0.25 - 0.02)
    bowl = np.tile([0.7, 0.2, 0.03], (n, 1))
    return {"t": t, "gripper_cmd": cmd, "gripper_pos": hand, "objects_center": np.stack([bowl, obj], axis=1),
            "object_names": np.asarray(["bowl", "cube"])}


class CalibrationTest(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(grasp_windows({"gripper_cmd": [0, 1, 1, 0, 0, 1, 0]}), [(1, 3), (5, 6)])

    def test_held_and_missed_grasps_are_labelled_and_judged(self):
        crit = TaskCheckCriteria()
        for held, verdict in ((True, "held"), (False, "empty")):
            tr = synthetic_trace(held)
            r = analyse_grasp(tr, ["bowl", "cube"], 20, len(tr["t"]), "cube", crit, 1 / 15, 0.03)
            self.assertEqual(r["truly_held"], held)
            self.assertEqual(r["monitor_verdict"], verdict, r["monitor_reason"])
            if held:
                self.assertAlmostEqual(r["obj_rise_at_hand_8cm"], 0.06, places=2)


if __name__ == "__main__":
    unittest.main()
