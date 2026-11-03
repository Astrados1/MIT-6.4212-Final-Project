import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from grasp_eval import run_suite
from grasp_eval.preflight import goal_problems, inspection_problems
from grasp_eval.summarize import build_rows, merge_shards, wilson_ci


def _args(**kw):
    a = SimpleNamespace(scene=1, variants=[0, 1, 2, 3], trials=2, instruction="x", target_object="cube", seed_base=0)
    a.__dict__.update(kw)
    return a


class ShardTest(unittest.TestCase):
    def test_shards_partition_runs_and_keep_replays_with_source(self):
        a = _args(variants=[0, 1, 2], distances_cm=[2], directions=["x+"], yaws_deg=[], repeats=1)
        runs = run_suite.build_displacement(a)
        run_suite.assign_ids(runs)
        s0, s1 = run_suite.shard_runs(runs, (0, 2)), run_suite.shard_runs(runs, (1, 2))
        self.assertEqual(sorted(r.run_id for r in s0 + s1), sorted(r.run_id for r in runs))
        for shard in (s0, s1):
            ids = {r.run_id for r in shard}
            self.assertTrue(all(r.plan_from.run_id in ids for r in shard if r.plan_from))


class ResumeTest(unittest.TestCase):
    def test_resume_runs_only_unfinished(self):
        calls = []

        def fake(cmd, cwd, log, timeout):
            if "observed_eval.py" in " ".join(cmd):
                calls.append(cmd[cmd.index("--run-id") + 1])
            return 0

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(run_suite, "PROJECT_ROOT", Path(d)), \
                mock.patch.object(run_suite, "run_process", fake), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            args = ["pilot", "--tiptop-dir", d, "--target-object", "cube", "--variants", "0", "1", "--trials", "1",
                    "--no-health-check", "--skip-attribution", "--out-name", "t"]
            out = Path(d) / "results" / "t"
            out.mkdir(parents=True)
            # First session: both runs scheduled, only the first finished before the machine went away.
            (out / "schedule.jsonl").write_text("".join(json.dumps({"run_id": i, "variant": v}) + "\n"
                                                        for i, v in (("000_s1v0_baseline_t0", 0),
                                                                     ("001_s1v1_baseline_t0", 1))))
            (out / "exit_codes.tsv").write_text("000_s1v0_baseline_t0\t0\n")
            self.assertEqual(run_suite.main(args), 1)  # refuses without --resume
            run_suite.main(args + ["--resume"])
            schedule = (out / "schedule.jsonl").read_text().splitlines()
        self.assertEqual(calls, ["001_s1v1_baseline_t0"])
        self.assertEqual(len(schedule), 2)  # the interrupted run was not scheduled twice


class SummaryScaleTest(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_ci(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(wilson_ci(0, 10)[0], 0.0)
        self.assertIsNone(wilson_ci(0, 0))

    def test_merge_shards(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for k, ids in ((0, ["a", "b"]), (1, ["c"])):
                s = d / f"s{k}"
                s.mkdir()
                (s / "schedule.jsonl").write_text("".join(json.dumps({"run_id": i, "variant": 0}) + "\n" for i in ids))
            merge_shards([d / "s0", d / "s1"], d / "all")
            rows, _ = build_rows(d / "all")
            self.assertEqual([r["run_id"] for r in rows], ["a", "b", "c"])
            (d / "s1" / "schedule.jsonl").write_text(json.dumps({"run_id": "a", "variant": 0}) + "\n")
            with self.assertRaises(ValueError):
                merge_shards([d / "s0", d / "s1"], d / "all")


class PreflightTest(unittest.TestCase):
    def test_goal_problems(self):
        cfg = {"1": {"goal": [{"objects": ["$target"], "relation": "in", "container": None}],
                     "object_aliases": "configs/object_aliases.json"},
               "4": {"goal": [{"objects": ["cube"], "relation": "on", "container": "mug"}]}}
        self.assertEqual(goal_problems(cfg, [1, 4]), ["scene 1 goal[0]: container not set"])

    def test_inspection_problems(self):
        info = {"robot_body_names": ["left_inner_finger", "right_inner_finger"],
                "rigid_objects_after_settle": {"cube": [0, 0, 0], "bowl": [0, 0, 0]},
                "rigid_object_local_bounds": {"cube": [[-0.03] * 3, [0.03] * 3], "bowl": [[-5, -5, 0], [5, 5, 1]]}}
        cfg = {"goal": [{"objects": ["$target"], "relation": "in", "container": "big_bowl"}]}
        problems = inspection_problems(info, cfg, ["left_inner_finger", "right_inner_finger"])
        self.assertEqual(len(problems), 2)
        self.assertIn("implausible box for bowl", problems[0])
        self.assertIn("'big_bowl' is not a simulator object", problems[1])


if __name__ == "__main__":
    unittest.main()
