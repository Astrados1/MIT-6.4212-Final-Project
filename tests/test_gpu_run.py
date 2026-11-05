import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from grasp_eval import run_suite
from grasp_eval.preflight import _sh, goal_problems, inspection_problems
from grasp_eval.summarize import build_rows, merge_shards, wilson_ci


def _args(**kw):
    a = SimpleNamespace(scene=1, variants=[0, 1, 2, 3], trials=2, instruction="x", target_object="cube", seed_base=0)
    a.__dict__.update(kw)
    return a


class NativeCrashOutputTest(unittest.TestCase):
    def test_invalid_utf8_preserves_exit_code_and_diagnostic(self):
        code, output = _sh([
            sys.executable, "-c",
            "import os; os.write(2, b'RTX crash: \\xff\\n'); raise SystemExit(139)",
        ])
        self.assertEqual(code, 139)
        self.assertIn("RTX crash:", output)
        self.assertIn("\ufffd", output)


class RemotePlannerTest(unittest.TestCase):
    def test_rejects_artifacts_outside_planner_output_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, tiptop = root / "results", root / "tiptop"
            rd = out / "runs" / "r1"
            rd.mkdir(parents=True)
            (rd / "record.json").write_text(json.dumps({"server_save_dir": str(root / "other")}))
            args = SimpleNamespace(planner_ssh_host="planner", planner_ssh_config=root / "ssh_config")
            with mock.patch.object(run_suite, "run_process") as run, self.assertRaises(ValueError):
                run_suite.attribute(SimpleNamespace(run_id="r1"), out, root / "sim", tiptop, args)
            run.assert_not_called()

    def test_extracts_on_planner_and_fetches_before_attribution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, tiptop = root / "results", root / "tiptop"
            rd = out / "runs" / "r1"
            rd.mkdir(parents=True)
            save_dir = tiptop / "tiptop_server_outputs" / "20261105_000000"
            (rd / "record.json").write_text(json.dumps({"server_save_dir": str(save_dir)}))
            args = SimpleNamespace(planner_ssh_host="planner", planner_ssh_config=root / "ssh_config")
            commands = []

            def run(cmd, cwd, log, timeout):
                commands.append(cmd)
                if cmd[0] == "scp":
                    Path(cmd[-1]).write_text('{}')
                return 0

            with mock.patch.object(run_suite, "run_process", run):
                run_suite.attribute(SimpleNamespace(run_id="r1"), out, root / "sim", tiptop, args)
            self.assertEqual([c[0] for c in commands], ["ssh", "rsync", "scp", "uv"])
            self.assertIn("--perceived", commands[-1])


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


class ClusterPlanTest(unittest.TestCase):
    def test_gpu_roles(self):
        from grasp_eval.preflight import gpu_roles

        self.assertEqual(gpu_roles("NVIDIA RTX A6000", 8.6), {"simulator": None, "m2t2": None, "tiptop": None})
        bw = gpu_roles("NVIDIA RTX PRO 6000 Blackwell Server Edition", 12.0)
        self.assertIsNone(bw["tiptop"])
        self.assertIn("compute capability 12", bw["simulator"])
        self.assertIn("compute capability 12", bw["m2t2"])
        h100 = gpu_roles("NVIDIA H100 80GB HBM3", 9.0)
        self.assertIn("RT cores", h100["simulator"])
        self.assertIsNone(h100["m2t2"])

    def test_os_release(self):
        from grasp_eval.preflight import parse_os_release

        self.assertEqual(parse_os_release('NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="22.04"\n'), ("ubuntu", "22.04"))
        self.assertEqual(parse_os_release('ID="rocky"\nVERSION_ID="9.4"\n'), ("rocky", "9.4"))


class VideoAndReviewTest(unittest.TestCase):
    def test_plan_step_text(self):
        from grasp_eval.video_overlay import caption_line, plan_step_text

        plan = [{"type": "trajectory", "label": "Pick(cube, g, q)"}, {"type": "gripper", "action": "close", "label": "Pick(cube, g, q)"}]
        self.assertEqual(plan_step_text(None, 0, None), "waiting for plan")
        self.assertEqual(plan_step_text(plan, 1, None), "step 1/2: Pick(cube, g, q)")
        self.assertEqual(plan_step_text(plan, 1, "close"), "step 2/2: gripper close (Pick(cube, g, q))")
        self.assertEqual(plan_step_text(plan, 3, None), "plan finished (2 steps)")
        self.assertIn("target +5.0 cm", caption_line("x", 1.0, 0.05, None))

    def test_review_page(self):
        from grasp_eval.review_page import render_review

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            rows = [{"run_id": "r1", "variant": 0, "condition": "baseline", "stable_lift": False, "task_success": False,
                     "task_success_source": "auto", "paper_category": "grasp_failure/heuristic_unstable_grasp",
                     "paper_category_source": "auto", "paper_evidence": ["no stable lift (missed_grasp)", "grasp source: heuristic"]},
                    {"run_id": "r2", "variant": 1, "condition": "baseline", "stable_lift": True, "task_success": True,
                     "task_success_source": "auto", "paper_category": "success", "paper_category_source": "auto",
                     "paper_evidence": []}]
            recs = {"r1": {"scene": 1, "trial": 0, "instruction": "Put <cube> in bowl",
                           "artifacts": {"video": str(base / "runs" / "r1" / "video.mp4")}}}
            attrs = {"r1": {"grasp_source": "heuristic", "gripper_xy_offset_at_close_m": 0.031}}
            page = render_review(base, rows, recs, attrs)
        self.assertIn('src="runs/r1/video.mp4"', page)
        self.assertIn("Heuristic Unstable Grasp", page)
        self.assertIn("3.1 cm", page)
        self.assertIn("Put &lt;cube&gt; in bowl", page)  # escaped
        self.assertLess(page.index('id="r1"'), page.index('id="r2"'))  # failures first
