import json
import tempfile
import unittest
from pathlib import Path

from grasp_eval.records import RunRecord, auto_label
from grasp_eval.summarize import build_rows, render


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _record(run_id, variant, **kw):
    rec = RunRecord(run_id=run_id, scene=1, variant=variant, trial=0, instruction="x", target_object="cube")
    for k, v in kw.items():
        setattr(rec, k, v)
    rec.failure_category, rec.failure_detail = auto_label(rec)
    return rec.to_dict()


class AutoLabelTest(unittest.TestCase):
    def test_labels(self):
        def label(**kw):
            rec = RunRecord(run_id="r", scene=1, variant=0, trial=0, instruction="x", target_object="c")
            for k, v in kw.items():
                setattr(rec, k, v)
            return auto_label(rec)[0]

        self.assertEqual(label(harness_exception="boom"), "infrastructure")
        self.assertEqual(label(), "infrastructure")
        self.assertEqual(label(server_response_received=True, server_success=False,
                               server_error="cuTAMP failed to find a plan: x"), "planning")
        self.assertEqual(label(server_response_received=True, server_success=False, server_error="gemini"), "unknown")
        ok = dict(server_response_received=True, server_success=True, executed_steps=10)
        self.assertIsNone(label(**ok, stable_lift=True))
        self.assertEqual(label(**ok, stable_lift=False, truncated=True, score={"num_close_commands": 0}), "timeout")
        self.assertEqual(label(**ok, stable_lift=False,
                               score={"num_close_commands": 1, "failure_hint": "missed_grasp"}), "missed_grasp")


class SummarizeTest(unittest.TestCase):
    def test_denominators_and_overrides(self):
        ok = dict(server_response_received=True, server_success=True, executed_steps=100)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_jsonl(d / "schedule.jsonl", [{"run_id": f"r{i}", "variant": i % 2} for i in range(5)])
            _write_jsonl(d / "runs.jsonl", [
                _record("r0", 0, **ok, stable_lift=True, planning_time_s=30.0),
                _record("r1", 1, **ok, stable_lift=False,
                        score={"num_close_commands": 1, "failure_hint": "missed_grasp"}),
                _record("r2", 0, server_response_received=True, server_success=False,
                        server_error="cuTAMP failed to find a plan"),
                _record("r3", 1, **ok, stable_lift=True),
                # r4 never wrote a record (crash)
            ])
            (d / "manual_labels.csv").write_text(
                "run_id,stable_lift,failure_category,task_success,note\n"
                "r3,false,execution_collision,,pushed the cube up against the bowl\n"
                "r0,,,true,\n"
            )
            rows, warnings = build_rows(d)
            by_id = {r["run_id"]: r for r in rows}
            self.assertEqual(len(rows), 5)
            self.assertEqual(sum(r["stable_lift"] for r in rows), 1)
            self.assertEqual(sum(r["valid_executed"] for r in rows), 3)
            self.assertEqual(by_id["r4"]["failure_category"], "infrastructure")
            self.assertEqual(by_id["r2"]["failure_category"], "planning")
            self.assertEqual(by_id["r3"]["failure_category"], "execution_collision")
            self.assertTrue(by_id["r0"]["task_success"])
            text = render(rows, warnings, "t")
            self.assertIn("overall (per scheduled trial): 1/5 (20%)", text)
            self.assertIn("conditional (per valid executed plan): 1/3 (33%)", text)
            self.assertEqual(warnings, [])

    def test_bad_manual_label_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_jsonl(d / "schedule.jsonl", [{"run_id": "r0", "variant": 0}])
            (d / "manual_labels.csv").write_text("run_id,stable_lift,failure_category,task_success,note\nr0,,oops,,\n")
            with self.assertRaises(ValueError):
                build_rows(d)


if __name__ == "__main__":
    unittest.main()
