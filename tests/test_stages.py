"""CPU tests for the staged GPU checks: the pass/fail rules, the arm test plan, and the stage
runner with the simulator replaced by fake worker processes."""

import contextlib
import io
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grasp_eval import stages
from grasp_eval.preflight import FAIL, PASS, WARN, Report
from grasp_eval.stage_checks import (
    LIMIT_PROBES, PLANNER_LIMITS_FILE, Q_HOME, SIM_LIMITS_FILE, arm_problems, arm_test_plan, driver_verdict,
    frame_problems, load_limits, missing_graphics_libs, parse_nvidia_smi_csv, scene_problems, vulkan_verdict,
)
from grasp_eval.stage_worker import end_mean, frame_stats

ROOT = Path(__file__).resolve().parents[1]
INSPECTIONS = sorted((ROOT / "results" / "preflight_inspect_20261105T050322Z" / "runs").glob("*/scene_inspection.json"))
SIM = load_limits(SIM_LIMITS_FILE)
FR3 = load_limits(PLANNER_LIMITS_FILE)


class PodRulesTest(unittest.TestCase):
    def test_driver(self):
        self.assertEqual(driver_verdict("595.91.07")[0], FAIL)  # the Nov-4 crash
        self.assertEqual(driver_verdict("590.48.01")[0], PASS)  # what the baseline ran on
        self.assertEqual(driver_verdict("580.95.05")[0], PASS)
        self.assertEqual(driver_verdict("600.10")[0], WARN)  # untested
        self.assertEqual(driver_verdict("")[0], FAIL)

    def test_nvidia_smi_parse(self):
        gpus = parse_nvidia_smi_csv("NVIDIA RTX A6000, 595.91.07, 49140 MiB, 8.6\n\n")
        self.assertEqual(gpus, [{"name": "NVIDIA RTX A6000", "driver": "595.91.07", "memory": "49140 MiB",
                                 "compute_cap": 8.6}])

    def test_vulkan(self):
        self.assertEqual(vulkan_verdict("GPU0:\n\tdeviceName = NVIDIA RTX A6000\n")[0], PASS)
        status, why = vulkan_verdict("GPU0:\n\tdeviceName = llvmpipe (LLVM 15.0.7, 256 bits)\n")
        self.assertEqual(status, FAIL)
        self.assertIn("CPU renderer", why)
        self.assertEqual(vulkan_verdict(None)[0], WARN)

    def test_graphics_libs(self):
        text = "\tlibEGL.so.1 (libc6,x86-64) => /lib/libEGL.so.1\n\tlibGL.so.1 (libc6,x86-64) => /lib/libGL.so.1\n"
        self.assertEqual(missing_graphics_libs(text), {"libXt.so.6": "libxt6", "libGLU.so.1": "libglu1-mesa"})


class FrameAndSceneRulesTest(unittest.TestCase):
    def test_blank_frame_fails_textured_frame_passes(self):
        self.assertTrue(frame_problems(frame_stats(np.zeros((180, 320, 3), np.uint8))))
        img = np.random.default_rng(0).integers(0, 255, (180, 320, 3), dtype=np.uint8)
        self.assertEqual(frame_problems(frame_stats(img)), [])
        self.assertEqual(frame_problems({}), ["no frame"])

    def test_recorded_variant0_inspections_pass(self):
        self.assertEqual(len(INSPECTIONS), 5)
        for path in INSPECTIONS:
            fails, warns = scene_problems(json.loads(path.read_text()))
            self.assertEqual((fails, warns), ([], []), path.parent.name)

    def test_cube_below_table_fails(self):
        info = json.loads(INSPECTIONS[3].read_text())  # scene 4
        info["geometry_centers_after_settle"]["rubiks_cube"] = [0.33, -0.18, -218.0]  # as the Nov-5 paper run saw
        fails, _ = scene_problems(info)
        self.assertEqual(len(fails), 1)
        self.assertTrue(any("below the table" in f for f in fails))

    def test_unsettled_and_changed_objects_warn(self):
        info = json.loads(INSPECTIONS[0].read_text())
        info["settle_motion_m"] = {"rubiks_cube": 0.03, "_24_bowl": 0.0}
        fails, warns = scene_problems(info, reference_objects=["_24_bowl", "rubiks_cube", "_25_mug"])
        self.assertEqual(fails, [])
        self.assertEqual(len(warns), 2)


class ArmRulesTest(unittest.TestCase):
    def test_plan_stays_inside_limits_except_the_probes(self):
        plan = arm_test_plan(SIM)
        self.assertEqual(plan[0]["label"], "home")
        self.assertEqual(plan[0]["q"], Q_HOME)
        kinds = [s["kind"] for s in plan]
        self.assertIn("gripper_closed", kinds)
        self.assertIn("gripper_open", kinds)
        for s in plan:
            for j, v in enumerate(s["q"]):
                lo, hi = SIM[j]
                if s["kind"] == "limit_probe" and j == s["joint"]:
                    self.assertTrue(v > hi, s["label"])  # past the simulator's limit...
                    self.assertTrue(v <= FR3[j][1], s["label"])  # ...but a command the FR3 planner may produce
                else:
                    self.assertTrue(lo <= v <= hi, (s["label"], j, v))
        self.assertEqual({s["joint"] for s in plan if s["kind"] == "limit_probe"}, set(LIMIT_PROBES))
        self.assertFalse(any(s["kind"] == "limit_probe" for s in arm_test_plan(SIM, limit_probes={})))

    def test_planner_limits_file(self):
        self.assertAlmostEqual(FR3[5][1], 4.5169, places=4)
        self.assertAlmostEqual(SIM[5][1], 3.7525, places=4)

    def summary(self, **over):
        """A perfect run of the plan: every move reached, probes stop at the simulator's limits."""
        steps = []
        for s in arm_test_plan(SIM):
            q = list(s["q"])
            if s["kind"] == "limit_probe":
                q[s["joint"]] = SIM[s["joint"]][1]
            closed = s["kind"] == "gripper_closed"
            steps.append({"label": s["label"], "kind": s["kind"], "joint": s.get("joint"), "q_target": s["q"],
                          "q_meas_end": q, "gripper_obs_end": 0.8 if closed else 0.0,
                          "finger_closure_end": 0.95 if closed else 0.0})
        out = {"steps": steps, "sim_joint_limits_live": [list(x) for x in SIM], "object_motion_m": {"cube": 0.0},
               "limit_violations": []}
        out.update(over)
        return out

    def test_perfect_run_passes_and_explains_the_mismatch(self):
        v = arm_problems(self.summary(), SIM, FR3)
        self.assertEqual((v["fails"], v["warns"]), ([], []))
        text = " ".join(v["notes"])
        self.assertIn("joint 6 commanded 4.219 rad stopped at 3.7525", text)
        self.assertIn("FR3 model allows 4.5169", text)

    def test_tracking_error_fails(self):
        s = self.summary()
        s["steps"][3]["q_meas_end"] = [v + (0.05 if k == 2 else 0) for k, v in enumerate(s["steps"][3]["q_meas_end"])]
        v = arm_problems(s, SIM, FR3)
        self.assertEqual(len(v["fails"]), 1)
        self.assertIn("joint 3 ended 0.050 rad", v["fails"][0])

    def test_gripper_that_does_not_move_fails(self):
        s = self.summary()
        for st in s["steps"]:
            st["gripper_obs_end"] = 0.0
        self.assertTrue(any("gripper did not move" in f for f in arm_problems(s, SIM, FR3)["fails"]))

    def test_unenforced_limit_and_moved_object_warn(self):
        s = self.summary(object_motion_m={"cube": 0.04})
        probe = next(st for st in s["steps"] if st["kind"] == "limit_probe" and st["joint"] == 5)
        probe["q_meas_end"] = list(probe["q_target"])  # reached 4.219: no limit at 3.7525
        warns = arm_problems(s, SIM, FR3)["warns"]
        self.assertTrue(any("went past the simulator limit" in w for w in warns))
        self.assertTrue(any("cube moved 4.0 cm" in w for w in warns))

    def test_live_limits_differ_warns(self):
        live = [list(x) for x in SIM]
        live[5][1] = 4.5169
        warns = arm_problems(self.summary(sim_joint_limits_live=live), SIM, FR3)["warns"]
        self.assertTrue(any("joint 6 live limits" in w for w in warns))

    def test_end_mean(self):
        self.assertEqual(end_mean([[0, 0], [1, 1], [3, 3]], n=2), [2.0, 2.0])
        self.assertIsNone(end_mean([]))


def args(tmp, **kw):
    base = dict(tiptop_dir=Path(tmp), timeout_s=10, scene=1, variant=0, no_limit_probe=False, scenes=[1, 4],
                variants=[0, 1], first_variants=5, all_variants=False, resume=False,
                scenes_config=str(ROOT / "configs" / "scenes.json"),
                gripper_bodies=["left_inner_finger", "right_inner_finger"])
    base.update(kw)
    return SimpleNamespace(**base)


class FakeWorker:
    """Stands in for `uv run python stage_worker.py <mode>`: writes <mode>_result.json and returns a code."""

    def __init__(self, results: dict, codes: dict | None = None):
        self.results, self.codes, self.calls, self.envs = results, codes or {}, [], []

    def __call__(self, cmd, cwd, stdout, stderr, timeout, env):
        mode = cmd[cmd.index(str(stages.WORKER)) + 1]
        out = Path(cmd[cmd.index("--out-dir") + 1])
        key = mode
        if "--scene" in cmd:
            key = f"{mode}:s{cmd[cmd.index('--scene') + 1]}v{cmd[cmd.index('--variant') + 1]}"
        self.calls.append(key)
        self.envs.append(env)
        res = self.results.get(key, self.results.get(mode))
        if res is not None:
            (out / f"{mode}_result.json").write_text(json.dumps(res))
        return SimpleNamespace(returncode=self.codes.get(key, self.codes.get(mode, 0)))


class StageRunnerTest(unittest.TestCase):
    def setUp(self):
        self._out = contextlib.redirect_stdout(io.StringIO())
        self._out.__enter__()

    def tearDown(self):
        self._out.__exit__(None, None, None)

    def test_r595_crash_is_reported_with_the_driver_hint(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "droid-sim-evals").mkdir()
            fake = FakeWorker({"launch": {"timings_s": {"before_app_start": 9.0}}}, codes={"launch": 139})
            rep = Report()
            stages.stage_sim(args(d), rep, Path(d) / "out", sh=lambda *a, **k: (0, "CUDA_MATMUL_PASSED x"), run=fake)
            self.assertEqual(fake.calls, ["launch"])  # no render attempt after a failed launch
            launch = next(c for c in rep.checks if c["check"] == "launch")
            self.assertEqual(launch["status"], FAIL)
            self.assertIn("driver 590 or older", launch["detail"])
            self.assertIn("before_app_start", launch["detail"])
            self.assertEqual(fake.envs[0]["UV_NO_SYNC"], "1")

    def test_sim_success_and_blank_camera(self):
        good = frame_stats(np.random.default_rng(1).integers(0, 255, (180, 320, 3), dtype=np.uint8))
        blank = frame_stats(np.zeros((180, 320, 3), np.uint8))
        render = {"ok": True, "timings_s": {"env_created": 40, "settled": 55}, "object_names": ["cube"],
                  "frames": {"external_cam": good, "wrist_cam": blank}}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "droid-sim-evals").mkdir()
            fake = FakeWorker({"launch": {"app_started": True, "timings_s": {"app_started": 30}},
                               "render:s1v0": render})
            rep = Report()
            stages.stage_sim(args(d), rep, Path(d) / "out", sh=lambda *a, **k: (0, "CUDA_MATMUL_PASSED x"), run=fake)
            status = {c["check"]: c["status"] for c in rep.checks}
            self.assertEqual(status, {"cuda": PASS, "launch": PASS, "env": PASS, "camera:external_cam": PASS,
                                      "camera:wrist_cam": FAIL})

    def test_scenes_flags_the_broken_variant(self):
        s1 = json.loads(INSPECTIONS[0].read_text())
        s4 = json.loads(INSPECTIONS[3].read_text())
        s4_bad = json.loads(json.dumps(s4))
        s4_bad["geometry_centers_after_settle"]["rubiks_cube"][2] = -218.0
        results = {f"scene:s{s}v{v}": dict(info, ok=True, object_names=sorted(info["rigid_objects_after_settle"]))
                   for (s, v), info in {(1, 0): s1, (1, 1): s1, (4, 0): s4, (4, 1): s4_bad}.items()}
        with tempfile.TemporaryDirectory() as d:
            fake = FakeWorker(results)
            rep = Report()
            extra = stages.stage_scenes(args(d), rep, Path(d) / "out", run=fake)
            self.assertEqual(fake.calls, ["scene:s1v0", "scene:s1v1", "scene:s4v0", "scene:s4v1"])
            self.assertEqual(extra["do_not_use"], ["s4v1"])
            self.assertEqual([c["status"] for c in rep.checks], [PASS, PASS, PASS, FAIL])
            # --resume keeps what is already there and reruns nothing
            fake2 = FakeWorker({})
            rep2 = Report()
            stages.stage_scenes(args(d, resume=True), rep2, Path(d) / "out", run=fake2)
            self.assertEqual(fake2.calls, [])
            self.assertEqual([c["status"] for c in rep2.checks], [PASS, PASS, PASS, FAIL])

    def test_default_variants_cover_pilot_and_paper(self):
        cfg = json.loads((ROOT / "configs" / "scenes.json").read_text())
        todo = stages.scene_variants(cfg, [1, 2, 3, 4, 5], None, 5)
        self.assertEqual(len(todo), 25)
        self.assertIn((1, 4), todo)  # the pilot's last variant
        # 51 scene files; scene 4 variant 9 is excluded in the config (cube ejected on load, Nov 6)
        self.assertEqual(len(stages.scene_variants(cfg, [1, 2, 3, 4, 5], None, None)), 50)
        self.assertIn("9", cfg["4"]["excluded_variants"])

    def test_arm_stage_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            res = ArmRulesTest.summary(ArmRulesTest())
            res.update(ok=True, plan=arm_test_plan(SIM), sim_duration_s=120.0, video="arm_check.mp4")
            rep = Report()
            stages.stage_arm(args(d), rep, Path(d) / "out", run=FakeWorker({"arm:s1v0": res}))
            self.assertFalse(rep.failed)
            self.assertTrue(any("3.7525" in c["detail"] for c in rep.checks))

    def test_main_writes_reports(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pod"
            with unittest.mock.patch.object(stages, "stage_pod", lambda a, rep, out: rep.add("x", PASS, "ok")):
                rc = stages.main(["pod", "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn("| x | PASS | ok |", (out / "report.md").read_text())
            self.assertEqual(json.loads((out / "report.json").read_text())["stage"], "pod")

if __name__ == "__main__":
    unittest.main()


class PandaPlannerModelTest(unittest.TestCase):
    """The planner should plan for the arm the simulator has (patches/tiptop/0002)."""

    def test_planner_robot_type_is_read_from_tiptop_yml(self):
        from grasp_eval.preflight import PLANNER_ROBOT, planner_robot_type

        panda = (ROOT / "configs" / "tiptop_panda_sim.yml").read_text()
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "tiptop" / "config" / "tiptop.yml"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(panda)
            self.assertEqual(planner_robot_type(Path(d)), PLANNER_ROBOT)
            cfg.write_text(panda.replace("type: panda_robotiq", "type: fr3_robotiq"))
            self.assertEqual(planner_robot_type(Path(d)), "fr3_robotiq")
            self.assertIsNone(planner_robot_type(Path(d) / "missing"))

    def test_patch_changes_only_the_robot_type(self):
        text = (ROOT / "patches" / "tiptop" / "0002-panda-robot-model.patch").read_text()
        changed = [ln for ln in text.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        self.assertEqual(changed, ["-  type: fr3_robotiq", "+  type: panda_robotiq"])
        self.assertIn("+++ b/tiptop/config/tiptop.yml", text)
        # Its context lines are the upstream file as copied into configs/tiptop_panda_sim.yml.
        panda = (ROOT / "configs" / "tiptop_panda_sim.yml").read_text().splitlines()
        context = [ln[1:] for ln in text.splitlines() if ln.startswith(" ")]
        start = panda.index(context[0], 1)
        self.assertEqual(panda[start:start + 3], context[:3])


class FakeCuda:
    """Stands in for libcuda: each call returns a CUDA result code; outputs are written through byref."""

    def __init__(self, fail=None, count=1):
        self.fail, self.count, self.calls = fail or {}, count, []

    def _call(self, name, *args):
        self.calls.append(name)
        return self.fail.get(name, 0)

    def cuInit(self, flags):
        return self._call("cuInit")

    def cuDeviceGetCount(self, ref):
        ref._obj.value = self.count
        return self._call("cuDeviceGetCount")

    def cuDeviceGet(self, ref, i):
        return self._call("cuDeviceGet")

    def cuCtxCreate_v2(self, ref, flags, dev):
        if not self.fail.get("cuCtxCreate_v2"):
            ref._obj.value = 1234
        return self._call("cuCtxCreate_v2")

    def cuMemAlloc_v2(self, ref, n):
        return self._call("cuMemAlloc_v2")

    def cuMemFree_v2(self, ptr):
        return self._call("cuMemFree_v2")

    def cuCtxDestroy_v2(self, ctx):
        return self._call("cuCtxDestroy_v2")


class Stage0CudaAndEnvTest(unittest.TestCase):
    def test_healthy_gpu_passes(self):
        from grasp_eval.stage_checks import cuda_driver_probe

        lib = FakeCuda(count=2)
        status, why = cuda_driver_probe(lib)
        self.assertEqual(status, PASS)
        self.assertIn("2 GPU(s)", why)
        self.assertEqual(lib.calls.count("cuCtxDestroy_v2"), 2)

    def test_nov6_broken_host_fails(self):
        """RTX 5000 Ada, driver 590: nvidia-smi fine, CUDA 'busy or unavailable' (error 46)."""
        from grasp_eval.stage_checks import cuda_driver_probe

        status, why = cuda_driver_probe(FakeCuda(fail={"cuCtxCreate_v2": 46}))
        self.assertEqual(status, FAIL)
        self.assertIn("cuCtxCreate failed with 46 CUDA_ERROR_DEVICE_UNAVAILABLE", why)
        self.assertEqual(cuda_driver_probe(FakeCuda(fail={"cuInit": 100}))[0], FAIL)
        self.assertEqual(cuda_driver_probe(FakeCuda(count=0))[0], FAIL)
        lib = FakeCuda(fail={"cuMemAlloc_v2": 2})
        self.assertEqual(cuda_driver_probe(lib)[0], FAIL)
        self.assertIn("cuCtxDestroy_v2", lib.calls)  # context released on failure

    def test_container_env_falls_back_to_pid1(self):
        from grasp_eval.stage_checks import container_env, driver_capabilities_verdict

        with tempfile.TemporaryDirectory() as d:
            proc = Path(d) / "environ"
            proc.write_bytes(b"PATH=/usr/bin\0NVIDIA_DRIVER_CAPABILITIES=compute,display,graphics,utility,video\0")
            caps = container_env("NVIDIA_DRIVER_CAPABILITIES", environ={}, proc_environ=str(proc))
            self.assertEqual(caps, "compute,display,graphics,utility,video")
            self.assertEqual(container_env("X", environ={"X": "1"}, proc_environ=str(proc)), "1")
            self.assertIsNone(container_env("MISSING", environ={}, proc_environ=str(proc)))
        self.assertEqual(driver_capabilities_verdict(caps)[0], PASS)
        self.assertEqual(driver_capabilities_verdict("all")[0], PASS)
        self.assertEqual(driver_capabilities_verdict("compute,utility")[0], FAIL)
        self.assertEqual(driver_capabilities_verdict(None)[0], WARN)

    def test_stage_pod_reports_the_cuda_failure(self):
        smi = "NVIDIA RTX 5000 Ada Generation, 590.48.01, 30712 MiB, 8.9"

        def sh(cmd, cwd=None, timeout=60):
            return (0, smi) if cmd[0] == "nvidia-smi" else (0, "libEGL.so.1 libGL.so.1 libXt.so.6 libGLU.so.1")

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            rep = Report()
            stages.stage_pod(SimpleNamespace(disk_path=d), rep, Path(d), sh=sh,
                             cuda_probe=lambda: (FAIL, "GPU 0: cuCtxCreate failed with 46"),
                             env=lambda v: "all" if v == "NVIDIA_DRIVER_CAPABILITIES" else None)
        status = {c["check"]: c["status"] for c in rep.checks}
        self.assertEqual(status["driver:gpu0"], PASS)
        self.assertEqual(status["cuda"], FAIL)
        self.assertEqual(status["env:NVIDIA_DRIVER_CAPABILITIES"], PASS)


class ArmVideoWarningTest(unittest.TestCase):
    def test_missing_video_warns_with_the_reason(self):
        """Nov 6: no ffmpeg on the pod, so the arm check passed but silently saved no video."""
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            res = ArmRulesTest.summary(ArmRulesTest())
            res.update(ok=True, plan=arm_test_plan(SIM), sim_duration_s=108.9,
                       video_error="RuntimeError: Program 'ffmpeg' is not found")
            rep = Report()
            stages.stage_arm(args(d), rep, Path(d) / "out", run=FakeWorker({"arm:s1v0": res}))
        self.assertFalse(rep.failed)
        video = [c for c in rep.checks if c["check"] == "arm:video"]
        self.assertEqual(len(video), 1)
        self.assertEqual(video[0]["status"], WARN)
        self.assertIn("ffmpeg", video[0]["detail"])


class SceneAssetFixTest(unittest.TestCase):
    def test_only_the_released_bad_offset_is_changed(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fix_scene_assets", ROOT / "scripts" / "fix_scene_assets.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.plan_fix(-0.19863615566511325), "fix")
        self.assertEqual(mod.plan_fix(0.20057768036734341), "ok")
        self.assertEqual(mod.plan_fix(0.0), "unknown")
        self.assertEqual(mod.FIXES, {"scene4_1.usd", "scene4_2.usd", "scene4_3.usd"})
