"""The container-frame 'in' check on real scene-3 baseline end states (labels from the end frames)."""

import json
import unittest
from pathlib import Path

from grasp_eval.task_goal import check_goal

CASES = json.loads((Path(__file__).parent / "data" / "scene3_goal_cases.json").read_text())
GOAL = [{"objects": ["$target"], "relation": "in", "container": "small_KLT_visual_collision", "frame": "container"}]
OLD_GOAL = [{k: v for k, v in GOAL[0].items() if k != "frame"}]


class ContainerFrameTest(unittest.TestCase):
    def test_matches_the_end_frames(self):
        for rid, case in CASES.items():
            with self.subTest(run=rid):
                self.assertEqual(check_goal(GOAL, case, True)["task_success"], case["visual_in_bin"])

    def test_the_old_world_box_check_got_these_wrong(self):
        # 021/023: tilted banana inside the bin rejected; 020/024: banana beside the turned bin accepted.
        for rid in ("021_s3v1_scene3_t0", "023_s3v3_scene3_t0", "024_s3v4_scene3_t0", "020_s3v0_scene3_t0"):
            with self.subTest(run=rid):
                self.assertNotEqual(check_goal(OLD_GOAL, CASES[rid], True)["task_success"], CASES[rid]["visual_in_bin"])

    def test_closed_gripper_and_missing_pose(self):
        case = CASES["021_s3v1_scene3_t0"]
        self.assertFalse(check_goal(GOAL, case, False)["task_success"])
        broken = dict(case, final_object_quats={})
        self.assertIsNone(check_goal(GOAL, broken, True)["task_success"])

    def test_container_frame_rejects_on(self):
        goal = [dict(GOAL[0], relation="on")]
        self.assertIsNone(check_goal(goal, CASES["021_s3v1_scene3_t0"], True)["task_success"])


if __name__ == "__main__":
    unittest.main()
