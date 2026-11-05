# Upstream code-reading notes (pinned revisions)

Initial code reading: October 29, 2026 at TiPToP `d8f5afd` and droid-sim-evals `69a914c`.
The historical notes below are supplemented by the November 5 execution findings at the end.
Items marked **CHECK** require experiment-specific verification.

## Robot, gripper and frames

- The simulator loads `assets/franka_robotiq_2f_85_flattened.usd` with joints `panda_joint1..7` plus
  Robotiq `finger_joint` (`src/sim_evals/environments/nvidia_droid.py`). TiPToP's default config selects
  `robot.type: fr3_robotiq` (`tiptop/config/tiptop.yml`).
- The simulator's initial arm pose (0, -π/5, 0, -4π/5, 0, 3π/5, 0) equals TiPToP's `q_home`
  (0, -0.628, 0, -2.513, 0, 1.885, 0) to the printed precision. This is consistent, but it does not prove the kinematic models match.
- **CHECK**: Panda and FR3 differ in joint limits and some link geometry. Compare FK of the cuRobo `fr3_robotiq`
  model against the simulator's link poses at several plan waypoints, and compare the TCP/grasp frame with the
  Robotiq finger geometry. Record the result in `docs/baseline_report.md`. Do not "fix" a mismatch silently.
  TiPToP also supports `robot.type: panda_robotiq` (`tiptop/motion_planning.py`, backed by cuTAMP's
  `load_panda_robotiq_container`), which matches the simulated arm by name. The simulation guide does not say to
  switch, so the default `fr3_robotiq` is what upstream sim runs use. If `gripper_xy_offset_at_close_m`
  (`docs/probes.md`) is systematically non-zero while perception error is small, test `panda_robotiq` as a
  separate, recorded condition.
- The client shifts the wrist-camera pose down by 1.5 cm before sending it to TiPToP
  (`world_from_cam[:3, 3] -= [0, 0, 0.015]`, `tiptop_websocket.py`), with a comment saying it matches
  TiPToP's grasp frame. Preserve it unless evidence shows a bug.
- Wrist camera pose is sent in the ROS convention (`quat_w_ros`). Depth is rendered `distance_to_image_plane`.
- **CHECK**: robot body names. `observed_eval.py` uses the midpoint of `left_inner_finger` and
  `right_inner_finger` as the gripper reference point. Confirm these names with `--inspect-only`.

## Timing and execution (simulator client)

- Control runs at 15 Hz: `sim.dt = 1/120`, `decimation = 8`. `tiptop_eval.py` overrides the episode
  length from 30 s to 90 s (1350 control steps). The only termination is the time limit; there is no reward.
- Before each episode, `settle_sim` holds the pose for 100 steps and then zeroes the episode counter.
- Trajectories are subsampled by `round(50 / 15) = 3`, always keeping the last waypoint. This assumes
  cuRobo interpolates at 50 Hz. **CHECK** the actual interpolation dt in the returned plan; if it differs,
  the executed motion is faster or slower than planned.
- Only `positions` are executed. The returned `velocities` are unused. The arm tracks joint targets with implicit
  PD control (stiffness 400, damping 80).
- Each gripper step holds the measured joint position for 20 control steps (≈1.33 s). The gripper is binary:
  `finger_joint` goes to π/4 (closed) or 0 (open), with velocity limit 1.0 rad/s.
- After the plan, the client holds the measured joints and gripper. The loop breaks on `client.plan_done`, so no
  frames are recorded after the last step finishes.
- The `gripper_pos` observation has Gaussian noise (σ = 0.05, clipped to [0, 1]). It feeds back into actions
  only during settling and after the plan finishes. Seeding torch makes this noise repeatable.
- The robot asset has `disable_gravity=True`.
- On a planning error, upstream logs it, drops the video and moves on, so no record is kept.
  `grasp_eval/observed_eval.py` keeps a record for every run.

## Server

- One websocket connection per episode. The server serializes pipeline runs with a lock (shared cuRobo state).
  The client reconnects on `reset()` to get a fresh handler.
- `success: true` means only that a plan was produced. The response also carries `save_dir`, the per-run
  output directory (`tiptop_server_outputs/<timestamp>/` with `tiptop_run.log` and `tiptop_plan.json`).
  `observed_eval.py` stores this path, which links each episode to its server artifacts.
- The server also saves `metadata.json` (grounded atoms, perception/planning durations, failure reason),
  `perception/cutamp_env.pkl` (the perceived scene: movable and static objects with geometry and poses),
  `perception/grasps.pt` (M2T2 grasps) and a copy of `tiptop.yml`. It does **not** save the RGB, depth, bounding
  boxes or masks: `recording.save_perception_outputs` exists but the server never calls it. Stage attribution
  therefore reads `cutamp_env.pkl` (`grasp_eval/extract_perceived_scene.py`, tiptop pixi env).
- Plan steps carry a `label` (e.g. `Pick(Rubik's_cube, grasp1, q1)`) and trajectory steps a `dt`, which lets us
  identify the grasped object and check the client's 50 Hz assumption without running anything.
- If the pipeline raises outside its own handler, the server sends a plain-text error and closes the connection.
  The client then fails to parse JSON. This is recorded as a non-JSON server reply.
- `cuTAMP failed to find a plan: ...` is the planning-failure message. Other errors (Gemini, SAM2, M2T2, CUDA)
  come through as exception strings and need the server log to classify.
- The Gemini default model is `gemini-robotics-er-1.6-preview` (`tiptop/perception/gemini.py`), called with
  `thinking_budget=0`. **Google shut that model down on Aug 31, 2026.** Our patch
  `patches/tiptop/0001-gemini-robotics-er-2.patch` switches it to `gemini-robotics-er-2-preview` with
  `thinking_level=MINIMAL` (see `patches/README.md`). No call site passes a model ID explicitly, so the defaults
  are the only place to change. Upstream TiPToP had not updated this as of Oct 30, 2026.
  The Google key is needed by the server. The sim client imports `google.genai` and loads `.env` but did not
  appear to call Gemini in the code read.

## Installation details

- `pixi run setup-planners` clones cuRobo from `williamshen-nz/curobo` (default branch, unpinned) and cuTAMP
  from `tiptop-robot/cuTAMP` at the tag required in `tiptop/utils.py` (v0.0.6). Both land inside the tiptop
  checkout (`tiptop/curobo`, `tiptop/cutamp`). `record_provenance.sh` records both SHAs.
- The SAM2 checkpoint `sam2.1_hiera_large.pt` is downloaded on first use into TiPToP's cache directory.
- Asset archive URLs disagree: the TiPToP simulation guide uses `pi-sim-assets.s3...`, the simulator README
  uses `tiptop-sim-assets.s3...`. **CHECK** which one is reachable and record its sha256.
- The three environments are separate processes with different PyTorch/CUDA builds. This decides which GPUs
  each can run on (preflight's `gpu` check encodes it; see docs/setup.md section G):

  | environment | locked build (from its lockfile at the pins) | newest GPU arch |
  |---|---|---|
  | droid-sim-evals (`uv.lock`) | isaacsim 5.0.0.0, isaaclab 2.2.0, torch 2.7.0 from PyPI with CUDA 12.6 libraries (`nvidia-cuda-runtime-cu12` 12.6.77). `pyproject.toml` declares a CUDA 11.8 index, but the lock resolved torch from PyPI. | sm 9.0 (no Blackwell) |
  | M2T2 (`pixi.lock`, williamshen-nz/M2T2 master) | pytorch 2.4.1 (cuda120 build), cuda-nvcc 12.1; `pointnet2_ops` is compiled at install time for the GPU present (no fixed arch list in its setup.py) | sm 9.0 (no Blackwell) |
  | tiptop (`pixi.lock`) | pytorch 2.7.1 (cuda129 build), cuda-nvcc 12.9; cuRobo compiled at install time | expected to cover Blackwell (sm 12.0) |

  Isaac Sim also needs RT cores (NVIDIA lists A100/H100 as unsupported), so the simulator needs an RTX-class
  card. Build M2T2 and cuRobo on the machine that runs them, so their compiled kernels target that GPU.
- `tiptop_eval.py`'s docstring says `python -m tiptop.websocket_server`; use the pixi task `pixi run tiptop-server`.

## Not yet read

`replay_json_traj.py`, `save_h5_obs.py`, `tiptop/tiptop_offline.py` (episode limits and the execution path
for the offline route), `tiptop/planning.py` (cuTAMP settings, serialization) and the perception wrapper.
Read these before using the offline route or interpreting planner-side failures.

## November 5 execution findings

- All five configured scenes load on the Blackwell/R590 split-host deployment. Both inner-finger
  body names exist. The cube is approximately 7.21 cm across. Scene 4 has two cans and scene 5
  has five blocks; object/container mappings are now populated from inspection.
- The original 10 cm finger-origin proximity threshold rejected a visually successful carry.
  The smoke's held cube was about 15.47 cm from the reference. See the pilot validation and
  explicit criteria in [scoring.md](scoring.md).
- Some scene-4/5 geometry centers are offset from body origins. Scoring now uses geometry-box
  centers while preserving raw body poses. Direct mesh-vertex inspection confirms the offsets.
- Recorded trajectory `dt=0.02`; stride 3 at 15 Hz yields approximately 1.111 times planned
  duration. Original timing and the 1.5 cm camera adjustment remain unchanged.
- FR3 and Panda grasp-frame FK agree at all 467 measured joint configurations in the successful
  smoke (zero translation difference, maximum angular difference below 0.000003 degrees).
  **Their joint limits do not agree.** The planner's FR3 joint 6 upper bound is 4.5169 rad,
  versus 3.7525 rad in both the Panda URDF and the actual simulator USD. Pilot variant 2
  saturated at the Panda bound while the command reached 4.219 rad before release. This caused
  a large pose-tracking error despite a stable grasp. Joint 7 limits also differ. These checks
  do not constitute a complete direct comparison of every USD link's FK.
- Contrary to the initial code-reading note above, actual saved pipeline directories include
  `rgb.png`, `bboxes_viz.png`, `masks_viz.png`, `perception/bboxes.json`, masks, depth, intrinsics,
  point clouds, grasp tensors and the perceived scene, as well as plan/config/provenance files.
  Preserve the complete per-run directory for failure review.
- The server's tracked diff consists of the supplied Gemini patch, including the model default
  in `compute_gripper_mask.py`. Saved per-run `git.diff` and metadata provide direct evidence.
