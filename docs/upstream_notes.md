# Upstream code-reading notes (pinned revisions)

Read on October 29, 2026 at TiPToP `d8f5afd` and droid-sim-evals `69a914c`. Nothing here has been
executed yet. Items marked **CHECK** must be verified on the GPU machine before baseline results are interpreted.

## Robot, gripper and frames

- The simulator loads `assets/franka_robotiq_2f_85_flattened.usd` with joints `panda_joint1..7` plus
  Robotiq `finger_joint` (`src/sim_evals/environments/nvidia_droid.py`). TiPToP's default config selects
  `robot.type: fr3_robotiq` (`tiptop/config/tiptop.yml`).
- The simulator's initial arm pose (0, -π/5, 0, -4π/5, 0, 3π/5, 0) equals TiPToP's `q_home`
  (0, -0.628, 0, -2.513, 0, 1.885, 0) to the printed precision. This is consistent, but it does not prove the kinematic models match.
- **CHECK**: Panda and FR3 differ in joint limits and some link geometry. Compare FK of the cuRobo `fr3_robotiq`
  model against the simulator's link poses at several plan waypoints, and compare the TCP/grasp frame with the
  Robotiq finger geometry. Record the result in `docs/baseline_report.md`. Do not "fix" a mismatch silently.
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
- If the pipeline raises outside its own handler, the server sends a plain-text error and closes the connection.
  The client then fails to parse JSON. This is recorded as a non-JSON server reply.
- `cuTAMP failed to find a plan: ...` is the planning-failure message. Other errors (Gemini, SAM2, M2T2, CUDA)
  come through as exception strings and need the server log to classify.
- The Gemini default model is `gemini-robotics-er-1.6-preview` (`tiptop/perception/gemini.py`).
  The Google key is needed by the server. The sim client imports `google.genai` and loads `.env` but did not
  appear to call Gemini in the code read.

## Installation details

- `pixi run setup-planners` clones cuRobo from `williamshen-nz/curobo` (default branch, unpinned) and cuTAMP
  from `tiptop-robot/cuTAMP` at the tag required in `tiptop/utils.py` (v0.0.6). Both land inside the tiptop
  checkout (`tiptop/curobo`, `tiptop/cutamp`). `record_provenance.sh` records both SHAs.
- The SAM2 checkpoint `sam2.1_hiera_large.pt` is downloaded on first use into TiPToP's cache directory.
- Asset archive URLs disagree: the TiPToP simulation guide uses `pi-sim-assets.s3...`, the simulator README
  uses `tiptop-sim-assets.s3...`. **CHECK** which one is reachable and record its sha256.
- The droid-sim-evals uv environment pins `isaaclab[all,isaacsim]==2.2.0` with CUDA 11.8 torch wheels. TiPToP's
  pixi environment targets CUDA 12. These are separate environments and processes.
- `tiptop_eval.py`'s docstring says `python -m tiptop.websocket_server`; use the pixi task `pixi run tiptop-server`.

## Not yet read

`replay_json_traj.py`, `save_h5_obs.py`, `tiptop/tiptop_offline.py` (episode limits and the execution path
for the offline route), `tiptop/planning.py` (cuTAMP settings, serialization) and the perception wrapper.
Read these before using the offline route or interpreting planner-side failures.
