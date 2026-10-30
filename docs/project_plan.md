# TiPToP grasp project: policy specification and week-one handoff

Prepared October 29, 2026. This is an implementation plan grounded in the repositories and documentation below. The software has not been installed or executed in this chat, and GPU compatibility has not been tested here.

## Project objective

Improve TiPToP's grasp success by replacing the local approach–close–lift execution segment with a learned policy that observes and adjusts during execution. Keep TiPToP's perception, target selection, grasp generation, and task and motion planning as the surrounding system.

The target hardware in the proposal is an IIWA. Start baseline reproduction with the supplied DROID/Franka simulator. IIWA deployment is a separate adaptation involving the robot model, gripper, cameras, controller, and potentially training data. It is not required to finish week one.

The main experiment ends at a verified stable lift. Transport, placement, and estimation of the object-to-gripper attachment after a changed grasp are outside the primary grasp evaluation.

## What the learned grasp policy will be

This is a goal-conditioned visuomotor controller, trained by imitation. It is not a new task planner or an online reinforcement learning system.

Its invocation starts at a verified pregrasp pose near the target, with the gripper empty. Its nominal target comes from TiPToP's selected grasp. Its operating domain initially consists of accessible top-down grasps of a small set of rigid objects in a clear local workspace.

Proposed first implementation:

- Inputs: recent wrist RGB images; measured joint and gripper state; the nominal grasp expressed in a consistent robot-relative frame. Compute any end-effector pose needed for this representation by forward kinematics.
- Model: an existing Diffusion Policy image encoder and conditional temporal 1D U-Net, adapted to our observation/action dimensions. A ResNet-18 image encoder is a reasonable initial choice, not an established optimum.
- Actions: a sequence of four-dimensional commands: small translations in robot-base axes, in meters, plus gripper opening/closing. Keep the planned gripper orientation fixed during this initial skill.
- Control adapter: bounded Cartesian targets are converted to joint-position targets by constrained IK or a suitable Cartesian controller. The existing low-level servo tracks those targets.
- Execution: predict a short sequence, execute a small prefix, read fresh observations, and predict again. Three observation frames, an eight-step prediction, and a two-step execution prefix are initial tunable settings. Measure latency before choosing the deployment update rate.
- Bounds: restrict displacement, velocity, and the local workspace. The original trajectory's collision checks do not guarantee that learned deviations are collision-free. Reject infeasible IK targets and stop on defined violations.
- Termination: an external supervisor identifies a stable lift or declares failure/timeout and enforces the attempt budget. Success checking is not automatically learned by the action policy.

The supervisor may request another attempt. A reopen-and-regrasp performed by the policy also counts toward the same attempt budget. Count every new closing attempt consistently.

### Learning procedure

1. Validate a scripted expert in the same simulated robot, camera, and action setup. For this narrow domain, it can use known object geometry and simulator poses to generate approach, closure, and lift commands. Verify actual outcomes rather than assuming scripted grasps succeed.
2. Record ordinary successful demonstrations, including observation histories, the initial TiPToP grasp condition, expert Cartesian commands, gripper commands, timestamps, and episode boundaries.
3. Introduce small, reproducible object displacements or approach offsets. Record expert corrections from those states, including reopening and realignment after missed grasps when feasible.
4. Train the diffusion policy on these observation–action sequences. At a high level, training adds noise to an expert action sequence and teaches the conditional network to denoise it using the observations and goal. This is imitation learning, not reward-based trial-and-error optimization.
5. Run the learner on development scenes; collect expert corrections from states it actually encounters; add these data and retrain. This is a simplified DAgger-inspired process, not a claim to implement the complete DAgger algorithm.
6. Compare ordinary-only training, extra ordinary training, and an equal amount of extra corrective training. Hold architecture and training budgets comparable. Evaluate on held-out configurations and perturbation realizations.

Privileged object poses are allowed for the training expert. Do not pass updated ground-truth object poses or oracle corrections into the evaluated policy. In particular, do not update the policy's nominal grasp with ground truth after a perturbation: doing so would remove the visual correction problem we intend to study.

The policy responds to recent visible evidence, including evidence that a grasp missed. Its weights stay fixed during a trial. It has only a short observation history, not an explicit persistent memory of all previously failed grasps. It can repeat an unsuccessful action. Improvement across runs requires deliberate correction collection and retraining.

Start with small position errors and missed grasps. Fast or hidden slip inside the fingers may not be observable or correctable with the proposed camera-only inputs and bounded action space.

Architecture reference: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [image-policy implementation](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py). Correction-collection reference: [DAgger](https://proceedings.mlr.press/v15/ross11a.html).

## Week-one objective

Run an upstream TiPToP baseline in the supplied simulator, record its configuration, and measure a small, reproducible set of grasp outcomes. Do not implement learned control or checkpoint retries yet.

This reproduces the upstream pipeline on selected scenes. It does not establish numerical reproduction of the paper's entire benchmark. Current upstream code also differs from the paper's original release, so record that distinction explicitly.

### How the baseline works

1. The simulator provides initial wrist RGB, depth, camera calibration/pose, and current arm configuration, together with the instruction.
2. Gemini detects objects and grounds the instruction into symbolic goals. SAM2 provides object masks. Depth supplies a point cloud; M2T2 predicts candidate grasps. The perception pipeline creates an object-centric geometric scene.
3. cuTAMP searches for a task sequence and jointly feasible grasps, placements, and robot configurations. cuRobo generates connecting robot trajectories.
4. The server returns joint-position trajectories and gripper actions. The simulation client steps through the returned plan.
5. A separate evaluator records what happened. Finishing the plan is not evidence that the target was grasped.

The client sends one perception/planning request per episode and then executes the stored plan. Reading fresh observations to track joints or render video is not visual grasp correction. Motor-level feedback can still be present in this baseline.

Simulation uses rendered ground-truth depth as documented upstream. It still uses perception and grasp prediction; do not replace those with ground-truth object identities, object meshes, or oracle grasps in the baseline.

## Concrete work sequence

### A. Establish the execution machine and freeze provenance

Use an available Linux NVIDIA GPU machine. The supplied TiPToP environment targets linux-64/CUDA 12; the simulation documentation requires Ubuntu 22.04 or later. Do not assume native Windows or a CPU-only machine is supported by this workflow. WSL would need its own compatibility verification.

Record OS, driver, GPU model/VRAM, available disk, Python/environment versions, and whether the simulator can run headlessly with cameras. Native CUDA extensions, model downloads, and Isaac Sim rendering all need checking. There is no verified combined VRAM minimum from this chat. The simulator and perception/planning components may require separate GPU processes or sequential execution.

Source revisions inspected on October 29:

- TiPToP: d8f5afdaa94a7432220c3042f9f80be5ab45aae8.
- droid-sim-evals: 69a914cf56e0da6adafe9428de39b6ceffe9bcc6.
- The inspected TiPToP code requires cuTAMP 0.0.6.

These are a reproducible starting point, not a tested-compatible release bundle. Recheck the repositories before setup. Pin the selected commits, preserve lockfiles, and record every dependency/checkpoint/configuration change. The cuRobo installer currently fetches its main branch, so record and subsequently freeze its actual installed SHA too. Record M2T2 and SAM2 versions and checkpoint hashes.

The inspected Gemini code defaults to gemini-robotics-er-1.6-preview. Confirm API access and model availability. A necessary model substitution changes provenance and must be reported, rather than silently described as an exact paper reproduction. Configure credentials without printing or committing their values.

### B. Install the simulation and planning components

Use a user-approved project workspace and keep the environments separate. In the commands below, TIPTOP_DIR is the parent of the three repositories. All command examples are Bash for the Linux execution machine, not PowerShell commands to run on the current desktop.

~~~bash
export TIPTOP_DIR="/path/to/approved/workspace/tiptop-project"
mkdir -p "$TIPTOP_DIR"
cd "$TIPTOP_DIR"
git clone https://github.com/tiptop-robot/tiptop.git
git clone https://github.com/williamshen-nz/M2T2.git
git clone --recurse-submodules https://github.com/tiptop-robot/droid-sim-evals.git
~~~

Check out the selected TiPToP and simulator commits before dependency installation. Install pixi and uv using their official instructions if absent. Then, in the appropriate repository:

~~~bash
cd "$TIPTOP_DIR/tiptop"
pixi install
pixi run setup-planners

cd "$TIPTOP_DIR/M2T2"
pixi run setup
pixi run download-weights

cd "$TIPTOP_DIR/droid-sim-evals"
uv sync
~~~

Download the scene assets and confirm the expected USD files exist. The simulation guide and simulator README currently give different S3 asset URLs (pi-sim-assets versus tiptop-sim-assets); validate the actual accessible archive and record its source/hash before relying on it. Do not overwrite an existing asset directory blindly.

SAM2 is an additional dependency/model used by perception. The inspected code downloads its checkpoint on first use. Verify this succeeds rather than interpreting a model-download problem as a grasp failure.

For simulation, skip the physical Franka controller daemon, ZED hardware configuration, and FoundationStereo service. The simulator supplies depth and camera parameters. Python client libraries may still be import dependencies; skipping physical hardware installation does not imply stripping package dependencies.

The simulator README contains a pi0.5 policy-server example. That is not the TiPToP baseline. Use the TiPToP server below. The lightweight openpi-client dependency does not mean we need VLA checkpoints or training.

Setup sources: [installation guide](https://tiptop-robot.readthedocs.io/en/latest/installation/), [simulation guide](https://tiptop-robot.readthedocs.io/en/latest/simulation/), [simulator README](https://github.com/tiptop-robot/droid-sim-evals/blob/main/README.md), and the pinned repositories.

### C. Verify geometry and run independent smoke tests

Before judging grasp success, verify that the simulated robot and planner use compatible arm/gripper geometry, joint order, TCP/grasp frame, and camera transforms. The simulator code names Panda joints and loads a Franka/Robotiq USD, while the inspected TiPToP default configuration selects fr3_robotiq. Inspect the assets and FK to determine the correct configuration; do not assume those defaults match or compensate for a model mismatch by training a policy.

Preserve the simulator's documented coordinate conventions and existing camera-frame offsets unless evidence demonstrates a bug. Log any change. Confirm the point cloud and grasp poses align visually with the scene.

Run planner import/demo checks and an M2T2 smoke test using the official instructions. On a headless host, use appropriate visualization-free options after inspecting actual CLI help. Do not make a graphical viewer a prerequisite for successful computation.

Then run the provided fixed-plan replay for scene 1, variant 0:

~~~bash
cd "$TIPTOP_DIR/droid-sim-evals"
uv run replay_json_traj.py --json-path tiptop_assets/tiptop_scene1_plan.json --scene 1 --variant 0
~~~

Inspect its saved video. This checks simulator assets, execution, and gripper behavior. It does not reproduce perception or planning. Run one episode at a time for this initial diagnostic, and verify the initial robot/object state matches the saved plan.

### D. Generate a fresh plan and execute it

Start M2T2 in one terminal:

~~~bash
cd "$TIPTOP_DIR/M2T2"
pixi run server
~~~

Set the Google API credential securely in the TiPToP server environment. In another terminal, start TiPToP:

~~~bash
cd "$TIPTOP_DIR/tiptop"
pixi run tiptop-server --host 127.0.0.1 --port 8765
~~~

Run an actual scene in a third terminal:

~~~bash
cd "$TIPTOP_DIR/droid-sim-evals"
uv run tiptop_eval.py --scene 1 --variant 0 --instruction "Put the Rubik's cube in the bowl." --episodes 1 --ws-host localhost --ws-port 8765
~~~

Inspect real CLI help on the pinned environment if a flag differs. Keep upstream planning/execution settings initially; document them. Leave RecGen and experimental next-to goals disabled for this baseline.

Verify the server built a fresh scene and returned a fresh plan, and the simulator executed it. Match the server run directory to the episode/video. The server writes planning artifacts under tiptop_server_outputs; the evaluation script saves videos under runs/date/time.

If GPU concurrency prevents online execution, use the documented offline route: capture a scene with save_h5_obs.py, run tiptop-h5 on the observation, and replay the resulting plan on the same scene/variant. Inspect its CLI and disable viewer spawning when necessary. Record that offline replay and the online client have different episode limits/execution details in the inspected code; do not combine them as equivalent measurements without checking.

### E. Add observation-only scoring and run a small baseline set

The inspected simulator has no task reward and only timeout termination. The evaluation script saves videos but does not compute grasp success. Neither response.success from the planning server nor client.plan_done means the object was lifted.

Implement a separate observer/evaluation wrapper that reads target object state and logs outcomes without changing actions, inserting retries, pausing the robot, or supplying oracle state to perception/planning. Inspect the scene assets to map the instructed object to its rigid-body entry.

Suggested initial scoring criterion: the target rises at least 5 cm above its settled starting height and remains lifted for at least 0.5 seconds while held near the closed gripper. Validate these thresholds against the native motion and geometry in pilot runs, then freeze them before comparison. A pushed or tossed object must not count as a successful grasp. If a hold criterion would require modifying baseline motion, use an observation-only criterion supported by that motion and document it instead.

For a pick-and-place trajectory, score the first grasp/lift segment before intentional release. Also record final task success separately if it can be labeled reliably. Do not add a dwell or stop at the lift to the original baseline merely to simplify scoring.

Start with scene 1, variants 0–4, three complete trials per variant: 15 pilot runs. This is a manageable project-local pilot, not a statistical reproduction of the paper. Each repetition must obtain a fresh plan rather than replaying the same cached plan. Reserve separate variants/configurations for later held-out evaluation. Additional scenes are useful only once this pilot is reproducible.

Record every scheduled run, including API/perception failures, no-plan outcomes, timeouts, and crashes. Do not report only successful planned executions. Report both overall grasp success per attempted trial and conditional success given a valid executed plan, with counts and explicit denominators. These answer different questions.

At minimum, each run record needs: unique ID; scene/variant; actual seed settings if available; instruction; repository/config/model versions; perception and planning status; planning time; number of closes; observed stable-lift result; execution duration; failure category; and paths to plan, logs, and video. Native CLI does not expose every desired seed/logging flag, so implement and document wrapper support rather than invent flags.

Inspect the video-saving/error path: planning failures currently skip normal video output. Preserve their error records regardless. Save traces or diagnostic images where practical. Review the native executor's gripper dwell, waypoint subsampling, plan_done timing, reset, and timeout behavior before interpreting failures. Any behavior-changing fix must be separately recorded and must not be silently folded into the original baseline.

Suggested failure labels are perception/grounding, planning, missed grasp, drop after closure/lift, execution/collision, timeout, and infrastructure. Use an unknown label when evidence is insufficient.

## Week-one completion criteria

- A documented GPU/Linux environment and pinned code/config/dependency provenance.
- A successful simulator/replay smoke test and a fresh end-to-end TiPToP run.
- A clear trace from initial observation through predicted grasps and returned plan to execution.
- An observation-only grasp evaluator with a verified target mapping and written success definition.
- A small baseline results file containing successes and failures, with reviewable videos/logs and clear denominators.
- A short account of the major observed failure modes and the local execution segment to replace next.
- Reproduction instructions that a teammate can follow.

Do not train the new policy, add checkpoint retries, change grasp models, port to IIWA, or attempt the complete MolmoSpaces benchmark as part of this week-one milestone. Those are later tasks. A working installation alone is a partial milestone, not the completed baseline experiment.

## Code-reading map

- TiPToP tiptop/tiptop_websocket_server.py: snapshot request, perception/planning pipeline, serialization, saved outputs. The success field means a plan was produced.
- TiPToP tiptop/perception_wrapper.py and tiptop/perception/: depth/point cloud, Gemini grounding, SAM2 masks, M2T2 grasps.
- TiPToP tiptop/planning.py: cuTAMP configuration, invocation, and trajectory/gripper serialization.
- TiPToP tiptop/config/tiptop.yml: robot selection, service URLs, depth/grasp association settings, and experimental flags.
- TiPToP tiptop/tiptop_offline.py: H5 observation loading and offline planning.
- Simulator tiptop_eval.py: episode/reset loop, client calls, timeout, and video saving.
- Simulator src/sim_evals/inference/tiptop_websocket.py: one planning request per episode and subsequent stored-plan execution. This is the eventual execution integration point.
- Simulator src/sim_evals/environments/droid_environment.py: observation/action definitions, dynamic scene registration, simulation timing, and absent task-success reward.
- Simulator src/sim_evals/environments/nvidia_droid.py: robot asset and actuator configuration.
- Simulator replay_json_traj.py and save_h5_obs.py: independent execution and observation-capture diagnostics.

Sources: [pinned TiPToP](https://github.com/tiptop-robot/tiptop/tree/d8f5afdaa94a7432220c3042f9f80be5ab45aae8), [pinned simulator](https://github.com/tiptop-robot/droid-sim-evals/tree/69a914cf56e0da6adafe9428de39b6ceffe9bcc6), and [TiPToP paper](https://arxiv.org/html/2603.09971v2).
