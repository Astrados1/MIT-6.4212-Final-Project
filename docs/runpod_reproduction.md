# Reproducing the November 5 split-host baseline

The verified deployment uses two Ubuntu 22.04 containers. The planner host is an RTX A6000
with driver 595.91.07; the simulator host is an RTX PRO 6000 Blackwell Max-Q with driver
590.48.01. The original Isaac Sim 5 runtime crashed before scene creation on the A6000/R595
host. GPU model alone is therefore not a sufficient compatibility check.

The simulator uses Isaac Sim 5.0.0.0 and Isaac Lab 2.2.0. Only the CUDA builds of Torch and
torchvision differ from the original simulator lock. The full environment, source pins,
asset hash and initial budget are in [split-host provenance](../provenance/split_host_20261105.json).
The tested experiment harness revision is `e4bb12239c0f5e52820e57320bd6bb582d159665`.
Subsequent report and audit changes do not change the active robot policy.

## Install and validate the simulator

Use identical paths on both hosts: project `/workspace/MIT-6.4212-Final-Project` and upstream
parent `/opt/tiptop-project`. Clone the pinned upstream repositories using `scripts/setup_upstream.sh`.
Install the planner and M2T2 environments on the A6000 following [setup.md](setup.md).
The simulator host only needs the simulator environment; its TiPToP checkout supplies an
artifact path, not a second local planner.

The tested simulator container image was `nvidia/cuda:12.8.1-devel-ubuntu22.04`, with
`NVIDIA_DRIVER_CAPABILITIES=all`. Container graphics dependencies included `libegl1`, `libgl1`,
`libxt6` and `libglu1-mesa`. Verify that Vulkan enumerates the NVIDIA GPU, not only a CPU renderer.

```bash
cd /opt/tiptop-project/droid-sim-evals
uv sync --frozen
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128'

export UV_NO_SYNC=1
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export XDG_RUNTIME_DIR=/tmp/xdg-tiptop
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
x = torch.ones((32, 32), device='cuda')
assert (x @ x).sum().item() == 32768
PY
```

Keep `UV_NO_SYNC=1` for every simulator command; an ordinary `uv run` synchronization can
restore the incompatible CUDA 12.6 build. Do not treat a successful CUDA test as a rendering
test: also inspect actual camera images and complete a robot replay. Cold shader initialization
on the tested host took more than 100 seconds.

Use the second runbook asset archive, `https://pi-sim-assets.s3.us-east-1.amazonaws.com/assets.zip`.
Its SHA-256 is `b791aef3585fea0b56605e9ac847970d399866e909f8349f358749321fb9567d`.
It includes all 51 configured scene files. The other archive lacked scene 3 variants 10 and 11.

## Connect the planner services

Start M2T2 on port 8123 and TiPToP on loopback port 8765 on the A6000, using the existing
Gemini ER 2 patch. Supply the Google credential only through the planner process environment.
The simulator does not need that credential.

Create a dedicated SSH identity and a `planner` host entry on the simulator. The authorized key
can be limited to forwarding to `127.0.0.1:8765` and `127.0.0.1:8123`, while permitting the
read/extraction commands needed to retrieve server artifacts. Keep keys outside this repository.

```bash
ssh -F /root/.ssh/planner_ssh_config -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:8765:127.0.0.1:8765 \
  -L 127.0.0.1:8123:127.0.0.1:8123 planner
```

In the evaluation terminal on the simulator:

```bash
export GRASP_EVAL_PLANNER_SSH_HOST=planner
export GRASP_EVAL_PLANNER_SSH_CONFIG=/root/.ssh/planner_ssh_config
export UV_NO_SYNC=1
cd /workspace/MIT-6.4212-Final-Project
python3 -m grasp_eval.preflight --tiptop-dir /opt/tiptop-project \
  --remote-planners --probe-simulator-runtime --sim-check
```

The missing-local-credential WARN is expected in this topology. Verify model access from the
planner host and then use a full fresh end-to-end request as the credential and pipeline gate.
The runtime GPU probe intentionally checks the installed CUDA build instead of assuming the
original lock's architecture support. All five scene inspections passed in this deployment.

## Run and preserve evidence

Use the explicit threshold arguments documented in [scoring.md](scoring.md), and a fresh
provenance file for each changed environment or planner configuration. The pilot command used:

```bash
python3 -m grasp_eval.run_suite pilot --tiptop-dir /opt/tiptop-project \
  --target-object rubiks_cube --out-name pilot_scene1_20261105 --max-hours 1.25 \
  --provenance-file provenance/split_host_20261105.json \
  --eval-args --max-obj-to-gripper-m 0.20
```

Record `nvidia-smi` timestamp, memory use and utilization on both hosts at approximately 1 Hz.
The scheduler's time limit stops new trials; allow for the per-trial timeout when setting an
outer deadline. No simulator assets, physics settings, actions, dwell times or retry policy
were changed in these baseline runs.

Copy the whole results folder, including gitignored videos and traces. Also copy each referenced
TiPToP output directory into `results/<suite>/server_outputs/<timestamp>/`; the local review
generator resolves those copies while preserving the original paths in immutable records.
Regenerate the local review with `python3 -m grasp_eval.summarize results/<suite>`.

Before sharing native crash logs, redact credential-like vendor build metadata; omit crash
memory dumps. Preserve hashes of the sanitized exported files, and verify downloads before
terminating the pods. Large server outputs, videos and traces are excluded from Git.

## Interpretation limits

The baseline planner still uses the upstream `fr3_robotiq` configuration. Although FR3 and Panda
grasp-frame FK agree in the checked configurations, their joint limits differ. The simulator's
Panda joint 6 upper limit is 3.7525 rad; the planner FR3 model allows 4.5169 rad. A pilot failure
commanded 4.219 rad just before release, producing saturation and a placement error. This is a
model/asset integration defect, not evidence that the grasp was unstable. Any Panda-config
validation must be identified as a separate condition rather than mixed into this baseline.

The scheduler now stops on upstream API quota/rate-limit errors even when both local services
remain healthy. Resolve the API limit before using `--resume`; do not count rejected calls as
robot grasp failures. The November 5 key's project reported a free-tier daily request limit of
20 for Gemini ER 2. The first run lacked this stop condition and retained all 35 quota failures.

The scene inspections covered variant 0 of each scene. Later records exposed cubes falling
below the table in scene 4 variants 1–3. Inspect the actual variants before a future survey;
a passing variant-0 inspection does not establish valid initial states for every variant.

For the separate cached-scene Panda diagnostic, run the following in the pinned TiPToP pixi
environment, then replay the resulting plan in scene 1 variant 2 with seed 1200 and frozen
scoring. The script uses saved geometry, goals and grasp candidates and makes no VLM calls.

```bash
pixi run python /workspace/MIT-6.4212-Final-Project/scripts/replan_saved_scene.py \
  --source /opt/tiptop-project/tiptop/tiptop_server_outputs/2026-11-05_05-13-21 \
  --config /workspace/MIT-6.4212-Final-Project/configs/tiptop_panda_sim.yml \
  --out /opt/tiptop-project/tiptop/tiptop_server_outputs/panda_cached_check \
  --seed 1200
```

This script rebuilds the original movable/static roles because the saved post-planning pickle
contains an in-place mutation by cuTAMP. The configuration selects Panda's full robot model,
including limits; this changes more than one joint bound. Preserve it as a diagnostic condition.
