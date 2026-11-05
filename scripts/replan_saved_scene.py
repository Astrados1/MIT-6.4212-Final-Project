"""Replan a saved TiPToP scene with an explicit robot config, without perception/API calls.

Run in the pinned TiPToP pixi environment. This is a separate diagnostic condition,
not part of the baseline. The scene, goal and M2T2 candidates are reused verbatim;
the planner model and random seed are recorded in the output metadata.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', type=int, default=1200)
    a = p.parse_args()
    import dill
    import numpy as np
    import torch
    from tiptop.config import set_tiptop_cfg_from_file

    cfg = set_tiptop_cfg_from_file(a.config.resolve())
    from tiptop.motion_planning import build_curobo_solvers
    from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
    from tiptop.recording import save_run_metadata
    from tiptop.tiptop_run import create_tamp_environment
    from tiptop.utils import add_file_handler, setup_logging

    a.out.mkdir(parents=True, exist_ok=False)
    setup_logging()
    add_file_handler(a.out/'tiptop_run.log')
    original = json.loads((a.source/'metadata.json').read_text())
    source_files = ['metadata.json', 'tiptop.yml', 'perception/cutamp_env.pkl', 'perception/grasps.pt']
    hashes = {name: hashlib.sha256((a.source/name).read_bytes()).hexdigest() for name in source_files}
    with (a.source/'perception/cutamp_env.pkl').open('rb') as f:
        saved_env = dill.load(f)
    # run_cutamp mutates env.movables/type_to_objects in place by appending
    # statics. The server pickles that post-planning object in its finally block.
    # Rebuild the original roles through the same factory as fresh perception;
    # geometry, poses, grounded atoms and cached grasp candidates are unchanged.
    objects = {obj.name: obj for obj in [*saved_env.movables, *saved_env.statics]}
    table = objects.pop('table')
    env, surfaces = create_tamp_environment(
        objects, table, original['perception']['grounded_atoms'], include_workspace=False)
    input_roles = {'movables': [o.name for o in env.movables],
                   'statics': [o.name for o in env.statics]}
    grasps = torch.load(a.source/'perception/grasps.pt', map_location='cuda', weights_only=False)
    q_init = np.asarray(original['observation']['q_at_capture'], dtype=np.float32)
    shutil.copytree(a.source/'perception', a.out/'perception')
    for name in ['rgb.png', 'bboxes_viz.png', 'masks_viz.png']:
        if (a.source/name).exists():
            shutil.copy2(a.source/name, a.out/name)
    shutil.copy2(a.config, a.out/'tiptop.yml')
    config = build_tamp_config(num_particles=256, max_planning_time=60., opt_steps=500,
                               robot_type=cfg.robot.type, time_dilation_factor=cfg.robot.time_dilation_factor,
                               near_placement=cfg.experimental.pick_place_next_to)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ik, motion, _ = build_curobo_solvers(256, config.coll_n_spheres, include_workspace=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    plan, elapsed, reason = run_planning(env, config, q_init, ik, grasps, motion, surfaces,
                                         experiment_dir=a.out/'cutamp')
    save_run_metadata(a.out, datetime.now(timezone.utc).isoformat(), original['task_instruction'],
                      q_init, original['observation']['world_from_cam'], 0.,
                      original['perception']['grounded_atoms'], plan is not None, reason, elapsed)
    meta = json.loads((a.out/'metadata.json').read_text())
    meta['diagnostic'] = {'source': str(a.source.resolve()), 'source_hashes': hashes,
                          'perception_reused': True, 'planner_seed': a.seed, 'robot_type': cfg.robot.type,
                          'input_roles': input_roles,
                          'environment_rebuilt': 'Original factory, cached geometry and grounded atoms; remove post-planning role mutation.',
                          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'scope': 'Cached-scene replanning; not a fresh end-to-end baseline trial.'}
    (a.out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    if plan is not None:
        save_tiptop_plan(serialize_plan(plan, q_init), a.out/'tiptop_plan.json')
    print(json.dumps({'success': plan is not None, 'planning_s': elapsed, 'failure': reason,
                      'output': str(a.out), 'robot': cfg.robot.type}), flush=True)
    return 0 if plan is not None else 2


if __name__ == '__main__':
    result = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(result)  # upstream also avoids CUDA-destructor shutdown crashes
