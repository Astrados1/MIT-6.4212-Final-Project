"""Read immutable traces and telemetry; write a separate scoring/resource audit.

Usage: python scripts/analyze_measurements.py results/SUITE \
    --sim-telemetry SIM_GPU.csv --planner-telemetry PLANNER_GPU.csv

This does not update records, attribution, labels, or summary outcomes. Distances
are measured to the recorded inner-finger-link origin midpoint, not the TCP.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_eval.scoring import LiftCriteria, score_first_grasp


def read_telemetry(path):
    if not path:
        return []
    rows = []
    with Path(path).open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                t = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                rows.append((t.timestamp(), float(row['memory.used [MiB]'].split()[0])))
            except (KeyError, ValueError):
                continue
    return rows


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite', type=Path)
    p.add_argument('--sim-telemetry', type=Path)
    p.add_argument('--planner-telemetry', type=Path)
    p.add_argument('--joint-limits', type=Path, help='USD joint-limit JSON with name/lower_rad/upper_rad')
    p.add_argument('--out', type=Path, help='separate output path when an earlier audit is frozen')
    a = p.parse_args()
    limits = {r['name']: r for r in json.loads(a.joint_limits.read_text())} if a.joint_limits else {}
    telemetry = {'simulator': read_telemetry(a.sim_telemetry), 'planner': read_telemetry(a.planner_telemetry)}
    schedule = {r['run_id']: r for r in (json.loads(line) for line in (a.suite/'schedule.jsonl').read_text().splitlines() if line.strip())}
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'notes': ['Read-only audit; hypothetical threshold scores do not replace recorded outcomes.',
                        'GPU peaks cover scheduling through record completion, sampled at approximately 1 Hz.',
                        'Summing separate-host peak memory is an upper bound, not a tested single-host deployment.'],
              'telemetry_files': {k: str(v) if v else None for k, v in [('simulator', a.sim_telemetry), ('planner', a.planner_telemetry)]},
              'runs': []}
    for path in sorted((a.suite/'runs').glob('*/record.json')):
        r = json.loads(path.read_text()); run_id = r['run_id']
        item = {'run_id': run_id, 'recorded_stable_lift': r.get('stable_lift'),
                'recorded_wall_s': r.get('wall_duration_s'), 'planning_s': r.get('planning_time_s')}
        start = timestamp(schedule.get(run_id, {}).get('scheduled_utc') or r['started_utc'])
        end = timestamp(r['finished_utc'])
        item['schedule_to_finish_s'] = end-start
        item['peak_memory_mib'] = {name: max((m for t, m in rows if start <= t <= end+1), default=None)
                                   for name, rows in telemetry.items()}
        trace_path = path.with_name('trace.npz')
        if trace_path.exists():
            with np.load(trace_path) as trace:
                names = list(trace['object_names']); target = r.get('target_object')
                if target in names and len(trace['t']):
                    key = 'objects_center' if 'objects_center' in trace else 'objects_pos'
                    initial = r.get('initial_object_centers') if key == 'objects_center' else r.get('initial_object_positions')
                    obj = trace[key][:, names.index(target)]; baseline = initial[target][2]
                    t, grip, cmd = trace['t'], trace['gripper_pos'], trace['gripper_cmd']
                    closed = cmd > .5; indices = np.flatnonzero(closed)
                    mask = np.zeros(len(t), dtype=bool)
                    if len(indices):
                        first = int(indices[0]); opened = np.flatnonzero(~closed[first:])
                        last = first+int(opened[0]) if len(opened) else len(t)
                        mask[first:last] = True
                    raised = mask & (obj[:, 2]-baseline >= .05)
                    dist = np.linalg.norm(obj-grip, axis=1)
                    item['position_reference'] = key
                    item['max_rise_m'] = float(np.max(obj[:, 2]-baseline))
                    item['raised_closed_distance_m_min_median_max'] = np.quantile(dist[raised], [0, .5, 1]).tolist() if raised.any() else None
                    item['proximity_sensitivity'] = {}
                    for limit in [.10, .15, .175, .20, .225]:
                        score = score_first_grasp(t, obj, grip, cmd, baseline, LiftCriteria(max_obj_to_gripper_m=limit))
                        item['proximity_sensitivity'][str(limit)] = {'stable_lift': score.lifted, 'longest_hold_s': score.longest_hold_s}
                    if 'joint_cmd' in trace and 'joint_pos' in trace:
                        command, measured = trace['joint_cmd'], trace['joint_pos']
                        error = np.max(np.abs(command-measured), axis=1)
                        item['max_joint_tracking_error_rad'] = float(error.max())
                        item['first_grasp_max_joint_tracking_error_rad'] = float(error[mask].max()) if mask.any() else None
                        item['joint_limit_violations'] = {}
                        for j in range(command.shape[1]):
                            name = f'panda_joint{j+1}'
                            if name not in limits:
                                continue
                            lo, hi = limits[name]['lower_rad'], limits[name]['upper_rad']
                            # Ignore floating-point noise at an exactly specified limit.
                            excess = np.maximum(lo-command[:, j], command[:, j]-hi)
                            violated = excess > 1e-4
                            if violated.any():
                                item['joint_limit_violations'][name] = {
                                    'simulator_limits_rad': [lo, hi],
                                    'command_min_max_rad': [float(command[:, j].min()), float(command[:, j].max())],
                                    'measured_min_max_rad': [float(measured[:, j].min()), float(measured[:, j].max())],
                                    'max_excess_rad': float(excess.max()),
                                    'violating_samples': int(violated.sum()),
                                    'violating_samples_during_first_grasp': int((violated & mask).sum()),
                                }
        result['runs'].append(item)
    output = a.out or a.suite/'measurement_audit.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(result["runs"])} records audited -> {output}')


if __name__ == '__main__':
    main()
