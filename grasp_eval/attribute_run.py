"""Compute stage attribution for one finished run (needs numpy; run in the simulator's uv env).

    uv run python $PROJECT/grasp_eval/attribute_run.py --run-dir results/<suite>/runs/<run_id> \
        [--perceived results/<suite>/perceived/<server timestamp>.json] [--append-to results/<suite>/attribution.jsonl]

Writes <run-dir>/attribution.json. Never modifies record.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp_eval.attribution import attribute  # noqa: E402
from grasp_eval.records import append_jsonl  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--perceived", type=Path, default=None)
    p.add_argument("--append-to", type=Path, default=None)
    args = p.parse_args(argv)

    record_path = args.run_dir / "record.json"
    if not record_path.exists():
        print(f"No record at {record_path}; nothing to attribute.")
        return 1
    record = json.loads(record_path.read_text())

    trace = None
    trace_path = args.run_dir / "trace.npz"
    if trace_path.exists():
        with np.load(trace_path) as z:
            trace = {k: z[k] for k in z.files}

    plan = None
    plan_path = record.get("artifacts", {}).get("server_plan")
    if plan_path and Path(plan_path).exists():
        plan = json.loads(Path(plan_path).read_text())

    perceived = json.loads(args.perceived.read_text()) if args.perceived and args.perceived.exists() else None

    attr = attribute(record, trace, perceived, plan)
    (args.run_dir / "attribution.json").write_text(json.dumps(attr, indent=2))
    if args.append_to:
        append_jsonl(args.append_to, attr)
    print(json.dumps(attr, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
