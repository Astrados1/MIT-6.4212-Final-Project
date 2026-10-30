"""Per-run record schema, automatic failure labeling, and JSONL I/O.

Raw records are written once by observed_eval.py and never edited. Human review
goes in a separate manual_labels.csv that summarize.py applies on top.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FAILURE_LABELS = (
    "perception_grounding",
    "planning",
    "missed_grasp",
    "drop_after_closure_or_lift",
    "execution_collision",
    "timeout",
    "infrastructure",
    "unknown",
)


@dataclass
class RunRecord:
    run_id: str
    scene: int
    variant: int
    trial: int
    instruction: str
    target_object: str

    started_utc: str = ""
    finished_utc: str = ""
    wall_duration_s: float | None = None

    # Seeds we set. Gemini, M2T2 and cuTAMP run in other processes and are not
    # seeded by this harness, so repeated runs are not expected to be identical.
    seed: int | None = None

    versions: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)

    # Perception/planning, as reported by the TiPToP server.
    server_response_received: bool = False
    server_success: bool | None = None  # True means a plan was produced, nothing more
    server_error: str | None = None
    server_save_dir: str | None = None
    planning_time_s: float | None = None
    plan_num_steps: int | None = None
    plan_num_close_steps: int | None = None

    # Execution in the simulator.
    executed_steps: int = 0
    sim_duration_s: float | None = None
    plan_done: bool = False
    truncated: bool = False  # hit the episode time limit
    terminated: bool = False

    # Observation-only outcome.
    score: dict[str, Any] | None = None
    stable_lift: bool | None = None  # None when no plan was executed
    failure_category: str | None = None  # None on success
    failure_category_source: str = "auto"
    failure_detail: str | None = None
    task_success: bool | None = None  # final place success; labeled manually

    initial_object_positions: dict[str, list[float]] = field(default_factory=dict)
    final_object_positions: dict[str, list[float]] = field(default_factory=dict)

    artifacts: dict[str, str] = field(default_factory=dict)
    harness_exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def valid_executed_plan(self) -> bool:
        return bool(self.server_success) and self.executed_steps > 0 and self.harness_exception is None


def auto_label(rec: RunRecord) -> tuple[str | None, str | None]:
    """Best-effort (failure_category, detail) from the record alone.

    Collisions and grounding mistakes are not detectable here; they stay
    "unknown" or a grasp label until someone reviews the video and server log.
    """
    if rec.harness_exception is not None:
        return "infrastructure", "harness/simulator exception; see record.harness_exception"
    if not rec.server_response_received:
        return "infrastructure", rec.server_error or "no response from TiPToP server"
    if not rec.server_success:
        err = rec.server_error or ""
        if "cuTAMP failed" in err:
            return "planning", err
        return "unknown", f"server returned no plan; check server log: {err}"
    if rec.stable_lift:
        return None, None
    hint = (rec.score or {}).get("failure_hint")
    if rec.score and rec.score.get("num_close_commands", 0) == 0:
        if rec.truncated:
            return "timeout", "episode time limit reached before any close command"
        return "unknown", "plan executed without a close command"
    if hint:
        detail = "trace-based hint; confirm on video"
        if rec.truncated:
            detail += " (episode also hit the time limit)"
        return hint, detail
    return "unknown", None


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
