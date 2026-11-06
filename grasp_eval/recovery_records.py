"""Trial and attempt records for week-two checkpoint recovery.

A trial holds one or two grasp attempts in one continuous episode. Trials go to
recovery_trials.jsonl, attempts to attempts.jsonl, never to the week-one runs.jsonl:
attempts are not independent trials and must not share that file's denominators.

Like week one, raw records are written once and never edited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TRIALS_FILE = "recovery_trials.jsonl"
ATTEMPTS_FILE = "attempts.jsonl"
SCHEMA_VERSION = 1


@dataclass
class AttemptRecord:
    trial_id: str
    attempt_index: int  # 0-based

    started_sim_s: float | None = None  # simulated time since the trial clock started
    started_wall_s: float | None = None
    start_joint_pos: list[float] | None = None  # measured, as sent to TiPToP as q_init
    # Evaluator-only ground truth at the start of the attempt (never sent to a sensor detector).
    object_positions: dict[str, list[float]] = field(default_factory=dict)
    object_quats: dict[str, list[float]] = field(default_factory=dict)
    object_centers: dict[str, list[float]] = field(default_factory=dict)  # geometry-box centers (scored)
    # Fingerprint of the observation sent to TiPToP, to show each attempt planned from a new view.
    request: dict[str, Any] = field(default_factory=dict)

    plan_source: str = "server"  # server | replay (paired first plan)
    replayed_plan: str | None = None
    server_response_received: bool = False
    server_success: bool | None = None
    server_error: str | None = None
    server_save_dir: str | None = None
    planning_time_s: float | None = None  # server-reported; the simulator is paused meanwhile
    planning_wall_s: float | None = None  # client-measured wall time of the blocking request
    plan_num_steps: int | None = None
    segment: dict[str, Any] | None = None  # plan_segments.GraspSegment
    segment_error: str | None = None

    actions_applied: int = 0
    retraction_applied: bool = False
    final_joint_cmd: list[float] | None = None
    final_joint_pos: list[float] | None = None  # measured at the checkpoint
    final_joint_error_max_rad: float | None = None
    hold_steps: int = 0
    hold_s: float | None = None
    closes_in_attempt: int | None = None
    displacement: dict[str, Any] | None = None  # probe applied in this attempt (attempt 0 only)

    # Outcome. `historical_lift` is the week-one score_first_grasp on this attempt's slice:
    # held long enough at some earlier point. `checkpoint` is whether it is held *now*.
    historical_lift: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None  # checkpoint.CheckResult
    outcome: str | None = None  # holding_target | failed_empty | uncertain | None (not reached)
    outcome_source: str | None = None  # detector name; "oracle-state" is privileged
    outcome_privileged: bool | None = None
    reference_outcome: str | None = None  # oracle verdict, when a non-oracle detector decided

    retry: dict[str, Any] | None = None  # empty-hand return after this attempt, if any
    stop: str | None = None  # stop reason if the trial ended in this attempt
    trace_slice: list[int] | None = None  # [start, end) indices into the trial trace
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryTrialRecord:
    trial_id: str
    scene: int
    variant: int
    trial: int
    instruction: str  # the original instruction, used unchanged for every attempt
    target_object: str  # simulator object scored in every attempt
    target_label: str | None = None  # TiPToP's perceived name, if given

    schema_version: int = SCHEMA_VERSION
    mode: str = ""  # checkpoint_one_attempt | checkpoint_recovery_2
    detector: str = "oracle-state"
    detector_privileged: bool = True
    variant_name: str = "grasp-checkpoint recovery (not the paper's placement-level checkpoint)"
    limits: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)  # LiftCriteria (historical score)
    check_criteria: dict[str, Any] = field(default_factory=dict)  # OracleCheckCriteria
    return_config: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=lambda: {"type": "baseline"})
    seed: int | None = None
    versions: dict[str, Any] = field(default_factory=dict)

    started_utc: str = ""
    finished_utc: str = ""
    process_startup_s: float | None = None  # process start -> trial clock start (sim launch, settle)
    sim_duration_s: float | None = None  # trial clock: first planning request -> stop
    wall_duration_s: float | None = None
    episode_length_s: float | None = None

    first_attempt_success: bool | None = None  # oracle-confirmed at the first checkpoint
    first_attempt_historical_lift: bool | None = None  # week-one criterion on attempt 0
    final_success: bool = False
    num_attempts: int = 0  # attempts that reached execution
    num_closes: int = 0
    num_planning_calls: int = 0
    retry_triggered: bool = False
    stop_reason: str = ""
    stop_detail: str | None = None
    time_to_success_sim_s: float | None = None
    time_to_success_wall_s: float | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)

    object_names: list[str] = field(default_factory=list)
    initial_object_positions: dict[str, list[float]] = field(default_factory=dict)
    final_object_positions: dict[str, list[float]] = field(default_factory=dict)
    # Geometry-box centers, used for all scoring; raw rigid-body origins above are kept unchanged.
    # Some scene-4/5 meshes sit >10 cm from their body origin (docs/scoring.md).
    initial_object_centers: dict[str, list[float]] = field(default_factory=dict)
    final_object_centers: dict[str, list[float]] = field(default_factory=dict)
    object_local_bounds: dict[str, Any] = field(default_factory=dict)
    object_bounds_notes: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    harness_exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TASK_TRIALS_FILE = "task_trials.jsonl"


@dataclass
class TaskTrialRecord:
    """One task-level checkpoint-recovery trial (grasp_eval/task_recovery_eval.py)."""

    trial_id: str
    scene: int
    variant: int
    trial: int
    instruction: str
    goal: list[dict[str, Any]] | None = None  # configs/scenes.json goal, scored with task_goal.check_goal
    target_object: str | None = None  # resolves "$target" in the goal (scenes 1-3)

    schema_version: int = 1
    mode: str = ""  # task_recovery_rN | task_checkpoint_only
    lift_mode: str = ""  # reverse_approach | vertical_ik
    detector: str = "oracle-state"
    detector_privileged: bool = True
    limits: dict[str, Any] = field(default_factory=dict)
    check_criteria: dict[str, Any] = field(default_factory=dict)
    return_config: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=lambda: {"type": "baseline"})
    seed: int | None = None
    versions: dict[str, Any] = field(default_factory=dict)
    paired_baseline_run: str | None = None  # run_id of the baseline trial whose plan attempt 0 replays
    paired_baseline_dir: str | None = None
    first_plan_source: str | None = None  # replay | server

    started_utc: str = ""
    finished_utc: str = ""
    process_startup_s: float | None = None
    sim_duration_s: float | None = None
    wall_duration_s: float | None = None
    episode_length_s: float | None = None

    stop_reason: str = ""
    stop_detail: str | None = None
    task_success: bool | None = None  # the goal check on the final state (same function as the baseline)
    task_success_at_baseline_limit: bool | None = None  # the same check on the state at the baseline's episode end
    task_check: dict[str, Any] = field(default_factory=dict)
    task_check_at_baseline_limit: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)  # per attempt: source, server dir, units, timing
    num_attempts: int = 0
    num_retries: int = 0
    num_planning_calls: int = 0
    first_checkpoint_verdict: str | None = None
    recovered: bool = False  # a retry happened and the task succeeded
    events: list[dict[str, Any]] = field(default_factory=list)  # carry drops, recovery steps, supervisor events

    object_names: list[str] = field(default_factory=list)
    object_local_bounds: dict[str, Any] = field(default_factory=dict)
    initial_object_positions: dict[str, list[float]] = field(default_factory=dict)
    initial_object_quats: dict[str, list[float]] = field(default_factory=dict)
    final_object_positions: dict[str, list[float]] = field(default_factory=dict)
    final_object_quats: dict[str, list[float]] = field(default_factory=dict)
    final_object_centers: dict[str, list[float]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    harness_exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
