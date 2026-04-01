"""
models.py — Typed Pydantic contracts for the ETL Pipeline Agent environment.
Defines Action, Observation, State, and Reward models per OpenEnv spec.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────
#  ACTION
# ─────────────────────────────────────────────

class ETLAction(BaseModel):
    """
    One action the agent can take per step.
    tool       → which tool to call
    params     → tool-specific arguments
    reasoning  → chain-of-thought (graded for partial credit)
    """
    tool: Literal[
        "profile_column",
        "inspect_sample",
        "write_transform",
        "execute_transform",
        "validate",
        "fix_transform",
        "load_to_target",
        "submit",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


# ─────────────────────────────────────────────
#  OBSERVATION  (what the agent sees)
# ─────────────────────────────────────────────

class ETLObservation(BaseModel):
    """
    Everything the agent can observe after reset() or step().
    df_gold is NEVER included here — it lives only in ETLState server-side.
    """
    dataset_sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First 5 rows of the current working DataFrame as dicts"
    )
    schema_current: Dict[str, str] = Field(
        default_factory=dict,
        description="{'col': 'dtype'} of the current working df"
    )
    schema_target: Dict[str, Any] = Field(
        default_factory=dict,
        description="The target schema contract the agent must satisfy"
    )
    quality_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Null rates, type error counts, value ranges found so far"
    )
    last_tool_output: str = Field(
        default="",
        description="Text result of the last tool call"
    )
    validation_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest validation check scores (0.0–1.0 per check)"
    )
    transform_history: List[str] = Field(
        default_factory=list,
        description="Python code strings tried so far (most recent last)"
    )
    errors_seen: List[str] = Field(
        default_factory=list,
        description="Execution errors encountered so far"
    )
    steps_remaining: int = Field(
        default=15,
        description="How many step() calls remain before forced episode end"
    )
    # Hard task only — None for Easy/Medium
    schema_drift_event: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Schema change notification injected at step 8 (Hard task only)"
    )


# ─────────────────────────────────────────────
#  STATE  (server-side — never fully sent to agent)
# ─────────────────────────────────────────────

class ETLState(BaseModel):
    """
    Full server-side state. df_gold and faults_planted are hidden from agent.
    state() endpoint returns a safe subset (no df_gold, no faults_planted).
    """
    task_id: str = Field(description="'easy' | 'medium' | 'hard'")
    episode_id: str = Field(description="UUID for reproducibility")
    step_count: int = Field(default=0)
    max_steps: int = Field(default=15)

    # Hidden from agent
    df_gold: Any = Field(default=None, description="Gold-standard clean DataFrame")
    faults_planted: List[str] = Field(
        default_factory=list,
        description="List of fault type names injected at reset()"
    )

    # Visible via state() endpoint
    batches_processed: int = Field(
        default=0,
        description="Number of batches committed (Hard task incremental tracking)"
    )
    schema_drift_step: Optional[int] = Field(
        default=None,
        description="Step number when schema drift was injected (Hard task only)"
    )
    episode_done: bool = Field(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)  # allows pandas DataFrame


# ─────────────────────────────────────────────
#  REWARD
# ─────────────────────────────────────────────

class ETLReward(BaseModel):
    """
    Reward returned after each step().
    step_reward   → immediate per-action signal
    final_score   → filled in only after submit() (0.0–1.0)
    breakdown     → component scores for logging
    """
    step_reward: float = Field(default=0.0)
    final_score: Optional[float] = Field(
        default=None,
        description="Filled by grader after submit() — None during episode"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Score components for logging/debugging"
    )


# ─────────────────────────────────────────────
#  STEP RESULT  (what step() returns)
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """Complete result returned by env.step()."""
    observation: ETLObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
