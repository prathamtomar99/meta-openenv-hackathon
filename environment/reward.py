"""
reward.py — Per-step and final reward computation.
Dense signal: every action returns a float immediately.
No LLM judge. No sparse end-of-episode only reward.
"""

from typing import Dict, List, Optional


STEP_REWARDS = {
    "profile_column_first":    +0.05,
    "profile_column_repeat":   -0.02,
    "inspect_sample_first":    +0.02,
    "inspect_sample_repeat":   -0.01,
    "write_transform":          0.00,
    "execute_success":         +0.10,
    "execute_syntax_error":    -0.05,
    "execute_runtime_error":   -0.03,
    "validate_check_pass":     +0.10,
    "fix_transform":           +0.05,
    "load_to_target_success":  +0.05,
    "load_to_target_fail":     -0.02,
    "drift_detected":          +0.15,
    "step_budget_penalty":     -0.01,
}


def compute_step_reward(
    tool: str,
    result: Dict,
    profiled_cols: List[str],   # list BEFORE this step's col was appended
    inspected: bool,            # True if inspect_sample was called BEFORE this step
    executed_before: bool,
    validation_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute the immediate reward for a single step() call.
    NOTE: profiled_cols must be the state BEFORE the current tool ran,
    so repeat-detection is accurate. etl_env passes pre-append snapshot.
    """
    r = 0.0

    if tool == "profile_column":
        # already_profiled is set by etl_env before appending to profiled_cols
        already = result.get("already_profiled", False)
        r = STEP_REWARDS["profile_column_repeat"] if already else STEP_REWARDS["profile_column_first"]

    elif tool == "inspect_sample":
        r = STEP_REWARDS["inspect_sample_repeat"] if inspected else STEP_REWARDS["inspect_sample_first"]

    elif tool == "write_transform":
        r = STEP_REWARDS["write_transform"]

    elif tool == "execute_transform":
        if result.get("success"):
            r = STEP_REWARDS["execute_success"]
        elif result.get("syntax_error"):
            r = STEP_REWARDS["execute_syntax_error"]
        else:
            r = STEP_REWARDS["execute_runtime_error"]

    elif tool == "validate":
        if validation_scores:
            for score in validation_scores.values():
                if score >= 0.8:
                    r += STEP_REWARDS["validate_check_pass"]
        r = min(r, 0.50)   # cap per validate call

    elif tool == "fix_transform":
        r = STEP_REWARDS["fix_transform"]

    elif tool == "load_to_target":
        r = STEP_REWARDS["load_to_target_success"] if result.get("success") else STEP_REWARDS["load_to_target_fail"]

    elif tool == "submit":
        r = 0.0  # final score replaces this via compute_final_reward

    # Small per-step budget penalty
    r += STEP_REWARDS["step_budget_penalty"]
    return round(r, 4)


def compute_final_reward(
    final_score: float,
    steps_used: int,
    max_steps: int,
    checks: Dict[str, float],
) -> float:
    """Episode return = grader score + small efficiency bonus (up to +0.05)."""
    efficiency = max(0.0, (max_steps - steps_used) / max_steps) * 0.05
    return round(min(final_score + efficiency, 1.0), 4)
