"""
etl_env.py — Core ETLEnvironment class.
Implements reset(), step(), state() per OpenEnv spec.
Physics engine = pure pandas. Deterministic given same seed.

Fixes applied:
  - profile_column: captures already_profiled BEFORE appending, so reward is correct
  - _build_observation: guards against _state=None on first reset call
  - Medium task: _tables stored correctly for grader
  - Hard drift: drift_notification passed to observation builder
  - steps_remaining in observation is correct at all times
  - traceback import removed (unused)
"""

import io
import uuid
import contextlib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    ETLAction, ETLObservation, ETLState, StepResult
)
from environment.fault_injector import generate_easy, generate_medium, generate_hard
from environment.grader import grade_easy, grade_medium, grade_hard, run_validation_checks
from environment.reward import compute_step_reward, compute_final_reward


TASK_CONFIG = {
    "easy":   {"max_steps": 15, "difficulty": 1},
    "medium": {"max_steps": 20, "difficulty": 2},
    "hard":   {"max_steps": 25, "difficulty": 3},
}


class ETLEnvironment:
    """
    OpenEnv-compliant ETL Pipeline Agent environment.

    Quickstart:
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        result = env.step(ETLAction(tool="profile_column", params={"column": "amount"}))
        state  = env.state()
    """

    def __init__(self, task_id: str = "easy"):
        assert task_id in TASK_CONFIG, f"task_id must be one of {list(TASK_CONFIG)}"
        self.task_id = task_id
        self._state: Optional[ETLState] = None
        self._df_working: Optional[pd.DataFrame] = None
        self._stored_code: str = ""
        self._profiled_cols: List[str] = []
        self._inspected: bool = False
        self._executed_before: bool = False
        self._validation_scores: Dict[str, float] = {}
        self._transform_history: List[str] = []
        self._errors_seen: List[str] = []
        self._cumulative_reward: float = 0.0
        self._tables: Optional[Dict[str, pd.DataFrame]] = None
        self._schema_target: Dict = {}
        # Hard task
        self._drift_event: Optional[Dict] = None
        self._drift_injected: bool = False
        self._drift_detected: bool = False
        self._steps_wasted_after_drift: int = 0
        self._pre_drift_row_count: int = 0

    # ──────────────────────────────────────────────────────────────
    #  reset()
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed: int = 42) -> StepResult:
        """Start a new episode. Returns initial observation."""
        cfg = TASK_CONFIG[self.task_id]

        if self.task_id == "easy":
            df_broken, df_gold, schema_target, faults = generate_easy(seed)
            df_working = df_broken.copy()
            tables = None
            drift_event = None

        elif self.task_id == "medium":
            tables, df_gold, schema_target, faults = generate_medium(seed)
            df_working = tables["orders"].copy()
            drift_event = None

        else:  # hard
            df_broken, df_gold, schema_target, faults, drift_event = generate_hard(seed)
            df_working = df_broken.copy()
            tables = None

        episode_id = str(uuid.uuid4())

        self._state = ETLState(
            task_id=self.task_id,
            episode_id=episode_id,
            step_count=0,
            max_steps=cfg["max_steps"],
            df_gold=df_gold,
            faults_planted=faults,
            batches_processed=0,
            schema_drift_step=None,
            episode_done=False,
        )

        # Reset all episode state
        self._df_working = df_working
        self._stored_code = ""
        self._profiled_cols = []
        self._inspected = False
        self._executed_before = False
        self._validation_scores = {}
        self._transform_history = []
        self._errors_seen = []
        self._cumulative_reward = 0.0
        self._tables = tables
        self._schema_target = schema_target
        self._drift_event = drift_event
        self._drift_injected = False
        self._drift_detected = False
        self._steps_wasted_after_drift = 0
        self._pre_drift_row_count = 0

        # Medium task: tell the agent which tables are available and their columns
        reset_msg = ""
        if self.task_id == "medium" and tables:
            table_info = []
            for tname, tdf in tables.items():
                table_info.append(f"  tables['{tname}']: columns={list(tdf.columns)}, rows={len(tdf)}")
            reset_msg = (
                "MEDIUM TASK — 3 tables available in execute_transform() via `tables` dict:\n"
                + "\n".join(table_info)
                + "\nJoin them to build the fact table. `df` starts as tables['orders']."
            )

        obs = self._build_observation(last_tool_output=reset_msg, drift_event=None)
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={"episode_id": episode_id, "task_id": self.task_id, "seed": seed},
        )

    # ──────────────────────────────────────────────────────────────
    #  step()
    # ──────────────────────────────────────────────────────────────

    def step(self, action: ETLAction) -> StepResult:
        """Execute one action. Returns observation, reward, done, info."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._state.step_count += 1

        # ── Hard task: inject schema drift at step 8 ─────────────
        drift_notification = None
        if (
            self.task_id == "hard"
            and self._drift_event is not None
            and self._state.step_count == self._drift_event["step_injected"]
            and not self._drift_injected
        ):
            self._drift_injected = True
            self._state.schema_drift_step = self._state.step_count
            self._pre_drift_row_count = len(self._df_working)
            drift_notification = self._drift_event

        # ── Snapshot profiled_cols BEFORE dispatch (for repeat detection) ──
        cols_before_step = list(self._profiled_cols)
        inspected_before_step = self._inspected

        # ── Dispatch tool ────────────────────────────────────────
        tool_result, tool_output_text = self._dispatch(action)

        # ── Compute step reward ──────────────────────────────────
        step_reward = compute_step_reward(
            tool=action.tool,
            result=tool_result,
            profiled_cols=cols_before_step,       # pre-step snapshot
            inspected=inspected_before_step,      # pre-step snapshot
            executed_before=self._executed_before,
            validation_scores=tool_result.get("validation_scores"),
        )

        # ── Hard task: count wasted steps after drift ────────────
        if (
            self.task_id == "hard"
            and self._drift_injected
            and not self._drift_detected
            and action.tool != "submit"
        ):
            self._steps_wasted_after_drift += 1

        # ── Detect drift via inspect_schema or profile after drift ─
        if self.task_id == "hard" and self._drift_injected and not self._drift_detected:
            if action.tool in ("inspect_sample", "validate", "write_transform", "fix_transform"):
                self._drift_detected = True
                # Do not penalise the step where agent first responds to drift
                self._steps_wasted_after_drift = max(0, self._steps_wasted_after_drift - 1)

        # ── Check done ───────────────────────────────────────────
        done = False
        final_score = None
        grader_breakdown = {}
        steps_remaining = self._state.max_steps - self._state.step_count

        if action.tool == "submit" or steps_remaining <= 0:
            done = True
            self._state.episode_done = True
            final_score, grader_breakdown = self._grade()
            step_reward = compute_final_reward(
                final_score, self._state.step_count,
                self._state.max_steps, grader_breakdown
            )

        self._cumulative_reward += step_reward

        obs = self._build_observation(
            last_tool_output=tool_output_text,
            drift_event=drift_notification,
        )

        info: Dict[str, Any] = {
            "step":               self._state.step_count,
            "tool":               action.tool,
            "step_reward":        step_reward,
            "cumulative_reward":  round(self._cumulative_reward, 4),
        }
        if final_score is not None:
            info["final_score"] = final_score
            info["grader_breakdown"] = grader_breakdown

        return StepResult(observation=obs, reward=step_reward, done=done, info=info)

    # ──────────────────────────────────────────────────────────────
    #  state()
    # ──────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """Returns safe episode metadata — no df_gold, no faults_planted."""
        if self._state is None:
            return {"status": "not_started"}
        return {
            "task_id":           self._state.task_id,
            "episode_id":        self._state.episode_id,
            "step_count":        self._state.step_count,
            "max_steps":         self._state.max_steps,
            "steps_remaining":   self._state.max_steps - self._state.step_count,
            "episode_done":      self._state.episode_done,
            "batches_processed": self._state.batches_processed,
            "schema_drift_step": self._state.schema_drift_step,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "validation_scores": dict(self._validation_scores),
            "profiled_cols":     list(self._profiled_cols),
        }

    # ──────────────────────────────────────────────────────────────
    #  TOOL HANDLERS
    # ──────────────────────────────────────────────────────────────

    def _dispatch(self, action: ETLAction) -> Tuple[Dict, str]:
        handlers = {
            "profile_column":    self._tool_profile_column,
            "inspect_sample":    self._tool_inspect_sample,
            "write_transform":   self._tool_write_transform,
            "execute_transform": self._tool_execute_transform,
            "validate":          self._tool_validate,
            "fix_transform":     self._tool_fix_transform,
            "load_to_target":    self._tool_load_to_target,
            "submit":            self._tool_submit,
        }
        handler = handlers.get(action.tool)
        if handler is None:
            return (
                {"success": False, "error": f"Unknown tool: {action.tool}"},
                f"ERROR: Unknown tool '{action.tool}'. Valid tools: {list(handlers)}",
            )
        return handler(action.params)

    def _tool_profile_column(self, params: Dict) -> Tuple[Dict, str]:
        col = params.get("column", "")
        if not col:
            return {"success": False, "column": "", "already_profiled": False}, \
                   "ERROR: params['column'] is required."
        if col not in self._df_working.columns:
            return (
                {"success": False, "column": col, "already_profiled": False},
                f"ERROR: Column '{col}' not found. Available columns: {list(self._df_working.columns)}",
            )

        already = col in self._profiled_cols   # capture BEFORE appending
        s = self._df_working[col]

        profile: Dict[str, Any] = {
            "column":       col,
            "dtype":        str(s.dtype),
            "null_rate":    round(float(s.isna().mean()), 4),
            "null_count":   int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "total_rows":   len(s),
        }
        if pd.api.types.is_numeric_dtype(s):
            profile.update({
                "min":  float(s.min()),
                "max":  float(s.max()),
                "mean": round(float(s.mean()), 4),
                "std":  round(float(s.std()), 4),
            })
        else:
            top = s.value_counts().head(5).to_dict()
            profile["top_values"]    = {str(k): int(v) for k, v in top.items()}
            profile["sample_values"] = [str(x) for x in s.dropna().head(3).tolist()]

        if not already:
            self._profiled_cols.append(col)

        text = f"Profile for '{col}':\n" + "\n".join(f"  {k}: {v}" for k, v in profile.items())
        return {"success": True, "column": col, "already_profiled": already, "profile": profile}, text

    def _tool_inspect_sample(self, params: Dict) -> Tuple[Dict, str]:
        n = min(int(params.get("n_rows", 5)), 20)
        sample = self._df_working.head(n)
        was_inspected = self._inspected
        self._inspected = True
        text = f"First {n} rows:\n{sample.to_string()}"
        return {"success": True, "n_rows": n, "was_inspected": was_inspected}, text

    def _tool_write_transform(self, params: Dict) -> Tuple[Dict, str]:
        code = params.get("code", "").strip()
        if not code:
            return {"success": False}, "ERROR: params['code'] is required and must not be empty."
        self._stored_code = code
        self._transform_history.append(code)
        return {"success": True}, f"Transform code stored ({len(code)} chars). Call execute_transform() to run it."

    def _tool_execute_transform(self, params: Dict) -> Tuple[Dict, str]:
        if not self._stored_code:
            return (
                {"success": False, "syntax_error": False},
                "ERROR: No code stored. Call write_transform() first.",
            )

        rows_in = len(self._df_working)
        df_copy = self._df_working.copy()
        namespace: Dict[str, Any] = {"df": df_copy, "pd": pd, "np": np}
        # Medium task: expose all 3 raw tables so the agent can do joins
        if self._tables:
            namespace["tables"] = {k: v.copy() for k, v in self._tables.items()}
        stdout_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(self._stored_code, namespace)  # nosec B102
        except SyntaxError as e:
            err = f"SyntaxError: {e}"
            self._errors_seen.append(err)
            return {"success": False, "syntax_error": True, "error": err}, f"SYNTAX ERROR: {e}\nFix your code and call fix_transform()."
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self._errors_seen.append(err)
            return {"success": False, "syntax_error": False, "error": err}, f"RUNTIME ERROR: {e}\nFix your code and call fix_transform()."

        result_df = namespace.get("df")
        if not isinstance(result_df, pd.DataFrame):
            err = "Your code must reassign the variable `df` to a pandas DataFrame."
            self._errors_seen.append(err)
            return {"success": False, "syntax_error": False, "error": err}, f"ERROR: {err}"

        self._df_working = result_df
        self._executed_before = True
        rows_out = len(result_df)

        stdout_snippet = stdout_buf.getvalue()[:200]
        text = (
            f"Execute successful.\n"
            f"  rows_in:  {rows_in}\n"
            f"  rows_out: {rows_out}\n"
            f"  columns:  {list(result_df.columns)}\n"
            f"  errors:   []"
        )
        if stdout_snippet:
            text += f"\n  stdout:   {stdout_snippet}"
        return {"success": True, "syntax_error": False, "rows_in": rows_in, "rows_out": rows_out, "errors": []}, text

    def _tool_validate(self, params: Dict) -> Tuple[Dict, str]:
        checks = params.get("checks", ["null_check", "type_check", "range_check"])
        scores = run_validation_checks(
            df=self._df_working,
            schema=self._schema_target,
            checks=checks,
        )
        self._validation_scores.update(scores)
        lines = [f"  {k}: {v:.4f}" for k, v in scores.items()]
        text = "Validation results:\n" + "\n".join(lines)
        return {"success": True, "validation_scores": scores}, text

    def _tool_fix_transform(self, params: Dict) -> Tuple[Dict, str]:
        code = params.get("code", "").strip()
        if not code:
            return {"success": False}, "ERROR: params['code'] is required."
        self._stored_code = code
        self._transform_history.append(code)
        error_msg = params.get("error_msg", "")
        return {"success": True}, f"Transform revised. Error addressed: {error_msg[:120] or 'N/A'}"

    def _tool_load_to_target(self, params: Dict) -> Tuple[Dict, str]:
        target_cols = set(self._schema_target.keys())
        agent_cols  = set(self._df_working.columns)
        missing = target_cols - agent_cols
        if missing:
            err = f"Missing required columns: {sorted(missing)}"
            return {"success": False, "error": err}, f"LOAD FAILED: {err}"
        self._state.batches_processed += 1
        extra = agent_cols - target_cols
        text = (
            f"Loaded to target successfully.\n"
            f"  rows:               {len(self._df_working)}\n"
            f"  extra_cols_ignored: {sorted(extra)}\n"
            f"  batches_committed:  {self._state.batches_processed}"
        )
        return {"success": True}, text

    def _tool_submit(self, params: Dict) -> Tuple[Dict, str]:
        reasoning = params.get("reasoning", "")
        return (
            {"success": True, "reasoning": reasoning},
            f"Episode submitted.\nReasoning: {reasoning[:200] or 'N/A'}\nRunning final grader...",
        )

    # ──────────────────────────────────────────────────────────────
    #  GRADER DISPATCH
    # ──────────────────────────────────────────────────────────────

    def _grade(self) -> Tuple[float, Dict]:
        df_agent = self._df_working
        df_gold  = self._state.df_gold

        if self.task_id == "easy":
            return grade_easy(df_agent, df_gold, self._schema_target)

        elif self.task_id == "medium":
            return grade_medium(df_agent, df_gold, self._schema_target, self._tables)

        else:
            return grade_hard(
                df_agent=df_agent,
                df_gold=df_gold,
                schema=self._schema_target,
                drift_event=self._drift_event,
                steps_wasted_after_drift=self._steps_wasted_after_drift,
                drift_detected=self._drift_detected,
                pre_drift_rows=self._pre_drift_row_count,
            )

    # ──────────────────────────────────────────────────────────────
    #  BUILD OBSERVATION
    # ──────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        last_tool_output: str = "",
        drift_event: Optional[Dict] = None,
    ) -> ETLObservation:
        df = self._df_working
        step_count   = self._state.step_count if self._state else 0
        max_steps    = self._state.max_steps  if self._state else 15
        steps_remaining = max(0, max_steps - step_count)

        quality_profile: Dict[str, Any] = {}
        for col in list(df.columns)[:10]:
            quality_profile[col] = {
                "null_rate": round(float(df[col].isna().mean()), 4),
                "dtype":     str(df[col].dtype),
            }

        return ETLObservation(
            dataset_sample=df.head(5).fillna("NULL").to_dict(orient="records"),
            schema_current={col: str(df[col].dtype) for col in df.columns},
            schema_target=self._schema_target,
            quality_profile=quality_profile,
            last_tool_output=last_tool_output,
            validation_scores=dict(self._validation_scores),
            transform_history=list(self._transform_history[-3:]),
            errors_seen=list(self._errors_seen[-5:]),
            steps_remaining=steps_remaining,
            schema_drift_event=drift_event,
        )
