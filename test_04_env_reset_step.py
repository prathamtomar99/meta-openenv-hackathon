"""
TEST 4 — Environment: reset() and step()
Run: python -m pytest tests/test_04_env_reset_step.py -v
Tests the full reset/step loop for all 3 task difficulties.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from environment.etl_env import ETLEnvironment
from environment.models import ETLAction, StepResult


class TestReset:
    def test_easy_reset_returns_stepresult(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert isinstance(result, StepResult)

    def test_reset_reward_is_zero(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert result.reward == 0.0

    def test_reset_not_done(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert result.done is False

    def test_reset_observation_has_sample(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert len(result.observation.dataset_sample) > 0

    def test_reset_steps_remaining_correct_easy(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert result.observation.steps_remaining == 15

    def test_reset_steps_remaining_correct_medium(self):
        env = ETLEnvironment(task_id="medium")
        result = env.reset(seed=42)
        assert result.observation.steps_remaining == 20

    def test_reset_steps_remaining_correct_hard(self):
        env = ETLEnvironment(task_id="hard")
        result = env.reset(seed=42)
        assert result.observation.steps_remaining == 25

    def test_reset_schema_target_present(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert len(result.observation.schema_target) > 0

    def test_reset_has_episode_id(self):
        env = ETLEnvironment(task_id="easy")
        result = env.reset(seed=42)
        assert "episode_id" in result.info

    def test_reset_reproducible(self):
        env = ETLEnvironment(task_id="easy")
        r1 = env.reset(seed=42)
        r2 = env.reset(seed=42)
        assert r1.observation.dataset_sample == r2.observation.dataset_sample

    def test_invalid_task_raises(self):
        with pytest.raises(AssertionError):
            ETLEnvironment(task_id="impossible")

    def test_step_before_reset_raises(self):
        env = ETLEnvironment(task_id="easy")
        with pytest.raises(RuntimeError):
            env.step(ETLAction(tool="submit"))


class TestStepTools:
    def setup_method(self):
        self.env = ETLEnvironment(task_id="easy")
        self.env.reset(seed=42)

    def test_profile_column_returns_reward(self):
        result = self.env.step(ETLAction(
            tool="profile_column", params={"column": "amount"}
        ))
        assert isinstance(result.reward, float)
        assert result.reward > 0  # first profile = +0.05 - 0.01 budget = +0.04

    def test_profile_column_output_has_stats(self):
        result = self.env.step(ETLAction(
            tool="profile_column", params={"column": "amount"}
        ))
        assert "min" in result.observation.last_tool_output or "null_rate" in result.observation.last_tool_output

    def test_profile_bad_column(self):
        result = self.env.step(ETLAction(
            tool="profile_column", params={"column": "nonexistent_col"}
        ))
        assert "ERROR" in result.observation.last_tool_output
        assert result.done is False  # error doesn't end episode

    def test_profile_repeat_penalized(self):
        self.env.step(ETLAction(tool="profile_column", params={"column": "amount"}))
        r2 = self.env.step(ETLAction(tool="profile_column", params={"column": "amount"}))
        assert r2.reward < 0  # repeat penalty -0.02 - 0.01 budget

    def test_inspect_sample(self):
        result = self.env.step(ETLAction(
            tool="inspect_sample", params={"n_rows": 3}
        ))
        assert result.done is False
        assert "rows" in result.observation.last_tool_output.lower()

    def test_write_transform_neutral_reward(self):
        result = self.env.step(ETLAction(
            tool="write_transform",
            params={"code": "df = df.dropna(subset=['customer_id'])"},
        ))
        # write alone = 0.0 - 0.01 budget = -0.01
        assert result.reward <= 0.0

    def test_execute_transform_success(self):
        self.env.step(ETLAction(
            tool="write_transform",
            params={"code": "df = df.dropna(subset=['customer_id'])"},
        ))
        result = self.env.step(ETLAction(tool="execute_transform", params={}))
        assert result.reward > 0.0  # execute success +0.10 - 0.01 budget
        assert "successful" in result.observation.last_tool_output.lower()

    def test_execute_without_code_errors(self):
        env2 = ETLEnvironment(task_id="easy")
        env2.reset(seed=42)
        result = env2.step(ETLAction(tool="execute_transform", params={}))
        assert "ERROR" in result.observation.last_tool_output

    def test_execute_syntax_error_penalized(self):
        self.env.step(ETLAction(
            tool="write_transform",
            params={"code": "df = df.this_is_not_valid python code!!!"},
        ))
        result = self.env.step(ETLAction(tool="execute_transform", params={}))
        assert result.reward < 0.0

    def test_validate_returns_scores(self):
        # Run a valid transform first
        self.env.step(ETLAction(
            tool="write_transform",
            params={"code": "df = df.dropna(subset=['customer_id'])"},
        ))
        self.env.step(ETLAction(tool="execute_transform", params={}))
        result = self.env.step(ETLAction(
            tool="validate",
            params={"checks": ["null_check", "type_check"]},
        ))
        assert "null_check" in result.observation.validation_scores
        assert isinstance(result.observation.validation_scores["null_check"], float)

    def test_steps_remaining_decrements(self):
        r1 = self.env.step(ETLAction(tool="profile_column", params={"column": "amount"}))
        r2 = self.env.step(ETLAction(tool="profile_column", params={"column": "status"}))
        assert r2.observation.steps_remaining == r1.observation.steps_remaining - 1

    def test_submit_ends_episode(self):
        result = self.env.step(ETLAction(
            tool="submit",
            params={"reasoning": "Done — cleaned data"},
        ))
        assert result.done is True
        assert "final_score" in result.info
        assert 0.0 <= result.info["final_score"] <= 1.0

    def test_step_after_done_raises(self):
        self.env.step(ETLAction(tool="submit", params={}))
        with pytest.raises(RuntimeError):
            self.env.step(ETLAction(tool="profile_column", params={"column": "amount"}))


class TestBudgetExhaustion:
    def test_episode_ends_when_budget_exhausted(self):
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        # Easy has 15 steps — use them all
        for i in range(14):
            result = env.step(ETLAction(
                tool="profile_column",
                params={"column": "amount"},
            ))
            if result.done:
                break
        # Last step should end it
        result = env.step(ETLAction(
            tool="profile_column", params={"column": "amount"}
        ))
        assert result.done is True
        assert "final_score" in result.info


class TestStateMethod:
    def test_state_before_reset(self):
        env = ETLEnvironment(task_id="easy")
        s = env.state()
        assert s["status"] == "not_started"

    def test_state_after_reset(self):
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        s = env.state()
        assert s["task_id"] == "easy"
        assert s["step_count"] == 0
        assert s["max_steps"] == 15

    def test_state_step_count_increments(self):
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        env.step(ETLAction(tool="profile_column", params={"column": "amount"}))
        s = env.state()
        assert s["step_count"] == 1

    def test_state_no_gold_exposed(self):
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        s = env.state()
        assert "df_gold" not in s
        assert "faults_planted" not in s


if __name__ == "__main__":
    print("Running environment reset/step tests...")

    r = TestReset()
    r.test_easy_reset_returns_stepresult(); print("  ✓ reset returns StepResult")
    r.test_reset_reward_is_zero(); print("  ✓ reset reward = 0")
    r.test_reset_not_done(); print("  ✓ reset not done")
    r.test_reset_observation_has_sample(); print("  ✓ reset has sample")
    r.test_reset_steps_remaining_correct_easy(); print("  ✓ easy steps = 15")
    r.test_reset_steps_remaining_correct_medium(); print("  ✓ medium steps = 20")
    r.test_reset_steps_remaining_correct_hard(); print("  ✓ hard steps = 25")
    r.test_reset_reproducible(); print("  ✓ reset reproducible")
    r.test_invalid_task_raises(); print("  ✓ invalid task raises")

    s = TestStepTools()
    s.setup_method()
    s.test_profile_column_returns_reward(); print("  ✓ profile reward positive")
    s.test_profile_bad_column(); print("  ✓ bad column error handled")

    s2 = TestStepTools()
    s2.setup_method()
    s2.test_write_transform_neutral_reward(); print("  ✓ write neutral reward")

    s3 = TestStepTools()
    s3.setup_method()
    s3.test_execute_transform_success(); print("  ✓ execute success reward")

    s4 = TestStepTools()
    s4.setup_method()
    s4.test_validate_returns_scores(); print("  ✓ validate returns scores")

    s5 = TestStepTools()
    s5.setup_method()
    s5.test_submit_ends_episode(); print("  ✓ submit ends episode")

    st = TestStateMethod()
    st.test_state_before_reset(); print("  ✓ state before reset")
    st.test_state_after_reset(); print("  ✓ state after reset")
    st.test_state_no_gold_exposed(); print("  ✓ state hides gold")

    print("\n✅ All environment tests passed!")
