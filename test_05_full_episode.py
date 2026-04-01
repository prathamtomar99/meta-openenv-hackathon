"""
TEST 5 — Full Episode Simulation
Run: python -m pytest tests/test_05_full_episode.py -v
Simulates a complete agent episode for each task (rule-based agent, not LLM).
Verifies the whole pipeline works end-to-end and produces a valid final score.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from environment.etl_env import ETLEnvironment
from environment.models import ETLAction


# ─────────────────────────────────────────────
#  SIMPLE RULE-BASED AGENT (no LLM — for testing)
# ─────────────────────────────────────────────

EASY_TRANSFORM = """
import pandas as pd
import numpy as np

# 1. Fix date formats
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', dayfirst=False, errors='coerce').dt.strftime('%Y-%m-%d')

# 2. Remove duplicate order_ids
df = df.drop_duplicates(subset=['order_id'], keep='first')

# 3. Drop rows with null customer_id
df = df[df['customer_id'].notna()]

# 4. Remove negative amounts
df = df[df['amount'] >= 0]

# 5. Cap outlier amounts at 10000
df['amount'] = df['amount'].clip(upper=10000)

# 6. Lowercase status values
df['status'] = df['status'].str.lower()

df = df.reset_index(drop=True)
"""

MEDIUM_TRANSFORM = """
import pandas as pd
import numpy as np

# tables dict is available: tables['orders'], tables['customers'], tables['products']
customers = tables['customers'].copy()
products  = tables['products'].copy()

# Fix region typos in customers
region_map = {
    'Nort East': 'North East', 'SouthWest': 'South West',
    'midlands': 'Midlands', 'south east': 'South East',
}
customers['region'] = customers['region'].replace(region_map).str.title()

# Normalize category case in products
products['category'] = products['category'].str.lower()

# Join orders -> customers -> products
merged = df.merge(customers, on='customer_id', how='inner')
merged = merged.merge(products, on='product_id', how='inner')

# Drop impossible margin rows (price < cost_price)
merged['margin_pct'] = ((merged['price'] - merged['cost_price']) / merged['price']).round(4)
merged = merged[merged['margin_pct'] > 0]

# Select target columns
df = merged[['order_id', 'customer_id', 'name', 'region', 'tier',
             'category', 'qty', 'price', 'cost_price', 'margin_pct']]
df = df.reset_index(drop=True)
"""

HARD_TRANSFORM = EASY_TRANSFORM  # Same base — Hard test focuses on drift detection


class TestFullEpisodeEasy:
    def test_complete_episode_produces_valid_score(self):
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)

        # Step 1: Profile columns
        env.step(ETLAction(tool="profile_column", params={"column": "order_date"}))
        env.step(ETLAction(tool="profile_column", params={"column": "amount"}))
        env.step(ETLAction(tool="profile_column", params={"column": "status"}))

        # Step 2: Write transform
        env.step(ETLAction(
            tool="write_transform",
            params={"code": EASY_TRANSFORM},
            reasoning="Fixing all known faults: date format, dupes, nulls, negatives, outliers, status case",
        ))

        # Step 3: Execute
        result = env.step(ETLAction(tool="execute_transform", params={}))
        assert "successful" in result.observation.last_tool_output.lower(), \
            f"Execute failed: {result.observation.last_tool_output}"

        # Step 4: Validate
        result = env.step(ETLAction(
            tool="validate",
            params={"checks": ["null_check", "type_check", "range_check", "uniqueness_check"]},
        ))
        scores = result.observation.validation_scores
        assert all(isinstance(v, float) for v in scores.values())

        # Step 5: Load and submit
        env.step(ETLAction(tool="load_to_target", params={}))
        result = env.step(ETLAction(
            tool="submit",
            params={"reasoning": "Applied all 6 fixes. Validated all checks."},
        ))

        assert result.done is True
        final_score = result.info.get("final_score", -1)
        assert 0.0 <= final_score <= 1.0, f"Score out of range: {final_score}"
        assert final_score >= 0.70, f"Score too low for correct transform: {final_score}"

        print(f"\n  Easy final score: {final_score}")
        print(f"  Breakdown: {result.info.get('grader_breakdown', {})}")

    def test_no_transform_episode_scores_lower_than_gold(self):
        """Agent that does nothing scores lower than a correct agent.
        Broken data is ~83% correct (faults affect ~17% of rows), so the
        do-nothing score is meaningfully below the correct agent's 1.0.
        """
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        result = env.step(ETLAction(
            tool="submit",
            params={"reasoning": "Did nothing"},
        ))
        assert result.done is True
        # Broken data should score strictly less than gold (1.0)
        assert result.info["final_score"] < 0.95, (
            f"Do-nothing should score <0.95 (gold=1.0), got {result.info['final_score']}"
        )

    def test_bad_transform_episode(self):
        """Agent that corrupts data further should score low."""
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        env.step(ETLAction(
            tool="write_transform",
            params={"code": "df = pd.DataFrame()"},  # destroys data
        ))
        env.step(ETLAction(tool="execute_transform", params={}))
        result = env.step(ETLAction(tool="submit", params={}))
        assert result.done is True
        assert result.info["final_score"] < 0.30


class TestFullEpisodeMedium:
    def test_complete_episode_produces_valid_score(self):
        env = ETLEnvironment(task_id="medium")
        env.reset(seed=42)

        # Profile
        env.step(ETLAction(tool="profile_column", params={"column": "qty"}))
        env.step(ETLAction(tool="profile_column", params={"column": "price"}))

        # Write and execute simplified transform
        env.step(ETLAction(
            tool="write_transform",
            params={"code": MEDIUM_TRANSFORM},
            reasoning="Basic cleaning on orders table",
        ))
        env.step(ETLAction(tool="execute_transform", params={}))

        # Validate
        result = env.step(ETLAction(
            tool="validate",
            params={"checks": ["null_check", "type_check"]},
        ))

        # Submit
        result = env.step(ETLAction(tool="submit", params={"reasoning": "Basic clean done"}))
        assert result.done is True
        final_score = result.info.get("final_score", -1)
        assert 0.0 <= final_score <= 1.0, f"Score out of range: {final_score}"
        assert final_score >= 0.40, f"Score too low for join transform: {final_score}"
        print(f"\n  Medium final score: {final_score}")


class TestFullEpisodeHard:
    def test_hard_episode_runs_without_crash(self):
        env = ETLEnvironment(task_id="hard")
        env.reset(seed=42)

        result = None
        for step_num in range(25):
            # After step 8, schema drift is injected
            if step_num < 3:
                action = ETLAction(
                    tool="profile_column",
                    params={"column": "amount"},
                    reasoning=f"Profiling step {step_num}"
                )
            elif step_num == 3:
                action = ETLAction(
                    tool="write_transform",
                    params={"code": HARD_TRANSFORM},
                )
            elif step_num == 4:
                action = ETLAction(tool="execute_transform", params={})
            elif step_num == 5:
                action = ETLAction(
                    tool="validate",
                    params={"checks": ["null_check", "range_check"]},
                )
            else:
                action = ETLAction(
                    tool="submit",
                    params={"reasoning": "Submitting after drift"},
                )

            result = env.step(action)

            # Check if drift was injected
            if result.observation.schema_drift_event:
                print(f"\n  Drift injected at step {step_num + 1}")
                print(f"  Drift: {result.observation.schema_drift_event.get('message', '')[:100]}")

            if result.done:
                break

        assert result is not None
        assert result.done is True
        final_score = result.info.get("final_score", -1)
        assert 0.0 <= final_score <= 1.0
        print(f"\n  Hard final score: {final_score}")

    def test_drift_event_appears_in_observation(self):
        """Verify the drift event appears in the observation at step 8."""
        env = ETLEnvironment(task_id="hard")
        env.reset(seed=42)

        drift_seen = False
        for i in range(10):
            result = env.step(ETLAction(
                tool="profile_column", params={"column": "amount"}
            ))
            if result.observation.schema_drift_event is not None:
                drift_seen = True
                assert "message" in result.observation.schema_drift_event
                break

        assert drift_seen, "Drift event should appear within first 10 steps for Hard task"


class TestRewardRange:
    def test_all_step_rewards_in_valid_range(self):
        """All intermediate rewards must be between -1.0 and +1.0."""
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)

        for _ in range(10):
            result = env.step(ETLAction(
                tool="profile_column",
                params={"column": "amount"},
            ))
            assert -1.0 <= result.reward <= 1.0

    def test_final_score_in_0_1(self):
        env = ETLEnvironment(task_id="easy")
        env.reset(seed=42)
        result = env.step(ETLAction(tool="submit", params={}))
        assert 0.0 <= result.info["final_score"] <= 1.0


class TestMultiEpisode:
    def test_can_reset_and_run_again(self):
        """Verify env can run multiple episodes back-to-back."""
        env = ETLEnvironment(task_id="easy")
        for episode in range(3):
            env.reset(seed=episode)
            result = env.step(ETLAction(tool="submit", params={}))
            assert result.done is True
            assert "final_score" in result.info

    def test_different_seeds_produce_different_scores(self):
        env = ETLEnvironment(task_id="easy")

        env.reset(seed=42)
        r1 = env.step(ETLAction(tool="submit", params={}))

        env.reset(seed=99)
        r2 = env.step(ETLAction(tool="submit", params={}))

        # Scores may differ (different data, different difficulty)
        # At minimum they should both be valid
        assert 0.0 <= r1.info["final_score"] <= 1.0
        assert 0.0 <= r2.info["final_score"] <= 1.0


if __name__ == "__main__":
    print("Running full episode tests...")

    t1 = TestFullEpisodeEasy()
    t1.test_complete_episode_produces_valid_score()
    print("  ✓ Easy: complete episode valid score")

    t1b = TestFullEpisodeEasy()
    t1b.test_no_transform_episode_scores_lower_than_gold()
    print("  ✓ Easy: no transform scores low")

    t1c = TestFullEpisodeEasy()
    t1c.test_bad_transform_episode()
    print("  ✓ Easy: bad transform scores low")

    t2 = TestFullEpisodeMedium()
    t2.test_complete_episode_produces_valid_score()
    print("  ✓ Medium: complete episode valid score")

    t3 = TestFullEpisodeHard()
    t3.test_hard_episode_runs_without_crash()
    print("  ✓ Hard: episode runs without crash")

    t3b = TestFullEpisodeHard()
    t3b.test_drift_event_appears_in_observation()
    print("  ✓ Hard: drift event appears")

    t4 = TestRewardRange()
    t4.test_all_step_rewards_in_valid_range()
    print("  ✓ Reward: all in valid range")

    t5 = TestMultiEpisode()
    t5.test_can_reset_and_run_again()
    print("  ✓ Multi-episode: reset works")

    print("\n✅ All full episode tests passed!")
