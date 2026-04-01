"""
TEST 3 — Grader
Run: python -m pytest tests/test_03_grader.py -v
Tests that all graders return valid 0.0–1.0 scores and are deterministic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from environment.grader import grade_easy, grade_medium, grade_hard, run_validation_checks
from environment.fault_injector import generate_easy, generate_medium, generate_hard


def _in_range(score, lo=0.0, hi=1.0):
    return lo <= score <= hi


class TestGradeEasy:
    def setup_method(self):
        self.df_broken, self.df_gold, self.schema, self.faults = generate_easy(seed=42)

    def test_gold_scores_near_1(self):
        score, checks = grade_easy(self.df_gold, self.df_gold, self.schema)
        assert score >= 0.80, f"Gold should score high, got {score}"
        assert all(_in_range(v) for v in checks.values())

    def test_broken_scores_lower_than_gold(self):
        gold_score, _ = grade_easy(self.df_gold, self.df_gold, self.schema)
        broken_score, _ = grade_easy(self.df_broken, self.df_gold, self.schema)
        assert broken_score < gold_score

    def test_empty_df_scores_low(self):
        df_empty = pd.DataFrame(columns=self.df_gold.columns)
        score, checks = grade_easy(df_empty, self.df_gold, self.schema)
        assert score < 0.5

    def test_scores_in_range(self):
        score, checks = grade_easy(self.df_broken, self.df_gold, self.schema)
        assert _in_range(score)
        for k, v in checks.items():
            assert _in_range(v), f"Check '{k}' out of range: {v}"

    def test_returns_dict_with_expected_keys(self):
        _, checks = grade_easy(self.df_gold, self.df_gold, self.schema)
        expected = {"null_check", "type_check", "range_check", "uniqueness_check", "row_count_match", "schema_match"}
        assert expected.issubset(set(checks.keys()))

    def test_deterministic(self):
        s1, c1 = grade_easy(self.df_gold, self.df_gold, self.schema)
        s2, c2 = grade_easy(self.df_gold, self.df_gold, self.schema)
        assert s1 == s2
        assert c1 == c2


class TestGradeMedium:
    def setup_method(self):
        self.tables, self.df_gold, self.schema, self.faults = generate_medium(seed=42)

    def test_gold_scores_high(self):
        score, checks = grade_medium(self.df_gold, self.df_gold, self.schema, self.tables)
        assert score >= 0.70, f"Gold should score high, got {score}"

    def test_scores_in_range(self):
        score, checks = grade_medium(self.df_gold, self.df_gold, self.schema, self.tables)
        assert _in_range(score)
        for k, v in checks.items():
            assert _in_range(v), f"Check '{k}' out of range: {v}"

    def test_missing_margin_col_scores_low(self):
        df_bad = self.df_gold.drop(columns=["margin_pct"], errors="ignore")
        score, checks = grade_medium(df_bad, self.df_gold, self.schema, self.tables)
        assert checks.get("business_rule_margin", 0) == 0.0

    def test_medium_specific_checks_present(self):
        _, checks = grade_medium(self.df_gold, self.df_gold, self.schema, self.tables)
        assert "referential_integrity" in checks
        assert "business_rule_margin" in checks
        assert "join_completeness" in checks


class TestGradeHard:
    def setup_method(self):
        _, self.df_gold, self.schema, self.faults, self.drift = generate_hard(seed=42)

    def test_scores_in_range(self):
        score, checks = grade_hard(
            df_agent=self.df_gold,
            df_gold=self.df_gold,
            schema=self.schema,
            drift_event=self.drift,
            steps_wasted_after_drift=0,
            drift_detected=True,
            pre_drift_rows=len(self.df_gold),
        )
        assert _in_range(score)
        for k, v in checks.items():
            assert _in_range(v), f"Check '{k}' out of range: {v}"

    def test_drift_detected_bonus(self):
        score_detected, _ = grade_hard(
            df_agent=self.df_gold, df_gold=self.df_gold, schema=self.schema,
            drift_event=self.drift, steps_wasted_after_drift=0,
            drift_detected=True, pre_drift_rows=len(self.df_gold),
        )
        score_missed, _ = grade_hard(
            df_agent=self.df_gold, df_gold=self.df_gold, schema=self.schema,
            drift_event=self.drift, steps_wasted_after_drift=0,
            drift_detected=False, pre_drift_rows=len(self.df_gold),
        )
        assert score_detected > score_missed

    def test_wasted_steps_penalty(self):
        score_fast, _ = grade_hard(
            df_agent=self.df_gold, df_gold=self.df_gold, schema=self.schema,
            drift_event=self.drift, steps_wasted_after_drift=0,
            drift_detected=True, pre_drift_rows=len(self.df_gold),
        )
        score_slow, _ = grade_hard(
            df_agent=self.df_gold, df_gold=self.df_gold, schema=self.schema,
            drift_event=self.drift, steps_wasted_after_drift=5,
            drift_detected=True, pre_drift_rows=len(self.df_gold),
        )
        assert score_fast > score_slow


class TestRunValidationChecks:
    def setup_method(self):
        self.df, self.df_gold, self.schema, _ = generate_easy(seed=42)

    def test_basic_checks_return_floats(self):
        results = run_validation_checks(
            df=self.df_gold,
            schema=self.schema,
            checks=["null_check", "type_check", "range_check"],
        )
        assert "null_check" in results
        assert all(isinstance(v, float) for v in results.values())

    def test_unknown_check_returns_zero(self):
        results = run_validation_checks(
            df=self.df_gold,
            schema=self.schema,
            checks=["nonexistent_check"],
        )
        assert results.get("nonexistent_check", 0.0) == 0.0

    def test_empty_checks_list_returns_empty(self):
        results = run_validation_checks(df=self.df_gold, schema=self.schema, checks=[])
        assert results == {}


if __name__ == "__main__":
    print("Running grader tests...")

    t = TestGradeEasy()
    t.setup_method()
    t.test_gold_scores_near_1(); print("  ✓ Easy: gold scores near 1")
    t.test_broken_scores_lower_than_gold(); print("  ✓ Easy: broken < gold")
    t.test_empty_df_scores_low(); print("  ✓ Easy: empty df scores low")
    t.test_scores_in_range(); print("  ✓ Easy: scores in range")
    t.test_returns_dict_with_expected_keys(); print("  ✓ Easy: check keys")
    t.test_deterministic(); print("  ✓ Easy: deterministic")

    t2 = TestGradeMedium()
    t2.setup_method()
    t2.test_gold_scores_high(); print("  ✓ Medium: gold scores high")
    t2.test_scores_in_range(); print("  ✓ Medium: scores in range")
    t2.test_medium_specific_checks_present(); print("  ✓ Medium: specific checks")

    t3 = TestGradeHard()
    t3.setup_method()
    t3.test_scores_in_range(); print("  ✓ Hard: scores in range")
    t3.test_drift_detected_bonus(); print("  ✓ Hard: drift detected bonus")
    t3.test_wasted_steps_penalty(); print("  ✓ Hard: wasted steps penalty")

    t4 = TestRunValidationChecks()
    t4.setup_method()
    t4.test_basic_checks_return_floats(); print("  ✓ Validation: returns floats")
    t4.test_unknown_check_returns_zero(); print("  ✓ Validation: unknown check = 0")

    print("\n✅ All grader tests passed!")
