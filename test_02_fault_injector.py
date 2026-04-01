"""
TEST 2 — Fault Injector
Run: python -m pytest tests/test_02_fault_injector.py -v
Tests that generate_easy/medium/hard produce valid, broken, and reproducible datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from environment.fault_injector import generate_easy, generate_medium, generate_hard


class TestGenerateEasy:
    def setup_method(self):
        self.df_broken, self.df_gold, self.schema, self.faults = generate_easy(seed=42)

    def test_returns_dataframes(self):
        assert isinstance(self.df_broken, pd.DataFrame)
        assert isinstance(self.df_gold, pd.DataFrame)

    def test_row_count_in_range(self):
        assert 500 <= len(self.df_broken) <= 700

    def test_required_columns_present(self):
        cols = {"order_id", "customer_id", "amount", "order_date", "status"}
        assert cols.issubset(set(self.df_broken.columns))

    def test_faults_are_planted(self):
        assert len(self.faults) >= 4
        assert "inconsistent_date_format" in self.faults
        assert "duplicate_primary_key" in self.faults

    def test_gold_has_fewer_rows_than_broken(self):
        # Gold removes nulls, dupes — should have fewer rows
        assert len(self.df_gold) <= len(self.df_broken)

    def test_gold_no_nulls_in_required_cols(self):
        assert self.df_gold["customer_id"].isna().sum() == 0
        assert self.df_gold["order_id"].isna().sum() == 0

    def test_gold_no_negative_amounts(self):
        assert (self.df_gold["amount"] < 0).sum() == 0

    def test_gold_amounts_capped(self):
        assert (self.df_gold["amount"] > 10_000).sum() == 0

    def test_gold_status_lowercase(self):
        assert (self.df_gold["status"] == self.df_gold["status"].str.lower()).all()

    def test_schema_target_keys(self):
        assert "order_id" in self.schema
        assert "amount" in self.schema
        assert self.schema["amount"]["min"] == 0

    def test_reproducible(self):
        df2, _, _, _ = generate_easy(seed=42)
        assert len(self.df_broken) == len(df2)

    def test_different_seeds_differ(self):
        df2, _, _, _ = generate_easy(seed=99)
        # Different seeds should produce different data
        assert not self.df_broken.equals(df2)


class TestGenerateMedium:
    def setup_method(self):
        self.tables, self.df_gold, self.schema, self.faults = generate_medium(seed=42)

    def test_returns_three_tables(self):
        assert set(self.tables.keys()) == {"orders", "customers", "products"}

    def test_all_tables_are_dataframes(self):
        for name, df in self.tables.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"

    def test_orders_has_required_cols(self):
        assert {"order_id", "customer_id", "product_id", "qty", "price"}.issubset(
            set(self.tables["orders"].columns)
        )

    def test_customers_has_required_cols(self):
        assert {"customer_id", "name", "region", "tier"}.issubset(
            set(self.tables["customers"].columns)
        )

    def test_gold_has_margin_pct(self):
        assert "margin_pct" in self.df_gold.columns

    def test_gold_margin_positive(self):
        assert (self.df_gold["margin_pct"] > 0).all()

    def test_faults_planted(self):
        assert "orphaned_customer_fk" in self.faults
        assert "region_typos" in self.faults


class TestGenerateHard:
    def setup_method(self):
        self.df_broken, self.df_gold, self.schema, self.faults, self.drift = generate_hard(seed=42)

    def test_returns_dataframe(self):
        assert isinstance(self.df_broken, pd.DataFrame)

    def test_drift_event_has_required_keys(self):
        assert "rename" in self.drift
        assert "new_col" in self.drift
        assert "type_chg" in self.drift
        assert "step_injected" in self.drift
        assert "message" in self.drift

    def test_drift_injected_at_step_8(self):
        assert self.drift["step_injected"] == 8

    def test_drift_rename_is_tuple(self):
        old, new = self.drift["rename"]
        assert isinstance(old, str)
        assert isinstance(new, str)
        assert old != new

    def test_drift_message_is_string(self):
        assert isinstance(self.drift["message"], str)
        assert len(self.drift["message"]) > 10


if __name__ == "__main__":
    print("Running fault injector tests...")

    t = TestGenerateEasy()
    t.setup_method()
    tests = [
        ("returns dataframes", t.test_returns_dataframes),
        ("row count in range", t.test_row_count_in_range),
        ("required columns", t.test_required_columns_present),
        ("faults planted", t.test_faults_are_planted),
        ("gold fewer rows", t.test_gold_has_fewer_rows_than_broken),
        ("gold no nulls", t.test_gold_no_nulls_in_required_cols),
        ("gold no negatives", t.test_gold_no_negative_amounts),
        ("gold amounts capped", t.test_gold_amounts_capped),
        ("gold status lowercase", t.test_gold_status_lowercase),
        ("schema target", t.test_schema_target_keys),
        ("reproducible", t.test_reproducible),
        ("diff seeds differ", t.test_different_seeds_differ),
    ]
    for name, fn in tests:
        fn(); print(f"  ✓ Easy: {name}")

    t2 = TestGenerateMedium()
    t2.setup_method()
    t2.test_returns_three_tables(); print("  ✓ Medium: three tables")
    t2.test_gold_has_margin_pct(); print("  ✓ Medium: gold has margin_pct")
    t2.test_gold_margin_positive(); print("  ✓ Medium: margin positive")

    t3 = TestGenerateHard()
    t3.setup_method()
    t3.test_drift_event_has_required_keys(); print("  ✓ Hard: drift event keys")
    t3.test_drift_injected_at_step_8(); print("  ✓ Hard: drift at step 8")

    print("\n✅ All fault injector tests passed!")
