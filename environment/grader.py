"""
grader.py — Deterministic graders for all 3 tasks.
Pure pandas. No LLM judge. Every score is reproducible.
Returns a dict of check_name → float (0.0–1.0) and a final weighted score.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────

def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _null_check(df: pd.DataFrame, required_cols: List[str]) -> float:
    """Fraction of required columns with zero nulls."""
    if not required_cols:
        return 1.0
    if len(df) == 0:
        return 0.0  # empty df has no valid rows
    scores = []
    for col in required_cols:
        if col not in df.columns:
            scores.append(0.0)
        else:
            null_rate = df[col].isna().mean()
            null_rate = 0.0 if pd.isna(null_rate) else null_rate
            scores.append(1.0 - null_rate)
    return float(np.mean(scores)) if scores else 1.0


def _type_check(df: pd.DataFrame, schema: Dict[str, Dict]) -> float:
    """Fraction of columns whose dtype matches the target schema type."""
    # pandas 3.x uses "str" dtype for string columns (older versions used "object")
    type_map = {
        "int":    ["int8","int16","int32","int64","Int8","Int16","Int32","Int64"],
        "float":  ["float16","float32","float64","Float32","Float64"],
        "string": ["object","string","str"],          # pandas 3.x = "str"
        "date":   ["object","datetime64","str"],      # allow string date representation
    }
    scores = []
    for col, spec in schema.items():
        if col not in df.columns:
            scores.append(0.0)
            continue
        expected = spec.get("type", "string")
        actual = str(df[col].dtype)
        allowed = type_map.get(expected, ["object"])
        scores.append(1.0 if any(actual.startswith(a) for a in allowed) else 0.0)
    return float(np.mean(scores)) if scores else 1.0


def _range_check(df: pd.DataFrame, schema: Dict[str, Dict]) -> float:
    """Fraction of numeric columns where all values are in [min, max]."""
    if len(df) == 0:
        return 0.0
    scores = []
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        if "min" not in spec and "max" not in spec:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        in_range = pd.Series([True] * len(df), index=df.index)
        if "min" in spec:
            in_range &= series >= spec["min"]
        if "max" in spec:
            in_range &= series <= spec["max"]
        val = in_range.mean()
        scores.append(0.0 if pd.isna(val) else float(val))
    return float(np.mean(scores)) if scores else 1.0


def _uniqueness_check(df: pd.DataFrame, unique_cols: List[str]) -> float:
    """Fraction of designated unique columns that have no duplicates."""
    if len(df) == 0:
        return 0.0
    scores = []
    for col in unique_cols:
        if col not in df.columns:
            scores.append(0.0)
        else:
            dup_rate = df[col].duplicated().mean()
            dup_rate = 0.0 if pd.isna(dup_rate) else float(dup_rate)
            scores.append(1.0 - dup_rate)
    return float(np.mean(scores)) if scores else 1.0


def _row_count_match(df_agent: pd.DataFrame, df_gold: pd.DataFrame, tolerance: float = 0.05) -> float:
    """
    Score 1.0 if row count is within tolerance % of gold, else partial credit.
    """
    if len(df_gold) == 0:
        return 1.0 if len(df_agent) == 0 else 0.0
    if len(df_agent) == 0:
        return 0.0
    diff = abs(len(df_agent) - len(df_gold)) / len(df_gold)
    if diff <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (diff - tolerance) * 5)


def _schema_match(df_agent: pd.DataFrame, schema: Dict[str, Dict]) -> float:
    """Fraction of target columns present in agent output."""
    target_cols = set(schema.keys())
    agent_cols = set(df_agent.columns)
    if not target_cols:
        return 1.0
    if len(df_agent) == 0:
        return 0.0  # empty df doesn't count as schema match
    return len(target_cols & agent_cols) / len(target_cols)


def _values_check(df: pd.DataFrame, schema: Dict[str, Dict]) -> float:
    """Fraction of enum columns where all values are in the allowed set."""
    if len(df) == 0:
        return 0.0
    scores = []
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        if "values" not in spec:
            continue
        allowed = set(spec["values"])
        actual = set(df[col].dropna().unique())
        invalid = actual - allowed
        invalid_rate = len(df[df[col].isin(invalid)]) / max(len(df), 1)
        scores.append(1.0 - invalid_rate)
    return float(np.mean(scores)) if scores else 1.0


# ─────────────────────────────────────────────
#  EASY GRADER
# ─────────────────────────────────────────────

EASY_WEIGHTS = {
    "null_check":       0.20,
    "type_check":       0.20,
    "range_check":      0.20,
    "uniqueness_check": 0.20,
    "row_count_match":  0.10,
    "schema_match":     0.10,
}

def grade_easy(df_agent: pd.DataFrame, df_gold: pd.DataFrame, schema: Dict) -> Tuple[float, Dict[str, float]]:
    required_cols = [c for c, s in schema.items() if not s.get("nullable", True)]
    unique_cols   = [c for c, s in schema.items() if s.get("unique", False)]

    checks = {
        "null_check":       _null_check(df_agent, required_cols or list(schema.keys())),
        "type_check":       _type_check(df_agent, schema),
        "range_check":      _range_check(df_agent, schema),
        "uniqueness_check": _uniqueness_check(df_agent, unique_cols or ["order_id"]),
        "row_count_match":  _row_count_match(df_agent, df_gold),
        "schema_match":     _schema_match(df_agent, schema),
    }

    final = sum(checks[k] * EASY_WEIGHTS[k] for k in EASY_WEIGHTS)
    return round(final, 4), checks


# ─────────────────────────────────────────────
#  MEDIUM GRADER
# ─────────────────────────────────────────────

MEDIUM_WEIGHTS = {
    "null_check":             0.10,
    "type_check":             0.10,
    "range_check":            0.10,
    "uniqueness_check":       0.10,
    "referential_integrity":  0.15,
    "business_rule_margin":   0.15,
    "region_standardized":    0.10,
    "category_normalized":    0.05,
    "join_completeness":      0.10,
    "column_derivation":      0.05,
}

APPROVED_REGIONS = {"North East", "South East", "North West", "South West", "Midlands"}

def grade_medium(
    df_agent: pd.DataFrame,
    df_gold: pd.DataFrame,
    schema: Dict,
    tables: Dict[str, pd.DataFrame],
) -> Tuple[float, Dict[str, float]]:
    """
    tables = {"orders": df, "customers": df, "products": df}
    """
    valid_cids = set(tables["customers"]["customer_id"].unique())
    valid_pids = set(tables["products"]["product_id"].unique())

    # Referential integrity — no orphaned FKs
    if "customer_id" in df_agent.columns:
        orphan_c = df_agent["customer_id"].isin(valid_cids)
        ref_score = orphan_c.mean()
    else:
        ref_score = 0.0

    # Business rule: margin_pct > 0
    if "margin_pct" in df_agent.columns:
        biz_score = (pd.to_numeric(df_agent["margin_pct"], errors="coerce") > 0).mean()
    else:
        biz_score = 0.0

    # Region standardized
    if "region" in df_agent.columns:
        reg_score = df_agent["region"].isin(APPROVED_REGIONS).mean()
    else:
        reg_score = 0.0

    # Category normalized (lowercase)
    if "category" in df_agent.columns:
        cat_score = (df_agent["category"] == df_agent["category"].str.lower()).mean()
    else:
        cat_score = 0.0

    # Join completeness — % of valid orders that made it through
    n_valid_orders = len(tables["orders"][
        tables["orders"]["customer_id"].isin(valid_cids) &
        tables["orders"]["product_id"].isin(valid_pids)
    ])
    join_score = min(len(df_agent) / max(n_valid_orders, 1), 1.0)

    # Column derivation: margin_pct computed correctly
    if "margin_pct" in df_agent.columns and "price" in df_agent.columns and "cost_price" in df_agent.columns:
        computed = ((df_agent["price"] - df_agent["cost_price"]) / df_agent["price"]).round(2)
        actual = pd.to_numeric(df_agent["margin_pct"], errors="coerce").round(2)
        deriv_score = (computed == actual).mean()
    else:
        deriv_score = 0.0

    checks = {
        "null_check":            _null_check(df_agent, ["order_id", "customer_id"]),
        "type_check":            _type_check(df_agent, schema),
        "range_check":           _range_check(df_agent, schema),
        "uniqueness_check":      _uniqueness_check(df_agent, ["order_id"]),
        "referential_integrity": _safe_float(ref_score),
        "business_rule_margin":  _safe_float(biz_score),
        "region_standardized":   _safe_float(reg_score),
        "category_normalized":   _safe_float(cat_score),
        "join_completeness":     _safe_float(join_score),
        "column_derivation":     _safe_float(deriv_score),
    }

    final = sum(checks[k] * MEDIUM_WEIGHTS[k] for k in MEDIUM_WEIGHTS)
    return round(final, 4), checks


# ─────────────────────────────────────────────
#  HARD GRADER
# ─────────────────────────────────────────────

HARD_WEIGHTS = {
    "null_check":              0.10,
    "schema_drift_detected":   0.15,
    "recovery_speed":          0.15,
    "incremental_correctness": 0.20,
    "unified_output_schema":   0.15,
    "default_value_applied":   0.15,
    "no_data_loss":            0.10,
}

def grade_hard(
    df_agent: pd.DataFrame,
    df_gold: pd.DataFrame,
    schema: Dict,
    drift_event: Dict,
    steps_wasted_after_drift: int,
    drift_detected: bool,
    pre_drift_rows: int,
) -> Tuple[float, Dict[str, float]]:
    new_col, default_val = drift_event["new_col"]
    old_col, new_col_name = drift_event["rename"]

    # Schema drift detected
    drift_score = 1.0 if drift_detected else 0.0

    # Recovery speed: 1 - (wasted_steps / 5), min 0
    recovery = max(0.0, 1.0 - steps_wasted_after_drift / 5)

    # Incremental correctness: pre-drift rows present and not reprocessed wrongly
    # Proxy: check that row count >= pre_drift_rows (didn't lose prior work)
    incr_score = 1.0 if len(df_agent) >= pre_drift_rows else len(df_agent) / max(pre_drift_rows, 1)

    # Unified output schema: renamed column present, old column absent
    has_new_name = new_col_name in df_agent.columns
    lacks_old_name = old_col not in df_agent.columns
    unified = 1.0 if (has_new_name and lacks_old_name) else 0.5 if has_new_name else 0.0

    # Default value applied (new required column filled)
    if new_col in df_agent.columns:
        filled = (df_agent[new_col].notna() & (df_agent[new_col] != "")).mean()
    else:
        filled = 0.0

    # No data loss
    no_loss = _row_count_match(df_agent, df_gold, tolerance=0.05)

    checks = {
        "null_check":              _null_check(df_agent, [new_col_name]),
        "schema_drift_detected":   drift_score,
        "recovery_speed":          round(recovery, 4),
        "incremental_correctness": round(incr_score, 4),
        "unified_output_schema":   unified,
        "default_value_applied":   round(float(filled), 4),
        "no_data_loss":            no_loss,
    }

    final = sum(checks[k] * HARD_WEIGHTS[k] for k in HARD_WEIGHTS)
    return round(final, 4), checks


# ─────────────────────────────────────────────
#  VALIDATE — live checks during episode
# ─────────────────────────────────────────────

def run_validation_checks(
    df: pd.DataFrame,
    schema: Dict,
    checks: List[str],
    extra: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Run a subset of checks during an episode (before submit).
    Returns partial scores the agent can observe.
    extra = {"tables": {...}} for medium task cross-table checks.
    """
    results = {}
    extra = extra or {}

    check_map = {
        "null_check":        lambda: _null_check(df, list(schema.keys())),
        "type_check":        lambda: _type_check(df, schema),
        "range_check":       lambda: _range_check(df, schema),
        "uniqueness_check":  lambda: _uniqueness_check(df, [c for c,s in schema.items() if s.get("unique")]),
        "values_check":      lambda: _values_check(df, schema),
        "schema_match":      lambda: _schema_match(df, schema),
        "referential_integrity": lambda: _null_check(df, ["customer_id"]),  # simplified live check
        "business_rule_margin":  lambda: (
            (pd.to_numeric(df["margin_pct"], errors="coerce") > 0).mean()
            if "margin_pct" in df.columns else 0.0
        ),
        "region_standardized": lambda: (
            df["region"].isin(APPROVED_REGIONS).mean() if "region" in df.columns else 0.0
        ),
        "category_normalized": lambda: (
            (df["category"] == df["category"].str.lower()).mean() if "category" in df.columns else 0.0
        ),
    }

    for check in checks:
        if check in check_map:
            try:
                results[check] = round(float(check_map[check]()), 4)
            except Exception as e:
                results[check] = 0.0
        else:
            results[check] = 0.0

    return results
