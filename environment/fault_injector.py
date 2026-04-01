"""
fault_injector.py — Generates deliberately broken datasets for each task.
Physics engine = pure pandas. No external simulator.
Every reset() calls generate_easy/medium/hard() to produce fresh broken data.

Fixes:
  - Removed %-d strftime (Linux-only); use lstrip("0") instead
  - Replaced deprecated infer_datetime_format=True with dateutil parser
  - Added missing List import
  - Boolean mask indexing replaced with list-comprehension for safety
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  EASY TASK
# ─────────────────────────────────────────────

def generate_easy(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List]:
    """Returns (df_broken, df_gold, schema_target, faults_planted)."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n           = rng.randint(500, 700)
    order_ids   = list(range(1001, 1001 + n))
    customer_ids = [f"C_{rng.randint(10, 99):03d}" for _ in range(n)]
    amounts     = np_rng.uniform(10, 5000, n).round(2).tolist()
    statuses    = [rng.choice(["completed", "pending", "cancelled"]) for _ in range(n)]
    base_dates  = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "order_id":    order_ids,
        "customer_id": customer_ids,
        "amount":      amounts,
        "order_date":  base_dates.strftime("%Y-%m-%d"),
        "status":      statuses,
    })

    faults_planted: List[str] = []

    # Fault 1: Inconsistent date format — natural language (cross-platform safe)
    nat_lang = [d.strftime("%d %b %Y").lstrip("0") for d in base_dates]
    indices = [i for i, m in enumerate(np_rng.random(n) < 0.45) if m]
    for i in indices:
        df.at[i, "order_date"] = nat_lang[i]
    faults_planted.append("inconsistent_date_format")

    # Fault 2: Duplicate primary key (~3 rows)
    dupe_idx = rng.sample(range(n // 2), max(3, n // 20))
    for pos, idx in enumerate(dupe_idx):
        df.at[n - 1 - pos, "order_id"] = df.at[idx, "order_id"]
    faults_planted.append("duplicate_primary_key")

    # Fault 3: Null customer_id (~2%)
    for i in [i for i, m in enumerate(np_rng.random(n) < 0.20) if m]:
        df.at[i, "customer_id"] = None
    faults_planted.append("null_fk")

    # Fault 4: Negative amounts (~3%)
    for i in [i for i, m in enumerate(np_rng.random(n) < 0.12) if m]:
        df.at[i, "amount"] = -abs(float(df.at[i, "amount"]))
    faults_planted.append("negative_amount")

    # Fault 5: Outlier amounts (3 rows → 1,500,000)
    for idx in rng.sample(range(n), min(n, max(3, n//15))):
        df.at[idx, "amount"] = 1_500_000.0
    faults_planted.append("outlier_amount")

    # Fault 6: Status case inconsistency
    for i in [i for i, m in enumerate(np_rng.random(n) < 0.35) if m]:
        df.at[i, "status"] = str(df.at[i, "status"]).upper()
    for i in [i for i, m in enumerate(np_rng.random(n) < 0.10) if m]:
        df.at[i, "status"] = str(df.at[i, "status"]).title()
    faults_planted.append("status_case_inconsistency")

    # Gold standard — what a perfect agent produces
    df_gold = df.copy()
    # pandas 3.x requires format="mixed" for columns with multiple date formats
    df_gold["order_date"] = pd.to_datetime(
        df_gold["order_date"], format="mixed", dayfirst=False, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    df_gold = df_gold.drop_duplicates(subset=["order_id"], keep="first")
    df_gold = df_gold[df_gold["customer_id"].notna()]
    df_gold = df_gold[df_gold["amount"] >= 0]
    df_gold["amount"] = df_gold["amount"].clip(upper=10_000)
    df_gold["status"] = df_gold["status"].str.lower()
    df_gold = df_gold.reset_index(drop=True)

    schema_target: Dict[str, Any] = {
        "order_id":    {"type": "int",    "unique": True,  "nullable": False},
        "customer_id": {"type": "string", "nullable": False, "pattern": r"C_\d+"},
        "amount":      {"type": "float",  "min": 0,        "max": 10_000},
        "order_date":  {"type": "date",   "format": "%Y-%m-%d"},
        "status":      {"type": "string", "values": ["completed", "pending", "cancelled"]},
    }
    return df, df_gold, schema_target, faults_planted


# ─────────────────────────────────────────────
#  MEDIUM TASK
# ─────────────────────────────────────────────

def generate_medium(seed: int = 42) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict, List]:
    """Returns (tables_dict, df_gold_fact, schema_target, faults_planted)."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n_orders    = rng.randint(300, 500)
    n_customers = 80
    n_products  = 40

    regions_clean   = ["North East", "South East", "North West", "South West", "Midlands"]
    tiers           = ["gold", "silver", "bronze"]
    categories_clean = ["electronics", "clothing", "home", "sports", "food"]

    customers = pd.DataFrame({
        "customer_id": [f"CU{i:03d}" for i in range(n_customers)],
        "name":        [f"Customer {i}" for i in range(n_customers)],
        "region":      [rng.choice(regions_clean) for _ in range(n_customers)],
        "tier":        [rng.choice(tiers)          for _ in range(n_customers)],
    })

    cost_prices  = np_rng.uniform(5, 200, n_products).round(2)
    sale_prices  = cost_prices + np_rng.uniform(5, 50, n_products).round(2)
    products = pd.DataFrame({
        "product_id": [f"P{i:04d}" for i in range(n_products)],
        "category":   [rng.choice(categories_clean) for _ in range(n_products)],
        "cost_price": cost_prices,
        "sale_price": sale_prices,
    })

    valid_cids = customers["customer_id"].tolist()
    valid_pids = products["product_id"].tolist()
    orders = pd.DataFrame({
        "order_id":    range(5001, 5001 + n_orders),
        "customer_id": [rng.choice(valid_cids) for _ in range(n_orders)],
        "product_id":  [rng.choice(valid_pids) for _ in range(n_orders)],
        "qty":         np_rng.integers(1, 10, n_orders),
        "price":       np_rng.uniform(20, 500, n_orders).round(2),
    })

    faults_planted: List[str] = []

    # Orphaned customer FKs
    for i in [i for i, m in enumerate(np_rng.random(n_orders) < 0.05) if m]:
        orders.at[i, "customer_id"] = "CU999"
    faults_planted.append("orphaned_customer_fk")

    # Orphaned product FKs
    for i in [i for i, m in enumerate(np_rng.random(n_orders) < 0.04) if m]:
        orders.at[i, "product_id"] = "P9999"
    faults_planted.append("orphaned_product_fk")

    # Impossible margin (price < cost)
    for i in [i for i, m in enumerate(np_rng.random(n_orders) < 0.06) if m]:
        orders.at[i, "price"] = 2.0
    faults_planted.append("impossible_margin")

    # Region typos in customers
    typo_map = {"North East": "Nort East", "South West": "SouthWest", "Midlands": "midlands"}
    for i in [i for i, m in enumerate(np_rng.random(n_customers) < 0.20) if m]:
        r = customers.at[i, "region"]
        customers.at[i, "region"] = typo_map.get(r, r)
    faults_planted.append("region_typos")

    # Category ALL_CAPS
    for i in [i for i, m in enumerate(np_rng.random(n_products) < 0.30) if m]:
        products.at[i, "category"] = str(products.at[i, "category"]).upper()
    faults_planted.append("category_case_inconsistency")

    tables = {"orders": orders, "customers": customers, "products": products}

    # Build gold fact table from clean versions
    df_m = orders.merge(customers, on="customer_id", how="inner")
    df_m = df_m.merge(products, on="product_id", how="inner")
    df_gold = df_m.copy()
    df_gold["margin_pct"] = ((df_gold["price"] - df_gold["cost_price"]) / df_gold["price"]).round(4)
    df_gold = df_gold[df_gold["margin_pct"] > 0]
    df_gold["region"]   = df_gold["region"].str.strip().str.title()
    df_gold["category"] = df_gold["category"].str.lower()
    output_cols = ["order_id", "customer_id", "name", "region", "tier",
                   "category", "qty", "price", "cost_price", "margin_pct"]
    df_gold = df_gold[output_cols].reset_index(drop=True)

    schema_target: Dict[str, Any] = {
        "order_id":    {"type": "int",   "unique": True},
        "customer_id": {"type": "string"},
        "name":        {"type": "string"},
        "region":      {"type": "string", "values": ["North East", "South East", "North West", "South West", "Midlands"]},
        "tier":        {"type": "string", "values": ["gold", "silver", "bronze"]},
        "category":    {"type": "string", "lowercase": True},
        "qty":         {"type": "int",   "min": 1},
        "price":       {"type": "float", "min": 0},
        "cost_price":  {"type": "float", "min": 0},
        "margin_pct":  {"type": "float", "min": 0, "description": "(price-cost)/price"},
    }
    return tables, df_gold, schema_target, faults_planted


# ─────────────────────────────────────────────
#  HARD TASK — Schema drift mid-episode
# ─────────────────────────────────────────────

DRIFT_TEMPLATES: List[Dict] = [
    {"rename": ("order_date",  "created_at"),   "new_col": ("currency_code",  "USD"),    "type_chg": ("amount",  "DECIMAL(10,2)")},
    {"rename": ("customer_id", "client_id"),    "new_col": ("source_system",  "LEGACY"), "type_chg": ("qty",     "SMALLINT")},
    {"rename": ("status",      "order_status"), "new_col": ("processed_flag", "0"),      "type_chg": ("amount",  "NUMERIC(12,4)")},
]


def generate_hard(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List, Dict]:
    """Returns (df_broken, df_gold, schema_target, faults_planted, drift_event)."""
    df_broken, df_gold_pre, schema_target, faults = generate_easy(seed)

    rng   = random.Random(seed)
    drift = rng.choice(DRIFT_TEMPLATES)
    old_col, new_col_name = drift["rename"]
    new_col, default_val  = drift["new_col"]

    drift_event: Dict[str, Any] = {
        "step_injected": 8,
        "rename":        drift["rename"],
        "new_col":       drift["new_col"],
        "type_chg":      drift["type_chg"],
        "message": (
            f"⚠️ Schema change detected at step 8:\n"
            f"  - Column '{old_col}' renamed to '{new_col_name}'\n"
            f"  - New required column '{new_col}' added (NOT NULL, default '{default_val}')\n"
            f"  - Column '{drift['type_chg'][0]}' type changed to {drift['type_chg'][1]}"
        ),
    }

    # Post-drift gold: apply rename + new required column
    df_gold = df_gold_pre.rename(columns={old_col: new_col_name}).copy()
    df_gold[new_col] = default_val

    return df_broken, df_gold, schema_target, faults, drift_event
