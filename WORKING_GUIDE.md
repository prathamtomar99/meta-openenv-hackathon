# ETL Pipeline Agent Environment — Complete Working Guide

## 📋 Project Summary

You've built an **OpenEnv environment** where LLM agents learn to act as data engineers. The environment provides broken datasets with injected faults, and the agent must diagnose issues, write pandas transformations, validate quality, and submit clean output.

### Key Innovation

Unlike static benchmarks (that only measure failure), your environment **trains agents** with dense step-by-step rewards, tight feedback loops, and randomized difficulty — making it viable for GRPO training.

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd /Users/prathamtomar/Desktop/HRIT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import pandas; import fastapi; import pydantic; print('✓ All dependencies installed')"
```

---

## 🧪 Testing (Run in Order)

Each test file is independent and tests one layer. Run them sequentially to verify correctness:

### Test 1: Pydantic Models (Data Serialization)

```bash
python tests/test_01_models.py
```

**What it tests:**

- ETLAction, ETLObservation, ETLState serialization
- JSON round-trip encoding/decoding
- Model validation

**Expected output:**

```
test_action_serialization ✓
test_observation_serialization ✓
test_state_serialization ✓
test_step_result_serialization ✓
All model tests passed!
```

---

### Test 2: Fault Injector (Data Generation)

```bash
python tests/test_02_fault_injector.py
```

**What it tests:**

- Easy/Medium/Hard dataset generation
- Fault distribution (nulls, foreign key violations, value errors, etc.)
- Gold dataset quality

**Expected output:**

```
Easy dataset: 100 rows, 5 columns
- Null faults: 8-12%
- Min faults: 2-4%
- FK faults: 2-4%
✓ Easy generation passed

Medium dataset: 300 rows across 3 tables
- Customers: 78 rows
- Products: 22 rows
- Orders: 200 rows
✓ Medium generation passed

Hard dataset: Same as Medium, will inject drift at step 8 during episode
✓ Hard generation passed
```

---

### Test 3: Grader (Quality Scoring)

```bash
python tests/test_03_grader.py
```

**What it tests:**

- Null check scoring (0.0–1.0)
- Schema match detection
- Row count preservation
- Data integrity checks
- Final episode scoring

**Expected output:**

```
Testing null_check...
  Clean data → 1.0 ✓
  50% nulls → 0.5 ✓

Testing schema_match...
  Columns match → 1.0 ✓
  Missing column → 0.0 ✓

Testing row_count_match...
  Same rows → 1.0 ✓
  50% rows dropped → 0.5 ✓

Testing final_episode_score...
  Gold data → 1.0 ✓
  Updated data → 0.87–0.93 ✓

All grader tests passed! ✓
```

---

### Test 4: Environment Loop (API Contract)

```bash
python tests/test_04_env_reset_step.py
```

**What it tests:**

- reset() → returns valid observation + budget
- step() → all 8 tools work correctly
- Budget exhaustion behavior
- State isolation between episodes

**Expected output:**

```
Testing reset()...
  Observation keys valid: ✓
  Budget = 15 steps: ✓
  Schema target provided: ✓

Testing profile_column action...
  Returns null_rate, dtype, stats: ✓
  Immediate reward +0.05: ✓

Testing write_transform...
  Code stored (not executed): ✓
  No immediate execution: ✓

Testing execute_transform...
  Code executed against df: ✓
  If syntax error → reward -0.05, error message returned: ✓

Testing validate action...
  5 checks run: ✓
  Each check score 0.0–1.0: ✓
  Reward = 0.05 per check: ✓

Testing budget depletion...
  After 15 steps → done=True: ✓
  Next reset() resets to 15: ✓

All environment loop tests passed! ✓
```

---

### Test 5: Full Episode (End-to-End)

```bash
python tests/test_05_full_episode.py
```

**What it tests:**

- Complete Easy episode (agent transforms broken → clean)
- Complete Medium episode (3-table join with business rules)
- Complete Hard episode (agent must detect schema drift at step 8)
- Score distribution verification

**Expected output:**

```
=== EASY EPISODE ===
Step 1: profile_column('amount') → null_rate=0.08, reward=+0.05
Step 2: write_transform(...) → stored, reward=0.0
Step 3: execute_transform() → 10 rows cleaned, reward=+0.10
Step 4: validate() → 8 checks, 7 pass, reward=+0.35
Step 5: submit() → Episode complete
Final score: 0.92 ✓

=== MEDIUM EPISODE ===
(3-table join workflow)
Step 1–8: Profiling + transform design
Step 9: execute_transform() → 200 rows joined across 3 tables
Step 10: validate() → FK integrity check: 0.95
Final score: 0.89 ✓

=== HARD EPISODE ===
Step 1–7: Normal workflow
Step 8: profile_column() → ⚠️  SCHEMA DRIFT DETECTED!
       (Column 'order_date' changed dtype: object → datetime64)
Step 9–15: Agent adapts, rewrites transforms
Final score: 0.78 (lower due to drift adaptation cost) ✓

All full episode tests passed! ✓
```

---

## 🏃 Running All Tests at Once

```bash
# With pytest (faster, clearer output)
pytest tests/ -v

# Or run sequentially (recommended first time)
for test in tests/test_0{1,2,3,4,5}_*.py; do
  echo "Running $test..."
  python "$test" || exit 1
done
```

---

## 🌐 Start the Server

Once tests pass, start the FastAPI server:

```bash
python -m uvicorn environment.server:app --host 0.0.0.0 --port 8000
```

**Expected output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
# Returns: {"status": "ok", "version": "1.0.0"}
```

#### 2. Reset Environment

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Returns:
# {
#   "observation": {
#     "dataset_sample": [[...], ...],
#     "schema_target": {"column1": "int64", ...},
#     "steps_remaining": 15,
#     ...
#   },
#   "episode_id": "episode_001"
# }
```

#### 3. Step (Call a Tool)

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id": "episode_001",
    "action": {
      "tool": "profile_column",
      "params": {"column": "amount"}
    }
  }'

# Returns:
# {
#   "observation": {...},
#   "reward": 0.05,
#   "done": false,
#   "tool_output": "Column: amount | null_rate: 0.08 | dtype: float64 | min: 0.5, max: 9999.99"
# }
```

#### 4. Get State

```bash
curl http://localhost:8000/state \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "episode_001"}'

# Returns:
# {
#   "task_id": "easy",
#   "step_count": 3,
#   "steps_remaining": 12,
#   "drift_detected": false,
#   "drift_step": null,
#   "budget_exceeded": false
# }
```

---

## 🤖 Running Inference (LLM Agent)

The `inference.py` script runs a baseline LLM agent against the environment.

### Setup

```bash
# 1. Get your Hugging Face token
# Go to https://huggingface.co/settings/tokens → Create new token

# 2. Set environment variables
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"  # Or another model
export API_BASE_URL="https://router.huggingface.co/v1"
```

### Run Agent on One Task

```bash
python inference.py --task easy --num_episodes 1
```

**Expected output:**

```
Episode 1/1 [EASY]
Step 1: Agent calls profile_column('amount')
  → Reward: +0.05
Step 2: Agent calls write_transform(code='df["amount"] = pd.to_numeric(...)')
  → Reward: 0.0 (stored, not executed)
Step 3: Agent calls execute_transform()
  → Reward: +0.10
Step 4: Agent calls validate(checks=['null_check', 'range_check'])
  → Rewards: +0.10 each
...
Episode done. Final score: 0.85
```

### Run Baseline on All Tasks

```bash
python inference.py --num_episodes 10
```

This runs 10 episodes each on Easy, Medium, Hard and saves results to `baseline_results.json`:

```json
{
  "easy": {
    "episode_1": {"final_score": 0.92, "steps_taken": 5},
    "episode_2": {"final_score": 0.88, "steps_taken": 6},
    ...
    "mean_score": 0.89,
    "std_score": 0.04
  },
  "medium": {...},
  "hard": {...}
}
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t etl-agent .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_your_token" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  etl-agent
```

Server will start on `http://localhost:8000`

---

## 📁 File Reference

| File                  | Purpose                  | Input                      | Output                          |
| --------------------- | ------------------------ | -------------------------- | ------------------------------- |
| **models.py**         | Pydantic data classes    | —                          | ETLAction, ETLObservation, etc. |
| **fault_injector.py** | Generate broken datasets | task_id, seed              | Broken df + gold df             |
| **grader.py**         | Score transformed data   | agent_df, gold_df, task_id | Score 0.0–1.0                   |
| **reward.py**         | Calculate step rewards   | tool, params, success      | Reward float                    |
| **etl_env.py**        | Main environment         | Action → observation       | (obs, reward, done, step_info)  |
| **server.py**         | FastAPI REST API         | HTTP requests              | JSON responses                  |
| **inference.py**      | LLM agent baseline       | Task + env                 | Episode results                 |

---

## 🔍 Understanding Each Tool

### profile_column

```python
# Agent calls:
action = ETLAction(
    tool="profile_column",
    params={"column": "amount"}
)

# Environment returns:
"""
Column: amount
  Type: float64
  Null rate: 0.08 (8 of 100 nulls)
  Range: [5.50, 9999.99]
  Top values: [100.0, 50.0, 75.5]
"""
# Reward: +0.05
```

### write_transform

```python
# Agent stores code WITHOUT executing it:
action = ETLAction(
    tool="write_transform",
    params={
        "code": """
df = df.dropna(subset=['amount'])
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
"""
    }
)

# Reward: 0.0 (preparation action, no immediate signal)
```

### execute_transform

```python
# Agent runs the stored code against `df`:
action = ETLAction(tool="execute_transform", params={})

# If successful:
# Rows went from 100 → 92 (8 nulls dropped)
# Reward: +0.10

# If syntax error:
# Returns error message, agent can call fix_transform()
# Reward: -0.05
```

### validate

```python
# Agent requests quality checks:
action = ETLAction(
    tool="validate",
    params={
        "checks": ["null_check", "range_check", "schema_match"]
    }
)

# Environment runs 3 checks:
# null_check: 0.0 (still has nulls) ✗
# range_check: 1.0 (all values in valid range) ✓
# schema_match: 1.0 (matches target schema) ✓
# Reward: +0.30 (0.10 per check)
```

### submit

```python
# Agent ends episode and triggers final grading:
action = ETLAction(
    tool="submit",
    params={"reasoning": "Dropped nulls, validated schema, ready for production"}
)

# Environment compares agent_df vs gold_df across 6 dimensions:
# - null_check: 1.0
# - row_count_match: 0.95 (kept 95% of gold rows)
# - schema_match: 1.0
# - uniqueness_check: 1.0
# - referential_integrity: 0.90
# - range_check: 1.0
# Final score: (1.0 + 0.95 + 1.0 + 1.0 + 0.90 + 1.0) / 6 = 0.975

# Episode ends, done=True
```

---

## 🎯 Reward Structure

### Immediate Rewards (Per Step)

| Action            | Reward      | Condition                                   |
| ----------------- | ----------- | ------------------------------------------- |
| profile_column    | +0.05       | Always                                      |
| write_transform   | 0.0         | Always (no execution yet)                   |
| execute_transform | +0.10       | Success; -0.05 if syntax error              |
| validate          | +0.10/check | Each check that passes                      |
| fix_transform     | +0.08       | Fix is syntactically correct                |
| inspect_sample    | +0.02       | Always (low reward for passive observation) |
| load_to_target    | -0.05       | If schema doesn't match (penalty)           |
| submit            | —           | Triggers final grading                      |

### Final Episode Reward (Easy Task)

```
Score = (null_check + row_count + schema_match +
         uniqueness_check + referential_integrity + range_check) / 6

Range: 0.0–1.0
Gold data: 1.0
Medium cleaning job: 0.85–0.92
Poor cleaning: 0.3–0.5
No cleaning: ~0.70 (broken data itself has 70% quality)
```

---

## 🧩 Understanding Task Difficulty

### EASY (1 Table, Deterministic)

```
Broken CSV: 100 rows, 5 columns
Faults (known):
  - Nulls in numeric columns (8%)
  - Negative values where invalid (3%)
  - Dtype mismatches (string instead of int)
  - Duplicate rows (2%)
  - Case inconsistencies (category column)

Target: Fix all faults, pass 6 quality checks
Steps: 15 (usually needs 4–6)
Gold score: 1.0
Broken score: ~0.70
```

### MEDIUM (3 Tables, FK Reasoning)

```
Broken CSVs:
  - customers.csv: 78 rows
  - products.csv: 22 rows
  - orders.csv: 200 rows (with FK violations)

Faults:
  - Missing foreign keys (orders.customer_id not in customers)
  - Missing product references
  - Cross-table nulls in join columns
  - Business rule violations (e.g., order qty > product stock)

Target: Fix and join all 3 tables
Precision/Recall tension:
  - Drop all suspicious rows → high precision, low recall
  - Keep everything → high recall, FK violations
  - Optimal: ~95% precision, ~90% recall

Steps: 20
Grader checks: 10 quality dimensions
Gold score: 1.0 (or 0.97 if one final row is borderline)
```

### HARD (Schema Drift Mid-Episode)

```
Same as Medium, but at step 8:
  - order_date column changes dtype: object → datetime64
  - New column added: discount_pct (random values)
  - One column dropped: internal_id

Agent must:
  1. Detect the drift (via profile_column or validate)
  2. Adapt transforms (rewrite code to handle new dtypes)
  3. Complete episode with new schema

Steps: 20 (2–3 extra for adaptation)
Gold score: 1.0 (if drift detected correctly)
Score if drift missed: 0.6–0.7 (transform code breaks)

Randomization:
  - Drift step: always step 8
  - Drift type: random (dtype change, column add/drop, rename)
  - Drift field: random column
  - Can't memorize the fix across episodes
```

---

## 🧪 Common Testing Workflows

### Test 1: Does the environment work end-to-end?

```bash
python tests/test_05_full_episode.py
# Runs one episode each on Easy, Medium, Hard
# All should complete without errors
```

### Test 2: Does my LLM agent work?

```bash
# Start server in one terminal:
python -m uvicorn environment.server:app --port 8000

# In another terminal, run agent:
python inference.py --task easy --num_episodes 1 --verbose
```

### Test 3: What's the baseline performance?

```bash
python inference.py --num_episodes 100
cat baseline_results.json | python -m json.tool
# See Mean±Std scores for each task
```

### Test 4: Debug a specific episode

```python
python
>>> from environment.etl_env import ETLEnvironment
>>> import json
>>>
>>> env = ETLEnvironment(task_id="easy", seed=42)
>>> obs, step_info = env.reset()
>>>
>>> # Profile the first column manually
>>> action_dict = {
...     "tool": "profile_column",
...     "params": {"column": "amount"}
... }
>>> obs, reward, done, step_info = env.step(action_dict)
>>> print(json.dumps(step_info["tool_output"], indent=2))
>>> print(f"Reward: {reward}")
>>> print(f"Steps remaining: {obs['steps_remaining']}")
```

---

## ✅ Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'environment'"

**Solution:** Ensure `.venv` is activated and you're in the `/Users/prathamtomar/Desktop/HRIT` directory.

```bash
cd /Users/prathamtomar/Desktop/HRIT
source .venv/bin/activate
python tests/test_01_models.py
```

### Issue: "KeyError: 'column_name'" in profile_column

**Solution:** Agent tried to profile a column that doesn't exist. Check the schema first.

```python
obs, _ = env.reset()
print(obs["schema_current"])  # Print all available columns
```

### Issue: execute_transform fails with "NameError: name 'pd' is not defined"

**Solution:** pd, np, and df are auto-injected into the exec namespace. Just use them directly:

```python
# Agent code should be:
df['col'] = pd.to_numeric(df['col'])  # ✓ Correct
# NOT:
import pandas as pd  # ✗ Will fail, pd already available
```

### Issue: Hard task scores are suspiciously low

**Solution:** Check if drift was detected. Use `state()` to see `drift_detected` flag.

```bash
curl http://localhost:8000/state -d '{"episode_id": "..."}'
# If drift_detected=false, agent missed the schema change
```

---

## 📊 Performance Expectations

### Gold Data (No Faults)

- Easy: 1.0
- Medium: 0.97–1.0 (1 borderline row might be dropped)
- Hard: 1.0 (if drift detected) / 0.6 (if drift missed)

### Simple Agent ("Drop all nulls, validate")

- Easy: 0.88–0.92 (too aggressive, loses rows)
- Medium: 0.75–0.82 (joins drop too much)
- Hard: 0.40–0.50 (breaks on drift)

### Smart Agent (profiles → inspects → surgically fixes)

- Easy: 0.92–0.98
- Medium: 0.88–0.94
- Hard: 0.85–0.92 (detects drift, adapts)

### Target (GRPO-Trained Agent)

- Easy: 0.95+ (near-gold)
- Medium: 0.92+ (balanced precision/recall)
- Hard: 0.90+ (detects and adapts to drift)

---

## 🚦 Next Steps

1. **Verify setup**: Run `pip install -r requirements.txt`
2. **Run tests**: `python tests/test_01_models.py` through `test_05_full_episode.py`
3. **Start server**: `python -m uvicorn environment.server:app --port 8000`
4. **Run baseline**: `python inference.py --num_episodes 5` (test run)
5. **Train GRPO**: Use `openenv.yaml` spec to configure GRPO setup in TRL

---

## 📖 References

- **OpenEnv Spec**: `openenv.yaml`
- **Problem Statement Research**:
  - ELT-Bench (2025) — Spider-Agent at 3.9% success rate
  - Spider 2.0 — Multi-relational query challenges
  - Wordle-to-ETL analogy: Tight feedback loops enable learning
- **GRPO Training**: Use the environment with TRL's GRPO trainer
  - Baseline: https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb

---

**Questions?** Check the files directly or print observations with `json.dumps()` to debug interactively.
