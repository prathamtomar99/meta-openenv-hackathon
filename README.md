---
title: OpenEnv ETL Pipeline Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: OpenEnv ETL environment for training data engineering agents
---

# ETL Pipeline Agent — OpenEnv Environment

An RL training environment where an LLM agent acts as a data engineer.
The agent receives a broken dataset and must identify faults, write transformation
code, validate quality, and submit a clean output.

## Motivation

ELT-Bench (2025) found that the best available agent — Spider-Agent with Claude 3.7
Sonnet extended thinking — correctly builds a complete ETL pipeline in only **3.9% of
cases**. This environment is designed to close that gap by providing a dense-reward,
POMDP-structured training ground for data engineering agents.

## Environment Description

The environment simulates the Extract-Transform-Load workflow that data engineers
perform daily. The agent observes a broken CSV (or set of CSVs), diagnoses faults
through profiling actions, writes pandas transformation code, validates quality
metrics, and submits the cleaned output for scoring.

## Action Space

8 discrete tool calls:

| Tool | Params | Description |
|------|--------|-------------|
| `profile_column` | `{"column": "<name>"}` | Stats: null rate, min/max, dtype, top values |
| `inspect_sample` | `{"n_rows": 5}` | View raw rows |
| `write_transform` | `{"code": "<python>"}` | Store pandas code (not run yet) |
| `execute_transform` | `{}` | Run stored code against `df` |
| `validate` | `{"checks": [...]}` | Quality checks → float per check |
| `fix_transform` | `{"code": "...", "error_msg": "..."}` | Revise code after error |
| `load_to_target` | `{}` | Write output (schema match required) |
| `submit` | `{"reasoning": "..."}` | End episode, trigger final grader |

## Observation Space

Mixed text + tabular. Each step returns:
- `dataset_sample` — first 5 rows as dicts
- `schema_current` — column → dtype of working df
- `schema_target` — target contract to satisfy
- `quality_profile` — null rate + dtype per column
- `last_tool_output` — text result of last action
- `validation_scores` — partial check scores
- `transform_history` — last 3 code snippets tried
- `errors_seen` — last 5 execution errors
- `steps_remaining` — budget left
- `schema_drift_event` — drift notification (Hard only)

## Tasks

### Task 1 — Easy: Single-table data cleaning
- Input: 1 CSV, 500–700 rows, 6 known fault types
- Faults: date format, duplicates, null FKs, negative amounts, outliers, case inconsistency
- Max steps: 15 | Expected score range: 0.7–1.0 for correct agent

### Task 2 — Medium: Multi-table join + business rules
- Input: 3 CSVs (orders, customers, products) with cross-table inconsistencies
- Faults: orphaned FKs, impossible margin, region typos, category case
- Max steps: 20 | Key challenge: precision vs recall on which rows to drop

### Task 3 — Hard: Schema drift + incremental repair
- Input: single table + schema drift event injected at step 8
- Challenge: detect drift, revise transform, preserve already-committed rows
- Max steps: 25 | Even GPT-4o fails 86% of the time on this class of task

## Reward Design

Dense multi-objective:
- `profile_column` (first call): +0.05
- `execute_transform` (success): +0.10
- `validate` (per passing check): +0.10
- `execute_transform` (syntax error): −0.05
- Every step: −0.01 budget penalty

Final score weights: accuracy 30%, completeness 25%, schema match 20%, referential integrity 15%, efficiency 10%.

## Setup

```bash
pip install -r requirements.txt
python -m uvicorn environment.server:app --port 8000
```

## Running Tests

```bash
# Run each test suite individually (recommended order):
python tests/test_01_models.py
python tests/test_02_fault_injector.py
python tests/test_03_grader.py
python tests/test_04_env_reset_step.py
python tests/test_05_full_episode.py

# Or all at once:
pytest tests/ -v
```

## Baseline Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx
python inference.py
```

## Docker

```bash
docker build -t etl-pipeline-agent .
docker run -p 8000:8000 etl-pipeline-agent
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute one action |
| GET  | `/state` | Episode metadata |
| GET  | `/tasks` | List all tasks |

## Baseline Scores

| Task | Expected Score | Notes |
|------|---------------|-------|
| Easy | ~0.70–0.90 | Depends on model quality |
| Medium | ~0.40–0.65 | Cross-table reasoning required |
| Hard | ~0.20–0.45 | Schema drift handling required |
