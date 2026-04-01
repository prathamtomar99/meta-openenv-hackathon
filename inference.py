"""
inference.py — Baseline inference script (MANDATORY per OpenEnv spec).
Uses OpenAI client to run an LLM agent against all 3 tasks.
Reads credentials from environment variables.

Required env vars:
    API_BASE_URL — LLM endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME   — model identifier
    HF_TOKEN     — your HF / API key

Run:
    python inference.py

Output: prints score for each task + summary table.
Runtime target: < 20 min on 2vCPU / 8GB.
"""

import os
import re
import json
import time
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from environment.etl_env import ETLEnvironment
from environment.models import ETLAction

# ─────────────────────────────────────────────
#  CONFIG — read from env vars
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS    = {"easy": 15, "medium": 20, "hard": 25}
TEMPERATURE  = 0.2
MAX_TOKENS   = 256
SEED         = 42

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer operating an ETL pipeline agent.
You receive a broken dataset and a target schema contract.
Your goal is to clean the data and make it match the target schema.

## Available tools (one per turn)
- profile_column: inspect stats for one column (params: {"column": "<name>"})
- inspect_sample: see raw rows (params: {"n_rows": 5})
- write_transform: store pandas code (params: {"code": "<python code>"})
- execute_transform: run the stored code against df (params: {})
- validate: run quality checks (params: {"checks": ["null_check","type_check","range_check","uniqueness_check"]})
- fix_transform: revise code (params: {"code": "<new python code>", "error_msg": "<error>"})
- load_to_target: write output (params: {})
- submit: end episode (params: {"reasoning": "<your explanation>"})

## IMPORTANT RULES for write_transform / fix_transform code:
- The dataframe is available as the variable `df`
- You MUST reassign `df` at the end: `df = df.reset_index(drop=True)`
- Use pandas and numpy only (pd and np are pre-imported)
- Do not use print() or import statements

## Response format — ALWAYS respond with valid JSON:
{
  "tool": "<tool_name>",
  "params": {<tool_params>},
  "reasoning": "<your chain-of-thought>"
}

No markdown, no extra text — just the JSON object.
""").strip()


# ─────────────────────────────────────────────
#  AGENT LOOP
# ─────────────────────────────────────────────

def build_user_message(obs, step_num: int) -> str:
    """Build the user message for the LLM from the current observation."""
    lines = [
        f"=== Step {step_num} | Steps remaining: {obs.steps_remaining} ===",
        "",
        "--- Dataset sample (first 5 rows) ---",
        json.dumps(obs.dataset_sample[:5], default=str, indent=2),
        "",
        "--- Current schema ---",
        json.dumps(obs.schema_current, indent=2),
        "",
        "--- Target schema contract ---",
        json.dumps(obs.schema_target, indent=2),
        "",
        "--- Quality profile ---",
        json.dumps(obs.quality_profile, indent=2),
    ]

    if obs.last_tool_output:
        lines += ["", "--- Last tool output ---", obs.last_tool_output[:1000]]

    if obs.validation_scores:
        lines += ["", "--- Validation scores ---", json.dumps(obs.validation_scores, indent=2)]

    if obs.errors_seen:
        lines += ["", "--- Errors seen ---", "\n".join(obs.errors_seen[-3:])]

    if obs.transform_history:
        lines += ["", "--- Last transform code ---", obs.transform_history[-1][:500]]

    if obs.schema_drift_event:
        lines += [
            "",
            "⚠️  SCHEMA DRIFT DETECTED:",
            obs.schema_drift_event.get("message", str(obs.schema_drift_event)),
        ]

    lines += ["", "Respond with your next action as JSON."]
    return "\n".join(lines)


def parse_action(response_text: str) -> Optional[ETLAction]:
    """Parse LLM response into ETLAction. Handles messy outputs gracefully."""
    text = response_text.strip()

    # Try to extract JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Try to find a JSON object anywhere in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
        tool = data.get("tool", "submit")
        params = data.get("params", {})
        reasoning = data.get("reasoning", "")
        return ETLAction(tool=tool, params=params, reasoning=reasoning)
    except Exception:
        # Fallback: try to detect tool name from raw text
        for tool_name in ["profile_column", "inspect_sample", "write_transform",
                          "execute_transform", "validate", "fix_transform",
                          "load_to_target", "submit"]:
            if tool_name in text:
                return ETLAction(tool=tool_name, params={}, reasoning="Parsed from malformed response")
        return ETLAction(tool="submit", params={"reasoning": "Could not parse response"})


def run_episode(task_id: str, seed: int = SEED) -> Dict:
    """Run one full episode for a given task. Returns episode results."""
    env = ETLEnvironment(task_id=task_id)
    result = env.reset(seed=seed)
    obs = result.observation

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_log = []
    cumulative_reward = 0.0

    print(f"\n{'='*60}")
    print(f"Task: {task_id.upper()} | Seed: {seed}")
    print(f"{'='*60}")

    for step_num in range(1, MAX_STEPS[task_id] + 1):
        user_msg = build_user_message(obs, step_num)
        messages.append({"role": "user", "content": user_msg})

        # LLM call
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [Step {step_num}] LLM error: {e}")
            response_text = '{"tool": "submit", "params": {"reasoning": "LLM error"}, "reasoning": ""}'

        messages.append({"role": "assistant", "content": response_text})

        # Parse and execute action
        action = parse_action(response_text)
        result = env.step(action)
        obs = result.observation
        cumulative_reward += result.reward

        step_log = {
            "step": step_num,
            "tool": action.tool,
            "reward": result.reward,
            "done": result.done,
        }
        episode_log.append(step_log)

        print(f"  Step {step_num:02d} | tool={action.tool:<20} | reward={result.reward:+.4f} | done={result.done}")
        if action.reasoning:
            print(f"            reasoning: {action.reasoning}")

        if result.done:
            final_score = result.info.get("final_score", 0.0)
            breakdown = result.info.get("grader_breakdown", {})
            print(f"\n  Final score:  {final_score:.4f}")
            print(f"  Breakdown:")
            for k, v in breakdown.items():
                bar = "█" * int(v * 20)
                print(f"    {k:<30} {v:.3f}  {bar}")
            return {
                "task_id": task_id,
                "seed": seed,
                "final_score": final_score,
                "breakdown": breakdown,
                "steps_used": step_num,
                "cumulative_reward": round(cumulative_reward, 4),
                "episode_log": episode_log,
            }

    # Should not reach here — budget enforced inside env
    return {
        "task_id": task_id,
        "seed": seed,
        "final_score": 0.0,
        "breakdown": {},
        "steps_used": MAX_STEPS[task_id],
        "cumulative_reward": round(cumulative_reward, 4),
        "episode_log": episode_log,
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"ETL Pipeline Agent — Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    print(f"Seed:     {SEED}")

    tasks = ["easy", "medium", "hard"]
    results = []

    start = time.time()
    for task_id in tasks:
        ep = run_episode(task_id=task_id, seed=SEED)
        results.append(ep)

    elapsed = time.time() - start

    # Summary table
    print(f"\n{'='*60}")
    print(f"BASELINE SCORES SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<12} {'Score':>8} {'Steps':>7} {'Cumulative Reward':>18}")
    print(f"{'-'*12} {'-'*8} {'-'*7} {'-'*18}")
    for ep in results:
        print(f"{ep['task_id']:<12} {ep['final_score']:>8.4f} {ep['steps_used']:>7} {ep['cumulative_reward']:>18.4f}")
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results to file for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "seed": SEED,
            "results": results,
            "runtime_seconds": round(elapsed, 2),
        }, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
