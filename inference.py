"""
inference.py — Round-1 compliant inference script.

Required env vars:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN

Optional env vars:
    LOCAL_IMAGE_NAME (only needed for from_docker_image flows)

Stdout contract:
    [START] ...
    [STEP]  ...
    [END]   ...
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
HF_TOKEN     = os.getenv("HF_TOKEN")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK    = os.getenv("BENCHMARK", "etl-pipeline-agent")

MAX_STEPS    = {"easy": 15, "medium": 20, "hard": 25}
TEMPERATURE  = 0.2
MAX_TOKENS   = 512
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_SECONDS = 1.5
SEED         = 42

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be set in the environment.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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


def _sanitize_log_value(value: str) -> str:
    """Keep logs on one line and stable for parser consumption."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def _action_to_str(action: ETLAction) -> str:
    params = json.dumps(action.params, separators=(",", ":"), sort_keys=True)
    return _sanitize_log_value(f"{action.tool}({params})")


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={_sanitize_log_value(model)}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = _sanitize_log_value(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def get_llm_response(messages: List[Dict[str, str]]) -> tuple[str, Optional[str]]:
    """Call the model with bounded retries and return response text + last error."""
    last_error: Optional[str] = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = response.choices[0].message.content or ""
            return response_text, None
        except Exception as exc:
            last_error = f"llm_api_error_attempt_{attempt}: {exc}"
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF_SECONDS * attempt)

    fallback = '{"tool": "submit", "params": {"reasoning": "LLM error"}, "reasoning": ""}'
    return fallback, last_error


def run_episode(task_id: str, seed: int = SEED) -> Dict:
    """Run one full episode for a given task. Returns episode results."""
    env = ETLEnvironment(task_id=task_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_log = []
    cumulative_reward = 0.0
    rewards: List[float] = []
    steps_used = 0
    final_score = 0.0
    breakdown: Dict[str, float] = {}
    success = False

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(seed=seed)
        obs = result.observation

        for step_num in range(1, MAX_STEPS[task_id] + 1):
            user_msg = build_user_message(obs, step_num)
            messages.append({"role": "user", "content": user_msg})

            response_text, llm_error = get_llm_response(messages)

            messages.append({"role": "assistant", "content": response_text})

            action = parse_action(response_text)
            prev_errors_count = len(obs.errors_seen)
            result = env.step(action)
            obs = result.observation

            reward = float(result.reward)
            done = bool(result.done)
            cumulative_reward += reward
            rewards.append(reward)
            steps_used = step_num

            current_error = llm_error
            if len(obs.errors_seen) > prev_errors_count:
                env_error = obs.errors_seen[-1]
                current_error = f"{current_error}; {env_error}" if current_error else env_error

            step_log = {
                "step": step_num,
                "tool": action.tool,
                "reward": reward,
                "done": done,
                "error": current_error,
            }
            episode_log.append(step_log)

            log_step(
                step=step_num,
                action_str=_action_to_str(action),
                reward=reward,
                done=done,
                error=current_error,
            )

            if done:
                final_score = float(result.info.get("final_score", 0.0))
                breakdown = result.info.get("grader_breakdown", {})
                success = final_score >= 0.0
                break

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        final_score = min(max(float(final_score), 0.0), 1.0)
        success = bool(success and final_score >= 0.0)
        log_end(success=success, steps=steps_used, score=final_score, rewards=rewards)

    return {
        "task_id": task_id,
        "seed": seed,
        "final_score": final_score,
        "breakdown": breakdown,
        "steps_used": steps_used,
        "cumulative_reward": round(cumulative_reward, 4),
        "episode_log": episode_log,
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    tasks = ["easy", "medium", "hard"]
    results = []

    start = time.time()
    for task_id in tasks:
        ep = run_episode(task_id=task_id, seed=SEED)
        results.append(ep)

    elapsed = time.time() - start

    # Save results to file for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "seed": SEED,
            "results": results,
            "runtime_seconds": round(elapsed, 2),
            "api_base_url": API_BASE_URL,
            "benchmark": BENCHMARK,
            "local_image_name": LOCAL_IMAGE_NAME,
        }, f, indent=2)


if __name__ == "__main__":
    main()
