#!/usr/bin/env python3
"""Quick demo of the ETL environment"""

from environment.etl_env import ETLEnvironment
from environment.models import ETLAction

print("=" * 70)
print("ETL PIPELINE AGENT - QUICK DEMO")
print("=" * 70)

# Create environment and reset
env = ETLEnvironment(task_id="easy")
result = env.reset()
obs = result.observation

print(f"\n✓ Episode started")
print(f"  - Steps remaining: {obs.steps_remaining}")
print(f"  - Dataset rows: {len(obs.dataset_sample)}")
print(f"  - Columns: {list(obs.schema_current.keys())}")

# Profile amount column
action = ETLAction(tool="profile_column", params={"column": "amount"})
result = env.step(action)
print(f"\n✓ Profiled 'amount' column")
print(f"  - Reward: {result.reward}")
print(f"  - Info: {result.info.get('tool_output', '')[:100]}...")

# Write transformation
action = ETLAction(
    tool="write_transform",
    params={
        "code": """
df = df.dropna(subset=['customer_id', 'amount'])
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df = df[df['amount'] > 0]
df['status'] = df['status'].str.lower()
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed')
"""
    }
)
result = env.step(action)
print(f"\n✓ Wrote transformation")
print(f"  - Reward: {result.reward}")

# Execute
action = ETLAction(tool="execute_transform", params={})
result = env.step(action)
print(f"\n✓ Executed transformation")
print(f"  - Reward: {result.reward}")
print(f"  - {result.info.get('tool_output', '')[:100]}...")

# Validate
action = ETLAction(
    tool="validate",
    params={"checks": ["null_check", "range_check"]}
)
result = env.step(action)
print(f"\n✓ Validated")
print(f"  - Reward: {result.reward}")
print(f"  - Checks: {result.info.get('validation_results', {})}")

# Submit
action = ETLAction(
    tool="submit",
    params={"reasoning": "Cleaned and validated"}
)
result = env.step(action)
print(f"\n✓ Submitted")
print(f"  - Done: {result.done}")
print(f"  - Final score: {result.info.get('final_score', 0):.3f}")
print(f"  - Breakdown: {result.info.get('score_breakdown', {})}")

print("\n" + "=" * 70)
print("Setup complete! Try running:")
print("  • python test_01_models.py")
print("  • python test_02_fault_injector.py")
print("  • python test_03_grader.py")
print("  • python test_04_env_reset_step.py")
print("  • python test_05_full_episode.py")
print("\nOr start the server:")
print("  • python -m uvicorn environment.server:app --port 8000")
print("=" * 70)
