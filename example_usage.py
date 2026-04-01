#!/usr/bin/env python3
"""
Quick Start Example — ETL Agent Environment
This script shows how to use the environment locally (no server needed)
"""

import json
from environment.etl_env import ETLEnvironment

def main():
    print("=" * 70)
    print("QUICK START: ETL Pipeline Agent Environment")
    print("=" * 70)
    
    # 1. Create environment
    print("\n1️⃣  Creating environment (EASY task)...")
    env = ETLEnvironment(task_id="easy")
    
    # 2. Reset and get initial observation
    print("\n2️⃣  Calling reset()...")
    obs, step_info = env.reset()
    episode_id = step_info.get("episode_id", "unknown")
    print(f"   Episode ID: {episode_id}")
    print(f"   Steps remaining: {obs['steps_remaining']}")
    print(f"   Dataset sample (first 2 rows):")
    for i, row in enumerate(obs['dataset_sample'][:2]):
        print(f"     Row {i}: {row}")
    
    print(f"\n   Schema (current):")
    for col, dtype in obs['schema_current'].items():
        print(f"     {col}: {dtype}")
    
    print(f"\n   Schema (target):")
    target = obs['schema_target']
    for col, spec in list(target.items())[:3]:
        print(f"     {col}: {spec}")
    
    # 3. Profile a column
    print("\n3️⃣  Profiling 'amount' column (action 1 of 3)...")
    action = {
        "tool": "profile_column",
        "params": {"column": "amount"}
    }
    obs, reward, done, step_info = env.step(action)
    print(f"   ✓ Reward: {reward}")
    print(f"   Output: {step_info['tool_output'][:150]}...")
    print(f"   Steps remaining: {obs['steps_remaining']}")
    
    # 4. Write transform
    print("\n4️⃣  Writing transform code (action 2 of 3)...")
    action = {
        "tool": "write_transform",
        "params": {
            "code": """
# Clean the data
df = df.dropna(subset=['customer_id', 'amount'])
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df = df[df['amount'] > 0]
df['status'] = df['status'].str.lower()
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed')
"""
        }
    }
    obs, reward, done, step_info = env.step(action)
    print(f"   ✓ Code stored. Reward: {reward}")
    print(f"   Steps remaining: {obs['steps_remaining']}")
    
    # 5. Execute transform
    print("\n5️⃣  Executing transform (action 3 of 3)...")
    action = {"tool": "execute_transform", "params": {}}
    obs, reward, done, step_info = env.step(action)
    print(f"   ✓ Reward: {reward}")
    print(f"   Output: {step_info['tool_output'][:150]}...")
    print(f"   Steps remaining: {obs['steps_remaining']}")
    
    # 6. Validate
    print("\n6️⃣  Validating quality (action 4 of 3)...")
    action = {
        "tool": "validate",
        "params": {
            "checks": ["null_check", "range_check", "schema_match"]
        }
    }
    obs, reward, done, step_info = env.step(action)
    print(f"   ✓ Reward: {reward}")
    print(f"   Validation results:")
    for check_name, score in step_info.get('validation_results', {}).items():
        print(f"     {check_name}: {score:.2f}")
    print(f"   Steps remaining: {obs['steps_remaining']}")
    
    # 7. Submit
    print("\n7️⃣  Submitting solution (action 5 of 3)...")
    action = {
        "tool": "submit",
        "params": {"reasoning": "Cleaned nulls, fixed amounts, normalized dates"}
    }
    obs, reward, done, step_info = env.step(action)
    print(f"   ✓ Done: {done}")
    print(f"   Final score: {step_info.get('final_score', 'N/A')}")
    print(f"\n   Score breakdown:")
    for check, score in step_info.get('score_breakdown', {}).items():
        print(f"     {check}: {score:.3f}")
    
    # 8. Show final state
    print(f"\n   🎉 Episode complete!")
    print(f"   Total steps taken: {5}")
    print(f"   Final reward: {step_info.get('final_score', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Try MEDIUM and HARD tasks (see different complexity)")
    print("  2. Use server API: python -m uvicorn environment.server:app")
    print("  3. Run inference: python inference.py --task easy --num_episodes 5")
    print("=" * 70)

if __name__ == "__main__":
    main()
