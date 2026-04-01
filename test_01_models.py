"""
TEST 1 — Models
Run: python -m pytest tests/test_01_models.py -v
Tests that ETLAction, ETLObservation, ETLState, StepResult can be created and serialized.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from environment.models import ETLAction, ETLObservation, ETLState, StepResult


class TestETLAction:
    def test_valid_tool(self):
        a = ETLAction(tool="profile_column", params={"column": "amount"})
        assert a.tool == "profile_column"
        assert a.params["column"] == "amount"

    def test_all_tools_valid(self):
        tools = [
            "profile_column", "inspect_sample", "write_transform",
            "execute_transform", "validate", "fix_transform",
            "load_to_target", "submit"
        ]
        for t in tools:
            a = ETLAction(tool=t)
            assert a.tool == t

    def test_invalid_tool_raises(self):
        with pytest.raises(Exception):
            ETLAction(tool="hack_database")

    def test_defaults(self):
        a = ETLAction(tool="submit")
        assert a.params == {}
        assert a.reasoning == ""

    def test_serialization(self):
        a = ETLAction(tool="validate", params={"checks": ["null_check"]}, reasoning="testing")
        d = a.model_dump()
        assert d["tool"] == "validate"
        assert d["params"]["checks"] == ["null_check"]


class TestETLObservation:
    def test_empty_defaults(self):
        obs = ETLObservation()
        assert obs.dataset_sample == []
        assert obs.schema_current == {}
        assert obs.steps_remaining == 15
        assert obs.schema_drift_event is None

    def test_with_data(self):
        obs = ETLObservation(
            dataset_sample=[{"order_id": 1, "amount": 99.9}],
            schema_current={"order_id": "int64", "amount": "float64"},
            steps_remaining=12,
        )
        assert len(obs.dataset_sample) == 1
        assert obs.steps_remaining == 12

    def test_serialization(self):
        obs = ETLObservation(steps_remaining=5)
        d = obs.model_dump()
        assert d["steps_remaining"] == 5


class TestStepResult:
    def test_basic(self):
        obs = ETLObservation()
        result = StepResult(observation=obs, reward=0.05, done=False)
        assert result.reward == 0.05
        assert result.done is False
        assert result.info == {}

    def test_serialization(self):
        obs = ETLObservation()
        result = StepResult(observation=obs, reward=0.1, done=True, info={"step": 5})
        d = result.model_dump()
        assert d["reward"] == 0.1
        assert d["done"] is True
        assert d["info"]["step"] == 5


if __name__ == "__main__":
    print("Running model tests...")
    t1 = TestETLAction()
    t1.test_valid_tool(); print("  ✓ valid tool")
    t1.test_all_tools_valid(); print("  ✓ all tools valid")
    t1.test_invalid_tool_raises(); print("  ✓ invalid tool raises")
    t1.test_defaults(); print("  ✓ defaults")
    t1.test_serialization(); print("  ✓ serialization")

    t2 = TestETLObservation()
    t2.test_empty_defaults(); print("  ✓ observation defaults")
    t2.test_with_data(); print("  ✓ observation with data")

    t3 = TestStepResult()
    t3.test_basic(); print("  ✓ step result basic")
    t3.test_serialization(); print("  ✓ step result serialization")

    print("\n✅ All model tests passed!")
