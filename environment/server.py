"""
server.py — FastAPI server exposing the ETL Pipeline Agent as an OpenEnv REST API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

from environment.etl_env import ETLEnvironment
from environment.models import ETLAction, StepResult

app = FastAPI(
    title="ETL Pipeline Agent — OpenEnv",
    description="RL training environment for LLM data engineering agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per task_id (stateful server)
_envs: Dict[str, ETLEnvironment] = {}


def _get_env(task_id: str) -> ETLEnvironment:
    if task_id not in _envs:
        _envs[task_id] = ETLEnvironment(task_id=task_id)
    return _envs[task_id]


# ─────────────────────────────────────────────
#  REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    task_id: str = "easy"
    tool: str
    params: Dict[str, Any] = {}
    reasoning: str = ""


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "etl-pipeline-agent", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    """
    Start a new episode.
    Returns: observation (broken dataset + target schema), reward=0, done=False.
    """
    try:
        env = _get_env(req.task_id)
        result = env.reset(seed=req.seed)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """
    Execute one action.
    Returns: observation, reward (float), done (bool), info (dict).
    """
    try:
        env = _get_env(req.task_id)
        action = ETLAction(tool=req.tool, params=req.params, reasoning=req.reasoning)
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(task_id: str = "easy"):
    """
    Returns safe episode metadata (no gold data, no planted faults).
    """
    try:
        env = _get_env(task_id)
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata (used by openenv validate)."""
    return {
        "tasks": [
            {"id": "easy",   "name": "Single-table data cleaning",        "difficulty": 1, "max_steps": 15},
            {"id": "medium", "name": "Multi-table join + business rules",  "difficulty": 2, "max_steps": 20},
            {"id": "hard",   "name": "Schema drift + incremental repair",  "difficulty": 3, "max_steps": 25},
        ]
    }


if __name__ == "__main__":
    uvicorn.run("environment.server:app", host="0.0.0.0", port=8000, reload=False)
