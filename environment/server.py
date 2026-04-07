"""
server.py — FastAPI server exposing the ETL Pipeline Agent as an OpenEnv REST API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

import json
from typing import Any, Dict, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from environment.etl_env import ETLEnvironment
from environment.models import ETLAction

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
def reset(req: Optional[ResetRequest] = Body(default=None)):
    """
    Start a new episode.
    Returns: observation (broken dataset + target schema), reward=0, done=False.
    """
    try:
        req = req or ResetRequest()
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


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=True)


def ui_health() -> str:
    return _pretty(health())


def ui_tasks() -> str:
    return _pretty(list_tasks())


def ui_reset(task_id: str, seed: int) -> str:
    result = reset(ResetRequest(task_id=task_id, seed=int(seed)))
    return _pretty(result)


def ui_step(task_id: str, tool: str, params_json: str, reasoning: str) -> str:
    parsed_params: Dict[str, Any] = {}
    raw = (params_json or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed_params = parsed
            else:
                return _pretty({"error": "params_json must decode to a JSON object"})
        except json.JSONDecodeError as exc:
            return _pretty({"error": f"Invalid JSON in params_json: {exc}"})

    result = step(StepRequest(task_id=task_id, tool=tool, params=parsed_params, reasoning=reasoning or ""))
    return _pretty(result)


def ui_state(task_id: str) -> str:
    return _pretty(state(task_id=task_id))


with gr.Blocks(title="OpenEnv ETL Pipeline Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # OpenEnv ETL Pipeline Agent
        Use this UI to call all core environment operations while keeping API endpoints available.
        API docs: `/docs` | OpenEnv health: `/health`
        """
    )

    with gr.Row():
        task = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task")
        seed = gr.Number(value=42, precision=0, label="Seed")

    with gr.Tab("Reset"):
        reset_btn = gr.Button("POST /reset", variant="primary")
        reset_out = gr.Code(label="Reset Response", language="json")
        reset_btn.click(ui_reset, inputs=[task, seed], outputs=reset_out)

    with gr.Tab("Step"):
        tool = gr.Dropdown(
            [
                "profile_column",
                "inspect_sample",
                "write_transform",
                "execute_transform",
                "validate",
                "fix_transform",
                "load_to_target",
                "submit",
            ],
            value="profile_column",
            label="Tool",
        )
        params_json = gr.Textbox(
            value='{"column": "amount"}',
            lines=4,
            label="Params JSON",
        )
        reasoning = gr.Textbox(value="", lines=3, label="Reasoning (optional)")
        step_btn = gr.Button("POST /step", variant="primary")
        step_out = gr.Code(label="Step Response", language="json")
        step_btn.click(ui_step, inputs=[task, tool, params_json, reasoning], outputs=step_out)

    with gr.Tab("State"):
        state_btn = gr.Button("GET /state", variant="primary")
        state_out = gr.Code(label="State Response", language="json")
        state_btn.click(ui_state, inputs=[task], outputs=state_out)

    with gr.Tab("Health & Tasks"):
        with gr.Row():
            health_btn = gr.Button("GET /health")
            tasks_btn = gr.Button("GET /tasks")
        health_out = gr.Code(label="Health Response", language="json")
        tasks_out = gr.Code(label="Tasks Response", language="json")
        health_btn.click(ui_health, outputs=health_out)
        tasks_btn.click(ui_tasks, outputs=tasks_out)


app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    uvicorn.run("environment.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
