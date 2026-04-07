"""
server.py — FastAPI server exposing the ETL Pipeline Agent as an OpenEnv REST API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def home():
        return """
        <!doctype html>
        <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>OpenEnv ETL Pipeline Agent</title>
                <style>
                    :root {
                        color-scheme: dark;
                        --bg: #0b1020;
                        --panel: #10192f;
                        --panel-2: #14213d;
                        --text: #e5eefc;
                        --muted: #93a4c3;
                        --accent: #4ade80;
                        --accent-2: #38bdf8;
                        --border: rgba(255,255,255,0.08);
                    }
                    * { box-sizing: border-box; }
                    body {
                        margin: 0;
                        min-height: 100vh;
                        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                        background:
                            radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 30%),
                            radial-gradient(circle at 80% 20%, rgba(74,222,128,0.16), transparent 28%),
                            linear-gradient(180deg, #060913 0%, #0b1020 100%);
                        color: var(--text);
                        display: grid;
                        place-items: center;
                        padding: 32px;
                    }
                    .card {
                        width: min(920px, 100%);
                        background: rgba(16,25,47,0.82);
                        border: 1px solid var(--border);
                        border-radius: 24px;
                        box-shadow: 0 24px 80px rgba(0,0,0,0.35);
                        backdrop-filter: blur(16px);
                        overflow: hidden;
                    }
                    .hero {
                        padding: 40px;
                        border-bottom: 1px solid var(--border);
                        background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(74,222,128,0.04));
                    }
                    .eyebrow {
                        text-transform: uppercase;
                        letter-spacing: 0.18em;
                        color: var(--accent-2);
                        font-size: 0.78rem;
                        margin-bottom: 14px;
                    }
                    h1 {
                        margin: 0;
                        font-size: clamp(2rem, 5vw, 3.6rem);
                        line-height: 1.02;
                    }
                    .sub {
                        margin-top: 16px;
                        max-width: 70ch;
                        color: var(--muted);
                        font-size: 1.04rem;
                        line-height: 1.7;
                    }
                    .grid {
                        display: grid;
                        gap: 16px;
                        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                        padding: 24px 40px 40px;
                    }
                    .tile {
                        background: rgba(20,33,61,0.88);
                        border: 1px solid var(--border);
                        border-radius: 18px;
                        padding: 18px;
                    }
                    .tile h2 {
                        margin: 0 0 8px;
                        font-size: 1rem;
                    }
                    .tile p, .tile a {
                        color: var(--muted);
                        margin: 0;
                        line-height: 1.6;
                        font-size: 0.95rem;
                    }
                    .links a {
                        color: var(--accent);
                        text-decoration: none;
                    }
                    .links a:hover { text-decoration: underline; }
                    .pill {
                        display: inline-block;
                        margin-top: 10px;
                        padding: 6px 12px;
                        border-radius: 999px;
                        background: rgba(74,222,128,0.12);
                        border: 1px solid rgba(74,222,128,0.25);
                        color: #b9f6ca;
                        font-size: 0.85rem;
                    }
                </style>
            </head>
            <body>
                <main class="card">
                    <section class="hero">
                        <div class="eyebrow">OpenEnv Space</div>
                        <h1>ETL Pipeline Agent</h1>
                        <p class="sub">
                            This Space exposes the OpenEnv data-engineering benchmark and REST API.
                            Use it to reset episodes, step through tool calls, and inspect the current
                            training environment.
                        </p>
                        <div class="pill">Status: running</div>
                    </section>
                    <section class="grid">
                        <div class="tile">
                            <h2>API Endpoints</h2>
                            <p class="links">
                                <a href="/health">/health</a><br />
                                <a href="/docs">/docs</a><br />
                                <a href="/redoc">/redoc</a><br />
                                <a href="/tasks">/tasks</a>
                            </p>
                        </div>
                        <div class="tile">
                            <h2>Tasks</h2>
                            <p>Easy: single-table cleaning. Medium: multi-table reasoning. Hard: schema drift repair.</p>
                        </div>
                        <div class="tile">
                            <h2>Inference</h2>
                            <p>Use <code>inference.py</code> as the baseline LLM agent runner against the environment.</p>
                        </div>
                    </section>
                </main>
            </body>
        </html>
        """

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
