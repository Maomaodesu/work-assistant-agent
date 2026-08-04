# Work Assistant

> A local-first web assistant that helps developers recover project context, turn requests into plans, and assess implementation progress from local development evidence.

Work Assistant is a Python and FastAPI application for individual developers who frequently switch between projects or AI coding conversations. It combines a conversational agent with local project inspection, task planning, progress analysis, and a workspace for organizing local Codex and Claude conversation history.

## Why Work Assistant?

When development is interrupted, the relevant context is usually scattered across Git history, modified files, terminal sessions, IDE state, and long AI conversations. Work Assistant brings those signals together locally and uses an LLM to generate a concise recovery view:

- what the current task is;
- what evidence exists in the project;
- what is likely completed or at risk; and
- what the next action should be.

## Core capabilities

- **Conversational task management** — classifies a request as a new task, existing-project assessment, progress check, or normal chat; asks for missing project context when necessary.
- **LLM-generated plans** — creates task plans and sub-tasks from a goal and one or more project paths.
- **Local project snapshots** — collects Git metadata, changed files, commit summaries, project structure, selected IDE/work-session signals, terminal history, and development process information.
- **Evidence-based progress reports** — compares a saved task plan with the latest local snapshot to produce completion estimates, evidence, risks, and next actions.
- **Persistent multi-turn sessions** — uses LangGraph checkpoints and SQLite to preserve web chat state across requests.
- **Workspace context recovery** — incrementally imports local Codex and Claude histories, associates them with projects, semantically segments long conversations, and derives reusable work items and context packages.
- **Safe continuation controls** — supports cancellation of streamed requests and keeps raw external histories read-only while storing Work Assistant analysis separately.

## Architecture

```text
Browser UI
    │ HTTP + Server-Sent Events (SSE)
    ▼
FastAPI server (server.py)
    ├── LangGraph agent workflow (agent_graph.py)
    │       ├── intent routing and context gathering
    │       ├── task planning (task_manager.py)
    │       ├── local snapshot collection (snapshot_collector.py)
    │       └── progress analysis (progress_analyzer.py)
    ├── Workspace services
    │       ├── Codex / Claude history synchronization
    │       ├── project matching and semantic segmentation
    │       └── work-item and context-package management
    └── SQLite persistence

LLM endpoint (OpenAI-compatible)
    └── Default development configuration: AMD Radeon Token Factory
        └── Qwen3.6-35B-A3B
```

The application owns orchestration, local evidence collection, and persistence. The LLM is used for intent classification, plan generation, summarization, and progress reasoning.

## Technology stack

- Python 3.10+
- FastAPI + Uvicorn + Jinja2
- LangGraph + LangChain OpenAI integration
- SQLite
- OpenAI-compatible LLM API
- Vanilla HTML, CSS, and JavaScript

## Quick start

### 1. Create a virtual environment and install dependencies

```powershell
cd work_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Start the application

```powershell
.\start.ps1
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. On the first launch, the setup wizard asks for the model endpoint, model name, API key, and optional default project paths.

### 3. Configure an LLM endpoint

The default development configuration is AMD Radeon Token Factory:

```dotenv
AMD_BASE_URL=https://developer.amd.com.cn/radeon/api/v1
AMD_MODEL=Qwen3.6-35B-A3B
AMD_API_KEY=your_key_here
```

You may either configure these values in the web setup wizard or copy `.env.example` to `.env` and fill in the values. Never commit `.env`, API keys, snapshots, or local databases.

For a headless configuration, set `WORK_ASSISTANT_SETUP_COMPLETED=true` after providing a valid API key.

## Running tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Data and privacy

Work Assistant runs locally and stores its task, conversation, workspace, and checkpoint state in SQLite under `data/`. Snapshot files are stored under `snapshots/`.

Depending on enabled settings, model-analysis prompts can include Git commit information and a summary of code differences. Before using a third-party endpoint, select only projects that may be analyzed and turn off `SEND_COMMIT_INFO` or `SEND_DIFF_SUMMARY` when appropriate. API keys are not stored in the repository; on Windows, the application can use Windows Credential Manager through `keyring`.

## Project layout

```text
server.py                       FastAPI routes, SSE, and application startup
agent_graph.py                  LangGraph agent workflow
task_manager.py                 Task plans and persistence
snapshot_collector.py           Local project evidence collection
progress_analyzer.py            Evidence-to-progress analysis
settings.py                     Endpoint, key, and local configuration management
workspace_store.py              Workspace SQLite data layer
external_conversation_sync.py   Codex/Claude history import
templates/                      HTML pages
static/                         CSS and browser-side JavaScript
tests/                          Unit and integration-style tests
```

## AMD Radeon deployment path

The current development baseline uses the official Radeon Token Factory API. The LLM layer is deliberately built around the OpenAI-compatible client interface, so it can be switched to a Radeon Cloud self-hosted vLLM endpoint by changing the following configuration values rather than rewriting the agent workflow:

```dotenv
AMD_BASE_URL=https://your-radeon-cloud-endpoint/v1
AMD_MODEL=the-model-exposed-by-your-vllm-service
AMD_API_KEY=your_endpoint_key_if_required
```

The final hackathon deliverables will document the deployed Radeon Cloud/vLLM configuration, measured runtime behavior, and reproducible demonstration steps. This README does not claim that a dedicated GPU deployment or benchmark has already been completed.

## Hackathon delivery status

| Deliverable | Status |
|---|---|
| Complete source code | Available in this repository |
| English README | Available (this document) |
| English Project Specification | Planned |
| Radeon Cloud / vLLM deployment evidence | Planned |
| 3–5 minute demo video | Planned |
| Optional poster or presentation | Planned |

## Documentation

- `ARCHITECTURE_CN.md` — Chinese architecture guide for the project author. It is a learning and maintenance document, not an official English contest deliverable.
