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
- **Incremental workspace analysis** — synchronizes local Codex and Claude histories without reprocessing stable conversation evidence; only changed conversation tails are re-segmented and reclassified.
- **Bounded retrieval for long histories** — chunks imported conversations, retrieves relevant historical evidence locally, and uses a hierarchical summary path when a full history is too large for one model request.
- **Work-item candidate retrieval** — selects a bounded Top-K set of locally relevant work items before asking the model to classify a conversation segment, reducing irrelevant context and repeated model calls.
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
    │       ├── incremental Codex / Claude history synchronization
    │       ├── project matching and stable semantic segmentation
    │       ├── local retrieval chunks and hierarchical summaries
    │       └── bounded work-item discovery and context-package management
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

## Measuring a dedicated Radeon Cloud vLLM endpoint

After launching a dedicated Radeon Cloud Model API, run the repeatable benchmark below from PowerShell. The script prompts for the API key without saving it, sends one warm-up request and three measured non-streaming requests, then writes a non-secret JSON result under `bench-results/`.

```powershell
.\scripts\benchmark_vllm.ps1 `
  -BaseUrl "https://your-radeon-cloud-endpoint/v1" `
  -Model "Qwen/Qwen3-14B"
```

Record the generated JSON together with a redacted platform screenshot for the final Project Specification and demo. Do not commit an API key.

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
workspace_store.py              Workspace SQLite data layer and incremental evidence lifecycle
external_conversation_sync.py   Codex/Claude history import and change detection
conversation_retriever.py       Local lexical retrieval over imported conversation chunks
retrieval_chunker.py            Bounded, overlapping conversation-chunk construction
summary_hierarchy.py            Token-budgeted hierarchical summary construction
work_item_retriever.py          Top-K local candidate selection for work-item discovery
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

A dedicated Radeon Cloud vLLM Model API has been deployed for the current submission using `Qwen/Qwen3-14B` on an AMD Radeon Pro W7900. The local Work Assistant is configured through its setup page with the endpoint's OpenAI-compatible `/v1` base URL, model name, and locally stored API key; no application code change is required. The non-secret benchmark result in `bench-results/` records a three-run measurement of this endpoint.

## Hackathon delivery status

| Deliverable | Status |
|---|---|
| Complete source code | Available in this repository |
| English README | Available (this document) |
| English Project Specification | Available; measured deployment data included |
| Radeon Cloud / vLLM deployment evidence | Available; redacted screenshots and demo video pending |
| 3–5 minute demo video | Planned |
| Optional poster or presentation | Planned |

## Documentation

- `ARCHITECTURE_CN.md` — Chinese architecture guide for the project author. It is a learning and maintenance document, not an official English contest deliverable.
