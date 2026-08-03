"""
Work Assistant Agent - FastAPI 服务层

接口：
  POST /api/chat          接收用户消息，SSE 流式返回 agent 回复
  GET  /api/tasks         任务列表
  GET  /api/tasks/{id}    单任务详情
  GET  /api/tasks/{id}/progress  触发快照+进度分析，SSE 流式返回报告

UI：
  GET  /           主聊天页
  GET  /tasks      任务列表页
"""

import asyncio
import dataclasses
import json
import sys
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

if sys.platform == "win32":
    # Windows 下 asyncio 需要 Proactor event loop 才能跑子进程
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 把 work_assistant 目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from agent_graph import build_graph_v2
from task_manager import (
    init_db, list_tasks, load_task, Task,
    calculate_completion_percent,
)
from snapshot_collector import collect_snapshot, save_snapshot
from progress_analyzer import analyze_and_persist_progress
from session_store import (
    build_turn_input,
    create_sqlite_checkpointer,
    thread_config,
)
from settings import ConfigurationError, get_settings, settings_service
from conversation_manager import conversation_store
from streaming_runtime import (
    GenerationCancelled,
    GraphStreamCallback,
    bind_generation_cancel,
)
from workspace_launcher import (
    WorkspaceLaunchError,
    discover_local_tools,
    launch_ai_with_context,
    launch_project_tool,
    resume_external_session,
)
from work_item_context import WorkItemContextError, WorkItemContextService
from workspace_store import WorkspaceStoreError, workspace_store
from external_cli_chat import ExternalCliChatError, external_cli_chat_runner
from external_conversation_sync import external_conversation_sync
from project_matcher import conversation_project_matcher
from semantic_segmenter import semantic_conversation_segmenter
from analysis_job_manager import analysis_job_manager
from amd_health import amd_health_monitor

# ──────────────────────────────────────────────
# 初始化
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
init_db()
session_checkpointer, session_db_conn = create_sqlite_checkpointer(
    get_settings().checkpoint_db_path
)
agent_app = build_graph_v2(checkpointer=session_checkpointer)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 静态文件目录（CSS/JS）
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)

app = FastAPI(title="Work Assistant Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
_session_locks: dict[str, asyncio.Lock] = {}
_active_generations: dict[str, "ActiveGeneration"] = {}
SERVICE_STARTED_AT = datetime.now(timezone.utc).isoformat()


@dataclasses.dataclass
class ActiveGeneration:
    request_id: str
    cancel_event: threading.Event
    started_at: float


@app.middleware("http")
async def require_initial_setup(request: Request, call_next):
    """首次启动时只开放初始化页、静态资源和配置接口。"""
    path = request.url.path
    public_paths = {
        "/setup", "/api/setup", "/api/setup/test",
        "/api/setup/models", "/api/settings", "/api/amd/status",
        "/api/amd/config", "/health", "/api/health",
    }
    public_prefixes = ("/static/", "/docs", "/openapi.json", "/redoc")
    if (
        not settings_service.is_setup_complete()
        and path not in public_paths
        and not path.startswith(public_prefixes)
    ):
        if path.startswith("/api/"):
            return JSONResponse(
                {"error": "尚未完成初始化，请访问 /setup"},
                status_code=503,
            )
        return RedirectResponse("/setup", status_code=303)
    return await call_next(request)

# ──────────────────────────────────────────────
# 请求体
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    request_id: str | None = None


class ExternalCliChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    request_id: str | None = None


class SetupRequest(BaseModel):
    api_key: str = ""
    amd_base_url: str
    amd_model: str
    default_project_paths: list[str] = Field(default_factory=list)
    request_timeout_seconds: int = 180
    send_diff_summary: bool = True
    send_commit_info: bool = True


class ModelDiscoveryRequest(BaseModel):
    api_key: str = ""
    amd_base_url: str
    request_timeout_seconds: int = 60


class AMDQuickConfigRequest(BaseModel):
    api_key: str = ""
    amd_base_url: str
    amd_model: str


class ConversationCreateRequest(BaseModel):
    session_id: str | None = None
    title: str = "新对话"


class ConversationRenameRequest(BaseModel):
    title: str | None = None
    archived: bool | None = None


class ToolSettingsRequest(BaseModel):
    preferred_editor: str = "auto"
    tool_paths: dict[str, str] = Field(default_factory=dict)


class ProjectLaunchRequest(BaseModel):
    project_key: str
    action: str


class WorkspaceProjectCreateRequest(BaseModel):
    name: str
    root_paths: list[str] = Field(min_length=1)
    description: str = ""


class WorkspaceProjectRootsUpdateRequest(BaseModel):
    root_paths: list[str] = Field(min_length=1)


class WorkspaceWorkItemCreateRequest(BaseModel):
    project_id: str
    title: str
    item_type: str = "feature"
    goal: str = ""
    description: str = ""
    priority: str = "P1"
    deadline: str | None = None


class WorkspaceWorkItemIgnoreRequest(BaseModel):
    reason: str = ""


class WorkspaceWorkItemCompleteRequest(BaseModel):
    completion_note: str = Field(default="", max_length=4_000)
    acceptance_result: str = Field(default="", max_length=4_000)


class WorkspaceWorkItemMergeRequest(BaseModel):
    source_work_item_ids: list[str] = Field(min_length=1)


class ContextPackageLaunchRequest(BaseModel):
    source: str


def _context_service() -> WorkItemContextService:
    """保持上下文包与当前工作区数据库一致，便于隔离测试和未来切换数据目录。"""
    return WorkItemContextService(workspace_store)


class WorkspaceProjectActionRequest(BaseModel):
    action: str


class ConversationProjectAssignRequest(BaseModel):
    project_id: str


class WorkItemDiscoveryRequest(BaseModel):
    project_id: str | None = None
    force: bool = False
    limit: int | None = Field(default=None, ge=1, le=2000)


# ──────────────────────────────────────────────
# SSE 工具
# ──────────────────────────────────────────────

def sse_event(data: str, event: str = "message") -> str:
    """格式化单条 SSE 消息。"""
    lines = [f"event: {event}"]
    normalized = data.replace("\r\n", "\n").replace("\r", "\n")
    # split("\n") 保留结尾空项，避免模型单独输出换行 token 时被吞掉。
    for line in normalized.split("\n"):
        lines.append(f"data: {line}")
    lines.append("\n")
    return "\n".join(lines)


def run_agent_stream_worker(
    invoke_input: dict,
    session_id: str,
    callback: GraphStreamCallback,
) -> dict:
    """在线程池执行 LangGraph；回调会实时推送阶段和模型 token。"""
    config = thread_config(session_id)
    config["callbacks"] = [callback]
    with bind_generation_cancel(callback.cancel_event):
        return agent_app.invoke(invoke_input, config=config)


# ──────────────────────────────────────────────
# API 路由
# ──────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
async def api_health():
    """无需访问模型的本地服务健康检查。"""
    runtime_settings = get_settings()
    return JSONResponse({
        "status": "ok",
        "service": "work-assistant",
        "version": app.version,
        "pid": os.getpid(),
        "started_at": SERVICE_STARTED_AT,
        "setup_complete": settings_service.is_setup_complete(),
        "active_requests": len(_active_generations),
        "server": {
            "host": runtime_settings.server_host,
            "port": runtime_settings.server_port,
            "reload": runtime_settings.server_reload,
        },
    })

@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    """
    接收用户消息，SSE 流式返回 agent 回复。

    SSE 事件类型：request / stage / message / cancelled / error / done。
    chat 意图的 message 是模型原始 token；其他流程会实时推送节点阶段，
    并在结构化分析完成后发送格式化结果。
    """
    invoke_input = build_turn_input(req.message)
    conversation_store.ensure_work_assistant(req.session_id, req.message)
    session_lock = _session_locks.setdefault(req.session_id, asyncio.Lock())
    if session_lock.locked():
        return JSONResponse(
            {"error": "该会话正在处理上一条消息，请等待完成后再发送"},
            status_code=409,
        )
    await session_lock.acquire()

    request_id = req.request_id or f"request-{uuid.uuid4()}"
    cancel_event = threading.Event()
    active_generation = ActiveGeneration(
        request_id=request_id,
        cancel_event=cancel_event,
        started_at=time.monotonic(),
    )
    _active_generations[req.session_id] = active_generation
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()
    callback = GraphStreamCallback(loop, event_queue, cancel_event)

    def execute_graph():
        try:
            result = run_agent_stream_worker(invoke_input, req.session_id, callback)
            outcome = ("complete", result)
        except GenerationCancelled:
            outcome = ("cancelled", None)
        except Exception as exc:
            outcome = ("failed", exc)
        loop.call_soon_threadsafe(event_queue.put_nowait, outcome)

    worker_future = loop.run_in_executor(None, execute_graph)

    async def cleanup_generation():
        try:
            await worker_future
        except Exception:
            pass
        if _active_generations.get(req.session_id) is active_generation:
            _active_generations.pop(req.session_id, None)
        if session_lock.locked():
            session_lock.release()

    async def generate() -> AsyncGenerator[str, None]:
        terminal_event_sent = False
        deadline = loop.time() + get_settings().request_timeout_seconds
        last_heartbeat = loop.time()
        try:
            yield sse_event(
                json.dumps({"request_id": request_id}, ensure_ascii=False),
                event="request",
            )
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    cancel_event.set()
                    error_message = "请求超时，后台操作已停止；请检查 AMD API 后重试。"
                    conversation_store.append_exchange(req.session_id, req.message, error_message)
                    yield sse_event(error_message, event="error")
                    yield sse_event("", event="done")
                    terminal_event_sent = True
                    break

                try:
                    event_type, data = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=min(0.25, remaining),
                    )
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        cancel_event.set()
                        break
                    if loop.time() - last_heartbeat >= 15:
                        yield ": keep-alive\n\n"
                        last_heartbeat = loop.time()
                    continue

                if event_type == "stage":
                    yield sse_event(json.dumps(data, ensure_ascii=False), event="stage")
                elif event_type == "token":
                    yield sse_event(data, event="message")
                elif event_type == "complete":
                    result = data or {}
                    output = result.get("output", "") or "（Agent 没有返回内容）"
                    conversation_store.append_exchange(
                        req.session_id,
                        req.message,
                        output,
                        linked_task_id=result.get("task_id"),
                        project_path=(
                            ((result.get("task") or {}).get("project_paths") or [None])[0]
                        ),
                    )
                    if not callback.streamed_answer:
                        yield sse_event(output, event="message")
                    yield sse_event("", event="done")
                    terminal_event_sent = True
                    break
                elif event_type == "cancelled":
                    partial_output = callback.answer_text or "（已停止生成）"
                    conversation_store.append_exchange(
                        req.session_id,
                        req.message,
                        partial_output,
                    )
                    yield sse_event("已停止生成", event="cancelled")
                    yield sse_event("", event="done")
                    terminal_event_sent = True
                    break
                elif event_type == "failed":
                    error_message = f"Agent 出错：{data}"
                    conversation_store.append_exchange(req.session_id, req.message, error_message)
                    yield sse_event(error_message, event="error")
                    yield sse_event("", event="done")
                    terminal_event_sent = True
                    break
        finally:
            if not terminal_event_sent:
                cancel_event.set()
            if worker_future.done():
                await cleanup_generation()
            else:
                asyncio.create_task(cleanup_generation())

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/{session_id}/cancel")
async def api_cancel_chat(session_id: str, request_id: str | None = None):
    active = _active_generations.get(session_id)
    if not active:
        return JSONResponse({"error": "该会话当前没有正在执行的请求"}, status_code=404)
    if request_id and active.request_id != request_id:
        return JSONResponse({"error": "请求标识不匹配"}, status_code=409)
    active.cancel_event.set()
    return JSONResponse({
        "ok": True,
        "status": "cancelling",
        "request_id": active.request_id,
    })


@app.post("/api/external-conversations/{conversation_id:path}/chat")
async def api_external_conversation_chat(
    conversation_id: str,
    req: ExternalCliChatRequest,
):
    """在网页中续接可信的本地 Codex / Claude 导入会话。"""
    conversation = conversation_store.get(conversation_id)
    if not conversation:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    if not conversation.get("readonly") or conversation.get("archived"):
        return JSONResponse({"error": "只有未归档的外部导入会话可在网页续接"}, status_code=400)
    if conversation.get("source") not in {"codex", "claude"}:
        return JSONResponse({"error": "该会话不是 Codex 或 Claude 会话"}, status_code=400)
    if external_cli_chat_runner.active(conversation_id):
        return JSONResponse({"error": "该会话正在处理上一条消息"}, status_code=409)

    runtime_settings = get_settings()

    async def generate() -> AsyncGenerator[str, None]:
        terminal_sent = False
        try:
            yield sse_event(
                json.dumps({"request_id": req.request_id or f"external-{uuid.uuid4()}"}, ensure_ascii=False),
                event="request",
            )
            async for event_type, payload in external_cli_chat_runner.stream(
                conversation, req.message, configured_paths=runtime_settings.tool_paths,
            ):
                if event_type == "stage":
                    yield sse_event(json.dumps({"node": "external_cli", "label": payload}, ensure_ascii=False), event="stage")
                elif event_type == "message":
                    yield sse_event(payload, event="message")
                elif event_type == "log":
                    yield sse_event(payload, event="log")
                elif event_type == "error":
                    yield sse_event(payload, event="error")
                    yield sse_event("", event="done")
                    terminal_sent = True
                elif event_type == "cancelled":
                    yield sse_event(payload, event="cancelled")
                    yield sse_event("", event="done")
                    terminal_sent = True
                elif event_type == "done":
                    # CLI 已将新回合写回原始本地会话；同步只会重读有变化的来源文件。
                    await asyncio.get_running_loop().run_in_executor(None, conversation_store.import_external)
                    yield sse_event("", event="done")
                    terminal_sent = True
        except ExternalCliChatError as exc:
            yield sse_event(str(exc), event="error")
            yield sse_event("", event="done")
            terminal_sent = True
        except Exception as exc:
            yield sse_event(f"本地 CLI 桥接失败：{exc}", event="error")
            yield sse_event("", event="done")
            terminal_sent = True
        finally:
            if not terminal_sent:
                yield sse_event("已停止本地 CLI 会话", event="cancelled")
                yield sse_event("", event="done")

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/external-conversations/{conversation_id:path}/cancel")
async def api_cancel_external_conversation_chat(conversation_id: str):
    cancelled = await external_cli_chat_runner.cancel(conversation_id)
    if not cancelled:
        return JSONResponse({"error": "该外部会话当前没有正在执行的请求"}, status_code=404)
    return JSONResponse({"ok": True})


@app.get("/api/tasks")
async def api_list_tasks(status: str | None = None):
    tasks = list_tasks(status=status)
    return JSONResponse([_task_to_dict(t) for t in tasks])


@app.get("/api/conversations")
async def api_list_conversations():
    return JSONResponse(conversation_store.list())


@app.get("/api/projects")
async def api_list_projects():
    return JSONResponse(conversation_store.projects())


@app.get("/api/tools")
async def api_local_tools():
    runtime_settings = get_settings()
    return JSONResponse({
        "preferred_editor": runtime_settings.preferred_editor,
        "tools": discover_local_tools(runtime_settings.tool_paths),
    })


@app.post("/api/tools/discover")
async def api_discover_local_tools():
    runtime_settings = get_settings()
    return JSONResponse({
        "ok": True,
        "preferred_editor": runtime_settings.preferred_editor,
        "tools": discover_local_tools(runtime_settings.tool_paths),
    })


@app.post("/api/tools")
async def api_save_local_tools(req: ToolSettingsRequest):
    try:
        settings_service.save_tool_settings(req.model_dump())
        runtime_settings = get_settings()
        return JSONResponse({
            "ok": True,
            "preferred_editor": runtime_settings.preferred_editor,
            "tools": discover_local_tools(runtime_settings.tool_paths),
        })
    except ConfigurationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/projects/launch")
async def api_launch_project(req: ProjectLaunchRequest):
    project = conversation_store.get_project(req.project_key)
    if not project:
        return JSONResponse({"ok": False, "error": "项目不存在或尚未归入项目会话"}, status_code=404)
    runtime_settings = get_settings()
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: launch_project_tool(
                project["project_path"],
                req.action,
                preferred_editor=runtime_settings.preferred_editor,
                configured_paths=runtime_settings.tool_paths,
            ),
        )
        return JSONResponse({"ok": True, **result})
    except WorkspaceLaunchError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/workspace/schema")
async def api_workspace_schema():
    return JSONResponse(workspace_store.schema_info())


@app.get("/api/workspace/conversations")
async def api_workspace_conversations(source: str | None = None):
    try:
        return JSONResponse(workspace_store.list_conversations(source=source))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/conversations/sync")
async def api_workspace_sync_conversations():
    result = await asyncio.get_running_loop().run_in_executor(
        None, external_conversation_sync.sync
    )
    return JSONResponse({"ok": result["errors"] == 0, **result})


@app.post("/api/conversations/sync-all")
async def api_sync_all_conversations():
    """同步会话资料，并更新项目侧用于分析和归类的索引。"""
    loop = asyncio.get_running_loop()
    legacy_result = await loop.run_in_executor(None, conversation_store.import_external)
    workspace_result = await loop.run_in_executor(None, external_conversation_sync.sync)
    return JSONResponse({
        "ok": workspace_result["errors"] == 0,
        "conversations": legacy_result,
        "workspace": workspace_result,
    })


@app.post("/api/workspace/conversations/match-projects")
async def api_workspace_match_conversation_projects():
    result = await asyncio.get_running_loop().run_in_executor(
        None, conversation_project_matcher.match_all
    )
    return JSONResponse({"ok": True, **result})


@app.post("/api/workspace/conversations/segment")
async def api_workspace_segment_conversations():
    result = await asyncio.get_running_loop().run_in_executor(
        None, semantic_conversation_segmenter.segment_all
    )
    return JSONResponse({"ok": result["errors"] == 0, **result})


@app.post("/api/workspace/conversations/discover-work-items")
async def api_workspace_discover_work_items(req: WorkItemDiscoveryRequest):
    try:
        run = analysis_job_manager.start(
            project_id=req.project_id,
            force=req.force,
            limit=req.limit,
        )
        return JSONResponse({"ok": True, "run": run}, status_code=202)
    except (WorkspaceStoreError, ConfigurationError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/workspace/classification-runs")
async def api_workspace_classification_runs(limit: int = 20):
    return JSONResponse(analysis_job_manager.list(limit=limit))


@app.get("/api/workspace/classification-runs/{run_id}")
async def api_workspace_classification_run(run_id: str):
    run = analysis_job_manager.get(run_id)
    if not run:
        return JSONResponse({"error": "分析任务不存在"}, status_code=404)
    return JSONResponse(run)


@app.post("/api/workspace/classification-runs/{run_id}/pause")
async def api_workspace_pause_classification_run(run_id: str):
    try:
        return JSONResponse({"ok": True, "run": analysis_job_manager.pause(run_id)})
    except WorkspaceStoreError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/workspace/classification-runs/{run_id}/resume")
async def api_workspace_resume_classification_run(run_id: str):
    try:
        return JSONResponse({"ok": True, "run": analysis_job_manager.resume(run_id)})
    except WorkspaceStoreError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/workspace/classification-runs/{run_id}/cancel")
async def api_workspace_cancel_classification_run(run_id: str):
    try:
        return JSONResponse({"ok": True, "run": analysis_job_manager.cancel(run_id)})
    except WorkspaceStoreError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/workspace/classification-runs/{run_id}/retry")
async def api_workspace_retry_classification_run(run_id: str):
    try:
        return JSONResponse(
            {"ok": True, "run": analysis_job_manager.retry(run_id)}, status_code=202
        )
    except WorkspaceStoreError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/workspace/conversations/unassigned")
async def api_workspace_unassigned_conversations():
    conversations = workspace_store.list_conversations()
    return JSONResponse([
        conversation for conversation in conversations
        if conversation["project_match_state"] != "matched"
    ])


@app.post("/api/workspace/conversations/{conversation_id}/project")
async def api_workspace_assign_conversation_project(
    conversation_id: str,
    req: ConversationProjectAssignRequest,
):
    try:
        matches = workspace_store.set_manual_conversation_project(
            conversation_id, req.project_id
        )
        return JSONResponse({"ok": True, "matches": matches})
    except WorkspaceStoreError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/workspace/projects")
async def api_workspace_projects(status: str | None = None):
    try:
        return JSONResponse(workspace_store.list_projects(status=status))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/projects")
async def api_workspace_create_project(req: WorkspaceProjectCreateRequest):
    try:
        project = workspace_store.create_project(
            req.name,
            req.root_paths,
            description=req.description,
            source="manual",
        )
        return JSONResponse(project, status_code=201)
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.put("/api/workspace/projects/{project_id}/roots")
async def api_workspace_update_project_roots(
    project_id: str,
    req: WorkspaceProjectRootsUpdateRequest,
):
    try:
        return JSONResponse(workspace_store.update_project_roots(project_id, req.root_paths))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/workspace/overview")
async def api_workspace_overview():
    return JSONResponse(workspace_store.list_project_overviews())


@app.get("/api/workspace/projects/{project_id}")
async def api_workspace_project_detail(project_id: str):
    project = workspace_store.get_project_overview(project_id)
    if not project:
        return JSONResponse({"error": "项目不存在"}, status_code=404)
    return JSONResponse(project)


@app.get("/api/workspace/projects/{project_id}/segments")
async def api_workspace_project_segments(
    project_id: str,
    segment_kind: str | None = None,
    limit: int = 200,
):
    try:
        return JSONResponse(workspace_store.list_project_segment_details(
            project_id, segment_kind=segment_kind, limit=limit
        ))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/workspace/segments/{segment_id}")
async def api_workspace_segment_detail(segment_id: str):
    segment = workspace_store.get_segment_review_detail(segment_id)
    if not segment:
        return JSONResponse({"error": "会话片段不存在"}, status_code=404)
    return JSONResponse(segment)


@app.post("/api/workspace/projects/{project_id}/launch")
async def api_workspace_launch_project(
    project_id: str,
    req: WorkspaceProjectActionRequest,
):
    project = workspace_store.get_project(project_id)
    if not project:
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    primary_root = next(
        (root for root in project["roots"] if root["is_primary"]),
        project["roots"][0] if project["roots"] else None,
    )
    if not primary_root:
        return JSONResponse({"ok": False, "error": "项目没有可用目录"}, status_code=400)
    runtime_settings = get_settings()
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: launch_project_tool(
                primary_root["path"],
                req.action,
                preferred_editor=runtime_settings.preferred_editor,
                configured_paths=runtime_settings.tool_paths,
            ),
        )
        return JSONResponse({"ok": True, **result})
    except WorkspaceLaunchError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/workspace/work-items")
async def api_workspace_work_items(
    project_id: str | None = None,
    status: str | None = None,
    include_ignored: bool = False,
):
    try:
        return JSONResponse(workspace_store.list_work_items(
            project_id,
            status=status,
            include_ignored=include_ignored,
        ))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items")
async def api_workspace_create_work_item(req: WorkspaceWorkItemCreateRequest):
    try:
        item = workspace_store.create_work_item(
            req.project_id,
            req.title,
            item_type=req.item_type,
            goal=req.goal,
            description=req.description,
            priority=req.priority,
            deadline=req.deadline,
            source="manual",
        )
        return JSONResponse(item, status_code=201)
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/workspace/work-items/{work_item_id}")
async def api_workspace_work_item_detail(work_item_id: str):
    item = workspace_store.get_work_item(work_item_id)
    if not item:
        return JSONResponse({"error": "工作项不存在"}, status_code=404)
    return JSONResponse(item)


@app.get("/api/workspace/work-items/{work_item_id}/sources")
async def api_workspace_work_item_sources(work_item_id: str):
    try:
        return JSONResponse(workspace_store.list_work_item_source_details(work_item_id))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


@app.get("/api/workspace/work-items/{work_item_id}/context-packages")
async def api_workspace_context_packages(work_item_id: str):
    try:
        return JSONResponse(workspace_store.list_context_packages(work_item_id))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


@app.post("/api/workspace/work-items/{work_item_id}/context-packages")
async def api_workspace_generate_context_package(work_item_id: str):
    try:
        package = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _context_service().generate(work_item_id)
        )
        return JSONResponse({"ok": True, "package": package})
    except (WorkspaceStoreError, WorkItemContextError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{work_item_id}/continue-prompt")
async def api_workspace_continue_prompt(work_item_id: str):
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _context_service().generate_continue_prompt(work_item_id)
        )
        return JSONResponse({"ok": True, **result})
    except (WorkspaceStoreError, WorkItemContextError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/workspace/context-packages/{context_id}/content")
async def api_workspace_context_package_content(context_id: str):
    try:
        package, content = _context_service().read_content(context_id)
        return JSONResponse({"package": package, "content": content})
    except WorkItemContextError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


@app.post("/api/workspace/context-packages/{context_id}/launch")
async def api_workspace_launch_context_package(
    context_id: str,
    req: ContextPackageLaunchRequest,
): 
    try:
        package, _ = _context_service().read_content(context_id)
        work_item = workspace_store.get_work_item(package["work_item_id"])
        project = workspace_store.get_project(work_item["project_id"]) if work_item else None
        if not project:
            raise WorkItemContextError("上下文包所属项目不存在")
        primary_root = next(
            (root for root in project["roots"] if root["is_primary"]),
            project["roots"][0] if project["roots"] else None,
        )
        if not primary_root:
            raise WorkItemContextError("项目没有可用目录")
        runtime_settings = get_settings()
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: launch_ai_with_context(
                primary_root["path"],
                req.source,
                package["canonical_path"],
                configured_paths=runtime_settings.tool_paths,
            ),
        )
        return JSONResponse({"ok": True, "package": package, **result})
    except (WorkItemContextError, WorkspaceLaunchError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{target_work_item_id}/merge")
async def api_workspace_merge_work_items(
    target_work_item_id: str,
    req: WorkspaceWorkItemMergeRequest,
):
    try:
        return JSONResponse({
            "ok": True,
            "work_item": workspace_store.merge_work_items(
                target_work_item_id, req.source_work_item_ids
            ),
        })
    except WorkspaceStoreError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{work_item_id}/confirm")
async def api_workspace_confirm_work_item(work_item_id: str):
    try:
        return JSONResponse(workspace_store.confirm_work_item(work_item_id))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{work_item_id}/ignore")
async def api_workspace_ignore_work_item(
    work_item_id: str,
    req: WorkspaceWorkItemIgnoreRequest,
):
    try:
        return JSONResponse(workspace_store.ignore_work_item(
            work_item_id,
            reason=req.reason,
        ))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{work_item_id}/restore")
async def api_workspace_restore_work_item(work_item_id: str):
    try:
        return JSONResponse(workspace_store.restore_work_item(work_item_id))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{work_item_id}/complete")
async def api_workspace_complete_work_item(
    work_item_id: str,
    req: WorkspaceWorkItemCompleteRequest,
):
    try:
        return JSONResponse(workspace_store.complete_work_item(
            work_item_id,
            completion_note=req.completion_note,
            acceptance_result=req.acceptance_result,
        ))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/work-items/{work_item_id}/reopen")
async def api_workspace_reopen_work_item(work_item_id: str):
    try:
        return JSONResponse(workspace_store.reopen_work_item(work_item_id))
    except WorkspaceStoreError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/workspace/conversations/{conversation_id}/resume")
async def api_workspace_resume_external_conversation(conversation_id: str):
    conversation = workspace_store.get_conversation(conversation_id)
    if not conversation:
        return JSONResponse({"ok": False, "error": "会话不存在"}, status_code=404)
    if conversation["source"] not in {"codex", "claude"} or not conversation["resume_capable"]:
        return JSONResponse({"ok": False, "error": "这个会话不支持恢复"}, status_code=400)

    project_path = conversation.get("original_project_path", "")
    if not project_path or not Path(project_path).expanduser().is_dir():
        primary_match = next(
            (
                match for match in workspace_store.get_conversation_project_matches(conversation_id)
                if match["is_primary"]
            ),
            None,
        )
        if primary_match:
            project = workspace_store.get_project(primary_match["project_id"])
            primary_root = next(
                (root for root in project["roots"] if root["is_primary"]),
                project["roots"][0] if project["roots"] else None,
            )
            project_path = primary_root["path"] if primary_root else ""
    runtime_settings = get_settings()
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: resume_external_session(
                project_path,
                conversation["source"],
                conversation["external_session_id"],
                configured_paths=runtime_settings.tool_paths,
            ),
        )
        return JSONResponse({"ok": True, **result})
    except WorkspaceLaunchError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/conversations")
async def api_create_conversation(req: ConversationCreateRequest):
    session_id = req.session_id or f"session-{uuid.uuid4()}"
    conversation = conversation_store.ensure_work_assistant(session_id, req.title)
    if req.title and req.title != "新对话":
        conversation = conversation_store.rename(session_id, req.title)
    return JSONResponse(conversation, status_code=201)


@app.post("/api/conversations/import")
async def api_import_conversations():
    result = await asyncio.get_event_loop().run_in_executor(
        None, conversation_store.import_external
    )
    return JSONResponse({"ok": True, **result})


@app.get("/api/conversations/{conversation_id:path}/messages")
async def api_conversation_messages(conversation_id: str):
    conversation = conversation_store.get(conversation_id)
    if not conversation:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    return JSONResponse({
        "conversation": conversation,
        "messages": conversation_store.messages(conversation_id),
    })


@app.patch("/api/conversations/{conversation_id:path}")
async def api_rename_conversation(
    conversation_id: str,
    req: ConversationRenameRequest,
):
    try:
        conversation = conversation_store.get(conversation_id)
        if not conversation:
            raise KeyError(conversation_id)
        if req.title is not None:
            conversation = conversation_store.rename(conversation_id, req.title)
        if req.archived is not None:
            conversation = conversation_store.set_archived(conversation_id, req.archived)
        if req.title is None and req.archived is None:
            return JSONResponse({"error": "没有提供要更新的字段"}, status_code=400)
        return JSONResponse(conversation)
    except KeyError:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.delete("/api/conversations/{conversation_id:path}")
async def api_delete_conversation(conversation_id: str):
    if conversation_id in _active_generations:
        return JSONResponse(
            {"error": "该会话正在生成回答，请先停止生成再删除"},
            status_code=409,
        )
    conversation = conversation_store.delete(conversation_id)
    if not conversation:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    if conversation["source"] == "work_assistant":
        await asyncio.get_event_loop().run_in_executor(
            None, session_checkpointer.delete_thread, conversation_id
        )
        _session_locks.pop(conversation_id, None)
    return JSONResponse({"ok": True})


@app.get("/api/tasks/{task_id}")
async def api_get_task(task_id: str):
    task = load_task(task_id)
    if not task:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return JSONResponse(_task_to_dict(task))


@app.get("/api/tasks/{task_id}/progress")
async def api_task_progress(task_id: str):
    """
    触发快照采集 + 进度分析，SSE 流式返回报告。
    """
    task = load_task(task_id)
    if not task:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    async def generate() -> AsyncGenerator[str, None]:
        yield sse_event("正在采集快照...", event="thinking")
        try:
            loop = asyncio.get_event_loop()
            snapshot_obj = await loop.run_in_executor(
                None,
                lambda: collect_snapshot(
                    task_id=task_id,
                    project_paths=task.project_paths,
                    idea_project_path=task.project_paths[0],
                    interrupt_reason="Web 进度查询",
                ),
            )
            snapshot = dataclasses.asdict(snapshot_obj)
        except Exception as e:
            yield sse_event(f"快照采集失败：{e}", event="error")
            yield sse_event("", event="done")
            return

        yield sse_event("正在分析进度...", event="thinking")
        try:
            report = await loop.run_in_executor(
                None, analyze_and_persist_progress, task, snapshot
            )
        except Exception as e:
            yield sse_event(f"进度分析失败：{e}", event="error")
            yield sse_event("", event="done")
            return

        yield sse_event(json.dumps(report, ensure_ascii=False), event="report")
        yield sse_event("", event="done")

    return StreamingResponse(generate(), media_type="text/event-stream")


# ──────────────────────────────────────────────
# 页面路由
# ──────────────────────────────────────────────

@app.get("/setup", response_class=HTMLResponse)
async def page_setup(request: Request):
    return templates.TemplateResponse(request, "setup.html", {
        "settings": settings_service.public_view(),
        "is_setup": True,
    })


@app.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request):
    return templates.TemplateResponse(request, "setup.html", {
        "settings": settings_service.public_view(),
        "is_setup": False,
    })


@app.get("/api/settings")
async def api_settings():
    return JSONResponse(settings_service.public_view())


@app.get("/api/amd/status")
async def api_amd_status(force: bool = False):
    result = await asyncio.get_running_loop().run_in_executor(
        None, lambda: amd_health_monitor.check(force=force)
    )
    return JSONResponse(result)


@app.post("/api/amd/config")
async def api_save_amd_quick_config(req: AMDQuickConfigRequest):
    try:
        values = {
            "amd_base_url": req.amd_base_url,
            "amd_model": req.amd_model,
        }
        settings_service.validate_amd_settings(values)
        settings_service.save_api_key(req.api_key)
        settings_service.save_amd_settings(values, mark_complete=True)
        amd_health_monitor.invalidate()
        health = await asyncio.get_running_loop().run_in_executor(
            None, lambda: amd_health_monitor.check(force=True)
        )
        return JSONResponse({
            "ok": True,
            "settings": settings_service.public_view(),
            "health": health,
        })
    except ConfigurationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        message = str(exc)
        if req.api_key:
            message = message.replace(req.api_key, "***")
        return JSONResponse(
            {"ok": False, "error": f"保存 AMD 配置失败：{message}"}, status_code=500
        )


def _test_amd_connection(req: SetupRequest) -> str:
    values = settings_service.validate_public_settings(req.model_dump())
    api_key = req.api_key.strip() or settings_service.get_api_key()
    client = OpenAI(api_key=api_key, base_url=values["amd_base_url"])
    response = client.chat.completions.create(
        model=values["amd_model"],
        messages=[{"role": "user", "content": "Reply with OK only."}],
        temperature=0,
        max_tokens=8,
        timeout=values["request_timeout_seconds"],
    )
    return (response.choices[0].message.content or "").strip()


def _discover_amd_models(req: ModelDiscoveryRequest) -> list[dict]:
    base_url = req.amd_base_url.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigurationError("AMD API 地址必须是有效的 http/https URL")
    api_key = req.api_key.strip() or settings_service.get_api_key()
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.models.list(timeout=req.request_timeout_seconds)
    models = []
    seen = set()
    for item in response.data:
        model_id = str(item.id).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append({
            "id": model_id,
            "owned_by": getattr(item, "owned_by", None),
        })
    if not models:
        raise ConfigurationError("API 没有返回可用模型")
    return sorted(models, key=lambda item: item["id"].lower())


@app.post("/api/setup/models")
async def api_discover_models(req: ModelDiscoveryRequest):
    try:
        models = await asyncio.get_event_loop().run_in_executor(
            None, _discover_amd_models, req
        )
        return JSONResponse({"ok": True, "models": models})
    except Exception as exc:
        error_message = str(exc)
        if req.api_key:
            error_message = error_message.replace(req.api_key, "***")
        return JSONResponse({"ok": False, "error": error_message}, status_code=400)


@app.post("/api/setup/test")
async def api_test_setup(req: SetupRequest):
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _test_amd_connection, req
        )
        return JSONResponse({"ok": True, "response": result})
    except Exception as exc:
        error_message = str(exc)
        if req.api_key:
            error_message = error_message.replace(req.api_key, "***")
        return JSONResponse({"ok": False, "error": error_message}, status_code=400)


@app.post("/api/setup")
async def api_save_setup(req: SetupRequest):
    try:
        values = req.model_dump(exclude={"api_key"})
        settings_service.validate_public_settings(values)
        settings_service.save_api_key(req.api_key)
        settings_service.save_public_settings(values, mark_complete=True)
        return JSONResponse({"ok": True, "settings": settings_service.public_view()})
    except ConfigurationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"保存配置失败：{exc}"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def page_workspace_home(request: Request):
    """项目工作台是应用首页；前端会恢复上次选中的项目。"""
    return templates.TemplateResponse(request, "workspace.html", {
        "settings": settings_service.public_view(),
    })


@app.get("/conversations", response_class=HTMLResponse)
async def page_conversations(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "settings": settings_service.public_view(),
    })


@app.get("/tasks", response_class=HTMLResponse)
async def page_tasks(request: Request):
    tasks = list_tasks()
    return templates.TemplateResponse(request, "tasks.html", {
        "tasks": [_task_to_dict(t) for t in tasks],
        "settings": settings_service.public_view(),
    })


@app.get("/workspace", response_class=HTMLResponse)
async def page_workspace(request: Request):
    """保留旧地址，避免已有书签和链接失效。"""
    return templates.TemplateResponse(request, "workspace.html", {
        "settings": settings_service.public_view(),
    })


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _task_to_dict(task: Task) -> dict:
    d = dataclasses.asdict(task)
    # 计算整体完成百分比
    d["completion_percent"] = calculate_completion_percent(task.plan)
    return d


# ──────────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────────

def _port_owners(port: int) -> list[str]:
    """尽力查找监听端口的进程，查询失败时返回空列表。"""
    try:
        import psutil

        owners = []
        seen = set()
        for connection in psutil.net_connections(kind="inet"):
            if not connection.laddr or connection.laddr.port != port:
                continue
            if str(connection.status).upper() != "LISTEN":
                continue
            pid = connection.pid
            if pid in seen:
                continue
            seen.add(pid)
            if pid is None:
                owners.append("系统进程")
                continue
            try:
                owners.append(f"PID {pid} ({psutil.Process(pid).name()})")
            except (psutil.Error, OSError):
                owners.append(f"PID {pid}（进程信息不可用）")
        return owners
    except Exception:
        return []


def check_port_available(host: str, port: int) -> tuple[bool, str]:
    """在启动 Uvicorn 前检查监听地址，避免产生第二个服务进程。"""
    probe_host = host.strip() or "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind((probe_host, port))
        return True, ""
    except OSError as exc:
        owners = _port_owners(port)
        owner_text = "、".join(owners) if owners else "未能识别占用进程"
        return False, f"{probe_host}:{port} 已被占用（{owner_text}；{exc}）"


def run_server() -> int:
    """运行单实例 Web 服务，并提供可操作的启动提示。"""
    import uvicorn

    runtime_settings = get_settings()
    available, reason = check_port_available(
        runtime_settings.server_host,
        runtime_settings.server_port,
    )
    if not available:
        print("\n[启动失败] Work Assistant 无法监听配置的端口。", file=sys.stderr)
        print(f"原因：{reason}", file=sys.stderr)
        print("请关闭已有服务，或在 .env 中修改 SERVER_PORT 后重试。\n", file=sys.stderr)
        return 1

    browser_host = (
        "localhost"
        if runtime_settings.server_host in {"0.0.0.0", "127.0.0.1"}
        else runtime_settings.server_host
    )
    print(f"\nWork Assistant 正在启动：http://{browser_host}:{runtime_settings.server_port}")
    print(f"健康检查：http://{browser_host}:{runtime_settings.server_port}/health")
    print(f"热重载：{'开启' if runtime_settings.server_reload else '关闭（单进程稳定模式）'}\n")
    uvicorn.run(
        "server:app" if runtime_settings.server_reload else app,
        host=runtime_settings.server_host,
        port=runtime_settings.server_port,
        reload=runtime_settings.server_reload,
        reload_dirs=[str(BASE_DIR)] if runtime_settings.server_reload else None,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(run_server())
