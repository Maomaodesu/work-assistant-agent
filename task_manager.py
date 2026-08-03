"""
任务管理器：Task / PlanStep 的 SQLite 持久化 + LLM 任务初始化对话。
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from settings import get_api_key, get_settings
from streaming_runtime import stream_chat_completion_text

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装：pip install openai")


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class SubTask:
    name: str
    status: str = "pending"   # pending | completed


@dataclass
class PlanStep:
    step_id: str
    step_index: int
    step_name: str
    description: str
    status: str = "pending"   # pending | in_progress | completed | skipped
    sub_tasks: list = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    notes: str = ""


@dataclass
class Task:
    task_id: str
    task_name: str
    project_paths: list          # 支持多项目
    project_types: list          # spring-boot / vue / python ...
    goal: str                    # 用户的目标描述
    tech_stack: dict             # {backend, frontend, database, ...}
    status: str                  # pending | active | suspended | completed
    priority: str                # P0 | P1 | P2
    created_at: str
    last_active_at: str
    total_work_seconds: int
    interrupt_count: int
    plan: list                   # list[PlanStep]
    current_step_index: int
    notes: str = ""


# ──────────────────────────────────────────────
# 数据库
# ──────────────────────────────────────────────

DB_PATH = get_settings().agent_db_path


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id            TEXT PRIMARY KEY,
            task_name          TEXT NOT NULL,
            project_paths_json TEXT,
            project_types_json TEXT,
            goal               TEXT,
            tech_stack_json    TEXT,
            status             TEXT NOT NULL DEFAULT 'pending',
            priority           TEXT NOT NULL DEFAULT 'P1',
            created_at         TEXT NOT NULL,
            last_active_at     TEXT,
            total_work_seconds INTEGER DEFAULT 0,
            interrupt_count    INTEGER DEFAULT 0,
            plan_json          TEXT,
            current_step_index INTEGER DEFAULT 0,
            notes              TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS work_sessions (
            session_id       TEXT PRIMARY KEY,
            task_id          TEXT NOT NULL REFERENCES tasks(task_id),
            started_at       TEXT NOT NULL,
            ended_at         TEXT,
            duration_seconds INTEGER,
            end_reason       TEXT
        );
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────

def save_task(task: Task):
    conn = _get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO tasks VALUES (
            :task_id, :task_name, :project_paths_json, :project_types_json,
            :goal, :tech_stack_json, :status, :priority,
            :created_at, :last_active_at, :total_work_seconds,
            :interrupt_count, :plan_json, :current_step_index, :notes
        )
    """, {
        "task_id":             task.task_id,
        "task_name":           task.task_name,
        "project_paths_json":  json.dumps(task.project_paths, ensure_ascii=False),
        "project_types_json":  json.dumps(task.project_types, ensure_ascii=False),
        "goal":                task.goal,
        "tech_stack_json":     json.dumps(task.tech_stack, ensure_ascii=False),
        "status":              task.status,
        "priority":            task.priority,
        "created_at":          task.created_at,
        "last_active_at":      task.last_active_at,
        "total_work_seconds":  task.total_work_seconds,
        "interrupt_count":     task.interrupt_count,
        "plan_json":           json.dumps([asdict(s) for s in task.plan], ensure_ascii=False),
        "current_step_index":  task.current_step_index,
        "notes":               task.notes,
    })
    conn.commit()
    conn.close()


def load_task(task_id: str) -> Optional[Task]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_task(row)


def list_tasks(status: Optional[str] = None) -> list:
    conn = _get_conn()
    if status:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY last_active_at DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY last_active_at DESC"
        ).fetchall()
    conn.close()
    return [_row_to_task(r) for r in rows]


def _row_to_task(row) -> Task:
    raw_plan = json.loads(row["plan_json"] or "[]")
    plan = [
        PlanStep(
            step_id=s["step_id"],
            step_index=s["step_index"],
            step_name=s["step_name"],
            description=s["description"],
            status=s.get("status", "pending"),
            sub_tasks=s.get("sub_tasks", []),
            started_at=s.get("started_at"),
            completed_at=s.get("completed_at"),
            notes=s.get("notes", ""),
        )
        for s in raw_plan
    ]
    return Task(
        task_id=row["task_id"],
        task_name=row["task_name"],
        project_paths=json.loads(row["project_paths_json"] or "[]"),
        project_types=json.loads(row["project_types_json"] or "[]"),
        goal=row["goal"] or "",
        tech_stack=json.loads(row["tech_stack_json"] or "{}"),
        status=row["status"],
        priority=row["priority"],
        created_at=row["created_at"],
        last_active_at=row["last_active_at"] or row["created_at"],
        total_work_seconds=row["total_work_seconds"] or 0,
        interrupt_count=row["interrupt_count"] or 0,
        plan=plan,
        current_step_index=row["current_step_index"] or 0,
        notes=row["notes"] or "",
    )


def update_step_status(task_id: str, step_index: int, status: str, notes: str = ""):
    task = load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    for step in task.plan:
        if step.step_index == step_index:
            step.status = status
            step.notes = notes
            if status == "in_progress" and not step.started_at:
                step.started_at = datetime.now().isoformat()
            if status == "completed":
                step.completed_at = datetime.now().isoformat()
                task.current_step_index = step_index + 1
    task.last_active_at = datetime.now().isoformat()
    save_task(task)


PROGRESS_STATUS_WEIGHTS = {
    "pending": 0.0,
    "in_progress": 0.5,
    "completed": 1.0,
    "skipped": 1.0,
}


def calculate_completion_percent(plan: list) -> int:
    """根据持久化的步骤状态计算任务进度，保证所有入口口径一致。"""
    if not plan:
        return 0
    total = sum(PROGRESS_STATUS_WEIGHTS.get(step.status, 0.0) for step in plan)
    return round(total / len(plan) * 100)


def apply_progress_report(task_id: str, report: dict) -> dict:
    """把 LLM 进度报告原子地应用到任务，并返回数据库口径的规范化报告。"""
    if not isinstance(report, dict):
        raise TypeError("progress report must be a dict")

    raw_statuses = report.get("step_statuses")
    if not isinstance(raw_statuses, list):
        raise ValueError("progress report is missing step_statuses")

    task = load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    steps_by_index = {step.step_index: step for step in task.plan}
    updates = {}
    evidence_by_index = {}
    for item in raw_statuses:
        if not isinstance(item, dict):
            continue
        try:
            step_index = int(item.get("step_index"))
        except (TypeError, ValueError):
            continue

        status = item.get("status")
        if step_index not in steps_by_index or status not in PROGRESS_STATUS_WEIGHTS:
            continue

        updates[step_index] = status
        evidence = item.get("evidence", "")
        if isinstance(evidence, str):
            evidence_by_index[step_index] = evidence.strip()

    if task.plan and not updates:
        raise ValueError("progress report contains no valid step statuses")

    now = datetime.now().isoformat()
    for step in task.plan:
        if step.step_index not in updates:
            continue

        new_status = updates[step.step_index]
        step.status = new_status
        evidence = evidence_by_index.get(step.step_index, "")
        if evidence:
            step.notes = evidence

        if new_status in ("in_progress", "completed") and not step.started_at:
            step.started_at = now
        if new_status in ("completed", "skipped"):
            if not step.completed_at:
                step.completed_at = now
        else:
            step.completed_at = None
        if new_status == "pending":
            step.started_at = None

    current_step = next(
        (step for step in task.plan if step.status == "in_progress"),
        None,
    ) or next(
        (step for step in task.plan if step.status == "pending"),
        None,
    )

    task.current_step_index = current_step.step_index if current_step else len(task.plan)
    if task.plan and all(step.status in ("completed", "skipped") for step in task.plan):
        task.status = "completed"
    elif task.status == "completed":
        task.status = "active"
    task.last_active_at = now
    save_task(task)

    normalized = dict(report)
    normalized["current_step_index"] = task.current_step_index
    normalized["current_step_name"] = current_step.step_name if current_step else "全部完成"
    normalized["completion_percent"] = calculate_completion_percent(task.plan)
    normalized["step_statuses"] = [
        {
            "step_index": step.step_index,
            "step_name": step.step_name,
            "status": step.status,
            "evidence": evidence_by_index.get(step.step_index, ""),
        }
        for step in task.plan
    ]
    return normalized


# ──────────────────────────────────────────────
# LLM 任务初始化
# ──────────────────────────────────────────────

def _load_api_key() -> str:
    return get_api_key()


def _load_prompt(filename: str) -> str:
    return (Path(__file__).parent / filename).read_text(encoding="utf-8")


def create_task_via_llm(
    user_description: str,
    project_paths: list,
    model: str | None = None,) -> Task:
    """
    用户输入任务描述，LLM解析并生成执行计划，返回已存库的 Task。
    """
    from snapshot_collector import detect_project_type

    project_types = [detect_project_type(p) for p in project_paths]

    settings = get_settings()
    api_key = _load_api_key()
    model = model or settings.amd_model
    sys_prompt = _load_prompt("task_init_prompt.txt")

    user_content = (
        f"任务描述：{user_description}\n\n"
        f"项目路径：{json.dumps(project_paths, ensure_ascii=False)}\n"
        f"项目类型：{json.dumps(project_types, ensure_ascii=False)}"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=settings.amd_base_url,
    )
    raw = stream_chat_completion_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2048,
        timeout=settings.request_timeout_seconds,
    )
    raw = raw.strip()
    # 去掉可能的 markdown 代码块
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[:-1])

    parsed = json.loads(raw)

    now = datetime.now().isoformat()
    task_id = f"TASK-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    plan = []
    for i, s in enumerate(parsed.get("plan", [])):
        plan.append(PlanStep(
            step_id=f"{task_id}-STEP-{i+1:02d}",
            step_index=i,
            step_name=s["step_name"],
            description=s.get("description", ""),
            sub_tasks=s.get("sub_tasks", []),
        ))

    task = Task(
        task_id=task_id,
        task_name=parsed.get("task_name", user_description[:40]),
        project_paths=project_paths,
        project_types=project_types,
        goal=parsed.get("goal", user_description),
        tech_stack=parsed.get("tech_stack", {}),
        status="active",
        priority=parsed.get("priority", "P1"),
        created_at=now,
        last_active_at=now,
        total_work_seconds=0,
        interrupt_count=0,
        plan=plan,
        current_step_index=0,
        notes=parsed.get("notes", ""),
    )

    init_db()
    save_task(task)
    return task


def init_existing_project_via_llm(
    conversation_context: str,
    project_paths: list,
    snapshot: dict,
    model: str | None = None,) -> Task:
    """
    对已有项目，结合对话上下文 + 快照，让LLM推断任务目标和计划。
    """
    from snapshot_collector import detect_project_type

    project_types = [detect_project_type(p) for p in project_paths]
    settings = get_settings()
    api_key = _load_api_key()
    model = model or settings.amd_model
    sys_prompt = _load_prompt("task_init_prompt.txt")

    # 简化快照，只传模块列表和最近提交
    summary = {
        "projects": [
            {
                "name": p["project_name"],
                "type": p["project_type"],
                "branch": p["branch"],
                "latest_commit": (
                    p["commits"][0]
                    if settings.send_commit_info and p["commits"]
                    else None
                ),
                "recent_modules": [
                    {"module": m["module_name"], "layers": m["layers_present"], "status": m["git_status"]}
                    for m in p["modules"][:10]
                ],
                "uncommitted_count": len(p["uncommitted_tracked"]),
            }
            for p in snapshot.get("projects", [])
        ]
    }

    user_content = (
        f"这是一个已有项目，请根据对话上下文和快照信息推断任务目标并生成执行计划。\n\n"
        f"对话上下文：\n{conversation_context}\n\n"
        f"项目路径：{json.dumps(project_paths, ensure_ascii=False)}\n"
        f"项目类型：{json.dumps(project_types, ensure_ascii=False)}\n\n"
        f"项目快照摘要：\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=settings.amd_base_url,
    )
    raw = stream_chat_completion_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2048,
        timeout=settings.request_timeout_seconds,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[:-1])

    parsed = json.loads(raw)

    now = datetime.now().isoformat()
    task_id = f"TASK-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    plan = []
    for i, s in enumerate(parsed.get("plan", [])):
        plan.append(PlanStep(
            step_id=f"{task_id}-STEP-{i+1:02d}",
            step_index=i,
            step_name=s["step_name"],
            description=s.get("description", ""),
            sub_tasks=s.get("sub_tasks", []),
        ))

    task = Task(
        task_id=task_id,
        task_name=parsed.get("task_name", "已有项目任务"),
        project_paths=project_paths,
        project_types=project_types,
        goal=parsed.get("goal", conversation_context[:100]),
        tech_stack=parsed.get("tech_stack", {}),
        status="active",
        priority=parsed.get("priority", "P1"),
        created_at=now,
        last_active_at=now,
        total_work_seconds=0,
        interrupt_count=0,
        plan=plan,
        current_step_index=0,
        notes=parsed.get("notes", ""),
    )

    init_db()
    save_task(task)
    return task
