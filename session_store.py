"""LangGraph 对话线程的 SQLite 持久化配置。"""

import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from settings import get_settings


CHECKPOINT_DB_PATH = get_settings().checkpoint_db_path


def create_sqlite_checkpointer(
    db_path: Path | str = CHECKPOINT_DB_PATH,
) -> tuple[SqliteSaver, sqlite3.Connection]:
    """创建可跨线程使用的 SQLite Checkpointer。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer, conn


def thread_config(session_id: str) -> dict:
    """把 Web session_id 映射为 LangGraph thread_id。"""
    normalized = session_id.strip()
    if not normalized:
        raise ValueError("session_id cannot be empty")
    return {"configurable": {"thread_id": normalized}}


def build_turn_input(user_message: str) -> dict:
    """只提交本轮增量；历史状态由 Checkpointer 自动恢复。"""
    return {
        "messages": [HumanMessage(content=user_message)],
        "snapshot": None,
        "snapshot_path": None,
        "progress_report": None,
        "output": "",
        "error": None,
    }
