"""本地会话索引，以及 Claude Code / Codex JSONL 会话导入。"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from settings import get_settings


MAX_IMPORTED_SESSIONS_PER_SOURCE = 200
MAX_MESSAGES_PER_SESSION = 500
MAX_MESSAGE_CHARS = 20_000
PROJECT_MARKERS = (
    ".git", ".idea", "pom.xml", "pyproject.toml", "package.json",
    "requirements.txt", "Cargo.toml", "go.mod", "build.gradle",
)


def _now() -> str:
    return datetime.now().isoformat()


def _truncate(text: str) -> str:
    normalized = text.strip()
    if len(normalized) <= MAX_MESSAGE_CHARS:
        return normalized
    return normalized[:MAX_MESSAGE_CHARS] + "\n...[内容已截断]"


def _title_from_messages(messages: list[dict], fallback: str) -> str:
    first_user = next((item["content"] for item in messages if item["role"] == "user"), "")
    title = " ".join(first_user.split())[:48]
    return title or fallback


def _text_blocks(content) -> str:
    if isinstance(content, str):
        return _truncate(content)
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return _truncate("\n".join(texts)) if texts else ""


def detect_project(path_value: str, *, force_project: bool = False) -> dict:
    """从会话 cwd 向上寻找最近的项目根；找不到则视为随意会话。"""
    if not path_value:
        return {"group_type": "casual", "project_key": "", "project_name": ""}
    path = Path(path_value).expanduser()
    if path.is_file():
        path = path.parent
    candidates = [path, *path.parents]
    project_root = next(
        (
            candidate
            for candidate in candidates
            if any((candidate / marker).exists() for marker in PROJECT_MARKERS)
        ),
        None,
    )
    if project_root is None and force_project:
        project_root = path
    if project_root is None:
        return {"group_type": "casual", "project_key": "", "project_name": ""}
    absolute = os.path.abspath(str(project_root))
    return {
        "group_type": "project",
        "project_key": os.path.normcase(absolute),
        "project_name": Path(absolute).name or absolute,
        "project_path": absolute,
    }


def parse_codex_session(path: Path, title_index: dict[str, str] | None = None) -> dict | None:
    metadata = {}
    messages = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_type = item.get("type")
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if record_type == "session_meta":
                metadata = payload
                continue
            if record_type != "event_msg":
                continue
            event_type = payload.get("type")
            if event_type == "user_message":
                role, content = "user", payload.get("message", "")
            elif event_type == "agent_message":
                role, content = "assistant", payload.get("message", "")
            else:
                continue
            if isinstance(content, str) and content.strip():
                messages.append({
                    "role": role,
                    "content": _truncate(content),
                    "created_at": item.get("timestamp") or metadata.get("timestamp") or _now(),
                })
            if len(messages) >= MAX_MESSAGES_PER_SESSION:
                break

    source = metadata.get("source")
    if metadata.get("parent_thread_id") or (isinstance(source, dict) and source.get("subagent")):
        return None
    session_id = metadata.get("id") or metadata.get("session_id") or path.stem
    if not messages:
        return None
    indexed_title = (title_index or {}).get(session_id)
    return {
        "id": f"codex:{session_id}",
        "source": "codex",
        "title": indexed_title or _title_from_messages(messages, "Codex 会话"),
        "project_path": metadata.get("cwd") or "",
        "created_at": metadata.get("timestamp") or messages[0]["created_at"],
        "updated_at": messages[-1]["created_at"],
        "messages": messages,
    }


def parse_claude_session(path: Path) -> dict | None:
    session_id = path.stem
    project_path = ""
    messages = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("isSidechain"):
                continue
            record_type = item.get("type")
            if record_type not in {"user", "assistant"}:
                continue
            message = item.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role") or record_type
            if role not in {"user", "assistant"}:
                continue
            content = _text_blocks(message.get("content"))
            if not content:
                continue
            session_id = item.get("sessionId") or session_id
            project_path = item.get("cwd") or project_path
            messages.append({
                "role": role,
                "content": content,
                "created_at": item.get("timestamp") or _now(),
            })
            if len(messages) >= MAX_MESSAGES_PER_SESSION:
                break
    if not messages:
        return None
    return {
        "id": f"claude:{session_id}",
        "source": "claude",
        "title": _title_from_messages(messages, "Claude 会话"),
        "project_path": project_path,
        "created_at": messages[0]["created_at"],
        "updated_at": messages[-1]["created_at"],
        "messages": messages,
    }


def load_codex_title_index(codex_root: Path) -> dict[str, str]:
    index_path = codex_root / "session_index.jsonl"
    if not index_path.exists():
        return {}
    titles = {}
    with index_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("id") and item.get("thread_name"):
                titles[str(item["id"])] = str(item["thread_name"])
    return titles


class ConversationStore:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or get_settings().conversation_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id                 TEXT PRIMARY KEY,
                    source             TEXT NOT NULL,
                    title              TEXT NOT NULL,
                    project_path       TEXT DEFAULT '',
                    source_path        TEXT UNIQUE,
                    source_modified_ns INTEGER DEFAULT 0,
                    created_at         TEXT NOT NULL,
                    updated_at         TEXT NOT NULL,
                    message_count      INTEGER DEFAULT 0,
                    preview            TEXT DEFAULT '',
                    linked_task_id     TEXT,
                    readonly           INTEGER NOT NULL DEFAULT 0,
                    group_type         TEXT NOT NULL DEFAULT 'casual',
                    project_key        TEXT DEFAULT '',
                    project_name       TEXT DEFAULT '',
                    archived           INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    message_index   INTEGER NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, message_index)
                );
                CREATE TABLE IF NOT EXISTS conversation_comments (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    prompt          TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_comments_conversation
                ON conversation_comments(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
                CREATE TABLE IF NOT EXISTS ignored_external_sources (
                    source_path TEXT PRIMARY KEY,
                    ignored_at TEXT NOT NULL
                );
            """)
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(conversations)")
            }
            migrations = {
                "group_type": "ALTER TABLE conversations ADD COLUMN group_type TEXT NOT NULL DEFAULT 'casual'",
                "project_key": "ALTER TABLE conversations ADD COLUMN project_key TEXT DEFAULT ''",
                "project_name": "ALTER TABLE conversations ADD COLUMN project_name TEXT DEFAULT ''",
                "archived": "ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0",
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    conn.execute(statement)
        self.reclassify_all()

    def reclassify_all(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT id, project_path, linked_task_id FROM conversations").fetchall()
            for row in rows:
                classification = detect_project(
                    row["project_path"] or "",
                    force_project=bool(row["linked_task_id"] and row["project_path"]),
                )
                conn.execute(
                    """
                    UPDATE conversations
                    SET group_type = ?, project_key = ?, project_name = ?,
                        project_path = CASE WHEN ? <> '' THEN ? ELSE project_path END
                    WHERE id = ?
                    """,
                    (
                        classification["group_type"],
                        classification["project_key"],
                        classification["project_name"],
                        classification.get("project_path", ""),
                        classification.get("project_path", ""),
                        row["id"],
                    ),
                )

    def ensure_work_assistant(self, session_id: str, first_message: str = "") -> dict:
        now = _now()
        title = " ".join(first_message.split())[:48] or "新对话"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(
                    id, source, title, created_at, updated_at, readonly
                ) VALUES (?, 'work_assistant', ?, ?, ?, 0)
                ON CONFLICT(id) DO NOTHING
                """,
                (session_id, title, now, now),
            )
        return self.get(session_id)

    def append_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        linked_task_id: str | None = None,
        project_path: str | None = None,
    ):
        self.ensure_work_assistant(session_id, user_message)
        now = _now()
        classification = detect_project(
            project_path or "",
            force_project=bool(project_path and linked_task_id),
        )
        with self._connect() as conn:
            next_index = conn.execute(
                "SELECT COALESCE(MAX(message_index), -1) + 1 FROM conversation_messages WHERE conversation_id = ?",
                (session_id,),
            ).fetchone()[0]
            rows = [
                (session_id, next_index, "user", _truncate(user_message), now),
                (session_id, next_index + 1, "assistant", _truncate(assistant_message), now),
            ]
            conn.executemany(
                "INSERT INTO conversation_messages VALUES (?, ?, ?, ?, ?)", rows
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?, message_count = message_count + 2,
                    preview = ?, linked_task_id = COALESCE(?, linked_task_id),
                    project_path = CASE WHEN ? <> '' THEN ? ELSE project_path END,
                    group_type = CASE WHEN ? <> '' THEN ? ELSE group_type END,
                    project_key = CASE WHEN ? <> '' THEN ? ELSE project_key END,
                    project_name = CASE WHEN ? <> '' THEN ? ELSE project_name END,
                    title = CASE
                        WHEN message_count = 0 AND title = '新对话' THEN ?
                        ELSE title
                    END
                WHERE id = ?
                """,
                (
                    now,
                    _truncate(assistant_message)[:160],
                    linked_task_id,
                    classification.get("project_path", ""),
                    classification.get("project_path", ""),
                    project_path or "",
                    classification["group_type"],
                    project_path or "",
                    classification["project_key"],
                    project_path or "",
                    classification["project_name"],
                    " ".join(user_message.split())[:48] or "新对话",
                    session_id,
                ),
            )

    def list(self, limit: int = 300) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def projects(self) -> list[dict]:
        """聚合全部项目会话，作为本地项目工作台的可信项目来源。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversations
                WHERE group_type = 'project' AND project_path <> ''
                ORDER BY updated_at DESC
                """
            ).fetchall()
        projects = {}
        for row in rows:
            item = self._row(row)
            key = item.get("project_key") or os.path.normcase(
                os.path.abspath(item["project_path"])
            )
            project = projects.setdefault(key, {
                "project_key": key,
                "project_name": item.get("project_name") or Path(item["project_path"]).name,
                "project_path": item["project_path"],
                "updated_at": item["updated_at"],
                "conversation_count": 0,
                "active_conversation_count": 0,
                "sources": set(),
            })
            project["conversation_count"] += 1
            if not item.get("archived"):
                project["active_conversation_count"] += 1
            project["sources"].add(item["source"])
        result = []
        for project in projects.values():
            project["sources"] = sorted(project["sources"])
            project["path_available"] = Path(project["project_path"]).is_dir()
            result.append(project)
        return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def get_project(self, project_key: str) -> dict | None:
        normalized = str(project_key or "").strip()
        return next(
            (project for project in self.projects() if project["project_key"] == normalized),
            None,
        )

    def get(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return self._row(row) if row else None

    def messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at FROM conversation_messages
                WHERE conversation_id = ? ORDER BY message_index
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def comments(self, conversation_id: str) -> list[dict]:
        """返回 work_assistant 针对导入历史留下的分析评论。

        这些评论与原始 Codex / Claude 记录分表保存，避免同步外部会话时覆盖
        或改写原始历史。
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, prompt, content, created_at
                FROM conversation_comments
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_comment(self, conversation_id: str, prompt: str, content: str) -> dict:
        if not self.get(conversation_id):
            raise KeyError(conversation_id)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_comments(conversation_id, prompt, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, _truncate(prompt), _truncate(content), _now()),
            )
            row = conn.execute(
                """
                SELECT id, prompt, content, created_at
                FROM conversation_comments WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        return dict(row)

    def delete_comment(self, conversation_id: str, comment_id: int) -> bool:
        """Delete only one analysis comment that belongs to this conversation."""
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM conversation_comments WHERE id = ? AND conversation_id = ?",
                (int(comment_id), conversation_id),
            ).rowcount
        return bool(deleted)

    def rename(self, conversation_id: str, title: str) -> dict:
        normalized = " ".join(title.split())[:80]
        if not normalized:
            raise ValueError("会话名称不能为空")
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (normalized, _now(), conversation_id),
            ).rowcount
        if not changed:
            raise KeyError(conversation_id)
        return self.get(conversation_id)

    def set_archived(self, conversation_id: str, archived: bool) -> dict:
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE conversations SET archived = ?, updated_at = ? WHERE id = ?",
                (int(archived), _now(), conversation_id),
            ).rowcount
        if not changed:
            raise KeyError(conversation_id)
        return self.get(conversation_id)

    @staticmethod
    def _default_source_roots(source: str) -> tuple[Path, ...]:
        if source == "codex":
            return (Path.home() / ".codex" / "sessions",)
        if source == "claude":
            return (Path.home() / ".claude" / "projects",)
        return ()

    def delete(
        self,
        conversation_id: str,
        *,
        delete_source: bool = False,
        allowed_source_roots: tuple[Path, ...] | None = None,
    ) -> dict | None:
        conversation = self.get(conversation_id)
        if not conversation:
            return None
        source_deleted = False
        if delete_source:
            if conversation["source"] not in {"codex", "claude"} or not conversation["readonly"]:
                raise ValueError("只有导入的 Codex 或 Claude 历史可删除本机原始记录")
            source_path = Path(str(conversation.get("source_path") or "")).expanduser()
            if source_path.suffix.lower() != ".jsonl" or not source_path.is_file():
                raise ValueError("找不到可信的本机原始会话文件，无法删除")
            resolved_source = source_path.resolve()
            trusted_roots = allowed_source_roots or self._default_source_roots(conversation["source"])
            if not any(
                resolved_source.is_relative_to(Path(root).expanduser().resolve())
                for root in trusted_roots
                if Path(root).expanduser().is_dir()
            ):
                raise ValueError("原始会话文件不在允许的 Codex/Claude 历史目录中")
            # This is deliberately a single, verified JSONL file; never delete a directory or glob.
            resolved_source.unlink()
            source_deleted = True
        with self._connect() as conn:
            if conversation["readonly"] and conversation.get("source_path") and not source_deleted:
                conn.execute(
                    "INSERT OR REPLACE INTO ignored_external_sources VALUES (?, ?)",
                    (conversation["source_path"], _now()),
                )
            elif source_deleted:
                conn.execute(
                    "DELETE FROM ignored_external_sources WHERE source_path = ?",
                    (conversation["source_path"],),
                )
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return {**conversation, "source_deleted": source_deleted}

    def _source_is_ignored(self, path: Path) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ignored_external_sources WHERE source_path = ?",
                (str(path),),
            ).fetchone()
        return bool(row)

    def _source_is_current(self, path: Path) -> bool:
        modified_ns = path.stat().st_mtime_ns
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source_modified_ns FROM conversations WHERE source_path = ?",
                (str(path),),
            ).fetchone()
        return bool(row and row[0] == modified_ns)

    def _upsert_external(self, parsed: dict, path: Path):
        messages = parsed.pop("messages")
        classification = detect_project(parsed["project_path"])
        if classification["group_type"] == "project":
            parsed["project_path"] = classification["project_path"]
        modified_ns = path.stat().st_mtime_ns
        preview = messages[-1]["content"][:160] if messages else ""
        with self._connect() as conn:
            old = conn.execute(
                "SELECT id FROM conversations WHERE source_path = ?", (str(path),)
            ).fetchone()
            if old and old["id"] != parsed["id"]:
                conn.execute("DELETE FROM conversations WHERE id = ?", (old["id"],))
            conn.execute(
                """
                INSERT INTO conversations(
                    id, source, title, project_path, source_path, source_modified_ns,
                    created_at, updated_at, message_count, preview, readonly,
                    group_type, project_key, project_name, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    project_path = excluded.project_path,
                    source_path = excluded.source_path,
                    source_modified_ns = excluded.source_modified_ns,
                    updated_at = excluded.updated_at,
                    message_count = excluded.message_count,
                    preview = excluded.preview,
                    group_type = excluded.group_type,
                    project_key = excluded.project_key,
                    project_name = excluded.project_name
                """,
                (
                    parsed["id"], parsed["source"], parsed["title"],
                    parsed["project_path"], str(path), modified_ns,
                    parsed["created_at"], parsed["updated_at"], len(messages), preview,
                    classification["group_type"],
                    classification["project_key"],
                    classification["project_name"],
                ),
            )
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (parsed["id"],),
            )
            conn.executemany(
                "INSERT INTO conversation_messages VALUES (?, ?, ?, ?, ?)",
                [
                    (parsed["id"], index, item["role"], item["content"], item["created_at"])
                    for index, item in enumerate(messages)
                ],
            )

    def import_external(
        self,
        codex_root: Path | None = None,
        claude_root: Path | None = None,
    ) -> dict:
        result = {"codex": 0, "claude": 0, "skipped": 0, "errors": 0}
        codex_root = codex_root or Path.home() / ".codex"
        title_index = load_codex_title_index(codex_root)
        codex_files = sorted(
            (codex_root / "sessions").rglob("*.jsonl") if (codex_root / "sessions").exists() else [],
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:MAX_IMPORTED_SESSIONS_PER_SOURCE]
        claude_root = claude_root or Path.home() / ".claude" / "projects"
        claude_files = sorted(
            claude_root.glob("*/*.jsonl") if claude_root.exists() else [],
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:MAX_IMPORTED_SESSIONS_PER_SOURCE]

        for source, files, parser in (
            ("codex", codex_files, lambda path: parse_codex_session(path, title_index)),
            ("claude", claude_files, parse_claude_session),
        ):
            for path in files:
                try:
                    if self._source_is_ignored(path):
                        result["skipped"] += 1
                        continue
                    if self._source_is_current(path):
                        result["skipped"] += 1
                        continue
                    parsed = parser(path)
                    if not parsed:
                        result["skipped"] += 1
                        continue
                    self._upsert_external(parsed, path)
                    result[source] += 1
                except Exception:
                    result["errors"] += 1
        return result

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["readonly"] = bool(item["readonly"])
        item["archived"] = bool(item["archived"])
        return item


conversation_store = ConversationStore()
