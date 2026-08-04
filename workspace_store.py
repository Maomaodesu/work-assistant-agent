"""统一的项目、工作项、会话片段与自动整理任务存储层。"""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from settings import get_settings


SCHEMA_VERSION = 7
PROJECT_STATUSES = {"active", "archived"}
WORK_ITEM_TYPES = {
    "feature", "function", "bug", "refactor", "maintenance", "research", "other"
}
WORK_ITEM_STATUSES = {
    "suggested", "backlog", "planned", "in_progress", "blocked",
    "completed", "archived", "ignored",
}
WORK_ITEM_SOURCES = {"manual", "inferred"}
CONVERSATION_SOURCES = {"work_assistant", "codex", "claude"}
SEGMENT_KINDS = {"project", "work_item", "casual", "unclassified"}
SEGMENT_REVIEW_STATUSES = {"suggested", "confirmed", "ignored"}
SEGMENT_LINK_RELATIONS = {"primary", "mentioned", "evidence"}
CLASSIFICATION_RUN_STATUSES = {
    "queued", "running", "paused", "completed", "failed", "cancelled"
}
PRIVACY_MODES = {"strict_local", "balanced", "high_accuracy"}


class WorkspaceStoreError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _normalize_path(value: str | Path) -> str:
    expanded = os.path.expandvars(str(value).strip().strip('"'))
    if not expanded:
        raise WorkspaceStoreError("项目目录不能为空")
    return str(Path(expanded).expanduser().resolve())


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _message_content_hash(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Validate imported messages and attach the deterministic content hash."""
    normalized = []
    ordinals = set()
    for index, message in enumerate(messages):
        ordinal = int(message.get("ordinal", index))
        role = str(message.get("role", "")).lower()
        if role not in {"user", "assistant", "system"}:
            raise WorkspaceStoreError(f"消息角色不正确：{role}")
        if ordinal in ordinals:
            raise WorkspaceStoreError("会话消息序号不能重复")
        ordinals.add(ordinal)
        content = str(message.get("content", ""))
        normalized.append({
            "ordinal": ordinal,
            "role": role,
            "content": content,
            "created_at": str(message.get("created_at") or _now()),
            "content_hash": str(message.get("content_hash") or _message_content_hash(content)),
        })
    return normalized


def _segment_content_fingerprint(messages: list[dict]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(json.dumps(
            [
                int(message["ordinal"]), str(message["role"]).lower(),
                str(message["created_at"]),
                str(message.get("content_hash") or _message_content_hash(message.get("content", ""))),
            ],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
    return digest.hexdigest()


class WorkspaceStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or get_settings().workspace_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    project_id    TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    description   TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'active'
                                  CHECK(status IN ('active', 'archived')),
                    source        TEXT NOT NULL DEFAULT 'manual'
                                  CHECK(source IN ('manual', 'inferred')),
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    last_active_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_roots (
                    root_id         TEXT PRIMARY KEY,
                    project_id      TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    path            TEXT NOT NULL,
                    normalized_path TEXT NOT NULL UNIQUE,
                    label           TEXT NOT NULL DEFAULT '',
                    root_type       TEXT NOT NULL DEFAULT 'code',
                    is_primary      INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0, 1)),
                    created_at      TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_project_one_primary_root
                ON project_roots(project_id) WHERE is_primary = 1;

                CREATE TABLE IF NOT EXISTS project_definitions (
                    project_id           TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
                    status               TEXT NOT NULL DEFAULT 'draft'
                                         CHECK(status IN ('draft','confirmed','ignored')),
                    goal                 TEXT NOT NULL DEFAULT '',
                    scope                TEXT NOT NULL DEFAULT '',
                    non_goals            TEXT NOT NULL DEFAULT '',
                    acceptance_criteria  TEXT NOT NULL DEFAULT '',
                    constraints          TEXT NOT NULL DEFAULT '',
                    summary              TEXT NOT NULL DEFAULT '',
                    source               TEXT NOT NULL DEFAULT 'inferred'
                                         CHECK(source IN ('inferred','manual')),
                    source_segment_ids_json TEXT NOT NULL DEFAULT '[]',
                    source_run_id        TEXT NOT NULL DEFAULT '',
                    created_at           TEXT NOT NULL,
                    updated_at           TEXT NOT NULL,
                    confirmed_at         TEXT,
                    ignored_at           TEXT
                );

                CREATE TABLE IF NOT EXISTS work_items (
                    work_item_id   TEXT PRIMARY KEY,
                    project_id     TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    title          TEXT NOT NULL,
                    item_type      TEXT NOT NULL DEFAULT 'feature'
                                   CHECK(item_type IN ('feature','function','bug','refactor','maintenance','research','other')),
                    goal           TEXT NOT NULL DEFAULT '',
                    description    TEXT NOT NULL DEFAULT '',
                    status         TEXT NOT NULL DEFAULT 'suggested'
                                   CHECK(status IN ('suggested','backlog','planned','in_progress','blocked','completed','archived','ignored')),
                    source         TEXT NOT NULL DEFAULT 'manual'
                                   CHECK(source IN ('manual','inferred')),
                    confidence     REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                    priority       TEXT NOT NULL DEFAULT 'P1',
                    deadline       TEXT,
                    created_at     TEXT NOT NULL,
                    discovered_at  TEXT NOT NULL,
                    updated_at     TEXT NOT NULL,
                    last_active_at TEXT NOT NULL,
                    confirmed_at   TEXT,
                    ignored_at     TEXT,
                    ignore_reason  TEXT NOT NULL DEFAULT '',
                    completed_at   TEXT,
                    metadata_json  TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_work_items_project_status
                ON work_items(project_id, status, last_active_at DESC);

                CREATE TABLE IF NOT EXISTS work_item_steps (
                    step_id       TEXT PRIMARY KEY,
                    work_item_id  TEXT NOT NULL REFERENCES work_items(work_item_id) ON DELETE CASCADE,
                    step_index    INTEGER NOT NULL CHECK(step_index >= 0),
                    title         TEXT NOT NULL,
                    description   TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'pending'
                                  CHECK(status IN ('pending','in_progress','completed','skipped')),
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    completed_at  TEXT,
                    UNIQUE(work_item_id, step_index)
                );

                CREATE TABLE IF NOT EXISTS work_item_events (
                    event_id      TEXT PRIMARY KEY,
                    work_item_id  TEXT NOT NULL REFERENCES work_items(work_item_id) ON DELETE CASCADE,
                    event_type    TEXT NOT NULL,
                    actor         TEXT NOT NULL DEFAULT 'system',
                    details_json  TEXT NOT NULL DEFAULT '{}',
                    created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_work_item_events_item
                ON work_item_events(work_item_id, created_at);

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id    TEXT PRIMARY KEY,
                    source             TEXT NOT NULL
                                       CHECK(source IN ('work_assistant','codex','claude')),
                    external_session_id TEXT NOT NULL,
                    title              TEXT NOT NULL DEFAULT '',
                    original_project_path TEXT NOT NULL DEFAULT '',
                    source_path        TEXT NOT NULL DEFAULT '',
                    started_at         TEXT NOT NULL,
                    updated_at         TEXT NOT NULL,
                    imported_at        TEXT NOT NULL,
                    source_modified_ns INTEGER NOT NULL DEFAULT 0,
                    content_hash       TEXT NOT NULL DEFAULT '',
                    resume_capable     INTEGER NOT NULL DEFAULT 0 CHECK(resume_capable IN (0, 1)),
                    metadata_json      TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(source, external_session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_source_updated
                ON conversations(source, updated_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_source_path
                ON conversations(source_path) WHERE source_path <> '';

                CREATE TABLE IF NOT EXISTS conversation_project_matches (
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    project_id      TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    root_id         TEXT REFERENCES project_roots(root_id) ON DELETE SET NULL,
                    match_method    TEXT NOT NULL
                                    CHECK(match_method IN ('exact_root','root_descendant','manual')),
                    confidence      REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    is_primary      INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0, 1)),
                    match_source    TEXT NOT NULL DEFAULT 'auto'
                                    CHECK(match_source IN ('auto','manual')),
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    PRIMARY KEY(conversation_id, project_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_one_primary_project
                ON conversation_project_matches(conversation_id) WHERE is_primary = 1;
                CREATE INDEX IF NOT EXISTS idx_conversation_project_lookup
                ON conversation_project_matches(project_id, is_primary, conversation_id);

                CREATE TABLE IF NOT EXISTS external_source_state (
                    source_path        TEXT PRIMARY KEY,
                    source             TEXT NOT NULL CHECK(source IN ('codex','claude')),
                    source_modified_ns INTEGER NOT NULL DEFAULT 0,
                    content_hash       TEXT NOT NULL DEFAULT '',
                    import_status      TEXT NOT NULL
                                       CHECK(import_status IN ('imported','skipped','error')),
                    conversation_id    TEXT REFERENCES conversations(conversation_id) ON DELETE SET NULL,
                    last_error         TEXT NOT NULL DEFAULT '',
                    checked_at         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_external_source_status
                ON external_source_state(import_status, checked_at);

                CREATE TABLE IF NOT EXISTS messages (
                    message_id      TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    ordinal         INTEGER NOT NULL CHECK(ordinal >= 0),
                    role            TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    content_hash    TEXT NOT NULL DEFAULT '',
                    UNIQUE(conversation_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
                ON messages(conversation_id, ordinal);

                CREATE TABLE IF NOT EXISTS conversation_segments (
                    segment_id       TEXT PRIMARY KEY,
                    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    start_ordinal     INTEGER NOT NULL CHECK(start_ordinal >= 0),
                    end_ordinal       INTEGER NOT NULL CHECK(end_ordinal >= start_ordinal),
                    segment_kind     TEXT NOT NULL DEFAULT 'unclassified'
                                     CHECK(segment_kind IN ('project','work_item','casual','unclassified')),
                    project_id       TEXT REFERENCES projects(project_id) ON DELETE SET NULL,
                    title            TEXT NOT NULL DEFAULT '',
                    summary          TEXT NOT NULL DEFAULT '',
                    confidence       REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                    classification_source TEXT NOT NULL DEFAULT 'rules'
                                          CHECK(classification_source IN ('rules','amd','manual')),
                    review_status    TEXT NOT NULL DEFAULT 'suggested'
                                     CHECK(review_status IN ('suggested','confirmed','ignored')),
                    boundary_reason  TEXT NOT NULL DEFAULT 'conversation_start',
                    segmenter_version TEXT NOT NULL DEFAULT '',
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL DEFAULT '',
                    is_current       INTEGER NOT NULL DEFAULT 1 CHECK(is_current IN (0, 1)),
                    superseded_at    TEXT,
                    superseded_reason TEXT NOT NULL DEFAULT '',
                    UNIQUE(conversation_id, start_ordinal, end_ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_segments_project_review
                ON conversation_segments(project_id, is_current, review_status, conversation_id);

                CREATE TABLE IF NOT EXISTS conversation_segmentation_state (
                    conversation_id    TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    message_fingerprint TEXT NOT NULL,
                    segmenter_version   TEXT NOT NULL,
                    status              TEXT NOT NULL CHECK(status IN ('completed','error')),
                    segment_count       INTEGER NOT NULL DEFAULT 0 CHECK(segment_count >= 0),
                    error_message       TEXT NOT NULL DEFAULT '',
                    processed_at        TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_retrieval_chunks (
                    chunk_id          TEXT PRIMARY KEY,
                    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    segment_id       TEXT NOT NULL REFERENCES conversation_segments(segment_id) ON DELETE CASCADE,
                    chunk_index      INTEGER NOT NULL CHECK(chunk_index >= 0),
                    start_ordinal    INTEGER NOT NULL CHECK(start_ordinal >= 0),
                    end_ordinal      INTEGER NOT NULL CHECK(end_ordinal >= start_ordinal),
                    content          TEXT NOT NULL,
                    content_hash     TEXT NOT NULL,
                    char_count       INTEGER NOT NULL CHECK(char_count > 0),
                    chunker_version  TEXT NOT NULL,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL,
                    UNIQUE(segment_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_conversation
                ON conversation_retrieval_chunks(conversation_id, segment_id, chunk_index);

                CREATE TABLE IF NOT EXISTS conversation_retrieval_index_state (
                    conversation_id    TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    message_fingerprint TEXT NOT NULL,
                    chunker_version    TEXT NOT NULL,
                    status             TEXT NOT NULL CHECK(status IN ('completed','error')),
                    chunk_count        INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
                    error_message      TEXT NOT NULL DEFAULT '',
                    processed_at       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS segment_work_item_links (
                    segment_id    TEXT NOT NULL REFERENCES conversation_segments(segment_id) ON DELETE CASCADE,
                    work_item_id  TEXT NOT NULL REFERENCES work_items(work_item_id) ON DELETE CASCADE,
                    relation      TEXT NOT NULL DEFAULT 'primary'
                                  CHECK(relation IN ('primary','mentioned','evidence')),
                    confidence    REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                    created_at    TEXT NOT NULL,
                    PRIMARY KEY(segment_id, work_item_id, relation)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_segment_one_primary_work_item
                ON segment_work_item_links(segment_id) WHERE relation = 'primary';
                CREATE INDEX IF NOT EXISTS idx_segment_links_work_item
                ON segment_work_item_links(work_item_id, relation);

                CREATE TABLE IF NOT EXISTS classification_runs (
                    run_id               TEXT PRIMARY KEY,
                    run_type             TEXT NOT NULL CHECK(run_type IN ('full','incremental')),
                    status               TEXT NOT NULL DEFAULT 'queued'
                                         CHECK(status IN ('queued','running','paused','completed','failed','cancelled')),
                    privacy_mode         TEXT NOT NULL DEFAULT 'balanced'
                                         CHECK(privacy_mode IN ('strict_local','balanced','high_accuracy')),
                    stage                TEXT NOT NULL DEFAULT 'queued',
                    total_sources        INTEGER NOT NULL DEFAULT 0 CHECK(total_sources >= 0),
                    processed_sources    INTEGER NOT NULL DEFAULT 0 CHECK(processed_sources >= 0),
                    discovered_count     INTEGER NOT NULL DEFAULT 0 CHECK(discovered_count >= 0),
                    unclassified_count   INTEGER NOT NULL DEFAULT 0 CHECK(unclassified_count >= 0),
                    amd_call_count       INTEGER NOT NULL DEFAULT 0 CHECK(amd_call_count >= 0),
                    credential_redaction_count INTEGER NOT NULL DEFAULT 0
                                               CHECK(credential_redaction_count >= 0),
                    request_json         TEXT NOT NULL DEFAULT '{}',
                    retry_of_run_id      TEXT,
                    error_message        TEXT NOT NULL DEFAULT '',
                    created_at           TEXT NOT NULL,
                    started_at           TEXT,
                    finished_at          TEXT,
                    updated_at           TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_classification_runs_status
                ON classification_runs(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS segment_classification_state (
                    segment_id          TEXT PRIMARY KEY
                                        REFERENCES conversation_segments(segment_id) ON DELETE CASCADE,
                    segment_fingerprint TEXT NOT NULL,
                    classifier_version  TEXT NOT NULL,
                    status              TEXT NOT NULL
                                        CHECK(status IN ('classified','skipped','error')),
                    decision_json       TEXT NOT NULL DEFAULT '{}',
                    redaction_json      TEXT NOT NULL DEFAULT '{}',
                    run_id              TEXT REFERENCES classification_runs(run_id) ON DELETE SET NULL,
                    error_message       TEXT NOT NULL DEFAULT '',
                    processed_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_segment_classification_status
                ON segment_classification_state(status, processed_at);

                CREATE TABLE IF NOT EXISTS context_packages (
                    context_id        TEXT PRIMARY KEY,
                    work_item_id      TEXT NOT NULL REFERENCES work_items(work_item_id) ON DELETE CASCADE,
                    version           INTEGER NOT NULL CHECK(version > 0),
                    canonical_path    TEXT NOT NULL,
                    project_copy_path TEXT NOT NULL DEFAULT '',
                    content_hash      TEXT NOT NULL,
                    created_at        TEXT NOT NULL,
                    UNIQUE(work_item_id, version)
                );

                CREATE TABLE IF NOT EXISTS context_package_segments (
                    context_id  TEXT NOT NULL REFERENCES context_packages(context_id) ON DELETE CASCADE,
                    segment_id  TEXT NOT NULL REFERENCES conversation_segments(segment_id) ON DELETE CASCADE,
                    PRIMARY KEY(context_id, segment_id)
                );
            """)
            segment_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(conversation_segments)")
            }
            if "boundary_reason" not in segment_columns:
                conn.execute(
                    "ALTER TABLE conversation_segments ADD COLUMN boundary_reason TEXT NOT NULL DEFAULT 'conversation_start'"
                )
            if "segmenter_version" not in segment_columns:
                conn.execute(
                    "ALTER TABLE conversation_segments ADD COLUMN segmenter_version TEXT NOT NULL DEFAULT ''"
                )
            if "content_fingerprint" not in segment_columns:
                conn.execute(
                    "ALTER TABLE conversation_segments ADD COLUMN content_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            if "is_current" not in segment_columns:
                conn.execute(
                    "ALTER TABLE conversation_segments ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1"
                )
            if "superseded_at" not in segment_columns:
                conn.execute(
                    "ALTER TABLE conversation_segments ADD COLUMN superseded_at TEXT"
                )
            if "superseded_reason" not in segment_columns:
                conn.execute(
                    "ALTER TABLE conversation_segments ADD COLUMN superseded_reason TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_segments_current_project_review
                ON conversation_segments(project_id, is_current, review_status, conversation_id)
                """
            )
            # Existing databases predate content fingerprints.  Fill them
            # before the next source sync so a later edit cannot be mistaken
            # for an unchanged segment merely because its range is the same.
            missing_fingerprints = conn.execute(
                """
                SELECT segment_id, conversation_id, start_ordinal, end_ordinal
                FROM conversation_segments WHERE content_fingerprint=''
                """
            ).fetchall()
            for segment in missing_fingerprints:
                messages = [dict(row) for row in conn.execute(
                    """
                    SELECT ordinal, role, content, created_at, content_hash
                    FROM messages
                    WHERE conversation_id=? AND ordinal BETWEEN ? AND ?
                    ORDER BY ordinal
                    """,
                    (segment["conversation_id"], segment["start_ordinal"], segment["end_ordinal"]),
                )]
                if messages:
                    conn.execute(
                        "UPDATE conversation_segments SET content_fingerprint=? WHERE segment_id=?",
                        (_segment_content_fingerprint(messages), segment["segment_id"]),
                    )
            run_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(classification_runs)")
            }
            if "credential_redaction_count" not in run_columns:
                conn.execute(
                    "ALTER TABLE classification_runs ADD COLUMN credential_redaction_count "
                    "INTEGER NOT NULL DEFAULT 0 CHECK(credential_redaction_count >= 0)"
                )
            if "request_json" not in run_columns:
                conn.execute(
                    "ALTER TABLE classification_runs ADD COLUMN request_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            if "retry_of_run_id" not in run_columns:
                conn.execute(
                    "ALTER TABLE classification_runs ADD COLUMN retry_of_run_id TEXT"
                )
            now = _now()
            conn.execute(
                """
                INSERT INTO schema_meta(key, value, updated_at) VALUES ('schema_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (str(SCHEMA_VERSION), now),
            )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for key in (
            "metadata_json", "details_json", "decision_json", "redaction_json",
            "request_json", "source_segment_ids_json",
        ):
            if key in item:
                try:
                    item[key[:-5] if key.endswith("_json") else key] = json.loads(item.pop(key))
                except (TypeError, json.JSONDecodeError):
                    item[key[:-5]] = {}
        for key in ("is_primary", "resume_capable"):
            if key in item:
                item[key] = bool(item[key])
        return item

    def schema_info(self) -> dict:
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
        return {"version": version, "tables": tables, "db_path": str(self.db_path)}

    def create_project(
        self,
        name: str,
        root_paths: list[str],
        *,
        description: str = "",
        source: str = "manual",
        project_id: str | None = None,
        created_at: str | None = None,
    ) -> dict:
        normalized_name = " ".join(str(name).split())[:120]
        if not normalized_name:
            raise WorkspaceStoreError("项目名称不能为空")
        if source not in {"manual", "inferred"}:
            raise WorkspaceStoreError("项目来源不正确")
        roots = [_normalize_path(path) for path in root_paths if str(path).strip()]
        if not roots:
            raise WorkspaceStoreError("项目至少需要一个代码目录")
        roots = list(dict.fromkeys(roots))
        if source == "manual":
            missing_root = next((path for path in roots if not Path(path).is_dir()), None)
            if missing_root:
                raise WorkspaceStoreError(f"项目目录不存在：{missing_root}")
        project_id = project_id or _new_id("PRJ")
        created_at = created_at or _now()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO projects VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
                    """,
                    (project_id, normalized_name, description.strip(), source,
                     created_at, created_at, created_at),
                )
                for index, path in enumerate(roots):
                    conn.execute(
                        """
                        INSERT INTO project_roots
                        (root_id, project_id, path, normalized_path, label, root_type, is_primary, created_at)
                        VALUES (?, ?, ?, ?, ?, 'code', ?, ?)
                        """,
                        (_new_id("ROOT"), project_id, path, os.path.normcase(path),
                         Path(path).name, int(index == 0), created_at),
                    )
        except sqlite3.IntegrityError as exc:
            raise WorkspaceStoreError(f"项目目录已被其他项目使用：{exc}") from exc
        return self.get_project(project_id)

    def update_project_roots(self, project_id: str, root_paths: list[str]) -> dict:
        """保存用户维护的目录列表；列表首项是工具打开与交接文件的工作目录。"""
        roots = [_normalize_path(path) for path in root_paths if str(path).strip()]
        roots = list(dict.fromkeys(roots))
        if not roots:
            raise WorkspaceStoreError("项目至少需要一个代码目录")
        missing_root = next((path for path in roots if not Path(path).is_dir()), None)
        if missing_root:
            raise WorkspaceStoreError(f"项目目录不存在：{missing_root}")

        normalized_roots = [os.path.normcase(path) for path in roots]
        now = _now()
        try:
            with self._connect() as conn:
                if not conn.execute(
                    "SELECT 1 FROM projects WHERE project_id=?", (project_id,)
                ).fetchone():
                    raise WorkspaceStoreError("项目不存在")
                placeholders = ", ".join("?" for _ in normalized_roots)
                conflict = conn.execute(
                    f"SELECT path FROM project_roots WHERE normalized_path IN ({placeholders}) AND project_id <> ? LIMIT 1",
                    (*normalized_roots, project_id),
                ).fetchone()
                if conflict:
                    raise WorkspaceStoreError(f"项目目录已被其他项目使用：{conflict['path']}")

                existing = {
                    row["normalized_path"]: self._row(row)
                    for row in conn.execute(
                        "SELECT * FROM project_roots WHERE project_id=?", (project_id,)
                    )
                }
                conn.execute("UPDATE project_roots SET is_primary=0 WHERE project_id=?", (project_id,))
                conn.execute(
                    f"DELETE FROM project_roots WHERE project_id=? AND normalized_path NOT IN ({placeholders})",
                    (project_id, *normalized_roots),
                )
                for index, path in enumerate(roots):
                    normalized_path = os.path.normcase(path)
                    root = existing.get(normalized_path)
                    if root:
                        conn.execute(
                            """UPDATE project_roots
                               SET path=?, label=?, root_type='code', is_primary=?
                               WHERE root_id=?""",
                            (path, Path(path).name, int(index == 0), root["root_id"]),
                        )
                    else:
                        conn.execute(
                            """INSERT INTO project_roots
                               (root_id, project_id, path, normalized_path, label, root_type, is_primary, created_at)
                               VALUES (?, ?, ?, ?, ?, 'code', ?, ?)""",
                            (_new_id("ROOT"), project_id, path, normalized_path,
                             Path(path).name, int(index == 0), now),
                        )
                conn.execute(
                    "UPDATE projects SET updated_at=?, last_active_at=? WHERE project_id=?",
                    (now, now, project_id),
                )
        except sqlite3.IntegrityError as exc:
            raise WorkspaceStoreError(f"保存项目目录失败：{exc}") from exc
        return self.get_project(project_id)

    def get_project_definition(self, project_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_definitions WHERE project_id=?", (project_id,)
            ).fetchone()
        return self._row(row)

    def save_project_definition(
        self,
        project_id: str,
        *,
        goal: str = "",
        scope: str = "",
        non_goals: str = "",
        acceptance_criteria: str = "",
        constraints: str = "",
        summary: str = "",
        status: str = "draft",
        source: str = "manual",
        source_segment_ids: list[str] | None = None,
        source_run_id: str = "",
    ) -> dict:
        if not self.get_project(project_id):
            raise WorkspaceStoreError("项目不存在")
        if status not in {"draft", "confirmed", "ignored"}:
            raise WorkspaceStoreError("项目定义状态不正确")
        if source not in {"manual", "inferred"}:
            raise WorkspaceStoreError("项目定义来源不正确")
        now = _now()
        source_ids = list(dict.fromkeys(
            str(item).strip() for item in (source_segment_ids or []) if str(item).strip()
        ))
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM project_definitions WHERE project_id=?", (project_id,)
            ).fetchone()
            conn.execute(
                """
                INSERT INTO project_definitions(
                    project_id, status, goal, scope, non_goals, acceptance_criteria,
                    constraints, summary, source, source_segment_ids_json, source_run_id,
                    created_at, updated_at, confirmed_at, ignored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    status=excluded.status, goal=excluded.goal, scope=excluded.scope,
                    non_goals=excluded.non_goals,
                    acceptance_criteria=excluded.acceptance_criteria,
                    constraints=excluded.constraints, summary=excluded.summary,
                    source=excluded.source,
                    source_segment_ids_json=excluded.source_segment_ids_json,
                    source_run_id=excluded.source_run_id, updated_at=excluded.updated_at,
                    confirmed_at=excluded.confirmed_at, ignored_at=excluded.ignored_at
                """,
                (
                    project_id, status, goal.strip()[:4000], scope.strip()[:4000],
                    non_goals.strip()[:4000], acceptance_criteria.strip()[:4000],
                    constraints.strip()[:4000], summary.strip()[:4000], source,
                    _json(source_ids), source_run_id.strip()[:160],
                    existing["created_at"] if existing else now, now,
                    now if status == "confirmed" else None,
                    now if status == "ignored" else None,
                ),
            )
            conn.execute(
                "UPDATE projects SET updated_at=?, last_active_at=? WHERE project_id=?",
                (now, now, project_id),
            )
        return self.get_project_definition(project_id)

    def confirm_project_definition(self, project_id: str) -> dict:
        definition = self.get_project_definition(project_id)
        if not definition:
            raise WorkspaceStoreError("没有可确认的项目定义草稿")
        return self.save_project_definition(
            project_id,
            goal=definition["goal"], scope=definition["scope"],
            non_goals=definition["non_goals"],
            acceptance_criteria=definition["acceptance_criteria"],
            constraints=definition["constraints"], summary=definition["summary"],
            status="confirmed", source=definition["source"],
            source_segment_ids=definition.get("source_segment_ids", []),
            source_run_id=definition.get("source_run_id", ""),
        )

    def ignore_project_definition(self, project_id: str) -> dict:
        definition = self.get_project_definition(project_id)
        if not definition:
            raise WorkspaceStoreError("没有可忽略的项目定义草稿")
        return self.save_project_definition(
            project_id,
            goal=definition["goal"], scope=definition["scope"],
            non_goals=definition["non_goals"],
            acceptance_criteria=definition["acceptance_criteria"],
            constraints=definition["constraints"], summary=definition["summary"],
            status="ignored", source=definition["source"],
            source_segment_ids=definition.get("source_segment_ids", []),
            source_run_id=definition.get("source_run_id", ""),
        )

    def get_project(self, project_id: str) -> dict | None:
        with self._connect() as conn:
            project = self._row(conn.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone())
            if not project:
                return None
            roots = [self._row(row) for row in conn.execute(
                "SELECT * FROM project_roots WHERE project_id = ? ORDER BY is_primary DESC, created_at",
                (project_id,),
            )]
        project["roots"] = roots
        project["definition"] = self.get_project_definition(project_id)
        return project

    def list_projects(self, status: str | None = None) -> list[dict]:
        query = "SELECT project_id FROM projects"
        params = []
        if status:
            if status not in PROJECT_STATUSES:
                raise WorkspaceStoreError("项目状态不正确")
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY last_active_at DESC"
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(query, params)]
        return [self.get_project(project_id) for project_id in ids]

    def get_project_overview(self, project_id: str) -> dict | None:
        project = self.get_project(project_id)
        if not project:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM work_items WHERE project_id=? GROUP BY status",
                (project_id,),
            ).fetchall()
            conversation_rows = conn.execute(
                """
                SELECT c.source, COUNT(*) AS count
                FROM conversation_project_matches m
                JOIN conversations c ON c.conversation_id=m.conversation_id
                WHERE m.project_id=? AND m.is_primary=1
                GROUP BY c.source
                """,
                (project_id,),
            ).fetchall()
        counts = {status: 0 for status in WORK_ITEM_STATUSES}
        counts.update({row["status"]: row["count"] for row in rows})
        project["work_item_counts"] = counts
        project["work_item_total"] = sum(counts.values())
        project["active_work_item_count"] = sum(
            counts[status] for status in {"backlog", "planned", "in_progress", "blocked"}
        )
        conversation_counts = {"codex": 0, "claude": 0, "work_assistant": 0}
        conversation_counts.update({row["source"]: row["count"] for row in conversation_rows})
        project["conversation_counts"] = conversation_counts
        project["conversation_count"] = sum(conversation_counts.values())
        return project

    def list_project_overviews(self, status: str | None = "active") -> list[dict]:
        return [
            self.get_project_overview(project["project_id"])
            for project in self.list_projects(status=status)
        ]

    def create_work_item(
        self,
        project_id: str,
        title: str,
        *,
        item_type: str = "feature",
        goal: str = "",
        description: str = "",
        source: str = "manual",
        status: str | None = None,
        confidence: float | None = None,
        priority: str = "P1",
        deadline: str | None = None,
        created_at: str | None = None,
        metadata: dict | None = None,
        work_item_id: str | None = None,
    ) -> dict:
        if not self.get_project(project_id):
            raise WorkspaceStoreError("所属项目不存在")
        normalized_title = " ".join(str(title).split())[:160]
        if not normalized_title:
            raise WorkspaceStoreError("工作项名称不能为空")
        if item_type not in WORK_ITEM_TYPES:
            raise WorkspaceStoreError("工作项类型不正确")
        if source not in WORK_ITEM_SOURCES:
            raise WorkspaceStoreError("工作项来源不正确")
        status = status or ("suggested" if source == "inferred" else "backlog")
        if status not in WORK_ITEM_STATUSES:
            raise WorkspaceStoreError("工作项状态不正确")
        if confidence is not None and not 0 <= confidence <= 1:
            raise WorkspaceStoreError("识别置信度必须在 0–1 之间")
        created_at = created_at or _now()
        discovered_at = _now()
        work_item_id = work_item_id or _new_id("WI")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO work_items(
                    work_item_id, project_id, title, item_type, goal, description,
                    status, source, confidence, priority, deadline, created_at,
                    discovered_at, updated_at, last_active_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (work_item_id, project_id, normalized_title, item_type, goal.strip(),
                 description.strip(), status, source, confidence, priority, deadline,
                 created_at, discovered_at, discovered_at, created_at, _json(metadata or {})),
            )
            self._insert_work_item_event(
                conn, work_item_id, "created", "system",
                {"source": source, "status": status}, discovered_at,
            )
        return self.get_work_item(work_item_id)

    def _insert_work_item_event(
        self, conn, work_item_id: str, event_type: str, actor: str,
        details: dict, created_at: str | None = None,
    ):
        conn.execute(
            "INSERT INTO work_item_events VALUES (?, ?, ?, ?, ?, ?)",
            (_new_id("EVT"), work_item_id, event_type, actor,
             _json(details), created_at or _now()),
        )

    def get_work_item(self, work_item_id: str) -> dict | None:
        with self._connect() as conn:
            item = self._row(conn.execute(
                "SELECT * FROM work_items WHERE work_item_id = ?", (work_item_id,)
            ).fetchone())
            if not item:
                return None
            item["steps"] = [self._row(row) for row in conn.execute(
                "SELECT * FROM work_item_steps WHERE work_item_id = ? ORDER BY step_index",
                (work_item_id,),
            )]
            item["events"] = [self._row(row) for row in conn.execute(
                "SELECT * FROM work_item_events WHERE work_item_id=? ORDER BY created_at",
                (work_item_id,),
            )]
        if not item["steps"]:
            item["completion_percent"] = None
        else:
            weights = {"completed": 1.0, "skipped": 1.0, "in_progress": 0.5, "pending": 0.0}
            completed = sum(weights.get(step["status"], 0) for step in item["steps"])
            item["completion_percent"] = round(completed * 100 / len(item["steps"]))
        return item

    def replace_work_item_steps(self, work_item_id: str, steps: list[dict]) -> dict:
        if not self.get_work_item(work_item_id):
            raise WorkspaceStoreError("工作项不存在")
        now = _now()
        with self._connect() as conn:
            conn.execute("DELETE FROM work_item_steps WHERE work_item_id=?", (work_item_id,))
            for index, step in enumerate(steps):
                status = str(step.get("status", "pending"))
                if status not in {"pending", "in_progress", "completed", "skipped"}:
                    raise WorkspaceStoreError("工作项步骤状态不正确")
                title = " ".join(str(step.get("title", "")).split())
                if not title:
                    raise WorkspaceStoreError("工作项步骤名称不能为空")
                conn.execute(
                    """
                    INSERT INTO work_item_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (step.get("step_id") or _new_id("STEP"), work_item_id, index,
                     title, str(step.get("description", "")), status, now, now,
                     now if status == "completed" else None),
                )
            conn.execute(
                "UPDATE work_items SET status=?, updated_at=? WHERE work_item_id=?",
                ("planned" if steps else "backlog", now, work_item_id),
            )
            self._insert_work_item_event(
                conn, work_item_id, "plan_replaced", "user", {"step_count": len(steps)}, now,
            )
        return self.get_work_item(work_item_id)

    def list_work_items(
        self,
        project_id: str | None = None,
        *,
        status: str | None = None,
        include_ignored: bool = False,
    ) -> list[dict]:
        clauses, params = [], []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if status:
            if status not in WORK_ITEM_STATUSES:
                raise WorkspaceStoreError("工作项状态不正确")
            clauses.append("status = ?")
            params.append(status)
        elif not include_ignored:
            clauses.append("status <> 'ignored'")
        query = "SELECT work_item_id FROM work_items"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY last_active_at DESC, discovered_at DESC"
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(query, params)]
        return [self.get_work_item(work_item_id) for work_item_id in ids]

    def _transition_work_item(
        self,
        work_item_id: str,
        new_status: str,
        *,
        actor: str = "user",
        ignore_reason: str = "",
    ) -> dict:
        item = self.get_work_item(work_item_id)
        if not item:
            raise WorkspaceStoreError("工作项不存在")
        now = _now()
        confirmed_at = now if new_status == "backlog" else item.get("confirmed_at")
        ignored_at = now if new_status == "ignored" else None
        reason = ignore_reason.strip() if new_status == "ignored" else ""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE work_items
                SET status=?, confirmed_at=?, ignored_at=?, ignore_reason=?, updated_at=?
                WHERE work_item_id=?
                """,
                (new_status, confirmed_at, ignored_at, reason, now, work_item_id),
            )
            self._insert_work_item_event(
                conn, work_item_id, "status_changed", actor,
                {"from": item["status"], "to": new_status, "reason": reason}, now,
            )
        return self.get_work_item(work_item_id)

    def confirm_work_item(self, work_item_id: str, *, actor: str = "user") -> dict:
        item = self.get_work_item(work_item_id)
        if not item:
            raise WorkspaceStoreError("工作项不存在")
        if item["status"] != "suggested":
            raise WorkspaceStoreError("只有待确认工作项可以确认")
        return self._transition_work_item(work_item_id, "backlog", actor=actor)

    def ignore_work_item(
        self, work_item_id: str, *, reason: str = "", actor: str = "user"
    ) -> dict:
        return self._transition_work_item(
            work_item_id, "ignored", actor=actor, ignore_reason=reason
        )

    def restore_work_item(self, work_item_id: str, *, actor: str = "user") -> dict:
        item = self.get_work_item(work_item_id)
        if not item:
            raise WorkspaceStoreError("工作项不存在")
        if item["status"] != "ignored":
            raise WorkspaceStoreError("只有已忽略工作项可以恢复")
        restored_status = "suggested" if item["source"] == "inferred" else "backlog"
        return self._transition_work_item(work_item_id, restored_status, actor=actor)

    def complete_work_item(
        self,
        work_item_id: str,
        *,
        completion_note: str = "",
        acceptance_result: str = "",
        actor: str = "user",
    ) -> dict:
        """完成正式工作项，并把人工验收说明保留为不可丢失的事件记录。"""
        item = self.get_work_item(work_item_id)
        if not item:
            raise WorkspaceStoreError("工作项不存在")
        if item["status"] not in {"backlog", "planned", "in_progress", "blocked"}:
            raise WorkspaceStoreError("只有待办、已规划、进行中或阻塞的工作项可以标记完成")
        now = _now()
        completion_note = str(completion_note).strip()
        acceptance_result = str(acceptance_result).strip()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE work_items
                SET status='completed', completed_at=?, updated_at=?, last_active_at=?
                WHERE work_item_id=?
                """,
                (now, now, now, work_item_id),
            )
            self._insert_work_item_event(
                conn, work_item_id, "completed", actor,
                {
                    "from": item["status"],
                    "completion_note": completion_note,
                    "acceptance_result": acceptance_result,
                }, now,
            )
        return self.get_work_item(work_item_id)

    def reopen_work_item(self, work_item_id: str, *, actor: str = "user") -> dict:
        """允许完成后的返工；完成历史保留在事件中。"""
        item = self.get_work_item(work_item_id)
        if not item:
            raise WorkspaceStoreError("工作项不存在")
        if item["status"] != "completed":
            raise WorkspaceStoreError("只有已完成工作项可以恢复为进行中")
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE work_items
                SET status='in_progress', completed_at=NULL, updated_at=?, last_active_at=?
                WHERE work_item_id=?
                """,
                (now, now, work_item_id),
            )
            self._insert_work_item_event(
                conn, work_item_id, "reopened", actor,
                {"from": "completed", "to": "in_progress"}, now,
            )
        return self.get_work_item(work_item_id)

    def merge_work_items(
        self,
        target_work_item_id: str,
        source_work_item_ids: list[str],
        *,
        actor: str = "user",
    ) -> dict:
        target = self.get_work_item(target_work_item_id)
        if not target:
            raise WorkspaceStoreError("保留的工作项不存在")
        if target["status"] == "archived":
            raise WorkspaceStoreError("不能合并到已归档工作项")
        source_ids = list(dict.fromkeys(str(item_id) for item_id in source_work_item_ids))
        source_ids = [item_id for item_id in source_ids if item_id != target_work_item_id]
        if not source_ids:
            raise WorkspaceStoreError("至少选择一个需要合并的工作项")
        sources = []
        for source_id in source_ids:
            source = self.get_work_item(source_id)
            if not source:
                raise WorkspaceStoreError(f"待合并工作项不存在：{source_id}")
            if source["project_id"] != target["project_id"]:
                raise WorkspaceStoreError("只能合并同一项目下的工作项")
            if source["status"] == "archived":
                raise WorkspaceStoreError("待合并工作项已经归档")
            sources.append(source)

        now = _now()
        with self._connect() as conn:
            for source in sources:
                links = conn.execute(
                    "SELECT * FROM segment_work_item_links WHERE work_item_id=?",
                    (source["work_item_id"],),
                ).fetchall()
                for row in links:
                    segment_id, relation, confidence = (
                        row["segment_id"], row["relation"], row["confidence"]
                    )
                    conn.execute(
                        "DELETE FROM segment_work_item_links "
                        "WHERE segment_id=? AND work_item_id=? AND relation=?",
                        (segment_id, source["work_item_id"], relation),
                    )
                    target_links = conn.execute(
                        "SELECT relation, confidence FROM segment_work_item_links "
                        "WHERE segment_id=? AND work_item_id=?",
                        (segment_id, target_work_item_id),
                    ).fetchall()
                    target_relations = {item["relation"] for item in target_links}
                    if relation == "primary":
                        conn.execute(
                            "DELETE FROM segment_work_item_links "
                            "WHERE segment_id=? AND work_item_id=?",
                            (segment_id, target_work_item_id),
                        )
                    elif "primary" in target_relations:
                        continue
                    existing = conn.execute(
                        "SELECT confidence FROM segment_work_item_links "
                        "WHERE segment_id=? AND work_item_id=? AND relation=?",
                        (segment_id, target_work_item_id, relation),
                    ).fetchone()
                    if existing:
                        values = [value for value in (existing["confidence"], confidence) if value is not None]
                        conn.execute(
                            "UPDATE segment_work_item_links SET confidence=? "
                            "WHERE segment_id=? AND work_item_id=? AND relation=?",
                            (max(values) if values else None, segment_id,
                             target_work_item_id, relation),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO segment_work_item_links VALUES (?, ?, ?, ?, ?)",
                            (segment_id, target_work_item_id, relation, confidence, now),
                        )

                metadata = dict(source.get("metadata") or {})
                metadata.update({
                    "merged_into_work_item_id": target_work_item_id,
                    "merged_at": now,
                })
                conn.execute(
                    """
                    UPDATE work_items
                    SET status='archived', metadata_json=?, updated_at=?
                    WHERE work_item_id=?
                    """,
                    (_json(metadata), now, source["work_item_id"]),
                )
                self._insert_work_item_event(
                    conn, source["work_item_id"], "merged_into", actor,
                    {"target_work_item_id": target_work_item_id}, now,
                )
            conn.execute(
                "UPDATE work_items SET updated_at=?, last_active_at=? WHERE work_item_id=?",
                (now, now, target_work_item_id),
            )
            self._insert_work_item_event(
                conn, target_work_item_id, "merged_sources", actor,
                {
                    "source_work_item_ids": source_ids,
                    "source_titles": [source["title"] for source in sources],
                },
                now,
            )
        merged_target = self.get_work_item(target_work_item_id)
        merged_target["merged_source_ids"] = source_ids
        return merged_target

    def upsert_conversation(
        self,
        source: str,
        external_session_id: str,
        *,
        title: str = "",
        original_project_path: str = "",
        source_path: str = "",
        started_at: str | None = None,
        updated_at: str | None = None,
        source_modified_ns: int = 0,
        content_hash: str = "",
        resume_capable: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        if source not in CONVERSATION_SOURCES:
            raise WorkspaceStoreError("会话来源不正确")
        external_session_id = str(external_session_id).strip()
        if not external_session_id:
            raise WorkspaceStoreError("外部会话 ID 不能为空")
        now = _now()
        started_at = started_at or updated_at or now
        updated_at = updated_at or started_at
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT conversation_id FROM conversations WHERE source=? AND external_session_id=?",
                (source, external_session_id),
            ).fetchone()
            conversation_id = existing[0] if existing else _new_id("CONV")
            conn.execute(
                """
                INSERT INTO conversations(
                    conversation_id, source, external_session_id, title,
                    original_project_path, source_path, started_at, updated_at,
                    imported_at, source_modified_ns, content_hash, resume_capable, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, external_session_id) DO UPDATE SET
                    title=excluded.title,
                    original_project_path=excluded.original_project_path,
                    source_path=excluded.source_path,
                    started_at=excluded.started_at,
                    updated_at=excluded.updated_at,
                    imported_at=excluded.imported_at,
                    source_modified_ns=excluded.source_modified_ns,
                    content_hash=excluded.content_hash,
                    resume_capable=excluded.resume_capable,
                    metadata_json=excluded.metadata_json
                """,
                (conversation_id, source, external_session_id, title.strip(),
                 original_project_path, source_path, started_at, updated_at, now,
                 int(source_modified_ns), content_hash, int(resume_capable), _json(metadata or {})),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            return self._row(conn.execute(
                "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone())

    def find_conversation(self, source: str, external_session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE source=? AND external_session_id=?",
                (source, external_session_id),
            ).fetchone()
        return self._row(row)

    def touch_conversation_source(
        self,
        conversation_id: str,
        *,
        source_modified_ns: int,
        content_hash: str,
    ):
        with self._connect() as conn:
            changed = conn.execute(
                """
                UPDATE conversations
                SET source_modified_ns=?, content_hash=?, imported_at=?
                WHERE conversation_id=?
                """,
                (int(source_modified_ns), content_hash, _now(), conversation_id),
            ).rowcount
        if not changed:
            raise WorkspaceStoreError("会话不存在")

    def list_conversations(self, source: str | None = None) -> list[dict]:
        query = """
            SELECT c.*, COUNT(m.message_id) AS message_count
            FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.conversation_id
        """
        params = []
        if source:
            if source not in CONVERSATION_SOURCES:
                raise WorkspaceStoreError("会话来源不正确")
            query += " WHERE c.source=?"
            params.append(source)
        query += " GROUP BY c.conversation_id ORDER BY c.updated_at DESC"
        with self._connect() as conn:
            conversations = [self._row(row) for row in conn.execute(query, params)]
        for conversation in conversations:
            matches = self.get_conversation_project_matches(conversation["conversation_id"])
            conversation["project_matches"] = matches
            conversation["primary_project"] = next(
                (match for match in matches if match["is_primary"]), None
            )
            conversation["project_match_state"] = (
                "matched" if conversation["primary_project"]
                else "ambiguous" if matches
                else "unassigned"
            )
        return conversations

    def list_project_roots(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, p.name AS project_name, p.status AS project_status
                FROM project_roots r JOIN projects p ON p.project_id=r.project_id
                WHERE p.status='active'
                ORDER BY LENGTH(r.normalized_path) DESC
                """
            ).fetchall()
        return [self._row(row) for row in rows]

    def get_conversation_project_matches(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, p.name AS project_name, r.path AS root_path
                FROM conversation_project_matches m
                JOIN projects p ON p.project_id=m.project_id
                LEFT JOIN project_roots r ON r.root_id=m.root_id
                WHERE m.conversation_id=?
                ORDER BY m.is_primary DESC, m.confidence DESC, p.name
                """,
                (conversation_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def replace_auto_project_matches(
        self,
        conversation_id: str,
        matches: list[dict],
    ) -> list[dict]:
        if not self.get_conversation(conversation_id):
            raise WorkspaceStoreError("会话不存在")
        now = _now()
        with self._connect() as conn:
            manual = conn.execute(
                """
                SELECT 1 FROM conversation_project_matches
                WHERE conversation_id=? AND match_source='manual' AND is_primary=1
                """,
                (conversation_id,),
            ).fetchone()
            if manual:
                return self.get_conversation_project_matches(conversation_id)
            conn.execute(
                "DELETE FROM conversation_project_matches WHERE conversation_id=? AND match_source='auto'",
                (conversation_id,),
            )
            for match in matches:
                conn.execute(
                    """
                    INSERT INTO conversation_project_matches VALUES (?, ?, ?, ?, ?, ?, 'auto', ?, ?)
                    ON CONFLICT(conversation_id, project_id) DO UPDATE SET
                        root_id=excluded.root_id,
                        match_method=excluded.match_method,
                        confidence=excluded.confidence,
                        is_primary=excluded.is_primary,
                        match_source='auto',
                        updated_at=excluded.updated_at
                    """,
                    (conversation_id, match["project_id"], match.get("root_id"),
                     match["match_method"], float(match["confidence"]),
                     int(match.get("is_primary", False)), now, now),
                )
        return self.get_conversation_project_matches(conversation_id)

    def set_manual_conversation_project(
        self,
        conversation_id: str,
        project_id: str,
    ) -> list[dict]:
        if not self.get_conversation(conversation_id):
            raise WorkspaceStoreError("会话不存在")
        project = self.get_project(project_id)
        if not project:
            raise WorkspaceStoreError("项目不存在")
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_project_matches WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.execute(
                """
                INSERT INTO conversation_project_matches
                VALUES (?, ?, NULL, 'manual', 1.0, 1, 'manual', ?, ?)
                """,
                (conversation_id, project_id, now, now),
            )
        return self.get_conversation_project_matches(conversation_id)

    def get_source_state(self, source_path: str | Path) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM external_source_state WHERE source_path=?",
                (str(Path(source_path)),),
            ).fetchone()
        return self._row(row)

    def save_source_state(
        self,
        source_path: str | Path,
        *,
        source: str,
        source_modified_ns: int,
        content_hash: str,
        import_status: str,
        conversation_id: str | None = None,
        last_error: str = "",
    ) -> dict:
        if source not in {"codex", "claude"}:
            raise WorkspaceStoreError("外部会话来源不正确")
        if import_status not in {"imported", "skipped", "error"}:
            raise WorkspaceStoreError("外部文件导入状态不正确")
        path = str(Path(source_path))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO external_source_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    source=excluded.source,
                    source_modified_ns=excluded.source_modified_ns,
                    content_hash=excluded.content_hash,
                    import_status=excluded.import_status,
                    conversation_id=excluded.conversation_id,
                    last_error=excluded.last_error,
                    checked_at=excluded.checked_at
                """,
                (path, source, int(source_modified_ns), content_hash, import_status,
                 conversation_id, last_error[:1000], _now()),
            )
            row = conn.execute(
                "SELECT * FROM external_source_state WHERE source_path=?", (path,)
            ).fetchone()
        return self._row(row)

    def replace_messages(self, conversation_id: str, messages: list[dict]):
        """Legacy destructive replacement for callers that explicitly need it.

        External Codex / Claude sync uses :meth:`sync_messages` instead so an
        appended journal entry cannot destroy segment evidence.
        """
        if not self.get_conversation(conversation_id):
            raise WorkspaceStoreError("会话不存在")
        normalized = _normalize_messages(messages)
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            conn.execute("DELETE FROM conversation_segments WHERE conversation_id=?", (conversation_id,))
            conn.execute(
                "DELETE FROM conversation_segmentation_state WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversation_retrieval_index_state WHERE conversation_id=?",
                (conversation_id,),
            )
            for message in normalized:
                conn.execute(
                    """
                    INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (_new_id("MSG"), conversation_id, message["ordinal"], message["role"],
                     message["content"], message["created_at"], message["content_hash"]),
                )

    def sync_messages(self, conversation_id: str, messages: list[dict]) -> dict:
        """Apply an external conversation snapshot without replacing its prefix.

        JSONL sources are normally append-only.  For an edited source, the
        longest common prefix is still preserved and only the changed suffix is
        rewritten.  Derived segment state is invalidated, but the segment rows
        themselves are reconciled later by the semantic segmenter.
        """
        if not self.get_conversation(conversation_id):
            raise WorkspaceStoreError("会话不存在")
        incoming = _normalize_messages(messages)
        existing = self.list_messages(conversation_id)

        def same_message(before: dict, after: dict) -> bool:
            before_hash = before.get("content_hash") or _message_content_hash(before.get("content", ""))
            return (
                int(before["ordinal"]) == int(after["ordinal"])
                and str(before["role"]).lower() == after["role"]
                and str(before["created_at"]) == after["created_at"]
                and before_hash == after["content_hash"]
            )

        prefix_size = 0
        while (
            prefix_size < len(existing)
            and prefix_size < len(incoming)
            and same_message(existing[prefix_size], incoming[prefix_size])
        ):
            prefix_size += 1
        if prefix_size == len(existing) == len(incoming):
            return {
                "state": "unchanged", "first_changed_ordinal": None,
                "stable_message_count": prefix_size,
                "old_message_count": len(existing), "new_message_count": len(incoming),
            }

        if prefix_size == len(existing):
            state = "appended"
        elif prefix_size == len(incoming):
            state = "truncated"
        else:
            state = "rewritten"
        first_changed_ordinal = (
            incoming[prefix_size]["ordinal"]
            if prefix_size < len(incoming)
            else existing[prefix_size]["ordinal"]
        )
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id=? AND ordinal>=?",
                (conversation_id, first_changed_ordinal),
            )
            for message in incoming[prefix_size:]:
                conn.execute(
                    """
                    INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (_new_id("MSG"), conversation_id, message["ordinal"], message["role"],
                     message["content"], message["created_at"], message["content_hash"]),
                )
            # Retrieval chunks are derived data and must never be used against
            # a changed transcript.  Segments remain until reconciliation so
            # their IDs and evidence links can be retained where still valid.
            conn.execute(
                "DELETE FROM conversation_segmentation_state WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversation_retrieval_index_state WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversation_retrieval_chunks WHERE conversation_id=?",
                (conversation_id,),
            )
        return {
            "state": state, "first_changed_ordinal": first_changed_ordinal,
            "stable_message_count": prefix_size,
            "old_message_count": len(existing), "new_message_count": len(incoming),
        }

    def list_messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY ordinal",
                (conversation_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def get_segmentation_state(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_segmentation_state WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        return self._row(row)

    def get_retrieval_index_state(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_retrieval_index_state WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        return self._row(row)

    def list_segments_for_conversation(
        self, conversation_id: str, *, include_superseded: bool = False
    ) -> list[dict]:
        current_filter = "" if include_superseded else " AND is_current=1"
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(
                f"""
                SELECT segment_id FROM conversation_segments
                WHERE conversation_id=?{current_filter} ORDER BY start_ordinal, created_at
                """,
                (conversation_id,),
            )]
        return [self.get_segment(segment_id) for segment_id in ids]

    def list_retrieval_chunks_for_conversation(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, s.title AS segment_title
                FROM conversation_retrieval_chunks r
                JOIN conversation_segments s ON s.segment_id=r.segment_id
                WHERE r.conversation_id=? AND s.is_current=1
                ORDER BY s.start_ordinal, r.chunk_index
                """,
                (conversation_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_retrieval_chunks_for_segment(self, segment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversation_retrieval_chunks
                WHERE segment_id=? ORDER BY chunk_index
                """,
                (segment_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_segments(self, project_id: str | None = None) -> list[dict]:
        query = """
            SELECT s.segment_id
            FROM conversation_segments s
            JOIN conversations c ON c.conversation_id=s.conversation_id
        """
        params = []
        if project_id:
            query += " WHERE s.project_id=?"
            params.append(project_id)
        query += " AND s.is_current=1" if project_id else " WHERE s.is_current=1"
        query += " ORDER BY c.started_at, s.start_ordinal"
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(query, params)]
        return [self.get_segment(segment_id) for segment_id in ids]

    def list_messages_for_segment(self, segment_id: str) -> list[dict]:
        segment = self.get_segment(segment_id)
        if not segment:
            raise WorkspaceStoreError("会话片段不存在")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id=? AND ordinal BETWEEN ? AND ?
                ORDER BY ordinal
                """,
                (segment["conversation_id"], segment["start_ordinal"], segment["end_ordinal"]),
            ).fetchall()
        return [self._row(row) for row in rows]

    def replace_retrieval_chunks(
        self,
        conversation_id: str,
        chunks: list[dict],
        *,
        message_fingerprint: str,
        chunker_version: str,
    ) -> list[dict]:
        """Atomically replace one conversation's derived retrieval chunks."""
        segmentation = self.get_segmentation_state(conversation_id)
        if not segmentation or segmentation["status"] != "completed":
            raise WorkspaceStoreError("会话尚未完成语义分段，无法建立检索索引")

        segments = {
            segment["segment_id"]: segment
            for segment in self.list_segments_for_conversation(conversation_id)
        }
        expected_indexes: dict[str, int] = {}
        for chunk in chunks:
            segment_id = str(chunk.get("segment_id") or "")
            segment = segments.get(segment_id)
            if not segment:
                raise WorkspaceStoreError("检索块引用了不属于该会话的语义片段")
            chunk_index = int(chunk.get("chunk_index", -1))
            if chunk_index != expected_indexes.get(segment_id, 0):
                raise WorkspaceStoreError("同一语义片段的检索块序号必须连续")
            expected_indexes[segment_id] = chunk_index + 1
            start_ordinal = int(chunk.get("start_ordinal", -1))
            end_ordinal = int(chunk.get("end_ordinal", -1))
            if (
                start_ordinal < segment["start_ordinal"]
                or end_ordinal > segment["end_ordinal"]
                or end_ordinal < start_ordinal
            ):
                raise WorkspaceStoreError("检索块消息范围超出所属语义片段")
            content = str(chunk.get("content") or "")
            if not content:
                raise WorkspaceStoreError("检索块内容不能为空")
            if len(content) != int(chunk.get("char_count", -1)):
                raise WorkspaceStoreError("检索块字符数与内容不一致")

        now = _now()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_retrieval_chunks WHERE conversation_id=?",
                (conversation_id,),
            )
            for chunk in chunks:
                content = str(chunk["content"])
                conn.execute(
                    """
                    INSERT INTO conversation_retrieval_chunks(
                        chunk_id, conversation_id, segment_id, chunk_index,
                        start_ordinal, end_ordinal, content, content_hash,
                        char_count, chunker_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _new_id("RCH"), conversation_id, chunk["segment_id"],
                        int(chunk["chunk_index"]), int(chunk["start_ordinal"]),
                        int(chunk["end_ordinal"]), content,
                        hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        int(chunk["char_count"]), chunker_version, now, now,
                    ),
                )
            conn.execute(
                """
                INSERT INTO conversation_retrieval_index_state(
                    conversation_id, message_fingerprint, chunker_version,
                    status, chunk_count, error_message, processed_at
                ) VALUES (?, ?, ?, 'completed', ?, '', ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    message_fingerprint=excluded.message_fingerprint,
                    chunker_version=excluded.chunker_version,
                    status='completed', chunk_count=excluded.chunk_count,
                    error_message='', processed_at=excluded.processed_at
                """,
                (conversation_id, message_fingerprint, chunker_version, len(chunks), now),
            )
        return self.list_retrieval_chunks_for_conversation(conversation_id)

    def save_retrieval_index_error(
        self,
        conversation_id: str,
        *,
        message_fingerprint: str,
        chunker_version: str,
        error_message: str,
    ) -> dict:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_retrieval_index_state(
                    conversation_id, message_fingerprint, chunker_version,
                    status, chunk_count, error_message, processed_at
                ) VALUES (?, ?, ?, 'error', 0, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    message_fingerprint=excluded.message_fingerprint,
                    chunker_version=excluded.chunker_version,
                    status='error', chunk_count=0,
                    error_message=excluded.error_message,
                    processed_at=excluded.processed_at
                """,
                (conversation_id, message_fingerprint, chunker_version, error_message[:1000], now),
            )
        return self.get_retrieval_index_state(conversation_id)

    def update_segment_classification(
        self,
        segment_id: str,
        *,
        segment_kind: str,
        title: str,
        summary: str,
        confidence: float | None,
        classification_source: str,
    ) -> dict:
        if segment_kind not in SEGMENT_KINDS:
            raise WorkspaceStoreError("会话片段类型不正确")
        if classification_source not in {"rules", "amd", "manual"}:
            raise WorkspaceStoreError("会话片段识别来源不正确")
        if confidence is not None and not 0 <= confidence <= 1:
            raise WorkspaceStoreError("识别置信度必须在 0–1 之间")
        with self._connect() as conn:
            changed = conn.execute(
                """
                UPDATE conversation_segments
                SET segment_kind=?, title=?, summary=?, confidence=?,
                    classification_source=?, updated_at=?
                WHERE segment_id=?
                """,
                (segment_kind, title.strip()[:160], summary.strip(), confidence,
                 classification_source, _now(), segment_id),
            ).rowcount
        if not changed:
            raise WorkspaceStoreError("会话片段不存在")
        return self.get_segment(segment_id)

    def get_segment_classification_state(self, segment_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM segment_classification_state WHERE segment_id=?",
                (segment_id,),
            ).fetchone()
        return self._row(row)

    def save_segment_classification_state(
        self,
        segment_id: str,
        *,
        segment_fingerprint: str,
        classifier_version: str,
        status: str,
        decision: dict | None = None,
        redactions: dict | None = None,
        run_id: str | None = None,
        error_message: str = "",
    ) -> dict:
        if status not in {"classified", "skipped", "error"}:
            raise WorkspaceStoreError("片段分析状态不正确")
        if not self.get_segment(segment_id):
            raise WorkspaceStoreError("会话片段不存在")
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO segment_classification_state(
                    segment_id, segment_fingerprint, classifier_version, status,
                    decision_json, redaction_json, run_id, error_message, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    segment_fingerprint=excluded.segment_fingerprint,
                    classifier_version=excluded.classifier_version,
                    status=excluded.status,
                    decision_json=excluded.decision_json,
                    redaction_json=excluded.redaction_json,
                    run_id=excluded.run_id,
                    error_message=excluded.error_message,
                    processed_at=excluded.processed_at
                """,
                (segment_id, segment_fingerprint, classifier_version, status,
                 _json(decision or {}), _json(redactions or {}), run_id,
                 error_message[:1000], now),
            )
            row = conn.execute(
                "SELECT * FROM segment_classification_state WHERE segment_id=?",
                (segment_id,),
            ).fetchone()
        return self._row(row)

    def replace_conversation_segments(
        self,
        conversation_id: str,
        segments: list[dict],
        *,
        message_fingerprint: str,
        segmenter_version: str,
    ) -> list[dict]:
        messages = self.list_messages(conversation_id)
        if not messages:
            raise WorkspaceStoreError("会话没有可切分的消息")
        if not segments:
            raise WorkspaceStoreError("切分结果不能为空")
        first_ordinal = messages[0]["ordinal"]
        last_ordinal = messages[-1]["ordinal"]
        expected_start = first_ordinal
        for segment in segments:
            start = int(segment["start_ordinal"])
            end = int(segment["end_ordinal"])
            if start != expected_start or end < start:
                raise WorkspaceStoreError("切分结果必须连续且不能重叠")
            expected_start = end + 1
        if expected_start - 1 != last_ordinal:
            raise WorkspaceStoreError("切分结果必须覆盖全部消息")

        messages_by_ordinal = {int(message["ordinal"]): message for message in messages}

        def fingerprint_for(segment: dict) -> str:
            start = int(segment["start_ordinal"])
            end = int(segment["end_ordinal"])
            source_messages = [messages_by_ordinal[ordinal] for ordinal in range(start, end + 1)]
            return str(segment.get("content_fingerprint") or _segment_content_fingerprint(source_messages))

        prepared = [{**segment, "content_fingerprint": fingerprint_for(segment)} for segment in segments]
        now = _now()
        with self._connect() as conn:
            existing = [dict(row) for row in conn.execute(
                "SELECT * FROM conversation_segments WHERE conversation_id=?",
                (conversation_id,),
            ).fetchall()]
            used_ids: set[str] = set()

            def take_candidate(predicate, *, prefer_current: bool = True) -> dict | None:
                candidates = [
                    row for row in existing
                    if row["segment_id"] not in used_ids and predicate(row)
                ]
                if not candidates:
                    return None
                candidates.sort(key=lambda row: (int(row["is_current"]) if prefer_current else 0, row["updated_at"]), reverse=True)
                return candidates[0]

            for segment in prepared:
                start = int(segment["start_ordinal"])
                end = int(segment["end_ordinal"])
                content_fingerprint = segment["content_fingerprint"]
                exact = take_candidate(
                    lambda row: (
                        int(row["start_ordinal"]) == start
                        and int(row["end_ordinal"]) == end
                        and row.get("content_fingerprint") == content_fingerprint
                    )
                )
                if exact:
                    segment_id = exact["segment_id"]
                    used_ids.add(segment_id)
                    # An old range can become current again after a source
                    # rewrite is reverted.  Its matching classification state
                    # and evidence remain valid because the content hash also
                    # matches.
                    conn.execute(
                        """
                        UPDATE conversation_segments
                        SET is_current=1, superseded_at=NULL, superseded_reason='', updated_at=?
                        WHERE segment_id=?
                        """,
                        (now, segment_id),
                    )
                    continue

                # A segment with the same starting message is the affected
                # tail segment.  Keep its identity and evidence links, but
                # invalidate its classification cache so it is analysed again.
                candidate = take_candidate(lambda row: int(row["start_ordinal"]) == start)
                if candidate:
                    segment_id = candidate["segment_id"]
                    used_ids.add(segment_id)
                    conn.execute(
                        """
                        UPDATE conversation_segments
                        SET start_ordinal=?, end_ordinal=?, project_id=?, title=?,
                            boundary_reason=?, segmenter_version=?, content_fingerprint=?,
                            is_current=1, superseded_at=NULL, superseded_reason='', updated_at=?
                        WHERE segment_id=?
                        """,
                        (
                            start, end, segment.get("project_id"),
                            str(segment.get("title", ""))[:160],
                            str(segment.get("boundary_reason", "conversation_start"))[:80],
                            segmenter_version, content_fingerprint, now, segment_id,
                        ),
                    )
                    conn.execute(
                        "DELETE FROM segment_classification_state WHERE segment_id=?",
                        (segment_id,),
                    )
                    continue

                segment_id = _new_id("SEG")
                used_ids.add(segment_id)
                conn.execute(
                    """
                    INSERT INTO conversation_segments(
                        segment_id, conversation_id, start_ordinal, end_ordinal,
                        segment_kind, project_id, title, summary, confidence,
                        classification_source, review_status, boundary_reason,
                        segmenter_version, created_at, updated_at, content_fingerprint,
                        is_current, superseded_at, superseded_reason
                    ) VALUES (?, ?, ?, ?, 'unclassified', ?, ?, '', NULL,
                              'rules', 'suggested', ?, ?, ?, ?, ?, 1, NULL, '')
                    """,
                    (
                        segment_id, conversation_id, start, end, segment.get("project_id"),
                        str(segment.get("title", ""))[:160],
                        str(segment.get("boundary_reason", "conversation_start"))[:80],
                        segmenter_version, now, now, content_fingerprint,
                    ),
                )

            # Do not physically delete segments which may be referenced by a
            # context package.  They remain available as historical evidence,
            # but all normal analysis queries read only current segments.
            conn.execute(
                f"""
                UPDATE conversation_segments
                SET is_current=0, superseded_at=?, superseded_reason='source_resegmented', updated_at=?
                WHERE conversation_id=? AND is_current=1
                  AND segment_id NOT IN ({','.join('?' for _ in used_ids)})
                """,
                (now, now, conversation_id, *used_ids),
            )
            conn.execute(
                "DELETE FROM conversation_retrieval_chunks WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversation_retrieval_index_state WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.execute(
                """
                INSERT INTO conversation_segmentation_state VALUES (?, ?, ?, 'completed', ?, '', ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    message_fingerprint=excluded.message_fingerprint,
                    segmenter_version=excluded.segmenter_version,
                    status='completed',
                    segment_count=excluded.segment_count,
                    error_message='',
                    processed_at=excluded.processed_at
                """,
                (conversation_id, message_fingerprint, segmenter_version, len(prepared), now),
            )
        return self.list_segments_for_conversation(conversation_id)

    def save_segmentation_error(
        self,
        conversation_id: str,
        *,
        message_fingerprint: str,
        segmenter_version: str,
        error_message: str,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_segmentation_state
                VALUES (?, ?, ?, 'error', 0, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    message_fingerprint=excluded.message_fingerprint,
                    segmenter_version=excluded.segmenter_version,
                    status='error',
                    segment_count=0,
                    error_message=excluded.error_message,
                    processed_at=excluded.processed_at
                """,
                (conversation_id, message_fingerprint, segmenter_version,
                 error_message[:1000], _now()),
            )

    def create_segment(
        self,
        conversation_id: str,
        start_ordinal: int,
        end_ordinal: int,
        *,
        segment_kind: str = "unclassified",
        project_id: str | None = None,
        title: str = "",
        summary: str = "",
        confidence: float | None = None,
        classification_source: str = "rules",
        review_status: str = "suggested",
        boundary_reason: str = "conversation_start",
        segmenter_version: str = "",
    ) -> dict:
        if segment_kind not in SEGMENT_KINDS:
            raise WorkspaceStoreError("会话片段类型不正确")
        if review_status not in SEGMENT_REVIEW_STATUSES:
            raise WorkspaceStoreError("会话片段审核状态不正确")
        if classification_source not in {"rules", "amd", "manual"}:
            raise WorkspaceStoreError("会话片段识别来源不正确")
        if confidence is not None and not 0 <= confidence <= 1:
            raise WorkspaceStoreError("识别置信度必须在 0–1 之间")
        with self._connect() as conn:
            bounds = conn.execute(
                "SELECT MIN(ordinal), MAX(ordinal) FROM messages WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if bounds[0] is None:
                raise WorkspaceStoreError("会话没有可切分的消息")
            if start_ordinal < bounds[0] or end_ordinal > bounds[1] or end_ordinal < start_ordinal:
                raise WorkspaceStoreError("会话片段消息范围不正确")
            overlap = conn.execute(
                """
                SELECT 1 FROM conversation_segments
                WHERE conversation_id=? AND is_current=1
                  AND NOT(end_ordinal < ? OR start_ordinal > ?)
                LIMIT 1
                """,
                (conversation_id, start_ordinal, end_ordinal),
            ).fetchone()
            if overlap:
                raise WorkspaceStoreError("同一会话的片段范围不能重叠")
            if project_id and not conn.execute(
                "SELECT 1 FROM projects WHERE project_id=?", (project_id,)
            ).fetchone():
                raise WorkspaceStoreError("片段所属项目不存在")
            now = _now()
            segment_id = _new_id("SEG")
            segment_messages = [dict(row) for row in conn.execute(
                """
                SELECT ordinal, role, content, created_at, content_hash FROM messages
                WHERE conversation_id=? AND ordinal BETWEEN ? AND ? ORDER BY ordinal
                """,
                (conversation_id, start_ordinal, end_ordinal),
            )]
            conn.execute(
                """
                INSERT INTO conversation_segments(
                    segment_id, conversation_id, start_ordinal, end_ordinal, segment_kind,
                    project_id, title, summary, confidence, classification_source,
                    review_status, boundary_reason, segmenter_version, created_at, updated_at,
                    content_fingerprint, is_current, superseded_at, superseded_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, '')
                """,
                (segment_id, conversation_id, start_ordinal, end_ordinal, segment_kind,
                 project_id, title.strip(), summary.strip(), confidence,
                 classification_source, review_status, boundary_reason, segmenter_version,
                 now, now, _segment_content_fingerprint(segment_messages)),
            )
        return self.get_segment(segment_id)

    def get_segment(self, segment_id: str) -> dict | None:
        with self._connect() as conn:
            segment = self._row(conn.execute(
                "SELECT * FROM conversation_segments WHERE segment_id=?", (segment_id,)
            ).fetchone())
            if not segment:
                return None
            segment["work_item_links"] = [self._row(row) for row in conn.execute(
                """
                SELECT l.*, w.title AS work_item_title, w.project_id
                FROM segment_work_item_links l
                JOIN work_items w ON w.work_item_id=l.work_item_id
                WHERE l.segment_id=? ORDER BY CASE l.relation WHEN 'primary' THEN 0 ELSE 1 END
                """,
                (segment_id,),
            )]
        return segment

    def link_segment_work_item(
        self,
        segment_id: str,
        work_item_id: str,
        *,
        relation: str = "primary",
        confidence: float | None = None,
    ) -> dict:
        if relation not in SEGMENT_LINK_RELATIONS:
            raise WorkspaceStoreError("片段与工作项关系不正确")
        if confidence is not None and not 0 <= confidence <= 1:
            raise WorkspaceStoreError("关联置信度必须在 0–1 之间")
        segment = self.get_segment(segment_id)
        work_item = self.get_work_item(work_item_id)
        if not segment or not work_item:
            raise WorkspaceStoreError("会话片段或工作项不存在")
        if segment.get("project_id") and segment["project_id"] != work_item["project_id"]:
            raise WorkspaceStoreError("会话片段与工作项不属于同一个项目")
        try:
            with self._connect() as conn:
                if not segment.get("project_id"):
                    conn.execute(
                        "UPDATE conversation_segments SET project_id=?, updated_at=? WHERE segment_id=?",
                        (work_item["project_id"], _now(), segment_id),
                    )
                conn.execute(
                    """
                    INSERT INTO segment_work_item_links VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(segment_id, work_item_id, relation) DO UPDATE SET
                        confidence=excluded.confidence
                    """,
                    (segment_id, work_item_id, relation, confidence, _now()),
                )
        except sqlite3.IntegrityError as exc:
            raise WorkspaceStoreError("一个片段只能有一个主要工作项") from exc
        return self.get_segment(segment_id)

    def clear_segment_primary_work_item(self, segment_id: str) -> dict:
        if not self.get_segment(segment_id):
            raise WorkspaceStoreError("会话片段不存在")
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM segment_work_item_links WHERE segment_id=? AND relation='primary'",
                (segment_id,),
            )
        return self.get_segment(segment_id)

    def list_segments_for_work_item(self, work_item_id: str) -> list[dict]:
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(
                """
                SELECT s.segment_id
                FROM conversation_segments s
                JOIN segment_work_item_links l ON l.segment_id=s.segment_id
                WHERE l.work_item_id=?
                ORDER BY s.created_at
                """,
                (work_item_id,),
            )]
        return [self.get_segment(segment_id) for segment_id in ids]

    def get_segment_review_detail(
        self, segment_id: str, *, include_messages: bool = True
    ) -> dict | None:
        segment = self.get_segment(segment_id)
        if not segment:
            return None
        conversation = self.get_conversation(segment["conversation_id"])
        if not conversation:
            return None
        segment["conversation"] = {
            "conversation_id": conversation["conversation_id"],
            "source": conversation["source"],
            "external_session_id": conversation["external_session_id"],
            "title": conversation["title"],
            "original_project_path": conversation["original_project_path"],
            "started_at": conversation["started_at"],
            "updated_at": conversation["updated_at"],
            "resume_capable": conversation["resume_capable"],
        }
        if include_messages:
            segment["messages"] = self.list_messages_for_segment(segment_id)
        else:
            segment["message_count"] = segment["end_ordinal"] - segment["start_ordinal"] + 1
        return segment

    def list_work_item_source_details(self, work_item_id: str) -> list[dict]:
        if not self.get_work_item(work_item_id):
            raise WorkspaceStoreError("工作项不存在")
        return [
            self.get_segment_review_detail(segment["segment_id"])
            for segment in self.list_segments_for_work_item(work_item_id)
        ]

    def list_project_segment_details(
        self,
        project_id: str,
        *,
        segment_kind: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        if not self.get_project(project_id):
            raise WorkspaceStoreError("项目不存在")
        if segment_kind and segment_kind not in SEGMENT_KINDS:
            raise WorkspaceStoreError("会话片段类型不正确")
        query = """
            SELECT s.segment_id
            FROM conversation_segments s
            JOIN conversations c ON c.conversation_id=s.conversation_id
            WHERE s.project_id=? AND s.is_current=1
        """
        params = [project_id]
        if segment_kind:
            query += " AND s.segment_kind=?"
            params.append(segment_kind)
        query += " ORDER BY c.updated_at DESC, s.start_ordinal DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(query, params)]
        return [
            self.get_segment_review_detail(segment_id, include_messages=False)
            for segment_id in ids
        ]

    def create_classification_run(
        self,
        *,
        run_type: str,
        privacy_mode: str = "balanced",
        total_sources: int = 0,
        request: dict | None = None,
        retry_of_run_id: str | None = None,
    ) -> dict:
        if run_type not in {"full", "incremental"}:
            raise WorkspaceStoreError("整理任务类型不正确")
        if privacy_mode not in PRIVACY_MODES:
            raise WorkspaceStoreError("隐私模式不正确")
        now = _now()
        run_id = _new_id("RUN")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO classification_runs(
                    run_id, run_type, status, privacy_mode, stage, total_sources,
                    processed_sources, discovered_count, unclassified_count,
                    amd_call_count, credential_redaction_count, request_json,
                    retry_of_run_id, error_message, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, 'queued', ?, 0, 0, 0, 0, 0, ?, ?, '', ?, ?)
                """,
                (run_id, run_type, privacy_mode, int(total_sources), _json(request or {}),
                 retry_of_run_id, now, now),
            )
        return self.get_classification_run(run_id)

    def update_classification_run(self, run_id: str, **changes) -> dict:
        allowed = {
            "status", "stage", "total_sources", "processed_sources", "discovered_count",
            "unclassified_count", "amd_call_count", "credential_redaction_count",
            "error_message",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise WorkspaceStoreError(f"不支持更新字段：{sorted(unknown)[0]}")
        if "status" in changes and changes["status"] not in CLASSIFICATION_RUN_STATUSES:
            raise WorkspaceStoreError("整理任务状态不正确")
        for key in {
            "total_sources", "processed_sources", "discovered_count",
            "unclassified_count", "amd_call_count", "credential_redaction_count",
        }:
            if key in changes and int(changes[key]) < 0:
                raise WorkspaceStoreError("整理任务计数不能小于零")
            if key in changes:
                changes[key] = int(changes[key])
        current = self.get_classification_run(run_id)
        if not current:
            raise WorkspaceStoreError("整理任务不存在")
        resulting_total = changes.get("total_sources", current["total_sources"])
        resulting_processed = changes.get("processed_sources", current["processed_sources"])
        if resulting_total and resulting_processed > resulting_total:
            raise WorkspaceStoreError("已处理会话数不能超过总数")
        now = _now()
        if changes.get("status") == "running" and not current.get("started_at"):
            changes["started_at"] = now
        if changes.get("status") in {"completed", "failed", "cancelled"}:
            changes["finished_at"] = now
        changes["updated_at"] = now
        assignments = ", ".join(f"{key}=?" for key in changes)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE classification_runs SET {assignments} WHERE run_id=?",
                [*changes.values(), run_id],
            )
        return self.get_classification_run(run_id)

    def get_classification_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            item = self._row(conn.execute(
                "SELECT * FROM classification_runs WHERE run_id=?", (run_id,)
            ).fetchone())
            if item and not item.get("request"):
                project_ids = [row[0] for row in conn.execute(
                    """
                    SELECT DISTINCT s.project_id
                    FROM segment_classification_state cs
                    JOIN conversation_segments s ON s.segment_id=cs.segment_id
                    WHERE cs.run_id=? AND s.project_id IS NOT NULL
                    """,
                    (run_id,),
                )]
                if len(project_ids) == 1:
                    item["request"] = {
                        "project_id": project_ids[0],
                        "force": item["run_type"] == "full",
                        "limit": item["total_sources"],
                    }
        if item:
            total = item["total_sources"]
            item["completion_percent"] = round(item["processed_sources"] * 100 / total) if total else 0
        return item

    def list_classification_runs(self, limit: int = 20) -> list[dict]:
        normalized_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(
                "SELECT run_id FROM classification_runs ORDER BY created_at DESC LIMIT ?",
                (normalized_limit,),
            )]
        return [self.get_classification_run(run_id) for run_id in ids]

    def create_context_package(
        self,
        work_item_id: str,
        *,
        canonical_path: str,
        content_hash: str,
        segment_ids: list[str],
        project_copy_path: str = "",
    ) -> dict:
        if not self.get_work_item(work_item_id):
            raise WorkspaceStoreError("工作项不存在")
        for segment_id in segment_ids:
            if not self.get_segment(segment_id):
                raise WorkspaceStoreError(f"来源片段不存在：{segment_id}")
        with self._connect() as conn:
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM context_packages WHERE work_item_id=?",
                (work_item_id,),
            ).fetchone()[0]
            context_id = _new_id("CTX")
            created_at = _now()
            conn.execute(
                "INSERT INTO context_packages VALUES (?, ?, ?, ?, ?, ?, ?)",
                (context_id, work_item_id, version, canonical_path,
                 project_copy_path, content_hash, created_at),
            )
            for segment_id in dict.fromkeys(segment_ids):
                conn.execute(
                    "INSERT INTO context_package_segments VALUES (?, ?)",
                    (context_id, segment_id),
                )
        return self.get_context_package(context_id)

    def get_context_package(self, context_id: str) -> dict | None:
        with self._connect() as conn:
            item = self._row(conn.execute(
                "SELECT * FROM context_packages WHERE context_id=?", (context_id,)
            ).fetchone())
            if not item:
                return None
            item["segment_ids"] = [row[0] for row in conn.execute(
                "SELECT segment_id FROM context_package_segments WHERE context_id=? ORDER BY segment_id",
                (context_id,),
            )]
        return item

    def list_context_packages(self, work_item_id: str, limit: int = 30) -> list[dict]:
        if not self.get_work_item(work_item_id):
            raise WorkspaceStoreError("工作项不存在")
        normalized_limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(
                """
                SELECT context_id FROM context_packages
                WHERE work_item_id=?
                ORDER BY version DESC
                LIMIT ?
                """,
                (work_item_id, normalized_limit),
            )]
        return [self.get_context_package(context_id) for context_id in ids]

    def get_latest_context_package(self, work_item_id: str) -> dict | None:
        packages = self.list_context_packages(work_item_id, limit=1)
        return packages[0] if packages else None


workspace_store = WorkspaceStore()
