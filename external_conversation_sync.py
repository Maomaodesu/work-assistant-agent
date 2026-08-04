"""将 Claude/Codex 原始会话增量同步到新的 workspace 数据库。"""

import hashlib
from pathlib import Path

from conversation_manager import (
    load_codex_title_index,
    parse_claude_session,
    parse_codex_session,
)
from workspace_store import WorkspaceStore, workspace_store


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalConversationSync:
    def __init__(self, store: WorkspaceStore | None = None):
        self.store = store or workspace_store

    @staticmethod
    def discover_files(
        codex_root: Path | None = None,
        claude_root: Path | None = None,
    ) -> dict[str, list[Path]]:
        codex_root = codex_root or Path.home() / ".codex"
        codex_sessions = codex_root / "sessions"
        claude_root = claude_root or Path.home() / ".claude" / "projects"
        codex_files = list(codex_sessions.rglob("*.jsonl")) if codex_sessions.exists() else []
        claude_files = list(claude_root.rglob("*.jsonl")) if claude_root.exists() else []
        return {
            "codex": sorted(codex_files, key=lambda path: str(path).lower()),
            "claude": sorted(claude_files, key=lambda path: str(path).lower()),
        }

    def sync(
        self,
        *,
        codex_root: Path | None = None,
        claude_root: Path | None = None,
        max_sources: int | None = None,
    ) -> dict:
        files_by_source = self.discover_files(codex_root, claude_root)
        codex_index_root = codex_root or Path.home() / ".codex"
        title_index = load_codex_title_index(codex_index_root)
        result = {
            "total_files": sum(len(files) for files in files_by_source.values()),
            "processed_files": 0,
            "imported": 0,
            "updated": 0,
            "unchanged": 0,
            "skipped": 0,
            "errors": 0,
            "messages": 0,
            "sources": {
                "codex": {"imported": 0, "updated": 0, "unchanged": 0, "skipped": 0, "errors": 0},
                "claude": {"imported": 0, "updated": 0, "unchanged": 0, "skipped": 0, "errors": 0},
            },
            "error_details": [],
        }
        remaining = max_sources
        for source in ("codex", "claude"):
            parser = (
                (lambda path: parse_codex_session(path, title_index))
                if source == "codex" else parse_claude_session
            )
            for path in files_by_source[source]:
                if remaining is not None and remaining <= 0:
                    return result
                if remaining is not None:
                    remaining -= 1
                result["processed_files"] += 1
                try:
                    self._sync_file(path, source, parser, result)
                except Exception as exc:
                    result["errors"] += 1
                    result["sources"][source]["errors"] += 1
                    try:
                        stat = path.stat()
                        content_hash = _file_sha256(path)
                        self.store.save_source_state(
                            path,
                            source=source,
                            source_modified_ns=stat.st_mtime_ns,
                            content_hash=content_hash,
                            import_status="error",
                            last_error=str(exc),
                        )
                    except Exception:
                        pass
                    if len(result["error_details"]) < 20:
                        result["error_details"].append({"path": str(path), "error": str(exc)})
        return result

    def _sync_file(self, path: Path, source: str, parser, result: dict):
        stat = path.stat()
        state = self.store.get_source_state(path)
        if (
            state
            and state["import_status"] != "error"
            and state["source_modified_ns"] == stat.st_mtime_ns
        ):
            result["unchanged"] += 1
            result["sources"][source]["unchanged"] += 1
            return

        content_hash = _file_sha256(path)
        if (
            state
            and state["import_status"] != "error"
            and state["content_hash"] == content_hash
        ):
            if state.get("conversation_id"):
                self.store.touch_conversation_source(
                    state["conversation_id"],
                    source_modified_ns=stat.st_mtime_ns,
                    content_hash=content_hash,
                )
            self.store.save_source_state(
                path,
                source=source,
                source_modified_ns=stat.st_mtime_ns,
                content_hash=content_hash,
                import_status=state["import_status"],
                conversation_id=state.get("conversation_id"),
            )
            result["unchanged"] += 1
            result["sources"][source]["unchanged"] += 1
            return

        parsed = parser(path)
        if not parsed:
            self.store.save_source_state(
                path,
                source=source,
                source_modified_ns=stat.st_mtime_ns,
                content_hash=content_hash,
                import_status="skipped",
            )
            result["skipped"] += 1
            result["sources"][source]["skipped"] += 1
            return

        external_session_id = parsed["id"].split(":", 1)[-1]
        existing = self.store.find_conversation(source, external_session_id)
        messages = parsed["messages"]
        conversation = self.store.upsert_conversation(
            source,
            external_session_id,
            title=parsed["title"],
            original_project_path=parsed.get("project_path", ""),
            source_path=str(path),
            started_at=parsed["created_at"],
            updated_at=parsed["updated_at"],
            source_modified_ns=stat.st_mtime_ns,
            content_hash=content_hash,
            resume_capable=True,
            metadata={"importer": "external_conversation_sync_v1"},
        )
        # Keep the stable source prefix intact.  Replacing the whole message
        # list would cascade-delete previously classified segments and their
        # work-item/context evidence whenever a live session simply appends.
        self.store.sync_messages(conversation["conversation_id"], messages)
        self.store.save_source_state(
            path,
            source=source,
            source_modified_ns=stat.st_mtime_ns,
            content_hash=content_hash,
            import_status="imported",
            conversation_id=conversation["conversation_id"],
        )
        outcome = "updated" if existing else "imported"
        result[outcome] += 1
        result["sources"][source][outcome] += 1
        result["messages"] += len(messages)


external_conversation_sync = ExternalConversationSync()
