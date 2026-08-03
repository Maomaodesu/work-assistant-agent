"""根据外部会话工作目录匹配新数据库中的项目根目录。"""

import os
import re
from pathlib import Path

from workspace_store import WorkspaceStore, workspace_store


def normalize_session_path(value: str | Path) -> str:
    """统一 Windows、MSYS(`/c/...`) 与 WSL(`/mnt/c/...`) 路径。"""
    raw = os.path.expandvars(str(value or "").strip().strip('"'))
    if not raw:
        return ""
    raw = raw.removeprefix("\\\\?\\")
    wsl = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", raw)
    msys = re.match(r"^/([a-zA-Z])(?:/(.*))?$", raw)
    if wsl:
        raw = f"{wsl.group(1).upper()}:/{wsl.group(2) or ''}"
    elif msys:
        raw = f"{msys.group(1).upper()}:/{msys.group(2) or ''}"
    path = Path(raw).expanduser()
    if path.exists() and path.is_file():
        path = path.parent
    return os.path.normcase(os.path.normpath(str(path.resolve())))


def _is_descendant(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError):
        return False


class ConversationProjectMatcher:
    def __init__(self, store: WorkspaceStore | None = None):
        self.store = store or workspace_store

    def candidates_for_path(self, path_value: str) -> list[dict]:
        conversation_path = normalize_session_path(path_value)
        if not conversation_path:
            return []
        by_project = {}
        for root in self.store.list_project_roots():
            root_path = normalize_session_path(root["path"])
            if not root_path:
                continue
            if conversation_path == root_path:
                method, confidence = "exact_root", 1.0
            elif _is_descendant(conversation_path, root_path):
                method, confidence = "root_descendant", 0.98
            else:
                continue
            candidate = {
                "project_id": root["project_id"],
                "project_name": root["project_name"],
                "root_id": root["root_id"],
                "root_path": root["path"],
                "match_method": method,
                "confidence": confidence,
                "root_length": len(root_path),
            }
            previous = by_project.get(root["project_id"])
            if not previous or (candidate["root_length"], confidence) > (
                previous["root_length"], previous["confidence"]
            ):
                by_project[root["project_id"]] = candidate
        if not by_project:
            return []
        candidates = list(by_project.values())
        longest = max(candidate["root_length"] for candidate in candidates)
        return sorted(
            [candidate for candidate in candidates if candidate["root_length"] == longest],
            key=lambda candidate: candidate["project_name"].lower(),
        )

    def match_conversation(self, conversation: dict) -> dict:
        existing = conversation.get("project_matches") or self.store.get_conversation_project_matches(
            conversation["conversation_id"]
        )
        manual = next(
            (match for match in existing if match["match_source"] == "manual" and match["is_primary"]),
            None,
        )
        if manual:
            return {"state": "manual", "matches": existing}

        candidates = self.candidates_for_path(conversation.get("original_project_path", ""))
        if len(candidates) == 1:
            candidate = candidates[0]
            matches = self.store.replace_auto_project_matches(
                conversation["conversation_id"],
                [{**candidate, "is_primary": True}],
            )
            return {"state": "matched", "matches": matches}
        if len(candidates) > 1:
            matches = self.store.replace_auto_project_matches(
                conversation["conversation_id"],
                [{**candidate, "is_primary": False} for candidate in candidates],
            )
            return {"state": "ambiguous", "matches": matches}
        self.store.replace_auto_project_matches(conversation["conversation_id"], [])
        return {
            "state": "no_path" if not conversation.get("original_project_path") else "unassigned",
            "matches": [],
        }

    def match_all(self) -> dict:
        conversations = self.store.list_conversations()
        result = {
            "total": len(conversations),
            "matched": 0,
            "manual": 0,
            "ambiguous": 0,
            "unassigned": 0,
            "no_path": 0,
        }
        for conversation in conversations:
            outcome = self.match_conversation(conversation)
            result[outcome["state"]] += 1
        result["project_count"] = len(self.store.list_projects(status="active"))
        return result


conversation_project_matcher = ConversationProjectMatcher()
