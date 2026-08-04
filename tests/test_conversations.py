import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server
from conversation_manager import (
    ConversationStore,
    detect_project,
    parse_claude_session,
    parse_codex_session,
)


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def codex_records(session_id="codex-session", *, subagent=False):
    source = {"subagent": {"name": "worker"}} if subagent else "desktop"
    return [
        {
            "timestamp": "2026-07-18T10:00:00",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "session_id": session_id,
                "timestamp": "2026-07-18T10:00:00",
                "cwd": "C:/workspace/codex-demo",
                "source": source,
                "parent_thread_id": "parent" if subagent else None,
            },
        },
        {
            "timestamp": "2026-07-18T10:00:01",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "Please inspect the project."},
        },
        {
            "timestamp": "2026-07-18T10:00:02",
            "type": "response_item",
            "payload": {"type": "reasoning", "encrypted_content": "private"},
        },
        {
            "timestamp": "2026-07-18T10:00:03",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "The project is ready."},
        },
    ]


def claude_records(session_id="claude-session"):
    return [
        {
            "type": "file-history-snapshot",
            "sessionId": session_id,
            "snapshot": {"secret": "ignored"},
        },
        {
            "type": "user",
            "sessionId": session_id,
            "cwd": "C:/workspace/claude-demo",
            "timestamp": "2026-07-18T11:00:00",
            "message": {"role": "user", "content": "Implement the controller."},
        },
        {
            "type": "assistant",
            "sessionId": session_id,
            "cwd": "C:/workspace/claude-demo",
            "timestamp": "2026-07-18T11:00:01",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I prepared the controller."},
                    {"type": "tool_use", "name": "Write", "input": {"path": "secret"}},
                ],
            },
        },
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": "2026-07-18T11:00:02",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "ignored tool output"}],
            },
        },
    ]


class ConversationTests(unittest.TestCase):
    def test_page_bootstrap_does_not_auto_sync_external_sessions(self):
        template = Path("templates/index.html").read_text(encoding="utf-8")
        bootstrap_body = template.split(
            "async function bootstrapConversations() {", 1
        )[1].split("}", 1)[0]
        self.assertNotIn("syncExternalSessions()", bootstrap_body)
        self.assertNotIn("ensureCurrentConversation()", bootstrap_body)
        self.assertIn("onclick=\"syncExternalSessions()\"", template)

    def test_conversation_page_uses_issue_list_and_readonly_history_comments(self):
        template = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn('id="conversationList" class="conversation-issue-list"', template)
        self.assertIn("新建 work_assistant 会话", template)
        self.assertIn('id="syncSessionsBtn"', template)
        self.assertIn("/api/conversations/sync-all", template)
        self.assertIn("analyzeHistory()", template)
        self.assertIn("分析历史", template)
        self.assertIn("请生成开发工作回顾", template)
        self.assertIn("文件变更、验证情况和会话内结论", template)
        self.assertIn('id="sendBtn" class="send-btn" onclick="sendMessage()">发送</button>', template)
        self.assertIn("只读历史", template)
        self.assertIn("conversationProjectFilter", template)
        self.assertIn("CONVERSATION_FILTER_STATE", template)
        self.assertIn('requestKind === "summary"', template)
        self.assertIn("/summary", template)
        self.assertIn("work-assistant-entry", template)
        self.assertIn("historyAnalysisActions", template)
        self.assertIn('analysisLabel: "会话分析"', template)
        self.assertIn("deleteAnalysisComment", template)
        self.assertIn("commentId: comment.id", template)
        self.assertNotIn('appendMessage("user", comment.prompt', template)
        self.assertIn("formatConversationTime(item.updated_at)", template)

    def test_project_detection_separates_project_and_casual_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "demo-project"
            nested = project / "src" / "main"
            nested.mkdir(parents=True)
            (project / ".git").mkdir()
            casual = root / "random-chat"
            casual.mkdir()

            project_result = detect_project(str(nested))
            casual_result = detect_project(str(casual))

        self.assertEqual(project_result["group_type"], "project")
        self.assertEqual(project_result["project_name"], "demo-project")
        self.assertEqual(casual_result["group_type"], "casual")

    def test_codex_parser_keeps_chat_and_skips_reasoning_and_subagents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_path = root / "main.jsonl"
            subagent_path = root / "subagent.jsonl"
            write_jsonl(main_path, codex_records())
            write_jsonl(subagent_path, codex_records("subagent-session", subagent=True))

            parsed = parse_codex_session(main_path, {"codex-session": "Indexed title"})
            subagent = parse_codex_session(subagent_path)

        self.assertEqual(parsed["title"], "Indexed title")
        self.assertEqual(parsed["project_path"], "C:/workspace/codex-demo")
        self.assertEqual(
            [(item["role"], item["content"]) for item in parsed["messages"]],
            [
                ("user", "Please inspect the project."),
                ("assistant", "The project is ready."),
            ],
        )
        self.assertNotIn("private", str(parsed))
        self.assertIsNone(subagent)

    def test_claude_parser_keeps_text_and_skips_tools_and_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "claude.jsonl"
            write_jsonl(path, claude_records())
            parsed = parse_claude_session(path)

        self.assertEqual(parsed["project_path"], "C:/workspace/claude-demo")
        self.assertEqual(len(parsed["messages"]), 2)
        self.assertEqual(parsed["messages"][1]["content"], "I prepared the controller.")
        self.assertNotIn("tool output", str(parsed))
        self.assertNotIn("snapshot", str(parsed))

    def test_store_crud_and_incremental_external_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ConversationStore(root / "conversations.db")
            store.ensure_work_assistant("session-local", "My first local message")
            store.append_exchange("session-local", "hello", "hi", "TASK-1")
            renamed = store.rename("session-local", "Renamed conversation")
            archived = store.set_archived("session-local", True)
            restored = store.set_archived("session-local", False)

            codex_root = root / ".codex"
            codex_path = codex_root / "sessions" / "2026" / "07" / "18" / "rollout.jsonl"
            write_jsonl(codex_path, codex_records())
            write_jsonl(
                codex_root / "session_index.jsonl",
                [{"id": "codex-session", "thread_name": "Codex imported"}],
            )
            claude_root = root / ".claude" / "projects"
            claude_path = claude_root / "demo" / "claude-session.jsonl"
            write_jsonl(claude_path, claude_records())

            imported = store.import_external(codex_root, claude_root)
            imported_again = store.import_external(codex_root, claude_root)
            conversations = store.list()
            codex_conversation = store.get("codex:codex-session")
            comment = store.add_comment("codex:codex-session", "总结这段对话", "项目已经准备完成。")
            comments = store.comments("codex:codex-session")
            deleted_comment = store.delete_comment("codex:codex-session", comment["id"])
            comments_after_delete = store.comments("codex:codex-session")
            store.delete("codex:codex-session")
            after_delete_sync = store.import_external(codex_root, claude_root)

        self.assertEqual(renamed["title"], "Renamed conversation")
        self.assertTrue(archived["archived"])
        self.assertFalse(restored["archived"])
        self.assertEqual(imported["codex"], 1)
        self.assertEqual(imported["claude"], 1)
        self.assertGreaterEqual(imported_again["skipped"], 2)
        self.assertEqual({item["source"] for item in conversations}, {"work_assistant", "codex", "claude"})
        self.assertTrue(codex_conversation["readonly"])
        self.assertEqual(comment["prompt"], "总结这段对话")
        self.assertEqual([item["content"] for item in comments], ["项目已经准备完成。"])
        self.assertTrue(deleted_comment)
        self.assertEqual(comments_after_delete, [])
        self.assertEqual(after_delete_sync["codex"], 0)

    def test_permanent_delete_removes_only_the_verified_external_source_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ConversationStore(root / "conversations.db")
            codex_root = root / ".codex"
            source_path = codex_root / "sessions" / "2026" / "07" / "18" / "remove-me.jsonl"
            write_jsonl(source_path, codex_records("remove-me"))
            store.import_external(codex_root, root / ".claude" / "projects")
            store.set_archived("codex:remove-me", True)

            deleted = store.delete(
                "codex:remove-me",
                delete_source=True,
                allowed_source_roots=(codex_root / "sessions",),
            )

            self.assertTrue(deleted["source_deleted"])
            self.assertFalse(source_path.exists())
            self.assertIsNone(store.get("codex:remove-me"))

    def test_conversation_api_create_list_messages_rename_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.db")
            checkpointer = MagicMock()
            with (
                patch.object(server, "conversation_store", store),
                patch.object(server, "session_checkpointer", checkpointer),
                patch.object(server.settings_service, "is_setup_complete", return_value=True),
            ):
                client = TestClient(server.app)
                created = client.post(
                    "/api/conversations",
                    json={"session_id": "session-api", "title": "API conversation"},
                )
                store.append_exchange("session-api", "user text", "assistant text")
                comment = store.add_comment("session-api", "分析", "可删除的分析")
                listed = client.get("/api/conversations")
                messages = client.get("/api/conversations/session-api/messages")
                deleted_comment = client.delete(
                    f"/api/conversations/session-api/comments/{comment['id']}"
                )
                missing_comment = client.delete(
                    f"/api/conversations/session-api/comments/{comment['id']}"
                )
                renamed = client.patch(
                    "/api/conversations/session-api",
                    json={"title": "Renamed through API"},
                )
                archived = client.patch(
                    "/api/conversations/session-api",
                    json={"archived": True},
                )
                restored = client.patch(
                    "/api/conversations/session-api",
                    json={"archived": False},
                )
                rejected_delete = client.delete("/api/conversations/session-api")
                client.patch(
                    "/api/conversations/session-api",
                    json={"archived": True},
                )
                deleted = client.delete("/api/conversations/session-api")
                client.close()

        self.assertEqual(created.status_code, 201)
        self.assertEqual(len(listed.json()), 1)
        self.assertEqual(len(messages.json()["messages"]), 2)
        self.assertEqual(deleted_comment.json(), {"ok": True, "comment_id": comment["id"]})
        self.assertEqual(missing_comment.status_code, 404)
        self.assertEqual(renamed.json()["title"], "Renamed through API")
        self.assertTrue(archived.json()["archived"])
        self.assertFalse(restored.json()["archived"])
        self.assertEqual(rejected_delete.status_code, 400)
        self.assertIn("请先归档", rejected_delete.json()["error"])
        self.assertEqual(deleted.json(), {"ok": True, "source_deleted": False})
        checkpointer.delete_thread.assert_called_once_with("session-api")

    def test_busy_session_rejects_concurrent_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.db")
            busy_lock = asyncio.Lock()
            asyncio.run(busy_lock.acquire())
            server._session_locks["session-busy"] = busy_lock
            try:
                with (
                    patch.object(server, "conversation_store", store),
                    patch.object(server.settings_service, "is_setup_complete", return_value=True),
                ):
                    client = TestClient(server.app)
                    response = client.post(
                        "/api/chat",
                        json={"message": "second request", "session_id": "session-busy"},
                    )
                    client.close()
            finally:
                if busy_lock.locked():
                    busy_lock.release()
                server._session_locks.pop("session-busy", None)

        self.assertEqual(response.status_code, 409)
        self.assertIn("正在处理", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
