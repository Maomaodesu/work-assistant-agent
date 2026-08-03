import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from conversation_manager import ConversationStore
from external_cli_chat import (
    ExternalCliChatError,
    ExternalCliChatRunner,
    build_cli_environment,
    build_external_cli_command,
    external_session_id,
    extract_stream_text,
)


class _FakeRunner:
    def active(self, conversation_id):
        return False

    async def cancel(self, conversation_id):
        return True

    async def stream(self, conversation, prompt, *, configured_paths=None):
        yield "stage", "正在调用本机 Codex 会话"
        yield "message", "已从原会话继续。"
        yield "done", ""


class ExternalCliChatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.tool = self.root / "codex.cmd"
        self.tool.write_text("@echo off", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fixed_resume_commands_reject_unsafe_session_or_tool(self):
        command = build_external_cli_command(
            "codex", str(self.tool), "019f-session", "继续实现接口"
        )
        self.assertEqual(command[-5:], ["exec", "resume", "--json", "019f-session", "继续实现接口"])
        with self.assertRaises(ExternalCliChatError):
            build_external_cli_command("codex", str(self.tool), "bad & command", "hello")
        with self.assertRaises(ExternalCliChatError):
            build_external_cli_command("claude", str(self.root / "missing.exe"), "session", "hello")

    def test_extracts_codex_and_claude_machine_readable_text(self):
        self.assertEqual(
            extract_stream_text("codex", {"type": "item.completed", "item": {"type": "agent_message", "text": "Codex 回复"}}),
            "Codex 回复",
        )

    def test_runner_streams_a_local_cli_json_event(self):
        fake_cli = self.root / "fake-codex.cmd"
        fake_cli.write_text(
            '@echo {"type":"item.completed","item":{"type":"agent_message","text":"bridge works"}}\r\n',
            encoding="utf-8",
        )
        conversation = {
            "id": "codex:session-1", "source": "codex", "project_path": str(self.project),
        }

        async def collect():
            runner = ExternalCliChatRunner()
            return [event async for event in runner.stream(
                conversation, "continue", configured_paths={"codex": str(fake_cli)}
            )]

        events = asyncio.run(collect())
        self.assertIn(("message", "bridge works"), events)
        self.assertIn(("done", ""), events)
        self.assertEqual(
            extract_stream_text("claude", {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "Claude 增量"}}}),
            "Claude 增量",
        )

    def test_external_session_id_only_uses_trusted_source_prefix(self):
        self.assertEqual(external_session_id({"source": "claude", "id": "claude:session-1"}), "session-1")
        with self.assertRaises(ExternalCliChatError):
            external_session_id({"source": "codex", "id": "claude:session-1"})

    def test_codex_uses_the_configuration_home_that_owns_imported_session(self):
        codex_home = self.root / ".codex"
        source_path = codex_home / "sessions" / "2026" / "session.jsonl"
        source_path.parent.mkdir(parents=True)
        source_path.write_text("{}\n", encoding="utf-8")
        environment = build_cli_environment(
            "codex", {"source_path": str(source_path)}, {"CODEX_HOME": "C:/sandbox/.codex"}
        )

        self.assertEqual(environment["CODEX_HOME"], str(codex_home.resolve()))

    def test_api_only_bridges_existing_readonly_external_conversation(self):
        store = ConversationStore(self.root / "conversations.db")
        source_path = self.root / "session.jsonl"
        source_path.write_text("{}\n", encoding="utf-8")
        store._upsert_external({
            "id": "codex:session-1", "source": "codex", "title": "Imported",
            "project_path": str(self.project), "created_at": "2026-07-29T00:00:00+00:00",
            "updated_at": "2026-07-29T00:00:01+00:00",
            "messages": [{"role": "user", "content": "历史消息", "created_at": "2026-07-29T00:00:00+00:00"}],
        }, source_path)
        fake_runner = _FakeRunner()
        with (
            patch.object(server, "conversation_store", store),
            patch.object(server, "external_cli_chat_runner", fake_runner),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
            patch.object(store, "import_external", return_value={"codex": 1, "claude": 0}),
        ):
            with TestClient(server.app) as client:
                response = client.post(
                    "/api/external-conversations/codex%3Asession-1/chat",
                    json={"message": "继续", "request_id": "test-request"},
                )
                missing = client.post(
                    "/api/external-conversations/missing/chat", json={"message": "继续"}
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn("已从原会话继续", response.text)
        self.assertIn("event: done", response.text)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
