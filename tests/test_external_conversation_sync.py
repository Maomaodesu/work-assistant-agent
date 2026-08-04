import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from external_conversation_sync import ExternalConversationSync
from tests.test_conversations import claude_records, codex_records, write_jsonl
from workspace_store import WorkspaceStore


class ExternalConversationSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.codex_root = self.root / "codex"
        self.claude_root = self.root / "claude-projects"
        self.codex_path = self.codex_root / "sessions" / "2026" / "07" / "codex.jsonl"
        self.claude_path = self.claude_root / "demo" / "claude.jsonl"
        write_jsonl(self.codex_path, codex_records())
        write_jsonl(self.claude_path, claude_records())
        self.store = WorkspaceStore(self.root / "workspace.db")
        self.sync = ExternalConversationSync(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self):
        return self.sync.sync(
            codex_root=self.codex_root,
            claude_root=self.claude_root,
        )

    def test_first_sync_imports_both_sources_into_new_database(self):
        result = self._run()
        conversations = self.store.list_conversations()

        self.assertEqual(result["total_files"], 2)
        self.assertEqual(result["processed_files"], 2)
        self.assertEqual(result["imported"], 2)
        self.assertEqual(result["messages"], 4)
        self.assertEqual({item["source"] for item in conversations}, {"codex", "claude"})
        self.assertEqual({item["external_session_id"] for item in conversations}, {
            "codex-session", "claude-session"
        })
        self.assertTrue(all(item["resume_capable"] for item in conversations))
        self.assertTrue(all(item["content_hash"] for item in conversations))
        self.assertTrue(all(item["source_modified_ns"] > 0 for item in conversations))
        self.assertTrue(all(item["message_count"] == 2 for item in conversations))
        self.assertIn("C:/workspace", " ".join(item["original_project_path"] for item in conversations))

        codex = next(item for item in conversations if item["source"] == "codex")
        messages = self.store.list_messages(codex["conversation_id"])
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["created_at"], "2026-07-18T10:00:01")

    def test_second_sync_skips_unchanged_files_without_reparsing(self):
        first = self._run()
        with (
            patch("external_conversation_sync.parse_codex_session") as codex_parser,
            patch("external_conversation_sync.parse_claude_session") as claude_parser,
        ):
            second = self._run()

        self.assertEqual(first["imported"], 2)
        self.assertEqual(second["unchanged"], 2)
        self.assertEqual(second["imported"], 0)
        self.assertEqual(second["updated"], 0)
        codex_parser.assert_not_called()
        claude_parser.assert_not_called()

    def test_mtime_change_with_same_hash_updates_metadata_without_message_rewrite(self):
        self._run()
        conversation = self.store.find_conversation("codex", "codex-session")
        before_messages = self.store.list_messages(conversation["conversation_id"])
        stat = self.codex_path.stat()
        os.utime(self.codex_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

        with patch.object(self.store, "replace_messages", wraps=self.store.replace_messages) as replace:
            result = self._run()

        refreshed = self.store.find_conversation("codex", "codex-session")
        self.assertEqual(result["unchanged"], 2)
        replace.assert_not_called()
        self.assertGreater(refreshed["source_modified_ns"], conversation["source_modified_ns"])
        self.assertEqual(
            [item["content"] for item in self.store.list_messages(conversation["conversation_id"])],
            [item["content"] for item in before_messages],
        )

    def test_changed_content_updates_only_changed_conversation(self):
        self._run()
        records = codex_records()
        records.append({
            "timestamp": "2026-07-18T10:00:04",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "One more request."},
        })
        write_jsonl(self.codex_path, records)

        result = self._run()
        codex = self.store.find_conversation("codex", "codex-session")

        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(len(self.store.list_messages(codex["conversation_id"])), 3)

    def test_subagent_or_empty_source_is_recorded_as_skipped_incrementally(self):
        subagent_path = self.codex_root / "sessions" / "subagent.jsonl"
        write_jsonl(subagent_path, codex_records("subagent", subagent=True))

        first = self._run()
        second = self._run()
        state = self.store.get_source_state(subagent_path)

        self.assertEqual(first["skipped"], 1)
        self.assertEqual(state["import_status"], "skipped")
        self.assertEqual(second["unchanged"], 3)

    def test_discovery_has_no_legacy_two_hundred_session_cap(self):
        for index in range(205):
            path = self.codex_root / "sessions" / "bulk" / f"{index:03d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        files = self.sync.discover_files(self.codex_root, self.claude_root)
        self.assertEqual(len(files["codex"]), 206)

    def test_sync_api_exposes_counts_without_invoking_amd(self):
        fake_sync = SimpleNamespace(sync=lambda: {
            "total_files": 2, "processed_files": 2, "imported": 2, "updated": 0,
            "unchanged": 0, "skipped": 0, "errors": 0, "messages": 4,
            "sources": {}, "error_details": [],
        })
        with (
            patch.object(server, "external_conversation_sync", fake_sync),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                response = client.post("/api/workspace/conversations/sync")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["imported"], 2)

    def test_global_sync_updates_conversation_library_and_workspace_index(self):
        fake_sync = SimpleNamespace(sync=lambda: {
            "total_files": 2, "processed_files": 1, "imported": 1, "updated": 0,
            "unchanged": 1, "skipped": 0, "errors": 0, "messages": 3,
            "sources": {}, "error_details": [],
        })
        fake_conversations = SimpleNamespace(import_external=lambda: {"claude": 1, "codex": 2})
        fake_matcher = SimpleNamespace(match_all=lambda: {
            "total": 3, "matched": 2, "manual": 0, "ambiguous": 0,
            "unassigned": 1, "no_path": 0, "project_count": 1,
        })
        with (
            patch.object(server, "external_conversation_sync", fake_sync),
            patch.object(server, "conversation_store", fake_conversations),
            patch.object(server, "conversation_project_matcher", fake_matcher),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                response = client.post("/api/conversations/sync-all")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["conversations"]["codex"], 2)
        self.assertEqual(response.json()["workspace"]["imported"], 1)
        self.assertEqual(response.json()["project_matching"]["matched"], 2)


if __name__ == "__main__":
    unittest.main()
