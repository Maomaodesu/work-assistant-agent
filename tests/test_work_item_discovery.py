import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from work_item_discovery import (
    CLASSIFIER_VERSION,
    WorkItemDiscovery,
    redact_sensitive_text,
)
from workspace_store import WorkspaceStore


class FakeAMDClient:
    def __init__(self, existing_item_id=None):
        self.existing_item_id = existing_item_id
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        decisions = []
        for segment in payload["segments"]:
            text = segment["conversation"].lower()
            if "rolling pe" in text:
                decisions.append({
                    "segment_id": segment["segment_id"],
                    "segment_kind": "work_item",
                    "action": "create",
                    "matched_work_item_id": None,
                    "title": "Rolling PE API",
                    "item_type": "feature",
                    "goal": "提供滚动市盈率接口",
                    "summary": "实现并测试滚动市盈率接口",
                    "confidence": 0.93,
                })
            elif "login" in text:
                decisions.append({
                    "segment_id": segment["segment_id"],
                    "segment_kind": "work_item",
                    "action": "match",
                    "matched_work_item_id": self.existing_item_id,
                    "title": "Fix login bug",
                    "item_type": "bug",
                    "goal": "修复登录",
                    "summary": "继续修复已有登录问题",
                    "confidence": 0.9,
                })
            else:
                decisions.append({
                    "segment_id": segment["segment_id"],
                    "segment_kind": "project",
                    "action": "none",
                    "matched_work_item_id": None,
                    "title": "Project discussion",
                    "item_type": "other",
                    "goal": "",
                    "summary": "项目级讨论",
                    "confidence": 0.8,
                })
        response = json.dumps({"decisions": decisions}, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class WorkItemDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        project_root = root / "demo"
        project_root.mkdir()
        self.store = WorkspaceStore(root / "workspace.db")
        self.project = self.store.create_project("Demo", [str(project_root)])
        conversation = self.store.upsert_conversation(
            "codex", "discovery-session", title="mixed work",
            original_project_path=str(project_root),
            started_at="2026-07-18T10:00:00+00:00",
            updated_at="2026-07-18T10:06:00+00:00",
        )
        self.store.set_manual_conversation_project(
            conversation["conversation_id"], self.project["project_id"]
        )
        self.store.replace_messages(conversation["conversation_id"], [
            {"ordinal": 0, "role": "user", "content": "Implement rolling PE API api_key=supersecretvalue", "created_at": "2026-07-18T10:00:00+00:00"},
            {"ordinal": 1, "role": "assistant", "content": "Use C:\\demo\\RollingPE.java and run mvn test", "created_at": "2026-07-18T10:01:00+00:00"},
            {"ordinal": 2, "role": "user", "content": "Continue fixing the login bug", "created_at": "2026-07-18T10:02:00+00:00"},
            {"ordinal": 3, "role": "assistant", "content": "Checking LoginService", "created_at": "2026-07-18T10:03:00+00:00"},
            {"ordinal": 4, "role": "user", "content": "好的", "created_at": "2026-07-18T10:04:00+00:00"},
        ])
        self.segments = [
            self.store.create_segment(
                conversation["conversation_id"], 0, 1,
                project_id=self.project["project_id"], title="Rolling PE",
            ),
            self.store.create_segment(
                conversation["conversation_id"], 2, 3,
                project_id=self.project["project_id"], title="Login",
            ),
            self.store.create_segment(
                conversation["conversation_id"], 4, 4,
                project_id=self.project["project_id"], title="确认",
            ),
        ]
        self.login_item = self.store.create_work_item(
            self.project["project_id"], "Fix login bug", item_type="bug"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_redactor_removes_credentials_but_keeps_development_context(self):
        source = (
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n"
            "api_key=secret-value-123\n"
            "POSTGRES_PASSWORD: local-password-123\n"
            "clone https://alice:password@example.test/repo.git\n"
            "run mvn test in C:\\demo\\pom.xml"
        )
        result = redact_sensitive_text(source)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", result.text)
        self.assertNotIn("secret-value-123", result.text)
        self.assertNotIn("local-password-123", result.text)
        self.assertNotIn("alice:password", result.text)
        self.assertIn("mvn test", result.text)
        self.assertIn("C:\\demo\\pom.xml", result.text)
        self.assertGreaterEqual(result.total, 3)

    def test_discovers_suggestion_matches_existing_and_is_incremental(self):
        fake_client = FakeAMDClient(self.login_item["work_item_id"])
        client_factory_calls = []

        def factory():
            client_factory_calls.append(True)
            return fake_client

        discovery = WorkItemDiscovery(self.store, factory)
        first = discovery.discover()

        self.assertEqual(first["created_work_items"], 1)
        self.assertEqual(first["matched_work_items"], 1)
        self.assertEqual(first["casual_segments"], 1)
        self.assertEqual(first["amd_analyzed_segments"], 2)
        self.assertEqual(first["credential_redaction_count"], 1)
        self.assertEqual(len(fake_client.calls), 1)
        sent_payload = fake_client.calls[0]["messages"][1]["content"]
        self.assertNotIn("supersecretvalue", sent_payload)
        self.assertIn("RollingPE.java", sent_payload)

        suggestions = self.store.list_work_items(
            self.project["project_id"], status="suggested"
        )
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["title"], "Rolling PE API")
        self.assertEqual(suggestions[0]["created_at"], "2026-07-18T10:00:00+00:00")
        self.assertEqual(
            self.store.get_segment(self.segments[1]["segment_id"])["work_item_links"][0]["work_item_id"],
            self.login_item["work_item_id"],
        )
        state = self.store.get_segment_classification_state(self.segments[0]["segment_id"])
        self.assertEqual(state["classifier_version"], CLASSIFIER_VERSION)
        self.assertEqual(state["redaction"]["API_KEY"], 1)

        second = discovery.discover()
        self.assertEqual(second["unchanged_segments"], 3)
        self.assertEqual(len(client_factory_calls), 1)
        self.assertEqual(len(self.store.list_work_items(self.project["project_id"], status="suggested")), 1)

    def test_handoff_marker_links_new_conversation_to_existing_work_item_without_amd(self):
        conversation = self.store.upsert_conversation(
            "codex", "handoff-session", title="继续实现登录修复",
            original_project_path=str(self.project["roots"][0]["path"]),
        )
        self.store.set_manual_conversation_project(conversation["conversation_id"], self.project["project_id"])
        self.store.replace_messages(conversation["conversation_id"], [{
            "ordinal": 0, "role": "user",
            "content": f"请继续开发。Work Assistant 工作项 ID：`{self.login_item['work_item_id']}`。",
        }])
        segment = self.store.create_segment(
            conversation["conversation_id"], 0, 0, project_id=self.project["project_id"], title="继续开发"
        )
        discovery = WorkItemDiscovery(self.store, lambda: FakeAMDClient(self.login_item["work_item_id"]))

        result = discovery.discover()
        detail = self.store.get_segment(segment["segment_id"])

        self.assertEqual(result["matched_work_items"], 2)
        self.assertEqual(result["amd_analyzed_segments"], 2)
        self.assertEqual(detail["classification_source"], "rules")
        self.assertEqual(detail["work_item_links"][0]["work_item_id"], self.login_item["work_item_id"])

    def test_discovery_api_returns_run_result(self):
        fake_run = {
            "run_id": "RUN-1", "status": "queued", "total_sources": 4,
            "processed_sources": 0, "discovered_count": 0,
        }
        fake_manager = SimpleNamespace(start=lambda **kwargs: fake_run)
        with (
            patch.object(server, "analysis_job_manager", fake_manager),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                response = client.post(
                    "/api/workspace/conversations/discover-work-items",
                    json={"project_id": self.project["project_id"], "force": False},
                )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["run"]["run_id"], "RUN-1")

    def test_client_initialization_failure_is_recorded_without_stuck_run(self):
        def fail_factory():
            raise RuntimeError("AMD unavailable")

        result = WorkItemDiscovery(self.store, fail_factory).discover()

        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["processed_segments"], 3)
        self.assertEqual(result["run"]["status"], "failed")
        self.assertIn("AMD unavailable", result["run"]["error_message"])
        self.assertEqual(
            self.store.get_segment_classification_state(self.segments[0]["segment_id"])["status"],
            "error",
        )


if __name__ == "__main__":
    unittest.main()
