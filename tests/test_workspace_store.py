import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from settings import SettingsService
from workspace_store import WorkspaceStore, WorkspaceStoreError


class WorkspaceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend = self.root / "backend"
        self.frontend = self.root / "frontend"
        self.backend.mkdir()
        self.frontend.mkdir()
        self.store = WorkspaceStore(self.root / "work_assistant.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _project(self, name="Demo"):
        return self.store.create_project(
            name,
            [str(self.backend), str(self.frontend)],
        )

    def _conversation(self):
        conversation = self.store.upsert_conversation(
            "codex",
            "019f-session",
            title="mixed work",
            original_project_path=str(self.backend),
            started_at="2026-07-12T20:00:00+00:00",
            updated_at="2026-07-12T21:00:00+00:00",
            resume_capable=True,
        )
        self.store.replace_messages(conversation["conversation_id"], [
            {"role": "user", "content": "project architecture", "created_at": "2026-07-12T20:00:00+00:00"},
            {"role": "assistant", "content": "architecture answer", "created_at": "2026-07-12T20:01:00+00:00"},
            {"role": "user", "content": "build rolling PE", "created_at": "2026-07-12T20:10:00+00:00"},
            {"role": "assistant", "content": "PE implementation", "created_at": "2026-07-12T20:20:00+00:00"},
            {"role": "user", "content": "also fix login", "created_at": "2026-07-12T20:30:00+00:00"},
            {"role": "assistant", "content": "login fix", "created_at": "2026-07-12T20:40:00+00:00"},
        ])
        return conversation

    def test_schema_is_new_and_does_not_reuse_legacy_task_tables(self):
        info = self.store.schema_info()
        self.assertEqual(info["version"], 4)
        self.assertIn("projects", info["tables"])
        self.assertIn("work_items", info["tables"])
        self.assertIn("conversation_segments", info["tables"])
        self.assertIn("classification_runs", info["tables"])
        self.assertIn("conversation_segmentation_state", info["tables"])
        self.assertIn("segment_classification_state", info["tables"])
        self.assertNotIn("tasks", info["tables"])
        self.assertNotIn("plan_steps", info["tables"])

    def test_project_supports_multiple_roots_and_root_belongs_to_one_project(self):
        project = self._project()
        self.assertEqual(len(project["roots"]), 2)
        self.assertTrue(project["roots"][0]["is_primary"])

        with self.assertRaises(WorkspaceStoreError):
            self.store.create_project("Duplicate", [str(self.backend)])

    def test_inferred_work_item_uses_conversation_time_and_recoverable_ignore(self):
        project = self._project()
        conversation_time = "2026-07-12T20:10:00+00:00"
        item = self.store.create_work_item(
            project["project_id"],
            "滚动市盈率功能",
            item_type="feature",
            source="inferred",
            confidence=0.87,
            created_at=conversation_time,
        )
        self.assertEqual(item["status"], "suggested")
        self.assertEqual(item["created_at"], conversation_time)
        self.assertNotEqual(item["discovered_at"], conversation_time)
        self.assertIsNone(item["completion_percent"])

        confirmed = self.store.confirm_work_item(item["work_item_id"])
        ignored = self.store.ignore_work_item(confirmed["work_item_id"], reason="识别错误")
        self.assertEqual(ignored["status"], "ignored")
        self.assertEqual(self.store.list_work_items(project["project_id"]), [])
        self.assertEqual(
            self.store.list_work_items(project["project_id"], status="ignored")[0]["ignore_reason"],
            "识别错误",
        )

        restored = self.store.restore_work_item(item["work_item_id"])
        self.assertEqual(restored["status"], "suggested")
        event_types = [event["event_type"] for event in restored["events"]]
        self.assertEqual(event_types.count("status_changed"), 3)

    def test_work_item_has_no_percent_without_plan_and_plan_drives_percent(self):
        project = self._project()
        item = self.store.create_work_item(project["project_id"], "Feature A")
        self.assertIsNone(item["completion_percent"])

        planned = self.store.replace_work_item_steps(item["work_item_id"], [
            {"title": "设计", "status": "completed"},
            {"title": "开发", "status": "in_progress"},
            {"title": "测试", "status": "pending"},
        ])
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(planned["completion_percent"], 50)

    def test_complete_and_reopen_work_item_preserves_completion_event(self):
        project = self._project()
        item = self.store.create_work_item(project["project_id"], "完成状态测试")

        completed = self.store.complete_work_item(
            item["work_item_id"],
            completion_note="实现已合并",
            acceptance_result="相关测试通过",
        )
        reopened = self.store.reopen_work_item(item["work_item_id"])

        self.assertEqual(completed["status"], "completed")
        self.assertTrue(completed["completed_at"])
        self.assertEqual(reopened["status"], "in_progress")
        self.assertIsNone(reopened["completed_at"])
        events = {event["event_type"]: event for event in reopened["events"]}
        self.assertEqual(events["completed"]["details"]["completion_note"], "实现已合并")
        self.assertEqual(events["completed"]["details"]["acceptance_result"], "相关测试通过")
        self.assertIn("reopened", events)

    def test_one_conversation_can_have_segments_for_different_work_items(self):
        project = self._project()
        conversation = self._conversation()
        rolling_pe = self.store.create_work_item(project["project_id"], "Rolling PE")
        login = self.store.create_work_item(project["project_id"], "Login bug", item_type="bug")

        project_segment = self.store.create_segment(
            conversation["conversation_id"], 0, 1,
            segment_kind="project", project_id=project["project_id"],
        )
        pe_segment = self.store.create_segment(
            conversation["conversation_id"], 2, 3,
            segment_kind="work_item", project_id=project["project_id"],
        )
        login_segment = self.store.create_segment(
            conversation["conversation_id"], 4, 5,
            segment_kind="work_item", project_id=project["project_id"],
        )
        self.store.link_segment_work_item(pe_segment["segment_id"], rolling_pe["work_item_id"])
        self.store.link_segment_work_item(login_segment["segment_id"], login["work_item_id"])
        self.store.link_segment_work_item(
            pe_segment["segment_id"], login["work_item_id"], relation="mentioned"
        )

        self.assertEqual(project_segment["work_item_links"], [])
        self.assertEqual(
            self.store.list_segments_for_work_item(rolling_pe["work_item_id"])[0]["start_ordinal"],
            2,
        )
        with self.assertRaises(WorkspaceStoreError):
            self.store.link_segment_work_item(
                pe_segment["segment_id"], login["work_item_id"], relation="primary"
            )
        with self.assertRaises(WorkspaceStoreError):
            self.store.create_segment(
                conversation["conversation_id"], 1, 2, segment_kind="unclassified"
            )

    def test_segment_cannot_link_to_work_item_from_another_project(self):
        project = self._project("Project A")
        other_root = self.root / "other"
        other_root.mkdir()
        other = self.store.create_project("Project B", [str(other_root)])
        item = self.store.create_work_item(other["project_id"], "Other feature")
        conversation = self._conversation()
        segment = self.store.create_segment(
            conversation["conversation_id"], 0, 1,
            segment_kind="project", project_id=project["project_id"],
        )
        with self.assertRaises(WorkspaceStoreError):
            self.store.link_segment_work_item(segment["segment_id"], item["work_item_id"])

    def test_merge_work_items_moves_sources_and_archives_duplicates(self):
        project = self._project()
        conversation = self._conversation()
        target = self.store.create_work_item(
            project["project_id"], "资产负债表功能", source="inferred"
        )
        duplicate = self.store.create_work_item(
            project["project_id"], "获取资产负债表", source="inferred"
        )
        first = self.store.create_segment(
            conversation["conversation_id"], 0, 1,
            segment_kind="work_item", project_id=project["project_id"],
        )
        second = self.store.create_segment(
            conversation["conversation_id"], 2, 3,
            segment_kind="work_item", project_id=project["project_id"],
        )
        self.store.link_segment_work_item(first["segment_id"], target["work_item_id"])
        self.store.link_segment_work_item(second["segment_id"], duplicate["work_item_id"])

        merged = self.store.merge_work_items(
            target["work_item_id"], [duplicate["work_item_id"]]
        )

        self.assertEqual(merged["merged_source_ids"], [duplicate["work_item_id"]])
        self.assertEqual(len(self.store.list_work_item_source_details(target["work_item_id"])), 2)
        archived = self.store.get_work_item(duplicate["work_item_id"])
        self.assertEqual(archived["status"], "archived")
        self.assertEqual(
            archived["metadata"]["merged_into_work_item_id"], target["work_item_id"]
        )
        self.assertEqual(self.store.list_segments_for_work_item(duplicate["work_item_id"]), [])

    def test_review_details_include_messages_and_project_discussions(self):
        project = self._project()
        conversation = self._conversation()
        item = self.store.create_work_item(project["project_id"], "Rolling PE")
        work_segment = self.store.create_segment(
            conversation["conversation_id"], 2, 3,
            segment_kind="work_item", project_id=project["project_id"], title="Rolling PE",
        )
        discussion = self.store.create_segment(
            conversation["conversation_id"], 0, 1,
            segment_kind="project", project_id=project["project_id"], title="架构讨论",
        )
        self.store.link_segment_work_item(work_segment["segment_id"], item["work_item_id"])

        sources = self.store.list_work_item_source_details(item["work_item_id"])
        discussions = self.store.list_project_segment_details(
            project["project_id"], segment_kind="project"
        )
        self.assertEqual([message["ordinal"] for message in sources[0]["messages"]], [2, 3])
        self.assertEqual(sources[0]["conversation"]["source"], "codex")
        self.assertTrue(sources[0]["conversation"]["resume_capable"])
        self.assertEqual([segment["segment_id"] for segment in discussions], [discussion["segment_id"]])

    def test_classification_run_reports_loading_progress_and_privacy_mode(self):
        run = self.store.create_classification_run(
            run_type="full", privacy_mode="balanced", total_sources=124
        )
        running = self.store.update_classification_run(
            run["run_id"],
            status="running",
            stage="classifying",
            processed_sources=62,
            discovered_count=8,
            amd_call_count=40,
        )
        self.assertEqual(running["completion_percent"], 50)
        self.assertEqual(running["privacy_mode"], "balanced")
        self.assertIsNotNone(running["started_at"])
        with self.assertRaises(WorkspaceStoreError):
            self.store.update_classification_run(run["run_id"], processed_sources=125)

    def test_context_package_versions_and_preserves_segment_sources(self):
        project = self._project()
        item = self.store.create_work_item(project["project_id"], "Context feature")
        conversation = self._conversation()
        segment = self.store.create_segment(
            conversation["conversation_id"], 2, 3,
            segment_kind="work_item", project_id=project["project_id"],
        )
        self.store.link_segment_work_item(segment["segment_id"], item["work_item_id"])

        first = self.store.create_context_package(
            item["work_item_id"],
            canonical_path="data/contexts/context.md",
            content_hash="hash-1",
            segment_ids=[segment["segment_id"]],
        )
        second = self.store.create_context_package(
            item["work_item_id"],
            canonical_path="data/contexts/context.md",
            content_hash="hash-2",
            segment_ids=[segment["segment_id"]],
        )
        self.assertEqual((first["version"], second["version"]), (1, 2))
        self.assertEqual(first["segment_ids"], [segment["segment_id"]])

    def test_workspace_api_uses_new_store_and_supports_ignore_restore(self):
        with (
            patch.object(server, "workspace_store", self.store),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                project_response = client.post("/api/workspace/projects", json={
                    "name": "API project",
                    "root_paths": [str(self.backend), str(self.frontend)],
                })
                project_id = project_response.json()["project_id"]
                inferred = self.store.create_work_item(
                    project_id, "Detected item", source="inferred", confidence=0.8
                )
                ignored = client.post(
                    f"/api/workspace/work-items/{inferred['work_item_id']}/ignore",
                    json={"reason": "not a task"},
                )
                restored = client.post(
                    f"/api/workspace/work-items/{inferred['work_item_id']}/restore"
                )
                listed = client.get(
                    "/api/workspace/work-items", params={"project_id": project_id}
                )

        self.assertEqual(project_response.status_code, 201)
        self.assertEqual(ignored.json()["status"], "ignored")
        self.assertEqual(restored.json()["status"], "suggested")
        self.assertEqual(len(listed.json()), 1)

    def test_workspace_api_completes_and_reopens_manual_work_item(self):
        with (
            patch.object(server, "workspace_store", self.store),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                project = client.post("/api/workspace/projects", json={
                    "name": "Completion API project", "root_paths": [str(self.backend)],
                }).json()
                item = client.post("/api/workspace/work-items", json={
                    "project_id": project["project_id"], "title": "完成接口测试",
                }).json()
                completed = client.post(
                    f"/api/workspace/work-items/{item['work_item_id']}/complete",
                    json={"completion_note": "实现完成", "acceptance_result": "测试通过"},
                )
                reopened = client.post(
                    f"/api/workspace/work-items/{item['work_item_id']}/reopen"
                )

        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "completed")
        self.assertTrue(completed.json()["completed_at"])
        self.assertEqual(reopened.status_code, 200)
        self.assertEqual(reopened.json()["status"], "in_progress")

    def test_settings_exposes_separate_workspace_database_path(self):
        service = SettingsService(self.root / "settings.db")
        view = service.public_view()
        self.assertTrue(view["workspace_db_path"].endswith("work_assistant.db"))
        self.assertNotEqual(view["workspace_db_path"], view["conversation_db_path"])


if __name__ == "__main__":
    unittest.main()
