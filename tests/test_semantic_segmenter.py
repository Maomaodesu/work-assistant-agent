import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from semantic_segmenter import (
    SEGMENTER_VERSION,
    SemanticConversationSegmenter,
    build_semantic_segments,
)
from workspace_store import WorkspaceStore


def msg(ordinal, role, content, created_at):
    return {
        "ordinal": ordinal,
        "role": role,
        "content": content,
        "created_at": created_at,
    }


class SemanticSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "project"
        self.project_root.mkdir()
        self.store = WorkspaceStore(self.root / "workspace.db")
        self.project = self.store.create_project("Demo", [str(self.project_root)])
        self.segmenter = SemanticConversationSegmenter(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _conversation(self, session_id="session", path=None):
        conversation = self.store.upsert_conversation(
            "codex", session_id,
            title="Demo conversation",
            original_project_path=str(path or self.project_root),
            started_at="2026-07-18T10:00:00+00:00",
            updated_at="2026-07-18T10:20:00+00:00",
        )
        self.store.set_manual_conversation_project(
            conversation["conversation_id"], self.project["project_id"]
        )
        return next(
            item for item in self.store.list_conversations()
            if item["conversation_id"] == conversation["conversation_id"]
        )

    def test_explicit_topic_change_starts_new_segment_at_user_message(self):
        messages = [
            msg(0, "user", "实现滚动市盈率接口", "2026-07-18T10:00:00+00:00"),
            msg(1, "assistant", "接口已经完成", "2026-07-18T10:01:00+00:00"),
            msg(2, "user", "好，接下来修复登录错误", "2026-07-18T10:02:00+00:00"),
            msg(3, "assistant", "开始修复登录", "2026-07-18T10:03:00+00:00"),
        ]
        segments = build_semantic_segments(messages)
        self.assertEqual([(item["start_ordinal"], item["end_ordinal"]) for item in segments], [(0, 1), (2, 3)])
        self.assertEqual(segments[1]["boundary_reason"], "explicit_topic_change")

    def test_long_time_gap_and_task_id_change_create_boundaries(self):
        messages = [
            msg(0, "user", "处理 TASK-AAA", "2026-07-18T10:00:00+00:00"),
            msg(1, "assistant", "done", "2026-07-18T10:01:00+00:00"),
            msg(2, "user", "继续检查 TASK-AAA 结果", "2026-07-18T13:00:00+00:00"),
            msg(3, "assistant", "checked", "2026-07-18T13:01:00+00:00"),
            msg(4, "user", "处理 WI-BBB", "2026-07-18T13:02:00+00:00"),
        ]
        segments = build_semantic_segments(messages)
        self.assertEqual([item["boundary_reason"] for item in segments], [
            "conversation_start", "time_gap", "task_id_changed"
        ])

    def test_semantic_shift_requires_context_and_ignores_short_acknowledgement(self):
        messages = [
            msg(0, "user", "Implement rolling price earnings calculation service", "2026-07-18T10:00:00"),
            msg(1, "assistant", "ok", "2026-07-18T10:01:00"),
            msg(2, "user", "Add rolling earnings mapper and controller endpoint", "2026-07-18T10:02:00"),
            msg(3, "assistant", "ok", "2026-07-18T10:03:00"),
            msg(4, "user", "Test earnings calculation with quarterly profit data", "2026-07-18T10:04:00"),
            msg(5, "assistant", "ok", "2026-07-18T10:05:00"),
            msg(6, "user", "好的", "2026-07-18T10:06:00"),
            msg(7, "assistant", "continue", "2026-07-18T10:07:00"),
            msg(8, "user", "Design authentication password reset email security tokens", "2026-07-18T10:08:00"),
        ]
        segments = build_semantic_segments(messages)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1]["start_ordinal"], 8)
        self.assertEqual(segments[1]["boundary_reason"], "semantic_shift")

    def test_saved_segments_cover_all_messages_and_inherit_primary_project(self):
        conversation = self._conversation()
        messages = [
            msg(0, "user", "first task", "2026-07-18T10:00:00"),
            msg(1, "assistant", "first answer", "2026-07-18T10:01:00"),
            msg(2, "user", "Next task: second feature", "2026-07-18T10:02:00"),
            msg(3, "assistant", "second answer", "2026-07-18T10:03:00"),
        ]
        self.store.replace_messages(conversation["conversation_id"], messages)
        conversation = next(item for item in self.store.list_conversations() if item["conversation_id"] == conversation["conversation_id"])

        outcome = self.segmenter.segment_conversation(conversation)
        segments = self.store.list_segments_for_conversation(conversation["conversation_id"])

        self.assertEqual(outcome["state"], "segmented")
        self.assertEqual([(item["start_ordinal"], item["end_ordinal"]) for item in segments], [(0, 1), (2, 3)])
        self.assertTrue(all(item["project_id"] == self.project["project_id"] for item in segments))
        self.assertTrue(all(item["segment_kind"] == "unclassified" for item in segments))
        self.assertTrue(all(item["segmenter_version"] == SEGMENTER_VERSION for item in segments))

    def test_unchanged_conversation_is_not_segmented_again(self):
        conversation = self._conversation()
        self.store.replace_messages(conversation["conversation_id"], [
            msg(0, "user", "single topic", "2026-07-18T10:00:00"),
            msg(1, "assistant", "answer", "2026-07-18T10:01:00"),
        ])
        conversation = next(item for item in self.store.list_conversations() if item["conversation_id"] == conversation["conversation_id"])
        first = self.segmenter.segment_conversation(conversation)
        with patch.object(self.store, "replace_conversation_segments", wraps=self.store.replace_conversation_segments) as replace:
            second = self.segmenter.segment_conversation(conversation)

        self.assertEqual(first["state"], "segmented")
        self.assertEqual(second["state"], "unchanged")
        replace.assert_not_called()

    def test_changed_messages_invalidate_previous_segmentation_state(self):
        conversation = self._conversation()
        self.store.replace_messages(conversation["conversation_id"], [
            msg(0, "user", "old topic", "2026-07-18T10:00:00"),
        ])
        conversation = next(item for item in self.store.list_conversations() if item["conversation_id"] == conversation["conversation_id"])
        self.segmenter.segment_conversation(conversation)
        self.store.replace_messages(conversation["conversation_id"], [
            msg(0, "user", "new topic", "2026-07-18T10:00:00"),
            msg(1, "assistant", "new answer", "2026-07-18T10:01:00"),
        ])
        self.assertIsNone(self.store.get_segmentation_state(conversation["conversation_id"]))
        self.assertEqual(self.store.list_segments_for_conversation(conversation["conversation_id"]), [])

    def test_segment_api_returns_statistics_without_amd(self):
        fake_segmenter = SimpleNamespace(segment_all=lambda: {
            "total_conversations": 2,
            "segmented_conversations": 2,
            "unchanged_conversations": 0,
            "protected_conversations": 0,
            "empty_conversations": 0,
            "errors": 0,
            "segments_created": 3,
            "boundary_reasons": {"conversation_start": 2, "explicit_topic_change": 1},
            "error_details": [],
            "segmenter_version": SEGMENTER_VERSION,
        })
        with (
            patch.object(server, "semantic_conversation_segmenter", fake_segmenter),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                response = client.post("/api/workspace/conversations/segment")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["segments_created"], 3)


if __name__ == "__main__":
    unittest.main()
