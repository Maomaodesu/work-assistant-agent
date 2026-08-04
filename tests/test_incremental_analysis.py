import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from semantic_segmenter import SemanticConversationSegmenter
from work_item_discovery import CLASSIFIER_VERSION, WorkItemDiscovery, _segment_fingerprint
from work_item_retriever import DEFAULT_CANDIDATE_CHAR_BUDGET, WorkItemRetriever
from workspace_store import WorkspaceStore


class CaptureAMDClient:
    def __init__(self):
        self.payloads = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        self.payloads.append(payload)
        decisions = [{
            "segment_id": segment["segment_id"],
            "segment_kind": "project",
            "action": "none",
            "matched_work_item_id": None,
            "title": segment["current_title"],
            "item_type": "other",
            "goal": "",
            "summary": "测试分类结果",
            "confidence": 0.8,
        } for segment in payload["segments"]]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"decisions": decisions}, ensure_ascii=False),
            ))]
        )


class IncrementalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "demo"
        self.project_root.mkdir()
        self.store = WorkspaceStore(self.root / "workspace.db")
        self.project = self.store.create_project("Demo", [str(self.project_root)])
        self.segmenter = SemanticConversationSegmenter(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _conversation(self, suffix: str) -> dict:
        conversation = self.store.upsert_conversation(
            "codex", f"incremental-{suffix}", title="incremental test",
            original_project_path=str(self.project_root),
        )
        self.store.set_manual_conversation_project(
            conversation["conversation_id"], self.project["project_id"],
        )
        return conversation

    def _segment(self, conversation_id: str):
        conversation = next(
            item for item in self.store.list_conversations()
            if item["conversation_id"] == conversation_id
        )
        return self.segmenter.segment_conversation(conversation)

    def test_append_preserves_stable_segment_evidence_and_only_reanalyzes_tail(self):
        conversation = self._conversation("append")
        original_messages = [
            {"ordinal": 0, "role": "user", "content": "Implement login validation", "created_at": "2026-08-01T10:00:00+00:00"},
            {"ordinal": 1, "role": "assistant", "content": "I will update LoginService", "created_at": "2026-08-01T10:01:00+00:00"},
            {"ordinal": 2, "role": "user", "content": "Next task: implement metrics endpoint", "created_at": "2026-08-01T10:02:00+00:00"},
            {"ordinal": 3, "role": "assistant", "content": "I will add the endpoint", "created_at": "2026-08-01T10:03:00+00:00"},
        ]
        self.store.replace_messages(conversation["conversation_id"], original_messages)
        self._segment(conversation["conversation_id"])
        first, tail = self.store.list_segments_for_conversation(conversation["conversation_id"])

        item = self.store.create_work_item(self.project["project_id"], "Login validation")
        self.store.link_segment_work_item(first["segment_id"], item["work_item_id"])
        self.store.create_context_package(
            item["work_item_id"], canonical_path="contexts/login.md",
            content_hash="context-v1", segment_ids=[first["segment_id"]],
        )
        for segment in (first, tail):
            messages = self.store.list_messages_for_segment(segment["segment_id"])
            self.store.save_segment_classification_state(
                segment["segment_id"], segment_fingerprint=_segment_fingerprint(messages),
                classifier_version=CLASSIFIER_VERSION, status="classified",
            )

        before_messages = self.store.list_messages(conversation["conversation_id"])
        delta = self.store.sync_messages(conversation["conversation_id"], [
            *original_messages,
            {"ordinal": 4, "role": "user", "content": "Add Prometheus counter to the metrics endpoint", "created_at": "2026-08-01T10:04:00+00:00"},
        ])
        self.assertEqual(delta["state"], "appended")
        self.assertEqual(
            self.store.list_messages(conversation["conversation_id"])[0]["message_id"],
            before_messages[0]["message_id"],
        )

        self._segment(conversation["conversation_id"])
        current = self.store.list_segments_for_conversation(conversation["conversation_id"])
        self.assertEqual(current[0]["segment_id"], first["segment_id"])
        self.assertEqual(current[1]["segment_id"], tail["segment_id"])
        self.assertEqual(current[1]["end_ordinal"], 4)
        self.assertIsNotNone(self.store.get_segment_classification_state(first["segment_id"]))
        self.assertIsNone(self.store.get_segment_classification_state(tail["segment_id"]))
        self.assertEqual(
            self.store.get_context_package(
                self.store.list_context_packages(item["work_item_id"])[0]["context_id"]
            )["segment_ids"],
            [first["segment_id"]],
        )

        client = CaptureAMDClient()
        WorkItemDiscovery(self.store, lambda: client).discover()
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(
            [segment["segment_id"] for segment in client.payloads[0]["segments"]],
            [tail["segment_id"]],
        )

    def test_rewrite_supersedes_obsolete_segment_without_losing_context_source(self):
        conversation = self._conversation("rewrite")
        original_messages = [
            {"ordinal": 0, "role": "user", "content": "Fix login validation", "created_at": "2026-08-01T10:00:00+00:00"},
            {"ordinal": 1, "role": "assistant", "content": "Starting login fix", "created_at": "2026-08-01T10:01:00+00:00"},
            {"ordinal": 2, "role": "user", "content": "Next task: implement metrics endpoint", "created_at": "2026-08-01T10:02:00+00:00"},
            {"ordinal": 3, "role": "assistant", "content": "Starting metrics endpoint", "created_at": "2026-08-01T10:03:00+00:00"},
        ]
        self.store.replace_messages(conversation["conversation_id"], original_messages)
        self._segment(conversation["conversation_id"])
        first, obsolete = self.store.list_segments_for_conversation(conversation["conversation_id"])
        item = self.store.create_work_item(self.project["project_id"], "Metrics endpoint")
        self.store.link_segment_work_item(obsolete["segment_id"], item["work_item_id"])
        package = self.store.create_context_package(
            item["work_item_id"], canonical_path="contexts/metrics.md",
            content_hash="context-v1", segment_ids=[obsolete["segment_id"]],
        )

        rewritten = [*original_messages]
        rewritten[2] = {
            **rewritten[2], "content": "Continue improving the login validation flow",
        }
        delta = self.store.sync_messages(conversation["conversation_id"], rewritten)
        self.assertEqual(delta["state"], "rewritten")
        self._segment(conversation["conversation_id"])

        current = self.store.list_segments_for_conversation(conversation["conversation_id"])
        historical = self.store.get_segment(obsolete["segment_id"])
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["segment_id"], first["segment_id"])
        self.assertFalse(historical["is_current"])
        self.assertEqual(historical["superseded_reason"], "source_resegmented")
        self.assertEqual(self.store.get_context_package(package["context_id"])["segment_ids"], [obsolete["segment_id"]])

    def test_discovery_sends_top_k_candidates_not_all_project_items(self):
        conversation = self._conversation("retrieval")
        self.store.replace_messages(conversation["conversation_id"], [{
            "ordinal": 0, "role": "user",
            "content": "Fix LoginService timeout in auth/api.py",
            "created_at": "2026-08-01T10:00:00+00:00",
        }])
        self.store.create_segment(
            conversation["conversation_id"], 0, 0,
            project_id=self.project["project_id"], title="LoginService timeout",
        )
        matching = self.store.create_work_item(
            self.project["project_id"], "Repair LoginService timeout",
            goal="修复 auth/api.py 中的登录超时",
        )
        for index in range(24):
            self.store.create_work_item(
                self.project["project_id"], f"Unrelated maintenance {index}",
                goal="常规维护任务" * 40,
            )

        client = CaptureAMDClient()
        WorkItemDiscovery(self.store, lambda: client).discover()
        payload = client.payloads[0]
        self.assertNotIn("existing_work_items", payload)
        candidates = payload["segments"][0]["work_item_candidates"]
        self.assertLess(len(candidates), 25)
        self.assertIn(matching["work_item_id"], {item["work_item_id"] for item in candidates})
        self.assertLessEqual(
            sum(len(json.dumps(item, ensure_ascii=False, separators=(",", ":"))) for item in candidates),
            DEFAULT_CANDIDATE_CHAR_BUDGET,
        )

    def test_retriever_enforces_limit_and_character_budget(self):
        items = [{
            "work_item_id": f"WI-{index}",
            "title": "Fix LoginService timeout" if index == 0 else f"Other work {index}",
            "item_type": "bug", "status": "in_progress" if index < 5 else "backlog",
            "goal": "auth/api.py LoginService timeout" if index == 0 else "regular work" * 80,
            "description": "detail" * 100,
        } for index in range(20)]
        result = WorkItemRetriever().retrieve(
            "Fix LoginService timeout in auth/api.py", items,
            limit=3, char_budget=900,
        )
        self.assertLessEqual(result["candidate_count"], 3)
        self.assertLessEqual(result["candidate_char_count"], 900)
        self.assertIn("WI-0", {item["work_item_id"] for item in result["candidates"]})


if __name__ == "__main__":
    unittest.main()
