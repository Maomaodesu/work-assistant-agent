import tempfile
import unittest
from pathlib import Path

from retrieval_chunker import RetrievalChunker, build_retrieval_chunks
from semantic_segmenter import SemanticConversationSegmenter
from workspace_store import WorkspaceStore


def msg(ordinal, role, content):
    return {
        "ordinal": ordinal,
        "role": role,
        "content": content,
        "created_at": f"2026-08-04T10:{ordinal:02d}:00+00:00",
    }


class RetrievalChunkerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = WorkspaceStore(Path(self.temp_dir.name) / "workspace.db")
        self.segmenter = SemanticConversationSegmenter(self.store)
        self.chunker = RetrievalChunker(self.store)
        self.conversation = self.store.upsert_conversation(
            "codex", "retrieval-session", title="检索块测试"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _segment(self, messages):
        conversation_id = self.conversation["conversation_id"]
        self.store.replace_messages(conversation_id, messages)
        return self.segmenter.segment_conversation(
            self.store.get_conversation(conversation_id)
        )

    def test_chunk_builder_bounds_content_and_repeats_overlap(self):
        chunks = build_retrieval_chunks(
            {"segment_id": "SEG-1"},
            [msg(0, "user", "甲" * 1_100)],
            max_chars=500,
            overlap_chars=80,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["char_count"] <= 500 for chunk in chunks))
        self.assertTrue(all(chunk["start_ordinal"] == 0 for chunk in chunks))
        self.assertIn(chunks[0]["content"][-80:], chunks[1]["content"])
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], list(range(len(chunks))))

    def test_changed_messages_invalidate_then_next_index_run_rebuilds_chunks(self):
        conversation_id = self.conversation["conversation_id"]
        self._segment([
            msg(0, "user", "实现认证接口。" * 500),
            msg(1, "assistant", "已完成第一版。" * 300),
        ])

        first = self.chunker.chunk_conversation(self.store.get_conversation(conversation_id))
        saved = self.store.list_retrieval_chunks_for_conversation(conversation_id)
        self.assertEqual(first["state"], "indexed")
        self.assertGreater(first["chunk_count"], 1)
        self.assertEqual(len(saved), first["chunk_count"])
        self.assertEqual(
            self.chunker.chunk_conversation(self.store.get_conversation(conversation_id))["state"],
            "unchanged",
        )

        self.store.replace_messages(conversation_id, [
            msg(0, "user", "重新设计认证接口。"),
            msg(1, "assistant", "新实现已经完成。"),
        ])
        self.assertEqual(self.store.list_retrieval_chunks_for_conversation(conversation_id), [])
        self.assertIsNone(self.store.get_retrieval_index_state(conversation_id))

        self._segment([
            msg(0, "user", "重新设计认证接口。"),
            msg(1, "assistant", "新实现已经完成。"),
        ])
        rebuilt = self.chunker.chunk_conversation(self.store.get_conversation(conversation_id))
        self.assertEqual(rebuilt["state"], "indexed")
        self.assertEqual(rebuilt["chunk_count"], 1)
        self.assertIn(
            "重新设计认证接口",
            self.store.list_retrieval_chunks_for_conversation(conversation_id)[0]["content"],
        )

    def test_chunker_waits_for_semantic_segments(self):
        conversation_id = self.conversation["conversation_id"]
        self.store.replace_messages(conversation_id, [msg(0, "user", "尚未切分")])

        outcome = self.chunker.chunk_conversation(self.store.get_conversation(conversation_id))

        self.assertEqual(outcome, {"state": "pending_segmentation", "chunk_count": 0})


if __name__ == "__main__":
    unittest.main()
