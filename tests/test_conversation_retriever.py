import tempfile
import unittest
from pathlib import Path

from conversation_retriever import ConversationRetriever
from retrieval_chunker import RetrievalChunker
from semantic_segmenter import SemanticConversationSegmenter
from workspace_store import WorkspaceStore


def msg(ordinal, role, content):
    return {
        "ordinal": ordinal,
        "role": role,
        "content": content,
        "created_at": f"2026-08-05T10:{ordinal:02d}:00+00:00",
    }


class ConversationRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = WorkspaceStore(Path(self.temp_dir.name) / "workspace.db")
        self.segmenter = SemanticConversationSegmenter(self.store)
        self.chunker = RetrievalChunker(self.store)
        self.retriever = ConversationRetriever(self.store)
        self.conversation = self.store.upsert_conversation(
            "codex", "retrieval-search", title="检索测试"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _index(self, messages):
        conversation_id = self.conversation["conversation_id"]
        self.store.replace_messages(conversation_id, messages)
        self.segmenter.segment_conversation(self.store.get_conversation(conversation_id))
        self.chunker.chunk_conversation(self.store.get_conversation(conversation_id))
        return conversation_id

    def test_lexical_retrieval_selects_relevant_historical_evidence(self):
        conversation_id = self._index([
            msg(0, "user", "修复 auth.py 中的登录鉴权报错，token 刷新失败。"),
            msg(1, "assistant", "定位到 auth.py 的 refresh token 校验分支。"),
            msg(2, "user", "接下来实现报表 dashboard.py 的导出功能。"),
            msg(3, "assistant", "已增加 dashboard.py 的 CSV 导出。"),
        ])

        result = self.retriever.retrieve(
            conversation_id, "auth.py 登录鉴权 token 报错原因", limit=3
        )

        self.assertEqual(result["state"], "ok")
        self.assertGreaterEqual(result["matched_count"], 1)
        evidence = "\n".join(chunk["content"] for chunk in result["chunks"])
        self.assertIn("auth.py", evidence)
        self.assertNotIn("dashboard.py", evidence)

    def test_retrieval_adds_same_segment_neighbor_for_context(self):
        conversation_id = self._index([
            msg(0, "user", "JWTREFRESH-42 的失败原因是签名过期。" + "补充细节。" * 900),
            msg(1, "assistant", "后续处理是刷新密钥并重新验证。" + "处理说明。" * 900),
        ])

        result = self.retriever.retrieve(
            conversation_id, "JWTREFRESH-42", limit=1, neighbor_window=1
        )

        self.assertEqual(result["state"], "ok")
        self.assertEqual(result["matched_count"], 1)
        self.assertGreater(result["selected_count"], 1)
        self.assertTrue(any(chunk["is_neighbor"] for chunk in result["chunks"]))

    def test_retrieval_reports_missing_index_without_falling_back_to_unrelated_chunks(self):
        conversation_id = self.conversation["conversation_id"]
        self.store.replace_messages(conversation_id, [msg(0, "user", "尚未建立索引")])

        result = self.retriever.retrieve(conversation_id, "索引")

        self.assertEqual(result["state"], "index_unavailable")
        self.assertEqual(result["chunks"], [])


if __name__ == "__main__":
    unittest.main()
