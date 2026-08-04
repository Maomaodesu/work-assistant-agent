import asyncio
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
import agent_graph
from conversation_manager import ConversationStore
from streaming_runtime import (
    GenerationCancelled,
    GraphStreamCallback,
    bind_generation_cancel,
    stream_chat_completion_text,
)
from semantic_segmenter import message_fingerprint
from workspace_store import WorkspaceStore


class StreamingControlTests(unittest.TestCase):
    def test_summary_auto_route_is_conservative_and_can_be_overridden(self):
        self.assertEqual(
            server._resolve_summary_mode("请生成开发工作回顾", "auto"),
            "full",
        )
        self.assertEqual(
            server._resolve_summary_mode("总结 auth.py 的登录报错原因", "auto"),
            "focused",
        )
        self.assertEqual(
            server._resolve_summary_mode("第 12 轮讨论最终结论是什么？", "auto"),
            "focused",
        )
        self.assertEqual(
            server._resolve_summary_mode("帮我看一下", "auto"),
            "full",
        )
        self.assertEqual(
            server._resolve_summary_mode("完整回顾全部历史", "focused"),
            "focused",
        )

    def test_summary_request_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            server.ConversationSummaryRequest(message="请总结", mode="all")

    def test_summary_maps_legacy_external_id_to_workspace_conversation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkspaceStore(Path(temp_dir) / "workspace.db")
            workspace_conversation = store.upsert_conversation(
                "codex", "session:with:colon", title="Imported history"
            )
            store.replace_messages(workspace_conversation["conversation_id"], [
                {"ordinal": 0, "role": "user", "content": "请检查登录问题"},
                {"ordinal": 1, "role": "assistant", "content": "正在检查"},
            ])

            with patch.object(server, "workspace_store", store):
                context = server._summary_workspace_context({
                    "id": "codex:session:with:colon", "source": "codex",
                })

            self.assertEqual(context["state"], "pending_segmentation")
            self.assertEqual(
                context["workspace_conversation_id"],
                workspace_conversation["conversation_id"],
            )
            self.assertEqual(context["segment_count"], 0)
            self.assertEqual(context["retrieval_index_state"], "missing")

    def test_summary_workspace_mapping_is_best_effort_when_not_synced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkspaceStore(Path(temp_dir) / "workspace.db")
            with patch.object(server, "workspace_store", store):
                context = server._summary_workspace_context({
                    "id": "claude:missing-session", "source": "claude",
                })

            self.assertEqual(context, {
                "state": "not_synced",
                "workspace_conversation_id": None,
                "segment_count": 0,
                "retrieval_index_state": "missing",
                "retrieval_chunk_count": 0,
            })

    def test_direct_structured_openai_stream_can_be_cancelled_mid_response(self):
        cancel_event = threading.Event()

        class FakeStream:
            def __init__(self):
                self.index = 0
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.index == 0:
                    self.index += 1
                    return SimpleNamespace(choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="first")
                    )])
                cancel_event.set()
                raise StopIteration

            def close(self):
                self.closed = True

        fake_stream = FakeStream()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: fake_stream
        )))
        with bind_generation_cancel(cancel_event):
            with self.assertRaises(GenerationCancelled):
                stream_chat_completion_text(client, model="test", messages=[])

        self.assertTrue(fake_stream.closed)

    def test_sse_preserves_standalone_newline_token(self):
        encoded = server.sse_event("\n", event="message")
        self.assertEqual(encoded.count("data: "), 2)

    def test_chat_model_is_configured_for_provider_token_streaming(self):
        runtime_settings = type("RuntimeSettings", (), {
            "amd_model": "stream-model",
            "amd_base_url": "https://example.test/v1",
            "request_timeout_seconds": 45,
        })()
        with (
            patch.object(agent_graph, "get_settings", return_value=runtime_settings),
            patch.object(agent_graph, "_load_api_key", return_value="test-key"),
            patch.object(agent_graph, "ChatOpenAI") as chat_openai,
        ):
            agent_graph.get_llm()

        kwargs = chat_openai.call_args.kwargs
        self.assertTrue(kwargs["streaming"])
        self.assertEqual(kwargs["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertEqual(kwargs["timeout"], 300)

    def test_callback_emits_stage_and_only_chat_answer_tokens(self):
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = threading.Event()
        callback = GraphStreamCallback(loop, queue, cancel_event)
        chat_run = uuid.uuid4()
        route_run = uuid.uuid4()
        try:
            callback.on_chain_start(
                {}, {}, run_id=uuid.uuid4(), metadata={"langgraph_node": "chat"}
            )
            callback.on_chat_model_start(
                {}, [], run_id=route_run, metadata={"langgraph_node": "route"}
            )
            callback.on_llm_new_token("hidden", run_id=route_run)
            callback.on_chat_model_start(
                {}, [], run_id=chat_run, metadata={"langgraph_node": "chat"}
            )
            callback.on_llm_new_token("你", run_id=chat_run)
            callback.on_llm_new_token("好", run_id=chat_run)
            loop.run_until_complete(asyncio.sleep(0))

            events = [queue.get_nowait(), queue.get_nowait(), queue.get_nowait()]
        finally:
            loop.close()

        self.assertEqual(events[0][0], "stage")
        self.assertEqual([event[1] for event in events[1:]], ["你", "好"])
        self.assertEqual(callback.answer_text, "你好")

    def test_callback_raises_when_user_cancels(self):
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        cancel_event = threading.Event()
        callback = GraphStreamCallback(loop, queue, cancel_event)
        cancel_event.set()
        try:
            with self.assertRaises(GenerationCancelled):
                callback.on_chain_start(
                    {}, {}, run_id=uuid.uuid4(), metadata={"langgraph_node": "chat"}
                )
        finally:
            loop.close()

    def test_chat_api_forwards_tokens_without_repeating_final_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.db")

            def fake_stream_worker(invoke_input, session_id, callback):
                run_id = uuid.uuid4()
                callback.on_chain_start(
                    {}, {}, run_id=uuid.uuid4(), metadata={"langgraph_node": "chat"}
                )
                callback.on_chat_model_start(
                    {}, [], run_id=run_id, metadata={"langgraph_node": "chat"}
                )
                callback.on_llm_new_token("实时", run_id=run_id)
                callback.on_llm_new_token("回答", run_id=run_id)
                callback.on_llm_end(None, run_id=run_id)
                return {"output": "实时回答", "task": None, "task_id": None}

            with (
                patch.object(server, "conversation_store", store),
                patch.object(server, "run_agent_stream_worker", side_effect=fake_stream_worker),
                patch.object(server.settings_service, "is_setup_complete", return_value=True),
            ):
                with TestClient(server.app) as client:
                    response = client.post(
                        "/api/chat",
                        json={
                            "message": "请回答",
                            "session_id": "session-stream",
                            "request_id": "request-stream",
                        },
                    )

            self.assertEqual(response.status_code, 200)
            self.assertIn("event: stage", response.text)
            self.assertEqual(response.text.count("data: 实时"), 1)
            self.assertEqual(response.text.count("data: 回答"), 1)
            self.assertIn("event: done", response.text)
            self.assertEqual(store.messages("session-stream")[-1]["content"], "实时回答")
            self.assertFalse(server._session_locks["session-stream"].locked())
            server._session_locks.pop("session-stream", None)

    def test_summary_persists_streamed_text_when_graph_output_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "conversations.db")
            store.ensure_work_assistant("imported-summary")
            with store._connect() as conn:
                conn.execute(
                    "UPDATE conversations SET source = 'codex', readonly = 1 WHERE id = ?",
                    ("imported-summary",),
                )
            del conn

            def fake_stream_worker(invoke_input, graph_session_id, callback):
                self.assertTrue(invoke_input["force_chat"])
                self.assertTrue(graph_session_id.startswith("summary:imported-summary:"))
                self.assertIn("请求模式：focused", invoke_input["messages"][0].content)
                run_id = uuid.uuid4()
                callback.on_chain_start(
                    {}, {}, run_id=uuid.uuid4(), metadata={"langgraph_node": "chat"}
                )
                callback.on_chat_model_start(
                    {}, [], run_id=run_id, metadata={"langgraph_node": "chat"}
                )
                callback.on_llm_new_token("完整", run_id=run_id)
                callback.on_llm_new_token("总结", run_id=run_id)
                callback.on_llm_end(None, run_id=run_id)
                return {"output": ""}

            with (
                patch.object(server, "conversation_store", store),
                patch.object(server, "run_agent_stream_worker", side_effect=fake_stream_worker),
                patch.object(server.settings_service, "is_setup_complete", return_value=True),
            ):
                with TestClient(server.app) as client:
                    response = client.post(
                        "/api/conversations/imported-summary/summary",
                        json={
                            "message": "请总结",
                            "mode": "focused",
                            "request_id": "summary-stream",
                        },
                    )

            self.assertEqual(response.status_code, 200)
            self.assertIn('"name": "summary_route", "mode": "focused"', response.text)
            self.assertEqual(response.text.count("data: 完整"), 1)
            self.assertEqual(response.text.count("data: 总结"), 1)
            self.assertEqual(store.comments("imported-summary")[-1]["content"], "完整总结")
            for _ in range(100):
                if "summary:imported-summary" not in server._active_generations:
                    break
                time.sleep(0.01)
            self.assertNotIn("summary:imported-summary", server._active_generations)
            self.assertFalse(server._session_locks["summary:imported-summary"].locked())
            server._session_locks.pop("summary:imported-summary", None)

    def test_summary_prompt_keeps_latest_history_within_conservative_budget(self):
        messages = [
            {"role": "user", "content": "旧消息" * 8_000},
            {"role": "assistant", "content": "最新结论"},
        ]
        prompt = server._conversation_summary_prompt(
            {"source": "codex", "project_name": "示例", "title": "长历史"},
            messages,
            "请总结",
        )

        self.assertIn("最新结论", prompt)
        self.assertIn("（更早的历史已省略）", prompt)
        self.assertIn("# 事项清单", prompt)
        self.assertIn("# 文件变更汇总", prompt)
        self.assertIn("不得猜测", prompt)
        self.assertLess(len(prompt), 22_000)

    def test_focused_summary_prompt_uses_retrieved_evidence_instead_of_latest_window(self):
        prompt = server._conversation_summary_prompt(
            {"source": "codex", "project_name": "示例", "title": "检索历史"},
            [{"role": "assistant", "content": "无关的最新讨论"}],
            "auth.py 的登录错误原因是什么？",
            analysis_mode="focused",
            retrieved_chunks=[{
                "segment_id": "SEG-auth",
                "start_ordinal": 4,
                "end_ordinal": 6,
                "score": 3.2,
                "is_neighbor": False,
                "content": "用户：修复 auth.py 的登录 token 校验失败。",
            }],
        )

        self.assertIn("检索到的原始历史证据", prompt)
        self.assertIn("[证据 1｜检索命中｜片段 SEG-auth｜消息 4–6", prompt)
        self.assertIn("只能依据这些证据下结论", prompt)
        self.assertNotIn("无关的最新讨论", prompt)

    def test_full_summary_uses_hierarchical_context_when_index_is_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_store = ConversationStore(root / "conversations.db")
            legacy_store.ensure_work_assistant("codex:full-summary")
            with legacy_store._connect() as conn:
                conn.execute(
                    "UPDATE conversations SET source='codex', readonly=1 WHERE id=?",
                    ("codex:full-summary",),
                )
            del conn

            workspace = WorkspaceStore(root / "workspace.db")
            workspace_conversation = workspace.upsert_conversation(
                "codex", "full-summary", title="完整历史"
            )
            workspace_messages = [
                {"ordinal": 0, "role": "user", "content": "实现认证接口", "created_at": "2026-08-05T10:00:00"},
                {"ordinal": 1, "role": "assistant", "content": "认证接口已完成", "created_at": "2026-08-05T10:01:00"},
            ]
            workspace.replace_messages(workspace_conversation["conversation_id"], workspace_messages)
            segments = workspace.replace_conversation_segments(
                workspace_conversation["conversation_id"],
                [{"start_ordinal": 0, "end_ordinal": 1, "title": "认证", "boundary_reason": "conversation_start"}],
                message_fingerprint=message_fingerprint(workspace_messages),
                segmenter_version="test",
            )
            chunk_content = "用户：实现认证接口\n\n助手：认证接口已完成"
            workspace.replace_retrieval_chunks(
                workspace_conversation["conversation_id"],
                [{
                    "segment_id": segments[0]["segment_id"], "chunk_index": 0,
                    "start_ordinal": 0, "end_ordinal": 1,
                    "content": chunk_content, "char_count": len(chunk_content),
                }],
                message_fingerprint=message_fingerprint(workspace_messages),
                chunker_version="test",
            )

            hierarchy = SimpleNamespace(build_final_prompt=lambda **kwargs: (
                "层级汇总后的完整上下文",
                {"strategy": "hierarchical_map_reduce", "map_call_count": 1},
            ))

            def fake_stream_worker(invoke_input, graph_session_id, callback):
                self.assertIn("层级汇总后的完整上下文", invoke_input["messages"][0].content)
                return {"output": "完整回顾"}

            with (
                patch.object(server, "conversation_store", legacy_store),
                patch.object(server, "workspace_store", workspace),
                patch.object(server, "summary_hierarchy", hierarchy),
                patch.object(server, "run_agent_stream_worker", side_effect=fake_stream_worker),
                patch.object(server.settings_service, "is_setup_complete", return_value=True),
            ):
                with TestClient(server.app) as client:
                    response = client.post(
                        "/api/conversations/codex%3Afull-summary/summary",
                        json={"message": "请完整回顾", "mode": "full"},
                    )

            self.assertEqual(response.status_code, 200)
            self.assertIn("data: 完整回顾", response.text)
            for _ in range(100):
                if "summary:codex:full-summary" not in server._active_generations:
                    break
                time.sleep(0.01)
            self.assertNotIn("summary:codex:full-summary", server._active_generations)
            server._session_locks.pop("summary:codex:full-summary", None)

    def test_cancel_endpoint_sets_matching_request_event(self):
        cancel_event = threading.Event()
        active = server.ActiveGeneration(
            request_id="request-cancel",
            cancel_event=cancel_event,
            started_at=time.monotonic(),
        )
        server._active_generations["session-cancel"] = active
        try:
            with patch.object(server.settings_service, "is_setup_complete", return_value=True):
                with TestClient(server.app) as client:
                    mismatch = client.post(
                        "/api/chat/session-cancel/cancel?request_id=wrong"
                    )
                    cancelled = client.post(
                        "/api/chat/session-cancel/cancel?request_id=request-cancel"
                    )
        finally:
            server._active_generations.pop("session-cancel", None)

        self.assertEqual(mismatch.status_code, 409)
        self.assertEqual(cancelled.status_code, 200)
        self.assertTrue(cancel_event.is_set())

    def test_frontend_exposes_stop_retry_and_stage_handling(self):
        template = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn('id="stopBtn"', template)
        self.assertIn('id="retryBtn"', template)
        self.assertIn("async function stopGeneration()", template)
        self.assertIn('eventType === "stage"', template)
        self.assertIn("answerText += data", template)
        self.assertIn("summary/cancel?request_id=", template)
        self.assertIn("receivedDone && !requestFailed && !requestCancelled", template)


if __name__ == "__main__":
    unittest.main()
