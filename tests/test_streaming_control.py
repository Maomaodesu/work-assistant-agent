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


class StreamingControlTests(unittest.TestCase):
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
        self.assertEqual(kwargs["timeout"], 45)

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


if __name__ == "__main__":
    unittest.main()
