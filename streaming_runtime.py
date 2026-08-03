"""LangGraph 执行过程到 SSE 的线程安全事件桥接。"""

import asyncio
import threading
from contextlib import contextmanager
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


NODE_STAGE_LABELS = {
    "route": "正在识别请求类型",
    "gather_info": "正在整理任务信息",
    "create_task": "正在生成任务计划",
    "init_existing": "正在分析现有项目",
    "load_task": "正在读取任务",
    "collect_snapshot": "正在采集项目快照",
    "analyze_progress": "正在分析任务进度",
    "format_output": "正在整理分析结果",
    "chat": "正在生成回答",
}


class GenerationCancelled(RuntimeError):
    """用户主动停止本轮生成。"""


_generation_context = threading.local()


@contextmanager
def bind_generation_cancel(cancel_event: threading.Event):
    """让当前工作线程内的直接 OpenAI 流也能响应同一个取消信号。"""
    previous = getattr(_generation_context, "cancel_event", None)
    _generation_context.cancel_event = cancel_event
    try:
        yield
    finally:
        _generation_context.cancel_event = previous


def _current_cancel_event() -> threading.Event | None:
    return getattr(_generation_context, "cancel_event", None)


def stream_chat_completion_text(client, **request_options) -> str:
    """流式读取 OpenAI 兼容响应并拼接文本；结构化调用也可及时取消。"""
    cancel_event = _current_cancel_event()
    if cancel_event and cancel_event.is_set():
        raise GenerationCancelled("用户已停止生成")
    stream = client.chat.completions.create(stream=True, **request_options)
    chunks: list[str] = []
    try:
        for event in stream:
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled("用户已停止生成")
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str):
                chunks.append(content)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if cancel_event and cancel_event.is_set():
        raise GenerationCancelled("用户已停止生成")
    return "".join(chunks)


class GraphStreamCallback(BaseCallbackHandler):
    """把 LangGraph 节点阶段和 chat 模型 token 推入 asyncio 队列。"""

    raise_error = True

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        event_queue: asyncio.Queue,
        cancel_event: threading.Event,
    ):
        self.loop = loop
        self.event_queue = event_queue
        self.cancel_event = cancel_event
        self._started_nodes: set[str] = set()
        self._chat_model_runs: set[UUID] = set()
        self.answer_chunks: list[str] = []

    @property
    def answer_text(self) -> str:
        return "".join(self.answer_chunks)

    @property
    def streamed_answer(self) -> bool:
        return bool(self.answer_chunks)

    def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise GenerationCancelled("用户已停止生成")

    def _emit(self, event_type: str, data):
        self.loop.call_soon_threadsafe(
            self.event_queue.put_nowait,
            (event_type, data),
        )

    def on_chain_start(self, serialized, inputs, *, metadata=None, **kwargs):
        self._check_cancelled()
        node = (metadata or {}).get("langgraph_node")
        if node in NODE_STAGE_LABELS and node not in self._started_nodes:
            self._started_nodes.add(node)
            self._emit("stage", {"node": node, "label": NODE_STAGE_LABELS[node]})

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        metadata=None,
        **kwargs,
    ):
        self._check_cancelled()
        if (metadata or {}).get("langgraph_node") == "chat":
            self._chat_model_runs.add(run_id)

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        self._check_cancelled()
        if run_id not in self._chat_model_runs or not token:
            return
        if isinstance(token, str):
            text = token
        else:
            text = "".join(part for part in token if isinstance(part, str))
        if text:
            self.answer_chunks.append(text)
            self._emit("token", text)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._chat_model_runs.discard(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._chat_model_runs.discard(run_id)
