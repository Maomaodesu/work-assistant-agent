"""把已导入的本地 Codex / Claude 会话安全桥接到网页聊天界面。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import AsyncIterator

from workspace_launcher import discover_local_tools


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,240}$")
MAX_MESSAGE_CHARS = 12_000


class ExternalCliChatError(ValueError):
    pass


def _command_prefix(executable: str) -> list[str]:
    """为已检测到的 CLI 构建非 shell 命令前缀。"""
    suffix = Path(executable).suffix.lower()
    if suffix in {".cmd", ".bat"}:
        command_processor = os.getenv("COMSPEC")
        if not command_processor:
            raise ExternalCliChatError("未找到 cmd.exe，无法启动本地 CLI")
        return [command_processor, "/d", "/c", executable]
    if suffix == ".ps1":
        powershell = Path(os.getenv("SystemRoot", "C:/Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if not powershell.is_file():
            raise ExternalCliChatError("未找到 PowerShell，无法启动本地 CLI")
        return [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", executable]
    return [executable]


def external_session_id(conversation: dict) -> str:
    """外部导入记录的 id 采用 `source:session-id`，只取可信数据库中的后半段。"""
    source = str(conversation.get("source", ""))
    value = str(conversation.get("id", ""))
    prefix = source + ":"
    if source not in {"codex", "claude"} or not value.startswith(prefix):
        raise ExternalCliChatError("这不是可续接的 Codex 或 Claude 会话")
    session_id = value[len(prefix):]
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ExternalCliChatError("外部会话 ID 格式不安全，无法启动")
    return session_id


def build_external_cli_command(
    source: str,
    executable: str,
    session_id: str,
    prompt: str,
) -> list[str]:
    """只允许固定的恢复会话命令，浏览器不能影响可执行路径或额外参数。"""
    if source not in {"codex", "claude"}:
        raise ExternalCliChatError("仅支持 Codex 或 Claude 会话")
    if not SESSION_ID_PATTERN.fullmatch(str(session_id)):
        raise ExternalCliChatError("外部会话 ID 格式不安全，无法启动")
    normalized_prompt = str(prompt).strip()
    if not normalized_prompt or len(normalized_prompt) > MAX_MESSAGE_CHARS:
        raise ExternalCliChatError("消息不能为空，且不能超过 12000 个字符")
    tool_path = Path(executable).expanduser()
    if not tool_path.is_file():
        raise ExternalCliChatError("本地 CLI 路径不存在，请在设置页重新配置")
    command = _command_prefix(str(tool_path.resolve()))
    if source == "codex":
        return command + ["exec", "resume", "--json", str(session_id), normalized_prompt]
    return command + [
        "--print", "--output-format", "stream-json", "--include-partial-messages",
        "--resume", str(session_id), normalized_prompt,
    ]


def build_cli_environment(source: str, conversation: dict, base_environment: dict | None = None) -> dict:
    """让 CLI 使用导入该会话时对应的本地配置根目录。

    Work Assistant 有时由受控宿主启动，可能继承临时的 CODEX_HOME；此时必须
    优先使用会话来源文件所在的真实 `.codex` 目录，否则无法找到原线程。
    """
    environment = dict(base_environment or os.environ)
    if source != "codex":
        return environment
    source_path = Path(str(conversation.get("source_path", ""))).expanduser()
    for candidate in (source_path, *source_path.parents):
        if candidate.name.lower() == ".codex" and candidate.is_dir():
            environment["CODEX_HOME"] = str(candidate.resolve())
            break
    return environment


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return ""


def extract_stream_text(source: str, event: dict) -> str:
    """抽取两种 CLI 机器可读输出中的可显示增量文本。"""
    if source == "codex":
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        if item.get("type") in {"agent_message", "agentMessage"}:
            return str(item.get("text") or _content_text(item.get("content")) or "")
        if event.get("type") in {"agent_message", "agentMessage"}:
            return str(event.get("text") or _content_text(event.get("content")) or "")
        return ""

    event_type = event.get("type")
    if event_type == "stream_event":
        stream_event = event.get("event") if isinstance(event.get("event"), dict) else {}
        if stream_event.get("type") == "content_block_delta":
            delta = stream_event.get("delta") if isinstance(stream_event.get("delta"), dict) else {}
            return str(delta.get("text") or "")
    if event_type == "assistant":
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        return _content_text(message.get("content"))
    if event_type == "result":
        return str(event.get("result") or "")
    return ""


class ExternalCliChatRunner:
    """每条消息启动一次可恢复 CLI 回合；CLI 自己持久化原始会话。"""

    def __init__(self):
        self._active: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    def active(self, conversation_id: str) -> bool:
        return conversation_id in self._active

    async def cancel(self, conversation_id: str) -> bool:
        process = self._active.get(conversation_id)
        if not process or process.returncode is not None:
            return False
        self._cancelled.add(conversation_id)
        process.terminate()
        return True

    async def stream(
        self,
        conversation: dict,
        prompt: str,
        *,
        configured_paths: dict | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        conversation_id = str(conversation.get("id", ""))
        source = str(conversation.get("source", ""))
        if self.active(conversation_id):
            raise ExternalCliChatError("该外部会话正在处理上一条消息")
        raw_project_path = str(conversation.get("project_path", "")).strip()
        if not raw_project_path:
            raise ExternalCliChatError("原会话没有可信项目目录，无法安全续接")
        project_path = Path(raw_project_path).expanduser().resolve()
        if not project_path.is_dir():
            raise ExternalCliChatError("原会话的项目目录不存在，无法安全续接")
        session_id = external_session_id(conversation)
        tool = discover_local_tools(configured_paths).get(source)
        if not tool or not tool.get("available"):
            raise ExternalCliChatError(f"未检测到 {source} CLI，请在设置页配置路径")
        command = build_external_cli_command(source, tool["path"], session_id, prompt)
        environment = build_cli_environment(source, conversation)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(project_path),
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ExternalCliChatError(f"启动 {source} CLI 失败：{exc}") from exc

        self._active[conversation_id] = process
        stderr_task = asyncio.create_task(process.stderr.read())
        emitted_text = False
        try:
            yield "stage", f"正在调用本机 {source.title()} 会话"
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    yield "log", raw
                    continue
                text = extract_stream_text(source, event)
                if text:
                    # Claude 的 result 会复述已经发送的增量，避免网页重复显示。
                    if source == "claude" and event.get("type") == "result" and emitted_text:
                        continue
                    emitted_text = True
                    yield "message", text

            return_code = await process.wait()
            stderr = (await stderr_task).decode("utf-8", errors="replace").strip()
            if conversation_id in self._cancelled:
                yield "cancelled", "已停止本地 CLI 会话"
            elif return_code != 0:
                detail = stderr[-1500:] or f"CLI 退出码：{return_code}"
                yield "error", f"{source.title()} 会话执行失败：{detail}"
            elif not emitted_text:
                detail = stderr[-800:]
                yield "error", detail or f"{source.title()} 没有返回可显示的消息"
            else:
                yield "done", ""
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
            raise
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await process.wait()
            except ProcessLookupError:
                pass
            if not stderr_task.done():
                stderr_task.cancel()
            self._active.pop(conversation_id, None)
            self._cancelled.discard(conversation_id)


external_cli_chat_runner = ExternalCliChatRunner()
