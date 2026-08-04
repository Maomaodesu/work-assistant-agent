"""从已关联的项目级 AI 会话中提炼待确认的项目定义。"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Callable

from openai import OpenAI

from settings import get_api_key, get_settings
from work_item_discovery import redact_sensitive_text
from workspace_store import WorkspaceStore, workspace_store


EXTRACTOR_VERSION = "amd-project-definition-v1"
MAX_SEGMENTS = 8
MAX_SEGMENT_CHARS = 6_000
MAX_TOTAL_CHARS = 24_000

SYSTEM_PROMPT = """你是本地开发工作助手中的项目定义整理器。
仅根据给出的项目级讨论，提炼一份供用户确认的项目定义草稿。不要补全、猜测或把技术细节虚构成需求；没有明确提到的字段请返回空字符串。
只返回 JSON，不使用 Markdown，格式如下：
{"summary":"一句话概述","goal":"要解决的问题和预期结果","scope":"首期包含的范围","non_goals":"明确暂不做的内容","acceptance_criteria":"可验证的完成标准","constraints":"已明确的技术、时间、兼容性或其他约束"}
"""


def _parse_response(content: str) -> dict:
    value = str(content or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AMD 模型未返回项目定义 JSON 对象")
    parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("AMD 模型返回的项目定义格式不正确")
    return parsed


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[片段已截断]"


class ProjectDefinitionDiscovery:
    def __init__(
        self,
        store: WorkspaceStore | None = None,
        client_factory: Callable[[], object] | None = None,
    ):
        self.store = store or workspace_store
        self.client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client():
        settings = get_settings()
        return OpenAI(
            api_key=get_api_key(), base_url=settings.amd_base_url,
            timeout=settings.request_timeout_seconds, max_retries=0,
        )

    def _source_payload(self, project_id: str) -> tuple[list[dict], Counter]:
        details = self.store.list_project_segment_details(
            project_id, segment_kind="project", limit=MAX_SEGMENTS
        )
        payload: list[dict] = []
        counts: Counter[str] = Counter()
        remaining = MAX_TOTAL_CHARS
        for detail in details:
            if remaining <= 0:
                break
            messages = self.store.list_messages_for_segment(detail["segment_id"])
            blocks = []
            for message in messages:
                redacted = redact_sensitive_text(message.get("content", ""))
                counts.update(redacted.counts)
                blocks.append(f"[{message.get('role', 'unknown')}]\n{redacted.text}")
            text = _truncate("\n\n".join(blocks), min(MAX_SEGMENT_CHARS, remaining))
            remaining -= len(text)
            if text.strip():
                payload.append({
                    "segment_id": detail["segment_id"],
                    "title": detail.get("title", ""),
                    "summary": detail.get("summary", ""),
                    "conversation": text,
                })
        return payload, counts

    def discover(
        self,
        *,
        project_id: str | None = None,
        run_id: str = "",
        control=None,
    ) -> dict:
        project_ids = [project_id] if project_id else [
            project["project_id"] for project in self.store.list_projects(status="active")
        ]
        candidates: list[tuple[dict, list[dict], Counter]] = []
        result = {
            "created": 0, "updated": 0, "skipped": 0,
            "amd_call_count": 0, "credential_redaction_count": 0,
            "errors": [],
        }
        for current_project_id in project_ids:
            if control is not None and not control.checkpoint():
                return result
            existing = self.store.get_project_definition(current_project_id)
            # 用户已确认或明确忽略的定义只能由用户主动编辑，不能被扫描覆盖。
            if existing and existing["status"] in {"confirmed", "ignored"}:
                result["skipped"] += 1
                continue
            segments, redactions = self._source_payload(current_project_id)
            if not segments:
                result["skipped"] += 1
                continue
            if existing and set(existing.get("source_segment_ids", [])) == {
                segment["segment_id"] for segment in segments
            }:
                # 同一份草稿的来源没有变化时，遵循增量分析约定，不重复请求模型。
                result["skipped"] += 1
                continue
            project = self.store.get_project(current_project_id)
            candidates.append((project, segments, redactions))

        if not candidates:
            return result
        client = self.client_factory()
        settings = get_settings()
        for project, segments, redactions in candidates:
            if control is not None and not control.checkpoint():
                return result
            try:
                response = client.chat.completions.create(
                    model=settings.amd_model,
                    temperature=0.1,
                    max_tokens=1200,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({
                            "project": {"project_id": project["project_id"], "name": project["name"]},
                            "segments": segments,
                        }, ensure_ascii=False)},
                    ],
                )
                result["amd_call_count"] += 1
                result["credential_redaction_count"] += sum(redactions.values())
                definition = _parse_response(response.choices[0].message.content)
                fields = {
                    key: str(definition.get(key) or "").strip()
                    for key in ("summary", "goal", "scope", "non_goals", "acceptance_criteria", "constraints")
                }
                if not any(fields.values()):
                    raise ValueError("AMD 未从讨论中提取到可确认的项目定义")
                previous = self.store.get_project_definition(project["project_id"])
                self.store.save_project_definition(
                    project["project_id"], **fields, status="draft", source="inferred",
                    source_segment_ids=[segment["segment_id"] for segment in segments],
                    source_run_id=run_id,
                )
                result["updated" if previous else "created"] += 1
            except Exception as exc:
                result["errors"].append({"project_id": project["project_id"], "error": str(exc)})
        return result


project_definition_discovery = ProjectDefinitionDiscovery()
