"""从会话片段识别并自动发现待确认工作项。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

from openai import OpenAI

from settings import get_api_key, get_settings
from workspace_store import WorkspaceStore, workspace_store


CLASSIFIER_VERSION = "rules-amd-v1"
MAX_SEGMENT_CHARS = 12_000
BATCH_SIZE = 4
MAX_BATCH_CHARS = 24_000
CREATE_CONFIDENCE_THRESHOLD = 0.62


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


_SECRET_PATTERNS = (
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
        r"-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
        re.DOTALL | re.IGNORECASE,
    )),
    ("AUTHORIZATION", re.compile(
        r"(?im)^\s*(?:authorization|proxy-authorization)\s*:\s*[^\r\n]+"
    )),
    ("COOKIE", re.compile(r"(?im)^\s*(?:cookie|set-cookie)\s*:\s*[^\r\n]+")),
    ("URL_CREDENTIALS", re.compile(
        r"(?i)(https?://)([^\s/:@]+):([^\s/@]+)@"
    )),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("KNOWN_API_KEY", re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|amd[_-][A-Za-z0-9_-]{16,})\b"
    )),
    ("API_KEY", re.compile(
        r"(?i)\b(?:[A-Za-z][A-Za-z0-9]*[_-])?"
        r"(api[_-]?key|apikey|access[_-]?token|secret|token|password|passwd|pwd)\b"
        r"\s*[:=]\s*[\"']?[^\s\"'`;]{6,}[\"']?"
    )),
)


def redact_sensitive_text(text: str) -> RedactionResult:
    """删除凭据，但保留代码、命令、工具输出与普通路径。"""
    redacted = str(text)
    counts: Counter[str] = Counter()
    for label, pattern in _SECRET_PATTERNS:
        if label == "URL_CREDENTIALS":
            def replace_url(match: re.Match) -> str:
                counts[label] += 1
                return f"{match.group(1)}[REDACTED:URL_CREDENTIALS]@"
            redacted = pattern.sub(replace_url, redacted)
            continue
        redacted, count = pattern.subn(f"[REDACTED:{label}]", redacted)
        if count:
            counts[label] += count
    return RedactionResult(redacted, dict(counts))


def _segment_fingerprint(messages: list[dict]) -> str:
    payload = [
        [message.get("ordinal"), message.get("role"), message.get("content"), message.get("created_at")]
        for message in messages
    ]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_ACK_PATTERN = re.compile(
    r"^(?:好|好的|可以|行|嗯|收到|知道了|谢谢|感谢|ok|okay|yes|no|continue|继续|开始吧|没问题)"
    r"[。.!！?？\s]*$",
    re.IGNORECASE,
)
HANDOFF_WORK_ITEM_PATTERN = re.compile(
    r"work\s*assistant\s*工作项\s*id\s*[:：]?\s*`?(WI-[A-Za-z0-9-]+)`?",
    re.IGNORECASE,
)


def local_screen(messages: list[dict], project_id: str | None) -> tuple[str, str]:
    """返回 amd / casual / unassigned；只跳过确定不需要模型的片段。"""
    if not project_id:
        return "unassigned", "片段尚未识别所属项目"
    user_texts = [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    if not user_texts:
        return "casual", "片段没有用户文本"
    if all(_ACK_PATTERN.fullmatch(text) for text in user_texts):
        return "casual", "片段仅包含简短确认"
    return "amd", "需要语义判断"


def _handoff_work_item(store: WorkspaceStore, messages: list[dict], project_id: str | None) -> dict | None:
    """识别由 Work Assistant 生成的交接提示词，可靠地回链到原工作项。"""
    if not project_id:
        return None
    for message in messages:
        match = HANDOFF_WORK_ITEM_PATTERN.search(str(message.get("content") or ""))
        if not match:
            continue
        item = store.get_work_item(match.group(1))
        if item and item["project_id"] == project_id and item["status"] not in {"ignored", "archived"}:
            return item
    return None


def _truncate(text: str, limit: int = MAX_SEGMENT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    tail = limit - head
    return text[:head] + "\n...[内容因模型上下文长度被截断]...\n" + text[-tail:]


def _messages_for_model(messages: list[dict]) -> tuple[str, Counter]:
    blocks = []
    counts: Counter[str] = Counter()
    for message in messages:
        result = redact_sensitive_text(message.get("content", ""))
        counts.update(result.counts)
        blocks.append(f"[{message.get('role', 'unknown')}]\n{result.text}")
    return _truncate("\n\n".join(blocks)), counts


def _bounded_batches(entries: list[dict]):
    batch, char_count = [], 0
    for entry in entries:
        entry_size = len(entry["model_text"])
        if batch and (len(batch) >= BATCH_SIZE or char_count + entry_size > MAX_BATCH_CHARS):
            yield batch
            batch, char_count = [], 0
        batch.append(entry)
        char_count += entry_size
    if batch:
        yield batch


def _normalize_title(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value).lower())


def _find_similar_work_item(title: str, items: list[dict]) -> dict | None:
    normalized = _normalize_title(title)
    if not normalized:
        return None
    best, best_score = None, 0.0
    for item in items:
        candidate = _normalize_title(item["title"])
        if not candidate:
            continue
        score = SequenceMatcher(None, normalized, candidate).ratio()
        if normalized in candidate or candidate in normalized:
            score = max(score, min(len(normalized), len(candidate)) / max(len(normalized), len(candidate)))
        if score > best_score:
            best, best_score = item, score
    return best if best_score >= 0.82 else None


def _json_from_response(content: str) -> dict:
    normalized = str(content or "").strip()
    normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```$", "", normalized)
    start, end = normalized.find("{"), normalized.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AMD 模型未返回 JSON 对象")
    value = json.loads(normalized[start:end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
        raise ValueError("AMD 模型返回的 decisions 格式不正确")
    return value


SYSTEM_PROMPT = """你是本地开发工作助手中的会话整理器。分析每个会话片段属于：
1. work_item：明确的功能、缺陷、重构、维护或研究工作；
2. project：项目级讨论、总体规划、架构或状态讨论，不应凭空创建具体工作项；
3. casual：闲聊、简单确认或与项目工作无关的内容。

对于 work_item，优先匹配已有工作项；只有确实是新工作时才 create。一个片段只能选择一个主要工作项。
只返回 JSON，不使用 Markdown。格式：
{"decisions":[{"segment_id":"...","segment_kind":"work_item|project|casual","action":"match|create|none","matched_work_item_id":null,"title":"简洁工作项标题","item_type":"feature|function|bug|refactor|maintenance|research|other","goal":"目标","summary":"片段摘要","confidence":0.0}]}
不得编造不存在于输入中的需求。"""


class WorkItemDiscovery:
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
            api_key=get_api_key(),
            base_url=settings.amd_base_url,
            timeout=settings.request_timeout_seconds,
            max_retries=0,
        )

    def _call_amd(self, client, project: dict, items: list[dict], batch: list[dict]) -> dict:
        settings = get_settings()
        prompt = {
            "project": {"project_id": project["project_id"], "name": project["name"]},
            "existing_work_items": [
                {
                    "work_item_id": item["work_item_id"],
                    "title": item["title"],
                    "type": item["item_type"],
                    "status": item["status"],
                    "goal": item["goal"][:500],
                }
                for item in items
            ],
            "segments": [
                {
                    "segment_id": entry["segment"]["segment_id"],
                    "current_title": entry["segment"]["title"],
                    "conversation": entry["model_text"],
                }
                for entry in batch
            ],
        }
        response = client.chat.completions.create(
            model=settings.amd_model,
            temperature=0.1,
            max_tokens=1800,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        return _json_from_response(response.choices[0].message.content)

    def _apply_decision(
        self,
        entry: dict,
        decision: dict,
        existing_items: list[dict],
        run_id: str,
    ) -> tuple[str, dict | None, bool]:
        segment = entry["segment"]
        messages = entry["messages"]
        kind = str(decision.get("segment_kind", "unclassified"))
        if kind not in {"work_item", "project", "casual"}:
            kind = "unclassified"
        confidence = decision.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None
        title = str(decision.get("title") or segment["title"]).strip()[:160]
        summary = str(decision.get("summary") or "").strip()
        action = str(decision.get("action") or "none")
        linked_item = None
        created_item = False

        if kind == "work_item" and action in {"match", "create"}:
            requested_id = str(decision.get("matched_work_item_id") or "")
            if requested_id:
                linked_item = next(
                    (item for item in existing_items if item["work_item_id"] == requested_id), None
                )
            if not linked_item:
                linked_item = _find_similar_work_item(title, existing_items)
            if (
                not linked_item and action in {"create", "match"} and title
                and confidence is not None and confidence >= CREATE_CONFIDENCE_THRESHOLD
            ):
                item_type = str(decision.get("item_type") or "feature")
                if item_type not in {"feature", "function", "bug", "refactor", "maintenance", "research", "other"}:
                    item_type = "other"
                linked_item = self.store.create_work_item(
                    segment["project_id"],
                    title,
                    item_type=item_type,
                    goal=str(decision.get("goal") or "").strip(),
                    description=summary,
                    source="inferred",
                    confidence=confidence,
                    created_at=messages[0].get("created_at") if messages else None,
                    metadata={
                        "classifier_version": CLASSIFIER_VERSION,
                        "classification_run_id": run_id,
                        "source_segment_id": segment["segment_id"],
                    },
                )
                existing_items.append(linked_item)
                created_item = True

        self.store.update_segment_classification(
            segment["segment_id"],
            segment_kind=kind,
            title=title,
            summary=summary,
            confidence=confidence,
            classification_source="amd",
        )
        self.store.clear_segment_primary_work_item(segment["segment_id"])
        if linked_item:
            self.store.link_segment_work_item(
                segment["segment_id"], linked_item["work_item_id"],
                relation="primary", confidence=confidence,
            )
        return kind, linked_item, created_item

    def discover(
        self,
        *,
        project_id: str | None = None,
        force: bool = False,
        limit: int | None = None,
        run_id: str | None = None,
        control=None,
        finalize_run: bool = True,
    ) -> dict:
        segments = self.store.list_segments(project_id)
        if limit is not None:
            segments = segments[:max(0, int(limit))]
        if run_id:
            run = self.store.get_classification_run(run_id)
            if not run:
                raise ValueError("分析任务不存在")
            self.store.update_classification_run(run_id, total_sources=len(segments))
        else:
            run = self.store.create_classification_run(
                run_type="full" if force else "incremental",
                total_sources=len(segments),
                request={"project_id": project_id, "force": force, "limit": limit},
            )
            run_id = run["run_id"]
        result = {
            "run_id": run_id,
            "total_segments": len(segments),
            "processed_segments": 0,
            "unchanged_segments": 0,
            "locally_skipped_segments": 0,
            "amd_analyzed_segments": 0,
            "amd_call_count": 0,
            "created_work_items": 0,
            "matched_work_items": 0,
            "work_item_segments": 0,
            "project_segments": 0,
            "casual_segments": 0,
            "unclassified_segments": 0,
            "credential_redaction_count": 0,
            "errors": 0,
            "error_details": [],
            "classifier_version": CLASSIFIER_VERSION,
            "cancelled": False,
        }
        pending_by_project: dict[str, list[dict]] = defaultdict(list)

        def progress():
            self.store.update_classification_run(
                run_id,
                processed_sources=result["processed_segments"],
                discovered_count=result["created_work_items"],
                unclassified_count=result["unclassified_segments"],
                amd_call_count=result["amd_call_count"],
                credential_redaction_count=result["credential_redaction_count"],
            )

        def checkpoint() -> bool:
            return control is None or control.checkpoint()

        def cancelled_result() -> dict:
            result["cancelled"] = True
            progress()
            self.store.update_classification_run(
                run_id, status="cancelled", stage="cancelled",
                error_message="用户取消了分析任务",
            )
            result["run"] = self.store.get_classification_run(run_id)
            return result

        if not checkpoint():
            return cancelled_result()
        self.store.update_classification_run(
            run_id, status="running", stage="local_screening"
        )

        for segment in segments:
            if not checkpoint():
                return cancelled_result()
            messages = self.store.list_messages_for_segment(segment["segment_id"])
            fingerprint = _segment_fingerprint(messages)
            state = self.store.get_segment_classification_state(segment["segment_id"])
            if not force and state and state["status"] in {"classified", "skipped"} and (
                state["segment_fingerprint"] == fingerprint
                and state["classifier_version"] == CLASSIFIER_VERSION
            ):
                result["unchanged_segments"] += 1
                result["processed_segments"] += 1
                continue
            if segment["review_status"] == "confirmed" or segment["classification_source"] == "manual":
                self.store.save_segment_classification_state(
                    segment["segment_id"], segment_fingerprint=fingerprint,
                    classifier_version=CLASSIFIER_VERSION, status="skipped",
                    decision={"reason": "manual_or_confirmed"}, run_id=run_id,
                )
                result["locally_skipped_segments"] += 1
                result["processed_segments"] += 1
                continue
            handoff_item = _handoff_work_item(self.store, messages, segment.get("project_id"))
            if handoff_item:
                self.store.update_segment_classification(
                    segment["segment_id"], segment_kind="work_item",
                    title=handoff_item["title"],
                    summary="由 Work Assistant 交接提示词标记，自动关联到原工作项。",
                    confidence=1.0, classification_source="rules",
                )
                self.store.clear_segment_primary_work_item(segment["segment_id"])
                self.store.link_segment_work_item(
                    segment["segment_id"], handoff_item["work_item_id"],
                    relation="primary", confidence=1.0,
                )
                self.store.save_segment_classification_state(
                    segment["segment_id"], segment_fingerprint=fingerprint,
                    classifier_version=CLASSIFIER_VERSION, status="classified",
                    decision={
                        "segment_kind": "work_item", "action": "match",
                        "matched_work_item_id": handoff_item["work_item_id"],
                        "reason": "work_assistant_handoff_marker",
                    },
                    run_id=run_id,
                )
                result["matched_work_items"] += 1
                result["work_item_segments"] += 1
                result["processed_segments"] += 1
                continue
            screen, reason = local_screen(messages, segment.get("project_id"))
            if screen != "amd":
                kind = "casual" if screen == "casual" else "unclassified"
                self.store.update_segment_classification(
                    segment["segment_id"], segment_kind=kind,
                    title=segment["title"], summary=reason, confidence=1.0,
                    classification_source="rules",
                )
                self.store.save_segment_classification_state(
                    segment["segment_id"], segment_fingerprint=fingerprint,
                    classifier_version=CLASSIFIER_VERSION, status="skipped",
                    decision={"segment_kind": kind, "action": "none", "reason": reason},
                    run_id=run_id,
                )
                result["locally_skipped_segments"] += 1
                result[f"{kind}_segments"] += 1
                result["processed_segments"] += 1
                continue
            model_text, redactions = _messages_for_model(messages)
            result["credential_redaction_count"] += sum(redactions.values())
            pending_by_project[segment["project_id"]].append({
                "segment": segment,
                "messages": messages,
                "fingerprint": fingerprint,
                "model_text": model_text,
                "redactions": dict(redactions),
            })

        progress()
        if not checkpoint():
            return cancelled_result()
        try:
            client = self.client_factory() if pending_by_project else None
        except Exception as exc:
            if not checkpoint():
                return cancelled_result()
            pending = [entry for entries in pending_by_project.values() for entry in entries]
            result["errors"] += len(pending)
            result["unclassified_segments"] += len(pending)
            for entry in pending:
                segment_id = entry["segment"]["segment_id"]
                self.store.save_segment_classification_state(
                    segment_id,
                    segment_fingerprint=entry["fingerprint"],
                    classifier_version=CLASSIFIER_VERSION,
                    status="error",
                    redactions=entry["redactions"],
                    run_id=run_id,
                    error_message=str(exc),
                )
                result["processed_segments"] += 1
                if len(result["error_details"]) < 20:
                    result["error_details"].append({
                        "segment_id": segment_id, "error": str(exc)
                    })
            progress()
            self.store.update_classification_run(
                run_id, status="failed", stage="finished",
                error_message=f"AMD 客户端初始化失败：{exc}",
            )
            result["run"] = self.store.get_classification_run(run_id)
            return result
        self.store.update_classification_run(run_id, stage="amd_classification")

        for current_project_id, entries in pending_by_project.items():
            project = self.store.get_project(current_project_id)
            existing_items = self.store.list_work_items(
                current_project_id, include_ignored=True
            )
            for batch in _bounded_batches(entries):
                if not checkpoint():
                    return cancelled_result()
                try:
                    response = self._call_amd(client, project, existing_items, batch)
                    result["amd_call_count"] += 1
                    if not checkpoint():
                        return cancelled_result()
                    decisions = {
                        str(decision.get("segment_id")): decision
                        for decision in response["decisions"] if isinstance(decision, dict)
                    }
                    missing_ids = [
                        entry["segment"]["segment_id"] for entry in batch
                        if entry["segment"]["segment_id"] not in decisions
                    ]
                    if missing_ids:
                        raise ValueError(f"AMD 响应缺少片段：{missing_ids[0]}")
                    for entry in batch:
                        segment_id = entry["segment"]["segment_id"]
                        decision = decisions[segment_id]
                        kind, linked_item, created_item = self._apply_decision(
                            entry, decision, existing_items, run_id
                        )
                        self.store.save_segment_classification_state(
                            segment_id,
                            segment_fingerprint=entry["fingerprint"],
                            classifier_version=CLASSIFIER_VERSION,
                            status="classified",
                            decision=decision,
                            redactions=entry["redactions"],
                            run_id=run_id,
                        )
                        result["amd_analyzed_segments"] += 1
                        result[f"{kind}_segments"] += 1
                        if linked_item:
                            if created_item:
                                result["created_work_items"] += 1
                            else:
                                result["matched_work_items"] += 1
                        elif kind == "work_item":
                            result["unclassified_segments"] += 1
                        result["processed_segments"] += 1
                except Exception as exc:
                    if not checkpoint():
                        return cancelled_result()
                    result["errors"] += len(batch)
                    for entry in batch:
                        segment_id = entry["segment"]["segment_id"]
                        self.store.save_segment_classification_state(
                            segment_id,
                            segment_fingerprint=entry["fingerprint"],
                            classifier_version=CLASSIFIER_VERSION,
                            status="error",
                            redactions=entry["redactions"],
                            run_id=run_id,
                            error_message=str(exc),
                        )
                        result["processed_segments"] += 1
                        result["unclassified_segments"] += 1
                        if len(result["error_details"]) < 20:
                            result["error_details"].append({
                                "segment_id": segment_id, "error": str(exc)
                            })
                progress()

        if finalize_run:
            final_status = "failed" if result["errors"] else "completed"
            self.store.update_classification_run(
                run_id,
                status=final_status,
                stage="finished",
                error_message=(
                    f"{result['errors']} 个片段分析失败" if result["errors"] else ""
                ),
            )
        result["run"] = self.store.get_classification_run(run_id)
        return result


work_item_discovery = WorkItemDiscovery()
