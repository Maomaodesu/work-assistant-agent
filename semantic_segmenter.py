"""不调用模型的会话语义切分器。"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime

from workspace_store import WorkspaceStore, workspace_store


SEGMENTER_VERSION = "local-rules-v1"
TIME_GAP_SECONDS = 2 * 60 * 60
MAX_SEGMENT_MESSAGES = 48
MAX_SEGMENT_CHARS = 30_000

NEW_TOPIC_PATTERN = re.compile(
    r"^\s*(?:好的?[，,。\s]*)?(?:"
    r"接下来|下一步|现在(?:开始|来|需要|先)|另外|另一个|还有(?:一个|个)|顺便|"
    r"换个|切换到|新任务|新需求|再来|开始做|"
    r"next\b|now\b|another\b|new\s+(?:task|feature|issue)\b|"
    r"switch\s+to\b|let['’]?s\s+move\b|separately\b"
    r")",
    re.IGNORECASE,
)
TASK_ID_PATTERN = re.compile(r"\b(?:TASK|WI)-[A-Z0-9-]+\b", re.IGNORECASE)
ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.+#-]{2,}")
CHINESE_SEQUENCE_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
TOOL_BLOCK_PATTERN = re.compile(
    r"\[external_agent_tool_(?:call|result)[^\]]*\][\s\S]*?"
    r"\[/external_agent_tool_(?:call|result)\]",
    re.IGNORECASE,
)
STOP_TOKENS = {
    "the", "and", "for", "this", "that", "with", "from", "have", "please", "help",
    "一下", "现在", "这个", "那个", "可以", "需要", "帮我", "继续", "已经", "进行",
    "项目", "功能", "问题", "工作", "代码", "实现", "看看", "好的", "然后", "我们",
}


def message_fingerprint(messages: list[dict]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(json.dumps(
            [message.get("ordinal"), message.get("role"), message.get("created_at"), message.get("content")],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"))
    return digest.hexdigest()


def _timestamp(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, OSError):
        return None


def _clean_for_topic(text: str) -> str:
    text = TOOL_BLOCK_PATTERN.sub(" ", text)
    text = CODE_FENCE_PATTERN.sub(" ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s]+", " ", text)
    return text


def topic_tokens(text: str) -> set[str]:
    cleaned = _clean_for_topic(text)
    tokens = {
        token.lower() for token in ENGLISH_TOKEN_PATTERN.findall(cleaned)
        if token.lower() not in STOP_TOKENS
    }
    for sequence in CHINESE_SEQUENCE_PATTERN.findall(cleaned):
        if sequence in STOP_TOKENS:
            continue
        if len(sequence) <= 4:
            tokens.add(sequence)
        else:
            tokens.update(
                sequence[index:index + 2]
                for index in range(len(sequence) - 1)
                if sequence[index:index + 2] not in STOP_TOKENS
            )
    return tokens


def _semantic_shift(recent_user_texts: list[str], new_text: str) -> bool:
    if len(recent_user_texts) < 3 or len(new_text.strip()) < 10:
        return False
    existing = set().union(*(topic_tokens(text) for text in recent_user_texts[-3:]))
    incoming = topic_tokens(new_text)
    if len(existing) < 5 or len(incoming) < 4:
        return False
    similarity = len(existing & incoming) / len(existing | incoming)
    return similarity < 0.045


def _first_user_title(messages: list[dict], fallback: str) -> str:
    content = next(
        (message["content"] for message in messages if message["role"] == "user"),
        fallback,
    )
    content = TOOL_BLOCK_PATTERN.sub("", content)
    normalized = " ".join(content.split())
    return normalized[:100] or fallback[:100] or "未命名片段"


def build_semantic_segments(
    messages: list[dict],
    *,
    fallback_title: str = "会话片段",
    project_id: str | None = None,
) -> list[dict]:
    if not messages:
        return []
    boundaries = [(0, "conversation_start")]
    segment_user_texts: list[str] = []
    segment_char_count = 0
    previous_user_timestamp = None
    previous_task_ids: set[str] = set()

    for index, message in enumerate(messages):
        content = str(message.get("content", ""))
        if message.get("role") != "user":
            segment_char_count += len(content)
            continue

        reason = None
        current_timestamp = _timestamp(message.get("created_at", ""))
        current_task_ids = {value.upper() for value in TASK_ID_PATTERN.findall(content)}
        segment_start_index = boundaries[-1][0]
        segment_message_count = index - segment_start_index

        if index > segment_start_index:
            if (
                previous_user_timestamp is not None
                and current_timestamp is not None
                and current_timestamp - previous_user_timestamp >= TIME_GAP_SECONDS
            ):
                reason = "time_gap"
            elif previous_task_ids and current_task_ids and previous_task_ids.isdisjoint(current_task_ids):
                reason = "task_id_changed"
            elif NEW_TOPIC_PATTERN.search(content):
                reason = "explicit_topic_change"
            elif segment_message_count >= MAX_SEGMENT_MESSAGES or segment_char_count >= MAX_SEGMENT_CHARS:
                reason = "segment_size_limit"
            elif _semantic_shift(segment_user_texts, content):
                reason = "semantic_shift"

        if reason:
            boundaries.append((index, reason))
            segment_user_texts = []
            segment_char_count = 0
            previous_task_ids = set()

        segment_user_texts.append(content)
        segment_char_count += len(content)
        if current_timestamp is not None:
            previous_user_timestamp = current_timestamp
        if current_task_ids:
            previous_task_ids = current_task_ids

    segments = []
    for boundary_index, (start_index, reason) in enumerate(boundaries):
        end_index = (
            boundaries[boundary_index + 1][0] - 1
            if boundary_index + 1 < len(boundaries)
            else len(messages) - 1
        )
        segment_messages = messages[start_index:end_index + 1]
        segments.append({
            "start_ordinal": segment_messages[0]["ordinal"],
            "end_ordinal": segment_messages[-1]["ordinal"],
            "boundary_reason": reason,
            "title": _first_user_title(segment_messages, fallback_title),
            "project_id": project_id,
        })
    return segments


class SemanticConversationSegmenter:
    def __init__(self, store: WorkspaceStore | None = None):
        self.store = store or workspace_store

    def segment_conversation(self, conversation: dict) -> dict:
        conversation_id = conversation["conversation_id"]
        messages = self.store.list_messages(conversation_id)
        if not messages:
            return {"state": "empty", "segment_count": 0, "boundary_reasons": {}}
        fingerprint = message_fingerprint(messages)
        state = self.store.get_segmentation_state(conversation_id)
        if (
            state
            and state["status"] == "completed"
            and state["message_fingerprint"] == fingerprint
            and state["segmenter_version"] == SEGMENTER_VERSION
        ):
            return {
                "state": "unchanged",
                "segment_count": state["segment_count"],
                "boundary_reasons": {},
            }

        primary_project = conversation.get("primary_project")
        segments = build_semantic_segments(
            messages,
            fallback_title=conversation.get("title") or "会话片段",
            project_id=primary_project["project_id"] if primary_project else None,
        )
        try:
            saved = self.store.replace_conversation_segments(
                conversation_id,
                segments,
                message_fingerprint=fingerprint,
                segmenter_version=SEGMENTER_VERSION,
            )
        except Exception as exc:
            self.store.save_segmentation_error(
                conversation_id,
                message_fingerprint=fingerprint,
                segmenter_version=SEGMENTER_VERSION,
                error_message=str(exc),
            )
            raise
        reasons = Counter(segment["boundary_reason"] for segment in segments)
        return {
            "state": "segmented",
            "segment_count": len(saved),
            "boundary_reasons": dict(reasons),
        }

    def segment_all(self) -> dict:
        conversations = self.store.list_conversations()
        result = {
            "total_conversations": len(conversations),
            "segmented_conversations": 0,
            "unchanged_conversations": 0,
            "protected_conversations": 0,
            "empty_conversations": 0,
            "errors": 0,
            "segments_created": 0,
            "boundary_reasons": {},
            "error_details": [],
            "segmenter_version": SEGMENTER_VERSION,
        }
        boundary_counter = Counter()
        for conversation in conversations:
            try:
                outcome = self.segment_conversation(conversation)
                if outcome["state"] == "segmented":
                    result["segmented_conversations"] += 1
                    result["segments_created"] += outcome["segment_count"]
                    boundary_counter.update(outcome["boundary_reasons"])
                elif outcome["state"] == "unchanged":
                    result["unchanged_conversations"] += 1
                else:
                    result["empty_conversations"] += 1
            except Exception as exc:
                result["errors"] += 1
                if len(result["error_details"]) < 20:
                    result["error_details"].append({
                        "conversation_id": conversation["conversation_id"],
                        "error": str(exc),
                    })
        result["boundary_reasons"] = dict(boundary_counter)
        return result


semantic_conversation_segmenter = SemanticConversationSegmenter()
