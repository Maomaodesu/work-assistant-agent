"""Build bounded, overlapping source chunks from semantic conversation segments."""

from collections import Counter

from semantic_segmenter import message_fingerprint
from workspace_store import WorkspaceStore, workspace_store


RETRIEVAL_CHUNKER_VERSION = "char-overlap-v1"
MAX_RETRIEVAL_CHUNK_CHARS = 2_000
RETRIEVAL_CHUNK_OVERLAP_CHARS = 200
_OVERLAP_MARKER_RESERVE = 64


def _role_label(role: str) -> str:
    return {
        "user": "用户",
        "assistant": "助手",
        "system": "系统",
    }.get(str(role).lower(), "消息")


def _split_at_natural_boundary(text: str, maximum: int) -> tuple[str, str]:
    """Take a bounded prefix without dropping characters when possible."""
    if len(text) <= maximum:
        return text, ""
    minimum = max(1, maximum * 3 // 5)
    best_end = 0
    for separator in ("\n", "。", "！", "？", ". ", "; ", "，", ", ", " "):
        position = text.rfind(separator, minimum, maximum + 1)
        if position >= minimum:
            best_end = max(best_end, position + len(separator))
    cut_at = best_end or maximum
    return text[:cut_at], text[cut_at:]


def _message_blocks(
    message: dict,
    *,
    block_content_limit: int,
) -> list[tuple[int, str]]:
    """Render one message into bounded blocks while retaining its role and ordinal."""
    ordinal = int(message["ordinal"])
    role = _role_label(message.get("role", ""))
    remaining = str(message.get("content") or "")
    if not remaining:
        return []

    blocks = []
    continuation = False
    while remaining:
        prefix = f"{role}{'（续）' if continuation else ''}："
        piece_limit = max(1, block_content_limit - len(prefix))
        piece, remaining = _split_at_natural_boundary(remaining, piece_limit)
        blocks.append((ordinal, prefix + piece))
        continuation = True
    return blocks


def build_retrieval_chunks(
    segment: dict,
    messages: list[dict],
    *,
    max_chars: int = MAX_RETRIEVAL_CHUNK_CHARS,
    overlap_chars: int = RETRIEVAL_CHUNK_OVERLAP_CHARS,
) -> list[dict]:
    """Create source-preserving chunks for one semantic segment.

    Chunks never cross semantic segment boundaries.  Adjacent chunks repeat a
    trailing raw-text window marked as context, so a retrieved chunk retains
    the transition that led into it.
    """
    if max_chars <= overlap_chars + _OVERLAP_MARKER_RESERVE:
        raise ValueError("检索块最大字符数必须明显大于重叠字符数")
    if overlap_chars < 0:
        raise ValueError("检索块重叠字符数不能为负数")

    block_content_limit = max_chars - overlap_chars - _OVERLAP_MARKER_RESERVE
    blocks = []
    for message in messages:
        blocks.extend(_message_blocks(message, block_content_limit=block_content_limit))
    if not blocks:
        return []

    chunks = []
    current: list[tuple[int, str]] = []

    def render(items: list[tuple[int, str]]) -> str:
        return "\n\n".join(text for _, text in items)

    def emit_current():
        if not current:
            return
        content = render(current)
        chunks.append({
            "segment_id": segment["segment_id"],
            "chunk_index": len(chunks),
            "start_ordinal": current[0][0],
            "end_ordinal": current[-1][0],
            "content": content,
            "char_count": len(content),
        })

    for ordinal, block in blocks:
        separator_size = 2 if current else 0
        if current and len(render(current)) + separator_size + len(block) > max_chars:
            previous_content = render(current)
            previous_end_ordinal = current[-1][0]
            emit_current()
            current = []
            if overlap_chars:
                prefix = f"（上文衔接，来自消息 {previous_end_ordinal}）："
                available_overlap = max(0, max_chars - len(prefix) - len(block) - 2)
                trailing = (
                    previous_content[-min(overlap_chars, available_overlap):]
                    if available_overlap else ""
                )
                if trailing:
                    current.append((previous_end_ordinal, prefix + trailing))
        current.append((ordinal, block))
    emit_current()
    return chunks


class RetrievalChunker:
    def __init__(self, store: WorkspaceStore | None = None):
        self.store = store or workspace_store

    def chunk_conversation(self, conversation: dict) -> dict:
        conversation_id = conversation["conversation_id"]
        messages = self.store.list_messages(conversation_id)
        if not messages:
            return {"state": "empty", "chunk_count": 0}

        fingerprint = message_fingerprint(messages)
        segmentation = self.store.get_segmentation_state(conversation_id)
        if not segmentation or segmentation["status"] != "completed":
            return {"state": "pending_segmentation", "chunk_count": 0}

        state = self.store.get_retrieval_index_state(conversation_id)
        if (
            state
            and state["status"] == "completed"
            and state["message_fingerprint"] == fingerprint
            and state["chunker_version"] == RETRIEVAL_CHUNKER_VERSION
        ):
            return {"state": "unchanged", "chunk_count": state["chunk_count"]}

        try:
            chunks = []
            for segment in self.store.list_segments_for_conversation(conversation_id):
                chunks.extend(build_retrieval_chunks(
                    segment,
                    self.store.list_messages_for_segment(segment["segment_id"]),
                ))
            saved = self.store.replace_retrieval_chunks(
                conversation_id,
                chunks,
                message_fingerprint=fingerprint,
                chunker_version=RETRIEVAL_CHUNKER_VERSION,
            )
        except Exception as exc:
            self.store.save_retrieval_index_error(
                conversation_id,
                message_fingerprint=fingerprint,
                chunker_version=RETRIEVAL_CHUNKER_VERSION,
                error_message=str(exc),
            )
            raise
        return {"state": "indexed", "chunk_count": len(saved)}

    def chunk_all(self) -> dict:
        result = {
            "total_conversations": 0,
            "indexed_conversations": 0,
            "unchanged_conversations": 0,
            "pending_segmentation_conversations": 0,
            "empty_conversations": 0,
            "errors": 0,
            "chunks_created": 0,
            "error_details": [],
            "chunker_version": RETRIEVAL_CHUNKER_VERSION,
        }
        states = Counter()
        for conversation in self.store.list_conversations():
            result["total_conversations"] += 1
            try:
                outcome = self.chunk_conversation(conversation)
                state = outcome["state"]
                states[state] += 1
                result["chunks_created"] += outcome["chunk_count"] if state == "indexed" else 0
            except Exception as exc:
                result["errors"] += 1
                if len(result["error_details"]) < 20:
                    result["error_details"].append({
                        "conversation_id": conversation["conversation_id"],
                        "error": str(exc),
                    })
        result["indexed_conversations"] = states["indexed"]
        result["unchanged_conversations"] = states["unchanged"]
        result["pending_segmentation_conversations"] = states["pending_segmentation"]
        result["empty_conversations"] = states["empty"]
        return result


retrieval_chunker = RetrievalChunker()
