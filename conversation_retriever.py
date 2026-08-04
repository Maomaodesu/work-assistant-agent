"""Local lexical retrieval over bounded conversation source chunks."""

import math
import re
from collections import Counter

from semantic_segmenter import topic_tokens
from workspace_store import WorkspaceStore, workspace_store


RETRIEVER_VERSION = "lexical-bm25-v1"
DEFAULT_RETRIEVAL_LIMIT = 6
DEFAULT_RETRIEVAL_CHAR_BUDGET = 10_000
_ANCHOR_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.#/-]{2,}")


def _anchors(text: str) -> set[str]:
    """Keep file names, APIs and error identifiers as high-value exact terms."""
    return {value.lower() for value in _ANCHOR_PATTERN.findall(text)}


class ConversationRetriever:
    def __init__(self, store: WorkspaceStore | None = None):
        self.store = store or workspace_store

    def retrieve(
        self,
        conversation_id: str,
        query: str,
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        char_budget: int = DEFAULT_RETRIEVAL_CHAR_BUDGET,
        neighbor_window: int = 1,
    ) -> dict:
        """Return relevant chunks plus same-segment neighbors for continuity.

        This is RAG retrieval without a remote service or vector database.  It
        uses BM25-style token weighting, exact identifier boosts, and a small
        recency tie-break.  A later embedding retriever can implement the same
        result contract.
        """
        normalized_query = " ".join(str(query or "").split())
        if not normalized_query:
            return self._result("empty_query")
        if limit <= 0 or char_budget <= 0:
            return self._result("empty_request")

        state = self.store.get_retrieval_index_state(conversation_id)
        if not state or state["status"] != "completed":
            return self._result("index_unavailable")

        chunks = self.store.list_retrieval_chunks_for_conversation(conversation_id)
        if not chunks:
            return self._result("index_empty")

        query_tokens = topic_tokens(normalized_query)
        query_anchors = _anchors(normalized_query)
        if not query_tokens and not query_anchors:
            return self._result("query_has_no_search_terms")

        token_counts = [Counter(topic_tokens(chunk["content"])) for chunk in chunks]
        document_frequency = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())
        average_length = max(
            1,
            sum(sum(counts.values()) for counts in token_counts) / len(token_counts),
        )
        max_ordinal = max(chunk["end_ordinal"] for chunk in chunks) or 1

        ranked = []
        for position, (chunk, counts) in enumerate(zip(chunks, token_counts)):
            document_length = max(1, sum(counts.values()))
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency[token]
                idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
                score += idf * (frequency * 2.2) / (
                    frequency + 1.2 * (0.25 + 0.75 * document_length / average_length)
                )
            content_lower = chunk["content"].lower()
            title_lower = str(chunk.get("segment_title") or "").lower()
            for anchor in query_anchors:
                if anchor in content_lower:
                    score += 2.0
                if anchor in title_lower:
                    score += 1.0
            if score > 0:
                # Only resolves otherwise equal evidence in favor of the newer
                # discussion; it must never outweigh an actual lexical match.
                score += 0.01 * chunk["end_ordinal"] / max_ordinal
                ranked.append((score, position, chunk))

        if not ranked:
            return self._result("no_match")

        ranked.sort(key=lambda item: (-item[0], -item[2]["end_ordinal"], item[2]["chunk_id"]))
        by_segment_and_index = {
            (chunk["segment_id"], chunk["chunk_index"]): (position, chunk)
            for position, chunk in enumerate(chunks)
        }
        selected: list[dict] = []
        selected_ids: set[str] = set()
        used_chars = 0

        def add(chunk: dict, *, score: float, rank: int | None, is_neighbor: bool) -> bool:
            nonlocal used_chars
            if chunk["chunk_id"] in selected_ids:
                return False
            if selected and used_chars + chunk["char_count"] > char_budget:
                return False
            if not selected and chunk["char_count"] > char_budget:
                return False
            selected_ids.add(chunk["chunk_id"])
            used_chars += chunk["char_count"]
            selected.append({
                **chunk,
                "score": round(score, 4),
                "rank": rank,
                "is_neighbor": is_neighbor,
            })
            return True

        primary_count = 0
        for score, _, chunk in ranked:
            if primary_count >= limit:
                break
            if not add(chunk, score=score, rank=primary_count + 1, is_neighbor=False):
                continue
            primary_count += 1
            for offset in range(1, max(0, neighbor_window) + 1):
                for neighbor_index in (chunk["chunk_index"] - offset, chunk["chunk_index"] + offset):
                    neighbor = by_segment_and_index.get((chunk["segment_id"], neighbor_index))
                    if neighbor:
                        add(
                            neighbor[1], score=score, rank=primary_count,
                            is_neighbor=True,
                        )

        selected.sort(key=lambda chunk: (chunk["start_ordinal"], chunk["end_ordinal"], chunk["chunk_index"]))
        return {
            "state": "ok",
            "retriever_version": RETRIEVER_VERSION,
            "query_token_count": len(query_tokens),
            "matched_count": primary_count,
            "selected_count": len(selected),
            "selected_char_count": used_chars,
            "chunks": selected,
        }

    @staticmethod
    def _result(state: str) -> dict:
        return {
            "state": state,
            "retriever_version": RETRIEVER_VERSION,
            "query_token_count": 0,
            "matched_count": 0,
            "selected_count": 0,
            "selected_char_count": 0,
            "chunks": [],
        }


conversation_retriever = ConversationRetriever()
