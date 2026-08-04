"""Bounded local retrieval of likely work-item matches for a conversation segment."""

from __future__ import annotations

import json
import re

from semantic_segmenter import topic_tokens


WORK_ITEM_RETRIEVER_VERSION = "lexical-work-item-v1"
DEFAULT_CANDIDATE_LIMIT = 12
DEFAULT_CANDIDATE_CHAR_BUDGET = 8_000
RECENT_FALLBACK_LIMIT = 3
MAX_GOAL_CHARS = 500
MAX_DESCRIPTION_CHARS = 500

_ANCHOR_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.#/-]{2,}")
_ACTIVE_STATUSES = {"suggested", "backlog", "planned", "in_progress", "blocked"}


def _anchors(text: str) -> set[str]:
    return {value.lower() for value in _ANCHOR_PATTERN.findall(str(text or ""))}


def _compact(value: str, limit: int) -> str:
    value = str(value or "").strip()
    return value if len(value) <= limit else value[:limit] + "…"


def _candidate_payload(item: dict, *, score: float, reason: str) -> dict:
    return {
        "work_item_id": item["work_item_id"],
        "title": _compact(item.get("title", ""), 160),
        "type": item.get("item_type", "other"),
        "status": item.get("status", ""),
        "goal": _compact(item.get("goal", ""), MAX_GOAL_CHARS),
        "description": _compact(item.get("description", ""), MAX_DESCRIPTION_CHARS),
        "retrieval_reason": reason,
        "retrieval_score": round(score, 4),
    }


class WorkItemRetriever:
    """Rank project work items with local lexical evidence and a hard budget."""

    def retrieve(
        self,
        query: str,
        items: list[dict],
        *,
        limit: int = DEFAULT_CANDIDATE_LIMIT,
        char_budget: int = DEFAULT_CANDIDATE_CHAR_BUDGET,
        recent_fallback_limit: int = RECENT_FALLBACK_LIMIT,
    ) -> dict:
        if limit <= 0 or char_budget <= 0:
            return self._result("empty_request")
        normalized_query = " ".join(str(query or "").split())
        query_tokens = topic_tokens(normalized_query)
        query_anchors = _anchors(normalized_query)

        scored = []
        for position, item in enumerate(items):
            title = str(item.get("title") or "")
            goal = str(item.get("goal") or "")
            description = str(item.get("description") or "")
            title_tokens = topic_tokens(title)
            goal_tokens = topic_tokens(goal)
            description_tokens = topic_tokens(description)
            score = (
                3.0 * len(query_tokens & title_tokens)
                + 1.8 * len(query_tokens & goal_tokens)
                + 1.2 * len(query_tokens & description_tokens)
            )
            title_lower = title.lower()
            combined_lower = f"{title}\n{goal}\n{description}".lower()
            anchor_hits = 0
            for anchor in query_anchors:
                if anchor in combined_lower:
                    score += 2.0
                    anchor_hits += 1
                if anchor in title_lower:
                    score += 1.0
            if item.get("status") in _ACTIVE_STATUSES:
                score += 0.15
            if score > 0:
                reason = "keyword_anchor" if anchor_hits else "keyword"
                scored.append((score, position, item, reason))

        # Work-item rows already arrive in last-active order.  Add a small
        # active/recent fallback so a short, vague follow-up can still match a
        # task being worked on, without allowing the entire project into the
        # model prompt.
        selected: list[dict] = []
        selected_ids: set[str] = set()
        used_chars = 0

        def add(item: dict, score: float, reason: str) -> bool:
            nonlocal used_chars
            item_id = str(item.get("work_item_id") or "")
            if not item_id or item_id in selected_ids:
                return False
            payload = _candidate_payload(item, score=score, reason=reason)
            payload_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            if selected and used_chars + payload_chars > char_budget:
                return False
            if not selected and payload_chars > char_budget:
                # Keep an actionable candidate even under an unusually small
                # caller budget, but shrink verbose fields rather than exceed it.
                payload["goal"] = _compact(payload["goal"], max(0, char_budget // 4))
                payload["description"] = ""
                payload_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                if payload_chars > char_budget:
                    return False
            selected_ids.add(item_id)
            used_chars += payload_chars
            selected.append(payload)
            return True

        scored.sort(key=lambda value: (-value[0], value[1], str(value[2].get("work_item_id", ""))))
        for score, _, item, reason in scored:
            if len(selected) >= limit:
                break
            add(item, score, reason)

        fallback_count = 0
        for item in items:
            if len(selected) >= limit or fallback_count >= recent_fallback_limit:
                break
            if item.get("status") not in _ACTIVE_STATUSES:
                continue
            if add(item, 0.0, "recent_active"):
                fallback_count += 1

        return {
            "state": "ok" if selected else "empty",
            "retriever_version": WORK_ITEM_RETRIEVER_VERSION,
            "query_token_count": len(query_tokens),
            "query_anchor_count": len(query_anchors),
            "candidate_count": len(selected),
            "candidate_char_count": used_chars,
            "candidates": selected,
        }

    @staticmethod
    def _result(state: str) -> dict:
        return {
            "state": state,
            "retriever_version": WORK_ITEM_RETRIEVER_VERSION,
            "query_token_count": 0,
            "query_anchor_count": 0,
            "candidate_count": 0,
            "candidate_char_count": 0,
            "candidates": [],
        }


work_item_retriever = WorkItemRetriever()
