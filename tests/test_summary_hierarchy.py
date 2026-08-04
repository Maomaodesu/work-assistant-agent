import threading
import unittest

from summary_hierarchy import (
    FINAL_INPUT_TOKEN_BUDGET,
    HierarchicalSummaryBuilder,
    TokenBudget,
    estimate_tokens,
)
from streaming_runtime import GenerationCancelled


class SummaryHierarchyTests(unittest.TestCase):
    def test_conservative_token_budget_rejects_overflow(self):
        budget = TokenBudget(100)
        self.assertTrue(budget.try_add("认证接口" * 10))
        self.assertFalse(budget.try_add("x" * 1_000))
        self.assertGreater(estimate_tokens("中文 English /api/auth.py"), 0)

    def test_map_reduce_uses_all_source_groups_and_bounds_final_prompt(self):
        calls = []

        def generator(system, prompt, max_output_tokens, cancel_event):
            calls.append((system, prompt, max_output_tokens))
            return "证据摘要：登录、验证、遗留项。" * 35

        chunks = [{
            "chunk_id": f"RCH-{index}",
            "segment_id": f"SEG-{index // 3}",
            "start_ordinal": index,
            "end_ordinal": index,
            "content": "原始会话证据。" * 180,
        } for index in range(210)]
        builder = HierarchicalSummaryBuilder(generator=generator)
        prompt, details = builder.build_final_prompt(
            chunks=chunks,
            final_prefix="最终工作回顾：\n",
            final_suffix="\n请输出结论。",
            cancel_event=threading.Event(),
        )

        self.assertGreater(details["map_call_count"], 1)
        self.assertGreaterEqual(details["reduction_rounds"], 1)
        self.assertLessEqual(details["final_estimated_input_tokens"], FINAL_INPUT_TOKEN_BUDGET)
        self.assertLessEqual(estimate_tokens(prompt), FINAL_INPUT_TOKEN_BUDGET)
        self.assertTrue(calls)

    def test_cancelled_build_stops_before_model_call(self):
        called = False

        def generator(system, prompt, max_output_tokens, cancel_event):
            nonlocal called
            called = True
            return "不应生成"

        cancel_event = threading.Event()
        cancel_event.set()
        builder = HierarchicalSummaryBuilder(generator=generator)
        with self.assertRaises(GenerationCancelled) as raised:
            builder.build_final_prompt(
                chunks=[{
                    "chunk_id": "RCH-1", "segment_id": "SEG-1",
                    "start_ordinal": 0, "end_ordinal": 0, "content": "原文",
                }],
                final_prefix="前缀", final_suffix="后缀", cancel_event=cancel_event,
            )
        self.assertIn("停止", str(raised.exception))
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
