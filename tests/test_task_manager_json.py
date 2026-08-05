import unittest

from task_manager import _parse_task_json


class TaskManagerJsonTests(unittest.TestCase):
    def test_parses_json_inside_markdown_and_explanation(self):
        parsed = _parse_task_json(
            "我会按下面的计划执行：\n```json\n"
            '{"task_name":"测试任务","plan":[]}\n```'
        )

        self.assertEqual(parsed["task_name"], "测试任务")
        self.assertEqual(parsed["plan"], [])

    def test_rejects_empty_model_response_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, "没有返回任务计划内容"):
            _parse_task_json("")


if __name__ == "__main__":
    unittest.main()
