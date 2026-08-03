import tempfile
import unittest
from pathlib import Path

from work_item_context import WorkItemContextService
from workspace_store import WorkspaceStore


class WorkItemContextServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "demo"
        self.project_root.mkdir()
        self.store = WorkspaceStore(self.root / "workspace.db")
        self.service = WorkItemContextService(self.store, self.root / "contexts")
        self.project = self.store.create_project("Demo", [str(self.project_root)])
        self.item = self.store.create_work_item(
            self.project["project_id"], "实现滚动市盈率", goal="提供 TTM PE 接口与图表"
        )
        self.conversation = self.store.upsert_conversation(
            "codex", "context-session", title="滚动市盈率开发",
            original_project_path=str(self.project_root), resume_capable=True,
        )
        self.store.replace_messages(self.conversation["conversation_id"], [
            {"role": "user", "content": "实现 TTM PE，api_key=very-secret-token-value"},
            {"role": "assistant", "content": "先检查财务数据表和已有测试。"},
        ])
        self.segment = self.store.create_segment(
            self.conversation["conversation_id"], 0, 1, segment_kind="work_item",
            project_id=self.project["project_id"], title="TTM PE 数据源确认",
        )
        self.store.link_segment_work_item(self.segment["segment_id"], self.item["work_item_id"])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_redacts_sensitive_text_and_reuses_unchanged_package(self):
        first = self.service.generate(self.item["work_item_id"])
        package, content = self.service.read_content(first["context_id"])
        second = self.service.generate(self.item["work_item_id"])

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["context_id"], second["context_id"])
        self.assertEqual(package["version"], 1)
        self.assertTrue(Path(package["canonical_path"]).is_file())
        self.assertTrue(Path(first["handoff_path"]).is_file())
        self.assertIn(".work-assistant", first["handoff_path"])
        self.assertIn("# 工作项上下文：实现滚动市盈率", content)
        self.assertIn("[REDACTED:API_KEY]", content)
        self.assertNotIn("very-secret-token-value", content)
        self.assertEqual(package["segment_ids"], [self.segment["segment_id"]])

    def test_changed_source_creates_a_new_context_version(self):
        first = self.service.generate(self.item["work_item_id"])
        self.store.replace_messages(self.conversation["conversation_id"], [
            {"role": "user", "content": "实现 TTM PE 接口"},
            {"role": "assistant", "content": "新增了数据对齐验证步骤。"},
        ])
        second = self.service.generate(self.item["work_item_id"])

        self.assertTrue(first["created"])
        self.assertTrue(second["created"])
        self.assertEqual(second["version"], 2)

    def test_continue_prompt_includes_current_progress_and_redacted_context(self):
        self.store.replace_work_item_steps(self.item["work_item_id"], [
            {"title": "确认财务数据源", "status": "completed"},
            {"title": "实现 TTM PE 计算接口", "status": "in_progress"},
            {"title": "补齐接口测试", "status": "pending"},
        ])

        result = self.service.generate_continue_prompt(self.item["work_item_id"])

        self.assertEqual(result["work_item_id"], self.item["work_item_id"])
        self.assertIn("## 当前任务进度", result["prompt"])
        self.assertIn("[已完成] 确认财务数据源", result["prompt"])
        self.assertIn("[进行中] 实现 TTM PE 计算接口", result["prompt"])
        self.assertIn("## 刚采集的本地代码状态", result["prompt"])
        self.assertIn("## 已压缩的 Codex / Claude 历史上下文", result["prompt"])
        self.assertIn("[需要用户补充：验收标准]", result["prompt"])
        self.assertIn("完整本机上下文附件", result["prompt"])
        self.assertIn("Work Assistant 工作项 ID", result["prompt"])
        self.assertIn("[REDACTED:API_KEY]", result["prompt"])
        self.assertNotIn("very-secret-token-value", result["prompt"])
        self.assertIn("context_package", result)
        self.assertTrue(result["handoff_available"])
        self.assertTrue(Path(result["handoff_path"]).is_file())
        self.assertGreaterEqual(len(result["missing_information"]), 2)

    def test_continue_prompt_does_not_request_already_provided_business_details(self):
        detailed = self.store.create_work_item(
            self.project["project_id"], "带完整验收的功能", item_type="feature",
            goal="让用户按条件筛选报表", metadata={
                "acceptance_criteria": ["筛选条件生效", "空结果有明确反馈"],
                "key_metrics": "筛选响应时间小于 500ms",
            },
        )

        result = self.service.generate_continue_prompt(detailed["work_item_id"])

        self.assertEqual(result["missing_information"], [])
        self.assertIn("业务信息检查通过", result["prompt"])
        self.assertNotIn("[需要用户补充", result["prompt"])

    def test_continue_prompt_rejects_completed_work_item(self):
        completed = self.store.create_work_item(
            self.project["project_id"], "已完成工作", status="completed"
        )

        with self.assertRaisesRegex(Exception, "只有待确认"):
            self.service.generate_continue_prompt(completed["work_item_id"])

    def test_continue_prompt_allows_suggested_work_item_for_scope_review(self):
        suggested = self.store.create_work_item(
            self.project["project_id"], "待确认的需求", status="suggested"
        )

        result = self.service.generate_continue_prompt(suggested["work_item_id"])

        self.assertIn("该工作项仍待确认", result["prompt"])


if __name__ == "__main__":
    unittest.main()
