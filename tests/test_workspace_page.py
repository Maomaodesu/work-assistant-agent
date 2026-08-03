import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from workspace_store import WorkspaceStore


class WorkspacePageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend = self.root / "backend"
        self.frontend = self.root / "frontend"
        self.backend.mkdir()
        self.frontend.mkdir()
        self.store = WorkspaceStore(self.root / "workspace.db")
        self.setup_patch = patch.object(
            server.settings_service, "is_setup_complete", return_value=True
        )
        self.store_patch = patch.object(server, "workspace_store", self.store)
        self.setup_patch.start()
        self.store_patch.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close()
        self.store_patch.stop()
        self.setup_patch.stop()
        self.temp_dir.cleanup()

    def test_workspace_page_explains_project_and_work_item_levels(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("一个项目可以包含多个代码目录", response.text)
        self.assertIn("项目级信息只展示工作项统计", response.text)
        self.assertIn("待确认", response.text)
        self.assertIn("已忽略", response.text)
        self.assertIn("新建工作项", response.text)
        self.assertIn("会话工作项发现", response.text)
        self.assertIn("开始增量分析", response.text)
        self.assertIn("style.css?v=workspace-source-review-20260802", response.text)
        self.assertIn("已请求暂停", response.text)
        self.assertIn("增量重试", response.text)
        self.assertIn("项目级讨论", response.text)
        self.assertIn("查看来源片段", response.text)
        self.assertIn("合并选中项", response.text)
        self.assertIn("copyProjectDirectories", response.text)
        self.assertIn("saveProjectDirectories", response.text)
        self.assertIn("Agent 项目目录", response.text)
        self.assertIn("workspace-analysis-technical", response.text)
        self.assertIn("copyAnalysisRunId", response.text)
        self.assertIn("setWorkItemFilter", response.text)
        self.assertIn("查看 ${suggestedCount} 个待确认工作项", response.text)
        self.assertIn("toggleProjectOpenMenu", response.text)
        self.assertIn("在资源管理器中显示", response.text)
        self.assertNotIn("resumeConversation", response.text)
        self.assertIn("上下文包", response.text)
        self.assertNotIn("launchContextPackage", response.text)

    def test_overview_aggregates_counts_without_project_percentage(self):
        project = self.store.create_project(
            "Multi root",
            [str(self.backend), str(self.frontend)],
        )
        self.store.create_work_item(project["project_id"], "Manual item")
        suggested = self.store.create_work_item(
            project["project_id"], "Detected item", source="inferred", confidence=0.9
        )
        self.store.ignore_work_item(suggested["work_item_id"])

        response = self.client.get("/api/workspace/overview")
        overview = response.json()[0]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(overview["work_item_total"], 2)
        self.assertEqual(overview["active_work_item_count"], 1)
        self.assertEqual(overview["work_item_counts"]["ignored"], 1)
        self.assertNotIn("completion_percent", overview)

    def test_page_api_creates_manual_work_item_as_backlog_without_percent(self):
        project = self.client.post("/api/workspace/projects", json={
            "name": "API project",
            "root_paths": [str(self.backend), str(self.frontend)],
        }).json()
        response = self.client.post("/api/workspace/work-items", json={
            "project_id": project["project_id"],
            "title": "New feature",
            "item_type": "feature",
            "goal": "Deliver a concrete feature",
        })
        item = response.json()

        self.assertEqual(response.status_code, 201)
        self.assertEqual(item["status"], "backlog")
        self.assertIsNone(item["completion_percent"])

    def test_project_directories_can_be_replaced_by_user_configuration(self):
        project = self.store.create_project(
            "Configured roots", [str(self.backend), str(self.frontend)]
        )

        response = self.client.put(
            f"/api/workspace/projects/{project['project_id']}/roots",
            json={"root_paths": [str(self.frontend), str(self.backend)]},
        )

        self.assertEqual(response.status_code, 200)
        roots = response.json()["roots"]
        self.assertEqual([item["path"] for item in roots], [str(self.frontend), str(self.backend)])
        self.assertTrue(roots[0]["is_primary"])
        self.assertFalse(roots[1]["is_primary"])

    def test_context_package_api_generates_reads_and_reuses_local_package(self):
        project = self.store.create_project("Context API", [str(self.backend)])
        item = self.store.create_work_item(project["project_id"], "Context work item")

        first = self.client.post(
            f"/api/workspace/work-items/{item['work_item_id']}/context-packages"
        )
        package = first.json()["package"]
        content = self.client.get(
            f"/api/workspace/context-packages/{package['context_id']}/content"
        )
        second = self.client.post(
            f"/api/workspace/work-items/{item['work_item_id']}/context-packages"
        )

        self.assertEqual(first.status_code, 200)
        self.assertTrue(package["created"])
        self.assertEqual(content.status_code, 200)
        self.assertIn("# 工作项上下文：Context work item", content.json()["content"])
        self.assertEqual(second.status_code, 200)
        self.assertFalse(second.json()["package"]["created"])
        self.assertEqual(second.json()["package"]["context_id"], package["context_id"])

    def test_continue_prompt_api_is_available_for_open_work_items(self):
        project = self.store.create_project("Continue API", [str(self.backend)])
        item = self.store.create_work_item(
            project["project_id"], "继续实现 API", goal="完成接口与验证"
        )
        self.store.replace_work_item_steps(item["work_item_id"], [
            {"title": "实现接口", "status": "in_progress"},
        ])

        response = self.client.post(
            f"/api/workspace/work-items/{item['work_item_id']}/continue-prompt"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("继续开发：继续实现 API", payload["prompt"])
        self.assertIn("计划完成度", payload["prompt"])
        self.assertIn("需要用户补充", payload["prompt"])
        self.assertIn("missing_information", payload)

    def test_workspace_page_has_continue_prompt_action(self):
        template = Path("templates/workspace.html").read_text(encoding="utf-8")
        self.assertIn("继续开发提示词", template)
        self.assertIn("/continue-prompt", template)
        self.assertIn("copyContinuePrompt", template)
        self.assertIn("missing_information", template)

    def test_workspace_page_exports_prompt_and_groups_source_message_types(self):
        template = Path("templates/workspace.html").read_text(encoding="utf-8")
        stylesheet = Path("static/style.css").read_text(encoding="utf-8")
        self.assertIn("导出提示词", template)
        self.assertNotIn("继续工作（复制到 ChatGPT）", template)
        self.assertIn("renderAssistantReply", template)
        self.assertIn("人类输入", template)
        self.assertIn("AI 执行的命令", template)
        self.assertIn("AI 回复中的代码", template)
        self.assertIn("AI 执行的操作", template)
        self.assertIn("AI 操作结果", template)
        self.assertIn("width: min(80vw, 1520px)", stylesheet)
        self.assertIn(".workspace-source-message.role-human", stylesheet)
        self.assertIn(".workspace-source-message.role-ai", stylesheet)
        self.assertIn(".workspace-source-code", stylesheet)
        self.assertIn(".workspace-source-command", stylesheet)
        self.assertIn(".workspace-source-tool-result", stylesheet)

    def test_narrow_workspace_layout_preserves_project_actions_and_analysis(self):
        stylesheet = Path("static/style.css").read_text(encoding="utf-8")
        self.assertIn("@media (max-width: 900px)", stylesheet)
        self.assertIn(
            ".workspace-project-open-options { right: auto; left: 0; }", stylesheet
        )
        self.assertIn(".workspace-tabs { width: 100%; }", stylesheet)
        self.assertIn(
            ".workspace-analysis-stats { grid-template-columns: 1fr; }", stylesheet
        )

    def test_workspace_page_has_complete_and_reopen_actions(self):
        template = Path("templates/workspace.html").read_text(encoding="utf-8")
        self.assertIn("标记完成", template)
        self.assertIn("恢复为进行中", template)
        self.assertIn("completionNote", template)
        self.assertIn("acceptanceResult", template)
        self.assertIn("/complete", template)
        self.assertIn("/reopen", template)

    def test_manual_project_rejects_missing_local_directory(self):
        response = self.client.post("/api/workspace/projects", json={
            "name": "Broken project",
            "root_paths": [str(self.root / "missing")],
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("项目目录不存在", response.json()["error"])

    def test_navigation_uses_workspace_as_home_and_keeps_conversations_separate(self):
        index = Path("templates/index.html").read_text(encoding="utf-8")
        workspace = Path("templates/workspace.html").read_text(encoding="utf-8")
        settings = Path("templates/setup.html").read_text(encoding="utf-8")
        self.assertIn('href="/" class="nav-link active">工作台</a>', workspace)
        self.assertIn('href="/conversations" class="nav-link active">会话</a>', index)
        self.assertIn('href="/conversations" class="nav-link">会话</a>', settings)
        self.assertIn("syncAllConversations", workspace)
        self.assertIn("context-packages", workspace)

    def test_review_api_sources_merge_discussions_and_resume(self):
        project = self.store.create_project("Review", [str(self.backend)])
        conversation = self.store.upsert_conversation(
            "codex", "019f-review", title="Review session",
            original_project_path=str(self.backend), resume_capable=True,
        )
        self.store.replace_messages(conversation["conversation_id"], [
            {"ordinal": 0, "role": "user", "content": "讨论项目架构"},
            {"ordinal": 1, "role": "assistant", "content": "架构建议"},
            {"ordinal": 2, "role": "user", "content": "实现资产负债表"},
            {"ordinal": 3, "role": "assistant", "content": "已经实现"},
        ])
        discussion = self.store.create_segment(
            conversation["conversation_id"], 0, 1,
            segment_kind="project", project_id=project["project_id"], title="架构讨论",
        )
        source_segment = self.store.create_segment(
            conversation["conversation_id"], 2, 3,
            segment_kind="work_item", project_id=project["project_id"], title="资产负债表",
        )
        target = self.store.create_work_item(
            project["project_id"], "资产负债表", source="inferred"
        )
        duplicate = self.store.create_work_item(
            project["project_id"], "获取资产负债表", source="inferred"
        )
        self.store.link_segment_work_item(source_segment["segment_id"], duplicate["work_item_id"])

        sources = self.client.get(
            f"/api/workspace/work-items/{duplicate['work_item_id']}/sources"
        )
        discussions = self.client.get(
            f"/api/workspace/projects/{project['project_id']}/segments?segment_kind=project"
        )
        merged = self.client.post(
            f"/api/workspace/work-items/{target['work_item_id']}/merge",
            json={"source_work_item_ids": [duplicate["work_item_id"]]},
        )
        launch_result = {
            "source": "codex", "tool_name": "Codex",
            "external_session_id": "019f-review", "project_path": str(self.backend), "pid": 1,
        }
        with (
            patch.object(server, "resume_external_session", return_value=launch_result) as resume,
            patch.object(server, "get_settings", return_value=SimpleNamespace(tool_paths={})),
        ):
            resumed = self.client.post(
                f"/api/workspace/conversations/{conversation['conversation_id']}/resume"
            )

        self.assertEqual(sources.status_code, 200)
        self.assertEqual(sources.json()[0]["messages"][0]["ordinal"], 2)
        self.assertEqual(discussions.json()[0]["segment_id"], discussion["segment_id"])
        self.assertTrue(merged.json()["ok"])
        self.assertEqual(
            self.store.get_work_item(duplicate["work_item_id"])["status"], "archived"
        )
        self.assertTrue(resumed.json()["ok"])
        resume.assert_called_once_with(
            str(self.backend), "codex", "019f-review", configured_paths={}
        )


if __name__ == "__main__":
    unittest.main()
