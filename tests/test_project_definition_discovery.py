import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from project_definition_discovery import ProjectDefinitionDiscovery
from workspace_store import WorkspaceStore


class FakeDefinitionClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({
                "summary": "项目记忆层",
                "goal": "将 Codex 讨论沉淀为可确认的项目定义",
                "scope": "关联会话、生成草稿和工作项",
                "non_goals": "不代替 Codex 编程",
                "acceptance_criteria": "用户能确认并在后续查看定义",
                "constraints": "仅使用已关联本机会话",
            }, ensure_ascii=False)
        ))])


class ProjectDefinitionDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        project_root = root / "demo"
        project_root.mkdir()
        self.store = WorkspaceStore(root / "workspace.db")
        self.project = self.store.create_project("Demo", [str(project_root)])
        conversation = self.store.upsert_conversation(
            "codex", "definition-session", original_project_path=str(project_root)
        )
        self.store.set_manual_conversation_project(
            conversation["conversation_id"], self.project["project_id"]
        )
        self.store.replace_messages(conversation["conversation_id"], [
            {"ordinal": 0, "role": "user", "content": "我们要让 Work Assistant 记录 Codex 项目讨论。api_key=secret-value-123"},
            {"ordinal": 1, "role": "assistant", "content": "首期只做会话关联、定义草稿和工作项，不替代编程 Agent。"},
        ])
        self.segment = self.store.create_segment(
            conversation["conversation_id"], 0, 1, project_id=self.project["project_id"],
            segment_kind="project", title="项目定位", summary="讨论项目目标与范围",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discovers_draft_from_project_discussion_and_preserves_confirmed_definition(self):
        client = FakeDefinitionClient()
        discovery = ProjectDefinitionDiscovery(self.store, lambda: client)

        first = discovery.discover(project_id=self.project["project_id"], run_id="RUN-1")
        definition = self.store.get_project_definition(self.project["project_id"])

        self.assertEqual(first["created"], 1)
        self.assertEqual(definition["status"], "draft")
        self.assertEqual(definition["source_segment_ids"], [self.segment["segment_id"]])
        self.assertIn("Codex 讨论", definition["goal"])
        self.assertNotIn("secret-value-123", client.calls[0]["messages"][1]["content"])

        unchanged_draft = discovery.discover(project_id=self.project["project_id"], run_id="RUN-2")
        self.assertEqual(unchanged_draft["skipped"], 1)
        self.assertEqual(len(client.calls), 1)

        self.store.confirm_project_definition(self.project["project_id"])
        second = discovery.discover(project_id=self.project["project_id"], run_id="RUN-3")

        self.assertEqual(second["skipped"], 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            self.store.get_project_definition(self.project["project_id"])["status"], "confirmed"
        )


if __name__ == "__main__":
    unittest.main()
