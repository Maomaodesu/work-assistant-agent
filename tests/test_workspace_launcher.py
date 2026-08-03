import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server
import workspace_launcher
from conversation_manager import ConversationStore
from settings import ConfigurationError, SettingsService
from workspace_launcher import (
    WorkspaceLaunchError,
    launch_ai_with_context,
    launch_project_tool,
    resume_external_session,
)


class WorkspaceLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "demo-project"
        self.project.mkdir()
        (self.project / ".git").mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_settings_persist_only_existing_allowlisted_files(self):
        executable = self.root / "Code.exe"
        executable.write_bytes(b"test")
        service = SettingsService(self.root / "settings.db")
        service.save_tool_settings({
            "preferred_editor": "vscode",
            "tool_paths": {"vscode": str(executable), "unknown": str(executable)},
        })

        loaded = service.load()
        self.assertEqual(loaded.preferred_editor, "vscode")
        self.assertEqual(loaded.tool_paths, {"vscode": str(executable.resolve())})
        with self.assertRaises(ConfigurationError):
            service.save_tool_settings({
                "preferred_editor": "idea",
                "tool_paths": {"idea": str(self.root / "missing.exe")},
            })

    def test_launcher_uses_fixed_action_and_project_as_working_directory(self):
        process = SimpleNamespace(pid=4321)
        popen = MagicMock(return_value=process)
        tools = {
            name: {"available": name == "vscode", "path": str(self.root / f"{name}.exe")}
            for name in workspace_launcher.TOOL_KEYS
        }
        with patch.object(workspace_launcher, "discover_local_tools", return_value=tools):
            result = launch_project_tool(
                str(self.project),
                "default_editor",
                preferred_editor="vscode",
                popen=popen,
            )

        command = popen.call_args.args[0]
        options = popen.call_args.kwargs
        self.assertEqual(command, [tools["vscode"]["path"], str(self.project.resolve())])
        self.assertEqual(options["cwd"], str(self.project.resolve()))
        self.assertFalse(options["shell"])
        self.assertEqual(result["tool_name"], "VS Code")

    def test_launcher_rejects_unknown_action_and_missing_project(self):
        with self.assertRaises(WorkspaceLaunchError):
            launch_project_tool(str(self.project), "run_arbitrary_command")
        with self.assertRaises(WorkspaceLaunchError):
            launch_project_tool(str(self.root / "missing"), "explorer")

    def test_resume_external_session_uses_fixed_cli_arguments(self):
        process = SimpleNamespace(pid=9123)
        popen = MagicMock(return_value=process)
        codex = self.root / "codex.cmd"
        codex.write_text("@echo off", encoding="utf-8")
        tools = {
            name: {"available": name == "codex", "path": str(codex)}
            for name in workspace_launcher.TOOL_KEYS
        }
        with (
            patch.object(workspace_launcher, "discover_local_tools", return_value=tools),
            patch.object(workspace_launcher, "_which", return_value="C:/Windows/System32/cmd.exe"),
        ):
            result = resume_external_session(
                str(self.project), "codex", "019f-session-id", popen=popen
            )

        command = popen.call_args.args[0]
        self.assertEqual(command[-2:], ["resume", "019f-session-id"])
        self.assertEqual(popen.call_args.kwargs["cwd"], str(self.project.resolve()))
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(result["tool_name"], "Codex")
        with self.assertRaises(WorkspaceLaunchError):
            resume_external_session(str(self.project), "codex", "bad & calc", popen=popen)

    def test_context_launch_uses_fixed_cli_and_local_markdown_prompt(self):
        process = SimpleNamespace(pid=7123)
        popen = MagicMock(return_value=process)
        codex = self.root / "codex.cmd"
        codex.write_text("@echo off", encoding="utf-8")
        context = self.root / "context.md"
        context.write_text("# Context", encoding="utf-8")
        tools = {
            name: {"available": name == "codex", "path": str(codex)}
            for name in workspace_launcher.TOOL_KEYS
        }
        with (
            patch.object(workspace_launcher, "discover_local_tools", return_value=tools),
            patch.object(workspace_launcher, "_which", return_value="C:/Windows/System32/cmd.exe"),
        ):
            result = launch_ai_with_context(
                str(self.project), "codex", str(context), popen=popen
            )

        command = popen.call_args.args[0]
        self.assertEqual(command[1:3], ["/d", "/k"])
        self.assertEqual(command[3], str(codex))
        self.assertIn(str(context.resolve()), command[-1])
        self.assertEqual(popen.call_args.kwargs["cwd"], str(self.project.resolve()))
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(result["tool_name"], "Codex")
        with self.assertRaises(WorkspaceLaunchError):
            launch_ai_with_context(str(self.project), "codex", str(self.root / "missing.md"), popen=popen)

    def test_projects_are_aggregated_from_trusted_conversation_metadata(self):
        store = ConversationStore(self.root / "conversations.db")
        store.ensure_work_assistant("session-project", "demo")
        store.append_exchange(
            "session-project",
            "work",
            "done",
            project_path=str(self.project),
        )

        projects = store.projects()
        selected = store.get_project(projects[0]["project_key"])

        self.assertEqual(len(projects), 1)
        self.assertEqual(selected["project_name"], "demo-project")
        self.assertEqual(selected["conversation_count"], 1)
        self.assertTrue(selected["path_available"])

    def test_project_launch_api_resolves_key_without_accepting_browser_path(self):
        store = ConversationStore(self.root / "api-conversations.db")
        store.ensure_work_assistant("session-api-project", "demo")
        store.append_exchange(
            "session-api-project",
            "work",
            "done",
            project_path=str(self.project),
        )
        project = store.projects()[0]
        runtime_settings = SimpleNamespace(
            preferred_editor="auto",
            tool_paths={},
        )
        launch_result = {
            "action": "terminal",
            "tool_name": "终端",
            "project_path": str(self.project.resolve()),
            "pid": 123,
        }
        with (
            patch.object(server, "conversation_store", store),
            patch.object(server, "get_settings", return_value=runtime_settings),
            patch.object(server, "launch_project_tool", return_value=launch_result) as launcher,
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                response = client.post("/api/projects/launch", json={
                    "project_key": project["project_key"],
                    "action": "terminal",
                    "project_path": "C:/browser-must-not-control-this",
                })

        self.assertEqual(response.status_code, 200)
        launcher.assert_called_once_with(
            str(self.project.resolve()),
            "terminal",
            preferred_editor="auto",
            configured_paths={},
        )

    def test_workspace_and_settings_expose_workspace_controls(self):
        workspace = Path("templates/workspace.html").read_text(encoding="utf-8")
        setup = Path("templates/setup.html").read_text(encoding="utf-8")
        self.assertIn("launchWorkspaceProject", workspace)
        self.assertIn("资源管理器", workspace)
        self.assertIn("终端", workspace)
        self.assertIn("默认编辑器", workspace)
        self.assertIn("编辑器与 AI 工具", setup)
        self.assertIn("discoverLocalTools", setup)


if __name__ == "__main__":
    unittest.main()
