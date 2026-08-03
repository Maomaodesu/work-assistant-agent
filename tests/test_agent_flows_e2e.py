import dataclasses
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

import agent_graph
import server
import task_manager
from conversation_manager import ConversationStore
from session_store import create_sqlite_checkpointer
from task_manager import PlanStep, Task


@dataclasses.dataclass
class FakeSnapshot:
    snapshot_id: str
    task_id: str
    interrupt_time: str
    interrupt_reason: str
    projects: list
    work_sessions: list
    total_work_hours: float
    idea_changed_files: list
    progress: dict | None
    terminal_history: list
    dev_processes: list
    current_thought: str
    blockers: list
    next_actions: list


class FakeWorkflowLLM:
    """Deterministic route/gather responses for API-level flow tests."""

    def invoke(self, messages):
        system_text = messages[0].content if messages else ""
        human_texts = [
            message.content
            for message in messages
            if isinstance(message, HumanMessage)
        ]
        latest_human = human_texts[-1] if human_texts else ""
        all_human = "\n".join(human_texts)

        if "意图识别器" in system_text:
            lowered = latest_human.lower()
            if "new task" in lowered:
                return SimpleNamespace(content="new_task")
            if "check project" in lowered:
                return SimpleNamespace(content="check")
            if "progress" in lowered:
                return SimpleNamespace(content="progress")
            return SimpleNamespace(content="chat")

        if "初始化任务信息" in system_text:
            if "C:/demo" not in all_human:
                return SimpleNamespace(
                    content='{"ready": false, "question": "Please provide the project path."}'
                )
            if "check project" in all_human.lower():
                return SimpleNamespace(
                    content=(
                        '{"ready": true, "project_paths": ["C:/demo"], '
                        '"description": "", "context": "existing demo project"}'
                    )
                )
            return SimpleNamespace(
                content=(
                    '{"ready": true, "project_paths": ["C:/demo"], '
                    '"description": "build demo feature", "context": ""}'
                )
            )

        raise AssertionError(f"Unexpected LLM prompt: {system_text[:80]}")


class AgentFlowE2ETests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            task_manager,
            "DB_PATH",
            Path(self.temp_dir.name) / "agent.db",
        )
        self.db_patch.start()
        task_manager.init_db()

        self.patches = [
            patch.object(agent_graph, "get_llm", return_value=FakeWorkflowLLM()),
            patch.object(
                agent_graph,
                "create_task_via_llm",
                side_effect=self._create_new_task,
            ),
            patch.object(
                agent_graph,
                "init_existing_project_via_llm",
                side_effect=self._init_existing_task,
            ),
            patch.object(
                agent_graph,
                "collect_snapshot",
                side_effect=self._collect_snapshot,
            ),
            patch.object(
                agent_graph,
                "save_snapshot",
                return_value=str(Path(self.temp_dir.name) / "snapshot.json"),
            ),
            patch.object(
                agent_graph,
                "analyze_and_persist_progress",
                side_effect=self._analyze_and_persist,
            ),
        ]
        for active_patch in self.patches:
            active_patch.start()

        self.checkpointer, self.checkpoint_conn = create_sqlite_checkpointer(
            Path(self.temp_dir.name) / "checkpoints.db"
        )
        self.test_graph = agent_graph.build_graph_v2(checkpointer=self.checkpointer)
        self.server_graph_patch = patch.object(server, "agent_app", self.test_graph)
        self.server_graph_patch.start()
        self.conversation_store_patch = patch.object(
            server,
            "conversation_store",
            ConversationStore(Path(self.temp_dir.name) / "conversations.db"),
        )
        self.conversation_store_patch.start()
        self.setup_gate_patch = patch.object(
            server.settings_service,
            "is_setup_complete",
            return_value=True,
        )
        self.setup_gate_patch.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close()
        self.conversation_store_patch.stop()
        self.setup_gate_patch.stop()
        self.server_graph_patch.stop()
        self.checkpoint_conn.close()
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _make_task(task_id: str, task_name: str, project_paths: list) -> Task:
        now = datetime.now().isoformat()
        return Task(
            task_id=task_id,
            task_name=task_name,
            project_paths=project_paths,
            project_types=["python"],
            goal=f"Goal for {task_name}",
            tech_stack={"backend": "Python"},
            status="active",
            priority="P1",
            created_at=now,
            last_active_at=now,
            total_work_seconds=0,
            interrupt_count=0,
            plan=[
                PlanStep(f"{task_id}-STEP-01", 0, "Foundation", "Create foundation"),
                PlanStep(f"{task_id}-STEP-02", 1, "Delivery", "Finish delivery"),
            ],
            current_step_index=0,
        )

    def _create_new_task(self, description, project_paths):
        task = self._make_task("TASK-20990101-000001", "New demo task", project_paths)
        task_manager.save_task(task)
        return task

    def _init_existing_task(self, conversation_context, project_paths, snapshot):
        task = self._make_task(
            "TASK-20990101-000002",
            "Existing demo task",
            project_paths,
        )
        task_manager.save_task(task)
        return task

    @staticmethod
    def _collect_snapshot(task_id, project_paths, idea_project_path, interrupt_reason):
        return FakeSnapshot(
            snapshot_id=f"SNAP-{task_id}",
            task_id=task_id,
            interrupt_time=datetime.now().isoformat(),
            interrupt_reason=interrupt_reason,
            projects=[],
            work_sessions=[],
            total_work_hours=0,
            idea_changed_files=[],
            progress=None,
            terminal_history=[],
            dev_processes=[],
            current_thought="",
            blockers=[],
            next_actions=[],
        )

    @staticmethod
    def _analyze_and_persist(task, snapshot):
        report = {
            "current_step_index": 1,
            "current_step_name": "Delivery",
            "completion_percent": 75,
            "step_statuses": [
                {
                    "step_index": 0,
                    "step_name": "Foundation",
                    "status": "completed",
                    "evidence": "foundation exists",
                },
                {
                    "step_index": 1,
                    "step_name": "Delivery",
                    "status": "in_progress",
                    "evidence": "delivery is in progress",
                },
            ],
            "summary": "The workflow is active.",
            "next_action": "Finish delivery.",
            "risks": [],
        }
        return task_manager.apply_progress_report(task.task_id, report)

    def _chat(self, message: str, session_id: str) -> tuple[str, str]:
        response = self.client.post(
            "/api/chat",
            json={"message": message, "session_id": session_id},
        )
        self.assertEqual(response.status_code, 200)

        message_lines = []
        event_types = []
        for block in response.text.split("\n\n"):
            event_type = ""
            data_lines = []
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_type = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    data_lines.append(line.removeprefix("data: "))
            if event_type:
                event_types.append(event_type)
            if event_type == "message":
                message_lines.append("\n".join(data_lines))
        return "\n".join(message_lines), ",".join(event_types)

    def test_new_task_multi_turn_flow_creates_and_persists_task(self):
        first_output, first_events = self._chat(
            "new task: build a demo feature",
            "session-new-task",
        )
        self.assertIn("Please provide the project path", first_output)
        self.assertIn("done", first_events)

        final_output, final_events = self._chat(
            "project path is C:/demo",
            "session-new-task",
        )
        task = task_manager.load_task("TASK-20990101-000001")
        conversation = server.conversation_store.get("session-new-task")

        self.assertIn("New demo task", final_output)
        self.assertIn("75%", final_output)
        self.assertIn("done", final_events)
        self.assertIsNotNone(task)
        self.assertEqual([step.status for step in task.plan], ["completed", "in_progress"])
        self.assertEqual(task.current_step_index, 1)
        self.assertEqual(conversation["group_type"], "project")
        self.assertEqual(conversation["project_name"], "demo")

        saved_state = self.test_graph.get_state(
            {"configurable": {"thread_id": "session-new-task"}}
        ).values
        self.assertEqual(saved_state["intent"], "new_task")
        self.assertGreaterEqual(len(saved_state["messages"]), 4)

    def test_check_flow_initializes_existing_project_and_reports_progress(self):
        output, events = self._chat(
            "check project C:/demo with existing work",
            "session-check",
        )
        task = task_manager.load_task("TASK-20990101-000002")

        self.assertIn("Existing demo task", output)
        self.assertIn("75%", output)
        self.assertIn("done", events)
        self.assertIsNotNone(task)
        self.assertEqual(task.project_paths, ["C:/demo"])
        self.assertEqual([step.status for step in task.plan], ["completed", "in_progress"])

    def test_progress_flow_handles_unknown_then_loads_existing_task(self):
        existing = self._make_task(
            "TASK-20990101-000003",
            "Progress demo task",
            ["C:/demo"],
        )
        task_manager.save_task(existing)

        missing_output, _ = self._chat(
            "progress TASK-20990101-999999",
            "session-progress",
        )
        self.assertIn("TASK-20990101-000003", missing_output)
        self.assertIn("请输入要查看的 task_id", missing_output)

        final_output, final_events = self._chat(
            "progress TASK-20990101-000003",
            "session-progress",
        )
        reloaded = task_manager.load_task("TASK-20990101-000003")

        self.assertIn("Progress demo task", final_output)
        self.assertIn("75%", final_output)
        self.assertIn("done", final_events)
        self.assertEqual(
            [step.status for step in reloaded.plan],
            ["completed", "in_progress"],
        )


if __name__ == "__main__":
    unittest.main()
