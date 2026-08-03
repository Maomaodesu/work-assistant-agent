import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import progress_analyzer
import task_manager
from task_manager import PlanStep, Task


class ProgressPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = task_manager.DB_PATH
        task_manager.DB_PATH = Path(self.temp_dir.name) / "agent.db"
        task_manager.init_db()

        self.task = Task(
            task_id="TASK-TEST",
            task_name="持久化测试",
            project_paths=["C:/workspace/demo"],
            project_types=["python"],
            goal="验证进度报告持久化",
            tech_stack={"backend": "Python"},
            status="active",
            priority="P0",
            created_at="2026-07-18T00:00:00",
            last_active_at="2026-07-18T00:00:00",
            total_work_seconds=0,
            interrupt_count=0,
            plan=[
                PlanStep("STEP-1", 0, "第一步", "完成基础工作"),
                PlanStep("STEP-2", 1, "第二步", "继续实现"),
                PlanStep("STEP-3", 2, "第三步", "最终验证"),
            ],
            current_step_index=0,
        )
        task_manager.save_task(self.task)

    def tearDown(self):
        task_manager.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_report_is_persisted_and_normalized(self):
        report = {
            "current_step_index": 2,
            "current_step_name": "错误的模型判断",
            "completion_percent": 87,
            "step_statuses": [
                {
                    "step_index": 0,
                    "step_name": "第一步",
                    "status": "completed",
                    "evidence": "基础代码已经存在",
                },
                {
                    "step_index": 1,
                    "step_name": "第二步",
                    "status": "in_progress",
                    "evidence": "核心文件正在修改",
                },
                {
                    "step_index": 2,
                    "step_name": "第三步",
                    "status": "pending",
                    "evidence": "尚未发现测试结果",
                },
            ],
            "summary": "正在进行第二步",
            "next_action": "完成第二步",
            "risks": [],
        }

        normalized = task_manager.apply_progress_report("TASK-TEST", report)
        reloaded = task_manager.load_task("TASK-TEST")

        self.assertEqual(normalized["completion_percent"], 50)
        self.assertEqual(normalized["current_step_index"], 1)
        self.assertEqual(normalized["current_step_name"], "第二步")
        self.assertEqual(
            [step.status for step in reloaded.plan],
            ["completed", "in_progress", "pending"],
        )
        self.assertEqual(reloaded.current_step_index, 1)
        self.assertEqual(reloaded.plan[0].notes, "基础代码已经存在")
        self.assertIsNotNone(reloaded.plan[0].completed_at)
        self.assertIsNotNone(reloaded.plan[1].started_at)
        self.assertGreater(reloaded.last_active_at, "2026-07-18T00:00:00")

    def test_completed_report_marks_task_completed_after_reload(self):
        report = {
            "completion_percent": 100,
            "step_statuses": [
                {"step_index": 0, "status": "completed"},
                {"step_index": 1, "status": "completed"},
                {"step_index": 2, "status": "skipped"},
            ],
        }

        normalized = task_manager.apply_progress_report("TASK-TEST", report)
        reloaded = task_manager.load_task("TASK-TEST")

        self.assertEqual(normalized["completion_percent"], 100)
        self.assertEqual(normalized["current_step_index"], 3)
        self.assertEqual(normalized["current_step_name"], "全部完成")
        self.assertEqual(reloaded.status, "completed")
        self.assertEqual(reloaded.current_step_index, 3)

    def test_analysis_entrypoint_persists_without_network(self):
        llm_report = {
            "completion_percent": 10,
            "step_statuses": [
                {"step_index": 0, "status": "completed", "evidence": "已完成"},
                {"step_index": 1, "status": "pending", "evidence": "未开始"},
                {"step_index": 2, "status": "pending", "evidence": "未开始"},
            ],
        }

        with patch.object(progress_analyzer, "analyze_progress", return_value=llm_report):
            normalized = progress_analyzer.analyze_and_persist_progress(
                self.task,
                {"projects": []},
            )

        # load_task 每次都会新建 SQLite 连接，等价验证服务重启后的磁盘读取。
        reloaded = task_manager.load_task("TASK-TEST")
        self.assertEqual(normalized["completion_percent"], 33)
        self.assertEqual(reloaded.plan[0].status, "completed")
        self.assertEqual(reloaded.current_step_index, 1)


if __name__ == "__main__":
    unittest.main()
