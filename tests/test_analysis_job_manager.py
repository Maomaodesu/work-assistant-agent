import tempfile
import threading
import time
import unittest
from pathlib import Path

from analysis_job_manager import AnalysisJobManager
from workspace_store import WorkspaceStore, WorkspaceStoreError


def wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


class ControlledDiscovery:
    def __init__(self, store):
        self.store = store
        self.calls = 0

    def discover(self, *, run_id, control, **request):
        self.calls += 1
        self.store.update_classification_run(run_id, status="running", stage="local_screening")
        for index in range(1, 21):
            if not control.checkpoint():
                self.store.update_classification_run(
                    run_id, status="cancelled", stage="cancelled"
                )
                return
            self.store.update_classification_run(run_id, processed_sources=index)
            time.sleep(0.02)
        self.store.update_classification_run(run_id, status="completed", stage="finished")


class FailsOnceDiscovery:
    def __init__(self, store):
        self.store = store
        self.calls = 0

    def discover(self, *, run_id, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("AMD unavailable")
        run = self.store.get_classification_run(run_id)
        self.store.update_classification_run(
            run_id, status="running", stage="local_screening",
            processed_sources=run["total_sources"],
        )
        self.store.update_classification_run(run_id, status="completed", stage="finished")


class RecordingStep:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def sync(self):
        self.events.append(self.name)
        return {}

    def match_all(self):
        self.events.append(self.name)
        return {}

    def segment_all(self):
        self.events.append(self.name)
        return {}


class RecordingDiscovery:
    def __init__(self, store, events):
        self.store = store
        self.events = events

    def discover(self, *, run_id, **kwargs):
        self.events.append("discover")
        self.store.update_classification_run(
            run_id, status="completed", stage="finished", total_sources=0,
        )


class AnalysisJobManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        project_root = root / "project"
        project_root.mkdir()
        self.store = WorkspaceStore(root / "workspace.db")
        self.project = self.store.create_project("Demo", [str(project_root)])
        conversation = self.store.upsert_conversation(
            "codex", "session", original_project_path=str(project_root)
        )
        self.store.replace_messages(conversation["conversation_id"], [
            {"ordinal": index, "role": "user", "content": f"feature {index}"}
            for index in range(20)
        ])
        for index in range(20):
            self.store.create_segment(
                conversation["conversation_id"], index, index,
                project_id=self.project["project_id"], title=f"feature {index}",
            )
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown()
        self.temp_dir.cleanup()

    def manager(self, discovery, *, recover_interrupted=False, events=None):
        events = events if events is not None else []
        manager = AnalysisJobManager(
            self.store, discovery,
            conversation_sync=RecordingStep(events, "sync"),
            project_matcher=RecordingStep(events, "match"),
            segmenter=RecordingStep(events, "segment"),
            recover_interrupted=recover_interrupted,
        )
        self.managers.append(manager)
        return manager

    def test_background_run_can_pause_resume_and_cancel(self):
        manager = self.manager(ControlledDiscovery(self.store))
        started_at = time.time()
        run = manager.start(project_id=self.project["project_id"])
        self.assertLess(time.time() - started_at, 0.2)
        wait_until(lambda: manager.get(run["run_id"])["processed_sources"] >= 2)
        with self.assertRaises(WorkspaceStoreError):
            manager.start(project_id=self.project["project_id"])

        paused = manager.pause(run["run_id"])
        self.assertEqual(paused["status"], "paused")
        time.sleep(0.06)
        stable_count = manager.get(run["run_id"])["processed_sources"]
        time.sleep(0.08)
        self.assertEqual(manager.get(run["run_id"])["processed_sources"], stable_count)

        resumed = manager.resume(run["run_id"])
        self.assertEqual(resumed["status"], "running")
        wait_until(lambda: manager.get(run["run_id"])["processed_sources"] >= stable_count + 1)
        cancelled = manager.cancel(run["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        wait_until(lambda: manager.get(run["run_id"])["stage"] == "cancelled")

    def test_failed_run_retries_incrementally_with_original_request(self):
        discovery = FailsOnceDiscovery(self.store)
        manager = self.manager(discovery)
        first = manager.start(project_id=self.project["project_id"], limit=5)
        def failed_run():
            run = manager.get(first["run_id"])
            return run if run["status"] == "failed" else None
        failed = wait_until(failed_run)
        self.assertIn("AMD unavailable", failed["error_message"])

        retry = manager.retry(first["run_id"])
        self.assertEqual(retry["retry_of_run_id"], first["run_id"])
        self.assertEqual(retry["request"]["project_id"], self.project["project_id"])
        self.assertFalse(retry["request"]["force"])
        def completed_run():
            run = manager.get(retry["run_id"])
            return run if run["status"] == "completed" else None
        completed = wait_until(completed_run)
        self.assertEqual(completed["processed_sources"], 5)

    def test_service_restart_marks_orphaned_run_retryable(self):
        run = self.store.create_classification_run(
            run_type="incremental", total_sources=3,
            request={"project_id": self.project["project_id"]},
        )
        self.store.update_classification_run(run["run_id"], status="running")
        manager = self.manager(
            ControlledDiscovery(self.store), recover_interrupted=True
        )
        recovered = manager.get(run["run_id"])
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["stage"], "interrupted")

    def test_run_refreshes_local_conversations_before_discovery(self):
        events = []
        manager = self.manager(RecordingDiscovery(self.store, events), events=events)
        run = manager.start(project_id=self.project["project_id"])
        wait_until(lambda: manager.get(run["run_id"])["status"] == "completed")
        self.assertEqual(events, ["sync", "match", "segment", "discover"])


if __name__ == "__main__":
    unittest.main()
