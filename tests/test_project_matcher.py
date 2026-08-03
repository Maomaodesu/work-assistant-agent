import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from project_matcher import ConversationProjectMatcher, normalize_session_path
from workspace_store import WorkspaceStore


class ProjectMatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend = self.root / "backend"
        self.frontend = self.root / "frontend"
        self.other = self.root / "other"
        self.backend.mkdir()
        self.frontend.mkdir()
        self.other.mkdir()
        (self.backend / "src" / "main").mkdir(parents=True)
        self.store = WorkspaceStore(self.root / "workspace.db")
        self.project = self.store.create_project(
            "Demo", [str(self.backend), str(self.frontend)]
        )
        self.matcher = ConversationProjectMatcher(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _conversation(self, session_id: str, path: str, source="codex"):
        return self.store.upsert_conversation(
            source,
            session_id,
            title=session_id,
            original_project_path=path,
            started_at="2026-07-18T10:00:00+00:00",
            updated_at="2026-07-18T10:01:00+00:00",
        )

    def test_normalizes_windows_msys_and_wsl_drive_paths(self):
        windows = normalize_session_path("C:/workspace/demo")
        self.assertEqual(windows, normalize_session_path("/c/workspace/demo"))
        self.assertEqual(windows, normalize_session_path("/mnt/c/workspace/demo"))

    def test_exact_root_and_descendant_match_multi_root_project(self):
        exact = self._conversation("exact", str(self.frontend))
        nested = self._conversation("nested", str(self.backend / "src" / "main"), "claude")

        result = self.matcher.match_all()
        exact_match = self.store.get_conversation_project_matches(exact["conversation_id"])[0]
        nested_match = self.store.get_conversation_project_matches(nested["conversation_id"])[0]

        self.assertEqual(result["matched"], 2)
        self.assertEqual(exact_match["project_id"], self.project["project_id"])
        self.assertEqual(exact_match["match_method"], "exact_root")
        self.assertEqual(exact_match["confidence"], 1.0)
        self.assertEqual(nested_match["match_method"], "root_descendant")
        self.assertTrue(nested_match["is_primary"])

    def test_longest_nested_project_root_wins(self):
        parent_project = self.store.create_project("Parent", [str(self.root)])
        conversation = self._conversation("nested-project", str(self.backend / "src"))

        outcome = self.matcher.match_conversation(conversation)
        match = outcome["matches"][0]

        self.assertEqual(outcome["state"], "matched")
        self.assertEqual(match["project_id"], self.project["project_id"])
        self.assertNotEqual(match["project_id"], parent_project["project_id"])

    def test_similar_path_prefix_does_not_false_match(self):
        repo = self.root / "repo"
        repo_other = self.root / "repo-other"
        repo.mkdir()
        repo_other.mkdir()
        project = self.store.create_project("Repo", [str(repo)])
        conversation = self._conversation("prefix", str(repo_other))

        outcome = self.matcher.match_conversation(conversation)

        self.assertEqual(outcome["state"], "unassigned")
        self.assertEqual(self.store.get_conversation_project_matches(conversation["conversation_id"]), [])
        self.assertIsNotNone(project)

    def test_missing_path_is_reported_separately(self):
        self._conversation("no-path", "")
        result = self.matcher.match_all()
        self.assertEqual(result["no_path"], 1)

    def test_manual_project_assignment_survives_automatic_rematch(self):
        other_project = self.store.create_project("Other", [str(self.other)])
        conversation = self._conversation("manual", str(self.backend))
        self.store.set_manual_conversation_project(
            conversation["conversation_id"], other_project["project_id"]
        )

        result = self.matcher.match_all()
        match = self.store.get_conversation_project_matches(conversation["conversation_id"])[0]

        self.assertEqual(result["manual"], 1)
        self.assertEqual(match["project_id"], other_project["project_id"])
        self.assertEqual(match["match_source"], "manual")
        self.assertEqual(match["match_method"], "manual")

    def test_conversation_list_exposes_match_state_and_project(self):
        self._conversation("matched-list", str(self.backend))
        self._conversation("unassigned-list", str(self.other))
        self.matcher.match_all()

        conversations = self.store.list_conversations()
        states = {item["external_session_id"]: item["project_match_state"] for item in conversations}
        matched = next(item for item in conversations if item["external_session_id"] == "matched-list")

        self.assertEqual(states["matched-list"], "matched")
        self.assertEqual(states["unassigned-list"], "unassigned")
        self.assertEqual(matched["primary_project"]["project_name"], "Demo")

    def test_match_api_and_manual_assignment_api(self):
        conversation = self._conversation("api", str(self.backend))
        fake_matcher = ConversationProjectMatcher(self.store)
        with (
            patch.object(server, "workspace_store", self.store),
            patch.object(server, "conversation_project_matcher", fake_matcher),
            patch.object(server.settings_service, "is_setup_complete", return_value=True),
        ):
            with TestClient(server.app) as client:
                match_response = client.post("/api/workspace/conversations/match-projects")
                unassigned = client.get("/api/workspace/conversations/unassigned")
                manual = client.post(
                    f"/api/workspace/conversations/{conversation['conversation_id']}/project",
                    json={"project_id": self.project["project_id"]},
                )

        self.assertEqual(match_response.status_code, 200)
        self.assertEqual(match_response.json()["matched"], 1)
        self.assertEqual(unassigned.json(), [])
        self.assertEqual(manual.status_code, 200)
        self.assertEqual(manual.json()["matches"][0]["match_source"], "manual")


if __name__ == "__main__":
    unittest.main()
