import tempfile
import unittest
from pathlib import Path

from snapshot_collector import collect_project_info


class SnapshotCollectorTests(unittest.TestCase):
    def test_skips_virtual_environment_and_reports_root_project_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "server.py").write_text("app = object()", encoding="utf-8")
            (root / "requirements.txt").write_text("fastapi", encoding="utf-8")
            (root / "README.md").write_text("# Demo", encoding="utf-8")
            dependency = root / ".venv" / "Lib" / "site-packages"
            dependency.mkdir(parents=True)
            (dependency / "third_party.py").write_text("x = 1", encoding="utf-8")

            project = collect_project_info(str(root))

        self.assertEqual(project.project_type, "python")
        self.assertEqual(project.root_files, ["README.md", "requirements.txt", "server.py"])
        scanned_paths = [file.path for module in project.modules for file in module.files]
        self.assertNotIn(".venv/Lib/site-packages/third_party.py", scanned_paths)


if __name__ == "__main__":
    unittest.main()
