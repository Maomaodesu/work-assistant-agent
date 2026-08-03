import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node.js is only required for formatter tests")
class MessageFormatterTests(unittest.TestCase):
    def test_chat_page_loads_and_uses_formatter_module(self):
        project_root = Path(__file__).parent.parent
        template = (project_root / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('src="/static/message_formatter.js"', template)
        self.assertIn("WorkAssistantMessageFormatter.format(text)", template)

    def test_browser_formatter_semantics_and_xss_safety(self):
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["node", "tests/message_formatter_test.js"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MESSAGE_FORMATTER_TEST_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
