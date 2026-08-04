import unittest
from pathlib import Path


class ConversationMessageColorTests(unittest.TestCase):
    def test_conversation_detail_uses_distinct_human_ai_and_analysis_colors(self):
        stylesheet = Path("static/style.css").read_text(encoding="utf-8")

        self.assertIn(".conversation-timeline-entry.user-entry { border-color: #7589ff;", stylesheet)
        self.assertIn(".conversation-timeline-entry.assistant-entry { border-color: #47bd89;", stylesheet)
        self.assertIn(".conversation-timeline-entry.analysis-entry { border-color: #bb86fc;", stylesheet)
        self.assertIn(".conversation-timeline-entry.original-entry { border-left-width: 4px; }", stylesheet)
