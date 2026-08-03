import tempfile
import unittest
from pathlib import Path
from typing import Annotated

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from session_store import (
    build_turn_input,
    create_sqlite_checkpointer,
    thread_config,
)


class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    gathered_info: dict
    output: str


def remember_state(state: TestState) -> dict:
    if not state.get("intent"):
        return {
            "intent": "new_task",
            "gathered_info": {"project_paths": ["C:/workspace/demo"]},
            "output": "first turn",
        }
    return {"output": f"messages={len(state['messages'])}"}


def build_test_graph(checkpointer):
    graph = StateGraph(TestState)
    graph.add_node("remember", remember_state)
    graph.set_entry_point("remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


class SessionPersistenceTests(unittest.TestCase):
    def test_thread_state_survives_new_connection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "checkpoints.db"
            config = thread_config("session-restart-test")

            saver1, conn1 = create_sqlite_checkpointer(db_path)
            graph1 = build_test_graph(saver1)
            first = graph1.invoke(build_turn_input("first message"), config=config)
            conn1.close()

            saver2, conn2 = create_sqlite_checkpointer(db_path)
            graph2 = build_test_graph(saver2)
            second = graph2.invoke(build_turn_input("second message"), config=config)
            conn2.close()

            self.assertEqual(first["intent"], "new_task")
            self.assertEqual(second["intent"], "new_task")
            self.assertEqual(
                second["gathered_info"],
                {"project_paths": ["C:/workspace/demo"]},
            )
            self.assertEqual(len(second["messages"]), 2)
            self.assertEqual(second["output"], "messages=2")

    def test_different_session_ids_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saver, conn = create_sqlite_checkpointer(Path(temp_dir) / "checkpoints.db")
            graph = build_test_graph(saver)
            graph.invoke(build_turn_input("session A"), config=thread_config("session-a"))
            state_b = graph.invoke(
                build_turn_input("session B"),
                config=thread_config("session-b"),
            )
            conn.close()

            self.assertEqual(len(state_b["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
