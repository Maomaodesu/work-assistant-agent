import io
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from settings import SettingsService


class ServerStartupTests(unittest.TestCase):
    def test_stable_server_defaults_and_reload_environment_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                stable = SettingsService(Path(temp_dir) / "stable.db").load()
            with patch.dict(os.environ, {"SERVER_RELOAD": "true"}, clear=True):
                development = SettingsService(Path(temp_dir) / "development.db").load()

        self.assertEqual(stable.server_host, "127.0.0.1")
        self.assertFalse(stable.server_reload)
        self.assertTrue(development.server_reload)

    def test_health_endpoint_is_public_before_setup(self):
        fake_settings = SimpleNamespace(
            server_host="127.0.0.1",
            server_port=8000,
            server_reload=False,
        )
        fake_service = SimpleNamespace(
            is_setup_complete=lambda: False,
            public_view=lambda: {},
        )
        with (
            patch.object(server, "settings_service", fake_service),
            patch.object(server, "get_settings", return_value=fake_settings),
        ):
            with TestClient(server.app) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertFalse(response.json()["server"]["reload"])

    def test_port_check_detects_listener_and_recovers_after_close(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            available, message = server.check_port_available("127.0.0.1", port)
            self.assertFalse(available)
            self.assertIn(str(port), message)
        finally:
            listener.close()

        available, message = server.check_port_available("127.0.0.1", port)
        self.assertTrue(available)
        self.assertEqual(message, "")

    def test_run_server_stops_before_uvicorn_when_port_is_busy(self):
        fake_settings = SimpleNamespace(
            server_host="127.0.0.1",
            server_port=8000,
            server_reload=False,
        )
        stderr = io.StringIO()
        with (
            patch.object(server, "get_settings", return_value=fake_settings),
            patch.object(server, "check_port_available", return_value=(False, "busy")),
            patch("uvicorn.run") as uvicorn_run,
            redirect_stderr(stderr),
        ):
            exit_code = server.run_server()

        self.assertEqual(exit_code, 1)
        uvicorn_run.assert_not_called()
        self.assertIn("启动失败", stderr.getvalue())

    def test_normal_start_uses_app_object_without_reloader(self):
        fake_settings = SimpleNamespace(
            server_host="127.0.0.1",
            server_port=8000,
            server_reload=False,
        )
        with (
            patch.object(server, "get_settings", return_value=fake_settings),
            patch.object(server, "check_port_available", return_value=(True, "")),
            patch("uvicorn.run") as uvicorn_run,
        ):
            exit_code = server.run_server()

        self.assertEqual(exit_code, 0)
        self.assertIs(uvicorn_run.call_args.args[0], server.app)
        self.assertFalse(uvicorn_run.call_args.kwargs["reload"])

    def test_page_has_bounded_bootstrap_and_visible_reconnect(self):
        template = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn('id="appStatusBanner"', template)
        self.assertIn("function fetchWithTimeout(", template)
        self.assertIn('fetchWithTimeout("/health", {}, 5000)', template)
        self.assertIn("重新连接", template)
        self.assertIn('id="amdConnectionCard"', template)
        self.assertIn('id="amdHost"', template)
        self.assertIn('id="amdPort"', template)
        self.assertIn('id="amdQuickApiKey"', template)
        self.assertIn('"/api/amd/config"', template)
        self.assertIn("checkAMDConnection", template)

    def test_start_script_uses_project_virtual_environment(self):
        script = Path("start.ps1").read_text(encoding="utf-8")
        self.assertIn('.venv\\Scripts\\python.exe', script)
        self.assertIn('& $Python "server.py"', script)
        self.assertTrue(script.isascii(), "start.ps1 must remain PowerShell 5.1-safe ASCII")


if __name__ == "__main__":
    unittest.main()
