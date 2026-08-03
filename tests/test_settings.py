import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server
import settings
from settings import ConfigurationError, SettingsService


class FakeKeyring:
    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[(service, username)] = password


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_dir = self.root / "demo-project"
        self.project_dir.mkdir()
        self.fake_keyring = FakeKeyring()
        self.keyring_patches = [
            patch.object(
                settings.keyring,
                "get_password",
                side_effect=self.fake_keyring.get_password,
            ),
            patch.object(
                settings.keyring,
                "set_password",
                side_effect=self.fake_keyring.set_password,
            ),
        ]
        for active_patch in self.keyring_patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.keyring_patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    def _payload(self, api_key="secret-value-for-test"):
        return {
            "api_key": api_key,
            "amd_base_url": "https://example.test/v1",
            "amd_model": "test-model",
            "default_project_paths": [str(self.project_dir)],
            "request_timeout_seconds": 120,
            "send_diff_summary": False,
            "send_commit_info": True,
        }

    def test_secret_is_stored_in_keyring_not_sqlite(self):
        db_path = self.root / "settings.db"
        service = SettingsService(db_path)
        payload = self._payload()

        service.save_api_key(payload["api_key"])
        service.save_public_settings(payload, mark_complete=True)

        self.assertEqual(service.get_api_key(), payload["api_key"])
        self.assertTrue(service.is_setup_complete())
        self.assertNotIn("api_key", service.public_view())

        database_bytes = db_path.read_bytes()
        self.assertNotIn(payload["api_key"].encode(), database_bytes)
        with sqlite3.connect(db_path) as conn:
            keys = {row[0] for row in conn.execute("SELECT key FROM app_settings")}
        self.assertNotIn("api_key", keys)

        reloaded = SettingsService(db_path).load()
        self.assertEqual(reloaded.amd_model, "test-model")
        self.assertEqual(reloaded.default_project_paths, (str(self.project_dir),))
        self.assertFalse(reloaded.send_diff_summary)

    def test_invalid_project_path_is_rejected(self):
        service = SettingsService(self.root / "settings.db")
        payload = self._payload()
        payload["default_project_paths"] = [str(self.root / "missing")]
        with self.assertRaises(ConfigurationError):
            service.validate_public_settings(payload)

        with self.assertRaises(ConfigurationError):
            service.validate_amd_settings({
                "amd_base_url": "http://127.0.0.1:70000/v1",
                "amd_model": "model",
            })

    def test_first_run_redirect_and_web_save(self):
        service = SettingsService(self.root / "settings.db")
        with patch.object(server, "settings_service", service):
            client = TestClient(server.app)
            redirect = client.get("/", follow_redirects=False)
            setup_page = client.get("/setup")
            response = client.post("/api/setup", json=self._payload())
            home = client.get("/", follow_redirects=False)
            client.close()

        self.assertEqual(redirect.status_code, 303)
        self.assertEqual(redirect.headers["location"], "/setup")
        self.assertEqual(setup_page.status_code, 200)
        self.assertIn("初始化工作助手", setup_page.text)
        self.assertIn("获取可用模型", setup_page.text)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertNotIn("secret-value-for-test", response.text)
        self.assertEqual(home.status_code, 200)

    def test_connection_endpoint_uses_unsaved_key_without_persisting_it(self):
        service = SettingsService(self.root / "settings.db")
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with (
            patch.object(server, "settings_service", service),
            patch.object(server, "OpenAI", return_value=fake_client),
        ):
            client = TestClient(server.app)
            response = client.post("/api/setup/test", json=self._payload())
            client.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "response": "OK"})
        self.assertNotIn("secret-value-for-test", (self.root / "settings.db").read_text(
            encoding="latin-1"
        ))

    def test_model_discovery_returns_real_api_model_ids(self):
        service = SettingsService(self.root / "settings.db")
        fake_client = MagicMock()
        fake_client.models.list.return_value = SimpleNamespace(data=[
            SimpleNamespace(id="Qwen/Qwen3-32B", owned_by="AMD"),
            SimpleNamespace(id="Qwen3.6-35B-A3B", owned_by="Radeon Cloud"),
        ])
        request = {
            "api_key": "unsaved-discovery-key",
            "amd_base_url": "https://example.test/v1",
            "request_timeout_seconds": 60,
        }

        with (
            patch.object(server, "settings_service", service),
            patch.object(server, "OpenAI", return_value=fake_client),
        ):
            client = TestClient(server.app)
            response = client.post("/api/setup/models", json=request)
            client.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [model["id"] for model in response.json()["models"]],
            ["Qwen/Qwen3-32B", "Qwen3.6-35B-A3B"],
        )
        self.assertNotIn("unsaved-discovery-key", response.text)
        self.assertNotIn(
            "unsaved-discovery-key",
            (self.root / "settings.db").read_text(encoding="latin-1"),
        )

    def test_home_quick_config_updates_only_amd_settings_and_rechecks(self):
        service = SettingsService(self.root / "quick-settings.db")
        service.save_api_key("old-secret-key")
        service.save_public_settings(self._payload(api_key=""), mark_complete=True)
        health = {
            "online": True, "status": "online", "message": "AMD API 在线",
            "base_url": "http://10.0.0.8:9000/v1", "model": "new-model",
            "latency_ms": 12,
        }
        monitor = MagicMock()
        monitor.check.return_value = health
        with (
            patch.object(server, "settings_service", service),
            patch.object(server, "amd_health_monitor", monitor),
        ):
            with TestClient(server.app) as client:
                status = client.get("/api/amd/status?force=true")
                response = client.post("/api/amd/config", json={
                    "amd_base_url": "http://10.0.0.8:9000/v1",
                    "amd_model": "new-model",
                    "api_key": "new-secret-key",
                })

        self.assertEqual(status.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["health"]["online"])
        self.assertEqual(service.load().amd_base_url, "http://10.0.0.8:9000/v1")
        self.assertEqual(service.load().amd_model, "new-model")
        self.assertEqual(service.get_api_key(), "new-secret-key")
        self.assertEqual(service.load().default_project_paths, (str(self.project_dir),))
        self.assertNotIn("new-secret-key", (self.root / "quick-settings.db").read_text(encoding="latin-1"))
        monitor.invalidate.assert_called_once()
        monitor.check.assert_any_call(force=True)


if __name__ == "__main__":
    unittest.main()
