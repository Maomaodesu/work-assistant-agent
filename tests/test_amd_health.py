import unittest
from types import SimpleNamespace

from amd_health import AMDHealthMonitor
from settings import ConfigurationError


class FakeModels:
    def __init__(self, model_ids=None, error=None):
        self.model_ids = model_ids or []
        self.error = error
        self.calls = 0

    def list(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=model_id) for model_id in self.model_ids])


class AMDHealthTests(unittest.TestCase):
    def settings(self, model="Qwen-Test"):
        return SimpleNamespace(
            amd_base_url="http://10.0.0.8:8001/v1",
            amd_model=model,
        )

    def monitor(self, models, **kwargs):
        client = SimpleNamespace(models=models)
        return AMDHealthMonitor(
            settings_provider=self.settings,
            api_key_provider=lambda: "test-secret-key",
            api_key_source_provider=lambda: "system_keyring",
            client_factory=lambda **options: client,
            cache_seconds=30,
            **kwargs,
        )

    def test_online_status_checks_models_and_uses_cache(self):
        models = FakeModels(["Qwen-Test", "Other"])
        monitor = self.monitor(models)

        first = monitor.check()
        second = monitor.check()
        forced = monitor.check(force=True)

        self.assertTrue(first["online"])
        self.assertEqual(first["status"], "online")
        self.assertEqual(first["models_count"], 2)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(models.calls, 2)
        self.assertFalse(forced["cached"])

    def test_model_missing_is_distinct_from_server_offline(self):
        result = self.monitor(FakeModels(["Different-Model"])).check()
        self.assertFalse(result["online"])
        self.assertEqual(result["status"], "model_missing")
        self.assertIn("模型", result["message"])

    def test_timeout_and_missing_key_return_safe_failure(self):
        timeout = self.monitor(FakeModels(error=TimeoutError("request timed out"))).check()
        self.assertEqual(timeout["status"], "timeout")
        self.assertNotIn("test-secret-key", timeout["error"])

        monitor = AMDHealthMonitor(
            settings_provider=self.settings,
            api_key_provider=lambda: (_ for _ in ()).throw(ConfigurationError("no key")),
            api_key_source_provider=lambda: "none",
            client_factory=lambda **kwargs: None,
        )
        missing = monitor.check()
        self.assertEqual(missing["status"], "not_configured")
        self.assertFalse(missing["api_key_configured"])


if __name__ == "__main__":
    unittest.main()
