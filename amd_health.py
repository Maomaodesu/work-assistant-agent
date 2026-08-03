"""AMD OpenAI 兼容服务的轻量连接状态检测。"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable

from openai import OpenAI

from settings import ConfigurationError, get_api_key, get_settings, settings_service


class AMDHealthMonitor:
    def __init__(
        self,
        *,
        settings_provider: Callable = get_settings,
        api_key_provider: Callable = get_api_key,
        api_key_source_provider: Callable = settings_service.api_key_source,
        client_factory: Callable = OpenAI,
        cache_seconds: int = 20,
        check_timeout_seconds: int = 8,
    ):
        self.settings_provider = settings_provider
        self.api_key_provider = api_key_provider
        self.api_key_source_provider = api_key_source_provider
        self.client_factory = client_factory
        self.cache_seconds = cache_seconds
        self.check_timeout_seconds = check_timeout_seconds
        self._cached: dict | None = None
        self._cached_at = 0.0
        self._lock = threading.Lock()

    def invalidate(self):
        with self._lock:
            self._cached = None
            self._cached_at = 0.0

    def check(self, *, force: bool = False) -> dict:
        with self._lock:
            if (
                not force and self._cached
                and time.monotonic() - self._cached_at < self.cache_seconds
            ):
                return dict(self._cached, cached=True)
            result = self._perform_check()
            self._cached = result
            self._cached_at = time.monotonic()
            return dict(result, cached=False)

    def _perform_check(self) -> dict:
        started = time.perf_counter()
        settings = self.settings_provider()
        key_source = self.api_key_source_provider()
        base = {
            "base_url": settings.amd_base_url,
            "model": settings.amd_model,
            "api_key_configured": key_source != "none",
            "api_key_source": key_source,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        api_key = ""
        try:
            api_key = self.api_key_provider()
            client = self.client_factory(
                api_key=api_key,
                base_url=settings.amd_base_url,
                timeout=self.check_timeout_seconds,
                max_retries=0,
            )
            response = client.models.list(timeout=self.check_timeout_seconds)
            model_ids = [str(item.id) for item in getattr(response, "data", [])]
            model_available = settings.amd_model in model_ids
            status = "online" if model_available else "model_missing"
            message = (
                "AMD API 在线，当前模型可用"
                if model_available
                else "AMD API 在线，但当前配置的模型不在可用列表中"
            )
            return {
                **base,
                "online": model_available,
                "status": status,
                "message": message,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "models_count": len(model_ids),
                "model_available": model_available,
                "error": "",
            }
        except Exception as exc:
            error = str(exc)
            if api_key:
                error = error.replace(api_key, "***")
            error = error[:600]
            status_code = getattr(exc, "status_code", None)
            class_name = exc.__class__.__name__.lower()
            lowered = error.lower()
            if isinstance(exc, ConfigurationError):
                status, message = "not_configured", "AMD API Key 尚未配置"
            elif status_code in {401, 403} or "unauthorized" in lowered or "authentication" in lowered:
                status, message = "unauthorized", "AMD API Key 无效或没有访问权限"
            elif "timeout" in class_name or "timed out" in lowered or "timeout" in lowered:
                status, message = "timeout", "连接 AMD API 超时"
            elif "connection" in class_name or "connection" in lowered or "connect" in lowered:
                status, message = "offline", "无法连接 AMD API 服务器"
            else:
                status, message = "error", "AMD API 检测失败"
            return {
                **base,
                "online": False,
                "status": status,
                "message": message,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "models_count": 0,
                "model_available": False,
                "error": error,
            }


amd_health_monitor = AMDHealthMonitor()
