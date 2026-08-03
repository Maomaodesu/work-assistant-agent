"""应用配置：SQLite 保存普通设置，系统凭据库保存 API Key。"""

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import keyring
from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=False)

DEFAULT_SETTINGS_DB = BASE_DIR / "data" / "settings.db"
KEYRING_SERVICE = "work-assistant"
KEYRING_USERNAME = "amd-api-key"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class AppSettings:
    amd_base_url: str = "https://developer.amd.com.cn/radeon/api/v1"
    amd_model: str = "Qwen3.6-35B-A3B"
    agent_db_path: Path = BASE_DIR / "data" / "agent.db"
    checkpoint_db_path: Path = BASE_DIR / "data" / "checkpoints.db"
    conversation_db_path: Path = BASE_DIR / "data" / "conversations.db"
    workspace_db_path: Path = BASE_DIR / "data" / "work_assistant.db"
    snapshot_dir: Path = BASE_DIR / "snapshots"
    default_project_paths: tuple[str, ...] = ()
    request_timeout_seconds: int = 180
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_reload: bool = False
    send_diff_summary: bool = True
    send_commit_info: bool = True
    setup_completed: bool = False
    preferred_editor: str = "auto"
    tool_paths: dict[str, str] = field(default_factory=dict)


def _resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class SettingsService:
    PUBLIC_KEYS = {
        "amd_base_url",
        "amd_model",
        "agent_db_path",
        "checkpoint_db_path",
        "conversation_db_path",
        "workspace_db_path",
        "snapshot_dir",
        "default_project_paths",
        "request_timeout_seconds",
        "server_host",
        "server_port",
        "server_reload",
        "send_diff_summary",
        "send_commit_info",
        "setup_completed",
        "preferred_editor",
        "tool_paths",
    }

    ENV_KEYS = {
        "amd_base_url": "AMD_BASE_URL",
        "amd_model": "AMD_MODEL",
        "agent_db_path": "AGENT_DB_PATH",
        "checkpoint_db_path": "CHECKPOINT_DB_PATH",
        "conversation_db_path": "CONVERSATION_DB_PATH",
        "workspace_db_path": "WORKSPACE_DB_PATH",
        "snapshot_dir": "SNAPSHOT_DIR",
        "request_timeout_seconds": "REQUEST_TIMEOUT_SECONDS",
        "server_host": "SERVER_HOST",
        "server_port": "SERVER_PORT",
        "server_reload": "SERVER_RELOAD",
        "send_diff_summary": "SEND_DIFF_SUMMARY",
        "send_commit_info": "SEND_COMMIT_INFO",
        "setup_completed": "WORK_ASSISTANT_SETUP_COMPLETED",
    }

    def __init__(self, db_path: Path | str | None = None):
        configured_path = os.getenv("WORK_ASSISTANT_SETTINGS_DB")
        self.db_path = _resolve_path(db_path or configured_path or DEFAULT_SETTINGS_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key        TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def _stored_values(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM app_settings").fetchall()
        values = {}
        for row in rows:
            if row["key"] in self.PUBLIC_KEYS:
                values[row["key"]] = json.loads(row["value_json"])
        return values

    def load(self) -> AppSettings:
        values = asdict(AppSettings())
        values.update(self._stored_values())

        for key, env_name in self.ENV_KEYS.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                values[key] = env_value

        default_paths_env = os.getenv("DEFAULT_PROJECT_PATHS")
        if default_paths_env is not None:
            values["default_project_paths"] = [
                item.strip() for item in default_paths_env.split(os.pathsep) if item.strip()
            ]

        return AppSettings(
            amd_base_url=str(values["amd_base_url"]).rstrip("/"),
            amd_model=str(values["amd_model"]),
            agent_db_path=_resolve_path(values["agent_db_path"]),
            checkpoint_db_path=_resolve_path(values["checkpoint_db_path"]),
            conversation_db_path=_resolve_path(values["conversation_db_path"]),
            workspace_db_path=_resolve_path(values["workspace_db_path"]),
            snapshot_dir=_resolve_path(values["snapshot_dir"]),
            default_project_paths=tuple(values.get("default_project_paths") or ()),
            request_timeout_seconds=int(values["request_timeout_seconds"]),
            server_host=str(values["server_host"]),
            server_port=int(values["server_port"]),
            server_reload=_as_bool(values["server_reload"]),
            send_diff_summary=_as_bool(values["send_diff_summary"]),
            send_commit_info=_as_bool(values["send_commit_info"]),
            setup_completed=_as_bool(values["setup_completed"]),
            preferred_editor=str(values.get("preferred_editor") or "auto"),
            tool_paths=dict(values.get("tool_paths") or {}),
        )

    def validate_public_settings(self, values: dict) -> dict:
        amd_values = self.validate_amd_settings(values)
        base_url = amd_values["amd_base_url"]
        model = amd_values["amd_model"]

        try:
            timeout = int(values.get("request_timeout_seconds", 180))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("请求超时时间必须是整数") from exc
        if timeout < 10 or timeout > 900:
            raise ConfigurationError("请求超时时间必须在 10–900 秒之间")

        raw_paths = values.get("default_project_paths", [])
        if isinstance(raw_paths, str):
            raw_paths = raw_paths.splitlines()
        project_paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        missing_paths = [path for path in project_paths if not Path(path).expanduser().exists()]
        if missing_paths:
            raise ConfigurationError(f"项目路径不存在：{missing_paths[0]}")

        return {
            "amd_base_url": base_url,
            "amd_model": model,
            "default_project_paths": project_paths,
            "request_timeout_seconds": timeout,
            "send_diff_summary": _as_bool(values.get("send_diff_summary", True)),
            "send_commit_info": _as_bool(values.get("send_commit_info", True)),
        }

    def validate_amd_settings(self, values: dict) -> dict:
        base_url = str(values.get("amd_base_url", "")).strip().rstrip("/")
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or not parsed_url.hostname
        ):
            raise ConfigurationError("AMD API 地址必须是有效的 http/https URL")
        if parsed_url.username or parsed_url.password:
            raise ConfigurationError("AMD API 地址不能包含用户名或密码")
        try:
            port = parsed_url.port
        except ValueError as exc:
            raise ConfigurationError("AMD API 端口必须在 1–65535 之间") from exc
        if port is not None and not 1 <= port <= 65535:
            raise ConfigurationError("AMD API 端口必须在 1–65535 之间")

        model = str(values.get("amd_model", "")).strip()
        if not model:
            raise ConfigurationError("模型名称不能为空")
        return {"amd_base_url": base_url, "amd_model": model}

    def save_public_settings(self, values: dict, *, mark_complete: bool = False):
        validated = self.validate_public_settings(values)
        if mark_complete:
            validated["setup_completed"] = True
        now = datetime.now().isoformat()
        with self._connect() as conn:
            for key, value in validated.items():
                conn.execute(
                    """
                    INSERT INTO app_settings(key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value, ensure_ascii=False), now),
                )

    def save_amd_settings(
        self, values: dict, *, mark_complete: bool = True
    ) -> dict:
        validated = self.validate_amd_settings(values)
        if mark_complete:
            validated["setup_completed"] = True
        now = datetime.now().isoformat()
        with self._connect() as conn:
            for key, value in validated.items():
                conn.execute(
                    """
                    INSERT INTO app_settings(key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json=excluded.value_json,
                        updated_at=excluded.updated_at
                    """,
                    (key, json.dumps(value, ensure_ascii=False), now),
                )
        return validated

    def validate_tool_settings(self, values: dict) -> dict:
        preferred_editor = str(values.get("preferred_editor", "auto")).strip().lower()
        if preferred_editor not in {"auto", "vscode", "idea"}:
            raise ConfigurationError("默认编辑器必须是 auto、vscode 或 idea")

        allowed_tools = {"terminal", "vscode", "idea", "codex", "claude"}
        raw_paths = values.get("tool_paths") or {}
        if not isinstance(raw_paths, dict):
            raise ConfigurationError("本地工具路径格式不正确")
        tool_paths = {}
        for name, raw_path in raw_paths.items():
            if name not in allowed_tools:
                continue
            normalized = str(raw_path or "").strip().strip('"')
            if not normalized:
                continue
            path = Path(os.path.expandvars(normalized)).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ConfigurationError(f"{name} 可执行文件不存在：{normalized}")
            tool_paths[name] = str(path.resolve())
        return {"preferred_editor": preferred_editor, "tool_paths": tool_paths}

    def save_tool_settings(self, values: dict):
        validated = self.validate_tool_settings(values)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            for key, value in validated.items():
                conn.execute(
                    """
                    INSERT INTO app_settings(key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value, ensure_ascii=False), now),
                )

    def _legacy_key_file(self) -> Path:
        configured = os.getenv("AMD_API_KEY_FILE")
        return _resolve_path(configured) if configured else BASE_DIR.parent / "AMD_apikey.txt"

    def api_key_source(self) -> str:
        if os.getenv("AMD_API_KEY"):
            return "environment"
        try:
            if keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME):
                return "system_keyring"
        except Exception:
            pass
        if self._legacy_key_file().exists():
            return "legacy_file"
        return "none"

    def get_api_key(self) -> str:
        env_key = os.getenv("AMD_API_KEY", "").strip()
        if env_key:
            return env_key
        try:
            stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception as exc:
            raise ConfigurationError(f"无法读取系统凭据库：{exc}") from exc
        if stored:
            return stored.strip()
        legacy_file = self._legacy_key_file()
        if legacy_file.exists():
            return legacy_file.read_text(encoding="utf-8").strip()
        raise ConfigurationError("尚未配置 AMD API Key，请访问 /setup")

    def save_api_key(self, api_key: str):
        normalized = api_key.strip()
        if not normalized:
            source = self.api_key_source()
            if source == "legacy_file":
                normalized = self._legacy_key_file().read_text(encoding="utf-8").strip()
            elif source == "none":
                raise ConfigurationError("AMD API Key 不能为空")
            else:
                return
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, normalized)
        except Exception as exc:
            raise ConfigurationError(f"无法写入系统凭据库：{exc}") from exc

    def is_setup_complete(self) -> bool:
        return self.load().setup_completed and self.api_key_source() != "none"

    def public_view(self) -> dict:
        settings = self.load()
        return {
            "amd_base_url": settings.amd_base_url,
            "amd_model": settings.amd_model,
            "agent_db_path": str(settings.agent_db_path),
            "checkpoint_db_path": str(settings.checkpoint_db_path),
            "conversation_db_path": str(settings.conversation_db_path),
            "workspace_db_path": str(settings.workspace_db_path),
            "snapshot_dir": str(settings.snapshot_dir),
            "settings_db_path": str(self.db_path),
            "default_project_paths": list(settings.default_project_paths),
            "request_timeout_seconds": settings.request_timeout_seconds,
            "server_host": settings.server_host,
            "server_port": settings.server_port,
            "server_reload": settings.server_reload,
            "send_diff_summary": settings.send_diff_summary,
            "send_commit_info": settings.send_commit_info,
            "setup_completed": settings.setup_completed,
            "preferred_editor": settings.preferred_editor,
            "tool_paths": settings.tool_paths,
            "api_key_configured": self.api_key_source() != "none",
            "api_key_source": self.api_key_source(),
        }


settings_service = SettingsService()


def get_settings() -> AppSettings:
    return settings_service.load()


def get_api_key() -> str:
    return settings_service.get_api_key()
