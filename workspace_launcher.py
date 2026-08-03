"""本地项目工作台：发现并安全启动编辑器、终端与 AI CLI。"""

import os
import shutil
import subprocess
import re
from pathlib import Path
from typing import Callable


TOOL_KEYS = ("terminal", "vscode", "idea", "codex", "claude")
LAUNCH_ACTIONS = {
    "explorer", "terminal", "default_editor", "vscode", "idea", "codex", "claude"
}
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,240}$")


class WorkspaceLaunchError(ValueError):
    pass


def _first_existing(candidates) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
    return ""


def _which(*commands: str) -> str:
    for command in commands:
        found = shutil.which(command)
        if found:
            return str(Path(found).resolve())
    return ""


def _idea_candidates() -> list[str]:
    candidates = []
    program_files = Path(os.getenv("ProgramFiles", "C:/Program Files")) / "JetBrains"
    if program_files.is_dir():
        candidates.extend(
            str(path) for path in sorted(
                program_files.glob("IntelliJ IDEA*/bin/idea64.exe"),
                reverse=True,
            )
        )
    toolbox_root = Path(os.getenv("LOCALAPPDATA", "")) / "JetBrains" / "Toolbox" / "apps"
    if toolbox_root.is_dir():
        candidates.extend(
            str(path) for path in sorted(
                toolbox_root.glob("**/bin/idea64.exe"),
                reverse=True,
            )
        )
    return candidates


def discover_local_tools(configured_paths: dict | None = None) -> dict[str, dict]:
    """返回固定工具白名单的可用状态；配置路径优先于自动发现。"""
    configured = configured_paths or {}
    local_app_data = Path(os.getenv("LOCALAPPDATA", ""))
    program_files = Path(os.getenv("ProgramFiles", "C:/Program Files"))
    system_root = Path(os.getenv("SystemRoot", "C:/Windows"))

    detected = {
        "terminal": _first_existing([
            configured.get("terminal"),
            _which("wt.exe", "powershell.exe", "pwsh.exe"),
            system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe",
        ]),
        "vscode": _first_existing([
            configured.get("vscode"),
            local_app_data / "Programs" / "Microsoft VS Code" / "Code.exe",
            program_files / "Microsoft VS Code" / "Code.exe",
            _which("code.exe", "code.cmd"),
        ]),
        "idea": _first_existing([
            configured.get("idea"),
            _which("idea64.exe", "idea.exe"),
            *_idea_candidates(),
        ]),
        "codex": _first_existing([
            configured.get("codex"),
            _which("codex.cmd", "codex.ps1", "codex.exe", "codex"),
        ]),
        "claude": _first_existing([
            configured.get("claude"),
            _which("claude.exe", "claude.cmd", "claude.ps1", "claude"),
        ]),
    }
    return {
        name: {"available": bool(path), "path": path, "configured": bool(configured.get(name))}
        for name, path in detected.items()
    }


def _interactive_command(executable: str) -> list[str]:
    suffix = Path(executable).suffix.lower()
    if suffix == ".ps1":
        powershell = _which("powershell.exe", "pwsh.exe")
        if not powershell:
            raise WorkspaceLaunchError("未找到可运行 PowerShell 脚本的终端")
        return [powershell, "-NoExit", "-ExecutionPolicy", "Bypass", "-File", executable]
    if suffix in {".cmd", ".bat"}:
        command_processor = os.getenv("COMSPEC") or _which("cmd.exe")
        if not command_processor:
            raise WorkspaceLaunchError("未找到 cmd.exe，无法启动该 CLI")
        return [command_processor, "/d", "/k", executable]
    return [executable]


def launch_project_tool(
    project_path: str,
    action: str,
    *,
    preferred_editor: str = "auto",
    configured_paths: dict | None = None,
    popen: Callable = subprocess.Popen,
) -> dict:
    """启动固定动作；不接受额外命令或来自浏览器的任意可执行文件路径。"""
    if action not in LAUNCH_ACTIONS:
        raise WorkspaceLaunchError(f"不支持的启动动作：{action}")
    project = Path(project_path).expanduser().resolve()
    if not project.is_dir():
        raise WorkspaceLaunchError(f"项目目录不存在：{project}")

    tools = discover_local_tools(configured_paths)
    selected_action = action
    if action == "default_editor":
        if preferred_editor in {"vscode", "idea"} and tools[preferred_editor]["available"]:
            selected_action = preferred_editor
        elif tools["vscode"]["available"]:
            selected_action = "vscode"
        elif tools["idea"]["available"]:
            selected_action = "idea"
        else:
            raise WorkspaceLaunchError("未检测到可用编辑器，请先在设置页配置")

    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    popen_options = {"cwd": str(project)}
    if selected_action == "explorer":
        explorer = _which("explorer.exe") or str(Path(os.getenv("SystemRoot", "C:/Windows")) / "explorer.exe")
        command = [explorer, str(project)]
        display_name = "文件资源管理器"
    else:
        tool = tools.get(selected_action)
        if not tool or not tool["available"]:
            raise WorkspaceLaunchError(f"未检测到 {selected_action}，请先在设置页配置路径")
        executable = tool["path"]
        if selected_action == "terminal":
            if Path(executable).name.lower() in {"wt.exe", "wt"}:
                command = [executable, "-d", str(project)]
            else:
                command = _interactive_command(executable)
                popen_options["creationflags"] = creation_flags
            display_name = "终端"
        elif selected_action in {"codex", "claude"}:
            command = _interactive_command(executable)
            display_name = "Codex" if selected_action == "codex" else "Claude Code"
            popen_options["creationflags"] = creation_flags
        else:
            command = [executable, str(project)]
            display_name = "VS Code" if selected_action == "vscode" else "IntelliJ IDEA"

    try:
        process = popen(command, shell=False, **popen_options)
    except OSError as exc:
        raise WorkspaceLaunchError(f"启动 {display_name} 失败：{exc}") from exc
    return {
        "action": selected_action,
        "tool_name": display_name,
        "project_path": str(project),
        "pid": getattr(process, "pid", None),
    }


def resume_external_session(
    project_path: str,
    source: str,
    external_session_id: str,
    *,
    configured_paths: dict | None = None,
    popen: Callable = subprocess.Popen,
) -> dict:
    """使用数据库中的可信会话 ID 恢复 Codex/Claude 会话。"""
    if source not in {"codex", "claude"}:
        raise WorkspaceLaunchError("只有 Codex 或 Claude 会话可以恢复")
    session_id = str(external_session_id).strip()
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise WorkspaceLaunchError("外部会话 ID 格式不安全，无法启动")
    project = Path(project_path).expanduser().resolve()
    if not project.is_dir():
        raise WorkspaceLaunchError(f"项目目录不存在：{project}")
    tool = discover_local_tools(configured_paths).get(source)
    if not tool or not tool["available"]:
        raise WorkspaceLaunchError(f"未检测到 {source}，请先在设置页配置路径")

    command = _interactive_command(tool["path"])
    command.extend(
        ["resume", session_id] if source == "codex" else ["--resume", session_id]
    )
    try:
        process = popen(
            command,
            cwd=str(project),
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    except OSError as exc:
        label = "Codex" if source == "codex" else "Claude Code"
        raise WorkspaceLaunchError(f"恢复 {label} 会话失败：{exc}") from exc
    return {
        "source": source,
        "tool_name": "Codex" if source == "codex" else "Claude Code",
        "external_session_id": session_id,
        "project_path": str(project),
        "pid": getattr(process, "pid", None),
    }


def launch_ai_with_context(
    project_path: str,
    source: str,
    context_path: str,
    *,
    configured_paths: dict | None = None,
    popen: Callable = subprocess.Popen,
) -> dict:
    """启动新的 Codex/Claude 会话，并要求其先读取本地上下文包。"""
    if source not in {"codex", "claude"}:
        raise WorkspaceLaunchError("只支持使用 Codex 或 Claude 续接工作项")
    project = Path(project_path).expanduser().resolve()
    context = Path(context_path).expanduser().resolve()
    if not project.is_dir():
        raise WorkspaceLaunchError(f"项目目录不存在：{project}")
    if not context.is_file() or context.suffix.lower() != ".md":
        raise WorkspaceLaunchError("上下文包文件不存在或格式不正确")
    tool = discover_local_tools(configured_paths).get(source)
    if not tool or not tool["available"]:
        raise WorkspaceLaunchError(f"未检测到 {source}，请先在设置页配置路径")

    prompt = (
        f"Read the local work context file at {context} before responding. "
        "Use it to continue the work item, verify the current repository state first, "
        "and do not attempt to reconstruct redacted secrets."
    )
    command = _interactive_command(tool["path"])
    command.append(prompt)
    try:
        process = popen(
            command,
            cwd=str(project),
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    except OSError as exc:
        label = "Codex" if source == "codex" else "Claude Code"
        raise WorkspaceLaunchError(f"使用上下文启动 {label} 失败：{exc}") from exc
    return {
        "source": source,
        "tool_name": "Codex" if source == "codex" else "Claude Code",
        "context_path": str(context),
        "project_path": str(project),
        "pid": getattr(process, "pid", None),
    }
