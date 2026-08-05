"""
中断快照采集器 (Interrupt Snapshot Collector)

支持多项目（前后端分离 / monorepo / 多服务）跨项目采集。

采集内容：
  - 多项目 Git 状态（remote / 分支 / commit / 未提交 / diff）
  - 多项目文件按功能模块聚合（MVC/组件完整度 + git 状态）
  - IDEA workspace.xml（变更文件 + 工作会话时长）
  - 跨项目工作会话：每次工作改了哪些项目的哪些模块
  - .agent/progress.json（任务进度）
  - Shell 历史 / 开发进程
"""

import json
import os
import re
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from xml.etree import ElementTree


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _run(cmd: list[str], cwd: Optional[str] = None) -> str:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _truncate(text: str, max_lines: int = 200) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... （已截断，共 {len(lines)} 行）"


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class GitCommitInfo:
    hash: str
    message: str
    author: str
    date: str


@dataclass
class GitChangeInfo:
    status: str
    path: str


@dataclass
class ModuleFile:
    path: str
    layer: str
    fs_modified: str
    in_git: bool


@dataclass
class ProjectModule:
    module_name: str
    last_modified: str
    git_status: str             # committed / partial / untracked
    layers_present: list[str]
    layers_missing: list[str]
    files: list[ModuleFile]


@dataclass
class ProjectInfo:
    project_name: str           # 项目名（目录名）
    project_path: str
    project_type: str           # spring-boot / vue / react / python / unknown
    remote_url: str
    branch: str
    commits: list[GitCommitInfo]
    root_files: list[str]      # 根目录中的关键源码与项目配置文件
    modules: list[ProjectModule]
    uncommitted_tracked: list[GitChangeInfo]
    diff_unstaged: str
    diff_staged: str


@dataclass
class WorkSession:
    session_index: int
    started_at: str
    duration_minutes: float
    projects_touched: dict      # { project_name: [module_name, ...] }


@dataclass
class WorkProgress:
    current_step: str
    completed_steps: list[str]
    pending_steps: list[str]
    notes: str


@dataclass
class InterruptSnapshot:
    snapshot_id: str
    task_id: str
    interrupt_time: str
    interrupt_reason: str

    # 多项目信息
    projects: list[ProjectInfo]

    # 跨项目工作会话
    work_sessions: list[WorkSession]
    total_work_hours: float

    # IDEA 变更文件（原始列表，中文可读）
    idea_changed_files: list[str]

    # 任务进度
    progress: Optional[WorkProgress]

    # 环境
    terminal_history: list[str]
    dev_processes: list[str]

    current_thought: str = ""
    blockers: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 项目类型检测
# ──────────────────────────────────────────────

def detect_project_type(project_path: str) -> str:
    root = Path(project_path)
    if (root / "pom.xml").exists():
        return "spring-boot"
    if (root / "package.json").exists():
        pkg = json.loads((root / "package.json").read_text(encoding="utf-8", errors="ignore"))
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        if "vue" in deps:
            return "vue"
        if "react" in deps or "react-dom" in deps:
            return "react"
        return "node"
    if list(root.glob("*.py")) or (root / "pyproject.toml").exists():
        return "python"
    return "unknown"


# ──────────────────────────────────────────────
# 层识别规则（按项目类型）
# ──────────────────────────────────────────────

# Spring Boot MVC 层
_JAVA_LAYER_RULES: list[tuple[str, str]] = [
    (r"/entity/|\\entity\\|Entity\.java",   "entity"),
    (r"Controller",                          "controller"),
    (r"ServiceImpl",                         "service_impl"),
    (r"/service/|\\service\\|Service\.java", "service"),
    (r"Mapper|Repository",                   "mapper"),
    (r"/dto/|\\dto\\|DTO|Dto",               "dto"),
    (r"Config",                              "config"),
    (r"Util|Utils",                          "util"),
]
_JAVA_MVC_LAYERS = ["controller", "service", "service_impl", "mapper", "entity"]

# Vue/React 前端层
_FRONTEND_LAYER_RULES: list[tuple[str, str]] = [
    (r"/views?/|\\views?\\",    "view"),
    (r"/pages?/|\\pages?\\",    "view"),
    (r"/components?/|\\components?\\", "component"),
    (r"/api/|\\api\\",          "api"),
    (r"/stores?/|\\stores?\\",  "store"),
    (r"/router/|\\router\\",    "router"),
    (r"/utils?/|\\utils?\\",    "util"),
    (r"/hooks?/|\\hooks?\\",    "hook"),
]
_FRONTEND_MVC_LAYERS = ["view", "component", "api", "store", "router"]

# Python 层
_PYTHON_LAYER_RULES: list[tuple[str, str]] = [
    (r"/api/|\\api\\|router",   "api"),
    (r"/service|\\service",     "service"),
    (r"/model|\\model",         "model"),
    (r"/schema|\\schema",       "schema"),
    (r"/utils?/|\\utils?\\",    "util"),
]
_PYTHON_MVC_LAYERS = ["api", "service", "model", "schema"]

_SKIP_DIRS = {
    ".git", "target", "build", "dist", "node_modules",
    "__pycache__", ".idea", ".vscode", ".gradle", ".agent",
    ".venv", "venv", ".tox", ".mypy_cache",
}

_SOURCE_SUFFIXES = {
    ".java", ".py", ".ts", ".js", ".vue", ".jsx", ".tsx",
    ".html", ".css", ".scss", ".xml", ".yml", ".yaml",
    ".sql", ".json", ".kt", ".properties", ".md",
}


def _get_layer_rules(project_type: str) -> tuple[list[tuple[str, str]], list[str]]:
    if project_type == "spring-boot":
        return _JAVA_LAYER_RULES, _JAVA_MVC_LAYERS
    if project_type in ("vue", "react", "node"):
        return _FRONTEND_LAYER_RULES, _FRONTEND_MVC_LAYERS
    if project_type == "python":
        return _PYTHON_LAYER_RULES, _PYTHON_MVC_LAYERS
    return _JAVA_LAYER_RULES, _JAVA_MVC_LAYERS


def _detect_layer(path_str: str, rules: list[tuple[str, str]]) -> str:
    for pattern, layer in rules:
        if re.search(pattern, path_str):
            return layer
    return "other"


def _extract_module_name(path_str: str, project_type: str) -> str:
    filename = Path(path_str).stem

    if project_type == "spring-boot":
        for suffix in ["ServiceImpl", "Service", "Controller", "Mapper",
                       "Repository", "Entity", "DTO", "Dto", "Config", "Utils", "Util"]:
            if filename.endswith(suffix):
                return filename[: -len(suffix)]
        if path_str.endswith(".sql"):
            return f"SQL:{filename}"
        return filename

    if project_type in ("vue", "react", "node"):
        # 从路径推断：views/StockList.vue → StockList
        parts = Path(path_str).parts
        for i, part in enumerate(parts):
            if part.lower() in ("views", "view", "pages", "page",
                                "components", "component", "api",
                                "stores", "store", "router"):
                if i + 1 < len(parts):
                    # 取子目录名或文件名
                    name = Path(parts[i + 1]).stem
                    # 去掉 index / main 等无意义名称
                    if name.lower() not in ("index", "main", "app"):
                        return name
        return filename

    return filename


# ──────────────────────────────────────────────
# 单项目采集
# ──────────────────────────────────────────────

def collect_project_info(project_path: str) -> ProjectInfo:
    root = Path(project_path)
    project_name = root.name
    project_type = detect_project_type(project_path)
    layer_rules, mvc_layers = _get_layer_rules(project_type)

    # Git 信息
    remote_url = _run(["git", "remote", "get-url", "origin"], project_path) or "（无 remote）"
    branch     = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], project_path) or "unknown"

    log_out = _run(["git", "log", "--format=%H\t%s\t%an\t%ai"], project_path)
    commits = []
    for line in log_out.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            commits.append(GitCommitInfo(
                hash=parts[0][:8], message=parts[1],
                author=parts[2], date=parts[3],
            ))

    status_out = _run(["git", "status", "--porcelain"], project_path)
    uncommitted, untracked = [], []
    for line in status_out.splitlines():
        if not line:
            continue
        xy   = line[:2].strip()
        path = line[3:].strip().strip('"')
        if xy == "??":
            untracked.append(path)
        else:
            uncommitted.append(GitChangeInfo(status=xy, path=path))

    diff_unstaged = _truncate(_run(["git", "diff"], project_path))
    diff_staged   = _truncate(_run(["git", "diff", "--cached"], project_path))

    # 文件 → 模块聚合
    tracked_set = set(_run(["git", "ls-files"], project_path).splitlines())
    root_files = sorted(
        fp.name for fp in root.iterdir()
        if fp.is_file() and (
            fp.suffix.lower() in {".py", ".js", ".ts", ".java", ".go", ".rs"}
            or fp.name.lower() in {
                "readme.md", "requirements.txt", "pyproject.toml", "package.json",
                "pom.xml", "dockerfile", "compose.yaml", "docker-compose.yml",
            }
        )
    )
    module_map: dict[str, list[ModuleFile]] = {}

    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        if any(part in _SKIP_DIRS for part in fp.parts):
            continue
        if fp.suffix.lower() not in _SOURCE_SUFFIXES:
            continue

        rel   = fp.relative_to(root).as_posix()
        mtime = datetime.fromtimestamp(fp.stat().st_mtime).isoformat(timespec="seconds")
        layer = _detect_layer(rel, layer_rules)
        mod   = _extract_module_name(rel, project_type)

        module_map.setdefault(mod, []).append(
            ModuleFile(path=rel, layer=layer, fs_modified=mtime, in_git=(rel in tracked_set))
        )

    modules = []
    for mod_name, files in module_map.items():
        files.sort(key=lambda f: f.fs_modified, reverse=True)
        layers_present = sorted(set(f.layer for f in files if f.layer in mvc_layers))
        layers_missing = [l for l in mvc_layers if l not in layers_present]
        all_git  = all(f.in_git for f in files)
        none_git = all(not f.in_git for f in files)
        git_status = "committed" if all_git else ("untracked" if none_git else "partial")

        modules.append(ProjectModule(
            module_name=mod_name,
            last_modified=files[0].fs_modified,
            git_status=git_status,
            layers_present=layers_present,
            layers_missing=layers_missing,
            files=files,
        ))
    modules.sort(key=lambda m: m.last_modified, reverse=True)

    return ProjectInfo(
        project_name=project_name,
        project_path=project_path,
        project_type=project_type,
        remote_url=remote_url,
        branch=branch,
        commits=commits,
        root_files=root_files,
        modules=modules,
        uncommitted_tracked=uncommitted,
        diff_unstaged=diff_unstaged,
        diff_staged=diff_staged,
    )


# ──────────────────────────────────────────────
# IDEA workspace.xml + 跨项目工作会话
# ──────────────────────────────────────────────

def collect_idea_info(
    main_project_path: str,
    all_projects: list[ProjectInfo],
) -> tuple[list[str], list[WorkSession], float]:
    """
    解析 IDEA workspace.xml，生成跨项目工作会话。
    对每个 workItem 时间窗口，在所有项目里匹配被修改的文件，推断活跃模块。
    """
    workspace_xml = Path(main_project_path) / ".idea" / "workspace.xml"
    if not workspace_xml.exists():
        return [], [], 0.0

    try:
        tree = ElementTree.parse(workspace_xml)
        root = tree.getroot()

        # 变更文件
        changed_files: list[str] = []
        for comp in root.iter("component"):
            if comp.get("name") == "ChangeListManager":
                for lst in comp.iter("list"):
                    for change in lst.iter("change"):
                        after = change.get("afterPath", "")
                        if after:
                            path = after.replace("$PROJECT_DIR$", main_project_path)
                            changed_files.append(f"[IDEA变更] {path}")

        # workItem 原始数据
        raw_sessions: list[tuple[datetime, float]] = []
        for comp in root.iter("component"):
            if comp.get("name") == "TaskManager":
                for task in comp.iter("task"):
                    if task.get("active") == "true":
                        for wi in task.iter("workItem"):
                            from_ms = int(wi.get("from", "0"))
                            dur_ms  = int(wi.get("duration", "0"))
                            if from_ms and dur_ms:
                                raw_sessions.append((
                                    datetime.fromtimestamp(from_ms / 1000),
                                    dur_ms / 1000 / 60,
                                ))
        raw_sessions.sort(key=lambda x: x[0])

        # 预构建：所有项目的文件修改时间表
        # { project_name: [(mtime, module_name), ...] }
        project_file_mtimes: dict[str, list[tuple[datetime, str]]] = {}
        for proj in all_projects:
            root_p = Path(proj.project_path)
            entries = []
            for fp in root_p.rglob("*"):
                if not fp.is_file():
                    continue
                if any(part in _SKIP_DIRS for part in fp.parts):
                    continue
                if fp.suffix.lower() not in _SOURCE_SUFFIXES:
                    continue
                mtime = datetime.fromtimestamp(fp.stat().st_mtime)
                mod   = _extract_module_name(
                    fp.relative_to(root_p).as_posix(), proj.project_type
                )
                entries.append((mtime, mod))
            project_file_mtimes[proj.project_name] = entries

        # 构建跨项目工作会话
        tolerance = 5 * 60  # 5 分钟容差
        work_sessions: list[WorkSession] = []
        for i, (start_dt, dur_min) in enumerate(raw_sessions):
            end_ts = start_dt.timestamp() + dur_min * 60 + tolerance
            projects_touched: dict[str, list[str]] = {}

            for proj_name, file_entries in project_file_mtimes.items():
                mods = list(dict.fromkeys(
                    mod for mtime, mod in file_entries
                    if start_dt.timestamp() <= mtime.timestamp() <= end_ts
                ))
                if mods:
                    projects_touched[proj_name] = mods

            work_sessions.append(WorkSession(
                session_index=i + 1,
                started_at=start_dt.isoformat(timespec="seconds"),
                duration_minutes=round(dur_min, 1),
                projects_touched=projects_touched,
            ))

        total_hours = round(sum(s.duration_minutes for s in work_sessions) / 60, 1)
        changed_files.append(
            f"[IDEA工时] 累计工作时长：{total_hours} 小时，共 {len(work_sessions)} 次"
        )
        return changed_files, work_sessions, total_hours

    except Exception:
        return [], [], 0.0


# ──────────────────────────────────────────────
# 其他采集函数
# ──────────────────────────────────────────────

def collect_work_progress(workspace_path: str) -> Optional[WorkProgress]:
    progress_file = Path(workspace_path) / ".agent" / "progress.json"
    if not progress_file.exists():
        return None
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WorkProgress(
            current_step=data.get("current_step", ""),
            completed_steps=data.get("completed_steps", []),
            pending_steps=data.get("pending_steps", []),
            notes=data.get("notes", ""),
        )
    except Exception as e:
        return WorkProgress(current_step=f"[读取失败: {e}]",
                            completed_steps=[], pending_steps=[], notes="")


def collect_terminal_history(lines: int = 20) -> list[str]:
    ps_history  = Path(os.environ.get("APPDATA", "")) / \
        "Microsoft" / "Windows" / "PowerShell" / "PSReadLine" / "ConsoleHost_history.txt"
    bash_history = Path.home() / ".bash_history"
    zsh_history  = Path.home() / ".zsh_history"
    for hist_file in [ps_history, bash_history, zsh_history]:
        if hist_file.exists():
            try:
                text = hist_file.read_text(encoding="utf-8", errors="ignore")
                all_lines = [l.strip() for l in text.splitlines() if l.strip()]
                return all_lines[-lines:]
            except Exception:
                continue
    return []


_DEV_KEYWORDS = {
    "python", "node", "uvicorn", "vite", "webpack",
    "pytest", "docker", "idea", "pycharm", "java", "gradle", "mvn", "code",
}


def collect_dev_processes() -> list[str]:
    try:
        import psutil
    except ImportError:
        return ["psutil 未安装"]
    found = set()
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if any(kw in name for kw in _DEV_KEYWORDS):
                found.add(proc.info["name"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(found)


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def collect_snapshot(
    task_id: str,
    project_paths: list[str],           # 支持多个项目路径
    idea_project_path: Optional[str] = None,  # .idea/ 所在的项目（通常是主项目）
    interrupt_reason: str = "manual",
) -> "InterruptSnapshot":
    now = datetime.now()
    snapshot_id = f"SNAP-{now.strftime('%Y%m%d-%H%M%S')}"

    # 1. 采集所有项目
    projects: list[ProjectInfo] = []
    for path in project_paths:
        print(f"[采集项目] {Path(path).name} ({path}) ...")
        proj = collect_project_info(path)
        print(f"  类型: {proj.project_type}  模块: {len(proj.modules)}  commits: {len(proj.commits)}")
        projects.append(proj)

    # 2. IDEA 工作会话（跨项目）
    idea_path = idea_project_path or project_paths[0]
    print(f"[IDEA工作会话] 读取 {idea_path}/.idea/workspace.xml ...")
    idea_changed, work_sessions, total_hours = collect_idea_info(idea_path, projects)
    print(f"  变更文件: {len([f for f in idea_changed if '变更' in f])}  会话: {len(work_sessions)}  累计: {total_hours}h")

    # 3. 任务进度
    print("[任务进度] 读取 .agent/progress.json ...")
    progress = collect_work_progress(idea_path)

    # 4. 环境
    print("[环境] Shell历史 + 开发进程 ...")
    terminal_history = collect_terminal_history()
    dev_processes    = collect_dev_processes()

    return InterruptSnapshot(
        snapshot_id=snapshot_id,
        task_id=task_id,
        interrupt_time=now.isoformat(),
        interrupt_reason=interrupt_reason,
        projects=projects,
        work_sessions=work_sessions,
        total_work_hours=total_hours,
        idea_changed_files=idea_changed,
        progress=progress,
        terminal_history=terminal_history,
        dev_processes=dev_processes,
    )


def save_snapshot(snapshot: "InterruptSnapshot", output_dir: str = ".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{snapshot.snapshot_id}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(snapshot), f, ensure_ascii=False, indent=2)
    return filepath
