"""本地生成工作项上下文包，并供 Codex/Claude 新会话使用。"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from work_item_discovery import redact_sensitive_text
from workspace_store import WorkspaceStore, WorkspaceStoreError, workspace_store


MAX_TOTAL_SOURCE_CHARS = 24_000
MAX_MESSAGE_CHARS = 2_000
CONTINUABLE_STATUSES = {"suggested", "backlog", "planned", "in_progress", "blocked"}
MAX_HISTORY_BRIEF_CHARS = 6_000
MAX_HISTORY_FALLBACK_CHARS = 900

# 这些字段来自工作项 metadata；值可以是字符串、列表，或显式写为 not_applicable。
INFORMATION_REQUIREMENTS = {
    "feature": (
        ("验收标准", "acceptance_criteria", "说明用户可见的完成条件、关键场景和边界情况"),
        ("关键业务指标", "key_metrics", "说明要改善或保护的指标；若确实不适用，请明确写 not_applicable"),
    ),
    "function": (
        ("验收标准", "acceptance_criteria", "说明输入、输出、异常处理和完成判定"),
        ("关键业务指标", "key_metrics", "说明成功指标；若确实不适用，请明确写 not_applicable"),
    ),
    "bug": (
        ("复现与期望行为", "reproduction_steps", "提供复现步骤、实际结果、期望结果和影响范围"),
        ("回归验收标准", "acceptance_criteria", "说明用于证明修复有效的测试或场景"),
    ),
    "refactor": (
        ("不变约束", "non_functional_constraints", "说明必须保持的外部行为、兼容性或性能约束"),
        ("回归验收标准", "acceptance_criteria", "说明需要通过的测试、性能基线或人工验收场景"),
    ),
    "maintenance": (
        ("验收标准", "acceptance_criteria", "说明维护完成的判定和需要验证的范围"),
    ),
    "research": (
        ("研究问题与输出形式", "acceptance_criteria", "说明需要回答的问题、结论格式和决策依据"),
    ),
}


class WorkItemContextError(WorkspaceStoreError):
    pass


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-") or "work-item"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n…（此消息已截断）"


def _markdown_code(value: str) -> str:
    return "```text\n" + value.replace("```", "``\\`") + "\n```"


class WorkItemContextService:
    def __init__(self, store: WorkspaceStore | None = None, context_root: str | Path | None = None):
        self.store = store or workspace_store
        default_root = self.store.db_path.parent / "contexts"
        self.context_root = Path(context_root or default_root).resolve()

    def _context_path(self, work_item_id: str, version: int) -> Path:
        return self.context_root / _slug(work_item_id) / f"context-v{version}.md"

    @staticmethod
    def _project_handoff_path(project: dict, work_item_id: str) -> Path | None:
        """返回 Agent 已打开项目时可读取的交接文件位置。"""
        roots = project.get("roots") or []
        primary_root = next((root for root in roots if root.get("is_primary")), None)
        root = primary_root or (roots[0] if roots else None)
        if not root or not str(root.get("path") or "").strip():
            return None
        return Path(root["path"]).expanduser().resolve() / ".work-assistant" / "handoff" / f"{_slug(work_item_id)}.md"

    def _write_project_handoff(self, project: dict, work_item_id: str, content: str) -> tuple[str, str]:
        """导出脱敏上下文到项目内，供 ChatGPT Codex 新会话主动读取。"""
        path = self._project_handoff_path(project, work_item_id)
        if not path:
            return "", "项目没有可用目录，无法生成项目内交接文件"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            return "", f"无法写入项目内交接文件：{exc}"
        return str(path), ""

    def _next_version(self, work_item_id: str) -> int:
        latest = self.store.get_latest_context_package(work_item_id)
        return (latest["version"] + 1) if latest else 1

    def _build_content(self, work_item: dict, project: dict, sources: list[dict]) -> tuple[str, int]:
        lines = [
            f"# 工作项上下文：{work_item['title']}",
            "",
            "> 此文件由 Work Assistant 在本地生成。开始工作前先阅读；实际代码状态仍以项目目录为准。",
            f"> 工作项 ID：`{work_item['work_item_id']}`。",
            "",
            "## 工作项",
            f"- 项目：{project['name']}（`{project['project_id']}`）",
            f"- 类型：{work_item['item_type']}；状态：{work_item['status']}；优先级：{work_item['priority']}",
            f"- 目标：{work_item['goal'] or '未填写'}",
            f"- 描述：{work_item['description'] or '未填写'}",
        ]
        if work_item.get("deadline"):
            lines.append(f"- 截止日期：{work_item['deadline']}")
        if project.get("roots"):
            lines.extend(["", "## 项目目录"])
            for root in project["roots"]:
                lines.append(f"- `{root['path']}`")

        steps = work_item.get("steps") or []
        if steps:
            lines.extend(["", "## 当前计划"])
            for step in steps:
                marker = {"completed": "x", "skipped": "-", "in_progress": ">", "pending": " "}.get(step["status"], " ")
                lines.append(f"- [{marker}] {step['title']}{(' — ' + step['description']) if step.get('description') else ''}")

        next_step = next(
            (step for step in steps if step.get("status") not in {"completed", "skipped"}),
            None,
        )
        lines.extend(["", "## 建议续接方式"])
        if work_item["status"] == "suggested":
            lines.append("- 该工作项仍待确认；先核对来源片段与范围，再决定确认、合并或忽略。")
        elif next_step:
            lines.append(f"- 先处理计划中的下一步：{next_step['title']}。")
        else:
            lines.append("- 先核对项目当前代码与来源片段，再明确下一步实现或验证动作。")

        redaction_count = 0
        remaining = MAX_TOTAL_SOURCE_CHARS
        lines.extend(["", "## 来源会话片段"])
        if not sources:
            lines.append("- 尚未关联历史会话片段；此上下文仅包含手工填写的工作项信息。")
        for index, source in enumerate(sources, 1):
            conversation = source.get("conversation") or {}
            label = {"codex": "Codex", "claude": "Claude", "work_assistant": "Work Assistant"}.get(
                conversation.get("source"), conversation.get("source", "未知来源")
            )
            lines.extend([
                "",
                f"### 片段 {index}：{source.get('title') or conversation.get('title') or '未命名片段'}",
                f"- 来源：{label}；会话：`{conversation.get('conversation_id', '')}`",
                f"- 时间：{conversation.get('started_at') or source.get('created_at') or '未知'}",
            ])
            if source.get("summary"):
                lines.append(f"- 摘要：{source['summary']}")
            for message in source.get("messages") or []:
                if remaining <= 0:
                    break
                result = redact_sensitive_text(message.get("content", ""))
                redaction_count += result.total
                content = _truncate(result.text, min(MAX_MESSAGE_CHARS, remaining))
                remaining -= len(content)
                role = {"user": "用户", "assistant": "AI", "system": "系统"}.get(
                    message.get("role"), message.get("role", "消息")
                )
                lines.extend([f"#### {role}", _markdown_code(content)])
            if remaining <= 0:
                lines.extend(["", "_来源消息达到上下文长度上限，后续内容未写入此版本。_"])
                break
        lines.extend([
            "",
            "## 使用规则",
            "- 不要把此上下文中的假设当作已验证事实；先检查当前代码、测试和 Git 状态。",
            "- 不要输出或恢复 `[REDACTED:*]` 占位符所代表的敏感凭据。",
            "- 完成关键变更后，更新工作项计划或重新生成上下文包。",
            "",
        ])
        return "\n".join(lines), redaction_count

    def generate(self, work_item_id: str) -> dict:
        work_item = self.store.get_work_item(work_item_id)
        if not work_item:
            raise WorkItemContextError("工作项不存在")
        project = self.store.get_project(work_item["project_id"])
        if not project:
            raise WorkItemContextError("所属项目不存在")
        sources = self.store.list_work_item_source_details(work_item_id)
        content, redaction_count = self._build_content(work_item, project, sources)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        latest = self.store.get_latest_context_package(work_item_id)
        source_ids = [source["segment_id"] for source in sources]
        if (
            latest and latest["content_hash"] == content_hash
            and set(latest["segment_ids"]) == set(source_ids) and Path(latest["canonical_path"]).is_file()
        ):
            handoff_path, handoff_error = self._write_project_handoff(project, work_item_id, content)
            return {
                **latest, "created": False, "redaction_count": redaction_count,
                "handoff_path": handoff_path, "handoff_error": handoff_error,
            }

        version = self._next_version(work_item_id)
        path = self._context_path(work_item_id, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        try:
            package = self.store.create_context_package(
                work_item_id,
                canonical_path=str(path),
                content_hash=content_hash,
                segment_ids=source_ids,
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        handoff_path, handoff_error = self._write_project_handoff(project, work_item_id, content)
        return {
            **package, "created": True, "redaction_count": redaction_count,
            "handoff_path": handoff_path, "handoff_error": handoff_error,
        }

    def read_content(self, context_id: str) -> tuple[dict, str]:
        package = self.store.get_context_package(context_id)
        if not package:
            raise WorkItemContextError("上下文包不存在")
        path = Path(package["canonical_path"]).resolve()
        try:
            path.relative_to(self.context_root)
        except ValueError as exc:
            raise WorkItemContextError("上下文包路径不安全") from exc
        if not path.is_file():
            raise WorkItemContextError("上下文文件已丢失，请重新生成")
        return package, path.read_text(encoding="utf-8")

    @staticmethod
    def _git_progress(project: dict) -> list[str]:
        """采集轻量、只读的 Git 状态，帮助 Agent 从真实代码状态继续。"""
        lines: list[str] = []
        for root in project.get("roots", []):
            root_path = Path(root["path"])
            if not root_path.is_dir():
                lines.append(f"- `{root_path}`：目录当前不可访问")
                continue
            try:
                status = subprocess.run(
                    ["git", "status", "--short"], cwd=root_path, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=5, check=False,
                )
                latest = subprocess.run(
                    ["git", "log", "-1", "--oneline"], cwd=root_path, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=5, check=False,
                )
            except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                lines.append(f"- `{root_path}`：未能读取 Git 状态；请在开始前自行执行 `git status`。")
                continue
            if status.returncode != 0:
                lines.append(f"- `{root_path}`：不是可读取的 Git 仓库；请在开始前检查目录。")
                continue
            changed = [line for line in status.stdout.splitlines() if line.strip()]
            latest_line = latest.stdout.strip() if latest.returncode == 0 else "未读取到提交记录"
            lines.append(f"- `{root_path}`：最近提交 `{latest_line}`；未提交变更 {len(changed)} 个文件。")
            lines.extend(f"  - `{line}`" for line in changed[:30])
            if len(changed) > 30:
                lines.append(f"  - …其余 {len(changed) - 30} 个变更文件未列出")
        return lines or ["- 项目尚未配置代码目录；先确认工作目录。"]

    @staticmethod
    def _metadata_value(metadata: dict, key: str) -> str:
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            value = "；".join(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, dict):
            value = "；".join(
                f"{name}：{detail}" for name, detail in value.items() if str(detail).strip()
            )
        return str(value or "").strip()

    def _information_gaps(self, work_item: dict) -> list[dict]:
        """只标注开发前真正影响范围、验收或决策的信息，不把每项都伪装成必填。"""
        metadata = work_item.get("metadata") or {}
        gaps: list[dict] = []
        if not (work_item.get("goal") or work_item.get("description")):
            gaps.append({
                "label": "目标与范围", "key": "goal",
                "guidance": "说明要交付什么、哪些内容不在本次范围内。",
            })
        for label, key, guidance in INFORMATION_REQUIREMENTS.get(
            work_item.get("item_type"), INFORMATION_REQUIREMENTS["maintenance"]
        ):
            value = self._metadata_value(metadata, key)
            if not value:
                gaps.append({"label": label, "key": key, "guidance": guidance})
        return gaps

    def _history_brief(self, sources: list[dict]) -> tuple[list[str], int]:
        """优先使用语义片段摘要；缺失时只摘取最近关键对话，避免提示词退化为原始日志。"""
        if not sources:
            return ["- 未关联来源会话；先结合当前代码确认历史决策。"], 0
        lines: list[str] = []
        redaction_count = 0
        remaining = MAX_HISTORY_BRIEF_CHARS
        for index, source in enumerate(sources, 1):
            if remaining <= 0:
                lines.append("- …其余历史片段未展开；如有需要请读取下方的完整本机上下文文件。")
                break
            conversation = source.get("conversation") or {}
            source_name = {"codex": "Codex", "claude": "Claude", "work_assistant": "Work Assistant"}.get(
                conversation.get("source"), conversation.get("source", "未知来源")
            )
            title = source.get("title") or conversation.get("title") or "未命名会话片段"
            summary = str(source.get("summary") or "").strip()
            if not summary:
                candidates = [
                    str(message.get("content", "")).strip()
                    for message in (source.get("messages") or [])
                    if message.get("role") in {"user", "assistant"} and str(message.get("content", "")).strip()
                ]
                summary = "\n".join(candidates[-2:]) or "该片段未提供可读摘要。"
            result = redact_sensitive_text(_truncate(summary, min(MAX_HISTORY_FALLBACK_CHARS, remaining)))
            redaction_count += result.total
            remaining -= len(result.text)
            lines.extend([
                f"### 历史片段 {index}：{title}",
                f"- 来源：{source_name}；时间：{conversation.get('started_at') or source.get('created_at') or '未知'}",
                f"- 摘要：{result.text}",
            ])
        return lines, redaction_count

    def generate_continue_prompt(self, work_item_id: str) -> dict:
        """生成可直接粘贴到 Codex/Claude Code 的续接提示词，不调用模型。"""
        work_item = self.store.get_work_item(work_item_id)
        if not work_item:
            raise WorkItemContextError("工作项不存在")
        if work_item["status"] not in CONTINUABLE_STATUSES:
            raise WorkItemContextError("只有待确认、待办、已规划、进行中或阻塞的工作项可以生成继续开发提示词")
        project = self.store.get_project(work_item["project_id"])
        if not project:
            raise WorkItemContextError("所属项目不存在")

        package = self.generate(work_item_id)
        handoff_path = package.get("handoff_path") or package["canonical_path"]
        handoff_note = (
            "该文件已写入项目目录，ChatGPT App 的 Codex 打开此项目后可以读取。"
            if package.get("handoff_path") else
            "项目内交接文件未生成；请将下方提示词和上下文包内容一并粘贴到新会话。"
        )
        sources = self.store.list_work_item_source_details(work_item_id)
        history_lines, history_redaction_count = self._history_brief(sources)
        information_gaps = self._information_gaps(work_item)
        suggested_note = (
            "该工作项仍待确认：先核对来源、范围和验收口径；除非用户明确确认，不要直接实施代码变更。"
            if work_item["status"] == "suggested" else ""
        )
        steps = work_item.get("steps") or []
        completion = work_item.get("completion_percent")
        progress_lines = [
            f"- 工作项状态：{work_item['status']}；计划完成度：{completion if completion is not None else '尚未建立步骤'}",
            f"- 最近更新：{work_item.get('updated_at') or '未知'}",
        ]
        if steps:
            for step in steps:
                marker = {"completed": "已完成", "skipped": "跳过", "in_progress": "进行中", "pending": "待处理"}.get(step["status"], step["status"])
                progress_lines.append(f"- [{marker}] {step['title']}{('：' + step['description']) if step.get('description') else ''}")
        else:
            progress_lines.append("- 尚未建立可量化步骤；请先基于当前代码和目标确认下一步。")

        information_lines = [
            "- 业务信息检查通过：仍应在修改前核对当前代码、Git 状态和测试。"
        ]
        if information_gaps:
            information_lines = [
                f"- ⚠ [需要用户补充：{gap['label']}] {gap['guidance']}"
                for gap in information_gaps
            ]

        prompt = "\n".join([
            f"# 继续开发：{work_item['title']}",
            "",
            "请作为该项目的编程 Agent 继续开发。先读取下方指定的 Work Assistant 交接文件，再检查当前代码、Git 状态和相关测试；不要假设历史结论仍然成立。",
            f"Work Assistant 工作项 ID：`{work_item['work_item_id']}`。",
            "",
            "## 本次目标",
            f"- 项目：{project['name']}",
            f"- 工作项：{work_item['title']}（{work_item['item_type']}，优先级 {work_item['priority']}）",
            f"- 交付目标：{work_item['goal'] or work_item['description'] or '请先根据上下文确认具体交付物'}",
            *([f"- ⚠ {suggested_note}"] if suggested_note else []),
            "",
            "## 当前任务进度",
            *progress_lines,
            "",
            "## 刚采集的本地代码状态",
            *self._git_progress(project),
            "",
            "## 开始前需确认的业务信息",
            *information_lines,
            "",
            "## 已压缩的 Codex / Claude 历史上下文",
            *history_lines,
            "",
            "## 执行要求",
            "1. 若上方列出了待补充业务信息，先向用户提出最少必要问题；未经确认不要擅自把业务指标或验收口径当作事实。",
            "2. 先阅读相关实现、测试与未提交变更，确认本工作项的真实边界。",
            "3. 不要覆盖、丢弃或格式化与此工作项无关的现有改动。",
            "4. 完成后运行与改动相符的测试或验证，并说明修改了什么、验证结果和仍需用户决策的事项。",
            "5. 如上下文与当前代码冲突，以当前代码和 Git 状态为准，并明确指出差异。",
            "",
            "## 完整本机上下文附件",
            f"- 请优先读取：`{handoff_path}`",
            f"- {handoff_note}",
            "- 附件由 Work Assistant 本机生成，包含更完整的脱敏来源片段；不要把其中的假设视为已验证事实。",
        ])
        redacted = redact_sensitive_text(prompt)
        return {
            "work_item_id": work_item_id,
            "project_id": project["project_id"],
            "prompt": redacted.text,
            "context_package": package,
            "handoff_path": handoff_path,
            "handoff_available": bool(package.get("handoff_path")),
            "handoff_error": package.get("handoff_error", ""),
            "missing_information": information_gaps,
            "redaction_count": package.get("redaction_count", 0) + history_redaction_count + redacted.total,
        }
