"""
进度分析器：对比任务计划和当前快照，推断进度并输出结构化报告。
"""

import json
from pathlib import Path

from settings import get_api_key, get_settings
from streaming_runtime import stream_chat_completion_text

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装：pip install openai")


def _load_api_key() -> str:
    return get_api_key()


def _load_prompt() -> str:
    return (Path(__file__).parent / "progress_analysis_prompt.txt").read_text(encoding="utf-8")


def analyze_progress(
    task: "Task",
    snapshot: dict,
    model: str | None = None,
) -> dict:
    """
    对比 task.plan 和当前快照，返回进度分析结果。

    返回结构：
    {
        "current_step_index": int,
        "current_step_name": str,
        "completion_percent": int,        # 0-100
        "step_statuses": [                # 每步的判断
            {"step_index": 0, "step_name": "...", "status": "completed|in_progress|pending", "evidence": "..."},
            ...
        ],
        "summary": "一句话进度描述",
        "next_action": "建议的下一步操作",
        "risks": ["风险1", "风险2"]
    }
    """
    from task_manager import Task

    settings = get_settings()
    api_key = _load_api_key()
    model = model or settings.amd_model
    sys_prompt = _load_prompt()

    # 构建计划摘要
    plan_summary = [
        {
            "step_index": s.step_index,
            "step_name": s.step_name,
            "description": s.description,
            "current_status": s.status,
            "sub_tasks": s.sub_tasks,
        }
        for s in task.plan
    ]

    # 构建快照摘要（控制token）
    snapshot_summary = {
        "interrupt_time": snapshot.get("interrupt_time"),
        "projects": [
            {
                "name": p["project_name"],
                "type": p["project_type"],
                "branch": p["branch"],
                "latest_commit": (
                    p["commits"][0]
                    if settings.send_commit_info and p["commits"]
                    else None
                ),
                "modules": [
                    {
                        "name": m["module_name"],
                        "layers_present": m["layers_present"],
                        "layers_missing": m["layers_missing"],
                        "git_status": m["git_status"],
                        "last_modified": m["last_modified"],
                    }
                    for m in p["modules"][:15]
                ],
                "uncommitted_count": len(p["uncommitted_tracked"]),
                "diff_summary": (
                    _summarize_diff(p.get("diff_unstaged", "") + p.get("diff_staged", ""))
                    if settings.send_diff_summary
                    else ""
                ),
            }
            for p in snapshot.get("projects", [])
        ],
        "progress_file": snapshot.get("progress"),
        "work_sessions_count": len(snapshot.get("work_sessions", [])),
        "total_work_hours": snapshot.get("total_work_hours", 0),
    }

    user_content = (
        f"任务目标：{task.goal}\n\n"
        f"执行计划：\n{json.dumps(plan_summary, ensure_ascii=False, indent=2)}\n\n"
        f"当前快照：\n{json.dumps(snapshot_summary, ensure_ascii=False, indent=2)}"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=settings.amd_base_url,
    )
    raw = stream_chat_completion_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=settings.request_timeout_seconds,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[:-1])

    return json.loads(raw)


def analyze_and_persist_progress(
    task: "Task",
    snapshot: dict,
    model: str | None = None,
) -> dict:
    """分析当前进度，将结果持久化后返回数据库口径的报告。"""
    from task_manager import apply_progress_report

    report = analyze_progress(task, snapshot, model=model)
    return apply_progress_report(task.task_id, report)


def _summarize_diff(diff_text: str, max_lines: int = 30) -> str:
    if not diff_text:
        return ""
    lines = diff_text.splitlines()
    # 只保留 +/- 变更行和文件名行
    important = [l for l in lines if l.startswith(("+++", "---", "@@", "+", "-")) and not l.startswith(("+++", "---"))]
    kept = important[:max_lines]
    if len(important) > max_lines:
        kept.append(f"...（共 {len(important)} 行变更）")
    return "\n".join(kept)


def print_progress_report(report: dict, task: "Task"):
    """格式化打印进度报告"""
    SEP = "─" * 54
    print(f"\n{'=' * 54}")
    print(f"  任务：{task.task_name}")
    print(f"  目标：{task.goal}")
    print(f"{'=' * 54}")

    pct = report.get("completion_percent", 0)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  进度  [{bar}] {pct}%")
    print(f"  摘要  {report.get('summary', '')}")

    print(f"\n{SEP}")
    print("  各步骤状态")
    status_icon = {"completed": "[✓]", "in_progress": "[→]", "pending": "[ ]", "skipped": "[–]"}
    for s in report.get("step_statuses", []):
        icon = status_icon.get(s["status"], "[ ]")
        print(f"  {icon} {s['step_index']+1}. {s['step_name']}")
        if s.get("evidence"):
            print(f"       └ {s['evidence']}")

    print(f"\n{SEP}")
    print(f"  当前步骤：{report.get('current_step_name', '-')}")
    print(f"  下一步  ：{report.get('next_action', '-')}")

    risks = report.get("risks", [])
    if risks:
        print(f"\n{SEP}")
        print("  风险提示")
        for r in risks:
            print(f"  ! {r}")

    print(f"{'=' * 54}\n")
