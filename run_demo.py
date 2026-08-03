"""
Work Assistant Agent - 主入口

模式：
  1. snapshot   采集快照 + LLM中断摘要（原有功能）
  2. new-task   新建任务：输入描述 → LLM生成计划 → 采集快照 → 进度分析
  3. check      评估已有项目：对话补充上下文 → LLM推断计划 → 进度分析
  4. progress   对已有任务重新分析进度（传 task_id）

用法：
  python run_demo.py snapshot
  python run_demo.py new-task
  python run_demo.py check
  python run_demo.py progress <task_id>
  python run_demo.py           # 默认 snapshot 模式
"""

import sys
import dataclasses
from pathlib import Path
from snapshot_collector import collect_snapshot, save_snapshot
from llm_analyzer import analyze_snapshot
from task_manager import (
    init_db, list_tasks, load_task,
    create_task_via_llm, init_existing_project_via_llm,
)
from progress_analyzer import analyze_and_persist_progress, print_progress_report
from settings import get_settings

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "─" * 54

DEFAULT_PROJECTS = list(get_settings().default_project_paths)


# ──────────────────────────────────────────────
# 快照打印（保留原有功能）
# ──────────────────────────────────────────────

def pretty_print(s: dict):
    print(f"\n{'=' * 58}")
    print(f"  快照 ID   : {s['snapshot_id']}")
    print(f"  中断时间  : {s['interrupt_time']}")
    print(f"  中断原因  : {s['interrupt_reason']}")
    print(f"  项目数量  : {len(s['projects'])} 个")
    print(f"  累计工时  : {s['total_work_hours']} 小时")

    for proj in s["projects"]:
        print(f"\n{'─'*58}")
        print(f"  [{proj['project_type'].upper()}] {proj['project_name']}")
        print(f"  Remote : {proj['remote_url']}")
        print(f"  分支   : {proj['branch']}")
        if proj["commits"]:
            c = proj["commits"][0]
            print(f"  最新提交: [{c['hash']}] {c['date'][:10]}  {c['message']}")
        print(f"  历史共 : {len(proj['commits'])} 次提交")

        print(f"\n  模块视图（共 {len(proj['modules'])} 个）")
        status_tag = {"committed": "[已提交]", "partial": "[部分]", "untracked": "[未跟踪]"}
        for m in proj["modules"][:15]:
            tag     = status_tag.get(m["git_status"], m["git_status"])
            present = "/".join(m["layers_present"]) or "-"
            missing = "  缺: " + "/".join(m["layers_missing"]) if m["layers_missing"] else ""
            print(f"    {m['last_modified'][:16]}  {tag:10}  {m['module_name']}")
            print(f"      层: {present}{missing}")
        if len(proj["modules"]) > 15:
            print(f"    ... 共 {len(proj['modules'])} 个模块，完整内容见 JSON")

        if proj["uncommitted_tracked"]:
            print(f"\n  未提交已跟踪（{len(proj['uncommitted_tracked'])} 个）")
            for c in proj["uncommitted_tracked"][:8]:
                print(f"    [{c['status']}] {c['path']}")

        if proj["diff_unstaged"]:
            print(f"\n  diff 未 staged（前10行）")
            for line in proj["diff_unstaged"].splitlines()[:10]:
                print(f"    {line}")
            total = len(proj["diff_unstaged"].splitlines())
            if total > 10:
                print(f"    ... 共 {total} 行，完整内容见 JSON")

    print(f"\n{'─'*58}")
    print(f"  工作会话（共 {len(s['work_sessions'])} 次，总计 {s['total_work_hours']}h）")
    for ws in s["work_sessions"][-15:]:
        dur = ws["duration_minutes"]
        dur_str = f"{dur:.0f}分钟" if dur < 60 else f"{dur/60:.1f}小时"
        touched = ws["projects_touched"]
        if touched:
            parts = []
            for pname, mods in touched.items():
                parts.append(f"{pname}: {', '.join(mods[:3])}{'...' if len(mods) > 3 else ''}")
            touched_str = "  |  ".join(parts)
        else:
            touched_str = "无文件改动"
        print(f"  [{ws['session_index']:>3}] {ws['started_at'][:16]}  {dur_str:8}  {touched_str}")

    print(f"\n{'─'*58}")
    work_hours   = [f for f in s["idea_changed_files"] if "工时" in f]
    changed_list = [f for f in s["idea_changed_files"] if "变更" in f]
    for h in work_hours:
        print(f"  {h}")
    print(f"  IDEA 变更文件（{len(changed_list)} 个）")
    for f in changed_list[:10]:
        print(f"    {f.replace('[IDEA变更] ', '')}")

    print(f"\n{'─'*58}")
    print("  任务进度")
    prog = s.get("progress")
    if prog:
        print(f"  当前: {prog['current_step']}")
        print(f"  已完成: {' > '.join(prog['completed_steps']) or '（无）'}")
        print(f"  待完成: {' > '.join(prog['pending_steps']) or '（无）'}")
        if prog["notes"]:
            print(f"  备注: {prog['notes']}")
    else:
        print("  （未找到 .agent/progress.json）")

    print(f"\n{'─'*58}")
    print(f"  Shell 历史（最近5条）")
    for cmd in s["terminal_history"][-5:]:
        print(f"    $ {cmd}")
    print(f"  开发进程: {', '.join(s['dev_processes']) or '（无）'}")
    print(f"{'=' * 58}\n")


# ──────────────────────────────────────────────
# 快照采集（公共步骤）
# ──────────────────────────────────────────────

def do_collect(project_paths: list, task_id: str = "TASK-MANUAL", reason: str = "手动触发") -> tuple:
    print(f"\n[采集快照] 项目: {project_paths}")
    snapshot = collect_snapshot(
        task_id=task_id,
        project_paths=project_paths,
        idea_project_path=project_paths[0],
        interrupt_reason=reason,
    )
    s_dict = dataclasses.asdict(snapshot)
    output_path = save_snapshot(snapshot, output_dir=str(get_settings().snapshot_dir))
    print(f"快照已保存: {output_path}")
    return s_dict, output_path


# ──────────────────────────────────────────────
# 模式：snapshot（原有）
# ──────────────────────────────────────────────

def mode_snapshot(project_paths: list):
    s_dict, output_path = do_collect(project_paths)
    pretty_print(s_dict)

    print(f"\n{SEP}")
    print("正在生成中断摘要...")
    try:
        analysis = analyze_snapshot(s_dict)
        print(analysis)
        analysis_path = output_path.replace(".json", "_analysis.txt")
        Path(analysis_path).write_text(analysis, encoding="utf-8")
        print(f"\n摘要已保存: {analysis_path}")
    except Exception as e:
        print(f"LLM 分析失败: {e}")


# ──────────────────────────────────────────────
# 模式：new-task
# ──────────────────────────────────────────────

def mode_new_task(project_paths: list):
    print("\n=== 新建任务 ===")
    print(f"项目路径: {project_paths}")
    print("\n请描述你的任务目标（输入后回车）：")
    description = input("> ").strip()
    if not description:
        print("未输入描述，退出。")
        return

    print("\n[LLM] 正在解析任务并生成执行计划...")
    try:
        task = create_task_via_llm(description, project_paths)
    except Exception as e:
        print(f"任务创建失败: {e}")
        return

    print(f"\n任务已创建: {task.task_id}")
    print(f"名称: {task.task_name}")
    print(f"目标: {task.goal}")
    print(f"优先级: {task.priority}")
    print(f"\n执行计划（{len(task.plan)} 步）:")
    for s in task.plan:
        print(f"  {s.step_index+1}. {s.step_name}")
        print(f"     {s.description}")
        for sub in s.sub_tasks:
            print(f"     - {sub}")

    print(f"\n{SEP}")
    print("正在采集当前快照并分析初始进度...")
    s_dict, output_path = do_collect(project_paths, task_id=task.task_id, reason="任务创建")

    try:
        report = analyze_and_persist_progress(task, s_dict)
        print_progress_report(report, task)
        report_path = output_path.replace(".json", "_progress.json")
        Path(report_path).write_text(
            __import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"进度报告已保存: {report_path}")
    except Exception as e:
        print(f"进度分析失败: {e}")


# ──────────────────────────────────────────────
# 模式：check（评估已有项目）
# ──────────────────────────────────────────────

def mode_check(project_paths: list):
    print("\n=== 评估已有项目 ===")
    print(f"项目路径: {project_paths}")
    print("\n请简单描述这个项目在做什么，以及你目前的目标（多行输入，空行结束）：")

    lines = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        lines.append(line)
    context = "\n".join(lines)

    if not context:
        print("未输入上下文，退出。")
        return

    print("\n[采集] 正在采集项目快照...")
    s_dict, output_path = do_collect(project_paths, reason="已有项目评估")

    print("\n[LLM] 正在根据上下文和快照推断任务计划...")
    try:
        task = init_existing_project_via_llm(context, project_paths, s_dict)
    except Exception as e:
        print(f"任务初始化失败: {e}")
        return

    print(f"\n任务已创建: {task.task_id}")
    print(f"名称: {task.task_name}")
    print(f"推断目标: {task.goal}")
    print(f"\n推断执行计划（{len(task.plan)} 步）:")
    for s in task.plan:
        print(f"  {s.step_index+1}. {s.step_name}")
        print(f"     {s.description}")

    print(f"\n{SEP}")
    print("正在分析当前进度...")
    try:
        report = analyze_and_persist_progress(task, s_dict)
        print_progress_report(report, task)
        report_path = output_path.replace(".json", "_progress.json")
        Path(report_path).write_text(
            __import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"进度报告已保存: {report_path}")
    except Exception as e:
        print(f"进度分析失败: {e}")


# ──────────────────────────────────────────────
# 模式：progress（已有任务重新分析）
# ──────────────────────────────────────────────

def mode_progress(task_id: str, project_paths: list):
    task = load_task(task_id)
    if not task:
        # 列出已有任务供参考
        tasks = list_tasks()
        if tasks:
            print(f"未找到 task_id={task_id}，已有任务：")
            for t in tasks:
                print(f"  {t.task_id}  [{t.status}]  {t.task_name}")
        else:
            print(f"未找到 task_id={task_id}，且数据库中无任务记录。")
        return

    paths = task.project_paths if task.project_paths else project_paths
    print(f"\n[任务] {task.task_name}  ({task.status})")
    s_dict, output_path = do_collect(paths, task_id=task_id, reason="进度检查")

    print("\n[LLM] 正在分析进度...")
    try:
        report = analyze_and_persist_progress(task, s_dict)
        print_progress_report(report, task)
    except Exception as e:
        print(f"进度分析失败: {e}")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    args = sys.argv[1:]
    mode = args[0] if args else "snapshot"

    # 项目路径：mode 后面的参数，或默认
    path_args = args[1:] if len(args) > 1 else []

    # progress 模式第一个参数是 task_id
    if mode == "progress":
        task_id_arg = path_args[0] if path_args else ""
        extra_paths = path_args[1:] if len(path_args) > 1 else DEFAULT_PROJECTS
        mode_progress(task_id_arg, extra_paths)
    else:
        project_paths = path_args if path_args else DEFAULT_PROJECTS
        if not project_paths:
            print("未配置默认项目路径。请先访问 Web /setup，或在命令后传入项目路径。")
            sys.exit(1)
        if mode == "snapshot":
            mode_snapshot(project_paths)
        elif mode == "new-task":
            mode_new_task(project_paths)
        elif mode == "check":
            mode_check(project_paths)
        else:
            print(f"未知模式: {mode}")
            print("可用模式: snapshot | new-task | check | progress <task_id>")
