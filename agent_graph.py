"""
Work Assistant Agent - LangGraph 主图

状态流：
  用户输入
    → [route] 意图识别
        → new_task:    [gather_task_info] 追问 → [create_task] 生成计划
        → check:       [gather_task_info] 追问 → [init_existing] 推断计划
        → progress:    直接进入 [collect_snapshot]
        → chat:        [chat_reply] 直接回复
    → [collect_snapshot] 采集快照
    → [analyze_progress] 进度分析
    → [format_output] 格式化输出
    → END
"""

import json
import dataclasses
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from task_manager import (
    init_db, load_task, list_tasks, save_task,
    create_task_via_llm, init_existing_project_via_llm,
    Task,
)
from snapshot_collector import collect_snapshot, save_snapshot
from progress_analyzer import analyze_and_persist_progress

import os
from pathlib import Path
from settings import get_api_key, get_settings

# ──────────────────────────────────────────────
# LLM 客户端
# ──────────────────────────────────────────────

def _load_api_key() -> str:
    return get_api_key()


def get_llm(model: str | None = None) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.amd_model,
        api_key=_load_api_key(),
        base_url=settings.amd_base_url,
        temperature=0.1,
        streaming=True,
        timeout=settings.request_timeout_seconds,
    )


# ──────────────────────────────────────────────
# 全局状态定义
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    # 对话历史（LangGraph 用 add_messages 自动追加）
    messages: Annotated[list, add_messages]

    # 意图：new_task | check | progress | chat
    intent: str

    # 信息收集阶段积累的上下文
    gathered_info: dict          # {"description": ..., "project_paths": [...], "context": ...}
    info_ready: bool             # 信息是否足够，可以进入下一步

    # 任务对象（创建/加载后填入）
    task: Optional[dict]         # Task asdict 后的字典（TypedDict 不支持 dataclass）
    task_id: Optional[str]

    # 快照
    snapshot: Optional[dict]
    snapshot_path: Optional[str]

    # 进度报告
    progress_report: Optional[dict]

    # 最终输出文本
    output: str

    # 错误信息
    error: Optional[str]


# ──────────────────────────────────────────────
# 节点：路由（意图识别）
# ──────────────────────────────────────────────

ROUTE_PROMPT = """你是一个开发助手的意图识别器。根据用户输入判断意图，只输出以下四个词之一，不要输出其他任何内容：

- new_task   : 用户想新建一个任务/功能/需求
- check      : 用户想让agent评估/接管一个已有的项目或已在进行的工作
- progress   : 用户想查看某个已有任务的当前进度
- chat       : 其他问题或闲聊

示例：
"我想做一个用户登录功能" → new_task
"帮我看看我这个项目做到哪了" → check
"quantitativeInvestment 任务现在进度怎样" → progress
"LangGraph 和 LangChain 有什么区别" → chat
"""

def node_route(state: AgentState) -> dict:
    """意图识别，决定走哪条分支。

    如果上一轮已经识别了意图且还在收集信息中（info_ready=False, intent已设置），
    直接沿用上一轮的意图，不重新路由，避免把用户补充的信息误判为新意图。
    """
    existing_intent = state.get("intent", "")
    if existing_intent in ("new_task", "check") and not state.get("info_ready", True):
        # 还在信息收集阶段，保持原意图
        return {}

    llm = get_llm()
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    response = llm.invoke([
        SystemMessage(content=ROUTE_PROMPT),
        HumanMessage(content=last_user_msg),
    ])
    intent = response.content.strip().lower()
    if intent not in ("new_task", "check", "progress", "chat"):
        intent = "chat"

    return {"intent": intent, "info_ready": False, "gathered_info": {}}


# ──────────────────────────────────────────────
# 节点：信息收集（new_task 和 check 共用）
# ──────────────────────────────────────────────

GATHER_PROMPT = """你是一个开发任务助手，正在帮用户初始化任务信息。

你需要收集以下信息：
1. 项目路径（至少一个，如 C:/workspace/myproject）
2. 任务描述或目标（如果是新任务）或当前工作背景（如果是已有项目）

规则：
- 如果用户已提供了项目路径和描述/背景，输出 JSON：
  {"ready": true, "project_paths": ["路径1", "路径2"], "description": "...", "context": "..."}
- 如果信息不足，输出 JSON：
  {"ready": false, "question": "你需要追问用户的问题"}

只输出 JSON，不要输出其他内容。
"""

def node_gather_info(state: AgentState) -> dict:
    """
    判断是否已收集到足够信息。
    不够则生成追问并设 info_ready=False（本轮结束，等用户下一轮回复）。
    够了则设 info_ready=True，让图继续往下走。
    """
    llm = get_llm()

    configured_paths = list(get_settings().default_project_paths)
    gather_prompt = GATHER_PROMPT
    if configured_paths:
        gather_prompt += (
            "\n用户已在本地设置中配置默认项目路径："
            f"{json.dumps(configured_paths, ensure_ascii=False)}。"
            "若用户没有另行指定路径，可以使用这些默认路径。"
        )

    response = llm.invoke([
        SystemMessage(content=gather_prompt),
        *state["messages"],
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[:-1])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "messages": [AIMessage(content="请告诉我项目路径和任务描述，例如：项目在 C:/workspace/myproject，我想实现用户登录功能。")],
            "info_ready": False,
        }

    if parsed.get("ready"):
        return {
            "gathered_info": {
                "project_paths": parsed.get("project_paths", []),
                "description":   parsed.get("description", ""),
                "context":       parsed.get("context", ""),
            },
            "info_ready": True,
        }
    else:
        question = parsed.get("question", "请提供项目路径和任务描述。")
        return {
            "messages": [AIMessage(content=question)],
            "info_ready": False,
            # output 设置追问内容，让 CLI 打印出来后等待用户输入
            "output": question,
        }


# ──────────────────────────────────────────────
# 节点：创建新任务
# ──────────────────────────────────────────────

def node_create_task(state: AgentState) -> dict:
    """调用 LLM 解析任务描述，生成计划，存库。"""
    info = state["gathered_info"]
    project_paths = info.get("project_paths", [])
    description   = info.get("description", "")

    if not project_paths or not description:
        return {"error": "缺少项目路径或任务描述"}

    try:
        task = create_task_via_llm(description, project_paths)
        return {
            "task": dataclasses.asdict(task),
            "task_id": task.task_id,
            "error": None,
        }
    except Exception as e:
        return {"error": f"任务创建失败: {e}"}


# ──────────────────────────────────────────────
# 节点：初始化已有项目
# ──────────────────────────────────────────────

def node_init_existing(state: AgentState) -> dict:
    """结合对话上下文 + 快照，LLM 推断任务计划。"""
    info = state["gathered_info"]
    project_paths = info.get("project_paths", [])
    context = info.get("context", "") or info.get("description", "")

    # 先采集快照作为输入
    try:
        snapshot_obj = collect_snapshot(
            task_id="TASK-INIT",
            project_paths=project_paths,
            idea_project_path=project_paths[0],
            interrupt_reason="已有项目评估",
        )
        snapshot = dataclasses.asdict(snapshot_obj)
    except Exception as e:
        return {"error": f"快照采集失败: {e}"}

    # 对话历史拼成字符串传给 LLM
    history_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in state["messages"]
    )

    try:
        task = init_existing_project_via_llm(history_text, project_paths, snapshot)
        return {
            "task": dataclasses.asdict(task),
            "task_id": task.task_id,
            "snapshot": snapshot,
            "error": None,
        }
    except Exception as e:
        return {"error": f"任务初始化失败: {e}"}


# ──────────────────────────────────────────────
# 节点：加载已有任务（progress 模式）
# ──────────────────────────────────────────────

def node_load_task(state: AgentState) -> dict:
    """从对话中提取 task_id 或列出任务供选择。"""
    # 从最后一条用户消息里找 task_id
    last_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content
            break

    # 先看 state 里有没有
    task_id = state.get("task_id")

    # 没有则尝试从消息里解析
    if not task_id:
        import re
        match = re.search(r"TASK-\d{8}-\d{6}", last_msg)
        if match:
            task_id = match.group()

    if task_id:
        task = load_task(task_id)
        if task:
            return {"task": dataclasses.asdict(task), "task_id": task_id, "error": None}

    # 找不到则列出所有任务
    tasks = list_tasks()
    if not tasks:
        msg = "数据库中没有任务记录，请先新建任务。"
        return {"error": msg, "output": msg}

    task_list = "\n".join(
        f"  {t.task_id}  [{t.status}]  {t.task_name}"
        for t in tasks
    )
    msg = f"没有找到该任务，已有任务如下：\n{task_list}\n\n请输入要查看的 task_id："
    return {
        "messages": [AIMessage(content=msg)],
        "output": msg,
        "intent": "progress",   # 保持 progress 意图，下一轮继续走 load_task
        "info_ready": False,
    }


# ──────────────────────────────────────────────
# 节点：采集快照
# ──────────────────────────────────────────────

def node_collect_snapshot(state: AgentState) -> dict:
    """采集当前工作快照。"""
    # 如果已有快照（init_existing 里采集过），跳过
    if state.get("snapshot"):
        return {}

    task_dict = state.get("task")
    if not task_dict:
        return {"error": "没有任务信息，无法采集快照"}

    project_paths = task_dict.get("project_paths", [])
    task_id       = task_dict.get("task_id", "TASK-UNKNOWN")

    try:
        snapshot_obj = collect_snapshot(
            task_id=task_id,
            project_paths=project_paths,
            idea_project_path=project_paths[0],
            interrupt_reason="进度检查",
        )
        snapshot = dataclasses.asdict(snapshot_obj)
        path = save_snapshot(snapshot_obj, output_dir=str(get_settings().snapshot_dir))
        return {"snapshot": snapshot, "snapshot_path": path, "error": None}
    except Exception as e:
        return {"error": f"快照采集失败: {e}"}


# ──────────────────────────────────────────────
# 节点：进度分析
# ──────────────────────────────────────────────

def node_analyze_progress(state: AgentState) -> dict:
    """对比计划和快照，LLM 推断进度。"""
    task_dict = state.get("task")
    snapshot  = state.get("snapshot")

    if not task_dict or not snapshot:
        return {"error": "缺少任务或快照数据"}

    # 从 dict 重建 Task 对象
    from task_manager import Task, PlanStep
    plan = [
        PlanStep(
            step_id=s["step_id"],
            step_index=s["step_index"],
            step_name=s["step_name"],
            description=s["description"],
            status=s.get("status", "pending"),
            sub_tasks=s.get("sub_tasks", []),
            started_at=s.get("started_at"),
            completed_at=s.get("completed_at"),
            notes=s.get("notes", ""),
        )
        for s in task_dict.get("plan", [])
    ]
    task = Task(
        task_id=task_dict["task_id"],
        task_name=task_dict["task_name"],
        project_paths=task_dict["project_paths"],
        project_types=task_dict["project_types"],
        goal=task_dict["goal"],
        tech_stack=task_dict["tech_stack"],
        status=task_dict["status"],
        priority=task_dict["priority"],
        created_at=task_dict["created_at"],
        last_active_at=task_dict["last_active_at"],
        total_work_seconds=task_dict["total_work_seconds"],
        interrupt_count=task_dict["interrupt_count"],
        plan=plan,
        current_step_index=task_dict["current_step_index"],
        notes=task_dict.get("notes", ""),
    )

    try:
        report = analyze_and_persist_progress(task, snapshot)
        return {"progress_report": report, "error": None}
    except Exception as e:
        return {"error": f"进度分析失败: {e}"}


# ──────────────────────────────────────────────
# 节点：闲聊回复
# ──────────────────────────────────────────────

CHAT_SYSTEM = """你是一个开发工作助手，帮助开发者管理任务进度、分析代码状态。
回答要简洁专业。如果用户想新建任务或查看进度，提示他们告诉你项目路径和目标。"""

def node_chat(state: AgentState) -> dict:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=CHAT_SYSTEM),
        *state["messages"],
    ])
    return {
        "messages": [AIMessage(content=response.content)],
        "output": response.content,
    }


# ──────────────────────────────────────────────
# 节点：格式化输出
# ──────────────────────────────────────────────

def node_format_output(state: AgentState) -> dict:
    """把进度报告格式化为可读文本。"""
    error = state.get("error")
    if error:
        msg = f"出错了：{error}"
        return {"output": msg, "messages": [AIMessage(content=msg)]}

    report = state.get("progress_report")
    task_dict = state.get("task")

    if not report or not task_dict:
        msg = "分析完成，但没有生成报告。"
        return {"output": msg, "messages": [AIMessage(content=msg)]}

    pct   = report.get("completion_percent", 0)
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    lines = [
        f"任务：{task_dict['task_name']}",
        f"目标：{task_dict['goal']}",
        f"",
        f"进度  [{bar}] {pct}%",
        f"摘要  {report.get('summary', '')}",
        f"",
        "各步骤状态：",
    ]

    icon_map = {"completed": "✓", "in_progress": "→", "pending": "○", "skipped": "–"}
    for s in report.get("step_statuses", []):
        icon = icon_map.get(s["status"], "○")
        lines.append(f"  [{icon}] {s['step_index']+1}. {s['step_name']}")
        if s.get("evidence"):
            lines.append(f"       {s['evidence']}")

    lines += [
        f"",
        f"当前步骤：{report.get('current_step_name', '-')}",
        f"下一步  ：{report.get('next_action', '-')}",
    ]

    risks = report.get("risks", [])
    if risks:
        lines.append("")
        lines.append("风险提示：")
        for r in risks:
            lines.append(f"  ! {r}")

    output = "\n".join(lines)
    return {"output": output, "messages": [AIMessage(content=output)]}


# ──────────────────────────────────────────────
# 条件边
# ──────────────────────────────────────────────

def edge_after_route(state: AgentState) -> str:
    """路由后决定走哪个分支。"""
    return state["intent"]   # new_task | check | progress | chat


def edge_after_gather(state: AgentState) -> str:
    """信息收集后：够了继续，不够继续追问（回到 gather）。"""
    if state.get("info_ready"):
        return "ready"
    return "need_more"


def edge_after_load_task(state: AgentState) -> str:
    """加载任务后：有任务继续，没有等用户回复。"""
    if state.get("task"):
        return "has_task"
    return "wait_user"


def edge_check_error(state: AgentState) -> str:
    """出错了就直接输出，否则继续。"""
    if state.get("error"):
        return "error"
    return "ok"


# ──────────────────────────────────────────────
# 构建图
# ──────────────────────────────────────────────

def build_graph():
    init_db()
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("route",            node_route)
    graph.add_node("gather_info",      node_gather_info)
    graph.add_node("create_task",      node_create_task)
    graph.add_node("init_existing",    node_init_existing)
    graph.add_node("load_task",        node_load_task)
    graph.add_node("collect_snapshot", node_collect_snapshot)
    graph.add_node("analyze_progress", node_analyze_progress)
    graph.add_node("format_output",    node_format_output)
    graph.add_node("chat",             node_chat)

    # 入口
    graph.set_entry_point("route")

    # route → 四个分支
    graph.add_conditional_edges("route", edge_after_route, {
        "new_task":  "gather_info",
        "check":     "gather_info",
        "progress":  "load_task",
        "chat":      "chat",
    })

    # gather_info → 信息够了走不同节点，不够继续追问
    graph.add_conditional_edges("gather_info", edge_after_gather, {
        "ready":     "create_task",   # new_task 路径；check 路径见下方重载
        "need_more": "gather_info",   # 继续追问（等用户下一轮输入时重新进入）
    })

    # create_task → 检查错误 → collect_snapshot
    graph.add_conditional_edges("create_task", edge_check_error, {
        "ok":    "collect_snapshot",
        "error": "format_output",
    })

    # init_existing → 检查错误 → analyze_progress（已有快照）
    graph.add_conditional_edges("init_existing", edge_check_error, {
        "ok":    "analyze_progress",
        "error": "format_output",
    })

    # load_task → 有任务继续，没有等下一轮
    graph.add_conditional_edges("load_task", edge_after_load_task, {
        "has_task":  "collect_snapshot",
        "wait_user": END,
    })

    # collect_snapshot → 检查错误 → analyze_progress
    graph.add_conditional_edges("collect_snapshot", edge_check_error, {
        "ok":    "analyze_progress",
        "error": "format_output",
    })

    # analyze_progress → format_output
    graph.add_edge("analyze_progress", "format_output")

    # 终点
    graph.add_edge("format_output", END)
    graph.add_edge("chat",          END)

    return graph.compile()


# ──────────────────────────────────────────────
# check 路径的 gather_info → init_existing 修复
# ──────────────────────────────────────────────
# 问题：gather_info 的 ready 边固定指向 create_task，
# 但 check 意图应该走 init_existing。
# 解决：在 gather_info 节点里根据 intent 输出不同的 next 字段，
# 条件边读取这个字段来路由。

def node_gather_info_v2(state: AgentState) -> dict:
    """gather_info 升级版：保留多轮收集结果，路由由条件边处理。"""
    return node_gather_info(state)


def edge_after_gather_v2(state: AgentState) -> str:
    if not state.get("info_ready"):
        return "need_more"   # → END，本轮结束，等用户下一轮输入
    return "init_existing" if state.get("intent") == "check" else "create_task"


def build_graph_v2(checkpointer=None):
    """最终版：支持 intent-aware 的 gather 路由。"""
    init_db()
    graph = StateGraph(AgentState)

    graph.add_node("route",            node_route)
    graph.add_node("gather_info",      node_gather_info_v2)
    graph.add_node("create_task",      node_create_task)
    graph.add_node("init_existing",    node_init_existing)
    graph.add_node("load_task",        node_load_task)
    graph.add_node("collect_snapshot", node_collect_snapshot)
    graph.add_node("analyze_progress", node_analyze_progress)
    graph.add_node("format_output",    node_format_output)
    graph.add_node("chat",             node_chat)

    graph.set_entry_point("route")

    graph.add_conditional_edges("route", edge_after_route, {
        "new_task":  "gather_info",
        "check":     "gather_info",
        "progress":  "load_task",
        "chat":      "chat",
    })

    graph.add_conditional_edges("gather_info", edge_after_gather_v2, {
        "create_task":   "create_task",
        "init_existing": "init_existing",
        "need_more":     END,            # 本轮结束，追问输出给用户，等下一轮输入
    })

    graph.add_conditional_edges("create_task", edge_check_error, {
        "ok":    "collect_snapshot",
        "error": "format_output",
    })

    graph.add_conditional_edges("init_existing", edge_check_error, {
        "ok":    "analyze_progress",
        "error": "format_output",
    })

    graph.add_conditional_edges("load_task", edge_after_load_task, {
        "has_task":  "collect_snapshot",
        "wait_user": END,
    })

    graph.add_conditional_edges("collect_snapshot", edge_check_error, {
        "ok":    "analyze_progress",
        "error": "format_output",
    })

    graph.add_edge("analyze_progress", "format_output")
    graph.add_edge("format_output",    END)
    graph.add_edge("chat",             END)

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────
# CLI 测试入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import threading
    import itertools
    import time
    import concurrent.futures

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    TIMEOUT_SECONDS = get_settings().request_timeout_seconds

    def _spinner(stop_event: threading.Event):
        """在后台线程里持续输出等待动画，直到 stop_event 被设置。"""
        frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        elapsed = 0
        while not stop_event.is_set():
            print(f"\r  {next(frames)} 思考中... {elapsed}s", end="", flush=True)
            time.sleep(0.1)
            elapsed = round(elapsed + 0.1, 1)
        # 清空这一行
        print("\r" + " " * 30 + "\r", end="", flush=True)

    app = build_graph_v2()

    print("Work Assistant Agent 已启动（输入 exit 退出）")
    print("示例输入：")
    print("  我想给 quantitativeInvestment 项目实现股票筛选功能")
    print("  帮我看看我这个项目现在做到哪了")
    print("  TASK-20260712-103045 的进度怎么样\n")

    # 跨轮次持久化的状态（信息收集阶段需要保留）
    persist = {
        "intent":        "",
        "gathered_info": {},
        "info_ready":    False,
        "task":          None,
        "task_id":       None,
    }
    history = []

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))

        invoke_input = {
            "messages":        history,
            "intent":          persist["intent"],
            "gathered_info":   persist["gathered_info"],
            "info_ready":      persist["info_ready"],
            "task":            persist["task"],
            "task_id":         persist["task_id"],
            "snapshot":        None,
            "snapshot_path":   None,
            "progress_report": None,
            "output":          "",
            "error":           None,
        }

        # 启动等待动画
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=_spinner, args=(stop_event,), daemon=True)
        spinner_thread.start()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(app.invoke, invoke_input)
                try:
                    result = future.result(timeout=TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    stop_event.set()
                    spinner_thread.join()
                    print(f"\nAgent: 请求超时（超过 {TIMEOUT_SECONDS} 秒），请检查网络或 API 状态。\n")
                    history.pop()
                    continue
        except KeyboardInterrupt:
            stop_event.set()
            spinner_thread.join()
            print("\n已取消。\n")
            history.pop()
            continue
        finally:
            stop_event.set()
            spinner_thread.join()

        # 更新跨轮次状态
        persist["intent"]        = result.get("intent", persist["intent"])
        persist["gathered_info"] = result.get("gathered_info", persist["gathered_info"])
        persist["info_ready"]    = result.get("info_ready", False)
        persist["task"]          = result.get("task") or persist["task"]
        persist["task_id"]       = result.get("task_id") or persist["task_id"]

        # 任务完成后重置收集状态，准备接受下一个新意图
        if result.get("progress_report") or result.get("error"):
            persist["intent"]        = ""
            persist["gathered_info"] = {}
            persist["info_ready"]    = False

        output = result.get("output", "")
        if output:
            print(f"\nAgent: {output}\n")

        history = result.get("messages", history)
