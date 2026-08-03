# Work Assistant Agent — 开发进度文档

> 项目路径：`C:\workspace\AIAgent\work_assistant`  
> 最后更新：2026-07-19

---

## 项目定位

面向开发者的个人工作助手 Agent，核心能力：
- 中断快照采集（读取本机 git/IDEA/文件状态）
- LLM 分析生成中断摘要和恢复建议
- 任务创建与执行计划生成
- 任务进度自动推断
- LangGraph 多轮对话编排

参赛方向：AMD 赛道二「私有 AI Agent 开发与本地部署」  
推理后端：AMD Radeon Cloud 共享 API（`https://developer.amd.com.cn/radeon/api/v1`，模型 `Qwen3.6-35B-A3B`）  
Agent 本体运行在用户 Windows 本机，读取本机真实文件和工作状态。

---

## 2026-07-18 当前状态

### 最新完成

- [x] FastAPI 服务层与 Web 聊天/任务列表页面
- [x] 进度分析结果持久化：步骤状态、证据、时间、当前步骤写回 SQLite
- [x] Web、LangGraph、CLI 使用统一的进度计算口径
- [x] 合并重复的滚动市盈率任务，保留快照引用的主任务
- [x] 使用真实 AMD API 完成一次端到端进度验收
- [x] 使用 LangGraph `SqliteSaver` 持久化 Web 对话与 Agent 状态
- [x] 浏览器使用稳定的 `session_id`，刷新页面后继续同一线程
- [x] 自动化测试覆盖进度持久化、会话重建和线程隔离
- [x] API 级端到端测试覆盖 `new_task`、`check`、`progress` 及任务选择错误路径
- [x] 修复 `check` 流程因临时路由字段被状态模型丢弃而误入 `create_task` 的问题
- [x] Web 首次初始化向导与 `/settings` 本地配置页面
- [x] 普通配置写入 `data/settings.db`，API Key 写入 Windows Credential Manager
- [x] API 地址、模型、超时、数据库、快照目录和默认项目路径统一配置化
- [x] AMD 摘要发送支持 commit 信息与 diff 摘要隐私开关
- [x] `.env.example` 与敏感文件 `.gitignore` 规则
- [x] 初始化页通过 OpenAI 兼容 `/models` 自动发现真实模型并下拉选择
- [x] ChatGPT 风格会话侧边栏：新建、切换、重命名、删除、历史消息恢复
- [x] 本地增量导入 Claude Code 与 Codex 会话，只读展示纯文本消息
- [x] 过滤外部会话中的 reasoning、tool_use/tool_result、文件快照和子代理日志
- [x] 同一会话并发请求保护
- [x] 会话自动识别项目根目录，并按项目/随意会话/已归档分组
- [x] 会话归档、恢复和删除；外部原始 Claude/Codex 文件保持不变
- [x] Claude/Codex 改为手动同步，页面启动只读取本地会话数据库
- [x] 会话侧边栏升级为项目级折叠树，会话按最近时间排列并保留来源标签
- [x] 新建任务、评估项目、查看进度三个主入口固定在侧边栏顶部
- [x] Web 服务默认关闭热重载并使用单进程稳定模式，启动前检测端口占用
- [x] 增加无需访问 AMD 的 `/health` 健康检查和前端超时/重新连接提示
- [x] 增加 `start.ps1` 统一启动入口，直接使用项目 `.venv`
- [x] `start.ps1` 使用 PowerShell 5.1 安全的 ASCII 文本，避免无 BOM UTF-8 中文导致解析失败
- [x] AMD 普通问答升级为模型 token 级真实流式输出，移除按行延迟的伪流式
- [x] LangGraph 各节点实时推送执行阶段，SSE 增加心跳和断线识别
- [x] 增加停止生成、失败重试、请求 ID、取消接口及会话锁安全释放
- [x] 任务规划与进度 JSON 等结构化 AMD 调用也改为内部流式读取，可在响应中途取消

### 后续完善清单（每完成一项后确认再继续）

1. **三条主流程自动化端到端测试（已完成）**
   - 覆盖 `new_task`、`check`、`progress`
   - Mock LLM 与快照采集，避免测试依赖外部 API 和真实项目
   - 验证追问、多轮恢复、任务创建、进度写回和错误路径

2. **配置与敏感信息治理（已完成）**
   - API Key、模型、base URL、数据库路径、超时时间改为环境变量
   - 提供 `.env.example`，启动时校验必需配置
   - 删除业务代码中的硬编码 Windows 路径

3. **会话管理能力（已完成）**
   - 新建/重置会话接口与前端按钮
   - 会话列表、最后活跃时间和过期清理
   - 同一 `session_id` 并发请求保护

4. **真正的流式响应与交互完善（已完成）**
   - 将当前“完整结果按行拆分”升级为模型/图事件实时流式输出
   - 增加取消、重试、断线提示和错误恢复

5. **快照准确性与隐私控制**
   - 校验 Vue 项目的 Git/模块状态识别
   - 增加发送 AMD API 前的摘要预览、脱敏和字段开关
   - 对 diff、终端历史和路径设置明确的截断与过滤规则

6. **自动触发快照与进度更新**
   - watchdog 监听文件变化
   - 支持定时采集、去抖和失败重试
   - 在页面展示最近一次自动分析时间

7. **RAG 本地知识检索**
   - 索引代码和项目文档
   - 支持模块职责、调用关系和恢复上下文问答

8. **交付与参赛材料**
   - README、环境配置、启动指南和常见问题
   - Agent/AMD 推理链路架构图
   - 3–5 分钟演示脚本与视频

---

## 文件结构

```
work_assistant/
├── snapshot_collector.py     # 多项目快照采集（git/IDEA/模块聚合）
├── llm_analyzer.py           # 快照→中断摘要（AMD API）
├── task_manager.py           # Task/PlanStep SQLite持久化 + LLM任务初始化
├── progress_analyzer.py      # 计划 vs 快照→进度推断（AMD API）
├── agent_graph.py            # LangGraph 主图（多轮对话编排）
├── run_demo.py               # CLI 入口（snapshot/new-task/check/progress 四种模式）
├── streaming_runtime.py      # LangGraph 阶段与模型 token 的线程安全事件桥接
├── start.ps1                 # Windows 单进程稳定启动入口
├── interrupt_analysis_prompt.txt   # 中断摘要提示词
├── task_init_prompt.txt            # 任务初始化提示词
├── progress_analysis_prompt.txt    # 进度分析提示词
├── requirements.txt          # psutil / openai / langgraph / langchain-openai
├── data/agent.db             # SQLite 任务数据库
└── snapshots/                # 历史快照 JSON 文件
```

---

## 已完成模块

### 1. 快照采集（snapshot_collector.py）
- 多项目支持（前后端分离/多服务）
- Git 状态：remote、分支、commit 历史、未提交文件、diff
- 文件按功能模块聚合，MVC 层完整度检测（Spring Boot / Vue / React / Python）
- IDEA workspace.xml 解析：变更文件列表、工作会话（workItem）
- 跨项目工作会话：每次会话改了哪些项目的哪些模块
- Shell 历史、开发进程采集
- `.agent/progress.json` 任务进度读取

### 2. LLM 分析（llm_analyzer.py）
- 快照 JSON → 中断摘要文本
- diff 截断至 50 行控制 token

### 3. 任务管理（task_manager.py）
- Task / PlanStep dataclass + SQLite 持久化
- `create_task_via_llm()`：自然语言描述→结构化任务+执行计划
- `init_existing_project_via_llm()`：已有项目+对话上下文→推断计划
- `update_step_status()`：步骤状态更新

### 4. 进度分析（progress_analyzer.py）
- 对比 task.plan 和当前快照，LLM 推断每步完成度
- 返回结构化报告：completion_percent / step_statuses / next_action / risks
- `print_progress_report()`：带进度条的格式化输出

### 5. LangGraph 主图（agent_graph.py）
- AgentState 定义（messages / intent / gathered_info / task / snapshot / progress_report）
- 节点：route / gather_info / create_task / init_existing / load_task / collect_snapshot / analyze_progress / format_output / chat
- 多轮对话：跨轮次保留 intent / gathered_info / task，信息不足时追问后 END 等待用户
- 修复了 gather_info 无限循环 bug（need_more → END）
- CLI 带 spinner 动画和 180s 超时

### 6. API 切换
- 从阿里云通义千问切换到 AMD Radeon Cloud API
- base_url、model 由 Web `/setup` 或环境变量配置
- API Key 保存在 Windows Credential Manager，不写入 SQLite
- 旧版 `AMD_apikey.txt` 仅用于首次迁移兼容

---

## 已知问题 / 待修复

- `node_load_task` 找不到任务时 output 字段未设置（已修复，待验证）
- `gather_info` 的 `need_more` 曾导致无限循环（已修复）
- SSH 连接 AMD 云实例始终 Connection refused（平台问题，暂搁置，用共享 API 替代）

---

## 接下来的开发任务

### 优先级 P0（参赛核心）

1. **验证 agent_graph.py 完整链路**
   - 测试 `new-task` 流程：输入描述→追问路径→生成计划→采集快照→进度分析→输出
   - 测试 `check` 流程：已有项目评估
   - 测试 `progress` 流程：查已有任务进度

2. **FastAPI 服务层**
   - `server.py`：把 agent_graph 包成 HTTP 接口
   - `POST /api/chat`：接收用户消息，返回 agent 回复（SSE 流式）
   - `GET /api/tasks`：任务列表
   - `GET /api/tasks/{task_id}/progress`：进度报告

3. **最简 Web UI**
   - FastAPI + Jinja2 + HTMX
   - 聊天界面：输入框 + 消息流
   - 任务列表页：展示任务状态和进度条
   - 这是演示视频的主要界面

### 优先级 P1（加分项）

4. **RAG 本地知识检索**
   - ChromaDB 向量化代码库/文档
   - 支持"这个模块是干什么的"语义问答

5. **Memory 持久化**
   - LangGraph checkpointer（SQLite）
   - 跨会话保留对话上下文和任务状态

6. **自动触发快照**
   - watchdog 监听文件变化
   - 定时（每30分钟）自动采集快照并更新进度

### 优先级 P2（文档/提交）

7. **README.md**：环境配置、启动指南、依赖列表
8. **架构图**：Agent 架构 + AMD GPU 推理链路
9. **演示视频**：3-5 分钟，展示完整使用流程

---

## 环境信息

- Python: 3.x（work_assistant/.venv）
- 主要依赖: psutil / openai / langgraph / langchain-openai / langchain-core
- 项目数据库: `work_assistant/data/agent.db`
- 配置数据库: `work_assistant/data/settings.db`
- 对话数据库: `work_assistant/data/checkpoints.db`
- 会话索引数据库: `work_assistant/data/conversations.db`
- AMD API Key: Windows Credential Manager（服务名 `work-assistant`）
- 阿里云 API Key（备用）: `C:\workspace\AIAgent\阿里云apikey.txt`
- 测试项目（后端）: `C:\workspace\javaProject\quantitativeInvestment`
- 测试项目（前端）: `C:\workspace\frontProject\quantitativeInvestment-web`
