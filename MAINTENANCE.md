# mul-agent 项目维护说明

> Universal Multi-Agent Collaboration Framework — 基于 ReAct 推理循环的多智能体协作框架，支持飞书 Bot、CLI (TUI/Headless)、HTTP API 多通道交付。

---

## 1. 架构总览

```
             ┌─── 飞书 Bot (WebSocket) ───┐
             │                             │
用户 ────────┼─── CLI TUI / Headless ─────┼──→ SessionManager ──→ run_auto()
             │                             │          │
             └─── HTTP API (FastAPI) ─────┘          ▼
                                              ┌─────────────┐
                                              │ _classify()  │  ← 自动分类
                                              └──────┬───────┘
                                           single/   \project
                                              ▼        ▼
                                      ┌──────────┐  ┌──────────────┐
                                      │ ReAct    │  │ ProjectPilot │
                                      │ Loop     │  │ 迭代式 DAG    │
                                      │(LLM决策) │  │ Plan→Exec→   │
                                      └──────┬───┘  │ Review→Replan│
                                             │      └──────┬───────┘
                                             │             │ 子任务
                                             │             ▼
                                             │      ┌──────────┐
                                             │      │ ReAct    │
                                             │      │ Loop×N   │
                                             │      └──────┬───┘
                                                     │ 工具调用
                                    ┌────────────────┼────────────────┐
                                    ▼                ▼                 ▼
                              ┌──────────┐   ┌──────────┐    ┌──────────────┐
                              │ 搜索/检索 │   │ 文件/代码 │    │   delegate   │
                              │ web_search│   │ read_file │    │  (子 agent)  │
                              │ knowledge │   │ glob/grep │    │  11 种角色    │
                              │ glob/grep │   │ code_run  │    │  background  │
                              └──────────┘   │ git_ops   │    │  worktree    │
                                              │ sql_query │    └──────┬───────┘
                                              │ browser   │           │
                                              └──────────┘      再次进入
                                                               ReAct Loop
```

**核心设计**：
- 单任务：LLM 在推理循环中自主决定调用什么工具
- 大项目：ProjectPilot 迭代式 DAG，子任务复用 ReAct Loop 执行
- 多通道共享同一个 `SessionManager`，会话数据互通（文件级 JSON 存储）

---

## 2. 目录结构

```
mul-agent/
├── src/
│   ├── main.py                    # FastAPI 入口
│   ├── cli/                       # CLI 模块（v0.5.0 新增）
│   │   ├── main.py                #   CLI 入口（mulagent 命令）
│   │   ├── runner.py              #   AgentRunner（全栈任务执行器 + 状态输出）
│   │   ├── tui.py                 #   Textual TUI（可选文本复制、编辑浮层）
│   │   ├── headless.py            #   Headless REPL（异步 Spinner + 进度回调）
│   │   └── init_wizard.py         #   交互式初始化向导（mulagent init）
│   ├── graph/                     # 核心编排层
│   │   ├── orchestrator.py        #   入口：run_react()
│   │   ├── react_orchestrator.py  #   ReAct 推理循环（核心）
│   │   ├── memory.py              #   三层工作记忆
│   │   ├── conversation.py        #   多轮对话管理 + 实体提取 + 上下文 CRUD
│   │   ├── context_compressor.py  #   三维智能上下文压缩（语义分类/话题归档/相关性压缩）
│   │   └── checkpoint.py          #   Redis 断点续作
│   ├── tools/                     # 工具层（23 个工具，含 load_tool meta-tool）
│   │   ├── registry.py            #   工具注册表
│   │   ├── base.py                #   ToolDef 基类
│   │   ├── isolation.py           #   delegate 子 agent（含后台执行 + worktree 隔离）+ check_background
│   │   ├── skill_loader.py        #   Skill 自动发现与注册
│   │   ├── discovery.py           #   web_search, knowledge_recall
│   │   ├── injection.py           #   web_fetch, read_file, list_dir, glob_search, grep_search
│   │   ├── generation.py          #   execute_shell, code_run, write_file, edit_file（支持 replace_all）
│   │   ├── task_manager.py        #   todo_manage, plan_submit（计划模式）
│   │   ├── research.py            #   deep_research
│   │   ├── codemap.py             #   codemap (AST 分析)
│   │   ├── docs_lookup.py         #   docs_lookup
│   │   ├── browser.py             #   browser_fetch (Playwright)
│   │   ├── sql_query.py           #   sql_query (只读)
│   │   ├── git_tools.py           #   git_ops, github_ops
│   │   ├── knowledge_rag.py       #   Chunk + Embedding RAG
│   │   ├── sandbox.py             #   Docker 沙箱执行
│   │   ├── security.py            #   工具安全钩子 + 用户可配置 Shell Hooks（含危险操作确认）
│   │   └── plugins.py             #   插件系统
│   ├── gateway/                   # 接入层
│   │   ├── adapter.py             #   SessionManager + 进度协议（v0.5.0 新增）
│   │   ├── feishu_bot.py          #   飞书 Bot（WebSocket 长连接）
│   │   ├── routes.py              #   HTTP API 路由
│   │   └── streaming.py           #   SSE 流式输出
│   ├── common/                    # 基础设施
│   │   ├── config.py              #   配置加载 + PROJECT_ROOT 解析
│   │   ├── llm.py                 #   多模型 LLM 管理
│   │   ├── vector.py              #   Qdrant 向量 + 三级 Embedding
│   │   ├── observability.py       #   Prometheus 指标 + 分布式追踪
│   │   ├── trace_context.py       #   请求级 trace_id 上下文（contextvars）
│   │   ├── redis_client.py        #   Redis 客户端
│   │   ├── db.py                  #   PostgreSQL 异步连接
│   │   ├── tokenizer.py           #   Token 计数与截断
│   │   ├── retry.py               #   指数退避重试
│   │   └── logging_config.py      #   结构化日志
│   ├── evolution/                 # 自进化层
│   │   ├── tool_learning.py       #   UCB1 工具学习算法
│   │   ├── experience.py          #   经验提取与存储
│   │   ├── diagnostician.py       #   系统诊断（任务统计/弱项分析）
│   │   ├── prescriber.py          #   进化处方（规则+LLM 推荐改进）
│   │   ├── applier.py             #   进化执行（备份/回滚/热重载）
│   │   ├── absorber.py            #   外部项目吸收
│   │   ├── controller.py          #   进化编排（诊断→处方→执行流水线）
│   │   ├── feedback_loop.py       #   用户反馈驱动进化
│   │   ├── feedback.py            #   反馈模型
│   │   └── trace.py               #   执行轨迹持久化
│   └── models/                    # ORM 模型
│       ├── trace.py               #   task_traces, subtask_traces
│       └── feedback.py            #   feedbacks
├── config/
│   ├── settings.yaml              # 运行时配置（含 API Key，不入 git）
│   ├── settings.yaml.example      # 配置模板
│   ├── agents.yaml                # 11 个专业角色定义（支持 knowledge: auto 动态知识选择）
│   ├── tools.yaml                 # 工具配置
│   ├── knowledge/                 # 角色知识库（18 个 .md 文件：python/go/rust/java 等）
│   ├── skills/                    # 外部 Skill 目录（自动注册为 delegate 角色）
├── scripts/
│   ├── setup.sh                   # Linux 一键安装脚本（含数据库选择）
│   ├── setup.ps1                  # Windows 一键安装脚本（PowerShell）
│   └── hooks/                     # 工具安全钩子脚本
│       ├── guard_shell.sh         #   Shell 命令：危险操作弹确认 + 审计日志
│       ├── guard_file.sh          #   文件写入/编辑：自动备份 + 审计日志
│       └── guard_git.sh           #   Git 操作：危险操作弹确认 + auto-stash
├── tests/
│   ├── unit/                      # 单元测试（17 个文件，308+ 个用例）
│   ├── integration/               # 集成测试（trace 持久化）
│   └── e2e/                       # 端到端测试（HTTP + ReAct 流程）
├── docker/
│   ├── docker-compose.yaml        # 基础设施编排（PG/Redis/Qdrant）
│   └── sandbox/                   # 沙箱环境配置
├── .github/workflows/ci.yml       # CI/CD 流水线（lint + test + docker）
├── Dockerfile                     # 应用镜像
├── Makefile                       # 开发命令
├── alembic/                       # 数据库迁移
├── alembic.ini                    # Alembic 配置
├── mulagent-feishu.service        # systemd 服务定义（飞书 Bot）
├── README.md                      # 项目说明
└── pyproject.toml                 # 依赖定义 + CLI entry point
```

---

## 3. 核心流程

### 3.1 ReAct 推理循环

```
react_loop(user_input, tools, llm)
  │
  ├─ Step 0: 继承父级指令（子 agent）
  ├─ Step 1: 提取用户约束（规则 + LLM）
  ├─ Step 2: 准备工具 Schema
  └─ Step 3: 循环（最多 max_rounds 轮）
       │
       ├─ 动态构建 System Prompt（7 层：base + directives + git(TTL60s) + memory + history + facts + reminders）
       ├─ Deferred tools rebind（load_tool 后下轮自动重新绑定 schema）
       ├─ System reminders 注入（工具恢复/todo 提醒/deferred 加载通知）
       ├─ LLM 推理 → 返回工具调用或最终回答
       ├─ 如无工具调用 → 返回最终回答（可选验证）
       ├─ 循环检测（重复调用 3 次 → 强制切换策略）
       ├─ 并行执行工具（带缓存、幂等键、超时）
       ├─ 结果压缩 → 存入工作记忆（各段有 token 预算）
       ├─ 每 3 轮保存 Redis 检查点
       └─ 记忆超 15 条 → LLM 摘要压缩
```

### 3.2 四层记忆体系

| 层 | 内容 | 生命周期 | 存储 |
|----|------|----------|------|
| **Persistent Memory** | 跨会话用户偏好/项目上下文 | 永久（跨会话） | `~/.mulagent/memory/` |
| **Directives** | 用户约束（如"删除前要确认"） | 永不压缩（会话内） | WorkingMemory |
| **State** | 结构化进度（last_tool, rounds） | 原子更新（会话内） | WorkingMemory |
| **Facts** | 工具返回结果 | 按相关度衰减，可压缩 | WorkingMemory |

### 3.3 子 Agent 委派

`delegate` 工具会启动独立 ReAct 循环（最多 5 轮、90s 超时），支持 11 个内置角色 + 自动加载的 Skill 角色：

| 类别 | 角色 |
|------|------|
| 战略 | planner, architect |
| 研究 | researcher, analyst |
| 代码 | coder, code_reviewer, build_resolver, tdd_guide |
| 安全 | security_auditor |
| 内容 | writer, executor |
| Skill | 自动从 config/skills/ 和 SKILL_DIRS 环境变量加载 |

**内置角色**定义在 `config/agents.yaml`，每个角色有：
- `prompt`: 系统提示词
- `tools`: 允许的工具子集
- `knowledge`: 注入的知识库文件（来自 `config/knowledge/*.md`）

**Skill 角色**通过 `skill_loader.py` 自动发现，只需放入 skill 目录即可注册（见 10.4）。

**后台执行**：`delegate(background=true)` 使用 `asyncio.ensure_future()` 异步执行子 agent，返回 `bg_id`，通过 `check_background` 工具查询状态和结果。

**Worktree 隔离**：`delegate(isolation="worktree")` 在临时 Git worktree 中执行子 agent，隔离文件变更。无变更时自动清理 worktree，有变更时保留供用户 review。

---

## 4. 23 个工具清单

工具按**功能分类**，选择依据是「任务需要什么」而非「哪个便宜」。

| 分类 | 工具 | 可逆性 | 用途 |
|------|------|--------|------|
| 搜索与发现 | `glob_search` | 只读 | 文件名模式匹配搜索 |
| | `grep_search` | 只读 | 文件内容正则搜索 |
| | `read_file` | 只读 | 读取文件（支持 offset/limit） |
| | `list_dir` | 只读 | 浏览目录 |
| | `codemap` | 只读 | AST 代码结构提取（仅作 glob/grep 补充） |
| | `knowledge_recall` | 只读 | 知识库语义检索（含分层经验） |
| 外部查询 | `web_search` | 只读 | 网络搜索 |
| | `web_fetch` | 只读 | 抓取网页内容 |
| | `docs_lookup` | 只读 | 官方文档查询 |
| 文件操作 | `edit_file` | 可逆 | 文件局部编辑（支持 replace_all），优先于 write_file |
| | `write_file` | 可逆 | 创建新文件或完整重写 |
| 执行 | `execute_shell` | 有副作用 | Shell 命令（仅用于计算/系统操作，禁止搜索/读取） |
| | `code_run` | 有副作用 | 多语言代码执行（Python/JS/TS/Go/Rust/Java） |
| | `sql_query` | 只读 | 只读 SQL 查询 |
| | `browser_fetch` | 只读 | Playwright JS 渲染抓取 |
| 版本控制 | `git_ops` | 部分可逆 | Git 操作（push/force push 需确认） |
| | `github_ops` | 影响外部 | GitHub PR/Issue 管理 |
| 任务管理 | `todo_manage` | 可逆 | 任务管理（create/done/update/list） |
| | `plan_submit` | 可逆 | 提交执行计划供用户确认 |
| | `check_background` | 只读 | 查询后台子 agent 状态和结果 |
| 研究与委派 | `deep_research` | 只读 | 多角度深度研究 [deferred] |
| | `delegate` | 视子任务 | 委派给专业子 agent（后台执行 + worktree 隔离） |
| 元工具 | `load_tool` | 只读 | 加载 deferred 工具的完整 schema（按需激活扩展工具） |

---

## 5. 自进化机制

### 5.1 自我进化系统（5 模块闭环）

```
Diagnostician → Prescriber → Applier
    ↑                          │
    └── Controller 编排 ←──────┘
                ↓
         Absorber (外部项目吸收)
```

| 模块 | 文件 | 职责 |
|------|------|------|
| Diagnostician | `evolution/diagnostician.py` | 聚合 trace/feedback/会话数据，输出系统弱点报告 |
| Prescriber | `evolution/prescriber.py` | 基于诊断结果，LLM + 规则引擎生成改进方案 |
| Applier | `evolution/applier.py` | 安全写入配置，支持备份/回滚/热加载 |
| Absorber | `evolution/absorber.py` | clone Git 项目 → 分析结构 → 识别能力 → 生成融合方案 |
| Controller | `evolution/controller.py` | 串联以上模块，3 种执行模式（propose/auto/full） |

**安全边界**：`tune_params` → 自动执行 | `prompt/skill/tool/knowledge` → 需确认 | 核心代码 → 永不自动修改

### 5.2 工具学习（UCB1 算法）

```
score = success_rate + C × √(ln(total_trials) / tool_trials)
```

- 平衡已验证工具（exploitation）与低频工具（exploration）
- 每日衰减因子 0.995，防止马太效应
- 状态持久化到 Redis

### 5.3 分层分级经验系统

经验按复杂度分为三层，每层有独立的积累、评分和晋升机制：

```
┌─────────────────────────────────────────────────┐
│  L3 — 领域知识 (Domain)                          │
│  多个 L2 经验由 LLM 综合提炼而成                    │
│  tier_bonus = 1.5                                │
├─────────────────────────────────────────────────┤
│  L2 — 策略经验 (Strategy)                        │  ← L1 晋升条件：
│  多步骤、多工具协同的经验                           │    use_count ≥ 5
│  tier_bonus = 1.2                                │    success_rate ≥ 60%
├─────────────────────────────────────────────────┤
│  L1 — 原子经验 (Atomic)                          │
│  单工具单步骤的经验                                │
│  tier_bonus = 1.0                                │
└─────────────────────────────────────────────────┘
```

**多维评分公式**：
```
effective_score = similarity × quality_score × freshness × use_bonus × success_bonus × tier_bonus
```

| 维度 | 计算方式 |
|------|----------|
| `quality_score` | 初始 0.5，反馈驱动调整 |
| `freshness` | 30 天半衰期指数衰减 |
| `use_bonus` | min(1 + use_count × 0.1, 2.0) |
| `success_bonus` | 0.5 + success_rate × 0.5 |
| `tier_bonus` | L1=1.0, L2=1.2, L3=1.5 |

**经验生命周期**：
```
任务执行 → 自动提取（含 domain_tags, tier 分类）
    → 去重合并（同层 similarity > 0.85 时合并）
    → 多维评分排序
    → 晋升检查（L1→L2: use_count≥5, success_rate≥60%）
    → 领域综合（多个 L2 → LLM 提炼为 L3）
```

**飞书 `/experience` 命令**：查看各层经验数量、Top 领域标签、平均质量分。

### 5.4 用户反馈

- 低评分（1-2）→ LLM 回顾分析 → 存入负面经验（含 tier 元数据和评分字段）
- 高评分（4-5）→ 提升经验质量权重 + 更新 success_count + last_used_at
- 反馈处理后自动触发 `maybe_promote()` 晋升检查

### 5.5 三维智能上下文压缩

```
维度 1: 语义角色分类 (requirement/correction/error_attempt/final_result/intermediate/directive/question)
维度 2: 话题分组与归档 (hot → cold → recalled)
维度 3: 相关性动态压缩 (Full ≥0.7 → Summary 0.3–0.7 → Title 0.1–0.3 → Hidden <0.1)
```

- 三信号相关性评分：关键词重叠 (0.5) + 召回意图检测 (0.3) + 时间衰减 (0.2)
- 超过 30 轮自动归档冷话题，`/recall` 可随时召回
- 模块：`graph/context_compressor.py`（TurnClassifier / TopicGrouper / SmartCompressor / ContextAssembler）

---

## 6. 安装与部署

### 6.1 一键安装

核心功能**仅需 Python 3.10+ 和 LLM API Key**，数据库全部可选。

**Linux / macOS：**

```bash
# 一键安装 + 启动（交互式选择是否安装数据库）
./scripts/setup.sh

# 安装时自动包含所有数据库
./scripts/setup.sh --with-db

# 仅安装不启动
./scripts/setup.sh --infra
```

**Windows（PowerShell）：**

```powershell
# 一键安装 + 启动（交互式选择是否安装数据库）
.\scripts\setup.ps1

# 安装时自动包含所有数据库
.\scripts\setup.ps1 -WithDB

# 仅安装不启动
.\scripts\setup.ps1 -Infra
```

**前提条件**：Python 3.10+（Windows 安装时勾选 "Add Python to PATH"）

安装流程（两个平台均使用 Python 虚拟环境隔离）：

```
Step 1: 检查 Python → 创建 .venv 虚拟环境
Step 2: 在 venv 中安装 mul-agent 包（pip install -e .[cli]）
Step 3: 注册 PATH（Windows 自动将 .venv\Scripts 加入用户 PATH，安装后可直接使用 mulagent 命令）
Step 4: 数据库选择（交互式询问/--with-db 自动安装）
         ┌─ PostgreSQL — 任务追踪与反馈存储（可选）
         ├─ Redis      — 缓存、检查点、幂等键（可选）
         └─ Qdrant     — 向量存储、经验库、知识RAG（可选）
Step 5: 环境摘要 → 首次自动运行 mulagent init 配置 API Key 和 Base URL → 完成（再次运行时直接启动 CLI）
```

> **不安装数据库也能正常使用**。PostgreSQL 不可用时 trace 功能降级，Redis 不可用时缓存/检查点降级，Qdrant 不可用时使用内存向量。
> Windows 上数据库通过 Docker 安装（需安装 Docker Desktop），Linux 上支持 apt/dnf 原生安装或 Docker。

### 6.2 数据库组件

| 组件 | 用途 | 安装方式 | 不安装时的影响 |
|------|------|----------|---------------|
| PostgreSQL 16 | 执行轨迹、反馈存储 | apt/dnf/Docker | trace 功能不可用 |
| Redis 7 | 缓存、检查点、幂等键 | apt/dnf/Docker | 无缓存、无断点续作 |
| Qdrant | 向量存储、经验库、知识 RAG | Docker | fallback 到内存向量 |

```bash
# 也可通过 Docker Compose 一次性启动所有数据库
docker compose -f docker/docker-compose.yaml up -d
```

### 6.3 运维命令

```bash
# Linux
./scripts/setup.sh --status        # 查看所有服务状态
./scripts/setup.sh --restart       # 重启飞书 Bot
./scripts/setup.sh --stop          # 停止飞书 Bot
./scripts/setup.sh --logs [N]      # 查看最近日志

# Windows
.\scripts\setup.ps1 -Status        # 查看服务状态
```

---

## 7. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tasks` | 提交任务 |
| POST | `/api/v1/tasks/stream` | 流式执行（SSE） |
| POST | `/api/v1/feedback` | 提交评分反馈 |
| GET | `/api/v1/health` | 健康检查（含组件状态 + 延迟） |
| GET | `/api/v1/health/liveness` | 存活探针 |
| GET | `/api/v1/health/readiness` | 就绪探针 |
| GET | `/api/v1/models` | 列出可用模型 |
| GET | `/api/v1/metrics` | 系统指标（JSON） |
| GET | `/api/v1/metrics/prometheus` | Prometheus 格式指标 |
| GET | `/api/v1/traces` | 最近分布式追踪 |
| GET | `/api/v1/checkpoints` | 可恢复的任务检查点 |
| POST | `/api/v1/config/reload` | 热重载配置 |

---

## 8. 使用方式

### 8.1 CLI 命令

```bash
mulagent                         # TUI 模式（富终端，支持文本选择复制）
mulagent --headless              # Headless REPL（纯文本）
mulagent -c "帮我分析这段代码"     # 单次执行
mulagent --model deepseek        # 指定模型
mulagent --session <id>          # 恢复历史会话
mulagent --evolve                # 自我进化（diagnose/propose/auto/full）
mulagent --absorb <git_url>      # 吸收外部项目能力
```

### 8.2 REPL 内置命令

| 命令 | 说明 |
|------|------|
| `/new` | 新建会话 |
| `/resume <id>` | 恢复历史会话 |
| `/model <id>` | 切换模型 |
| `/sessions` | 会话列表 |
| `/modify` | 上下文管理（list/view/edit/del/clear/compress/topics/expand/collapse） |
| `/recall <keyword>` | 召回已归档的对话话题 |
| `/directives` | 持久指令管理 |
| `/evolve` | 自我进化（diagnose/propose/auto/full/history） |
| `/absorb <url>` | 吸收外部 Git 项目 |
| `/quit` | 退出 |

TUI 快捷键：**Ctrl+N** 新会话、**Ctrl+Q** 退出、**Esc** 聚焦输入框。输入 `/` 触发命令自动补全。

### 8.3 开发命令（Make）

```bash
make install          # 安装依赖
make up / make down   # 启动/停止基础设施 (Docker)
make dev              # 启动 API 服务（热重载）
make test             # 运行全部测试
make lint / make format  # 代码检查/格式化
make migrate          # 数据库迁移
```

---

## 9. 配置说明

主配置文件：`config/settings.yaml`（不入 git，含 API Key）
模板：`config/settings.yaml.example`

关键配置段：

```yaml
llm:
  default: "qwen"                     # 默认模型 ID
  models:
    qwen:
      name: "Qwen 3.5 Plus"
      model: "qwen3.5-plus"
      api_key: ""                     # 填入 API Key
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      max_tokens: 65536
    # deepseek / openai / ollama 等其他模型按相同格式配置

database:
  url: "postgresql+asyncpg://mulagent:mulagent@localhost:5432/mulagent"

redis:
  url: "redis://localhost:6379/0"

qdrant:
  url: "http://localhost:6333"
  collection_name: "case_library"

react:
  max_rounds: 30                      # 最大推理轮次（复杂任务需要更多轮）
  timeout: 600                        # 整体超时（秒）— 单工具超时自动推导为 timeout/10
  max_parallel_tools: 5               # 并行工具数
  max_conversation_pairs: 4           # 上下文保留轮数

hooks:                                # 用户可配置的工具钩子（可选）
  pre:
    write_file: "bash scripts/hooks/guard_file.sh '{path}' write"    # 自动备份
    edit_file: "bash scripts/hooks/guard_file.sh '{path}' edit"      # 自动备份
    execute_shell: "bash scripts/hooks/guard_shell.sh '{command}'"   # 危险命令确认
    git_ops: "bash scripts/hooks/guard_git.sh '{action}' '{args}'"   # Git 安全防护
  post:
    write_file: "echo ... >> data/audit/file_audit.jsonl"            # 审计日志

react:
  compress:                             # 压缩参数（全部可配置）
    facts_compact_trigger: 15           # facts 超过此数触发压缩
    facts_keep_recent: 5               # 压缩时保留最近 N 条
    tool_result_max_tokens: 1500       # 单个工具结果截断上限 (token)
    context_compress_ratio: 0.15       # 上下文压缩阈值 = context_window × ratio（动态）
    context_max_chars: 0               # 显式覆盖 (>0 时优先于 ratio，向后兼容)
    level_full: 0.7                    # ≥此值 → 完整保留
    level_summary: 0.3                 # ≥此值 → 摘要
    level_title: 0.1                   # ≥此值 → 仅标题；低于此值 → 隐藏
    weight_keyword: 0.5                # 关键词重叠权重 (Jaccard)
    weight_recall: 0.3                 # 召回意图检测权重
    weight_decay: 0.2                  # 时间衰减权重
    archive_threshold: 30              # 自动归档：超过 N 轮归档冷话题
    archive_manual_threshold: 6        # 手动压缩时的归档阈值
    decay_half_life_hours: 24.0        # 时间衰减半衰期（小时）

embedding:
  api_key: "..."                      # Embedding API Key
  base_url: "..."                     # Embedding API 地址
  model: "text-embedding-v3"

feishu:                               # 飞书 Bot（可选）
  app_id: "..."
  app_secret: "..."
```

> 首次使用运行 `mulagent init` 交互式生成配置，支持通义千问/DeepSeek/OpenAI/Ollama/自定义 5 种 LLM 提供商。

---

## 10. 添加新功能指南

### 添加新工具

1. 在 `src/tools/` 下创建文件，定义 `ToolDef`
2. 在 `src/tools/registry.py` 的 `ALL_TOOLS` 列表中注册
3. 在 `config/agents.yaml` 中将工具添加到相关角色的 `tools` 列表
4. 在 `src/gateway/feishu_bot.py` 添加中文标签（可选）
5. 在 `tests/unit/test_tools.py` 添加测试

### 添加新角色

1. 在 `config/agents.yaml` 的 `roles:` 下添加角色定义
2. 可选：在 `config/knowledge/` 下创建知识库 `.md` 文件
3. 在角色定义中引用知识库文件
4. delegate 工具会自动发现新角色，无需改代码

### 添加新知识库

1. 创建 `config/knowledge/<name>.md`
2. 在角色的 `knowledge:` 列表中引用
3. 多个角色可共享同一知识库文件

### 10.4 注册外部 Skill（零代码）

Skill 是自包含的能力包，遵循标准目录结构：

```
skill-dir/
├── SKILL.md        # YAML frontmatter (name, description) + prompt body
├── references/     # 参考资料文件，子 agent 通过 read_file 读取
└── scripts/        # 工具脚本，子 agent 通过 execute_shell 执行
```

**注册方式**（任选一种）：

1. **放入 `config/skills/`**（推荐）
   ```bash
   # 符号链接
   ln -s /path/to/my-skill config/skills/my-skill
   # 或直接复制
   cp -r /path/to/my-skill config/skills/
   ```

2. **设置 SKILL_DIRS 环境变量**
   ```bash
   export SKILL_DIRS="/path/to/skills-dir1:/path/to/skills-dir2"
   ```

注册后 delegate 工具自动发现，无需编辑任何代码或配置文件。

**SKILL.md frontmatter 格式**：
```yaml
---
name: my-skill-name
description: 技能描述，用于路由决策
metadata:
  trigger: 触发关键词
---
# Prompt body here...
```

`references/` 和 `scripts/` 中的相对路径会自动解析为绝对路径。

---

## 11. 关键设计决策

| 决策 | 原因 |
|------|------|
| ReAct 循环替代 DAG 流水线 | LLM 自主决策比固定流程更灵活 |
| 三层工作记忆 | 不同信息有不同生命周期，避免上下文爆炸 |
| 用户约束（Directives）永不压缩 | 安全性保证，跨委派边界继承 |
| 三级 Embedding 降级 | API Embedding → LLM 关键词提取 → SHA-512 哈希 |
| UCB1 工具学习 | 平衡探索与利用，避免马太效应 |
| 动态知识注入（70% 上下文预算） | 防止知识注入占满上下文窗口 |
| 写操作幂等键 | Redis SET NX 防止重复执行 |
| 每 3 轮保存检查点 | 平衡性能与恢复粒度 |
| SessionManager 适配器层 | 飞书/CLI/API 共享会话存储，切换通道不丢上下文 |
| 所有数据库组件可选 | PG/Redis/Qdrant 不可用时优雅降级，核心功能仅需 LLM |
| 三维上下文压缩 | 语义分类 + 话题归档 + 相关性压缩，长对话不丢失关键信息 |
| 自我进化三级安全 | 参数自动→配置需确认→代码永不修改，防止系统自毁 |
| 动态任务超时 | 不同任务类型配不同超时，避免长任务被误杀或短任务空等 |
| Textual 作为可选依赖 | headless 模式零额外依赖，TUI 仅 `pip install mulagent[cli]` |
| /modify 上下文 CRUD | 用户可精细控制对话上下文，配合智能压缩降低 token 消耗 |
| PROJECT_ROOT 统一解析 | 环境变量→CWD→源码→~/.mulagent 四级降级，支持全局安装 |
| 一键安装 + 交互式数据库选择 | 降低入门门槛，按需安装基础设施，venv 隔离避免依赖冲突 |
| 分层分级经验系统 (L1/L2/L3) | 区分原子经验/策略经验/领域知识，自动晋升和综合，提升经验复用质量 |
| 计划模式 (plan_submit) | 复杂任务先提交计划供用户确认，避免盲目执行 |
| 后台子 agent 执行 | 独立任务不阻塞主循环，提升并发效率 |
| Worktree 隔离 | 文件变更在临时 worktree 中执行，保护主工作目录 |
| 用户可配置 Shell Hooks | pre/post 钩子支持审计、备份、危险操作确认（exit 2 触发飞书确认卡片） |
| 结构化审计日志 | JSON 格式记录每次工具调用，便于合规审计和问题追踪 |
| 任务适配选工具（非成本驱动） | 对标 Claude Code：按功能分类、可逆性决策，而非按价格递增 |
| 强制并行独立工具调用 | 减少轮次浪费，独立调用在同一轮发出 |
| 专用工具替代 shell 搜索/读取 | glob/grep/read_file 更安全高效，减少 Shell 注入风险 |
| 搜索驱动代码理解（对标 Claude Code） | codemap 全扫描不 scale，Glob+Grep+Read 定位→精确→阅读→追踪 更高效 |
| 压缩参数可配置化（13 项） | 不同模型上下文窗口差异大，硬编码无法适配 Qwen 1M vs GPT-4 128K |
| 读后才能改（read-before-edit） | 防止 LLM 凭记忆修改文件导致内容错误，对标 Claude Code 的强制前置 Read |
| 禁用工具 TTL 自动恢复 | 临时网络错误不应永久禁用工具，300s 后自动重新启用 |
| 项目级指令文件（.mulagent.md） | 对标 Claude Code 的 CLAUDE.md，支持项目/全局两级规则注入 |
| 自动 git 上下文注入 | 减少首轮冗余 git_ops 调用，LLM 直接获得 branch/status/commit 信息 |
| Git 上下文 TTL 刷新（60s） | 进程级缓存会导致长时间运行后 git 状态过期，TTL 自动刷新 |
| 工具延迟加载（Deferred Tools） | 22 个工具全量注入浪费 token，扩展工具按需加载 schema 节省 ~800 tokens/轮 |
| System Reminder 动态注入 | 对标 Claude Code 的 system-reminder，支持循环中间插入上下文提醒 |
| 持久化跨会话记忆（PersistentMemory） | 对标 Claude Code 的 MEMORY.md，跨会话保持用户偏好和项目上下文 |
| Prompt 分层预算控制（按可用输入空间） | 基于 context_window - max_tokens 按比例分配，确保输出空间不被输入挤占 |
| 动态环境上下文（每轮刷新） | 替代静态 current_date，提供实时时间、工作目录、平台信息 |
| write_file 后置校验（.mulagent.md 规则） | 纯 prompt 约束在长任务中会漂移，框架层硬校验才能保证格式一致性 |
| Directive 每 5 轮重注入 | LLM 对长上下文中间部分的 attention 权重衰减，定期 reminder 补偿 |
| 完整性门控（Completion Gate） | LLM 倾向于"差不多就结束"，必须检查 todo 全部完成才允许返回最终回答 |
| 数值计算强制 code_run | LLM 数学能力差，百分比/统计/字数必须用 Python 计算，不能估算 |
| Prompt 分层注入（BASE+EXTENDED） | 安全/经验等规则首轮注入即可，不需每轮重复消耗 token |
| 工具描述按 category 分组 | 分类展示帮助 LLM 快速定位工具，减少描述冗余 |
| 对话历史作为消息列表注入 | 放在 system prompt 中注意力衰减快，独立消息对 LLM attention 更强 |
| Knowledge auto 动态选择 | 10 个知识库全量注入浪费 ~50K token，按任务语言/领域动态选 ≤4 个 |
| Skill trigger 自动路由 | 用户输入匹配 skill trigger 时主动提示，降低 skill 使用门槛 |
| pre_tool_hook 语义分类匹配 | 全量 directive 对全量工具逐一匹配误拦截率高，按操作类型分类精准匹配 |
| Fact 衰减基于 round 距离 | 累积乘法在高轮次时所有 fact 趋近于零，距离衰减更符合"最近事实更重要"的直觉 |
| 重复检测用参数相似度 | 同名工具不同参数不应判重复，bigram Jaccard > 0.8 兼顾效率和准确性 |

---

## 12. 变更日志

### v0.19.0 — System Prompt 与架构深度优化（当前）

对标 Claude Code 最佳实践，18 项 prompt/工具/架构优化，分 4 批实施：

**Batch 1 — Prompt 分层与工具描述优化：**
- **Prompt 分层注入**：`ORCHESTRATOR_PROMPT` 拆分为 `BASE`（每轮）+ `EXTENDED`（仅首轮），减少重复 token 消耗
- **工具描述去重 + 分类展示**：ToolDef 新增 `category` 字段（search/file/execution/vcs/task/delegation/meta），system prompt 中按分类分组展示工具描述
- **对话历史改为消息列表**：从 system prompt 字符串注入改为 `HumanMessage`/`AIMessage` 对，提升 LLM 注意力
- **删除 Guardian 角色**：11 个内置角色（原 12 个），代码审查功能已由 code_reviewer 覆盖
- **compact_facts_llm 改进**：按来源分组（每源最多 5 条×400 字符），LLM 摘要 2-3 句/源，500 字符预算，超时 20s
- **空响应分级处理**：facts 为空时直接返回友好提示，不调用 LLM 强制总结

**Batch 2 — 循环检测与幂等性增强：**
- **Todo nudge 阈值提升**：从 `round>=3` 改为 `round>=5 或 tool_calls>8`，减少简单任务的干扰提醒
- **重复检测改为参数相似度**：新增 `_arg_similarity()` 基于字符 bigram Jaccard 相似度（>0.8 判为重复），替代单纯工具名匹配
- **幂等性 fallback**：Redis 不可用时自动降级到会话级 `_idem_local: set[str]`

**Batch 3 — 记忆衰减与缓存优化：**
- **Fact 衰减模型重写**：从累积 `*=0.95` 改为 `0.95^(current_round - fact.round_num)`，距离越远衰减越大
- **Facts budget 从 config 读取**：基于 `context_window` 的 6% 动态分配，不再硬编码
- **System prompt 静态缓存**：`_static_prefix_cache` 缓存 project_directives + persistent_memory 块，300s TTL

**Batch 4 — 知识动态化与安全增强：**
- **Knowledge auto 动态选择**：coder/code_reviewer/build_resolver 的 `knowledge` 改为 `auto`，从全部 18 个知识库中按任务内容自动选择最相关的 ≤4 个
- **Skill trigger 自动路由提示**：用户输入匹配 skill 的 trigger 正则时，round 0 注入 system reminder 建议使用对应 skill
- **Skill 默认工具扩展**：`_DEFAULT_SKILL_TOOLS` 新增 `glob_search`、`grep_search`，skill 子 agent 具备代码搜索能力
- **pre_tool_hook 语义分类匹配**：directive 按操作类型分类（destructive/modify/send/scope），仅对相关工具生效，减少误拦截。新增 send 类操作（push/deploy）的确认拦截

### v0.18.0 — ProjectPilot 大项目支持

新增 ProjectPilot 引擎，支持大型多步骤项目的自动拆解、迭代执行和自我纠正：

- **迭代式 DAG 引擎**（`src/graph/project_pilot.py`）：
  - 执行模型：Plan → Execute → Review → Re-plan → Execute → ...
  - LLM 自动将大任务拆解为带依赖关系的子任务 DAG
  - 独立子任务通过 `asyncio.gather` 并行执行，复用现有 `run_react()`
  - 每批子任务完成后 LLM 审查打分，可插入修正任务、调整后续计划或回退重做
- **三道收敛防线**（防止无限循环）：
  - 最大迭代轮次（`max_iterations`，默认 3，可配置）：到达上限强制结束
  - 收敛检测：连续 N 轮评分无提升 → 已达能力极限，停止迭代
  - 降级到用户决策：子任务重做超过 `max_subtask_retries` 次 → 推送决策卡片给用户
- **自动分类路由**（`src/graph/orchestrator.py`）：
  - `run_auto()` 统一入口，关键词启发式分类（不消耗 LLM 调用）
  - 单任务 → 原有 `run_react()` 路径不变
  - 多步骤项目 → ProjectPilot 迭代 DAG
- **飞书项目级进度卡片**（`src/gateway/feishu_bot.py`）：
  - `ProjectProgressTracker`：实时展示所有子任务状态 + 迭代轮次 + 进度条
  - 决策点交互卡片：子任务需要用户决策时推送按钮卡片，支持「继续执行」「跳过」
  - 决策超时（300s）自动继续
- **可配置项**（`project_pilot` 配置段）：
  - `max_iterations`: 最大迭代轮次（默认 3）
  - `max_subtask_retries`: 单个子任务最多重做次数（默认 2）
  - `convergence_threshold`: 连续无提升轮次阈值（默认 2）
  - `max_parallel_subtasks`: 最大并行子任务数（默认 3）
  - 无项目总超时 — 子任务继承 `react.timeout`，项目只要在推进就不会中断
- **项目 Checkpoint**：
  - Redis 持久化（key 前缀 `project:`，TTL 24h）
  - 支持中断后恢复：`resume_project()` 从断点继续执行

### v0.17.0 — System Prompt 动态化

对标 Claude Code 动态 system prompt 机制，6 项架构优化：

- **Git 上下文 TTL 刷新**：
  - 从进程级一次性缓存改为 60s TTL 自动刷新
  - 长时间运行的服务不再使用过期的 branch/status 信息
- **工具延迟加载（Deferred Tools）**：
  - 核心工具（16 个）始终发送完整 schema
  - 扩展工具（6 个：deep_research/codemap/browser_fetch/sql_query/docs_lookup/check_background）仅列名称+描述
  - 新增 `load_tool` meta-tool，LLM 按需加载 deferred 工具的完整 schema
  - ToolDef 新增 `deferred: bool` 字段，下轮自动 rebind tools
- **System Reminder 动态注入**：
  - 新增 `_pending_reminders` 队列，在 ReAct 循环中间注入 `<system-reminder>` 提醒
  - 触发场景：工具自动恢复、todo_manage 使用提醒、deferred 工具加载通知
  - 通过 `_build_system_prompt(reminders=...)` 注入到 system prompt 尾部
- **持久化跨会话记忆（PersistentMemory）**：
  - 存储路径：`~/.mulagent/memory/`（MEMORY.md 索引 + 独立 .md 文件）
  - 支持 save/remove/load_index 操作
  - MEMORY.md 内容以 `## Persistent Memory` 段注入 system prompt
  - 300s TTL 缓存避免每轮读磁盘
  - LLM 可通过 read_file/write_file 直接操作记忆目录
- **Prompt 分层预算控制（按模型 context window 比例）**：
  - `ModelConfig` 新增 `context_window` 配置项（0=自动检测）
  - 内置 50+ 常见模型 context window 查找表（GPT/Claude/Qwen/DeepSeek/Gemini/GLM/Moonshot/Doubao）
  - 优先级：显式配置 → 模型名前缀匹配 → 保守默认值 32K
  - 预算基于**可用输入空间**（context_window - max_tokens），而非 context_window 本身
  - 按输入空间比例分配：directives(2.5%), git(0.5%), memory(2%), history(9%), facts(6%)
  - 总动态预算 ≈ 20% 输入空间，其余留给 base prompt + tool schemas + 工具结果
  - 每段有最低保底值（如 conversation_history 至少 500 tokens）
- **动态环境上下文**：
  - 替换静态 `{current_date}` 为 `{environment_context}`
  - 每轮刷新：date(精确到分钟) + cwd + platform
  - 未来可扩展：活跃子 agent 数、内存使用等

### v0.17.1 — 长任务执行质量保障

针对长任务（如批量章节生成）中暴露的格式漂移、数据缺失、数值错误等问题，4 项框架级改进：

- **write_file 后置校验钩子**：
  - `.mulagent.md` 中新增 `## Validation Rules` 段，定义文件级校验规则
  - write_file/edit_file 后自动用 code_run(python) 校验（JSON 合法性、行数、字数等）
  - 校验失败时将错误信息追加到 ToolMessage，LLM 必须修正
  - 支持 fnmatch 通配符匹配（如 `chapter_*.md: 中文字数3000-4000`）
- **Directive 衰减补偿**：
  - 每 5 轮自动将所有 directives 作为 system-reminder 重新注入
  - 对抗长对话中注意力衰减导致的规则遗忘
- **完整性校验门控**（Completion Gate）：
  - LLM 试图返回最终回答时，先检查 todo_manage 任务列表是否全部完成
  - 有未完成任务 → 不允许结束，注入提醒继续执行
  - `_verify_answer` 增加任务完成度和文件写入清单信息
- **数值计算强制 code_run**：
  - Prompt 新增硬规则：百分比/字数/统计等计算必须用 code_run(python)
  - 检测 write_file 内容含百分比但同轮无 code_run → 注入 system-reminder 警告
  - 避免 LLM "凭感觉估算"导致进度百分比严重偏离

### v0.16.0 — 可靠性与智能化增强

对标 Claude Code 核心机制，7 项架构改进：

- **edit_file diff + 唯一性校验**：
  - 多处匹配时返回错误并显示各匹配位置（行号），而非静默替换
  - 编辑成功后返回 unified diff 输出，用户可清晰看到变更内容
  - mtime 冲突检测：文件被外部修改后拒绝编辑，要求重新 read
- **读后才能改的硬约束**（read-before-edit enforcement）：
  - edit_file/write_file 执行前检查该文件是否已通过 read_file 读取过
  - 未读取则返回错误："you must read_file before edit_file"
  - 防止 LLM 凭记忆或假设修改文件导致内容错误
  - 新建文件（write_file + 文件不存在）跳过此检查
- **并行调用检测 + nudge**：
  - 连续 3 轮仅发出 1 个工具调用时，自动注入提示：
    "combine independent calls in ONE round for efficiency"
  - 减少不必要的串行轮次，提升任务执行效率
- **LLM 指数退避重试**：
  - 从 `max_retries=1` 升级为 `max_retries=3, base_delay=2s, max_delay=30s`
  - 429/503/timeout 等可重试错误自动指数退避（2s → 4s → 8s）
- **禁用工具自动恢复**（TTL 300s）：
  - 工具连续失败被禁用后，300 秒后自动恢复（重置失败计数）
  - 防止临时网络问题导致工具永久不可用
- **项目级指令文件**（.mulagent.md）：
  - 启动时自动搜索 `.mulagent.md` 或 `AGENT.md`（CWD → 向上 5 级 → ~/）
  - 内容作为 `## Project Directives` 注入 system prompt
  - 支持层级继承：全局 `~/.mulagent.md` + 项目 `./AGENT.md`
- **自动 git 上下文注入**：
  - 启动时自动检测 git repo，注入 branch、status、last commit 到 system prompt
  - 减少 LLM 在首轮调用 git_ops 获取基本信息的浪费

### v0.15.0 — Claude Code 对标优化

对标 Claude Code，从 prompt、tools、架构三个维度补齐 11 项差距：

- **5 个新工具**（17→22）：
  - `glob_search`：文件名模式匹配（替代 execute_shell + find）
  - `grep_search`：文件内容正则搜索（替代 execute_shell + grep）
  - `todo_manage`：任务管理（create/done/update/list），结构化追踪进度
  - `plan_submit`：提交执行计划，ReAct 循环检测 `PLAN_PENDING_MARKER` 后暂停等待用户确认
  - `check_background`：查询后台子 agent 状态和结果
- **edit_file 增强**：新增 `replace_all` 参数，支持批量替换（同名变量重命名等场景）
- **delegate 增强**：
  - `background=true`：`asyncio.ensure_future()` 异步执行，返回 bg_id
  - `isolation="worktree"`：临时 Git worktree 隔离执行，无变更自动清理
- **用户可配置 Shell Hooks + 危险操作确认**：
  - `settings.yaml` 中定义 pre/post 工具钩子
  - 内置 3 个安全脚本（`scripts/hooks/`）：
    - `guard_shell.sh`：危险 Shell 命令弹飞书确认卡片（如 `rm -rf /home`、`curl|bash`、`DROP TABLE`），安全命令记审计日志
    - `guard_file.sh`：写入/编辑前自动备份到 `data/backups/`（带时间戳，保留最近 50 版本）
    - `guard_git.sh`：force push/reset --hard 弹确认，checkout/rebase 前自动 git stash
  - Hook exit code 协议：0=放行, 1=阻止, 2=需用户确认（飞书弹卡片）
  - 确认卡片 120 秒超时自动跳过
- **飞书 TodoList 实时进度**：
  - ReAct 执行期间，`todo_manage` 调用实时推送到飞书卡片（含创建时间、耗时）
  - 最终结果前展示任务完成概览（完成率 + 每步耗时 + 总耗时）
- **飞书 `/model` 命令**：查看可用模型列表 / 切换模型
- **结构化审计日志**：`mulagent.audit` logger + `data/audit/*.jsonl`，JSON 格式记录工具调用、文件变更、Git 操作
- **分层分级经验系统**（L1 原子 / L2 策略 / L3 领域）：
  - 多维评分：quality × freshness(30天半衰期) × use_bonus × success_bonus × tier_bonus
  - 自动去重合并（同层 similarity > 0.85）
  - 自动晋升（L1→L2: use_count≥5, success_rate≥60%）
  - 领域综合（多个 L2 → LLM 提炼为 L3）
  - 飞书 `/experience` 命令查看统计
- **13 项压缩参数可配置化**（`react.compress`）：
  - Facts 压缩：`facts_compact_trigger`, `facts_keep_recent`
  - 工具结果截断：`tool_result_max_tokens`
  - 上下文压缩阈值：`context_compress_ratio`（context_window × ratio 动态计算）
  - 向后兼容：`context_max_chars` > 0 时仍优先使用显式值
  - 四级压缩阈值：`level_full`, `level_summary`, `level_title`
  - 三信号权重：`weight_keyword`, `weight_recall`, `weight_decay`
  - 话题归档：`archive_threshold`, `archive_manual_threshold`, `decay_half_life_hours`
  - 新增 `CompressSettings` 类（`common/config.py`），`context_compressor.py`、`conversation.py`、`memory.py` 全部从配置读取
- **搜索驱动代码理解策略**（对标 Claude Code）：
  - Prompt 从扫描驱动（codemap 为主）重写为搜索驱动（Glob+Grep+Read 为主）
  - 5 步工作流：定位→精确→阅读→追踪→修改
  - codemap 降级为补充工具，不再作为首选
  - 明确禁止 codemap 全目录扫描和盲读整个文件
- **工具选择原则重写**（对标 Claude Code）：
  - 从「成本递增」改为「任务适配 + 可逆性感知」：按功能分类，不按价格排序
  - 可逆性三级：只读（随便用）→ 可逆/有副作用（说明变更）→ 影响外部/不可逆（确认后执行）
  - 强制并行：无依赖的工具调用必须在同一轮并行发出
  - 最小变更原则：不加多余功能、不加没请求的注释、不为假设的需求设计
  - 8 条负向规则（禁止用 execute_shell 搜索文件/内容、禁止文本化任务跟踪等）
  - 新增经验系统引导：复杂任务前先 knowledge_recall 查询历史经验
  - ReAct 循环步骤更新：Step 1 检查经验 → Step 3 使用 todo_manage

### v0.14.0 — 三维智能上下文压缩

- **语义角色分类**：每条对话自动标记为 requirement/correction/error_attempt/final_result/intermediate/directive/question
- **话题分组与归档**：自动检测话题边界，冷话题归档（cold），热话题保留（hot），支持 `/recall` 召回
- **相关性驱动动态压缩**：基于关键词重叠 + 召回意图检测 + 时间衰减三信号计算相关性，四级压缩梯度：
  - Full (≥0.7) → Summary (0.3–0.7) → Title (0.1–0.3) → Hidden (<0.1)
- **新增模块 `graph/context_compressor.py`**：TurnClassifier、TopicGrouper、SmartCompressor、ContextAssembler
- **ConversationStore 增强**：`smart_compress()`, `recall_topic()`, `list_topics()`, `expand_topic()`, `collapse_topic()`
- **`get_history_for_prompt()` 升级**：接受 `current_query` 参数，自动按相关性组装上下文
- **自动归档**：`append_turn()` 超过 30 轮时自动归档冷话题
- **CLI 新命令**：`/recall <keyword>`, `/modify topics`, `/modify expand <id>`, `/modify collapse <id>`
- 新增 41 个单元测试（总测试数 308）

### v0.13.0 — 自我进化系统

- **5 大进化模块**，实现从被动学习到主动自我改进的闭环：
  - `evolution/diagnostician.py` — 诊断器：聚合 trace、feedback、tool learning、会话数据，输出系统弱点报告
  - `evolution/prescriber.py` — 处方器：基于诊断结果，通过 LLM + 规则引擎生成具体改进方案（prompt 优化、新技能、新工具、知识更新、参数调优）
  - `evolution/applier.py` — 执行器：安全写入配置文件，支持备份、回滚、热加载，不修改核心代码
  - `evolution/absorber.py` — 吸收器：给定 Git URL，自动 clone → 分析结构 → 识别能力 → 生成融合方案
  - `evolution/controller.py` — 控制器：串联以上模块，支持 3 种执行模式
- **3 种进化模式**：
  - `propose`（默认）— 只诊断 + 建议，不修改任何文件
  - `auto` — 自动执行安全改动（参数调优），高风险改动需确认
  - `full` — 执行所有改动（prompt、技能、工具、知识、参数）
- **3 种触发入口**：
  - CLI 参数：`mulagent --evolve [propose|diagnose|auto|full]`、`mulagent --absorb <git_url>`
  - Headless 命令：`/evolve`、`/absorb <url>`
  - TUI 命令：`/evolve`、`/absorb <url>`（支持命令补全）
- **安全边界**：tune_params 自动执行 → prompt/skill/tool/knowledge 需确认 → 核心代码永不自动修改
- 进化日志持久化到 `data/evolution_logs/`，支持 `/evolve history` 查看历史
- 新增 20+ 单元测试，总测试数 267

### v0.12.0 — 动态超时 + 进度条单行刷新

- 任务类型智能超时（`estimate_timeout` / `classify_task_type`）：
  - 根据用户输入自动识别任务类别：writing(600s)、coding(480s)、translation(480s)、analysis(420s)、search(180s)、summary(180s)、general(默认)
  - 长输入自动追加缓冲时间（每 100 字 +30s，上限 120s）
  - 优先级：显式指定 > 任务估算 > 配置默认值
- Headless 进度单行刷新：
  - 使用 `\r` 回车符原地覆盖更新，不再逐行换行输出
  - 带动画旋转符号（⠋⠙⠹...）和实时耗时
  - 显示任务类型、剩余超时时间
  - 非 TTY 环境自动降级为换行输出
- 启动时显示任务类型和超时配置：`[writing] timeout=600s starting...`
- 完成时显示任务类型：`[completed · 12.3s · writing]`

### v0.11.0 — TUI 命令自动补全 + 编辑器 UX 改进

- 命令自动补全（Codex 风格）：
  - 输入 `/` 时自动弹出命令列表，显示所有可用命令及说明
  - 支持实时过滤：随着输入内容缩小候选范围
  - 上/下箭头键在候选列表中导航，Tab 或 Enter 确认选择
  - Esc 关闭弹窗，继续输入自动隐藏
  - 覆盖所有命令：`/new`、`/help`、`/sessions`、`/resume`、`/model`、`/modify *`、`/directives *`、`/evolve *`、`/absorb`、`/recall`、`/quit`
  - 需要参数的命令（如 `/resume`、`/modify edit`）自动追加空格
- 编辑器保存 UX 改进（`/modify edit`）：
  - 新增醒目的操作提示栏（高亮背景）："Ctrl+S = Save and close | Esc = Cancel without saving"
  - 新增可点击按钮："💾 Save (Ctrl+S)" 和 "Cancel (Esc)"，降低键盘快捷键学习门槛
- 使用 Textual `OptionList` 组件实现弹出菜单，兼容 v8.1.1+

### v0.10.0 — Windows 一键安装 + 跨平台兼容

- 新增 `scripts/setup.ps1`：Windows PowerShell 一键安装脚本
  - 自动创建 venv、安装依赖、检查基础设施端口、数据库迁移、启动 CLI
  - 支持 `-Status` / `-Headless` / `-Infra` / `-c` 等参数
  - 基础设施检测通过 TCP 端口探测（替代 Linux systemctl）
- 跨平台路径修复：
  - `tools/generation.py`：`/tmp` → `tempfile.gettempdir()`，Rust 编译输出路径跨平台
  - `tools/injection.py`：`_ALLOWED_ROOTS` 使用 `tempfile.gettempdir()` 替代硬编码 `/tmp`
  - `tools/skill_loader.py`：路径拼接改用 `Path()` 替代字符串 `/`
  - `tools/generation.py`：Python 命令自动检测 `python3` vs `python`（Windows 无 python3）
- Makefile clean 目标使用 `$TMPDIR` 环境变量替代硬编码 `/tmp`

### v0.9.0 — Health API 增强 + trace_id 全链路传播

- Health API 增强：
  - 每组件返回 `ComponentStatus`（ok/latency_ms/error），替代简单布尔值
  - 新增 PostgreSQL 实际连接测试（`SELECT 1`），不再仅检查 factory 是否存在
  - 版本号动态读取（package metadata → pyproject.toml 回退），不再硬编码
  - 状态三级：`ok` / `degraded`（基础设施降级）/ `error`（LLM 不可用）
- trace_id 全链路传播：
  - 新增 `common/trace_context.py`：基于 `contextvars.ContextVar` 的请求级追踪上下文
  - HTTP API（routes + streaming）、飞书 Bot、CLI Runner 三个入口点均自动生成 trace_id
  - `JSONFormatter` 自动注入 trace_id 到所有日志输出
  - `react_loop` 启动时记录 trace_id 到日志
  - `TaskTrace` ORM 模型新增 `trace_id` 字段（可选，含索引）
  - `record_task_trace` 自动从上下文提取 trace_id
  - `TaskResponse` 返回 trace_id，客户端可用于日志关联
  - SSE streaming 结果事件包含 trace_id
- 新增 8 个 trace_context 单元测试 + 3 个 health API 测试（共 236 个）

### v0.8.0 — 错误分类 + 委派深度控制 + 降级测试

- 工具错误分类系统（`classify_tool_error` + `ToolErrorKind`）：
  - 区分 `RETRYABLE`（超时/限速/连接错误）和 `FATAL`（权限/无效参数/未找到）
  - 可重试错误允许 5 次重试（原 3 次），并提示 LLM 可重试
  - 致命错误保持 3 次即禁用策略
- delegate 显式深度控制（`MAX_DELEGATE_DEPTH = 3`）：
  - 替代原来的简单工具排除，支持多级委派（depth 0→1→2 可递归，depth 3 禁止）
  - `delegate_depth` 通过 deps 传递，每次递增
- 降级测试套件（`test_degradation.py`，7 个用例）：
  - Redis 不可用时 ReAct 正常运行 + 幂等检查跳过
  - Qdrant 不可用时 ReAct 正常运行 + delegate 正常工作
  - PostgreSQL 不可用时 checkpoint 静默失败
  - tool learning 不可用时静默跳过
  - 全部基础设施同时不可用的综合测试
- 新增 11 个错误分类测试 + 3 个深度控制测试（共 225 个）

### v0.7.0 — 初始化向导 + 启动状态 + 跨会话指令 + Spinner

- 新增 `mulagent init`：交互式初始化向导，支持 5 种 LLM 提供商（通义千问/DeepSeek/OpenAI/Ollama/自定义）
- `AgentRunner.print_status()`：启动时输出组件状态（LLM/PostgreSQL/Redis/Qdrant），带颜色标记 ✓/○
- 跨会话持久化指令（persistent directives）：用户规则跨 session 生效
  - ConversationStore 新增：`load_persistent_directives`, `add_persistent_directive`, `remove_persistent_directive`, `get_all_directives`
  - TUI + Headless 新增 `/directives` 命令（list/add/del/clear）
  - `AgentRunner.run()` 使用 `get_all_directives()` 自动合并持久+会话指令
- Headless 异步 Spinner：LLM 思考时每 2 秒打印耗时，工具调用实时输出
- 新增 7 个持久化指令单元测试（共 212 个）

### v0.6.0 — 上下文管理 + TUI 增强 + 代码质量修复

- TUI 聊天面板从 `RichLog` 迁移到 `TextArea(read_only=True)`，支持鼠标选择文本复制
- 新增 `/modify` 命令：对话上下文增删改查（list/view/edit/del/clear/summary/compress）
- TUI 编辑浮层（`EditScreen`）：`/modify edit <n>` 弹出全屏编辑器，Ctrl+S 保存 / Esc 取消
- Headless 编辑：`/modify edit <n>` 打开 `$EDITOR`（默认 nano）
- ConversationStore 新增 CRUD 方法：`list_turns`, `delete_turn`, `delete_turns_range`, `edit_turn`, `clear_turns`, `get_summary`
- 新增 8 个上下文 CRUD 单元测试（共 205 个）
- 修复 `setup.sh` 启动顺序：venv 检查提前到数据库迁移之前
- 修复 `SessionManager._make_id()` 生成冗余 ID（去掉重复的 context_id 片段）
- 统一 `_ensure_src_path()` 到 `cli/__init__.py`，消除三处重复定义
- pyproject.toml 版本号同步为 0.6.0
- 明确 `setup.sh`（运维）与 `mulagent`（用户交互）的职责边界

### v0.5.0 — CLI + 全局安装 + SessionManager

- 新增 `src/cli/` 包：TUI (Textual)、Headless REPL、单次执行三种模式
- 新增 `src/gateway/adapter.py`：SessionManager + ProgressEvent 协议
- 新增 `scripts/setup.sh`：统一服务管理
- `common/config.py` 引入 `_find_project_root()`，支持 `mulagent` 全局安装（任意目录运行）
- 统一 6 个模块的路径引用为 `PROJECT_ROOT`/`CONFIG_DIR`/`DATA_DIR`
- pyproject.toml 新增 `mulagent` 入口点和 `[cli]` 可选依赖
- 重构 `feishu_bot.py` 使用 SessionManager

### v0.4.0 — Skill 自动加载机制

- 新增 `src/tools/skill_loader.py`：扫描 Skill 目录，解析 SKILL.md frontmatter，自动注册为 delegate 角色
- 新增 `config/skills/` 目录：放入或符号链接 Skill 即可注册，零代码
- 支持 `SKILL_DIRS` 环境变量指定额外 Skill 目录
- delegate 工具的角色枚举和描述动态生成（不再硬编码）
- SKILL.md 中的相对路径（references/、scripts/）自动解析为绝对路径
- 已通过 chinese-novelist-skill v2.0 验证（符号链接注册，100% 功能可用）

### v0.3.0 — 清理 Legacy 代码

- 移除 OpenClaw agent 运行时及 legacy pipeline（dispatch→plan→execute→quality_check）
- 移除 `src/agents/` 下所有适配器、注册表、技能获取器
- 移除 `src/graph/` 下 dispatcher、dag_builder、quality_gate、state
- 简化 `orchestrator.py` 为纯 ReAct 入口
- 简化 `routes.py`、`streaming.py`、`main.py` 的依赖注入
- 移除 `OpenClawSettings` 配置
- 清理 `config/agents.yaml` 中 legacy agents 定义
- 清理 15 个 legacy 测试文件
- 统一错误检测为 `_is_error_result()` 公共函数
- 移除 `compress_tool_result` 冗余分支
- 移除 `_force_conclude` 冗余包装函数

### v0.2.0 — Phase 2+3：15 个功能实现

1. 多轮对话记忆增强（实体提取、跨会话上下文）
2. UCB1 工具学习算法（参数优化、反马太效应）
3. Chunk + Embedding 知识 RAG
4. 动态知识注入（上下文预算感知，70% 上限）
5. 浏览器工具（Playwright JS 渲染）
6. 沙箱增强（持久工作区、多文件项目、依赖预安装）
7. SQL 查询工具（只读模式、Schema 自动发现）
8. Git/GitHub 集成工具
9. 写操作幂等键
10. 可观测性三件套（Prometheus 指标、分布式追踪、告警）
11. 用户对话隔离（per-user 目录）
12. CI/CD 流水线（GitHub Actions）
13. 任务断点续作（Redis 检查点）
14. 渐进式输出检测（自动完成信号词）
15. 多模态支持基础

### v0.1.0 — 初始发布

- LangGraph 编排 + ReAct 推理循环
- 飞书 Bot 交付
- 13 个基础工具
- 经验库自进化
- 多模型 LLM 支持
