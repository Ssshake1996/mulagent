# 开发架构：框架 vs 胶水代码

## 核心原则

> **框架做重活，胶水做连接。** 每个模块先问"有没有现成的开源框架能做"，只有框架不覆盖的连接逻辑才自己写。

---

## 框架职责划分总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     我们要写的（胶水代码）                        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ API路由   │  │ Agent包装 │  │ 安全审查   │  │ 进化/反馈    │  │
│  │ +WebSocket│  │ OpenClaw  │  │ 编排管线   │  │ 收集+存储    │  │
│  │          │  │ →LangGraph│  │           │  │              │  │
│  └──────────┘  └──────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   框架做的（直接用，不重写）                      │
│                                                                  │
│  FastAPI        LangGraph          OpenClaw         Qdrant      │
│  ├ REST路由     ├ DAG构建/执行      ├ Agent运行时    ├ 向量存储   │
│  ├ WebSocket    ├ 并行扇出/汇入     ├ Markdown记忆   ├ 语义检索   │
│  ├ 中间件       ├ 状态机管理        ├ Skill插件      └───────────│
│  └ 依赖注入     ├ 检查点持久化      ├ 多Agent路由                │
│                 │ (PG+Redis)       └ WebSocket API              │
│  Celery        ├ 人工干预节点                                    │
│  ├ 分布式任务   ├ Supervisor模式    semgrep/bandit               │
│  ├ 重试/超时    ├ 子图嵌套          ├ AST静态分析                │
│  └ 优先级队列   └ 流式输出          └ 安全漏洞扫描               │
│                                                                  │
│  Docker SDK     Playwright         SQLAlchemy      structlog    │
│  ├ 容器管理     ├ 浏览器自动化      ├ ORM           ├ 结构化日志 │
│  └ 沙箱隔离     └ 页面操作          └ 迁移(Alembic) └ JSON输出   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 逐模块拆解：框架 vs 胶水

### 1. 用户接口层

| 功能 | 谁做 | 说明 |
|------|------|------|
| REST API | **FastAPI** | 直接用，零胶水 |
| WebSocket推送 | **FastAPI** | 原生支持 |
| 请求校验 | **Pydantic** | 模型定义即校验 |
| 认证/限流 | **FastAPI中间件** | 标准中间件 |
| **胶水：路由定义** | 自己写 | 定义API端点，连接到调度层 |
| **胶水：SSE流式输出** | 自己写 | 把LangGraph的流式事件转成SSE推给前端 |

**胶水代码量估算：~150行**

### 2. 调度层（Dispatcher）

| 功能 | 谁做 | 说明 |
|------|------|------|
| 意图识别/问题分类 | **LangGraph节点** | 一个LLM调用节点，输出分类结果 |
| 模板匹配/案例检索 | **Qdrant** | 语义相似度搜索，SDK直接调用 |
| 路由决策 | **LangGraph条件边** | `conditional_edges` 根据分类结果路由 |
| **胶水：分类prompt** | 自己写 | 定义分类的prompt模板和输出schema |
| **胶水：Qdrant查询封装** | 自己写 | 把案例库查询结果格式化为LangGraph状态 |

**胶水代码量估算：~100行**

### 3. 编排层（Orchestrator）—— 框架覆盖最多的层

| 功能 | 谁做 | 说明 |
|------|------|------|
| 任务DAG构建 | **LangGraph** | `StateGraph` + 节点 + 边 |
| 并行执行（扇出/汇入） | **LangGraph** | 多边自动并行，reducer合并结果 |
| 串行流水线 | **LangGraph** | 顺序边 |
| 辩论/投票 | **LangGraph** | 扇出到多Agent → 汇总节点投票 |
| 递归分解 | **LangGraph** | 子图 + 条件边实现递归 |
| 状态管理 | **LangGraph** | TypedDict/Pydantic状态，reducer合并 |
| 检查点/容错 | **LangGraph** | PostgresSaver + RedisSaver |
| 人工干预 | **LangGraph** | `interrupt()` 暂停等待人类输入 |
| Supervisor模式 | **langgraph-supervisor** | 开箱即用的supervisor多Agent |
| 超时控制 | **Celery** | 任务级超时 + 重试策略 |
| 分布式调度 | **Celery** | Worker分发 + 优先级队列 |
| **胶水：Agent选派逻辑** | 自己写 | 多因素打分，决定哪个Agent执行哪个子任务 |
| **胶水：质量门控** | 自己写 | 调用检查LLM，判断输出是否合格 |
| **胶水：动态DAG生成** | 自己写 | 根据LLM分析结果动态构建LangGraph图 |

**胶水代码量估算：~300行**

### 4. Agent池

| 功能 | 谁做 | 说明 |
|------|------|------|
| Agent运行时 | **OpenClaw** | 通过 `openclaw-sdk` Python SDK 调用 |
| Agent记忆 | **OpenClaw** | Markdown文件 + 内置语义检索 |
| Skill管理 | **OpenClaw** | SKILL.md格式，目录即插件 |
| 多Agent隔离 | **OpenClaw** | 每个Agent独立workspace |
| **胶水：OpenClaw→LangGraph适配器** | 自己写 | 把OpenClaw Agent包装成LangGraph节点 |
| **胶水：Agent注册表** | 自己写 | 维护Agent元信息（技能标签、历史评分） |
| **胶水：能力获取调度** | 自己写 | 三级优先级：经验→搜索→Skill引入 |

**胶水代码量估算：~250行**

### 5. 安全审查层 —— 胶水最多的层（但调用的都是现成工具）

| 功能 | 谁做 | 说明 |
|------|------|------|
| 静态分析执行 | **semgrep + bandit** | CLI工具，subprocess调用 |
| 沙箱环境 | **Docker SDK** | 创建隔离容器，限制网络/资源 |
| 行为监控 | **Docker SDK** | 容器日志 + 网络流量审计 |
| **胶水：审查管线编排** | 自己写 | 串联4个Stage，汇总打分 |
| **胶水：权限评估规则** | 自己写 | 最小权限校验逻辑 |
| **胶水：信任评级算法** | 自己写 | 综合各Stage结果计算信任等级 |
| **胶水：质量验证** | 自己写 | benchmark运行 + 基线对比 |

**胶水代码量估算：~400行**

### 6. 记忆系统

| 功能 | 谁做 | 说明 |
|------|------|------|
| 会话记忆 | **LangGraph RedisSaver** | 检查点即会话记忆，原生支持 |
| 长期记忆 | **OpenClaw Memory** | Markdown + SQLite语义索引 |
| 向量存储/检索 | **Qdrant** | SDK直接调用 |
| 结构化存储 | **SQLAlchemy + PostgreSQL** | ORM标准用法 |
| 数据库迁移 | **Alembic** | SQLAlchemy生态 |
| **胶水：跨Agent共享记忆** | 自己写 | Agent执行结果写入共享知识库 |

**胶水代码量估算：~150行**

### 7. 工具层

| 功能 | 谁做 | 说明 |
|------|------|------|
| 代码沙箱 | **Docker SDK** | 创建容器、执行、获取输出 |
| Web搜索 | **Tavily SDK** | `tavily-python`，一行代码搜索 |
| 浏览器操作 | **Playwright** | 原生Python API |
| HTTP请求 | **httpx** | 异步HTTP客户端 |
| 文件操作 | **Python标准库** | pathlib + aiofiles |
| **胶水：工具注册表** | 自己写 | 统一接口，让Agent声明式调用工具 |

**胶水代码量估算：~100行**

### 8. 进化层

| 功能 | 谁做 | 说明 |
|------|------|------|
| 轨迹存储 | **SQLAlchemy + PostgreSQL** | 标准ORM写入 |
| 案例语义检索 | **Qdrant** | 向量存储+检索 |
| 经验提取 | **LLM调用** | 通过LangGraph节点调用LLM做抽象 |
| **胶水：反馈收集** | 自己写 | 显式评分 + 隐式信号采集 |
| **胶水：轨迹记录** | 自己写 | 执行链路结构化存储 |
| **胶水：经验提取prompt** | 自己写 | 定义抽象模式的prompt模板 |
| **胶水：技能缺口分析** | 自己写 | 统计失败模式，识别能力缺口 |

**胶水代码量估算：~300行**

---

## 框架依赖关系图

```
                    ┌─────────────┐
                    │   FastAPI   │ ← HTTP入口
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  LangGraph  │ ← 编排大脑
                    │  StateGraph │
                    │  Supervisor │
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          ▼                ▼                 ▼
   ┌─────────────┐  ┌───────────┐   ┌─────────────┐
   │   Celery    │  │  OpenClaw │   │   Qdrant    │
   │  分布式任务  │  │  SDK      │   │  向量检索    │
   │  队列       │  │  Agent运行 │   │  案例库      │
   └──────┬──────┘  └─────┬─────┘   └─────────────┘
          │                │
          ▼                ▼
   ┌─────────────┐  ┌───────────┐
   │    Redis    │  │  OpenClaw │
   │  消息+缓存  │  │  Daemon   │
   │  +检查点    │  │  (Node.js)│
   └─────────────┘  └───────────┘
          │
          ▼
   ┌─────────────┐
   │ PostgreSQL  │
   │ 持久化+检查点│
   └─────────────┘
```

---

## 核心接口设计

### 接口1：OpenClaw Agent ↔ LangGraph 适配器

这是最关键的胶水——把OpenClaw Agent包装成LangGraph可调度的节点。

```
LangGraph State
    │
    ▼
┌─────────────────────────────┐
│  OpenClawAgentNode (胶水)    │
│                              │
│  1. 从State提取任务描述       │
│  2. 调openclaw-sdk发送给Agent│
│  3. 等待Agent返回结果         │
│  4. 结果写回State             │
└─────────────────────────────┘
    │
    ▼
LangGraph State (更新后)
```

### 接口2：动态DAG生成

用户输入 → LLM分析 → 生成DAG描述(JSON) → 转换为LangGraph StateGraph

```
用户: "帮我分析竞品并写一份报告"
    │
    ▼
LLM输出任务分解JSON:
{
  "tasks": [
    {"id": "t1", "name": "搜索竞品信息", "agent_type": "research", "deps": []},
    {"id": "t2", "name": "搜索市场数据", "agent_type": "research", "deps": []},
    {"id": "t3", "name": "分析数据", "agent_type": "data", "deps": ["t1","t2"]},
    {"id": "t4", "name": "撰写报告", "agent_type": "writing", "deps": ["t3"]}
  ]
}
    │
    ▼
胶水代码: JSON → LangGraph StateGraph
(t1和t2并行, t3等t1+t2完成, t4等t3完成)
```

### 接口3：安全审查管线

```
Skill候选
    │
    ▼
┌──────────────────────────────────────────┐
│  SecurityPipeline (胶水，串联现成工具)     │
│                                           │
│  Stage1: subprocess.run(["semgrep", ...]) │
│  Stage2: 自定义权限规则引擎                │
│  Stage3: docker.create_container(...)     │
│  Stage4: 综合打分 → 信任等级               │
└──────────────────────────────────────────┘
    │
    ▼
TrustLevel: trusted | sandboxed | human_review | blocked
```

### 接口4：进化层反馈回路

```
任务完成
    │
    ├─→ trace_recorder: 执行轨迹 → PostgreSQL (SQLAlchemy)
    │
    ├─→ feedback_collector: 用户评分 → PostgreSQL
    │
    └─→ experience_extractor: LLM调用提取模式 → Qdrant (向量化存入案例库)
                │
                ▼
         案例库更新 → 下次调度层检索时命中
```

---

## 项目目录结构（更新版）

```
mul-agent/
├── docs/
│   ├── architecture.md           # 整体架构设计
│   ├── architecture.mmd          # Mermaid架构图
│   └── dev-architecture.md       # 本文件：开发架构
│
├── config/
│   ├── settings.yaml             # 系统配置（端口、模型、阈值等）
│   ├── agents.yaml               # Agent注册表（技能标签、评分）
│   ├── security_rules.yaml       # 安全审查规则（权限基线、黑名单模式）
│   └── prompts/                  # Prompt模板集中管理
│       ├── dispatcher.yaml       # 意图分类prompt
│       ├── dag_planner.yaml      # 任务分解prompt
│       ├── quality_check.yaml    # 质量门控prompt
│       └── experience_extract.yaml # 经验提取prompt
│
├── src/
│   ├── main.py                   # 入口：FastAPI app + 启动配置
│   │
│   ├── gateway/                  # 胶水：API层 (~150行)
│   │   ├── routes.py             # FastAPI路由定义
│   │   └── streaming.py          # LangGraph事件 → SSE流式输出
│   │
│   ├── graph/                    # 胶水：LangGraph图定义 (~300行)
│   │   ├── state.py              # 全局State定义 (Pydantic)
│   │   ├── dispatcher.py         # 调度子图：意图识别→分类→路由
│   │   ├── orchestrator.py       # 编排主图：动态DAG生成+执行
│   │   ├── dag_builder.py        # JSON任务描述 → LangGraph StateGraph
│   │   └── quality_gate.py       # 质量门控节点
│   │
│   ├── agents/                   # 胶水：Agent池 (~250行)
│   │   ├── adapter.py            # OpenClaw Agent → LangGraph节点适配器
│   │   ├── registry.py           # Agent注册表（技能、评分、元信息）
│   │   ├── selector.py           # Agent选派（多因素打分）
│   │   └── skill_acquirer.py     # 三级能力获取调度
│   │
│   ├── security/                 # 胶水：安全审查 (~400行)
│   │   ├── pipeline.py           # 审查管线编排（串联4个Stage）
│   │   ├── static_analyzer.py    # 调用semgrep+bandit
│   │   ├── permission_checker.py # 权限评估规则引擎
│   │   ├── sandbox_runner.py     # Docker沙箱试运行
│   │   ├── trust_rater.py        # 信任评级算法
│   │   └── runtime_monitor.py    # 运行时异常检测
│   │
│   ├── evolution/                # 胶水：进化层 (~300行)
│   │   ├── feedback.py           # 反馈收集（显式+隐式）
│   │   ├── trace.py              # 轨迹记录
│   │   ├── experience.py         # 经验提取（LLM调用+Qdrant写入）
│   │   └── gap_analyzer.py       # 技能缺口分析
│   │
│   ├── models/                   # SQLAlchemy模型定义
│   │   ├── trace.py              # 执行轨迹表
│   │   ├── feedback.py           # 反馈表
│   │   ├── case.py               # 案例表
│   │   └── skill_audit.py        # Skill审查记录表
│   │
│   └── common/                   # 通用工具 (~100行)
│       ├── config.py             # pydantic-settings配置加载
│       ├── db.py                 # SQLAlchemy引擎+会话
│       └── vector.py             # Qdrant客户端封装
│
├── openclaw_agents/              # OpenClaw Agent定义（非Python代码）
│   ├── code_agent/
│   │   ├── AGENT.md              # Agent人设+指令
│   │   ├── MEMORY.md             # 长期记忆
│   │   └── skills/
│   │       └── code_review/
│   │           └── SKILL.md
│   ├── research_agent/
│   │   ├── AGENT.md
│   │   └── skills/
│   ├── data_agent/
│   │   ├── AGENT.md
│   │   └── skills/
│   ├── writing_agent/
│   │   ├── AGENT.md
│   │   └── skills/
│   └── reasoning_agent/
│       ├── AGENT.md
│       └── skills/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docker/
│   ├── docker-compose.yaml       # Redis + PostgreSQL + Qdrant
│   └── sandbox/
│       └── Dockerfile.sandbox    # Skill沙箱隔离镜像
│
├── alembic/                      # 数据库迁移
│   └── alembic.ini
│
├── pyproject.toml
└── Makefile                      # 常用命令快捷方式
```

---

## 自写代码量估算

| 模块 | 行数估算 | 性质 |
|------|---------|------|
| gateway/ | ~150 | API路由 + SSE转换 |
| graph/ | ~300 | LangGraph图定义 + 动态DAG |
| agents/ | ~250 | OpenClaw适配 + 选派逻辑 |
| security/ | ~400 | 安全管线编排 |
| evolution/ | ~300 | 反馈+轨迹+经验提取 |
| models/ | ~150 | SQLAlchemy模型 |
| common/ | ~100 | 配置+DB+向量客户端 |
| config/prompts/ | ~200 | Prompt模板(YAML) |
| openclaw_agents/ | ~300 | Agent定义(Markdown) |
| **总计** | **~2150** | **其中Python胶水约1650行** |

---

## 核心依赖（精简版）

```toml
[project]
name = "mul-agent"
requires-python = ">=3.11"
dependencies = [
    # === 编排核心 ===
    "langgraph>=0.3",               # DAG编排+状态机+多Agent模式
    "langgraph-supervisor",          # Supervisor多Agent模式
    "langgraph-checkpoint-postgres", # PostgreSQL检查点
    "langgraph-checkpoint-redis",    # Redis检查点

    # === Agent运行时 ===
    "openclaw-sdk",                  # OpenClaw Python SDK

    # === API网关 ===
    "fastapi>=0.115",
    "uvicorn[standard]",
    "sse-starlette",                 # Server-Sent Events

    # === 分布式任务 ===
    "celery[redis]>=5.4",

    # === 存储 ===
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg",                       # PostgreSQL异步驱动
    "alembic",                       # 数据库迁移
    "qdrant-client>=1.12",           # 向量数据库
    "redis>=5.0",

    # === 安全审查 ===
    "semgrep",                       # 静态分析
    "bandit",                        # Python安全扫描
    "docker>=7.0",                   # Docker SDK (沙箱)

    # === 工具 ===
    "tavily-python",                 # AI搜索
    "playwright",                    # 浏览器自动化
    "httpx",                         # HTTP客户端

    # === 基础 ===
    "pydantic>=2.0",
    "pydantic-settings",
    "pyyaml",
    "structlog",
]
```

---

## Phase 1 开发顺序（精确到文件）

**目标：跑通 "用户输入 → 调度 → 单Agent执行 → 返回结果" 的最简闭环**

```
Step 1: 基础设施
  ├─ docker/docker-compose.yaml      (Redis + PostgreSQL + Qdrant)
  ├─ src/common/config.py            (配置加载)
  ├─ src/common/db.py                (SQLAlchemy引擎)
  ├─ pyproject.toml                  (依赖声明)
  └─ Makefile                        (dev/up/down等快捷命令)

Step 2: Agent适配
  ├─ openclaw_agents/code_agent/     (第一个Agent定义)
  ├─ src/agents/adapter.py           (OpenClaw→LangGraph适配器)
  └─ src/agents/registry.py          (Agent注册表，硬编码版)

Step 3: LangGraph核心图
  ├─ src/graph/state.py              (全局State)
  ├─ src/graph/dispatcher.py         (简单分类→直接路由)
  ├─ src/graph/orchestrator.py       (单Agent串行执行)
  └─ src/graph/quality_gate.py       (基础质量检查)

Step 4: API网关
  ├─ src/main.py                     (FastAPI入口)
  ├─ src/gateway/routes.py           (提交任务/获取结果)
  └─ src/gateway/streaming.py        (SSE流式输出)

Step 5: 端到端测试
  └─ tests/e2e/test_basic_flow.py    (完整流程验证)
```
