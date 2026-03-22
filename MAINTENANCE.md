# mul-agent 项目维护说明

> Universal Multi-Agent Collaboration Framework — 基于 ReAct 推理循环的多智能体协作框架，通过飞书 Bot 交付。

---

## 1. 架构总览

```
用户 ──→ 飞书 Bot (WebSocket) ──→ run_react()
                                     │
                                     ▼
                              ┌─────────────┐
                              │  ReAct Loop  │  ← 核心推理循环
                              │  (LLM 决策)  │
                              └──────┬───────┘
                                     │ 工具调用
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
              ┌──────────┐   ┌──────────┐    ┌──────────────┐
              │ 搜索/检索 │   │ 文件/代码 │    │   delegate   │
              │ web_search│   │ read_file │    │  (子 agent)  │
              │ knowledge │   │ code_run  │    │  12 种角色    │
              └──────────┘   │ git_ops   │    └──────┬───────┘
                              │ sql_query │           │
                              │ browser   │      再次进入
                              └──────────┘     ReAct Loop
```

**核心设计**：LLM 在推理循环中自主决定调用什么工具，无需意图分类或 DAG 规划。

---

## 2. 目录结构

```
mul-agent/
├── src/
│   ├── main.py                    # FastAPI 入口
│   ├── graph/                     # 核心编排层
│   │   ├── orchestrator.py        #   入口：run_react()
│   │   ├── react_orchestrator.py  #   ReAct 推理循环（核心）
│   │   ├── memory.py              #   三层工作记忆
│   │   ├── conversation.py        #   多轮对话管理 + 实体提取
│   │   └── checkpoint.py          #   Redis 断点续作
│   ├── tools/                     # 工具层（17 个工具）
│   │   ├── registry.py            #   工具注册表
│   │   ├── base.py                #   ToolDef 基类
│   │   ├── isolation.py           #   delegate 子 agent + 知识注入
│   │   ├── skill_loader.py        #   Skill 自动发现与注册
│   │   ├── discovery.py           #   web_search, knowledge_recall
│   │   ├── injection.py           #   web_fetch, read_file, list_dir
│   │   ├── generation.py          #   execute_shell, code_run, write_file, edit_file
│   │   ├── research.py            #   deep_research
│   │   ├── codemap.py             #   codemap (AST 分析)
│   │   ├── docs_lookup.py         #   docs_lookup
│   │   ├── browser.py             #   browser_fetch (Playwright)
│   │   ├── sql_query.py           #   sql_query (只读)
│   │   ├── git_tools.py           #   git_ops, github_ops
│   │   ├── knowledge_rag.py       #   Chunk + Embedding RAG
│   │   ├── sandbox.py             #   Docker 沙箱执行
│   │   ├── security.py            #   工具安全钩子
│   │   └── plugins.py             #   插件系统
│   ├── gateway/                   # 接入层
│   │   ├── feishu_bot.py          #   飞书 Bot（主交付通道）
│   │   ├── routes.py              #   HTTP API 路由
│   │   └── streaming.py           #   SSE 流式输出
│   ├── common/                    # 基础设施
│   │   ├── config.py              #   配置加载（pydantic-settings）
│   │   ├── llm.py                 #   多模型 LLM 管理
│   │   ├── vector.py              #   Qdrant 向量 + 三级 Embedding
│   │   ├── observability.py       #   Prometheus 指标 + 分布式追踪
│   │   ├── redis_client.py        #   Redis 客户端
│   │   ├── db.py                  #   PostgreSQL 异步连接
│   │   ├── tokenizer.py           #   Token 计数与截断
│   │   ├── retry.py               #   指数退避重试
│   │   ├── timing.py              #   性能计时
│   │   └── logging_config.py      #   结构化日志
│   ├── evolution/                 # 自进化层
│   │   ├── tool_learning.py       #   UCB1 工具学习算法
│   │   ├── experience.py          #   经验提取与存储
│   │   ├── feedback_loop.py       #   用户反馈驱动进化
│   │   ├── feedback.py            #   反馈模型
│   │   └── trace.py               #   执行轨迹持久化
│   └── models/                    # ORM 模型
│       ├── trace.py               #   task_traces, subtask_traces
│       └── feedback.py            #   feedbacks
├── config/
│   ├── settings.yaml              # 运行时配置（含 API Key，不入 git）
│   ├── settings.yaml.example      # 配置模板
│   ├── agents.yaml                # 12 个专业角色定义
│   ├── tools.yaml                 # 工具配置
│   ├── knowledge/                 # 角色知识库（18 个 .md 文件）
│   ├── skills/                    # 外部 Skill 目录（自动注册为 delegate 角色）
│   └── prompts/                   # LLM 提示词模板
├── tests/
│   ├── unit/                      # 单元测试（12 个文件）
│   ├── integration/               # 集成测试
│   └── e2e/                       # 端到端测试
├── docker/docker-compose.yaml     # 基础设施编排
├── .github/workflows/ci.yml       # CI/CD 流水线
├── Dockerfile                     # 应用镜像
├── Makefile                       # 开发命令
├── alembic/                       # 数据库迁移
└── pyproject.toml                 # 依赖定义
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
       ├─ 构建 System Prompt（含工作记忆）
       ├─ LLM 推理 → 返回工具调用或最终回答
       ├─ 如无工具调用 → 返回最终回答（可选验证）
       ├─ 循环检测（重复调用 3 次 → 强制切换策略）
       ├─ 并行执行工具（带缓存、幂等键、超时）
       ├─ 结果压缩 → 存入工作记忆
       ├─ 每 3 轮保存 Redis 检查点
       └─ 记忆超 15 条 → LLM 摘要压缩
```

### 3.2 三层工作记忆

| 层 | 内容 | 生命周期 |
|----|------|----------|
| **Directives** | 用户约束（如"删除前要确认"） | 永不压缩 |
| **State** | 结构化进度（last_tool, rounds） | 原子更新 |
| **Facts** | 工具返回结果 | 按相关度衰减，可压缩 |

### 3.3 子 Agent 委派

`delegate` 工具会启动独立 ReAct 循环（最多 5 轮、90s 超时），支持 12 个内置角色 + 自动加载的 Skill 角色：

| 类别 | 角色 |
|------|------|
| 战略 | planner, architect |
| 研究 | researcher, analyst |
| 代码 | coder, code_reviewer, build_resolver, tdd_guide |
| 安全 | security_auditor |
| 内容 | writer, executor, guardian |
| Skill | 自动从 config/skills/ 和 SKILL_DIRS 环境变量加载 |

**内置角色**定义在 `config/agents.yaml`，每个角色有：
- `prompt`: 系统提示词
- `tools`: 允许的工具子集
- `knowledge`: 注入的知识库文件（来自 `config/knowledge/*.md`）

**Skill 角色**通过 `skill_loader.py` 自动发现，只需放入 skill 目录即可注册（见 10.4）。

---

## 4. 17 个工具清单

| 工具 | 成本 | 用途 |
|------|------|------|
| `knowledge_recall` | ⚡ 免费 | 知识库语义检索 |
| `read_file` | ⚡ 免费 | 读取文件 |
| `list_dir` | ⚡ 免费 | 浏览目录 |
| `codemap` | ⚡ 免费 | AST 代码结构提取 |
| `web_search` | 🔍 中等 | 网络搜索 |
| `web_fetch` | 🌐 中等 | 抓取网页内容 |
| `docs_lookup` | 📚 中等 | 官方文档查询 |
| `execute_shell` | 🔧 重 | 执行 Shell 命令 |
| `code_run` | 🔧 重 | 多语言代码执行 |
| `write_file` | ✏️ 即时 | 写入文件 |
| `edit_file` | ✏️ 即时 | 文件局部编辑 |
| `browser_fetch` | 🌐 重 | Playwright JS 渲染抓取 |
| `sql_query` | 🗄️ 中等 | 只读 SQL 查询 |
| `git_ops` | 🔧 中等 | Git 操作 |
| `github_ops` | 🐙 中等 | GitHub PR/Issue 管理 |
| `deep_research` | 🔬 重 | 多角度深度研究 |
| `delegate` | 🤖 昂贵 | 委派给专业子 agent |

---

## 5. 自进化机制

### 5.1 工具学习（UCB1 算法）

```
score = success_rate + C × √(ln(total_trials) / tool_trials)
```

- 平衡已验证工具（exploitation）与低频工具（exploration）
- 每日衰减因子 0.995，防止马太效应
- 状态持久化到 Redis

### 5.2 经验循环

```
任务执行 → 提取经验模式 → 存入 Qdrant 向量库
     ↑                              │
     └──── 语义检索相似经验 ←────────┘
```

### 5.3 用户反馈

- 低评分（1-2）→ LLM 回顾分析 → 存入负面经验
- 高评分（4-5）→ 提升经验质量权重

---

## 6. 基础设施

| 组件 | 用途 | 默认端口 |
|------|------|----------|
| PostgreSQL 16 | 执行轨迹、反馈存储 | 5432 |
| Redis 7 | 缓存、检查点、幂等键、工具学习状态 | 6379 |
| Qdrant | 向量存储（经验库、知识 RAG） | 6333 |
| FastAPI | HTTP API 服务 | 8000 |
| 飞书 Bot | WebSocket 长连接（主交付通道） | — |

---

## 7. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tasks` | 提交任务 |
| POST | `/api/v1/tasks/stream` | 流式执行（SSE） |
| POST | `/api/v1/feedback` | 提交评分反馈 |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/models` | 列出可用模型 |
| GET | `/api/v1/metrics` | 系统指标（JSON） |
| GET | `/api/v1/metrics/prometheus` | Prometheus 格式指标 |
| GET | `/api/v1/traces` | 最近分布式追踪 |
| GET | `/api/v1/checkpoints` | 可恢复的任务检查点 |
| POST | `/api/v1/config/reload` | 热重载配置 |

---

## 8. 开发命令

```bash
make install          # 安装依赖
make up               # 启动基础设施（Qdrant/Redis/PostgreSQL）
make dev              # 启动 API 服务（热重载）
make bot              # 启动飞书 Bot
make test             # 运行全部测试
make test-unit        # 仅单元测试
make lint             # 代码检查（ruff）
make format           # 代码格式化
make migrate          # 运行数据库迁移
make health           # 检查服务健康状态
make reload           # 热重载配置
make clean            # 清理缓存
```

---

## 9. 配置说明

主配置文件：`config/settings.yaml`（不入 git，含 API Key）
模板：`config/settings.yaml.example`

关键配置段：

```yaml
llm:
  default: "qwen"                     # 默认模型
  models:
    qwen:
      provider: "openai_compat"
      base_url: "..."
      api_key: "..."
      model: "qwen3.5-plus"

react:
  max_rounds: 10                      # 最大推理轮次
  timeout: 180                        # 总超时（秒）
  tool_timeout: 60                    # 单工具超时
  max_parallel_tools: 5               # 并行工具数
  max_conversation_pairs: 4           # 上下文保留轮数

embedding:
  api_key: "..."                      # Embedding API Key
  base_url: "..."                     # Embedding API 地址
  model: "text-embedding-v3"

feishu:
  app_id: "..."                       # 飞书应用 ID
  app_secret: "..."                   # 飞书应用密钥
```

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

---

## 12. 变更日志

### v0.4.0 — Skill 自动加载机制（当前）

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
