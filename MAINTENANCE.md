# mul-agent 项目维护说明

> Universal Multi-Agent Collaboration Framework — 基于 ReAct 推理循环的多智能体协作框架，支持飞书 Bot、CLI (TUI/Headless)、HTTP API 多通道交付。

---

## 1. 架构总览

```
             ┌─── 飞书 Bot (WebSocket) ───┐
             │                             │
用户 ────────┼─── CLI TUI / Headless ─────┼──→ SessionManager ──→ run_react()
             │                             │          │
             └─── HTTP API (FastAPI) ─────┘          ▼
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

**核心设计**：
- LLM 在推理循环中自主决定调用什么工具，无需意图分类或 DAG 规划
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
├── scripts/
│   ├── setup.sh                   # Linux 统一管理脚本（systemd/服务状态/日志）
│   └── setup.ps1                  # Windows 一键安装脚本（PowerShell）
├── tests/
│   ├── unit/                      # 单元测试（19 个文件，236 个用例）
│   ├── integration/               # 集成测试
│   └── e2e/                       # 端到端测试
├── docker/docker-compose.yaml     # 容器化部署编排
├── .github/workflows/ci.yml       # CI/CD 流水线
├── Dockerfile                     # 应用镜像
├── Makefile                       # 开发命令
├── alembic/                       # 数据库迁移
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

## 6. 基础设施与服务管理

### 6.1 服务清单

| 组件 | 用途 | 管理方式 | 必需 |
|------|------|----------|------|
| PostgreSQL 16 | 执行轨迹、反馈存储 | systemd 系统服务（`postgresql@16-main`） | 是 |
| Redis 7 | 缓存、检查点、幂等键、工具学习状态 | 可选部署 | 否（优雅降级） |
| Qdrant | 向量存储（经验库、知识 RAG） | 可选部署 | 否（fallback 内存） |
| 飞书 Bot | WebSocket 长连接 | systemd 用户服务（`mulagent-feishu`） | 是 |
| FastAPI | HTTP API 服务 | 手动/Docker | 按需 |
| CLI | TUI / Headless / 单次执行 | `mulagent` 命令 | 按需 |

### 6.2 运维脚本（scripts/setup.sh）

`scripts/setup.sh` 负责**基础设施运维**（检查/启停服务、查看日志）。默认模式附带启动 CLI，但核心职责是服务管理：

```bash
# 运维操作
./scripts/setup.sh --status        # 查看所有服务状态
./scripts/setup.sh --restart       # 重启飞书 Bot（代码更新后）
./scripts/setup.sh --stop          # 停止飞书 Bot
./scripts/setup.sh --logs [N]      # 查看 Bot 最近 N 行日志
./scripts/setup.sh --infra         # 仅检查基础设施，不启动 CLI

# 便捷启动（检查服务 → 自动修复 → 启动 CLI）
./scripts/setup.sh                 # 检查服务 + 启动 TUI
./scripts/setup.sh --headless      # 检查服务 + 启动 Headless
```

> **与 `mulagent` 的区别**：`mulagent` 是纯用户交互入口（不管服务状态），`setup.sh` 是运维工具（确保服务就绪后可选启动 CLI）。日常使用直接 `mulagent`，首次部署或排障用 `setup.sh`。

### 6.3 Windows 一键安装（scripts/setup.ps1）

Windows 用户使用 PowerShell 脚本安装：

```powershell
# 一键安装 + 启动
.\scripts\setup.ps1                     # 安装依赖 + 启动 TUI
.\scripts\setup.ps1 -Headless           # 安装依赖 + Headless REPL
.\scripts\setup.ps1 -c "帮我查天气"     # 单次执行

# 检查服务状态
.\scripts\setup.ps1 -Status             # 检查 PG/Redis/Qdrant/API 端口

# 仅安装不启动
.\scripts\setup.ps1 -Infra              # 安装 + 检查基础设施
```

**前提条件**：
- Python 3.10+（安装时勾选 "Add Python to PATH"）
- 可选：Docker Desktop（用于 PostgreSQL/Redis/Qdrant）

> 脚本自动处理：创建 venv → 安装包 → 检查基础设施端口 → 数据库迁移 → 启动 CLI。所有基础设施组件均为可选，核心 ReAct 循环仅需 LLM 即可运行。

### 6.4 systemd 服务（仅 Linux）

飞书 Bot 作为用户级 systemd 服务自启动：

```
~/.config/systemd/user/mulagent-feishu.service
```

常用命令：
```bash
systemctl --user status mulagent-feishu    # 查看状态
systemctl --user restart mulagent-feishu   # 重启（代码更新后）
journalctl --user -u mulagent-feishu -f    # 实时日志
```

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

## 8. 开发与运维命令

### 8.1 Make 命令

```bash
make install          # 安装依赖
make up               # 启动基础设施（Qdrant/Redis/PostgreSQL）
make dev              # 启动 API 服务（热重载）
make bot              # 启动飞书 Bot
make test             # 运行全部测试（205 个用例）
make test-unit        # 仅单元测试
make lint             # 代码检查（ruff）
make format           # 代码格式化
make migrate          # 运行数据库迁移
make health           # 检查服务健康状态
make reload           # 热重载配置
make clean            # 清理缓存
```

### 8.2 CLI 命令

```bash
mulagent                         # TUI 模式（Textual 富终端，支持文本选择复制）
mulagent --headless              # Headless REPL（纯文本）
mulagent -c "帮我分析这段代码"     # 单次执行
mulagent --model deepseek        # 指定模型
mulagent --session <id>          # 恢复历史会话
```

**全局安装**（任意目录可用）：
```bash
# 方式 1：符号链接（推荐，开发环境）
ln -s $(pwd)/.venv/bin/mulagent ~/.local/bin/mulagent

# 方式 2：pip install（新环境部署）
pip install -e ".[cli]"
```

路径解析策略（`_find_project_root()`）：
1. `MULAGENT_ROOT` 环境变量（显式指定）
2. 从 CWD 向上查找 `config/settings.yaml`
3. 从源码文件位置向上查找（editable install）
4. `~/.mulagent/`（全局安装兜底）

**REPL 内置命令**：

| 命令 | 说明 |
|------|------|
| `/new` | 新建会话 |
| `/resume <id>` | 恢复历史会话 |
| `/model <id>` | 切换模型 |
| `/sessions` | 会话列表 |
| `/modify` | 上下文管理（见下方） |
| `/quit` | 退出 |

**`/modify` 上下文管理命令**：

| 命令 | 说明 |
|------|------|
| `/modify` 或 `/modify list` | 列出所有对话轮次（带索引和预览） |
| `/modify view <n>` | 查看第 n 轮完整内容 |
| `/modify edit <n>` | 交互式编辑第 n 轮（TUI: 编辑浮层 Ctrl+S/Esc；Headless: 打开 $EDITOR） |
| `/modify del <n>` | 删除第 n 轮 |
| `/modify del <n-m>` | 批量删除第 n~m 轮 |
| `/modify clear` | 清空所有对话 |
| `/modify summary` | 查看对话摘要 |
| `/modify compress` | 强制压缩旧对话为摘要 |

TUI 模式快捷键：**Ctrl+N** 新会话、**Ctrl+Q** 退出、**Esc** 聚焦输入框。聊天面板支持鼠标选择文本复制。

> `mulagent` 不检查基础设施状态（LLM 无配置时会报错退出，PG/Redis/Qdrant 不可用时优雅降级）。首次部署请先用 `./scripts/setup.sh --infra` 确认服务就绪（见 6.2）。

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
| SessionManager 适配器层 | 飞书/CLI/API 共享会话存储，切换通道不丢上下文 |
| CLI 使用全栈依赖（PG/Redis/Qdrant） | 与服务端行为一致，trace 和 checkpoint 完整可用 |
| systemd 用户服务管理 Bot | 开机自启、崩溃自恢复、日志归档，比 nohup 可靠 |
| Textual 作为可选依赖 | headless 模式零额外依赖，TUI 仅 `pip install mulagent[cli]` |
| TextArea 替代 RichLog | 牺牲富文本标记换取原生文本选择复制能力 |
| /modify 上下文 CRUD | 用户可精细控制对话上下文，手动压缩降低 token 消耗 |
| PROJECT_ROOT 统一解析 | 环境变量→CWD→源码→~/.mulagent 四级降级，支持全局安装 |

---

## 12. 变更日志

### v0.12.0 — 动态超时 + 进度条单行刷新（当前）

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
  - 覆盖 17 个命令：`/new`、`/help`、`/sessions`、`/resume`、`/model`、`/modify *`、`/directives *`、`/quit`
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
