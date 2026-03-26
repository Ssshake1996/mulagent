# mul-agent — 通用多Agent协作框架

基于 ReAct 推理循环的多智能体协作框架，支持飞书 Bot、CLI (TUI/Headless)、HTTP API 多通道交付。具备自我进化能力。

## 快速开始

### 一键安装（推荐）

**Linux / macOS：**

```bash
git clone https://github.com/Ssshake1996/mulagent.git
cd mulagent
./scripts/setup.sh
```

**Windows（PowerShell）：**

```powershell
git clone https://github.com/Ssshake1996/mulagent.git
cd mulagent
.\scripts\setup.ps1
```

安装脚本会自动完成：
1. 创建 Python 虚拟环境 (`.venv`)
2. 安装依赖包
3. 交互式询问是否安装数据库（PostgreSQL/Redis/Qdrant，**全部可选**）
4. 启动 CLI

> 核心功能仅需 **Python 3.10+** 和 **LLM API Key**，不安装任何数据库也能正常使用。

### 手动安装

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cli]"

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[cli]"
```

### 配置

首次使用运行初始化向导：

```bash
mulagent init
```

支持通义千问、DeepSeek、OpenAI、Ollama、自定义 5 种 LLM 提供商。

也可手动编辑 `config/settings.yaml`（参考 `config/settings.yaml.example`）：

```yaml
llm:
  default: "qwen"
  models:
    qwen:
      name: "Qwen 3.5 Plus"
      model: "qwen3.5-plus"
      api_key: "your-api-key"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      max_tokens: 65536
```

### 启动

```bash
mulagent                         # TUI 模式（富终端）
mulagent --headless              # Headless REPL（纯文本）
mulagent -c "帮我写一段排序代码"   # 单次执行
```

## 架构

```
             +--- Feishu Bot (WebSocket) ---+
             |                              |
User --------+--- CLI TUI / Headless ------+---> SessionManager ---> run_react()
             |                              |          |
             +--- HTTP API (FastAPI) -------+          v
                                              +---------------+
                                              |  ReAct Loop   |  <-- Core reasoning
                                              |  (LLM driven) |
                                              +-------+-------+
                                                      | Tool calls
                                     +----------------+----------------+
                                     v                v                v
                              +-----------+   +-----------+    +-------------+
                              | Search    |   | File/Code |    |  delegate   |
                              | web_search|   | read_file |    | (sub-agent) |
                              | knowledge |   | code_run  |    | 12 roles    |
                              +-----------+   | git_ops   |    +------+------+
                                              | sql_query |           |
                                              | browser   |     Re-enter
                                              +-----------+     ReAct Loop
```

**核心设计**：LLM 在推理循环中自主决定调用什么工具，无需意图分类或 DAG 规划。

## 主要功能

- **17 个内置工具**：搜索、文件操作、代码执行、浏览器、SQL、Git/GitHub、深度研究等
- **12 个专业子 Agent**：planner、architect、researcher、coder、security_auditor 等
- **自我进化系统**：诊断 → 处方 → 执行闭环（`mulagent --evolve`）
- **外部项目吸收**：给定 Git URL 自动分析并融合能力（`mulagent --absorb <url>`）
- **智能上下文压缩**：语义分类 + 话题归档 + 相关性驱动动态压缩
- **动态任务超时**：根据任务类型自动调整超时时间
- **多通道交付**：CLI TUI、Headless REPL、HTTP API、飞书 Bot
- **全部数据库可选**：PostgreSQL/Redis/Qdrant 不可用时优雅降级

## REPL 命令

| 命令 | 说明 |
|------|------|
| `/new` | 新建会话 |
| `/resume <id>` | 恢复历史会话 |
| `/model <id>` | 切换模型 |
| `/modify` | 上下文管理（list/view/edit/del/compress/topics） |
| `/recall <keyword>` | 召回已归档的对话话题 |
| `/evolve` | 自我进化（diagnose/propose/auto/full） |
| `/absorb <url>` | 吸收外部 Git 项目 |
| `/directives` | 持久指令管理 |
| `/quit` | 退出 |

## API

```bash
# 提交任务
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"input": "写一个Python快速排序函数"}'

# 流式执行 (SSE)
curl -N -X POST http://localhost:8000/api/v1/tasks/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "解释量子计算"}'

# 健康检查
curl http://localhost:8000/api/v1/health
```

## 数据库（可选）

所有数据库组件均为可选，不安装也能使用核心功能：

| 组件 | 用途 | 不安装时的影响 |
|------|------|---------------|
| PostgreSQL | 任务追踪、反馈存储 | trace 功能不可用 |
| Redis | 缓存、检查点、幂等键 | 无缓存、无断点续作 |
| Qdrant | 向量存储、经验库、知识 RAG | fallback 到内存向量 |

```bash
# 通过 Docker Compose 一键启动所有数据库
docker compose -f docker/docker-compose.yaml up -d

# 或在安装时选择
./scripts/setup.sh --with-db          # Linux
.\scripts\setup.ps1 -WithDB           # Windows
```

## 开发

```bash
make test             # 运行测试（308+ 用例）
make lint             # 代码检查
make dev              # 启动 API 服务（热重载）
make up / make down   # 启动/停止基础设施 (Docker)
```

## 文档

详细架构设计、工具清单、配置说明、变更日志等见 [MAINTENANCE.md](MAINTENANCE.md)。

## License

MIT
