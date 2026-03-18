# mul-agent — 通用多Agent协作框架

自进化的多Agent协作系统，基于 LangGraph 编排 + OpenClaw Agent 运行时。

## 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL（本地运行）
- Qdrant（可选，自动降级为内存模式）

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 配置

编辑 `config/settings.yaml`：

```yaml
llm:
  default: "qwen"
  models:
    qwen:
      name: "Qwen 3.5 Plus"
      model: "qwen3.5-plus"
      api_key: "your-api-key"
      base_url: "https://coding.dashscope.aliyuncs.com/v1"
      max_tokens: 65536

database:
  url: "postgresql+asyncpg://mulagent:mulagent@localhost:5432/mulagent"
```

支持多模型配置，API 请求时可通过 `model` 参数选择。

### 数据库初始化

```bash
# 创建用户和数据库
sudo -u postgres createuser mulagent -P  # 密码: mulagent
sudo -u postgres createdb mulagent -O mulagent

# 创建表
PGPASSWORD=mulagent psql -h localhost -U mulagent -d mulagent <<'SQL'
CREATE TABLE task_traces (
    id UUID PRIMARY KEY, session_id VARCHAR(64), user_input TEXT,
    intent_category VARCHAR(64), dag_plan JSON, final_output TEXT,
    status VARCHAR(32) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(), completed_at TIMESTAMPTZ
);
CREATE TABLE subtask_traces (
    id UUID PRIMARY KEY, task_id UUID REFERENCES task_traces(id),
    agent_id VARCHAR(64), subtask_name VARCHAR(256),
    input_data JSON, output_data JSON, status VARCHAR(32) DEFAULT 'pending',
    retry_count INT DEFAULT 0, duration_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE feedbacks (
    id UUID PRIMARY KEY, task_id UUID, rating INT,
    comment TEXT, feedback_type VARCHAR(32) DEFAULT 'explicit',
    created_at TIMESTAMP DEFAULT NOW()
);
SQL
```

### 启动服务

```bash
PYTHONPATH=src uvicorn main:app --host 0.0.0.0 --port 8000
```

### 运行测试

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## API 接口

### 提交任务

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"input": "写一个Python快速排序函数"}'
```

可选参数：
- `session_id`: 会话 ID，用于关联同一对话
- `model`: 指定 LLM 模型 ID（如 `"qwen"`, `"deepseek"`）

响应包含 `timing` 字段，展示各阶段耗时（ms）。

### 流式输出（SSE）

```bash
curl -N -X POST http://localhost:8000/api/v1/tasks/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "解释量子计算"}'
```

实时推送 4 个阶段事件：`dispatch` → `plan` → `execute` → `quality_check` → `done`。

### 提交反馈

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"task_id": "uuid-from-trace", "rating": 5, "comment": "很好"}'
```

反馈会触发自进化：
- rating 4-5：提升相关经验质量权重
- rating 1-2：降低 Agent 成功率 + LLM 复盘存入反面经验
- rating 3：不做调整

### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查（含 DB/Qdrant/LLM 状态） |
| `/api/v1/agents` | GET | 查看所有 Agent 及其成功率 |
| `/api/v1/models` | GET | 查看可用 LLM 模型列表 |

## 架构

```
用户请求 → FastAPI Gateway
              ↓
         LangGraph 编排
         ┌─ dispatch（意图识别 + 经验检索）
         ├─ plan（DAG 任务分解）
         ├─ execute（Agent 执行 + Skill 获取）
         └─ quality_check（质量检查）
              ↓
         结果 + Trace → PostgreSQL
         经验提取 → Qdrant 案例库
              ↓
         用户反馈 → Agent 评分调整 + 经验质量管理
```

### 自进化机制

1. **经验积累**：每次任务完成后 LLM 提取可复用模式，存入 Qdrant
2. **经验检索**：新任务分发时检索相似经验，注入 LLM 上下文辅助决策
3. **反馈闭环**：用户评分驱动 Agent 成功率调整和经验质量升降
4. **Skill 获取**：3 级优先链（历史经验 → Web 搜索 → LLM 生成 + 安全审查）

### 错误恢复

- 所有 LLM 调用支持 2 次自动重试（指数退避）
- Dispatcher/Quality Gate 失败自动降级到关键字/自动通过
- Qdrant 不可用自动降级为内存模式
- DB 不可用不阻塞任务执行（trace 记录静默失败）

## 目录结构

```
config/
  settings.yaml          # 全局配置
  agents.yaml            # Agent 注册表
  prompts/               # LLM 提示词模板
src/
  main.py                # FastAPI 入口
  common/                # 公共模块（config, db, llm, vector, retry, timing）
  agents/                # Agent 层（registry, adapter, skill_acquirer, skill_security）
  graph/                 # LangGraph 编排（dispatcher, dag_builder, orchestrator, quality_gate, state）
  gateway/               # API 网关（routes, streaming）
  evolution/             # 进化层（experience, trace, feedback, feedback_loop）
  models/                # SQLAlchemy 模型
tests/
  unit/                  # 单元测试
  integration/           # 集成测试（需要 PostgreSQL）
  e2e/                   # 端到端测试
```
