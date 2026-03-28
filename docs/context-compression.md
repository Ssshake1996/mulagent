# 三维智能上下文压缩方案

## 概述

mul-agent 使用三维智能上下文压缩系统管理多轮对话的历史记录。该系统在保留关键信息的同时，将上下文控制在 LLM 的 token 预算内，避免信息丢失和上下文溢出。

核心源码：
- `src/graph/context_compressor.py` — 压缩引擎（分类器、分组器、压缩器、组装器）
- `src/graph/conversation.py` — 会话存储，集成压缩能力

---

## 架构总览

```
用户输入 (current_query)
    │
    ▼
┌──────────────────────────────────────────────────┐
│              ContextAssembler                     │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────┐│
│  │ Dimension 1  │  │ Dimension 2  │  │Dimension 3││
│  │ 语义角色分类 │→│ 话题分组归档 │→│ 相关性压缩││
│  │TurnClassifier│  │ TopicGrouper │  │SmartCompr.││
│  └─────────────┘  └─────────────┘  └───────────┘│
│                                                   │
│          ┌────────────────────────┐               │
│          │ Token 预算组装 (8000字符)│              │
│          └────────────────────────┘               │
└──────────────────────────────────────────────────┘
    │
    ▼
  压缩后上下文 → 送入 LLM
```

---

## 维度一：语义角色分类 (TurnClassifier)

对每条对话消息进行语义分类，标记其在对话中的角色。分类影响后续的保留优先级。

### 语义类型（按保留优先级排序）

| 类型 | 说明 | 示例 |
|------|------|------|
| `requirement` | 用户的原始需求/目标 | "帮我写一个排序函数" |
| `correction` | 用户纠正/重定向 | "不对，应该是降序" |
| `directive` | 持久性指令 | "以后都用中文回答" |
| `final_result` | 已确认的最终输出 | 包含代码块的长回复 |
| `question` | 用户提问（非任务） | "什么是快速排序？" |
| `error_attempt` | 失败的尝试/报错输出 | "Traceback... Error" |
| `intermediate` | 中间思考/部分进展 | 短回复、思考过程 |

### 分类规则

- **用户消息**：基于关键词模式匹配，按 directive → correction → requirement → question 优先级检查
- **助手消息**：结合内容特征 + 前一条用户消息的语义类型推断
  - 包含代码块且长度 > 300 字符 → `final_result`
  - 前一条为 `error_attempt` → `error_attempt`
  - 短回复 (< 100 字符) → `intermediate`

### 支持的关键词（中英文）

```python
# requirement: "帮我", "请", "我想", "implement", "create", ...
# correction:  "不对", "错了", "应该是", "fix", "redo", ...
# directive:   "以后", "记住", "永远", "always", "never", ...
# question:    "什么是", "为什么", "how", "explain", ...
```

---

## 维度二：话题分组与归档 (TopicGrouper)

将连续的对话轮次按话题边界分组，形成 Topic 对象，并管理 hot → cold → recalled 生命周期。

### Topic 数据结构

```python
@dataclass
class Topic:
    id: str                    # MD5 哈希 (12位)
    title: str                 # 话题标题（取自首条 requirement）
    keywords: list[str]        # 提取的关键词（用于相关性计算）
    summary: str               # 归档时生成的摘要
    requirement: str           # 首条用户需求（前200字符）
    final_result_preview: str  # 最终结果预览（前200字符）
    lessons: str               # 从 error → correction 链提取的教训
    turns: list[dict]          # 包含的对话轮次
    status: str                # "hot" | "cold" | "recalled"
    created_at: str            # ISO 时间戳
    updated_at: str            # ISO 时间戳
```

### 话题边界检测

当以下任一条件满足时，认为开始了一个新话题：

1. **显式边界信号**：用户消息包含 "另外"、"换个"、"接下来"、"new topic"、"by the way" 等
2. **需求-结果切换**：当前分组最后一条助手消息是 `final_result`，且新消息是 `requirement`
3. **长度阈值**：当前话题已累积 ≥ 6 轮对话

### 话题生命周期

```
                 append_turn (>30轮)          /recall <query>
    ┌─────┐    ──────────────────→    ┌──────┐    ─────────→    ┌──────────┐
    │ hot │                           │ cold │                   │ recalled │
    └─────┘    ←──────────────────    └──────┘    ←─────────    └──────────┘
                                                   /collapse
```

- **hot**：活跃话题，保留在 `turns` 数组中
- **cold**：归档话题，移入 `archive.topics`，生成摘要
- **recalled**：被用户召回的归档话题，重新参与上下文组装

### 自动归档触发

```python
# 在 ConversationStore.append_turn() 中
# 当 turns 超过 30 条时，自动归档除最新话题外的所有话题
remaining, newly_archived = assembler.auto_archive(turns, archive_threshold=30)

# 手动压缩 (/modify compress) 使用更低的阈值
remaining, newly_archived = assembler.auto_archive(turns, archive_threshold=6)
```

### 关键词提取

基于词频的简单关键词提取：
- 支持中文（≥2字词）和英文（≥3字母词）
- 过滤中英文停用词
- 按频率排序，取前 8 个

---

## 维度三：相关性驱动的动态压缩

根据每个话题与当前用户查询的相关性，决定其压缩级别。

### 相关性评分（三信号加权）

```
relevance = 0.5 × keyword_overlap + 0.3 × recall_intent + 0.2 × time_decay
```

| 信号 | 权重 | 算法 | 说明 |
|------|------|------|------|
| 关键词重叠 | 0.5 | Jaccard 相似度 | query 关键词与 topic 关键词的交集/并集 |
| 召回意图 | 0.3 | 模式匹配 | 检测 "之前"、"上次"、"earlier" 等回顾性词语 |
| 时间衰减 | 0.2 | 指数衰减 (半衰期24小时) | `e^(-0.693 × age_hours / 24)` |

### 四级压缩

| 级别 | 相关性区间 | 输出内容 | 示例 |
|------|-----------|---------|------|
| **Full** | ≥ 0.7 | 完整对话（intermediate 截断到80字符，error 截断到150字符） | 完整的代码讨论过程 |
| **Summary** | 0.3 - 0.7 | 标题 + 需求 + 结果预览 + 教训 | `[Topic: 排序函数] Requirement: ... Result: ...` |
| **Title** | 0.1 - 0.3 | 仅话题标题 | `[Topic: 排序函数]` |
| **Hidden** | < 0.1 | 完全隐藏 | (不输出) |

### 特殊规则

- **最近话题始终 Full**：最后一个 hot topic 强制 relevance = 1.0
- **预算降级**：如果某话题按当前级别超出剩余字符预算，自动降级到更低的压缩级别

---

## 上下文组装流程 (ContextAssembler)

`ContextAssembler.assemble()` 是整个系统的入口，执行以下步骤：

```
Step 1: 语义分类 → 给每条 turn 标记 sem_type
    │
Step 2: 话题分组 → 将 turns 分组为 hot Topics
    │
Step 3: 合并归档 → 将 archive 中的 cold/recalled Topics 加入
    │
Step 4: 相关性评分 → 对每个 Topic 计算 relevance score，分配压缩级别
    │
Step 5: 预算组装 → 按相关性从高到低填充，超预算则降级
    │
    ▼
  输出: 压缩后上下文字符串 (≤ max_chars)
```

### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_chars` | 自动 (`max_tokens × 0.5 × 4`) | 字符预算，从配置中 LLM 的 max_tokens 自动计算；如 max_tokens=65536 则 max_chars=131072 |
| `archive_threshold` | 30 | 自动归档触发的轮次阈值 |
| `max_gap_turns` | 6 | 单个话题最大轮次数 |
| `max_kw` | 8 | 每个话题提取的最大关键词数 |
| Hard cap | 50 turns | 即使不归档，也最多保留50轮 |

> **配置方式**：在 `config/settings.yaml` 中调整 LLM 的 `max_tokens`，压缩预算自动按 `max_tokens × 0.5` 计算。
> 计算逻辑：`max_tokens × 0.5`（token 预算）× `4`（每 token 约 4 字符）= `max_chars`。
> 如果配置读取失败，回退到默认值 8000 字符。

---

## 用户交互命令

通过 CLI (headless/TUI) 可以手动管理上下文：

| 命令 | 功能 |
|------|------|
| `/modify compress` | 手动触发智能压缩（archive_threshold=6） |
| `/modify topics` | 列出所有话题（hot + cold + recalled）及其状态 |
| `/modify expand <topic_id>` | 将归档话题标记为 recalled，重新纳入上下文 |
| `/modify collapse <topic_id>` | 将 recalled 话题重新归档为 cold |
| `/recall <query>` | 按关键词搜索并召回相关的归档话题 |

---

## 数据存储结构

会话数据以 JSON 文件存储在 `data/conversations/` 目录下：

```json
{
  "session_id": "session_abc123",
  "user_id": "user_001",
  "turns": [
    {"role": "user", "content": "...", "ts": "2026-03-28T10:00:00", "sem_type": "requirement"},
    {"role": "assistant", "content": "...", "ts": "2026-03-28T10:00:05", "sem_type": "final_result"}
  ],
  "archive": {
    "topics": [
      {
        "id": "a1b2c3d4e5f6",
        "title": "排序函数实现",
        "keywords": ["排序", "函数", "降序"],
        "summary": "Task: 帮我写排序函数; Result: 实现了降序快排",
        "requirement": "帮我写一个降序排序函数",
        "final_result_preview": "def sort_desc(arr): ...",
        "lessons": "",
        "turns": [...],
        "status": "cold",
        "created_at": "2026-03-28T09:00:00",
        "updated_at": "2026-03-28T09:05:00"
      }
    ]
  },
  "summary": "",
  "directives": ["以后都用中文回答"],
  "entities": {"preferences": [...], "decisions": [...], "topics": [...]},
  "created_at": "2026-03-28T09:00:00",
  "updated_at": "2026-03-28T10:00:05"
}
```

---

## 调用链路

### 正常对话流程

```
用户发送消息
  → runner.py / feishu_bot.py
    → ConversationStore.get_history_for_prompt(session_id, current_query=user_input)
      → ContextAssembler.assemble(turns, current_query, archived_topics, summary)
        → 1. classify_turns()      — 语义分类
        → 2. grouper.group()       — 话题分组
        → 3. compute_relevance()   — 相关性评分
        → 4. compressor.compress() — 按级别压缩
        → 5. 预算组装              — 拼接输出
      ← 返回压缩后上下文字符串
    → 拼入 system prompt 发送给 LLM
```

### 消息追加流程

```
LLM 返回结果
  → ConversationStore.append_turn(session_id, role, content)
    → 追加到 turns[]
    → ContextAssembler.auto_archive(turns, threshold=30)
      → 如果 turns > 30:
        → classify + group
        → 归档除最新外的所有话题到 archive.topics
        → turns 只保留最新话题的轮次
    → Hard cap: turns 最多保留 50 条
    → 写入 JSON 文件
```

---

## 设计决策

| 决策 | 理由 |
|------|------|
| 基于规则而非 LLM 的分类/压缩 | 零延迟、零成本、可离线运行，无需额外 API 调用 |
| Jaccard + 时间衰减 + 召回意图三信号 | 平衡语义相关性、时效性和用户意图 |
| 半衰期 24 小时 | 适合典型工作会话节奏，太旧的话题自然淡出 |
| 最新话题强制 Full | 保证当前上下文完整性 |
| 字符预算而非 token 预算 | 简化计算，约 4 字符 ≈ 1 token |
| 30 轮自动归档 + 50 轮硬上限 | 双重保护，防止上下文无限增长 |
| 支持手动 recall/expand/collapse | 用户可随时调取或隐藏历史话题 |
