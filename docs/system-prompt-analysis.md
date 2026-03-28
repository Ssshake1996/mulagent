# System Prompt 对比分析与融合方案

## 一、结构对比

| 维度 | Claude Code | mul-agent (融合前) | 融合方案 |
|------|-------------|-------------------|---------|
| **角色定位** | "interactive agent for software engineering" | "universal task assistant" | "task execution agent" — 强调执行而非建议 |
| **工具策略** | 专用工具优先 + 负面规则（不要用 Bash 读文件） | 按成本排序 cost guide | 保留 cost guide + 新增负面规则（不要用 shell 做 read_file 能做的事） |
| **执行安全** | 可逆性三级分级：自由执行/谨慎/需确认 | 全自动或全确认，无分级 | 采用三级分级：freely/with care/ask before |
| **输出风格** | 详细的简洁性指导 + 格式规范 | "Be concise" 一句话 | 补充具体指导：一句话能说的不用三句 |
| **错误处理** | "不要暴力重试" + 调查根因 | Playbook 5 条规则 | 合并：保留 Playbook + 加入"不要暴力重试"原则 |
| **代码修改** | 最小变更、不过度重构、不加无用注释 | 无 | 新增 Code Modification Principles |
| **安全意识** | XSS/SQL注入/OWASP/凭据保护 | 无 | 新增 Security 段落 |
| **上下文管理** | 提到压缩感知，要求记下关键信息 | 三维压缩系统（语义分类+话题归档+相关性压缩）+ 三层 WorkingMemory | 保留 mul-agent 的完整方案 + 新增 prompt 层面的压缩感知提示 |
| **任务分解** | TaskCreate/TaskUpdate 跟踪 | "plan briefly" | 升级为"numbered todolist + check off steps" |
| **并行策略** | "independent calls in parallel" | 有 max_parallel_tools 但 prompt 未提 | 新增 Parallel Execution 段落 |
| **delegation** | Agent tool 精细规则 | 列了角色但规则简单 | 保留角色体系 + 加入"≤3调用不要 delegate"的负面规则 |

## 二、mul-agent 的上下文管理优势（Claude Code 无对应方案）

mul-agent 在上下文管理方面有完整的多层方案，Claude Code 仅有简单的压缩感知提示：

### 1. 三层 WorkingMemory（单轮 ReAct 内的上下文管理）

```
Layer 1 — Directives: 用户约束，永不压缩
Layer 2 — State: 结构化进度（JSON），原地更新
Layer 3 — Facts: 工具结果，按 relevance 衰减，可压缩
```

- Facts 层有 relevance 衰减机制（每轮 ×0.95），相同工具调用时 boost +0.1
- 支持 LLM-based 和 rule-based 两种压缩路径
- Pinned facts 永不压缩
- 压缩时按工具分组，保留每组 top-3 高 relevance 结果

### 2. 三维对话历史压缩（跨轮的会话级上下文管理）

```
维度 1: 语义角色分类 → requirement/correction/directive/final_result/...
维度 2: 话题分组归档 → hot/cold/recalled 生命周期
维度 3: 相关性驱动压缩 → full/summary/title/hidden 四级
```

- 压缩预算自动从配置的 max_tokens × 0.5 计算
- 支持 /recall 跨话题召回
- 最新话题强制 Full 级别

### 3. Directive 提取（用户约束自动识别）

- 双路径：快速正则 + LLM 补充
- 自动识别"删除前要确认"、"只处理某类数据"等约束
- Directives 永不压缩，始终在 system prompt 最顶部

**融合要点**：保留 mul-agent 的全部上下文管理机制，仅在 prompt 层面新增一句提醒 LLM 主动记下关键信息（来自 Claude Code）。

## 三、融合后的 Prompt 已应用

融合后的 prompt 位于 `src/graph/react_orchestrator.py` 的 `ORCHESTRATOR_PROMPT` 变量。

### 关键改动清单

| 改动 | 来源 | 解决的实际问题 |
|------|------|---------------|
| "task execution agent" | Claude Code 启发 | LLM 不再有"助手习惯"（总想问用户确认） |
| Tool 负面规则段落 | Claude Code | 日志中频繁出现 `cat`/`sed -n` 代替 read_file |
| 可逆性三级分类 | Claude Code | 平衡自主性和安全性 |
| Code Modification Principles | Claude Code | 防止过度重构 |
| Parallel Execution 段落 | Claude Code | 利用 max_parallel_tools=5 |
| Context Awareness 提醒 | Claude Code + mul-agent | 长任务中间结果压缩感知 |
| Security 段落 | Claude Code | 安全意识缺失 |
| 任务类型策略路由 | mul-agent 保留 | 按任务类型选择执行策略 |
| Error Recovery 增强 | 合并两者 | 结构化恢复 + 不暴力重试 |
| Delegation 角色体系 | mul-agent 保留 | 多 agent 协作 |

## 四、还可以改进的地方

### 已发现的可优化点

| 领域 | 当前状态 | 改进方案 | 优先级 |
|------|---------|---------|--------|
| **read_file limit** | 默认读整个文件，大文件导致重复 read_file (日志中反复 read_file offset=N) | 工具描述里提醒：大文件先查 wc -l 或用 limit 参数分段 | 高 |
| **自我评估偏保守** | self-eval 经常 score=2, completeness=2，即使任务已基本完成 | 调整 self-eval prompt，给出更具体的评分标准 | 中 |
| **强制结束质量** | max_rounds 耗尽后 _force_conclude_llm 生成的总结质量不稳定 | force_conclude 应该带上 todolist 执行进度，让总结更准确 | 中 |
| **工具结果截断** | compress_tool_result 对所有工具统一 1500 token 限制 | 按工具类型和任务类型动态调整：代码生成任务允许更大的 code_run 结果 | 低 |
| **跨任务经验复用** | knowledge_recall 存在但日志中几乎从没命中 | 检查经验库入库和检索逻辑，可能是 embedding API 404 导致 | 高 |
| **进度反馈粒度** | 飞书只有"思考中..."一个状态 | 中间结果可以增量推送到飞书卡片，用户能实时看到执行进度 | 中 |
| **Directive 持久性** | 每次新任务需要用户重复说约束 | 已有 persistent directives，但飞书端没暴露 /directives 命令 | 高 |

### 1. knowledge_recall 命中率问题（高优先级）

日志中每次任务结束都出现 `Embedding API returned 404`，说明向量存储的经验入库和检索都不工作：

```
2026-03-28 19:41:46 [common.vector] WARNING: Embedding API returned 404:
```

这导致 knowledge_recall 形同虚设，LLM 每次都得从零开始。需要检查 embedding 配置。

### 2. 飞书 /directives 命令（高优先级）

用户在飞书端无法管理 persistent directives（如"删除文件前要确认"），每次新会话约束都丢失。
建议在飞书 bot 中增加 `/directives` 命令（与 headless 对齐）。

### 3. 飞书增量进度推送（中优先级）

当前飞书只在最终完成时更新卡片。可以在 on_progress 回调中增量更新飞书卡片内容，
让用户实时看到"步骤 1/5: 读取配置文件... ✅" 这样的进度。
