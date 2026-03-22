# Chinese Novelist Knowledge Base

## Reference Files Index

All detailed reference files are located in `config/knowledge/novelist/`. Read them via `read_file` when needed during specific creation phases.

| File | Content | When to Use |
|------|---------|-------------|
| `bestseller-techniques.md` | 爆款网文核心技法（一句话人设、力量体系、伏笔经济学、名场面、情绪节奏、配角生态、金句密度） | Planning phase + every 10 chapters audit |
| `chapter-guide.md` | 章节结构设计指南（10种强力开头技巧、节奏控制） | Every chapter opening |
| `chapter-template.md` | 章节文件模板（含元数据和笔记区） | Creating chapter files |
| `character-building.md` | 人物塑造方法（原型、深度技巧、成长弧线） | Planning phase |
| `character-template.md` | 人物档案模板（含成长轨迹表） | Creating character profiles |
| `consistency.md` | 连贯性保证（滑动窗口法、跨卷连贯性） | Post-writing check |
| `content-expansion.md` | 内容扩充技巧（7种自然扩充方法） | When chapter < 3000 chars |
| `dialogue-writing.md` | 对话写作规范（潜台词、角色语言差异化） | Writing dialogue |
| `hook-techniques.md` | 悬念设置技巧（10种钩子类型 + 层级应用指南） | Chapter endings |
| `outline-template.md` | 标准模式大纲模板（<=50章） | Standard mode planning |
| `plot-structures.md` | 情节结构模板（三幕式、英雄之旅、超长篇结构） | Planning phase |
| `punctuation-guide.md` | 标点符号规范（中文标点、节奏控制） | Post-writing check |
| `quality-checklist.md` | 质量检查清单（11维度评分） | Post-writing check |
| `redundancy-check.md` | 冗余描述检测与优化指南（6类冗余、首详后略原则） | Post-writing check |
| `sentence-variety.md` | 句式多样性指南（7种陷阱、8种工具） | Post-writing check |
| `suspense-tracker-template.md` | 四级悬念追踪器模板（全书/卷/篇章/章节） | Planning + per-chapter update |
| `volume-arc-template.md` | 巨著模式卷-篇-章模板（100+章） | Epic mode planning |

## Quality Scripts

| Script | Usage |
|--------|-------|
| `scripts/check_chapter_wordcount.py` | `python scripts/check_chapter_wordcount.py <file>` or `--all <dir>` |
| `scripts/detect_redundancy.py` | `python scripts/detect_redundancy.py <dir> [--threshold N] [--top N] [--output FILE]` |

## Seven Golden Rules

1. **展示而非讲述** - 用动作和对话表现，不要直接陈述
2. **冲突驱动剧情** - 每章必须有冲突或转折
3. **悬念承上启下** - 每章结尾必须留下钩子
4. **首详后略，拒绝冗余** - 同一描述只在首次出现时浓墨重彩，之后点到为止
5. **句式多变，标点精准** - 避免句式单调，标点服务于节奏
6. **伏笔即投资** - 每条伏笔都有回收计划，埋设时像闲笔，回收时震撼
7. **无水章** - 每章必须有"信息增量"。日常章也要展示性格/埋伏笔/推进感情线

## Deep Polish Checklist (去除AI味)

- 去除过度修饰的形容词（"璀璨"、"瑰丽"、"绚烂"等堆砌）
- 减少抽象陈述（把"心中涌起复杂的情感"改为具体动作/对话）
- 打破四字格律（避免"心潮澎湃、热血沸腾"等陈词滥调）
- 增加口语化表达（人物对话要有个性）
- 细节具象化（用具体的视觉/听觉/嗅觉细节替代笼统描述）

## Four-Level Suspense System

| Level | Scope | Count | Lifecycle |
|-------|-------|-------|-----------|
| Book-level | Entire book | 1-3 | Ch.1 plant → final reveal |
| Volume-level | One volume | 2-5 per vol | Vol start → vol end |
| Arc-level | One arc | 1-3 per arc | Within arc |
| Chapter-level | Single chapter | 1-2 per ch | Next chapter response |

Lifecycle: **planted → fermenting → revealed → closed**
