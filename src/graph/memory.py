"""Three-layer Working Memory for ReAct orchestrator.

Layer 1 — Directives: User constraints/rules that must NEVER be compressed.
Layer 2 — State: Structured progress data, updated in place.
Layer 3 — Facts: Tool results and observations, compressible.

The key insight: different information has different lifetimes.
Directives live forever. State is updated atomically. Facts decay and merge.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """A piece of information gathered during execution."""
    source: str        # Which tool produced this
    content: str       # The actual content
    round_num: int     # Which round produced this
    relevance: float = 1.0  # Decays over time
    pinned: bool = False    # Pinned facts are never compacted


@dataclass
class WorkingMemory:
    """Three-layer structured memory.

    - directives: never compressed, always at top of context
    - state: structured dict, updated in place (not appended)
    - facts: tool results, compressible when too many
    """
    directives: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)

    def add_directive(self, directive: str) -> None:
        """Add a user constraint (deduplicated)."""
        if directive not in self.directives:
            self.directives.append(directive)

    def update_state(self, key: str, value: Any) -> None:
        """Update a state field (atomic, in-place)."""
        self.state[key] = value

    def add_fact(self, source: str, content: str, round_num: int,
                 pinned: bool = False) -> None:
        """Add a tool result to the facts layer. Apply relevance decay to older facts."""
        # Decay existing facts slightly (older facts become less relevant)
        for f in self.facts:
            if not f.pinned:
                f.relevance *= 0.95
        # Boost relevance if same tool is called again (confirms importance)
        for f in self.facts:
            if f.source == source:
                f.relevance = min(1.0, f.relevance + 0.1)

        self.facts.append(Fact(
            source=source,
            content=content,
            round_num=round_num,
            relevance=1.0,
            pinned=pinned,
        ))

    def compact_facts(self, keep_recent: int = 5) -> None:
        """Compress the facts layer: merge old facts, keep recent ones intact.

        Uses relevance scores to decide what to keep.
        Only the Facts layer is touched. Directives and State are NEVER compressed.
        Pinned facts are always preserved intact.
        """
        if len(self.facts) <= keep_recent:
            return

        # Separate pinned facts — they are never compacted
        pinned = [f for f in self.facts if f.pinned]
        unpinned = [f for f in self.facts if not f.pinned]

        if len(unpinned) <= keep_recent:
            return

        recent = unpinned[-keep_recent:]
        old = unpinned[:-keep_recent]

        # Group old facts by source tool, preserving high-relevance items
        grouped: dict[str, list[Fact]] = {}
        for f in old:
            grouped.setdefault(f.source, []).append(f)

        # Merge each group into a single summary fact
        merged = []
        for source, facts in grouped.items():
            # Sort by relevance — keep the most important info
            facts.sort(key=lambda f: f.relevance, reverse=True)
            # Keep top 3 by relevance as brief summaries
            previews = [f.content[:120] for f in facts[:3]]
            summary = f"[{len(facts)} calls, best results] " + " | ".join(previews)
            merged.append(Fact(
                source=source,
                content=summary,
                round_num=old[-1].round_num,
                relevance=0.3,
            ))

        self.facts = pinned + merged + recent
        logger.debug("Compacted facts: %d old → %d merged + %d recent + %d pinned",
                      len(old), len(merged), len(recent), len(pinned))

    async def compact_facts_llm(self, llm: Any, keep_recent: int = 5) -> None:
        """LLM-based fact summarization: smarter than simple truncation.

        Preserves key insights and relationships between facts.
        Falls back to compact_facts() on failure.
        Pinned facts are always preserved intact.
        """
        # Filter out pinned facts for compaction
        unpinned = [f for f in self.facts if not f.pinned]
        if len(unpinned) <= keep_recent:
            return

        old = unpinned[:-keep_recent]
        if not old:
            return

        # Build text for LLM summarization
        fact_text = "\n".join(
            f"[{f.source}] {f.content[:300]}" for f in old[-10:]
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=(
                    "请将以下工具调用结果压缩为简洁摘要。保留：\n"
                    "1. 关键事实和数据\n"
                    "2. 重要发现和结论\n"
                    "3. 失败的尝试（避免重复）\n"
                    "去掉重复信息和冗余细节。输出不超过 300 字。"
                )),
                HumanMessage(content=fact_text),
            ]
            import asyncio
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=15)
            summary = response.content.strip()
            if summary:
                pinned = [f for f in self.facts if f.pinned]
                recent = [f for f in self.facts if not f.pinned][-keep_recent:]
                self.facts = pinned + [
                    Fact(source="[summary]", content=summary,
                         round_num=old[-1].round_num, relevance=0.6),
                ] + recent
                logger.debug("LLM-compacted facts: %d old → 1 summary + %d recent",
                            len(old), len(recent))
                return
        except Exception as e:
            logger.debug("LLM fact compaction failed, using fallback: %s", e)

        # Fallback to simple compaction
        self.compact_facts(keep_recent)

    def build_context_message(self) -> str:
        """Assemble all three layers into a single context string.

        Structure:
        1. Directives (top, most prominent)
        2. State (structured progress)
        3. Facts (recent observations)
        """
        parts = []

        # Layer 1: Directives — always first, always visible
        if self.directives:
            rules = "\n".join(f"  - {d}" for d in self.directives)
            parts.append(f"## RULES (must follow at all times)\n{rules}")

        # Layer 2: State — structured, compact
        if self.state:
            state_json = json.dumps(self.state, ensure_ascii=False, indent=2)
            parts.append(f"## Current Progress\n```json\n{state_json}\n```")

        # Layer 3: Facts — sorted by relevance × recency, truncated by tokens
        if self.facts:
            from common.tokenizer import estimate_tokens, truncate_to_tokens
            # Score = relevance * recency_boost (recent rounds score higher)
            max_round = max((f.round_num for f in self.facts), default=0)
            recent = sorted(
                self.facts,
                key=lambda f: f.relevance * (1.0 + 0.1 * (f.round_num - max_round + 10)),
                reverse=True,
            )[:10]
            fact_lines = []
            budget = 3000  # token budget for facts section
            used = 0
            for f in recent:
                line = f"- [{f.source}] {f.content}"
                line_tokens = estimate_tokens(line)
                if used + line_tokens > budget:
                    line = truncate_to_tokens(f"- [{f.source}] {f.content}", budget - used)
                    fact_lines.append(line)
                    break
                fact_lines.append(line)
                used += line_tokens
            parts.append(f"## Gathered Information\n" + "\n".join(fact_lines))

        return "\n\n".join(parts) if parts else ""


# ── Directive Extraction ──────────────────────────────────────────

# Regex patterns that indicate user constraints
_DIRECTIVE_PATTERNS = [
    # Approval / confirmation required
    r"(经过|需要|必须).{0,10}(同意|确认|审批|批准|允许)",
    r"(删除|修改|发送|执行).{0,10}(之?前|先).{0,10}(问我|告诉我|确认|通知)",
    r"不要.{0,10}(直接|自动|擅自).{0,10}(删|改|发|执行)",
    # Scope constraints
    r"(只|仅|不要|不能|禁止).{0,10}(处理|清理|删除|修改|访问|操作)",
    r"(保留|跳过|忽略|排除).{0,6}",
    # Safety
    r"(备份|保存).{0,10}(之?后|再|然后)",
    # Ordering
    r"(先|之?前).{0,10}(再|然后|之?后)",
]


def extract_directives_fast(user_input: str) -> list[str]:
    """Quick rule-based extraction of user constraints from input.

    Returns sentences that match constraint patterns.
    This is the fast path — no LLM call needed.
    """
    # Split into sentences
    sentences = re.split(r"[。！？；\n,.!?;]", user_input)
    directives = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 4:
            continue
        for pattern in _DIRECTIVE_PATTERNS:
            if re.search(pattern, sentence):
                directives.append(sentence)
                break

    return directives


async def extract_directives_llm(user_input: str, llm: Any) -> list[str]:
    """Use LLM to extract user constraints from input.

    More accurate but slower. Used as a complement to fast extraction.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "从用户消息中提取所有约束条件和必须遵守的规则。\n"
            "只提取限制性指令（如：删除前要确认、只处理某类数据、不能执行某操作），"
            "不提取任务描述本身。\n"
            "返回 JSON 数组，每项是一条简洁的指令。没有约束则返回 []。\n\n"
            "示例:\n"
            '输入: "帮我清理邮箱，30天前的广告邮件可以删，但删除前要经过我同意"\n'
            '输出: ["删除任何邮件前必须经过用户确认", "只清理30天前的邮件", "只删除广告类邮件"]'
        )),
        HumanMessage(content=user_input),
    ]

    try:
        from common.retry import retry_async
        response = await retry_async(llm.ainvoke, messages, max_retries=1)
        content = response.content.strip()
        # Handle markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(content)
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.debug("LLM directive extraction failed: %s", e)
        return []


def _needs_llm_extraction(user_input: str, fast_results: list[str]) -> bool:
    """Decide whether to invoke LLM for directive extraction.

    Skip LLM when:
    - Input is very short (simple questions like "今天天气怎么样")
    - Fast extraction already found directives (LLM unlikely to add more)
    - Input has no sentence-like structure (just a keyword)
    """
    # Very short inputs are rarely directive-bearing
    if len(user_input) < 15:
        return False
    # Fast extraction already captured something — usually sufficient
    if fast_results:
        return False
    # Medium-length inputs with multiple clauses may have hidden directives
    # Look for comma/semicolon separators suggesting complex instructions
    separators = sum(1 for c in user_input if c in "，。；,;")
    return separators >= 1 and len(user_input) >= 30


async def extract_directives(user_input: str, llm: Any = None) -> list[str]:
    """Extract user directives using fast rules, optionally enhanced by LLM.

    Fast rules run always. LLM is invoked only when the input is complex
    enough to potentially contain hidden directives that regex missed.
    """
    fast = extract_directives_fast(user_input)

    if llm is not None and _needs_llm_extraction(user_input, fast):
        llm_directives = await extract_directives_llm(user_input, llm)
        # Merge, deduplicate
        seen = set(fast)
        for d in llm_directives:
            if d not in seen:
                fast.append(d)
                seen.add(d)

    return fast


def compress_tool_result(raw: str, tool_name: str, max_tokens: int = 0) -> str:
    """Compress a tool result before storing in Facts.

    Different tools have different compression strategies.
    Default limit from config (react.compress.tool_result_max_tokens, default 1500).

    Uses token-based truncation for accurate context management.
    """
    if max_tokens <= 0:
        try:
            from common.config import get_settings
            max_tokens = get_settings().react.compress.tool_result_max_tokens
        except Exception:
            max_tokens = 1500
    from common.tokenizer import estimate_tokens, truncate_to_tokens, truncate_middle

    if estimate_tokens(raw) <= max_tokens:
        return raw

    if tool_name == "web_search":
        # Keep first 5 result blocks
        blocks = raw.split("---")
        kept = "---".join(blocks[:5])
        result = truncate_to_tokens(kept, max_tokens)
        if len(blocks) > 5:
            result += f"\n... ({len(blocks)} results total)"
        return result

    elif tool_name == "execute_shell":
        # Keep head (exit code) + tail (recent output)
        return truncate_middle(raw, max_tokens)

    else:
        return truncate_to_tokens(raw, max_tokens)
