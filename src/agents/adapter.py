"""OpenClaw Agent → LangGraph node adapter.

This is the core glue: wraps an OpenClaw agent so it can be used as a
LangGraph node. Supports three execution backends:
  1. OpenClaw CLI (`openclaw agent --local --json`) — real agent runtime
  2. LLM-direct (LangChain ChatModel) — fallback when OpenClaw unavailable
  3. Mock — for testing without any backend
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from typing import Any

from agents.registry import AgentMeta

logger = logging.getLogger(__name__)

# OpenClaw agent ID mapping: our agent_id → OpenClaw agent_id
# These must match agents registered via `openclaw agents add`
OPENCLAW_AGENT_MAP: dict[str, str] = {
    "thinker": "thinker",
    "retriever": "retriever",
    "executor": "executor",
}


def _openclaw_available() -> bool:
    """Check if the openclaw CLI is installed and on PATH."""
    return shutil.which("openclaw") is not None


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_id: str
    output: str
    success: bool
    metadata: dict[str, Any] | None = None


# ── Intent-specific structured prompt templates ──────────────────────
#
# Each template contains:
#   role       — what the agent is good at
#   thinking   — how the agent should approach the problem
#   format     — output format constraints
#   quality    — quality bar and edge cases to cover

_INTENT_PROFILES: dict[str, dict[str, str]] = {
    "code": {
        "role": "你是一个高级软件工程师。",
        "thinking": (
            "1. 先理解需求，明确输入输出和边界条件\n"
            "2. 选择最合适的语言/框架，说明选择理由（一句话）\n"
            "3. 编写代码，处理边界情况和异常\n"
            "4. 如果代码超过 20 行，附上简要使用说明"
        ),
        "format": (
            "- 代码放在 ```language 代码块中\n"
            "- 关键逻辑处加注释，不要逐行注释\n"
            "- 如有多个文件，用文件名标注分隔"
        ),
        "quality": (
            "- 代码必须可运行，不要伪代码或省略号\n"
            "- 处理空输入、类型错误等边界情况\n"
            "- 遵循该语言的主流编码规范"
        ),
    },
    "research": {
        "role": "你是一个高级研究分析师。",
        "thinking": (
            "1. 明确需要回答什么问题\n"
            "2. 从提供的参考资料中提取关键事实\n"
            "3. 交叉验证，标注来源可信度\n"
            "4. 综合分析，区分事实、推测和观点"
        ),
        "format": (
            "- 先给结论（1-2 句），再展开分析\n"
            "- 引用来源时标注 [来源名称]\n"
            "- 如有多个观点，用对比表格呈现"
        ),
        "quality": (
            "- 区分'已证实'和'据报道'\n"
            "- 信息有时效性时标注日期\n"
            "- 不确定时明确说明而非编造"
        ),
    },
    "data": {
        "role": "你是一个数据分析专家。",
        "thinking": (
            "1. 理解数据背景和分析目标\n"
            "2. 选择合适的分析方法，说明理由\n"
            "3. 执行分析，呈现关键发现\n"
            "4. 给出可操作的建议"
        ),
        "format": (
            "- 数值结果用表格呈现\n"
            "- 关键指标加粗标注\n"
            "- 建议可视化时说明图表类型和坐标轴\n"
            "- 如涉及计算，附上计算代码"
        ),
        "quality": (
            "- 标注数据来源和样本量\n"
            "- 说明统计显著性和置信区间\n"
            "- 避免因果推断，除非有充分证据"
        ),
    },
    "writing": {
        "role": "你是一个资深内容创作者和语言专家。",
        "thinking": (
            "1. 确认目标读者、场景和语气\n"
            "2. 确定核心信息和结构\n"
            "3. 撰写内容，注重可读性和节奏\n"
            "4. 自查：有无歧义、冗余或不当表达"
        ),
        "format": (
            "- 使用清晰的标题和段落结构\n"
            "- 翻译任务保留原文术语，在括号中注释\n"
            "- 长文按章节组织，附目录概要"
        ),
        "quality": (
            "- 无语法和标点错误\n"
            "- 风格前后一致\n"
            "- 翻译忠实原意，兼顾目标语言的自然表达"
        ),
    },
    "reasoning": {
        "role": "你是一个逻辑推理和规划专家。",
        "thinking": (
            "1. 将问题拆解为子问题\n"
            "2. 对每个子问题逐步推理，展示过程\n"
            "3. 检查推理链是否有跳跃或隐含假设\n"
            "4. 综合子结论，给出最终答案"
        ),
        "format": (
            "- 用编号步骤展示推理过程\n"
            "- 数学公式用 LaTeX 格式\n"
            "- 最终结论单独一段加粗"
        ),
        "quality": (
            "- 每一步推理必须有依据\n"
            "- 考虑反例和边界情况\n"
            "- 如果有多种解法，说明最优解及理由"
        ),
    },
    "execute": {
        "role": "你是一个运维和自动化专家。",
        "thinking": (
            "1. 分析要执行的操作及其影响范围\n"
            "2. 评估安全性：是否有副作用、是否可回滚\n"
            "3. 给出具体命令/操作步骤\n"
            "4. 说明预期输出和异常处理"
        ),
        "format": (
            "- 命令放在 ```bash 代码块中\n"
            "- 每个命令附一句说明\n"
            "- 危险操作用 ⚠️ 标注警告"
        ),
        "quality": (
            "- 命令必须可直接复制执行\n"
            "- 涉及删除/覆盖时提供回滚方案\n"
            "- 避免 sudo 或高危操作，除非明确要求"
        ),
    },
    "general": {
        "role": "你是一个通用智能助手。",
        "thinking": (
            "1. 理解用户的真实意图\n"
            "2. 选择最合适的回答方式\n"
            "3. 给出清晰、有条理的回答"
        ),
        "format": (
            "- 根据内容自动选择最佳格式（列表/表格/段落）\n"
            "- 简单问题简短回答，复杂问题结构化展开"
        ),
        "quality": (
            "- 直接回答问题，不绕弯子\n"
            "- 不确定时坦诚说明"
        ),
    },
}


def _build_dynamic_prompt(agent_meta: "AgentMeta", intent: str, context: dict | None) -> str:
    """Dynamically generate system prompt based on agent type, intent, and context.

    Assembled per-task from:
    - Agent identity (thinker/retriever/executor)
    - Intent-specific profile (role, thinking chain, output format, quality bar)
    - Context awareness (retrieved knowledge, dependency results)
    - Language detection (match user's language)
    """
    profile = _INTENT_PROFILES.get(intent, _INTENT_PROFILES["general"])

    has_refs = bool(context and context.get("_skill_context"))
    has_deps = bool(context and any(k != "_skill_context" for k in context))

    # Build context awareness hints
    ctx_hints = []
    if has_refs:
        ctx_hints.append("- 用户提供了参考资料，请充分利用并标注引用")
    if has_deps:
        ctx_hints.append("- 有前置步骤的输出作为上下文，在此基础上继续工作")
    ctx_section = "\n".join(ctx_hints)
    ctx_block = f"\n\n上下文提示:\n{ctx_section}" if ctx_hints else ""

    return (
        f"{profile['role']}\n\n"
        f"思考方式:\n{profile['thinking']}\n\n"
        f"输出格式:\n{profile['format']}\n\n"
        f"质量要求:\n{profile['quality']}"
        f"{ctx_block}\n\n"
        "重要: 用与用户相同的语言回复。"
    )


# ── Executor safety ──────────────────────────────────────────────────

_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",      # rm -rf /
    r"\bsudo\b",             # any sudo
    r"\bdd\s+if=",           # dd disk operations
    r"\bmkfs\b",             # format filesystem
    r"\b:()\s*\{",           # fork bomb
    r"\bshutdown\b",         # shutdown
    r"\breboot\b",           # reboot
    r"\binit\s+0",           # init 0
    r"\bkill\s+-9\s+1\b",   # kill init
    r">\s*/dev/sd",          # overwrite disk
    r"\bchmod\s+-R\s+777\s+/",  # chmod 777 /
    r"\bchown\s+-R\s+.*\s+/\s*$",  # chown /
]


def _is_dangerous_command(cmd: str) -> bool:
    """Check if a shell command matches known dangerous patterns."""
    import re
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return True
    return False


class OpenClawAdapter:
    """Adapts OpenClaw agent into a callable for LangGraph nodes.

    Execution priority:
      1. If use_openclaw=True and the agent has an OpenClaw mapping → CLI call
      2. If llm_client is provided → LLM-direct
      3. Otherwise → mock
    """

    def __init__(
        self,
        agent_meta: AgentMeta,
        llm_client: Any = None,
        use_openclaw: bool = False,
        openclaw_timeout: int = 120,
    ):
        self.agent_meta = agent_meta
        self._llm_client = llm_client
        self._use_openclaw = use_openclaw
        self._openclaw_timeout = openclaw_timeout

    async def execute(self, task: str, context: dict[str, Any] | None = None, intent: str = "general") -> AgentResult:
        """Execute a task through the agent."""
        try:
            openclaw_id = OPENCLAW_AGENT_MAP.get(self.agent_meta.id)

            if self._use_openclaw and openclaw_id and _openclaw_available():
                result = await self._execute_via_openclaw(openclaw_id, task, context)
            elif self.agent_meta.agent_type == "retriever" and self._llm_client is not None:
                result = await self._execute_retriever(task, context, intent)
            elif self.agent_meta.agent_type == "executor" and self._llm_client is not None:
                result = await self._execute_executor(task, context, intent)
            elif self._llm_client is not None:
                result = await self._execute_via_llm(task, context, intent)
            else:
                result = await self._execute_mock(task, context)

            return AgentResult(
                agent_id=self.agent_meta.id,
                output=result,
                success=True,
            )
        except Exception as e:
            logger.error("Agent %s failed: %s", self.agent_meta.id, e)
            return AgentResult(
                agent_id=self.agent_meta.id,
                output=str(e),
                success=False,
            )

    async def _execute_via_openclaw(
        self, openclaw_id: str, task: str, context: dict[str, Any] | None
    ) -> str:
        """Execute via OpenClaw CLI: `openclaw agent --local --json`."""
        # Build the message with context
        if context:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            message = f"Context from prior steps:\n{ctx_str}\n\nTask: {task}"
        else:
            message = task

        cmd = [
            "openclaw", "agent",
            "--agent", openclaw_id,
            "--local",
            "--json",
            "--message", message,
            "--timeout", str(self._openclaw_timeout),
        ]

        logger.info("Calling OpenClaw agent %s for task: %.80s...", openclaw_id, task)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=self._openclaw_timeout + 10
        )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            logger.warning("OpenClaw agent %s returned code %d: %s", openclaw_id, proc.returncode, err_msg)
            # Fallback to LLM-direct
            if self._llm_client is not None:
                logger.info("Falling back to LLM-direct for agent %s", self.agent_meta.id)
                return await self._execute_via_llm(task, context)
            raise RuntimeError(f"OpenClaw agent {openclaw_id} failed: {err_msg}")

        # Parse JSON output — extract text from payloads
        raw = stdout.decode(errors="replace")
        # The JSON may be preceded by plugin log lines; find the first '{'
        json_start = raw.find("{")
        if json_start == -1:
            return raw.strip()

        try:
            data = json.loads(raw[json_start:])
        except json.JSONDecodeError:
            return raw.strip()

        payloads = data.get("payloads", [])
        texts = [p.get("text", "") for p in payloads if p.get("text")]
        result = "\n".join(texts)

        duration = data.get("meta", {}).get("durationMs", "?")
        logger.info("OpenClaw agent %s completed in %sms", openclaw_id, duration)

        return result

    async def _execute_via_llm(self, task: str, context: dict[str, Any] | None, intent: str = "general") -> str:
        """Execute via LLM with dynamically generated system prompt.

        The prompt is assembled from:
        1. Agent type description (thinker/retriever/executor)
        2. Intent-specific focus guidance
        3. Retrieved knowledge context (from skill_acquirer, passed via context)
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from common.retry import retry_async

        system_prompt = _build_dynamic_prompt(self.agent_meta, intent, context)
        messages = [SystemMessage(content=system_prompt)]

        # Build user message — separate retrieved knowledge from task
        parts = []
        if context:
            skill_ctx = context.pop("_skill_context", None)
            if skill_ctx:
                parts.append(f"参考资料:\n{skill_ctx}")
            if context:
                dep_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
                parts.append(f"前置步骤结果:\n{dep_str}")
        parts.append(f"任务: {task}")

        messages.append(HumanMessage(content="\n\n".join(parts)))

        response = await retry_async(self._llm_client.ainvoke, messages, max_retries=2)
        return response.content

    async def _execute_retriever(self, task: str, context: dict[str, Any] | None, intent: str) -> str:
        """Retriever: autonomous multi-round search + synthesis.

        1. LLM generates search queries from the task
        2. Execute each query via mcporter
        3. LLM synthesizes results into a coherent answer
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from common.retry import retry_async

        # Step 1: Generate search queries
        query_messages = [
            SystemMessage(content=(
                "你是一个搜索策略专家。根据用户任务生成 1-3 个最有效的搜索查询词。\n"
                "返回 JSON 数组，每项是一个搜索词。只返回 JSON，不要解释。\n"
                '示例: ["Python async best practices", "asyncio vs threading 2024"]'
            )),
            HumanMessage(content=f"任务: {task}"),
        ]
        try:
            query_resp = await retry_async(self._llm_client.ainvoke, query_messages, max_retries=1)
            queries = json.loads(query_resp.content)
            if not isinstance(queries, list):
                queries = [task]
        except Exception:
            queries = [task]

        # Step 2: Search each query via mcporter
        all_results = []
        if shutil.which("mcporter"):
            for q in queries[:3]:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "mcporter", "call", "WebSearch.bailian_web_search",
                        f"query={q}", "--output", "json",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode == 0:
                        data = json.loads(stdout.decode())
                        for p in data.get("pages", [])[:3]:
                            all_results.append(
                                f"**{p.get('title', '')}**\n"
                                f"{p.get('snippet', '')[:500]}\n"
                                f"Source: {p.get('url', '')}"
                            )
                except Exception as e:
                    logger.debug("Retriever search failed for query '%s': %s", q, e)

        # Also include any pre-fetched context from skill_acquirer
        skill_ctx = ""
        if context:
            skill_ctx = context.pop("_skill_context", "") or ""

        search_material = "\n\n---\n\n".join(all_results) if all_results else ""
        combined_refs = "\n\n".join(filter(None, [skill_ctx, search_material]))

        if not combined_refs:
            combined_refs = "(未找到相关搜索结果，请基于已有知识回答)"

        # Step 3: Synthesize with LLM
        system_prompt = _build_dynamic_prompt(self.agent_meta, intent, context)
        synth_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"参考资料:\n{combined_refs}\n\n任务: {task}"),
        ]
        response = await retry_async(self._llm_client.ainvoke, synth_messages, max_retries=2)
        return response.content

    async def _execute_executor(self, task: str, context: dict[str, Any] | None, intent: str) -> str:
        """Executor: LLM generates commands, then executes them in a sandboxed shell.

        Safety model:
        - LLM generates commands with ```bash blocks
        - Blocked: rm -rf /, sudo, dd, mkfs, and other destructive patterns
        - Each command has a 30s timeout
        - All commands and outputs are logged
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from common.retry import retry_async
        import re

        # Step 1: LLM generates the commands
        system_prompt = _build_dynamic_prompt(self.agent_meta, intent, context)
        parts = []
        if context:
            skill_ctx = context.pop("_skill_context", None)
            if skill_ctx:
                parts.append(f"参考资料:\n{skill_ctx}")
            if context:
                dep_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
                parts.append(f"前置步骤结果:\n{dep_str}")
        parts.append(f"任务: {task}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n\n".join(parts)),
        ]
        response = await retry_async(self._llm_client.ainvoke, messages, max_retries=2)
        llm_output = response.content

        # Step 2: Extract bash commands from ```bash blocks
        bash_blocks = re.findall(r"```bash\n(.*?)```", llm_output, re.DOTALL)
        if not bash_blocks:
            return llm_output  # No commands to execute, return as-is

        # Step 3: Safety check + execute
        results = [llm_output, "\n---\n**执行结果:**\n"]
        for block in bash_blocks:
            commands = [line.strip() for line in block.strip().split("\n")
                        if line.strip() and not line.strip().startswith("#")]
            for cmd in commands:
                if _is_dangerous_command(cmd):
                    results.append(f"⛔ `{cmd}` — 已拦截（危险操作）")
                    logger.warning("Executor blocked dangerous command: %s", cmd)
                    continue

                logger.info("Executor running: %s", cmd)
                try:
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                    out = stdout.decode(errors="replace").strip()
                    err = stderr.decode(errors="replace").strip()

                    if proc.returncode == 0:
                        output = out[:2000] if out else "(无输出)"
                        results.append(f"✅ `{cmd}`\n```\n{output}\n```")
                    else:
                        results.append(f"❌ `{cmd}` (exit {proc.returncode})\n```\n{err[:1000]}\n```")
                except asyncio.TimeoutError:
                    results.append(f"⏰ `{cmd}` — 超时 (30s)")
                except Exception as e:
                    results.append(f"❌ `{cmd}` — {e}")

        return "\n\n".join(results)

    async def _execute_mock(self, task: str, context: dict[str, Any] | None) -> str:
        """Mock execution for testing without LLM."""
        await asyncio.sleep(0.01)  # simulate latency
        ctx_info = f" (with context: {list(context.keys())})" if context else ""
        return f"[{self.agent_meta.name}] completed: {task}{ctx_info}"


class AdapterFactory:
    """Creates OpenClawAdapter instances for registered agents."""

    def __init__(self, llm_client: Any = None, use_openclaw: bool = False, openclaw_timeout: int = 120):
        self._llm_client = llm_client
        self._use_openclaw = use_openclaw
        self._openclaw_timeout = openclaw_timeout
        self._cache: dict[str, OpenClawAdapter] = {}

    def get_adapter(self, agent_meta: AgentMeta) -> OpenClawAdapter:
        if agent_meta.id not in self._cache:
            self._cache[agent_meta.id] = OpenClawAdapter(
                agent_meta,
                self._llm_client,
                use_openclaw=self._use_openclaw,
                openclaw_timeout=self._openclaw_timeout,
            )
        return self._cache[agent_meta.id]
