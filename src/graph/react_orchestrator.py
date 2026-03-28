"""ReAct Orchestrator: the core reasoning loop.

Replaces the rigid dispatch → plan → execute → quality_check pipeline
with a single LLM that thinks, acts (calls tools), observes, and repeats.

Key design decisions:
1. LLM decides what to do via tool_use (no intent classification needed)
2. Three-layer WorkingMemory prevents context explosion
3. Directives (user constraints) are NEVER compressed
4. Sub-agents (delegate tool) get isolated context windows
5. Self-healing: errors returned to LLM for strategy adjustment

IMPORTANT: The messages list is append-only within each round. Context from
WorkingMemory is injected into the system prompt, NOT as mid-conversation
HumanMessages, to avoid breaking the AI→ToolMessage pairing required by
the OpenAI tool_use protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from graph.memory import (
    WorkingMemory,
    compress_tool_result,
    extract_directives,
)
from tools.security import pre_tool_hook, post_tool_hook

logger = logging.getLogger(__name__)


# ── Progressive Output Detection ─────────────────────────────────

_AUTO_COMPLETE_SIGNALS = [
    "自行判断", "自动完成", "全部完成", "一次性完成",
    "自己决定", "你来决定", "不用问我", "直接做",
    "auto", "do it all", "just do it",
]


def _should_auto_complete(user_input: str) -> bool:
    """Check if user wants fully automatic mode (no outline confirmation)."""
    lower = user_input.lower()
    return any(signal in lower for signal in _AUTO_COMPLETE_SIGNALS)

# ── Structured Audit Logger ───────────────────────────────────────

_audit_logger = logging.getLogger("mulagent.audit")


def _audit_tool_call(
    tool_name: str, args: dict, round_num: int,
    elapsed: float, is_error: bool, result_len: int,
    trace_id: str = "",
) -> None:
    """Write a structured JSON audit log entry for each tool call.

    Logged at INFO level to a dedicated 'mulagent.audit' logger,
    so it can be routed to a separate file or sink.
    """
    entry = {
        "event": "tool_call",
        "tool": tool_name,
        "args_summary": _brief_args(args)[:120] if args else "",
        "round": round_num + 1,
        "elapsed_s": round(elapsed, 3),
        "status": "error" if is_error else "ok",
        "result_chars": result_len,
    }
    if trace_id:
        entry["trace_id"] = trace_id

    _audit_logger.info(json.dumps(entry, ensure_ascii=False))


# ── System Prompt ─────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """\
You are a task execution agent. You complete tasks autonomously by reasoning and using tools.
You handle diverse tasks: code, research, writing, data processing, file operations, and more.

**Current date: {current_date}**

## Available Tools (sorted by cost: cheap → expensive)
{tool_descriptions}

**Tool cost guide:**
- ⚡ knowledge_recall: instant, free — always try FIRST for known patterns
- ⚡ read_file / list_dir / glob_search / grep_search: instant, free — file discovery and content search
- ⚡ codemap: instant, free — extract code structure (classes, functions, routes) via AST
- ⚡ todo_manage: instant — manage your task list (create/done/list)
- 🔍 web_search: 2-5s, moderate — use when knowledge_recall has no match
- 🌐 web_fetch: 3-10s, moderate — use when you have a specific URL
- 📚 docs_lookup: 3-10s, moderate — fetch official library/framework documentation
- 🔧 execute_shell / code_run: 5-30s, heavy — use for computation, not lookup. code_run supports Python/JS/TS/Go/Rust/Java.
- ✏️ edit_file: instant — surgical find-and-replace, supports replace_all for batch rename (safer than write_file)
- ✏️ write_file: instant — full file write (use for new files or complete rewrites only)
- 🌐 browser_fetch: 10-30s, heavy — JS-rendered page fetch (for SPAs/dynamic content)
- 🗄️ sql_query: 2-30s, moderate — read-only SQL. Use 'schema' query to discover tables.
- 🔧 git_ops: 2-10s — Git operations (diff, status, log, commit, branch)
- 🐙 github_ops: 5-15s — GitHub PR/issue management via gh CLI
- 🔬 deep_research: 30-60s, heavy — multi-angle research with source verification
- 🤖 delegate: 30-120s, expensive — sub-agent for complex multi-step work. Available roles:
  - Strategic: `planner` (decomposition), `architect` (system design)
  - Research: `researcher` (multi-source), `analyst` (data/SQL)
  - Code: `coder` (gen/debug), `code_reviewer` (review), `build_resolver` (fix builds)
  - Quality: `tdd_guide` (testing), `security_auditor` (OWASP)
  - Content: `writer` (docs/content), `executor` (shell/file ops), `guardian` (quality gate)
  - Skills: additional roles auto-loaded from config/skills/ and SKILL_DIRS

**Tool selection — what NOT to do:**
- Do NOT use execute_shell to read files (cat/head/tail) — use read_file instead
- Do NOT use execute_shell for grep/find/ls — use grep_search, glob_search, or list_dir
- Do NOT use write_file when edit_file can make the change (prefer surgical edits over full rewrites)
- Do NOT delegate tasks you can handle in ≤3 tool calls
- Do NOT repeat the same tool call with identical arguments — it will produce the same result
- Do NOT use execute_shell for file search patterns — use glob_search("**/*.py") instead of execute_shell("find . -name '*.py'")
- Do NOT use execute_shell for content search — use grep_search(pattern="def main", file_glob="*.py") instead of execute_shell("grep -r 'def main' *.py")
- Do NOT create tasks in your text output — use todo_manage(action="create") to make them trackable

## Experience System — Learn from History

Before starting complex tasks, use knowledge_recall to check if similar tasks were done before.
Past experiences contain: what strategy worked, what tools to use, what to avoid, estimated rounds.
This saves time and avoids repeating past mistakes.

## How You Work (ReAct Loop)

1. **Check experience**: For non-trivial tasks, call knowledge_recall first to find relevant patterns.
2. **Assess complexity**: simple (0-1 tools) / medium (2-3 tools) / complex (4+ tools)?
3. **Plan**: For complex tasks (4+ steps), use todo_manage to create a task list. Mark each done as you go.
4. **Act**: Call tools — start cheap, escalate if needed. When calls are independent, request them in parallel.
5. **Observe**: Read the result. What did you learn? What's still missing?
6. **Continue or conclude**: All steps done → summary report. More steps → execute next. Do NOT ask permission.

## Autonomous Execution — CRITICAL

**You are an executor, not an advisor. Do the work, don't describe it.**

- When given a task, EXECUTE it fully. Do NOT pause to ask "是否继续?" / "要我开始吗?" / "shall I proceed?".
- "执行", "开始", "do it", "go ahead" authorizes ALL steps through completion.
- Only ask when there is GENUINE ambiguity: missing required info, or multiple valid interpretations.
- If a step fails, fix it yourself. Only report failure after 2-3 attempts with different approaches.
- NEVER end with questions like "是否需要我...?", "您需要我...?", "请确认". Instead, execute and report results.

**Task decomposition:**
- For complex tasks (4+ steps): call todo_manage(action="create", items=[...]) to create a tracked plan.
- For each step: use the appropriate tool(s), then call todo_manage(action="done", task_id=N).
- If a step produces important data you'll need later, write it down explicitly — context may be compressed in long tasks.
- The task list is shown in the context window so you can always see your progress.

## Action Safety — Know What's Reversible

**Freely execute** (low risk, reversible):
- Reading files, browsing directories, running search/lookup queries
- Writing/editing files, running computations, code execution
- Git status/diff/log (read-only git operations)

**Execute with care** (mention what you changed):
- Deleting files, overwriting existing data, modifying configs
- Installing packages, changing system settings

**Ask before executing** (affects external systems or bulk-destructive):
- git push, creating PRs/issues, sending messages to external services
- rm -rf on directories, DROP TABLE, bulk deletions with unclear scope
- Operations the user explicitly asked to confirm (check RULES section)

## Parallel Execution

When multiple tool calls are independent, call them in parallel for efficiency.
Only sequence calls when one result feeds into the next.
Example: reading 3 files → parallel. Reading a file then editing based on content → sequential.

## Task-Type Strategies

**Information gathering**: knowledge_recall → web_search → extract facts → synthesize
**Analysis / comparison**: gather data → code_run for computation → structure findings
**Code generation**: read existing code first → knowledge_recall for patterns → write → test → refine
**Creative / writing**: gather reference if needed → draft directly → minimize tool use
**Multi-step operations**: todolist → execute ALL steps → report summary
**File/data operations**: list structure → read relevant files → process → write → verify

## Code Modification Principles

- Read and understand existing code BEFORE modifying it
- Make the minimal change needed — don't refactor surrounding code or add features not asked for
- Prefer edit_file (surgical) over write_file (full rewrite) for existing files
- Don't add error handling for impossible scenarios or abstractions for one-time operations
- Don't add docstrings/comments/type annotations to code you didn't change

## Error Recovery

When a tool fails, read the error carefully and fix the root cause:
1. **search fails**: simplify query, use core keywords only
2. **fetch fails (timeout/403)**: try different URL or different search query
3. **shell/code fails**: check error message, fix syntax, try alternative approach
4. **file not found**: check path with list_dir, then retry with correct path
5. **After 3 failed attempts on same goal**: stop, report what you tried honestly

Do NOT brute-force retry. Do NOT loop retrying the same failing approach.
When blocked, step back and consider a completely different strategy.

## Output Style

- Respond in the same language as the user
- Lead with the answer or action, not the reasoning process
- Be concise: one sentence > three sentences when possible
- For complex tasks: brief todolist at start → execute → structured summary at end
- When citing info, include the source (URL or tool name)
- NEVER fabricate information. Say "I don't know" rather than guess.

## Security

- Do not introduce vulnerabilities (injection, XSS, path traversal) when writing/editing code
- Do not expose secrets (API keys, passwords, tokens) in output or committed files
- Validate external input at system boundaries; use parameterized queries for SQL

## Context Awareness

- Earlier tool results may be compressed in long-running tasks. Write down key findings as you go.
- The RULES section contains user constraints extracted from this conversation — follow them strictly.
- If conversation history is provided, use it to understand prior context and avoid repeating work.
"""


# ── Task-type aware timeout ────────────────────────────────────────

_TASK_TIMEOUT_RULES: list[tuple[list[str], int, str]] = [
    # (keywords, timeout_seconds, category)
    (["写作", "写一篇", "撰写", "write an essay", "write a report", "write article",
      "写报告", "写文章", "长文", "blog post", "论文", "paper"], 600, "writing"),
    (["翻译", "translate", "全文翻译", "翻译成"], 480, "translation"),
    (["代码", "code", "实现", "implement", "重构", "refactor", "开发",
      "develop", "编程", "programming", "写一个程序"], 480, "coding"),
    (["分析", "analyze", "analysis", "调研", "research", "深度分析",
      "对比", "compare", "评测", "benchmark"], 420, "analysis"),
    (["搜索", "search", "查一下", "查询", "look up", "find"], 180, "search"),
    (["总结", "summarize", "摘要", "summary", "概括"], 180, "summary"),
]


def estimate_timeout(user_input: str, default: int = 300) -> int:
    """Estimate appropriate timeout based on task type and input length.

    Longer tasks (writing, coding) get more time; short queries get less.
    """
    text = user_input.lower()
    for keywords, timeout, _category in _TASK_TIMEOUT_RULES:
        if any(kw in text for kw in keywords):
            # Longer input → likely more complex, add buffer
            length_bonus = min(120, len(user_input) // 100 * 30)
            return timeout + length_bonus
    # Fallback: scale by input length
    if len(user_input) > 500:
        return max(default, 420)
    return default


def classify_task_type(user_input: str) -> str:
    """Return the task category for display purposes."""
    text = user_input.lower()
    for keywords, _timeout, category in _TASK_TIMEOUT_RULES:
        if any(kw in text for kw in keywords):
            return category
    return "general"


def _build_system_prompt(
    tool_descriptions: str,
    memory: WorkingMemory,
    conversation_history: str = "",
) -> str:
    """Build system prompt with memory context and conversation history.

    Memory context and history go into the system prompt (not as mid-conversation
    HumanMessages) to avoid breaking the AI→ToolMessage pairing.
    """
    from datetime import date
    base = ORCHESTRATOR_PROMPT.format(
        tool_descriptions=tool_descriptions,
        current_date=date.today().isoformat(),
    )
    parts = [base]

    if conversation_history:
        parts.append(f"## Conversation History\n{conversation_history}")

    ctx = memory.build_context_message()
    if ctx:
        parts.append(ctx)

    return "\n\n".join(parts)


# ── ReAct Loop ────────────────────────────────────────────────────

async def react_loop(
    user_input: str,
    tools: dict[str, Any],
    llm: Any,
    deps: dict[str, Any] | None = None,
    max_rounds: int = 10,
    timeout: int = 180,
    tool_timeout: int = 60,
    max_parallel_tools: int = 5,
    max_conversation_pairs: int = 4,
    is_sub_agent: bool = False,
    parent_directives: list[str] | None = None,
    conversation_history: str = "",
    on_progress: Any = None,
    result_meta: dict[str, Any] | None = None,
    task_id: str = "",
    session_id: str = "",
) -> str:
    """Execute the ReAct reasoning loop.

    Args:
        user_input: The user's task/question.
        tools: Dict of tool_name → ToolDef.
        llm: LangChain ChatModel with tool_use support.
        deps: Shared dependencies (qdrant, collection_name, etc.).
        max_rounds: Maximum thinking rounds.
        timeout: Overall timeout in seconds.
        tool_timeout: Per-tool execution timeout in seconds.
        max_parallel_tools: Max number of tools to execute in parallel.
        max_conversation_pairs: Max conversation rounds to keep in context.
        is_sub_agent: If True, skip directive extraction (inherited from parent).
        parent_directives: Directives inherited from parent agent (never lost).
        conversation_history: Previous turns for multi-turn context.
        on_progress: Optional async callback(round_num, action, detail).
        task_id: Unique task ID for checkpoint/idempotency.
        session_id: Session ID for checkpoint association.

    Returns:
        The final answer string.
    """
    deps = deps or {}
    memory = WorkingMemory()

    # ── Observability: track task metrics with trace_id ──
    _trace_id = ""
    try:
        from common.trace_context import get_trace_id
        _trace_id = get_trace_id()
    except Exception:
        pass
    try:
        from common.observability import metrics, tracer
        metrics.inc("task_starts_total")
        metrics.set_gauge("active_tasks", metrics.get_counter("task_starts_total") - metrics.get_counter("task_completions_total"))
    except Exception:
        pass
    if _trace_id:
        logger.info("react_loop start (trace=%s, sub_agent=%s)", _trace_id, is_sub_agent)

    # ── Step 0: Inherit parent directives (if sub-agent) ──
    if parent_directives:
        for d in parent_directives:
            memory.add_directive(d)
        logger.info("Inherited %d directives from parent", len(parent_directives))

    # ── Step 1: Extract directives (only for main agent) ──
    if not is_sub_agent:
        directives = await extract_directives(user_input, llm=llm)
        for d in directives:
            memory.add_directive(d)
        if directives:
            logger.info("Extracted %d directives: %s", len(directives), directives)

    # ── Step 2: Prepare tool schemas ──
    from tools.base import ToolDef
    tool_defs: dict[str, ToolDef] = tools
    tool_schemas = [t.to_openai_schema() for t in tool_defs.values()]

    # Build tool descriptions for the system prompt
    tool_desc_lines = []
    for t in tool_defs.values():
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}"
            for k, v in t.parameters.get("properties", {}).items()
        )
        tool_desc_lines.append(f"- **{t.name}**({params}): {t.description}")
    tool_descriptions = "\n".join(tool_desc_lines)

    # ── Step 3: ReAct loop ──
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    # Messages: append-only list. We NEVER insert messages in the middle.
    # Structure: [SystemMessage, HumanMessage, AI, Tool, Tool, AI, Tool, ...]
    # The SystemMessage is rebuilt each round (with updated memory context).
    # The conversation pairs (AI + ToolMessages) are kept intact.
    conversation: list[Any] = []  # AI/Tool pairs, grows each round
    recent_calls: list[str] = []  # Track recent tool call signatures for repeat detection

    # Strategy tracking: record what has been tried and outcomes
    strategies_tried: list[dict] = []  # [{tool, args_summary, outcome, round}]
    tool_fail_counts: dict[str, int] = {}  # tool_name → consecutive failure count
    disabled_tools: set[str] = set()  # tools disabled after repeated failures

    # Cache: bind_tools once since schemas don't change across rounds
    llm_with_tools = llm.bind_tools(tool_schemas)
    from common.retry import retry_async

    # Tool result cache: avoid re-executing identical calls
    _tool_cache: dict[str, str] = {}  # hash(tool_name + args) → result

    async def _run_loop() -> str:
        for round_num in range(max_rounds):
            # Rebuild messages: system (with context + history) + user + conversation
            system_prompt = _build_system_prompt(
                tool_descriptions, memory, conversation_history,
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]

            # Trim conversation to last N complete rounds to avoid context overflow.
            # A "round" = 1 AIMessage + its ToolMessages.
            trimmed = _trim_conversation(conversation, max_pairs=max_conversation_pairs)
            messages.extend(trimmed)

            # Progress callback
            if on_progress:
                try:
                    await on_progress(round_num + 1, "thinking", "")
                except Exception:
                    pass

            # ── Think + Act ──
            start = time.perf_counter()
            try:
                response = await retry_async(
                    llm_with_tools.ainvoke, messages, max_retries=1,
                )
            except Exception as e:
                logger.error("LLM call failed at round %d: %s", round_num + 1, e)
                if round_num > 0:
                    return await _force_conclude_llm(memory, user_input, llm)
                raise

            elapsed = time.perf_counter() - start
            logger.info("[round %d] LLM responded in %.1fs", round_num + 1, elapsed)

            # ── Check for tool calls ──
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # No tool calls → final answer
                answer = response.content or ""
                if answer:
                    # ── Verification loop: self-check before returning ──
                    # Only verify complex answers (multi-tool, multi-round) to
                    # avoid wasting tokens on simple lookups.
                    should_verify = (
                        not is_sub_agent
                        and round_num > 1           # At least 2 rounds of tool use
                        and len(memory.facts) >= 3  # Enough facts to cross-check
                    )
                    if should_verify:
                        verified = await _verify_answer(
                            answer, user_input, memory, llm,
                        )
                        if verified:
                            return verified
                    return answer
                # Empty response with no tool calls
                if round_num > 0:
                    return await _force_conclude_llm(memory, user_input, llm)
                continue

            # Add AI message to conversation (this MUST come before ToolMessages)
            conversation.append(response)

            # ── Report intermediate text (step descriptions/todolist) ──
            if on_progress and response.content:
                text_preview = response.content.strip()[:120]
                if text_preview:
                    try:
                        await on_progress(round_num + 1, "step_text", text_preview)
                    except Exception:
                        pass

            # ── Smart repeat / loop detection ──
            call_sig = "|".join(
                f"{tc['name']}:{sorted(tc.get('args', {}).items())}"
                for tc in tool_calls
            )
            recent_calls.append(call_sig)

            # Check for exact repeats (3x same call)
            is_exact_repeat = (
                len(recent_calls) >= 3
                and recent_calls[-1] == recent_calls[-2] == recent_calls[-3]
            )
            # Check for "similar" repeats (same tool, slightly different args, 3x)
            is_similar_repeat = False
            if not is_exact_repeat and len(recent_calls) >= 3:
                recent_tool_names = [
                    sig.split(":")[0] for sig in recent_calls[-3:]
                ]
                is_similar_repeat = len(set(recent_tool_names)) == 1

            # Check if calling a disabled tool
            calling_disabled = any(tc["name"] in disabled_tools for tc in tool_calls)

            if is_exact_repeat or calling_disabled:
                nudge = (
                    "[System] This tool has been called repeatedly with the same or similar arguments. "
                    "The results won't change. Please synthesize a final answer from what you already have, "
                    "or try a COMPLETELY different approach."
                )
                if strategies_tried:
                    tried_summary = "; ".join(
                        f"{s['tool']}({s['args_summary']})→{s['outcome']}"
                        for s in strategies_tried[-5:]
                    )
                    nudge += f"\n\nStrategies already tried: {tried_summary}"

                logger.warning("Loop detected (round %d): %s", round_num + 1, call_sig[:100])
                for tc in tool_calls:
                    conversation.append(ToolMessage(
                        content=nudge,
                        tool_call_id=tc.get("id", f"call_{round_num}_{tc['name']}"),
                    ))
                continue

            if is_similar_repeat:
                # Soft nudge: same tool 3x with different args
                for tc in tool_calls:
                    logger.info("Similar-tool repeat detected for %s, adding guidance", tc["name"])
                # Don't block, but state will carry the warning to self-check

            # ── Execute tools (parallel when multiple, with caching) ──
            async def _exec_one(tc: dict) -> tuple[dict, str]:
                """Execute a single tool call, return (tc, compressed_result)."""
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_def = tool_defs.get(t_name)
                if t_def is None:
                    return tc, f"Error: unknown tool '{t_name}'"

                # Check cache (skip for stateful tools like execute_shell, code_run, write_file)
                cacheable = t_name not in ("execute_shell", "code_run", "write_file", "delegate")
                cache_key = f"{t_name}:{json.dumps(t_args, sort_keys=True)}" if cacheable else ""
                if cache_key and cache_key in _tool_cache:
                    logger.info("[cache hit] %s(%s)", t_name, _brief_args(t_args))
                    return tc, _tool_cache[cache_key] + "\n(cached result)"

                # ── Pre-tool hook: directive enforcement ──
                blocked = pre_tool_hook(t_name, t_args, list(memory.directives))
                if blocked:
                    logger.info("Tool %s blocked by pre-hook: %s", t_name, blocked[:100])
                    return tc, blocked

                # ── Idempotency key for write operations ──
                _idem_key = ""
                if t_name in ("write_file", "edit_file", "git_ops", "execute_shell") and task_id:
                    import hashlib as _hl
                    _idem_key = f"idem:{task_id}:{t_name}:{_hl.md5(json.dumps(t_args, sort_keys=True).encode()).hexdigest()[:12]}"
                    try:
                        from common.redis_client import is_duplicate
                        if await is_duplicate(_idem_key, ttl=3600):
                            logger.info("Idempotent skip: %s(%s)", t_name, _brief_args(t_args))
                            return tc, f"[idempotent] Operation already completed in this task"
                    except Exception:
                        pass

                _t_start = time.perf_counter()
                try:
                    t_deps = {**deps, "llm": llm, "tools": tool_defs,
                              "parent_directives": list(memory.directives),
                              "memory": memory}
                    raw = await asyncio.wait_for(
                        t_def.fn(t_args, **t_deps), timeout=tool_timeout,
                    )
                except asyncio.TimeoutError:
                    raw = f"Tool {t_name} timed out ({tool_timeout}s)"
                except Exception as e:
                    raw = f"Tool {t_name} error: {e}"
                    logger.warning("Tool %s failed: %s", t_name, e)
                _t_elapsed = time.perf_counter() - _t_start

                # ── Post-tool hook: output sanitization ──
                raw = post_tool_hook(t_name, raw)

                compressed = compress_tool_result(raw, t_name)

                # ── Tool learning: record outcome ──
                _is_err = _is_error_result(compressed)
                try:
                    from evolution.tool_learning import get_tool_learner
                    learner = await get_tool_learner()
                    learner.record_outcome(
                        tool_name=t_name, args=t_args,
                        success=not _is_err, latency_s=_t_elapsed,
                    )
                except Exception:
                    pass

                # ── Observability: record tool metrics ──
                try:
                    from common.observability import metrics as _obs_metrics
                    _obs_metrics.inc("tool_calls_total", tool=t_name)
                    _obs_metrics.observe("tool_duration_seconds", _t_elapsed, tool=t_name)
                    if _is_err:
                        _obs_metrics.inc("tool_errors_total", tool=t_name)
                except Exception:
                    pass

                # ── Structured audit log ──
                _audit_tool_call(
                    tool_name=t_name, args=t_args, round_num=round_num,
                    elapsed=_t_elapsed, is_error=_is_err,
                    result_len=len(compressed), trace_id=_trace_id,
                )

                # Store in cache (only successful, non-error results)
                if cache_key and not _is_err:
                    _tool_cache[cache_key] = compressed

                return tc, compressed

            for tc in tool_calls:
                logger.info("[round %d] Calling tool: %s(%s)",
                            round_num + 1, tc["name"], _brief_args(tc.get("args", {})))
                if on_progress:
                    try:
                        await on_progress(round_num + 1, "tool_call", tc["name"])
                    except Exception:
                        pass

            if max_parallel_tools > 1 and len(tool_calls) > 1:
                sem = asyncio.Semaphore(max_parallel_tools)
                async def _limited(tc):
                    async with sem:
                        return await _exec_one(tc)
                exec_results = await asyncio.gather(*[_limited(tc) for tc in tool_calls])
            else:
                exec_results = [await _exec_one(tc) for tc in tool_calls]

            for tc, compressed in exec_results:
                tool_name = tc["name"]
                tool_id = tc.get("id", f"call_{round_num}_{tool_name}")

                # ── Plan mode: if plan_submit was called, return plan for user review ──
                from tools.task_manager import PLAN_PENDING_MARKER
                if PLAN_PENDING_MARKER in compressed:
                    plan_text = compressed.replace(PLAN_PENDING_MARKER, "").strip()
                    if result_meta is not None:
                        result_meta["plan_pending"] = True
                    return plan_text

                memory.add_fact(tool_name, compressed, round_num)
                memory.update_state("last_tool", tool_name)
                memory.update_state("rounds_completed", round_num + 1)

                # ── Strategy tracking with error classification ──
                error_kind = classify_tool_error(compressed)
                is_failure = error_kind != ToolErrorKind.OK
                args_summary = _brief_args(tc.get("args", {}))[:80]
                outcome = "fail" if is_failure else "ok"
                strategies_tried.append({
                    "tool": tool_name, "args_summary": args_summary,
                    "outcome": outcome, "round": round_num,
                    "error_kind": error_kind,
                })

                # Track consecutive failures per tool
                # Retryable errors get a softer threshold (5 strikes)
                # Fatal errors disable after 3 strikes
                if is_failure:
                    tool_fail_counts[tool_name] = tool_fail_counts.get(tool_name, 0) + 1
                    disable_threshold = 5 if error_kind == ToolErrorKind.RETRYABLE else 3
                    if tool_fail_counts[tool_name] >= disable_threshold:
                        disabled_tools.add(tool_name)
                        logger.warning("Tool %s disabled after %d consecutive %s failures",
                                      tool_name, tool_fail_counts[tool_name], error_kind)
                        compressed += (
                            f"\n\n⚠️ [System] Tool '{tool_name}' has failed "
                            f"{tool_fail_counts[tool_name]} times consecutively "
                            f"({error_kind}). Try a different approach or tool."
                        )
                    elif error_kind == ToolErrorKind.RETRYABLE:
                        compressed += (
                            f"\n\n💡 [System] This was a transient error ({error_kind}). "
                            "You may retry with the same parameters."
                        )
                else:
                    tool_fail_counts[tool_name] = 0  # Reset on success

                conversation.append(ToolMessage(
                    content=compressed,
                    tool_call_id=tool_id,
                ))

            # ── Checkpoint: save state for resumption ──
            if task_id and round_num % 3 == 2 and not is_sub_agent:
                try:
                    from graph.checkpoint import save_checkpoint
                    await save_checkpoint(
                        task_id=task_id,
                        user_input=user_input,
                        memory=memory,
                        round_num=round_num,
                        strategies_tried=strategies_tried,
                        tool_cache=_tool_cache,
                        session_id=session_id,
                    )
                except Exception as _cp_err:
                    logger.debug("Checkpoint save failed: %s", _cp_err)

            # ── Periodic compaction (facts only, directives untouched) ──
            if len(memory.facts) > 15:
                # Try LLM-based summarization (falls back to simple compaction)
                try:
                    await memory.compact_facts_llm(llm, keep_recent=5)
                except Exception:
                    memory.compact_facts(keep_recent=5)

            # ── Self-check at ~50% of max_rounds ──
            if round_num == max_rounds // 2 and not is_sub_agent:
                # Build strategy summary for self-reflection
                strategy_lines = []
                for s in strategies_tried[-8:]:
                    emoji = "✅" if s["outcome"] == "ok" else "❌"
                    strategy_lines.append(f"  {emoji} {s['tool']}({s['args_summary']})")
                strategy_text = "\n".join(strategy_lines) if strategy_lines else "  (none yet)"

                disabled_text = ""
                if disabled_tools:
                    disabled_text = f"\n⚠️ Disabled tools (too many failures): {', '.join(disabled_tools)}"

                conversation.append(HumanMessage(content=(
                    f"[System] Progress check — round {round_num + 1}/{max_rounds}.\n\n"
                    f"**Strategies tried so far:**\n{strategy_text}{disabled_text}\n\n"
                    "Evaluate:\n"
                    "1. Have you completed all planned steps? → Give final summary report.\n"
                    "2. Are there remaining steps? → Continue executing, do NOT ask for permission.\n"
                    "3. Need more info? → Try a DIFFERENT tool or approach than what failed above.\n"
                    "4. Stuck on a step? → Skip it, continue with next steps, report partial results at the end."
                )))

        # Max rounds reached
        return await _force_conclude_llm(memory, user_input, llm, strategies_tried)

    def _populate_meta():
        """Copy memory metadata into result_meta for the caller."""
        if result_meta is not None:
            result_meta["directives"] = list(memory.directives)
            result_meta["facts_count"] = len(memory.facts)
            result_meta["tools_used"] = list({f.source for f in memory.facts})
            result_meta["strategies_tried"] = strategies_tried
            result_meta["disabled_tools"] = list(disabled_tools)

    # Run with overall timeout
    try:
        answer = await asyncio.wait_for(_run_loop(), timeout=timeout)
        _populate_meta()
        # ── Completion metrics + cleanup checkpoint ──
        try:
            from common.observability import metrics as _obs_m
            _obs_m.inc("task_completions_total")
        except Exception:
            pass
        if task_id:
            try:
                from graph.checkpoint import delete_checkpoint
                await delete_checkpoint(task_id)
            except Exception:
                pass
        # ── Save tool learning state periodically ──
        try:
            from evolution.tool_learning import get_tool_learner
            learner = await get_tool_learner()
            await learner.save_to_redis()
        except Exception:
            pass
        return answer
    except asyncio.TimeoutError:
        logger.warning("ReAct loop timed out after %ds", timeout)
        _populate_meta()
        # Timeout — try LLM synthesis with a short deadline
        try:
            return await asyncio.wait_for(
                _force_conclude_llm(memory, user_input, llm), timeout=30,
            )
        except Exception:
            return _force_conclude_fallback(memory, user_input)


async def _verify_answer(
    answer: str, user_input: str, memory: WorkingMemory, llm: Any,
) -> str | None:
    """Verification loop: LLM checks its own answer against gathered facts.

    Returns the (possibly corrected) answer, or None to use the original.
    Only called for non-trivial answers (>0 rounds with facts).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Build fact summary for verification
    fact_texts = []
    for f in memory.facts[-8:]:
        fact_texts.append(f"[{f.source}] {f.content[:300]}")
    facts_str = "\n".join(fact_texts)

    directives_str = ""
    if memory.directives:
        directives_str = "\n用户约束: " + "; ".join(memory.directives)

    messages = [
        SystemMessage(content=(
            "你是一个答案验证器。检查以下回答是否准确、完整，且符合用户约束。\n\n"
            "检查项：\n"
            "1. 回答是否基于实际收集到的信息（不是编造的）？\n"
            "2. 是否完整回答了用户的问题？\n"
            "3. 是否遵守了所有用户约束？\n"
            "4. 是否包含敏感信息需要脱敏？\n\n"
            "如果回答没有问题，返回 PASS\n"
            "如果需要修正，返回修正后的完整回答（不要说'修正后的回答是'，直接给出内容）"
        )),
        HumanMessage(content=(
            f"用户问题: {user_input[:300]}{directives_str}\n\n"
            f"收集到的信息:\n{facts_str}\n\n"
            f"待验证的回答:\n{answer[:1500]}"
        )),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=15)
        verification = response.content.strip()
        if verification.upper().startswith("PASS"):
            logger.debug("Verification passed")
            return None  # Use original answer
        if len(verification) > 50:  # Non-trivial correction
            logger.info("Verification corrected answer (len: %d → %d)",
                       len(answer), len(verification))
            return verification
    except Exception as e:
        logger.debug("Verification skipped: %s", e)

    return None  # Use original on any failure


def _trim_conversation(conversation: list, max_pairs: int = 4) -> list:
    """Trim conversation to the last N complete AI+Tool rounds.

    A "round" is: 1 AIMessage (with tool_calls) followed by 1+ ToolMessages.
    We NEVER cut in the middle of a round — that would break the
    AI→ToolMessage pairing required by the API.

    HumanMessages (like the round-5 self-check) are standalone and safe to include.
    """
    if not conversation:
        return []

    # Find round boundaries: each AIMessage with tool_calls starts a new round
    from langchain_core.messages import AIMessage, HumanMessage

    round_starts = []
    for i, msg in enumerate(conversation):
        if isinstance(msg, (AIMessage, HumanMessage)):
            round_starts.append(i)

    if len(round_starts) <= max_pairs:
        return list(conversation)

    # Keep only the last max_pairs rounds
    cut_at = round_starts[-max_pairs]
    return list(conversation[cut_at:])


async def _force_conclude_llm(
    memory: WorkingMemory, user_input: str, llm: Any,
    strategies_tried: list[dict] | None = None,
) -> str:
    """Use LLM to synthesize a final answer from collected facts."""
    from langchain_core.messages import HumanMessage, SystemMessage

    # Build a summary of collected facts
    fact_summary = []
    for fact in memory.facts[-10:]:
        content = fact.content[:800] if len(fact.content) > 800 else fact.content
        fact_summary.append(f"[{fact.source}] {content}")
    facts_text = "\n\n".join(fact_summary)

    directives_text = ""
    if memory.directives:
        directives_text = "\n用户约束: " + "; ".join(memory.directives)

    # Build progress summary from strategies tried
    progress_text = ""
    if strategies_tried:
        completed = [s for s in strategies_tried if s["outcome"] == "ok"]
        failed = [s for s in strategies_tried if s["outcome"] == "fail"]
        progress_text = f"\n\n执行进度: {len(completed)} 步成功, {len(failed)} 步失败"
        if completed:
            progress_text += "\n已完成的操作:\n" + "\n".join(
                f"  ✅ {s['tool']}({s['args_summary']})" for s in completed[-10:]
            )
        if failed:
            progress_text += "\n失败的操作:\n" + "\n".join(
                f"  ❌ {s['tool']}({s['args_summary']})" for s in failed[-5:]
            )

    # Build state summary
    state_text = ""
    if memory.state:
        state_text = f"\n\n当前状态: {json.dumps(memory.state, ensure_ascii=False)}"

    messages = [
        SystemMessage(content=(
            "你是一个任务助手。用户提出了一个任务，执行过程已达到轮数上限。\n"
            "请根据已收集的信息和执行进度，给出完整的总结报告。\n"
            "要求：\n"
            "- 用和用户相同的语言回答\n"
            "- 明确说明：已完成了哪些步骤、还有哪些未完成\n"
            "- 如果有具体的执行结果（文件已创建、数据已处理等），列出来\n"
            "- 如果有未完成的步骤，给出下一步建议\n"
            "- 不要编造信息"
        )),
        HumanMessage(content=(
            f"用户任务: {user_input}{directives_text}{progress_text}{state_text}\n\n"
            f"已收集的信息:\n{facts_text}\n\n"
            "请根据以上信息给出完整的执行报告。"
        )),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30)
        answer = response.content or ""
        if answer:
            return answer
    except Exception as e:
        logger.warning("LLM synthesis in _force_conclude failed: %s", e)

    # Fallback: structured dump
    return _force_conclude_fallback(memory, user_input)


def _force_conclude_fallback(memory: WorkingMemory, user_input: str) -> str:
    """Fallback when LLM synthesis is unavailable."""
    if not memory.facts:
        return "抱歉，在限定时间/轮次内未能完成任务。请尝试简化任务后重试。"

    grouped: dict[str, list[str]] = {}
    for fact in memory.facts[-8:]:
        grouped.setdefault(fact.source, []).append(fact.content)

    parts = ["根据已收集的信息，以下是结果：\n"]
    for source, contents in grouped.items():
        best = contents[-1]
        if len(best) > 1500:
            best = best[:1500] + "\n..."
        parts.append(f"**{source}**:\n{best}")

    parts.append("\n---\n*注意：任务在限定时间/轮次内未完全完成，以上为已收集到的部分结果。*")
    return "\n\n".join(parts)


class ToolErrorKind:
    """Categorized tool error classification."""
    OK = "ok"
    RETRYABLE = "retryable"      # Transient: timeout, rate limit, network error
    FATAL = "fatal"              # Permanent: invalid params, permission denied, not found

# Patterns matched against first 200 chars of tool output (lowercased)
_RETRYABLE_PATTERNS = [
    "timed out", "timeout", "rate limit", "429", "503",
    "connection refused", "connection reset", "connection error",
    "temporarily unavailable", "server error", "502", "504",
    "try again", "retry", "overloaded",
]
_FATAL_PATTERNS = [
    "permission denied", "403", "401", "unauthorized",
    "invalid", "not supported", "no such file",
    "syntax error", "parse error", "command not found",
]
_ERROR_PATTERNS = [
    "error", "failed", "no results", "not found",
    "unavailable", "empty", "0 results",
]


def classify_tool_error(text: str) -> str:
    """Classify a tool result into OK / RETRYABLE / FATAL.

    Returns one of ToolErrorKind constants.
    """
    prefix = text[:200].lower()

    # Check retryable first (transient errors deserve retry)
    if any(kw in prefix for kw in _RETRYABLE_PATTERNS):
        return ToolErrorKind.RETRYABLE

    # Check fatal (permanent errors should not retry)
    if any(kw in prefix for kw in _FATAL_PATTERNS):
        return ToolErrorKind.FATAL

    # Check generic error patterns
    if any(kw in prefix for kw in _ERROR_PATTERNS):
        return ToolErrorKind.FATAL  # Default unknown errors to fatal

    return ToolErrorKind.OK


def _is_error_result(text: str) -> bool:
    """Check if a tool result indicates an error/failure."""
    return classify_tool_error(text) != ToolErrorKind.OK


def _brief_args(args: dict) -> str:
    """Abbreviate tool args for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 50:
            s = s[:50] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)
