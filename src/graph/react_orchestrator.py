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

ORCHESTRATOR_PROMPT_BASE = """\
You are a task execution agent. You complete tasks autonomously by reasoning and using tools.

**Environment: {environment_context}**

## Available Tools
{tool_descriptions}

## Tool Selection Guidelines
- Prefer `glob_search` / `grep_search` / `read_file` over `execute_shell` for search and reading
- Prefer `edit_file` over `write_file` for modifying existing files
- Prefer delegating to a specialized skill when one exists for the domain
- Parallelize independent tool calls in the same round to save time
- Avoid repeating the same tool call with identical arguments

## How You Work (ReAct Loop)
1. **Assess**: What does the task need? How complex is it?
2. **Plan if needed**: For complex tasks (5+ steps), create a task list with todo_manage.
3. **Act**: Use the right tool for each step. Parallelize independent calls.
4. **Observe**: What did you learn? Did anything fail? Adjust approach if needed.
5. **Conclude**: Summarize results. The user cannot see intermediate tool calls.

## Autonomous Execution
- Execute tasks directly. Only ask when there is genuine ambiguity.
- If a step fails, diagnose and try a different approach before reporting failure.
- Don't end responses with unnecessary questions like "需要我继续吗?"
- Write down key findings as you go — earlier context may be compressed.

## Output Style
- Respond in the same language as the user
- Lead with the answer or action, not reasoning
- Be concise. NEVER fabricate information.
- For multi-step tasks, include a brief summary proportional to the task complexity.

## Context Awareness
- The RULES section contains user constraints — follow them strictly.
- Earlier tool results may be compressed. Write down key findings as you go.
"""

# ── First-round-only instructions (injected once in round 0) ──────
ORCHESTRATOR_PROMPT_EXTENDED = """\
## Action Safety — Reversibility is the Decision Axis

**Freely execute** — read-only or easily reversible:
- All search/read/browse operations
- Writing/editing local files, running computations, git status/diff/log

**Execute with care** — mention what changed:
- Deleting files, overwriting data, modifying configs, installing packages
- Git commit (local, reversible, but mention it)

**Confirm before executing** — hard to reverse or affects others:
- git push, creating PRs/issues, sending external messages
- rm -rf, DROP TABLE, force push, reset --hard, branch -D
- Operations the user explicitly asked to confirm (check RULES section)

## Minimal Change Principle
- Read and understand existing code BEFORE modifying it.
- Make the minimal change needed — don't refactor surrounding code.
- Don't add error handling for impossible scenarios or abstractions for one-time use.
- Don't add docstrings, comments, or type annotations to code you didn't change.
- Three similar lines of code is better than a premature abstraction.

## Error Recovery
When a tool fails, read the error and fix the root cause:
1. **search fails**: simplify query, use core keywords only
2. **fetch fails (timeout/403)**: try different URL or search query
3. **shell/code fails**: check error, fix syntax, try alternative approach
4. **file not found**: check path with list_dir, retry with correct path
5. **After 3 failed attempts**: stop, report what you tried honestly

## Numerical Accuracy
For non-trivial calculations, use code_run instead of mental math.

## Security
- Do not introduce vulnerabilities (injection, XSS, path traversal) in code
- Do not expose secrets (API keys, passwords, tokens) in output or commits
- Validate external input at system boundaries; use parameterized queries for SQL

## Experience System
For complex or recurring tasks, use knowledge_recall to check for relevant past experiences.
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


def _load_project_directives() -> str:
    """Load project-level directives from .mulagent.md or AGENT.md.

    Searches: CWD → parent directories (up to 5 levels) → home directory.
    Returns the content or empty string.
    """
    from pathlib import Path
    candidates = [".mulagent.md", "AGENT.md"]

    # Search CWD and up to 5 parent directories
    cwd = Path.cwd()
    for _ in range(6):
        for name in candidates:
            f = cwd / name
            if f.is_file():
                try:
                    content = f.read_text(errors="replace").strip()
                    if content:
                        logger.info("Loaded project directives from %s", f)
                        return content
                except Exception:
                    pass
        parent = cwd.parent
        if parent == cwd:
            break
        cwd = parent

    # Fallback: home directory global directives
    home = Path.home()
    for name in [".mulagent.md"]:
        f = home / name
        if f.is_file():
            try:
                content = f.read_text(errors="replace").strip()
                if content:
                    logger.info("Loaded global directives from %s", f)
                    return content
            except Exception:
                pass

    return ""


# ── Post-write validation rules (parsed from .mulagent.md) ──
_cached_validation_rules: dict[str, list[str]] | None = None


def _load_validation_rules() -> dict[str, list[str]]:
    """Parse ## Validation Rules from project directives.

    Expected format in .mulagent.md:
        ## Validation Rules
        - recent_chapters.md: 每章最多3行，总行数≤ chapter_count * 3
        - hooks_tracker.json: 必须是合法JSON
        - chapter_*.md: 中文字数3000-4000

    Returns: {filename_pattern: [rule1, rule2, ...]}
    """
    global _cached_validation_rules
    if _cached_validation_rules is not None:
        return _cached_validation_rules

    _cached_validation_rules = {}
    content = _cached_project_directives or ""
    if not content:
        return _cached_validation_rules

    import re
    # Find ## Validation Rules section
    match = re.search(r'##\s*Validation\s*Rules\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if not match:
        return _cached_validation_rules

    for line in match.group(1).strip().split("\n"):
        line = line.strip().lstrip("- ")
        if ":" not in line:
            continue
        pattern, _, rule = line.partition(":")
        pattern = pattern.strip()
        rule = rule.strip()
        if pattern and rule:
            _cached_validation_rules.setdefault(pattern, []).append(rule)

    if _cached_validation_rules:
        logger.info("Loaded %d validation rules for %d file patterns",
                    sum(len(v) for v in _cached_validation_rules.values()),
                    len(_cached_validation_rules))
    return _cached_validation_rules


def _match_validation_rules(filepath: str) -> list[str]:
    """Find validation rules matching a file path."""
    import fnmatch
    from pathlib import Path
    rules = _load_validation_rules()
    if not rules:
        return []
    filename = Path(filepath).name
    matched = []
    for pattern, rule_list in rules.items():
        if fnmatch.fnmatch(filename, pattern):
            matched.extend(rule_list)
    return matched


async def _run_post_write_validation(
    filepath: str, rules: list[str], tool_defs: dict, llm: Any,
) -> str | None:
    """Run validation rules after write_file/edit_file.

    Uses code_run(python) to check file content against rules.
    Returns error message if validation fails, None if passes.
    """
    rules_text = "\n".join(f"  - {r}" for r in rules)
    validation_code = f'''
import json, os, re

filepath = {filepath!r}
errors = []

# Read the file
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
except Exception as e:
    print(f"VALIDATION_ERROR: Cannot read file: {{e}}")
    raise SystemExit(0)

lines = content.strip().split("\\n")
line_count = len(lines)

# Check JSON validity for .json files
if filepath.endswith(".json"):
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {{e}}")

# Count Chinese characters
cjk_count = len(re.findall(r'[\\u4e00-\\u9fff]', content))

# Report basic stats for LLM to evaluate against rules
print(f"FILE_STATS: lines={{line_count}}, chinese_chars={{cjk_count}}, bytes={{len(content)}}")

if errors:
    for e in errors:
        print(f"VALIDATION_ERROR: {{e}}")
else:
    print("VALIDATION_OK")
'''
    # Execute validation via code_run
    code_run = tool_defs.get("code_run")
    if not code_run:
        return None

    try:
        result = await asyncio.wait_for(
            code_run.fn({"language": "python", "code": validation_code}),
            timeout=10,
        )
        # If JSON is invalid, return error immediately
        if "VALIDATION_ERROR" in result:
            errors = [l.replace("VALIDATION_ERROR: ", "") for l in result.split("\n") if "VALIDATION_ERROR" in l]
            return f"⚠️ Post-write validation failed for {filepath}:\n" + "\n".join(f"  - {e}" for e in errors) + f"\n\nRules to follow:\n{rules_text}\nPlease fix the file."
        return None
    except Exception as e:
        logger.debug("Post-write validation skipped: %s", e)
        return None


def _fetch_git_context() -> str:
    """Fetch fresh git repo context via subprocess.

    Returns a brief summary: branch, status, recent commit.
    Non-blocking: returns empty string if not in a git repo or git fails.
    """
    import subprocess
    try:
        # Check if we're in a git repo
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, timeout=3, check=True,
        )

        parts = []

        # Current branch
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, timeout=3, text=True,
        )
        if branch.returncode == 0 and branch.stdout.strip():
            parts.append(f"branch: {branch.stdout.strip()}")

        # Status summary (modified/untracked counts)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, timeout=5, text=True,
        )
        if status.returncode == 0:
            lines = [l for l in status.stdout.strip().split("\n") if l]
            if lines:
                modified = sum(1 for l in lines if l.startswith(" M") or l.startswith("M "))
                added = sum(1 for l in lines if l.startswith("A ") or l.startswith("??"))
                status_parts = []
                if modified:
                    status_parts.append(f"{modified} modified")
                if added:
                    status_parts.append(f"{added} untracked/added")
                if status_parts:
                    parts.append(f"status: {', '.join(status_parts)}")
            else:
                parts.append("status: clean")

        # Last commit (short)
        log = subprocess.run(
            ["git", "log", "-1", "--format=%h %s (%ar)"],
            capture_output=True, timeout=3, text=True,
        )
        if log.returncode == 0 and log.stdout.strip():
            parts.append(f"last commit: {log.stdout.strip()}")

        return " | ".join(parts) if parts else ""

    except Exception:
        return ""


# ── TTL caches ──
_cached_project_directives: str | None = None
_git_context_cache: tuple[str, float] = ("", 0.0)  # (context, timestamp)
_GIT_CONTEXT_TTL = 60  # seconds


def _get_git_context() -> str:
    """Get git context with 60s TTL cache (refreshes each session, not per-process)."""
    global _git_context_cache
    now = time.time()
    if _git_context_cache[0] and now - _git_context_cache[1] < _GIT_CONTEXT_TTL:
        return _git_context_cache[0]
    result = _fetch_git_context()
    _git_context_cache = (result, now)
    return result


def _get_environment_context() -> str:
    """Build dynamic environment context string (refreshed every round)."""
    import os
    import sys
    from datetime import datetime
    parts = [
        f"date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"cwd: {os.getcwd()}",
        f"platform: {sys.platform}",
    ]
    return " | ".join(parts)


# ── Persistent memory (cross-session) ──
_persistent_memory_cache: tuple[str, float] = ("", 0.0)
_PERSISTENT_MEMORY_TTL = 300  # seconds


def _get_persistent_memory() -> str:
    """Load persistent memory from ~/.mulagent/memory/MEMORY.md with 300s TTL cache."""
    global _persistent_memory_cache
    now = time.time()
    if _persistent_memory_cache[0] and now - _persistent_memory_cache[1] < _PERSISTENT_MEMORY_TTL:
        return _persistent_memory_cache[0]
    from graph.memory import PersistentMemory
    result = PersistentMemory.load_index()
    _persistent_memory_cache = (result, now)
    return result


# ── Cached static prefix for system prompt ──
_static_prefix_cache: dict[str, tuple[str, float]] = {}  # key → (content, timestamp)
_STATIC_PREFIX_TTL = 300  # seconds


def _build_system_prompt(
    tool_descriptions: str,
    memory: WorkingMemory,
    round_num: int = 0,
    reminders: list[str] | None = None,
) -> str:
    """Build system prompt with layered injection and caching.

    Cached (rebuilt only on tool_descriptions change or TTL expiry):
      - Base prompt + tool descriptions
      - Project directives, persistent memory

    Per-round (always fresh):
      - Environment context (date/time), git context
      - Working memory (facts + directives)
      - System reminders, extended instructions (round 0 only)
    """
    global _cached_project_directives
    from common.tokenizer import estimate_tokens, truncate_to_tokens

    # ── Cached static prefix: base + project_directives + persistent_memory ──
    _cache_key = f"td:{hash(tool_descriptions)}"
    _now = time.time()
    _cached = _static_prefix_cache.get(_cache_key)
    if _cached and (_now - _cached[1]) < _STATIC_PREFIX_TTL:
        static_prefix = _cached[0]
    else:
        # Load project directives (once per process)
        if _cached_project_directives is None:
            _cached_project_directives = _load_project_directives()
        persistent_mem = _get_persistent_memory()

        # Compute token budgets
        _ctx_window = 32_768
        _max_output = 4096
        try:
            from common.config import get_settings
            _model_cfg = get_settings().llm.get_model()
            if _model_cfg:
                _ctx_window = _model_cfg.get_context_window()
                _max_output = _model_cfg.max_tokens
        except Exception:
            pass
        _input_budget = _ctx_window - _max_output

        _budget_proj = max(200, int(_input_budget * 0.025))
        _budget_mem = max(150, int(_input_budget * 0.02))

        prefix_parts = []
        # Project directives
        if _cached_project_directives:
            seg = f"## Project Directives\n{_cached_project_directives}"
            if estimate_tokens(seg) > _budget_proj:
                seg = truncate_to_tokens(seg, _budget_proj)
            prefix_parts.append(seg)
        # Persistent memory
        if persistent_mem:
            seg = f"## Persistent Memory\n{persistent_mem}"
            if estimate_tokens(seg) > _budget_mem:
                seg = truncate_to_tokens(seg, _budget_mem)
            prefix_parts.append(seg)

        static_prefix = "\n\n".join(prefix_parts) if prefix_parts else ""
        _static_prefix_cache.clear()  # keep only one entry
        _static_prefix_cache[_cache_key] = (static_prefix, _now)

    # ── Per-round dynamic parts ──
    env_context = _get_environment_context()
    git_context = _get_git_context()

    base = ORCHESTRATOR_PROMPT_BASE.format(
        tool_descriptions=tool_descriptions,
        environment_context=env_context,
    )
    parts = [base]

    # Extended instructions only on first round
    if round_num == 0:
        parts.append(ORCHESTRATOR_PROMPT_EXTENDED)

    # Static prefix (cached)
    if static_prefix:
        parts.append(static_prefix)

    # Git context (TTL 60s, cheap)
    if git_context:
        parts.append(f"## Git Context\n{git_context}")

    # Working memory (always fresh)
    ctx = memory.build_context_message()
    if ctx:
        parts.append(ctx)

    # System reminders
    if reminders:
        reminder_text = "\n".join(f"<system-reminder>{r}</system-reminder>" for r in reminders)
        parts.append(f"## System Reminders\n{reminder_text}")

    return "\n\n".join(parts)


def _arg_similarity(sig_a: str, sig_b: str) -> float:
    """Compute similarity between two call signatures (0.0 to 1.0).

    Uses character-level overlap ratio. Quick and sufficient for
    detecting near-duplicate tool calls.
    """
    if sig_a == sig_b:
        return 1.0
    if not sig_a or not sig_b:
        return 0.0
    # Simple Jaccard on character bigrams
    def _bigrams(s: str) -> set[str]:
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}
    a, b = _bigrams(sig_a), _bigrams(sig_b)
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


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

    # ── Step 2: Prepare tool schemas (deferred tools loaded on demand) ──
    from tools.base import ToolDef
    tool_defs: dict[str, ToolDef] = tools
    # Only send full schemas for core (non-deferred) tools
    tool_schemas = [t.to_openai_schema() for t in tool_defs.values() if not t.deferred]
    _loaded_deferred: set[str] = set()  # tracks which deferred tools have been loaded
    _tools_dirty = False  # True when deferred tool loaded → rebind next round

    # Build tool descriptions grouped by category
    _CATEGORY_LABELS = {
        "search": "Search & Discovery (read-only)",
        "external": "External Lookup (network)",
        "file": "File Operations (local)",
        "execution": "Execution (side effects)",
        "vcs": "Version Control",
        "task": "Task Management",
        "delegation": "Delegation",
        "meta": "Meta",
        "general": "Other",
    }
    categorized: dict[str, list[str]] = {}
    for t in tool_defs.values():
        cat = getattr(t, "category", "general")
        if t.deferred:
            line = f"  - **{t.name}** [deferred]: {t.description}"
        else:
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in t.parameters.get("properties", {}).items()
            )
            line = f"  - **{t.name}**({params}): {t.description}"
        categorized.setdefault(cat, []).append(line)
    # Add skill descriptions to delegation category + collect triggers for routing hints
    _skill_triggers: list[tuple[str, str, str]] = []  # (skill_key, trigger_regex, description)
    try:
        from tools.skill_loader import load_skills
        _skills = load_skills()
        if _skills:
            for _sk, _sv in _skills.items():
                _desc = _sv.get("description", "")
                categorized.setdefault("delegation", []).append(
                    f"  - **{_sk}** (skill): {_desc}"
                )
                _trigger = _sv.get("_trigger", "")
                if _trigger:
                    _skill_triggers.append((_sk, _trigger, _desc))
    except Exception:
        pass

    # Skill trigger auto-routing: check if user input matches any skill trigger
    _skill_route_hint = ""
    if _skill_triggers:
        import re as _re
        _input_lower = user_input.lower()
        for _sk, _trigger, _desc in _skill_triggers:
            try:
                if _re.search(_trigger, _input_lower):
                    _skill_route_hint = (
                        f"The user's request matches the '{_sk}' skill ({_desc}). "
                        f"Consider using delegate(role='{_sk}') for this task."
                    )
                    logger.info("Skill trigger matched: '%s' for input", _sk)
                    break
            except _re.error:
                pass

    tool_desc_parts = []
    for cat_key in _CATEGORY_LABELS:
        if cat_key in categorized:
            tool_desc_parts.append(f"**{_CATEGORY_LABELS[cat_key]}**:")
            tool_desc_parts.extend(categorized[cat_key])
    tool_descriptions = "\n".join(tool_desc_parts)

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
    disabled_tools: dict[str, float] = {}  # tool_name → disabled_at timestamp (auto-recovery after TTL)
    _DISABLED_TTL = 300  # seconds before a disabled tool is auto-recovered

    # Read-before-edit tracking: files that have been read in this session
    _files_read: set[str] = set()  # paths read via read_file
    _files_written: set[str] = set()  # paths written via write_file/edit_file
    # Parallel call tracking: consecutive rounds with single tool call
    _single_call_rounds = 0

    # Cache: bind_tools (rebind when deferred tool is loaded)
    llm_with_tools = llm.bind_tools(tool_schemas)
    from common.retry import retry_async

    # Tool result cache: avoid re-executing identical calls
    _tool_cache: dict[str, str] = {}  # hash(tool_name + args) → result
    _idem_local: set[str] = set()  # session-local idempotency fallback (when Redis unavailable)

    # System reminders: dynamic mid-loop injection
    _pending_reminders: list[str] = []
    _todo_used = False  # track whether todo_manage has been called
    _total_tool_calls = 0  # cumulative tool calls across all rounds

    async def _run_loop() -> str:
        nonlocal _single_call_rounds, _tools_dirty, llm_with_tools, _todo_used, _total_tool_calls
        for round_num in range(max_rounds):
            # Rebind tools if deferred tool was loaded last round
            if _tools_dirty:
                tool_schemas_now = [t.to_openai_schema() for t in tool_defs.values()
                                    if not t.deferred or t.name in _loaded_deferred]
                llm_with_tools = llm.bind_tools(tool_schemas_now)
                _tools_dirty = False

            # ── System reminder: skill routing hint (round 0 only) ──
            if round_num == 0 and _skill_route_hint:
                _pending_reminders.append(_skill_route_hint)

            # ── System reminder: todo_manage nudge (only when truly complex) ──
            if not _todo_used and not is_sub_agent and (round_num >= 5 or _total_tool_calls > 8):
                _pending_reminders.append(
                    "You haven't created a task list yet. For multi-step tasks, "
                    "use todo_manage(action='create') to track progress."
                )

            # ── Directive reinforcement: re-inject every 5 rounds to combat attention decay ──
            if round_num > 0 and round_num % 5 == 0 and memory.directives:
                rules_text = " | ".join(memory.directives)
                _pending_reminders.append(
                    f"REMINDER — These rules are still in effect and MUST be followed: {rules_text}"
                )

            # Rebuild messages: system (with context + history + reminders) + user + conversation
            current_reminders = list(_pending_reminders)
            _pending_reminders.clear()
            system_prompt = _build_system_prompt(
                tool_descriptions, memory, round_num=round_num,
                reminders=current_reminders if current_reminders else None,
            )
            messages = [SystemMessage(content=system_prompt)]

            # Inject conversation history as proper Human/AI message pairs
            # (instead of stuffing into system prompt)
            if conversation_history and round_num == 0:
                messages.append(HumanMessage(content=(
                    "[Previous conversation context]\n" + conversation_history
                )))
                messages.append(AIMessage(content="Understood, I have the previous context."))

            messages.append(HumanMessage(content=user_input))

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

            # ── Think + Act (exponential backoff for rate limits) ──
            start = time.perf_counter()
            try:
                response = await retry_async(
                    llm_with_tools.ainvoke, messages,
                    max_retries=3, base_delay=2.0, max_delay=30.0,
                )
            except Exception as e:
                logger.error("LLM call failed at round %d: %s", round_num + 1, e)
                if round_num > 0:
                    if memory.facts:
                        return await _force_conclude_llm(memory, user_input, llm)
                    return "抱歉，未能获取到有效信息来完成此任务。请尝试更具体的描述或换个方式提问。"
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
                    # ── Completion gate: check if all tasks are done before concluding ──
                    _tasks = memory.state.get("_tasks", [])
                    _pending_tasks = [t for t in _tasks if t.get("status") != "done"]
                    if _pending_tasks and not is_sub_agent and round_num < max_rounds - 2:
                        # Tasks still pending — don't conclude, push back
                        _pending_names = [f"#{t['id']} {t['text']}" for t in _pending_tasks[:5]]
                        nudge_msg = (
                            f"[System] You have {len(_pending_tasks)} incomplete tasks: "
                            + "; ".join(_pending_names)
                            + ". Continue executing before giving the final answer."
                        )
                        conversation.append(response)
                        conversation.append(ToolMessage(
                            content=nudge_msg,
                            tool_call_id=f"gate_{round_num}",
                        ))
                        logger.info("Completion gate: %d tasks pending, pushing back", len(_pending_tasks))
                        continue

                    should_verify = (
                        not is_sub_agent
                        and round_num > 1           # At least 2 rounds of tool use
                        and len(memory.facts) >= 3  # Enough facts to cross-check
                    )
                    if should_verify:
                        verified = await _verify_answer(
                            answer, user_input, memory, llm,
                            files_written=_files_written,
                        )
                        if verified:
                            return verified
                    return answer
                # Empty response with no tool calls
                if round_num > 0:
                    if memory.facts:
                        return await _force_conclude_llm(memory, user_input, llm)
                    return "抱歉，未能获取到有效信息来完成此任务。请尝试更具体的描述或换个方式提问。"
                continue

            _total_tool_calls += len(tool_calls)

            # ── Parallel call detection: nudge LLM if repeatedly making single calls ──
            if len(tool_calls) == 1:
                _single_call_rounds += 1
            else:
                _single_call_rounds = 0

            if _single_call_rounds >= 3 and round_num >= 3:
                logger.info("Parallel nudge: %d consecutive single-call rounds", _single_call_rounds)
                # Inject hint as part of the response content
                _parallel_hint = (
                    "\n\n💡 [System] You've made {n} consecutive rounds with a single tool call. "
                    "If your next steps are independent, combine them in ONE round for efficiency."
                ).format(n=_single_call_rounds)
                if hasattr(response, 'content') and response.content:
                    response.content += _parallel_hint
                _single_call_rounds = 0  # reset after nudge

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
            # Check for "similar" repeats (same tool, args overlap >80%, 3x)
            is_similar_repeat = False
            if not is_exact_repeat and len(recent_calls) >= 3:
                recent_tool_names = [
                    sig.split(":")[0] for sig in recent_calls[-3:]
                ]
                if len(set(recent_tool_names)) == 1:
                    # Same tool 3x — check argument similarity
                    recent_args = [
                        set(sig.split(":", 1)[1]) if ":" in sig else set()
                        for sig in recent_calls[-3:]
                    ]
                    # Pairwise overlap: if args strings are >80% similar
                    sigs = recent_calls[-3:]
                    overlap_01 = _arg_similarity(sigs[0], sigs[1])
                    overlap_12 = _arg_similarity(sigs[1], sigs[2])
                    is_similar_repeat = overlap_01 > 0.8 and overlap_12 > 0.8

            # ── Auto-recover disabled tools after TTL ──
            _now = time.perf_counter()
            _recovered = [t for t, ts in disabled_tools.items() if _now - ts > _DISABLED_TTL]
            for t in _recovered:
                del disabled_tools[t]
                tool_fail_counts.pop(t, None)
                logger.info("Tool %s auto-recovered after %ds TTL", t, _DISABLED_TTL)
                _pending_reminders.append(f"Tool '{t}' has been re-enabled after recovery timeout.")

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

                # ── Track todo_manage usage ──
                if t_name == "todo_manage":
                    _todo_used = True

                # ── Deferred tool loading: load_tool triggers rebind ──
                if t_name == "load_tool":
                    loaded_name = t_args.get("name", "")
                    result = await t_def.fn(t_args, tools=tool_defs)
                    if loaded_name and loaded_name in tool_defs and tool_defs[loaded_name].deferred:
                        _loaded_deferred.add(loaded_name)
                        _tools_dirty = True
                        _pending_reminders.append(
                            f"Tool '{loaded_name}' is now available with full schema. "
                            f"You can call it in the next round."
                        )
                    return tc, result

                # ── Read-before-edit enforcement ──
                if t_name in ("edit_file", "write_file") and not is_sub_agent:
                    _target = t_args.get("path", "")
                    if _target and _target not in _files_read:
                        # For write_file creating new files, skip the check
                        from pathlib import Path as _P
                        if t_name == "edit_file" or _P(_target).expanduser().exists():
                            return tc, (
                                f"Error: you must read_file('{_target}') before {t_name}. "
                                "This prevents editing based on stale or assumed content."
                            )

                # ── Track read_file calls for read-before-edit ──
                if t_name == "read_file":
                    _read_path = t_args.get("path", "")
                    if _read_path:
                        _files_read.add(_read_path)

                # Check cache (skip for stateful tools like execute_shell, code_run, write_file)
                cacheable = t_name not in ("execute_shell", "code_run", "write_file", "delegate")
                cache_key = f"{t_name}:{json.dumps(t_args, sort_keys=True)}" if cacheable else ""
                if cache_key and cache_key in _tool_cache:
                    logger.info("[cache hit] %s(%s)", t_name, _brief_args(t_args))
                    return tc, _tool_cache[cache_key] + "\n(cached result)"

                # ── Pre-tool hook: directive enforcement ──
                blocked = pre_tool_hook(t_name, t_args, list(memory.directives))
                if blocked:
                    # Confirmation flow: dangerous ops need user approval
                    if "[CONFIRM_REQUIRED]" in blocked and on_progress:
                        try:
                            confirm_detail = json.dumps({
                                "tool": t_name,
                                "args": _brief_args(t_args, max_val=500),
                                "reason": blocked.replace("[CONFIRM_REQUIRED] ", ""),
                            }, ensure_ascii=False)
                            approved = await on_progress(round_num + 1, "confirm_required", confirm_detail)
                            if approved is True:
                                logger.info("User approved dangerous op: %s", t_name)
                            else:
                                logger.info("User rejected dangerous op: %s", t_name)
                                return tc, f"[SKIPPED] 用户拒绝执行: {blocked.replace('[CONFIRM_REQUIRED] ', '')}"
                        except Exception:
                            return tc, f"[SKIPPED] 确认超时，跳过: {t_name}"
                    else:
                        logger.info("Tool %s blocked by pre-hook: %s", t_name, blocked[:100])
                        return tc, blocked

                # ── Idempotency key for write operations ──
                _idem_key = ""
                if t_name in ("write_file", "edit_file", "git_ops", "execute_shell") and task_id:
                    import hashlib as _hl
                    _idem_key = f"idem:{task_id}:{t_name}:{_hl.md5(json.dumps(t_args, sort_keys=True).encode()).hexdigest()[:12]}"
                    _is_dup = False
                    try:
                        from common.redis_client import is_duplicate
                        _is_dup = await is_duplicate(_idem_key, ttl=3600)
                    except Exception:
                        # Redis unavailable — fallback to session-local set
                        _is_dup = _idem_key in _idem_local
                    if _is_dup:
                        logger.info("Idempotent skip: %s(%s)", t_name, _brief_args(t_args))
                        return tc, f"[idempotent] Operation already completed in this task"
                    _idem_local.add(_idem_key)

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

                # ── Post-write validation: check file against .mulagent.md rules ──
                if t_name in ("write_file", "edit_file") and not _is_error_result(raw):
                    _written_path = t_args.get("path", "")
                    if _written_path:
                        _files_written.add(_written_path)
                        _vrules = _match_validation_rules(_written_path)
                        if _vrules:
                            _verr = await _run_post_write_validation(
                                _written_path, _vrules, tool_defs, llm,
                            )
                            if _verr:
                                raw += f"\n\n{_verr}"
                                logger.info("Post-write validation failed for %s", _written_path)

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

                # ── TodoList progress: push task list to caller ──
                if tool_name == "todo_manage" and on_progress:
                    try:
                        import json as _json
                        _tasks = memory.state.get("_tasks", [])
                        if _tasks:
                            await on_progress(round_num + 1, "todo_update", _json.dumps(_tasks, ensure_ascii=False))
                    except Exception:
                        pass

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
                        disabled_tools[tool_name] = time.perf_counter()
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

            # ── Numerical accuracy check: detect percentage writes without code_run ──
            _round_tools = {tc["name"] for tc in tool_calls}
            _wrote_file = _round_tools & {"write_file", "edit_file"}
            _used_code = "code_run" in _round_tools
            if _wrote_file and not _used_code:
                # Check if any write args contain percentage patterns
                import re as _re
                for tc in tool_calls:
                    if tc["name"] in ("write_file", "edit_file"):
                        _content = str(tc.get("args", {}).get("content", "")) + str(tc.get("args", {}).get("new_text", ""))
                        if _re.search(r'\d+\.?\d*\s*%', _content):
                            _pending_reminders.append(
                                "WARNING: You wrote a file containing percentage values without using code_run to calculate them. "
                                "LLMs are unreliable at math. If these are computed values (not literal text), "
                                "use code_run(python) to calculate them first, then write the correct values."
                            )
                            break

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
            try:
                from common.config import get_settings as _gs
                _cmp = _gs().react.compress
                _compact_trigger = _cmp.facts_compact_trigger
                _keep_recent = _cmp.facts_keep_recent
            except Exception:
                _compact_trigger, _keep_recent = 15, 5
            if len(memory.facts) > _compact_trigger:
                # Try LLM-based summarization (falls back to simple compaction)
                try:
                    await memory.compact_facts_llm(llm, keep_recent=_keep_recent)
                except Exception:
                    memory.compact_facts(keep_recent=_keep_recent)

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
            result_meta["disabled_tools"] = list(disabled_tools.keys())
            result_meta["_tasks"] = memory.state.get("_tasks", [])

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
    files_written: set[str] | None = None,
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

    # Completeness info
    completeness_str = ""
    _tasks = memory.state.get("_tasks", [])
    if _tasks:
        done = sum(1 for t in _tasks if t.get("status") == "done")
        total = len(_tasks)
        completeness_str += f"\n任务完成度: {done}/{total}"
        if done < total:
            pending = [t["text"] for t in _tasks if t.get("status") != "done"]
            completeness_str += f" (未完成: {', '.join(pending[:5])})"
    if files_written:
        completeness_str += f"\n已写入文件: {', '.join(sorted(files_written))}"

    messages = [
        SystemMessage(content=(
            "你是一个答案验证器。检查以下回答是否准确、完整，且符合用户约束。\n\n"
            "检查项：\n"
            "1. 回答是否基于实际收集到的信息（不是编造的）？\n"
            "2. 是否完整回答了用户的问题？\n"
            "3. 是否遵守了所有用户约束？\n"
            "4. 任务列表中的所有任务是否都已完成？\n"
            "5. 是否包含敏感信息需要脱敏？\n\n"
            "如果回答没有问题，返回 PASS\n"
            "如果需要修正，返回修正后的完整回答（不要说'修正后的回答是'，直接给出内容）"
        )),
        HumanMessage(content=(
            f"用户问题: {user_input[:300]}{directives_str}{completeness_str}\n\n"
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


def _brief_args(args: dict, max_val: int = 50) -> str:
    """Abbreviate tool args for logging (default 50 chars per value)."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > max_val:
            s = s[:max_val] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)
