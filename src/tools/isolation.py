"""Isolation tools: delegate work to sub-agents with independent context.

- delegate: Spawn a sub-agent to handle a subtask, returns compressed result.
  Supports specialized roles from config/agents.yaml and auto-loaded skills.
  Supports background=true for non-blocking async delegation.
- check_background: Check results of background delegations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

# Cache for loaded role configs and knowledge bases
_role_configs: dict[str, dict] | None = None
_knowledge_cache: dict[str, str] = {}
from common.config import CONFIG_DIR as _CONFIG_DIR


def _load_roles() -> dict[str, dict]:
    """Load role configurations from agents.yaml + auto-loaded skills (cached)."""
    global _role_configs
    if _role_configs is not None:
        return _role_configs

    # 1. Load from agents.yaml
    yaml_roles: dict[str, dict] = {}
    try:
        import yaml
        config_path = _CONFIG_DIR / "agents.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        yaml_roles = data.get("roles", {})
        logger.info("Loaded %d role configs from agents.yaml", len(yaml_roles))
    except Exception as e:
        logger.warning("Failed to load role configs: %s", e)

    # 2. Merge auto-loaded skills (skills don't override yaml roles)
    try:
        from tools.skill_loader import load_skills
        skills = load_skills()
        for key, cfg in skills.items():
            if key not in yaml_roles:
                yaml_roles[key] = cfg
                logger.info("Skill '%s' registered as delegate role", key)
    except Exception as e:
        logger.warning("Skill loading failed: %s", e)

    _role_configs = yaml_roles
    return _role_configs


def _load_knowledge(names: list[str]) -> str:
    """Load and concatenate knowledge base files for a role.

    Knowledge files are Markdown files in config/knowledge/.
    Results are cached in memory after first load.

    Args:
        names: List of knowledge base names (without .md extension).

    Returns:
        Concatenated knowledge content, or empty string if none found.
    """
    if not names:
        return ""

    parts = []
    for name in names:
        if name in _knowledge_cache:
            parts.append(_knowledge_cache[name])
            continue

        kb_path = _CONFIG_DIR / "knowledge" / f"{name}.md"
        try:
            content = kb_path.read_text(encoding="utf-8")
            # Truncate very large knowledge files to ~3000 tokens (~7500 chars)
            if len(content) > 7500:
                content = content[:7500] + "\n... (knowledge base truncated)"
            _knowledge_cache[name] = content
            parts.append(content)
            logger.debug("Loaded knowledge base: %s (%d chars)", name, len(content))
        except FileNotFoundError:
            logger.debug("Knowledge base not found: %s", kb_path)
        except Exception as e:
            logger.warning("Failed to load knowledge %s: %s", name, e)

    return "\n\n---\n\n".join(parts) if parts else ""


# ── Smart Knowledge Selection ──

# Language detection patterns: keyword → knowledge base name
_LANG_SIGNALS: dict[str, list[str]] = {
    "python": [
        "python", ".py", "pip", "django", "flask", "fastapi", "pytest",
        "pandas", "numpy", "torch", "pep8", "virtualenv", "poetry",
    ],
    "typescript": [
        "typescript", ".ts", ".tsx", "react", "next.js", "nextjs",
        "angular", "vue", "node", "npm", "yarn", "deno", "bun",
    ],
    "go": [
        "golang", ".go", "go mod", "goroutine", "gin", "echo",
        "go build", "go run", "go test",
    ],
    "java": [
        "java", ".java", "spring", "maven", "gradle", "jpa",
        "hibernate", "springboot", "spring boot", "tomcat",
    ],
    "rust": [
        "rust", ".rs", "cargo", "crate", "tokio", "async-std",
        "borrow checker", "lifetime",
    ],
    "cpp": [
        "c++", "cpp", ".cpp", ".hpp", ".h", "cmake", "makefile",
        "raii", "stl", "boost",
    ],
    "kotlin": [
        "kotlin", ".kt", "jetpack", "compose", "android", "ktor",
        "coroutine",
    ],
    "flutter": [
        "flutter", "dart", ".dart", "widget", "bloc", "riverpod",
        "pubspec",
    ],
    "pytorch": [
        "pytorch", "torch", "cuda", "tensor", "nn.module",
        "dataloader", "autograd", "huggingface", "transformers",
    ],
}

# Non-language knowledge bases and their signals
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "refactor": ["refactor", "dead code", "cleanup", "重构", "清理"],
    "database": ["sql", "database", "index", "query", "rls", "schema", "数据库"],
    "security": ["security", "owasp", "xss", "injection", "安全", "漏洞"],
    "tdd": ["test", "tdd", "coverage", "测试", "覆盖率"],
    "e2e": ["e2e", "end-to-end", "playwright", "selenium", "端到端"],
    "architect": ["architecture", "design", "scalab", "架构", "设计"],
    "code_review": ["review", "审查", "代码审查"],
    "build_errors": ["build", "compile", "构建", "编译"],
}


def _select_knowledge(all_names: list[str], task: str, max_kb: int = 4) -> list[str]:
    """Smart knowledge selection: pick only relevant knowledge bases for a task.

    Instead of injecting ALL knowledge bases for a role (which can be 10+ files,
    ~75K chars), detect the language/domain from the task and select only the
    relevant ones. This saves ~50K+ tokens per delegate call.

    Args:
        all_names: Full list of knowledge base names from role config.
        task: The task description to analyze.
        max_kb: Maximum number of knowledge bases to inject.

    Returns:
        Filtered list of knowledge base names, ordered by relevance.
    """
    if len(all_names) <= max_kb:
        return all_names

    task_lower = task.lower()
    scores: dict[str, float] = {}

    # Score language knowledge bases
    for kb_name, signals in _LANG_SIGNALS.items():
        if kb_name not in all_names:
            continue
        score = sum(1.0 for s in signals if s in task_lower)
        if score > 0:
            scores[kb_name] = score

    # Score domain knowledge bases
    for kb_name, signals in _DOMAIN_SIGNALS.items():
        if kb_name not in all_names:
            continue
        score = sum(1.0 for s in signals if s in task_lower)
        if score > 0:
            scores[kb_name] = score

    if not scores:
        # No signals detected — return first max_kb as fallback
        # Prefer domain KBs over language KBs (more generally useful)
        domain_kbs = [n for n in all_names if n in _DOMAIN_SIGNALS]
        lang_kbs = [n for n in all_names if n in _LANG_SIGNALS]
        return (domain_kbs + lang_kbs)[:max_kb]

    # Sort by score descending, take top max_kb
    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    selected = ranked[:max_kb]

    # Always include domain-relevant KBs even if they scored lower
    for kb_name in all_names:
        if kb_name in _DOMAIN_SIGNALS and kb_name in scores and kb_name not in selected:
            if len(selected) < max_kb:
                selected.append(kb_name)

    logger.info("Smart KB selection: %s → %s (from %d candidates)",
                list(scores.keys()), selected, len(all_names))
    return selected


async def _dynamic_knowledge_inject(
    task: str,
    knowledge_names: list[str],
    qdrant: Any,
    context_limit_chars: int = 60000,
    max_budget_ratio: float = 0.7,
) -> str:
    """Dynamic knowledge injection with context-budget awareness.

    Instead of injecting entire KB files, retrieves only the most relevant
    chunks via embedding search. Respects context budget:
    - Retrieves up to top_k chunks
    - Injects as many as fit within 70% of context limit
    - Logs trimming decisions for debugging

    Args:
        task: The task description to search for.
        knowledge_names: KB source files to search within.
        qdrant: Qdrant client.
        context_limit_chars: Approximate context limit in chars.
        max_budget_ratio: Max fraction of context to use (default 0.7).

    Returns:
        Formatted knowledge context string.
    """
    from tools.knowledge_rag import retrieve_knowledge, format_knowledge_context

    max_chars = int(context_limit_chars * max_budget_ratio)

    # Retrieve relevant chunks from the knowledge bases
    chunks = await retrieve_knowledge(
        query=task[:500],
        qdrant=qdrant,
        top_k=15,
        source_filter=knowledge_names,
        score_threshold=0.25,
    )

    if not chunks:
        logger.info("RAG retrieval returned 0 chunks for task, falling back to file-based")
        return _load_knowledge(knowledge_names)

    # Format with budget constraint
    text, used_chunks = format_knowledge_context(chunks, max_chars=max_chars)

    trimmed_count = len(chunks) - len(used_chunks)
    if trimmed_count > 0:
        logger.info(
            "Dynamic knowledge injection: retrieved %d chunks, injected %d, "
            "trimmed %d (budget: %d/%d chars, %.0f%% of limit)",
            len(chunks), len(used_chunks), trimmed_count,
            len(text), context_limit_chars, len(text) / context_limit_chars * 100,
        )
    else:
        logger.info(
            "Dynamic knowledge injection: all %d chunks injected (%d chars, %.0f%% of limit)",
            len(chunks), len(text), len(text) / context_limit_chars * 100,
        )

    if not text:
        return _load_knowledge(knowledge_names)

    return text


def reload_roles():
    """Force reload of role configs, knowledge bases, and skills. Used by hot-reload."""
    global _role_configs
    _role_configs = None
    _knowledge_cache.clear()
    try:
        from tools.skill_loader import reload_skills
        reload_skills()
    except Exception:
        pass
    logger.info("Role configs, knowledge cache, and skills cleared")


MAX_DELEGATE_DEPTH = 3  # Maximum nesting level for delegate calls


async def _delegate(params: dict[str, Any], **deps: Any) -> str:
    """Delegate a subtask to a sub-agent with its own context window.

    The sub-agent runs a shorter ReAct loop (max 5 rounds) and returns
    a compressed summary. This prevents the main orchestrator's context
    from being polluted with detailed intermediate steps.

    Parent directives are inherited — user constraints like "删除前要经过我同意"
    are NEVER lost across delegation boundaries.

    Depth control: sub-agents can re-delegate up to MAX_DELEGATE_DEPTH levels.
    At max depth, the delegate tool is excluded to prevent runaway recursion.
    """
    task = params.get("task", "")
    if not task:
        return "Error: task is required"

    context_hint = params.get("context", "")
    role = params.get("role", "")
    background = params.get("background", False)
    isolation = params.get("isolation", "")

    # Import here to avoid circular dependency
    from graph.react_orchestrator import react_loop

    llm = deps.get("llm")
    tools = deps.get("tools")
    parent_directives = deps.get("parent_directives", [])
    current_depth = deps.get("delegate_depth", 0)

    if llm is None or tools is None:
        return "Error: delegate requires llm and tools dependencies"

    # Depth control: exclude delegate tool at max depth
    next_depth = current_depth + 1
    if next_depth >= MAX_DELEGATE_DEPTH:
        sub_tools = {name: tool for name, tool in tools.items() if name != "delegate"}
        logger.info("Delegate depth %d reached max (%d), sub-agent cannot re-delegate",
                    next_depth, MAX_DELEGATE_DEPTH)
    else:
        sub_tools = dict(tools)
        logger.info("Delegate depth %d/%d, sub-agent can re-delegate", next_depth, MAX_DELEGATE_DEPTH)

    # ── Apply role-specific configuration ──
    role_prompt = ""
    knowledge_content = ""
    if role:
        roles = _load_roles()
        role_cfg = roles.get(role)
        if role_cfg:
            role_prompt = role_cfg.get("prompt", "")
            allowed_tools = role_cfg.get("tools", [])
            knowledge_names = role_cfg.get("knowledge", [])

            # Handle knowledge: auto — dynamically select from ALL available KBs
            if knowledge_names == "auto":
                all_kb_names = list(_LANG_SIGNALS.keys()) + list(_DOMAIN_SIGNALS.keys())
                knowledge_names = _select_knowledge(all_kb_names, task)
                logger.info("Auto-knowledge for role '%s': selected %s", role, knowledge_names)
            elif knowledge_names:
                knowledge_names = _select_knowledge(knowledge_names, task)
                # Dynamic knowledge injection: use RAG if available, else file-based
                qdrant = deps.get("qdrant")
                if qdrant is not None:
                    try:
                        knowledge_content = await _dynamic_knowledge_inject(
                            task, knowledge_names, qdrant,
                        )
                    except Exception as _rag_err:
                        logger.debug("RAG knowledge failed, using file-based: %s", _rag_err)
                        knowledge_content = _load_knowledge(knowledge_names)
                else:
                    knowledge_content = _load_knowledge(knowledge_names)

            if allowed_tools:
                # Restrict sub-agent tools to role's allowed set
                sub_tools = {
                    name: tool for name, tool in sub_tools.items()
                    if name in allowed_tools
                }
                logger.info("Role '%s' restricts tools to: %s", role, list(sub_tools.keys()))

            if knowledge_names:
                logger.info("Role '%s' loaded knowledge: %s", role, knowledge_names)
        else:
            logger.debug("Unknown role '%s', using default tools", role)

    # Build sub-agent input with role context + knowledge
    parts = []
    if role_prompt:
        parts.append(role_prompt.strip())
    if knowledge_content:
        parts.append(f"## Reference Knowledge\n\n{knowledge_content}")
    if context_hint:
        parts.append(f"Context: {context_hint}")
    parts.append(f"任务: {task}")
    sub_input = "\n\n".join(parts)

    # ── Pre-load relevant experience for sub-agent context ──
    experience_hint = ""
    qdrant = deps.get("qdrant")
    collection_name = deps.get("collection_name", "case_library")
    if qdrant is not None:
        try:
            from common.vector import text_to_embedding_async
            from evolution.experience import search_similar_experiences
            query_vec = await text_to_embedding_async(task[:200])
            experiences = await search_similar_experiences(
                qdrant, collection_name, query_vec, top_k=2,
            )
            relevant = [e for e in experiences if e.get("score", 0) > 0.5 and not e.get("is_negative")]
            if relevant:
                exp_lines = []
                for e in relevant[:2]:
                    exp_lines.append(
                        f"- Strategy: {e.get('recommended_strategy', '')}\n"
                        f"  Tips: {e.get('tips', '')}"
                    )
                    if e.get("failure_patterns"):
                        exp_lines.append(f"  Avoid: {', '.join(e['failure_patterns'][:3])}")
                experience_hint = "\n\n[Past experience for this type of task:]\n" + "\n".join(exp_lines)
        except Exception:
            pass  # Experience retrieval is optional

    if experience_hint:
        sub_input += experience_hint

    # ── Worktree isolation: create temporary git worktree ──
    worktree_path: str | None = None
    worktree_branch: str | None = None
    if isolation == "worktree":
        worktree_path, worktree_branch = await _create_worktree()
        if worktree_path:
            logger.info("Sub-agent worktree created: %s (branch %s)", worktree_path, worktree_branch)

    # ── Dynamic timeout/rounds based on task complexity ──
    _BASE_TIMEOUT = 90
    _BASE_ROUNDS = 5
    try:
        from common.config import get_settings as _gs
        _react_cfg = _gs().react
        _main_timeout = _react_cfg.timeout
        _tool_timeout = _react_cfg.tool_timeout
    except Exception:
        _main_timeout = 600
        _tool_timeout = 120

    _task_lower = task.lower()
    _is_batch = any(kw in _task_lower for kw in [
        "批量", "全部", "所有", "每一", "逐个", "逐章", "1~", "1-", "第1到",
        "校验", "validate", "batch", "all chapters", "each",
    ])
    if _is_batch or len(task) > 500:
        sub_timeout = min(_BASE_TIMEOUT * 6, _main_timeout)
        sub_rounds = min(_BASE_ROUNDS * 3, 20)
        logger.info("Delegate dynamic scaling: batch/complex task detected, "
                     "timeout=%ds, max_rounds=%d", sub_timeout, sub_rounds)
    else:
        sub_timeout = _BASE_TIMEOUT
        sub_rounds = _BASE_ROUNDS

    async def _run_sub_agent() -> str:
        """Run the sub-agent and return compressed result."""
        meta_inner: dict[str, Any] = {}
        sub_deps_inner = {**deps, "delegate_depth": next_depth}
        if worktree_path:
            sub_deps_inner["worktree_path"] = worktree_path
        sub_result = await react_loop(
            user_input=sub_input,
            tools=sub_tools,
            llm=llm,
            deps=sub_deps_inner,
            max_rounds=sub_rounds,
            timeout=sub_timeout,
            is_sub_agent=True,
            parent_directives=parent_directives,
            result_meta=meta_inner,
        )

        strategies = meta_inner.get("strategies_tried", [])
        failed = [s for s in strategies if s.get("outcome") == "fail"]

        from common.tokenizer import truncate_to_tokens
        sub_result = truncate_to_tokens(sub_result, 1500)

        if failed:
            fail_summary = "; ".join(f"{s['tool']}({s['args_summary']})" for s in failed[:3])
            sub_result += f"\n\n[Sub-agent note: these approaches failed: {fail_summary}]"

        return sub_result

    # ── Background mode: start async and return immediately ──
    if background:
        memory = deps.get("memory")
        bg_id = f"bg_{int(time.time() * 1000) % 100000}"

        async def _bg_wrapper():
            try:
                return await _run_sub_agent()
            except Exception as e:
                return f"Background subtask failed: {e}"

        future = asyncio.ensure_future(_bg_wrapper())

        # Store in memory state so check_background can find it
        if memory is not None:
            bg_tasks = memory.state.get("_bg_tasks", {})
            bg_tasks[bg_id] = {
                "task": task[:100],
                "role": role,
                "future": future,
                "started_at": time.time(),
            }
            memory.update_state("_bg_tasks", bg_tasks)

        return (
            f"Background task started: {bg_id}\n"
            f"Task: {task[:100]}\n"
            f"Use check_background(task_id='{bg_id}') to get the result when ready."
        )

    # ── Foreground mode: wait for result ──
    try:
        result = await _run_sub_agent()
        if worktree_path:
            wt_info = await _cleanup_worktree(worktree_path, worktree_branch)
            if wt_info:
                result += f"\n\n[Worktree: {wt_info}]"
        return result
    except Exception as e:
        logger.warning("delegate failed: %s", e)
        if worktree_path:
            await _cleanup_worktree(worktree_path, worktree_branch)
        return f"Subtask failed: {e}"


# ── Git worktree helpers ────────────────────────────────────────

async def _create_worktree() -> tuple[str | None, str | None]:
    """Create a temporary git worktree for isolated sub-agent work.

    Returns (worktree_path, branch_name) or (None, None) on failure.
    """
    import tempfile

    branch = f"mulagent-wt-{int(time.time() * 1000) % 100000}"
    wt_dir = Path(tempfile.mkdtemp(prefix="mulagent_wt_"))

    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "add", "-b", branch, str(wt_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode == 0:
            return str(wt_dir), branch
        logger.warning("git worktree add failed: %s", stderr.decode()[:200])
    except Exception as e:
        logger.warning("Worktree creation failed: %s", e)

    # Cleanup on failure
    import shutil
    shutil.rmtree(wt_dir, ignore_errors=True)
    return None, None


async def _cleanup_worktree(wt_path: str, branch: str | None) -> str | None:
    """Remove a worktree. If it has changes, report them instead of deleting.

    Returns info string about what happened, or None.
    """
    import shutil

    try:
        # Check for changes
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", wt_path, "status", "--porcelain",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        has_changes = bool(stdout.decode().strip())

        if has_changes:
            return f"changes in worktree {wt_path} (branch {branch}), kept for review"

        # No changes — remove worktree and branch
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "remove", "--force", wt_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)

        if branch:
            proc = await asyncio.create_subprocess_exec(
                "git", "branch", "-D", branch,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5)

        return None  # Clean removal, no message needed

    except Exception as e:
        logger.debug("Worktree cleanup failed: %s", e)
        shutil.rmtree(wt_path, ignore_errors=True)
        return None


def _build_delegate_tool() -> ToolDef:
    """Build the delegate ToolDef with dynamically generated role enum/description.

    The role list is generated from agents.yaml + auto-loaded skills,
    so adding a skill never requires editing this file.
    """
    roles = _load_roles()

    # Generate description
    try:
        from tools.skill_loader import get_delegate_description, get_all_role_names
        yaml_roles = {k: v for k, v in roles.items() if "_skill_dir" not in v}
        description = get_delegate_description(yaml_roles)
        role_enum = get_all_role_names(yaml_roles)
    except Exception:
        # Fallback: use all loaded role keys
        description = (
            "Delegate a complex subtask to a specialized sub-agent. "
            f"Available roles: {', '.join(sorted(roles.keys()))}."
        )
        role_enum = sorted(roles.keys())

    return ToolDef(
        name="delegate",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear description of the subtask to delegate",
                },
                "role": {
                    "type": "string",
                    "description": "Specialist role for the sub-agent. Choose based on task type.",
                    "enum": role_enum,
                },
                "context": {
                    "type": "string",
                    "description": "Brief context/background for the sub-agent (optional)",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background (non-blocking). Use check_background to get result later. Default: false.",
                },
                "isolation": {
                    "type": "string",
                    "description": "Isolation mode. 'worktree' creates a temporary git worktree so the sub-agent works on an isolated copy.",
                    "enum": ["worktree"],
                },
            },
            "required": ["task"],
        },
        fn=_delegate,
        category="delegation",
    )


DELEGATE = _build_delegate_tool()


# ── Background task checker ────────────────────────────────────

async def _check_background(params: dict[str, Any], **deps: Any) -> str:
    """Check the status/result of a background delegation."""
    task_id = params.get("task_id", "")
    memory = deps.get("memory")

    if memory is None:
        return "Error: no memory context"

    bg_tasks = memory.state.get("_bg_tasks", {})

    if not task_id:
        # List all background tasks
        if not bg_tasks:
            return "No background tasks running."
        lines = ["**Background tasks:**"]
        for bid, info in bg_tasks.items():
            future = info.get("future")
            status = "done" if (future and future.done()) else "running"
            elapsed = time.time() - info.get("started_at", 0)
            lines.append(f"  [{bid}] {info.get('task', '?')[:60]} — {status} ({elapsed:.0f}s)")
        return "\n".join(lines)

    info = bg_tasks.get(task_id)
    if not info:
        return f"Error: background task '{task_id}' not found"

    future = info.get("future")
    if future is None:
        return f"Error: no future for task '{task_id}'"

    if not future.done():
        elapsed = time.time() - info.get("started_at", 0)
        return f"Task '{task_id}' still running ({elapsed:.0f}s elapsed). Check again later."

    # Get result and clean up
    try:
        result = future.result()
    except Exception as e:
        result = f"Background task failed: {e}"

    del bg_tasks[task_id]
    memory.update_state("_bg_tasks", bg_tasks)

    return f"[Background result for '{task_id}']\n{result}"


CHECK_BACKGROUND = ToolDef(
    name="check_background",
    description=(
        "Check the status or result of background delegate tasks. "
        "Call with no task_id to list all running tasks, or with task_id to get a specific result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Background task ID (from delegate with background=true). Omit to list all.",
            },
        },
        "required": [],
    },
    fn=_check_background,
    category="delegation",
)
