"""Isolation tools: delegate work to sub-agents with independent context.

- delegate: Spawn a sub-agent to handle a subtask, returns compressed result.
  Supports specialized roles from config/agents.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

# Cache for loaded role configs and knowledge bases
_role_configs: dict[str, dict] | None = None
_knowledge_cache: dict[str, str] = {}
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _load_roles() -> dict[str, dict]:
    """Load role configurations from config/agents.yaml (cached)."""
    global _role_configs
    if _role_configs is not None:
        return _role_configs

    try:
        import yaml
        config_path = _CONFIG_DIR / "agents.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        _role_configs = data.get("roles", {})
        logger.info("Loaded %d role configs from agents.yaml", len(_role_configs))
    except Exception as e:
        logger.warning("Failed to load role configs: %s", e)
        _role_configs = {}

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
    "novelist": [
        "小说", "novel", "章节", "chapter", "写作", "创作", "story",
        "悬念", "角色", "character", "剧情", "plot", "连载",
        "言情", "玄幻", "悬疑", "科幻", "武侠", "仙侠",
    ],
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
    """Force reload of role configs and knowledge bases. Used by hot-reload."""
    global _role_configs
    _role_configs = None
    _knowledge_cache.clear()
    logger.info("Role configs and knowledge cache cleared")


async def _delegate(params: dict[str, Any], **deps: Any) -> str:
    """Delegate a subtask to a sub-agent with its own context window.

    The sub-agent runs a shorter ReAct loop (max 5 rounds) and returns
    a compressed summary. This prevents the main orchestrator's context
    from being polluted with detailed intermediate steps.

    Parent directives are inherited — user constraints like "删除前要经过我同意"
    are NEVER lost across delegation boundaries.

    If a `role` is specified, the sub-agent uses the specialized system prompt
    and tool subset defined in config/agents.yaml.
    """
    task = params.get("task", "")
    if not task:
        return "Error: task is required"

    context_hint = params.get("context", "")
    role = params.get("role", "")

    # Import here to avoid circular dependency
    from graph.react_orchestrator import react_loop

    llm = deps.get("llm")
    tools = deps.get("tools")
    parent_directives = deps.get("parent_directives", [])

    if llm is None or tools is None:
        return "Error: delegate requires llm and tools dependencies"

    # Exclude delegate from sub-agent tools to prevent infinite recursion
    sub_tools = {name: tool for name, tool in tools.items() if name != "delegate"}

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

            # Smart knowledge selection: pick only relevant KBs
            if knowledge_names:
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

    try:
        meta: dict[str, Any] = {}
        result = await react_loop(
            user_input=sub_input,
            tools=sub_tools,
            llm=llm,
            deps=deps,
            max_rounds=5,        # Shorter loop for sub-agents
            timeout=90,          # Shorter timeout
            is_sub_agent=True,   # Skip directive extraction for sub-agents
            parent_directives=parent_directives,  # Inherit user constraints
            result_meta=meta,
        )

        # ── Structured result with metadata for parent ──
        strategies = meta.get("strategies_tried", [])
        failed = [s for s in strategies if s.get("outcome") == "fail"]

        # Compress result
        from common.tokenizer import truncate_to_tokens
        result = truncate_to_tokens(result, 1500)

        # Append sub-agent metadata for parent's awareness
        if failed:
            fail_summary = "; ".join(f"{s['tool']}({s['args_summary']})" for s in failed[:3])
            result += f"\n\n[Sub-agent note: these approaches failed: {fail_summary}]"

        return result
    except Exception as e:
        logger.warning("delegate failed: %s", e)
        return f"Subtask failed: {e}"


DELEGATE = ToolDef(
    name="delegate",
    description=(
        "Delegate a complex subtask to a specialized sub-agent with its own independent context. "
        "Use this when a task requires deep research (>3 searches), lengthy code generation, "
        "or multi-step operations. Specify a role for specialized behavior: "
        "'planner' (task decomposition), 'architect' (system design), "
        "'researcher' (multi-source research), 'analyst' (data analysis), "
        "'coder' (code gen/debug), 'code_reviewer' (code review), "
        "'build_resolver' (fix build errors), 'tdd_guide' (test-driven dev), "
        "'security_auditor' (security audit), 'novelist' (Chinese novel writing), "
        "'writer' (content creation), 'executor' (shell/file ops), 'guardian' (quality review)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of the subtask to delegate",
            },
            "role": {
                "type": "string",
                "description": (
                    "Specialist role for the sub-agent. Choose based on task type."
                ),
                "enum": [
                    "planner", "architect",
                    "researcher", "analyst",
                    "coder", "code_reviewer", "build_resolver", "tdd_guide",
                    "security_auditor",
                    "novelist", "writer", "executor", "guardian",
                ],
            },
            "context": {
                "type": "string",
                "description": "Brief context/background for the sub-agent (optional)",
            },
        },
        "required": ["task"],
    },
    fn=_delegate,
)
