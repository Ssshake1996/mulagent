"""Skill auto-loader: scan skill directories and register as delegate roles.

A skill is a directory containing:
  - SKILL.md: frontmatter (name, description, metadata.trigger) + prompt body
  - references/ (optional): knowledge files read by sub-agent via read_file
  - scripts/ (optional): utility scripts executed via execute_shell

Skills are auto-discovered from:
  1. config/skills/ — local project skills (committed to git)
  2. Additional paths from SKILL_DIRS env var or config (colon-separated)

The loader parses SKILL.md, resolves relative paths (references/, scripts/)
to absolute paths in the prompt, and returns role configs compatible with
the delegate tool's existing role system.

Usage:
    from tools.skill_loader import load_skills
    skills = load_skills()  # returns dict[str, dict] like agents.yaml roles
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOCAL_SKILLS_DIR = _PROJECT_ROOT / "config" / "skills"

# Default tools given to skill-based roles
_DEFAULT_SKILL_TOOLS = [
    "read_file", "write_file", "edit_file", "list_dir",
    "execute_shell", "knowledge_recall", "web_search",
]

# Cache
_skill_cache: dict[str, dict] | None = None


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md.

    Returns (frontmatter_dict, body_text).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 4:].strip()

    # Simple YAML-like parsing (avoid heavy yaml dependency at module level)
    try:
        import yaml
        fm = yaml.safe_load(fm_text)
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}

    return fm, body


def _resolve_paths(body: str, skill_dir: Path) -> str:
    """Replace relative references/ and scripts/ paths with absolute paths.

    Handles patterns like:
      - references/xxx.md → /absolute/path/references/xxx.md
      - scripts/xxx.py → /absolute/path/scripts/xxx.py
      - `references/xxx.md` (in backticks)
      - [text](references/xxx.md) (markdown links)
    """
    skill_abs = str(skill_dir.resolve())

    # Replace references/ and scripts/ paths
    # Match: references/anything.ext or scripts/anything.ext
    # In various contexts: backticks, markdown links, plain text
    def replacer(match: re.Match) -> str:
        prefix = match.group(1) or ""
        rel_path = match.group(2)
        abs_path = f"{skill_abs}/{rel_path}"
        return f"{prefix}{abs_path}"

    # Pattern: optional prefix char + (references/... or scripts/...)
    pattern = r'([`(\[]\s*|(?:读取|参考|运行|使用)\s+[`]?)?((?:references|scripts)/[^\s`)\]]+)'
    body = re.sub(pattern, replacer, body)

    return body


def _load_single_skill(skill_dir: Path) -> dict[str, Any] | None:
    """Load a single skill from its directory.

    Returns a role config dict compatible with agents.yaml format, or None.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read %s: %s", skill_md, e)
        return None

    fm, body = _parse_frontmatter(text)
    if not fm.get("name"):
        logger.warning("Skill %s has no name in frontmatter, skipping", skill_dir.name)
        return None

    # Resolve relative paths to absolute
    body = _resolve_paths(body, skill_dir)

    # Extract metadata
    name = fm["name"]
    description = fm.get("description", "").strip()
    metadata = fm.get("metadata", {}) or {}
    trigger = metadata.get("trigger", "")

    # Build tools list from frontmatter or use defaults
    tools = fm.get("tools", _DEFAULT_SKILL_TOOLS)

    # Build role config (same shape as agents.yaml roles)
    role_config = {
        "name": name,
        "description": description,
        "tools": tools,
        "knowledge": [],  # Skills use read_file for references instead of KB injection
        "prompt": body,
        # Extra metadata for dynamic description/enum generation
        "_skill_dir": str(skill_dir.resolve()),
        "_trigger": trigger,
    }

    return role_config


def _get_skill_dirs() -> list[Path]:
    """Collect all skill directory paths to scan."""
    dirs: list[Path] = []

    # 1. Local project skills
    if _LOCAL_SKILLS_DIR.is_dir():
        dirs.append(_LOCAL_SKILLS_DIR)

    # 2. Additional paths from env var (colon-separated)
    extra = os.environ.get("SKILL_DIRS", "")
    if extra:
        for p in extra.split(":"):
            p = p.strip()
            if p:
                path = Path(p).expanduser()
                if path.is_dir():
                    dirs.append(path)

    return dirs


def load_skills() -> dict[str, dict]:
    """Scan all skill directories and return role configs.

    Returns:
        Dict mapping skill name → role config (agents.yaml compatible).
        Cached after first call.
    """
    global _skill_cache
    if _skill_cache is not None:
        return _skill_cache

    skills: dict[str, dict] = {}

    for scan_dir in _get_skill_dirs():
        # Each subdirectory is a potential skill
        try:
            for child in sorted(scan_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name.startswith(".") or child.name.startswith("_"):
                    continue

                role_cfg = _load_single_skill(child)
                if role_cfg is None:
                    continue

                skill_name = role_cfg["name"]
                # Normalize name to valid identifier (lowercase, hyphens to underscores)
                role_key = skill_name.lower().replace("-", "_").replace(" ", "_")

                if role_key in skills:
                    logger.debug("Skill '%s' already loaded, skipping duplicate from %s",
                                 role_key, child)
                    continue

                skills[role_key] = role_cfg
                logger.info("Loaded skill: %s from %s", role_key, child)
        except Exception as e:
            logger.warning("Failed to scan skill dir %s: %s", scan_dir, e)

    _skill_cache = skills
    if skills:
        logger.info("Total skills loaded: %d (%s)", len(skills), ", ".join(skills.keys()))
    return skills


def reload_skills():
    """Clear skill cache, forcing re-scan on next load_skills() call."""
    global _skill_cache
    _skill_cache = None
    logger.info("Skill cache cleared")


def get_all_role_names(yaml_roles: dict[str, dict]) -> list[str]:
    """Get combined list of role names from agents.yaml + loaded skills.

    Used to dynamically generate the delegate tool's role enum.
    """
    skills = load_skills()
    all_names = list(yaml_roles.keys()) + [k for k in skills if k not in yaml_roles]
    return sorted(all_names)


def get_delegate_description(yaml_roles: dict[str, dict]) -> str:
    """Generate delegate tool description dynamically from all available roles.

    Includes both agents.yaml roles and auto-loaded skills.
    """
    base = (
        "Delegate a complex subtask to a specialized sub-agent with its own independent context. "
        "Use this when a task requires deep research (>3 searches), lengthy code generation, "
        "or multi-step operations. Specify a role for specialized behavior: "
    )

    # Built-in roles with short descriptions
    builtin_descriptions = {
        "planner": "task decomposition",
        "architect": "system design",
        "researcher": "multi-source research",
        "analyst": "data analysis",
        "coder": "code gen/debug",
        "code_reviewer": "code review",
        "build_resolver": "fix build errors",
        "tdd_guide": "test-driven dev",
        "security_auditor": "security audit",
        "writer": "content creation",
        "executor": "shell/file ops",
        "guardian": "quality review",
    }

    parts = []
    for role_key, desc in builtin_descriptions.items():
        if role_key in yaml_roles:
            parts.append(f"'{role_key}' ({desc})")

    # Add skill-based roles
    skills = load_skills()
    for skill_key, skill_cfg in skills.items():
        if skill_key not in yaml_roles:
            short_desc = skill_cfg.get("_trigger", "") or skill_cfg.get("description", "")[:30]
            parts.append(f"'{skill_key}' ({short_desc})")

    return base + ", ".join(parts) + "."
