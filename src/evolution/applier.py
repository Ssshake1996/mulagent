"""Evolution Applier — safely applies evolution actions to project files.

Handles backup, validation, rollback, and hot-reload for each change type.
Only modifies configuration files, never core source code.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    """Result of applying a single evolution."""
    evolution_type: str
    target: str
    success: bool
    message: str
    backup_path: str = ""


class Applier:
    """Applies evolution actions with backup and rollback."""

    def __init__(self, project_root: Path | None = None):
        from common.config import _find_project_root
        self._root = project_root or _find_project_root()
        self._backup_dir = self._root / "data" / "evolution_backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    async def apply(
        self,
        evolutions: list[Any],
        dry_run: bool = False,
    ) -> list[ApplyResult]:
        """Apply a list of Evolution actions.

        Args:
            evolutions: List of Evolution objects to apply.
            dry_run: If True, validate but don't write.
        """
        results = []
        for evo in evolutions:
            try:
                result = self._apply_one(evo, dry_run=dry_run)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to apply %s: %s", evo.type, e)
                results.append(ApplyResult(
                    evolution_type=evo.type,
                    target=evo.target,
                    success=False,
                    message=f"Exception: {e}",
                ))
        return results

    def _apply_one(self, evo: Any, dry_run: bool = False) -> ApplyResult:
        """Apply a single evolution action."""
        handlers = {
            "prompt_refine": self._apply_prompt_refine,
            "add_skill": self._apply_add_skill,
            "add_tool": self._apply_add_tool,
            "update_knowledge": self._apply_update_knowledge,
            "tune_params": self._apply_tune_params,
        }
        handler = handlers.get(evo.type)
        if handler is None:
            return ApplyResult(
                evolution_type=evo.type,
                target=evo.target,
                success=False,
                message=f"Unknown evolution type: {evo.type}",
            )
        return handler(evo, dry_run=dry_run)

    # ── Prompt refinement ────────────────────────────────────────

    def _apply_prompt_refine(self, evo: Any, dry_run: bool) -> ApplyResult:
        """Update a role's prompt in agents.yaml."""
        agents_path = self._root / "config" / "agents.yaml"
        if not agents_path.exists():
            return ApplyResult(evo.type, evo.target, False, "agents.yaml not found")

        backup = self._backup(agents_path)
        try:
            data = yaml.safe_load(agents_path.read_text())
            roles = data.get("roles", {})

            # Parse target like "roles.coder" → role_name = "coder"
            role_name = evo.target.replace("roles.", "").strip()
            if role_name not in roles:
                return ApplyResult(evo.type, evo.target, False,
                                   f"Role '{role_name}' not found in agents.yaml")

            if dry_run:
                return ApplyResult(evo.type, evo.target, True,
                                   f"[dry-run] Would update prompt for '{role_name}'",
                                   backup_path=str(backup))

            # Apply the patch — replace or append to prompt
            patch = evo.patch
            if isinstance(patch, str):
                if patch.startswith("+"):
                    # Append mode
                    roles[role_name]["prompt"] = (
                        roles[role_name].get("prompt", "") + "\n" + patch[1:]
                    )
                else:
                    roles[role_name]["prompt"] = patch

            agents_path.write_text(
                yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
            )
            self._trigger_reload("roles")
            return ApplyResult(evo.type, evo.target, True,
                               f"Updated prompt for role '{role_name}'",
                               backup_path=str(backup))
        except Exception as e:
            self._rollback(agents_path, backup)
            return ApplyResult(evo.type, evo.target, False, f"Rollback: {e}",
                               backup_path=str(backup))

    # ── Add skill ────────────────────────────────────────────────

    def _apply_add_skill(self, evo: Any, dry_run: bool) -> ApplyResult:
        """Create a new skill directory with SKILL.md."""
        skills_dir = self._root / "config" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Extract skill name from target
        skill_name = Path(evo.target).name or evo.target.strip("/").split("/")[-1]
        skill_dir = skills_dir / skill_name

        if skill_dir.exists():
            return ApplyResult(evo.type, evo.target, False,
                               f"Skill '{skill_name}' already exists")

        if dry_run:
            return ApplyResult(evo.type, evo.target, True,
                               f"[dry-run] Would create skill '{skill_name}'")

        try:
            skill_dir.mkdir(parents=True)
            (skill_dir / "references").mkdir()

            # Write SKILL.md
            if isinstance(evo.patch, dict):
                skill_md = self._build_skill_md(evo.patch)
            elif isinstance(evo.patch, str):
                skill_md = evo.patch
            else:
                skill_md = f"---\nname: {skill_name}\ndescription: Auto-generated skill\n---\n\n{evo.reason}\n"

            (skill_dir / "SKILL.md").write_text(skill_md)
            self._trigger_reload("skills")
            return ApplyResult(evo.type, evo.target, True,
                               f"Created skill '{skill_name}' at {skill_dir}")
        except Exception as e:
            # Cleanup on failure
            if skill_dir.exists():
                shutil.rmtree(skill_dir, ignore_errors=True)
            return ApplyResult(evo.type, evo.target, False, f"Failed: {e}")

    def _build_skill_md(self, spec: dict) -> str:
        """Build SKILL.md from a spec dict."""
        name = spec.get("name", "unnamed")
        desc = spec.get("description", "")
        tools = spec.get("tools", [])
        prompt = spec.get("prompt", "")

        tools_line = f"tools: [{', '.join(tools)}]" if tools else ""
        return (
            f"---\n"
            f"name: \"{name}\"\n"
            f"description: \"{desc}\"\n"
            f"{tools_line}\n"
            f"---\n\n"
            f"{prompt}\n"
        )

    # ── Add tool ─────────────────────────────────────────────────

    def _apply_add_tool(self, evo: Any, dry_run: bool) -> ApplyResult:
        """Add an API tool entry to tools.yaml."""
        tools_path = self._root / "config" / "tools.yaml"

        if dry_run:
            return ApplyResult(evo.type, evo.target, True,
                               f"[dry-run] Would add tool to tools.yaml")

        backup = self._backup(tools_path) if tools_path.exists() else ""

        try:
            if tools_path.exists():
                data = yaml.safe_load(tools_path.read_text()) or {}
            else:
                data = {}

            tools_list = data.setdefault("tools", [])

            if isinstance(evo.patch, dict):
                # Check for duplicate
                existing_names = {t.get("name") for t in tools_list if isinstance(t, dict)}
                new_name = evo.patch.get("name", "")
                if new_name in existing_names:
                    return ApplyResult(evo.type, evo.target, False,
                                       f"Tool '{new_name}' already exists")
                tools_list.append(evo.patch)
            elif isinstance(evo.patch, str):
                parsed = yaml.safe_load(evo.patch)
                if isinstance(parsed, dict):
                    tools_list.append(parsed)

            tools_path.write_text(
                yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
            )
            return ApplyResult(evo.type, evo.target, True,
                               f"Added tool to tools.yaml",
                               backup_path=str(backup))
        except Exception as e:
            if backup:
                self._rollback(tools_path, Path(backup))
            return ApplyResult(evo.type, evo.target, False, f"Failed: {e}")

    # ── Update knowledge ─────────────────────────────────────────

    def _apply_update_knowledge(self, evo: Any, dry_run: bool) -> ApplyResult:
        """Create or update a knowledge base file."""
        kb_dir = self._root / "config" / "knowledge"
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Determine target file
        target = evo.target
        if not target.endswith(".md"):
            target = target + ".md"
        # Strip directory prefix if present
        target_name = Path(target).name
        kb_path = kb_dir / target_name

        if dry_run:
            action = "update" if kb_path.exists() else "create"
            return ApplyResult(evo.type, evo.target, True,
                               f"[dry-run] Would {action} {target_name}")

        backup = self._backup(kb_path) if kb_path.exists() else ""

        try:
            content = evo.patch if isinstance(evo.patch, str) else str(evo.patch)

            if kb_path.exists():
                # Append to existing
                existing = kb_path.read_text()
                kb_path.write_text(existing + "\n\n" + content)
            else:
                kb_path.write_text(content)

            return ApplyResult(evo.type, evo.target, True,
                               f"Updated knowledge: {target_name}",
                               backup_path=str(backup))
        except Exception as e:
            if backup:
                self._rollback(kb_path, Path(backup))
            return ApplyResult(evo.type, evo.target, False, f"Failed: {e}")

    # ── Tune parameters ──────────────────────────────────────────

    def _apply_tune_params(self, evo: Any, dry_run: bool) -> ApplyResult:
        """Update ReAct or other config parameters."""
        settings_path = self._root / "config" / "settings.yaml"
        if not settings_path.exists():
            return ApplyResult(evo.type, evo.target, False, "settings.yaml not found")

        if dry_run:
            return ApplyResult(evo.type, evo.target, True,
                               f"[dry-run] Would tune {evo.target}: {evo.patch}")

        backup = self._backup(settings_path)
        try:
            data = yaml.safe_load(settings_path.read_text()) or {}

            # Parse target like "react.timeout" → data["react"]["timeout"]
            if isinstance(evo.patch, dict):
                section = evo.target.split(".")[0] if "." in evo.target else "react"
                if section not in data:
                    data[section] = {}
                for key, value in evo.patch.items():
                    data[section][key] = value
            else:
                parts = evo.target.split(".")
                if len(parts) == 2:
                    section, key = parts
                    if section not in data:
                        data[section] = {}
                    data[section][key] = evo.patch

            settings_path.write_text(
                yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
            )
            self._trigger_reload("config")
            return ApplyResult(evo.type, evo.target, True,
                               f"Tuned {evo.target}",
                               backup_path=str(backup))
        except Exception as e:
            self._rollback(settings_path, backup)
            return ApplyResult(evo.type, evo.target, False, f"Rollback: {e}",
                               backup_path=str(backup))

    # ── Backup / rollback ────────────────────────────────────────

    def _backup(self, path: Path) -> Path:
        """Create a timestamped backup of a file."""
        if not path.exists():
            return Path("")
        ts = int(time.time())
        backup = self._backup_dir / f"{path.name}.{ts}.bak"
        shutil.copy2(path, backup)
        return backup

    def _rollback(self, target: Path, backup: Path) -> None:
        """Restore a file from backup."""
        if backup and backup.exists():
            shutil.copy2(backup, target)
            logger.info("Rolled back %s from %s", target, backup)

    def _trigger_reload(self, scope: str) -> None:
        """Trigger hot-reload of cached resources."""
        try:
            if scope == "config":
                from common.config import reload_settings
                reload_settings()
            elif scope == "skills":
                from tools.skill_loader import reload_skills
                reload_skills()
            elif scope == "roles":
                # Role cache is function-local in isolation.py; clear via import hack
                import importlib
                import tools.isolation
                importlib.reload(tools.isolation)
        except Exception as e:
            logger.debug("Hot-reload (%s) skipped: %s", scope, e)
