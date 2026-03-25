"""Project Absorber — analyzes external Git projects and generates fusion plans.

Given a Git URL, clones the repo, analyzes its structure, identifies
reusable capabilities, and generates Evolution actions to integrate them.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Absorber:
    """Analyzes and absorbs external projects into mul-agent."""

    async def absorb(self, git_url: str, llm=None) -> list[Any]:
        """Analyze an external project and generate Evolution actions.

        Returns a list of Evolution objects ready for the Applier.
        """
        from evolution.prescriber import Evolution

        # 1. Clone to temp directory
        repo_path = self._clone(git_url)
        if repo_path is None:
            return [Evolution(
                type="add_skill", target="",
                reason=f"Failed to clone {git_url}",
                patch="", priority=1, confidence=0,
            )]

        try:
            # 2. Analyze project structure
            analysis = self._analyze_structure(repo_path)

            # 3. Use LLM for deep analysis if available
            if llm is not None:
                capabilities = await self._llm_analyze(repo_path, analysis, llm)
            else:
                capabilities = self._rule_based_analyze(analysis)

            # 4. Generate Evolution actions
            evolutions = self._generate_evolutions(capabilities, analysis, repo_path)
            return evolutions
        finally:
            # Cleanup temp directory
            shutil.rmtree(repo_path, ignore_errors=True)

    def _clone(self, git_url: str) -> Path | None:
        """Clone a git repository to a temp directory."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="mul_absorb_"))
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", git_url, str(tmp_dir / "repo")],
                capture_output=True, timeout=120, check=True,
            )
            return tmp_dir / "repo"
        except Exception as e:
            logger.warning("Git clone failed for %s: %s", git_url, e)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    def _analyze_structure(self, repo_path: Path) -> dict[str, Any]:
        """Analyze project structure without LLM."""
        analysis: dict[str, Any] = {
            "name": repo_path.parent.name if repo_path.name == "repo" else repo_path.name,
            "files": [],
            "languages": set(),
            "has_readme": False,
            "readme_content": "",
            "has_api": False,
            "has_cli": False,
            "has_pyproject": False,
            "has_package_json": False,
            "has_dockerfile": False,
            "entry_points": [],
            "dependencies": [],
        }

        # Scan files (limit depth to avoid huge repos)
        for f in sorted(repo_path.rglob("*")):
            if f.is_dir() or ".git" in f.parts:
                continue
            rel = f.relative_to(repo_path)
            depth = len(rel.parts)
            if depth > 4:
                continue

            analysis["files"].append(str(rel))

            # Language detection
            suffix = f.suffix.lower()
            lang_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
                ".sh": "shell", ".cpp": "cpp", ".c": "c",
            }
            if suffix in lang_map:
                analysis["languages"].add(lang_map[suffix])

        # README
        for name in ("README.md", "README.rst", "README.txt", "README"):
            readme = repo_path / name
            if readme.exists():
                analysis["has_readme"] = True
                try:
                    analysis["readme_content"] = readme.read_text()[:3000]
                except Exception:
                    pass
                break

        # Key files
        if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
            analysis["has_pyproject"] = True
        if (repo_path / "package.json").exists():
            analysis["has_package_json"] = True
        if (repo_path / "Dockerfile").exists():
            analysis["has_dockerfile"] = True

        # API detection
        api_indicators = ["fastapi", "flask", "express", "gin", "actix", "axum"]
        for f_path in analysis["files"][:200]:
            f_lower = f_path.lower()
            if any(ind in f_lower for ind in ["routes", "api", "endpoint", "server"]):
                analysis["has_api"] = True
            if any(ind in f_lower for ind in ["cli", "main", "__main__", "cmd"]):
                analysis["has_cli"] = True

        # Read requirements/dependencies
        for dep_file in ["requirements.txt", "pyproject.toml", "package.json"]:
            dep_path = repo_path / dep_file
            if dep_path.exists():
                try:
                    content = dep_path.read_text()[:2000]
                    analysis["dependencies"].append({
                        "file": dep_file,
                        "content": content,
                    })
                    for indicator in api_indicators:
                        if indicator in content.lower():
                            analysis["has_api"] = True
                except Exception:
                    pass

        analysis["languages"] = sorted(analysis["languages"])
        return analysis

    async def _llm_analyze(
        self, repo_path: Path, analysis: dict, llm
    ) -> list[dict[str, Any]]:
        """Use LLM to deeply analyze project capabilities."""
        from langchain_core.messages import HumanMessage, SystemMessage
        import asyncio

        # Build context from analysis
        context_parts = [
            f"Project: {analysis['name']}",
            f"Languages: {', '.join(analysis['languages'])}",
            f"Has API: {analysis['has_api']}",
            f"Has CLI: {analysis['has_cli']}",
            f"Has Docker: {analysis['has_dockerfile']}",
            f"Key files ({len(analysis['files'])} total): {', '.join(analysis['files'][:30])}",
        ]
        if analysis["readme_content"]:
            context_parts.append(f"\nREADME:\n{analysis['readme_content'][:2000]}")

        # Read a few key source files for deeper understanding
        source_snippets = []
        for f_str in analysis["files"][:100]:
            f_path = repo_path / f_str
            if f_path.suffix in (".py", ".js", ".ts", ".go") and f_path.stat().st_size < 5000:
                try:
                    content = f_path.read_text()[:1500]
                    source_snippets.append(f"\n--- {f_str} ---\n{content}")
                except Exception:
                    pass
            if len(source_snippets) >= 5:
                break

        if source_snippets:
            context_parts.append("\nSample source files:" + "".join(source_snippets[:3]))

        messages = [
            SystemMessage(content="""Analyze this external project and identify capabilities that can be integrated into a multi-agent AI framework (mul-agent).

For each capability, determine the best integration method:
- "api_tool": Project has an HTTP API → register as API tool in tools.yaml
- "cli_skill": Project is a CLI tool → wrap as a Skill with execute_shell
- "python_lib": Project is a Python library → suggest native tool integration
- "knowledge": Project has useful documentation → extract as knowledge base

Respond with ONLY a JSON array:
[
  {
    "name": "capability name",
    "description": "what it does",
    "integration_type": "api_tool|cli_skill|python_lib|knowledge",
    "api_endpoint": "http://... (if api_tool)",
    "cli_command": "command to invoke (if cli_skill)",
    "import_path": "python.module.path (if python_lib)",
    "doc_content": "extracted knowledge (if knowledge)",
    "confidence": 0.0-1.0
  }
]"""),
            HumanMessage(content="\n".join(context_parts)),
        ]

        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30)
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
            result = json.loads(content)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning("LLM project analysis failed: %s", e)
            return self._rule_based_analyze(analysis)

    def _rule_based_analyze(self, analysis: dict) -> list[dict[str, Any]]:
        """Fallback rule-based analysis without LLM."""
        capabilities = []
        name = analysis["name"]

        if analysis["has_api"]:
            capabilities.append({
                "name": f"{name}_api",
                "description": f"API service from {name}",
                "integration_type": "api_tool",
                "confidence": 0.5,
            })

        if analysis["has_cli"]:
            capabilities.append({
                "name": f"{name}_cli",
                "description": f"CLI tool from {name}",
                "integration_type": "cli_skill",
                "confidence": 0.5,
            })

        if "python" in analysis.get("languages", []) and analysis.get("has_pyproject"):
            capabilities.append({
                "name": f"{name}_lib",
                "description": f"Python library from {name}",
                "integration_type": "python_lib",
                "confidence": 0.4,
            })

        if analysis.get("readme_content"):
            capabilities.append({
                "name": f"{name}_docs",
                "description": f"Documentation from {name}",
                "integration_type": "knowledge",
                "doc_content": analysis["readme_content"][:2000],
                "confidence": 0.7,
            })

        return capabilities

    def _generate_evolutions(
        self,
        capabilities: list[dict],
        analysis: dict,
        repo_path: Path,
    ) -> list[Any]:
        """Convert capabilities into Evolution actions."""
        from evolution.prescriber import Evolution
        evolutions = []

        for cap in capabilities:
            itype = cap.get("integration_type", "")
            name = cap.get("name", "unknown")
            desc = cap.get("description", "")
            conf = cap.get("confidence", 0.5)

            if itype == "api_tool":
                endpoint = cap.get("api_endpoint", "http://localhost:8080/api")
                evolutions.append(Evolution(
                    type="add_tool",
                    target="config/tools.yaml",
                    reason=f"Absorb API capability from external project: {desc}",
                    patch={
                        "name": name,
                        "description": desc,
                        "endpoint": endpoint,
                        "method": "POST",
                        "params": {"input": {"type": "string", "required": True}},
                    },
                    priority=2,
                    confidence=conf,
                ))

            elif itype == "cli_skill":
                cli_cmd = cap.get("cli_command", name)
                skill_prompt = (
                    f"You are an expert at using the {name} tool.\n"
                    f"Description: {desc}\n\n"
                    f"Use execute_shell to run: {cli_cmd}\n"
                    f"Parse the output and present results clearly."
                )
                evolutions.append(Evolution(
                    type="add_skill",
                    target=f"config/skills/{name}",
                    reason=f"Absorb CLI tool from external project: {desc}",
                    patch={
                        "name": name,
                        "description": desc,
                        "tools": ["execute_shell", "read_file", "write_file"],
                        "prompt": skill_prompt,
                    },
                    priority=2,
                    confidence=conf,
                ))

            elif itype == "python_lib":
                import_path = cap.get("import_path", name)
                evolutions.append(Evolution(
                    type="add_skill",
                    target=f"config/skills/{name}",
                    reason=f"Absorb Python library: {desc} (needs manual tool code for deep integration)",
                    patch={
                        "name": name,
                        "description": desc,
                        "tools": ["code_run", "execute_shell", "read_file"],
                        "prompt": (
                            f"You are an expert at using the {name} Python library.\n"
                            f"Import path: {import_path}\n"
                            f"Description: {desc}\n\n"
                            f"Use code_run to write Python scripts that import and use this library.\n"
                            f"Ensure proper error handling and clear output."
                        ),
                    },
                    priority=2,
                    confidence=conf * 0.8,  # Lower confidence for lib integration
                ))

            elif itype == "knowledge":
                doc_content = cap.get("doc_content", "")
                if doc_content:
                    evolutions.append(Evolution(
                        type="update_knowledge",
                        target=f"config/knowledge/{name}.md",
                        reason=f"Absorb documentation from external project: {desc}",
                        patch=f"# {name}\n\n{doc_content}",
                        priority=3,
                        confidence=conf,
                    ))

        return evolutions
