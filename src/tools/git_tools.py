"""Git/GitHub integration tools.

Provides safe Git operations and GitHub API access:
- git_diff: View changes (staged, unstaged, between commits)
- git_commit: Stage and commit changes
- github_pr: Create/view pull requests
- github_issue: Create/view issues

All operations are non-destructive by default.
Uses `gh` CLI for GitHub operations when available.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 30


async def _run_git(args: list[str], timeout: int = _GIT_TIMEOUT) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
        )
    except asyncio.TimeoutError:
        return (-1, "", f"git command timed out ({timeout}s)")
    except Exception as e:
        return (-1, "", str(e))


async def _run_gh(args: list[str], timeout: int = _GIT_TIMEOUT) -> tuple[int, str, str]:
    """Run a gh CLI command and return (returncode, stdout, stderr)."""
    if not shutil.which("gh"):
        return (-1, "", "gh CLI not installed. Install from https://cli.github.com/")
    try:
        proc = await asyncio.create_subprocess_exec(
            "gh", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
        )
    except asyncio.TimeoutError:
        return (-1, "", f"gh command timed out ({timeout}s)")
    except Exception as e:
        return (-1, "", str(e))


async def _git_ops(params: dict[str, Any], **deps: Any) -> str:
    """Git operations: diff, log, status, commit, branch management."""
    action = params.get("action", "")
    if not action:
        return "Error: action is required"

    if action == "diff":
        target = params.get("target", "")
        if target == "staged":
            rc, out, err = await _run_git(["diff", "--cached", "--stat"])
            if rc == 0 and out:
                # Also get the actual diff (limited)
                rc2, diff, _ = await _run_git(["diff", "--cached"])
                if len(diff) > 5000:
                    diff = diff[:5000] + "\n... (diff truncated)"
                return f"Staged changes:\n{out}\n\n{diff}"
            return "No staged changes"
        elif target:
            rc, out, err = await _run_git(["diff", target])
        else:
            rc, out, err = await _run_git(["diff", "--stat"])
            if rc == 0 and out:
                rc2, diff, _ = await _run_git(["diff"])
                if len(diff) > 5000:
                    diff = diff[:5000] + "\n... (diff truncated)"
                return f"Unstaged changes:\n{out}\n\n{diff}"
            return "No changes"
        if rc != 0:
            return f"git diff error: {err}"
        if len(out) > 5000:
            out = out[:5000] + "\n... (diff truncated)"
        return out or "No changes"

    elif action == "status":
        rc, out, err = await _run_git(["status", "--short"])
        if rc != 0:
            return f"git status error: {err}"
        return out or "Working tree clean"

    elif action == "log":
        count = params.get("count", 10)
        rc, out, err = await _run_git([
            "log", f"-{min(count, 50)}",
            "--oneline", "--decorate",
        ])
        if rc != 0:
            return f"git log error: {err}"
        return out

    elif action == "commit":
        message = params.get("message", "")
        files = params.get("files", "")
        if not message:
            return "Error: commit message is required"

        # Stage files
        if files:
            file_list = [f.strip() for f in files.split(",") if f.strip()]
            rc, _, err = await _run_git(["add"] + file_list)
        else:
            rc, _, err = await _run_git(["add", "-A"])
        if rc != 0:
            return f"git add error: {err}"

        # Check if there are staged changes
        rc, staged, _ = await _run_git(["diff", "--cached", "--stat"])
        if not staged:
            return "Nothing to commit (no staged changes after add)"

        # Commit
        rc, out, err = await _run_git(["commit", "-m", message])
        if rc != 0:
            return f"git commit error: {err}"
        return f"Committed: {out}"

    elif action == "branch":
        subaction = params.get("target", "list")
        if subaction == "list":
            rc, out, _ = await _run_git(["branch", "-a", "--no-color"])
            return out or "No branches"
        elif subaction == "current":
            rc, out, _ = await _run_git(["branch", "--show-current"])
            return out or "HEAD detached"
        else:
            return f"Unknown branch subaction: {subaction}"

    else:
        return f"Unknown action: {action}. Supported: diff, status, log, commit, branch"


async def _github_ops(params: dict[str, Any], **deps: Any) -> str:
    """GitHub operations via gh CLI: PR and issue management."""
    action = params.get("action", "")
    if not action:
        return "Error: action is required"

    if action == "pr_list":
        rc, out, err = await _run_gh(["pr", "list", "--limit", "10"])
        if rc != 0:
            return f"gh pr list error: {err}"
        return out or "No open PRs"

    elif action == "pr_create":
        title = params.get("title", "")
        body = params.get("body", "")
        base = params.get("base", "main")
        if not title:
            return "Error: PR title is required"
        args = ["pr", "create", "--title", title, "--base", base]
        if body:
            args.extend(["--body", body])
        rc, out, err = await _run_gh(args)
        if rc != 0:
            return f"gh pr create error: {err}"
        return f"PR created: {out}"

    elif action == "pr_view":
        pr_number = params.get("target", "")
        if not pr_number:
            rc, out, err = await _run_gh(["pr", "view", "--web=false"])
        else:
            rc, out, err = await _run_gh(["pr", "view", str(pr_number)])
        if rc != 0:
            return f"gh pr view error: {err}"
        return out

    elif action == "issue_list":
        rc, out, err = await _run_gh(["issue", "list", "--limit", "10"])
        if rc != 0:
            return f"gh issue list error: {err}"
        return out or "No open issues"

    elif action == "issue_create":
        title = params.get("title", "")
        body = params.get("body", "")
        if not title:
            return "Error: issue title is required"
        args = ["issue", "create", "--title", title]
        if body:
            args.extend(["--body", body])
        rc, out, err = await _run_gh(args)
        if rc != 0:
            return f"gh issue create error: {err}"
        return f"Issue created: {out}"

    else:
        return f"Unknown action: {action}. Supported: pr_list, pr_create, pr_view, issue_list, issue_create"


GIT_OPS = ToolDef(
    name="git_ops",
    description=(
        "Git operations: view diffs, status, log, create commits, manage branches. "
        "Use action='diff' to see changes, action='status' for working tree status, "
        "action='log' for commit history, action='commit' to create a commit."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Git action to perform",
                "enum": ["diff", "status", "log", "commit", "branch"],
            },
            "target": {
                "type": "string",
                "description": "Target for the action (e.g., 'staged' for diff, branch name, commit range)",
            },
            "message": {
                "type": "string",
                "description": "Commit message (required for action='commit')",
            },
            "files": {
                "type": "string",
                "description": "Comma-separated file paths to stage (for commit, default: all)",
            },
        },
        "required": ["action"],
    },
    fn=_git_ops,
)

GITHUB_OPS = ToolDef(
    name="github_ops",
    description=(
        "GitHub operations via gh CLI: manage pull requests and issues. "
        "Requires gh CLI to be installed and authenticated. "
        "Use action='pr_list' to list PRs, 'pr_create' to create a PR, "
        "'issue_list' to list issues, 'issue_create' to create an issue."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "GitHub action to perform",
                "enum": ["pr_list", "pr_create", "pr_view", "issue_list", "issue_create"],
            },
            "title": {
                "type": "string",
                "description": "Title for PR or issue creation",
            },
            "body": {
                "type": "string",
                "description": "Body/description for PR or issue",
            },
            "target": {
                "type": "string",
                "description": "PR number for pr_view",
            },
            "base": {
                "type": "string",
                "description": "Base branch for PR (default: main)",
            },
        },
        "required": ["action"],
    },
    fn=_github_ops,
)
