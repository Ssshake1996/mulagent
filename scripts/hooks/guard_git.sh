#!/usr/bin/env bash
# guard_git.sh — Pre-hook for git_ops
# Safe ops → exit 0 (allow + auto-stash if needed)
# Dangerous ops → exit 2 (request user confirmation)
# All ops → audit log
#
# Like Claude Code: destructive git ops need user approval.

set -uo pipefail

ACTION="$1"
ARGS="${2:-}"

PROJECT_ROOT="${MULAGENT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="$PROJECT_ROOT/data/audit"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/git_audit.jsonl"

ACTION_LOWER=$(echo "$ACTION" | tr '[:upper:]' '[:lower:]')
ARGS_LOWER=$(echo "$ARGS" | tr '[:upper:]' '[:lower:]')
COMBINED="$ACTION_LOWER $ARGS_LOWER"

# ── Dangerous patterns: require user confirmation (exit 2) ──
NEED_CONFIRM=false
REASON=""

# Force push to main/master
if [ "$ACTION_LOWER" = "push" ] && echo "$ARGS_LOWER" | grep -qE "(--force|-f).*(main|master)" 2>/dev/null; then
    NEED_CONFIRM=true
    REASON="强制推送到主分支"
fi

# Hard reset
if [ "$ACTION_LOWER" = "reset" ] && echo "$ARGS_LOWER" | grep -qF "hard" 2>/dev/null; then
    NEED_CONFIRM=true
    REASON="硬重置 (git reset --hard) 会丢失未提交的更改"
fi

# Clean with force
if [ "$ACTION_LOWER" = "clean" ] && echo "$ARGS_LOWER" | grep -qE "(-f|--force)" 2>/dev/null; then
    NEED_CONFIRM=true
    REASON="强制清理未跟踪文件"
fi

# Delete branch
if echo "$COMBINED" | grep -qE "branch.*(-D|--delete.*--force)" 2>/dev/null; then
    NEED_CONFIRM=true
    REASON="强制删除分支"
fi

if $NEED_CONFIRM; then
    echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"CONFIRM_REQ\",\"action\":\"$ACTION\",\"args\":\"${ARGS:0:200}\",\"reason\":\"$REASON\"}" >> "$LOG"
    echo "$REASON: git $ACTION $ARGS" >&2
    exit 2
fi

# ── Auto-stash before risky operations (safety net, not blocking) ──
if echo "$ACTION_LOWER" | grep -qE "^(checkout|switch|rebase|merge|pull)$" 2>/dev/null; then
    if cd "$PROJECT_ROOT" 2>/dev/null && ! git diff --quiet 2>/dev/null; then
        STASH_MSG="mulagent-auto-stash-$(date +%Y%m%d_%H%M%S)-before-$ACTION"
        if git stash push -m "$STASH_MSG" 2>/dev/null; then
            echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"AUTO_STASH\",\"action\":\"$ACTION\",\"stash_msg\":\"$STASH_MSG\"}" >> "$LOG"
        fi
    fi
fi

# ── Audit log ──
LEVEL="INFO"
if echo "$COMBINED" | grep -qiE "(force|--hard|clean|-D)" 2>/dev/null; then
    LEVEL="WARN"
fi
echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"GIT_OP\",\"level\":\"$LEVEL\",\"action\":\"$ACTION\",\"args\":\"${ARGS:0:500}\"}" >> "$LOG"

exit 0
