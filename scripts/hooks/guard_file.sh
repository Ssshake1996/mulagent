#!/usr/bin/env bash
# guard_file.sh — Pre-hook for write_file / edit_file
# Auto-backup before modification + audit logging.
#
# Exceeds Claude Code: Claude Code has no auto-backup mechanism.
# This creates timestamped snapshots so any file change can be reverted
# even without git (e.g., config files, generated code outside repos).
#
# Usage: guard_file.sh <file_path> <operation>
#   operation: "write" or "edit"

set -euo pipefail

FILE_PATH="$1"
OPERATION="${2:-write}"

PROJECT_ROOT="${MULAGENT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BACKUP_DIR="$PROJECT_ROOT/data/backups"
LOG_DIR="$PROJECT_ROOT/data/audit"
mkdir -p "$BACKUP_DIR" "$LOG_DIR"
LOG="$LOG_DIR/file_audit.jsonl"

# ── Auto-backup if file exists ──
if [ -f "$FILE_PATH" ]; then
    # Create backup with timestamp: data/backups/<filename>.<timestamp>.bak
    BASENAME=$(basename "$FILE_PATH")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/${BASENAME}.${TIMESTAMP}.bak"

    cp "$FILE_PATH" "$BACKUP_PATH" 2>/dev/null || true

    # Get file size for logging
    FILE_SIZE=$(stat -c%s "$FILE_PATH" 2>/dev/null || stat -f%z "$FILE_PATH" 2>/dev/null || echo "?")

    echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"BACKUP\",\"op\":\"$OPERATION\",\"file\":\"$FILE_PATH\",\"backup\":\"$BACKUP_PATH\",\"size\":$FILE_SIZE}" >> "$LOG"
else
    echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"NEW_FILE\",\"op\":\"$OPERATION\",\"file\":\"$FILE_PATH\"}" >> "$LOG"
fi

# ── Cleanup old backups (keep last 50 per file) ──
if [ -d "$BACKUP_DIR" ]; then
    BNAME=$(basename "$FILE_PATH")
    OLD_BACKUPS=$(find "$BACKUP_DIR" -name "${BNAME}.*.bak" -type f 2>/dev/null | sort | head -n -50 || true)
    if [ -n "$OLD_BACKUPS" ]; then
        echo "$OLD_BACKUPS" | xargs rm -f 2>/dev/null || true
    fi
fi

exit 0
