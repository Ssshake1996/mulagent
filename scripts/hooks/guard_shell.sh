#!/usr/bin/env bash
# guard_shell.sh — Pre-hook for execute_shell
# Safe commands → exit 0 (allow)
# Dangerous commands → exit 2 (request user confirmation)
# All commands → audit log
#
# Like Claude Code: dangerous ops need user approval, never silently blocked.

set -uo pipefail

CMD="$1"
LOG_DIR="${MULAGENT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/data/audit"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/shell_audit.jsonl"

CMD_LOWER=$(echo "$CMD" | tr '[:upper:]' '[:lower:]')

# ── Dangerous patterns: require user confirmation (exit 2) ──
declare -A DANGER_PATTERNS=(
    ['rm -rf /$']="删除根目录"
    ['rm -rf /(home|etc|var|usr|opt|boot|root|srv|lib|bin|sbin)']="删除系统关键目录"
    ['rm -rf ~']="删除用户目录"
    ['rm -rf \*$']="删除所有文件"
    ['rm -rf \.$']="删除当前目录"
    ['mkfs\.']="格式化磁盘"
    ['dd if=.* of=/dev/']="写入设备"
    ['> /dev/sd']="覆盖磁盘设备"
    ['--no-preserve-root']="无保护删除根目录"
    ['chmod -R 777 /$']="全局权限变更"
    ['curl .* \| sh']="远程脚本执行"
    ['curl .* \| bash']="远程脚本执行"
    ['wget .* \| sh']="远程脚本执行"
    ['wget .* \| bash']="远程脚本执行"
    ['^shutdown']="关机"
    ['^reboot']="重启"
    ['^init 0$']="关机"
    ['^init 6$']="重启"
    ['^killall ']="批量终止进程"
    ['npm publish']="发布 npm 包"
    ['pip upload']="上传 pip 包"
    ['DROP DATABASE']="删除数据库"
    ['DROP TABLE']="删除数据表"
    ['TRUNCATE ']="清空数据表"
)

for pattern in "${!DANGER_PATTERNS[@]}"; do
    if echo "$CMD_LOWER" | grep -qiE "$pattern" 2>/dev/null; then
        reason="${DANGER_PATTERNS[$pattern]}"
        echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"CONFIRM_REQ\",\"cmd\":\"${CMD:0:200}\",\"reason\":\"$reason\"}" >> "$LOG"
        echo "$reason: ${CMD:0:200}" >&2
        exit 2
    fi
done

# ── Warn-level patterns: allow but tag in audit log ──
LEVEL="INFO"
WARN_PATTERNS=('rm -rf' 'rm -r' 'git push' 'git reset' 'pip install' 'pip uninstall'
               'apt install' 'apt remove' 'docker rm' 'docker rmi' 'chmod -R' 'chown -R' 'kill ')

for pattern in "${WARN_PATTERNS[@]}"; do
    if echo "$CMD" | grep -qiF "$pattern" 2>/dev/null; then
        LEVEL="WARN"
        break
    fi
done

echo "{\"ts\":\"$(date -Iseconds)\",\"event\":\"EXEC\",\"level\":\"$LEVEL\",\"cmd\":\"${CMD:0:500}\"}" >> "$LOG"
exit 0
