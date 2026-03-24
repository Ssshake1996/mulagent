#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  mul-agent 统一管理脚本
#
#  服务清单（systemd 管理）:
#    - postgresql@16-main   系统服务，开机自启
#    - mulagent-feishu      用户服务，开机自启（WebSocket bot）
#
#  用法:
#    ./scripts/setup.sh                 # 检查服务 + 启动 CLI (TUI)
#    ./scripts/setup.sh --headless      # 检查服务 + 启动 CLI (headless)
#    ./scripts/setup.sh -c "查天气"     # 检查服务 + 单次执行
#    ./scripts/setup.sh --status        # 查看所有服务状态
#    ./scripts/setup.sh --restart       # 重启应用服务（更新代码后）
#    ./scripts/setup.sh --stop          # 停止应用服务
#    ./scripts/setup.sh --logs [N]      # 查看 feishu bot 最近日志
#    ./scripts/setup.sh --infra         # 仅检查基础设施，不启动 CLI
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# systemd service names
SVC_PG="postgresql@16-main"          # system-level
SVC_BOT="mulagent-feishu"            # user-level

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; }

# ── Parse arguments ───────────────────────────────────────────────
MODE="cli"
CLI_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)       MODE="status";  shift ;;
        --restart)      MODE="restart"; shift ;;
        --stop)         MODE="stop";    shift ;;
        --infra)        MODE="infra";   shift ;;
        --logs)
            MODE="logs"
            shift
            LOG_LINES="${1:-50}"
            [[ "$LOG_LINES" =~ ^[0-9]+$ ]] && shift || LOG_LINES=50
            ;;
        --headless)     CLI_ARGS+=("--headless");      shift ;;
        --model)        CLI_ARGS+=("--model" "$2");    shift 2 ;;
        --config)       CLI_ARGS+=("--config" "$2");   shift 2 ;;
        --session)      CLI_ARGS+=("--session" "$2");  shift 2 ;;
        -c|--command)   CLI_ARGS+=("-c" "$2");         shift 2 ;;
        -h|--help)
            cat <<'HELP'
Usage: ./scripts/setup.sh [MODE] [CLI_OPTIONS]

Modes:
  (default)       Check services + launch CLI (TUI)
  --status        Show all service status
  --restart       Restart feishu bot (after code update)
  --stop          Stop feishu bot
  --logs [N]      Show last N lines of bot log (default 50)
  --infra         Check infrastructure only, don't launch CLI

CLI options (passed to mulagent):
  --headless      Plain REPL instead of TUI
  --model ID      Override default LLM model
  --config PATH   Custom config file
  --session ID    Resume a specific session
  -c COMMAND      Execute single command and exit
HELP
            exit 0 ;;
        *)  CLI_ARGS+=("$1"); shift ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────

check_system_svc() {
    systemctl is-active --quiet "$1" 2>/dev/null
}

check_user_svc() {
    systemctl --user is-active --quiet "$1" 2>/dev/null
}

print_svc_status() {
    local name="$1" active="$2"
    printf "  %-20s" "$name"
    if [[ "$active" == "true" ]]; then
        echo -e "${GREEN}running${NC}"
    else
        echo -e "${RED}stopped${NC}"
    fi
}

# ── Status mode ───────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    echo ""
    echo -e "${CYAN}mul-agent service status${NC}"
    echo ""

    # PostgreSQL
    pg_ok=false; check_system_svc "$SVC_PG" && pg_ok=true
    print_svc_status "PostgreSQL (system)" "$pg_ok"

    # Redis (check port, may be system or docker)
    redis_ok=false
    if command -v redis-cli &>/dev/null && redis-cli -h localhost -p 6379 ping &>/dev/null 2>&1; then
        redis_ok=true
    fi
    print_svc_status "Redis" "$redis_ok"

    # Qdrant (check HTTP)
    qdrant_ok=false
    if curl -sf http://localhost:6333/healthz &>/dev/null 2>&1; then
        qdrant_ok=true
    fi
    print_svc_status "Qdrant" "$qdrant_ok"

    # Feishu Bot
    bot_ok=false; check_user_svc "$SVC_BOT" && bot_ok=true
    print_svc_status "Feishu Bot (user)" "$bot_ok"
    if [[ "$bot_ok" == "true" ]]; then
        pid=$(systemctl --user show "$SVC_BOT" --property=MainPID --value 2>/dev/null)
        uptime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
        echo -e "  ${DIM}  PID=$pid  uptime=$uptime${NC}"
    fi

    echo ""

    # Summary
    if [[ "$pg_ok" == "true" && "$bot_ok" == "true" ]]; then
        ok "Core services running."
    else
        warn "Some services are not running."
    fi

    [[ "$redis_ok" == "false" ]] && echo -e "  ${DIM}Redis not available — checkpoint & cache disabled${NC}"
    [[ "$qdrant_ok" == "false" ]] && echo -e "  ${DIM}Qdrant not available — using in-memory fallback${NC}"
    echo ""
    exit 0
fi

# ── Logs mode ─────────────────────────────────────────────────────
if [[ "$MODE" == "logs" ]]; then
    journalctl --user -u "$SVC_BOT" -n "$LOG_LINES" --no-pager
    exit 0
fi

# ── Stop mode ─────────────────────────────────────────────────────
if [[ "$MODE" == "stop" ]]; then
    info "Stopping feishu bot..."
    systemctl --user stop "$SVC_BOT" 2>/dev/null && ok "Feishu bot stopped." || warn "Feishu bot was not running."
    exit 0
fi

# ── Restart mode ──────────────────────────────────────────────────
if [[ "$MODE" == "restart" ]]; then
    info "Restarting feishu bot..."
    systemctl --user restart "$SVC_BOT" 2>/dev/null
    sleep 2
    if check_user_svc "$SVC_BOT"; then
        pid=$(systemctl --user show "$SVC_BOT" --property=MainPID --value 2>/dev/null)
        ok "Feishu bot restarted (PID=$pid)."
    else
        err "Feishu bot failed to start. Check: ./scripts/setup.sh --logs"
    fi
    exit 0
fi

# ── Startup: check services & launch ─────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       mul-agent  一键启动                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# 0. Ensure venv & package (must be before DB migration)
if [[ ! -f "$PYTHON" ]]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
if ! "$PYTHON" -c "import cli.runner" &>/dev/null 2>&1; then
    info "Installing mul-agent..."
    "$PIP" install -e "$PROJECT_ROOT[cli]" --quiet
fi

# 1. Check PostgreSQL
printf "  %-20s" "PostgreSQL:"
if check_system_svc "$SVC_PG"; then
    echo -e "${GREEN}running${NC}"
else
    echo -e "${YELLOW}starting...${NC}"
    sudo systemctl start "$SVC_PG" 2>/dev/null || true
    sleep 1
    if check_system_svc "$SVC_PG"; then
        echo -e "  %-20s${GREEN}started${NC}\n" ""
    else
        err "PostgreSQL failed to start."
    fi
fi

# 2. Check Redis (optional)
printf "  %-20s" "Redis:"
if command -v redis-cli &>/dev/null && redis-cli -h localhost -p 6379 ping &>/dev/null 2>&1; then
    echo -e "${GREEN}running${NC}"
else
    echo -e "${DIM}not available (optional)${NC}"
fi

# 3. Check Qdrant (optional)
printf "  %-20s" "Qdrant:"
if curl -sf http://localhost:6333/healthz &>/dev/null 2>&1; then
    echo -e "${GREEN}running${NC}"
else
    echo -e "${DIM}not available (optional)${NC}"
fi

# 4. Feishu Bot
printf "  %-20s" "Feishu Bot:"
if check_user_svc "$SVC_BOT"; then
    pid=$(systemctl --user show "$SVC_BOT" --property=MainPID --value 2>/dev/null)
    echo -e "${GREEN}running${NC} ${DIM}(PID=$pid)${NC}"
else
    echo -e "${YELLOW}starting...${NC}"
    systemctl --user start "$SVC_BOT" 2>/dev/null
    sleep 2
    if check_user_svc "$SVC_BOT"; then
        pid=$(systemctl --user show "$SVC_BOT" --property=MainPID --value 2>/dev/null)
        printf "  %-20s${GREEN}started${NC} ${DIM}(PID=$pid)${NC}\n" ""
    else
        warn "Feishu bot failed to start. Check: ./scripts/setup.sh --logs"
    fi
fi

# 5. Database schema
info "Checking database schema..."
if [[ -f "$PROJECT_ROOT/alembic.ini" ]]; then
    VERSIONS_DIR="$PROJECT_ROOT/alembic/versions"
    if [[ -d "$VERSIONS_DIR" ]] && ls "$VERSIONS_DIR"/*.py &>/dev/null 2>&1; then
        cd "$PROJECT_ROOT"
        "$VENV_DIR/bin/alembic" upgrade head 2>/dev/null && ok "Database migrated." || true
    else
        "$PYTHON" -c "
import asyncio
from common.config import get_settings
from common.db import create_engine, Base
from models.trace import TaskTrace, SubtaskTrace
from models.feedback import Feedback

async def init():
    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

asyncio.run(init())
" 2>/dev/null && ok "Database tables ready." || true
    fi
fi

echo ""
ok "All services ready."
echo ""

# ── Infra-only mode ──────────────────────────────────────────────
if [[ "$MODE" == "infra" ]]; then
    info "Infrastructure-only mode. To launch CLI:"
    echo "  $VENV_DIR/bin/mulagent"
    echo "  $VENV_DIR/bin/mulagent --headless"
    exit 0
fi

# ── Launch CLI ────────────────────────────────────────────────────
info "Launching mul-agent CLI..."
echo ""
exec "$VENV_DIR/bin/mulagent" "${CLI_ARGS[@]}"
