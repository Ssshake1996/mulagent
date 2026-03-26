#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  mul-agent Linux 一键安装与启动脚本
#
#  用法:
#    ./scripts/setup.sh                 # 安装 + 启动 CLI (TUI)
#    ./scripts/setup.sh --headless      # 安装 + 启动 CLI (headless)
#    ./scripts/setup.sh -c "查天气"     # 安装 + 单次执行
#    ./scripts/setup.sh --status        # 查看服务状态
#    ./scripts/setup.sh --restart       # 重启应用服务
#    ./scripts/setup.sh --stop          # 停止应用服务
#    ./scripts/setup.sh --logs [N]      # 查看最近日志
#    ./scripts/setup.sh --infra         # 仅检查基础设施
#    ./scripts/setup.sh --with-db       # 安装时包含数据库
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# systemd service names
SVC_PG="postgresql@16-main"
SVC_BOT="mulagent-feishu"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; }

# ── Parse arguments ───────────────────────────────────────────────
MODE="cli"
CLI_ARGS=()
WITH_DB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)       MODE="status";  shift ;;
        --restart)      MODE="restart"; shift ;;
        --stop)         MODE="stop";    shift ;;
        --infra)        MODE="infra";   shift ;;
        --with-db)      WITH_DB="yes";  shift ;;
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
Usage: ./scripts/setup.sh [OPTIONS]

Modes:
  (default)       Install + launch CLI (TUI)
  --status        Show all service status
  --restart       Restart feishu bot
  --stop          Stop feishu bot
  --logs [N]      Show last N lines of log (default 50)
  --infra         Check infrastructure only

Install options:
  --with-db       Install and configure databases (PostgreSQL/Redis/Qdrant)
                  Without this flag, only the core agent is installed.

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
        echo -e "${DIM}not available${NC}"
    fi
}

ask_yes_no() {
    local prompt="$1" default="${2:-n}"
    local yn
    if [[ "$default" == "y" ]]; then
        prompt="$prompt [Y/n] "
    else
        prompt="$prompt [y/N] "
    fi
    read -r -p "$prompt" yn
    yn="${yn:-$default}"
    [[ "$yn" =~ ^[Yy]$ ]]
}

# ── Status mode ───────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    echo ""
    echo -e "${CYAN}mul-agent service status${NC}"
    echo ""

    pg_ok=false; check_system_svc "$SVC_PG" && pg_ok=true
    print_svc_status "PostgreSQL" "$pg_ok"

    redis_ok=false
    if command -v redis-cli &>/dev/null && redis-cli -h localhost -p 6379 ping &>/dev/null 2>&1; then
        redis_ok=true
    fi
    print_svc_status "Redis" "$redis_ok"

    qdrant_ok=false
    if curl -sf http://localhost:6333/healthz &>/dev/null 2>&1; then
        qdrant_ok=true
    fi
    print_svc_status "Qdrant" "$qdrant_ok"

    bot_ok=false; check_user_svc "$SVC_BOT" && bot_ok=true
    print_svc_status "Feishu Bot" "$bot_ok"
    if [[ "$bot_ok" == "true" ]]; then
        pid=$(systemctl --user show "$SVC_BOT" --property=MainPID --value 2>/dev/null)
        uptime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
        echo -e "  ${DIM}  PID=$pid  uptime=$uptime${NC}"
    fi

    echo ""
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

# ══════════════════════════════════════════════════════════════════
# Startup: install & launch
# ══════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           mul-agent  Linux 一键安装与启动                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Python & venv ────────────────────────────────────────
info "Step 1/4: Checking Python environment..."

if ! command -v python3 &>/dev/null; then
    err "Python 3 not found. Please install Python 3.10+:"
    echo "    sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  Python version:  ${GREEN}$PY_VER${NC}"

if [[ ! -f "$PYTHON" ]]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    ok "Virtual environment exists."
fi

# ── Step 2: Install package ──────────────────────────────────────
info "Step 2/4: Installing mul-agent..."

if "$PYTHON" -c "import cli.runner" &>/dev/null 2>&1; then
    ok "mul-agent already installed."
else
    "$PIP" install -e "$PROJECT_ROOT[cli]" --quiet
    ok "mul-agent installed."
fi

# ── Step 3: Database selection ───────────────────────────────────
info "Step 3/4: Database configuration..."

INSTALL_PG=false
INSTALL_REDIS=false
INSTALL_QDRANT=false

# Check what's already running
PG_RUNNING=false; check_system_svc "$SVC_PG" && PG_RUNNING=true
REDIS_RUNNING=false
command -v redis-cli &>/dev/null && redis-cli -h localhost -p 6379 ping &>/dev/null 2>&1 && REDIS_RUNNING=true
QDRANT_RUNNING=false
curl -sf http://localhost:6333/healthz &>/dev/null 2>&1 && QDRANT_RUNNING=true

if [[ "$WITH_DB" == "yes" ]]; then
    # Non-interactive: install all databases
    INSTALL_PG=true
    INSTALL_REDIS=true
    INSTALL_QDRANT=true
elif [[ -t 0 ]]; then
    # Interactive: ask user
    echo ""
    echo -e "  ${BOLD}数据库组件（全部可选，不安装也能正常使用核心功能）：${NC}"
    echo ""

    # PostgreSQL
    printf "  %-16s" "PostgreSQL:"
    if [[ "$PG_RUNNING" == "true" ]]; then
        echo -e "${GREEN}already running${NC}"
    else
        echo -e "${DIM}not running${NC}"
        if ask_yes_no "    Install PostgreSQL? (用于任务追踪与反馈存储)"; then
            INSTALL_PG=true
        fi
    fi

    # Redis
    printf "  %-16s" "Redis:"
    if [[ "$REDIS_RUNNING" == "true" ]]; then
        echo -e "${GREEN}already running${NC}"
    else
        echo -e "${DIM}not running${NC}"
        if ask_yes_no "    Install Redis? (用于缓存、检查点、幂等键)"; then
            INSTALL_REDIS=true
        fi
    fi

    # Qdrant
    printf "  %-16s" "Qdrant:"
    if [[ "$QDRANT_RUNNING" == "true" ]]; then
        echo -e "${GREEN}already running${NC}"
    else
        echo -e "${DIM}not running${NC}"
        if ask_yes_no "    Install Qdrant? (用于向量存储、经验库、知识RAG)"; then
            INSTALL_QDRANT=true
        fi
    fi

    echo ""
else
    # Non-interactive without --with-db: skip database installation
    echo -e "  ${DIM}Non-interactive mode: skipping database setup.${NC}"
    echo -e "  ${DIM}Use --with-db flag to install databases automatically.${NC}"
fi

# Install databases if requested
if [[ "$INSTALL_PG" == "true" && "$PG_RUNNING" == "false" ]]; then
    info "Installing PostgreSQL..."
    if command -v apt &>/dev/null; then
        sudo apt update -qq && sudo apt install -y -qq postgresql postgresql-client >/dev/null 2>&1
        sudo systemctl enable --now postgresql 2>/dev/null || true
        ok "PostgreSQL installed and started."
        PG_RUNNING=true
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y -q postgresql-server postgresql >/dev/null 2>&1
        sudo postgresql-setup --initdb 2>/dev/null || true
        sudo systemctl enable --now postgresql 2>/dev/null || true
        ok "PostgreSQL installed and started."
        PG_RUNNING=true
    else
        warn "Cannot auto-install PostgreSQL (unsupported package manager)."
        echo "    Please install manually or use Docker: docker compose -f docker/docker-compose.yaml up -d postgres"
    fi
fi

if [[ "$INSTALL_REDIS" == "true" && "$REDIS_RUNNING" == "false" ]]; then
    info "Installing Redis..."
    if command -v apt &>/dev/null; then
        sudo apt install -y -qq redis-server >/dev/null 2>&1
        sudo systemctl enable --now redis-server 2>/dev/null || true
        ok "Redis installed and started."
        REDIS_RUNNING=true
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y -q redis >/dev/null 2>&1
        sudo systemctl enable --now redis 2>/dev/null || true
        ok "Redis installed and started."
        REDIS_RUNNING=true
    else
        warn "Cannot auto-install Redis. Use Docker: docker compose -f docker/docker-compose.yaml up -d redis"
    fi
fi

if [[ "$INSTALL_QDRANT" == "true" && "$QDRANT_RUNNING" == "false" ]]; then
    info "Installing Qdrant (via Docker)..."
    if command -v docker &>/dev/null; then
        docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
            -v "$PROJECT_ROOT/data/qdrant_storage:/qdrant/storage" \
            qdrant/qdrant:latest >/dev/null 2>&1 || true
        sleep 2
        if curl -sf http://localhost:6333/healthz &>/dev/null 2>&1; then
            ok "Qdrant started."
            QDRANT_RUNNING=true
        else
            warn "Qdrant container started but not yet responding. It may need a moment."
        fi
    else
        warn "Docker not found. Qdrant requires Docker."
        echo "    Install Docker: https://docs.docker.com/engine/install/"
    fi
fi

# Run database migration if PostgreSQL is available
if [[ "$PG_RUNNING" == "true" && -f "$PROJECT_ROOT/alembic.ini" ]]; then
    info "Running database migration..."
    VERSIONS_DIR="$PROJECT_ROOT/alembic/versions"
    if [[ -d "$VERSIONS_DIR" ]] && ls "$VERSIONS_DIR"/*.py &>/dev/null 2>&1; then
        cd "$PROJECT_ROOT"
        "$VENV_DIR/bin/alembic" upgrade head 2>/dev/null && ok "Database migrated." || warn "Migration skipped."
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
" 2>/dev/null && ok "Database tables ready." || warn "Database setup skipped."
    fi
fi

# ── Step 4: Summary ──────────────────────────────────────────────
info "Step 4/4: Environment summary"
echo ""
printf "  %-16s" "Python:"
echo -e "${GREEN}$PY_VER${NC}"
printf "  %-16s" "PostgreSQL:"
[[ "$PG_RUNNING" == "true" ]] && echo -e "${GREEN}running${NC}" || echo -e "${DIM}not installed (optional)${NC}"
printf "  %-16s" "Redis:"
[[ "$REDIS_RUNNING" == "true" ]] && echo -e "${GREEN}running${NC}" || echo -e "${DIM}not installed (optional)${NC}"
printf "  %-16s" "Qdrant:"
[[ "$QDRANT_RUNNING" == "true" ]] && echo -e "${GREEN}running${NC}" || echo -e "${DIM}not installed (optional)${NC}"

# Check config
CONFIG_PATH="$PROJECT_ROOT/config/settings.yaml"
printf "  %-16s" "Config:"
if [[ -f "$CONFIG_PATH" ]]; then
    echo -e "${GREEN}found${NC}"
else
    echo -e "${YELLOW}not found${NC} — run 'mulagent init' to configure"
fi

echo ""
ok "Installation complete!"
echo ""

# ── Infra-only mode ──────────────────────────────────────────────
if [[ "$MODE" == "infra" ]]; then
    info "To launch CLI:"
    echo "  $VENV_DIR/bin/mulagent"
    echo "  $VENV_DIR/bin/mulagent --headless"
    exit 0
fi

# ── Run init if no config (first install) ────────────────────────
CONFIG_PATH="$PROJECT_ROOT/config/settings.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    info "No config found. Running first-time setup..."
    echo ""
    "$VENV_DIR/bin/mulagent" init
    echo ""
    # First install: do not auto-launch CLI
    exit 0
fi

# ── Launch CLI (only on subsequent runs) ─────────────────────────
info "Launching mul-agent CLI..."
echo ""
exec "$VENV_DIR/bin/mulagent" "${CLI_ARGS[@]}"
