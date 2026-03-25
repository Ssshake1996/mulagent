<#
.SYNOPSIS
    mul-agent Windows 一键安装与启动脚本

.DESCRIPTION
    在 Windows 系统上安装依赖、检查基础设施、启动 CLI。
    等价于 Linux 上的 scripts/setup.sh。

.EXAMPLE
    .\scripts\setup.ps1                 # 安装 + 启动 TUI
    .\scripts\setup.ps1 --headless      # 安装 + 启动 headless REPL
    .\scripts\setup.ps1 -c "查天气"     # 安装 + 单次执行
    .\scripts\setup.ps1 --status        # 检查服务状态
    .\scripts\setup.ps1 --infra         # 仅检查基础设施
#>

param(
    [switch]$Status,
    [switch]$Infra,
    [switch]$Headless,
    [string]$Model,
    [string]$Config,
    [string]$Session,
    [Alias("c")][string]$Command,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ── Paths ────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$Mulagent = Join-Path $VenvDir "Scripts\mulagent.exe"

# ── Colors ───────────────────────────────────────────────────
function Write-OK     { param($msg) Write-Host "  [ OK ] " -ForegroundColor Green -NoNewline; Write-Host $msg }
function Write-Info   { param($msg) Write-Host "  [INFO] " -ForegroundColor Cyan -NoNewline; Write-Host $msg }
function Write-Warn   { param($msg) Write-Host "  [WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Write-Err    { param($msg) Write-Host "  [ERR ] " -ForegroundColor Red -NoNewline; Write-Host $msg }

function Write-SvcStatus {
    param($Name, $Ok)
    $pad = $Name.PadRight(20)
    if ($Ok) {
        Write-Host "  $pad" -NoNewline; Write-Host "running" -ForegroundColor Green
    } else {
        Write-Host "  $pad" -NoNewline; Write-Host "not available" -ForegroundColor DarkGray
    }
}

# ── Help ─────────────────────────────────────────────────────
if ($Help) {
    @"

Usage: .\scripts\setup.ps1 [OPTIONS]

Options:
  -Status       Show service status
  -Infra        Check infrastructure only, don't launch CLI
  -Headless     Plain REPL instead of TUI
  -Model ID     Override default LLM model
  -Config PATH  Custom config file
  -Session ID   Resume a specific session
  -c COMMAND    Execute single command and exit
  -Help         Show this help

Examples:
  .\scripts\setup.ps1                    # Install + launch TUI
  .\scripts\setup.ps1 -Headless          # Install + headless REPL
  .\scripts\setup.ps1 -c "帮我查天气"     # Single command

"@
    exit 0
}

# ── Service check helpers ────────────────────────────────────
function Test-TcpPort {
    param($Host_, $Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.ConnectAsync($Host_, $Port).Wait(1000) | Out-Null
        $result = $tcp.Connected
        $tcp.Close()
        return $result
    } catch {
        return $false
    }
}

function Test-HttpEndpoint {
    param($Url)
    try {
        $resp = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

# ── Status mode ──────────────────────────────────────────────
if ($Status) {
    Write-Host ""
    Write-Host "  mul-agent service status" -ForegroundColor Cyan
    Write-Host ""

    # PostgreSQL (port 5432)
    $pgOk = Test-TcpPort "localhost" 5432
    Write-SvcStatus "PostgreSQL" $pgOk

    # Redis (port 6379)
    $redisOk = Test-TcpPort "localhost" 6379
    Write-SvcStatus "Redis" $redisOk

    # Qdrant (HTTP 6333)
    $qdrantOk = Test-HttpEndpoint "http://localhost:6333/healthz"
    Write-SvcStatus "Qdrant" $qdrantOk

    # API server (port 8000)
    $apiOk = Test-HttpEndpoint "http://localhost:8000/api/v1/health"
    Write-SvcStatus "API Server" $apiOk

    Write-Host ""
    if ($pgOk) { Write-OK "PostgreSQL available." }
    if (-not $redisOk) { Write-Host "  Redis not available — checkpoint & cache disabled" -ForegroundColor DarkGray }
    if (-not $qdrantOk) { Write-Host "  Qdrant not available — using in-memory fallback" -ForegroundColor DarkGray }
    Write-Host ""
    exit 0
}

# ── Startup ──────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║       mul-agent  Windows 一键启动         ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 0. Check Python
$sysPython = Get-Command python -ErrorAction SilentlyContinue
if (-not $sysPython) {
    Write-Err "Python not found. Please install Python 3.10+ from https://python.org"
    Write-Host "  Ensure 'Add Python to PATH' is checked during installation." -ForegroundColor Yellow
    exit 1
}

$pyVer = & python --version 2>&1
Write-Info "System Python: $pyVer"

# 1. Create/check venv
if (-not (Test-Path $Python)) {
    Write-Info "Creating virtual environment..."
    & python -m venv $VenvDir
    if (-not (Test-Path $Python)) {
        Write-Err "Failed to create venv at $VenvDir"
        exit 1
    }
    Write-OK "Virtual environment created."
}

# 2. Install package
$installed = $false
try {
    & $Python -c "import cli.runner" 2>$null
    $installed = $true
} catch {}

if (-not $installed) {
    Write-Info "Installing mul-agent (this may take a minute)..."
    & $Pip install -e "$ProjectRoot[cli]" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "pip install had warnings. Trying without [cli] extras..."
        & $Pip install -e "$ProjectRoot" --quiet
    }
    Write-OK "mul-agent installed."
}

# 3. Check infrastructure
Write-Host ""

# PostgreSQL
$pgOk = Test-TcpPort "localhost" 5432
$pad = "PostgreSQL:".PadRight(20)
Write-Host "  $pad" -NoNewline
if ($pgOk) {
    Write-Host "running" -ForegroundColor Green
} else {
    Write-Host "not available (optional)" -ForegroundColor DarkGray
}

# Redis
$redisOk = Test-TcpPort "localhost" 6379
$pad = "Redis:".PadRight(20)
Write-Host "  $pad" -NoNewline
if ($redisOk) {
    Write-Host "running" -ForegroundColor Green
} else {
    Write-Host "not available (optional)" -ForegroundColor DarkGray
}

# Qdrant
$qdrantOk = Test-HttpEndpoint "http://localhost:6333/healthz"
$pad = "Qdrant:".PadRight(20)
Write-Host "  $pad" -NoNewline
if ($qdrantOk) {
    Write-Host "running" -ForegroundColor Green
} else {
    Write-Host "not available (optional)" -ForegroundColor DarkGray
}

Write-Host ""

# 4. Database migration (if PostgreSQL available and alembic exists)
$alembicIni = Join-Path $ProjectRoot "alembic.ini"
if ($pgOk -and (Test-Path $alembicIni)) {
    Write-Info "Checking database schema..."
    $alembic = Join-Path $VenvDir "Scripts\alembic.exe"
    if (Test-Path $alembic) {
        Push-Location $ProjectRoot
        try {
            & $alembic upgrade head 2>$null
            Write-OK "Database migrated."
        } catch {
            Write-Warn "Database migration skipped."
        }
        Pop-Location
    }
}

# 5. Check config
$configPath = Join-Path $ProjectRoot "config\settings.yaml"
if (-not (Test-Path $configPath)) {
    Write-Warn "config\settings.yaml not found."
    Write-Host "  Run 'mulagent init' to create configuration." -ForegroundColor Yellow
    Write-Host ""
}

Write-OK "Setup complete."
Write-Host ""

# ── Infra-only mode ──────────────────────────────────────────
if ($Infra) {
    Write-Info "Infrastructure-only mode. To launch CLI:"
    Write-Host "  $Mulagent"
    Write-Host "  $Mulagent --headless"
    exit 0
}

# ── Launch CLI ───────────────────────────────────────────────
$cliArgs = @()

if ($Headless) { $cliArgs += "--headless" }
if ($Model)    { $cliArgs += "--model", $Model }
if ($Config)   { $cliArgs += "--config", $Config }
if ($Session)  { $cliArgs += "--session", $Session }
if ($Command)  { $cliArgs += "-c", $Command }

Write-Info "Launching mul-agent CLI..."
Write-Host ""

# Use mulagent entry point if available, otherwise python -m
if (Test-Path $Mulagent) {
    & $Mulagent @cliArgs
} else {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & $Python -m cli.main @cliArgs
}
