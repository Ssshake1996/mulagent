<#
.SYNOPSIS
    mul-agent Windows 一键安装与启动脚本

.DESCRIPTION
    在 Windows 系统上安装依赖、可选安装数据库、启动 CLI。

.EXAMPLE
    .\scripts\setup.ps1                     # 安装 + 启动 TUI
    .\scripts\setup.ps1 -Headless           # 安装 + Headless REPL
    .\scripts\setup.ps1 -c "查天气"         # 单次执行
    .\scripts\setup.ps1 -Status             # 检查服务状态
    .\scripts\setup.ps1 -WithDB             # 安装时包含数据库
#>

param(
    [switch]$Status,
    [switch]$Infra,
    [switch]$Headless,
    [switch]$WithDB,
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
    $pad = $Name.PadRight(16)
    if ($Ok) {
        Write-Host "  $pad" -NoNewline; Write-Host "running" -ForegroundColor Green
    } else {
        Write-Host "  $pad" -NoNewline; Write-Host "not available" -ForegroundColor DarkGray
    }
}

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

function Ask-YesNo {
    param($Prompt)
    $answer = Read-Host "$Prompt [y/N]"
    return $answer -match '^[Yy]$'
}

# ── Help ─────────────────────────────────────────────────────
if ($Help) {
    @"

Usage: .\scripts\setup.ps1 [OPTIONS]

Options:
  -Status       Show service status
  -Infra        Check infrastructure only, don't launch CLI
  -WithDB       Install databases (PostgreSQL/Redis/Qdrant) during setup
  -Headless     Plain REPL instead of TUI
  -Model ID     Override default LLM model
  -Config PATH  Custom config file
  -Session ID   Resume a specific session
  -c COMMAND    Execute single command and exit
  -Help         Show this help

Examples:
  .\scripts\setup.ps1                    # Install + launch TUI
  .\scripts\setup.ps1 -WithDB           # Install with databases
  .\scripts\setup.ps1 -Headless         # Install + headless REPL
  .\scripts\setup.ps1 -c "帮我查天气"   # Single command

"@
    exit 0
}

# ── Status mode ──────────────────────────────────────────────
if ($Status) {
    Write-Host ""
    Write-Host "  mul-agent service status" -ForegroundColor Cyan
    Write-Host ""

    $pgOk = Test-TcpPort "localhost" 5432
    Write-SvcStatus "PostgreSQL" $pgOk

    $redisOk = Test-TcpPort "localhost" 6379
    Write-SvcStatus "Redis" $redisOk

    $qdrantOk = Test-HttpEndpoint "http://localhost:6333/healthz"
    Write-SvcStatus "Qdrant" $qdrantOk

    $apiOk = Test-HttpEndpoint "http://localhost:8000/api/v1/health"
    Write-SvcStatus "API Server" $apiOk

    Write-Host ""
    if (-not $redisOk) { Write-Host "  Redis not available — checkpoint & cache disabled" -ForegroundColor DarkGray }
    if (-not $qdrantOk) { Write-Host "  Qdrant not available — using in-memory fallback" -ForegroundColor DarkGray }
    Write-Host ""
    exit 0
}

# ══════════════════════════════════════════════════════════════
# Startup: install & launch
# ══════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  +----------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "  |         mul-agent  Windows 一键安装与启动                  |" -ForegroundColor Cyan
Write-Host "  +----------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Python ───────────────────────────────────────────
Write-Info "Step 1/4: Checking Python environment..."

$sysPython = Get-Command python -ErrorAction SilentlyContinue
if (-not $sysPython) {
    Write-Err "Python not found. Please install Python 3.10+ from https://python.org"
    Write-Host "  Ensure 'Add Python to PATH' is checked during installation." -ForegroundColor Yellow
    exit 1
}

$pyVer = & python --version 2>&1
Write-Host "  Python version:  " -NoNewline; Write-Host "$pyVer" -ForegroundColor Green

if (-not (Test-Path $Python)) {
    Write-Info "Creating virtual environment..."
    & python -m venv $VenvDir
    if (-not (Test-Path $Python)) {
        Write-Err "Failed to create venv at $VenvDir"
        exit 1
    }
    Write-OK "Virtual environment created."
} else {
    Write-OK "Virtual environment exists."
}

# ── Step 2: Install package ──────────────────────────────────
Write-Info "Step 2/4: Installing mul-agent..."

$installed = $false
try {
    & $Python -c "import cli.runner" 2>$null
    $installed = $true
} catch {}

if (-not $installed) {
    Write-Info "Installing (this may take a minute)..."
    & $Pip install -e "$ProjectRoot[cli]" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Trying without [cli] extras..."
        & $Pip install -e "$ProjectRoot" --quiet
    }
    Write-OK "mul-agent installed."
} else {
    Write-OK "mul-agent already installed."
}

# ── Step 3: Database selection ───────────────────────────────
Write-Info "Step 3/4: Database configuration..."
Write-Host ""

$pgOk = Test-TcpPort "localhost" 5432
$redisOk = Test-TcpPort "localhost" 6379
$qdrantOk = Test-HttpEndpoint "http://localhost:6333/healthz"

$installPg = $false
$installRedis = $false
$installQdrant = $false

if ($WithDB) {
    $installPg = $true
    $installRedis = $true
    $installQdrant = $true
} else {
    Write-Host "  数据库组件（全部可选，不安装也能正常使用核心功能）：" -ForegroundColor White
    Write-Host ""

    # PostgreSQL
    $pad = "PostgreSQL:".PadRight(16)
    Write-Host "  $pad" -NoNewline
    if ($pgOk) {
        Write-Host "already running" -ForegroundColor Green
    } else {
        Write-Host "not running" -ForegroundColor DarkGray
        if (Ask-YesNo "    Install PostgreSQL? (用于任务追踪与反馈存储)") {
            $installPg = $true
        }
    }

    # Redis
    $pad = "Redis:".PadRight(16)
    Write-Host "  $pad" -NoNewline
    if ($redisOk) {
        Write-Host "already running" -ForegroundColor Green
    } else {
        Write-Host "not running" -ForegroundColor DarkGray
        if (Ask-YesNo "    Install Redis? (用于缓存、检查点、幂等键)") {
            $installRedis = $true
        }
    }

    # Qdrant
    $pad = "Qdrant:".PadRight(16)
    Write-Host "  $pad" -NoNewline
    if ($qdrantOk) {
        Write-Host "already running" -ForegroundColor Green
    } else {
        Write-Host "not running" -ForegroundColor DarkGray
        if (Ask-YesNo "    Install Qdrant? (用于向量存储、经验库、知识RAG)") {
            $installQdrant = $true
        }
    }
}

# Install via Docker if requested
$dockerAvailable = $null -ne (Get-Command docker -ErrorAction SilentlyContinue)

if (($installPg -or $installRedis -or $installQdrant) -and -not $dockerAvailable) {
    Write-Warn "Docker not found. Database installation requires Docker Desktop."
    Write-Host "    Download: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
} else {
    if ($installPg -and -not $pgOk) {
        Write-Info "Starting PostgreSQL via Docker..."
        & docker run -d --name mulagent-postgres -p 5432:5432 `
            -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=mulagent `
            postgres:16 2>$null
        Start-Sleep -Seconds 3
        $pgOk = Test-TcpPort "localhost" 5432
        if ($pgOk) { Write-OK "PostgreSQL started." } else { Write-Warn "PostgreSQL may still be starting." }
    }

    if ($installRedis -and -not $redisOk) {
        Write-Info "Starting Redis via Docker..."
        & docker run -d --name mulagent-redis -p 6379:6379 redis:7-alpine 2>$null
        Start-Sleep -Seconds 2
        $redisOk = Test-TcpPort "localhost" 6379
        if ($redisOk) { Write-OK "Redis started." } else { Write-Warn "Redis may still be starting." }
    }

    if ($installQdrant -and -not $qdrantOk) {
        Write-Info "Starting Qdrant via Docker..."
        $qdrantStorage = Join-Path $ProjectRoot "data\qdrant_storage"
        & docker run -d --name mulagent-qdrant -p 6333:6333 -p 6334:6334 `
            -v "${qdrantStorage}:/qdrant/storage" qdrant/qdrant:latest 2>$null
        Start-Sleep -Seconds 3
        $qdrantOk = Test-HttpEndpoint "http://localhost:6333/healthz"
        if ($qdrantOk) { Write-OK "Qdrant started." } else { Write-Warn "Qdrant may still be starting." }
    }
}

# Database migration
$alembicIni = Join-Path $ProjectRoot "alembic.ini"
if ($pgOk -and (Test-Path $alembicIni)) {
    Write-Info "Running database migration..."
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

Write-Host ""

# ── Step 4: Summary ──────────────────────────────────────────
Write-Info "Step 4/4: Environment summary"
Write-Host ""

$pad = "Python:".PadRight(16)
Write-Host "  $pad" -NoNewline; Write-Host "$pyVer" -ForegroundColor Green

$pad = "PostgreSQL:".PadRight(16)
Write-Host "  $pad" -NoNewline
if ($pgOk) { Write-Host "running" -ForegroundColor Green }
else { Write-Host "not installed (optional)" -ForegroundColor DarkGray }

$pad = "Redis:".PadRight(16)
Write-Host "  $pad" -NoNewline
if ($redisOk) { Write-Host "running" -ForegroundColor Green }
else { Write-Host "not installed (optional)" -ForegroundColor DarkGray }

$pad = "Qdrant:".PadRight(16)
Write-Host "  $pad" -NoNewline
if ($qdrantOk) { Write-Host "running" -ForegroundColor Green }
else { Write-Host "not installed (optional)" -ForegroundColor DarkGray }

# Check config
$configPath = Join-Path $ProjectRoot "config\settings.yaml"
$pad = "Config:".PadRight(16)
Write-Host "  $pad" -NoNewline
if (Test-Path $configPath) {
    Write-Host "found" -ForegroundColor Green
} else {
    Write-Host "not found" -ForegroundColor Yellow -NoNewline
    Write-Host " — run 'mulagent init' to configure"
}

Write-Host ""
Write-OK "Installation complete!"
Write-Host ""

# ── Infra-only mode ──────────────────────────────────────────
if ($Infra) {
    Write-Info "To launch CLI:"
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

if (Test-Path $Mulagent) {
    & $Mulagent @cliArgs
} else {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & $Python -m cli.main @cliArgs
}
