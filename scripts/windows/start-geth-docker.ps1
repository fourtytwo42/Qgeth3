# Q Geth Docker Node Starter
# Usage: ./start-geth-docker.ps1 [network] [options]
# Networks: testnet, devnet (default: testnet)
# Options: -Mining (enable mining mode), -Build (rebuild containers)

param(
    [Parameter(Position=0)]
    [ValidateSet("testnet", "devnet")]
    [string]$Network = "testnet",
    
    [switch]$Mining,
    [switch]$Build,
    [switch]$Foreground,
    [switch]$Help,
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if ($Help) {
    Write-Host "Q Geth Docker Node Starter" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./start-geth-docker.ps1 [network] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Networks:" -ForegroundColor Yellow
    Write-Host "  testnet   - Q Geth Testnet (Chain ID 73235) [DEFAULT]"
    Write-Host "  devnet    - Q Geth Dev Network (Chain ID 73234)"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Mining       - Enable mining mode (uses qgeth-miner container)"
    Write-Host "  -Build        - Rebuild Docker containers before starting"
    Write-Host "  -Foreground   - Run in foreground (see logs directly)"
    Write-Host "  -Help         - Show this help message"
    Write-Host ""
    Write-Host "Docker Features:" -ForegroundColor Green
    Write-Host "  🐳 Cross-Platform: Works on Linux, Windows, macOS"
    Write-Host "  🔒 Isolated Environment: Clean, secure container deployment"
    Write-Host "  📊 Health Checks: Built-in container health monitoring"
    Write-Host "  💾 Persistent Data: Blockchain data survives container restarts"
    Write-Host "  🔗 MetaMask Ready: HTTP RPC (8545) and WebSocket (8546) exposed"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./start-geth-docker.ps1                    # Start testnet node"
    Write-Host "  ./start-geth-docker.ps1 devnet -Mining     # Start dev node with mining"
    Write-Host "  ./start-geth-docker.ps1 -Build             # Rebuild and start testnet"
    Write-Host "  ./start-geth-docker.ps1 -Foreground        # Start with visible logs"
    Write-Host ""
    Write-Host "Management Commands:" -ForegroundColor Green
    Write-Host "  docker-compose ps                          # Check container status"
    Write-Host "  docker-compose logs -f qgeth-testnet       # View logs"
    Write-Host "  docker-compose stop                        # Stop containers"
    Write-Host "  docker-compose down                        # Stop and remove containers"
    exit 0
}

# Check if Docker is available
function Test-DockerAvailable {
    try {
        $dockerVersion = docker --version 2>$null
        if (-not $dockerVersion) {
            Write-Host "ERROR: Docker is not installed!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Install Docker Desktop:" -ForegroundColor Yellow
            Write-Host "  1. Download from: https://docs.docker.com/desktop/install/windows/"
            Write-Host "  2. Install Docker Desktop with WSL2 backend"
            Write-Host "  3. Start Docker Desktop"
            Write-Host "  4. Restart PowerShell after installation"
            exit 1
        }
    }
    catch {
        Write-Host "ERROR: Docker is not installed or not in PATH!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Install Docker Desktop:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://docs.docker.com/desktop/install/windows/"
        Write-Host "  2. Install Docker Desktop with WSL2 backend"
        Write-Host "  3. Start Docker Desktop"
        Write-Host "  4. Restart PowerShell after installation"
        exit 1
    }

    try {
        $composeVersion = docker-compose --version 2>$null
        if (-not $composeVersion) {
            Write-Host "ERROR: Docker Compose is not installed!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Docker Compose is usually included with Docker Desktop." -ForegroundColor Yellow
            Write-Host "If using Docker Engine without Desktop, install Docker Compose separately."
            exit 1
        }
    }
    catch {
        Write-Host "ERROR: Docker Compose is not available!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Docker Compose is usually included with Docker Desktop." -ForegroundColor Yellow
        Write-Host "If using Docker Engine without Desktop, install Docker Compose separately."
        exit 1
    }

    # Check if Docker daemon is running
    try {
        docker info >$null 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Docker daemon is not running!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Start Docker:" -ForegroundColor Yellow
            Write-Host "  1. Start Docker Desktop application"
            Write-Host "  2. Wait for Docker to be ready (green status icon)"
            Write-Host "  3. Or run: Start-Service docker (if using Docker Engine)"
            exit 1
        }
    }
    catch {
        Write-Host "ERROR: Cannot connect to Docker daemon!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Start Docker:" -ForegroundColor Yellow
        Write-Host "  1. Start Docker Desktop application"
        Write-Host "  2. Wait for Docker to be ready (green status icon)"
        exit 1
    }
}

# Navigate to project root (where docker-compose.yml is located)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "../..")

Write-Host "[DEBUG] Project root: $projectRoot" -ForegroundColor Cyan

$dockerComposeFile = Join-Path $projectRoot "docker-compose.yml"
if (-not (Test-Path $dockerComposeFile)) {
    Write-Host "ERROR: docker-compose.yml not found in project root!" -ForegroundColor Red
    Write-Host "Expected location: $dockerComposeFile" -ForegroundColor Yellow
    exit 1
}

Set-Location $projectRoot

# Check Docker installation
Test-DockerAvailable

# Determine container and port configuration
$configs = @{
    "testnet" = @{
        chainid = 73235
        name = "Q Geth Testnet"
        mining_container = "qgeth-miner"
        mining_service = "qgeth-miner"
        mining_http_port = 8547
        mining_ws_port = 8548
        mining_p2p_port = 30304
        mining_profile = "--profile mining"
        normal_container = "qgeth-testnet"
        normal_service = "qgeth-testnet"
        normal_http_port = 8545
        normal_ws_port = 8546
        normal_p2p_port = 30303
        normal_profile = ""
    }
    "devnet" = @{
        chainid = 73234
        name = "Q Geth Dev Network"
        container = "qgeth-dev"
        service = "qgeth-dev"
        http_port = 8549
        ws_port = 8550
        p2p_port = 30305
        profile = "--profile dev"
    }
}

$config = $configs[$Network]

# Configure specific container based on network and mining option
if ($Network -eq "testnet") {
    if ($Mining) {
        $containerName = $config.mining_container
        $serviceName = $config.mining_service
        $httpPort = $config.mining_http_port
        $wsPort = $config.mining_ws_port
        $p2pPort = $config.mining_p2p_port
        $profileArg = $config.mining_profile
    } else {
        $containerName = $config.normal_container
        $serviceName = $config.normal_service
        $httpPort = $config.normal_http_port
        $wsPort = $config.normal_ws_port
        $p2pPort = $config.normal_p2p_port
        $profileArg = $config.normal_profile
    }
} elseif ($Network -eq "devnet") {
    $containerName = $config.container
    $serviceName = $config.service
    $httpPort = $config.http_port
    $wsPort = $config.ws_port
    $p2pPort = $config.p2p_port
    $profileArg = $config.profile
    if ($Mining) {
        Write-Host "[INFO] Dev network has mining enabled by default" -ForegroundColor Yellow
    }
}

Write-Host "[START] Starting $($config.name) (Chain ID: $($config.chainid)) with Docker" -ForegroundColor Cyan

# Build containers if requested or if they don't exist
if ($Build) {
    Write-Host "[BUILD] Building Docker containers..." -ForegroundColor Yellow
    docker-compose build --no-cache
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Docker containers built successfully" -ForegroundColor Green
} else {
    # Check if image exists
    $imageExists = docker images qgeth:latest --format "{{.Repository}}" 2>$null
    if (-not $imageExists) {
        Write-Host "[BUILD] Q Geth Docker image not found, building..." -ForegroundColor Yellow
        docker-compose build
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Docker build failed!" -ForegroundColor Red
            exit 1
        }
    }
}

# Stop any existing containers
Write-Host "[CLEANUP] Stopping existing containers..." -ForegroundColor Yellow
docker-compose down --remove-orphans >$null 2>&1

# Prepare docker-compose command
$composeCmd = "docker-compose"
if ($profileArg) {
    $composeCmd += " $profileArg"
}

if ($Foreground) {
    $composeCmd += " up $serviceName"
    Write-Host "[INFO] Starting container in foreground (Ctrl+C to stop)..." -ForegroundColor Yellow
} else {
    $composeCmd += " up -d $serviceName"
    Write-Host "[INFO] Starting container in background..." -ForegroundColor Yellow
}

# Display configuration
Write-Host "[NETWORK] Network: $($config.name)" -ForegroundColor White
Write-Host "[CHAIN] Chain ID: $($config.chainid)" -ForegroundColor White
Write-Host "[CONTAINER] Container: $containerName" -ForegroundColor White
Write-Host "[HTTP] HTTP RPC: http://localhost:$httpPort" -ForegroundColor White
Write-Host "[WS] WebSocket: ws://localhost:$wsPort" -ForegroundColor White
Write-Host "[P2P] P2P Port: $p2pPort" -ForegroundColor White
if ($Mining -or $Network -eq "devnet") {
    Write-Host "[MINING] Mining: Enabled" -ForegroundColor White
} else {
    Write-Host "[MINING] Mining Interface: Available for external miners" -ForegroundColor White
}
Write-Host ""

# Execute the docker-compose command
Write-Host "Executing: $composeCmd" -ForegroundColor Gray
Invoke-Expression $composeCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to start Docker container!" -ForegroundColor Red
    exit 1
}

if (-not $Foreground) {
    Write-Host "[SUCCESS] Q Geth container started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "[INFO] Container Management:" -ForegroundColor Yellow
    Write-Host "  Check status: docker-compose ps"
    Write-Host "  View logs:    docker-compose logs -f $serviceName"
    Write-Host "  Stop:         docker-compose stop $serviceName"
    Write-Host "  Remove:       docker-compose down"
    Write-Host ""
    Write-Host "[INFO] MetaMask Connection:" -ForegroundColor Yellow
    Write-Host "  Network: Custom RPC"
    Write-Host "  RPC URL: http://localhost:$httpPort"
    Write-Host "  Chain ID: $($config.chainid)"
    Write-Host "  Currency: QGC"
    Write-Host ""
    Write-Host "[INFO] Mining Connection:" -ForegroundColor Yellow
    Write-Host "  HTTP RPC: http://localhost:$httpPort"
    Write-Host "  WebSocket: ws://localhost:$wsPort"
    Write-Host ""
    
    # Wait a moment for container to start, then check health
    Start-Sleep 3
    Write-Host "[HEALTH] Checking container health..." -ForegroundColor Cyan
    
    $containerStatus = docker-compose ps --format "table {{.Name}}\t{{.Status}}" 2>$null
    if ($containerStatus -match "Up.*healthy") {
        Write-Host "[HEALTH] ✅ Container is healthy and ready!" -ForegroundColor Green
    } elseif ($containerStatus -match "Up.*starting") {
        Write-Host "[HEALTH] 🔄 Container is starting... (health check pending)" -ForegroundColor Yellow
    } else {
        Write-Host "[HEALTH] ❌ Container may have issues. Check logs: docker-compose logs $serviceName" -ForegroundColor Red
    }
    
    # Test API connectivity
    Write-Host "[TEST] Testing API connectivity..." -ForegroundColor Cyan
    Start-Sleep 2
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$httpPort" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response) {
            Write-Host "[TEST] ✅ HTTP RPC API is accessible" -ForegroundColor Green
        } else {
            Write-Host "[TEST] ⏳ HTTP RPC API not yet ready (normal during startup)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "[TEST] ⏳ HTTP RPC API not yet ready (normal during startup)" -ForegroundColor Yellow
    }
}

Write-Host "[COMPLETE] Q Geth Docker deployment complete!" -ForegroundColor Green 