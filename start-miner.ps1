# Q Coin Miner Starter
# Usage: ./start-miner.ps1 [type] [network] [options]
# Types: cpu, gpu (default: cpu)
# Networks: mainnet, testnet, devnet (default: testnet)

param(
    [Parameter(Position=0)]
    [ValidateSet("cpu", "gpu")]
    [string]$Type = "cpu",
    
    [Parameter(Position=1)]
    [ValidateSet("mainnet", "testnet", "devnet")]
    [string]$Network = "testnet",
    
    [int]$Threads = 4,
    [string]$GethRpc = "http://localhost:8545",
    [string]$Etherbase = "",
    [switch]$Help
)

if ($Help) {
    Write-Host "Q Coin Miner Starter" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./start-miner.ps1 [type] [network] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Mining Types:" -ForegroundColor Yellow
    Write-Host "  cpu       - CPU Mining [DEFAULT]"
    Write-Host "  gpu       - GPU Mining (CUDA required)"
    Write-Host ""
    Write-Host "Networks:" -ForegroundColor Yellow
    Write-Host "  mainnet   - Q Coin Mainnet (Chain ID 73236)"
    Write-Host "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    Write-Host "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -threads <n>        Number of mining threads (default: 4)"
    Write-Host "  -gethRpc <url>      Geth RPC endpoint (default: http://localhost:8545)"
    Write-Host "  -etherbase <addr>   Mining reward address (auto-detected if empty)"
    Write-Host "  -help               Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./start-miner.ps1                       # Start CPU miner on testnet"
    Write-Host "  ./start-miner.ps1 gpu                   # Start GPU miner on testnet"
    Write-Host "  ./start-miner.ps1 cpu devnet            # Start CPU miner on devnet"
    Write-Host "  ./start-miner.ps1 cpu testnet -threads 8 # Start CPU miner with 8 threads"
    exit 0
}

# Build miner if it doesn't exist
$MinerPath = "quantum-miner\quantum-miner.exe"
if (-not (Test-Path $MinerPath)) {
    Write-Host "🔨 Building Q Coin Miner..." -ForegroundColor Yellow
    & .\build-linux.sh miner
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Miner build failed!" -ForegroundColor Red
        exit 1
    }
}

# Network configurations
$configs = @{
    "mainnet" = @{
        chainid = 73236
        name = "Q Coin Mainnet"
        description = "Production network with real Q Coin value"
    }
    "testnet" = @{
        chainid = 73235
        name = "Q Coin Testnet"
        description = "Testing network with test Q Coin"
    }
    "devnet" = @{
        chainid = 73234
        name = "Q Coin Dev Network"
        description = "Development network for testing"
    }
}

$config = $configs[$Network]
Write-Host "⛏️  Starting $Type Mining on $($config.name)" -ForegroundColor Cyan

# Get etherbase if not provided
if ($Etherbase -eq "") {
    Write-Host "🔍 Auto-detecting mining address..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' -ErrorAction Stop
        if ($response.result -and $response.result.Count -gt 0) {
            $Etherbase = $response.result[0]
            Write-Host "✅ Using account: $Etherbase" -ForegroundColor Green
        } else {
            Write-Host "⚠️  No accounts found. Creating new account..." -ForegroundColor Yellow
            $createResponse = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"personal_newAccount","params":[""],"id":1}' -ErrorAction Stop
            if ($createResponse.result) {
                $Etherbase = $createResponse.result
                Write-Host "✅ Created new account: $Etherbase" -ForegroundColor Green
            } else {
                Write-Host "❌ Failed to create account!" -ForegroundColor Red
                exit 1
            }
        }
    } catch {
        Write-Host "❌ Failed to connect to Geth RPC at $GethRpc" -ForegroundColor Red
        Write-Host "   Make sure Geth is running first!" -ForegroundColor Yellow
        exit 1
    }
}

# Prepare miner arguments
$minerArgs = @(
    "--rpc-url", $GethRpc,
    "--etherbase", $Etherbase,
    "--threads", $Threads
)

if ($Type -eq "gpu") {
    $minerArgs += "--gpu"
    Write-Host "🎮 GPU Mining enabled" -ForegroundColor Yellow
} else {
    Write-Host "🖥️  CPU Mining enabled" -ForegroundColor Yellow
}

Write-Host "🌐 Network: $($config.name)" -ForegroundColor White
Write-Host "🔗 Chain ID: $($config.chainid)" -ForegroundColor White
Write-Host "📡 Geth RPC: $GethRpc" -ForegroundColor White
Write-Host "💰 Mining Address: $Etherbase" -ForegroundColor White
Write-Host "🧵 Threads: $Threads" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Starting Q Coin $Type miner..." -ForegroundColor Green

# Start miner
& $MinerPath @minerArgs 