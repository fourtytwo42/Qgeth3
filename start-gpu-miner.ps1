# Q Coin GPU Miner
# Starts the Q Coin GPU miner for mining Q Coins with CUDA
# Usage: .\start-gpu-miner.ps1 [options]

param(
    [string]$address = "",                     # Mining address (required)
    [string]$rpcurl = "http://127.0.0.1:8545", # RPC URL
    [int]$gpus = 0,                            # Number of GPUs (0 = auto)
    [switch]$testnet = $false,                 # Connect to testnet (default: mainnet)
    [switch]$help = $false                     # Show help
)

# Show help
if ($help -or $address -eq "") {
    Write-Host "Q Coin GPU Miner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start-gpu-miner.ps1 -address <mining_address> [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Required:" -ForegroundColor Green
    Write-Host "  -address <addr>    Your Q Coin address to receive mining rewards"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -rpcurl <url>      RPC URL (default: http://127.0.0.1:8545)"
    Write-Host "  -gpus <num>        Number of GPUs to use (default: auto-detect)"
    Write-Host "  -testnet           Connect to testnet instead of mainnet"
    Write-Host "  -help              Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start-gpu-miner.ps1 -address 0x123..."
    Write-Host "  .\start-gpu-miner.ps1 -address 0x123... -gpus 2"
    Write-Host "  .\start-gpu-miner.ps1 -address 0x123... -testnet"
    Write-Host ""
    Write-Host "Q Coin GPU Mining Details:" -ForegroundColor Magenta
    Write-Host "  Algorithm: QMPoW (Quantum Proof of Work)"
    Write-Host "  GPU Support: NVIDIA CUDA"
    Write-Host "  Block Time: 12 seconds"
    Write-Host "  Difficulty: ASERT-Q (Adaptive)"
    Write-Host ""
    Write-Host "Requirements:" -ForegroundColor Red
    Write-Host "  - NVIDIA GPU with CUDA support"
    Write-Host "  - CUDA drivers installed"
    Write-Host ""
    if ($address -eq "") {
        Write-Host "ERROR: Mining address is required!" -ForegroundColor Red
        Write-Host "Get a Q Coin address from your wallet or geth console" -ForegroundColor Yellow
    }
    exit 0
}

# Find the latest quantum-miner release
Write-Host "Q Coin GPU Miner - Starting" -ForegroundColor Cyan
Write-Host ""

$MinerReleaseDir = Get-ChildItem -Path "releases\quantum-miner-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if (-not $MinerReleaseDir) {
    Write-Host "ERROR: No quantum-miner release found!" -ForegroundColor Red
    Write-Host "Please run: .\build-release.ps1 miner" -ForegroundColor Yellow
    exit 1
}

$MinerExecutable = "$($MinerReleaseDir.FullName)\quantum-miner.exe"
Write-Host "Using miner from: $($MinerReleaseDir.Name)" -ForegroundColor Green

# Determine network
if ($testnet) {
    $networkName = "Q Coin Testnet"
    $chainId = "73235"
} else {
    $networkName = "Q Coin Mainnet"
    $chainId = "73236"
}

# Auto-detect GPUs if not specified
if ($gpus -eq 0) {
    # Try to detect NVIDIA GPUs
    try {
        $nvidiaOutput = & nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaOutput) {
            $gpus = [int]$nvidiaOutput.Split("`n")[0].Trim()
            Write-Host "Auto-detected NVIDIA GPUs: $gpus" -ForegroundColor Green
        } else {
            $gpus = 1
            Write-Host "Could not detect GPUs, defaulting to 1" -ForegroundColor Yellow
        }
    } catch {
        $gpus = 1
        Write-Host "Could not detect GPUs, defaulting to 1" -ForegroundColor Yellow
    }
} else {
    Write-Host "Using $gpus GPUs" -ForegroundColor Green
}

# Build miner command
$minerArgs = @(
    "-rpc-url", "$rpcurl",
    "-address", "$address",
    "-cuda-devices", "$gpus"
)

# Display startup information
Write-Host ""
Write-Host "$networkName GPU Mining Configuration:" -ForegroundColor Cyan
Write-Host "  Mining Address: $address" -ForegroundColor Gray
Write-Host "  RPC URL: $rpcurl" -ForegroundColor Gray
Write-Host "  CUDA GPUs: $gpus" -ForegroundColor Gray
Write-Host "  Algorithm: QMPoW (Quantum Proof of Work)" -ForegroundColor Gray
Write-Host ""
Write-Host "Starting GPU miner..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start miner
try {
    & "$MinerExecutable" $minerArgs
} catch {
    Write-Host "ERROR: Failed to start miner: $_" -ForegroundColor Red
    Write-Host "Make sure CUDA drivers are installed and NVIDIA GPU is available" -ForegroundColor Yellow
    exit 1
} 