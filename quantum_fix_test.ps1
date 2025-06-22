# Quantum Blockchain Test Script
# This script tests the quantum blockchain with RLP encoding fixes

Write-Host "🔬 Starting Quantum Blockchain Test" -ForegroundColor Cyan

# Stop any running geth instances
Write-Host "Stopping any running geth instances..." -ForegroundColor Yellow
Stop-Process -Name geth -ErrorAction SilentlyContinue

# Rebuild geth with our changes
Write-Host "Building geth with quantum fixes..." -ForegroundColor Yellow
Set-Location quantum-geth
go build -o ../geth.exe ./cmd/geth
Set-Location ..

if (-not (Test-Path ./geth.exe)) {
    Write-Host "❌ Failed to build geth!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Successfully built geth with quantum fixes" -ForegroundColor Green

# Create output directory for logs
if (-not (Test-Path ./logs)) {
    New-Item -Path ./logs -ItemType Directory | Out-Null
}

# Initialize the blockchain with the quantum genesis block
Write-Host "Initializing quantum blockchain..." -ForegroundColor Yellow
./geth.exe --datadir ./qdata init ./quantum-geth/eth/configs/genesis_qmpow.json

# Create an account if needed
if (-not (Test-Path ./qdata/keystore/*)) {
    Write-Host "Creating new account..." -ForegroundColor Yellow
    ./geth.exe --datadir ./qdata account new --password ./qdata/password.txt
}

# Get the account address
$accountFile = Get-ChildItem -Path ./qdata/keystore/* | Select-Object -First 1
$accountJson = Get-Content $accountFile | ConvertFrom-Json
$accountAddress = $accountJson.address
Write-Host "Using account: 0x$accountAddress" -ForegroundColor Green

# Start mining with debug logs
Write-Host "🚀 Starting quantum blockchain with mining..." -ForegroundColor Cyan
Write-Host "Logs will be saved to ./logs directory" -ForegroundColor Yellow

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "./logs/quantum_test_$timestamp.log"

$cmd = "./geth.exe --datadir ./qdata --networkid 9248 --mine --miner.threads 1 " +
       "--miner.etherbase 0x$accountAddress " +
       "--password ./qdata/password.txt --unlock 0x$accountAddress " +
       "--http --http.api eth,web3,net,debug,qmpow --http.addr localhost --http.port 8545 " +
       "--verbosity 5 --vmodule qmpow=5,miner=5,worker=5,blockchain=5,consensus=5,core=5 " +
       "--qmpow.powmode 1 --qmpow.testmode --log.debug " +
       "--syncmode full --gcmode archive --txlookuplimit 0 " +
       "--allow-insecure-unlock"

Write-Host "Running command: $cmd" -ForegroundColor Gray
Invoke-Expression "$cmd 2>&1 | Tee-Object -FilePath $logFile" 