# Fixed Genesis Quantum Blockchain Test

Write-Host "🔬 Starting Fixed Genesis Quantum Blockchain Test" -ForegroundColor Cyan

# Stop any running geth instances
Write-Host "Stopping any running geth instances..." -ForegroundColor Yellow
Stop-Process -Name geth -ErrorAction SilentlyContinue

# Build geth
Write-Host "Building geth with quantum fixes..." -ForegroundColor Yellow
Set-Location quantum-geth
go build -o ../geth.exe ./cmd/geth
Set-Location ..

if (-not (Test-Path ./geth.exe)) {
    Write-Host "❌ Failed to build geth!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Successfully built geth with quantum fixes" -ForegroundColor Green

# Clean the data directory
Write-Host "Cleaning data directory..." -ForegroundColor Yellow
if (Test-Path ./qdata/geth/chaindata) {
    Remove-Item -Path ./qdata/geth/chaindata -Recurse -Force
}

# Initialize the blockchain with the fixed quantum genesis block
Write-Host "Initializing blockchain with fixed genesis..." -ForegroundColor Yellow
$initResult = ./geth.exe --datadir ./qdata init ./quantum-geth/eth/configs/genesis_qmpow_fixed.json
Write-Host $initResult -ForegroundColor Gray

# Create an account if needed
if (-not (Test-Path ./qdata/keystore/*)) {
    Write-Host "Creating new account..." -ForegroundColor Yellow
    $password = "password123"
    $password | Out-File -FilePath ./qdata/password.txt -Encoding ascii -NoNewline
    ./geth.exe --datadir ./qdata account new --password ./qdata/password.txt
}

# Get the account address
$accountFile = Get-ChildItem -Path ./qdata/keystore/* | Select-Object -First 1
$accountJson = Get-Content $accountFile | ConvertFrom-Json
$accountAddress = $accountJson.address
Write-Host "Using account: 0x$accountAddress" -ForegroundColor Green

# Start mining with console output
Write-Host "🚀 Starting quantum blockchain with mining..." -ForegroundColor Cyan

# Run geth with quantum mining enabled
$cmd = "./geth.exe --datadir ./qdata --networkid 73428 --mine --miner.threads 1 " +
       "--miner.etherbase 0x$accountAddress --password ./qdata/password.txt " +
       "--unlock 0x$accountAddress --allow-insecure-unlock " +
       "--verbosity 4 console"

Write-Host "Running command: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd 