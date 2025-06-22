# Realistic Quantum Mining Test Script

Write-Host "🔬 Starting Realistic Quantum Mining Test" -ForegroundColor Cyan
Write-Host "Target: 12-second block times with adaptive difficulty" -ForegroundColor Yellow

# Stop any running geth instances
Write-Host "Stopping any running geth instances..." -ForegroundColor Yellow
Stop-Process -Name geth -ErrorAction SilentlyContinue

# Build geth with realistic quantum mining
Write-Host "Building geth with realistic quantum mining..." -ForegroundColor Yellow
Set-Location quantum-geth
go build -o ../geth.exe ./cmd/geth
Set-Location ..

if (-not (Test-Path ./geth.exe)) {
    Write-Host "❌ Failed to build geth!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Successfully built geth with realistic quantum mining" -ForegroundColor Green

# Clean the data directory
Write-Host "Cleaning data directory..." -ForegroundColor Yellow
if (Test-Path ./qdata/geth/chaindata) {
    Remove-Item -Path ./qdata/geth/chaindata -Recurse -Force
}

# Initialize the blockchain with the realistic quantum genesis block
Write-Host "Initializing blockchain with realistic quantum genesis..." -ForegroundColor Yellow
./geth.exe --datadir ./qdata init ./quantum-geth/eth/configs/genesis_qmpow_realistic.json

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

Write-Host "🚀 Starting realistic quantum blockchain with mining..." -ForegroundColor Cyan
Write-Host "Account: 0x$accountAddress" -ForegroundColor Green
Write-Host "Expected block time: ~12 seconds" -ForegroundColor Yellow
Write-Host "Difficulty will adjust automatically based on actual mining times" -ForegroundColor Yellow

# Run geth with realistic quantum mining
./geth.exe --datadir ./qdata --networkid 73428 --mine --miner.threads 1 --miner.etherbase "0x$accountAddress" --password ./qdata/password.txt --unlock "0x$accountAddress" --allow-insecure-unlock --verbosity 4 console 