# Simple Realistic Quantum Mining Test

# Stop any running geth instances
Stop-Process -Name geth -ErrorAction SilentlyContinue

# Build geth
Set-Location quantum-geth
go build -o ../geth.exe ./cmd/geth
Set-Location ..

# Clean data directory
if (Test-Path ./qdata/geth/chaindata) {
    Remove-Item -Path ./qdata/geth/chaindata -Recurse -Force
}

# Initialize blockchain
./geth.exe --datadir ./qdata init ./quantum-geth/eth/configs/genesis_qmpow_realistic.json

# Create account if needed
if (-not (Test-Path ./qdata/keystore/*)) {
    $password = "password123"
    $password | Out-File -FilePath ./qdata/password.txt -Encoding ascii -NoNewline
    ./geth.exe --datadir ./qdata account new --password ./qdata/password.txt
}

# Get account address
$accountFile = Get-ChildItem -Path ./qdata/keystore/* | Select-Object -First 1
$accountJson = Get-Content $accountFile | ConvertFrom-Json
$accountAddress = $accountJson.address

Write-Host "Starting realistic quantum mining with account: 0x$accountAddress"
Write-Host "Target block time: 12 seconds"

# Start mining
./geth.exe --datadir ./qdata --networkid 73428 --mine --miner.threads 1 --miner.etherbase "0x$accountAddress" --password ./qdata/password.txt --unlock "0x$accountAddress" --allow-insecure-unlock console 