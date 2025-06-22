# Fixed Genesis Quantum Blockchain Test - Simple Version

# Stop any running geth instances
Stop-Process -Name geth -ErrorAction SilentlyContinue

# Build geth
Set-Location quantum-geth
go build -o ../geth.exe ./cmd/geth
Set-Location ..

# Clean the data directory
if (Test-Path ./qdata/geth/chaindata) {
    Remove-Item -Path ./qdata/geth/chaindata -Recurse -Force
}

# Initialize the blockchain with the fixed quantum genesis block
./geth.exe --datadir ./qdata init ./quantum-geth/eth/configs/genesis_qmpow_fixed.json

# Create an account if needed
if (-not (Test-Path ./qdata/keystore/*)) {
    $password = "password123"
    $password | Out-File -FilePath ./qdata/password.txt -Encoding ascii -NoNewline
    ./geth.exe --datadir ./qdata account new --password ./qdata/password.txt
}

# Get the account address
$accountFile = Get-ChildItem -Path ./qdata/keystore/* | Select-Object -First 1
$accountJson = Get-Content $accountFile | ConvertFrom-Json
$accountAddress = $accountJson.address

# Run geth with quantum mining enabled
./geth.exe --datadir ./qdata --networkid 73428 --mine --miner.threads 1 --miner.etherbase "0x$accountAddress" --password ./qdata/password.txt --unlock "0x$accountAddress" --allow-insecure-unlock --verbosity 4 console 