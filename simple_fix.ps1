# Simple Quantum Blockchain Fix
Write-Host "Fixing Quantum Blockchain Block Progression..." -ForegroundColor Cyan

# Stop processes
Stop-Process -Name "geth" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Clean database completely
if (Test-Path "qdata") {
    Remove-Item -Path "qdata" -Recurse -Force
    Write-Host "Database cleaned" -ForegroundColor Green
}

# Initialize fresh
Write-Host "Initializing fresh quantum blockchain..." -ForegroundColor Yellow
& ./geth.exe --datadir qdata init quantum-geth/eth/configs/genesis_qmpow.json

# Create password file
New-Item -Path "qdata" -ItemType Directory -Force -ErrorAction SilentlyContinue
"password123" | Out-File -FilePath "qdata/password.txt" -Encoding ascii

# Start mining
Write-Host "Starting quantum mining..." -ForegroundColor Yellow
Start-Process -FilePath "./geth.exe" -ArgumentList @(
    "--datadir", "qdata",
    "--mine",
    "--miner.threads", "1",
    "--unlock", "0x9052D4c9f8828E262B0559DCebC0ee310f570968",
    "--password", "qdata/password.txt",
    "--allow-insecure-unlock",
    "--miner.etherbase", "0x9052D4c9f8828E262B0559DCebC0ee310f570968",
    "--networkid", "73428",
    "--nodiscover",
    "--maxpeers", "0",
    "--verbosity", "4",
    "--http",
    "--http.port", "8545"
) -NoNewWindow

Write-Host "Quantum blockchain started. Waiting 20 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 20

# Check progression
Write-Host "Checking block progression..." -ForegroundColor Cyan
for ($i = 1; $i -le 10; $i++) {
    try {
        $body = '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
        $result = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 3
        $blockHex = $result.result
        $blockNumber = [Convert]::ToInt32($blockHex, 16)
        Write-Host "Check $i : Block $blockNumber" -ForegroundColor White
        
        if ($blockNumber -ge 2) {
            Write-Host "SUCCESS: Block progression confirmed!" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "Check $i : RPC not ready" -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 5
}

Write-Host "Fix complete. Monitor the blockchain for continued progression." -ForegroundColor Magenta 