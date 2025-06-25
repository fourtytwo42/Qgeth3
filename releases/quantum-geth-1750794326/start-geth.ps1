# Quantum-Geth Node Launcher (PowerShell)
param(
    [string]$DataDir = "qdata",
    [switch]$Help
)

if ($Help) {
    Write-Host "Quantum-Geth Node Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-geth.ps1 [-DataDir <path>] [-Help]" -ForegroundColor Yellow
    Write-Host "  -DataDir: Blockchain data directory (default: qdata)" -ForegroundColor White
    Write-Host "  -Help: Show this help" -ForegroundColor White
    exit 0
}

Write-Host "Starting Quantum-Geth Node..." -ForegroundColor Blue
Write-Host "Data Directory: $DataDir" -ForegroundColor Green
Write-Host "Network ID: 1337" -ForegroundColor Green  
Write-Host "Mining: DISABLED (use start-geth-mining.ps1 to mine)" -ForegroundColor Yellow
Write-Host ""

& ".\geth.exe" --datadir "$DataDir" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,qmpow,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,qmpow,admin" --miner.threads 0 --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
