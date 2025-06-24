# Quantum-Geth Mining Node Launcher (PowerShell)
param(
    [int]$Threads = 1,
    [string]$DataDir = "qdata",
    [switch]$Help
)

if ($Help) {
    Write-Host "Quantum-Geth Mining Node Launcher" -ForegroundColor Cyan
    Write-Host "Usage: .\start-geth-mining.ps1 [-Threads <n>] [-DataDir <path>] [-Help]" -ForegroundColor Yellow
    Write-Host "  -Threads: Number of mining threads (default: 1)" -ForegroundColor White
    Write-Host "  -DataDir: Blockchain data directory (default: qdata)" -ForegroundColor White
    Write-Host "  -Help: Show this help" -ForegroundColor White
    exit 0
}

Write-Host "Starting Quantum-Geth Mining Node..." -ForegroundColor Blue
Write-Host "Data Directory: $DataDir" -ForegroundColor Green
Write-Host "Network ID: 1337" -ForegroundColor Green
Write-Host "Mining: ENABLED with $Threads threads" -ForegroundColor Green
Write-Host ""

& ".\geth.exe" --datadir "$DataDir" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,admin" --mine --miner.threads $Threads --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
