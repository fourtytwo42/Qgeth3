# Start Geth with Mining - Quantum-Geth v0.9-rc3-hw0
# Starts the quantum geth node with Bitcoin-style nonce-level difficulty mining
# Usage: .\start-geth-mining.ps1 -threads 1 -verbosity 4

param(
    [int]$threads = 1,
    [string]$datadir = "qdata_quantum",
    [int]$networkid = 73428,
    [int]$port = 0,  # Disabled for isolated testing
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [int]$verbosity = 4,
    [string]$quantumSolver = ".\quantum-geth\tools\solver\qiskit_solver.py",
    [switch]$isolated = $true  # Run in isolated mode (no peers)
)

Write-Host "*** QUANTUM-GETH MINING STARTUP ***" -ForegroundColor Green
Write-Host "Bitcoin-Style Nonce-Level Difficulty Implementation" -ForegroundColor Cyan
Write-Host "Successfully fixed quality calculation and comparison logic!" -ForegroundColor Green
Write-Host ""

# Check if blockchain exists
if (-not (Test-Path "$datadir\geth\chaindata")) {
    Write-Host "ERROR: No blockchain found in $datadir" -ForegroundColor Red
    Write-Host ""
    Write-Host "You need to initialize a blockchain first:" -ForegroundColor Yellow
    Write-Host "   .\reset-blockchain.ps1 -difficulty 1    # Easy testing"
    Write-Host "   .\reset-blockchain.ps1 -difficulty 100  # Medium testing"
    Write-Host "   .\reset-blockchain.ps1 -difficulty 1000 # Hard testing"
    Write-Host ""
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data Directory: $datadir"
Write-Host "  Mining Threads: $threads"
Write-Host "  Network ID: $networkid"
Write-Host "  Etherbase: $etherbase"
Write-Host "  Verbosity: $verbosity"
Write-Host "  Quantum Solver: $quantumSolver"
Write-Host "  Isolated Mode: $isolated"
Write-Host ""

# Stop any existing geth processes
Write-Host "Stopping any existing geth processes..." -ForegroundColor Yellow
try {
    Get-Process geth -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep 2
    Write-Host "  Existing processes stopped" -ForegroundColor Green
} catch {
    Write-Host "  No existing processes found" -ForegroundColor Gray
}

# Build the geth command
$gethArgs = @(
    "--datadir", $datadir
    "--networkid", $networkid
    "--mine"
    "--miner.threads", $threads
    "--miner.etherbase", $etherbase
    "--quantum.solver", $quantumSolver
    "--http"
    "--http.api", "admin,eth,miner,net,txpool,personal,web3"
    "--http.addr", "localhost"
    "--http.port", $httpport
    "--http.corsdomain", "*"
    "--allow-insecure-unlock"
    "--verbosity", $verbosity
)

# Add isolation parameters if requested
if ($isolated) {
    $gethArgs += @(
        "--nodiscover"
        "--maxpeers", "0"
        "--port", $port
    )
    Write-Host "Running in isolated mode (no peer connections)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Expected Mining Behavior:" -ForegroundColor Cyan
Write-Host "  • Bitcoin-style nonce progression: qnonce=0,1,2,3,4..."
Write-Host "  • Quality must be less than Target for success (lower quality = better)"
Write-Host "  • Positive quality values (no more negative numbers)"
Write-Host "  • Multiple attempts required for higher difficulty"
Write-Host ""

Write-Host "Starting Quantum-Geth mining..." -ForegroundColor Green
Write-Host "   Use Ctrl+C to stop mining" -ForegroundColor Gray
Write-Host ""

# Start geth
try {
    & ".\quantum-geth\build\bin\geth.exe" @gethArgs
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to start geth: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Make sure the blockchain is initialized:"
    Write-Host "     .\reset-blockchain.ps1 -difficulty 1"
    Write-Host "  2. Check if the geth binary exists:"
    Write-Host "     .\quantum-geth\build\bin\geth.exe"
    Write-Host "  3. Verify the quantum solver exists:"
    Write-Host "     $quantumSolver"
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Mining stopped." -ForegroundColor Yellow 