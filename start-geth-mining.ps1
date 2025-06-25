# Start Geth with Mining - Quantum-Geth with Halving
# Starts the quantum geth node with quantum proof-of-work mining
# Usage: .\start-geth-mining.ps1 -threads 1 -verbosity 4

param(
    [int]$threads = 1,
    [string]$datadir = "qdata",
    [int]$networkid = 73428,
    [int]$port = 0,  # Disabled for isolated testing
    [int]$httpport = 8545,
    [int]$authrpcport = 8551,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [int]$verbosity = 3,
    [string]$quantumSolver = ".\quantum-geth\tools\solver\qiskit_solver.py",
    [switch]$isolated = $true  # Run in isolated mode (no peers)
)

Write-Host "*** QUANTUM-GETH MINING STARTUP ***" -ForegroundColor Green
Write-Host "48-Puzzle Sequential Quantum Proof-of-Work with Bitcoin-Style Halving" -ForegroundColor Cyan
Write-Host ""

# Check if blockchain exists
if (-not (Test-Path "$datadir\geth\chaindata")) {
    Write-Host "ERROR: No blockchain found in $datadir" -ForegroundColor Red
    Write-Host ""
    Write-Host "You need to initialize a blockchain first:" -ForegroundColor Yellow
    Write-Host "   .\reset-blockchain.ps1 -difficulty 1 -force" -ForegroundColor White
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

Write-Host "Quantum Mining Features:" -ForegroundColor Magenta
Write-Host "  * 48 sequential quantum puzzles per block" -ForegroundColor Gray
Write-Host "  * 16 qubits x 8,192 T-gates per puzzle" -ForegroundColor Gray
Write-Host "  * Seed chaining: Seed_{i+1} = SHA256(Seed_i || Outcome_i)" -ForegroundColor Gray
Write-Host "  * Bitcoin-style halving: 50 QGC -> 25 QGC -> 12.5 QGC..." -ForegroundColor Gray
Write-Host "  * ASERT-Q difficulty targeting 12-second blocks" -ForegroundColor Gray
Write-Host "  * Mahadev->CAPSS->Nova proof generation" -ForegroundColor Gray
Write-Host "  * Dilithium-2 self-attestation" -ForegroundColor Gray
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
    "--http.api", "admin,eth,miner,net,txpool,personal,web3,qmpow,debug,trace"
    "--http.addr", "0.0.0.0"
    "--http.port", $httpport
    "--http.corsdomain", "*"
    "--http.vhosts", "*"
    "--allow-insecure-unlock"
    "--verbosity", $verbosity
    "--log.vmodule", "rpc=1"
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
Write-Host "  • Sequential 48-puzzle execution with seed chaining" -ForegroundColor Gray
Write-Host "  • OutcomeRoot = MerkleRoot(Outcome_0...Outcome_47)" -ForegroundColor Gray
Write-Host "  • GateHash = SHA256(stream_0 || ... || stream_47)" -ForegroundColor Gray
Write-Host "  • Nova-Lite proof aggregation (3 proofs <=6kB each)" -ForegroundColor Gray
Write-Host "  • Dilithium signature binding prover to outcomes" -ForegroundColor Gray
Write-Host "  • ASERT-Q difficulty adjustment every block" -ForegroundColor Gray
Write-Host "  • Block rewards: 50 QGC + transaction fees" -ForegroundColor Gray
Write-Host ""

Write-Host "Starting Quantum-Geth mining..." -ForegroundColor Green
Write-Host "   Use Ctrl+C to stop mining" -ForegroundColor Gray
Write-Host ""

# Find the newest quantum-geth release or use development version
$GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if ($GethReleaseDir) {
    $GethExecutable = "$($GethReleaseDir.FullName)\geth.exe"
    Write-Host "Using geth from release: $($GethReleaseDir.Name)" -ForegroundColor Green
} else {
    $GethExecutable = ".\geth.exe"
    if (-not (Test-Path $GethExecutable)) {
        Write-Host "Geth executable not found. Building release..." -ForegroundColor Yellow
        try {
            & ".\build-release.ps1" geth
            $GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
            if ($GethReleaseDir) {
                $GethExecutable = "$($GethReleaseDir.FullName)\geth.exe"
                Write-Host "SUCCESS: Release built at $($GethReleaseDir.FullName)" -ForegroundColor Green
            } else {
                throw "Failed to create release"
            }
        } catch {
            Write-Host "ERROR: Failed to build quantum-geth release: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Using development geth from root directory" -ForegroundColor Yellow
    }
}

# Start geth with logging
try {
    & "$GethExecutable" @gethArgs | Tee-Object -FilePath "$datadir\geth.log"
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to start geth: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Make sure the blockchain is initialized:"
    Write-Host "     .\reset-blockchain.ps1 -difficulty 1 -force"
    Write-Host "  2. Check if the geth binary exists:"
    Write-Host "     .\geth.exe"
    Write-Host "  3. Verify the quantum solver exists:"
    Write-Host "     $quantumSolver"
    Write-Host "  4. Check if quantum genesis was used:"
    Write-Host "     Should contain 'qmpow' config section"
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Quantum mining stopped." -ForegroundColor Yellow 