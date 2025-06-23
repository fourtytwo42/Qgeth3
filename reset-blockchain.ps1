# Reset Blockchain - Quantum-Geth v0.9 BareBones+Halving
# Cleans the blockchain data, builds latest binary, and creates a new genesis block
# Usage: .\reset-blockchain-clean.ps1 -difficulty 1 -datadir "qdata_quantum"

param(
    [float]$difficulty = 0.0005,  # Starting difficulty (quantum-optimized: 0.0005 based on real testing)
    [string]$datadir = "qdata_quantum",
    [int]$networkid = 73428,
    [int]$chainid = 73428,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [string]$balance = "300000000000000000000000", # 300000 QGC
    [switch]$force = $false,     # Skip confirmation prompt
    [switch]$nobuild = $false    # Skip building geth binary
)

Write-Host "*** QUANTUM-GETH v0.9 BareBones+Halving BLOCKCHAIN RESET ***" -ForegroundColor Yellow
Write-Host "This will COMPLETELY WIPE the existing blockchain and rebuild geth!" -ForegroundColor Red
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data Directory: $datadir"
Write-Host "  Starting Difficulty: $difficulty (quantum-optimized for ~0.25 H/s)" -ForegroundColor Green
Write-Host "  Network ID: $networkid"
Write-Host "  Etherbase: $etherbase"
Write-Host "  Balance: $balance wei - 300000 QGC" -ForegroundColor Yellow
Write-Host "  Build Binary: $(if ($nobuild) { 'NO' } else { 'YES' })" -ForegroundColor $(if ($nobuild) { 'Yellow' } else { 'Green' })
Write-Host ""
Write-Host "v0.9 BareBones+Halving Features:" -ForegroundColor Magenta
Write-Host "  * Initial Subsidy: 50 QGC per block" -ForegroundColor Gray
Write-Host "  * Halving Interval: 600000 blocks - 6 months" -ForegroundColor Gray
Write-Host "  * Target Block Time: 12 seconds - ASERT-Q" -ForegroundColor Gray
Write-Host "  * Quantum Puzzles: 48 per block - 16 qubits x 8192 T-gates" -ForegroundColor Gray
Write-Host "  * Proof Stack: Mahadev-CAPSS-Nova" -ForegroundColor Gray
Write-Host "  * Self-Attestation: Dilithium-2" -ForegroundColor Gray
Write-Host ""

# Confirmation prompt (unless -force is used)
if (-not $force) {
    $confirmation = Read-Host "Are you sure you want to DELETE all blockchain data and rebuild? (type 'YES' to confirm)"
    if ($confirmation -ne "YES") {
        Write-Host "Operation cancelled." -ForegroundColor Red
        exit 1
    }
}

# Stop any running geth processes first
Write-Host "Stopping any running geth processes..." -ForegroundColor Yellow
try {
    Get-Process geth -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep 2
    Write-Host "  Geth processes stopped" -ForegroundColor Green
} catch {
    Write-Host "  No running geth processes found" -ForegroundColor Gray
}

# Build latest Quantum-Geth binary (unless skipped)
if (-not $nobuild) {
    Write-Host "Building latest Quantum-Geth v0.9 BareBones+Halving binary..." -ForegroundColor Cyan
    
    # Check if Go is available
    try {
        $goVersion = & go version 2>$null
        Write-Host "  Go detected: $goVersion" -ForegroundColor Green
    } catch {
        Write-Host "  ERROR: Go compiler not found!" -ForegroundColor Red
        Write-Host "  Please install Go from https://golang.org/dl/" -ForegroundColor Yellow
        Write-Host "  Or use -nobuild flag to skip building" -ForegroundColor Yellow
        exit 1
    }
    
    # Build in quantum-geth directory
    if (-not (Test-Path "quantum-geth")) {
        Write-Host "  ERROR: quantum-geth directory not found!" -ForegroundColor Red
        exit 1
    }
    
    Push-Location "quantum-geth"
    try {
        Write-Host "  Building geth binary..." -ForegroundColor Yellow
        $buildResult = & go build -o ..\geth.exe ./cmd/geth 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Geth binary built successfully" -ForegroundColor Green
            
            # Verify the binary has quantum features
            Pop-Location
            $helpOutput = & .\geth.exe --help 2>&1 | Out-String
            if ($helpOutput -match "quantum\.solver") {
                Write-Host "  Quantum features detected in binary" -ForegroundColor Green
            } else {
                Write-Host "  Warning: Quantum features not detected in binary" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Failed to build geth binary:" -ForegroundColor Red
            Write-Host $buildResult -ForegroundColor Red
            Pop-Location
            exit 1
        }
    } catch {
        Write-Host "  Build error: $_" -ForegroundColor Red
        Pop-Location
        exit 1
    }
} else {
    Write-Host "Skipping binary build (using existing geth.exe)..." -ForegroundColor Yellow
    if (-not (Test-Path "geth.exe")) {
        Write-Host "  ERROR: geth.exe not found in current directory!" -ForegroundColor Red
        Write-Host "  Remove -nobuild flag to build the binary" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Cleaning blockchain data..." -ForegroundColor Yellow

# Remove existing blockchain data
Write-Host "  Removing existing blockchain data..."
if (Test-Path $datadir) {
    try {
        Remove-Item -Recurse -Force $datadir
        Write-Host "  Blockchain data removed" -ForegroundColor Green
    } catch {
        Write-Host "  Failed to remove blockchain data: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  No existing blockchain data found" -ForegroundColor Gray
}

# Create genesis content with v0.9 BareBones+Halving specification
$genesisContent = @"
{
  "config": {
    "chainId": 73428,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "qmpow": {
      "qbits": 16,
      "tcount": 8192,
      "lnet": 48,
      "epochLen": 0
    }
  },
  "nonce": "0x0000000000000042",
  "timestamp": "0x0",
  "extraData": "0x",
  "gasLimit": "0x7D64161",
  "difficulty": "0x$([Convert]::ToString([Math]::Max(500, [Math]::Round($difficulty * 1000000000)), 16))",
  "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "coinbase": "0x0000000000000000000000000000000000000000",
  "alloc": {
    "$etherbase": {
      "balance": "$balance"
    }
  },
  "number": "0x0",
  "gasUsed": "0x0",
  "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "qbits": 16,
  "tcount": 8192,
  "lnet": 48,
  "qnonce64": "0x0000000000000000",
  "extranonce32": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "outcomeroot": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "branchnibbles": "0x000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
  "gatehash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "proofroot": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "attestmode": "0x00"
}
"@

try {
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText("genesis_quantum_v09.json", $genesisContent, $utf8NoBom)
    Write-Host "  v0.9 genesis file created successfully" -ForegroundColor Green
} catch {
    Write-Host "  Failed to create v0.9 genesis file: $_" -ForegroundColor Red
    exit 1
}

# Initialize blockchain with v0.9 genesis
Write-Host "Initializing blockchain with v0.9 BareBones+Halving genesis..." -ForegroundColor Yellow
try {
    $initResult = & ".\geth.exe" --datadir $datadir init genesis_quantum_v09.json 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Blockchain initialized successfully" -ForegroundColor Green
    } else {
        Write-Host "  Failed to initialize blockchain:" -ForegroundColor Red
        Write-Host $initResult -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  Failed to run geth init: $_" -ForegroundColor Red
    exit 1
}

# Clean up temporary genesis file
Write-Host "Cleaning up..." -ForegroundColor Yellow
try {
    Remove-Item genesis_quantum_v09.json -Force
    Write-Host "  Temporary genesis file removed" -ForegroundColor Green
} catch {
    Write-Host "  Warning: Could not remove temporary genesis file: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "v0.9 BareBones+Halving BLOCKCHAIN RESET COMPLETE!" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Binary: $(if ($nobuild) { 'Used existing' } else { 'Rebuilt from source' })"
Write-Host "  Starting Difficulty: $difficulty"
$difficultyDesc = if ($difficulty -eq 1) { 'Very easy - instant blocks' } elseif ($difficulty -le 10) { 'Easy - fast blocks' } elseif ($difficulty -le 100) { 'Medium - normal blocks' } else { 'Hard - slow blocks' }
Write-Host "  Target: $difficultyDesc"
Write-Host "  Data Directory: $datadir"
Write-Host "  Etherbase: $etherbase"
Write-Host "  Initial Subsidy: 50 QGC per block"
Write-Host "  Halving Schedule: Every 600000 blocks"
Write-Host ""
Write-Host "You can now start mining with:" -ForegroundColor Yellow
Write-Host "   .\start-geth-mining.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Pro Tips for v0.9:" -ForegroundColor Cyan
Write-Host "  * Use difficulty=1 for instant testing - 48 puzzles still execute"
Write-Host "  * Use difficulty=10-100 for normal testing"
Write-Host "  * Use difficulty=1000+ for realistic mining"
Write-Host "  * Monitor halving events at blocks 600k, 1200k, 1800k..."
Write-Host "  * Each block executes 48 sequential quantum puzzles"
Write-Host "  * ASERT-Q targets 12-second blocks automatically"
Write-Host "" 