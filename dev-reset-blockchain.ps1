# Reset Blockchain - Quantum-Geth with Halving
# Cleans the blockchain data, builds latest release binaries, and creates a new genesis block
# Usage: .\reset-blockchain.ps1 -difficulty 1 -force

param(
    [float]$difficulty = 200,  # Starting difficulty (quantum-optimized: 200 based on real testing)
    [string]$datadir = "qdata",
    [int]$networkid = 73234,
    [int]$chainid = 73234,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [string]$balance = "300000000000000000000000", # 300000 QGC
    [switch]$force = $false,     # Skip confirmation prompt
    [switch]$nobuild = $false    # Skip building new release packages
)

Write-Host "*** QUANTUM-GETH COMPLETE RESET ***" -ForegroundColor Yellow
Write-Host "This will COMPLETELY WIPE the existing blockchain and build new release packages!" -ForegroundColor Red
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data Directory: $datadir"
Write-Host "  Starting Difficulty: $difficulty (ASERT-Q optimized for testing)" -ForegroundColor Green
Write-Host "  Network ID: $networkid"
Write-Host "  Etherbase: $etherbase"
Write-Host "  Balance: $balance wei - 300000 QGC" -ForegroundColor Yellow
  Write-Host "  Build Releases: $(if ($nobuild) { 'NO' } else { 'YES' })" -ForegroundColor $(if ($nobuild) { 'Yellow' } else { 'Green' })
Write-Host ""
Write-Host "Quantum-Geth Features:" -ForegroundColor Magenta
Write-Host "  * Initial Subsidy: 50 QGC per block" -ForegroundColor Gray
Write-Host "  * Halving Interval: 600000 blocks - 6 months" -ForegroundColor Gray
Write-Host "  * Target Block Time: 12 seconds - ASERT-Q (per-block adjustment)" -ForegroundColor Gray
Write-Host "  * Quantum Puzzles: 128 chained per block - 16 qubits x 20 T-gates" -ForegroundColor Gray
Write-Host "  * Proof Stack: Mahadev-CAPSS-Nova" -ForegroundColor Gray
Write-Host "  * Self-Attestation: Dilithium-2" -ForegroundColor Gray
Write-Host ""

# Confirmation prompt (unless -force is used)
if (-not $force) {
    $confirmation = Read-Host "Are you sure you want to DELETE all blockchain data and build new releases? (type 'YES' to confirm)"
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

# Build latest release packages (unless skipped)
if (-not $nobuild) {
    Write-Host "Building new Quantum-Geth release packages..." -ForegroundColor Cyan
    
    try {
        Write-Host "  Building both quantum-geth and quantum-miner releases..." -ForegroundColor Yellow
        & ".\build-release.ps1" both
        
        # Check if releases were actually created (more reliable than exit codes)
        $GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        $MinerReleaseDir = Get-ChildItem -Path "releases\quantum-miner-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        
        if ($GethReleaseDir -and $MinerReleaseDir) {
            Write-Host "  Release packages built successfully" -ForegroundColor Green
            Write-Host "    Geth: $($GethReleaseDir.Name)" -ForegroundColor Gray
            Write-Host "    Miner: $($MinerReleaseDir.Name)" -ForegroundColor Gray
        } else {
            throw "Release directories not found after build"
        }
    } catch {
        Write-Host "  ERROR: Failed to build release packages: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  Or use -nobuild flag to skip building" -ForegroundColor Yellow
        exit 1
    }
    
} else {
    Write-Host "Skipping release build (using existing releases)..." -ForegroundColor Yellow
    
    # Find the newest geth release
    $GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $GethReleaseDir) {
        Write-Host "  ERROR: No quantum-geth release found!" -ForegroundColor Red
        Write-Host "  Remove -nobuild flag to build release packages" -ForegroundColor Yellow
        exit 1
    } else {
        Write-Host "  Using geth release: $($GethReleaseDir.Name)" -ForegroundColor Green
    }
    
    # Find the newest miner release
    $MinerReleaseDir = Get-ChildItem -Path "releases\quantum-miner-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $MinerReleaseDir) {
        Write-Host "  Warning: No quantum-miner release found!" -ForegroundColor Yellow
    } else {
        Write-Host "  Using miner release: $($MinerReleaseDir.Name)" -ForegroundColor Green
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

# Create genesis file with specified difficulty
Write-Host "Creating genesis file with difficulty $difficulty..." -ForegroundColor Yellow

# Convert difficulty to hex (direct conversion without scaling)
$difficultyInt = [math]::Round($difficulty)
$difficultyHex = "0x" + [Convert]::ToString($difficultyInt, 16).ToUpper()

Write-Host "  Converting difficulty: $difficulty -> $difficultyInt -> $difficultyHex" -ForegroundColor Gray

# Create dynamic genesis JSON with proper QMPoW consensus engine format
$genesisJson = @"
{
  "config": {
    "networkId": $chainid,
    "chainId": $chainid,
    "eip2FBlock": 0,
    "eip7FBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip160Block": 0,
    "eip161FBlock": 0,
    "eip170FBlock": 0,
    "qmpow": {
      "qbits": 16,
      "tcount": 20,
      "lnet": 128,
      "epochLen": 100,
      "testMode": false
    }
  },
  "difficulty": "$difficultyHex",
  "gasLimit": "0x2fefd8",
  "alloc": {
    "$etherbase": {
      "balance": "$balance"
    },
    "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A": {
      "balance": "$balance"
    },
    "0x1234567890123456789012345678901234567890": {
      "balance": "$balance"
    }
  }
}
"@

# Write the dynamic genesis file
$tempGenesisFile = "genesis_quantum_temp.json"
try {
    # Use UTF-8 encoding without BOM to prevent 'invalid character' errors
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($tempGenesisFile, $genesisJson, $utf8NoBom)
    Write-Host "  Dynamic genesis file created: $tempGenesisFile" -ForegroundColor Green
} catch {
    Write-Host "  Failed to create genesis file: $_" -ForegroundColor Red
    exit 1
}

# Find the newest geth release to use for initialization
$GethReleaseDir = Get-ChildItem -Path "releases\quantum-geth-*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if ($GethReleaseDir) {
    $GethExecutable = "$($GethReleaseDir.FullName)\geth.exe"
    Write-Host "Using geth from release: $($GethReleaseDir.Name)" -ForegroundColor Green
} else {
    Write-Host "ERROR: No quantum-geth release found for initialization!" -ForegroundColor Red
    Write-Host "Run the script without -nobuild flag to build releases" -ForegroundColor Yellow
    exit 1
}

# Initialize blockchain with dynamic genesis format
Write-Host "Initializing blockchain with custom difficulty genesis..." -ForegroundColor Yellow
try {
    $initResult = & "$GethExecutable" --datadir $datadir init $tempGenesisFile 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Blockchain initialized successfully with difficulty $difficulty" -ForegroundColor Green
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
    if (Test-Path $tempGenesisFile) {
        Remove-Item $tempGenesisFile -Force
        Write-Host "  Temporary genesis file removed" -ForegroundColor Green
    }
    if (Test-Path "genesis_quantum.json") {
        Remove-Item genesis_quantum.json -Force
        Write-Host "  Old genesis file removed" -ForegroundColor Green
    }
} catch {
    Write-Host "  Warning: Could not remove temporary genesis file: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "QUANTUM-GETH BLOCKCHAIN RESET COMPLETE!" -ForegroundColor Green
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
Write-Host "Pro Tips:" -ForegroundColor Cyan
Write-Host "  * Use difficulty=1 for instant testing - 128 chained puzzles still execute"
Write-Host "  * Use difficulty=10-100 for normal testing"
Write-Host "  * Use difficulty=1000+ for realistic mining"
Write-Host "  * Monitor halving events at blocks 600k, 1200k, 1800k..."
Write-Host "  * Each block executes 128 sequential chained quantum puzzles"
Write-Host "  * ASERT-Q targets 12-second blocks automatically"
Write-Host "" 
