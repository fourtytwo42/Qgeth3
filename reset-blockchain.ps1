# Reset Blockchain - Quantum-Geth v0.9-rc3-hw0
# Cleans the blockchain data and creates a new genesis block with specified difficulty
# Usage: .\reset-blockchain.ps1 -difficulty 1 -datadir "qdata_quantum"

param(
    [int]$difficulty = 1,        # Starting difficulty (1 = minimum for testing)
    [string]$datadir = "qdata_quantum",
    [int]$networkid = 73428,
    [int]$chainid = 73428,
    [string]$etherbase = "0x8b61271473f14c80f2B1381Db9CB13b2d5306200",
    [string]$balance = "300000000000000000000000",
    [switch]$force = $false      # Skip confirmation prompt
)

Write-Host "*** QUANTUM-GETH BLOCKCHAIN RESET UTILITY ***" -ForegroundColor Yellow
Write-Host "This will COMPLETELY WIPE the existing blockchain!" -ForegroundColor Red
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data Directory: $datadir"
Write-Host "  Starting Difficulty: $difficulty (0x$([Convert]::ToString($difficulty, 16)))"
Write-Host "  Network ID: $networkid"
Write-Host "  Etherbase: $etherbase"
Write-Host "  Balance: $balance wei"
Write-Host ""

# Confirmation prompt (unless -force is used)
if (-not $force) {
    $confirmation = Read-Host "Are you sure you want to DELETE all blockchain data? (type 'YES' to confirm)"
    if ($confirmation -ne "YES") {
        Write-Host "Operation cancelled." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Cleaning blockchain data..." -ForegroundColor Yellow

# Stop any running geth processes
Write-Host "  Stopping any running geth processes..."
try {
    Get-Process geth -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "  Geth processes stopped" -ForegroundColor Green
} catch {
    Write-Host "  No running geth processes found" -ForegroundColor Gray
}

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

# Create new genesis file
$genesisFile = "genesis_temp_d$difficulty.json"
Write-Host "  Creating genesis file: $genesisFile" -ForegroundColor Yellow

$difficultyHex = "0x$([Convert]::ToString($difficulty, 16))"

$genesisContent = @"
{
  "config": {
    "networkId": $networkid,
    "chainId": $chainid,
    "homesteadBlock": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "qmpow": {
      "qbits": 12,
      "tcount": 4096,
      "lnet": 48,
      "epochLength": 100
    }
  },
  "nonce": "0x0000000000000000",
  "timestamp": "0x00",
  "extraData": "0x00",
  "gasLimit": "0x8000000",
  "difficulty": "$difficultyHex",
  "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "coinbase": "0x0000000000000000000000000000000000000000",
  "baseFeePerGas": "0x3B9ACA00",
  "withdrawalsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
  "alloc": {
    "$etherbase": {
      "balance": "$balance"
    }
  }
}
"@

try {
    # Write file without BOM to avoid JSON parsing issues
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($genesisFile, $genesisContent, $utf8NoBom)
    Write-Host "  Genesis file created" -ForegroundColor Green
} catch {
    Write-Host "  Failed to create genesis file: $_" -ForegroundColor Red
    exit 1
}

# Initialize blockchain with new genesis
Write-Host "Initializing blockchain with new genesis..." -ForegroundColor Yellow
try {
    $initResult = & ".\quantum-geth\build\bin\geth.exe" --datadir $datadir init $genesisFile 2>&1
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
    Remove-Item $genesisFile -Force
    Write-Host "  Temporary genesis file removed" -ForegroundColor Green
} catch {
    Write-Host "  Warning: Could not remove temporary genesis file: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "BLOCKCHAIN RESET COMPLETE!" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Starting Difficulty: $difficulty"
Write-Host "  Target: $(if ($difficulty -eq 1) { 'Very easy (instant blocks)' } elseif ($difficulty -le 10) { 'Easy (fast blocks)' } elseif ($difficulty -le 100) { 'Medium (normal blocks)' } else { 'Hard (slow blocks)' })"
Write-Host "  Data Directory: $datadir"
Write-Host "  Etherbase: $etherbase"
Write-Host ""
Write-Host "You can now start mining with:" -ForegroundColor Yellow
Write-Host "   .\start-geth-mining.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Pro Tips:" -ForegroundColor Cyan
Write-Host "  * Use difficulty=1 for instant block testing"
Write-Host "  * Use difficulty=10-100 for normal testing"
Write-Host "  * Use difficulty=1000+ for realistic mining"
Write-Host "  * Bitcoin-style nonce progression: qnonce=0,1,2,3..."
Write-Host "  * Lower quality values win (Bitcoin-style)"
Write-Host "" 