# Q Coin Smart Miner - Uses Latest Release
# Auto-detects network from running Geth node
# Auto-detects GPU capability and falls back to CPU
# Usage: ./start-miner.ps1 [options]

param(
    [int]$Threads = 0,
    [string]$GethRpc = "http://localhost:8545",
    [string]$Node = "",
    [string]$Etherbase = "",
    [string]$Coinbase = "",
    [switch]$ForceCpu,
    [switch]$Help,
    [switch]$Log
)

if ($Help) {
    Write-Host "Q Coin Smart Miner (Latest Release)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Auto-detects network from running Geth node" -ForegroundColor Green
    Write-Host "Auto-detects GPU capability and falls back to CPU" -ForegroundColor Green
    Write-Host "Uses latest miner release from releases directory" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: ./start-miner.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -threads <n>        Number of mining threads (0 = auto-detect)"
    Write-Host "  -node <url>         Geth RPC endpoint (default: http://localhost:8545)"
    Write-Host "  -gethRpc <url>      Alias for -node"
    Write-Host "  -coinbase <addr>    Mining reward address (auto-detected if empty)"
    Write-Host "  -etherbase <addr>   Alias for -coinbase"
    Write-Host "  -forceCpu           Force CPU mining even if GPU available"
    Write-Host "  -log                Enable verbose logging"
    Write-Host "  -help               Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./start-miner.ps1                                    # Smart auto-detection"
    Write-Host "  ./start-miner.ps1 -threads 32                        # 32 threads"
    Write-Host "  ./start-miner.ps1 -node http://64.23.179.84:8545     # Mine to VPS"
    Write-Host "  ./start-miner.ps1 -coinbase 0x1234...890 -forceCpu   # CPU mining with specific address"
    exit 0
}

# Handle parameter aliases and overrides
if ($Node -ne "") {
    $GethRpc = $Node
}
if ($Coinbase -ne "") {
    $Etherbase = $Coinbase
}

Write-Host "Q Coin Smart Miner Starting..." -ForegroundColor Cyan
Write-Host "Auto-detecting optimal mining configuration..." -ForegroundColor Yellow

# Find latest miner release
function Get-LatestMinerRelease {
    $minerReleases = Get-ChildItem "../../releases" -Directory | Where-Object { $_.Name -like "quantum-miner-*" } | Sort-Object Name -Descending
    if ($minerReleases.Count -eq 0) {
        return $null
    }
    return Join-Path $minerReleases[0].FullName "quantum-miner.exe"
}

# Build miner if latest release doesn't exist
$MinerPath = Get-LatestMinerRelease
if (-not $MinerPath -or -not (Test-Path $MinerPath)) {
    Write-Host "Building Q Coin Miner Release..." -ForegroundColor Yellow
    & .\build-release.ps1 miner
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Miner build failed!" -ForegroundColor Red
        exit 1
    }
    $MinerPath = Get-LatestMinerRelease
    if (-not $MinerPath) {
        Write-Host "ERROR: No miner release found after build!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Using latest miner release: $MinerPath" -ForegroundColor Green

# Test Geth connection and get network info
Write-Host "Connecting to Geth at $GethRpc..." -ForegroundColor Yellow
try {
    $chainIdResponse = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' -ErrorAction Stop
    $chainIdHex = $chainIdResponse.result
    $chainId = [Convert]::ToInt32($chainIdHex, 16)
    
    # Determine network from chain ID
    switch ($chainId) {
        73234 { 
            $networkName = "Q Coin Dev Network"
            $networkColor = "Magenta"
        }
        73235 { 
            $networkName = "Q Coin Testnet"
            $networkColor = "Cyan"
        }
        73236 { 
            $networkName = "Q Coin Mainnet"
            $networkColor = "Green"
        }
        default { 
            $networkName = "Unknown Q Coin Network (Chain ID: $chainId)"
            $networkColor = "Yellow"
        }
    }
    
    Write-Host "SUCCESS: Connected to $networkName" -ForegroundColor $networkColor
} catch {
    Write-Host "ERROR: Failed to connect to Geth RPC at $GethRpc" -ForegroundColor Red
    Write-Host "       Make sure Geth is running first!" -ForegroundColor Yellow
    Write-Host "       Try: ./qcoin-geth.ps1" -ForegroundColor Cyan
    exit 1
}

# Get or create etherbase address
if ($Etherbase -eq "") {
    Write-Host "Auto-detecting mining address..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' -ErrorAction Stop
        if ($response.result -and $response.result.Count -gt 0) {
            $Etherbase = $response.result[0]
            Write-Host "SUCCESS: Using existing account: $Etherbase" -ForegroundColor Green
        } else {
            Write-Host "WARNING: No accounts found. Creating new account..." -ForegroundColor Yellow
            $createResponse = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"personal_newAccount","params":[""],"id":1}' -ErrorAction Stop
            if ($createResponse.result) {
                $Etherbase = $createResponse.result
                Write-Host "SUCCESS: Created new account: $Etherbase" -ForegroundColor Green
            } else {
                Write-Host "ERROR: Failed to create account!" -ForegroundColor Red
                exit 1
            }
        }
    } catch {
        Write-Host "ERROR: Failed to get/create mining address!" -ForegroundColor Red
        exit 1
    }
}

# CRITICAL FIX: Set the coinbase address on Geth node
Write-Host "Setting coinbase address on Geth node..." -ForegroundColor Yellow
try {
    $setEtherbaseBody = @{
        jsonrpc = "2.0"
        method = "miner_setEtherbase"
        params = @($Etherbase)
        id = 1
    } | ConvertTo-Json
    
    $setResponse = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body $setEtherbaseBody -ErrorAction Stop
    if ($setResponse.result -eq $true) {
        Write-Host "SUCCESS: Coinbase address set to $Etherbase" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Failed to set coinbase address" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Could not set coinbase address: $_" -ForegroundColor Yellow
}

# Verify the coinbase was set correctly
try {
    $verifyResponse = Invoke-RestMethod -Uri "$GethRpc" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_etherbase","params":[],"id":1}' -ErrorAction Stop
    $currentEtherbase = $verifyResponse.result
    if ($currentEtherbase -eq $Etherbase) {
        Write-Host "✅ VERIFIED: Geth coinbase correctly set to $Etherbase" -ForegroundColor Green
    } else {
        Write-Host "⚠️  WARNING: Geth coinbase is $currentEtherbase, expected $Etherbase" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Could not verify coinbase address" -ForegroundColor Yellow
}

# Auto-detect mining mode (GPU vs CPU)
$UseGpu = $false
$MiningMode = "CPU"
$MiningColor = "Yellow"

if (-not $ForceCpu) {
    Write-Host "Testing GPU mining capability..." -ForegroundColor Yellow
    
    # Test GPU mining by running a quick check
    try {
        $gpuTest = & $MinerPath -gpu -node $GethRpc -coinbase $Etherbase -threads 1 -help 2>&1
        if ($LASTEXITCODE -eq 0) {
            $UseGpu = $true
            $MiningMode = "GPU"
            $MiningColor = "Green"
            Write-Host "SUCCESS: GPU mining available - Using GPU mode" -ForegroundColor Green
        } else {
            Write-Host "WARNING: GPU mining not available - Falling back to CPU" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "WARNING: GPU test failed - Falling back to CPU" -ForegroundColor Yellow
    }
} else {
    Write-Host "INFO: CPU mining forced by user" -ForegroundColor Yellow
}

# Auto-detect thread count
if ($Threads -eq 0) {
    if ($UseGpu) {
        $Threads = 1
        Write-Host "Auto-detected GPU threads: $Threads" -ForegroundColor Green
    } else {
        $Threads = [Environment]::ProcessorCount
        Write-Host "Auto-detected CPU threads: $Threads" -ForegroundColor Green
    }
}

# Prepare miner arguments
$minerArgs = @(
    "-node", $GethRpc,
    "-coinbase", $Etherbase,
    "-threads", $Threads
)

if ($UseGpu) {
    $minerArgs += "-gpu"
}

if ($Log) {
    $minerArgs += "-log"
}

# Display configuration
Write-Host ""
Write-Host "Mining Configuration:" -ForegroundColor White
Write-Host "Network: $networkName" -ForegroundColor $networkColor
Write-Host "Chain ID: $chainId" -ForegroundColor White
Write-Host "Mining Mode: $MiningMode" -ForegroundColor $MiningColor
Write-Host "Geth RPC: $GethRpc" -ForegroundColor White
Write-Host "Mining Address: $Etherbase" -ForegroundColor White
Write-Host "Threads: $Threads" -ForegroundColor White
Write-Host ""
Write-Host "Starting Q Coin miner..." -ForegroundColor Green

# Start miner
& $MinerPath @minerArgs 