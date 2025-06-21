# Quantum Geth Mining Startup Script
# Starts quantum proof-of-work mining with QμPoW consensus

param(
    [int]$Threads = 4,
    [string]$DataDir = "qdata", 
    [int]$NetworkId = 73428,
    [switch]$InitGenesis,
    [switch]$VerboseLogging,
    [switch]$Force
)

Write-Host "=== Quantum Geth Mining Startup ===" -ForegroundColor Cyan
Write-Host "Threads: $Threads"
Write-Host "Data Directory: $DataDir"
Write-Host "Network ID: $NetworkId"

# Check for quantum geth binary
$gethPath = ""
if (Test-Path "./geth.exe") {
    $gethPath = "./geth.exe"
    Write-Host "Quantum geth binary found. Skipping build." -ForegroundColor Green
} elseif (Test-Path "./quantum-geth/build/bin/geth.exe") {
    $gethPath = "./quantum-geth/build/bin/geth.exe"
    Write-Host "Using quantum geth from build directory." -ForegroundColor Green
} else {
    Write-Host "Building quantum geth..." -ForegroundColor Yellow
    
    # Check if quantum-geth directory exists
    if (-not (Test-Path "./quantum-geth")) {
        Write-Host "Error: quantum-geth directory not found!" -ForegroundColor Red
        Write-Host "Please run this script from the Qgeth3 root directory." -ForegroundColor Red
        exit 1
    }
    
    # Build the quantum geth
    Push-Location "./quantum-geth"
    try {
        # Set environment for Windows build
        $env:CGO_ENABLED = "0"
        
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        go mod tidy
        
        Write-Host "Building quantum geth binary..." -ForegroundColor Yellow
        go build -o build/bin/geth.exe ./cmd/geth
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed!" -ForegroundColor Red
            exit 1
        }
        
        $gethPath = "./build/bin/geth.exe"
        Write-Host "Build successful!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

# Initialize genesis if requested
if ($InitGenesis) {
    Write-Host "`n=== Initializing Quantum Genesis ===" -ForegroundColor Yellow
    
    # Create data directory
    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir | Out-Null
    }
    
    # Initialize with quantum genesis
    $genesisPath = "./quantum-geth/eth/configs/genesis_qmpow.json"
    if (-not (Test-Path $genesisPath)) {
        Write-Host "Error: Genesis file not found at $genesisPath" -ForegroundColor Red
        exit 1
    }
    
    & $gethPath --datadir $DataDir init $genesisPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Genesis initialization failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Genesis initialized successfully!" -ForegroundColor Green
}

# Create mining account if it doesn't exist
Write-Host "`n=== Setting up Mining Account ===" -ForegroundColor Yellow

$accountFile = "$DataDir/mining-account.txt"
$passwordFile = "$DataDir/password.txt"
$etherbase = ""

if (Test-Path $accountFile) {
    $etherbase = Get-Content $accountFile -Raw
    $etherbase = $etherbase.Trim()
    Write-Host "Using existing mining account: $etherbase" -ForegroundColor Green
} else {
    # Create password file
    "mining123" | Out-File -FilePath $passwordFile -Encoding ASCII
    
    # Create new account
    Write-Host "Creating new mining account..." -ForegroundColor Yellow
    $accountOutput = & $gethPath --datadir $DataDir account new --password $passwordFile 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create mining account!" -ForegroundColor Red
        Write-Host $accountOutput
        exit 1
    }
    
    # Extract address from output
    $addressMatch = $accountOutput | Select-String "Public address of the key:\s+(0x[a-fA-F0-9]{40})"
    if ($addressMatch) {
        $etherbase = $addressMatch.Matches[0].Groups[1].Value
        $etherbase | Out-File -FilePath $accountFile -Encoding ASCII
        Write-Host "Created new mining account: $etherbase" -ForegroundColor Green
    } else {
        Write-Host "Failed to extract mining address!" -ForegroundColor Red
        Write-Host $accountOutput
        exit 1
    }
}

# Start mining
Write-Host "`n=== Starting Quantum Mining ===" -ForegroundColor Cyan

$miningArgs = @(
    "--datadir", $DataDir,
    "--networkid", $NetworkId,
    "--mine",
    "--miner.threads", $Threads,
    "--miner.etherbase", $etherbase,
    "--miner.gasprice", "1000000000",
    "--miner.gaslimit", "8000000",
    "--http",
    "--http.api", "eth,net,web3,personal,miner",
    "--http.corsdomain", "*",
    "--allow-insecure-unlock",
    "--unlock", $etherbase,
    "--password", $passwordFile
)

if ($VerboseLogging) {
    $miningArgs += "--verbosity", "4"
}

$commandStr = "$gethPath " + ($miningArgs -join " ")
Write-Host "Command: $commandStr"

Write-Host "`nQuantum Parameters:" -ForegroundColor Yellow
Write-Host "  QBits: 8"
Write-Host "  T-Gates: 25" 
Write-Host "  Puzzles per block (L_net): 64"
Write-Host "  Mining threads: $Threads"
Write-Host "  Mining account: $etherbase"
Write-Host "  Target block time: 12 seconds"

Write-Host "`nPress Ctrl+C to stop mining" -ForegroundColor Yellow
Write-Host "Monitor progress in the output below:" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Yellow

# Start the quantum mining process
& $gethPath @miningArgs 