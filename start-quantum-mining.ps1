param(
    [Parameter(Mandatory=$true)]
    [int]$Threads,
    [string]$DataDir = "qdata",
    [string]$NetworkId = "73428",
    [string]$MinerAccount = "",
    [string]$PasswordFile = "",
    [switch]$InitGenesis,
    [switch]$VerboseLogging
)

# Quantum Geth Mining Startup Script
Write-Host "=== Quantum Geth Mining Startup ===" -ForegroundColor Green
Write-Host "Threads: $Threads" -ForegroundColor Cyan
Write-Host "Data Directory: $DataDir" -ForegroundColor Cyan  
Write-Host "Network ID: $NetworkId" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "quantum-geth")) {
    Write-Host "Error: quantum-geth directory not found. Run this script from the project root." -ForegroundColor Red
    exit 1
}

# Navigate to quantum-geth directory
Set-Location "quantum-geth"

try {
    # Build geth if needed
    if (-not (Test-Path "build/bin/geth.exe") -and -not (Test-Path "build/bin/geth")) {
        Write-Host "Building quantum geth..." -ForegroundColor Yellow
        
        # Create build directory if it doesn't exist
        if (-not (Test-Path "build/bin")) {
            New-Item -ItemType Directory -Path "build/bin" -Force | Out-Null
        }
        
        if ($VerboseLogging) {
            if ($IsWindows -or $env:OS -eq "Windows_NT") {
                & go build -o build/bin/geth.exe ./cmd/geth
            } else {
                & make geth
            }
        } else {
            if ($IsWindows -or $env:OS -eq "Windows_NT") {
                & go build -o build/bin/geth.exe ./cmd/geth 2>&1 | Out-Null
            } else {
                & make geth 2>&1 | Out-Null
            }
        }
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed. Attempting to resolve dependencies..." -ForegroundColor Yellow
            
            # Try to fix the go-bip39 dependency issue
            Write-Host "Attempting dependency fix..." -ForegroundColor Yellow
            & go mod edit -replace github.com/tyler-smith/go-bip39=github.com/wealdtech/go-bip39@v1.8.0
            & go mod tidy
            
            Write-Host "Retrying build..." -ForegroundColor Yellow
            if ($VerboseLogging) {
                if ($IsWindows -or $env:OS -eq "Windows_NT") {
                    & go build -o build/bin/geth.exe ./cmd/geth
                } else {
                    & make geth
                }
            } else {
                if ($IsWindows -or $env:OS -eq "Windows_NT") {
                    & go build -o build/bin/geth.exe ./cmd/geth 2>&1 | Out-Null
                } else {
                    & make geth 2>&1 | Out-Null
                }
            }
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Build still failing. You may need to resolve dependencies manually." -ForegroundColor Red
                Write-Host "Try running: go mod download && go mod tidy" -ForegroundColor Yellow
                exit 1
            }
        }
        
        Write-Host "Build completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Geth binary already exists." -ForegroundColor Green
    }

    # Determine geth executable path
    $gethPath = ""
    if (Test-Path "build/bin/geth.exe") {
        $gethPath = "./build/bin/geth.exe"
    } elseif (Test-Path "build/bin/geth") {
        $gethPath = "./build/bin/geth"
    } else {
        Write-Host "Error: Could not find geth executable after build." -ForegroundColor Red
        exit 1
    }

    # Initialize genesis if requested or if datadir doesn't exist
    if ($InitGenesis -or -not (Test-Path $DataDir)) {
        Write-Host "Initializing quantum blockchain with genesis..." -ForegroundColor Yellow
        
        # Clean up existing data if reinitializing
        if ($InitGenesis -and (Test-Path $DataDir)) {
            Write-Host "Cleaning up existing blockchain data..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $DataDir
        }
        
        # Initialize with quantum genesis
        & $gethPath --datadir $DataDir init eth/configs/genesis_qmpow.json
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Genesis initialization failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Genesis initialized successfully!" -ForegroundColor Green
    }

    # Prepare mining command
    $miningArgs = @(
        "--datadir", $DataDir,
        "--networkid", $NetworkId,
        "--mine",
        "--miner.threads", $Threads,
        "--miner.gasprice", "1000000000",
        "--miner.gaslimit", "8000000",
        "--http",
        "--http.api", "eth,net,web3,personal,miner",
        "--http.corsdomain", "*",
        "--allow-insecure-unlock"
    )

    # Add account unlock if specified
    if ($MinerAccount -ne "") {
        $miningArgs += "--unlock"
        $miningArgs += $MinerAccount
        
        if ($PasswordFile -ne "") {
            $miningArgs += "--password"
            $miningArgs += $PasswordFile
        }
    }

    # Add verbose logging if requested
    if ($VerboseLogging) {
        $miningArgs += "--verbosity"
        $miningArgs += "4"
    }

    Write-Host "`n=== Starting Quantum Mining ===" -ForegroundColor Green
    Write-Host "Command: $gethPath $($miningArgs -join ' ')" -ForegroundColor Cyan
    Write-Host "`nQuantum Parameters:" -ForegroundColor Yellow
    Write-Host "  QBits: 8" -ForegroundColor White
    Write-Host "  T-Gates: 25" -ForegroundColor White  
    Write-Host "  Puzzles per block (L_net): 64" -ForegroundColor White
    Write-Host "  Mining threads: $Threads" -ForegroundColor White
    Write-Host "  Target block time: 12 seconds" -ForegroundColor White
    Write-Host "`nPress Ctrl+C to stop mining" -ForegroundColor Yellow
    Write-Host "Monitor progress in the output below:" -ForegroundColor Yellow
    Write-Host ("=" * 50) -ForegroundColor Green

    # Start mining
    & $gethPath @miningArgs

} catch {
    Write-Host "Error starting quantum mining: $_" -ForegroundColor Red
    exit 1
} finally {
    # Return to original directory
    Set-Location ".."
} 