# Q Coin Build Script - Creates timestamped releases
# Usage: ./build-release.ps1 [component]
# Components: geth, miner, both (default: both)

param(
    [Parameter(Position=0)]
    [ValidateSet("geth", "miner", "both")]
    [string]$Component = "both"
)

Write-Host "Building Q Coin Release..." -ForegroundColor Cyan
Write-Host ""

$timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

# Get absolute path to releases directory before changing directories
$releasesPath = (Resolve-Path "../../releases").Path

# Build geth
if ($Component -eq "geth" -or $Component -eq "both") {
    Write-Host "Building quantum-geth..." -ForegroundColor Yellow
    
    if (-not (Test-Path "../../quantum-geth")) {
        Write-Host "quantum-geth directory not found!" -ForegroundColor Red
        exit 1
    }
    
    # Build to regular location first
    Push-Location "../../quantum-geth"
    try {
        # CRITICAL: Always use CGO_ENABLED=0 for geth to ensure compatibility
        # This ensures Windows and Linux builds have identical quantum field handling
        $env:CGO_ENABLED = "0"
        Write-Host "ENFORCING: CGO_ENABLED=0 for geth build (quantum field compatibility)" -ForegroundColor Yellow
        & go build -o "geth.exe" "./cmd/geth"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-geth built successfully (CGO_ENABLED=0)" -ForegroundColor Green
            
            # Create timestamped release in geth subfolder
            $releaseDir = "$releasesPath\geth\quantum-geth-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "geth.exe" "$releaseDir\geth.exe" -Force
            
            # Create PowerShell launcher
            $gethPS1 = @'
# Q Coin Geth Launcher (PowerShell)
# Usage: .\start-geth.ps1 [network] [options]
# Networks: mainnet, testnet, devnet (default: testnet)

param(
    [Parameter(Position=0)]
    [ValidateSet("mainnet", "testnet", "devnet")]
    [string]$Network = "testnet",
    
    [switch]$Mining,
    [switch]$Help,
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if ($Help) {
    Write-Host "Q Coin Geth Launcher" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start-geth.ps1 [network] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Networks:" -ForegroundColor Yellow
    Write-Host "  mainnet   - Q Coin Mainnet (Chain ID 73236)"
    Write-Host "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    Write-Host "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -mining   - Enable CPU mining with 1 thread"
    Write-Host "  -help     - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\start-geth.ps1                 # Start testnet node"
    Write-Host "  .\start-geth.ps1 mainnet         # Start mainnet node"
    Write-Host "  .\start-geth.ps1 devnet -mining  # Start dev node with mining"
    exit 0
}

# Network configurations
$configs = @{
    "mainnet" = @{
        chainid = 73236
        datadir = "$env:APPDATA\Qcoin\mainnet"
        port = 30303
        name = "Q Coin Mainnet"
    }
    "testnet" = @{
        chainid = 73235
        datadir = "$env:APPDATA\Qcoin\testnet"  
        port = 30303
        name = "Q Coin Testnet"
    }
    "devnet" = @{
        chainid = 73234
        datadir = "$env:APPDATA\Qcoin\devnet"
        port = 30305
        name = "Q Coin Dev Network"
    }
}

$config = $configs[$Network]
Write-Host "Starting $($config.name) (Chain ID: $($config.chainid))" -ForegroundColor Cyan

# Create data directory if needed
if (-not (Test-Path $config.datadir)) {
    New-Item -ItemType Directory -Path $config.datadir -Force | Out-Null
    Write-Host "Created data directory: $($config.datadir)" -ForegroundColor Green
}

# Build geth arguments
$gethArgs = @(
    "--datadir", $config.datadir,
    "--networkid", $config.chainid,
    "--port", $config.port,
    "--nat", "any",
    "--http",
    "--http.addr", "0.0.0.0",
    "--http.port", "8545", 
    "--http.corsdomain", "*",
    "--http.api", "eth,net,web3,personal,admin,txpool,miner,qmpow",
    "--ws",
    "--ws.addr", "0.0.0.0",
    "--ws.port", "8546",
    "--ws.origins", "*", 
    "--ws.api", "eth,net,web3,personal,admin,txpool,miner,qmpow",
    "--maxpeers", "25",
    "--verbosity", "3"
)

# Add mining if requested
if ($Mining) {
    $gethArgs += @("--mine", "--miner.threads", "1", "--miner.etherbase", "0x0000000000000000000000000000000000000001")
    Write-Host "Mining enabled with 1 thread" -ForegroundColor Yellow
} else {
    $gethArgs += @("--mine", "--miner.threads", "0", "--miner.etherbase", "0x0000000000000000000000000000000000000001") 
    Write-Host "Mining interface enabled for external miners" -ForegroundColor Green
}

# Add extra arguments
if ($ExtraArgs) {
    $gethArgs += $ExtraArgs
}

Write-Host "Network: $($config.name)" -ForegroundColor White
Write-Host "Data Directory: $($config.datadir)" -ForegroundColor White
Write-Host "HTTP RPC: http://localhost:8545" -ForegroundColor White
Write-Host "WebSocket: ws://localhost:8546" -ForegroundColor White
Write-Host ""
Write-Host "Starting Q Coin Geth node..." -ForegroundColor Green

# Start geth
& ".\geth.exe" @gethArgs
'@
            $gethPS1 | Out-File -FilePath "$releaseDir\start-geth.ps1" -Encoding UTF8

            # Create Batch launcher
            $gethBAT = @'
@echo off
REM Q Coin Geth Launcher (Batch)
REM Usage: start-geth.bat [mainnet|testnet|devnet]

set NETWORK=%1
if "%NETWORK%"=="" set NETWORK=testnet

if "%NETWORK%"=="mainnet" (
    set CHAINID=73236
    set DATADIR=%APPDATA%\Qcoin\mainnet
    set PORT=30303
    set NETWORK_NAME=Q Coin Mainnet
) else if "%NETWORK%"=="testnet" (
    set CHAINID=73235
    set DATADIR=%APPDATA%\Qcoin\testnet
    set PORT=30303
    set NETWORK_NAME=Q Coin Testnet
) else if "%NETWORK%"=="devnet" (
    set CHAINID=73234
    set DATADIR=%APPDATA%\Qcoin\devnet
    set PORT=30305
    set NETWORK_NAME=Q Coin Dev Network
) else (
    echo Invalid network: %NETWORK%
    echo Usage: start-geth.bat [mainnet^|testnet^|devnet]
    exit /b 1
)

echo Starting %NETWORK_NAME% (Chain ID: %CHAINID%)

REM Create data directory if needed
if not exist "%DATADIR%" mkdir "%DATADIR%"

REM Start geth
geth.exe --datadir "%DATADIR%" --networkid %CHAINID% --port %PORT% --nat any --http --http.addr 0.0.0.0 --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,admin,txpool,miner,qmpow" --ws --ws.addr 0.0.0.0 --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,admin,txpool,miner,qmpow" --maxpeers 25 --verbosity 3 --mine --miner.threads 0 --miner.etherbase 0x0000000000000000000000000000000000000001
'@
            $gethBAT | Out-File -FilePath "$releaseDir\start-geth.bat" -Encoding ASCII

            # Create comprehensive README
            $gethReadme = @"
# Q Coin Geth Release $timestamp

**Built:** $(Get-Date)  
**Component:** Quantum-Geth (Q Coin Blockchain Node)  
**Version:** Latest  

## What is Q Coin Geth?

Q Coin Geth is the official Go implementation of the Q Coin quantum blockchain protocol. It provides a complete blockchain node that can connect to Q Coin networks, process transactions, and mine new blocks using quantum-resistant consensus.

## Quick Start

### Option 1: PowerShell (Recommended)
```powershell
# Start Q Coin Testnet (default)
.\start-geth.ps1

# Start Q Coin Mainnet  
.\start-geth.ps1 mainnet

# Start with CPU mining enabled
.\start-geth.ps1 testnet -mining

# Get help
.\start-geth.ps1 -help
```

### Option 2: Command Prompt/Batch
```cmd
# Start Q Coin Testnet (default)
start-geth.bat testnet

# Start Q Coin Mainnet
start-geth.bat mainnet

# Start Q Coin Dev Network
start-geth.bat devnet
```

### Option 3: Manual Command Line
```cmd
# Initialize blockchain (first time only)
geth.exe --datadir "%APPDATA%\Qcoin\testnet" init genesis_quantum_testnet.json

# Start node
geth.exe --datadir "%APPDATA%\Qcoin\testnet" --networkid 73235 --mine --miner.threads 0
```

## Network Information

| Network | Chain ID | Purpose | Port |
|---------|----------|---------|------|
| **Testnet** | 73235 | Testing & Development | 30303 |
| **Mainnet** | 73236 | Production Network | 30303 |  
| **Devnet** | 73234 | Local Development | 30305 |

## API Access

Once running, Q Coin Geth exposes APIs for interaction:

- **HTTP RPC:** `http://localhost:8545`
- **WebSocket:** `ws://localhost:8546`
- **Available APIs:** eth, net, web3, personal, admin, txpool, miner, qmpow

### Example API Calls

```javascript
// Get chain ID
curl -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' http://localhost:8545

// Get latest block
curl -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":["latest",false],"id":1}' http://localhost:8545

// Create new account
curl -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"personal_newAccount","params":["password"],"id":1}' http://localhost:8545
```

## Data Directories

Q Coin Geth stores blockchain data in these locations:

- **Windows:** `%APPDATA%\Qcoin\[network]\`
- **Example:** `C:\Users\YourName\AppData\Roaming\Qcoin\testnet\`

## Mining

### CPU Mining (Built-in)
```powershell
# Enable 1-thread CPU mining
.\start-geth.ps1 testnet -mining
```

### External Mining (Recommended)
The node runs with mining interface enabled (`--mine --miner.threads 0`) allowing external miners to connect:

1. Start geth node normally (mining interface auto-enabled)
2. Run external quantum miner: `quantum-miner.exe -node http://localhost:8545`

## Console Access

Attach to running node for direct interaction:

```cmd
# While geth is running in another window
geth.exe attach http://localhost:8545

# Or using IPC (if available)
geth.exe attach \\.\pipe\geth.ipc
```

### Useful Console Commands

```javascript
// Check account balance
eth.getBalance("0xYourAddress")

// Start/stop mining
miner.start(1)  // 1 thread
miner.stop()

// Get network info
admin.nodeInfo
net.peerCount

// Create account
personal.newAccount("password")
```

## Configuration Files

For advanced users, you can customize geth with:

- **Config File:** Create `config.toml` with your settings
- **Genesis File:** Required for private networks
- **Key Files:** Located in `keystore/` subdirectory

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change port: add `--port 30304` to command
   - Or stop other geth instances

2. **Slow Sync**
   - Ensure good internet connection
   - Try different bootnodes
   - Check firewall settings

3. **Can't Connect to Network**
   - Verify network (mainnet/testnet) is correct
   - Check NAT/firewall configuration
   - Ensure port 30303 is accessible

4. **Out of Disk Space**
   - Monitor blockchain size (grows over time)
   - Consider pruning: add `--syncmode "fast" --gcmode "archive"`

### Log Levels

Adjust verbosity for debugging:
- `--verbosity 1` - Error only
- `--verbosity 3` - Normal (default)  
- `--verbosity 5` - Very detailed

## Security Notes

- **Never share private keys** or keystore files
- **Use strong passwords** for accounts
- **Backup keystore files** regularly
- **Keep software updated** for security patches

## Support

- **Documentation:** See project README.md
- **Issues:** Report on project GitHub
- **Community:** Join Q Coin Discord/Telegram

---

**Q Coin Geth** - Quantum-Resistant Blockchain Technology
"@ | Out-File -FilePath "$releaseDir\README.md" -Encoding UTF8
            
            Write-Host "Created release: $releaseDir" -ForegroundColor Green
        } else {
            Write-Host "quantum-geth build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

# Build miner
if ($Component -eq "miner" -or $Component -eq "both") {
    Write-Host "Building quantum-miner..." -ForegroundColor Yellow
    
    if (-not (Test-Path "../../quantum-miner")) {
        Write-Host "quantum-miner directory not found!" -ForegroundColor Red
        exit 1
    }
    
    # Build to regular location first
    Push-Location "../../quantum-miner"
    try {
        # Use CGO_ENABLED=0 for Windows miner (uses CuPy instead of native CUDA)
        $env:CGO_ENABLED = "0"
        Write-Host "INFO: Using CGO_ENABLED=0 for Windows miner (CuPy GPU support)" -ForegroundColor Cyan
        & go build -o "quantum-miner.exe" "."
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "quantum-miner built successfully" -ForegroundColor Green
            
            # Create timestamped release in quantum-miner subfolder
            $releaseDir = "$releasesPath\quantum-miner\quantum-miner-$timestamp"
            New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
            Copy-Item "quantum-miner.exe" "$releaseDir\quantum-miner.exe" -Force
            
            # Copy pkg directory if it exists
            if (Test-Path "pkg") {
                Copy-Item "pkg" "$releaseDir\pkg" -Recurse -Force
            }
            
            # Create unified PowerShell launcher
            $minerPS1 = @'
# Q Coin Quantum Miner Launcher (PowerShell)
# Usage: .\start-miner.ps1 [options]

param(
    [int]$Threads = 0,
    [string]$Node = "http://localhost:8545", 
    [string]$Coinbase = "",
    [switch]$ForceCpu,
    [switch]$Help
)

if ($Help) {
    Write-Host "Q Coin Quantum Miner Launcher" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start-miner.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -threads <n>      Number of mining threads (0 = auto-detect)"
    Write-Host "  -node <url>       Geth RPC endpoint (default: http://localhost:8545)"
    Write-Host "  -coinbase <addr>  Mining reward address (auto-detected if empty)"
    Write-Host "  -forceCpu         Force CPU mining even if GPU available"
    Write-Host "  -help             Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\start-miner.ps1                           # Auto-detect everything"
    Write-Host "  .\start-miner.ps1 -threads 8                # Use 8 threads"
    Write-Host "  .\start-miner.ps1 -forceCpu                 # Force CPU mining"
    Write-Host "  .\start-miner.ps1 -node http://remote:8545  # Mine to remote node"
    exit 0
}

Write-Host "Q Coin Quantum Miner Starting..." -ForegroundColor Cyan
Write-Host "Auto-detecting optimal mining configuration..." -ForegroundColor Yellow

# Test node connection
Write-Host "Connecting to Geth at $Node..." -ForegroundColor Yellow
try {
    $chainIdResponse = Invoke-RestMethod -Uri "$Node" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' -ErrorAction Stop
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
    Write-Host "ERROR: Failed to connect to Geth RPC at $Node" -ForegroundColor Red
    Write-Host "       Make sure Geth is running first!" -ForegroundColor Yellow
    exit 1
}

# Get or auto-detect coinbase address
if ($Coinbase -eq "") {
    Write-Host "Auto-detecting mining address..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Uri "$Node" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' -ErrorAction Stop
        if ($response.result -and $response.result.Count -gt 0) {
            $Coinbase = $response.result[0]
            Write-Host "SUCCESS: Using existing account: $Coinbase" -ForegroundColor Green
        } else {
            Write-Host "WARNING: No accounts found. Creating new account..." -ForegroundColor Yellow
            $createResponse = Invoke-RestMethod -Uri "$Node" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"personal_newAccount","params":[""],"id":1}' -ErrorAction Stop
            if ($createResponse.result) {
                $Coinbase = $createResponse.result
                Write-Host "SUCCESS: Created new account: $Coinbase" -ForegroundColor Green
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

# Auto-detect mining mode (GPU vs CPU)
$UseGpu = $false
$MiningMode = "CPU"
$MiningColor = "Yellow"

if (-not $ForceCpu) {
    Write-Host "Testing GPU mining capability..." -ForegroundColor Yellow
    
    # Test GPU mining by running a quick check
    try {
        $gpuTest = & ".\quantum-miner.exe" -gpu -node $Node -coinbase $Coinbase -threads 1 -help 2>&1
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
    "-node", $Node,
    "-coinbase", $Coinbase,
    "-threads", $Threads
)

if ($UseGpu) {
    $minerArgs += "-gpu"
}

# Display configuration
Write-Host ""
Write-Host "Mining Configuration:" -ForegroundColor White
Write-Host "Network: $networkName" -ForegroundColor $networkColor
Write-Host "Chain ID: $chainId" -ForegroundColor White
Write-Host "Mining Mode: $MiningMode" -ForegroundColor $MiningColor
Write-Host "Geth RPC: $Node" -ForegroundColor White
Write-Host "Mining Address: $Coinbase" -ForegroundColor White
Write-Host "Threads: $Threads" -ForegroundColor White
Write-Host ""
Write-Host "Starting Q Coin quantum miner..." -ForegroundColor Green

# Start miner
& ".\quantum-miner.exe" @minerArgs
'@
            $minerPS1 | Out-File -FilePath "$releaseDir\start-miner.ps1" -Encoding UTF8

            # Create Batch launcher
            $minerBAT = @'
@echo off
REM Q Coin Quantum Miner Launcher (Batch)
REM Usage: start-miner.bat [threads] [node_url] [coinbase_address]

set THREADS=%1
set NODE=%2
set COINBASE=%3

if "%THREADS%"=="" set THREADS=0
if "%NODE%"=="" set NODE=http://localhost:8545
if "%COINBASE%"=="" set COINBASE=auto

echo Q Coin Quantum Miner Starting...
echo Threads: %THREADS% (0 = auto-detect)
echo Node: %NODE%
echo Coinbase: %COINBASE%
echo.

if "%COINBASE%"=="auto" (
    echo Auto-detecting coinbase address...
    quantum-miner.exe -node %NODE% -threads %THREADS%
) else (
    quantum-miner.exe -node %NODE% -coinbase %COINBASE% -threads %THREADS%
)
'@
            $minerBAT | Out-File -FilePath "$releaseDir\start-miner.bat" -Encoding ASCII

            # Create comprehensive miner README
            $minerReadme = @"
# Q Coin Quantum Miner Release $timestamp

**Built:** $(Get-Date)  
**Component:** Quantum-Miner (Q Coin Mining Software)  
**Version:** Latest  

## What is Q Coin Quantum Miner?

Q Coin Quantum Miner is the official mining software for the Q Coin quantum blockchain. It uses quantum-resistant algorithms to mine new blocks and secure the network while providing competitive mining rewards.

## Quick Start

### Option 1: PowerShell (Recommended)
```powershell
# Auto-detect everything (recommended)
.\start-miner.ps1

# Specify number of threads
.\start-miner.ps1 -threads 8

# Force CPU mining (disable GPU)
.\start-miner.ps1 -forceCpu

# Mine to remote node
.\start-miner.ps1 -node http://192.168.1.100:8545

# Get help
.\start-miner.ps1 -help
```

### Option 2: Command Prompt/Batch
```cmd
# Auto-detect threads and coinbase
start-miner.bat

# Specify threads
start-miner.bat 8

# Specify threads and node
start-miner.bat 8 http://localhost:8545

# Specify threads, node, and coinbase
start-miner.bat 8 http://localhost:8545 0x1234567890123456789012345678901234567890
```

### Option 3: Manual Command Line
```cmd
# Basic mining (auto-detect coinbase)
quantum-miner.exe -node http://localhost:8545 -threads 4

# Specify coinbase address
quantum-miner.exe -node http://localhost:8545 -coinbase 0x1234... -threads 4

# GPU mining (if available)
quantum-miner.exe -gpu -node http://localhost:8545 -threads 1
```

## Prerequisites

### Required: Running Q Coin Geth Node
The miner requires a running Q Coin Geth node to connect to:

1. **Local Node:** Start geth first with mining interface enabled
   ```powershell
   # In another terminal/window
   geth.exe --mine --miner.threads 0 --http --http.api "eth,net,web3,personal,admin,txpool,miner"
   ```

2. **Remote Node:** Connect to existing node
   ```powershell
   .\start-miner.ps1 -node http://your-node-ip:8545
   ```

### Optional: GPU Support
- **NVIDIA GPU:** Install latest NVIDIA drivers
- **CUDA Support:** Install CUDA Toolkit 11.0+ (optional, uses fallback if not available)
- **Python/CuPy:** For advanced GPU acceleration (auto-detected)

## Mining Modes

### GPU Mining (Recommended)
- **Requirements:** NVIDIA GPU with compute capability 6.0+
- **Performance:** 10-100x faster than CPU mining
- **Usage:** Auto-detected, or force with ``-gpu`` flag
- **Threads:** Typically 1 thread for GPU mining

### CPU Mining
- **Requirements:** Any modern CPU
- **Performance:** Good for testing and low-power mining
- **Usage:** Auto-detected, or force with ``-forceCpu`` flag  
- **Threads:** Defaults to number of CPU cores

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| ``-node <url>`` | Geth RPC endpoint | ``-node http://localhost:8545`` |
| ``-coinbase <addr>`` | Mining reward address | ``-coinbase 0x1234...`` |
| ``-threads <n>`` | Number of mining threads | ``-threads 8`` |
| ``-gpu`` | Enable GPU mining | ``-gpu`` |
| ``-help`` | Show help message | ``-help`` |
| ``-verbose`` | Enable detailed logging | ``-verbose`` |

## Network Information

The miner auto-detects which Q Coin network you're connected to:

| Network | Chain ID | Difficulty | Rewards |
|---------|----------|------------|---------|
| **Testnet** | 73235 | Low | Test coins |
| **Mainnet** | 73236 | High | Real Q Coins |  
| **Devnet** | 73234 | Very Low | Dev coins |

## Performance Optimization

### GPU Optimization
- **Update drivers** to latest NVIDIA version
- **Close other GPU applications** (games, video editing)
- **Monitor temperature** - keep GPU under 85°C
- **Adjust power limit** in MSI Afterburner for efficiency

### CPU Optimization
- **Set thread count** to CPU cores minus 1 (leave 1 for system)
- **Close unnecessary programs** to free up CPU
- **Enable performance mode** in Windows power settings
- **Ensure good cooling** to prevent thermal throttling

### General Tips
- **Stable internet connection** required for mining
- **UPS recommended** to prevent mining interruption
- **Monitor electricity costs** vs mining rewards
- **Join mining pools** for more consistent rewards

## Mining Rewards & Economics

### Reward Structure
- **Block Reward:** Varies by network difficulty
- **Transaction Fees:** Included in block rewards
- **Quantum Bonus:** Additional rewards for quantum-resistant proofs

### Calculating Profitability
```
Daily Earnings = (Hashrate / Network Hashrate) × Daily Block Rewards × Block Price
Daily Costs = Power Consumption (kW) × 24 hours × Electricity Rate
Daily Profit = Daily Earnings - Daily Costs
```

### Pool Mining (Coming Soon)
- **Solo Mining:** Mine directly to network (current mode)
- **Pool Mining:** Share rewards with other miners (lower variance)

## Troubleshooting

### Common Issues

1. **Can't Connect to Node**
   - Ensure Geth is running: ``geth.exe --mine --miner.threads 0 --http``
   - Check node URL is correct
   - Verify firewall allows connections

2. **Low Hashrate**
   - GPU: Update drivers, check temperature, close other GPU apps
   - CPU: Reduce background processes, check thermal throttling
   - Network: Ensure stable internet connection

3. **GPU Not Detected**
   - Install/update NVIDIA drivers
   - Check GPU compute capability (6.0+ required)
   - Try forcing CPU mode: ``-forceCpu``

4. **No Mining Rewards**
   - Check if you're on correct network (testnet vs mainnet)
   - Verify coinbase address is correct
   - Ensure node is fully synced

### Debug Commands

```cmd
# Test miner connection only
quantum-miner.exe -node http://localhost:8545 -help

# Verbose logging
quantum-miner.exe -verbose -node http://localhost:8545 -threads 1

# GPU capability test
quantum-miner.exe -gpu -help
```

## Security & Best Practices

### Wallet Security
- **Secure coinbase address:** Use hardware wallet or secure software wallet
- **Backup private keys:** Store safely offline
- **Monitor rewards:** Check mining address balance regularly

### Mining Security  
- **Firewall:** Only allow necessary connections
- **Antivirus:** Ensure mining software is whitelisted
- **Updates:** Keep miner and Geth updated
- **Monitoring:** Track hashrate and temperature

### Operational Security
- **Power management:** Use UPS for power outages
- **Cooling:** Maintain proper ventilation
- **Monitoring:** Set up alerts for issues
- **Backups:** Regular backup of configurations

## Performance Monitoring

### Key Metrics to Watch
- **Hashrate:** Mining speed (higher is better)
- **Accepted Shares:** Successfully submitted work
- **Rejected Shares:** Should be < 2%
- **Temperature:** Keep GPU < 85°C, CPU < 80°C
- **Power Consumption:** Monitor for efficiency

### Monitoring Tools
- **Built-in:** Miner displays real-time stats
- **GPU-Z:** Monitor GPU temperature and usage
- **HWiNFO64:** System monitoring and logging
- **MSI Afterburner:** GPU overclocking and monitoring

## Support & Community

- **Documentation:** See main project README.md
- **Issues:** Report bugs on project GitHub  
- **Community:** Join Q Coin Discord/Telegram
- **Mining Guides:** Check community mining guides
- **Pool Information:** Visit Q Coin mining pools (when available)

---

**Q Coin Quantum Miner** - Quantum-Resistant Mining Technology
"@
            $minerReadme | Out-File -FilePath "$releaseDir\README.md" -Encoding UTF8
            
            Write-Host "Created release: $releaseDir" -ForegroundColor Green
        } else {
            Write-Host "quantum-miner build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "" 