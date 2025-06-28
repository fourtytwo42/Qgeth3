# Windows Q Geth Node Guide

Complete guide for running Q Coin quantum blockchain nodes on Windows systems.

## 📋 Requirements

### System Requirements
- **OS**: Windows 10 (64-bit) or Windows 11
- **CPU**: 2+ cores (4+ recommended) 
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 20GB minimum (SSD recommended for better performance)
- **Network**: Stable internet connection with 1Mbps+ bandwidth

### Software Dependencies
- **Go**: Version 1.21 or later ([Download](https://golang.org/dl/))
- **Git**: For source code management ([Download](https://git-scm.com/download/win))
- **PowerShell**: 5.1+ (included with Windows)

### Optional (for development)
- **Visual Studio Build Tools** ([Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/))
- **Windows Terminal** ([Microsoft Store](https://aka.ms/terminal))

## 🚀 Installation

### Quick Installation (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/fourtytwo42/Qgeth3.git
Set-Location Qgeth3

# Build Q Geth
.\scripts\windows\build-release.ps1

# Start the node (testnet)
.\scripts\windows\start-geth.ps1 -Network testnet
```

### Manual Build Process
```powershell
# Navigate to quantum-geth directory
Set-Location quantum-geth

# Set build environment
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"

# Build geth binary
$gitCommit = git rev-parse HEAD
$buildTime = Get-Date -Format "yyyy-MM-dd_HH:mm:ss"
go build -ldflags "-s -w -X main.gitCommit=$gitCommit -X main.buildTime=$buildTime" -o ..\geth.exe .\cmd\geth

# Verify build
..\geth.exe version
```

### Installation Verification
```powershell
# Check if geth binary exists
Get-ChildItem geth.exe
Get-Command .\geth.exe

# Test basic functionality
.\geth.exe help
.\geth.exe version

# Check quantum consensus is available
.\geth.exe help | Select-String -Pattern "quantum"
```

## ⚙️ Configuration

### Network Selection
Q Coin supports three networks:

#### Testnet (Recommended for testing)
```powershell
# Start testnet node
.\scripts\windows\start-geth.ps1 -Network testnet

# Custom testnet configuration
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" `
  --networkid 1337 `
  --genesis "configs\genesis_quantum_testnet.json" `
  --http --http.addr "0.0.0.0" --http.port 8545 `
  --http.corsdomain "*" --http.api "eth,net,web3,personal,txpool" `
  --ws --ws.addr "0.0.0.0" --ws.port 8546 `
  --ws.origins "*" --ws.api "eth,net,web3" `
  --port 30303 --nat "any" `
  --allow-insecure-unlock `
  console
```

#### Devnet (Development)
```powershell
# Start development network
.\scripts\windows\start-geth.ps1 -Network devnet

# Custom devnet with mining
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\devnet" `
  --networkid 1338 `
  --genesis "configs\genesis_quantum_dev.json" `
  --http --http.addr "127.0.0.1" --http.port 8545 `
  --mine --miner.etherbase "0xYourCoinbaseAddress" `
  console
```

#### Mainnet (Production)
```powershell
# Start mainnet node (when available)
.\scripts\windows\start-geth.ps1 -Network mainnet

# Mainnet with conservative settings
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\mainnet" `
  --networkid 1339 `
  --genesis "configs\genesis_quantum_mainnet.json" `
  --http --http.addr "127.0.0.1" --http.port 8545 `
  --http.api "eth,net,web3" `
  --port 30303 `
  --cache 1024 --maxpeers 50 `
  console
```

### Data Directory Structure
```
%USERPROFILE%\.qcoin\
├── testnet\
│   ├── geth\
│   │   ├── chaindata\      # Blockchain database
│   │   ├── lightchaindata\ # Light client data
│   │   ├── nodes\          # Node discovery data
│   │   └── LOCK           # Database lock file
│   ├── keystore\          # Account keystore files
│   └── geth.ipc          # IPC communication pipe
├── devnet\               # Development network data
└── mainnet\             # Mainnet data (when available)
```

### Configuration Files
```powershell
# Create custom configuration directory
New-Item -ItemType Directory -Path "$env:USERPROFILE\.qcoin\config" -Force

# Create TOML configuration file
@"
[Eth]
NetworkId = 1337
DatabaseHandles = 1024
DatabaseCache = 1024
TrieCleanCache = 256
TrieCleanCacheJournal = "triecache"
TrieCleanCacheRejournal = 3600000000000
TrieDirtyCache = 256
TrieTimeout = 3600000000000
EnablePreimageRecording = false

[Node]
DataDir = "$env:USERPROFILE\.qcoin\testnet"
IPCPath = "geth.ipc"
HTTPHost = "127.0.0.1"
HTTPPort = 8545
HTTPCors = ["*"]
HTTPVirtualHosts = ["localhost"]
HTTPModules = ["eth", "net", "web3", "personal", "txpool"]
WSHost = "127.0.0.1"
WSPort = 8546
WSOrigins = ["*"]
WSModules = ["eth", "net", "web3"]

[Node.P2P]
MaxPeers = 50
NoDiscovery = false
BootstrapNodes = []
StaticNodes = []
TrustedNodes = []
ListenAddr = ":30303"
EnableMsgEvents = false
"@ | Out-File -FilePath "$env:USERPROFILE\.qcoin\config\geth.toml" -Encoding UTF8

# Use custom configuration
.\geth.exe --config "$env:USERPROFILE\.qcoin\config\geth.toml"
```

## 🌐 Networking

### Port Configuration
| Service | Default Port | Protocol | Purpose |
|---------|-------------|----------|---------|
| P2P | 30303 | TCP/UDP | Peer-to-peer networking |
| HTTP RPC | 8545 | TCP | JSON-RPC API |
| WebSocket | 8546 | TCP | WebSocket API |
| IPC | N/A | Named Pipe | Local IPC communication |

### Windows Firewall Setup
```powershell
# Add firewall rules for Q Geth
New-NetFirewallRule -DisplayName "Q Geth P2P TCP" -Direction Inbound -Protocol TCP -LocalPort 30303
New-NetFirewallRule -DisplayName "Q Geth P2P UDP" -Direction Inbound -Protocol UDP -LocalPort 30303
New-NetFirewallRule -DisplayName "Q Geth HTTP RPC" -Direction Inbound -Protocol TCP -LocalPort 8545
New-NetFirewallRule -DisplayName "Q Geth WebSocket" -Direction Inbound -Protocol TCP -LocalPort 8546

# Check firewall rules
Get-NetFirewallRule | Where-Object DisplayName -like "*Geth*"

# Check open ports
Get-NetTCPConnection | Where-Object {$_.LocalPort -in @(8545, 8546, 30303)}
```

### Router Configuration
```powershell
# For nodes behind router, configure port forwarding:
# Router Settings -> Port Forwarding:
# - Service Name: Q Geth P2P
# - External Port: 30303
# - Internal Port: 30303  
# - Internal IP: Your PC's local IP
# - Protocol: TCP and UDP

# Get your local IP address
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*"}

# Test external connectivity
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --nat "extip:YOUR_EXTERNAL_IP"

# Use UPnP for automatic port mapping
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --nat "upnp"
```

### Peer Discovery and Connectivity
```powershell
# Check peer count
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "net.peerCount"

# List connected peers
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "admin.peers"

# Add static peers
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec 'admin.addPeer("enode://NODEID@IP:PORT")'

# Check node info
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "admin.nodeInfo"
```

## 🔧 Operation

### Starting and Stopping
```powershell
# Start in background
Start-Process -FilePath ".\scripts\windows\start-geth.ps1" -ArgumentList "-Network testnet" -WindowStyle Hidden

# Start with custom options
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" `
  --networkid 1337 `
  --http --http.port 8545 `
  --port 30303 `
  --verbosity 3 `
  console

# Stop gracefully
# In console: exit
# Or stop process: Stop-Process -Name "geth" -Force
```

### Console Access
```powershell
# Attach to running node (IPC - fastest)
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc"

# Attach via HTTP
.\geth.exe attach http://localhost:8545

# JavaScript console examples
> eth.accounts
> eth.blockNumber
> net.peerCount
> personal.newAccount("password")
> eth.sendTransaction({from: eth.accounts[0], to: "0x...", value: web3.toWei(1, "ether")})
```

### Account Management
```powershell
# Create new account
.\geth.exe account new --datadir "$env:USERPROFILE\.qcoin\testnet"

# List accounts
.\geth.exe account list --datadir "$env:USERPROFILE\.qcoin\testnet"

# Import private key (create private_key.txt first)
.\geth.exe account import private_key.txt --datadir "$env:USERPROFILE\.qcoin\testnet"

# In console - unlock account
> personal.unlockAccount(eth.accounts[0], "password", 0)

# Check balance
> eth.getBalance(eth.accounts[0])
```

### Blockchain Operations
```powershell
# Check sync status
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.syncing"

# Get latest block
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.getBlock('latest')"

# Check chain ID
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.chainId()"

# Export/import blockchain data
.\geth.exe export blockchain.rlp --datadir "$env:USERPROFILE\.qcoin\testnet"
.\geth.exe import blockchain.rlp --datadir "$env:USERPROFILE\.qcoin\testnet"
```

## ⛏️ Mining Integration

### Solo Mining Setup
```powershell
# Start geth with mining enabled
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" `
  --networkid 1337 `
  --mine --miner.etherbase "0xYourCoinbaseAddress" `
  --miner.threads 1 `
  console

# Enable mining in console
> miner.setEtherbase(eth.accounts[0])
> miner.start(1)  # 1 thread
> miner.stop()
```

### External Miner Connection
```powershell
# Start geth as mining pool backend
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" `
  --http --http.addr "127.0.0.1" --http.port 8545 `
  --http.api "eth,net,web3,personal,txpool" `
  --allow-insecure-unlock

# Connect quantum-miner to geth
.\quantum-miner.exe --node http://localhost:8545 `
  --coinbase "0xYourAddress" `
  --threads 4
```

## 📊 Monitoring and Logging

### Logging Configuration
```powershell
# Start with specific log level
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --verbosity 4  # 0=silent, 5=debug

# Log to file
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" 2>&1 | Tee-Object -FilePath "geth.log"

# Structured logging
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --log.json 2>&1 | Tee-Object -FilePath "geth.json"

# View logs in real-time
Get-Content "geth.log" -Tail 20 -Wait
```

### Metrics and Monitoring
```powershell
# Enable metrics collection
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" `
  --metrics --metrics.addr "127.0.0.1" --metrics.port 6060

# View metrics in browser
Invoke-RestMethod -Uri "http://localhost:6060/debug/metrics"

# Performance monitoring
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
Get-Counter "\Memory\Available MBytes"
Get-Counter "\PhysicalDisk(_Total)\% Disk Time"

# Monitor blockchain sync
while ($true) {
    $blockNumber = .\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.blockNumber" 2>$null
    Write-Host "Current block: $blockNumber"
    Start-Sleep 5
}
```

### Health Checks
```powershell
# RPC health check
Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'

# Sync status check
Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}'

# Peer count check
Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}'
```

## 🔄 Service Management

### Windows Service Setup (with NSSM)
```powershell
# Download NSSM (Non-Sucking Service Manager)
# From: https://nssm.cc/download

# Install Q Geth as Windows service
nssm install "Q Geth Node" "$PWD\scripts\windows\start-geth.ps1"
nssm set "Q Geth Node" AppParameters "-Network testnet"
nssm set "Q Geth Node" AppDirectory "$PWD"
nssm set "Q Geth Node" DisplayName "Q Coin Geth Node"
nssm set "Q Geth Node" Description "Q Coin quantum blockchain node"
nssm set "Q Geth Node" Start SERVICE_AUTO_START

# Configure service logging
nssm set "Q Geth Node" AppStdout "$env:USERPROFILE\.qcoin\logs\geth-stdout.log"
nssm set "Q Geth Node" AppStderr "$env:USERPROFILE\.qcoin\logs\geth-stderr.log"

# Start the service
nssm start "Q Geth Node"

# Check service status
nssm status "Q Geth Node"
Get-Service "Q Geth Node"
```

### Task Scheduler (Alternative)
```powershell
# Create scheduled task for auto-start
$Action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File `"$PWD\scripts\windows\start-geth.ps1`" -Network testnet"
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "Q Geth Node" -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings

# Start the task
Start-ScheduledTask -TaskName "Q Geth Node"

# Check task status
Get-ScheduledTask -TaskName "Q Geth Node"
```

## 🔒 Security

### Network Security
```powershell
# Bind to localhost only (default)
.\geth.exe --http.addr "127.0.0.1" --ws.addr "127.0.0.1"

# Restrict API access
.\geth.exe --http.api "eth,net,web3"  # Remove personal, admin

# Use allowlist for CORS
.\geth.exe --http.corsdomain "https://your-dapp.com"

# Disable dangerous APIs
.\geth.exe --http.api "eth,net,web3"  # No debug, admin, personal
```

### Account Security
```powershell
# Backup keystore files
Copy-Item -Path "$env:USERPROFILE\.qcoin\testnet\keystore" -Destination "$env:USERPROFILE\keystore-backup-$(Get-Date -Format 'yyyyMMdd')" -Recurse

# Use strong passwords
# Generate secure password
[System.Web.Security.Membership]::GeneratePassword(32, 8)

# Lock accounts after use
.\geth.exe attach --exec "personal.lockAccount(eth.accounts[0])"
```

### File Permissions
```powershell
# Secure data directory (limit access to current user)
icacls "$env:USERPROFILE\.qcoin" /inheritance:d
icacls "$env:USERPROFILE\.qcoin" /grant "$env:USERNAME:(OI)(CI)F" /T
icacls "$env:USERPROFILE\.qcoin" /remove "Users" /T

# Check permissions
icacls "$env:USERPROFILE\.qcoin"
```

### Windows Defender Exclusions
```powershell
# Add Windows Defender exclusions for better performance
Add-MpPreference -ExclusionPath "$PWD"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.qcoin"
Add-MpPreference -ExclusionProcess "geth.exe"
Add-MpPreference -ExclusionProcess "quantum-miner.exe"

# Check exclusions
Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
Get-MpPreference | Select-Object -ExpandProperty ExclusionProcess
```

## 🛠️ Maintenance

### Regular Maintenance
```powershell
# Update Q Geth
Set-Location Qgeth3
git pull origin main
.\scripts\windows\build-release.ps1

# Restart service if using NSSM
nssm restart "Q Geth Node"

# Clean up disk space
# Remove old logs
Get-ChildItem "$env:USERPROFILE\.qcoin" -Filter "*.log" | Where-Object LastWriteTime -LT (Get-Date).AddDays(-30) | Remove-Item

# Compact database (requires stopping node)
Stop-Service "Q Geth Node" -ErrorAction SilentlyContinue
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" removedb
Start-Service "Q Geth Node" -ErrorAction SilentlyContinue

# Backup blockchain data
Compress-Archive -Path "$env:USERPROFILE\.qcoin\testnet" -DestinationPath "blockchain-backup-$(Get-Date -Format 'yyyyMMdd').zip"
```

### Performance Optimization
```powershell
# Optimize Windows for blockchain performance
# Disable Windows Search indexing for blockchain data
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows Search\CrawlScopeManager\Windows\SystemIndex\WorkingSetRules" -Name "$env:USERPROFILE\.qcoin" -Value 0

# Increase database cache
.\geth.exe --cache 2048  # Increase cache size

# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Optimize network settings
netsh int tcp set global autotuninglevel=normal
netsh int tcp set global chimney=enabled
```

## 📚 Advanced Usage

### Multiple Network Setup
```powershell
# Run multiple networks simultaneously
Start-Process -FilePath ".\geth.exe" -ArgumentList "--datadir", "$env:USERPROFILE\.qcoin\testnet", "--port", "30303", "--http.port", "8545"
Start-Process -FilePath ".\geth.exe" -ArgumentList "--datadir", "$env:USERPROFILE\.qcoin\devnet", "--port", "30304", "--http.port", "8546"

# Use different IPC paths
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --ipcpath "\\.\pipe\geth-testnet"
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\devnet" --ipcpath "\\.\pipe\geth-devnet"
```

### PowerShell Integration Examples
```powershell
# PowerShell Web3 integration
$web3Endpoint = "http://localhost:8545"
$jsonRpcRequest = @{
    jsonrpc = "2.0"
    method = "eth_blockNumber"
    params = @()
    id = 1
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri $web3Endpoint -Method Post -ContentType "application/json" -Body $jsonRpcRequest
$blockNumber = [Convert]::ToInt64($response.result, 16)
Write-Host "Current block number: $blockNumber"

# Monitor node status
function Get-QGethStatus {
    try {
        $version = Invoke-RestMethod -Uri $web3Endpoint -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'
        $blockNumber = Invoke-RestMethod -Uri $web3Endpoint -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
        $peerCount = Invoke-RestMethod -Uri $web3Endpoint -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}'
        
        [PSCustomObject]@{
            Version = $version.result
            BlockNumber = [Convert]::ToInt64($blockNumber.result, 16)
            PeerCount = [Convert]::ToInt64($peerCount.result, 16)
            Status = "Running"
        }
    }
    catch {
        [PSCustomObject]@{
            Status = "Stopped"
            Error = $_.Exception.Message
        }
    }
}

# Use the function
Get-QGethStatus
```

## 📖 Reference

### PowerShell Command Examples
```powershell
# View all geth options
.\geth.exe help

# Common startup patterns
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --networkid 1337 console
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --http --http.api "eth,net,web3"
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --mine --miner.etherbase "0xYourAddress"

# Useful PowerShell aliases
Set-Alias qgeth ".\geth.exe"
Set-Alias qminer ".\quantum-miner.exe"
```

### Useful PowerShell Scripts
```powershell
# Quick node status check script
function Check-QGethNode {
    $process = Get-Process -Name "geth" -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "Q Geth is running (PID: $($process.Id))"
        try {
            $status = Get-QGethStatus
            Write-Host "Block: $($status.BlockNumber), Peers: $($status.PeerCount)"
        }
        catch {
            Write-Host "Node running but RPC not responding"
        }
    }
    else {
        Write-Host "Q Geth is not running"
    }
}

# Create desktop shortcut
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Q Geth Console.lnk")
$Shortcut.TargetPath = "PowerShell.exe"
$Shortcut.Arguments = "-Command `"Set-Location '$PWD'; .\geth.exe attach '$env:USERPROFILE\.qcoin\testnet\geth.ipc'`""
$Shortcut.WorkingDirectory = $PWD
$Shortcut.Save()
```

For troubleshooting Windows geth issues, see [Windows Geth Troubleshooting](troubleshooting-windows-geth.md).

---

**Happy quantum blockchain exploring with Q Geth on Windows! 🪟⚡** 