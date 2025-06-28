# Windows Geth Troubleshooting

Solutions for Q Coin geth node issues on Windows systems.

## 🔧 Quick Geth Diagnostics (PowerShell)

### Node Status Check
```powershell
# Check if geth is running
Get-Process -Name "*geth*" -ErrorAction SilentlyContinue
Get-Process | Where-Object ProcessName -like "*geth*"

# Check geth process details
Get-Process -Name "geth" | Select-Object Id, ProcessName, CPU, WorkingSet, PagedMemorySize

# Quick health check via RPC
$response = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' -ErrorAction SilentlyContinue
if ($response) { Write-Host "Geth is responding: $($response.result)" } else { Write-Host "Geth is not responding" }

# Check Windows Event Logs
Get-WinEvent -LogName Application | Where-Object {$_.LevelDisplayName -eq "Error" -and $_.Message -like "*geth*"} | Select-Object -First 5
```

## 🚀 Node Startup Issues

### Geth Binary Not Found
```powershell
# Symptoms: "command not found", "file not found"
# Solution: Check binary location and permissions

# Verify binary exists
Get-ChildItem geth.exe
Test-Path .\geth.exe

# Check file properties
Get-ItemProperty .\geth.exe
Get-Command .\geth.exe -ErrorAction SilentlyContinue

# Check if binary is blocked by Windows
Get-Item .\geth.exe | Select-Object -ExpandProperty Properties
Unblock-File .\geth.exe  # If file is blocked

# Rebuild if necessary
.\scripts\windows\build-release.ps1
```

### PowerShell Execution Policy Issues
```powershell
# Check current execution policy
Get-ExecutionPolicy

# If scripts can't run, change policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run specific script with bypass
PowerShell -ExecutionPolicy Bypass -File .\scripts\windows\start-geth.ps1

# Check if script is signed
Get-AuthenticodeSignature .\scripts\windows\start-geth.ps1
```

### Genesis Block Initialization Issues
```powershell
# Symptoms: "genesis file not found", "invalid genesis"
# Solution: Verify genesis file and initialization

# Check genesis file exists
Test-Path configs\genesis_quantum_testnet.json
Get-Content configs\genesis_quantum_testnet.json | ConvertFrom-Json  # Validate JSON

# Initialize datadir with genesis (if needed)
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" init configs\genesis_quantum_testnet.json

# Check if genesis was properly initialized
Get-ChildItem "$env:USERPROFILE\.qcoin\testnet\geth\chaindata"

# If genesis mismatch, remove and reinitialize
Remove-Item "$env:USERPROFILE\.qcoin\testnet\geth\chaindata" -Recurse -Force
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" init configs\genesis_quantum_testnet.json
```

### Data Directory Issues
```powershell
# Check data directory permissions
Get-Acl "$env:USERPROFILE\.qcoin"
Get-ChildItem "$env:USERPROFILE\.qcoin" -Force

# Check disk space
Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object DeviceID -eq "C:"
Get-ChildItem "$env:USERPROFILE\.qcoin" | Measure-Object -Property Length -Sum

# If permission denied on IPC
icacls "$env:USERPROFILE\.qcoin" /grant "$env:USERNAME:(OI)(CI)F"

# Check for Windows Defender blocking
Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
```

## 🌐 Networking Issues

### Port Binding Failures
```powershell
# Symptoms: "bind: address already in use"
# Solution: Find and resolve port conflicts

# Check what's using geth ports
Get-NetTCPConnection | Where-Object LocalPort -in @(8545, 30303)
netstat -an | findstr "8545\|30303"

# Find process using specific port
Get-Process -Id (Get-NetTCPConnection -LocalPort 8545).OwningProcess -ErrorAction SilentlyContinue

# Kill conflicting processes
Stop-Process -Name "geth" -Force -ErrorAction SilentlyContinue
Get-Process | Where-Object ProcessName -like "*geth*" | Stop-Process -Force

# Use different ports if needed
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --http.port 8546 --port 30304

# Test port availability
Test-NetConnection -ComputerName localhost -Port 8545
```

### Windows Firewall Blocking
```powershell
# Check firewall status
Get-NetFirewallProfile | Select-Object Name, Enabled

# Check existing firewall rules
Get-NetFirewallRule | Where-Object DisplayName -like "*geth*"

# Add firewall rules for geth
New-NetFirewallRule -DisplayName "Q Geth P2P TCP" -Direction Inbound -Protocol TCP -LocalPort 30303
New-NetFirewallRule -DisplayName "Q Geth P2P UDP" -Direction Inbound -Protocol UDP -LocalPort 30303
New-NetFirewallRule -DisplayName "Q Geth HTTP RPC" -Direction Inbound -Protocol TCP -LocalPort 8545

# Test local connectivity
Test-NetConnection -ComputerName localhost -Port 8545
Test-NetConnection -ComputerName localhost -Port 30303

# Disable firewall temporarily for testing (NOT recommended for production)
Set-NetFirewallProfile -All -Enabled False
```

### P2P Connectivity Issues
```powershell
# Check peer count
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "net.peerCount"

# Check listening status
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "net.listening"

# Check node info
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "admin.nodeInfo"

# If no peers, check network connectivity
Test-Connection -ComputerName 8.8.8.8 -Count 4
nslookup github.com

# Check NAT settings
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --nat "upnp"
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --nat "extip:YOUR_EXTERNAL_IP"

# Get external IP
Invoke-RestMethod -Uri "https://api.ipify.org"
```

## 🔄 Sync Issues

### Blockchain Sync Problems
```powershell
# Check sync status
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.syncing"

# Check current block vs latest
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.blockNumber"

# Check peer sync info
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "admin.peers"

# If sync is stuck, restart with fresh database
Stop-Service "Q Geth Node" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.qcoin\testnet\geth\chaindata" -Recurse -Force
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" init configs\genesis_quantum_testnet.json
Start-Service "Q Geth Node" -ErrorAction SilentlyContinue
```

### Database Corruption
```powershell
# Symptoms: "database corruption", "bad block", "leveldb error"
# Solution: Rebuild database

# Stop geth
Stop-Process -Name "geth" -Force -ErrorAction SilentlyContinue
Stop-Service "Q Geth Node" -ErrorAction SilentlyContinue

# Backup corrupted data (optional)
Copy-Item "$env:USERPROFILE\.qcoin\testnet" "$env:USERPROFILE\.qcoin\testnet.backup" -Recurse

# Remove corrupted database
Remove-Item "$env:USERPROFILE\.qcoin\testnet\geth\chaindata" -Recurse -Force

# Reinitialize and restart
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" init configs\genesis_quantum_testnet.json
.\scripts\windows\start-geth.ps1 -Network testnet
```

### Sync Performance Issues
```powershell
# Monitor sync progress
while ($true) {
    $blockNumber = .\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.blockNumber" 2>$null
    Write-Host "Current block: $blockNumber"
    Start-Sleep 5
}

# Check system resources during sync
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
Get-Counter "\Memory\Available MBytes"
Get-Counter "\PhysicalDisk(_Total)\% Disk Time"

# Optimize cache settings
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --cache 2048

# Check disk I/O performance
$testFile = "test.tmp"
$data = New-Object byte[] 1MB
Measure-Command { [System.IO.File]::WriteAllBytes($testFile, $data) }
Remove-Item $testFile
```

## 💬 Console and IPC Issues

### Console Access Problems
```powershell
# Symptoms: "connection refused", "no such file"
# Solution: Check IPC pipe

# Verify IPC pipe exists
Test-Path "$env:USERPROFILE\.qcoin\testnet\geth.ipc"

# Check if geth is running
Get-Process -Name "geth" -ErrorAction SilentlyContinue

# Try HTTP attachment instead
.\geth.exe attach http://localhost:8545

# If HTTP also fails, check recent Windows Event logs
Get-WinEvent -LogName Application | Where-Object {$_.ProviderName -eq "geth" -or $_.Message -like "*geth*"} | Select-Object -First 5
```

### Console Commands Failing
```powershell
# In geth console, check basic functionality
> web3.version
> eth.accounts
> net.peerCount

# If commands fail, check API availability
> web3.admin  # Should show admin functions
> personal    # Should show personal functions

# Common fixes:
# 1. Restart geth with proper APIs enabled
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --http.api "eth,net,web3,personal,admin,txpool"

# 2. Check if account is unlocked
> personal.listAccounts
> personal.unlockAccount(eth.accounts[0], "password")
```

### Account Management Issues
```powershell
# Account creation problems
.\geth.exe account new --datadir "$env:USERPROFILE\.qcoin\testnet"

# If account creation fails, check keystore directory
Test-Path "$env:USERPROFILE\.qcoin\testnet\keystore"
New-Item -ItemType Directory -Path "$env:USERPROFILE\.qcoin\testnet\keystore" -Force

# List existing accounts
.\geth.exe account list --datadir "$env:USERPROFILE\.qcoin\testnet"

# Import account issues
.\geth.exe account import private_key.txt --datadir "$env:USERPROFILE\.qcoin\testnet"

# Check keystore file format
Get-Content "$env:USERPROFILE\.qcoin\testnet\keystore\UTC--*"
```

## ⛏️ Mining Integration Issues

### Mining Not Starting
```powershell
# Check if mining is enabled
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.mining"

# Check coinbase address
.\geth.exe attach "$env:USERPROFILE\.qcoin\testnet\geth.ipc" --exec "eth.coinbase"

# Set coinbase and start mining
> miner.setEtherbase(eth.accounts[0])
> miner.start(1)

# Check mining status
> eth.hashrate
> eth.mining
```

### External Miner Connection
```powershell
# Test if geth RPC is accessible for mining
Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"eth_getWork","params":[],"id":1}'

# If connection refused, check geth startup
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --http --http.addr "127.0.0.1" --http.port 8545 --http.api "eth,net,web3,personal,txpool" --allow-insecure-unlock

# Test miner connectivity
.\quantum-miner.exe --node http://localhost:8545 --test

# Check miner logs for connection issues
.\quantum-miner.exe --node http://localhost:8545 --verbose
```

## 📊 Performance Issues

### High CPU Usage
```powershell
# Monitor geth CPU usage
Get-Process -Name "geth" | Select-Object Id, ProcessName, CPU, WorkingSet

# Monitor CPU over time
Get-Counter "\Process(geth)\% Processor Time" -SampleInterval 5 -MaxSamples 12

# Check verbosity level (lower = less CPU)
.\geth.exe --verbosity 2  # Reduce from default 3

# Optimize cache settings
.\geth.exe --cache 1024

# Check for excessive logging
Get-ChildItem "*.log" | Measure-Object Length -Sum
```

### High Memory Usage
```powershell
# Monitor memory usage
Get-Process -Name "geth" | Select-Object Id, ProcessName, WorkingSet, PagedMemorySize, VirtualMemorySize

# Check total system memory
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory

# Reduce cache if needed
.\geth.exe --cache 512

# Monitor memory over time
while ($true) {
    Get-Process -Name "geth" | Select-Object WorkingSet, PagedMemorySize
    Start-Sleep 10
}
```

### Disk I/O Issues
```powershell
# Monitor disk usage
Get-Counter "\PhysicalDisk(_Total)\% Disk Time"
Get-Counter "\PhysicalDisk(_Total)\Disk Bytes/sec"

# Check disk space
Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object DeviceID -eq "C:"
Get-ChildItem "$env:USERPROFILE\.qcoin" | Measure-Object Length -Sum

# Optimize database settings
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --cache.database 75

# Move data to faster disk (if available)
Stop-Service "Q Geth Node" -ErrorAction SilentlyContinue
Move-Item "$env:USERPROFILE\.qcoin" "D:\.qcoin"
New-Item -ItemType SymbolicLink -Path "$env:USERPROFILE\.qcoin" -Target "D:\.qcoin"
Start-Service "Q Geth Node" -ErrorAction SilentlyContinue
```

## 🔧 Configuration Issues

### Invalid Command Line Parameters
```powershell
# Check available options
.\geth.exe help

# Verify parameter syntax
.\geth.exe --help | Select-String "your-option"

# Common parameter fixes:
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet"  # Use quotes for paths with spaces
.\geth.exe --http.api "eth,net,web3"   # Use quotes for multiple APIs
.\geth.exe --bootnodes "enode://..."  # Use quotes for node IDs
```

### TOML Configuration Issues
```powershell
# Validate TOML configuration
.\geth.exe dumpconfig | Out-File default.toml
Compare-Object (Get-Content default.toml) (Get-Content "$env:USERPROFILE\.qcoin\config\geth.toml")

# Test configuration file
.\geth.exe --config "$env:USERPROFILE\.qcoin\config\geth.toml" --help

# Check for syntax errors (requires PowerShell TOML module)
# Install-Module -Name powershell-toml
Import-Module powershell-toml
ConvertFrom-Toml (Get-Content "$env:USERPROFILE\.qcoin\config\geth.toml" -Raw)
```

## 🔄 Windows Service Issues

### NSSM Service Problems
```powershell
# Check if NSSM is installed
Get-Command nssm -ErrorAction SilentlyContinue

# Check service status
nssm status "Q Geth Node"
Get-Service "Q Geth Node"

# View service logs
nssm get "Q Geth Node" AppStdout
nssm get "Q Geth Node" AppStderr

# Restart service
nssm restart "Q Geth Node"

# Remove and recreate service if broken
nssm remove "Q Geth Node" confirm
# Then recreate with install commands
```

### Windows Event Log Analysis
```powershell
# Check Windows Event Logs for errors
Get-WinEvent -LogName Application | Where-Object {$_.LevelDisplayName -eq "Error"} | Select-Object -First 10

# Check system logs
Get-WinEvent -LogName System | Where-Object {$_.LevelDisplayName -eq "Error"} | Select-Object -First 10

# Filter for geth-related events
Get-WinEvent -LogName Application | Where-Object {$_.Message -like "*geth*" -or $_.ProcessId -eq (Get-Process geth).Id}

# Export event logs for analysis
Get-WinEvent -LogName Application | Where-Object {$_.TimeCreated -gt (Get-Date).AddHours(-1)} | Export-Csv "geth-events.csv"
```

### Task Scheduler Issues
```powershell
# Check scheduled task status
Get-ScheduledTask -TaskName "Q Geth Node"

# View task history
Get-ScheduledTaskInfo -TaskName "Q Geth Node"

# Start/stop task
Start-ScheduledTask -TaskName "Q Geth Node"
Stop-ScheduledTask -TaskName "Q Geth Node"

# Check task definition
Export-ScheduledTask -TaskName "Q Geth Node"
```

## 🚨 Emergency Recovery

### Complete Node Reset
```powershell
# Stop all geth processes
Stop-Process -Name "geth" -Force -ErrorAction SilentlyContinue
Stop-Service "Q Geth Node" -ErrorAction SilentlyContinue

# Backup important data
Copy-Item "$env:USERPROFILE\.qcoin\testnet\keystore" "$env:USERPROFILE\keystore-backup" -Recurse
Copy-Item "$env:USERPROFILE\.qcoin\config\geth.toml" "$env:USERPROFILE\geth.toml.backup"

# Remove all blockchain data
Remove-Item "$env:USERPROFILE\.qcoin\testnet\geth" -Recurse -Force

# Reinitialize
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" init configs\genesis_quantum_testnet.json

# Restore keystore
Copy-Item "$env:USERPROFILE\keystore-backup\*" "$env:USERPROFILE\.qcoin\testnet\keystore\" -Force

# Restart node
.\scripts\windows\start-geth.ps1 -Network testnet
```

### Log Analysis
```powershell
# Analyze geth logs for errors (if log files exist)
Select-String -Path "geth.log" -Pattern "error|fatal|panic" | Select-Object -Last 20

# Check for specific issues
Select-String -Path "geth.log" -Pattern "database|network|sync"

# Monitor logs in real-time (if log file exists)
Get-Content "geth.log" -Tail 20 -Wait | Where-Object {$_ -match "ERROR|WARN|FATAL"}

# If no log files, check Windows Event Logs
Get-WinEvent -LogName Application | Where-Object {$_.Message -like "*geth*" -and $_.LevelDisplayName -eq "Error"}
```

## 📚 Advanced Debugging

### Debug Logging
```powershell
# Enable debug logging
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --verbosity 5

# Structured JSON logging
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --log.json 2>&1 | Tee-Object -FilePath "geth.json"

# Log to file
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" 2>&1 | Tee-Object -FilePath "geth.log"

# View logs in real-time
Get-Content "geth.log" -Tail 20 -Wait
```

### Network Debugging
```powershell
# Test P2P connectivity
Test-NetConnection -ComputerName PEER_IP -Port 30303

# Check routing
Test-NetConnection -ComputerName PEER_IP -TraceRoute

# Monitor network traffic
Get-NetTCPConnection | Where-Object LocalPort -eq 30303
netstat -an | findstr "30303"

# Debug RPC calls
Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"admin_nodeInfo","params":[],"id":1}'
```

### Performance Profiling
```powershell
# Enable metrics collection
.\geth.exe --datadir "$env:USERPROFILE\.qcoin\testnet" --metrics --metrics.addr "127.0.0.1" --metrics.port 6060

# View metrics
Invoke-RestMethod -Uri "http://localhost:6060/debug/metrics"

# Monitor system performance
Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 5 -MaxSamples 12
Get-Counter "\Memory\Available MBytes" -SampleInterval 5 -MaxSamples 12
```

## 📋 Diagnostic Information Collection

### System Diagnostics
```powershell
# Collect comprehensive system info
$DiagFile = "windows-geth-diagnostics.txt"

"=== Windows Geth Diagnostics ===" | Out-File $DiagFile
"Date: $(Get-Date)" | Out-File $DiagFile -Append
"Computer: $env:COMPUTERNAME" | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"=== System Info ===" | Out-File $DiagFile -Append
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, Architecture | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"=== Geth Process ===" | Out-File $DiagFile -Append
Get-Process -Name "*geth*" | Select-Object Id, ProcessName, CPU, WorkingSet | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"=== Network ===" | Out-File $DiagFile -Append
Get-NetTCPConnection | Where-Object {$_.LocalPort -in @(8545, 30303)} | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"=== Disk Space ===" | Out-File $DiagFile -Append
Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID, Size, FreeSpace | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"=== Firewall Rules ===" | Out-File $DiagFile -Append
Get-NetFirewallRule | Where-Object DisplayName -like "*geth*" | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"=== Recent Events ===" | Out-File $DiagFile -Append
Get-WinEvent -LogName Application | Where-Object {$_.Message -like "*geth*"} | Select-Object -First 10 | Out-File $DiagFile -Append

Get-Content $DiagFile
```

## ✅ Windows Geth Checklist

### Pre-Startup
- [ ] Geth binary exists and is not blocked
- [ ] PowerShell execution policy allows scripts
- [ ] Genesis file properly configured
- [ ] Data directory has correct permissions
- [ ] Required ports are available

### Runtime
- [ ] Node starts without errors
- [ ] Windows Firewall configured correctly
- [ ] P2P networking functional
- [ ] Blockchain syncing properly
- [ ] RPC API accessible

### Performance
- [ ] System resources not overloaded
- [ ] Peer connections stable
- [ ] Sync progress reasonable
- [ ] No database corruption
- [ ] Windows Defender exclusions set

**For most Windows geth issues, checking Windows Event Logs and firewall settings resolves the problem!** 