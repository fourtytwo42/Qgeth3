# Windows Q Coin Deployment Troubleshooting

Comprehensive troubleshooting guide for Q Coin deployment issues on Windows systems.

## 🚨 Service Installation Issues

### NSSM Service Manager Problems

#### Error: `'nssm' is not recognized as an internal or external command`
**Problem:** NSSM not installed or not in PATH.

**Solutions:**
```powershell
# Download and install NSSM
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "nssm.zip"
Expand-Archive -Path "nssm.zip" -DestinationPath "C:\Tools"
$env:PATH += ";C:\Tools\nssm-2.24\win64"

# Add to permanent PATH
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
[System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\Tools\nssm-2.24\win64", "Machine")

# Verify installation
nssm version
```

#### Error: `Access is denied` when installing service
**Problem:** Insufficient privileges to install Windows services.

**Solutions:**
```powershell
# Run PowerShell as Administrator
Start-Process PowerShell -Verb RunAs

# Check current user privileges
whoami /priv | findstr SeServiceLogonRight

# Install service with proper admin rights
nssm install QGethNode "C:\QCoin\geth.exe"
```

#### Error: `The specified service already exists`
**Problem:** Service with same name already installed.

**Solutions:**
```powershell
# Check existing services
Get-Service | Where-Object {$_.Name -like "*QGeth*" -or $_.Name -like "*QMiner*"}

# Stop and remove existing service
Stop-Service QGethNode -Force -ErrorAction SilentlyContinue
nssm remove QGethNode confirm

# Reinstall service
nssm install QGethNode "C:\QCoin\geth.exe"
```

### Service Configuration Problems

#### Error: Service starts but immediately stops
**Problem:** Service configuration issues or binary problems.

**Solutions:**
```powershell
# Check service status and last error
Get-Service QGethNode
Get-WinEvent -LogName Application | Where-Object {$_.Source -eq "QGethNode"} | Select-Object -First 5

# Check NSSM service configuration
nssm get QGethNode Application
nssm get QGethNode Parameters
nssm get QGethNode AppDirectory

# Test binary manually
Set-Location "C:\QCoin"
.\geth.exe --help

# Check service logs
Get-Content "C:\QCoin\logs\geth-stdout.log" -Tail 20
Get-Content "C:\QCoin\logs\geth-stderr.log" -Tail 20
```

#### Error: `The service did not respond to the start or control request in a timely fashion`
**Problem:** Service taking too long to start or hanging.

**Solutions:**
```powershell
# Increase service timeout
nssm set QGethNode AppThrottle 5000  # 5 seconds

# Check if binary is responsive
$job = Start-Job -ScriptBlock { & "C:\QCoin\geth.exe" --help }
Wait-Job $job -Timeout 30
Receive-Job $job
Remove-Job $job

# Simplify service parameters for testing
nssm set QGethNode Parameters "--help"
Start-Service QGethNode
Start-Sleep 5
Stop-Service QGethNode

# Restore proper parameters
nssm set QGethNode Parameters "--http --http.addr 0.0.0.0 --http.port 8545"
```

## 🔐 Security and Permissions Issues

### Windows Firewall Problems

#### Error: Connection timeouts to geth ports
**Problem:** Windows Firewall blocking connections.

**Solutions:**
```powershell
# Check current firewall rules
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*Q*Geth*"}

# Add firewall rules for Q Geth
New-NetFirewallRule -DisplayName "Q Geth HTTP RPC" -Direction Inbound -Protocol TCP -LocalPort 8545 -Action Allow
New-NetFirewallRule -DisplayName "Q Geth WebSocket" -Direction Inbound -Protocol TCP -LocalPort 8546 -Action Allow
New-NetFirewallRule -DisplayName "Q Geth P2P" -Direction Inbound -Protocol TCP -LocalPort 30303 -Action Allow

# Test port connectivity
Test-NetConnection -ComputerName localhost -Port 8545
```

#### Error: External connections rejected
**Problem:** Services binding to localhost only.

**Solutions:**
```powershell
# Check service parameters
nssm get QGethNode Parameters

# Ensure binding to all interfaces
nssm set QGethNode Parameters "--http --http.addr 0.0.0.0 --http.port 8545 --ws --ws.addr 0.0.0.0 --ws.port 8546"

# Restart service
Restart-Service QGethNode

# Verify binding
netstat -an | findstr "8545"
netstat -an | findstr "8546"
```

### File System Permissions

#### Error: `Access to the path 'C:\QCoin\data' is denied`
**Problem:** Service account lacks permissions to data directory.

**Solutions:**
```powershell
# Check current permissions
Get-Acl "C:\QCoin"

# Grant full control to service account
$acl = Get-Acl "C:\QCoin"
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("qcoin-service","FullControl","ContainerInherit,ObjectInherit","None","Allow")
$acl.SetAccessRule($accessRule)
Set-Acl -Path "C:\QCoin" -AclObject $acl

# Alternative: Grant permissions to NETWORK SERVICE
$networkServiceRule = New-Object System.Security.AccessControl.FileSystemAccessRule("NETWORK SERVICE","FullControl","ContainerInherit,ObjectInherit","None","Allow")
$acl.SetAccessRule($networkServiceRule)
Set-Acl -Path "C:\QCoin" -AclObject $acl
```

#### Error: `The process cannot access the file because it is being used by another process`
**Problem:** File locking issues during service operation.

**Solutions:**
```powershell
# Find processes using Q Coin files
$qcoinProcesses = Get-Process | Where-Object {$_.Path -like "*QCoin*"}
$qcoinProcesses | Format-Table Name, Id, Path

# Check file handles (requires handle.exe from Sysinternals)
# handle.exe C:\QCoin

# Stop services gracefully
Stop-Service QGethNode -Force
Stop-Service QMiner -Force

# Wait for processes to terminate
Start-Sleep 10

# Check for remaining processes
Get-Process geth -ErrorAction SilentlyContinue
Get-Process quantum-miner -ErrorAction SilentlyContinue
```

## 🌐 Network and Connectivity Issues

### Port Binding Problems

#### Error: `bind: address already in use`
**Problem:** Port already occupied by another service.

**Solutions:**
```powershell
# Check what's using the port
netstat -ano | findstr "8545"
netstat -ano | findstr "30303"

# Find process by PID
$pid = (netstat -ano | findstr "8545" | ForEach-Object {($_ -split "\s+")[4]}) | Select-Object -First 1
Get-Process -Id $pid

# Kill conflicting process if safe
Stop-Process -Id $pid -Force

# Change Q Geth ports if needed
nssm set QGethNode Parameters "--http --http.port 8555 --ws --ws.port 8556"
```

#### Error: `connection refused` from external clients
**Problem:** Service not listening on external interfaces.

**Solutions:**
```powershell
# Check listening addresses
netstat -an | findstr "LISTENING" | findstr "8545"

# Ensure binding to all interfaces
nssm get QGethNode Parameters
# Should include: --http.addr 0.0.0.0

# Check Windows Firewall
Get-NetFirewallRule -DisplayName "*Q Geth*"

# Test from external machine
# telnet your-server-ip 8545
```

### DNS and Hostname Issues

#### Error: Unable to resolve hostnames in configuration
**Problem:** DNS resolution problems.

**Solutions:**
```powershell
# Test DNS resolution
nslookup pool.qcoin.org
Resolve-DnsName pool.qcoin.org

# Check DNS configuration
Get-DnsClientServerAddress

# Use IP addresses instead of hostnames
nssm set QMiner Parameters "--pool stratum+tcp://192.168.1.100:4444"

# Flush DNS cache
ipconfig /flushdns
```

## 🔧 Performance and Resource Issues

### Memory Problems

#### Error: `Out of memory` or service crashes
**Problem:** Insufficient memory allocation.

**Solutions:**
```powershell
# Check system memory
Get-WmiObject Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-Counter "\Memory\Available MBytes"

# Monitor service memory usage
Get-Process geth | Select-Object ProcessName, WorkingSet64, VirtualMemorySize64

# Increase virtual memory (page file)
# Control Panel -> System -> Advanced -> Performance Settings -> Advanced -> Virtual Memory

# Set process priority
nssm set QGethNode AppPriority BELOW_NORMAL_PRIORITY_CLASS

# Limit geth cache size
nssm set QGethNode Parameters "--cache 1024"  # 1GB cache
```

#### Error: Service becomes unresponsive over time
**Problem:** Memory leaks or resource exhaustion.

**Solutions:**
```powershell
# Monitor service over time
while ($true) {
    $proc = Get-Process geth -ErrorAction SilentlyContinue
    if ($proc) {
        $mem = [math]::Round($proc.WorkingSet64/1MB, 2)
        $cpu = $proc.CPU
        Write-Host "$(Get-Date): Memory: ${mem}MB, CPU: ${cpu}s"
    }
    Start-Sleep 60
}

# Set service restart policy
nssm set QGethNode AppExit Default Restart
nssm set QGethNode AppRestartDelay 30000
nssm set QGethNode AppThrottle 5000

# Schedule automatic restart
$action = New-ScheduledTaskAction -Execute "Restart-Service" -Argument "QGethNode"
$trigger = New-ScheduledTaskTrigger -Daily -At "03:00"
Register-ScheduledTask -TaskName "QGethNodeRestart" -Action $action -Trigger $trigger
```

### GPU Mining Issues

#### Error: GPU miner not detecting graphics cards
**Problem:** GPU drivers or CUDA issues.

**Solutions:**
```powershell
# Check GPU status
Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check miner configuration
nssm get QMiner Parameters
nssm get QMiner AppEnvironmentExtra

# Test GPU directly
Set-Location "C:\QCoin"
.\quantum-miner.exe --gpu --test

# Update GPU drivers
# Download from https://www.nvidia.com/drivers/
```

#### Error: Mining performance degradation
**Problem:** Thermal throttling or power limits.

**Solutions:**
```powershell
# Monitor GPU temperature and clocks
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,power.draw --format=csv -l 1

# Check for thermal throttling
nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active --format=csv -l 1

# Adjust power limits
nvidia-smi -pl 300  # Set 300W power limit

# Monitor mining performance
.\quantum-miner.exe --stats --interval 30s
```

## 📊 Monitoring and Logging Issues

### Event Log Problems

#### Error: Services not logging to Windows Event Log
**Problem:** Event log sources not properly configured.

**Solutions:**
```powershell
# Create event log sources (requires admin)
New-EventLog -LogName "QCoin" -Source "QGethNode"
New-EventLog -LogName "QCoin" -Source "QMiner"

# Check existing event logs
Get-EventLog -List | Where-Object {$_.Log -like "*QCoin*"}

# Configure NSSM to use event log
nssm set QGethNode AppEvents 1

# View recent events
Get-WinEvent -LogName "QCoin" | Select-Object -First 10
```

#### Error: Log files growing too large
**Problem:** Log rotation not configured properly.

**Solutions:**
```powershell
# Configure log rotation in NSSM
nssm set QGethNode AppRotateFiles 1
nssm set QGethNode AppRotateOnline 1
nssm set QGethNode AppRotateBytes 10485760  # 10MB

# Manual log rotation script
$logDir = "C:\QCoin\logs"
Get-ChildItem $logDir -Filter "*.log" | ForEach-Object {
    if ($_.Length -gt 50MB) {
        $archiveName = "$($_.BaseName)-$(Get-Date -Format 'yyyyMMdd-HHmmss')$($_.Extension)"
        Move-Item $_.FullName "$logDir\archive\$archiveName"
    }
}

# Schedule log rotation
$rotationAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\QCoin\scripts\log-rotation.ps1"
$rotationTrigger = New-ScheduledTaskTrigger -Daily -At "02:00"
Register-ScheduledTask -TaskName "QCoinLogRotation" -Action $rotationAction -Trigger $rotationTrigger
```

### Performance Monitoring

#### Error: Performance counters not available
**Problem:** Windows performance monitoring not configured.

**Solutions:**
```powershell
# Register performance counters (if available)
# This would require custom performance counter DLL

# Use WMI for basic monitoring
Get-WmiObject Win32_Process | Where-Object {$_.Name -like "*geth*" -or $_.Name -like "*quantum-miner*"}

# Use Performance Toolkit
# Install Windows Performance Toolkit from Windows SDK

# Monitor with built-in counters
Get-Counter "\Process(geth)\% Processor Time"
Get-Counter "\Process(geth)\Working Set"
```

## 🔄 Backup and Recovery Issues

### Backup Failures

#### Error: `The process cannot access the file because it is being used by another process`
**Problem:** Trying to backup while services are running.

**Solutions:**
```powershell
# Stop services before backup
Stop-Service QGethNode -Force
Stop-Service QMiner -Force

# Wait for file handles to close
Start-Sleep 10

# Use Volume Shadow Copy for hot backup
$shadow = (vssadmin create shadow /for=C:).Split("`n") | Where-Object {$_ -like "*Shadow Copy Volume*"}
$shadowPath = $shadow.Split(":")[2].Trim()

# Copy from shadow copy
robocopy "$shadowPath\QCoin\data" "E:\Backup\data" /MIR

# Delete shadow copy
$shadowId = $shadow.Split("{")[1].Split("}")[0]
vssadmin delete shadows /shadow={$shadowId}
```

#### Error: Backup script fails with permissions
**Problem:** Backup running without sufficient privileges.

**Solutions:**
```powershell
# Run backup as administrator
Start-Process PowerShell -Verb RunAs -ArgumentList "-File C:\QCoin\scripts\backup.ps1"

# Or configure backup service account
$password = ConvertTo-SecureString "BackupPassword123!" -AsPlainText -Force
New-LocalUser -Name "qcoin-backup" -Password $password
Add-LocalGroupMember -Group "Backup Operators" -Member "qcoin-backup"

# Grant backup privileges
secedit /export /cfg backup-rights.inf
# Edit backup-rights.inf to add SeBackupPrivilege
secedit /configure /db backup-rights.sdb /cfg backup-rights.inf
```

### Recovery Problems

#### Error: Service won't start after recovery
**Problem:** Corrupted data or configuration after restore.

**Solutions:**
```powershell
# Verify data integrity
geth.exe --datadir "C:\QCoin\data" --dev console --exec "eth.blockNumber"

# Reset if needed
Remove-Item "C:\QCoin\data\geth\chaindata" -Recurse -Force
geth.exe --datadir "C:\QCoin\data" init "C:\QCoin\configs\genesis_quantum_mainnet.json"

# Check service configuration
nssm get QGethNode Application
nssm get QGethNode AppDirectory
nssm get QGethNode Parameters

# Reinstall service if needed
nssm remove QGethNode confirm
nssm install QGethNode "C:\QCoin\geth.exe"
```

## 🎯 Advanced Troubleshooting

### Service Dependencies

#### Error: Service fails to start due to dependencies
**Problem:** Required services not running or missing.

**Solutions:**
```powershell
# Check service dependencies
Get-Service QGethNode | Select-Object -ExpandProperty RequiredServices
Get-Service QGethNode | Select-Object -ExpandProperty DependentServices

# Set service dependencies
sc config QGethNode depend= Tcpip/Dhcp/Dnscache

# Check dependent services status
Get-Service Tcpip, Dhcp, Dnscache

# Start dependencies manually
Start-Service Tcpip
Start-Service Dhcp
Start-Service Dnscache
```

### Registry Issues

#### Error: Service configuration lost after reboot
**Problem:** Registry corruption or permission issues.

**Solutions:**
```powershell
# Check service registry entries
Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Services\QGethNode"

# Backup service configuration
nssm dump QGethNode > "C:\QCoin\backup\qgethnode-config.txt"

# Restore service configuration
nssm install QGethNode < "C:\QCoin\backup\qgethnode-config.txt"

# Export registry for backup
reg export "HKLM\SYSTEM\CurrentControlSet\Services\QGethNode" "C:\QCoin\backup\qgethnode-registry.reg"
```

### System Integration

#### Error: Services conflict with other blockchain software
**Problem:** Port or resource conflicts.

**Solutions:**
```powershell
# Check for other blockchain services
Get-Service | Where-Object {$_.DisplayName -like "*bitcoin*" -or $_.DisplayName -like "*ethereum*"}

# Change Q Coin ports
nssm set QGethNode Parameters "--http.port 18545 --ws.port 18546 --port 40303"

# Use different data directories
nssm set QGethNode Parameters "--datadir C:\QCoin\mainnet-data"
```

## ✅ Deployment Health Check

### System Health Verification

```powershell
# Q Coin Windows Deployment Health Check
Write-Host "=== Q Coin Windows Deployment Health Check ==="

# Check services
$services = @("QGethNode", "QMiner")
foreach ($service in $services) {
    $svc = Get-Service $service -ErrorAction SilentlyContinue
    if ($svc) {
        $status = $svc.Status
        Write-Host "✅ $service service: $status"
        if ($status -ne "Running") {
            Write-Host "⚠️  Service not running"
        }
    } else {
        Write-Host "❌ $service service not found"
    }
}

# Check ports
$ports = @(8545, 8546, 30303)
foreach ($port in $ports) {
    $listening = netstat -an | findstr ":$port "
    if ($listening) {
        Write-Host "✅ Port $port is listening"
    } else {
        Write-Host "❌ Port $port not listening"
    }
}

# Check firewall rules
$rules = Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*Q Geth*"}
if ($rules) {
    Write-Host "✅ Firewall rules configured"
} else {
    Write-Host "⚠️  No firewall rules found"
}

# Check data directory
if (Test-Path "C:\QCoin\data") {
    $dataSize = (Get-ChildItem "C:\QCoin\data" -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "✅ Data directory exists ($([math]::Round($dataSize, 2)) GB)"
} else {
    Write-Host "❌ Data directory not found"
}

# Check logs
if (Test-Path "C:\QCoin\logs") {
    $logCount = (Get-ChildItem "C:\QCoin\logs" -Filter "*.log").Count
    Write-Host "✅ Log directory exists ($logCount log files)"
} else {
    Write-Host "⚠️  Log directory not found"
}

Write-Host "✅ Deployment health check complete!"
```

### Network Connectivity Test

```powershell
# Q Coin Network Connectivity Test
Write-Host "=== Q Coin Network Connectivity Test ==="

# Test local RPC
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8545" -Method POST -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' -ContentType "application/json"
    Write-Host "✅ Local RPC accessible: Block $($response.result)"
} catch {
    Write-Host "❌ Local RPC not accessible"
}

# Test WebSocket
try {
    $ws = New-Object System.Net.WebSockets.ClientWebSocket
    $uri = [System.Uri]::new("ws://localhost:8546")
    $token = [System.Threading.CancellationToken]::None
    $task = $ws.ConnectAsync($uri, $token)
    $task.Wait(5000)
    if ($ws.State -eq "Open") {
        Write-Host "✅ WebSocket accessible"
        $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "", $token).Wait()
    }
} catch {
    Write-Host "❌ WebSocket not accessible"
}

# Test P2P connectivity
$p2pTest = Test-NetConnection -ComputerName "localhost" -Port 30303
if ($p2pTest.TcpTestSucceeded) {
    Write-Host "✅ P2P port accessible"
} else {
    Write-Host "❌ P2P port not accessible"
}

Write-Host "✅ Network connectivity test complete!"
```

## 📞 Getting Help

### Community Support
- **GitHub Issues:** [Report deployment issues](https://github.com/fourtytwo42/Qgeth3/issues)
- **Documentation:** Check [deployment guide](windows-deploy.md)

### Self-Help Resources
- Run the health check scripts above
- Check Windows Event Viewer for service errors
- Review [VPS deployment troubleshooting](troubleshooting-vps-deployment.md)

### Debug Information Collection

```powershell
# Create comprehensive deployment debug log
$debugLog = "windows-deploy-debug.log"
@"
=== Windows Deployment Debug Information ===
Date: $(Get-Date)
OS: $((Get-WmiObject Win32_OperatingSystem).Caption)
User: $env:USERNAME
Computer: $env:COMPUTERNAME

=== Services Status ===
"@ | Out-File $debugLog

Get-Service | Where-Object {$_.Name -like "*QGeth*" -or $_.Name -like "*QMiner*"} | Out-File $debugLog -Append

@"

=== NSSM Configuration ===
"@ | Out-File $debugLog -Append

try {
    nssm get QGethNode Application 2>&1 | Out-File $debugLog -Append
    nssm get QGethNode Parameters 2>&1 | Out-File $debugLog -Append
} catch {}

@"

=== Network Status ===
"@ | Out-File $debugLog -Append

netstat -an | findstr "8545\|8546\|30303" | Out-File $debugLog -Append

@"

=== Recent Event Logs ===
"@ | Out-File $debugLog -Append

Get-WinEvent -LogName Application -MaxEvents 50 | Where-Object {$_.ProviderName -like "*QGeth*" -or $_.ProviderName -like "*QMiner*"} | Out-File $debugLog -Append

Write-Host "Debug log created: $debugLog"
Write-Host "Share this file when seeking help"
```

**Most Windows deployment issues are resolved by ensuring proper service configuration, firewall rules, and file permissions!** 🚀 