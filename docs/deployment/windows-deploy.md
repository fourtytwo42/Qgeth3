# Windows Q Coin Deployment Guide

Complete guide for deploying Q Coin quantum blockchain nodes and miners on Windows systems in production environments.

## 📋 Deployment Overview

### Deployment Scenarios
- **Desktop Development:** Single-user development and testing
- **Workstation Mining:** High-performance mining on workstations
- **Server Deployment:** Production blockchain nodes
- **Enterprise Mining Farms:** Large-scale mining operations
- **Cloud Deployment:** Azure/AWS Windows instances

### Architecture Components
```
Windows Q Coin Deployment
├── Q Geth Node (geth.exe)           # Blockchain node
├── Quantum Miner (quantum-miner.exe) # Mining software
├── Windows Services                  # Background services
├── Monitoring & Logging             # Performance tracking
└── Security & Firewall              # Network protection
```

## 🚀 Quick Production Deployment

### Automated Setup (Recommended)
```powershell
# Download and run automated installer
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/windows/deploy-production.ps1" -OutFile "deploy-production.ps1"
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\deploy-production.ps1 -Mode Production -Network mainnet
```

### Manual Production Setup
```powershell
# 1. Create deployment directory
$deployPath = "C:\QCoin"
New-Item -ItemType Directory -Force -Path $deployPath
Set-Location $deployPath

# 2. Download release binaries
Invoke-WebRequest -Uri "https://github.com/fourtytwo42/Qgeth3/releases/latest/download/qgeth-windows.zip" -OutFile "qgeth-windows.zip"
Expand-Archive -Path "qgeth-windows.zip" -DestinationPath "."

# 3. Install as Windows service
.\install-service.ps1 -ServiceName "QGethNode" -Network mainnet
.\install-service.ps1 -ServiceName "QMiner" -Network mainnet
```

## 🔧 Component Installation

### Q Geth Node Installation

#### Service Installation with NSSM
```powershell
# Download and install NSSM (Non-Sucking Service Manager)
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "nssm.zip"
Expand-Archive -Path "nssm.zip" -DestinationPath "C:\Tools"
$env:PATH += ";C:\Tools\nssm-2.24\win64"

# Install Q Geth as Windows service
nssm install QGethNode "C:\QCoin\geth.exe"
nssm set QGethNode Parameters "--http --http.addr 0.0.0.0 --http.port 8545 --http.corsdomain '*' --http.api eth,net,web3,personal,txpool --ws --ws.addr 0.0.0.0 --ws.port 8546 --mine --miner.threads 0"
nssm set QGethNode AppDirectory "C:\QCoin"
nssm set QGethNode DisplayName "Q Coin Blockchain Node"
nssm set QGethNode Description "Q Coin quantum blockchain node service"
nssm set QGethNode Start SERVICE_AUTO_START

# Set service recovery options
nssm set QGethNode AppExit Default Restart
nssm set QGethNode AppRestartDelay 30000
nssm set QGethNode AppStdout "C:\QCoin\logs\geth-stdout.log"
nssm set QGethNode AppStderr "C:\QCoin\logs\geth-stderr.log"
nssm set QGethNode AppRotateFiles 1
nssm set QGethNode AppRotateOnline 1
nssm set QGethNode AppRotateBytes 10485760  # 10MB

# Start the service
Start-Service QGethNode
```

#### Manual Service Configuration
```powershell
# Create service configuration script
@'
# Q Geth Node Service Configuration

# Service parameters
$serviceName = "QGethNode"
$serviceDisplayName = "Q Coin Blockchain Node"
$serviceDescription = "Q Coin quantum blockchain node service"
$executablePath = "C:\QCoin\geth.exe"
$serviceArgs = @(
    "--datadir", "C:\QCoin\data",
    "--http", "--http.addr", "0.0.0.0", "--http.port", "8545",
    "--http.corsdomain", "*",
    "--http.api", "eth,net,web3,personal,txpool",
    "--ws", "--ws.addr", "0.0.0.0", "--ws.port", "8546",
    "--mine", "--miner.threads", "0"
)

# Create and start service
nssm install $serviceName $executablePath
foreach ($arg in $serviceArgs) {
    nssm set $serviceName Parameters $arg
}

'@ | Out-File -Encoding UTF8 install-geth-service.ps1
```

### Quantum Miner Installation

#### Mining Service Setup
```powershell
# Install quantum miner as Windows service
nssm install QMiner "C:\QCoin\quantum-miner.exe"
nssm set QMiner Parameters "--gpu --coinbase 0xYourMiningAddress --pool stratum+tcp://pool.qcoin.org:4444"
nssm set QMiner AppDirectory "C:\QCoin"
nssm set QMiner DisplayName "Q Coin Quantum Miner"
nssm set QMiner Description "Q Coin quantum mining service"
nssm set QMiner Start SERVICE_AUTO_START

# Configure mining-specific settings
nssm set QMiner AppEnvironmentExtra "CUDA_VISIBLE_DEVICES=0,1"  # Multi-GPU setup
nssm set QMiner AppPriority ABOVE_NORMAL_PRIORITY_CLASS
nssm set QMiner AppStdout "C:\QCoin\logs\miner-stdout.log"
nssm set QMiner AppStderr "C:\QCoin\logs\miner-stderr.log"

# Start mining service
Start-Service QMiner
```

#### GPU-Optimized Configuration
```powershell
# Multi-GPU mining setup
$gpuCount = (Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}).Count
Write-Host "Detected $gpuCount NVIDIA GPUs"

# Configure for each GPU
for ($i = 0; $i -lt $gpuCount; $i++) {
    $serviceName = "QMiner-GPU$i"
    nssm install $serviceName "C:\QCoin\quantum-miner.exe"
    nssm set $serviceName Parameters "--gpu --gpu-id $i --coinbase 0xYourAddress"
    nssm set $serviceName AppEnvironmentExtra "CUDA_VISIBLE_DEVICES=$i"
    nssm set $serviceName DisplayName "Q Coin Miner GPU $i"
    nssm set $serviceName Start SERVICE_AUTO_START
}
```

## 🔐 Security Configuration

### Windows Firewall Setup
```powershell
# Allow Q Geth ports through Windows Firewall
New-NetFirewallRule -DisplayName "Q Geth HTTP RPC" -Direction Inbound -Protocol TCP -LocalPort 8545 -Action Allow
New-NetFirewallRule -DisplayName "Q Geth WebSocket" -Direction Inbound -Protocol TCP -LocalPort 8546 -Action Allow
New-NetFirewallRule -DisplayName "Q Geth P2P" -Direction Inbound -Protocol TCP -LocalPort 30303 -Action Allow
New-NetFirewallRule -DisplayName "Q Geth P2P UDP" -Direction Inbound -Protocol UDP -LocalPort 30303 -Action Allow

# Mining pool connections (if applicable)
New-NetFirewallRule -DisplayName "Mining Pool" -Direction Outbound -Protocol TCP -RemotePort 4444 -Action Allow

# Block unnecessary ports for security
New-NetFirewallRule -DisplayName "Block P2P Discovery" -Direction Inbound -Protocol UDP -LocalPort 30301 -Action Block
```

### User Account Configuration
```powershell
# Create dedicated service account
$password = ConvertTo-SecureString "SecureServicePassword123!" -AsPlainText -Force
New-LocalUser -Name "qcoin-service" -Password $password -FullName "Q Coin Service Account" -Description "Service account for Q Coin blockchain services"

# Set account properties
Set-LocalUser -Name "qcoin-service" -PasswordNeverExpires $true
Add-LocalGroupMember -Group "Log on as a service" -Member "qcoin-service"

# Configure service to run as service account
nssm set QGethNode ObjectName ".\qcoin-service" "SecureServicePassword123!"
nssm set QMiner ObjectName ".\qcoin-service" "SecureServicePassword123!"
```

### File System Permissions
```powershell
# Set proper permissions on Q Coin directory
$acl = Get-Acl "C:\QCoin"
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("qcoin-service","FullControl","ContainerInherit,ObjectInherit","None","Allow")
$acl.SetAccessRule($accessRule)
Set-Acl -Path "C:\QCoin" -AclObject $acl

# Restrict access to configuration files
$configAcl = Get-Acl "C:\QCoin\config"
$configRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Users","Read","ContainerInherit,ObjectInherit","None","Deny")
$configAcl.SetAccessRule($configRule)
Set-Acl -Path "C:\QCoin\config" -AclObject $configAcl
```

## 📊 Monitoring and Logging

### Event Log Configuration
```powershell
# Create custom event log for Q Coin
New-EventLog -LogName "QCoin" -Source "QGethNode"
New-EventLog -LogName "QCoin" -Source "QMiner"

# Configure log retention
Limit-EventLog -LogName "QCoin" -MaximumSize 100MB -OverflowAction OverwriteOlder
```

### Performance Monitoring
```powershell
# Create performance monitoring script
@'
# Q Coin Performance Monitor
param(
    [int]$IntervalSeconds = 60
)

while ($true) {
    $timestamp = Get-Date
    
    # Node metrics
    $gethProcess = Get-Process geth -ErrorAction SilentlyContinue
    if ($gethProcess) {
        $nodeMetrics = @{
            CPU = $gethProcess.CPU
            Memory = [math]::Round($gethProcess.WorkingSet64 / 1MB, 2)
            Threads = $gethProcess.Threads.Count
        }
        Write-EventLog -LogName "QCoin" -Source "QGethNode" -EntryType Information -EventId 1001 -Message "Node Metrics: CPU=$($nodeMetrics.CPU), Memory=$($nodeMetrics.Memory)MB, Threads=$($nodeMetrics.Threads)"
    }
    
    # Miner metrics
    $minerProcess = Get-Process quantum-miner -ErrorAction SilentlyContinue
    if ($minerProcess) {
        $minerMetrics = @{
            CPU = $minerProcess.CPU
            Memory = [math]::Round($minerProcess.WorkingSet64 / 1MB, 2)
        }
        Write-EventLog -LogName "QCoin" -Source "QMiner" -EntryType Information -EventId 2001 -Message "Miner Metrics: CPU=$($minerMetrics.CPU), Memory=$($minerMetrics.Memory)MB"
    }
    
    # GPU metrics (if available)
    try {
        $gpuInfo = nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            Write-EventLog -LogName "QCoin" -Source "QMiner" -EntryType Information -EventId 2002 -Message "GPU Metrics: $gpuInfo"
        }
    } catch {}
    
    Start-Sleep $IntervalSeconds
}
'@ | Out-File -Encoding UTF8 "C:\QCoin\scripts\performance-monitor.ps1"

# Install monitoring as scheduled task
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\QCoin\scripts\performance-monitor.ps1"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserID "qcoin-service" -LogonType ServiceAccount
Register-ScheduledTask -TaskName "QCoinPerformanceMonitor" -Action $action -Trigger $trigger -Principal $principal
```

### Log Rotation and Management
```powershell
# Create log rotation script
@'
# Q Coin Log Rotation
$logDir = "C:\QCoin\logs"
$maxLogSize = 50MB
$maxLogAge = 30  # days

Get-ChildItem $logDir -Filter "*.log" | ForEach-Object {
    if ($_.Length -gt $maxLogSize -or $_.CreationTime -lt (Get-Date).AddDays(-$maxLogAge)) {
        $archiveName = "$($_.BaseName)-$(Get-Date -Format 'yyyyMMdd-HHmmss')$($_.Extension)"
        $archivePath = Join-Path $logDir "archive\$archiveName"
        
        New-Item -ItemType Directory -Force -Path (Split-Path $archivePath)
        Move-Item $_.FullName $archivePath
        
        # Compress old logs
        Compress-Archive -Path $archivePath -DestinationPath "$archivePath.zip" -Force
        Remove-Item $archivePath
    }
}
'@ | Out-File -Encoding UTF8 "C:\QCoin\scripts\log-rotation.ps1"

# Schedule log rotation
$rotationAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\QCoin\scripts\log-rotation.ps1"
$rotationTrigger = New-ScheduledTaskTrigger -Daily -At "02:00"
Register-ScheduledTask -TaskName "QCoinLogRotation" -Action $rotationAction -Trigger $rotationTrigger
```

## 🌐 Network Configuration

### Load Balancer Setup (Multiple Nodes)
```powershell
# Configure IIS as load balancer for multiple Q Geth nodes
Install-WindowsFeature -Name IIS-WebServerRole, IIS-WebServer, IIS-ApplicationRequestRouting

# Configure ARR for load balancing
Import-Module WebAdministration

# Create server farm
New-IISServerFarm -Name "QGethFarm"
Add-IISServerFarmServer -Name "QGethFarm" -Address "127.0.0.1" -Port 8545
Add-IISServerFarmServer -Name "QGethFarm" -Address "127.0.0.1" -Port 8546  # If running multiple instances

# Create load balancing rule
$rule = @{
    Name = "QGethProxy"
    Pattern = ".*"
    Conditions = @(@{Input="{HTTP_HOST}"; Pattern="qgeth.yourdomain.com"})
    Action = @{Type="Rewrite"; Url="http://QGethFarm/{R:0}"}
}

Add-WebConfigurationProperty -Filter "system.webServer/rewrite/rules" -Name "." -Value $rule -PSPath "IIS:\"
```

### SSL/TLS Configuration
```powershell
# Install SSL certificate for secure RPC
$cert = Import-PfxCertificate -FilePath "C:\QCoin\ssl\qgeth.pfx" -CertStoreLocation Cert:\LocalMachine\My -Password (ConvertTo-SecureString "CertPassword" -AsPlainText -Force)

# Configure HTTPS binding
New-IISSiteBinding -Name "Default Web Site" -BindingInformation "*:443:qgeth.yourdomain.com" -Protocol "https" -CertificateThumbPrint $cert.Thumbprint
```

## 📈 Performance Optimization

### System-Level Optimizations
```powershell
# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable Windows Search indexing on data directory
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\ContentIndex" -Name "FilterFilesWithUnknownExtensions" -Value 0

# Optimize virtual memory
$totalRAM = (Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory / 1GB
$pagingFileSize = [math]::Round($totalRAM * 1.5, 0) * 1024  # 1.5x RAM in MB
wmic computersystem where name="%computername%" set AutomaticManagedPagefile=False
wmic pagefileset where name="C:\\pagefile.sys" set InitialSize=$pagingFileSize,MaximumSize=$pagingFileSize
```

### GPU Mining Optimizations
```powershell
# NVIDIA GPU optimization script
@'
# NVIDIA GPU Optimization for Q Coin Mining

# Set persistence mode
nvidia-smi -pm 1

# Optimize power and clock settings for each GPU
$gpuCount = (nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
for ($i = 0; $i -lt $gpuCount; $i++) {
    # Set power limit (adjust based on your GPU)
    nvidia-smi -i $i -pl 300  # 300W limit
    
    # Set memory clock offset (adjust based on stability testing)
    nvidia-smi -i $i -ac 5001,1000  # Memory and graphics clock
    
    # Set fan speed
    nvidia-smi -i $i -gtt 80  # Target temperature 80°C
}

# Monitor GPU status
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
'@ | Out-File -Encoding UTF8 "C:\QCoin\scripts\gpu-optimize.ps1"

# Run GPU optimization at startup
$gpuAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\QCoin\scripts\gpu-optimize.ps1"
$gpuTrigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "QCoinGPUOptimization" -Action $gpuAction -Trigger $gpuTrigger
```

## 🔄 Backup and Recovery

### Automated Backup System
```powershell
# Create backup script
@'
# Q Coin Backup Script
param(
    [string]$BackupPath = "E:\QCoinBackups",
    [int]$RetentionDays = 7
)

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path $BackupPath $timestamp

# Stop services
Stop-Service QGethNode -Force
Stop-Service QMiner -Force

try {
    # Create backup directory
    New-Item -ItemType Directory -Force -Path $backupDir
    
    # Backup blockchain data
    Write-Host "Backing up blockchain data..."
    robocopy "C:\QCoin\data" "$backupDir\data" /MIR /MT:8 /LOG:"$backupDir\backup.log"
    
    # Backup configuration
    Write-Host "Backing up configuration..."
    Copy-Item -Path "C:\QCoin\config" -Destination "$backupDir\config" -Recurse -Force
    
    # Backup keystore
    Write-Host "Backing up keystore..."
    Copy-Item -Path "C:\QCoin\keystore" -Destination "$backupDir\keystore" -Recurse -Force
    
    # Create backup info file
    @{
        Timestamp = Get-Date
        BlockHeight = "TBD"  # Could query from geth
        Size = (Get-ChildItem $backupDir -Recurse | Measure-Object -Property Length -Sum).Sum
    } | ConvertTo-Json | Out-File "$backupDir\backup-info.json"
    
    # Cleanup old backups
    Get-ChildItem $BackupPath -Directory | Where-Object {
        $_.CreationTime -lt (Get-Date).AddDays(-$RetentionDays)
    } | Remove-Item -Recurse -Force
    
    Write-Host "Backup completed successfully"
    
} finally {
    # Restart services
    Start-Service QGethNode
    Start-Service QMiner
}
'@ | Out-File -Encoding UTF8 "C:\QCoin\scripts\backup.ps1"

# Schedule daily backups
$backupAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\QCoin\scripts\backup.ps1"
$backupTrigger = New-ScheduledTaskTrigger -Daily -At "01:00"
Register-ScheduledTask -TaskName "QCoinDailyBackup" -Action $backupAction -Trigger $backupTrigger
```

### Disaster Recovery
```powershell
# Create recovery script
@'
# Q Coin Disaster Recovery Script
param(
    [string]$BackupPath,
    [switch]$Force
)

if (-not $BackupPath) {
    Write-Error "Please specify -BackupPath parameter"
    exit 1
}

if (-not (Test-Path $BackupPath)) {
    Write-Error "Backup path does not exist: $BackupPath"
    exit 1
}

if (-not $Force) {
    $confirm = Read-Host "This will overwrite existing Q Coin data. Type 'YES' to continue"
    if ($confirm -ne "YES") {
        Write-Host "Recovery cancelled"
        exit 0
    }
}

# Stop services
Stop-Service QGethNode -Force -ErrorAction SilentlyContinue
Stop-Service QMiner -Force -ErrorAction SilentlyContinue

try {
    # Restore data
    Write-Host "Restoring blockchain data..."
    robocopy "$BackupPath\data" "C:\QCoin\data" /MIR /MT:8
    
    # Restore configuration
    Write-Host "Restoring configuration..."
    robocopy "$BackupPath\config" "C:\QCoin\config" /MIR
    
    # Restore keystore
    Write-Host "Restoring keystore..."
    robocopy "$BackupPath\keystore" "C:\QCoin\keystore" /MIR
    
    Write-Host "Recovery completed successfully"
    
} finally {
    # Restart services
    Start-Service QGethNode
    Start-Service QMiner
}
'@ | Out-File -Encoding UTF8 "C:\QCoin\scripts\recovery.ps1"
```

## 🎯 Enterprise Deployment

### Group Policy Configuration
```powershell
# Create GPO template for Q Coin deployment
@'
# Q Coin Group Policy Configuration

# Registry settings for Q Coin services
Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SOFTWARE\QCoin]
"InstallPath"="C:\\QCoin"
"DataPath"="C:\\QCoin\\data"
"LogLevel"="info"
"EnableMonitoring"=dword:00000001

[HKEY_LOCAL_MACHINE\SOFTWARE\QCoin\Network]
"MainnetEnabled"=dword:00000001
"TestnetEnabled"=dword:00000000
"BootstrapNodes"="enode://..."

[HKEY_LOCAL_MACHINE\SOFTWARE\QCoin\Mining]
"AutoStart"=dword:00000001
"GPUEnabled"=dword:00000001
"MaxThreads"=dword:00000000
'@ | Out-File -Encoding ASCII "QCoin-GPO.reg"
```

### SCCM Deployment Package
```powershell
# Create SCCM deployment script
@'
# Q Coin SCCM Deployment Script

# Detection method
if (Test-Path "C:\QCoin\geth.exe") {
    Write-Host "Q Coin is installed"
    exit 0
} else {
    Write-Host "Q Coin is not installed"
    exit 1
}

# Installation script
function Install-QCoin {
    # Download and extract
    $source = "\\sccm-server\packages\QCoin\qcoin-latest.zip"
    $dest = "C:\QCoin"
    
    New-Item -ItemType Directory -Force -Path $dest
    Expand-Archive -Path $source -DestinationPath $dest -Force
    
    # Install services
    & "$dest\install-services.ps1"
    
    # Configure firewall
    & "$dest\configure-firewall.ps1"
    
    # Start services
    Start-Service QGethNode
    Start-Service QMiner
}

# Uninstallation script
function Uninstall-QCoin {
    Stop-Service QGethNode -Force -ErrorAction SilentlyContinue
    Stop-Service QMiner -Force -ErrorAction SilentlyContinue
    
    nssm remove QGethNode confirm
    nssm remove QMiner confirm
    
    Remove-Item "C:\QCoin" -Recurse -Force -ErrorAction SilentlyContinue
}
'@ | Out-File -Encoding UTF8 "QCoin-SCCM.ps1"
```

## 🔍 Troubleshooting

For deployment issues, see the [Windows Deployment Troubleshooting Guide](troubleshooting-windows-deploy.md).

Common deployment issues:
```powershell
# Service won't start
Get-EventLog -LogName System -Source "Service Control Manager" | Where-Object {$_.Message -like "*QGeth*"}

# Port conflicts
netstat -an | findstr "8545"
netstat -an | findstr "30303"

# Permission issues
icacls "C:\QCoin" /grant "qcoin-service:(OI)(CI)F"

# GPU not detected
nvidia-smi
Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
```

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Hardware meets requirements (CPU, RAM, GPU, storage)
- [ ] Windows 10/11 properly activated and updated
- [ ] Required software installed (Go, Python, Visual Studio Build Tools)
- [ ] Network connectivity and firewall configured
- [ ] Backup strategy planned

### Service Installation
- [ ] Q Geth service installed and configured
- [ ] Quantum miner service installed and configured
- [ ] Services set to start automatically
- [ ] Service recovery options configured
- [ ] Logging and monitoring configured

### Security Configuration
- [ ] Dedicated service account created
- [ ] File system permissions configured
- [ ] Windows Firewall rules applied
- [ ] SSL/TLS certificates installed (if applicable)
- [ ] Security monitoring enabled

### Production Readiness
- [ ] Performance monitoring active
- [ ] Backup system operational
- [ ] Log rotation configured
- [ ] GPU optimization applied (if applicable)
- [ ] Disaster recovery procedures tested

**Q Coin production deployment on Windows is now complete and operational!** 🚀 