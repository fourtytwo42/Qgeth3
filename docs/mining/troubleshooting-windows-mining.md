# Windows Mining Troubleshooting

Solutions for Windows mining issues with Q Coin quantum blockchain.

## 🔧 Quick Mining Diagnostics

### Check Mining Status (PowerShell)
```powershell
# Check if miner binary exists
Get-ChildItem quantum-miner.exe
Get-Command quantum-miner.exe -ErrorAction SilentlyContinue

# Check GPU availability
nvidia-smi

# Check system resources
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum

# Test quantum computing libraries
python -c "import cupy; print('CuPy available')"
python -c "import cupy; print('CUDA available:', cupy.cuda.is_available())"
```

## 🏗️ Visual Studio Build Tools Issues

### Visual Studio Not Found
```powershell
# Symptoms: "vcvarsall.bat not found", "Microsoft Visual C++ 14.0 is required"
# Solution: Install Visual Studio Build Tools

# Download Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Required workloads and components:
# - C++ build tools (workload)
# - MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
# - Windows 10/11 SDK (latest version)
# - CMake tools for Visual Studio

# Verify installation
where.exe cl  # Should show path to MSVC compiler
where.exe link  # Should show path to linker

# Check Visual Studio installation
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vs_installer.exe"

# Alternative: Install Visual Studio Community (full IDE)
# https://visualstudio.microsoft.com/vs/community/
```

### Build Tools Configuration
```powershell
# If cl.exe not found, setup build environment
# Find vcvarsall.bat
Get-ChildItem -Path "C:\Program Files*" -Recurse -Name "vcvarsall.bat" -ErrorAction SilentlyContinue

# Setup build environment manually
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

# Or use Developer PowerShell
# Start -> "Developer PowerShell for VS 2022"

# Verify compiler works
cl.exe /?
link.exe /?
```

### Windows SDK Issues
```powershell
# Check Windows SDK installation
Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\Include" -ErrorAction SilentlyContinue

# Install Windows SDK if missing
# Download from: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/

# Set SDK environment variables if needed
$env:WindowsSdkDir = "C:\Program Files (x86)\Windows Kits\10\"
$env:WindowsSdkVersion = "10.0.22621.0\"  # Use your version
```

## 🐍 Python and CuPy Issues

### Python Installation Problems
```powershell
# Check Python version (need 3.8+)
python --version

# If Python not found, install from Microsoft Store or python.org
# Microsoft Store: Search "Python 3.11"
# Or download from: https://www.python.org/downloads/windows/

# Verify pip is working
pip --version

# Update pip
python -m pip install --upgrade pip
```

### CuPy Installation Issues
```powershell
# Symptoms: "ModuleNotFoundError: No module named 'cupy'"
# Solution: Install CuPy with correct CUDA version

# Check CUDA version first
nvidia-smi

# Install CuPy for CUDA 11.x
pip install cupy-cuda11x

# Install CuPy for CUDA 12.x
pip install cupy-cuda12x

# If installation fails, try with --force-reinstall
pip install --force-reinstall cupy-cuda11x

# Test CuPy installation
python -c "import cupy; print('CuPy version:', cupy.__version__)"
python -c "import cupy; print('CUDA available:', cupy.cuda.is_available())"
python -c "import cupy; print('GPU count:', cupy.cuda.runtime.getDeviceCount())"
```

### CUDA on Windows
```powershell
# Check CUDA installation
nvcc --version
nvidia-smi

# If CUDA not found, download and install from:
# https://developer.nvidia.com/cuda-downloads

# Set CUDA environment variables
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
$env:PATH += ";$env:CUDA_PATH\bin"

# Add to permanent environment
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0", "Machine")
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$env:CUDA_PATH\bin", "Machine")

# Test CUDA compiler
nvcc --version
```

## 🎮 GPU and Driver Issues

### NVIDIA Drivers
```powershell
# Check current driver version
nvidia-smi

# Update NVIDIA drivers
# Method 1: NVIDIA GeForce Experience (for GeForce cards)
# Method 2: Download from NVIDIA website
# Method 3: Windows Update

# Check GPU detection
Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}

# If GPU not detected, check Device Manager
devmgmt.msc
# Look for: Display adapters -> NVIDIA GPU
# If showing error, right-click -> Update driver
```

### GPU Memory Issues
```powershell
# Check GPU memory usage
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Set GPU memory limits for mining
$env:CUPY_MEMPOOL_SIZE = "2048"  # 2GB
$env:CUDA_MEMORY_POOL_SIZE = "0.8"  # 80% of GPU memory

# Reduce GPU threads if memory errors
.\scripts\windows\start-miner.ps1 -Threads 2 -GPU

# Monitor GPU during mining
nvidia-smi dmon
```

## 🔨 Build and Binary Issues

### Go Build Errors on Windows
```powershell
# Check Go installation
go version

# If Go not found, install from https://golang.org/dl/
# Make sure Go bin directory is in PATH

# Check PATH
$env:PATH -split ';' | Where-Object {$_ -like "*go*"}

# Build miner for Windows
.\scripts\windows\build-release.ps1

# If build fails, try clean build
Remove-Item quantum-miner.exe -ErrorAction SilentlyContinue
.\scripts\windows\build-release.ps1

# Check if binary was created
Get-ChildItem quantum-miner.exe
Get-ChildItem .\releases\quantum-miner-*\quantum-miner.exe
```

### Binary Dependencies
```powershell
# Check if all required DLLs are available
# Use Dependency Walker or similar tool

# Common missing DLLs and solutions:
# - vcruntime140.dll: Install Visual C++ Redistributable
# - msvcp140.dll: Install Visual C++ Redistributable

# Download Visual C++ Redistributable from:
# https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

# Test miner binary
.\quantum-miner.exe --help
```

## 🔧 PowerShell Script Issues

### Execution Policy Problems
```powershell
# Check current execution policy
Get-ExecutionPolicy

# If scripts can't run, change policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run specific script with bypass
PowerShell -ExecutionPolicy Bypass -File .\scripts\windows\start-miner.ps1

# For permanent solution, run as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

### Script Parameter Issues
```powershell
# Check script parameters
Get-Help .\scripts\windows\start-miner.ps1

# Common parameter formats:
.\scripts\windows\start-miner.ps1 -GPU -Threads 4
.\scripts\windows\start-miner.ps1 -CPU -Threads 8
.\scripts\windows\start-miner.ps1 -Network testnet -CoinbaseAddress "0xYourAddress"

# If parameters not working, check script syntax
Get-Content .\scripts\windows\start-miner.ps1 | Select-Object -First 20
```

### Path and Directory Issues
```powershell
# Check current directory
Get-Location

# Navigate to correct directory
Set-Location "C:\Users\YourUser\Qgeth3"

# Fix relative paths in scripts
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Check if all required files exist
Test-Path ".\quantum-miner.exe"
Test-Path ".\scripts\windows\start-miner.ps1"
Test-Path ".\configs\genesis_quantum_testnet.json"
```

## 🌐 Network and Firewall Issues

### Windows Defender Firewall
```powershell
# Check if Windows Firewall is blocking mining
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*quantum*"}

# Add firewall rules for mining
New-NetFirewallRule -DisplayName "Q Geth HTTP" -Direction Inbound -Protocol TCP -LocalPort 8545
New-NetFirewallRule -DisplayName "Q Geth P2P" -Direction Inbound -Protocol TCP -LocalPort 30303
New-NetFirewallRule -DisplayName "Q Geth P2P UDP" -Direction Inbound -Protocol UDP -LocalPort 30303

# Or disable firewall temporarily for testing (NOT recommended for production)
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
```

### Windows Defender Antivirus
```powershell
# Check if Defender is blocking mining
Get-MpPreference | Select-Object -Property Exclusion*

# Add exclusions for Q Coin directory
Add-MpPreference -ExclusionPath "C:\Users\$env:USERNAME\Qgeth3"
Add-MpPreference -ExclusionProcess "quantum-miner.exe"
Add-MpPreference -ExclusionProcess "geth.exe"

# Check Windows Security Center
# Settings -> Update & Security -> Windows Security -> Virus & threat protection
```

### Network Connectivity
```powershell
# Test network connectivity to geth node
Test-NetConnection -ComputerName localhost -Port 8545

# Check if geth is responding
Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'

# Test with different endpoints
.\scripts\windows\start-miner.ps1 -Node "http://127.0.0.1:8545"
.\scripts\windows\start-miner.ps1 -Node "http://localhost:8545"
```

## 🔄 Windows Service Issues

### NSSM Service Problems
```powershell
# Check if NSSM is installed
Get-Command nssm -ErrorAction SilentlyContinue

# Install NSSM if needed
# Download from: https://nssm.cc/download
# Extract to C:\nssm\win64\nssm.exe
# Add to PATH

# Check service status
nssm status "Q Geth Miner"

# View service logs
nssm get "Q Geth Miner" stdout
nssm get "Q Geth Miner" stderr

# Restart service
nssm restart "Q Geth Miner"

# Remove and recreate service if broken
nssm remove "Q Geth Miner" confirm
# Then recreate with install script
```

### Windows Event Logs
```powershell
# Check Windows Event Logs for errors
Get-WinEvent -LogName Application | Where-Object {$_.LevelDisplayName -eq "Error"} | Select-Object -First 10

# Check system logs
Get-WinEvent -LogName System | Where-Object {$_.LevelDisplayName -eq "Error"} | Select-Object -First 10

# Filter for specific application
Get-WinEvent -LogName Application | Where-Object {$_.ProcessId -eq (Get-Process quantum-miner).Id}
```

## ⚡ Performance Issues

### Low Hash Rate
```powershell
# Check system performance
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 5

# Monitor GPU performance
nvidia-smi dmon

# Optimize thread count
.\scripts\windows\start-miner.ps1 -GPU -Threads 2
.\scripts\windows\start-miner.ps1 -GPU -Threads 4
.\scripts\windows\start-miner.ps1 -GPU -Threads 8

# Test CPU vs GPU performance
Measure-Command { .\scripts\windows\start-miner.ps1 -CPU -Threads 8 -Test }
Measure-Command { .\scripts\windows\start-miner.ps1 -GPU -Threads 4 -Test }
```

### System Overheating
```powershell
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader

# Monitor continuously
while ($true) { 
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv,noheader
    Start-Sleep 2 
}

# Reduce GPU power limit if overheating
nvidia-smi -pl 250  # Limit to 250W

# Alternative: Reduce mining intensity
.\scripts\windows\start-miner.ps1 -GPU -Threads 2 -Intensity 50
```

### Memory Issues
```powershell
# Check system memory usage
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory

# Check virtual memory
Get-CimInstance Win32_PageFileUsage

# Close other applications to free memory
Stop-Process -Name "chrome", "firefox", "steam" -ErrorAction SilentlyContinue

# Increase virtual memory if needed
# Control Panel -> System -> Advanced -> Performance Settings -> Advanced -> Virtual Memory
```

## 🚨 Emergency Recovery

### Complete Mining Reset
```powershell
# Stop all mining processes
Get-Process -Name "*quantum*", "*geth*" | Stop-Process -Force

# Remove any lock files
Remove-Item "C:\temp\quantum-miner.lock" -ErrorAction SilentlyContinue

# Rebuild miner
Remove-Item quantum-miner.exe -ErrorAction SilentlyContinue
.\scripts\windows\build-release.ps1

# Reset GPU state
nvidia-smi --gpu-reset

# Restart mining
.\scripts\windows\start-miner.ps1 -GPU -Threads 4
```

### System Recovery
```powershell
# Restart GPU drivers (requires Admin)
# Device Manager -> Display adapters -> NVIDIA GPU -> Right-click -> Disable
# Wait 5 seconds, then Enable

# Or use PowerShell (Admin required)
$gpu = Get-PnpDevice | Where-Object {$_.FriendlyName -like "*NVIDIA*"}
Disable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
Start-Sleep 5
Enable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false

# Check Windows Update for driver updates
Start-Process ms-settings:windowsupdate
```

## 📊 Monitoring and Diagnostics

### Performance Monitoring
```powershell
# Monitor mining statistics
Get-Content "mining.log" -Tail 20 -Wait

# System resource monitoring
while ($true) {
    $cpu = Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1
    $memory = Get-CimInstance Win32_OperatingSystem
    Write-Host "CPU: $($cpu.CounterSamples.CookedValue.ToString('N1'))% | Memory: $(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize * 100)%"
    Start-Sleep 2
}

# GPU monitoring
nvidia-smi dmon -s pucvmet -d 2  # Monitor every 2 seconds
```

### Diagnostic Information Collection
```powershell
# Collect system diagnostics
$DiagFile = "mining-diagnostics.txt"

"=== Windows Mining Diagnostics ===" | Out-File $DiagFile
"Timestamp: $(Get-Date)" | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"OS Information:" | Out-File $DiagFile -Append
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, Architecture | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"GPU Information:" | Out-File $DiagFile -Append
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"Python Information:" | Out-File $DiagFile -Append
python --version 2>&1 | Out-File $DiagFile -Append
python -c "import cupy; print('CuPy:', cupy.__version__)" 2>&1 | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"Memory Information:" | Out-File $DiagFile -Append
Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum | Out-File $DiagFile -Append
"" | Out-File $DiagFile -Append

"Mining Binary:" | Out-File $DiagFile -Append
Get-ChildItem quantum-miner.exe -ErrorAction SilentlyContinue | Out-File $DiagFile -Append

Get-Content $DiagFile
```

## 📚 Getting Help

### Information to Collect
When reporting Windows mining issues:

1. **Windows Version**: `Get-CimInstance Win32_OperatingSystem | Select Caption, Version`
2. **GPU Info**: `nvidia-smi --query-gpu=name,driver_version --format=csv`
3. **Python Version**: `python --version`
4. **CuPy Status**: `python -c "import cupy; print(cupy.__version__)"`
5. **Error Messages**: Full PowerShell output
6. **Event Logs**: Relevant Windows Event Log entries

## ✅ Windows Mining Checklist

### System Requirements
- [ ] Windows 10/11 (64-bit)
- [ ] NVIDIA GPU with CUDA support
- [ ] 8GB+ RAM (16GB+ recommended)
- [ ] Visual Studio Build Tools installed
- [ ] Python 3.8+ installed

### Dependencies
- [ ] NVIDIA drivers (latest)
- [ ] CUDA toolkit installed
- [ ] CuPy installed and working
- [ ] PowerShell execution policy allows scripts
- [ ] Windows Firewall configured
- [ ] Windows Defender exclusions set

### Performance
- [ ] Optimal thread count determined
- [ ] GPU acceleration working
- [ ] System temperatures stable
- [ ] No memory issues
- [ ] Network connectivity stable

**For persistent Windows mining issues, collect the diagnostic information above and seek help!** 