# Windows Mining Guide

Complete guide to quantum mining on Windows with Q Coin, including GPU acceleration with CuPy and CPU fallback options.

## ⚛️ Quantum Mining Overview

### Real Quantum Computing Features
- **16-qubit quantum circuits** per puzzle
- **20 T-gates per puzzle** for quantum complexity
- **128 quantum puzzles per block**
- **Bitcoin-style Proof-of-Work** with quantum difficulty
- **Dynamic difficulty adjustment** (ASERT-Q algorithm)
- **Real blockchain integration** with halving rewards

### Windows GPU Acceleration

Q Coin Windows supports **CuPy GPU acceleration**:

#### **CuPy GPU** (Windows GPU Acceleration)
- Python-based GPU acceleration using CuPy
- Good performance for Windows systems
- Works with NVIDIA GPUs on Windows
- **Performance**: ~2.0 puzzles/sec (RTX 4090)

#### **CPU Fallback**
- Universal compatibility
- Optimized for multi-core systems
- **Performance**: ~0.3-0.8 puzzles/sec

## 🔧 Prerequisites

### System Requirements
- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support (for GPU mining)
- **8GB+ RAM** (16GB+ recommended)
- **Python 3.8+**
- **Visual Studio Build Tools** (for compilation)

### Visual Studio Build Tools
```powershell
# Download and install Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install Visual Studio Community (includes build tools)
# https://visualstudio.microsoft.com/vs/community/

# Required components:
# - MSVC v143 compiler toolset
# - Windows 10/11 SDK
# - CMake tools for C++
```

### For GPU Mining
```powershell
# Install CUDA toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Install Python dependencies
pip install qiskit qiskit-aer numpy cupy-cuda11x

# Verify GPU support
python -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'No GPU')"
```

### For CPU-Only Mining
```powershell
# Install Go
# https://golang.org/dl/

# Install Python
# https://www.python.org/downloads/

# Install Git
# https://git-scm.com/download/win

# Basic Python dependencies
pip install qiskit qiskit-aer numpy
```

## 🏗️ Building the Miner

### Using the Build Script
```powershell
# Auto-detect and build with best available GPU support
.\scripts\windows\build-release.ps1

# Build miner only
.\scripts\windows\build-release.ps1 -Target miner

# Build with specific configuration
.\scripts\windows\build-release.ps1 -Target miner -Configuration Release
```

### Build Output Examples

**GPU Build (CuPy Available):**
```
🔍 Checking GPU capabilities...
✅ CuPy GPU support detected
🏗️  Build Configuration:
  GPU Type: CuPy
  CGO Enabled: 0
✅ Quantum-Miner built successfully
Created release: releases/quantum-miner-TIMESTAMP
```

**CPU Build:**
```
🔍 Checking GPU capabilities...
⚠️  No GPU acceleration available, building CPU-only version
🏗️  Build Configuration:
  GPU Type: CPU
  CGO Enabled: 0
✅ Quantum-Miner built successfully
Created release: releases/quantum-miner-TIMESTAMP
```

## 🚀 Mining Usage

### Using PowerShell Scripts

#### Quick Start
```powershell
# GPU mining (auto-detects if available)
.\scripts\windows\start-miner.ps1 -GPU -Coinbase 0xYourAddress

# CPU mining
.\scripts\windows\start-miner.ps1 -CPU -Threads 4 -Coinbase 0xYourAddress

# Auto-detect best method
.\scripts\windows\start-miner.ps1 -Coinbase 0xYourAddress -Verbose
```

#### Network Selection
```powershell
# Q Coin Testnet (default)
.\scripts\windows\start-miner.ps1 -Network testnet -Coinbase 0xYourAddress

# Q Coin Mainnet
.\scripts\windows\start-miner.ps1 -Network mainnet -Coinbase 0xYourAddress

# Custom node connection
.\scripts\windows\start-miner.ps1 -Node "http://192.168.1.100:8545" -Coinbase 0xYourAddress
```

#### Advanced Options
```powershell
# Full option example
.\scripts\windows\start-miner.ps1 `
  -GPU `
  -Threads 8 `
  -Coinbase "0x1234567890123456789012345678901234567890" `
  -Node "http://localhost:8545" `
  -Verbose

# Built-in mining with node
.\scripts\windows\start-geth.ps1 -Network testnet -Mining -Coinbase 0xYourAddress
```

### Using Release Packages

#### Extract and Run
```powershell
# Extract release package
Expand-Archive -Path "quantum-miner-TIMESTAMP.zip" -DestinationPath "C:\qcoin-mining"
cd "C:\qcoin-mining\quantum-miner-TIMESTAMP"

# Run with provided launcher
.\start-miner.ps1 -GPU -Coinbase 0xYourAddress

# Or run directly
.\quantum-miner.exe -threads 4 -coinbase 0xYourAddress -node http://localhost:8545
```

## 📊 Performance Comparison

| Method | RTX 4090 | RTX 3080 | GTX 1080 Ti | Setup Difficulty |
|--------|----------|----------|-------------|------------------|
| **CuPy GPU** | ~2.0 puzzles/sec | ~1.4 puzzles/sec | ~0.8 puzzles/sec | Medium |
| **CPU (16 cores)** | ~0.4 puzzles/sec | ~0.3 puzzles/sec | ~0.2 puzzles/sec | Easy |

*Note: Windows CuPy performance is slightly lower than Linux native CUDA but still significantly better than CPU-only mining.*

## 🔧 Optimization Tips

### GPU Optimization
```powershell
# Set GPU performance mode (requires admin privileges)
nvidia-smi -pm 1
nvidia-smi -pl 300  # Set power limit (adjust for your GPU)

# Windows power settings
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance mode
```

### Environment Variables
```powershell
# For CuPy GPU acceleration
$env:CUPY_CACHE_DIR = "C:\temp\cupy_cache"
$env:CUPY_DUMP_CUDA_SOURCE_ON_ERROR = "1"

# Memory optimization
$env:GOMEMLIMIT = "2GiB"

# Temporary directory (if needed)
$env:TEMP = "D:\mining-temp"  # Use fast drive
```

### PowerShell Execution Policy
```powershell
# Enable script execution (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine

# Or for current user only
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Multi-GPU Setup
```powershell
# Run multiple miners for multiple GPUs (each in separate PowerShell window)
Start-Process powershell -ArgumentList "-Command", "cd '$PWD'; $env:CUDA_VISIBLE_DEVICES=0; .\quantum-miner.exe -threads 4 -coinbase 0xAddr -node http://localhost:8545"
Start-Process powershell -ArgumentList "-Command", "cd '$PWD'; $env:CUDA_VISIBLE_DEVICES=1; .\quantum-miner.exe -threads 4 -coinbase 0xAddr -node http://localhost:8545"
```

## 🎮 Real-Time Monitoring

### GPU Monitoring
```powershell
# Monitor GPU usage (requires NVIDIA drivers)
nvidia-smi -l 1

# PowerShell GPU monitoring
while ($true) {
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    Start-Sleep 5
}
```

### Performance Monitoring
```powershell
# Monitor miner performance
.\scripts\windows\start-miner.ps1 -Verbose -Coinbase 0xYourAddress | Tee-Object -FilePath "mining-log.txt"

# Filter performance metrics
Get-Content "mining-log.txt" | Select-String "puzzles/sec"

# Monitor system resources
Get-Process | Where-Object {$_.ProcessName -like "*quantum*"} | Format-Table ProcessName,CPU,WorkingSet
```

### Windows Performance Toolkit
```powershell
# Task Manager monitoring
tasklist /fi "imagename eq quantum-miner.exe"

# Resource monitoring
Get-Counter "\Processor(*)\% Processor Time" -Continuous
Get-Counter "\Memory\Available MBytes" -Continuous
```

## 🔍 Troubleshooting

### GPU Not Detected
```powershell
# Check NVIDIA GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Check CuPy installation
python -c "import cupy; print(f'CuPy version: {cupy.__version__}'); print(f'CUDA available: {cupy.cuda.is_available()}')"

# Reinstall CuPy if needed
pip uninstall cupy-cuda11x
pip install cupy-cuda11x
```

### Build Issues

**Visual Studio Build Tools Missing:**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Check if build tools are available
where cl  # Should show path to MSVC compiler
```

**Go Build Failures:**
```powershell
# Check Go installation
go version

# Update Go if needed
# Download from: https://golang.org/dl/

# Clean Go module cache
go clean -modcache
```

**Python Dependencies:**
```powershell
# Reinstall Python dependencies
pip uninstall qiskit qiskit-aer cupy-cuda11x
pip install qiskit qiskit-aer cupy-cuda11x

# Check Python version
python --version  # Should be 3.8+
```

### Runtime Issues

**Permission Denied:**
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell -> "Run as Administrator"

# Check execution policy
Get-ExecutionPolicy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**GPU Memory Errors:**
```powershell
# Reduce thread count
.\scripts\windows\start-miner.ps1 -GPU -Threads 2 -Coinbase 0xYourAddress

# Close unnecessary applications
Stop-Process -Name "chrome", "firefox" -Force  # Close browsers
```

**Network Issues:**
```powershell
# Check if node is responding
Invoke-RestMethod -Uri "http://localhost:8545" -Method POST -ContentType "application/json" -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'

# Check Windows Firewall
Get-NetFirewallRule -DisplayName "*8545*"
New-NetFirewallRule -DisplayName "Q Coin RPC" -Direction Inbound -Protocol TCP -LocalPort 8545 -Action Allow
```

## 🏆 Best Practices

### Hardware Optimization
1. **Dedicated Mining PCs**: Use separate computer for mining if possible
2. **Multiple GPUs**: Install multiple NVIDIA cards for increased performance
3. **Adequate Cooling**: Ensure proper case ventilation
4. **Power Supply**: Use high-quality PSU with adequate wattage
5. **Fast Storage**: Use SSD for better I/O performance

### Software Optimization
1. **Keep Updated**: Regularly update drivers and rebuild miner
2. **Monitor Resources**: Use Task Manager and GPU monitoring tools
3. **Thread Tuning**: Start with CPU core count, adjust based on performance
4. **Antivirus Exclusions**: Add miner directory to antivirus exclusions
5. **Power Settings**: Use High Performance power plan

### Operational Best Practices
1. **Windows Services**: Create Windows services for automatic startup
2. **Monitoring**: Set up alerts for miner failures
3. **Security**: Use dedicated mining addresses, enable Windows Defender
4. **Backup Configurations**: Save working configurations
5. **Update Management**: Plan updates during low-activity periods

## 🎯 Windows Service Setup

### Create Mining Service
```powershell
# Install NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

# Create service (run as Administrator)
nssm install QCoinMiner
nssm set QCoinMiner Application "C:\path\to\quantum-miner.exe"
nssm set QCoinMiner Parameters "-threads 4 -coinbase 0xYourAddress -node http://localhost:8545"
nssm set QCoinMiner DisplayName "Q Coin Quantum Miner"
nssm set QCoinMiner Description "Q Coin Quantum Mining Service"
nssm set QCoinMiner Start SERVICE_AUTO_START

# Start service
nssm start QCoinMiner

# Check service status
Get-Service QCoinMiner
```

### Service Management
```powershell
# Start/Stop service
Start-Service QCoinMiner
Stop-Service QCoinMiner
Restart-Service QCoinMiner

# View service logs (if configured)
Get-WinEvent -LogName Application | Where-Object {$_.ProviderName -eq "QCoinMiner"}

# Remove service
nssm remove QCoinMiner confirm
```

## 🛡️ Security Considerations

### Windows Defender Exclusions
```powershell
# Add mining folder to exclusions (run as Administrator)
Add-MpPreference -ExclusionPath "C:\path\to\Qgeth3"
Add-MpPreference -ExclusionProcess "quantum-miner.exe"
Add-MpPreference -ExclusionProcess "geth.exe"

# Verify exclusions
Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
```

### Firewall Configuration
```powershell
# Allow mining applications through firewall
New-NetFirewallRule -DisplayName "Q Coin Miner" -Direction Inbound -Program "C:\path\to\quantum-miner.exe" -Action Allow
New-NetFirewallRule -DisplayName "Q Coin Geth" -Direction Inbound -Program "C:\path\to\geth.exe" -Action Allow

# Allow RPC ports
New-NetFirewallRule -DisplayName "Q Coin RPC" -Direction Inbound -Protocol TCP -LocalPort 8545 -Action Allow
New-NetFirewallRule -DisplayName "Q Coin P2P" -Direction Inbound -Protocol TCP -LocalPort 30303 -Action Allow
```

## 🔗 Next Steps

### Mining Setup Complete
After successful mining setup:

1. **Monitor Performance**: Use the monitoring tools above
2. **Scale Operations**: Consider multiple miners or dedicated hardware
3. **Join Community**: Connect with other miners for Windows-specific tips
4. **Stay Updated**: Keep drivers and software updated

### Performance Tuning
- **Thread Count**: Experiment with different thread counts
- **GPU Settings**: Use MSI Afterburner for GPU overclocking
- **System Optimization**: Disable unnecessary Windows services
- **Network Optimization**: Ensure fast connection to blockchain node

## ✅ Windows Mining Checklist

### Pre-Mining Setup
- [ ] Visual Studio Build Tools installed
- [ ] NVIDIA drivers installed (for GPU mining)
- [ ] CUDA toolkit installed (for GPU mining)
- [ ] Python and CuPy installed (for GPU mining)
- [ ] PowerShell execution policy configured
- [ ] Miner built and tested

### Mining Operation
- [ ] Miner starting successfully
- [ ] GPU utilization optimal (if using GPU)
- [ ] Thread count optimized
- [ ] Mining address configured
- [ ] Performance meeting expectations
- [ ] Windows Defender exclusions set

### Ongoing Monitoring
- [ ] GPU temperatures monitored
- [ ] Performance metrics tracked
- [ ] System resources monitored
- [ ] Mining logs reviewed regularly
- [ ] Software kept up to date
- [ ] Windows updates managed

**🎉 Happy Windows Quantum Mining! 🪟⚛️💎** 