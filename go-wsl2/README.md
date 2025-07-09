# WSL2 GPU Mining Setup for Quantum-Geth

This directory contains the WSL2 GPU mining setup for high-performance quantum circuit mining using Linux GPU acceleration on Windows.

## Overview

The WSL2 GPU miner leverages Windows Subsystem for Linux 2 (WSL2) to run quantum simulations using Linux GPU drivers, which often provide better performance and compatibility than native Windows GPU support.

## Features

- **Linux GPU Acceleration**: Uses WSL2 to access Linux GPU drivers for better performance
- **Automatic Fallback**: Falls back to CPU mining if GPU initialization fails
- **Qiskit Integration**: Uses Qiskit-Aer for quantum circuit simulation with GPU acceleration
- **Cross-Platform**: Works with NVIDIA GPUs that support CUDA in WSL2

## Requirements

1. **Windows 11** or **Windows 10 version 21H2** or later
2. **WSL2** installed and enabled
3. **Ubuntu 20.04 or 22.04** WSL2 distribution (recommended)
4. **NVIDIA GPU** with CUDA support
5. **NVIDIA Windows drivers** with WSL2 support (version 470.76 or later)

## Installation Steps

### 1. Install WSL2 and Ubuntu

```powershell
# Enable WSL2 (requires restart)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart your computer, then set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```

### 2. Install NVIDIA Drivers

Download and install the latest NVIDIA drivers with WSL2 support from:
https://developer.nvidia.com/cuda/wsl

### 3. Set Up WSL2 Environment

Run the automated setup script:

```batch
cd C:\Users\%USERNAME%\OneDrive\Documents\GitHub\Qgeth3\go-wsl2
setup-wsl2.bat
```

This script will:
- Verify WSL2 installation
- Install Python 3 and pip in Ubuntu
- Install Qiskit and GPU dependencies
- Set up the Python wrapper script
- Test the environment

### 4. Manual Setup (Alternative)

If the automated setup fails, you can set up manually:

```bash
# In WSL2 Ubuntu terminal
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Install quantum computing packages
python3 -m pip install --user qiskit qiskit-aer cuquantum-python cupy-cuda12x numpy

# Test GPU access
python3 -c "import qiskit; print('Qiskit available!')"
```

## Usage

### Option 1: Use the PowerShell Launcher

```powershell
# Navigate to the project root
cd C:\Users\%USERNAME%\OneDrive\Documents\GitHub\Qgeth3

# Run WSL2 GPU miner
.\go-wsl2\run-wsl2-miner.ps1 -Coinbase 0xYourAddress -Threads 2
```

### Option 2: Manual Environment Setup

```powershell
# Set WSL2 environment variables
$env:WSL2_MODE = "true"
$env:PYTHON_EXEC = "wsl /tmp/qgeth-wsl2/python-linux.sh"

# Run the quantum miner
cd quantum-miner
.\quantum-miner-wsl2.exe -gpu -coinbase 0xYourAddress -threads 2 -gpu-id 0
```

## Configuration

### Environment Variables

- `WSL2_MODE=true` - Enables WSL2 mode in the quantum miner
- `PYTHON_EXEC=wsl /tmp/qgeth-wsl2/python-linux.sh` - Points to the WSL2 Python wrapper

### GPU Settings

- `-gpu` - Enable GPU mining mode
- `-gpu-id 0` - Select GPU device (default: 0)
- `-threads 2` - Number of mining threads (recommended: 1-4 for GPU mode)

## Performance

### Expected Performance

- **CPU Mining**: ~0.1-0.5 puzzles/second
- **WSL2 GPU Mining**: ~2-10 puzzles/second (depending on GPU)
- **Memory Usage**: ~500MB-2GB (depending on batch size)

### GPU Compatibility

| GPU Series | Performance | Notes |
|------------|-------------|-------|
| RTX 40xx   | Excellent   | Best performance with CUDA 12+ |
| RTX 30xx   | Very Good   | Good performance, widely tested |
| RTX 20xx   | Good        | Solid performance |
| GTX 16xx   | Moderate    | Limited by compute capability |
| GTX 10xx   | Basic       | May require older CUDA versions |

## Troubleshooting

### Common Issues

#### "WSL2 is not available"
- Install WSL2: `wsl --install`
- Update Windows to latest version
- Enable virtualization in BIOS

#### "Python3 is not installed in WSL2"
```bash
sudo apt update
sudo apt install -y python3 python3-pip
```

#### "Qiskit not found"
```bash
python3 -m pip install --user qiskit qiskit-aer numpy
```

#### "GPU initialization failed"
- Update NVIDIA drivers
- Check GPU compatibility: `nvidia-smi` in WSL2
- Verify CUDA installation: `nvcc --version`

#### "Permission denied" errors
```bash
# In WSL2
chmod +x /tmp/qgeth-wsl2/python-linux.sh
```

### Performance Issues

#### Low GPU utilization
- Increase batch size in `qiskit_gpu.py`
- Reduce number of mining threads
- Check GPU memory usage

#### High memory usage
- Reduce `-threads` parameter
- Close unnecessary applications
- Check for memory leaks in logs

### Debugging

#### Enable verbose logging
```powershell
.\quantum-miner-wsl2.exe -gpu -coinbase 0xYourAddress -log
```

#### Check WSL2 GPU access
```bash
# In WSL2
nvidia-smi
python3 -c "import cupy; print(f'GPUs: {cupy.cuda.runtime.getDeviceCount()}')"
```

#### Test quantum simulation
```bash
# In WSL2
cd /tmp/qgeth-wsl2
python3 -c "from quantum_miner.pkg.quantum import qiskit_gpu; print('GPU test passed')"
```

## Performance Tuning

### Optimal Settings

For most systems:
```powershell
.\go-wsl2\run-wsl2-miner.ps1 -Coinbase 0xYourAddress -Threads 2 -GpuId 0
```

For high-end GPUs:
```powershell
.\go-wsl2\run-wsl2-miner.ps1 -Coinbase 0xYourAddress -Threads 4 -GpuId 0
```

### Memory Management

- Monitor GPU memory: `nvidia-smi` in WSL2
- Reduce threads if GPU memory is full
- Close other GPU applications while mining

## Support

### Getting Help

1. Check the main project README
2. Review troubleshooting section above
3. Check WSL2 installation: `wsl --status`
4. Verify GPU drivers: `nvidia-smi`

### Reporting Issues

When reporting issues, include:
- Windows version: `winver`
- WSL2 status: `wsl --status`
- GPU info: `nvidia-smi` (in WSL2)
- Error messages from miner logs

## Files in this Directory

- `python-linux.sh` - WSL2 Python wrapper script
- `setup-wsl2.bat` - Automated WSL2 setup script
- `run-wsl2-miner.ps1` - PowerShell launcher for WSL2 mining
- `README.md` - This documentation file

## Advanced Configuration

### Custom Python Path

```powershell
$env:PYTHON_EXEC = "wsl /usr/local/bin/python3.11"
```

### Multiple GPUs

```powershell
# Use GPU 1 instead of GPU 0
.\go-wsl2\run-wsl2-miner.ps1 -Coinbase 0xYourAddress -GpuId 1
```

### Custom CUDA Settings

```bash
# In python-linux.sh, add:
export CUDA_VISIBLE_DEVICES=0,1  # Use multiple GPUs
export CUDA_CACHE_PATH=/tmp/cuda_cache_custom
```

This setup provides a robust, high-performance quantum mining solution using WSL2 GPU acceleration. 