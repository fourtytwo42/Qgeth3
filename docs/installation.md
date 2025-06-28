# Installation Guide

Complete installation instructions for Q Coin on all supported platforms.

## 📋 System Requirements

### Minimum Requirements
- **CPU:** 1 vCPU, 2+ GHz
- **RAM:** 1GB (CPU mining: ~0.1-0.2 puzzles/sec)
- **Storage:** 10GB free space
- **OS:** Windows 10+ or Linux (Ubuntu 20.04+, CentOS 8+, Debian 11+)

### Recommended Requirements
- **CPU:** 2+ vCPU, 3+ GHz (CPU mining: ~0.4-0.8 puzzles/sec)
- **RAM:** 4GB+ (for building and mining)
- **Storage:** 20GB+ free space
- **GPU:** NVIDIA GPU with CUDA support (GPU mining: 1.5-3.0 puzzles/sec)
- **OS:** Ubuntu 22.04+ or Windows 11

## 🔧 Prerequisites

### Windows Prerequisites
- **Windows 10/11** (PowerShell scripts)
- **Visual Studio 2022 Build Tools** or **MinGW-w64** (for building from source)
- **Go 1.21+** (for building from source)
- **Python 3.8+** (for GPU mining only)

### Linux Prerequisites
- **Linux** (Ubuntu 20.04+, CentOS 8+, or similar)
- **Go 1.21+** (required for building)
- **Python 3.8+** (for GPU mining only)
- **Basic build tools** (`gcc`, `make`)
- **NVIDIA GPU + CUDA** (for maximum GPU performance)

## 🐧 Linux Installation

### Ubuntu/Debian Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Go compiler
sudo apt install -y golang-go

# Install Python for GPU mining (optional)
sudo apt install -y python3 python3-pip

# Install build tools
sudo apt install -y git build-essential curl wget

# Install quantum mining dependencies (for GPU acceleration)
pip3 install qiskit qiskit-aer numpy

# Verify installations
go version
python3 --version
```

### CentOS/RHEL/Fedora Installation
```bash
# Update system
sudo dnf update -y

# Install Go compiler
sudo dnf install -y golang

# Install Python for GPU mining (optional)
sudo dnf install -y python3 python3-pip

# Install build tools
sudo dnf install -y git gcc make curl wget

# Install quantum mining dependencies
pip3 install qiskit qiskit-aer numpy
```

### Manual Go Installation (if needed)
```bash
# Remove old Go installation
sudo rm -rf /usr/local/go

# Download and install latest Go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
rm go1.21.5.linux-amd64.tar.gz

# Add to PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version
```

### Linux Build & Setup
```bash
# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Make scripts executable
chmod +x quick-start.sh

# Build everything (auto-detects GPU capabilities)
./quick-start.sh build

# Expected output:
# ✅ Quantum-Geth built successfully: ./geth (CPU)
# ✅ Quantum-Miner built successfully: ./quantum-miner (GPU/CPU)
```

## 🪟 Windows Installation

### Visual Studio Build Tools
```powershell
# Download and install Visual Studio 2022 Build Tools
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Ensure these components are installed:
# - MSVC v143 - VS 2022 C++ x64/x86 build tools
# - Windows 10/11 SDK
# - CMake tools for Visual Studio
```

### Go Installation
```powershell
# Download and install Go from https://golang.org/dl/
# Or using Chocolatey:
choco install golang

# Verify installation
go version
```

### Python Installation (for GPU mining)
```powershell
# Download Python 3.8+ from https://python.org
# Or using Chocolatey:
choco install python

# Install quantum dependencies
pip install qiskit qiskit-aer numpy cupy-cuda11x

# Verify installation
python --version
```

### Windows Build & Setup
```powershell
# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Build releases (auto-detects Visual Studio)
.\scripts\windows\build-release.ps1

# The build system automatically:
# - Detects Visual Studio 2022 Build Tools
# - Sets up proper compiler environment  
# - Creates complete standalone packages
```

## 🌐 GPU Mining Setup

### NVIDIA GPU Support (Linux)
```bash
# Check for NVIDIA GPU
nvidia-smi

# Install CUDA toolkit (for native CUDA acceleration)
sudo apt install -y nvidia-cuda-toolkit

# OR install Qiskit-Aer for GPU (easier setup)
pip3 install qiskit-aer-gpu

# Verify GPU support
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

### GPU Build Detection
```bash
# The build system auto-detects GPU capabilities:
./scripts/linux/build-linux.sh miner

# Possible outputs:
# ✅ Quantum-Miner built successfully: ./quantum-miner (CUDA)      # Native CUDA
# ✅ Quantum-Miner built successfully: ./quantum-miner (Qiskit-GPU) # Python GPU
# ✅ Quantum-Miner built successfully: ./quantum-miner (CPU)       # CPU fallback
```

### Windows GPU Setup
```powershell
# Install CUDA toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Install CuPy for Python GPU acceleration
pip install cupy-cuda11x

# Verify GPU support
python -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'No GPU')"
```

## 🧪 Verification

### Test Installation
```bash
# Linux
./quick-start.sh build
./geth.bin version
./quantum-miner --help

# Windows
.\scripts\windows\build-release.ps1
.\geth.exe version
.\quantum-miner.exe --help
```

### Test GPU Support
```bash
# Linux GPU test
./scripts/linux/start-miner.sh --gpu --threads 1 --test

# Expected output if GPU working:
# ✅ GPU mining available - Using GPU mode
# ⚡ Linux GPU Batch Complete: 128 puzzles in 0.234s (547.0 puzzles/sec)
```

## 🎯 Post-Installation

### Quick Test Run
```bash
# Start testnet node
./scripts/linux/start-geth.sh testnet

# In another terminal, test mining
./scripts/linux/start-miner.sh --testnet --verbose
```

### Performance Optimization
```bash
# Set CPU governor to performance (Linux)
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf

# Optimize network
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🆘 Installation Troubleshooting

### Go Build Errors
```bash
# Go not found
export PATH=$PATH:/usr/local/go/bin
go version

# Go version too old
# Install newer Go following manual installation steps above
```

### GPU Build Issues
```bash
# CUDA not found (Linux)
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
export PATH=/usr/local/cuda/bin:$PATH

# Python GPU issues
pip3 install --upgrade qiskit qiskit-aer
python3 -c "import qiskit_aer; print('OK')"
```

### Windows Build Issues
```powershell
# Visual Studio not found
# Install Visual Studio 2022 Build Tools with C++ components

# Path issues
# Add Go to PATH in Environment Variables
```

### Memory Issues
```bash
# Low memory during build
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🔗 Next Steps

After successful installation:

1. **[Quick Start Guide](quick-start.md)** - Get your first node running
2. **[VPS Deployment](vps-deployment.md)** - Deploy to production
3. **[Mining Guide](mining.md)** - Optimize your mining setup
4. **[Troubleshooting](troubleshooting.md)** - Fix common issues

## ✅ Installation Checklist

### Windows
- [ ] Visual Studio 2022 Build Tools installed
- [ ] Go 1.21+ installed and in PATH
- [ ] Python 3.8+ installed (for GPU mining)
- [ ] GPU drivers and CUDA toolkit (for GPU mining)
- [ ] Repository cloned and build successful

### Linux
- [ ] Go 1.21+ installed
- [ ] Build tools installed (gcc, make)
- [ ] Python 3.8+ and pip installed
- [ ] GPU drivers and CUDA toolkit (for GPU mining)
- [ ] Repository cloned and build successful

### Both Platforms
- [ ] `geth` binary works and shows version
- [ ] `quantum-miner` binary works and shows help
- [ ] GPU support verified (if applicable)
- [ ] Firewall configured for ports 8545, 30303
- [ ] Ready to start mining! 🎉 