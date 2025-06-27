# Q Coin - Quantum Blockchain Platform

A complete quantum blockchain platform featuring **Q Coin** testnet with **Quantum-Geth** (quantum-enhanced Ethereum client) and **high-performance quantum miners** with both CPU and GPU acceleration support.

**🎉 NEW: Q Coin Testnet with User-Friendly Interface!**

## 🚀 Quick Start - Q Coin Testnet

### Windows Quick Start
```powershell
# Start the Q Coin testnet node (works like standard geth)
.\start-geth.ps1

# Start with mining enabled
.\start-geth.ps1 -mine -etherbase 0xYourAddress

# Show all options
.\start-geth.ps1 -help
```

### Linux Quick Start

**🎯 Option 1: Ultimate One-Command Setup (Easiest!)**
```bash
# One-liner: Downloads everything and sets up complete auto-service
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash

# Alternative with wget:
wget -qO- https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash

# This single command:
# ✅ Installs all dependencies (git, curl, golang, build tools)
# ✅ Downloads the Q Geth repository automatically
# ✅ Prepares VPS (memory checks, swap creation, firewall)
# ✅ Builds and configures Q Geth with auto-updating service
# ✅ Sets up crash recovery and GitHub monitoring
# No manual git clone, chmod, or dependency installation needed!
```

**Option 2: Full Auto-Service (Manual Clone)**
```bash
# Multi-step setup: Clone repo first, then run auto-service
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
sudo ./auto-geth-service.sh

# This automatically:
# ✅ Prepares VPS (memory checks, swap, dependencies)
# ✅ Clones, builds, and runs Q Geth
# ✅ Sets up GitHub monitoring (checks every 5 minutes)
# ✅ Auto-updates on new commits to main branch
# ✅ Crash recovery with 5-minute retry
# ✅ Persistent systemd service

# Management commands:
qgeth-service start     # Start the service
qgeth-service status    # Check status
qgeth-service logs geth # View logs
```

**Option 2: Manual Setup**
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y golang-go python3 python3-pip git

# Clone and build
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
chmod +x *.sh

# For low-memory VPS (<3GB RAM), prepare system first
sudo ./prepare-vps.sh  # Auto-checks memory and creates swap if needed

# Build with memory optimization
./build-linux.sh both

# Start testnet node
./start-geth.sh testnet

# Start mining (in another terminal)
./start-linux-miner.sh --testnet --verbose
```

## 🌐 Q Coin Testnet Details

- **Name:** Q Coin
- **Symbol:** Q  
- **Chain ID:** 73235
- **Block Time:** 12 seconds
- **Consensus:** QMPoW (Quantum Proof of Work)
- **Default Data Directory:** 
  - Windows: `%APPDATA%\Qcoin`
  - Linux: `~/.qcoin`

## 🐧 Linux Setup & Requirements

### System Requirements

**Minimum:**
- Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- 1 vCPU, 1GB RAM (CPU mining: ~0.1-0.2 puzzles/sec)
- 10GB storage

**Recommended:**
- Ubuntu 22.04+ 
- 2+ vCPU, 4GB+ RAM (CPU mining: ~0.4-0.8 puzzles/sec)
- 20GB+ storage
- NVIDIA GPU (GPU mining: 1.5-3.0 puzzles/sec)

### Linux Dependencies Installation

**Ubuntu/Debian:**
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

**CentOS/RHEL/Fedora:**
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

**Manual Go Installation (if needed):**
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
chmod +x *.sh

# Build everything (auto-detects GPU capabilities)
./build-linux.sh both

# Expected output:
# ✅ Quantum-Geth built successfully: ./geth (CPU)
# ✅ Quantum-Miner built successfully: ./quantum-miner (GPU/CPU)
```

### Linux GPU Mining Setup

**NVIDIA GPU Support:**
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

**Build with GPU Support:**
```bash
# The build system auto-detects GPU capabilities:
./build-linux.sh miner

# Possible outputs:
# ✅ Quantum-Miner built successfully: ./quantum-miner (CUDA)      # Native CUDA
# ✅ Quantum-Miner built successfully: ./quantum-miner (Qiskit-GPU) # Python GPU
# ✅ Quantum-Miner built successfully: ./quantum-miner (CPU)       # CPU fallback
```

### Linux Usage

**Start Q Coin Network:**
```bash
# Start testnet node (default)
./start-geth.sh testnet

# Start mainnet node
./start-geth.sh mainnet

# Start dev network
./start-geth.sh devnet

# Start with built-in mining
./start-geth.sh testnet --mining
```

**Start Mining:**
```bash
# Smart miner (auto-detects GPU/CPU)
./start-linux-miner.sh --testnet --verbose

# Force GPU mining
./start-linux-miner.sh --gpu --threads 8 --coinbase 0xYourAddress

# Force CPU mining
./start-linux-miner.sh --cpu --threads 4 --coinbase 0xYourAddress

# Connect to remote node
./start-linux-miner.sh --node http://192.168.1.100:8545 --verbose
```

**Manual Mining:**
```bash
# Direct miner usage
./quantum-miner -threads 8 -coinbase 0xYourAddress -node http://localhost:8545 -log
```

**Attach to the Geth JavaScript Console (Linux, using IPC):**
```bash
./geth.bin attach ipc:$HOME/.qcoin/geth.ipc
```
- Replace `$HOME/.qcoin/geth.ipc` with your actual IPC file path if different.
- For devnet: `$HOME/.qcoin/devnet/geth.ipc`
- This opens the interactive geth console for direct blockchain commands.

## 🤖 Auto-Updating Service (Recommended for VPS)

### Complete Auto-Service Setup

The `auto-geth-service.sh` script provides a complete "set it and forget it" solution for VPS deployments:

```bash
# One command sets up everything
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
sudo ./auto-geth-service.sh
```

**What it does:**
1. **VPS Preparation** - Runs `prepare-vps.sh` or creates basic setup
2. **UFW Firewall Setup** - Configures and enables firewall with required ports:
   - Port 22 (SSH) - Remote access
   - Port 8545 (HTTP RPC) - Geth API access
   - Port 30303 (P2P TCP/UDP) - Blockchain networking
   - Port 8546 (WebSocket) - WebSocket API access
3. **Memory Optimization** - Checks RAM, creates swap if needed (<3GB RAM)
4. **Dependency Installation** - Installs Go, Git, build tools, systemd
5. **Project Setup** - Clones repo to `/opt/qgeth/Qgeth3`
6. **Initial Build** - Builds geth with memory optimization
7. **Service Creation** - Creates 3 systemd services:
   - `qgeth-node.service` - Main geth service with crash recovery
   - `qgeth-github-monitor.service` - Monitors GitHub for updates
   - `qgeth-updater.service` - Handles updates when triggered
8. **Auto-Start Setup** - Enables services to start on boot

### Service Management Commands

```bash
# Service control
qgeth-service start         # Start both geth and GitHub monitor
qgeth-service stop          # Stop all services
qgeth-service restart       # Restart all services
qgeth-service status        # Show status of all services

# Monitoring
qgeth-service logs geth     # Follow geth logs
qgeth-service logs github   # Follow GitHub monitor logs
qgeth-service logs update   # Follow update logs
qgeth-service logs all      # Follow all logs

# Maintenance
qgeth-service update        # Trigger manual update
qgeth-service reset-crashes # Reset crash counter
qgeth-service version       # Show geth version
```

### Auto-Service Features

**✅ GitHub Auto-Updates:**
- Monitors `fourtytwo42/Qgeth3` main branch every 5 minutes
- Automatically detects new commits
- Stops geth → pulls latest → builds → restarts
- Keeps backups of last 5 versions
- Restores from backup if build fails

**✅ Crash Recovery:**
- Automatically restarts geth if it crashes
- 5-minute delay between restart attempts
- Infinite retries (until manually stopped)
- After 3+ crashes, triggers auto-update (fixes potential issues)
- Tracks crash count in `/opt/qgeth/logs/crash_count.txt`

**✅ Memory-Optimized Building:**
- Uses temporary directories for build artifacts
- Automatically creates 2GB swap on low-memory VPS
- Memory-efficient compiler flags
- Build verification before restarting service

**✅ Persistent Operation:**
- Services start automatically on VPS boot
- Runs with specific user (not root)
- Proper systemd integration
- Log rotation and backup management

### Configuration

The service runs with these defaults:
- **Network:** `testnet` (Q Coin Testnet)
- **API:** `--http.corsdomain "*" --http.api "eth,net,web3,personal,txpool"`
- **GitHub Check:** Every 5 minutes
- **Crash Retry:** Every 5 minutes
- **Logs:** `/opt/qgeth/logs/`
- **Backups:** `/opt/qgeth/backup/` (keeps last 5)

### File Locations

```
/opt/qgeth/
├── Qgeth3/                    # Main project directory
├── logs/
│   ├── geth-runner.log        # Geth service logs
│   ├── geth-output.log        # Geth stdout
│   ├── geth-error.log         # Geth stderr
│   ├── github-monitor.log     # GitHub monitoring
│   ├── update.log             # Update process logs
│   └── crash_count.txt        # Crash counter
├── backup/
│   ├── Qgeth3_20240101_120000/
│   └── ...                    # Auto-backups
└── scripts/
    ├── github-monitor.sh      # GitHub monitoring script
    ├── update-geth.sh         # Update handler
    └── run-geth.sh            # Geth runner with crash recovery
```

## 🌐 VPS Deployment Guide

### VPS Providers & Specs

**Recommended VPS Configurations:**

| Provider | vCPU | RAM | Storage | Est. Performance | Monthly Cost |
|----------|------|-----|---------|------------------|--------------|
| DigitalOcean | 2 | 2GB | 50GB | ~0.3 puzzles/sec | $12 |
| Vultr | 2 | 4GB | 80GB | ~0.4 puzzles/sec | $12 |
| Linode | 4 | 8GB | 160GB | ~0.6 puzzles/sec | $48 |
| AWS EC2 | 4 | 16GB | 100GB | ~0.8 puzzles/sec | $50 |
| **GPU VPS** | 4 | 16GB | 100GB + GPU | **2.0+ puzzles/sec** | $100+ |

### VPS Setup (One-Command Install)

```bash
# Ubuntu VPS quick setup
curl -fsSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/install-vps.sh | bash

# Or manual setup:
apt update && apt upgrade -y && \
apt install -y golang-go python3 python3-pip git build-essential && \
pip3 install qiskit qiskit-aer numpy && \
git clone https://github.com/fourtytwo42/Qgeth3.git && \
cd Qgeth3 && chmod +x *.sh && ./build-linux.sh both
```

### ⚠️ Important: Low-Memory VPS Build Fix

**If building on a VPS with 1-2GB RAM, you MUST add swap space to prevent build failures:**

```bash
# Add 3GB swap file (required for compilation)
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent (survives reboot)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
# Should show 3.0G swap available

# Now build normally
./build-linux.sh both
```

**Why is swap needed?**
- Go compilation uses significant memory during linking phase
- 1GB VPS runs out of memory during `quantum-geth` build
- 3GB swap provides enough virtual memory to complete compilation
- After build completes, swap usage returns to minimal levels

### VPS Security & Optimization

**Firewall Setup:**
```bash
# Allow Q Coin ports
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8545/tcp    # RPC
sudo ufw allow 30303/tcp   # P2P
sudo ufw enable
```

**Performance Optimization:**
```bash
# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf

# Optimize network
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Monitoring & Maintenance:**
```bash
# Monitor mining performance
./start-linux-miner.sh --verbose | grep "puzzles/sec"

# Monitor system resources
htop
iotop
nvidia-smi -l 1  # For GPU VPS

# Check geth sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545
```

## 🌐 VPS Deployment Guide

### VPS Providers & Specs

**Recommended VPS Configurations:**

| Provider | vCPU | RAM | Storage | Est. Performance | Monthly Cost |
|----------|------|-----|---------|------------------|--------------|
| DigitalOcean | 2 | 2GB | 50GB | ~0.3 puzzles/sec | $12 |
| Vultr | 2 | 4GB | 80GB | ~0.4 puzzles/sec | $12 |
| Linode | 4 | 8GB | 160GB | ~0.6 puzzles/sec | $48 |
| AWS EC2 | 4 | 16GB | 100GB | ~0.8 puzzles/sec | $50 |
| **GPU VPS** | 4 | 16GB | 100GB + GPU | **2.0+ puzzles/sec** | $100+ |

### VPS Setup (One-Command Install)

```bash
# Ubuntu VPS quick setup
curl -fsSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/install-vps.sh | bash

# Or manual setup:
apt update && apt upgrade -y && \
apt install -y golang-go python3 python3-pip git build-essential && \
pip3 install qiskit qiskit-aer numpy && \
git clone https://github.com/fourtytwo42/Qgeth3.git && \
cd Qgeth3 && chmod +x *.sh && ./build-linux.sh both
```

### ⚠️ Important: Low-Memory VPS Build Fix

**If building on a VPS with 1-2GB RAM, you MUST add swap space to prevent build failures:**

```bash
# Add 3GB swap file (required for compilation)
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent (survives reboot)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
# Should show 3.0G swap available

# Now build normally
./build-linux.sh both
```

**Why is swap needed?**
- Go compilation uses significant memory during linking phase
- 1GB VPS runs out of memory during `quantum-geth` build
- 3GB swap provides enough virtual memory to complete compilation
- After build completes, swap usage returns to minimal levels

### VPS Security & Optimization

**Firewall Setup:**
```bash
# Allow Q Coin ports
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8545/tcp    # RPC
sudo ufw allow 30303/tcp   # P2P
sudo ufw enable
```

**Performance Optimization:**
```bash
# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf

# Optimize network
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Monitoring & Maintenance:**
```bash
# Monitor mining performance
./start-linux-miner.sh --verbose | grep "puzzles/sec"

# Monitor system resources
htop
iotop
nvidia-smi -l 1  # For GPU VPS

# Check geth sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545
```

## 📁 Script Organization

**User Scripts (Testnet):**
- `start-geth.ps1` / `start-geth.sh` - User-friendly testnet node
- `start-miner.ps1` / `start-linux-miner.sh` - Smart mining with auto-detection

**Development Scripts:**
- `start-geth.ps1 devnet` / `start-geth.sh devnet` - Development node (uses $APPDATA\Qcoin\devnet\ or $HOME/.qcoin/devnet/)
- `dev-start-geth-mining.ps1` / `scripts/dev-start-geth-mining.sh` - Development mining
- `dev-reset-blockchain.ps1` / `dev-reset-blockchain.sh` - Reset development blockchain

**Build Scripts:**
- `build-release.ps1` - Windows release builder
- `build-linux.sh` - Linux build system with GPU auto-detection and memory optimization
- `prepare-vps.sh` - 🆕 VPS preparation script with automatic memory management
- `auto-geth-service.sh` - 🆕 Complete auto-updating service setup (one-command VPS setup)
- `bootstrap-qgeth.sh` - 🆕 **Ultimate one-command bootstrap** (downloads & sets up everything)

## 🏗️ Release Build System

The project now features a professional release system that creates distributable packages:

```powershell
# Windows: Build both quantum-geth and quantum-miner releases
.\build-release.ps1

# Build specific components
.\build-release.ps1 geth    # Build quantum-geth only
.\build-release.ps1 miner   # Build quantum-miner only
```

```bash
# Linux: Build both binaries to root directory
./build-linux.sh both

# Build specific components  
./build-linux.sh geth     # Build only quantum-geth -> ./geth
./build-linux.sh miner    # Build only quantum-miner -> ./quantum-miner

# Clean build
./build-linux.sh both --clean
```

**Release Features:**
- ✅ **Cross-Platform** - Windows releases, Linux direct binaries
- ✅ **GPU Auto-Detection** - Linux build detects CUDA/Qiskit-Aer automatically
- ✅ **Complete Standalone** - Includes executables, launchers, and documentation
- ✅ **Performance Optimized** - Release builds include all optimizations

## 📊 Performance Comparison

| Platform | Method | Performance | Dependencies | Best For |
|----------|--------|-------------|--------------|----------|
| **Linux** | **Native CUDA** | **~3.2 puzzles/sec** | CUDA toolkit, C++ dev tools | Maximum performance |
| **Linux** | **Qiskit-Aer GPU** | **~2.4 puzzles/sec** | Python, qiskit-aer | Good performance, easier setup |
| **Windows** | **CuPy GPU** | **~2.0 puzzles/sec** | Python, CuPy | Windows GPU acceleration |
| **Both** | **CPU** | ~0.3-0.8 puzzles/sec | None | Easy setup, any system |

### Expected VPS Performance

**CPU Mining:**
- 1 vCPU: ~0.1-0.2 puzzles/sec
- 2 vCPU: ~0.2-0.4 puzzles/sec  
- 4 vCPU: ~0.4-0.8 puzzles/sec
- 8 vCPU: ~0.6-1.2 puzzles/sec

**GPU Mining (Linux VPS):**
- GTX 1080 Ti: ~1.3 puzzles/sec (CUDA) / ~0.9 puzzles/sec (Qiskit)
- RTX 3080: ~2.1 puzzles/sec (CUDA) / ~1.6 puzzles/sec (Qiskit)
- RTX 4090: ~3.2 puzzles/sec (CUDA) / ~2.4 puzzles/sec (Qiskit)

## 🏗️ Project Structure

```
Qgeth3/
├── quantum-geth/          # Quantum-enhanced Ethereum client source
├── quantum-miner/         # Unified quantum miner source (CPU/GPU)
├── releases/              # 🆕 Built release packages (Windows)
│   ├── quantum-geth-*/   # Standalone geth distributions
│   └── quantum-miner-*/  # Standalone miner distributions
├── scripts/               # Blockchain management scripts
├── build-release.ps1     # 🆕 Professional build system (Windows)
├── build-linux.sh        # 🆕 Linux build system with GPU detection & memory optimization
├── prepare-vps.sh         # 🆕 VPS preparation script with automatic memory management
├── auto-geth-service.sh   # 🆕 Complete auto-updating service setup (one-command VPS)
├── bootstrap-qgeth.sh     # 🆕 **Ultimate one-command bootstrap** (downloads everything)
├── start-geth.sh         # 🐧 Linux geth launcher
├── start-linux-miner.sh  # 🐧 Linux smart miner launcher
├── geth                   # 🐧 Linux binary (created by build-linux.sh)
├── quantum-miner          # 🐧 Linux binary (created by build-linux.sh)
├── LINUX-GPU-MINING.md   # 🐧 Comprehensive Linux GPU guide
├── run-gpu-miner.ps1     # Windows GPU mining launcher
├── run-cpu-miner.ps1     # Windows CPU mining launcher
├── start-geth.ps1        # Windows blockchain node launcher
└── reset-blockchain.ps1  # Blockchain reset with dynamic genesis
```

## ⚛️ Quantum Mining Features

**✅ Real Quantum Circuits:**
- 16-qubit quantum circuits per puzzle
- 20 T-gates per puzzle for quantum complexity
- 128 quantum puzzles per block

**✅ Advanced Acceleration:**
- **Linux**: Native CUDA + Qiskit-Aer GPU support
- **Windows**: CuPy GPU acceleration
- **Both**: CPU fallback with optimization
- Batch processing optimization (128 puzzles in one call)
- Automatic fallback to CPU if GPU unavailable

**✅ Bitcoin-Style Mining:**
- Proof-of-Work consensus with quantum difficulty
- Dynamic difficulty adjustment (ASERT-Q algorithm)
- Real blockchain integration with halving rewards

## 🛠️ Setup Instructions

### Prerequisites

**Windows:**
- **Windows 10/11** (PowerShell scripts)
- **Visual Studio 2022 Build Tools** or **MinGW-w64** (for building from source)
- **Go 1.21+** (for building from source)
- **Python 3.8+** (for GPU mining only)

**Linux:**
- **Linux** (Ubuntu 20.04+, CentOS 8+, or similar)
- **Go 1.21+** (required for building)
- **Python 3.8+** (for GPU mining only)
- **Basic build tools** (`gcc`, `make`)
- **NVIDIA GPU + CUDA** (for maximum GPU performance)

### Windows Setup

**Option 1: Use Pre-Built Releases (Recommended)**
All root scripts automatically detect and use the newest releases. If no releases exist, they'll build them automatically.

**Option 2: Build Releases Manually**
```powershell
# Build both quantum-geth and quantum-miner releases
.\build-release.ps1

# The build system automatically:
# - Detects Visual Studio 2022 Build Tools
# - Sets up proper compiler environment  
# - Creates complete standalone packages
# - Includes launchers for Windows and Linux
```

### Linux Setup

**Quick Setup (Ubuntu/Debian):**
```bash
# Install dependencies
sudo apt update
sudo apt install -y golang-go python3 python3-pip git build-essential

# Clone and build
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
chmod +x *.sh
./build-linux.sh both

# Start mining
./start-geth.sh testnet
./start-linux-miner.sh --testnet --verbose
```

**GPU Mining Setup (Optional but Recommended):**
```bash
# For maximum performance (Native CUDA)
sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev

# For easier setup (Qiskit-Aer GPU)
pip3 install qiskit qiskit-aer numpy

# Verify GPU support
nvidia-smi
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

### VPS Setup (One Command)

```bash
# Complete VPS setup (Ubuntu)
curl -fsSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/install-vps.sh | bash
```

## 🎮 Detailed Usage

### Linux Usage Examples

**Basic Mining:**
```bash
# Start testnet node
./start-geth.sh testnet

# Start smart miner (auto-detects best acceleration)
./start-linux-miner.sh --testnet --verbose

# Expected output:
# ✅ GPU mining available - Using GPU mode
# ⚡ Linux GPU Batch Complete: 128 puzzles in 0.234s (547.0 puzzles/sec)
```

**Advanced Mining:**
```bash
# Force specific mining mode
./start-linux-miner.sh --gpu --threads 8 --coinbase 0xYourAddress
./start-linux-miner.sh --cpu --threads 16 --coinbase 0xYourAddress

# Connect to remote node
./start-linux-miner.sh --node http://192.168.1.100:8545 --mainnet

# Mining with custom settings
./quantum-miner -threads 4 -coinbase 0xYourAddress -node http://localhost:8545 -log
```

**Network Selection:**
```bash
# Q Coin Testnet (Chain ID 73235) - Default
./start-geth.sh testnet
./start-linux-miner.sh --testnet

# Q Coin Mainnet (Chain ID 73236)
./start-geth.sh mainnet  
./start-linux-miner.sh --mainnet

# Q Coin Dev Network (Chain ID 73234)
./start-geth.sh devnet
./start-linux-miner.sh --node http://localhost:8545
```

### Windows Usage (Release-Based)

All scripts now automatically use the newest release packages:

```powershell
# These scripts auto-detect the newest releases:
.\start-geth.ps1              # Uses releases/quantum-geth-*/geth.exe
.\start-geth-mining.ps1       # Uses releases/quantum-geth-*/geth.exe  
.\run-gpu-miner.ps1          # Uses releases/quantum-miner-*/quantum-miner.exe
.\run-cpu-miner.ps1          # Uses releases/quantum-miner-*/quantum-miner.exe

# If no releases exist, they automatically build them first
```

## 🔧 Advanced Configuration

### GPU Mining Optimization

**Linux CUDA Optimization:**
```bash
# Set GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export QISKIT_IN_PARALLEL=TRUE
export OPENBLAS_NUM_THREADS=1
```

**Multi-GPU Setup:**
```bash
# Mine on specific GPU
./start-linux-miner.sh --gpu --threads 4 --verbose
CUDA_VISIBLE_DEVICES=1 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545
```

### Custom Node Connection
```bash
# Connect to remote node
./start-linux-miner.sh --node http://192.168.1.100:8545 --coinbase 0xYourAddress

# Connect to different networks
./start-linux-miner.sh --node http://mainnet.qcoin.network:8545 --mainnet
```

### Performance Monitoring

```bash
# Monitor mining performance
./start-linux-miner.sh --verbose | grep "puzzles/sec"

# Monitor GPU usage (if available)
watch -n 1 nvidia-smi

# Monitor system resources
htop
iotop

# Check geth status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  http://localhost:8545
```

## 🏭 Build System Details

### Linux Build System
The Linux build system automatically detects GPU capabilities:

```bash
# Example build outputs:
🔍 NVIDIA GPU detected, checking CUDA availability...
✅ CUDA development environment found
🏗️  Build Configuration:
  GPU Type: CUDA
  Build Tags: cuda
  CGO Enabled: 1
✅ Quantum-Miner built successfully: ./quantum-miner (CUDA)
```

### Visual Studio 2022 Support (Windows)
The Windows build system automatically detects and uses Visual Studio 2022 Build Tools:
- Finds `vcvarsall.bat` automatically
- Sets up proper compiler environment
- Uses `CGO_ENABLED=0` for compatibility
- Handles complex project dependencies

## 📈 Mining Statistics

Both miners provide real-time statistics:
- **Puzzle Rate:** Quantum puzzles solved per second
- **Block Success Rate:** Percentage of successful block submissions
- **Hash Rate:** Equivalent traditional mining hash rate
- **Quantum Metrics:** Circuit execution time and success rates
- **GPU Utilization:** GPU memory and compute usage (if applicable)

## 🔍 Troubleshooting

### Linux Build Issues
```bash
# Go not found
sudo apt install golang-go
# OR
export PATH=$PATH:/usr/local/go/bin

# GPU build fails
sudo apt install nvidia-cuda-toolkit
pip3 install qiskit-aer

# Permission issues
chmod +x *.sh
sudo usermod -a -G video $USER  # For GPU access
```

### Linux Mining Issues
```bash
# Check if geth is running
curl http://localhost:8545

# Check GPU availability
nvidia-smi
python3 -c "import qiskit_aer; print('GPU OK')"

# Monitor miner logs
./start-linux-miner.sh --verbose | grep -E "(ERROR|puzzles/sec|GPU)"
```

### VPS Issues
```bash
# Check network connectivity
ping 8.8.8.8
curl -I google.com

# Check firewall
sudo ufw status
sudo ufw allow 8545/tcp
sudo ufw allow 30303/tcp

# Check system resources
free -h
df -h
top
```

### Performance Issues
```bash
# Optimize CPU performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check for thermal throttling
cat /proc/cpuinfo | grep MHz

# Monitor I/O
iotop -ao
```

## 🏆 Performance Tips

1. **Use GPU Mining:** 3-10x better performance than CPU-only
2. **VPS Selection:** Choose VPS with dedicated CPU cores vs shared
3. **Optimize Threads:** Start with CPU core count, adjust based on performance
4. **Monitor Resources:** Watch CPU/GPU usage to find optimal settings
5. **Network Optimization:** Use VPS in same region as other miners
6. **Keep Updated:** Rebuild regularly for latest optimizations

## 📚 Documentation

- **Linux GPU Mining:** See `LINUX-GPU-MINING.md` for comprehensive GPU setup
- **Build System:** Use `./build-linux.sh --help` for detailed build options
- **VPS Deployment:** This README covers complete VPS setup
- **Quantum-Miner:** See release packages for complete documentation
- **Quantum-Geth:** See `quantum-geth/README.md` for blockchain node documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test on both Windows and Linux
4. Test with both CPU and GPU miners
5. Use `./build-linux.sh` and `.\build-release.ps1` to create test builds
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the individual component licenses for details.

---

## 🎯 Getting Started Checklist

### Windows Users
- [ ] **Install Visual Studio 2022 Build Tools** (for building from source)
- [ ] **Install Python 3.8+** (for GPU mining)
- [ ] **Run `pip install qiskit qiskit-aer numpy`** (for GPU mining)
- [ ] **Build releases:** `.\build-release.ps1` (or let scripts auto-build)
- [ ] **Start quantum-geth:** `.\start-geth.ps1`
- [ ] **Start mining:** `.\run-gpu-miner.ps1 -Coinbase 0xYourAddress`

### Linux Users
- [ ] **Install dependencies:** `sudo apt install golang-go python3 python3-pip`
- [ ] **Install GPU support:** `pip3 install qiskit qiskit-aer numpy`
- [ ] **Clone repository:** `git clone https://github.com/fourtytwo42/Qgeth3.git`
- [ ] **Build everything:** `./build-linux.sh both`
- [ ] **Start quantum-geth:** `./start-geth.sh testnet`
- [ ] **Start mining:** `./start-linux-miner.sh --testnet --verbose`

### VPS Users
- [ ] **Choose VPS:** 2+ vCPU, 4GB+ RAM recommended
- [ ] **Run setup script:** `curl -fsSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/install-vps.sh | bash`
- [ ] **Configure firewall:** Allow ports 8545, 30303
- [ ] **Start mining:** `./start-geth.sh testnet && ./start-linux-miner.sh --testnet`
- [ ] **Monitor performance:** `htop` and `./start-linux-miner.sh --verbose`

**🎉 Professional quantum blockchain platform with cross-platform support!**
**Happy Quantum Mining! ⚛️💎** 