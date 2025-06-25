# Q Coin - Quantum Blockchain Platform

A complete quantum blockchain platform featuring **Q Coin** testnet with **Quantum-Geth** (quantum-enhanced Ethereum client) and **high-performance quantum miners** with both CPU and GPU acceleration support.

**🎉 NEW: Q Coin Testnet with User-Friendly Interface!**

## 🚀 Quick Start - Q Coin Testnet

### 1. Start Q Coin Testnet Node
```powershell
# Start the Q Coin testnet node (works like standard geth)
.\start-geth.ps1

# Start with mining enabled
.\start-geth.ps1 -mine -etherbase 0xYourAddress

# Show all options
.\start-geth.ps1 -help
```

### 2. Use External Miners (Optional)

**🎮 GPU Mining (Recommended - Best Performance)**
```powershell
# High-performance GPU mining with Qiskit acceleration
.\dev-run-gpu-miner.ps1 -Coinbase 0xYourAddress

# Show all GPU mining options
.\dev-run-gpu-miner.ps1 -Help
```

**💻 CPU Mining (No Dependencies)**
```powershell
# CPU-only mining (no additional setup required)  
.\dev-run-cpu-miner.ps1 -Coinbase 0xYourAddress

# Show all CPU mining options  
.\dev-run-cpu-miner.ps1 -Help
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

## 📁 Script Organization

**User Scripts (Testnet):**
- `start-geth.ps1` / `scripts/start-geth.sh` - User-friendly testnet node

**Development Scripts:**
- `dev-start-geth.ps1` / `scripts/dev-start-geth.sh` - Development node (uses qdata/)
- `dev-start-geth-mining.ps1` / `scripts/dev-start-geth-mining.sh` - Development mining
- `dev-run-gpu-miner.ps1` / `scripts/dev-run-gpu-miner.sh` - External GPU mining
- `dev-run-cpu-miner.ps1` / `scripts/dev-run-cpu-miner.sh` - External CPU mining
- `dev-reset-blockchain.ps1` / `scripts/dev-reset-blockchain.sh` - Reset development blockchain

## 🏗️ Release Build System

The project now features a professional release system that creates distributable packages:

```powershell
# Build both quantum-geth and quantum-miner releases
.\build-release.ps1

# Build specific components
.\build-release.ps1 geth    # Build quantum-geth only
.\build-release.ps1 miner   # Build quantum-miner only

# Get help
.\build-release.ps1 -Help
```

**Release Features:**
- ✅ **Visual Studio 2022 Support** - Works with VS Build Tools
- ✅ **Timestamped Packages** - Each build creates `releases/quantum-*-timestamp/`
- ✅ **Complete Standalone** - Includes executables, launchers, and documentation
- ✅ **Auto-Detection** - Root scripts automatically use newest releases
- ✅ **Cross-Platform** - Windows PowerShell and Linux bash launchers

## 🐧 Linux Compilation

For Linux users, Q Coin provides a streamlined build system that compiles binaries directly to the root directory:

```bash
# Build both geth and miner to root directory (recommended)
./build-linux.sh

# Build specific components
./build-linux.sh geth     # Build only quantum-geth -> ./geth
./build-linux.sh miner    # Build only quantum-miner -> ./quantum-miner

# Clean build both
./build-linux.sh both --clean

# Alternative: build from scripts directory
cd scripts/
./build-linux.sh          # Calls root build script
```

**Linux Build Features:**
- ✅ **Root Directory Output** - Binaries compile to `./geth` and `./quantum-miner`
- ✅ **No Release Packages** - Direct executable approach for easier usage
- ✅ **Automatic Dependencies** - Creates `./quantum_solver.py` helper script
- ✅ **Quick Start Commands** - Shows exact commands to run after build
- ✅ **Both Scripts Work** - Root and scripts versions produce same result

### Linux Quick Start
```bash
# 1. Build everything
./build-linux.sh

# 2. Initialize blockchain
./geth --datadir qdata init genesis_quantum_testnet.json

# 3. Start node (external mining)
./geth --datadir qdata --networkid 73235 --mine --miner.threads 0 \
       --http --http.api eth,net,web3,personal,admin,miner \
       --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

# 4. Start mining (in another terminal)
./quantum-miner -rpc-url http://127.0.0.1:8545 \
                -address 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
```

### Linux Script Management
Use the scripts directory for full blockchain management:
```bash
cd scripts/

# Initialize fresh blockchain
./dev-reset-blockchain.sh --difficulty 1 --force

# Start development node
./dev-start-geth.sh

# Start mining (in another terminal)
./start-cpu-miner.sh --address 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
```

## 📊 Performance Comparison

| Mining Method | Performance | Dependencies | Best For |
|---------------|-------------|--------------|----------|
| **GPU (Qiskit)** | **0.45 puzzles/sec** | Python 3.8+, Qiskit | Maximum performance |
| **CPU** | 0.36 puzzles/sec | None | Easy setup |

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
├── build-linux.sh        # 🆕 Linux build system (outputs to root)
├── geth                   # 🐧 Linux binary (created by build-linux.sh)
├── quantum-miner          # 🐧 Linux binary (created by build-linux.sh)
├── quantum_solver.py      # 🐧 Linux helper script
├── run-gpu-miner.ps1     # GPU mining launcher (auto-detects releases)
├── run-cpu-miner.ps1     # CPU mining launcher (auto-detects releases)
├── start-geth.ps1        # Blockchain node launcher (auto-detects releases)
├── start-geth-mining.ps1 # Built-in mining launcher (auto-detects releases)
└── reset-blockchain.ps1  # Blockchain reset with dynamic genesis
```

## ⚛️ Quantum Mining Features

**✅ Real Quantum Circuits:**
- 16-qubit quantum circuits per puzzle
- 8192 T-gates per puzzle for quantum complexity
- 48 quantum puzzles per block

**✅ Advanced Acceleration:**
- Unified executable with CPU/GPU modes
- Qiskit-based GPU quantum simulation  
- Batch processing optimization (48 puzzles in one call)
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

**Build and Use (Recommended)**
```bash
# Build both binaries to root directory
./build-linux.sh

# Start using immediately
./geth --datadir qdata init genesis_quantum_testnet.json
./geth --datadir qdata --networkid 73235 --mine --miner.threads 0 --http

# Start mining in another terminal
./quantum-miner -rpc-url http://127.0.0.1:8545 -address 0xYourAddress
```

### GPU Mining Setup (Optional)
```bash
# Linux
pip3 install qiskit qiskit-aer numpy

# Windows
pip install qiskit qiskit-aer numpy
```

## 🎮 Detailed Usage

### Release-Based Usage (Recommended)

All scripts now automatically use the newest release packages:

```powershell
# These scripts auto-detect the newest releases:
.\start-geth.ps1              # Uses releases/quantum-geth-*/geth.exe
.\start-geth-mining.ps1       # Uses releases/quantum-geth-*/geth.exe  
.\run-gpu-miner.ps1          # Uses releases/quantum-miner-*/quantum-miner.exe
.\run-cpu-miner.ps1          # Uses releases/quantum-miner-*/quantum-miner.exe

# If no releases exist, they automatically build them first
```

### Direct Release Usage

```powershell
# Use specific release directly
cd releases\quantum-geth-1234567890
.\start-geth.ps1

cd releases\quantum-miner-1234567890  
.\start-miner-gpu.ps1 -Coinbase 0xYourAddress
.\start-miner-cpu.ps1 -Coinbase 0xYourAddress
```

### Development Usage

```powershell
# Build and run from source (development)
cd quantum-miner
go build -o quantum-miner.exe .
.\quantum-miner.exe -coinbase 0xYourAddress

# GPU mode
.\quantum-miner.exe -gpu -coinbase 0xYourAddress

# CPU mode (default)
.\quantum-miner.exe -coinbase 0xYourAddress
```

## 🔧 Advanced Configuration

### Custom Node Connection
```powershell
# Connect to remote node
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -NodeURL http://192.168.1.100:8545

# Multiple GPU devices
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -GpuId 1
```

### Multi-threaded Mining
```powershell
# CPU mining with 8 threads
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress -Threads 8

# GPU mining with 4 parallel circuits
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -Threads 4
```

### Release Management
```powershell
# Build new releases
.\build-release.ps1 both

# Clean old releases and build fresh
.\build-release.ps1 both -Clean

# Reset blockchain with new releases
.\reset-blockchain.ps1 -difficulty 1 -force
```

## 🏭 Build System Details

### Visual Studio 2022 Support
The build system automatically detects and uses Visual Studio 2022 Build Tools:
- Finds `vcvarsall.bat` automatically
- Sets up proper compiler environment
- Uses `CGO_ENABLED=0` for compatibility
- Handles complex project dependencies

### Release Package Contents

**Quantum-Geth Release (`releases/quantum-geth-*/`):**
- `geth.exe` - Quantum-enhanced Ethereum client
- `genesis_quantum.json` - Default quantum blockchain genesis
- `start-geth.ps1/.bat` - Node launchers (Windows/Linux)
- `start-geth-mining.ps1/.bat` - Built-in mining launchers
- `README.md` - Complete usage documentation

**Quantum-Miner Release (`releases/quantum-miner-*/`):**
- `quantum-miner.exe` - Unified CPU/GPU quantum miner
- `pkg/` - All dependencies (CUDA libraries, Python scripts)
- `start-miner-cpu.ps1/.bat` - CPU mining launchers
- `start-miner-gpu.ps1/.bat` - GPU mining launchers  
- `README.md` - Complete mining documentation

## 📈 Mining Statistics

Both miners provide real-time statistics:
- **Puzzle Rate:** Quantum puzzles solved per second
- **Block Success Rate:** Percentage of successful block submissions
- **Hash Rate:** Equivalent traditional mining hash rate
- **Quantum Metrics:** Circuit execution time and success rates

## 🔍 Troubleshooting

### Build Issues
```powershell
# Visual Studio not found
# Install Visual Studio 2022 Build Tools from Microsoft

# Go compiler issues  
go version  # Ensure Go 1.21+

# Build manually if needed
.\build-release.ps1 -Help
```

### GPU Mining Issues
```powershell
# Check Python installation
python --version

# Install/update Qiskit
pip install --upgrade qiskit qiskit-aer numpy

# Test GPU miner help
.\run-gpu-miner.ps1 -Help
```

### CPU Mining Issues
```powershell
# Check if quantum-geth is running
curl http://localhost:8545

# Test CPU miner help
.\run-cpu-miner.ps1 -Help
```

### Release Issues
```powershell
# Rebuild releases if corrupted
.\build-release.ps1 both -Clean

# Check release contents
Get-ChildItem releases\ -Recurse
```

## 🏆 Performance Tips

1. **Use Release Packages:** Pre-built releases are optimized and include all dependencies
2. **Use GPU Mining:** 25% better performance than CPU-only
3. **Optimize Threads:** Start with 1-2 threads, increase based on system
4. **Monitor Resources:** Watch CPU/GPU usage to find optimal settings
5. **Keep Releases Updated:** Rebuild releases regularly for latest optimizations

## 📚 Documentation

- **Build System:** Use `.\build-release.ps1 -Help` for detailed build options
- **Quantum-Miner:** See release packages for complete documentation
- **Quantum-Geth:** See `quantum-geth/README.md` for blockchain node documentation
- **Scripts:** See `scripts/README.md` for blockchain management tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both CPU and GPU miners
5. Use `.\build-release.ps1` to create test releases
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the individual component licenses for details.

---

## 🎯 Getting Started Checklist

- [ ] **Install Visual Studio 2022 Build Tools** (for building from source)
- [ ] **Install Python 3.8+** (for GPU mining)
- [ ] **Run `pip install qiskit qiskit-aer numpy`** (for GPU mining)
- [ ] **Build releases:** `.\build-release.ps1` (or let scripts auto-build)
- [ ] **Start quantum-geth:** `.\start-geth.ps1`
- [ ] **Get your coinbase address ready**
- [ ] **Choose mining method:** GPU (`.\run-gpu-miner.ps1`) or CPU (`.\run-cpu-miner.ps1`)
- [ ] **Start mining with your coinbase address!**

**🎉 Professional quantum blockchain platform with enterprise-grade build system!**
**Happy Quantum Mining! ⚛️💎** 