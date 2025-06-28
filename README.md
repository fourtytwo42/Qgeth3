# Q Coin - Quantum Blockchain Platform

A complete quantum blockchain platform featuring **Q Coin** with **Quantum-Geth** (quantum-enhanced Ethereum client) and **high-performance quantum miners** with CPU and GPU acceleration support.

🎉 **NEW: Reorganized Documentation & One-Command VPS Setup!**

## 🚀 Quick Start

### Windows
```powershell
# Start Q Coin testnet node
.\scripts\windows\start-geth.ps1

# Start with mining
.\scripts\windows\start-geth.ps1 -mine -etherbase 0xYourAddress
```

### Linux - One Command Setup
```bash
# Complete VPS setup (downloads everything automatically)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y

# Or manual setup
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
./quick-start.sh build
./quick-start.sh start
```

### Using the Quick Start Wrapper
```bash
./quick-start.sh build          # Build Q Geth
./quick-start.sh start          # Start testnet node  
./quick-start.sh start-mining   # Start with mining
./quick-start.sh bootstrap      # VPS deployment
./quick-start.sh help           # Show all options
```

## 🌐 Q Coin Network Details

- **Name:** Q Coin
- **Symbol:** Q
- **Chain ID:** 73235 (Testnet), 73236 (Mainnet)
- **Block Time:** 12 seconds
- **Consensus:** QMPoW (Quantum Proof of Work)
- **Features:** Real quantum circuits, Bitcoin-style mining, GPU acceleration

## ⚛️ Quantum Mining Features

- **16-qubit quantum circuits** per puzzle
- **128 quantum puzzles per block**
- **Native CUDA + Qiskit-Aer GPU** support (Linux)
- **CuPy GPU acceleration** (Windows)
- **CPU fallback** with optimization
- **3-10x performance boost** with GPU mining

## 📊 Performance Overview

| Platform | Method | Performance | Best For |
|----------|--------|-------------|----------|
| **Linux** | **Native CUDA** | **~3.2 puzzles/sec** | Maximum performance |
| **Linux** | **Qiskit-Aer GPU** | **~2.4 puzzles/sec** | Easier setup |
| **Windows** | **CuPy GPU** | **~2.0 puzzles/sec** | Windows acceleration |
| **Both** | **CPU** | ~0.3-0.8 puzzles/sec | Universal compatibility |

## 📁 Project Structure

```
Qgeth3/
├── scripts/
│   ├── linux/          # Linux shell scripts  
│   ├── windows/        # Windows PowerShell scripts
│   └── deployment/     # VPS deployment & auto-service
├── configs/            # Network configurations
├── docs/               # 📚 Comprehensive documentation
├── quantum-geth/       # Quantum-enhanced Ethereum client
├── quantum-miner/      # High-performance quantum miner
└── quick-start.sh      # Convenience wrapper
```

## 📚 Documentation

### 🚀 Getting Started
- **[Quick Start Guide](docs/getting-started/quick-start.md)** - Get running in minutes
- **[Project Structure](docs/getting-started/project-structure.md)** - Codebase organization
- **[Quick Start Troubleshooting](docs/getting-started/troubleshooting-quick-start.md)** - Getting started problems

### 🔨 Building from Source
- **[Linux Q Geth Build](docs/building/linux-build-geth.md)** - Build Q Geth on Linux
- **[Windows Q Geth Build](docs/building/windows-build-geth.md)** - Build Q Geth on Windows  
- **[Linux Quantum Miner Build](docs/building/linux-build-quantum-miner.md)** - Build miner on Linux
- **[Windows Quantum Miner Build](docs/building/windows-build-quantum-miner.md)** - Build miner on Windows
- **[Linux Build Troubleshooting](docs/building/troubleshooting-linux-build-geth.md)** - Linux build issues
- **[Windows Build Troubleshooting](docs/building/troubleshooting-windows-build-geth.md)** - Windows build issues
- **[Linux Miner Build Troubleshooting](docs/building/troubleshooting-linux-build-quantum-miner.md)** - Linux miner issues
- **[Windows Miner Build Troubleshooting](docs/building/troubleshooting-windows-build-quantum-miner.md)** - Windows miner issues

### 🌐 Node Operation  
- **[Linux Geth Guide](docs/node-operation/linux-geth.md)** - Complete Linux node setup & operation
- **[Windows Geth Guide](docs/node-operation/windows-geth.md)** - Complete Windows node setup & operation
- **[Linux Geth Troubleshooting](docs/node-operation/troubleshooting-linux-geth.md)** - Linux node troubleshooting
- **[Windows Geth Troubleshooting](docs/node-operation/troubleshooting-windows-geth.md)** - Windows node troubleshooting

### ⚡ Mining Guides
- **[Linux Mining Guide](docs/mining/linux-mining.md)** - CUDA & GPU optimization for Linux
- **[Windows Mining Guide](docs/mining/windows-mining.md)** - CuPy & GPU optimization for Windows  
- **[Linux Mining Troubleshooting](docs/mining/troubleshooting-linux-mining.md)** - Linux mining troubleshooting
- **[Windows Mining Troubleshooting](docs/mining/troubleshooting-windows-mining.md)** - Windows mining troubleshooting

### 🚀 Production Deployment
- **[VPS Deployment Guide](docs/deployment/vps-deployment.md)** - Production server setup & management
- **[Windows Deployment Guide](docs/deployment/windows-deploy.md)** - Windows production deployment
- **[VPS Deployment Troubleshooting](docs/deployment/troubleshooting-vps-deployment.md)** - Production deployment troubleshooting
- **[Windows Deployment Troubleshooting](docs/deployment/troubleshooting-windows-deploy.md)** - Windows deployment troubleshooting

### 👨‍💻 Development
- **[Contributing Guidelines](docs/development/contributing.md)** - Development workflow

## 🎯 Use Cases

### For Developers
- Quantum-resistant blockchain development
- Smart contract deployment on quantum-secure network
- Research and experimentation with quantum consensus

### For Miners
- GPU-accelerated quantum mining
- Professional mining operations
- Educational quantum computing exploration

### For VPS Operators
- One-command production deployment
- Automated updates and monitoring
- Professional node operation

## 🚀 Production Features

### Auto-Updating VPS Service
- **GitHub Monitoring:** Auto-updates on new commits
- **Crash Recovery:** Automatic restart with 5-minute retry
- **Memory Optimization:** Handles low-memory VPS gracefully
- **Security:** UFW firewall configuration
- **Monitoring:** Comprehensive logging and status tracking

### Professional Build System
- **Multi-target builds:** Node-only, miner-only, or both
- **GPU Auto-detection:** CUDA, Qiskit-Aer, CuPy support
- **Cross-platform:** Linux, Windows, VPS optimization
- **Memory-efficient:** Temp directory management for VPS

### Enterprise-Ready
- **Systemd Integration:** Production service management
- **Backup System:** Automatic version backup and rollback
- **Lock File Protection:** Prevents installation conflicts
- **Log Management:** Structured logging and rotation

## 🛠️ Development Status

### Recent Improvements
- ✅ **Complete script reorganization** with platform-specific directories
- ✅ **Go temp directory build fixes** for VPS environments  
- ✅ **Comprehensive documentation breakdown** for better navigation
- ✅ **Enhanced .gitignore** to prevent build artifact commits
- ✅ **One-command VPS bootstrap** with non-interactive mode
- ✅ **Professional auto-service** with GitHub monitoring
- ✅ **Directory context and permission fixes** for seamless VPS deployment

### Core Features
- ✅ **Quantum Consensus (QMPoW)** - Production ready
- ✅ **GPU Mining** - CUDA, Qiskit-Aer, CuPy support
- ✅ **Cross-platform builds** - Linux, Windows
- ✅ **VPS Deployment** - One-command setup
- ✅ **Auto-updates** - GitHub monitoring & crash recovery

## 📈 Network Statistics

### Testnet (Chain ID: 73235)
- **Status:** Active
- **Purpose:** Testing and development
- **Mining:** Open to all participants
- **Faucet:** Available for test tokens

### Mainnet (Chain ID: 73236)  
- **Status:** Ready for deployment
- **Purpose:** Production blockchain
- **Mining:** Professional mining operations
- **Security:** Full quantum consensus validation

## 🔧 Technical Specifications

### Quantum Proof-of-Work (QMPoW)
- **Quantum Circuits:** 16 qubits, 20 T-gates per puzzle
- **Block Validation:** 128 quantum puzzles per block
- **Difficulty Adjustment:** ASERT-Q algorithm
- **Security:** Quantum-resistant validation

### Network Protocol
- **P2P Port:** 30303 (TCP/UDP)
- **HTTP RPC:** 8545
- **WebSocket:** 8546
- **Mining API:** Integrated
- **Monitoring:** Health check endpoints

## 🏆 Getting Started Checklist

### Quick Setup
- [ ] Clone repository or run bootstrap script
- [ ] Build Q Geth: `./quick-start.sh build`
- [ ] Start node: `./quick-start.sh start`
- [ ] Start mining: `./quick-start.sh start-mining`

### Production Deployment
- [ ] VPS with 2+ vCPU, 4GB+ RAM
- [ ] Run: `curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y`
- [ ] Verify service: `qgeth-service status`
- [ ] Monitor: `qgeth-service logs geth`

### Development Setup
- [ ] Read [Contributing Guidelines](docs/contributing.md)
- [ ] Set up development environment
- [ ] Review [Project Structure](docs/project-structure.md)
- [ ] Run tests and contribute!

## 📞 Support & Community

- **Documentation:** See `docs/` directory for comprehensive guides
- **Issues:** GitHub Issues for bug reports and feature requests
- **Questions:** GitHub Discussions for general questions
- **Emergency:** Use our topic-specific troubleshooting guides in the documentation section above

## 📄 License

This project is licensed under the MIT License - see individual component licenses for details.

---

**🎉 Professional quantum blockchain platform with enterprise-grade deployment!**

**Ready to explore quantum-resistant cryptocurrency? Start with our [Quick Start Guide](docs/quick-start.md)!**

**Happy Quantum Mining! ⚛️💎** 