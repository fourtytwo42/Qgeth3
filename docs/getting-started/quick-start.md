# Quick Start Guide

Get up and running with Q Coin in minutes!

## 🚀 Windows Quick Start

### One-Line Setup
```powershell
# Start the Q Coin testnet node (works like standard geth)
.\scripts\windows\start-geth.ps1

# Start with mining enabled
.\scripts\windows\start-geth.ps1 -mine -etherbase 0xYourAddress

# Show all options
.\scripts\windows\start-geth.ps1 -help
```

### Using the Quick Start Script
```bash
# Easy wrapper commands (from root directory)
./quick-start.sh build                # Build Q Geth
./quick-start.sh start                # Start testnet node
./quick-start.sh start-mining         # Start with mining  
./quick-start.sh bootstrap            # Complete VPS setup
```

## 🐧 Linux Quick Start

### Option 1: Ultimate One-Command Setup (Easiest!)
```bash
# Interactive mode (asks for confirmations)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash

# Non-interactive mode (auto-confirms all prompts - perfect for automation)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y

# This single command:
# ✅ Installs all dependencies (git, curl, golang, build tools)
# ✅ Downloads the Q Geth repository automatically
# ✅ Prepares VPS (memory checks, swap creation, firewall)
# ✅ Builds and configures Q Geth with auto-updating service
# ✅ Sets up crash recovery and GitHub monitoring
# ✅ Safe to run multiple times - detects existing installations gracefully
```

### Option 2: Manual Setup
```bash
# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Quick build and start
./quick-start.sh build
./quick-start.sh start

# Or use direct scripts
./scripts/linux/build-linux.sh geth    # Build node only
./scripts/linux/start-geth.sh testnet  # Start testnet node
```

### Option 3: Platform-Specific Scripts
```bash
# Direct script access
./scripts/linux/start-geth.sh         # Linux node
./scripts/windows/start-geth.ps1      # Windows node  
./scripts/deployment/bootstrap-qgeth.sh # VPS deployment
```

## 🌐 Q Coin Testnet Details

- **Name:** Q Coin
- **Symbol:** Q  
- **Chain ID:** 73235
- **Block Time:** 12 seconds
- **Consensus:** QMPoW (Quantum Proof of Work)
- **Default Data Directory:** 
  - Windows: `%APPDATA%\Qcoin`
  - Linux: `~/.qcoin/[network]` (e.g., `~/.qcoin/testnet`)

## 🎯 Build Target Guide

| Build Target | What Gets Built | Best For | Auto-Service Uses |
|--------------|-----------------|----------|-------------------|
| **`geth`** | Blockchain node only | VPS nodes, validators | ✅ **Yes** |
| **`miner`** | Mining software only | Dedicated mining rigs | ❌ No |
| **`both`** | Node + miner | Development, testing | ❌ No |

**💡 VPS Recommendation:** Use auto-service (builds only `geth`) for clean, efficient node deployment. Run miners separately on dedicated hardware for optimal performance.

## 🎮 Basic Usage Examples

### Start a Node
```bash
# Linux
./scripts/linux/start-geth.sh testnet   # Start testnet node
./scripts/linux/start-geth.sh mainnet   # Start mainnet node
./scripts/linux/start-geth.sh devnet    # Start dev network

# Windows (PowerShell)
.\scripts\windows\start-geth.ps1        # Start testnet node
```

### Start Mining
```bash
# Linux - Smart miner (auto-detects GPU/CPU)
./scripts/linux/start-miner.sh --testnet --verbose

# Start with built-in mining
./scripts/linux/start-geth.sh testnet --mining

# Connect to remote node
./scripts/linux/start-miner.sh --node http://192.168.1.100:8545 --verbose
```

### Attach to Console
```bash
# Linux (using IPC) - Replace with actual path from your network
./geth attach ipc:$HOME/.qcoin/testnet/geth.ipc    # Testnet
./geth attach ipc:$HOME/.qcoin/mainnet/geth.ipc    # Mainnet  
./geth attach ipc:$HOME/.qcoin/devnet/geth.ipc     # Devnet

# Or attach via HTTP RPC
./geth attach http://localhost:8545
```

## 🏆 Next Steps

Once you have Q Geth running:

1. **[Installation Guide](installation.md)** - Detailed setup for your platform
2. **[VPS Deployment](vps-deployment.md)** - Production deployment guide
3. **[Linux Mining Guide](linux-mining.md)** - Optimize your Linux mining setup
4. **[Windows Mining Guide](windows-mining.md)** - Optimize your Windows mining setup
5. **[Troubleshooting](troubleshooting.md)** - Fix common issues

## 🎯 Getting Started Checklist

### Windows Users
- [ ] **Build releases:** `.\scripts\windows\build-release.ps1` (or let scripts auto-build)
- [ ] **Start quantum-geth:** `.\scripts\windows\start-geth.ps1`
- [ ] **Start mining:** `.\scripts\windows\start-miner.ps1 -Coinbase 0xYourAddress`

### Linux Users
- [ ] **Install dependencies:** `sudo apt install golang-go python3 python3-pip`
- [ ] **Clone repository:** `git clone https://github.com/fourtytwo42/Qgeth3.git`
- [ ] **Build everything:** `./scripts/linux/build-linux.sh both`
- [ ] **Start quantum-geth:** `./scripts/linux/start-geth.sh testnet`
- [ ] **Start mining:** `./scripts/linux/start-miner.sh --testnet --verbose`

### VPS Users
- [ ] **Choose VPS:** 2+ vCPU, 4GB+ RAM recommended
- [ ] **Run setup script:** `curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y`
- [ ] **Configure firewall:** Allow ports 8545, 30303
- [ ] **Monitor performance:** `htop` and mining logs

**🎉 Professional quantum blockchain platform with cross-platform support!**
**Happy Quantum Mining! ⚛️💎** 