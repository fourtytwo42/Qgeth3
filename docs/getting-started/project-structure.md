# Project Structure

Understanding the Q Coin codebase organization and script layout.

## 📁 Directory Structure

```
Qgeth3/
├── 📁 scripts/
│   ├── 📁 linux/          # Linux shell scripts
│   │   ├── build-linux.sh     # Build system with GPU detection
│   │   ├── start-geth.sh      # Node launcher
│   │   ├── start-miner.sh     # Mining launcher
│   │   └── prepare-vps.sh     # VPS preparation
│   ├── 📁 windows/        # Windows PowerShell scripts  
│   │   ├── build-release.ps1  # Windows build system
│   │   ├── start-geth.ps1     # Node launcher
│   │   ├── start-miner.ps1    # Mining launcher
│   │   └── start-geth-sync.ps1 # Sync-only mode
│   └── 📁 deployment/     # Deployment & setup scripts
│       ├── bootstrap-qgeth.sh # One-command VPS setup
│       └── auto-geth-service.sh # Production service setup
├── 📁 configs/            # Configuration files
│   ├── genesis_quantum_mainnet.json  # Mainnet genesis
│   ├── genesis_quantum_testnet.json  # Testnet genesis
│   └── genesis_quantum_dev.json      # Development genesis
├── 📁 docs/               # Documentation
│   ├── quick-start.md         # Getting started guide
│   ├── installation.md       # Installation instructions
│   ├── vps-deployment.md     # VPS deployment guide
│   ├── linux-mining.md       # Linux mining guide (GPU/CPU)
│   ├── windows-mining.md     # Windows mining guide (GPU/CPU)
│   ├── troubleshooting.md    # Troubleshooting guide
│   ├── project-structure.md  # This file
│   └── contributing.md       # Contributing guidelines
├── 📁 quantum-geth/       # Quantum-enhanced Ethereum client
│   ├── cmd/geth/              # Main geth command
│   ├── core/                  # Blockchain core logic
│   ├── consensus/qmpow/       # QMPoW consensus engine
│   ├── eth/                   # Ethereum protocol
│   └── ... (full Ethereum codebase with quantum enhancements)
├── 📁 quantum-miner/      # Unified quantum miner
│   ├── cmd/quantum-miner/     # Main miner command
│   ├── internal/              # Miner implementation
│   └── pkg/                   # Reusable packages
├── 📁 tests/              # Test frameworks
│   └── hardness/              # Quantum hardness tests
├── quick-start.sh         # Convenience wrapper script
├── README.md              # Main documentation (now simplified)
└── .gitignore             # Git ignore rules
```

## 🛠️ Script Organization

### Platform-Specific Scripts

#### Linux Scripts (`scripts/linux/`)
- **`build-linux.sh`** - Comprehensive build system
  - Auto-detects GPU capabilities (CUDA, Qiskit-Aer)
  - Memory optimization for VPS
  - Builds geth, miner, or both
  - Clean build options

- **`start-geth.sh`** - Node launcher
  - Network selection (mainnet, testnet, devnet)
  - Configuration management
  - Auto-build if needed
  - Mining integration options

- **`start-miner.sh`** - Mining launcher
  - Smart GPU/CPU detection
  - Remote node connection
  - Performance monitoring
  - Network-specific configuration

- **`prepare-vps.sh`** - VPS preparation
  - Memory checks and swap creation
  - Dependency installation
  - System optimization
  - Security hardening

#### Windows Scripts (`scripts/windows/`)
- **`build-release.ps1`** - Windows build system
  - Visual Studio detection
  - GPU capability detection
  - Release package creation
  - Cross-platform compatibility

- **`start-geth.ps1`** - Windows node launcher
  - PowerShell-native implementation
  - Windows service integration
  - Configuration management

- **`start-miner.ps1`** - Windows mining launcher
  - GPU acceleration support
  - CuPy integration
  - Performance optimization

#### Deployment Scripts (`scripts/deployment/`)
- **`bootstrap-qgeth.sh`** - Ultimate one-command setup
  - Downloads entire project
  - Handles all dependencies
  - VPS preparation
  - Auto-service installation
  - Production-ready deployment

- **`auto-geth-service.sh`** - Production service setup
  - Systemd service creation
  - Service management
  - Crash recovery
  - Log management
  - Manual update support

### Root-Level Scripts

#### `quick-start.sh` - Convenience Wrapper
```bash
# Easy access to common operations
./quick-start.sh build          # Build Q Geth
./quick-start.sh start          # Start testnet node
./quick-start.sh start-mining   # Start with mining
./quick-start.sh bootstrap      # VPS setup
./quick-start.sh status         # Check status
./quick-start.sh help           # Show help
```

**Supported Commands:**
- `build` - Build geth and miner
- `build-clean` - Clean build
- `start` - Start testnet node
- `start-mainnet` - Start mainnet node
- `start-mining` - Start with mining
- `bootstrap` - Run VPS bootstrap
- `vps-setup` - VPS preparation
- `status` - Check system status
- `help` - Show usage information

## 🏗️ Core Components

### Quantum-Geth (`quantum-geth/`)
Enhanced Ethereum client with quantum consensus:

```
quantum-geth/
├── cmd/geth/              # Main geth command
├── consensus/qmpow/       # QMPoW consensus engine
│   ├── consensus.go           # Main consensus logic
│   ├── difficulty.go          # ASERT-Q difficulty adjustment
│   ├── quantum_validation.go  # Quantum proof validation
│   └── rewards.go             # Block reward calculation
├── core/
│   ├── blockchain.go          # Blockchain management
│   ├── genesis.go             # Genesis block handling
│   └── types/                 # Quantum block types
├── eth/                   # Ethereum protocol
├── internal/ethapi/       # RPC API
└── params/                # Network parameters
```

### Quantum-Miner (`quantum-miner/`)
High-performance quantum miner:

```
quantum-miner/
├── cmd/quantum-miner/     # Main miner command
├── internal/
│   ├── miner/                 # Core mining logic
│   ├── quantum/               # Quantum circuit handling
│   ├── gpu/                   # GPU acceleration
│   └── network/               # Network communication
└── pkg/
    ├── qiskit/                # Qiskit integration
    ├── cuda/                  # CUDA acceleration
    └── utils/                 # Utility functions
```

## 📋 Configuration Files

### Genesis Configurations (`configs/`)

#### `genesis_quantum_mainnet.json`
- **Chain ID:** 73236
- **Network:** Q Coin Mainnet
- **Consensus:** QMPoW
- **Initial Difficulty:** Production settings
- **Allocations:** Mainnet token distribution

#### `genesis_quantum_testnet.json`
- **Chain ID:** 73235
- **Network:** Q Coin Testnet
- **Consensus:** QMPoW
- **Initial Difficulty:** Testing-friendly
- **Allocations:** Testnet faucet addresses

#### `genesis_quantum_dev.json`
- **Chain ID:** 73234
- **Network:** Development
- **Consensus:** QMPoW
- **Initial Difficulty:** Very low for testing
- **Allocations:** Development accounts

### Script Configuration
Configuration is handled via command-line arguments and environment variables rather than separate config files. See the individual scripts for available options.

## 🔧 Build System Architecture

### Multi-Target Build System
The build system supports multiple targets:

1. **`geth`** - Blockchain node only
   - Optimized for VPS deployment
   - Minimal resource usage
   - Used by auto-service

2. **`miner`** - Mining software only
   - GPU acceleration support
   - Platform-specific optimizations
   - Standalone operation

3. **`both`** - Complete package
   - Development and testing
   - Full feature set
   - Local deployment

### Platform Detection
```bash
# Linux: GPU Detection Flow
1. Check for NVIDIA GPU (nvidia-smi)
2. Check for CUDA toolkit (nvcc)
3. Check for Python GPU libraries (qiskit-aer)
4. Build with best available acceleration
5. Tag binaries with acceleration type

# Windows: GPU Detection Flow
1. Check for Visual Studio Build Tools
2. Check for CUDA toolkit
3. Check for Python GPU libraries (cupy)
4. Build with appropriate flags
5. Create release packages
```

## 📦 Release System

### Windows Release Packages (`releases/`)
```
releases/
├── quantum-geth-YYYY-MM-DD-HHMMSS/
│   ├── geth.exe                   # Main binary
│   ├── start-geth.ps1             # Launcher
│   ├── configs/                   # Genesis files
│   └── README.txt                 # Usage instructions
└── quantum-miner-YYYY-MM-DD-HHMMSS/
    ├── quantum-miner.exe          # Miner binary
    ├── start-miner.ps1            # Launcher
    └── README.txt                 # Usage instructions
```

### Linux Direct Binaries
- `geth.bin` - Quantum-geth binary
- `geth` - Wrapper script with configuration
- `quantum-miner` - Miner binary
- Scripts automatically detect and use binaries

## 🔄 Script Interaction Flow

### Development Workflow
```
1. Clone repository
2. Run ./quick-start.sh build
3. Start node: ./quick-start.sh start
4. Start mining: ./quick-start.sh start-mining
5. Monitor and develop
```

### VPS Deployment Workflow
```
1. Run bootstrap script
   └── Downloads project
   └── Runs auto-geth-service.sh
       └── Runs prepare-vps.sh
       └── Builds geth only
       └── Creates systemd services
       └── Sets up monitoring
2. Services start automatically
3. Manual updates via documented procedures
4. Crash recovery ensures uptime
```

### Production Mining Workflow
```
1. VPS: Deploy node with auto-service
2. Local: Build miner with GPU support
3. Connect miner to VPS node
4. Monitor performance
5. Scale horizontally
```

## 🎯 Script Dependencies

### Dependency Graph
```
bootstrap-qgeth.sh
└── auto-geth-service.sh
    ├── prepare-vps.sh
    └── build-linux.sh
        └── quantum-geth/ (source)
        └── quantum-miner/ (source)

quick-start.sh
├── scripts/linux/build-linux.sh
├── scripts/linux/start-geth.sh
└── scripts/linux/start-miner.sh
```

### Runtime Dependencies
- **Go 1.21+** - Required for building
- **Python 3.8+** - Required for GPU mining
- **Git** - Required for source management
- **Build tools** - gcc, make (Linux)
- **CUDA toolkit** - Optional, for GPU acceleration
- **Visual Studio** - Required for Windows builds

## 📊 File Size & Resource Usage

### Typical File Sizes
- **quantum-geth binary**: ~50-80MB
- **quantum-miner binary**: ~20-40MB
- **Genesis files**: <1KB each
- **Scripts**: <100KB total
- **Documentation**: ~200KB total

### Resource Requirements
- **Build RAM**: 3-4GB (with swap)
- **Runtime RAM**: 512MB-2GB
- **Disk Space**: 10GB minimum, 20GB recommended
- **Network**: P2P (30303), RPC (8545), WebSocket (8546)

## 🔗 Integration Points

### Script Integration
- All scripts reference new `scripts/` directory structure
- Cross-platform compatibility maintained
- Consistent parameter passing
- Unified error handling

### Service Integration
- Systemd services for Linux VPS
- Windows Service wrapper support
- Docker container compatibility
- Kubernetes deployment ready

### API Integration
- JSON-RPC API on port 8545
- WebSocket API on port 8546
- Mining API for external miners
- Monitoring API for service health

## 🎛️ Configuration Management

### Network Selection
Scripts automatically select appropriate configuration:
```bash
# Network-specific data directories
~/.qcoin/mainnet/    # Mainnet data
~/.qcoin/testnet/    # Testnet data
~/.qcoin/devnet/     # Development data

# Corresponding genesis files
configs/genesis_quantum_mainnet.json
configs/genesis_quantum_testnet.json
configs/genesis_quantum_dev.json
```

### Environment Variables
```bash
# Build system
QGETH_BUILD_TEMP="/custom/temp/dir"
GOMEMLIMIT="2GiB"

# Runtime
GETH_NETWORK="testnet"
GETH_ARGS="--http.corsdomain \"*\""

# Mining
CUDA_VISIBLE_DEVICES="0"
QISKIT_IN_PARALLEL="TRUE"
```

## 🔄 Update & Maintenance

### Manual Update System
**Important**: Q Geth does not auto-update. All updates must be performed manually for security and stability.

- **Manual monitoring**: Check GitHub for new releases
- **Backup system**: Keep backups before updates
- **Rollback capability**: Manual rollback on build failure
- **Service continuity**: Follow documented update procedures

### Manual Maintenance
```bash
# Update to latest version
git pull origin main

# Clean rebuild
./quick-start.sh build-clean

# Update VPS installation
cd /opt/qgeth/Qgeth3
sudo git pull origin main
sudo ./scripts/linux/build-linux.sh geth
qgeth-service restart
```

This organized structure provides clear separation of concerns while maintaining ease of use and professional deployment capabilities. 