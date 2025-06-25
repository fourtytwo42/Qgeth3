# Q Coin Scripts Directory

This directory contains Linux shell scripts for building, running, and managing the Q Coin quantum blockchain system.

## 🔧 Recent Fixes (2025-06-25)

**All Linux scripts have been fixed** to properly handle:
- ✅ Correct path navigation to `quantum-geth` and `quantum-miner` directories  
- ✅ Proper genesis file references (`../genesis_quantum_testnet.json`, `../genesis_quantum_mainnet.json`)
- ✅ Release directory paths (`../releases/quantum-*`)
- ✅ Go module building from correct subdirectories
- ✅ Automatic fallback to building releases if executables not found

## 📁 Directory Structure

**Run these scripts FROM the `scripts/` directory:**
```bash
cd scripts/
./start-geth.sh --help
```

## 🚀 Quick Start

### 1. Build Everything
```bash
cd scripts/
./build-release.sh both          # Build timestamped releases (old approach)
./build-linux.sh                # 🆕 Build to root directory (new approach)
```

### 2. Start Q Coin Node 
```bash
./start-geth.sh                  # Testnet node (no mining)
./start-geth.sh --mainnet        # Mainnet node
./start-geth.sh --mine --etherbase 0x123...  # With mining
```

### 3. Start Mining
```bash
./start-cpu-miner.sh --address 0x123...     # CPU mining
./start-gpu-miner.sh --address 0x123...     # GPU mining (if available)
```

### 4. Development/Testing
```bash
./dev-reset-blockchain.sh --difficulty 1 --force   # Reset blockchain
./dev-start-geth.sh                                # Start dev node
./dev-run-cpu-miner.sh                             # Start dev miner
```

## 📜 Available Scripts

### Build Scripts
- `build-release.sh` - Build distributable releases (geth/miner/both)
- `build-linux.sh` - 🆕 Build binaries to root directory (calls `../build-linux.sh`)

### Node Scripts  
- `start-geth.sh` - Start Q Coin node (testnet/mainnet)
- `dev-start-geth.sh` - Start development node

### Mining Scripts
- `start-cpu-miner.sh` - Start CPU miner
- `start-gpu-miner.sh` - Start GPU miner  
- `start-mining.sh` - Start integrated mining
- `dev-run-cpu-miner.sh` - Development CPU mining
- `dev-run-gpu-miner.sh` - Development GPU mining

### Utility Scripts
- `dev-reset-blockchain.sh` - Reset blockchain with custom difficulty
- `basic-test.sh` - Basic functionality test
- `run-hardness-tests.sh` - Quantum hardness testing

## 🔧 Path Requirements

**Important**: These scripts assume the following directory structure:
```
Qgeth3/
├── quantum-geth/          # Quantum-Geth source code
├── quantum-miner/         # Quantum-Miner source code  
├── genesis_quantum_testnet.json
├── genesis_quantum_mainnet.json
├── build-linux.sh        # 🆕 Root Linux build script
├── geth                   # 🆕 Linux binary (from build-linux.sh)
├── quantum-miner          # 🆕 Linux binary (from build-linux.sh)
├── quantum_solver.py      # 🆕 Linux helper script
├── releases/              # Timestamped build outputs
└── scripts/               # ← Run scripts from here
    ├── start-geth.sh
    ├── build-release.sh
    ├── build-linux.sh     # 🆕 Calls ../build-linux.sh
    └── ...
```

## 🐛 Troubleshooting

### Build Issues
```bash
# If build fails, try:
go mod tidy                    # In quantum-geth/ or quantum-miner/
./build-release.sh --clean     # Clean rebuild
```

### Path Issues  
```bash
# Make sure you're in the scripts directory:
pwd                           # Should show: .../Qgeth3/scripts
ls ..                         # Should show: quantum-geth/ quantum-miner/
```

### Permission Issues
```bash
chmod +x *.sh                # Make scripts executable
```

## 📊 Network Details

### Testnet (Default)
- **Chain ID**: 73235
- **Currency**: Q (Q Coin)
- **Genesis**: `../genesis_quantum_testnet.json`
- **Ports**: P2P 4294, RPC 8545, WS 8546
- **Difficulty**: Starts at 200 (0xC8)

### Mainnet
- **Chain ID**: 73236  
- **Currency**: Q (Q Coin)
- **Genesis**: `../genesis_quantum_mainnet.json`
- **Ports**: P2P 4295, RPC 8545, WS 8546
- **Difficulty**: Starts at 200 (0xC8)

### Consensus: QMPoW (Quantum Proof of Work)
- **Algorithm**: 128 chained quantum puzzles per block
- **Qubits**: 16 per puzzle
- **T-gates**: 20 per puzzle  
- **Target Block Time**: 12 seconds
- **Difficulty Adjustment**: ASERT-Q (per-block exponential)

## 🎯 Usage Examples

### Basic Node Operation
```bash
# Start testnet node (external mining ready)
./start-geth.sh

# Start mainnet node with mining
./start-geth.sh --mainnet --mine --etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
```

### Mining Operations
```bash
# CPU mining to testnet
./start-cpu-miner.sh --address 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

# GPU mining to mainnet  
./start-gpu-miner.sh --address 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --mainnet
```

### Development Workflow
```bash
# 1. Reset blockchain for testing
./dev-reset-blockchain.sh --difficulty 1 --force

# 2. Start development node
./dev-start-geth.sh

# 3. Start development mining (in another terminal)
./dev-run-cpu-miner.sh
```

All scripts now properly handle paths and should work seamlessly when run from the scripts directory! 🎉 