# Q Coin Planck Network

## Overview
The **Planck Network** is the **default** Q Coin testnet with the same configuration as the existing testnet but with a unique chain ID and genesis hash for isolated testing.

## Network Details
- **Network Name**: Q Coin Planck Network
- **Chain ID**: 73237
- **Network ID**: 73237
- **Genesis File**: `configs/genesis_quantum_planck.json`
- **Default Port**: 30307
- **Data Directory**: 
  - **Linux**: `$HOME/.qcoin/planck`
  - **Windows**: `%APPDATA%\Qcoin\planck`

## Genesis Configuration
- **Difficulty**: 200 (0xC8) - Same as testnet
- **Gas Limit**: 3141592 (0x2fefd8)
- **Timestamp**: 0x67826000 (2025-06-30)
- **QMPoW Settings**:
  - Qubits: 16
  - T-Count: 20
  - L-Net: 128
  - Epoch Length: 600,000 blocks

## Pre-allocated Accounts
The planck network includes the same pre-allocated accounts as testnet:
- `0x8b61271473f14c80f2B1381Db9CB13b2d5306200`: 1000 QCOIN
- `0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A`: 1000 QCOIN
- `0x1234567890123456789012345678901234567890`: 1000 QCOIN

## Starting the Planck Network

### Linux
```bash
# Start planck network node (default - no network parameter needed)
./scripts/linux/start-geth.sh

# Start planck network with mining
./scripts/linux/start-geth.sh --mining

# Or explicitly specify planck
./scripts/linux/start-geth.sh planck

# Start miner for planck network (after node is running)
./scripts/linux/start-miner.sh --gpu --threads 32
```

### Windows
```powershell
# Start planck network node (default - no network parameter needed)
.\start-geth.ps1

# Start planck network with mining
.\start-geth.ps1 -mining

# Or explicitly specify planck
.\start-geth.ps1 planck
```

```batch
# Batch file method (planck is default)
start-geth.bat

# Or explicitly specify planck
start-geth.bat planck
```

## Mining on Planck Network
The planck network supports the same mining capabilities as other Q Coin networks:

```bash
# GPU mining (Linux)
./quantum-miner -gpu -coinbase 0xYourAddress -node http://localhost:8545

# CPU mining (Windows/Linux)  
./quantum-miner -cpu -coinbase 0xYourAddress -node http://localhost:8545 -threads 8
```

## Network Comparison

| Network | Chain ID | Default Port | Purpose |
|---------|----------|--------------|---------|
| **Planck** | **73237** | **30307** | **Default testnet (isolated)** |
| Testnet | 73235 | 30303 | Legacy testing network |
| Devnet | 73234 | 30305 | Development/debugging |

## Use Cases for Planck Network
- **Isolated Testing**: Test new features without affecting main testnet
- **Custom Experiments**: Run experiments with same genesis parameters but different network
- **Load Testing**: Test network performance with isolated environment
- **Development**: Develop and test applications without testnet interference
- **Research**: Quantum algorithm research and validation

## API Access
- **HTTP RPC**: http://localhost:8545
- **WebSocket**: ws://localhost:8546
- **APIs**: eth, net, web3, personal, admin, txpool, miner, qmpow, debug

## Auto-Reset Functionality
Like other Q Coin networks, planck supports auto-reset:
- Automatically detects genesis changes
- Wipes blockchain data when genesis differs
- Starts fresh from block 1 with new parameters

## Performance Expectations
Same as testnet/devnet:
- **CPU Mining**: ~400-800 PZ/s (8 threads)
- **GPU Mining**: ~8000-15000+ PZ/s (32 threads)
- **WSL2 GPU**: ~12000-20000+ PZ/s (Windows)

## Network Status
✅ **DEFAULT**: Planck network is now the default testnet - fully implemented and ready for use!

When you run geth without specifying a network, it automatically uses Planck.

## Benefits
- **Same Genesis Parameters**: Identical to testnet for consistency
- **Unique Chain ID**: Completely isolated from other networks
- **Cross-Platform**: Works on Linux and Windows
- **Full Feature Support**: Mining, auto-reset, all APIs
- **Easy Setup**: Same scripts and tools as existing networks

Perfect for testing scenarios where you need a clean, isolated environment with the same blockchain parameters as the main testnet! 