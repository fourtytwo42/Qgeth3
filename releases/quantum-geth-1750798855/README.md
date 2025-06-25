# Quantum-Geth Release Package

## Quick Start

### Windows (Batch Files)
- **start-geth.bat** - Start node (no mining)
- **start-geth-mining.bat [threads]** - Start mining node

### Windows/Linux (PowerShell)
- **start-geth.ps1** - Start node (no mining)  
- **start-geth-mining.ps1 -Threads <n>** - Start mining node

## First Time Setup

1. Initialize blockchain:
   `
   geth.exe --datadir qdata init genesis_quantum.json
   `

2. Start node:
   `
   .\start-geth.ps1
   `

3. Mine to this node using the optimized quantum-miner:
   `
   # Download quantum-miner release package
   # Then run:
   .\quantum-miner.exe -gpu -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -threads 64
   `

## Network Details
- **Network ID**: 1337
- **HTTP RPC**: http://localhost:8545
- **WebSocket**: ws://localhost:8546
- **Default Coinbase**: 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

## Mining
Use the **optimized quantum-miner package** for high-performance GPU mining (10-100x faster), or use start-geth-mining scripts for basic built-in mining.

Quantum-Geth v0.9 BareBones+Halving Features:
  * 32 chained quantum puzzles (16 qubits x 20 T-gates)
  * Bitcoin-style halving (50 QGC -> 25 QGC -> 12.5 QGC...)
  * 600,000 block epochs (approximately 6 months)
  * Branch-serial quantum circuit execution
  * Mahadev->CAPSS->Nova proof stack
  * Dilithium-2 self-attestation
  * ASERT-Q difficulty adjustment (12s target)
  * Single RLP quantum blob (197 bytes)
