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

3. Mine to this node from another terminal:
   `
   .\quantum-miner.exe -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
   `

## Network Details
- **Network ID**: 1337
- **HTTP RPC**: http://localhost:8545
- **WebSocket**: ws://localhost:8546
- **Default Coinbase**: 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

## Mining
Use the quantum-miner package for external mining, or use start-geth-mining scripts for built-in mining.
