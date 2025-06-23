# Quantum-Geth Standalone Miner

A standalone quantum miner for the Quantum-Geth v0.9 blockchain.

## Quick Start

### Windows
```powershell
# Build the miner
.\build-windows.ps1

# Start solo mining  
.\run-solo-miner.ps1 -Coinbase 0xYourAddressHere
```

### Linux
```bash
# Build the miner
chmod +x build-linux.sh
./build-linux.sh

# Start solo mining
chmod +x run-solo-miner.sh  
./run-solo-miner.sh -c 0xYourAddressHere
```

## Features

 **16-qubit quantum circuit mining** - Compatible with Quantum-Geth consensus  
 **Bitcoin-style difficulty** with simplified retargeting every 100 blocks  
 **Pool and Solo mining** support (pool mining coming soon)  
 **Cross-platform** - Windows and Linux support  
 **Multi-threaded** mining with configurable intensity  

## Requirements

- **Go 1.19+** for building
- **Quantum-Geth node** running with RPC enabled
- **4+ CPU cores** recommended for optimal performance

## Command Line Options

```bash
quantum-miner -coinbase 0xAddress -node http://localhost:8545 -threads 8
quantum-miner -version
```

## Current Status

 **Completed:**
- Cross-platform build system (Windows/Linux)
- Command-line interface and configuration
- Solo mining framework structure
- Build and run scripts

 **In Progress:**
- Full quantum mining implementation  
- RPC communication with geth node
- Quantum circuit computation backend

 **Coming Soon:**
- Pool mining with Stratum protocol
- Web-based mining dashboard
- GPU acceleration support

---

This is a foundational implementation. The quantum mining logic will be integrated from the main Quantum-Geth consensus engine.
