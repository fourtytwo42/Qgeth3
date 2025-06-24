# Quantum-Miner Release Package

## Quick Start

### Windows (Batch Files)
- **start-miner-cpu.bat <coinbase>** - CPU mining
- **start-miner-gpu.bat <coinbase>** - GPU mining

### Windows/Linux (PowerShell)  
- **start-miner-cpu.ps1 -Coinbase <address>** - CPU mining
- **start-miner-gpu.ps1 -Coinbase <address>** - GPU mining

## Examples

### CPU Mining
`
start-miner-cpu.bat 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
.\start-miner-cpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 4
`

### GPU Mining (Requires Python + Qiskit)
`
start-miner-gpu.bat 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
.\start-miner-gpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 2
`

## Performance
- **CPU Mining**: ~0.36 puzzles/sec
- **GPU Mining**: ~0.45 puzzles/sec (with Qiskit GPU acceleration)

## Requirements
- **CPU Mode**: No additional dependencies
- **GPU Mode**: Python 3.8+, Qiskit (pip install qiskit qiskit-aer numpy)

## Network
Connect to a running quantum-geth node:
- Default: http://localhost:8545
- Custom: Use -NodeURL parameter
