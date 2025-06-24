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

### GPU Mining (CUDA Optimized)
`
start-miner-gpu.bat 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
.\start-miner-gpu.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 64
`

## Performance (Batch CUDA Optimized)
- **CPU Mining**: ~0.36 puzzles/sec (1 thread)
- **GPU Mining**: ~10-100x faster (64-256 threads, batch processing)
- **Expected GPU Utilization**: 80%+ (vs 4% with old version)

## Requirements
- **CPU Mode**: No additional dependencies
- **GPU Mode**: NVIDIA GPU with CUDA support (optional, falls back to CPU)

## Network
Connect to a running quantum-geth node:
- Default: http://localhost:8545
- Custom: Use -NodeURL parameter
