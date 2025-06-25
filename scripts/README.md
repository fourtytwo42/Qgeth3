# Quantum-Geth Scripts

This directory contains shell scripts for Linux/Unix systems that provide equivalent functionality to the PowerShell scripts in the root directory.

## Available Scripts

### Core Node Management
- **`start-geth.sh`** - Start Quantum-Geth node without mining (equivalent to `start-geth.ps1`)
- **`start-geth-mining.sh`** - Start Quantum-Geth node with mining enabled (equivalent to `start-geth-mining.ps1`)
- **`start-mining.sh`** - Legacy mining script (existing)
- **`reset-blockchain.sh`** - Reset and initialize blockchain (existing)

### Build and Release
- **`build-release.sh`** - Build distributable release packages (equivalent to `build-release.ps1`)

### Mining Scripts
- **`run-cpu-miner.sh`** - Run quantum miner with CPU simulation (equivalent to `run-cpu-miner.ps1`)
- **`run-gpu-miner.sh`** - Run quantum miner with GPU acceleration (equivalent to `run-gpu-miner.ps1`)

### Testing
- **`basic-test.sh`** - Basic build test for both geth and miner (equivalent to `basic_test.ps1`)
- **`run-hardness-tests.sh`** - Comprehensive security test suite (equivalent to `run-hardness-tests.ps1`)

## Cross-Platform Equivalents

| Windows PowerShell | Linux Shell Script | Description |
|-------------------|-------------------|-------------|
| `build-release.ps1` | `build-release.sh` | Build release packages |
| `start-geth.ps1` | `start-geth.sh` | Start node (no mining) |
| `start-geth-mining.ps1` | `start-geth-mining.sh` | Start node with mining |
| `run-cpu-miner.ps1` | `run-cpu-miner.sh` | CPU quantum mining |
| `run-gpu-miner.ps1` | `run-gpu-miner.sh` | GPU quantum mining |
| `run-hardness-tests.ps1` | `run-hardness-tests.sh` | Security test suite |
| `basic_test.ps1` | `basic-test.sh` | Basic build tests |

## Usage Examples

### Quick Start (Linux)
```bash
# Initialize blockchain
./scripts/reset-blockchain.sh --difficulty 1 --force

# Start node without mining
./scripts/start-geth.sh

# Start node with mining
./scripts/start-geth-mining.sh --threads 1

# Run external CPU miner
./scripts/run-cpu-miner.sh --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

# Run external GPU miner
./scripts/run-gpu-miner.sh --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
```

### Build and Test (Linux)
```bash
# Build release packages
./scripts/build-release.sh both

# Run basic tests
./scripts/basic-test.sh

# Run security test suite
./scripts/run-hardness-tests.sh --category all --verbose
```

## Script Features

### Node Scripts
- **Auto-detection**: Automatically find and use the newest release or fall back to development builds
- **Parameter validation**: Validate addresses, ports, and other parameters
- **Comprehensive help**: Use `--help` with any script for detailed usage information
- **Error handling**: Clear error messages and troubleshooting guidance

### Mining Scripts
- **Hardware detection**: Automatically detect CUDA availability for GPU mining
- **Environment validation**: Check Python, qiskit-aer, and other dependencies
- **Performance information**: Display expected mining performance
- **Flexible configuration**: Support for custom node URLs, thread counts, and logging

### Build Scripts
- **Multi-target builds**: Build geth, miner, or both
- **Release packaging**: Create complete release packages with documentation
- **Dependency checking**: Verify Go installation and build requirements
- **Clean builds**: Optional cleanup of existing releases

### Test Scripts
- **Comprehensive coverage**: Test all security assumptions and attack vectors
- **Categorized testing**: Run specific test categories or all tests
- **Detailed reporting**: Success rates, timing, and failure analysis
- **CI/CD ready**: Appropriate exit codes for automation

## File Permissions

On Linux/Unix systems, make sure the scripts are executable:
```bash
chmod +x scripts/*.sh
```

## Requirements

### All Scripts
- Go 1.19 or later
- Bash shell

### Mining Scripts
- Python 3.8+
- qiskit and qiskit-aer packages
- For GPU mining: CUDA toolkit and compatible GPU

### Test Scripts
- bc (basic calculator) for floating point calculations
- Go test framework

## Quantum-Geth Features

All scripts support the current Quantum-Geth configuration:
- **128 sequential quantum puzzles** per block
- **16 qubits × 20 T-gates** per puzzle
- **Bitcoin-style halving** rewards (50 QGC → 25 QGC → 12.5 QGC...)
- **ASERT-Q difficulty adjustment** targeting 12-second blocks
- **Mahadev→CAPSS→Nova** proof stack
- **Dilithium-2 self-attestation**

## Troubleshooting

### Common Issues
1. **Permission denied**: Run `chmod +x scripts/*.sh` to make scripts executable
2. **Go not found**: Install Go 1.19+ and ensure it's in your PATH
3. **Python/qiskit issues**: Install Python 3.8+ and run `pip install qiskit qiskit-aer`
4. **CUDA not available**: For GPU mining, install NVIDIA CUDA toolkit

### Getting Help
Each script supports the `--help` flag for detailed usage information:
```bash
./scripts/start-geth.sh --help
./scripts/run-cpu-miner.sh --help
./scripts/build-release.sh --help
``` 