# Quantum-Geth Scripts Directory

This directory contains Linux/macOS shell script equivalents of the Windows PowerShell scripts in the root directory.

## Available Scripts

| Script | Purpose | Windows Equivalent |
|--------|---------|-------------------|
| `reset-blockchain.sh` | Reset blockchain with custom difficulty | `reset-blockchain.ps1` |
| `start-mining.sh` | Start quantum mining process | `start-geth-mining.ps1` |
| `monitor.sh` | Monitor mining progress in real-time | `monitor-mining.ps1` |

## Setup Instructions

### First Time Setup (Linux/macOS)

1. **Make scripts executable:**
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Install dependencies:**
   ```bash
   # Install Python dependencies
   pip install qiskit qiskit-aer numpy
   
   # Compile Quantum-Geth
   cd quantum-geth
   make geth
   cd ..
   ```

## Usage Examples

### Reset Blockchain
```bash
# Reset with default difficulty (100)
./scripts/reset-blockchain.sh

# Reset with custom difficulty
./scripts/reset-blockchain.sh --difficulty 1000

# Reset without confirmation prompt
./scripts/reset-blockchain.sh --difficulty 100 --force

# Get help
./scripts/reset-blockchain.sh --help
```

### Start Mining
```bash
# Start mining with default settings
./scripts/start-mining.sh

# Start with multiple threads
./scripts/start-mining.sh --threads 4

# Start with network connections enabled
./scripts/start-mining.sh --network

# Start with debug logging
./scripts/start-mining.sh --verbosity 5

# Get help
./scripts/start-mining.sh --help
```

### Monitor Mining
```bash
# Monitor with default settings (3s refresh)
./scripts/monitor.sh

# Monitor with custom refresh interval
./scripts/monitor.sh --interval 5

# Monitor with more log lines
./scripts/monitor.sh --lines 100

# Get help
./scripts/monitor.sh --help
```

## Script Features

### Cross-Platform Compatibility
- **Automatic binary detection**: Scripts automatically find `geth` binary in various locations
- **Color output**: Rich terminal colors for better readability
- **Error handling**: Comprehensive error checking and user feedback
- **Parameter validation**: Input validation with helpful error messages

### Configuration Options
- **Data directory**: Customizable blockchain data location
- **Network settings**: Isolated mode or peer connections
- **Mining parameters**: Configurable thread count and difficulty
- **Logging**: Adjustable verbosity levels

### Safety Features
- **Confirmation prompts**: Prevents accidental blockchain deletion
- **Process management**: Automatic cleanup of existing geth processes
- **Validation**: Parameter and dependency checking
- **Graceful shutdown**: Proper signal handling for clean exits

## Troubleshooting

### Common Issues

1. **Permission denied:**
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Geth binary not found:**
   ```bash
   cd quantum-geth
   make geth
   ```

3. **Python dependencies missing:**
   ```bash
   pip install qiskit qiskit-aer numpy
   ```

4. **Mining process not starting:**
   - Check if blockchain is initialized: `./scripts/reset-blockchain.sh`
   - Verify quantum solver exists: `ls quantum-geth/tools/solver/qiskit_solver.py`

### Log Locations
- **Blockchain data**: `qdata_quantum/`
- **Geth logs**: `qdata_quantum/geth/geth.log`
- **Mining output**: Terminal output or background logs

## Advanced Usage

### Custom Configuration
You can modify the default values at the top of each script:
- `DATADIR`: Change blockchain data directory
- `NETWORKID`: Change network ID for private networks
- `ETHERBASE`: Change mining reward address

### Background Mining
```bash
# Start mining in background
nohup ./scripts/start-mining.sh > mining.log 2>&1 &

# Monitor background mining
tail -f mining.log
```

### Production Deployment
For production use, consider:
- Using systemd services for automatic startup
- Implementing log rotation
- Setting up monitoring and alerting
- Configuring firewall rules for P2P networking

## Support

For issues with these scripts, please check:
1. This README for common solutions
2. The main project README.md for general setup
3. GitHub Issues for known problems
4. Script help output (`--help`) for usage details 