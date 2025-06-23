# Quantum-Miner: Standalone Quantum Mining Client

A high-performance standalone quantum miner for Quantum-Geth blockchain networks implementing **QMPoW (Quantum Merkle Proof of Work)** consensus with 16-qubit quantum circuits.

##  Key Features

- **16-qubit quantum circuits** with up to 8,192 T-gates per puzzle
- **48 quantum puzzles** per block providing 1,152-bit security
- **Multi-threaded mining** with configurable thread count
- **Remote mining** support for distributed operations
- **Real-time statistics** and progress reporting
- **Cross-platform** support (Windows/Linux)

##  Quick Start

### Running the Miner

**Basic Usage:**
`powershell
.\quantum-miner.exe -coinbase 0x742d35Cc6634C0532925a3b8D4B54c2A5e14CbE6
`

**Multi-threaded Mining:**
`powershell
.\quantum-miner.exe -coinbase 0x742d35Cc6634C0532925a3b8D4B54c2A5e14CbE6 -threads 8
`

**Remote Node Mining:**
`powershell
.\quantum-miner.exe -coinbase 0x742d35Cc6634C0532925a3b8D4B54c2A5e14CbE6 -ip 192.168.1.100
`

##  Configuration

### Command Line Options

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| -coinbase | Mining reward address (required) | - | -coinbase 0x742d35... |
| -threads | Number of mining threads | CPU cores | -threads 8 |
| -ip | Node IP address | localhost | -ip 192.168.1.100 |
| -port | Node RPC port | 8545 | -port 8545 |
| -node | Full node URL | - | -node http://node.com:8545 |
| -version | Show version information | - | -version |
| -help | Show help message | - | -help |

### Configuration Priority

1. -node flag takes highest priority (full URL)
2. -ip and -port flags combined if -node not specified  
3. Default: http://localhost:8545 if no flags specified

##  Quantum Mining Process

1. **Connection**: Connect to Quantum-Geth node via RPC
2. **Work Fetch**: Retrieve quantum mining work
3. **Circuit Execution**: Solve 48 x 16-qubit quantum circuits
4. **Proof Assembly**: Generate quantum proofs
5. **Work Submission**: Submit solution to node

### Mining Statistics

`
  Runtime: 41 seconds
  Puzzle Rate: 0.34 puzzles/sec
 Accepted Blocks: 12
 Rejected Blocks: 2
 Success Rate: 85.7%
`

##  Building from Source

**Windows:**
`powershell
go build -o quantum-miner.exe .
`

**Linux:**  
`ash
go build -o quantum-miner .
`

##  Performance Optimization

### Thread Configuration

| CPU Cores | Recommended Threads |
|-----------|-------------------|
| 4 cores | 3 threads |
| 8 cores | 6 threads |
| 16 cores | 12 threads |

### Performance Tips

- Monitor CPU temperature and throttling
- Use performance power plan on Windows
- Recommend 8GB+ RAM for optimal performance
- Use wired connection for stable latency

##  Troubleshooting

### Common Issues

**Cannot connect to geth node:**
`powershell
# Verify geth is running
Get-Process geth

# Test RPC connection  
curl http://localhost:8545 -Method POST -Body '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' -ContentType "application/json"
`

**Mining methods not available:**
- Use start-geth.ps1 for external mining
- Verify --http.api includes qmpow and eth

**High rejection rate:**
- Reduce thread count
- Check system resources

### Debug Information

`powershell
# Version information
.\quantum-miner.exe -version

# Help information
.\quantum-miner.exe -help
`

##  Available Scripts

| Script | Purpose |
|--------|---------|
| un-miner.ps1 | Start quantum miner |
| uild-windows.ps1 | Build for Windows |
| uild-linux.sh | Build for Linux |

##  Use Cases

### Development Mining
`powershell
.\run-miner.ps1 -Coinbase 0xTestAddress -Threads 4
`

### Production Mining
`powershell  
.\quantum-miner.exe -coinbase 0xAddress -threads 16 -node http://production-node:8545
`

### Pool Simulation
`powershell
# Multiple workers with different addresses
.\quantum-miner.exe -coinbase 0xAddress1 -threads 4
.\quantum-miner.exe -coinbase 0xAddress2 -threads 4
`

---

** High-Performance Quantum Mining for Everyone! **
