# Quantum Mining Setup

This guide shows how to start mining with the quantum proof of work algorithm.

## Quick Start

1. **Start mining with 4 threads (basic setup):**
   ```powershell
   .\start-quantum-mining.ps1 -Threads 4
   ```

2. **Initialize genesis and start mining:**
   ```powershell
   .\start-quantum-mining.ps1 -Threads 4 -InitGenesis
   ```

3. **Start mining with verbose output:**
   ```powershell
   .\start-quantum-mining.ps1 -Threads 8 -VerboseLogging
   ```

## Advanced Usage

### Full Parameter Example
```powershell
.\start-quantum-mining.ps1 -Threads 8 -DataDir "my-quantum-chain" -NetworkId "73428" -VerboseLogging
```

### With Account Unlocking
```powershell
# First create an account and password file
.\start-quantum-mining.ps1 -Threads 4 -MinerAccount "0x1234..." -PasswordFile "password.txt"
```

## Parameters

- **`-Threads`** (required): Number of mining threads (1-16 recommended)
- **`-DataDir`**: Blockchain data directory (default: "qdata")  
- **`-NetworkId`**: Network ID (default: "73428")
- **`-MinerAccount`**: Account address to unlock for mining rewards
- **`-PasswordFile`**: File containing password for account unlock
- **`-InitGenesis`**: Force genesis initialization (cleans existing data)
- **`-VerboseLogging`**: Enable detailed logging

## Quantum Parameters

The blockchain uses these quantum proof of work settings:
- **QBits**: 8 (qubits per puzzle)
- **T-Gates**: 25 (T-gates per puzzle)  
- **L_net**: 64 (puzzles per block)
- **Target Block Time**: 12 seconds
- **Difficulty Adjustment**: ±4 puzzles every 2048 blocks

## Mining Process

1. **Build**: Script automatically builds geth if needed
2. **Genesis**: Initializes quantum blockchain if no data exists  
3. **Mining**: Starts quantum proof of work mining
4. **Verification**: Each block contains quantum proofs for 64 puzzles

## Expected Output

When mining starts successfully, you should see:
```
INFO [timestamp] Starting quantum mining              qbits=8 tcount=25 lnet=64
INFO [timestamp] QMPoW: 64 puzzles, aggregate proof 2.1 kB, solved in 0.45s  
INFO [timestamp] Successfully sealed new quantum block number=1 hash=0x...
```

## Troubleshooting

### Build Issues
If build fails with dependency errors:
```powershell
cd quantum-geth
go mod download
go mod tidy
make geth
```

### Genesis Issues  
If genesis initialization fails, try:
```powershell
.\start-quantum-mining.ps1 -Threads 4 -InitGenesis -VerboseLogging
```

### Mining Issues
Check that Python is available for the quantum solver:
```powershell
python --version
python quantum-geth/tools/solver/solver.py
```

## Performance Tips

- **Threads**: Use 1-2x your CPU core count
- **Memory**: Quantum proofs use ~2KB per block
- **Storage**: Blockchain grows ~2KB per block for quantum data
- **Network**: Chain ID 73428 avoids conflicts with other networks

## Next Steps

Once mining is working:
1. Mine your first quantum block
2. Check block headers contain quantum fields
3. Verify quantum proofs are being generated
4. Monitor difficulty adjustments over time

The quantum blockchain is now ready for the future of decentralized computing! 🚀 