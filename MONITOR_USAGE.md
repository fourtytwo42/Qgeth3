# Quantum-Geth Blockchain Monitor Usage Guide

## Basic Usage

# Show current status once (no continuous monitoring)
.\monitor-blockchain.ps1 -once

# Start continuous monitoring (refreshes every 5 seconds)
.\monitor-blockchain.ps1

# Monitor with custom refresh rate (every 10 seconds)
.\monitor-blockchain.ps1 -refreshSeconds 10

# Monitor a remote node
.\monitor-blockchain.ps1 -nodeUrl "http://192.168.1.100:8545"

## What it shows:

 CURRENT STATUS
- Current block number
- Current difficulty (both fractional and raw values)
- Time since last block was mined

 DIFFICULTY RETARGETING  
- Retarget interval (100 blocks)
- Blocks remaining until next retarget
- Next retarget block number

 BLOCK TIMING ANALYSIS
- Target block time (12 seconds)
- Average block time from last 10 blocks
- Min/Max block times
- Deviation from target percentage

 QUANTUM MINING INFO
- Quantum puzzles per block (48)
- Circuit complexity (16 qubits  8192 T-gates)
- Security level (1,152-bit aggregate)

Press Ctrl+C to stop continuous monitoring.
