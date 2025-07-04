﻿# Quantum-Geth: Ethereum with Quantum Merkle Proof of Work

> A [CoreGeth/ethereum/go-ethereum](https://github.com/ethereum/go-ethereum) fork implementing **QMPoW (Quantum Merkle Proof of Work)** consensus with 16-qubit quantum circuits and 20 T-gates per puzzle.

Quantum-Geth introduces the first practical quantum-resistant proof-of-work consensus algorithm that combines quantum circuit verification with traditional blockchain mining. Each block requires solving 128 quantum puzzles with 16-qubit circuits, providing 2,048-bit aggregate quantum security.

##  Key Features

###  **Quantum Merkle Proof of Work (QMPoW)**
- **16-qubit circuits** with up to 8,192 T-gates per puzzle
- **128 quantum puzzles** per block providing 2,048-bit security
- **Quantum-resistant** mining algorithm
- **Deterministic** quantum seed generation

###  **Mining Infrastructure**
- **External mining APIs** (qmpow_* and eth_* namespaces)
- **Remote sealer** for standalone miners
- **Work preparation** and submission handling
- **Thread-safe** mining operations

###  **Block Validation & Security**
- **Comprehensive RLP validation** for external miner submissions
- **Multi-layer integrity checks** preventing malformed blocks
- **Quantum field validation** ensuring proper structure
- **Roundtrip encoding verification** preventing blockchain corruption

###  **Network Configuration**
- **Network ID**: 73428
- **Chain ID**: 73428
- **Block Time**: 12 seconds target
- **Difficulty Adjustment**: Every 100 blocks
- **Genesis**: Custom quantum-enabled genesis block
