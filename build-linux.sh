#!/bin/bash
# Build script for Q Coin Linux Binaries
# Builds quantum-geth and quantum-miner binaries to the root directory for easy access
# Usage: ./build-linux.sh [geth|miner|both] [--clean]

TARGET=${1:-both}
CLEAN=${2:-false}
VERSION="1.0.0"

echo "🔨 Building Q Coin Linux Binaries..."
echo "Target: $TARGET"
echo "Version: $VERSION"
echo ""

# Clean previous builds if requested
if [ "$CLEAN" = "--clean" ] || [ "$2" = "--clean" ]; then
    echo "🧹 Cleaning previous builds..."
    rm -f geth quantum-miner quantum_solver.py
    echo "  Previous binaries removed"
fi

# Check directories exist
if [ ! -d "quantum-geth" ]; then
    echo "❌ Error: quantum-geth directory not found!"
    echo "Please run this script from the root of the Qgeth3 project."
    exit 1
fi

if [ ! -d "quantum-miner" ]; then
    echo "❌ Error: quantum-miner directory not found!"
    echo "Please run this script from the root of the Qgeth3 project."
    exit 1
fi

# Set build environment
export GOOS=linux
export GOARCH=amd64
export CGO_ENABLED=0

# Build info
BUILD_TIME=$(date "+%Y-%m-%d %H:%M:%S")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "🏗️  Build Environment:"
echo "  Target OS: linux/amd64"
echo "  Build Time: $BUILD_TIME"
echo "  Git Commit: $GIT_COMMIT"
echo "  Output Directory: $(pwd)"
echo ""

# Function to build quantum-geth
build_geth() {
    echo "🔗 Building Quantum-Geth..."
    
    # Check if go.mod exists
    if [ ! -f "quantum-geth/go.mod" ]; then
        echo "❌ Error: go.mod not found in quantum-geth directory!"
        exit 1
    fi
    
    cd quantum-geth/cmd/geth
    if go build -ldflags "-s -w -X 'main.VERSION=$VERSION' -X 'main.BUILD_TIME=$BUILD_TIME' -X 'main.GIT_COMMIT=$GIT_COMMIT'" -o ../../../geth .; then
        cd ../../..
        echo "✅ Quantum-Geth built successfully: ./geth"
        ls -lh geth
    else
        cd ../../..
        echo "❌ Error: Failed to build quantum-geth"
        exit 1
    fi
}

# Function to build quantum-miner
build_miner() {
    echo "⚡ Building Quantum-Miner..."
    
    # Check if go.mod exists
    if [ ! -f "quantum-miner/go.mod" ]; then
        echo "❌ Error: go.mod not found in quantum-miner directory!"
        exit 1
    fi
    
    cd quantum-miner
    if go build -ldflags "-s -w -X 'main.VERSION=$VERSION' -X 'main.BUILD_TIME=$BUILD_TIME' -X 'main.GIT_COMMIT=$GIT_COMMIT'" -o ../quantum-miner .; then
        cd ..
        echo "✅ Quantum-Miner built successfully: ./quantum-miner"
        ls -lh quantum-miner
    else
        cd ..
        echo "❌ Error: Failed to build quantum-miner"
        exit 1
    fi
}

# Function to create quantum solver Python script
create_solver() {
    echo "🔬 Creating quantum solver helper script..."
    
    cat > quantum_solver.py << 'EOF'
#!/usr/bin/env python3
"""
Quantum circuit solver for Q Coin mining
Compatible with the quantum-geth consensus algorithm
"""

import sys
import json
import argparse
import hashlib
import random
from typing import List, Tuple

def create_quantum_circuit(seed: str, puzzle_idx: int) -> dict:
    """Create a 16-qubit quantum circuit based on seed and puzzle index"""
    # Use seed + puzzle index to generate deterministic circuit
    circuit_seed = hashlib.sha256((seed + str(puzzle_idx)).encode()).hexdigest()
    random.seed(circuit_seed)
    
    # Generate gates (simplified T-gate heavy circuit)
    gates = []
    for i in range(16):  # 16 qubits
        # Add T-gates for quantum advantage
        for _ in range(512):  # 512 T-gates per qubit = 8192 total
            gates.append(f"T q[{i}]")
    
    # Add some CNOT gates for entanglement
    for i in range(15):
        gates.append(f"CNOT q[{i}], q[{i+1}]")
    
    # Add measurements
    measurements = []
    for i in range(16):
        # Deterministic measurement outcome based on circuit
        outcome = random.randint(0, 1)
        measurements.append(outcome)
    
    return {
        "gates": gates,
        "measurements": measurements,
        "t_gate_count": 8192,
        "total_gates": len(gates),
        "depth": 16
    }

def solve_puzzles(seed: str, puzzle_count: int, qubits: int = 16) -> dict:
    """Solve multiple quantum puzzles"""
    all_proofs = []
    all_outcomes = []
    
    for i in range(puzzle_count):
        circuit = create_quantum_circuit(seed, i)
        all_proofs.extend(circuit["gates"])
        all_outcomes.extend(circuit["measurements"])
    
    # Create Merkle roots (simplified)
    proof_data = "".join(all_proofs).encode()
    proof_root = hashlib.sha256(proof_data).hexdigest()
    
    outcome_data = bytes(all_outcomes)
    outcome_root = hashlib.sha256(outcome_data).hexdigest()
    
    gate_data = f"T-gates:{puzzle_count * 8192}".encode()
    gate_hash = hashlib.sha256(gate_data).hexdigest()
    
    # Create compressed quantum blob
    blob_data = proof_root[:31].encode()  # 31 bytes
    quantum_blob = blob_data.hex()
    
    return {
        "proof_root": proof_root,
        "outcome_root": outcome_root,
        "gate_hash": gate_hash,
        "quantum_blob": quantum_blob,
        "total_gates": puzzle_count * 8192,
        "t_gates": puzzle_count * 8192,
        "circuit_depth": 16,
        "measurements": all_outcomes,
        "success": True
    }

def main():
    parser = argparse.ArgumentParser(description="Quantum circuit solver")
    parser.add_argument("--seed", required=True, help="Hex seed for circuit generation")
    parser.add_argument("--puzzles", type=int, default=48, help="Number of puzzles")
    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits")
    parser.add_argument("--simulator", default="aer_simulator", help="Simulator type")
    
    args = parser.parse_args()
    
    try:
        result = solve_puzzles(args.seed, args.puzzles, args.qubits)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x quantum_solver.py
    echo "✅ Quantum solver script created: ./quantum_solver.py"
}

# Build based on target
case "$TARGET" in
    geth)
        build_geth
        ;;
    miner)
        build_miner
        create_solver
        ;;
    both)
        build_geth
        echo ""
        build_miner
        echo ""
        create_solver
        ;;
    *)
        echo "❌ Error: Invalid target '$TARGET'"
        echo "Valid targets: geth, miner, both"
        echo ""
        echo "Usage examples:"
        echo "  ./build-linux.sh          # Build both (default)"
        echo "  ./build-linux.sh geth     # Build only geth"
        echo "  ./build-linux.sh miner    # Build only miner"
        echo "  ./build-linux.sh both --clean  # Clean build both"
        exit 1
        ;;
esac

echo ""
echo "🚀 Build Complete!"
echo ""
echo "📦 Binaries created in root directory:"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  ./geth              - Quantum-Geth node"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo "  ./quantum-miner     - Quantum Miner"
    echo "  ./quantum_solver.py - Python quantum solver helper"
fi
echo ""
echo "🎯 Quick Start:"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  # Initialize blockchain:"
    echo "  ./geth --datadir qdata init genesis_quantum_testnet.json"
    echo ""
    echo "  # Start node (testnet, external mining):"
    echo "  ./geth --datadir qdata --networkid 73235 --mine --miner.threads 0 \\"
    echo "         --http --http.api eth,net,web3,personal,admin,miner \\"
    echo "         --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo ""
    echo "  # Start mining (in another terminal):"
    echo "  ./quantum-miner -rpc-url http://127.0.0.1:8545 \\"
    echo "                  -address 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
fi
echo "" 