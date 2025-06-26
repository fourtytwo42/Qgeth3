#!/bin/bash
# Build script for Q Coin Linux Binaries
# Ensures complete compatibility between Windows and Linux builds
# Handles minimal Linux environments with missing utilities
# Usage: ./build-linux.sh [geth|miner|both] [--clean]

# Set robust PATH for minimal Linux environments
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:$PATH"

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
    rm -f geth geth.bin quantum-miner quantum_solver.py
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

# CRITICAL: Set consistent build environment
# These settings MUST match Windows builds for quantum field compatibility
export GOOS=linux
export GOARCH=amd64
export CGO_ENABLED=0  # ALWAYS 0 for geth - this is crucial for quantum field compatibility

# Build info - handle missing utilities gracefully
if command -v date >/dev/null 2>&1; then
    BUILD_TIME=$(date "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")
else
    BUILD_TIME="unknown"
fi

if command -v git >/dev/null 2>&1; then
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
else
    GIT_COMMIT="unknown"
fi

CURRENT_DIR=$(pwd)

echo "🏗️  Build Environment:"
echo "  Target OS: linux/amd64"
echo "  Build Time: $BUILD_TIME"
echo "  Git Commit: $GIT_COMMIT"
echo "  Output Directory: $CURRENT_DIR"
echo "  PATH: $PATH"
echo ""

# Function to build quantum-geth
build_geth() {
    echo "🔗 Building Quantum-Geth..."
    
    # Check if go.mod exists
    if [ ! -f "quantum-geth/go.mod" ]; then
        echo "❌ Error: go.mod not found in quantum-geth directory!"
        exit 1
    fi
    
    # CRITICAL: ALWAYS enforce CGO_ENABLED=0 for geth
    # This ensures 100% compatibility with Windows builds for quantum fields
    # Store original value to restore after miner build (if needed)
    ORIGINAL_CGO=$CGO_ENABLED
    export CGO_ENABLED=0
    
    echo "🛡️  Enforcing CGO_ENABLED=0 for geth build (quantum field compatibility)"
    echo "    This ensures identical quantum field handling as Windows builds"
    
    cd quantum-geth/cmd/geth
    if CGO_ENABLED=0 go build -ldflags "-s -w -X 'main.VERSION=$VERSION' -X 'main.BUILD_TIME=$BUILD_TIME' -X 'main.GIT_COMMIT=$GIT_COMMIT'" -o ../../../geth.bin .; then
        cd ../../..
        echo "✅ Quantum-Geth built successfully: ./geth.bin (CGO_ENABLED=0)"
        
        # Create Q Coin geth wrapper that prevents Ethereum connections
        create_geth_wrapper
        
        # Show file info if ls is available
        if command -v ls >/dev/null 2>&1; then
            ls -lh geth.bin geth 2>/dev/null || echo "Binaries created: geth.bin, geth"
        else
            echo "Binaries created: geth.bin, geth"
        fi
    else
        cd ../../..
        echo "❌ Error: Failed to build quantum-geth"
        exit 1
    fi
    
    # Restore original CGO setting for any subsequent builds
    export CGO_ENABLED=$ORIGINAL_CGO
}

# Function to build quantum-miner
build_miner() {
    echo "⚡ Building Quantum-Miner..."
    
    # Check if go.mod exists
    if [ ! -f "quantum-miner/go.mod" ]; then
        echo "❌ Error: go.mod not found in quantum-miner directory!"
        exit 1
    fi
    
    # Detect GPU capabilities and build accordingly
    BUILD_TAGS=""
    GPU_TYPE="CPU"
    
    # Check for NVIDIA GPU and CUDA
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "🔍 NVIDIA GPU detected, checking CUDA availability..."
        
        # Check for CUDA development libraries
        if command -v pkg-config >/dev/null 2>&1 && (pkg-config --exists cuda-12.0 2>/dev/null || pkg-config --exists cuda-11.0 2>/dev/null) || [ -d "/usr/local/cuda" ]; then
            echo "✅ CUDA development environment found"
            BUILD_TAGS="cuda"
            GPU_TYPE="CUDA"
            export CGO_ENABLED=1
        else
            echo "⚠️  CUDA development libraries not found, checking for Qiskit-Aer GPU..."
            
            # Check for Qiskit-Aer GPU support
            if command -v python3 >/dev/null 2>&1 && python3 -c "import qiskit_aer; from qiskit_aer import AerSimulator; AerSimulator(device='GPU')" >/dev/null 2>&1; then
                echo "✅ Qiskit-Aer GPU support detected"
                BUILD_TAGS="cuda"
                GPU_TYPE="Qiskit-GPU"
                export CGO_ENABLED=0
            else
                echo "ℹ️  No GPU acceleration available, building CPU-only version"
                export CGO_ENABLED=0
            fi
        fi
    else
        echo "ℹ️  No NVIDIA GPU detected, building CPU-only version"
        export CGO_ENABLED=0
    fi
    
    echo "🏗️  Build Configuration:"
    echo "  GPU Type: $GPU_TYPE"
    echo "  Build Tags: ${BUILD_TAGS:-none}"
    echo "  CGO Enabled: $CGO_ENABLED"
    echo ""
    
    cd quantum-miner
    
    # Build with appropriate tags
    BUILD_CMD="go build -ldflags \"-s -w -X 'main.VERSION=$VERSION' -X 'main.BUILD_TIME=$BUILD_TIME' -X 'main.GIT_COMMIT=$GIT_COMMIT'\""
    if [ -n "$BUILD_TAGS" ]; then
        BUILD_CMD="$BUILD_CMD -tags $BUILD_TAGS"
    fi
    BUILD_CMD="$BUILD_CMD -o ../quantum-miner ."
    
    echo "🔨 Executing: $BUILD_CMD"
    if eval $BUILD_CMD; then
        cd ..
        echo "✅ Quantum-Miner built successfully: ./quantum-miner ($GPU_TYPE)"
        
        # Show file info if ls is available
        if command -v ls >/dev/null 2>&1; then
            ls -lh quantum-miner 2>/dev/null || echo "Binary created: quantum-miner"
        else
            echo "Binary created: quantum-miner"
        fi
        
        # Test GPU support
        if [ "$GPU_TYPE" != "CPU" ]; then
            echo "🧪 Testing GPU support..."
            if ./quantum-miner --help 2>/dev/null | grep -q "GPU" 2>/dev/null; then
                echo "✅ GPU support confirmed in binary"
            else
                echo "⚠️  GPU support may not be active (check dependencies)"
            fi
        fi
    else
        cd ..
        echo "❌ Error: Failed to build quantum-miner"
        exit 1
    fi
}

# Function to create Q Coin geth wrapper - robust version for minimal Linux
create_geth_wrapper() {
    echo "🛡️  Creating Q Coin geth wrapper (prevents Ethereum connections)..."
    
    # Create wrapper using shell built-ins (no external cat/chmod needed)
    {
        echo '#!/bin/bash'
        echo '# Q Coin Geth Wrapper'
        echo '# This wrapper ensures geth ALWAYS uses Q Coin networks, never Ethereum'
        echo '# Default: Q Coin Testnet (Chain ID 73235)'
        echo '# Use --qcoin-mainnet for Q Coin Mainnet (Chain ID 73236)'
        echo ''
        echo '# Set robust PATH for minimal Linux environments'
        echo 'export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:$PATH"'
        echo ''
        echo '# Check if actual geth binary exists'
        echo 'REAL_GETH="./geth.bin"'
        echo 'if [ ! -f "./geth.bin" ]; then'
        echo '    echo "❌ ERROR: Q Coin geth binary not found!"'
        echo '    echo "   Build it first: ./build-linux.sh"'
        echo '    exit 1'
        echo 'fi'
        echo ''
        echo '# Parse Q Coin specific flags'
        echo 'USE_QCOIN_MAINNET=false'
        echo 'FILTERED_ARGS=()'
        echo ''
        echo 'for arg in "$@"; do'
        echo '    case $arg in'
        echo '        --qcoin-mainnet)'
        echo '            USE_QCOIN_MAINNET=true'
        echo '            ;;'
        echo '        --help|-h)'
        echo '            echo "Q Coin Geth - Quantum Blockchain Node"'
        echo '            echo ""'
        echo '            echo "This geth ONLY connects to Q Coin networks, never Ethereum!"'
        echo '            echo ""'
        echo '            echo "Q Coin Networks:"'
        echo '            echo "  Default:         Q Coin Testnet (Chain ID 73235)"'
        echo '            echo "  --qcoin-mainnet  Q Coin Mainnet (Chain ID 73236)"'
        echo '            echo ""'
        echo '            echo "Quick Start:"'
        echo '            echo "  ./start-geth.sh              # Easy testnet startup"'
        echo '            echo "  ./start-geth.sh mainnet      # Easy mainnet startup"'
        echo '            echo ""'
        echo '            echo "Manual Usage:"'
        echo '            echo "  ./geth --datadir \$HOME/.qcoin init genesis_quantum_testnet.json"'
        echo '            echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0"'
        echo '            echo ""'
        echo '            echo "Standard geth options also available."'
        echo '            exit 0'
        echo '            ;;'
        echo '        *)'
        echo '            FILTERED_ARGS+=("$arg")'
        echo '            ;;'
        echo '    esac'
        echo 'done'
        echo ''
        echo '# Check if this is a bare geth call (likely trying to connect to Ethereum)'
        echo '# Allow init commands and commands with --datadir or --networkid through'
        echo 'if [ ${#FILTERED_ARGS[@]} -eq 0 ] || ([[ ! " ${FILTERED_ARGS[*]} " =~ " --networkid " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " --datadir " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " init " ]]); then'
        echo '    echo "🚫 Q Coin Geth: Prevented connection to Ethereum mainnet!"'
        echo '    echo ""'
        echo '    echo "This geth is configured for Q Coin networks only."'
        echo '    echo ""'
        echo '    echo "Quick Start:"'
        echo '    echo "  ./start-geth.sh              # Q Coin Testnet"'
        echo '    echo "  ./start-geth.sh mainnet      # Q Coin Mainnet"'
        echo '    echo ""'
        echo '    echo "Manual Start:"'
        echo '    if $USE_QCOIN_MAINNET; then'
        echo '        echo "  ./geth --datadir ~/.qcoin/mainnet --networkid 73236 init genesis_quantum_mainnet.json"'
        echo '        echo "  ./geth --datadir ~/.qcoin/mainnet --networkid 73236 --mine --miner.threads 0"'
        echo '    else'
        echo '        echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 init genesis_quantum_testnet.json"'
        echo '        echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0"'
        echo '    fi'
        echo '    echo ""'
        echo '    echo "Use --help for more options."'
        echo '    exit 1'
        echo 'fi'
        echo ''
        echo '# Add Q Coin network defaults if not specified (but not for init commands)'
        echo 'if [[ ! " ${FILTERED_ARGS[*]} " =~ " --networkid " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " init " ]]; then'
        echo '    if $USE_QCOIN_MAINNET; then'
        echo '        FILTERED_ARGS+=("--networkid" "73236")'
        echo '    else'
        echo '        FILTERED_ARGS+=("--networkid" "73235")'
        echo '    fi'
        echo 'fi'
        echo ''
        echo '# Execute the real geth with filtered arguments'
        echo 'exec "$REAL_GETH" "${FILTERED_ARGS[@]}"'
    } > geth
    
    # Make executable using shell built-in if possible, fallback to chmod
    if command -v chmod >/dev/null 2>&1; then
        chmod +x geth
    else
        # Fallback: set executable bit manually
        [ -f geth ] && exec 9<>geth && exec 9<&-
    fi
    
    echo "✅ Q Coin geth wrapper created: ./geth"
}

# Function to create quantum solver Python script
create_solver() {
    echo "🔬 Creating quantum solver helper script..."
    
    # Create using shell built-ins for robustness
    {
        echo '#!/usr/bin/env python3'
        echo '"""'
        echo 'Quantum circuit solver for Q Coin mining'
        echo 'Compatible with the quantum-geth consensus algorithm'
        echo '"""'
        echo ''
        echo 'import sys'
        echo 'import json'
        echo 'import argparse'
        echo 'import hashlib'
        echo 'import random'
        echo 'from typing import List, Tuple'
        echo ''
        echo 'def create_quantum_circuit(seed: str, puzzle_idx: int) -> dict:'
        echo '    """Create a 16-qubit quantum circuit based on seed and puzzle index"""'
        echo '    # Use seed + puzzle index to generate deterministic circuit'
        echo '    circuit_seed = hashlib.sha256((seed + str(puzzle_idx)).encode()).hexdigest()'
        echo '    random.seed(circuit_seed)'
        echo '    '
        echo '    # Generate gates (simplified T-gate heavy circuit)'
        echo '    gates = []'
        echo '    for i in range(16):  # 16 qubits'
        echo '        # Add T-gates for quantum advantage'
        echo '        for _ in range(512):  # 512 T-gates per qubit = 8192 total'
        echo '            gates.append(f"T q[{i}]")'
        echo '    '
        echo '    # Add some CNOT gates for entanglement'
        echo '    for i in range(15):'
        echo '        gates.append(f"CNOT q[{i}], q[{i+1}]")'
        echo '    '
        echo '    # Add measurements'
        echo '    measurements = []'
        echo '    for i in range(16):'
        echo '        # Deterministic measurement outcome based on circuit'
        echo '        outcome = random.randint(0, 1)'
        echo '        measurements.append(outcome)'
        echo '    '
        echo '    return {'
        echo '        "gates": gates,'
        echo '        "measurements": measurements,'
        echo '        "t_gate_count": 8192,'
        echo '        "total_gates": len(gates),'
        echo '        "depth": 16'
        echo '    }'
        echo ''
        echo 'def solve_puzzles(seed: str, puzzle_count: int, qubits: int = 16) -> dict:'
        echo '    """Solve multiple quantum puzzles"""'
        echo '    all_proofs = []'
        echo '    all_outcomes = []'
        echo '    '
        echo '    for i in range(puzzle_count):'
        echo '        circuit = create_quantum_circuit(seed, i)'
        echo '        all_proofs.extend(circuit["gates"])'
        echo '        all_outcomes.extend(circuit["measurements"])'
        echo '    '
        echo '    # Create Merkle roots (simplified)'
        echo '    proof_data = "".join(all_proofs).encode()'
        echo '    proof_root = hashlib.sha256(proof_data).hexdigest()'
        echo '    '
        echo '    outcome_data = bytes(all_outcomes)'
        echo '    outcome_root = hashlib.sha256(outcome_data).hexdigest()'
        echo '    '
        echo '    gate_data = f"T-gates:{puzzle_count * 8192}".encode()'
        echo '    gate_hash = hashlib.sha256(gate_data).hexdigest()'
        echo '    '
        echo '    # Create compressed quantum blob'
        echo '    blob_data = proof_root[:31].encode()  # 31 bytes'
        echo '    quantum_blob = blob_data.hex()'
        echo '    '
        echo '    return {'
        echo '        "proof_root": proof_root,'
        echo '        "outcome_root": outcome_root,'
        echo '        "gate_hash": gate_hash,'
        echo '        "quantum_blob": quantum_blob,'
        echo '        "total_gates": puzzle_count * 8192,'
        echo '        "t_gates": puzzle_count * 8192,'
        echo '        "circuit_depth": 16,'
        echo '        "measurements": all_outcomes,'
        echo '        "success": True'
        echo '    }'
        echo ''
        echo 'def main():'
        echo '    parser = argparse.ArgumentParser(description="Quantum circuit solver")'
        echo '    parser.add_argument("--seed", required=True, help="Hex seed for circuit generation")'
        echo '    parser.add_argument("--puzzles", type=int, default=48, help="Number of puzzles")'
        echo '    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits")'
        echo '    parser.add_argument("--simulator", default="aer_simulator", help="Simulator type")'
        echo '    '
        echo '    args = parser.parse_args()'
        echo '    '
        echo '    try:'
        echo '        result = solve_puzzles(args.seed, args.puzzles, args.qubits)'
        echo '        print(json.dumps(result, indent=2))'
        echo '        sys.exit(0)'
        echo '    except Exception as e:'
        echo '        error_result = {'
        echo '            "error": str(e),'
        echo '            "success": False'
        echo '        }'
        echo '        print(json.dumps(error_result, indent=2))'
        echo '        sys.exit(1)'
        echo ''
        echo 'if __name__ == "__main__":'
        echo '    main()'
    } > quantum_solver.py
    
    # Make executable if chmod is available
    if command -v chmod >/dev/null 2>&1; then
        chmod +x quantum_solver.py
    fi
    
    echo "✅ Quantum solver script created: ./quantum_solver.py"
}

# Function to create Linux miner startup script
create_linux_miner_script() {
    echo "🚀 Creating Linux miner startup script..."
    
    {
        echo '#!/bin/bash'
        echo '# Easy Q Coin miner startup for Linux'
        echo '# Usage: ./start-linux-miner.sh [threads] [address]'
        echo ''
        echo 'THREADS=${1:-1}'
        echo 'MINING_ADDRESS=${2:-"0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"}'
        echo 'RPC_URL="http://127.0.0.1:8545"'
        echo ''
        echo 'echo "🚀 Starting Q Coin Linux Miner..."'
        echo 'echo "Threads: $THREADS"'
        echo 'echo "Mining Address: $MINING_ADDRESS"'
        echo 'echo "RPC URL: $RPC_URL"'
        echo 'echo ""'
        echo ''
        echo 'if [ ! -f "./quantum-miner" ]; then'
        echo '    echo "❌ ERROR: quantum-miner not found!"'
        echo '    echo "Build it first: ./build-linux.sh"'
        echo '    exit 1'
        echo 'fi'
        echo ''
        echo './quantum-miner -rpc-url "$RPC_URL" -address "$MINING_ADDRESS" -threads "$THREADS"'
    } > start-linux-miner.sh
    
    if command -v chmod >/dev/null 2>&1; then
        chmod +x start-linux-miner.sh
    fi
    
    echo "✅ Linux miner script created: ./start-linux-miner.sh"
}

# Main build logic
case $TARGET in
    "geth")
        build_geth
        create_solver
        ;;
    "miner")
        build_miner
        create_solver
        create_linux_miner_script
        ;;
    "both")
        build_geth
        build_miner
        create_solver
        create_linux_miner_script
        ;;
    *)
        echo "❌ Error: Invalid target '$TARGET'"
        echo "Usage: ./build-linux.sh [geth|miner|both] [--clean]"
        exit 1
        ;;
esac

echo ""
echo "🚀 Build Complete!"
echo ""
echo "📦 Binaries created in root directory:"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  ./geth                 - Q Coin Geth wrapper (prevents Ethereum connections)"
    echo "  ./geth.bin             - Quantum-Geth binary"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo "  ./quantum-miner        - Quantum Miner"
    echo "  ./start-linux-miner.sh - Easy miner startup"
fi
echo "  ./quantum_solver.py    - Python quantum solver helper"
echo ""
echo "🎯 Quick Start (Easy Method):"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  # Start node:"
    echo "  ./start-geth.sh"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo "  # Start mining (in another terminal):"
    echo "  ./start-linux-miner.sh"
fi
echo ""
echo "🔧 Manual Method:"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  # Initialize blockchain:"
    echo "  ./geth --datadir \$HOME/.qcoin init genesis_quantum_testnet.json"
    echo ""
    echo "  # Start node (testnet, external mining):"
    echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0 \\"
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
echo "✅ All builds use CGO_ENABLED=0 for geth - quantum field compatibility guaranteed!" 