#!/bin/bash
# Start Geth with Mining - Quantum-Geth with Halving
# Starts the quantum geth node with quantum proof-of-work mining
# Usage: ./start-geth-mining.sh --threads 1 --verbosity 4

# Default parameters
THREADS=1
DATADIR="qdata"
NETWORKID=1337
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
VERBOSITY=4
QUANTUM_SOLVER="cpu"
ISOLATED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --datadir)
            DATADIR="$2"
            shift 2
            ;;
        --networkid)
            NETWORKID="$2"
            shift 2
            ;;
        --etherbase)
            ETHERBASE="$2"
            shift 2
            ;;
        --verbosity)
            VERBOSITY="$2"
            shift 2
            ;;
        --quantum-solver)
            QUANTUM_SOLVER="$2"
            shift 2
            ;;
        --isolated)
            ISOLATED=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --threads NUM       Mining threads (default: 1)"
            echo "  --datadir DIR       Data directory (default: qdata)"
            echo "  --networkid ID      Network ID (default: 1337)"
            echo "  --etherbase ADDR    Coinbase address"
            echo "  --verbosity LEVEL   Log verbosity (default: 4)"
            echo "  --quantum-solver    Quantum solver (default: cpu)"
            echo "  --isolated          Run in isolated mode"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "*** QUANTUM-GETH MINING STARTUP ***"
echo "128-Puzzle Sequential Quantum Proof-of-Work with Bitcoin-Style Halving"
echo ""

# Check if blockchain exists
if [ ! -d "$DATADIR" ]; then
    echo "ERROR: No blockchain found in $DATADIR"
    echo ""
    echo "You need to initialize a blockchain first:"
    echo "   ./scripts/reset-blockchain.sh --difficulty 1 --force"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Data Directory: $DATADIR"
echo "  Mining Threads: $THREADS"
echo "  Network ID: $NETWORKID"
echo "  Etherbase: $ETHERBASE"
echo "  Verbosity: $VERBOSITY"
echo "  Quantum Solver: $QUANTUM_SOLVER"
echo "  Isolated Mode: $ISOLATED"
echo ""

echo "Quantum Mining Features:"
echo "  * 128 sequential quantum puzzles per block"
echo "  * 16 qubits x 20 T-gates per puzzle"
echo "  * Seed chaining: Seed_{i+1} = SHA256(Seed_i || Outcome_i)"
echo "  * Bitcoin-style halving: 50 QGC -> 25 QGC -> 12.5 QGC..."
echo "  * ASERT-Q difficulty targeting 12-second blocks"
echo "  * Mahadev->CAPSS->Nova proof generation"
echo "  * Dilithium-2 self-attestation"
echo ""

# Stop any existing geth processes
echo "Stopping any existing geth processes..."
if pkill -f "geth.*--mine" > /dev/null 2>&1; then
    echo "  Existing processes stopped"
else
    echo "  No existing processes found"
fi

# Build the geth command
GETH_ARGS=(
    "--datadir" "$DATADIR"
    "--networkid" "$NETWORKID"
    "--port" "30303"
    "--http"
    "--http.addr" "0.0.0.0"
    "--http.port" "8545"
    "--http.api" "eth,net,web3,personal,miner"
    "--http.corsdomain" "*"
    "--authrpc.addr" "0.0.0.0"
    "--authrpc.port" "8551"
    "--authrpc.vhosts" "*"
    "--ws"
    "--ws.addr" "0.0.0.0"
    "--ws.port" "8546"
    "--ws.api" "eth,net,web3"
    "--ws.origins" "*"
    "--miner.etherbase" "$ETHERBASE"
    "--mine"
    "--mine.threads" "$THREADS"
    "--verbosity" "$VERBOSITY"
)

# Add isolation parameters if requested
if [ "$ISOLATED" = true ]; then
    GETH_ARGS+=(
        "--nodiscover"
        "--maxpeers" "0"
    )
    echo "Running in isolated mode (no peer connections)"
fi

echo ""
echo "Expected Mining Behavior:"
echo "  • Sequential 128-puzzle execution with seed chaining"
echo "  • OutcomeRoot = MerkleRoot(Outcome_0...Outcome_127)"
echo "  • GateHash = SHA256(stream_0 || ... || stream_127)"
echo "  • Nova-Lite proof aggregation (3 proofs <=6kB each)"
echo "  • Dilithium signature binding prover to outcomes"
echo "  • ASERT-Q difficulty adjustment every block"
echo "  • Block rewards: 50 QGC + transaction fees"
echo ""

echo "Starting Quantum-Geth mining..."
echo "   Use Ctrl+C to stop mining"
echo ""

# Find the newest quantum-geth release or use development version
GETH_RELEASE_DIR=$(find releases -name "quantum-geth-*" -type d 2>/dev/null | sort -V | tail -1)

if [ -n "$GETH_RELEASE_DIR" ] && [ -f "$GETH_RELEASE_DIR/geth" ]; then
    echo "Using geth from release: $(basename $GETH_RELEASE_DIR)"
    GETH_PATH="$GETH_RELEASE_DIR/geth"
elif [ ! -f "geth" ]; then
    echo "Geth executable not found. Building release..."
    if ./build-linux.sh geth; then
        GETH_RELEASE_DIR=$(find releases -name "quantum-geth-*" -type d 2>/dev/null | sort -V | tail -1)
        echo "SUCCESS: Release built at $GETH_RELEASE_DIR"
        GETH_PATH="$GETH_RELEASE_DIR/geth"
    else
        echo "ERROR: Failed to build quantum-geth release"
        exit 1
    fi
else
    echo "Using development geth from root directory"
    GETH_PATH="./geth"
fi

# Execute geth with all arguments
exec "$GETH_PATH" "${GETH_ARGS[@]}" 