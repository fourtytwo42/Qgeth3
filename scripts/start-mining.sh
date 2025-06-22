#!/bin/bash

# Quantum-Geth Mining Startup Script
# Linux/macOS version of start-geth-mining.ps1

set -e  # Exit on any error

# Default configuration
DATADIR="qdata_quantum"
THREADS=1
NETWORKID=73428
ETHERBASE="0x8b61271473f14c80f2B1381Db9CB13b2d5306200"
VERBOSITY=4
QUANTUM_SOLVER="./quantum-geth/tools/solver/qiskit_solver.py"
ISOLATED=true
TESTMODE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --threads NUM       Set mining threads (default: 1)"
    echo "  --datadir DIR       Set data directory (default: qdata_quantum)"
    echo "  --networkid ID      Set network ID (default: 73428)"
    echo "  --etherbase ADDR    Set etherbase address"
    echo "  --verbosity LEVEL   Set logging verbosity 0-5 (default: 4)"
    echo "  --isolated          Run without peer connections (default)"
    echo "  --network           Enable peer connections"
    echo "  --testmode          Enable test mode with simplified verification"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --threads 4"
    echo "  $0 --verbosity 5 --network"
}

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
        --isolated)
            ISOLATED=true
            shift
            ;;
        --network)
            ISOLATED=false
            shift
            ;;
        --testmode)
            TESTMODE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate numeric parameters
if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [ "$THREADS" -lt 1 ]; then
    echo -e "${RED}Error: Threads must be a positive integer${NC}"
    exit 1
fi

if ! [[ "$VERBOSITY" =~ ^[0-5]$ ]]; then
    echo -e "${RED}Error: Verbosity must be between 0 and 5${NC}"
    exit 1
fi

echo "*** QUANTUM-GETH MINING STARTUP ***"
echo "Bitcoin-Style Nonce-Level Difficulty Implementation"
echo -e "${GREEN}Successfully fixed quality calculation and comparison logic!${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Data Directory: $DATADIR"
echo "  Mining Threads: $THREADS"
echo "  Network ID: $NETWORKID"
echo "  Etherbase: $ETHERBASE"
echo "  Verbosity: $VERBOSITY"
echo "  Quantum Solver: $QUANTUM_SOLVER"
echo "  Isolated Mode: $ISOLATED"
if [ "$TESTMODE" = true ]; then
    echo -e "  ${YELLOW}Test Mode: ENABLED${NC}"
fi
echo ""

# Check if blockchain is initialized
if [ ! -d "$DATADIR" ]; then
    echo -e "${RED}Error: Blockchain not initialized. Please run:${NC}"
    echo "  ./scripts/reset-blockchain.sh --difficulty 100"
    exit 1
fi

# Check for quantum solver
if [ ! -f "$QUANTUM_SOLVER" ]; then
    echo -e "${YELLOW}Warning: Quantum solver not found at $QUANTUM_SOLVER${NC}"
    echo "  Mining will use fallback classical simulation"
fi

# Stop any existing geth processes
echo "Stopping any existing geth processes..."
if pgrep -f "geth" > /dev/null; then
    pkill -f "geth" || true
    echo -e "  ${GREEN}Existing processes stopped${NC}"
else
    echo "  No existing processes found"
fi

# Wait a moment for processes to fully stop
sleep 2

# Build mining command
GETH_BIN=""
if [ -f "./quantum-geth/build/bin/geth" ]; then
    GETH_BIN="./quantum-geth/build/bin/geth"
elif [ -f "./geth" ]; then
    GETH_BIN="./geth"
elif [ -f "./geth.exe" ]; then
    GETH_BIN="./geth.exe"
else
    echo -e "${RED}Error: geth binary not found${NC}"
    echo "Please compile geth first with: cd quantum-geth && make geth"
    exit 1
fi

# Base command arguments
GETH_ARGS=(
    --datadir "$DATADIR"
    --networkid "$NETWORKID"
    --miner.etherbase "$ETHERBASE"
    --miner.threads "$THREADS"
    --mine
    --verbosity "$VERBOSITY"
    --quantum-solver-path "$QUANTUM_SOLVER"
)

# Add isolation or networking options
if [ "$ISOLATED" = true ]; then
    GETH_ARGS+=(--maxpeers 0 --nodiscover)
    echo "Running in isolated mode (no peer connections)"
else
    echo "Running with peer connections enabled"
fi

# Add test mode if enabled
if [ "$TESTMODE" = true ]; then
    GETH_ARGS+=(--quantum-testmode)
fi

echo ""
echo -e "${CYAN}Expected Mining Behavior:${NC}"
echo "  * Bitcoin-style nonce progression: qnonce=0,1,2,3,4..."
echo "  * Quality must be less than Target for success (lower quality = better)"
echo "  * Positive quality values (no more negative numbers)"
echo "  * Multiple attempts required for higher difficulty"
echo ""
echo -e "${YELLOW}Starting Quantum-Geth mining...${NC}"
echo "   Use Ctrl+C to stop mining"
echo ""

# Execute geth with all arguments
exec "$GETH_BIN" "${GETH_ARGS[@]}" 