#!/bin/bash
# Quantum-Geth CPU Miner

# Default parameters
COINBASE=""
NODE_URL="http://localhost:8545"
THREADS=1
LOG=false

# Function to show help
show_help() {
    echo "Quantum-Geth CPU Miner"
    echo ""
    echo "Description:"
    echo "  High-performance quantum mining with CPU-based simulation"
    echo ""
    echo "Usage:"
    echo "  $0 --coinbase <address> [options]"
    echo ""
    echo "Parameters:"
    echo "  --coinbase <address>   Coinbase address for mining rewards (required)"
    echo "  --node <url>           Quantum-Geth node URL (default: http://localhost:8545)"
    echo "  --threads <number>     Number of CPU mining threads (default: 1)"
    echo "  --log                  Enable detailed logging to quantum-miner.log file"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --threads 4"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --node http://192.168.1.100:8545"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --log  # Enable logging to file"
    echo ""
    echo "Features:"
    echo "  * CPU quantum simulation (0.36 puzzles/sec)"
    echo "  * Real quantum circuit simulation (16-qubit, 20 T-gates)"
    echo "  * Multi-threaded mining support"
    echo "  * No additional dependencies required"
    echo ""
    echo "Performance Comparison:"
    echo "  CPU Mining: 0.36 puzzles/sec (this miner)"
    echo "  GPU Mining: 0.45 puzzles/sec (use ./run-gpu-miner.sh)"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coinbase)
            COINBASE="$2"
            shift 2
            ;;
        --node)
            NODE_URL="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --log)
            LOG=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Quantum-Geth CPU Miner"
echo "CPU-based quantum circuit mining"

if [ -z "$COINBASE" ]; then
    echo "ERROR: Coinbase address required!"
    echo ""
    echo "Usage examples:"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --threads 4"
    echo "  $0 --help"
    echo ""
    echo "Tip: For better performance, try GPU mining with ./run-gpu-miner.sh"
    exit 1
fi

# Validate coinbase address format
if [[ ! "$COINBASE" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: Invalid coinbase address format!"
    echo "Expected format: 0x followed by 40 hex characters"
    exit 1
fi

# Find the newest quantum-miner release or use development version
RELEASE_DIR=$(find releases -name "quantum-miner-*" -type d 2>/dev/null | sort -V | tail -1)

if [ -n "$RELEASE_DIR" ] && [ -f "$RELEASE_DIR/quantum-miner" ]; then
    MINER_EXECUTABLE="$RELEASE_DIR/quantum-miner"
    WORKING_DIR="$RELEASE_DIR"
else
    MINER_EXECUTABLE="quantum-miner/quantum-miner"
    WORKING_DIR="quantum-miner"
fi

if [ ! -f "$MINER_EXECUTABLE" ]; then
    echo "CPU miner executable not found. Building release..."
    echo ""
    
    if ./scripts/build-release.sh miner; then
        # Re-find the newest release
        RELEASE_DIR=$(find releases -name "quantum-miner-*" -type d 2>/dev/null | sort -V | tail -1)
        if [ -n "$RELEASE_DIR" ] && [ -f "$RELEASE_DIR/quantum-miner" ]; then
            MINER_EXECUTABLE="$RELEASE_DIR/quantum-miner"
            WORKING_DIR="$RELEASE_DIR"
            echo "SUCCESS: Release built at $RELEASE_DIR"
        else
            echo "ERROR: Failed to create release"
            exit 1
        fi
    else
        echo "ERROR: Failed to build quantum-miner release"
        echo ""
        echo "Manual build options:"
        echo "  ./scripts/build-release.sh miner"
        echo "  cd quantum-miner && go build -o quantum-miner ."
        exit 1
    fi
fi

echo ""
echo "CPU Mining Configuration:"
echo "   Coinbase: $COINBASE"
echo "   Node URL: $NODE_URL"
echo "   CPU Threads: $THREADS"
echo "   Quantum Puzzles: 128 per block"
echo "   Circuit Size: 16 qubits, 20 T-gates"
echo ""

# Build miner arguments
MINER_ARGS=(-coinbase "$COINBASE" -node "$NODE_URL" -threads "$THREADS")
if [ "$LOG" = true ]; then
    MINER_ARGS+=(-log)
fi

echo "Starting CPU quantum miner..."
echo ""

# Change to working directory and run miner
cd "$WORKING_DIR" || exit 1

if ! ./quantum-miner "${MINER_ARGS[@]}"; then
    echo "ERROR: Failed to start CPU miner"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check quantum-geth node is running at $NODE_URL"
    echo "  2. Verify coinbase address format"
    echo "  3. Ensure quantum-miner is built and accessible"
    echo "  4. Consider GPU mining for better performance: ./scripts/run-gpu-miner.sh"
    exit 1
fi 