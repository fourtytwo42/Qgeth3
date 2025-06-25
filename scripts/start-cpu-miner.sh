#!/bin/bash

# Q Coin CPU Miner
# Starts the Q Coin CPU miner for mining Q Coins
# Usage: ./start-cpu-miner.sh [options]

# Default values
ADDRESS=""
RPCURL="http://127.0.0.1:8545"
THREADS=0
TESTNET=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --address)
            ADDRESS="$2"
            shift 2
            ;;
        --rpcurl)
            RPCURL="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --testnet)
            TESTNET=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Show help
if [ "$HELP" = true ] || [ -z "$ADDRESS" ]; then
    echo -e "\033[36mQ Coin CPU Miner\033[0m"
    echo ""
    echo -e "\033[33mUsage: ./start-cpu-miner.sh --address <mining_address> [options]\033[0m"
    echo ""
    echo -e "\033[32mRequired:\033[0m"
    echo "  --address <addr>     Your Q Coin address to receive mining rewards"
    echo ""
    echo -e "\033[32mOptions:\033[0m"
    echo "  --rpcurl <url>       RPC URL (default: http://127.0.0.1:8545)"
    echo "  --threads <num>      Number of CPU threads (default: auto-detect)"
    echo "  --testnet            Connect to testnet instead of mainnet"
    echo "  --help, -h           Show this help message"
    echo ""
    echo -e "\033[33mExamples:\033[0m"
    echo "  ./start-cpu-miner.sh --address 0x123..."
    echo "  ./start-cpu-miner.sh --address 0x123... --threads 4"
    echo "  ./start-cpu-miner.sh --address 0x123... --testnet"
    echo ""
    echo -e "\033[35mQ Coin Mining Details:\033[0m"
    echo "  Algorithm: QMPoW (Quantum Proof of Work)"
    echo "  Block Time: 12 seconds"
    echo "  Difficulty: ASERT-Q (Adaptive)"
    echo ""
    if [ -z "$ADDRESS" ]; then
        echo -e "\033[31mERROR: Mining address is required!\033[0m"
        echo -e "\033[33mGet a Q Coin address from your wallet or geth console\033[0m"
    fi
    exit 0
fi

# Find the latest quantum-miner release
echo -e "\033[36mQ Coin CPU Miner - Starting\033[0m"
echo ""

MINER_RELEASE_DIR=$(find ../releases -name "quantum-miner-*" -type d 2>/dev/null | sort -r | head -n 1)
if [ -z "$MINER_RELEASE_DIR" ]; then
    echo -e "\033[31mERROR: No quantum-miner release found!\033[0m"
    echo -e "\033[33mPlease run: ./build-release.sh miner\033[0m"
    exit 1
fi

MINER_EXECUTABLE="$MINER_RELEASE_DIR/quantum-miner"
if [ ! -f "$MINER_EXECUTABLE" ]; then
    echo -e "\033[31mERROR: Miner executable not found in release directory!\033[0m"
    exit 1
fi

echo -e "\033[32mUsing miner from: $(basename "$MINER_RELEASE_DIR")\033[0m"

# Determine network
if [ "$TESTNET" = true ]; then
    NETWORK_NAME="Q Coin Testnet"
    CHAIN_ID="73235"
else
    NETWORK_NAME="Q Coin Mainnet"
    CHAIN_ID="73236"
fi

# Auto-detect threads if not specified
if [ "$THREADS" -eq 0 ]; then
    THREADS=$(nproc 2>/dev/null || echo "4")
    echo -e "\033[32mAuto-detected CPU threads: $THREADS\033[0m"
else
    echo -e "\033[32mUsing $THREADS CPU threads\033[0m"
fi

# Build miner command arguments
MINER_ARGS=(
    -rpc-url "$RPCURL"
    -address "$ADDRESS"
    -cpu-threads "$THREADS"
)

# Display startup information
echo ""
echo -e "\033[36m$NETWORK_NAME CPU Mining Configuration:\033[0m"
echo -e "\033[37m  Mining Address: $ADDRESS\033[0m"
echo -e "\033[37m  RPC URL: $RPCURL\033[0m"
echo -e "\033[37m  CPU Threads: $THREADS\033[0m"
echo -e "\033[37m  Algorithm: QMPoW (Quantum Proof of Work)\033[0m"
echo ""
echo -e "\033[32mStarting CPU miner...\033[0m"
echo -e "\033[33mPress Ctrl+C to stop\033[0m"
echo ""

# Start miner
exec "$MINER_EXECUTABLE" "${MINER_ARGS[@]}" 