#!/bin/bash

# Q Coin GPU Miner
# Starts the Q Coin GPU miner for mining Q Coins with CUDA
# Usage: ./start-gpu-miner.sh [options]

# Default values
ADDRESS=""
RPCURL="http://127.0.0.1:8545"
GPUS=0
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
        --gpus)
            GPUS="$2"
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
    echo -e "\033[36mQ Coin GPU Miner\033[0m"
    echo ""
    echo -e "\033[33mUsage: ./start-gpu-miner.sh --address <mining_address> [options]\033[0m"
    echo ""
    echo -e "\033[32mRequired:\033[0m"
    echo "  --address <addr>     Your Q Coin address to receive mining rewards"
    echo ""
    echo -e "\033[32mOptions:\033[0m"
    echo "  --rpcurl <url>       RPC URL (default: http://127.0.0.1:8545)"
    echo "  --gpus <num>         Number of GPUs to use (default: auto-detect)"
    echo "  --testnet            Connect to testnet instead of mainnet"
    echo "  --help, -h           Show this help message"
    echo ""
    echo -e "\033[33mExamples:\033[0m"
    echo "  ./start-gpu-miner.sh --address 0x123..."
    echo "  ./start-gpu-miner.sh --address 0x123... --gpus 2"
    echo "  ./start-gpu-miner.sh --address 0x123... --testnet"
    echo ""
    echo -e "\033[35mQ Coin GPU Mining Details:\033[0m"
    echo "  Algorithm: QMPoW (Quantum Proof of Work)"
    echo "  GPU Support: NVIDIA CUDA"
    echo "  Block Time: 12 seconds"
    echo "  Difficulty: ASERT-Q (Adaptive)"
    echo ""
    echo -e "\033[31mRequirements:\033[0m"
    echo "  - NVIDIA GPU with CUDA support"
    echo "  - CUDA drivers installed"
    echo ""
    if [ -z "$ADDRESS" ]; then
        echo -e "\033[31mERROR: Mining address is required!\033[0m"
        echo -e "\033[33mGet a Q Coin address from your wallet or geth console\033[0m"
    fi
    exit 0
fi

# Find the latest quantum-miner release
echo -e "\033[36mQ Coin GPU Miner - Starting\033[0m"
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

# Auto-detect GPUs if not specified
if [ "$GPUS" -eq 0 ]; then
    # Try to detect NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
        if [ -n "$DETECTED_GPUS" ] && [ "$DETECTED_GPUS" -gt 0 ]; then
            GPUS=$DETECTED_GPUS
            echo -e "\033[32mAuto-detected NVIDIA GPUs: $GPUS\033[0m"
        else
            GPUS=1
            echo -e "\033[33mCould not detect GPUs, defaulting to 1\033[0m"
        fi
    else
        GPUS=1
        echo -e "\033[33mCould not detect GPUs (nvidia-smi not found), defaulting to 1\033[0m"
    fi
else
    echo -e "\033[32mUsing $GPUS GPUs\033[0m"
fi

# Build miner command arguments
MINER_ARGS=(
    -rpc-url "$RPCURL"
    -address "$ADDRESS"
    -cuda-devices "$GPUS"
)

# Display startup information
echo ""
echo -e "\033[36m$NETWORK_NAME GPU Mining Configuration:\033[0m"
echo -e "\033[37m  Mining Address: $ADDRESS\033[0m"
echo -e "\033[37m  RPC URL: $RPCURL\033[0m"
echo -e "\033[37m  CUDA GPUs: $GPUS\033[0m"
echo -e "\033[37m  Algorithm: QMPoW (Quantum Proof of Work)\033[0m"
echo ""
echo -e "\033[32mStarting GPU miner...\033[0m"
echo -e "\033[33mPress Ctrl+C to stop\033[0m"
echo ""

# Start miner
exec "$MINER_EXECUTABLE" "${MINER_ARGS[@]}" 