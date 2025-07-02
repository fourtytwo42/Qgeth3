#!/bin/bash
# Q Coin Smart Miner - Simplified Architecture
# CPU or GPU mode only - NO fallback behavior
# Usage: ./start-miner.sh [options]

THREADS=0  # 0 = auto-detect
GETH_RPC="http://localhost:8545"
ETHERBASE=""
FORCE_CPU=false
FORCE_GPU=false
HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --geth-rpc)
            GETH_RPC="$2"
            shift 2
            ;;
        --etherbase)
            ETHERBASE="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=true
            shift
            ;;
        --gpu)
            FORCE_GPU=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo -e "\033[1;36m[TARGET] Q Coin Smart Miner - Simplified Architecture\033[0m"
    echo ""
    echo -e "\033[1;32mCPU or GPU mode only - NO fallback behavior\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-miner.sh [options]\033[0m"
    echo ""
    echo -e "\033[1;33mMining Modes:\033[0m"
    echo "  --cpu                Force CPU mining only"
    echo "  --gpu                Force GPU mining (native Qiskit on Linux)"
    echo "  (default)            Auto-detect optimal mode"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --threads <n>        Number of mining threads (0 = auto-detect)"
    echo "  --geth-rpc <url>     Geth RPC endpoint (default: http://localhost:8545)"
    echo "  --etherbase <addr>   Mining reward address (auto-detected if empty)"
    echo "  --help               Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-miner.sh                    # Auto-detect optimal mode"
    echo "  ./start-miner.sh --cpu --threads 8  # CPU mode, 8 threads"
    echo "  ./start-miner.sh --gpu --threads 32 # GPU mode, 32 threads"
    echo ""
    echo -e "\033[1;35mPerformance Expectations:\033[0m"
    echo "  CPU Mode (8 threads):  ~400-800 PZ/s"
    echo "  GPU Mode (32 threads): ~8000-15000+ PZ/s"
    exit 0
fi

echo -e "\033[1;36m[TARGET] Q Coin Smart Miner Starting...\033[0m"
echo -e "\033[1;33m[CONFIG] Auto-detecting optimal mining configuration...\033[0m"

# Build miner if it doesn't exist
MINER_PATH="../../quantum-miner"
if [ ! -f "$MINER_PATH" ]; then
    echo -e "\033[1;33m[BUILD] Building Q Coin Miner...\033[0m"
    ./build-linux.sh miner
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m[ERROR] Miner build failed!\033[0m"
        exit 1
    fi
fi

# Test Geth connection and get network info
echo -e "\033[1;33m[NETWORK] Connecting to Geth at $GETH_RPC...\033[0m"

CHAIN_ID_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
    "$GETH_RPC" 2>/dev/null)

if [ $? -eq 0 ] && echo "$CHAIN_ID_RESPONSE" | grep -q '"result"'; then
    CHAIN_ID_HEX=$(echo "$CHAIN_ID_RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | tr -d '"')
    CHAIN_ID=$((CHAIN_ID_HEX))
    
    # Determine network from chain ID
    case $CHAIN_ID in
        73234)
            NETWORK_NAME="Q Coin Dev Network"
            NETWORK_COLOR="\033[1;35m"  # Magenta
            ;;
        73235)
            NETWORK_NAME="Q Coin Testnet"
            NETWORK_COLOR="\033[1;36m"  # Cyan
            ;;
        73236)
            NETWORK_NAME="Q Coin Mainnet"
            NETWORK_COLOR="\033[1;32m"  # Green
            ;;
        73237)
            NETWORK_NAME="Q Coin Planck Network"
            NETWORK_COLOR="\033[1;34m"  # Blue
            ;;
        *)
            NETWORK_NAME="Unknown Q Coin Network (Chain ID: $CHAIN_ID)"
            NETWORK_COLOR="\033[1;33m"  # Yellow
            ;;
    esac
    
    echo -e "\033[1;32m[SUCCESS] Connected to ${NETWORK_COLOR}${NETWORK_NAME}\033[0m"
else
    echo -e "\033[1;31m[ERROR] Failed to connect to Geth RPC at $GETH_RPC\033[0m"
    echo -e "\033[1;33m[INFO] Make sure Geth is running first!\033[0m"
    echo -e "\033[1;36m[INFO] Try: ./start-geth.sh\033[0m"
    exit 1
fi

# Get or create etherbase address
if [ -z "$ETHERBASE" ]; then
    echo -e "\033[1;33m[CONFIG] Auto-detecting mining address...\033[0m"
    
    # Try to get existing account
    RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
        "$GETH_RPC" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q '"result"'; then
        ETHERBASE=$(echo "$RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | head -1 | tr -d '"')
        
        if [ -n "$ETHERBASE" ]; then
            echo -e "\033[1;32m[SUCCESS] Using existing account: $ETHERBASE\033[0m"
        else
            echo -e "\033[1;33m[WARNING] No accounts found. Creating new account...\033[0m"
            CREATE_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
                --data '{"jsonrpc":"2.0","method":"personal_newAccount","params":[""],"id":1}' \
                "$GETH_RPC" 2>/dev/null)
            
            if [ $? -eq 0 ] && echo "$CREATE_RESPONSE" | grep -q '"result"'; then
                ETHERBASE=$(echo "$CREATE_RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | tr -d '"')
                echo -e "\033[1;32m[SUCCESS] Created new account: $ETHERBASE\033[0m"
            else
                echo -e "\033[1;31m[ERROR] Failed to create account!\033[0m"
                exit 1
            fi
        fi
    else
        echo -e "\033[1;31m[ERROR] Failed to get/create mining address!\033[0m"
        exit 1
    fi
fi

# Determine mining mode - NO fallback behavior
USE_GPU=false
MINING_MODE="CPU"
MINING_COLOR="\033[1;33m"  # Yellow

if [ "$FORCE_CPU" = true ] && [ "$FORCE_GPU" = true ]; then
    echo -e "\033[1;31m[ERROR] Cannot specify both --cpu and --gpu flags!\033[0m"
    exit 1
fi

if [ "$FORCE_CPU" = true ]; then
    echo -e "\033[1;33m[INFO] CPU mining forced by user\033[0m"
    USE_GPU=false
    MINING_MODE="CPU"
elif [ "$FORCE_GPU" = true ]; then
    echo -e "\033[1;33m[INFO] GPU mining forced by user\033[0m"
    USE_GPU=true
    MINING_MODE="GPU"
    MINING_COLOR="\033[1;32m"  # Green
else
    # Auto-detect mode (default to GPU for better performance)
    echo -e "\033[1;33m[CONFIG] Auto-detecting optimal mining mode...\033[0m"
    USE_GPU=true
    MINING_MODE="GPU"
    MINING_COLOR="\033[1;32m"  # Green
    echo -e "\033[1;32m[INFO] Auto-selected GPU mode for optimal performance\033[0m"
fi

# Auto-detect thread count
if [ "$THREADS" -eq 0 ]; then
    if [ "$USE_GPU" = true ]; then
        THREADS=32  # GPU typically uses more threads
        echo -e "\033[1;32m[CONFIG] Auto-detected GPU threads: $THREADS\033[0m"
    else
        THREADS=$(nproc)
        echo -e "\033[1;32m[CONFIG] Auto-detected CPU threads: $THREADS\033[0m"
    fi
fi

# Prepare miner arguments - using our simplified flags
MINER_ARGS=(
    "-coinbase" "$ETHERBASE"
    "-node" "$GETH_RPC"
    "-threads" "$THREADS"
)

if [ "$USE_GPU" = true ]; then
    # Use -gpu flag for GPU mode (native Qiskit on Linux)
    MINER_ARGS+=("-gpu")
else
    # Use -cpu flag to force CPU mode
    MINER_ARGS+=("-cpu")
fi

# Display configuration
echo ""
echo -e "\033[1;37m[CONFIG] Mining Configuration:\033[0m"
echo -e "\033[1;37m[NETWORK] Network: ${NETWORK_COLOR}${NETWORK_NAME}\033[0m"
echo -e "\033[1;37m[NETWORK] Chain ID: $CHAIN_ID\033[0m"
echo -e "\033[1;37m[MODE] Mining Mode: ${MINING_COLOR}${MINING_MODE}\033[0m"
echo -e "\033[1;37m[RPC] Geth RPC: $GETH_RPC\033[0m"
echo -e "\033[1;37m[WALLET] Mining Address: $ETHERBASE\033[0m"
echo -e "\033[1;37m[THREADS] Threads: $THREADS\033[0m"
if [ "$USE_GPU" = true ]; then
    echo -e "\033[1;37m[GPU] Linux Native Qiskit GPU acceleration\033[0m"
    echo -e "\033[1;37m[PERFORMANCE] Expected: ~8000-15000+ PZ/s\033[0m"
else
    echo -e "\033[1;37m[CPU] CPU processing only\033[0m"
    echo -e "\033[1;37m[PERFORMANCE] Expected: ~400-800 PZ/s\033[0m"
fi
echo ""
echo -e "\033[1;32m[START] Starting Q Coin miner...\033[0m"

# Start miner - let it handle GPU/CPU mode internally with no fallback
./"$MINER_PATH" "${MINER_ARGS[@]}" 