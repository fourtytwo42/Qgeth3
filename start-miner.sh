#!/bin/bash
# Q Coin Smart Miner
# Auto-detects network from running Geth node
# Auto-detects GPU capability and falls back to CPU
# Usage: ./start-miner.sh [options]

THREADS=0  # 0 = auto-detect
GETH_RPC="http://localhost:8545"
ETHERBASE=""
FORCE_CPU=false
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
        --force-cpu)
            FORCE_CPU=true
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
    echo -e "\033[1;36mðŸŽ¯ Q Coin Smart Miner\033[0m"
    echo ""
    echo -e "\033[1;32mAuto-detects network from running Geth node\033[0m"
    echo -e "\033[1;32mAuto-detects GPU capability and falls back to CPU\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-miner.sh [options]\033[0m"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --threads <n>        Number of mining threads (0 = auto-detect)"
    echo "  --geth-rpc <url>     Geth RPC endpoint (default: http://localhost:8545)"
    echo "  --etherbase <addr>   Mining reward address (auto-detected if empty)"
    echo "  --force-cpu          Force CPU mining even if GPU available"
    echo "  --help               Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-miner.sh                    # Smart auto-detection"
    echo "  ./start-miner.sh --threads 32       # 32 threads"
    echo "  ./start-miner.sh --force-cpu        # Force CPU mining"
    exit 0
fi

echo -e "\033[1;36mðŸŽ¯ Q Coin Smart Miner Starting...\033[0m"
echo -e "\033[1;33mðŸ” Auto-detecting optimal mining configuration...\033[0m"

# Build miner if it doesn't exist
MINER_PATH="quantum-miner/quantum-miner"
if [ ! -f "$MINER_PATH" ]; then
    echo -e "\033[1;33mðŸ”¨ Building Q Coin Miner...\033[0m"
    ./build-linux.sh miner
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31mâŒ Miner build failed!\033[0m"
        exit 1
    fi
fi

# Test Geth connection and get network info
echo -e "\033[1;33mðŸ“¡ Connecting to Geth at $GETH_RPC...\033[0m"

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
        *)
            NETWORK_NAME="Unknown Q Coin Network (Chain ID: $CHAIN_ID)"
            NETWORK_COLOR="\033[1;33m"  # Yellow
            ;;
    esac
    
    echo -e "\033[1;32mâœ… Connected to ${NETWORK_COLOR}${NETWORK_NAME}\033[0m"
else
    echo -e "\033[1;31mâŒ Failed to connect to Geth RPC at $GETH_RPC\033[0m"
    echo -e "\033[1;33m   Make sure Geth is running first!\033[0m"
    echo -e "\033[1;36m   Try: ./qcoin-geth.sh\033[0m"
    exit 1
fi

# Get or create etherbase address
if [ -z "$ETHERBASE" ]; then
    echo -e "\033[1;33mðŸ” Auto-detecting mining address...\033[0m"
    
    # Try to get existing account
    RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
        "$GETH_RPC" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q '"result"'; then
        ETHERBASE=$(echo "$RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | head -1 | tr -d '"')
        
        if [ -n "$ETHERBASE" ]; then
            echo -e "\033[1;32mâœ… Using existing account: $ETHERBASE\033[0m"
        else
            echo -e "\033[1;33mâš ï¸  No accounts found. Creating new account...\033[0m"
            CREATE_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
                --data '{"jsonrpc":"2.0","method":"personal_newAccount","params":[""],"id":1}' \
                "$GETH_RPC" 2>/dev/null)
            
            if [ $? -eq 0 ] && echo "$CREATE_RESPONSE" | grep -q '"result"'; then
                ETHERBASE=$(echo "$CREATE_RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | tr -d '"')
                echo -e "\033[1;32mâœ… Created new account: $ETHERBASE\033[0m"
            else
                echo -e "\033[1;31mâŒ Failed to create account!\033[0m"
                exit 1
            fi
        fi
    else
        echo -e "\033[1;31mâŒ Failed to get/create mining address!\033[0m"
        exit 1
    fi
fi

# Auto-detect mining mode (GPU vs CPU)
USE_GPU=false
MINING_MODE="CPU"
MINING_COLOR="\033[1;33m"  # Yellow

if [ "$FORCE_CPU" != true ]; then
    echo -e "\033[1;33mðŸŽ® Testing GPU mining capability...\033[0m"
    
    # Test GPU mining by running a quick check
    GPU_TEST_OUTPUT=$(./"$MINER_PATH" -gpu -node "$GETH_RPC" -coinbase "$ETHERBASE" -threads 1 -help 2>&1)
    GPU_TEST_EXIT=$?
    
    if [ $GPU_TEST_EXIT -eq 0 ]; then
        USE_GPU=true
        MINING_MODE="GPU"
        MINING_COLOR="\033[1;32m"  # Green
        echo -e "\033[1;32mâœ… GPU mining available - Using GPU mode\033[0m"
    else
        echo -e "\033[1;33mâš ï¸  GPU mining not available - Falling back to CPU\033[0m"
    fi
else
    echo -e "\033[1;33mðŸ–¥ï¸  CPU mining forced by user\033[0m"
fi

# Auto-detect thread count
if [ "$THREADS" -eq 0 ]; then
    if [ "$USE_GPU" = true ]; then
        THREADS=1  # GPU typically uses 1 thread
        echo -e "\033[1;32mðŸ§µ Auto-detected GPU threads: $THREADS\033[0m"
    else
        THREADS=$(nproc)
        echo -e "\033[1;32mðŸ§µ Auto-detected CPU threads: $THREADS\033[0m"
    fi
fi

# Prepare miner arguments
MINER_ARGS=(
    "-node" "$GETH_RPC"
    "-coinbase" "$ETHERBASE"
    "-threads" "$THREADS"
)

if [ "$USE_GPU" = true ]; then
    MINER_ARGS+=("-gpu")
fi

# Display configuration
echo ""
echo -e "\033[1;37mâš¡ Mining Configuration:\033[0m"
echo -e "\033[1;37mðŸŒ Network: ${NETWORK_COLOR}${NETWORK_NAME}\033[0m"
echo -e "\033[1;37mðŸ”— Chain ID: $CHAIN_ID\033[0m"
echo -e "\033[1;37mâ›ï¸  Mining Mode: ${MINING_COLOR}${MINING_MODE}\033[0m"
echo -e "\033[1;37mðŸ“¡ Geth RPC: $GETH_RPC\033[0m"
echo -e "\033[1;37mðŸ’° Mining Address: $ETHERBASE\033[0m"
echo -e "\033[1;37mðŸ§µ Threads: $THREADS\033[0m"
echo ""
echo -e "\033[1;32mðŸš€ Starting Q Coin miner...\033[0m"

# Start miner
./"$MINER_PATH" "${MINER_ARGS[@]}" 