#!/bin/bash
# Q Coin Miner Starter
# Usage: ./start-miner.sh [type] [network] [options]
# Types: cpu, gpu (default: cpu)
# Networks: mainnet, testnet, devnet (default: testnet)

TYPE="cpu"
NETWORK="testnet"
THREADS=4
GETH_RPC="http://localhost:8545"
ETHERBASE=""
HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        cpu|gpu)
            TYPE="$1"
            shift
            ;;
        mainnet|testnet|devnet)
            NETWORK="$1"
            shift
            ;;
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
    echo -e "\033[1;36mQ Coin Miner Starter\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-miner.sh [type] [network] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mMining Types:\033[0m"
    echo "  cpu       - CPU Mining [DEFAULT]"
    echo "  gpu       - GPU Mining (CUDA required)"
    echo ""
    echo -e "\033[1;33mNetworks:\033[0m"
    echo "  mainnet   - Q Coin Mainnet (Chain ID 73236)"
    echo "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    echo "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --threads <n>        Number of mining threads (default: 4)"
    echo "  --geth-rpc <url>     Geth RPC endpoint (default: http://localhost:8545)"
    echo "  --etherbase <addr>   Mining reward address (auto-detected if empty)"
    echo "  --help               Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-miner.sh                       # Start CPU miner on testnet"
    echo "  ./start-miner.sh gpu                   # Start GPU miner on testnet"
    echo "  ./start-miner.sh cpu devnet            # Start CPU miner on devnet"
    echo "  ./start-miner.sh cpu testnet --threads 8 # Start CPU miner with 8 threads"
    exit 0
fi

# Build miner if it doesn't exist
MINER_PATH="quantum-miner/quantum-miner"
if [ ! -f "$MINER_PATH" ]; then
    echo -e "\033[1;33m🔨 Building Q Coin Miner...\033[0m"
    ./build-linux.sh miner
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Miner build failed!\033[0m"
        exit 1
    fi
fi

# Network configurations
case $NETWORK in
    mainnet)
        CHAINID=73236
        NAME="Q Coin Mainnet"
        DESCRIPTION="Production network with real Q Coin value"
        ;;
    testnet)
        CHAINID=73235
        NAME="Q Coin Testnet"
        DESCRIPTION="Testing network with test Q Coin"
        ;;
    devnet)
        CHAINID=73234
        NAME="Q Coin Dev Network"
        DESCRIPTION="Development network for testing"
        ;;
esac

echo -e "\033[1;36m⛏️  Starting $TYPE Mining on $NAME\033[0m"

# Get etherbase if not provided
if [ -z "$ETHERBASE" ]; then
    echo -e "\033[1;33m🔍 Auto-detecting mining address...\033[0m"
    
    # Try to get existing account
    RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
        "$GETH_RPC" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q '"result"'; then
        ETHERBASE=$(echo "$RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | head -1 | tr -d '"')
        
        if [ -n "$ETHERBASE" ]; then
            echo -e "\033[1;32m✅ Using account: $ETHERBASE\033[0m"
        else
            echo -e "\033[1;33m⚠️  No accounts found. Creating new account...\033[0m"
            CREATE_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
                --data '{"jsonrpc":"2.0","method":"personal_newAccount","params":[""],"id":1}' \
                "$GETH_RPC" 2>/dev/null)
            
            if [ $? -eq 0 ] && echo "$CREATE_RESPONSE" | grep -q '"result"'; then
                ETHERBASE=$(echo "$CREATE_RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | tr -d '"')
                echo -e "\033[1;32m✅ Created new account: $ETHERBASE\033[0m"
            else
                echo -e "\033[1;31m❌ Failed to create account!\033[0m"
                exit 1
            fi
        fi
    else
        echo -e "\033[1;31m❌ Failed to connect to Geth RPC at $GETH_RPC\033[0m"
        echo -e "\033[1;33m   Make sure Geth is running first!\033[0m"
        exit 1
    fi
fi

# Prepare miner arguments
MINER_ARGS=(
    "--rpc-url" "$GETH_RPC"
    "--etherbase" "$ETHERBASE"
    "--threads" "$THREADS"
)

if [ "$TYPE" = "gpu" ]; then
    MINER_ARGS+=("--gpu")
    echo -e "\033[1;33m🎮 GPU Mining enabled\033[0m"
else
    echo -e "\033[1;33m🖥️  CPU Mining enabled\033[0m"
fi

echo -e "\033[1;37m🌐 Network: $NAME\033[0m"
echo -e "\033[1;37m🔗 Chain ID: $CHAINID\033[0m"
echo -e "\033[1;37m📡 Geth RPC: $GETH_RPC\033[0m"
echo -e "\033[1;37m💰 Mining Address: $ETHERBASE\033[0m"
echo -e "\033[1;37m🧵 Threads: $THREADS\033[0m"
echo ""
echo -e "\033[1;32m🎯 Starting Q Coin $TYPE miner...\033[0m"

# Start miner
./"$MINER_PATH" "${MINER_ARGS[@]}" 