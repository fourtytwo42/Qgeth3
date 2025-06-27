#!/bin/bash
# Q Coin Geth Node Starter
# Usage: ./start-geth.sh [network] [options]
# Networks: mainnet, testnet, devnet (default: testnet)
# Options: --mining (enable mining with single thread)

NETWORK="testnet"
MINING=false
HELP=false
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        mainnet|testnet|devnet)
            NETWORK="$1"
            shift
            ;;
        --mining)
            MINING=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        --*)
            # Collect all other -- arguments as extra args for geth
            EXTRA_ARGS+=("$1")
            shift
            # If this argument has a value, collect it too
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo -e "\033[1;36mQ Coin Geth Node Starter\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-geth.sh [network] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mNetworks:\033[0m"
    echo "  mainnet   - Q Coin Mainnet (Chain ID 73236)"
    echo "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    echo "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --mining  - Enable mining with single thread"
    echo "  --help    - Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-geth.sh                  # Start testnet node"
    echo "  ./start-geth.sh mainnet          # Start mainnet node"
    echo "  ./start-geth.sh devnet --mining  # Start dev node with mining"
    exit 0
fi

# Check for geth binary
if [ ! -f "./geth" ]; then
    echo -e "\033[1;33m🔨 Q Coin Geth binary not found. Building...\033[0m"
    if [ ! -f "./build-linux.sh" ]; then
        echo -e "\033[1;31m❌ build-linux.sh not found! Are you in the correct directory?\033[0m"
        exit 1
    fi
    ./build-linux.sh geth
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Build failed!\033[0m"
        exit 1
    fi
fi

# Make sure geth is executable
if [ ! -x "./geth" ]; then
    chmod +x "./geth"
    echo -e "\033[1;32m✅ Made geth executable\033[0m"
fi

# Network configurations
case $NETWORK in
    mainnet)
        CHAINID=73236
        DATADIR="$HOME/.qcoin/mainnet"
        GENESIS="genesis_quantum_mainnet.json"
        PORT=30303
        NAME="Q Coin Mainnet"
        BOOTNODE_PORT=30303
        ;;
    testnet)
        CHAINID=73235
        DATADIR="$HOME/.qcoin/testnet"
        GENESIS="genesis_quantum_testnet.json"
        PORT=30303
        NAME="Q Coin Testnet"
        BOOTNODE_PORT=30303
        ;;
    devnet)
        CHAINID=73234
        DATADIR="$HOME/.qcoin/devnet"
        GENESIS="genesis_quantum_dev.json"
        PORT=30305
        NAME="Q Coin Dev Network"
        BOOTNODE_PORT=30305
        ;;
esac

# Detect external IP for proper enode advertising
EXTERNAL_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || curl -s https://api.ipify.org 2>/dev/null || curl -s https://checkip.amazonaws.com 2>/dev/null)
if [ -z "$EXTERNAL_IP" ]; then
    echo -e "\033[1;33m⚠️  Could not detect external IP, using NAT discovery\033[0m"
    NAT_CONFIG="any"
else
    echo -e "\033[1;32m🌍 Detected external IP: $EXTERNAL_IP\033[0m"
    NAT_CONFIG="extip:$EXTERNAL_IP"
fi

# Bootnodes are automatically selected based on network ID (chainid)
# No need to specify --bootnodes flag - geth will use params/bootnodes_qcoin.go

echo -e "\033[1;36m🚀 Starting $NAME (Chain ID: $CHAINID)\033[0m"

# Create data directory if it doesn't exist
if [ ! -d "$DATADIR" ]; then
    mkdir -p "$DATADIR"
    echo -e "\033[1;32m📁 Created data directory: $DATADIR\033[0m"
fi

# Initialize with genesis if needed
if [ ! -d "$DATADIR/geth/chaindata" ]; then
    echo -e "\033[1;33m🔧 Initializing blockchain with genesis file...\033[0m"
    
    # Check if genesis file exists
    if [ ! -f "$GENESIS" ]; then
        echo -e "\033[1;31m❌ Genesis file not found: $GENESIS\033[0m"
        echo -e "\033[1;33m📋 Available genesis files:\033[0m"
        ls -la genesis_quantum_*.json 2>/dev/null || echo "No genesis files found!"
        exit 1
    fi
    
    # Initialize with correct argument order
    ./geth --datadir "$DATADIR" init "$GENESIS"
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Genesis initialization failed!\033[0m"
        echo -e "\033[1;33m💡 Debug info:\033[0m"
        echo "  Genesis file: $GENESIS"
        echo "  Data directory: $DATADIR"
        echo "  Command: ./geth --datadir \"$DATADIR\" init \"$GENESIS\""
        exit 1
    fi
    echo -e "\033[1;32m✅ Blockchain initialized successfully\033[0m"
fi

# Prepare geth arguments
GETH_ARGS=(
    "--datadir" "$DATADIR"
    "--networkid" "$CHAINID"
    "--port" "$PORT"
    "--nat" "$NAT_CONFIG"
    "--http"
    "--http.addr" "0.0.0.0"
    "--http.port" "8545"
    "--http.corsdomain" "*"
    "--http.api" "eth,net,web3,personal,admin,txpool,miner,qmpow"
    "--ws"
    "--ws.addr" "0.0.0.0"
    "--ws.port" "8546"
    "--ws.origins" "*"
    "--ws.api" "eth,net,web3,personal,admin,txpool,miner,qmpow"
    "--authrpc.addr" "127.0.0.1"
    "--authrpc.port" "8551"
    "--authrpc.vhosts" "localhost"
    "--authrpc.jwtsecret" "jwt.hex"
    "--maxpeers" "25"
    "--verbosity" "3"
)

# Add mining if requested
if [ "$MINING" = true ]; then
    GETH_ARGS+=("--mine" "--miner.threads" "1" "--miner.etherbase" "0x0000000000000000000000000000000000000001")
    echo -e "\033[1;33m⛏️  Mining enabled with 1 thread\033[0m"
    echo -e "\033[1;36mNOTE: Use miner_setEtherbase RPC call to set your actual mining address\033[0m"
else
    # Enable mining interface for external miners (0 threads = no CPU mining)
    GETH_ARGS+=("--mine" "--miner.threads" "0" "--miner.etherbase" "0x0000000000000000000000000000000000000001")
    echo -e "\033[1;33m🌐 Mining interface enabled for external miners (no CPU mining)\033[0m"
    echo -e "\033[1;36mNOTE: External miners will set their own coinbase addresses via RPC\033[0m"
fi

# Add any extra arguments passed to the script
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    GETH_ARGS+=("${EXTRA_ARGS[@]}")
    echo -e "\033[1;36m🔧 Extra arguments: ${EXTRA_ARGS[*]}\033[0m"
fi

# Create JWT file if it doesn't exist
if [ ! -f "jwt.hex" ]; then
    echo "0x$(openssl rand -hex 32)" > jwt.hex
    echo -e "\033[1;32m🔑 Created JWT secret file\033[0m"
fi

echo -e "\033[1;37m🌐 Network: $NAME\033[0m"
echo -e "\033[1;37m🔗 Chain ID: $CHAINID\033[0m"
echo -e "\033[1;37m📁 Data Directory: $DATADIR\033[0m"
echo -e "\033[1;37m🌍 Port: $PORT\033[0m"
echo -e "\033[1;37m🌐 NAT: $NAT_CONFIG\033[0m"
echo -e "\033[1;37m🌐 HTTP RPC: http://0.0.0.0:8545\033[0m"
echo -e "\033[1;37m🌐 WebSocket: ws://0.0.0.0:8546\033[0m"
echo -e "\033[1;37m📡 Bootnodes: Auto-selected for $NETWORK network\033[0m"
echo ""
echo -e "\033[1;32m🎯 Starting Q Coin Geth node...\033[0m"
echo -e "\033[1;33m💡 Use Ctrl+C to stop the node\033[0m"
echo ""

# Start geth with error handling
./geth "${GETH_ARGS[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "\033[1;31m❌ Geth exited with error code: $EXIT_CODE\033[0m"
    echo -e "\033[1;33m💡 Common solutions:\033[0m"
    echo "  - Check if ports 8545, 8546, $PORT are available"
    echo "  - Make sure you have enough disk space"
    echo "  - Try deleting the data directory and restarting"
    exit $EXIT_CODE
fi 