#!/bin/bash
# Q Coin Geth Node Starter
# Usage: ./start-geth.sh [network] [options]
# Networks: mainnet, testnet, devnet (default: testnet)
# Options: --mining (enable mining with single thread)

NETWORK="testnet"
MINING=false
HELP=false

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
        ;;
    testnet)
        CHAINID=73235
        DATADIR="$HOME/.qcoin"
        GENESIS="genesis_quantum_testnet.json"
        PORT=30303
        NAME="Q Coin Testnet"
        ;;
    devnet)
        CHAINID=73234
        DATADIR="qdata"
        GENESIS="genesis_quantum_dev.json"
        PORT=30305
        NAME="Q Coin Dev Network"
        ;;
esac

BOOTNODES="enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@69.243.132.233:4294"

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
    "--http"
    "--http.addr" "0.0.0.0"
    "--http.port" "8545"
    "--http.corsdomain" "*"
    "--http.api" "eth,net,web3,personal,admin,txpool,miner"
    "--ws"
    "--ws.addr" "0.0.0.0"
    "--ws.port" "8546"
    "--ws.origins" "*"
    "--ws.api" "eth,net,web3,personal,admin,txpool,miner"
    "--authrpc.addr" "127.0.0.1"
    "--authrpc.port" "8551"
    "--authrpc.vhosts" "localhost"
    "--authrpc.jwtsecret" "jwt.hex"
    "--bootnodes" "$BOOTNODES"
    "--maxpeers" "25"
    "--verbosity" "3"
)

# Add mining configuration
if [ "$MINING" = true ]; then
    GETH_ARGS+=("--mine" "--miner.threads" "1" "--miner.etherbase" "0x1234567890123456789012345678901234567890")
    echo -e "\033[1;33m⛏️  Mining enabled with 1 thread\033[0m"
else
    # Enable mining interface for external miners but don't mine locally
    GETH_ARGS+=("--mine" "--miner.threads" "0" "--miner.etherbase" "0x1234567890123456789012345678901234567890")
    echo -e "\033[1;33m🔌 Mining interface enabled for external miners\033[0m"
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
echo -e "\033[1;37m🌐 HTTP RPC: http://0.0.0.0:8545\033[0m"
echo -e "\033[1;37m🌐 WebSocket: ws://0.0.0.0:8546\033[0m"
echo -e "\033[1;37m📡 Bootnodes: $BOOTNODES\033[0m"
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