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

# Build if geth doesn't exist
if [ ! -f "quantum-geth/build/bin/geth" ]; then
    echo -e "\033[1;33m🔨 Building Q Coin Geth...\033[0m"
    ./build-linux.sh geth
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Build failed!\033[0m"
        exit 1
    fi
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

BOOTNODES="enode://0bc243936ebc13ebf57895dff1321695064ae4b0ac0c1e047d52d695c396b64c52847f852a9738f0d079af4ba109dfceafd1cf0924587b151765834caf13e5fd@69.243.132.233:30305"

echo -e "\033[1;36m🚀 Starting $NAME (Chain ID: $CHAINID)\033[0m"

# Create data directory if it doesn't exist
if [ ! -d "$DATADIR" ]; then
    mkdir -p "$DATADIR"
    echo -e "\033[1;32m📁 Created data directory: $DATADIR\033[0m"
fi

# Initialize with genesis if needed
if [ ! -d "$DATADIR/geth/chaindata" ]; then
    echo -e "\033[1;33m🔧 Initializing blockchain with genesis file...\033[0m"
    ./quantum-geth/build/bin/geth init "$GENESIS" --datadir "$DATADIR"
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Genesis initialization failed!\033[0m"
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
    "--http.api" "eth,net,web3,personal,admin,txpool"
    "--ws"
    "--ws.addr" "0.0.0.0"
    "--ws.port" "8546"
    "--ws.origins" "*"
    "--ws.api" "eth,net,web3,personal,admin,txpool"
    "--authrpc.addr" "127.0.0.1"
    "--authrpc.port" "8551"
    "--authrpc.vhosts" "localhost"
    "--authrpc.jwtsecret" "jwt.hex"
    "--bootnodes" "$BOOTNODES"
    "--maxpeers" "25"
    "--verbosity" "3"
)

# Add mining if requested
if [ "$MINING" = true ]; then
    GETH_ARGS+=("--mine" "--miner.threads" "1")
    echo -e "\033[1;33m⛏️  Mining enabled with 1 thread\033[0m"
else
    GETH_ARGS+=("--miner.threads" "-1")
    echo -e "\033[1;33m🚫 Local mining disabled (external miners only)\033[0m"
fi

echo -e "\033[1;37m🌐 Network: $NAME\033[0m"
echo -e "\033[1;37m🔗 Chain ID: $CHAINID\033[0m"
echo -e "\033[1;37m📁 Data Directory: $DATADIR\033[0m"
echo -e "\033[1;37m🌍 Port: $PORT\033[0m"
echo -e "\033[1;37m📡 Bootnodes: $BOOTNODES\033[0m"
echo ""
echo -e "\033[1;32m🎯 Starting Q Coin Geth node...\033[0m"

# Start geth
./quantum-geth/build/bin/geth "${GETH_ARGS[@]}" 