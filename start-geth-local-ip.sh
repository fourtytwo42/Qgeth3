#!/bin/bash
# Q Coin Geth Node Starter with Manual IP Override
# Usage: ./start-geth-local-ip.sh [network] [local_ip] [options]
# Example: ./start-geth-local-ip.sh testnet 192.168.50.254

NETWORK="testnet"
MANUAL_IP=""
MINING=false
HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        mainnet|testnet|devnet)
            NETWORK="$1"
            shift
            ;;
        --ip)
            MANUAL_IP="$2"
            shift 2
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
            # Check if it looks like an IP address
            if [[ $1 =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                MANUAL_IP="$1"
                shift
            else
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo -e "\033[1;36mQ Coin Geth Node Starter (Manual IP)\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-geth-local-ip.sh [network] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mNetworks:\033[0m"
    echo "  mainnet   - Q Coin Mainnet (Chain ID 73236)"
    echo "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    echo "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --ip IP   - Manually specify local IP address"
    echo "  --mining  - Enable mining with single thread"
    echo "  --help    - Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-geth-local-ip.sh testnet --ip 192.168.50.254"
    echo "  ./start-geth-local-ip.sh devnet 192.168.1.100 --mining"
    echo "  ./start-geth-local-ip.sh mainnet --ip 10.0.0.5"
    exit 0
fi

# Build if geth doesn't exist
if [ ! -f "./geth" ] || [ ! -f "./geth.bin" ]; then
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
        DATADIR="$HOME/.qcoin/devnet"
        GENESIS="genesis_quantum_dev.json"
        PORT=30305
        NAME="Q Coin Dev Network"
        ;;
esac

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
    ./geth init "$GENESIS" --datadir "$DATADIR"
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Genesis initialization failed!\033[0m"
        exit 1
    fi
    echo -e "\033[1;32m✅ Blockchain initialized successfully\033[0m"
fi

# Determine IP address to use
if [ -n "$MANUAL_IP" ]; then
    LOCAL_IP="$MANUAL_IP"
    echo -e "\033[1;33m🏠 Using manually specified IP: $LOCAL_IP\033[0m"
else
    # Auto-detect local IP address
    LOCAL_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+' 2>/dev/null || echo "192.168.50.254")
    echo -e "\033[1;33m🏠 Auto-detected local IP: $LOCAL_IP\033[0m"
fi

# Prepare geth arguments
GETH_ARGS=(
    "--datadir" "$DATADIR"
    "--networkid" "$CHAINID"
    "--port" "$PORT"
    "--nat" "extip:$LOCAL_IP"
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
echo -e "\033[1;37m🏠 Local IP: $LOCAL_IP\033[0m"
echo -e "\033[1;37m📡 Bootnodes: Auto-selected for $NETWORK network\033[0m"
echo ""
echo -e "\033[1;32m🎯 Starting Q Coin Geth node...\033[0m"

# Start geth
./geth "${GETH_ARGS[@]}" 