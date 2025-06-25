#!/bin/bash

# Q Coin Testnet - Start Geth Node
# Starts a Q Coin testnet node with mining support (mining disabled by default)
# Uses default blockchain location like standard geth
# Usage: ./start-geth.sh [options]

# Default values
MINE=false
ETHERBASE=""
PORT=4294
RPCPORT=8545
WSPORT=8546
DATADIR=""
MAINNET=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mine)
            MINE=true
            shift
            ;;
        --etherbase)
            ETHERBASE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --rpcport)
            RPCPORT="$2"
            shift 2
            ;;
        --wsport)
            WSPORT="$2"
            shift 2
            ;;
        --datadir)
            DATADIR="$2"
            shift 2
            ;;
        --mainnet)
            MAINNET=true
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
if [ "$HELP" = true ]; then
    echo -e "\033[36mQ Coin Testnet - Geth Node\033[0m"
    echo ""
    echo -e "\033[33mUsage: ./start-geth.sh [options]\033[0m"
    echo ""
    echo -e "\033[32mOptions:\033[0m"
    echo "  --mine               Enable mining (disabled by default)"
    echo "  --etherbase <addr>   Mining reward address (required if mining)"
    echo "  --port <port>        P2P network port (default: 4294 testnet, 4295 mainnet)"
    echo "  --rpcport <port>     HTTP-RPC server port (default: 8545)"
    echo "  --wsport <port>      WebSocket server port (default: 8546)"
    echo "  --datadir <path>     Custom data directory (default: system default)"
    echo "  --mainnet            Use mainnet instead of testnet"
    echo "  --help, -h           Show this help message"
    echo ""
    echo -e "\033[33mExamples:\033[0m"
    echo "  ./start-geth.sh                                      # Start node (no mining)"
    echo "  ./start-geth.sh --mine --etherbase 0x123...         # Start with mining"
    echo ""
    echo -e "\033[35mQ Coin Testnet Details:\033[0m"
    echo "  Chain ID: 73235"
    echo "  Currency: Q (Q Coin)"
    echo "  Block Time: 12 seconds"
    echo "  Consensus: QMPoW (Quantum Proof of Work)"
    exit 0
fi

# Determine network configuration
if [ "$MAINNET" = true ]; then
    NETWORK_NAME="Q Coin Mainnet"
    CHAIN_ID="73236"
    GENESIS_FILE="../genesis_quantum_mainnet.json"
    if [ -n "$HOME" ]; then
        DEFAULT_DATADIR="$HOME/.qcoin/mainnet"
    else
        DEFAULT_DATADIR="/tmp/qcoin/mainnet"
    fi
    if [ "$PORT" -eq 4294 ]; then
        PORT=4295  # Switch to mainnet port if using default
    fi
else
    NETWORK_NAME="Q Coin Testnet"
    CHAIN_ID="73235"
    GENESIS_FILE="../genesis_quantum_testnet.json"
    if [ -n "$HOME" ]; then
        DEFAULT_DATADIR="$HOME/.qcoin/testnet"
    else
        DEFAULT_DATADIR="/tmp/qcoin/testnet"
    fi
fi

# Validate mining parameters
if [ "$MINE" = true ] && [ -z "$ETHERBASE" ]; then
    echo -e "\033[31mERROR: Mining requires an etherbase address!\033[0m"
    echo -e "\033[33mUse: ./start-geth.sh --mine --etherbase <your_address>\033[0m"
    exit 1
fi

# Find the latest quantum-geth release
echo -e "\033[36m$NETWORK_NAME - Starting Geth Node\033[0m"
echo ""

# Check if genesis file exists
if [ ! -f "$GENESIS_FILE" ]; then
    echo -e "\033[31mERROR: Genesis file not found: $GENESIS_FILE\033[0m"
    echo -e "\033[33mMake sure you're running this script from the scripts directory\033[0m"
    exit 1
fi

GETH_RELEASE_DIR=$(find ../releases -name "quantum-geth-*" -type d 2>/dev/null | sort -r | head -n 1)
if [ -z "$GETH_RELEASE_DIR" ]; then
    echo -e "\033[31mERROR: No quantum-geth release found!\033[0m"
    echo -e "\033[33mPlease run: ./build-release.sh geth\033[0m"
    exit 1
fi

GETH_EXECUTABLE="$GETH_RELEASE_DIR/geth"
if [ ! -f "$GETH_EXECUTABLE" ]; then
    GETH_EXECUTABLE="$GETH_RELEASE_DIR/quantum-geth"
fi

if [ ! -f "$GETH_EXECUTABLE" ]; then
    echo -e "\033[31mERROR: Geth executable not found in release directory!\033[0m"
    echo -e "\033[33mFound release dir: $GETH_RELEASE_DIR\033[0m"
    echo -e "\033[33mTrying to build release...\033[0m"
    
    # Try to build a release
    if ./build-release.sh geth; then
        GETH_RELEASE_DIR=$(find ../releases -name "quantum-geth-*" -type d 2>/dev/null | sort -r | head -n 1)
        GETH_EXECUTABLE="$GETH_RELEASE_DIR/geth"
        if [ ! -f "$GETH_EXECUTABLE" ]; then
            echo -e "\033[31mERROR: Build succeeded but executable still not found!\033[0m"
            exit 1
        fi
    else
        echo -e "\033[31mERROR: Failed to build quantum-geth release!\033[0m"
        exit 1
    fi
fi

echo -e "\033[32mUsing geth from: $(basename "$GETH_RELEASE_DIR")\033[0m"

# Determine data directory
if [ -z "$DATADIR" ]; then
    # Use network-specific default directory
    DATADIR="$DEFAULT_DATADIR"
    echo -e "\033[32mUsing default data directory: $DATADIR\033[0m"
else
    echo -e "\033[32mUsing custom data directory: $DATADIR\033[0m"
fi

# Check if blockchain is initialized
if [ ! -d "$DATADIR/geth/chaindata" ]; then
    echo ""
    echo -e "\033[33mInitializing $NETWORK_NAME blockchain...\033[0m"
    
    # Create data directory if it doesn't exist
    mkdir -p "$DATADIR"
    
    # Initialize with genesis
    if "$GETH_EXECUTABLE" --datadir "$DATADIR" init "$GENESIS_FILE"; then
        echo -e "\033[32m$NETWORK_NAME blockchain initialized successfully!\033[0m"
    else
        echo -e "\033[31mERROR: Failed to initialize blockchain!\033[0m"
        exit 1
    fi
fi

# Build geth command arguments
GETH_ARGS=(
    --datadir "$DATADIR"
    --networkid "$CHAIN_ID"
    --port "$PORT"
    --http
    --http.addr 0.0.0.0
    --http.port "$RPCPORT"
    --http.api eth,net,web3,personal,admin,miner,debug,txpool
    --http.corsdomain "*"
    --ws
    --ws.addr 0.0.0.0
    --ws.port "$WSPORT"
    --ws.api eth,net,web3,personal,admin,miner,debug,txpool
    --ws.origins "*"
    --nat any
    --maxpeers 50
    --allow-insecure-unlock
    --syncmode full
    --gcmode archive
    --bootnodes "enode://0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000@192.168.50.254:30303,enode://0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000@192.168.50.152:30303"
)

# Add mining parameters if enabled
if [ "$MINE" = true ]; then
    GETH_ARGS+=(--mine --miner.etherbase "$ETHERBASE")
    echo -e "\033[33mMining enabled - rewards go to: $ETHERBASE\033[0m"
fi

# Display startup information
echo ""
echo -e "\033[36m$NETWORK_NAME Configuration:\033[0m"
echo -e "\033[37m  Chain ID: $CHAIN_ID\033[0m"
echo -e "\033[37m  Currency: Q (Q Coin)\033[0m"
echo -e "\033[37m  Data Directory: $DATADIR\033[0m"
echo -e "\033[37m  P2P Port: $PORT\033[0m"
echo -e "\033[37m  RPC Port: $RPCPORT\033[0m"
echo -e "\033[37m  WebSocket Port: $WSPORT\033[0m"
if [ "$MINE" = true ]; then
    echo -e "\033[37m  Mining: \033[32mENABLED\033[0m"
else
    echo -e "\033[37m  Mining: \033[33mDISABLED\033[0m"
fi
echo ""
echo -e "\033[32mStarting $NETWORK_NAME node...\033[0m"
echo -e "\033[33mPress Ctrl+C to stop\033[0m"
echo ""

# Start geth
exec "$GETH_EXECUTABLE" "${GETH_ARGS[@]}" 