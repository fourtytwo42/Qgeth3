#!/bin/bash
# Q Coin Linux Startup Script
# Initializes and starts Q Coin node (testnet by default, mainnet with --mainnet)
# Usage: ./start-linux-geth.sh [--mainnet] [--etherbase ADDRESS] [--port PORT]

set -e

# Default configuration (Q Coin Testnet)
DATADIR="qdata"
GENESIS_FILE="genesis_quantum_testnet.json"
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
NETWORK_ID="73235"
NETWORK_NAME="Q Coin Testnet"
PORT="30303"
HTTP_PORT="8545"
WS_PORT="8546"
USE_MAINNET=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mainnet)
            USE_MAINNET=true
            DATADIR="$HOME/.qcoin/mainnet"
            GENESIS_FILE="genesis_quantum_mainnet.json"
            NETWORK_ID="73236"
            NETWORK_NAME="Q Coin Mainnet"
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
        --help|-h)
            echo "Q Coin Node Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mainnet          Use Q Coin mainnet (default: testnet)"
            echo "  --etherbase ADDR   Mining reward address"
            echo "  --port PORT        P2P network port (default: 30303)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Networks:"
            echo "  Testnet (default): Chain ID 73235, genesis_quantum_testnet.json"
            echo "  Mainnet:          Chain ID 73236, genesis_quantum_mainnet.json"
            echo ""
            echo "Note: This geth ONLY connects to Q Coin networks, never Ethereum!"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🪙 $NETWORK_NAME - Linux Startup"
echo "================================="
echo ""

# Check if geth binary exists
if [ ! -f "./geth" ]; then
    echo "❌ ERROR: geth binary not found!"
    echo "   Run: ./build-linux.sh"
    exit 1
fi

# Check if genesis file exists
if [ ! -f "$GENESIS_FILE" ]; then
    echo "❌ ERROR: Genesis file not found: $GENESIS_FILE"
    exit 1
fi

# Initialize blockchain if not already done
if [ ! -d "$DATADIR/geth/chaindata" ]; then
    echo "🏗️  Initializing $NETWORK_NAME blockchain..."
    echo "   Data Directory: $DATADIR"
    echo "   Genesis File: $GENESIS_FILE"
    echo ""
    
    # Create data directory
    mkdir -p "$DATADIR"
    
    # Initialize with Q Coin genesis
    ./geth --datadir "$DATADIR" init "$GENESIS_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ $NETWORK_NAME blockchain initialized successfully!"
        echo ""
    else
        echo "❌ Failed to initialize blockchain!"
        exit 1
    fi
else
    echo "✅ Blockchain already initialized"
    echo ""
fi

# Display configuration
echo "🔧 Node Configuration:"
echo "   Network: $NETWORK_NAME"
echo "   Chain ID: $NETWORK_ID"
echo "   Data Directory: $DATADIR"
echo "   Mining Address: $ETHERBASE"
echo "   P2P Port: $PORT"
echo "   HTTP RPC: http://localhost:$HTTP_PORT"
echo "   WebSocket: ws://localhost:$WS_PORT"
echo "   Mining: External (0 internal threads)"
echo ""

echo "🚀 Starting $NETWORK_NAME node..."
echo "   Press Ctrl+C to stop"
echo ""

# Start geth with Q Coin testnet configuration
exec ./geth \
    --datadir "$DATADIR" \
    --networkid "$NETWORK_ID" \
    --port "$PORT" \
    --http \
    --http.addr "0.0.0.0" \
    --http.port "$HTTP_PORT" \
    --http.api "eth,net,web3,personal,admin,miner,debug,txpool,qmpow" \
    --http.corsdomain "*" \
    --ws \
    --ws.addr "0.0.0.0" \
    --ws.port "$WS_PORT" \
    --ws.api "eth,net,web3,personal,admin,miner,debug,txpool,qmpow" \
    --ws.origins "*" \
    --nat "any" \
    --maxpeers 50 \
    --syncmode "full" \
    --gcmode "archive" \
    --mine \
    --miner.threads 0 \
    --miner.etherbase "$ETHERBASE" \
    --allow-insecure-unlock 