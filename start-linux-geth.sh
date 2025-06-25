#!/bin/bash
# Q Coin Linux Testnet Startup Script
# Initializes and starts Q Coin testnet node with proper configuration

set -e

# Configuration
DATADIR="qdata"
GENESIS_FILE="genesis_quantum_testnet.json"
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
NETWORK_ID="73235"
PORT="30303"
HTTP_PORT="8545"
WS_PORT="8546"

echo "🪙 Q Coin Testnet - Linux Startup"
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
    echo "🏗️  Initializing Q Coin testnet blockchain..."
    echo "   Data Directory: $DATADIR"
    echo "   Genesis File: $GENESIS_FILE"
    echo ""
    
    # Create data directory
    mkdir -p "$DATADIR"
    
    # Initialize with Q Coin genesis
    ./geth --datadir "$DATADIR" init "$GENESIS_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Q Coin blockchain initialized successfully!"
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
echo "   Chain ID: $NETWORK_ID (Q Coin Testnet)"
echo "   Data Directory: $DATADIR"
echo "   Mining Address: $ETHERBASE"
echo "   P2P Port: $PORT"
echo "   HTTP RPC: http://localhost:$HTTP_PORT"
echo "   WebSocket: ws://localhost:$WS_PORT"
echo "   Mining: External (0 internal threads)"
echo ""

echo "🚀 Starting Q Coin testnet node..."
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