#!/bin/bash
# Q Coin Linux Miner Startup Script
# Connects to local Q Coin testnet node and starts mining

set -e

# Configuration
RPC_URL="http://127.0.0.1:8545"
MINER_ADDRESS="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
THREADS=1

echo "⚡ Q Coin Quantum Miner - Linux"
echo "==============================="
echo ""

# Check if miner binary exists
if [ ! -f "./quantum-miner" ]; then
    echo "❌ ERROR: quantum-miner binary not found!"
    echo "   Run: ./build-linux.sh"
    exit 1
fi

# Check if geth node is running
echo "🔍 Checking if Q Coin node is running..."
if ! curl -s -X POST -H "Content-Type: application/json" \
     --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
     "$RPC_URL" > /dev/null 2>&1; then
    echo "❌ ERROR: Q Coin node not running!"
    echo "   Start the node first: ./start-linux-geth.sh"
    exit 1
fi

echo "✅ Q Coin node is running"
echo ""

# Display configuration
echo "🔧 Miner Configuration:"
echo "   RPC Endpoint: $RPC_URL"
echo "   Mining Address: $MINER_ADDRESS"
echo "   Threads: $THREADS"
echo "   Quantum Algorithm: QMPoW (16 qubits × 20 T-gates)"
echo ""

echo "⚡ Starting Q Coin quantum miner..."
echo "   Press Ctrl+C to stop"
echo ""

# Start mining
exec ./quantum-miner \
    -rpc-url "$RPC_URL" \
    -address "$MINER_ADDRESS" \
    -threads "$THREADS" 