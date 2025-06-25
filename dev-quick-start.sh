#!/bin/bash
# Q Coin Dev Quick Start (Linux)
# One-command development shortcut: Build + Start Dev Network
# This script builds geth and immediately starts the Q Coin dev network with peer connections
# Usage: ./dev-quick-start.sh

# Default parameters
CLEAN=false
MINING=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
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

# Show help
if [ "$HELP" = true ]; then
    echo -e "\033[36m🚀 Q Coin Dev Quick Start (Linux)\033[0m"
    echo ""
    echo -e "\033[33mOne-command development shortcut: Build + Start Dev Network\033[0m"
    echo ""
    echo -e "\033[37mUsage: ./dev-quick-start.sh [OPTIONS]\033[0m"
    echo ""
    echo -e "\033[32mOptions:\033[0m"
    echo "  --clean     Clean build (rebuild everything)"
    echo "  --mining    Start with internal mining enabled"
    echo "  --help, -h  Show this help message"
    echo ""
    echo -e "\033[33mWhat this script does:\033[0m"
    echo "  1. Builds quantum-geth (clean if --clean specified)"
    echo "  2. Auto-initializes dev blockchain if needed"
    echo "  3. Starts Q Coin Dev Network (Chain ID 73234)"
    echo "  4. Connects to dev peer bootnodes automatically"
    echo "  5. Serves RPC endpoints for external miners"
    echo ""
    echo -e "\033[35mNetworks:\033[0m"
    echo "  Dev Network: Chain ID 73234, Port 30305"
    echo "  Peers: 192.168.50.254:30305 & 192.168.50.152:30305"
    echo "  RPC: http://127.0.0.1:8545"
    echo ""
    exit 0
fi

echo -e "\033[36m🚀 Q COIN DEV QUICK START (LINUX)\033[0m"
echo -e "\033[36m===================================\033[0m"
echo ""

# Step 1: Build quantum-geth
echo -e "\033[33m🔨 Step 1: Building Quantum-Geth...\033[0m"

if [ "$CLEAN" = true ]; then
    echo -e "\033[37m   Clean build requested - rebuilding everything\033[0m"
    if [ -f "geth" ]; then rm -f "geth"; fi
    if [ -f "geth.bin" ]; then rm -f "geth.bin"; fi
    if [ -f "quantum-geth" ]; then rm -f "quantum-geth"; fi
    echo -e "\033[32m   Cleaned previous builds\033[0m"
fi

# Build using existing build script
if [ -f "build-linux.sh" ]; then
    echo -e "\033[37m   Running build-linux.sh...\033[0m"
    
    # Make build script executable if needed
    if [ ! -x "build-linux.sh" ]; then
        echo -e "\033[37m   Making build-linux.sh executable...\033[0m"
        chmod +x build-linux.sh
    fi
    
    if ./build-linux.sh geth; then
        echo -e "\033[32m✅ Quantum-Geth built successfully!\033[0m"
    else
        echo -e "\033[31m❌ Build failed!\033[0m"
        exit 1
    fi
else
    echo -e "\033[31m❌ ERROR: build-linux.sh not found!\033[0m"
    echo -e "\033[33m   Make sure you're in the Q Coin root directory\033[0m"
    exit 1
fi

echo ""

# Step 2: Initialize and start dev network
echo -e "\033[33m🌐 Step 2: Starting Q Coin Dev Network...\033[0m"

# Data directory configuration
DATADIR="qdata"
NETWORKID=73234
PORT=30305
HTTPPORT=8545
WSPORT=8546
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"

# Find geth executable
GETH_EXECUTABLE=""
if [ -f "geth" ]; then
    GETH_EXECUTABLE="./geth"
    echo -e "\033[37m   Using Linux/wrapper geth executable\033[0m"
elif [ -f "geth.bin" ]; then
    GETH_EXECUTABLE="./geth.bin"
    echo -e "\033[37m   Using geth.bin executable\033[0m"
else
    echo -e "\033[31m❌ ERROR: No geth executable found!\033[0m"
    exit 1
fi

# Check for genesis file
if [ ! -f "genesis_quantum_dev.json" ]; then
    echo -e "\033[31m❌ ERROR: genesis_quantum_dev.json not found!\033[0m"
    echo -e "\033[33m   Make sure you're in the Q Coin root directory\033[0m"
    exit 1
fi

# Initialize blockchain if needed
if [ ! -d "$DATADIR/geth/chaindata" ]; then
    echo -e "\033[33m🏗️  Initializing Q Coin Dev blockchain...\033[0m"
    echo -e "\033[37m   Data Directory: $DATADIR\033[0m"
    echo -e "\033[37m   Genesis File: genesis_quantum_dev.json\033[0m"
    
    # Create data directory
    mkdir -p "$DATADIR"
    
    # Initialize with genesis
    if "$GETH_EXECUTABLE" --datadir "$DATADIR" init genesis_quantum_dev.json; then
        echo -e "\033[32m✅ Q Coin Dev blockchain initialized!\033[0m"
    else
        echo -e "\033[31m❌ ERROR: Failed to initialize blockchain!\033[0m"
        exit 1
    fi
else
    echo -e "\033[32m✅ Dev blockchain already initialized\033[0m"
fi

echo ""

# Step 3: Start the dev network
echo -e "\033[33m🚀 Step 3: Starting Dev Network Node...\033[0m"

# Display configuration
echo ""
echo -e "\033[36m🔧 Dev Network Configuration:\033[0m"
echo -e "\033[37m  Network: Q Coin Dev/Staging\033[0m"
echo -e "\033[37m  Chain ID: $NETWORKID\033[0m"
echo -e "\033[37m  Data Directory: $DATADIR\033[0m"
echo -e "\033[37m  P2P Port: $PORT\033[0m"
echo -e "\033[37m  RPC Port: $HTTPPORT\033[0m"
echo -e "\033[37m  WebSocket Port: $WSPORT\033[0m"
echo -e "\033[37m  Etherbase: $ETHERBASE\033[0m"
if [ "$MINING" = true ]; then
    echo -e "\033[37m  Mining: \033[32mINTERNAL THREADS ENABLED\033[0m"
else
    echo -e "\033[37m  Mining: \033[33mEXTERNAL MINERS ONLY\033[0m"
fi
echo ""

echo -e "\033[35m🪙 Quantum-Geth Features:\033[0m"
echo -e "\033[37m  ⚡ 128 sequential quantum puzzles (16 qubits × 20 T-gates)\033[0m"
echo -e "\033[37m  🪙 Bitcoin-style halving (50 QGC → 25 QGC → 12.5 QGC...)\033[0m"
echo -e "\033[37m  📅 600,000 block epochs (~6 months)\033[0m"
echo -e "\033[37m  🔄 ASERT-Q difficulty adjustment (12s target)\033[0m"
echo -e "\033[37m  🌐 Auto-connect to dev bootnodes\033[0m"
echo ""

echo -e "\033[32m🌐 Starting node and connecting to dev peers...\033[0m"
echo -e "\033[37m   Bootnodes: 192.168.50.254:30305 & 192.168.50.152:30305\033[0m"
echo -e "\033[33m   Press Ctrl+C to stop\033[0m"
echo ""

# Build geth arguments
GETH_ARGS=(
    --datadir "$DATADIR"
    --networkid "$NETWORKID"
    --port "$PORT"
    --http
    --http.addr 0.0.0.0
    --http.port "$HTTPPORT"
    --http.api "eth,net,web3,personal,miner,qmpow,admin,debug,trace"
    --http.corsdomain "*"
    --http.vhosts "*"
    --ws
    --ws.addr 0.0.0.0
    --ws.port "$WSPORT"
    --ws.api "eth,net,web3,personal,admin,miner"
    --ws.origins "*"
    --miner.etherbase "$ETHERBASE"
    --bootnodes "enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.254:30305,enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.50.152:30305"
    --nat any
    --maxpeers 25
    --verbosity 3
    --allow-insecure-unlock
)

# Add mining configuration
if [ "$MINING" = true ]; then
    GETH_ARGS+=(--mine --miner.threads 1)
else
    GETH_ARGS+=(--mine --miner.threads 0)
fi

# Start geth
"$GETH_EXECUTABLE" "${GETH_ARGS[@]}"

echo ""
echo -e "\033[32m🛑 Q Coin Dev Network stopped.\033[0m" 