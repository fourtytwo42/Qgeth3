#!/bin/bash
# Quantum-Geth Standalone Miner - Solo Mining Script (Linux)
# Runs the quantum miner in solo mining mode

# Default parameters
COINBASE=""
NODE_URL="http://localhost:8545"
THREADS=$(nproc)
INTENSITY=1
CONFIG_FILE="miner.json"
MINER_EXECUTABLE="quantum-miner"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--coinbase)
            COINBASE="$2"
            shift 2
            ;;
        -n|--node)
            NODE_URL="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -i|--intensity)
            INTENSITY="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coinbase ADDRESS    Coinbase address (required)"
            echo "  -n, --node URL           Node URL (default: http://localhost:8545)"
            echo "  -t, --threads NUMBER     Number of threads (default: CPU count)"
            echo "  -i, --intensity NUMBER   Mining intensity 1-10 (default: 1)"
            echo "      --config FILE        Config file path (default: miner.json)"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
            echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -t 8"
            echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -n http://192.168.1.100:8545"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting Quantum-Geth Standalone Miner (Solo Mode)"
echo ""

# Check if miner executable exists
if [ ! -f "$MINER_EXECUTABLE" ]; then
    echo "❌ Miner executable not found: $MINER_EXECUTABLE"
    echo "Please run build-linux.sh first to compile the miner."
    exit 1
fi

# Check if coinbase is provided
if [ -z "$COINBASE" ]; then
    echo "❌ Coinbase address is required for solo mining!"
    echo ""
    echo "Usage examples:"
    echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -t 8"
    echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -n http://192.168.1.100:8545"
    echo ""
    exit 1
fi

# Validate coinbase address format
if [[ ! $COINBASE =~ ^0x[0-9a-fA-F]{40}$ ]]; then
    echo "❌ Invalid coinbase address format!"
    echo "Expected format: 0x followed by 40 hex characters"
    echo "Example: 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    exit 1
fi

echo "📋 Mining Configuration:"
echo "  Mode: Solo Mining"
echo "  Coinbase: $COINBASE"
echo "  Node URL: $NODE_URL"
echo "  Threads: $THREADS"
echo "  Intensity: $INTENSITY"
echo "  Config File: $CONFIG_FILE"
echo ""

# Test connection to node
echo "🔌 Testing connection to geth node..."
if command -v curl >/dev/null 2>&1; then
    response=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
        --connect-timeout 5 \
        "$NODE_URL" 2>/dev/null)
    
    if echo "$response" | grep -q '"result"'; then
        block_hex=$(echo "$response" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)
        block_number=$((16#${block_hex#0x}))
        echo "✅ Connected to geth node successfully!"
        echo "  Current block: #$block_number"
    else
        echo "⚠️  Connected but got unexpected response"
    fi
else
    echo "⚠️  curl not found, skipping connection test"
fi

echo ""
echo "🚀 Starting quantum miner..."

# Build miner arguments
MINER_ARGS=(
    "-node" "$NODE_URL"
    "-coinbase" "$COINBASE"
    "-threads" "$THREADS"
    "-intensity" "$INTENSITY"
    "-config" "$CONFIG_FILE"
)

echo "Command: ./$MINER_EXECUTABLE ${MINER_ARGS[*]}"
echo ""

# Start the miner
exec "./$MINER_EXECUTABLE" "${MINER_ARGS[@]}" 