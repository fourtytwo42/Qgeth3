#!/bin/bash
# Q Coin Linux Miner Startup Script
# Automatically detects and configures GPU acceleration
# Usage: ./start-linux-miner.sh [options]

# Default values
THREADS=$(nproc)
COINBASE=""
NODE_URL="http://localhost:8545"
NETWORK="testnet"
GPU_MODE="auto"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -c|--coinbase)
            COINBASE="$2"
            shift 2
            ;;
        -n|--node)
            NODE_URL="$2"
            shift 2
            ;;
        --mainnet)
            NETWORK="mainnet"
            NODE_URL="http://localhost:8546"
            shift
            ;;
        --testnet)
            NETWORK="testnet"
            NODE_URL="http://localhost:8545"
            shift
            ;;
        --gpu)
            GPU_MODE="force"
            shift
            ;;
        --cpu)
            GPU_MODE="disable"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Q Coin Linux Miner - Quantum Blockchain Mining"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -t, --threads NUM    Number of mining threads (default: $(nproc))"
            echo "  -c, --coinbase ADDR  Mining reward address"
            echo "  -n, --node URL       Geth node URL (default: http://localhost:8545)"
            echo "  --mainnet            Connect to Q Coin Mainnet (Chain ID 73236)"
            echo "  --testnet            Connect to Q Coin Testnet (Chain ID 73235, default)"
            echo "  --gpu                Force GPU mining (fail if not available)"
            echo "  --cpu                Force CPU mining (disable GPU)"
            echo "  -v, --verbose        Enable verbose output"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --threads 8 --coinbase 0x1234...7890"
            echo "  $0 --mainnet --gpu --verbose"
            echo "  $0 --cpu --threads 16"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🪙 Q Coin Linux Miner Starting..."
echo "=================================="

# Check if miner binary exists
if [ ! -f "./quantum-miner" ]; then
    echo "❌ Error: quantum-miner binary not found!"
    echo "   Build it first: ./build-linux.sh miner"
    exit 1
fi

# Auto-detect coinbase if not provided
if [ -z "$COINBASE" ]; then
    echo "🔍 Auto-detecting mining address..."
    
    # Try to get account from geth
    COINBASE=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
        "$NODE_URL" | grep -o '"0x[a-fA-F0-9]\{40\}"' | head -1 | tr -d '"')
    
    if [ -n "$COINBASE" ]; then
        echo "✅ Using existing account: $COINBASE"
    else
        echo "⚠️  No accounts found. You may need to create one first:"
        echo "   ./geth account new"
        echo ""
        echo "Using default address for testing..."
        COINBASE="0x1234567890123456789012345678901234567890"
    fi
fi

# Detect GPU capabilities
GPU_AVAILABLE=false
GPU_TYPE="None"

if [ "$GPU_MODE" != "disable" ]; then
    echo "🔍 Detecting GPU capabilities..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$GPU_INFO" ]; then
            GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
            GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
            echo "✅ NVIDIA GPU detected: $GPU_NAME (${GPU_MEMORY}MB)"
            
            # Check for CUDA support
            if command -v nvcc >/dev/null 2>&1; then
                CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]\+\.[0-9]\+" | cut -d' ' -f2)
                echo "✅ CUDA toolkit detected: v$CUDA_VERSION"
                GPU_TYPE="CUDA"
                GPU_AVAILABLE=true
            else
                # Check for Qiskit-Aer GPU
                if python3 -c "import qiskit_aer; print('Qiskit-Aer available')" >/dev/null 2>&1; then
                    echo "✅ Qiskit-Aer available for GPU acceleration"
                    GPU_TYPE="Qiskit-GPU"
                    GPU_AVAILABLE=true
                else
                    echo "⚠️  GPU detected but no CUDA toolkit or Qiskit-Aer found"
                fi
            fi
        fi
    fi
    
    if [ "$GPU_AVAILABLE" = false ]; then
        if [ "$GPU_MODE" = "force" ]; then
            echo "❌ Error: GPU mining forced but no GPU acceleration available"
            exit 1
        else
            echo "ℹ️  No GPU acceleration available, using CPU mining"
        fi
    fi
else
    echo "ℹ️  GPU mining disabled by user"
fi

# Test connection to geth
echo "🔗 Testing connection to geth node..."
NETWORK_INFO=$(curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
    "$NODE_URL")

if echo "$NETWORK_INFO" | grep -q '"result"'; then
    CHAIN_ID=$(echo "$NETWORK_INFO" | grep -o '"0x[a-fA-F0-9]*"' | tr -d '"')
    CHAIN_ID_DEC=$((CHAIN_ID))
    
    case $CHAIN_ID_DEC in
        73234)
            DETECTED_NETWORK="Q Coin Dev Network"
            ;;
        73235)
            DETECTED_NETWORK="Q Coin Testnet"
            ;;
        73236)
            DETECTED_NETWORK="Q Coin Mainnet"
            ;;
        *)
            DETECTED_NETWORK="Unknown Network (Chain ID: $CHAIN_ID_DEC)"
            ;;
    esac
    
    echo "✅ Connected to: $DETECTED_NETWORK"
else
    echo "❌ Error: Cannot connect to geth node at $NODE_URL"
    echo "   Make sure geth is running:"
    echo "   ./start-linux-geth.sh"
    exit 1
fi

# Display mining configuration
echo ""
echo "🏗️  Mining Configuration:"
echo "=========================="
echo "Network: $DETECTED_NETWORK"
echo "Node URL: $NODE_URL"
echo "Mining Address: $COINBASE"
echo "Threads: $THREADS"
echo "GPU Type: $GPU_TYPE"
echo "GPU Available: $GPU_AVAILABLE"
echo ""

# Prepare miner arguments
MINER_ARGS="-threads $THREADS -coinbase $COINBASE -node $NODE_URL"

if [ "$VERBOSE" = true ]; then
    MINER_ARGS="$MINER_ARGS -log"
fi

# Set environment for optimal performance
export GOMAXPROCS=$THREADS

if [ "$GPU_AVAILABLE" = true ]; then
    export CUDA_VISIBLE_DEVICES=0
    export QISKIT_IN_PARALLEL=TRUE
    export OPENBLAS_NUM_THREADS=1
fi

echo "🚀 Starting Q Coin miner..."
echo "Press Ctrl+C to stop"
echo ""

# Start the miner
exec ./quantum-miner $MINER_ARGS 