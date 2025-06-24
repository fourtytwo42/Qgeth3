#!/bin/bash
# Quantum-Geth Standalone Miner - Solo Mining Script (Linux)

COINBASE=""
NODE_URL="http://localhost:8545"
THREADS=$(nproc)
INTENSITY=1
MINER_EXECUTABLE="quantum-miner"

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coinbase ADDRESS    Coinbase address (required)"
            echo "  -n, --node URL           Node URL (default: http://localhost:8545)"
            echo "  -t, --threads NUMBER     Number of threads (default: CPU count)"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
            echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -t 8"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo " Starting Quantum-Geth Standalone Miner (Solo Mode)"
echo ""

if [ ! -f "$MINER_EXECUTABLE" ]; then
    echo " Miner executable not found: $MINER_EXECUTABLE"
    echo "Please run build-linux.sh first to compile the miner."
    exit 1
fi

if [ -z "$COINBASE" ]; then
    echo " Coinbase address is required for solo mining!"
    echo ""
    echo "Usage examples:"
    echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    echo "  $0 -c 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -t 8"
    exit 1
fi

if [[ ! $COINBASE =~ ^0x[0-9a-fA-F]{40}$ ]]; then
    echo " Invalid coinbase address format!"
    echo "Expected format: 0x followed by 40 hex characters"
    exit 1
fi

echo " Mining Configuration:"
echo "  Mode: Solo Mining"
echo "  Coinbase: $COINBASE"
echo "  Node URL: $NODE_URL"
echo "  Threads: $THREADS"
echo ""

echo " Starting quantum miner..."
exec "./$MINER_EXECUTABLE" -coinbase "$COINBASE" -node "$NODE_URL" -threads "$THREADS"
