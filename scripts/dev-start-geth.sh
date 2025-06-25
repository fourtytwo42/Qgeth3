#!/bin/bash
# Start Geth (No Mining) - Quantum-Geth with Halving
# Starts the quantum geth node without mining enabled - serves work to external miners

# Default parameters
DATADIR="qdata"
NETWORKID=73235
PORT=30303
HTTPPORT=8545
AUTHRPCPORT=8551
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
VERBOSITY=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datadir)
            DATADIR="$2"
            shift 2
            ;;
        --networkid)
            NETWORKID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --httpport)
            HTTPPORT="$2"
            shift 2
            ;;
        --authrpcport)
            AUTHRPCPORT="$2"
            shift 2
            ;;
        --etherbase)
            ETHERBASE="$2"
            shift 2
            ;;
        --verbosity)
            VERBOSITY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --datadir DIR       Data directory (default: qdata)"
            echo "  --networkid ID      Network ID (default: 73235)"
            echo "  --port PORT         P2P port (default: 30303)"
            echo "  --httpport PORT     HTTP RPC port (default: 8545)"
            echo "  --authrpcport PORT  Auth RPC port (default: 8551)"
            echo "  --etherbase ADDR    Coinbase address"
            echo "  --verbosity LEVEL   Log verbosity (default: 4)"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "*** STARTING QUANTUM-GETH NODE (EXTERNAL MINING) ***"
echo "Configuration:"
echo "  * Data Directory: $DATADIR"
echo "  * Network ID: $NETWORKID"
echo "  * Port: $PORT"
echo "  * HTTP Port: $HTTPPORT"
echo "  * Auth RPC Port: $AUTHRPCPORT"
echo "  * Etherbase: $ETHERBASE"
echo "  * Verbosity: $VERBOSITY"
echo "  * Mining: EXTERNAL MINERS ONLY (0 internal threads)"
echo ""

echo "Quantum-Geth Features:"
echo "  * 128 sequential quantum puzzles (16 qubits x 20 T-gates)"
echo "  * Bitcoin-style halving (50 QGC -> 25 QGC -> 12.5 QGC...)"
echo "  * 600,000 block epochs (approximately 6 months)"
echo "  * Branch-serial quantum circuit execution"
echo "  * Mahadev->CAPSS->Nova proof stack"
echo "  * Dilithium-2 self-attestation"
echo "  * ASERT-Q difficulty adjustment (12s target)"
echo "  * Single RLP quantum blob (197 bytes)"
echo ""

# Check if data directory exists
if [ ! -d "$DATADIR" ]; then
    echo "Data directory '$DATADIR' not found!"
    echo "Initializing with quantum genesis..."
    echo "  ./dev-reset-blockchain.sh --difficulty 1 --force"
    echo ""
    echo "Please run the reset script first to initialize the blockchain:"
    echo "   ./dev-reset-blockchain.sh --difficulty 1 --force"
    echo ""
    exit 1
fi

echo "Starting quantum geth node (NO MINING)..."
echo "This node serves RPC/HTTP endpoints for external miners WITHOUT mining itself."
echo "Use ./start-mining.sh to start mining, or external miners to mine."
echo "Press Ctrl+C to stop the node."
echo ""

# Find the newest quantum-geth release or use development version
GETH_RELEASE_DIR=$(find ../releases -name "quantum-geth-*" -type d 2>/dev/null | sort -V | tail -1)

if [ -n "$GETH_RELEASE_DIR" ] && [ -f "$GETH_RELEASE_DIR/geth" ]; then
    echo "Using geth from release: $(basename $GETH_RELEASE_DIR)"
    GETH_PATH="$GETH_RELEASE_DIR/geth"
elif [ -f "../geth" ]; then
    echo "Using development geth from parent directory"
    GETH_PATH="../geth"
elif [ ! -f "geth" ]; then
    echo "Geth executable not found. Building release..."
    if ./build-release.sh geth; then
        GETH_RELEASE_DIR=$(find ../releases -name "quantum-geth-*" -type d 2>/dev/null | sort -V | tail -1)
        echo "SUCCESS: Release built at $GETH_RELEASE_DIR"
        GETH_PATH="$GETH_RELEASE_DIR/geth"
    else
        echo "ERROR: Failed to build quantum-geth release"
        exit 1
    fi
else
    echo "Using development geth from current directory"
    GETH_PATH="./geth"
fi

# Start geth WITHOUT mining - pure RPC node for external miners
exec "$GETH_PATH" \
    --datadir "$DATADIR" \
    --networkid "$NETWORKID" \
    --port "$PORT" \
    --http \
    --http.addr "0.0.0.0" \
    --http.port "$HTTPPORT" \
    --http.api "eth,net,web3,personal,miner" \
    --http.corsdomain "*" \
    --authrpc.addr "0.0.0.0" \
    --authrpc.port "$AUTHRPCPORT" \
    --authrpc.vhosts "*" \
    --ws \
    --ws.addr "0.0.0.0" \
    --ws.port 8546 \
    --ws.api "eth,net,web3" \
    --ws.origins "*" \
    --miner.etherbase "$ETHERBASE" \
    --mine.threads 0 \
    --verbosity "$VERBOSITY" \
    --nodiscover \
    --maxpeers 0 