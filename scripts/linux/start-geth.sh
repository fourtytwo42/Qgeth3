#!/bin/bash
# Q Coin Geth Node Starter
# Usage: ./start-geth.sh [network] [options]
# Networks: mainnet, testnet, devnet (default: testnet)
# Options: --mining (enable mining with single thread)

NETWORK="testnet"
MINING=false
HELP=false
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        testnet|devnet)
            NETWORK="$1"
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
        --*)
            # Collect all other -- arguments as extra args for geth
            EXTRA_ARGS+=("$1")
            shift
            # If this argument has a value, collect it too
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo -e "\033[1;36mQ Coin Geth Node Starter with Auto-Reset\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-geth.sh [network] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mNetworks:\033[0m"
    echo "  testnet   - Q Coin Testnet (Chain ID 73235) [DEFAULT]"
    echo "  devnet    - Q Coin Dev Network (Chain ID 73234)"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --mining  - Enable mining with single thread"
    echo "  --help    - Show this help message"
    echo ""
    echo -e "\033[1;32mFeatures:\033[0m"
    echo "  🔄 Auto-Reset: Automatically detects genesis changes and resets blockchain"
    echo "  🛡️ Minimum Difficulty: Protected against difficulty collapse (minimum 200)"
    echo "  🔗 External Miner Support: Full qmpow API for external mining"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-geth.sh                  # Start testnet node"
    echo "  ./start-geth.sh devnet --mining  # Start dev node with mining"
    exit 0
fi

# Check for geth binary with enhanced diagnostics
GETH_BINARY="../../geth"
GETH_BIN_BINARY="../../geth.bin"

echo -e "\033[1;36m[DEBUG] Checking for Q Geth binary...\033[0m"
echo -e "\033[1;37m[DEBUG] Current working directory: $(pwd)\033[0m"
echo -e "\033[1;37m[DEBUG] Current user: $(whoami)\033[0m"
echo -e "\033[1;37m[DEBUG] Looking for geth at: $GETH_BINARY\033[0m"

# Check if geth wrapper exists
if [ ! -f "$GETH_BINARY" ]; then
    echo -e "\033[1;31m[ERROR] Q Coin Geth wrapper not found at: $GETH_BINARY\033[0m"
    
    # Enhanced diagnostics
    echo -e "\033[1;33m[DEBUG] File system diagnostics:\033[0m"
    echo "  Project directory contents:"
    ls -la ../../ 2>/dev/null | head -10
    echo ""
    
    # Check if geth.bin exists
    if [ -f "$GETH_BIN_BINARY" ]; then
        echo -e "\033[1;33m[INFO] Found geth.bin but missing geth wrapper\033[0m"
        echo -e "\033[1;33m[INFO] Attempting to create wrapper...\033[0m"
        
        # Try to create wrapper if geth.bin exists
        cat > "$GETH_BINARY" << 'EOF'
#!/bin/bash
# Q Coin Geth Wrapper - Auto-generated
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_GETH="$SCRIPT_DIR/geth.bin"
exec "$REAL_GETH" "$@"
EOF
        
        if command -v chmod >/dev/null 2>&1; then
            chmod +x "$GETH_BINARY"
            echo -e "\033[1;32m[SUCCESS] Created geth wrapper\033[0m"
        else
            echo -e "\033[1;31m[ERROR] Could not make wrapper executable\033[0m"
        fi
    else
        echo -e "\033[1;31m[ERROR] Neither geth nor geth.bin found!\033[0m"
        echo -e "\033[1;33m[INFO] Available files in project root:\033[0m"
        find ../../ -maxdepth 1 -name "*geth*" 2>/dev/null || echo "  No geth files found"
    fi
    
    # Still missing after attempt to create
    if [ ! -f "$GETH_BINARY" ]; then
        echo -e "\033[1;33m[INFO] Please build geth first using:\033[0m"
        echo "  ./build-linux.sh geth"
        echo ""
        echo -e "\033[1;33m[INFO] Or use the bootstrap script for complete setup:\033[0m"
        echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y"
        exit 1
    fi
fi

# Make sure geth is executable
if [ ! -x "$GETH_BINARY" ]; then
    chmod +x "$GETH_BINARY"
    echo -e "\033[1;32m[SUCCESS] Made geth executable\033[0m"
fi

# Network configurations
case $NETWORK in
    testnet)
        CHAINID=73235
        DATADIR="$HOME/.qcoin/testnet"
        GENESIS="../../configs/genesis_quantum_testnet.json"
        PORT=30303
        NAME="Q Coin Testnet"
        BOOTNODE_PORT=30303
        ;;
    devnet)
        CHAINID=73234
        DATADIR="$HOME/.qcoin/devnet"
        GENESIS="../../configs/genesis_quantum_dev.json"
        PORT=30305
        NAME="Q Coin Dev Network"
        BOOTNODE_PORT=30305
        ;;
    *)
        echo -e "\033[1;31m[ERROR] Invalid network '$NETWORK'. Use: testnet, devnet\033[0m"
        exit 1
        ;;
esac

# Detect external IP for proper enode advertising
EXTERNAL_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || curl -s https://api.ipify.org 2>/dev/null || curl -s https://checkip.amazonaws.com 2>/dev/null)
if [ -z "$EXTERNAL_IP" ]; then
    echo -e "\033[1;33m[WARNING] Could not detect external IP, using NAT discovery\033[0m"
    NAT_CONFIG="any"
else
    echo -e "\033[1;32m[NETWORK] Detected external IP: $EXTERNAL_IP\033[0m"
    NAT_CONFIG="extip:$EXTERNAL_IP"
fi

# Bootnodes are automatically selected based on network ID (chainid)
# No need to specify --bootnodes flag - geth will use params/bootnodes_qcoin.go

echo -e "\033[1;36m[START] Starting $NAME (Chain ID: $CHAINID)\033[0m"

# Create data directory if it doesn't exist
if [ ! -d "$DATADIR" ]; then
    mkdir -p "$DATADIR"
    echo -e "\033[1;32m[CREATED] Data directory: $DATADIR\033[0m"
fi

# CRITICAL: Always initialize with genesis file for auto-reset functionality  
echo -e "\033[1;33m[INIT] Initializing with genesis file (auto-reset if changed)...\033[0m"

# Check if genesis file exists
if [ ! -f "$GENESIS" ]; then
    echo -e "\033[1;31m[ERROR] Genesis file not found: $GENESIS\033[0m"
    echo -e "\033[1;33m[INFO] Available genesis files:\033[0m"
    ls -la ../../configs/genesis_quantum_*.json 2>/dev/null || echo "No genesis files found!"
    exit 1
fi

# Always initialize to trigger auto-reset when genesis changes
if [ -f "$GETH_BIN_BINARY" ]; then
    # Use geth.bin directly for initialization (more reliable)
    $GETH_BIN_BINARY --datadir "$DATADIR" init "$GENESIS"
else
    # Fallback to wrapper if geth.bin not found
    $GETH_BINARY --datadir "$DATADIR" init "$GENESIS"
fi

if [ $? -ne 0 ]; then
    echo -e "\033[1;31m[ERROR] Genesis initialization failed!\033[0m"
    echo -e "\033[1;33m[DEBUG] Debug info:\033[0m"
    echo "  Genesis file: $GENESIS"
    echo "  Data directory: $DATADIR"
    echo "  Binary used: $([ -f "$GETH_BIN_BINARY" ] && echo "$GETH_BIN_BINARY" || echo "$GETH_BINARY")"
    exit 1
fi
echo -e "\033[1;32m[SUCCESS] Genesis initialization successful\033[0m"

# Prepare geth arguments
GETH_ARGS=(
    "--datadir" "$DATADIR"
    "--networkid" "$CHAINID"
    "--port" "$PORT"
    "--nat" "$NAT_CONFIG"
    "--http"
    "--http.addr" "0.0.0.0"
    "--http.port" "8545"
    "--http.corsdomain" "*"
    "--http.api" "eth,net,web3,personal,admin,txpool,miner,qmpow"
    "--ws"
    "--ws.addr" "0.0.0.0"
    "--ws.port" "8546"
    "--ws.origins" "*"
    "--ws.api" "eth,net,web3,personal,admin,txpool,miner,qmpow"
    "--maxpeers" "25"
    "--verbosity" "3"
)

# Add mining if requested
if [ "$MINING" = true ]; then
    GETH_ARGS+=("--mine" "--miner.threads" "1" "--miner.etherbase" "0x0000000000000000000000000000000000000001")
    echo -e "\033[1;33m[MINING] Mining enabled with 1 thread\033[0m"
    echo -e "\033[1;36mNOTE: Use miner_setEtherbase RPC call to set your actual mining address\033[0m"
else
    # Enable mining interface for external miners (0 threads = no CPU mining)
    GETH_ARGS+=("--mine" "--miner.threads" "0" "--miner.etherbase" "0x0000000000000000000000000000000000000001")
    echo -e "\033[1;33m[INTERFACE] Mining interface enabled for external miners (no CPU mining)\033[0m"
    echo -e "\033[1;36mNOTE: External miners will set their own coinbase addresses via RPC\033[0m"
fi

# Add any extra arguments passed to the script
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    GETH_ARGS+=("${EXTRA_ARGS[@]}")
    echo -e "\033[1;36m[CONFIG] Extra arguments: ${EXTRA_ARGS[*]}\033[0m"
fi

echo -e "\033[1;37m[NETWORK] Network: $NAME\033[0m"
echo -e "\033[1;37m[CHAIN] Chain ID: $CHAINID\033[0m"
echo -e "\033[1;37m[DATA] Data Directory: $DATADIR\033[0m"
echo -e "\033[1;37m[PORT] Port: $PORT\033[0m"
echo -e "\033[1;37m[NAT] NAT: $NAT_CONFIG\033[0m"
echo -e "\033[1;37m[HTTP] HTTP RPC: http://0.0.0.0:8545\033[0m"
echo -e "\033[1;37m[WS] WebSocket: ws://0.0.0.0:8546\033[0m"
echo -e "\033[1;37m[BOOT] Bootnodes: Auto-selected for $NETWORK network\033[0m"
echo ""
echo -e "\033[1;32m[LAUNCH] Starting Q Coin Geth node...\033[0m"
echo -e "\033[1;33m[INFO] Use Ctrl+C to stop the node\033[0m"
echo ""

# Start geth with error handling
$GETH_BINARY "${GETH_ARGS[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "\033[1;31m[ERROR] Geth exited with error code: $EXIT_CODE\033[0m"
    echo -e "\033[1;33m[HELP] Common solutions:\033[0m"
    echo "  - Check if ports 8545, 8546, $PORT are available"
    echo "  - Make sure you have enough disk space"
    echo "  - Try deleting the data directory and restarting"
    exit $EXIT_CODE
fi 