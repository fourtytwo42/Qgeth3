#!/usr/bin/env bash
# Q Coin Enhanced Cross-Distribution Geth Node Starter
# Universal Linux/Unix node launcher with automatic system detection
# Usage: ./start-geth-enhanced.sh [network] [options]
# Networks: mainnet, testnet, devnet (default: testnet)
# Options: --mining (enable mining with single thread)

# Get script directory for relative imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source system detection library
if [ -f "$SCRIPT_DIR/detect-system.sh" ]; then
    source "$SCRIPT_DIR/detect-system.sh" >/dev/null 2>&1
else
    # Fallback if detect-system.sh not available
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
fi

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
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo -e "\033[1;36mQ Coin Enhanced Cross-Distribution Geth Node Starter\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-geth-enhanced.sh [network] [options]\033[0m"
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
    echo "  🌐 Cross-Distribution: Compatible with multiple Linux distributions"
    echo ""
    if [ -n "${QGETH_DISTRO:-}" ]; then
        echo -e "\033[1;32mDetected System:\033[0m"
        echo "  OS: ${QGETH_OS:-unknown} (${QGETH_DISTRO:-unknown} ${QGETH_DISTRO_VERSION:-})"
        echo "  Architecture: ${QGETH_ARCH:-unknown}"
        echo "  Init System: ${QGETH_INIT_SYSTEM:-unknown}"
        echo ""
    fi
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-geth-enhanced.sh                  # Start testnet node"
    echo "  ./start-geth-enhanced.sh devnet --mining  # Start dev node with mining"
    exit 0
fi

# Cross-platform binary detection
GETH_BINARY=""
for binary_path in "../../geth" "../../geth.bin" "./geth" "./geth.bin"; do
    if [ -f "$binary_path" ] && [ -x "$binary_path" ]; then
        GETH_BINARY="$binary_path"
        break
    fi
done

if [ -z "$GETH_BINARY" ]; then
    log_error "Q Coin Geth binary not found!"
    log_info "Please build geth first using:"
    echo "  ./build-linux.sh geth"
    echo "  # OR"
    echo "  ./build-linux-enhanced.sh geth"
    echo ""
    log_info "Or use the bootstrap script for complete setup:"
    echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y"
    exit 1
fi

# Make sure geth is executable
if [ ! -x "$GETH_BINARY" ]; then
    if command -v chmod >/dev/null 2>&1; then
        chmod +x "$GETH_BINARY"
        log_success "Made geth executable"
    else
        log_warning "chmod not available - binary may not be executable"
    fi
fi

# Network configurations
case $NETWORK in
    testnet)
        CHAINID=73235
        DATADIR="$HOME/.qcoin/testnet"
        GENESIS="../../configs/genesis_quantum_testnet.json"
        PORT=30303
        NAME="Q Coin Testnet"
        ;;
    devnet)
        CHAINID=73234
        DATADIR="$HOME/.qcoin/devnet"
        GENESIS="../../configs/genesis_quantum_dev.json"
        PORT=30305
        NAME="Q Coin Dev Network"
        ;;
    *)
        log_error "Invalid network '$NETWORK'. Use: testnet, devnet"
        exit 1
        ;;
esac

# Enhanced external IP detection with fallbacks
detect_external_ip() {
    local external_ip=""
    local ip_services=(
        "https://ipinfo.io/ip"
        "https://api.ipify.org"
        "https://checkip.amazonaws.com"
        "https://icanhazip.com"
        "https://ifconfig.me/ip"
    )
    
    # Try different methods based on available tools
    for service in "${ip_services[@]}"; do
        if command -v curl >/dev/null 2>&1; then
            external_ip=$(curl -s --max-time 5 "$service" 2>/dev/null | tr -d '[:space:]')
        elif command -v wget >/dev/null 2>&1; then
            external_ip=$(wget -qO- --timeout=5 "$service" 2>/dev/null | tr -d '[:space:]')
        fi
        
        # Validate IP format
        if [[ "$external_ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
            echo "$external_ip"
            return 0
        fi
    done
    
    # If all services fail, try local methods
    if command -v ip >/dev/null 2>&1; then
        # Try to get IP from default route
        external_ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)
        if [[ "$external_ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
            echo "$external_ip"
            return 0
        fi
    fi
    
    # Fallback: return empty (will use NAT discovery)
    echo ""
    return 1
}

EXTERNAL_IP=$(detect_external_ip)
if [ -n "$EXTERNAL_IP" ]; then
    log_success "Detected external IP: $EXTERNAL_IP"
    NAT_CONFIG="extip:$EXTERNAL_IP"
else
    log_warning "Could not detect external IP, using NAT discovery"
    NAT_CONFIG="any"
fi

log_info "Starting $NAME (Chain ID: $CHAINID)"
if [ -n "${QGETH_DISTRO:-}" ]; then
    log_info "System: ${QGETH_OS:-unknown} (${QGETH_DISTRO:-unknown} ${QGETH_DISTRO_VERSION:-})"
fi

# Create data directory with proper ownership
if [ ! -d "$DATADIR" ]; then
    mkdir -p "$DATADIR"
    log_success "Created data directory: $DATADIR"
fi

# Always initialize with genesis file for auto-reset functionality  
log_info "Initializing with genesis file (auto-reset if changed)..."

# Check if genesis file exists
if [ ! -f "$GENESIS" ]; then
    log_error "Genesis file not found: $GENESIS"
    log_info "Available genesis files:"
    ls -la ../../configs/genesis_quantum_*.json 2>/dev/null || echo "No genesis files found!"
    exit 1
fi

# Determine which binary to use for init (prefer raw binary)
INIT_BINARY="$GETH_BINARY"
if [ -f "../../geth.bin" ]; then
    INIT_BINARY="../../geth.bin"
fi

# Always initialize to trigger auto-reset when genesis changes
$INIT_BINARY --datadir "$DATADIR" init "$GENESIS"
if [ $? -ne 0 ]; then
    log_error "Genesis initialization failed!"
    log_info "Debug info:"
    echo "  Genesis file: $GENESIS"
    echo "  Data directory: $DATADIR"
    echo "  Command: $INIT_BINARY --datadir \"$DATADIR\" init \"$GENESIS\""
    exit 1
fi
log_success "Genesis initialization successful"

# Prepare geth arguments with enhanced cross-platform compatibility
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

# Platform-specific optimizations
if [ -n "${QGETH_ARCH:-}" ]; then
    case "$QGETH_ARCH" in
        arm64|arm)
            log_info "Applying ARM optimizations..."
            # Reduce cache sizes for ARM systems
            GETH_ARGS+=("--cache" "256")
            ;;
        amd64)
            # Default cache settings work well for x86_64
            ;;
    esac
fi

# Add mining configuration
if [ "$MINING" = true ]; then
    GETH_ARGS+=("--mine" "--miner.threads" "1" "--miner.etherbase" "0x0000000000000000000000000000000000000001")
    log_info "Mining enabled with 1 thread"
    log_info "NOTE: Use miner_setEtherbase RPC call to set your actual mining address"
else
    # Enable mining interface for external miners (0 threads = no CPU mining)
    GETH_ARGS+=("--mine" "--miner.threads" "0" "--miner.etherbase" "0x0000000000000000000000000000000000000001")
    log_info "Mining interface enabled for external miners (no CPU mining)"
    log_info "NOTE: External miners will set their own coinbase addresses via RPC"
fi

# Add any extra arguments passed to the script
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    GETH_ARGS+=("${EXTRA_ARGS[@]}")
    log_info "Extra arguments: ${EXTRA_ARGS[*]}"
fi

# Display configuration
echo ""
log_info "Network Configuration:"
echo "  Network: $NAME"
echo "  Chain ID: $CHAINID"
echo "  Data Directory: $DATADIR"
echo "  Port: $PORT"
echo "  NAT: $NAT_CONFIG"
echo "  HTTP RPC: http://0.0.0.0:8545"
echo "  WebSocket: ws://0.0.0.0:8546"
echo "  Binary: $GETH_BINARY"
if [ -n "${QGETH_ARCH:-}" ]; then
    echo "  Architecture: $QGETH_ARCH"
fi
echo ""

log_success "Starting Q Coin Geth node..."
log_info "Use Ctrl+C to stop the node"
echo ""

# Enhanced error handling and monitoring
monitor_geth() {
    local geth_pid=$1
    local start_time=$(date +%s)
    
    # Monitor for the first 30 seconds
    for i in {1..30}; do
        if ! kill -0 $geth_pid 2>/dev/null; then
            log_error "Geth process died unexpectedly after $i seconds"
            return 1
        fi
        sleep 1
    done
    
    local run_time=$(($(date +%s) - start_time))
    log_success "Geth has been running successfully for $run_time seconds"
    return 0
}

# Start geth with enhanced error handling
"$GETH_BINARY" "${GETH_ARGS[@]}" &
GETH_PID=$!

# Brief monitoring
if monitor_geth $GETH_PID; then
    # Wait for the main process
    wait $GETH_PID
    EXIT_CODE=$?
else
    EXIT_CODE=1
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    log_error "Geth exited with error code: $EXIT_CODE"
    log_info "Common solutions:"
    echo "  - Check if ports 8545, 8546, $PORT are available"
    echo "  - Make sure you have enough disk space"
    echo "  - Try deleting the data directory and restarting"
    echo "  - Check system compatibility with enhanced script"
    
    # System-specific troubleshooting
    if [ -n "${QGETH_DISTRO:-}" ]; then
        echo ""
        log_info "System-specific troubleshooting for $QGETH_DISTRO:"
        case "$QGETH_DISTRO" in
            ubuntu|debian)
                echo "  - Check AppArmor restrictions: sudo aa-status"
                echo "  - Check systemd journal: journalctl -xe"
                ;;
            centos|rhel|fedora)
                echo "  - Check SELinux status: sestatus"
                echo "  - Check firewall: firewall-cmd --list-all"
                ;;
            arch)
                echo "  - Check systemd journal: journalctl -xe"
                echo "  - Check for missing dependencies: pacman -Qi glibc"
                ;;
            alpine)
                echo "  - Check musl compatibility"
                echo "  - Check available entropy: cat /proc/sys/kernel/random/entropy_avail"
                ;;
        esac
    fi
    
    exit $EXIT_CODE
fi 