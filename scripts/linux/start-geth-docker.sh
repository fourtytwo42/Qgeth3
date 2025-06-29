#!/bin/bash
# Q Geth Docker Node Starter
# Usage: ./start-geth-docker.sh [network] [options]
# Networks: testnet, devnet (default: testnet)
# Options: --mining (enable mining mode), --build (rebuild containers)

NETWORK="testnet"
MINING=false
BUILD=false
DETACH=true
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
        --build)
            BUILD=true
            shift
            ;;
        --foreground|-f)
            DETACH=false
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        --*)
            # Collect all other -- arguments as extra args
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
    echo -e "\033[1;36mQ Geth Docker Node Starter\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./start-geth-docker.sh [network] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mNetworks:\033[0m"
    echo "  testnet   - Q Geth Testnet (Chain ID 73235) [DEFAULT]"
    echo "  devnet    - Q Geth Dev Network (Chain ID 73234)"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --mining      - Enable mining mode (uses qgeth-miner container)"
    echo "  --build       - Rebuild Docker containers before starting"
    echo "  --foreground  - Run in foreground (see logs directly)"
    echo "  --help        - Show this help message"
    echo ""
    echo -e "\033[1;32mDocker Features:\033[0m"
    echo "  🐳 Cross-Platform: Works on Linux, Windows, macOS"
    echo "  🔒 Isolated Environment: Clean, secure container deployment"
    echo "  📊 Health Checks: Built-in container health monitoring"
    echo "  💾 Persistent Data: Blockchain data survives container restarts"
    echo "  🔗 MetaMask Ready: HTTP RPC (8545) and WebSocket (8546) exposed"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./start-geth-docker.sh                    # Start testnet node"
    echo "  ./start-geth-docker.sh devnet --mining    # Start dev node with mining"
    echo "  ./start-geth-docker.sh --build            # Rebuild and start testnet"
    echo "  ./start-geth-docker.sh --foreground       # Start with visible logs"
    echo ""
    echo -e "\033[1;32mManagement Commands:\033[0m"
    echo "  docker-compose ps                         # Check container status"
    echo "  docker-compose logs -f qgeth-testnet      # View logs"
    echo "  docker-compose stop                       # Stop containers"
    echo "  docker-compose down                       # Stop and remove containers"
    exit 0
fi

# Check if Docker is available
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "\033[1;31m[ERROR] Docker is not installed!\033[0m"
        echo ""
        echo -e "\033[1;33m[INFO] Install Docker:\033[0m"
        echo "  Ubuntu/Debian: curl -fsSL https://get.docker.com | sudo sh"
        echo "  Fedora: sudo dnf install docker docker-compose"
        echo "  Arch: sudo pacman -S docker docker-compose"
        echo ""
        echo -e "\033[1;33m[INFO] After installation:\033[0m"
        echo "  sudo systemctl start docker"
        echo "  sudo usermod -aG docker \$USER"
        echo "  newgrp docker"
        exit 1
    fi

    if ! command -v docker-compose >/dev/null 2>&1; then
        echo -e "\033[1;31m[ERROR] Docker Compose is not installed!\033[0m"
        echo ""
        echo -e "\033[1;33m[INFO] Install Docker Compose:\033[0m"
        echo "  sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose"
        echo "  sudo chmod +x /usr/local/bin/docker-compose"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "\033[1;31m[ERROR] Docker daemon is not running!\033[0m"
        echo ""
        echo -e "\033[1;33m[INFO] Start Docker daemon:\033[0m"
        echo "  sudo systemctl start docker"
        echo ""
        echo -e "\033[1;33m[INFO] Enable Docker to start on boot:\033[0m"
        echo "  sudo systemctl enable docker"
        exit 1
    fi
}

# Navigate to project root (where docker-compose.yml is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "\033[1;36m[DEBUG] Project root: $PROJECT_ROOT\033[0m"

if [ ! -f "$PROJECT_ROOT/docker-compose.yml" ]; then
    echo -e "\033[1;31m[ERROR] docker-compose.yml not found in project root!\033[0m"
    echo -e "\033[1;33m[INFO] Expected location: $PROJECT_ROOT/docker-compose.yml\033[0m"
    exit 1
fi

cd "$PROJECT_ROOT"

# Check Docker installation
check_docker

# Determine container and port configuration
case $NETWORK in
    testnet)
        if [ "$MINING" = true ]; then
            CONTAINER_NAME="qgeth-miner"
            SERVICE_NAME="qgeth-miner"
            HTTP_PORT=8547
            WS_PORT=8548
            P2P_PORT=30304
            PROFILE_ARG="--profile mining"
        else
            CONTAINER_NAME="qgeth-testnet"
            SERVICE_NAME="qgeth-testnet"
            HTTP_PORT=8545
            WS_PORT=8546
            P2P_PORT=30303
            PROFILE_ARG=""
        fi
        CHAINID=73235
        NAME="Q Geth Testnet"
        ;;
    devnet)
        CONTAINER_NAME="qgeth-dev"
        SERVICE_NAME="qgeth-dev"
        HTTP_PORT=8549
        WS_PORT=8550
        P2P_PORT=30305
        CHAINID=73234
        NAME="Q Geth Dev Network"
        PROFILE_ARG="--profile dev"
        if [ "$MINING" = true ]; then
            echo -e "\033[1;33m[INFO] Dev network has mining enabled by default\033[0m"
        fi
        ;;
    *)
        echo -e "\033[1;31m[ERROR] Invalid network '$NETWORK'. Use: testnet, devnet\033[0m"
        exit 1
        ;;
esac

echo -e "\033[1;36m[START] Starting $NAME (Chain ID: $CHAINID) with Docker\033[0m"

# Build containers if requested or if they don't exist
if [ "$BUILD" = true ]; then
    echo -e "\033[1;33m[BUILD] Building Docker containers...\033[0m"
    docker-compose build --no-cache
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m[ERROR] Docker build failed!\033[0m"
        exit 1
    fi
    echo -e "\033[1;32m[SUCCESS] Docker containers built successfully\033[0m"
elif ! docker images qgeth:latest >/dev/null 2>&1; then
    echo -e "\033[1;33m[BUILD] Q Geth Docker image not found, building...\033[0m"
    docker-compose build
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m[ERROR] Docker build failed!\033[0m"
        exit 1
    fi
fi

# Stop any existing containers
echo -e "\033[1;33m[CLEANUP] Stopping existing containers...\033[0m"
docker-compose down --remove-orphans >/dev/null 2>&1

# Prepare docker-compose command
COMPOSE_CMD="docker-compose"
if [ -n "$PROFILE_ARG" ]; then
    COMPOSE_CMD="$COMPOSE_CMD $PROFILE_ARG"
fi

if [ "$DETACH" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD up -d $SERVICE_NAME"
    echo -e "\033[1;33m[INFO] Starting container in background...\033[0m"
else
    COMPOSE_CMD="$COMPOSE_CMD up $SERVICE_NAME"
    echo -e "\033[1;33m[INFO] Starting container in foreground (Ctrl+C to stop)...\033[0m"
fi

# Start the container
echo -e "\033[1;37m[NETWORK] Network: $NAME\033[0m"
echo -e "\033[1;37m[CHAIN] Chain ID: $CHAINID\033[0m"
echo -e "\033[1;37m[CONTAINER] Container: $CONTAINER_NAME\033[0m"
echo -e "\033[1;37m[HTTP] HTTP RPC: http://localhost:$HTTP_PORT\033[0m"
echo -e "\033[1;37m[WS] WebSocket: ws://localhost:$WS_PORT\033[0m"
echo -e "\033[1;37m[P2P] P2P Port: $P2P_PORT\033[0m"
if [ "$MINING" = true ] || [ "$NETWORK" = "devnet" ]; then
    echo -e "\033[1;37m[MINING] Mining: Enabled\033[0m"
else
    echo -e "\033[1;37m[MINING] Mining Interface: Available for external miners\033[0m"
fi
echo ""

# Execute the docker-compose command
eval $COMPOSE_CMD
if [ $? -ne 0 ]; then
    echo -e "\033[1;31m[ERROR] Failed to start Docker container!\033[0m"
    exit 1
fi

if [ "$DETACH" = true ]; then
    echo -e "\033[1;32m[SUCCESS] Q Geth container started successfully!\033[0m"
    echo ""
    echo -e "\033[1;33m[INFO] Container Management:\033[0m"
    echo "  Check status: docker-compose ps"
    echo "  View logs:    docker-compose logs -f $SERVICE_NAME"
    echo "  Stop:         docker-compose stop $SERVICE_NAME"
    echo "  Remove:       docker-compose down"
    echo ""
    echo -e "\033[1;33m[INFO] MetaMask Connection:\033[0m"
    echo "  Network: Custom RPC"
    echo "  RPC URL: http://localhost:$HTTP_PORT"
    echo "  Chain ID: $CHAINID"
    echo "  Currency: QGC"
    echo ""
    echo -e "\033[1;33m[INFO] Mining Connection:\033[0m"
    echo "  HTTP RPC: http://localhost:$HTTP_PORT"
    echo "  WebSocket: ws://localhost:$WS_PORT"
    echo ""
    
    # Wait a moment for container to start, then check health
    sleep 3
    echo -e "\033[1;36m[HEALTH] Checking container health...\033[0m"
    if docker-compose ps | grep -q "Up.*healthy"; then
        echo -e "\033[1;32m[HEALTH] ✅ Container is healthy and ready!\033[0m"
    elif docker-compose ps | grep -q "Up.*starting"; then
        echo -e "\033[1;33m[HEALTH] 🔄 Container is starting... (health check pending)\033[0m"
    else
        echo -e "\033[1;31m[HEALTH] ❌ Container may have issues. Check logs: docker-compose logs $SERVICE_NAME\033[0m"
    fi
    
    # Test API connectivity
    echo -e "\033[1;36m[TEST] Testing API connectivity...\033[0m"
    sleep 2
    if curl -s http://localhost:$HTTP_PORT >/dev/null 2>&1; then
        echo -e "\033[1;32m[TEST] ✅ HTTP RPC API is accessible\033[0m"
    else
        echo -e "\033[1;33m[TEST] ⏳ HTTP RPC API not yet ready (normal during startup)\033[0m"
    fi
fi

echo -e "\033[1;32m[COMPLETE] Q Geth Docker deployment complete!\033[0m" 