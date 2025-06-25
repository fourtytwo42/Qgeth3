#!/bin/bash
# Q Coin Node Info Tool
# Usage: ./get-node-info.sh [options]

GETH_RPC="http://localhost:8545"
HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --geth-rpc)
            GETH_RPC="$2"
            shift 2
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

if [ "$HELP" = true ]; then
    echo -e "\033[1;36mQ Coin Node Info Tool\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./get-node-info.sh [options]\033[0m"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --geth-rpc <url>  - Geth RPC endpoint (default: http://localhost:8545)"
    echo "  --help            - Show this help message"
    echo ""
    echo -e "\033[1;32mExample:\033[0m"
    echo "  ./get-node-info.sh                           # Get local node info"
    echo "  ./get-node-info.sh --geth-rpc http://192.168.1.100:8545"
    exit 0
fi

echo -e "\033[1;36m🔍 Q Coin Node Information\033[0m"
echo ""

# Test connection
echo -e "\033[1;33m📡 Testing connection to Geth RPC...\033[0m"
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
    "$GETH_RPC" 2>/dev/null)

if [ $? -ne 0 ] || ! echo "$RESPONSE" | grep -q '"result"'; then
    echo -e "\033[1;31m❌ Cannot connect to Geth RPC at $GETH_RPC\033[0m"
    echo -e "\033[1;33m   Make sure Geth is running and RPC is enabled!\033[0m"
    exit 1
fi

echo -e "\033[1;32m✅ Connected to Geth RPC\033[0m"
echo ""

# Get network info
NETWORK_ID=$(echo "$RESPONSE" | grep -o '"[0-9]*"' | tr -d '"')
case $NETWORK_ID in
    73234) NETWORK_NAME="Q Coin Dev Network" ;;
    73235) NETWORK_NAME="Q Coin Testnet" ;;
    73236) NETWORK_NAME="Q Coin Mainnet" ;;
    *) NETWORK_NAME="Unknown Network" ;;
esac

# Get node info
NODEINFO_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"admin_nodeInfo","params":[],"id":1}' \
    "$GETH_RPC" 2>/dev/null)

# Get peer count
PEER_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
    "$GETH_RPC" 2>/dev/null)

# Extract node info
if echo "$NODEINFO_RESPONSE" | grep -q '"enode"'; then
    ENODE=$(echo "$NODEINFO_RESPONSE" | grep -o '"enode://[^"]*"' | tr -d '"')
    NODE_ID=$(echo "$ENODE" | cut -d'@' -f1 | cut -d'/' -f3)
    IP_PORT=$(echo "$ENODE" | cut -d'@' -f2)
    IP=$(echo "$IP_PORT" | cut -d':' -f1)
    PORT=$(echo "$IP_PORT" | cut -d':' -f2)
else
    ENODE="Not available"
    NODE_ID="Not available"
    IP="Not available"
    PORT="Not available"
fi

# Extract peer count
if echo "$PEER_RESPONSE" | grep -q '"result"'; then
    PEER_COUNT_HEX=$(echo "$PEER_RESPONSE" | grep -o '"0x[0-9a-fA-F]*"' | tr -d '"')
    PEER_COUNT=$((16#${PEER_COUNT_HEX#0x}))
else
    PEER_COUNT="Unknown"
fi

# Display information
echo -e "\033[1;36m📊 Node Information:\033[0m"
echo -e "\033[1;37m  Network: $NETWORK_NAME (ID: $NETWORK_ID)\033[0m"
echo -e "\033[1;37m  Node ID: $NODE_ID\033[0m"
echo -e "\033[1;37m  IP Address: $IP\033[0m"
echo -e "\033[1;37m  Port: $PORT\033[0m"
echo -e "\033[1;37m  Connected Peers: $PEER_COUNT\033[0m"
echo ""
echo -e "\033[1;36m🔗 Full Enode URL:\033[0m"
echo -e "\033[1;32m$ENODE\033[0m"
echo ""
echo -e "\033[1;36m📡 Default Bootnode:\033[0m"
echo -e "\033[1;35menode://0bc243936ebc13ebf57895dff1321695064ae4b0ac0c1e047d52d695c396b64c52847f852a9738f0d079af4ba109dfceafd1cf0924587b151765834caf13e5fd@69.243.132.233:30305\033[0m"
echo ""
echo -e "\033[1;33m💡 To connect other nodes to this one, use:\033[0m"
echo -e "\033[1;37m   ./connect-peers.sh '$ENODE'\033[0m" 