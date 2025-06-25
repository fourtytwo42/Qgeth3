#!/bin/bash
# Connect to Remote Q Coin Peer
# This script adds a peer to your running Q Coin node

# Default values
RPC_URL="http://127.0.0.1:8545"
ENODE=""
SHOW_HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --rpc-url)
            RPC_URL="$2"
            shift 2
            ;;
        *)
            if [[ -z "$ENODE" ]]; then
                ENODE="$1"
            else
                echo -e "\033[31m❌ Unknown option: $1\033[0m"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ "$SHOW_HELP" == "true" ]]; then
    echo -e "\033[36m🔗 Q COIN PEER CONNECTION\033[0m"
    echo -e "\033[36m=========================\033[0m"
    echo ""
    echo -e "\033[33mConnects your Q Coin node to a remote peer\033[0m"
    echo ""
    echo -e "\033[37mUsage:\033[0m"
    echo -e "\033[37m  ./connect-peer.sh <enode>\033[0m"
    echo ""
    echo -e "\033[37mExample:\033[0m"
    echo -e "\033[37m  ./connect-peer.sh 'enode://89df9647...@192.168.1.100:30303'\033[0m"
    echo ""
    echo -e "\033[37mOptions:\033[0m"
    echo -e "\033[37m  --rpc-url    RPC endpoint (default: http://127.0.0.1:8545)\033[0m"
    echo -e "\033[37m  --help, -h   Show this help message\033[0m"
    echo ""
    echo -e "\033[33mGet enode info from remote node:\033[0m"
    echo -e "\033[37m  ./get-node-info.sh\033[0m"
    echo ""
    exit 0
fi

if [[ -z "$ENODE" ]]; then
    echo -e "\033[31m❌ ERROR: Enode is required!\033[0m"
    echo ""
    echo -e "\033[33mUsage: ./connect-peer.sh <enode>\033[0m"
    echo -e "\033[33mUse --help for more information\033[0m"
    exit 1
fi

echo -e "\033[36m🔗 Q COIN PEER CONNECTION\033[0m"
echo -e "\033[36m=========================\033[0m"
echo ""

# Validate enode format
if [[ ! $ENODE =~ ^enode://[a-fA-F0-9]{128}@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$ ]]; then
    echo -e "\033[31m❌ Invalid enode format!\033[0m"
    echo ""
    echo -e "\033[33mExpected format:\033[0m"
    echo -e "\033[37m  enode://[128-char-hex]@[ip]:[port]\033[0m"
    echo ""
    echo -e "\033[33mExample:\033[0m"
    echo -e "\033[37m  enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.1.100:30303\033[0m"
    echo ""
    exit 1
fi

# Extract connection details
if [[ $ENODE =~ enode://([^@]+)@([^:]+):([0-9]+) ]]; then
    node_id="${BASH_REMATCH[1]}"
    ip="${BASH_REMATCH[2]}"
    port="${BASH_REMATCH[3]}"
    
    echo -e "\033[33m🎯 Target Peer:\033[0m"
    echo -e "\033[37m   Node ID: $node_id\033[0m"
    echo -e "\033[37m   IP:      $ip\033[0m"
    echo -e "\033[37m   Port:    $port\033[0m"
    echo ""
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "\033[31m❌ ERROR: jq is required but not installed\033[0m"
    echo -e "\033[33m   Install with: sudo apt install jq\033[0m"
    exit 1
fi

# Function to make RPC call
make_rpc_call() {
    local method=$1
    local params=${2:-"[]"}
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":1}" \
        $RPC_URL 2>/dev/null
}

# Check if local node is running
echo -e "\033[33m🔍 Checking local Q Coin node...\033[0m"
chain_response=$(make_rpc_call "eth_chainId")
chain_id=$(echo "$chain_response" | jq -r '.result // empty')

if [[ -z "$chain_id" || "$chain_id" == "null" ]]; then
    echo -e "\033[31m❌ No Q Coin node detected at $RPC_URL\033[0m"
    echo ""
    echo -e "\033[33mStart a Q Coin node first:\033[0m"
    echo -e "\033[37m  Dev Network:  ./dev-quick-start.sh\033[0m"
    echo -e "\033[37m  Testnet:      ./start-linux-geth.sh\033[0m"
    echo -e "\033[37m  Mainnet:      ./start-linux-geth.sh --mainnet\033[0m"
    exit 1
fi

chain_id_decimal=$((16#${chain_id#0x}))
case $chain_id_decimal in
    73234)
        network_name="Q Coin Dev Network"
        ;;
    73235)
        network_name="Q Coin Testnet"
        ;;
    73236)
        network_name="Q Coin Mainnet"
        ;;
    *)
        network_name="Unknown Q Coin Network"
        ;;
esac

echo -e "\033[32m✅ Local node detected: $network_name (Chain ID: $chain_id_decimal)\033[0m"
echo ""

# Get current peer count
peer_count_response=$(make_rpc_call "net_peerCount")
peer_count_hex=$(echo "$peer_count_response" | jq -r '.result // "0x0"')
peers_before=$((16#${peer_count_hex#0x}))

echo -e "\033[37m📊 Current peers: $peers_before\033[0m"
echo ""

# Attempt to add peer
echo -e "\033[33m🔗 Attempting to connect to peer...\033[0m"
add_response=$(make_rpc_call "admin_addPeer" "[\"$ENODE\"]")
add_result=$(echo "$add_response" | jq -r '.result // false')

if [[ "$add_result" == "true" ]]; then
    echo -e "\033[32m✅ Peer connection initiated successfully!\033[0m"
    echo ""
    
    # Wait a moment for connection to establish
    echo -e "\033[33m⏳ Waiting for connection to establish...\033[0m"
    sleep 3
    
    # Check new peer count
    peer_count_response_after=$(make_rpc_call "net_peerCount")
    peer_count_hex_after=$(echo "$peer_count_response_after" | jq -r '.result // "0x0"')
    peers_after=$((16#${peer_count_hex_after#0x}))
    
    echo -e "\033[37m📊 Peer count after connection: $peers_after\033[0m"
    
    if [[ $peers_after -gt $peers_before ]]; then
        echo -e "\033[32m🎉 New peer connected successfully!\033[0m"
        
        # Show current peers
        peers_response=$(make_rpc_call "admin_peers")
        peers_data=$(echo "$peers_response" | jq '.result // []')
        
        if [[ "$peers_data" != "[]" ]]; then
            echo ""
            echo -e "\033[36m👥 Current Peers:\033[0m"
            echo "$peers_data" | jq -r '.[] | "   • \(.name // "Unknown")\n     \(.network.remoteAddress // "N/A")"'
        fi
    else
        echo -e "\033[33m⚠️  Peer added but connection not yet established\033[0m"
        echo -e "\033[90m   This is normal - connections may take time to establish\033[0m"
    fi
elif [[ "$add_result" == "false" ]]; then
    echo -e "\033[33m⚠️  Peer already known or connection failed\033[0m"
    echo -e "\033[90m   The peer may already be in the node table\033[0m"
else
    echo -e "\033[31m❌ Failed to add peer\033[0m"
    error_msg=$(echo "$add_response" | jq -r '.error.message // "Unknown error"')
    if [[ "$error_msg" != "Unknown error" ]]; then
        echo -e "\033[31m   Error: $error_msg\033[0m"
    fi
fi

echo ""
echo -e "\033[34m💡 TIP: Use ./get-node-info.sh to check connection status\033[0m"
echo "" 