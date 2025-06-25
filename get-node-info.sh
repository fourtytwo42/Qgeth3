#!/bin/bash
# Get Q Coin Node Information for Peer Connections
# This script retrieves your local node's enode info so other nodes can connect

echo -e "\033[36m🔍 Q COIN NODE INFO RETRIEVAL\033[0m"
echo -e "\033[36m==============================\033[0m"
echo ""

# Configuration
RPC_URL="http://127.0.0.1:8545"

echo -e "\033[33m📡 Checking for running Q Coin nodes...\033[0m"

# Function to make RPC call
make_rpc_call() {
    local method=$1
    local params=${2:-"[]"}
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":1}" \
        $RPC_URL 2>/dev/null | jq -r '.result // empty' 2>/dev/null
}

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "\033[31m❌ ERROR: jq is required but not installed\033[0m"
    echo -e "\033[33m   Install with: sudo apt install jq\033[0m"
    exit 1
fi

# Check for running node
echo -e "\033[37m🔍 Checking for Q Coin node on port 8545...\033[0m"
chain_id=$(make_rpc_call "eth_chainId")

if [ -z "$chain_id" ] || [ "$chain_id" = "null" ]; then
    echo -e "\033[31m   ❌ No Q Coin node detected on port 8545\033[0m"
    echo ""
    echo -e "\033[33m🚀 Start a Q Coin node first:\033[0m"
    echo -e "\033[37m   Dev Network:  ./dev-quick-start.sh\033[0m"
    echo -e "\033[37m   Testnet:      ./start-linux-geth.sh\033[0m"
    echo -e "\033[37m   Mainnet:      ./start-linux-geth.sh --mainnet\033[0m"
    exit 1
fi

# Convert hex chain ID to decimal
chain_id_decimal=$((16#${chain_id#0x}))
echo -e "\033[32m   Found network with Chain ID: $chain_id_decimal\033[0m"

# Determine network name
case $chain_id_decimal in
    73234)
        network_name="Q Coin Dev Network"
        echo -e "\033[32m   ✅ Q Coin Dev Network detected!\033[0m"
        ;;
    73235)
        network_name="Q Coin Testnet"
        echo -e "\033[32m   ✅ Q Coin Testnet detected!\033[0m"
        ;;
    73236)
        network_name="Q Coin Mainnet"
        echo -e "\033[32m   ✅ Q Coin Mainnet detected!\033[0m"
        ;;
    *)
        network_name="Unknown Q Coin Network"
        echo -e "\033[33m   ⚠️  Unknown network (Chain ID: $chain_id_decimal)\033[0m"
        ;;
esac

echo ""
echo -e "\033[36m🌐 Network: $network_name\033[0m"
echo ""

# Get node info
echo -e "\033[33m📋 Retrieving node information...\033[0m"

# Get enode info
node_info=$(make_rpc_call "admin_nodeInfo")
if [ -n "$node_info" ] && [ "$node_info" != "null" ]; then
    enode=$(echo "$node_info" | jq -r '.enode // empty')
    node_id=$(echo "$node_info" | jq -r '.id // empty')
    
    if [ -n "$enode" ] && [ "$enode" != "null" ]; then
        echo -e "\033[32m✅ Node enode retrieved!\033[0m"
        
        # Get peer count
        peer_count_hex=$(make_rpc_call "net_peerCount")
        peer_count=$((16#${peer_count_hex#0x}))
        
        # Get listening status
        listening=$(make_rpc_call "net_listening")
        
        # Display results
        echo ""
        echo -e "\033[32m🔗 NODE CONNECTION INFO\033[0m"
        echo -e "\033[32m========================\033[0m"
        echo ""
        echo -e "\033[37mNetwork:     $network_name\033[0m"
        echo -e "\033[37mChain ID:    $chain_id_decimal\033[0m"
        echo -e "\033[37mNode ID:     $node_id\033[0m"
        echo -e "\033[37mListening:   $listening\033[0m"
        echo -e "\033[37mPeers:       $peer_count connected\033[0m"
        echo ""
        echo -e "\033[36m🌐 ENODE (for remote connections):\033[0m"
        echo -e "\033[33m$enode\033[0m"
        echo ""
        
        # Extract IP and port for convenience
        if [[ $enode =~ enode://([^@]+)@([^:]+):([0-9]+) ]]; then
            extracted_node_id="${BASH_REMATCH[1]}"
            ip="${BASH_REMATCH[2]}"
            port="${BASH_REMATCH[3]}"
            
            echo -e "\033[36m📡 Connection Details:\033[0m"
            echo -e "\033[37m   Node ID: $extracted_node_id\033[0m"
            echo -e "\033[37m   IP:      $ip\033[0m"
            echo -e "\033[37m   Port:    $port\033[0m"
            echo ""
            
            # Show commands for remote connection
            echo -e "\033[32m🔧 REMOTE CONNECTION COMMANDS\033[0m"
            echo -e "\033[32m==============================\033[0m"
            echo ""
            echo -e "\033[33mTo connect a remote node to this one, use:\033[0m"
            echo ""
            
            # Replace localhost/127.0.0.1 with actual IP if needed
            if [[ "$ip" == "127.0.0.1" || "$ip" == "localhost" ]]; then
                echo -e "\033[31m⚠️  Note: Replace 127.0.0.1 with your actual IP address!\033[0m"
                echo ""
                echo -e "\033[33mFind your IP with: ip addr show or hostname -I\033[0m"
                echo ""
            fi
            
            echo -e "\033[36mLinux (add to bootnode):\033[0m"
            echo -e "\033[37m   --bootnodes '$enode'\033[0m"
            echo ""
            echo -e "\033[36mWindows (add to bootnode):\033[0m"
            echo -e "\033[37m   --bootnodes \"$enode\"\033[0m"
            echo ""
            echo -e "\033[36mOr add peer manually via console:\033[0m"
            echo -e "\033[37m   admin.addPeer('$enode')\033[0m"
            echo ""
        fi
        
        # Show current peers if any
        if [ $peer_count -gt 0 ]; then
            echo -e "\033[32m👥 CURRENT PEERS\033[0m"
            echo -e "\033[32m=================\033[0m"
            
            peers=$(make_rpc_call "admin_peers")
            if [ -n "$peers" ] && [ "$peers" != "null" ]; then
                echo "$peers" | jq -r '.[] | "\n\u001b[37mPeer: \(.name // "Unknown")\u001b[0m\n\u001b[90m   Enode: \(.enode // "N/A")\u001b[0m\n\u001b[90m   Network: \(.network.remoteAddress // "N/A")\u001b[0m"'
            fi
        fi
        
    else
        echo -e "\033[31m❌ Failed to retrieve enode information\033[0m"
    fi
else
    echo -e "\033[31m❌ Failed to retrieve node information\033[0m"
    echo -e "\033[33m   Make sure your Q Coin node is running with RPC enabled\033[0m"
fi

echo ""
echo -e "\033[34m💡 TIP: Save this enode info to connect other Q Coin nodes!\033[0m"
echo "" 