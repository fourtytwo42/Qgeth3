#!/bin/bash
# Q Coin Peer Connection Tool
# Usage: ./connect-peers.sh [enode] [options]

ENODE=""
GETH_RPC="http://localhost:8545"
LIST=false
HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --geth-rpc)
            GETH_RPC="$2"
            shift 2
            ;;
        --list)
            LIST=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        enode://*)
            ENODE="$1"
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
    echo -e "\033[1;36mQ Coin Peer Connection Tool\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./connect-peers.sh [enode] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mArguments:\033[0m"
    echo "  enode             - Enode URL to connect to"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --geth-rpc <url>  - Geth RPC endpoint (default: http://localhost:8545)"
    echo "  --list            - List current peers"
    echo "  --help            - Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./connect-peers.sh --list                                    # List current peers"
    echo "  ./connect-peers.sh 'enode://abc123@192.168.1.100:30305'     # Connect to peer"
    echo ""
    echo -e "\033[1;35mDefault Bootnode:\033[0m"
    echo "  enode://0bc243936ebc13ebf57895dff1321695064ae4b0ac0c1e047d52d695c396b64c52847f852a9738f0d079af4ba109dfceafd1cf0924587b151765834caf13e5fd@69.243.132.233:30305"
    exit 0
fi

echo -e "\033[1;36m🔗 Q Coin Peer Connection Tool\033[0m"
echo ""

# Test Geth connection
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

# List peers if requested or no enode provided
if [ "$LIST" = true ] || [ -z "$ENODE" ]; then
    echo -e "\033[1;36m📋 Current Peers:\033[0m"
    
    PEERS_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"admin_peers","params":[],"id":1}' \
        "$GETH_RPC" 2>/dev/null)
    
    if echo "$PEERS_RESPONSE" | grep -q '"result"'; then
        # Count peers
        PEER_COUNT=$(echo "$PEERS_RESPONSE" | grep -o '"enode"' | wc -l)
        
        if [ "$PEER_COUNT" -gt 0 ]; then
            echo "  Found $PEER_COUNT connected peer(s):"
            echo ""
            
            # Extract and display peer info (simplified without jq)
            COUNT=1
            while read -r line; do
                if echo "$line" | grep -q '"remoteAddress"'; then
                    REMOTE_ADDR=$(echo "$line" | grep -o '"[^"]*"' | tail -1 | tr -d '"')
                    echo -e "  [$COUNT] \033[1;37m$REMOTE_ADDR\033[0m"
                    COUNT=$((COUNT + 1))
                fi
            done <<< "$(echo "$PEERS_RESPONSE" | tr ',' '\n')"
        else
            echo "  No peers connected"
        fi
    else
        echo "  Failed to get peer information"
    fi
    
    if [ -z "$ENODE" ]; then
        exit 0
    fi
    echo ""
fi

# Connect to peer if enode provided
if [ -n "$ENODE" ]; then
    echo -e "\033[1;33m🔗 Connecting to peer...\033[0m"
    echo -e "\033[1;37m   Enode: $ENODE\033[0m"
    echo ""
    
    # Validate enode format
    if ! echo "$ENODE" | grep -q "^enode://[0-9a-fA-F]\{128\}@[0-9a-zA-Z.-]\+:[0-9]\+$"; then
        echo -e "\033[1;31m❌ Invalid enode format!\033[0m"
        echo -e "\033[1;33m   Expected format: enode://pubkey@ip:port\033[0m"
        exit 1
    fi
    
    # Send add peer request
    ADD_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        --data "{\"jsonrpc\":\"2.0\",\"method\":\"admin_addPeer\",\"params\":[\"$ENODE\"],\"id\":1}" \
        "$GETH_RPC" 2>/dev/null)
    
    if echo "$ADD_RESPONSE" | grep -q '"result":true'; then
        echo -e "\033[1;32m✅ Peer connection request sent successfully!\033[0m"
        echo -e "\033[1;33m   Note: It may take a few moments to establish the connection.\033[0m"
    else
        echo -e "\033[1;31m❌ Failed to send peer connection request!\033[0m"
        exit 1
    fi
    
    echo ""
    echo -e "\033[1;33m🔍 Checking connection status in 5 seconds...\033[0m"
    sleep 5
    
    # Check if peer connected
    PEER_IP=$(echo "$ENODE" | cut -d'@' -f2 | cut -d':' -f1)
    PEERS_CHECK=$(curl -s -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"admin_peers","params":[],"id":1}' \
        "$GETH_RPC" 2>/dev/null)
    
    if echo "$PEERS_CHECK" | grep -q "$PEER_IP"; then
        echo -e "\033[1;32m✅ Successfully connected to peer!\033[0m"
        REMOTE_ADDR=$(echo "$PEERS_CHECK" | grep -A5 -B5 "$PEER_IP" | grep '"remoteAddress"' | grep -o '"[^"]*"' | tail -1 | tr -d '"')
        echo -e "\033[1;37m   Remote Address: $REMOTE_ADDR\033[0m"
    else
        echo -e "\033[1;33m⚠️  Peer not yet connected. This is normal and may take more time.\033[0m"
        echo -e "\033[1;33m   Use './connect-peers.sh --list' to check connection status later.\033[0m"
    fi
fi 