#!/bin/bash
# Q Coin Dev Blockchain Reset Tool
# Usage: ./dev-reset-blockchain.sh [options]

CHAIN_ID=73234
DIFFICULTY=200
HELP=false
CONFIRM=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --chain-id)
            CHAIN_ID="$2"
            shift 2
            ;;
        --difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        --yes)
            CONFIRM=true
            shift
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
    echo -e "\033[1;36mQ Coin Dev Blockchain Reset Tool\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./dev-reset-blockchain.sh [options]\033[0m"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --chain-id <id>     - Chain ID for genesis (default: 73234)"
    echo "  --difficulty <num>  - Starting difficulty (default: 200)"
    echo "  --yes               - Skip confirmation prompt"
    echo "  --help              - Show this help message"
    echo ""
    echo -e "\033[1;32mExample:\033[0m"
    echo "  ./dev-reset-blockchain.sh --yes                    # Reset with confirmation"
    echo "  ./dev-reset-blockchain.sh --difficulty 1 --yes     # Reset with difficulty 1"
    exit 0
fi

echo -e "\033[1;36m🔄 Q Coin Dev Blockchain Reset\033[0m"
echo ""

# Determine network name
case $CHAIN_ID in
    73234) NETWORK_NAME="Q Coin Dev Network" ;;
    73235) NETWORK_NAME="Q Coin Testnet" ;;
    73236) NETWORK_NAME="Q Coin Mainnet" ;;
    *) NETWORK_NAME="Custom Network" ;;
esac

echo -e "\033[1;33m⚠️  WARNING: This will completely reset the blockchain!\033[0m"
echo -e "\033[1;37m   Network: $NETWORK_NAME (Chain ID: $CHAIN_ID)\033[0m"
echo -e "\033[1;37m   Starting Difficulty: $DIFFICULTY\033[0m"
echo -e "\033[1;37m   Data Directory: \$HOME/.qcoin/devnet/\033[0m"
echo ""

if [ "$CONFIRM" = false ]; then
    echo -n "Are you sure you want to continue? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Reset cancelled."
        exit 0
    fi
fi

# Remove existing blockchain data
DEVNET_DIR="$HOME/.qcoin/devnet"
if [ -d "$DEVNET_DIR" ]; then
    echo -e "\033[1;33m🗑️  Removing existing blockchain data...\033[0m"
    rm -rf "$DEVNET_DIR"
    echo -e "\033[1;32m✅ Blockchain data removed\033[0m"
else
    echo -e "\033[1;33m📁 No existing blockchain data found\033[0m"
fi

# Build geth if needed
if [ ! -f "quantum-geth/build/bin/geth" ]; then
    echo -e "\033[1;33m🔨 Building Q Coin Geth...\033[0m"
    ./build-linux.sh geth
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31m❌ Build failed!\033[0m"
        exit 1
    fi
fi

# Create dynamic genesis file with specified difficulty
GENESIS_FILE="genesis_quantum_dev_temp.json"
echo -e "\033[1;33m🔧 Creating genesis file with difficulty $DIFFICULTY...\033[0m"

# Convert difficulty to hex
DIFFICULTY_HEX=$(printf "0x%X" $DIFFICULTY)
echo -e "\033[1;37m   Converting difficulty: $DIFFICULTY -> $DIFFICULTY_HEX\033[0m"

# Create genesis JSON with dynamic difficulty
cat > "$GENESIS_FILE" << EOF
{
  "config": {
    "networkId": $CHAIN_ID,
    "chainId": $CHAIN_ID,
    "eip2FBlock": 0,
    "eip7FBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip160Block": 0,
    "eip161FBlock": 0,
    "eip170FBlock": 0,
    "qmpow": {
      "qbits": 16,
      "tcount": 20,
      "lnet": 128,
      "epochLen": 100,
      "testMode": false
    }
  },
  "difficulty": "$DIFFICULTY_HEX",
  "gasLimit": "0x2fefd8",
  "alloc": {
    "0x8b61271473f14c80f2B1381Db9CB13b2d5306200": {
      "balance": "0x3635c9adc5dea00000"
    },
    "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A": {
      "balance": "0x3635c9adc5dea00000"
    },
    "0x1234567890123456789012345678901234567890": {
      "balance": "0x3635c9adc5dea00000"
    }
  }
}
EOF

# Initialize blockchain
echo -e "\033[1;33m🔧 Initializing fresh blockchain...\033[0m"
./quantum-geth/build/bin/geth init "$GENESIS_FILE" --datadir "$DEVNET_DIR"

if [ $? -eq 0 ]; then
    echo -e "\033[1;32m✅ Blockchain reset completed successfully!\033[0m"
    
    # Clean up temporary genesis file
    rm -f "$GENESIS_FILE"
    
    echo ""
    echo -e "\033[1;36m🚀 Next steps:\033[0m"
    echo -e "\033[1;37m   1. Start dev node: ./start-geth.sh devnet\033[0m"
    echo -e "\033[1;37m   2. Start mining: ./start-miner.sh cpu devnet\033[0m"
else
    echo -e "\033[1;31m❌ Blockchain initialization failed!\033[0m"
    rm -f "$GENESIS_FILE"  # Clean up on failure too
    exit 1
fi 