#!/bin/bash

# Quantum-Geth Blockchain Reset Utility
# Linux/macOS version of reset-blockchain.ps1

set -e  # Exit on any error

# Default configuration
DIFFICULTY=100
FORCE=false
DATADIR="qdata_quantum"
NETWORKID=73428
ETHERBASE="0x8b61271473f14c80f2B1381Db9CB13b2d5306200"
BALANCE="300000000000000000000000"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --difficulty DIFF    Set starting difficulty (default: 100)"
    echo "  --force             Skip confirmation prompt"
    echo "  --datadir DIR       Set data directory (default: qdata_quantum)"
    echo "  --networkid ID      Set network ID (default: 73428)"
    echo "  --etherbase ADDR    Set etherbase address"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --difficulty 1 --force"
    echo "  $0 --difficulty 10000"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --datadir)
            DATADIR="$2"
            shift 2
            ;;
        --networkid)
            NETWORKID="$2"
            shift 2
            ;;
        --etherbase)
            ETHERBASE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate difficulty is a number
if ! [[ "$DIFFICULTY" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Difficulty must be a positive integer${NC}"
    exit 1
fi

# Convert difficulty to hex
DIFFICULTY_HEX=$(printf "0x%x" "$DIFFICULTY")

echo "*** QUANTUM-GETH BLOCKCHAIN RESET UTILITY ***"
echo -e "${RED}This will COMPLETELY WIPE the existing blockchain!${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Data Directory: $DATADIR"
echo "  Starting Difficulty: $DIFFICULTY ($DIFFICULTY_HEX)"
echo "  Network ID: $NETWORKID"
echo "  Etherbase: $ETHERBASE"
echo "  Balance: $BALANCE wei"
echo ""

# Confirmation prompt unless --force is used
if [ "$FORCE" != true ]; then
    echo -n "Are you sure you want to DELETE all blockchain data? (type 'YES' to confirm): "
    read -r confirmation
    if [ "$confirmation" != "YES" ]; then
        echo -e "${RED}Operation cancelled.${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}Cleaning blockchain data...${NC}"

# Stop any running geth processes
echo "  Stopping any running geth processes..."
if pgrep -f "geth" > /dev/null; then
    pkill -f "geth" || true
    echo -e "  ${GREEN}Geth processes stopped${NC}"
else
    echo "  No running geth processes found"
fi

# Remove existing blockchain data
echo "  Removing existing blockchain data..."
if [ -d "$DATADIR" ]; then
    rm -rf "$DATADIR"
    echo -e "  ${GREEN}Blockchain data removed${NC}"
else
    echo "  No existing blockchain data found"
fi

# Create new genesis file
GENESIS_FILE="genesis_temp_d${DIFFICULTY}.json"
echo -e "${YELLOW}  Creating genesis file: $GENESIS_FILE${NC}"

# Generate genesis content
cat > "$GENESIS_FILE" << EOF
{
  "config": {
    "chainId": $NETWORKID,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "qmpow": {
      "period": 0,
      "epoch": 30000
    }
  },
  "difficulty": "$DIFFICULTY_HEX",
  "gasLimit": "0x1c9c380",
  "alloc": {
    "$ETHERBASE": {
      "balance": "$BALANCE"
    }
  }
}
EOF

echo -e "  ${GREEN}Genesis file created${NC}"

# Initialize blockchain with new genesis
echo -e "${YELLOW}Initializing blockchain with new genesis...${NC}"

# Determine geth binary path
GETH_BIN=""
if [ -f "./quantum-geth/build/bin/geth" ]; then
    GETH_BIN="./quantum-geth/build/bin/geth"
elif [ -f "./geth" ]; then
    GETH_BIN="./geth"
elif [ -f "./geth.exe" ]; then
    GETH_BIN="./geth.exe"
else
    echo -e "${RED}  Failed to find geth binary${NC}"
    echo "  Please compile geth first with: cd quantum-geth && make geth"
    exit 1
fi

if $GETH_BIN --datadir "$DATADIR" init "$GENESIS_FILE" 2>/dev/null; then
    echo -e "  ${GREEN}Blockchain initialized successfully${NC}"
else
    echo -e "${RED}  Failed to initialize blockchain${NC}"
    exit 1
fi

# Clean up temporary genesis file
echo -e "${YELLOW}Cleaning up...${NC}"
rm -f "$GENESIS_FILE"
echo -e "  ${GREEN}Temporary genesis file removed${NC}"

echo ""
echo -e "${GREEN}BLOCKCHAIN RESET COMPLETE!${NC}"
echo ""
echo -e "${CYAN}Summary:${NC}"
echo "  Starting Difficulty: $DIFFICULTY"
if [ "$DIFFICULTY" -eq 1 ]; then
    echo "  Target: Very easy (instant blocks)"
elif [ "$DIFFICULTY" -le 10 ]; then
    echo "  Target: Easy (fast blocks)"
elif [ "$DIFFICULTY" -le 100 ]; then
    echo "  Target: Medium (normal blocks)"
else
    echo "  Target: Hard (slow blocks)"
fi
echo "  Data Directory: $DATADIR"
echo "  Etherbase: $ETHERBASE"
echo ""
echo -e "${YELLOW}You can now start mining with:${NC}"
echo "  ./scripts/start-mining.sh"
echo ""
echo "Pro Tips:"
echo "  * Use difficulty=1 for instant block testing"
echo "  * Use difficulty=10-100 for normal testing"
echo "  * Use difficulty=1000+ for realistic mining"
echo "  * Bitcoin-style nonce progression: qnonce=0,1,2,3..."
echo "  * Lower quality values win (Bitcoin-style)"
echo "" 