#!/bin/bash
# Q Geth Quick Start Script
# This script provides easy access to the reorganized Q Geth scripts

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}🚀 Q Geth Quick Start${NC}"
    echo ""
}

print_help() {
    print_header
    echo "Usage: ./quick-start.sh [command]"
    echo ""
    echo -e "${YELLOW}Build Commands:${NC}"
    echo "  build           - Build Q Geth (Linux only)"
    echo "  build-clean     - Clean build Q Geth (Linux only)"
    echo ""
    echo -e "${YELLOW}Run Commands:${NC}"
    echo "  start           - Start Q Geth testnet node"
    echo "  start-mainnet   - Start Q Geth mainnet node"
    echo "  start-mining    - Start Q Geth testnet with mining"
    echo "  miner           - Start quantum miner"
    echo ""
    echo -e "${YELLOW}Setup Commands:${NC}"
    echo "  bootstrap       - Complete VPS setup (requires sudo)"
    echo "  prepare-vps     - Prepare VPS environment (Linux only)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./quick-start.sh build"
    echo "  ./quick-start.sh start"
    echo "  ./quick-start.sh bootstrap"
    echo ""
    echo -e "${YELLOW}Platform-Specific Scripts:${NC}"
    echo "  Linux:   scripts/linux/"
    echo "  Windows: scripts/windows/"
    echo "  Deploy:  scripts/deployment/"
}

case "${1:-help}" in
    build)
        echo -e "${GREEN}Building Q Geth...${NC}"
        ./scripts/linux/build-linux.sh geth
        ;;
    build-clean)
        echo -e "${GREEN}Clean building Q Geth...${NC}"
        ./scripts/linux/build-linux.sh geth --clean
        ;;
    start)
        echo -e "${GREEN}Starting Q Geth testnet node...${NC}"
        ./scripts/linux/start-geth.sh testnet
        ;;
    start-mainnet)
        echo -e "${GREEN}Starting Q Geth mainnet node...${NC}"
        ./scripts/linux/start-geth.sh mainnet
        ;;
    start-mining)
        echo -e "${GREEN}Starting Q Geth testnet with mining...${NC}"
        ./scripts/linux/start-geth.sh testnet --mining
        ;;
    miner)
        echo -e "${GREEN}Starting quantum miner...${NC}"
        ./scripts/linux/start-miner.sh
        ;;
    bootstrap)
        echo -e "${GREEN}Starting complete VPS bootstrap...${NC}"
        ./scripts/deployment/bootstrap-qgeth.sh
        ;;
    prepare-vps)
        echo -e "${GREEN}Preparing VPS environment...${NC}"
        ./scripts/linux/prepare-vps.sh
        ;;
    help|--help|-h|"")
        print_help
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac 