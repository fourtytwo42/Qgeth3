#!/bin/bash
# Q Coin Build Script
# Usage: ./build-release.sh [component] [options]
# Components: geth, miner, both (default: both)

COMPONENT="both"
CLEAN=false
HELP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        geth|miner|both)
            COMPONENT="$1"
            shift
            ;;
        --clean)
            CLEAN=true
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
    echo -e "\033[1;36mQ Coin Build Script\033[0m"
    echo ""
    echo -e "\033[1;37mUsage: ./build-release.sh [component] [options]\033[0m"
    echo ""
    echo -e "\033[1;33mComponents:\033[0m"
    echo "  geth      - Build quantum-geth only"
    echo "  miner     - Build quantum-miner only"
    echo "  both      - Build both geth and miner [DEFAULT]"
    echo ""
    echo -e "\033[1;33mOptions:\033[0m"
    echo "  --clean   - Clean build directories before building"
    echo "  --help    - Show this help message"
    echo ""
    echo -e "\033[1;32mExamples:\033[0m"
    echo "  ./build-release.sh           # Build both geth and miner"
    echo "  ./build-release.sh geth      # Build geth only"
    echo "  ./build-release.sh --clean   # Clean build and build both"
    exit 0
fi

echo -e "\033[1;36m🔨 Q Coin Build Script\033[0m"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "\033[1;33m🧹 Cleaning build directories...\033[0m"
    
    if [ -d "quantum-geth/build" ]; then
        rm -rf quantum-geth/build
        echo -e "\033[1;37m   Cleaned quantum-geth/build\033[0m"
    fi
    
    if [ -f "quantum-miner/quantum-miner" ]; then
        rm -f quantum-miner/quantum-miner
        echo -e "\033[1;37m   Cleaned quantum-miner\033[0m"
    fi
    
    echo -e "\033[1;32m✅ Clean completed\033[0m"
    echo ""
fi

# Build geth
if [ "$COMPONENT" = "geth" ] || [ "$COMPONENT" = "both" ]; then
    echo -e "\033[1;33m🔨 Building quantum-geth...\033[0m"
    
    if [ ! -d "quantum-geth" ]; then
        echo -e "\033[1;31m❌ quantum-geth directory not found!\033[0m"
        exit 1
    fi
    
    cd quantum-geth
    export CGO_ENABLED=1
    go build -o build/bin/geth ./cmd/geth
    
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m✅ quantum-geth built successfully\033[0m"
    else
        echo -e "\033[1;31m❌ quantum-geth build failed!\033[0m"
        cd ..
        exit 1
    fi
    cd ..
    echo ""
fi

# Build miner
if [ "$COMPONENT" = "miner" ] || [ "$COMPONENT" = "both" ]; then
    echo -e "\033[1;33m🔨 Building quantum-miner...\033[0m"
    
    if [ ! -d "quantum-miner" ]; then
        echo -e "\033[1;31m❌ quantum-miner directory not found!\033[0m"
        exit 1
    fi
    
    cd quantum-miner
    export CGO_ENABLED=1
    go build -o quantum-miner .
    
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m✅ quantum-miner built successfully\033[0m"
    else
        echo -e "\033[1;31m❌ quantum-miner build failed!\033[0m"
        cd ..
        exit 1
    fi
    cd ..
    echo ""
fi

echo -e "\033[1;32m🎉 Build completed successfully!\033[0m"

# Show built files
echo ""
echo -e "\033[1;36mBuilt files:\033[0m"
if [ "$COMPONENT" = "geth" ] || [ "$COMPONENT" = "both" ]; then
    if [ -f "quantum-geth/build/bin/geth" ]; then
        GETH_SIZE=$(du -h quantum-geth/build/bin/geth | cut -f1)
        echo -e "\033[1;37m  quantum-geth/build/bin/geth ($GETH_SIZE)\033[0m"
    fi
fi
if [ "$COMPONENT" = "miner" ] || [ "$COMPONENT" = "both" ]; then
    if [ -f "quantum-miner/quantum-miner" ]; then
        MINER_SIZE=$(du -h quantum-miner/quantum-miner | cut -f1)
        echo -e "\033[1;37m  quantum-miner/quantum-miner ($MINER_SIZE)\033[0m"
    fi
fi 