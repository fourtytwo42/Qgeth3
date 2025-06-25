#!/bin/bash
# Build script for Q Coin Linux Binaries (Scripts Directory Version)
# This script calls the root build-linux.sh to build binaries to the root directory
# Usage: ./build-linux.sh [geth|miner|both] [--clean]

# Pass all arguments to the root build script
TARGET=${1:-both}
CLEAN_FLAG=""

# Handle clean flag
if [ "$2" = "--clean" ] || [ "$1" = "--clean" ]; then
    CLEAN_FLAG="--clean"
fi

echo "🔨 Q Coin Linux Builder (Scripts Directory)"
echo "Calling root build script to compile binaries to project root..."
echo ""

# Check if we're in the scripts directory
if [ ! -f "../build-linux.sh" ]; then
    echo "❌ Error: Root build-linux.sh not found!"
    echo "Make sure you're running this from the scripts/ directory"
    exit 1
fi

# Call the root build script
cd ..
exec ./build-linux.sh "$TARGET" "$CLEAN_FLAG" 