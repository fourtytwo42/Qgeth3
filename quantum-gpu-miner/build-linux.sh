#!/bin/bash
# Build script for Quantum-Geth Standalone Miner (Linux)

OUTPUT_NAME=${1:-quantum-miner}
VERSION=${2:-1.0.0}
CLEAN=${3:-false}

echo " Building Quantum-Geth Standalone Miner for Linux..."
echo "Version: $VERSION"
echo ""

if [ "$CLEAN" = "true" ]; then
    echo " Cleaning previous builds..."
    rm -f quantum-miner
    rm -f quantum_solver.py
fi

export GOOS=linux
export GOARCH=amd64
export CGO_ENABLED=0

BUILD_TIME=$(date "+%Y-%m-%d %H:%M:%S")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "  Compiling Go binary..."
echo "  Target: linux/amd64"
echo "  Output: $OUTPUT_NAME"
echo ""

go build -ldflags "-s -w" -o "$OUTPUT_NAME" .

if [ $? -eq 0 ]; then
    echo " Build successful!"
    
    echo ""
    echo " Binary Information:"
    ls -lh "$OUTPUT_NAME"
    echo ""
    
    chmod +x "$OUTPUT_NAME"
    
    echo " Build complete! Ready to mine quantum blocks."
    echo ""
    echo "Usage examples:"
    echo "  ./$OUTPUT_NAME -coinbase 0x... -node http://localhost:8545"
    echo "  ./$OUTPUT_NAME -version"
    
else
    echo " Build failed!"
    exit 1
fi
