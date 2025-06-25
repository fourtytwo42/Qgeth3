#!/bin/bash
# Basic Test Script - Linux equivalent of basic_test.ps1

echo "Basic Quantum-Geth Test"
echo "======================="

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "ERROR: Go is not installed or not in PATH"
    echo "Please install Go 1.19 or later"
    exit 1
fi

echo "Go version: $(go version)"

# Test build of quantum-geth
echo "Testing quantum-geth build..."
if cd quantum-geth/cmd/geth && go build -o ../../../geth . && cd ../../..; then
    echo "SUCCESS: quantum-geth builds successfully"
else
    echo "ERROR: Failed to build quantum-geth"
    exit 1
fi

# Test build of quantum-miner
echo "Testing quantum-miner build..."
if cd quantum-miner && go build -o ../quantum-miner . && cd ..; then
    echo "SUCCESS: quantum-miner builds successfully"
else
    echo "ERROR: Failed to build quantum-miner"
    exit 1
fi

echo ""
echo "All basic tests passed!"
echo "Both quantum-geth and quantum-miner build successfully." 