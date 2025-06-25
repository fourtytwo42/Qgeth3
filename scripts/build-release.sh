#!/bin/bash
# Quantum-Geth Release Builder

# Default parameters
TARGET="both"
CLEAN=false

# Function to show help
show_help() {
    echo "Quantum-Geth Release Builder"
    echo ""
    echo "Description:"
    echo "  Builds distributable release packages with all dependencies"
    echo ""
    echo "Usage:"
    echo "  $0 [target] [options]"
    echo ""
    echo "Targets:"
    echo "  geth    - Build quantum-geth release only"
    echo "  miner   - Build quantum-miner release only"
    echo "  both    - Build both releases (default)"
    echo ""
    echo "Options:"
    echo "  --clean - Clean existing release folders before building"
    echo "  --help  - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 geth"
    echo "  $0 miner --clean"
    echo ""
    echo "Output:"
    echo "  releases/quantum-geth-<timestamp>/"
    echo "  releases/quantum-miner-<timestamp>/"
}

# Function to get Unix timestamp
get_unix_timestamp() {
    date +%s
}

# Function to create release folder
new_release_folder() {
    local base_name="$1"
    local timestamp=$(get_unix_timestamp)
    local release_dir="releases/$base_name-$timestamp"
    
    if [ ! -d "releases" ]; then
        mkdir -p "releases"
    fi
    
    if [ -d "$release_dir" ]; then
        rm -rf "$release_dir"
    fi
    
    mkdir -p "$release_dir"
    echo "$release_dir"
}

# Function to build quantum-geth
build_quantum_geth() {
    echo "Building Quantum-Geth Release..."
    
    # Create release directory
    local release_dir=$(new_release_folder "quantum-geth")
    echo "Release directory: $release_dir"
    
    # Check if we're in the scripts directory and need to go up
    if [ -d "../quantum-geth" ]; then
        QUANTUM_GETH_DIR="../quantum-geth"
    elif [ -d "quantum-geth" ]; then
        QUANTUM_GETH_DIR="quantum-geth"
    else
        echo "ERROR: quantum-geth directory not found"
        echo "Current directory: $(pwd)"
        echo "Looking for: ../quantum-geth or quantum-geth"
        exit 1
    fi
    
    # Build geth
    echo "Compiling quantum-geth from: $QUANTUM_GETH_DIR"
    
    cd "$QUANTUM_GETH_DIR/cmd/geth"
    if ! CGO_ENABLED=0 go build -ldflags "-s -w" -o ../../../"$release_dir"/geth .; then
        echo "ERROR: Failed to build quantum-geth"
        cd ../../..
        exit 1
    fi
    cd ../../..
    
    # Copy files to release
    echo "Preparing release package..."
    
    # Create a basic genesis file for the release
    echo "Creating basic genesis file for release..."
    cat > "$release_dir/genesis_quantum.json" << 'EOF'
{
  "config": {
    "chainId": 1337,
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
    "arrowGlacierBlock": 0,
    "grayGlacierBlock": 0,
    "mergeNetsplitBlock": 0,
    "shanghaiTime": 0,
    "cancunTime": 0,
    "pragueTime": null,
    "verkleTime": null,
    "qmpow": {
      "qbits": 16,
      "tcount": 20,
      "lnet": 128,
      "epochLen": 600000,
      "testMode": false
    }
  },
  "nonce": "0x0",
  "timestamp": "0x0",
  "extraData": "0x51756161746756756d2d476574682052656c65617365",
  "gasLimit": "0x2fefd8",
  "difficulty": "0x1f4",
  "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "coinbase": "0x0000000000000000000000000000000000000000",
  "alloc": {
    "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A": {
      "balance": "300000000000000000000000"
    }
  },
  "number": "0x0",
  "gasUsed": "0x0",
  "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "baseFeePerGas": "0x7",
  "excessBlobGas": "0x0",
  "blobGasUsed": "0x0"
}
EOF
    
    # Create startup scripts
    cat > "$release_dir/start-node.sh" << 'EOF'
#!/bin/bash
# Start Quantum-Geth Node (No Mining)

DATADIR="qdata"
NETWORKID=1337
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"

echo "Quantum-Geth Node Startup"
echo "========================="

# Initialize if needed
if [ ! -d "$DATADIR" ]; then
    echo "Initializing blockchain with quantum genesis..."
    ./geth --datadir "$DATADIR" init genesis_quantum.json
fi

echo "Starting Quantum-Geth node (RPC only, no mining)..."
echo "Press Ctrl+C to stop"

./geth \
    --datadir "$DATADIR" \
    --networkid "$NETWORKID" \
    --http \
    --http.addr "0.0.0.0" \
    --http.port 8545 \
    --http.api "eth,net,web3,personal,miner" \
    --http.corsdomain "*" \
    --ws \
    --ws.addr "0.0.0.0" \
    --ws.port 8546 \
    --ws.api "eth,net,web3" \
    --ws.origins "*" \
    --miner.etherbase "$ETHERBASE" \
    --mine.threads 0 \
    --nodiscover \
    --maxpeers 0
EOF
    
    cat > "$release_dir/start-mining.sh" << 'EOF'
#!/bin/bash
# Start Quantum-Geth with Mining

DATADIR="qdata"
NETWORKID=1337
ETHERBASE="0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
THREADS=1

echo "Quantum-Geth Mining Startup"
echo "==========================="

# Initialize if needed
if [ ! -d "$DATADIR" ]; then
    echo "Initializing blockchain with quantum genesis..."
    ./geth --datadir "$DATADIR" init genesis_quantum.json
fi

echo "Starting Quantum-Geth mining with $THREADS thread(s)..."
echo "Press Ctrl+C to stop"

./geth \
    --datadir "$DATADIR" \
    --networkid "$NETWORKID" \
    --http \
    --http.addr "0.0.0.0" \
    --http.port 8545 \
    --http.api "eth,net,web3,personal,miner" \
    --http.corsdomain "*" \
    --ws \
    --ws.addr "0.0.0.0" \
    --ws.port 8546 \
    --ws.api "eth,net,web3" \
    --ws.origins "*" \
    --miner.etherbase "$ETHERBASE" \
    --mine \
    --mine.threads "$THREADS" \
    --nodiscover \
    --maxpeers 0
EOF
    
    chmod +x "$release_dir/start-node.sh"
    chmod +x "$release_dir/start-mining.sh"
    
    # Create README
    cat > "$release_dir/README.md" << 'EOF'
# Quantum-Geth Release

This is a pre-built release of Quantum-Geth with quantum proof-of-work consensus.

## Quick Start

1. **Start a node (no mining):**
   ```bash
   ./start-node.sh
   ```

2. **Start mining:**
   ```bash
   ./start-mining.sh
   ```

## Features

- 128 sequential quantum puzzles per block
- 16 qubits × 20 T-gates per puzzle
- Bitcoin-style halving rewards
- ASERT-Q difficulty adjustment
- Mahadev→CAPSS→Nova proof stack
- Dilithium-2 self-attestation

## Files

- `geth` - Quantum-Geth executable
- `genesis_quantum.json` - Genesis block configuration
- `start-node.sh` - Start node without mining
- `start-mining.sh` - Start node with mining
- `README.md` - This file

## Configuration

The genesis file is pre-configured for:
- Chain ID: 1337
- Initial difficulty: 500
- Pre-funded account: 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A (300,000 QGC)
- Block reward: 50 QGC (halves every 600,000 blocks)

## Custom Usage

Initialize blockchain:
```bash
./geth --datadir qdata init genesis_quantum.json
```

Start with custom parameters:
```bash
./geth --datadir qdata --networkid 1337 --mine --mine.threads 2
```
EOF
    
    echo ""
    echo "SUCCESS: Quantum-Geth release built at $release_dir"
    echo "Files included:"
    ls -la "$release_dir"
}

# Function to build quantum-miner
build_quantum_miner() {
    echo "Building Quantum-Miner Release..."
    
    # Create release directory
    local release_dir=$(new_release_folder "quantum-miner")
    echo "Release directory: $release_dir"
    
    # Check if we're in the scripts directory and need to go up
    if [ -d "../quantum-miner" ]; then
        QUANTUM_MINER_DIR="../quantum-miner"
    elif [ -d "quantum-miner" ]; then
        QUANTUM_MINER_DIR="quantum-miner"
    else
        echo "ERROR: quantum-miner directory not found"
        echo "Current directory: $(pwd)"
        echo "Looking for: ../quantum-miner or quantum-miner"
        exit 1
    fi
    
    # Build miner
    echo "Compiling quantum-miner from: $QUANTUM_MINER_DIR"
    
    cd "$QUANTUM_MINER_DIR"
    if ! CGO_ENABLED=0 go build -ldflags "-s -w" -o ../"$release_dir"/quantum-miner .; then
        echo "ERROR: Failed to build quantum-miner"
        cd ..
        exit 1
    fi
    cd ..
    
    # Copy configuration files
    echo "Preparing release package..."
    
    # Copy miner.json if it exists in the parent directory
    if [ -f "../miner.json" ]; then
        cp ../miner.json "$release_dir/"
    elif [ -f "miner.json" ]; then
        cp miner.json "$release_dir/"
    else
        echo "Warning: miner.json not found, creating default configuration"
        cat > "$release_dir/miner.json" << 'EOF'
{
  "geth_rpc": "http://127.0.0.1:8545",
  "mining_address": "0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A",
  "cpu_threads": 4,
  "gpu_enabled": true,
  "gpu_device": 0,
  "log_level": "info"
}
EOF
    fi
    
    # Create startup script
    cat > "$release_dir/start-miner.sh" << 'EOF'
#!/bin/bash
# Start Quantum Miner

CONFIG_FILE="miner.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file $CONFIG_FILE not found"
    exit 1
fi

echo "Quantum Miner Startup"
echo "====================="
echo "Configuration: $CONFIG_FILE"
echo ""
echo "Make sure Quantum-Geth is running with RPC enabled:"
echo "  HTTP RPC: http://localhost:8545"
echo ""
echo "Press Ctrl+C to stop mining"
echo ""

./quantum-miner -config "$CONFIG_FILE"
EOF
    
    chmod +x "$release_dir/start-miner.sh"
    
    # Create README
    cat > "$release_dir/README.md" << 'EOF'
# Quantum-Miner Release

This is a pre-built release of the Quantum Miner for Quantum-Geth.

## Quick Start

1. **Make sure Quantum-Geth is running** with RPC enabled on port 8545
2. **Start mining:**
   ```bash
   ./start-miner.sh
   ```

## Configuration

Edit `miner.json` to configure:
- Geth RPC endpoint
- Mining address
- Number of worker threads
- GPU/CPU settings

## Files

- `quantum-miner` - Quantum Miner executable
- `miner.json` - Miner configuration
- `start-miner.sh` - Startup script
- `README.md` - This file

## Custom Usage

```bash
./quantum-miner -config miner.json
```

## Requirements

- Quantum-Geth node running with RPC enabled
- CUDA-compatible GPU (recommended) or CPU fallback
- Python 3.8+ with qiskit and qiskit-aer
EOF
    
    echo ""
    echo "SUCCESS: Quantum-Miner release built at $release_dir"
    echo "Files included:"
    ls -la "$release_dir"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        geth|miner|both)
            TARGET="$1"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clean releases if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning existing release folders..."
    if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
        rm -rf releases/quantum-geth-*
    fi
    if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
        rm -rf releases/quantum-miner-*
    fi
    echo "Cleanup completed"
    echo ""
fi

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "ERROR: Go is not installed or not in PATH"
    echo "Please install Go 1.19 or later"
    exit 1
fi

echo "Go version: $(go version)"
echo ""

# Build based on target
case "$TARGET" in
    geth)
        build_quantum_geth
        ;;
    miner)
        build_quantum_miner
        ;;
    both)
        build_quantum_geth
        echo ""
        build_quantum_miner
        ;;
    *)
        echo "ERROR: Invalid target '$TARGET'"
        echo "Valid targets: geth, miner, both"
        exit 1
        ;;
esac

echo ""
echo "Release build completed successfully!" 