#!/bin/bash
# Quantum-Geth GPU Miner

# Default parameters
COINBASE=""
NODE_URL="http://localhost:8545"
THREADS=1
LOG=false

# Function to show help
show_help() {
    echo "Quantum-Geth GPU Miner"
    echo ""
    echo "Description:"
    echo "  High-performance quantum mining with CUDA GPU acceleration"
    echo ""
    echo "Usage:"
    echo "  $0 --coinbase <address> [options]"
    echo ""
    echo "Parameters:"
    echo "  --coinbase <address>   Coinbase address for mining rewards (required)"
    echo "  --node <url>           Quantum-Geth node URL (default: http://localhost:8545)"
    echo "  --threads <number>     Number of GPU mining threads (default: 1)"
    echo "  --log                  Enable detailed logging to quantum-miner.log file"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --threads 2"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --node http://192.168.1.100:8545"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --log  # Enable logging to file"
    echo ""
    echo "Features:"
    echo "  * CUDA GPU acceleration (0.45 puzzles/sec)"
    echo "  * Real quantum circuit simulation (16-qubit, 20 T-gates)"
    echo "  * Multi-GPU mining support"
    echo "  * Automatic GPU detection and optimization"
    echo ""
    echo "Performance Comparison:"
    echo "  CPU Mining: 0.36 puzzles/sec (use ./run-cpu-miner.sh)"
    echo "  GPU Mining: 0.45 puzzles/sec (this miner)"
    echo ""
    echo "Requirements:"
    echo "  * CUDA-compatible GPU (compute capability 6.0+)"
    echo "  * NVIDIA CUDA toolkit"
    echo "  * Python 3.8+ with qiskit-aer[cuda]"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coinbase)
            COINBASE="$2"
            shift 2
            ;;
        --node)
            NODE_URL="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --log)
            LOG=true
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

echo "Quantum-Geth GPU Miner"
echo "CUDA GPU-accelerated quantum circuit mining"

if [ -z "$COINBASE" ]; then
    echo "ERROR: Coinbase address required!"
    echo ""
    echo "Usage examples:"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
    echo "  $0 --coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A --threads 2"
    echo "  $0 --help"
    echo ""
    echo "Tip: For CPU-only mining, use ./run-cpu-miner.sh"
    exit 1
fi

# Validate coinbase address format
if [[ ! "$COINBASE" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: Invalid coinbase address format!"
    echo "Expected format: 0x followed by 40 hex characters"
    exit 1
fi

# Check for CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1
else
    echo "  WARNING: nvidia-smi not found. GPU mining may not work properly."
    echo "  Consider using CPU mining: ./run-cpu-miner.sh"
fi

# Check Python and qiskit-aer
echo "Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "  Python version: $PYTHON_VERSION"
    
    if python3 -c "import qiskit_aer; print('  qiskit-aer version:', qiskit_aer.__version__)" 2>/dev/null; then
        echo "  qiskit-aer: Available"
    else
        echo "  WARNING: qiskit-aer not found. Install with: pip install qiskit-aer"
    fi
else
    echo "  ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Find the newest quantum-miner release or use development version
RELEASE_DIR=$(find releases -name "quantum-miner-*" -type d 2>/dev/null | sort -V | tail -1)

if [ -n "$RELEASE_DIR" ] && [ -f "$RELEASE_DIR/quantum-miner" ]; then
    MINER_EXECUTABLE="$RELEASE_DIR/quantum-miner"
    WORKING_DIR="$RELEASE_DIR"
else
    MINER_EXECUTABLE="quantum-miner/quantum-miner"
    WORKING_DIR="quantum-miner"
fi

if [ ! -f "$MINER_EXECUTABLE" ]; then
    echo "GPU miner executable not found. Building release..."
    echo ""
    
    if ./scripts/build-release.sh miner; then
        # Re-find the newest release
        RELEASE_DIR=$(find releases -name "quantum-miner-*" -type d 2>/dev/null | sort -V | tail -1)
        if [ -n "$RELEASE_DIR" ] && [ -f "$RELEASE_DIR/quantum-miner" ]; then
            MINER_EXECUTABLE="$RELEASE_DIR/quantum-miner"
            WORKING_DIR="$RELEASE_DIR"
            echo "SUCCESS: Release built at $RELEASE_DIR"
        else
            echo "ERROR: Failed to create release"
            exit 1
        fi
    else
        echo "ERROR: Failed to build quantum-miner release"
        echo ""
        echo "Manual build options:"
        echo "  ./scripts/build-release.sh miner"
        echo "  cd quantum-miner && go build -o quantum-miner ."
        exit 1
    fi
fi

echo ""
echo "GPU Mining Configuration:"
echo "   Coinbase: $COINBASE"
echo "   Node URL: $NODE_URL"
echo "   GPU Threads: $THREADS"
echo "   Quantum Puzzles: 128 per block"
echo "   Circuit Size: 16 qubits, 20 T-gates"
echo "   Acceleration: CUDA GPU"
echo ""

# Build miner arguments
MINER_ARGS=(-coinbase "$COINBASE" -node "$NODE_URL" -threads "$THREADS" -gpu)
if [ "$LOG" = true ]; then
    MINER_ARGS+=(-log)
fi

echo "Starting GPU quantum miner..."
echo "  Use Ctrl+C to stop mining"
echo ""

# Change to working directory and run miner
cd "$WORKING_DIR" || exit 1

if ! ./quantum-miner "${MINER_ARGS[@]}"; then
    echo "ERROR: Failed to start GPU miner"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check quantum-geth node is running at $NODE_URL"
    echo "  2. Verify CUDA installation and GPU compatibility"
    echo "  3. Install qiskit-aer with CUDA support: pip install qiskit-aer[cuda]"
    echo "  4. Check GPU memory availability with: nvidia-smi"
    echo "  5. Try CPU mining as fallback: ./scripts/run-cpu-miner.sh"
    exit 1
fi 