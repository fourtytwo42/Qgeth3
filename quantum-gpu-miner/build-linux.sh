#!/bin/bash
# Build script for Quantum-GPU-Miner (Linux)
# Usage: ./build-linux.sh [cpu|gpu|both] [clean]

MODE=${1:-cpu}
CLEAN=${2:-false}
HELP=${3:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function show_help() {
    echo -e "${BLUE}Quantum-GPU-Miner Build Script for Linux${NC}"
    echo ""
    echo "Usage: ./build-linux.sh [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  cpu    - Build CPU-only version (default)"
    echo "  gpu    - Build GPU-accelerated version (requires CUDA)"
    echo "  both   - Build both CPU and GPU versions"
    echo ""
    echo "Options:"
    echo "  clean  - Clean build artifacts before building"
    echo "  help   - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./build-linux.sh cpu"
    echo "  ./build-linux.sh gpu clean"
    echo "  ./build-linux.sh both"
    echo ""
    echo "Requirements:"
    echo "  CPU Mode: Go 1.19+, Python 3.8+, Qiskit"
    echo "  GPU Mode: CUDA Toolkit 12.0+, GCC"
}

function check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Go
    if ! command -v go &> /dev/null; then
        echo -e "${RED}Error: Go not found. Please install Go 1.19+${NC}"
        return 1
    fi
    echo -e "${GREEN}Go: $(go version)${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
        return 1
    fi
    PYTHON_CMD=$(command -v python3 || command -v python)
    echo -e "${GREEN}Python: $($PYTHON_CMD --version)${NC}"
    
    # Check Qiskit
    if ! $PYTHON_CMD -c "import qiskit" 2>/dev/null; then
        echo -e "${YELLOW}Warning: Qiskit not found. Installing...${NC}"
        pip3 install qiskit qiskit-aer numpy cupy-cuda12x
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error: Failed to install Qiskit dependencies${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}Qiskit: Available${NC}"
    fi
    
    return 0
}

function check_cuda_prerequisites() {
    echo -e "${BLUE}Checking CUDA prerequisites...${NC}"
    
    # Check NVCC
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Error: NVCC not found. Please install CUDA Toolkit 12.0+${NC}"
        return 1
    fi
    echo -e "${GREEN}NVCC: $(nvcc --version | grep 'release')${NC}"
    
    # Check GCC
    if ! command -v gcc &> /dev/null; then
        echo -e "${RED}Error: GCC not found. Please install build-essential${NC}"
        return 1
    fi
    echo -e "${GREEN}GCC: $(gcc --version | head -n1)${NC}"
    
    return 0
}

function clean_build_artifacts() {
    echo -e "${BLUE}Cleaning build artifacts...${NC}"
    
    # Remove executable
    rm -f quantum-gpu-miner
    
    # Remove CUDA libraries
    rm -f pkg/quantum/*.so
    rm -f pkg/quantum/*.dll
    
    # Clean Go cache
    go clean -cache
    
    echo -e "${GREEN}Build artifacts cleaned${NC}"
}

function build_cpu_version() {
    echo -e "${BLUE}Building quantum-gpu-miner (CPU/GPU capable)...${NC}"
    
    export CGO_ENABLED=1
    go build -ldflags "-s -w" -o quantum-gpu-miner .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Quantum miner built successfully: quantum-gpu-miner${NC}"
        echo -e "${BLUE}Note: This executable supports both CPU and GPU modes${NC}"
        
        # Test the build
        ./quantum-gpu-miner --version
        return 0
    else
        echo -e "${RED}Build failed${NC}"
        return 1
    fi
}

function build_cuda_library() {
    echo -e "${BLUE}Building CUDA library...${NC}"
    
    cd pkg/quantum
    
    # Compile CUDA library
    nvcc -shared -Xcompiler -fPIC -O3 -o libquantum_cuda.so quantum_cuda.cu
    
    cd ../..
    
    if [ -f "pkg/quantum/libquantum_cuda.so" ]; then
        echo -e "${GREEN}CUDA library built successfully${NC}"
        return 0
    else
        echo -e "${RED}CUDA library build failed${NC}"
        return 1
    fi
}

function build_gpu_version() {
    echo -e "${BLUE}Building quantum-gpu-miner with CUDA support...${NC}"
    
    # Build CUDA library first
    if ! build_cuda_library; then
        return 1
    fi
    
    # Build Go application with CUDA tags
    export CGO_ENABLED=1
    go build -tags cuda -ldflags "-s -w" -o quantum-gpu-miner .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Quantum miner with CUDA built successfully: quantum-gpu-miner${NC}"
        echo -e "${BLUE}Note: This executable supports both CPU and GPU modes${NC}"
        
        # Test the build
        ./quantum-gpu-miner --version
        return 0
    else
        echo -e "${RED}GPU build failed${NC}"
        return 1
    fi
}

# Main script logic
if [ "$MODE" = "help" ] || [ "$HELP" = "help" ]; then
    show_help
    exit 0
fi

echo -e "${BLUE}Quantum-GPU-Miner Build Script for Linux${NC}"
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo ""

# Clean if requested
if [ "$CLEAN" = "clean" ]; then
    clean_build_artifacts
fi

# Check basic prerequisites
if ! check_prerequisites; then
    exit 1
fi

# Build based on mode
case $MODE in
    cpu)
        build_cpu_version
        ;;
    gpu)
        if ! check_cuda_prerequisites; then
            exit 1
        fi
        build_gpu_version
        ;;
    both)
        # Build CPU first
        build_cpu_version
        if [ $? -ne 0 ]; then
            exit 1
        fi
        
        # Try GPU build
        if check_cuda_prerequisites; then
            echo -e "${BLUE}Attempting GPU build...${NC}"
            build_gpu_version
        else
            echo -e "${YELLOW}SKIPPING: GPU build (CUDA not available)${NC}"
        fi
        ;;
    *)
        echo -e "${RED}Error: Invalid mode '$MODE'. Use: cpu, gpu, both, or help${NC}"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${BLUE}Usage examples:${NC}"
    echo "  ./quantum-gpu-miner -node http://localhost:8545 -threads 4"
    echo "  ./quantum-gpu-miner -gpu -node http://localhost:8545 -threads 2"
    echo ""
    echo -e "${BLUE}Note: Single executable supports both CPU and GPU modes${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
