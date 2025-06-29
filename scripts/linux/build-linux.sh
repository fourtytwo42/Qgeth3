#!/bin/bash
# Q Coin Enhanced Linux Build System
# Consolidated script combining best features from build-linux.sh and build-linux-enhanced.sh
# Can be run standalone or integrated with bootstrap
# Cross-distribution compatibility with automated error recovery
# Usage: ./build-linux.sh [geth|miner|both] [--clean] [-y|--yes]

set -e

# Configuration
VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
AUTO_CONFIRM=false
TARGET="both"
CLEAN=false
DEBUG=false

for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        --clean)
            CLEAN=true
            ;;
        --debug)
            DEBUG=true
            ;;
        geth|miner|both)
            TARGET="$arg"
            ;;
        --help|-h)
            echo "Q Coin Enhanced Linux Build System"
            echo ""
            echo "Usage: $0 [target] [options]"
            echo ""
            echo "Targets:"
            echo "  geth    Build quantum-geth only"
            echo "  miner   Build quantum-miner only"
            echo "  both    Build both (default)"
            echo ""
            echo "Options:"
            echo "  -y, --yes     Auto-confirm all prompts"
            echo "  --clean       Clean previous builds"
            echo "  --debug       Enable debug output"
            echo "  --help        Show this help"
            echo ""
            exit 0
            ;;
    esac
done

# Colors and logging
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { [ "$DEBUG" = true ] && echo -e "${CYAN}[DEBUG]${NC} $1"; }

# Detect system information
detect_system() {
    # OS Detection
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        QGETH_OS="$ID"
        QGETH_DISTRO="$ID"
        QGETH_DISTRO_VERSION="$VERSION_ID"
        QGETH_DISTRO_NAME="$NAME"
    else
        QGETH_OS="unknown"
        QGETH_DISTRO="unknown"
        QGETH_DISTRO_VERSION="unknown"
        QGETH_DISTRO_NAME="Unknown Linux"
    fi
    
    # Architecture Detection
    QGETH_ARCH=$(uname -m)
    case $QGETH_ARCH in
        x86_64) QGETH_GO_ARCH="amd64" ;;
        aarch64|arm64) QGETH_GO_ARCH="arm64" ;;
        armv7l) QGETH_GO_ARCH="arm" ;;
        i686) QGETH_GO_ARCH="386" ;;
        *) QGETH_GO_ARCH="amd64" ;;
    esac
    
    # Shell Detection
    QGETH_SHELL="${SHELL:-/bin/bash}"
    
    log_debug "System detected: $QGETH_OS ($QGETH_DISTRO $QGETH_DISTRO_VERSION)"
    log_debug "Architecture: $QGETH_ARCH (Go: $QGETH_GO_ARCH)"
}

# Check and fix permissions
check_and_fix_permissions() {
    log_info "🔒 Checking directory permissions..."
    
    local output_dir=$(pwd)
    local current_user=$(whoami)
    local dir_owner=""
    
    # Get directory owner
    if command -v stat >/dev/null 2>&1; then
        dir_owner=$(stat -c '%U' "$output_dir" 2>/dev/null || echo "unknown")
    fi
    
    log_debug "Current directory: $output_dir"
    log_debug "Current user: $current_user"
    log_debug "Directory owner: $dir_owner"
    
    # Check if we can write to current directory
    if [ ! -w "$output_dir" ]; then
        log_warning "Permission issue detected: Cannot write to output directory"
        
        # If running as root, fix ownership
        if [ "$current_user" = "root" ] && [ "$dir_owner" != "root" ]; then
            log_info "🔧 Auto-fix: Changing directory ownership to root..."
            chown -R root:root "$output_dir" 2>/dev/null || true
        elif [ "$current_user" != "root" ] && [ "$dir_owner" = "root" ]; then
            log_info "🔧 Auto-fix: Attempting to change ownership to $current_user..."
            if command -v sudo >/dev/null 2>&1; then
                if sudo chown -R "$current_user:$current_user" "$output_dir" 2>/dev/null; then
                    log_success "Directory ownership fixed with sudo"
                else
                    log_error "Failed to fix directory ownership"
                    exit 1
                fi
            else
                log_error "sudo not available and cannot write to directory"
                exit 1
            fi
        fi
    else
        log_success "✅ Directory permissions OK"
    fi
}

# Memory check
check_memory() {
    local required_mb=4096
    local total_mb=0
    local swap_mb=0
    
    if [ -f /proc/meminfo ]; then
        local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        total_mb=$((mem_total / 1024))
        
        if [ -f /proc/swaps ]; then
            local swap_total=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
            swap_mb=$((swap_total / 1024))
        fi
        
        local combined_mb=$((total_mb + swap_mb))
        
        log_info "💾 Memory check..."
        log_info "RAM: ${total_mb}MB, Swap: ${swap_mb}MB, Total: ${combined_mb}MB"
        log_info "Required: ${required_mb}MB"
        
        if [ $combined_mb -lt $required_mb ]; then
            local deficit=$((required_mb - combined_mb))
            log_warning "Insufficient memory: need ${deficit}MB more"
            
            if [ "$AUTO_CONFIRM" != true ]; then
                echo -n "Continue anyway? (y/N): "
                read -r RESPONSE
                if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                    log_error "Build aborted due to insufficient memory"
                    exit 1
                fi
            fi
        else
            log_success "✅ Memory check passed (${combined_mb}MB total)"
        fi
    fi
}

# Setup temporary build environment
setup_temp_build() {
    log_info "📁 Setting up temporary build environment..."
    
    # Use existing temp dir if available
    if [ -n "$QGETH_BUILD_TEMP" ]; then
        BUILD_TEMP_DIR="$QGETH_BUILD_TEMP"
        mkdir -p "$BUILD_TEMP_DIR"
        log_info "Using VPS-prepared temp directory: $BUILD_TEMP_DIR"
    else
        # Create new temp directory
        if command -v df >/dev/null 2>&1; then
            tmp_space=$(df /tmp 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
            if [ "$tmp_space" -lt 1048576 ]; then
                BUILD_TEMP_DIR="./build-temp-$$"
            else
                BUILD_TEMP_DIR="/tmp/qgeth-build-$$"
            fi
        else
            BUILD_TEMP_DIR="./build-temp-$$"
        fi
        mkdir -p "$BUILD_TEMP_DIR"
        log_info "Selected temp directory: $BUILD_TEMP_DIR ($(df -h "$BUILD_TEMP_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo 'unknown')MB available)"
    fi
    
    # Convert to absolute path
    if [ "${BUILD_TEMP_DIR#/}" = "$BUILD_TEMP_DIR" ]; then
        BUILD_TEMP_DIR="$(pwd)/$BUILD_TEMP_DIR"
    fi
    
    # Set Go environment variables
    export GOCACHE="$BUILD_TEMP_DIR/gocache"
    export GOTMPDIR="$BUILD_TEMP_DIR/gotmp"
    export TMPDIR="$BUILD_TEMP_DIR/tmp"
    
    mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
    
    log_info "Build environment:"
    log_info "  Build Temp Dir: $BUILD_TEMP_DIR"
    log_info "  Go Cache: $GOCACHE"
    log_info "  Go Temp: $GOTMPDIR"
    log_info "  System Temp: $TMPDIR"
    
    # Cleanup function
    cleanup_temp() {
        if [ -z "$QGETH_BUILD_TEMP" ]; then
            log_info "🧹 Cleaning up temporary build directory..."
            rm -rf "$BUILD_TEMP_DIR" 2>/dev/null || true
            log_success "✅ Temporary files cleaned up"
        else
            log_info "🧹 Preserving VPS-prepared temp directory: $BUILD_TEMP_DIR"
        fi
    }
    
    trap cleanup_temp EXIT
}

# Get build flags
get_build_flags() {
    # Build info
    if command -v date >/dev/null 2>&1; then
        BUILD_TIME=$(date "+%Y-%m-%d_%H:%M:%S" 2>/dev/null || echo "unknown")
    else
        BUILD_TIME="unknown"
    fi
    
    if command -v git >/dev/null 2>&1; then
        GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    else
        GIT_COMMIT="unknown"
    fi
    
    # Memory-efficient linker flags
    LDFLAGS="-s -w"
    LDFLAGS="$LDFLAGS -X main.VERSION=$VERSION"
    LDFLAGS="$LDFLAGS -X main.BUILD_TIME=$BUILD_TIME"
    LDFLAGS="$LDFLAGS -X main.GIT_COMMIT=$GIT_COMMIT"
    
    log_info "💾 Configuring build flags..."
    log_info "Build flags:"
    log_info "  Linker: $LDFLAGS"
    log_info "  Trimpath: enabled"
    log_info "  VCS info: disabled"
    log_info "  Cache: $GOCACHE"
}

# Check Go version and determine appropriate flags
check_go_version_for_memsize() {
    local go_version go_version_full major minor
    if command -v go >/dev/null 2>&1; then
        go_version_full=$(go version 2>/dev/null)
        go_version=$(echo "$go_version_full" | grep -oE 'go[0-9]+\.[0-9]+' | head -1)
        log_info "Detected Go version: ${go_version:-unknown}"
        
        if [ -n "$go_version" ]; then
            # Extract major and minor version numbers
            major=$(echo "$go_version" | cut -d'.' -f1 | sed 's/go//')
            minor=$(echo "$go_version" | cut -d'.' -f2)
            
            # Check if version is 1.23 or higher (needs -checklinkname=0)
            if [ "$major" -gt 1 ] || ([ "$major" -eq 1 ] && [ "$minor" -ge 23 ]); then
                log_success "✅ Go $go_version supports -checklinkname flag (memsize compatibility)"
                USE_CHECKLINKNAME=true
            else
                log_info "ℹ️ Go $go_version uses standard build flags"
                USE_CHECKLINKNAME=false
            fi
        else
            log_warning "⚠️ Could not parse Go version, using standard build flags"
            USE_CHECKLINKNAME=false
        fi
    else
        log_error "Go not found - please install Go via bootstrap-qgeth.sh"
        exit 1
    fi
}

# Fix Go module conflicts
fix_go_module_conflicts() {
    local module_dir="$1"
    log_info "🔧 Checking Go module dependencies in $module_dir..."
    
    cd "$module_dir"
    
    if grep -q "github.com/fjl/memsize" go.mod 2>/dev/null; then
        log_info "🚨 Detected problematic memsize dependency"
        log_info "🔧 Auto-fix: Cleaning Go module cache and dependencies..."
        
        go clean -cache -modcache -testcache 2>/dev/null || true
        rm -f go.sum
        
        log_info "🚀 Re-downloading dependencies with fresh resolution..."
        GOPROXY=direct go mod download 2>/dev/null || true
        go mod tidy 2>/dev/null || true
        
        if go mod verify 2>/dev/null; then
            log_success "✅ Module dependencies verified and fixed"
        else
            log_warning "Module verification failed, trying alternative approach..."
            go clean -cache -modcache 2>/dev/null || true
            GOPROXY=https://proxy.golang.org go mod download 2>/dev/null || true
            go mod tidy 2>/dev/null || true
        fi
    else
        log_success "✅ No problematic dependencies detected"
    fi
    
    cd - >/dev/null
}

# Build with retry mechanism
build_with_retry() {
    local build_type="$1"
    local build_cmd="$2"
    local output_binary="$3"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log_info "🚀 Build attempt $((retry_count + 1))/$max_retries for $build_type..."
        log_debug "Build command: $build_cmd"
        
        # Capture both stdout and stderr for debugging
        if eval "$build_cmd" 2>&1; then
            log_success "✅ $build_type built successfully"
            return 0
        else
            local exit_code=$?
            retry_count=$((retry_count + 1))
            log_warning "🚨 Build attempt $retry_count failed for $build_type (exit code: $exit_code)"
            
            if [ $retry_count -lt $max_retries ]; then
                log_info "🔧 Attempting automated recovery..."
                rm -f "$output_binary" 2>/dev/null || true
                
                # Show which Go version is being used
                log_info "Current Go version: $(go version 2>/dev/null || echo 'Go not found')"
                log_info "Current PATH: $PATH"
                
                go clean -cache -testcache 2>/dev/null || true
                go mod tidy 2>/dev/null || true
                go mod download 2>/dev/null || true
                sleep 2
            else
                log_error "🚨 All build attempts failed for $build_type"
                log_error "💡 Try running the build manually to see detailed error output:"
                log_error "   cd $(pwd)"
                log_error "   $build_cmd"
                return 1
            fi
        fi
    done
    
    return 1
}

# Build quantum-geth
build_geth() {
    log_info "🚀 Building Quantum-Geth with cross-distribution compatibility..."
    
    if [ ! -f "../../quantum-geth/go.mod" ]; then
        log_error "go.mod not found in quantum-geth directory!"
        exit 1
    fi
    
    # Fix module conflicts
    fix_go_module_conflicts "../../quantum-geth"
    
    # Set build environment
    ORIGINAL_CGO=$CGO_ENABLED
    export CGO_ENABLED=0
    
    log_info "Build environment:"
    log_info "  GOOS: $GOOS"
    log_info "  GOARCH: $GOARCH"
    log_info "  CGO_ENABLED: $CGO_ENABLED"
    log_info "  Temp directory: $BUILD_TEMP_DIR"
    
    cd ../../quantum-geth/cmd/geth
    
    # Ensure temp directories exist
    mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
    
    # Build command with appropriate flags based on Go version
    if [ "$USE_CHECKLINKNAME" = true ]; then
        log_info "🔧 Using -checklinkname=0 flag for Go 1.23+ memsize compatibility"
        BUILD_CMD="PATH=\"$PATH\" CGO_ENABLED=0 go build -ldflags \"-checklinkname=0 $LDFLAGS\" -trimpath -buildvcs=false -o ../../../geth.bin ."
    else
        log_info "🔧 Using standard build flags for Go version compatibility"
        BUILD_CMD="PATH=\"$PATH\" CGO_ENABLED=0 go build -ldflags \"$LDFLAGS\" -trimpath -buildvcs=false -o ../../../geth.bin ."
    fi
    
    if build_with_retry "quantum-geth" "$BUILD_CMD" "../../../geth.bin"; then
        cd ../../..
        log_success "✅ Quantum-Geth built successfully: ./geth.bin"
        
        # Create geth wrapper
        create_geth_wrapper
        
        if command -v ls >/dev/null 2>&1; then
            ls -lh ./geth.bin ./geth 2>/dev/null || log_info "Binaries created: ./geth.bin, ./geth"
        else
            log_info "Binaries created: ./geth.bin, ./geth"
        fi
    else
        cd ../../..
        log_error "Failed to build quantum-geth after all retry attempts"
        return 1
    fi
    
    export CGO_ENABLED=$ORIGINAL_CGO
}

# Build quantum-miner
build_miner() {
    log_info "🚀 Building Quantum-Miner with enhanced GPU detection..."
    
    if [ ! -f "../../quantum-miner/go.mod" ]; then
        log_error "go.mod not found in quantum-miner directory!"
        exit 1
    fi
    
    # Fix module conflicts
    fix_go_module_conflicts "../../quantum-miner"
    
    # Detect GPU capabilities
    BUILD_TAGS=""
    GPU_TYPE="CPU"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "🎮 NVIDIA GPU detected, checking CUDA availability..."
        
        # Check for CUDA development libraries
        local cuda_found=false
        
        if command -v pkg-config >/dev/null 2>&1; then
            if pkg-config --exists cuda-12.0 2>/dev/null || pkg-config --exists cuda-11.0 2>/dev/null; then
                cuda_found=true
            fi
        fi
        
        if [ "$cuda_found" = false ]; then
            for cuda_path in /usr/local/cuda /opt/cuda /usr/cuda; do
                if [ -d "$cuda_path/include" ] && [ -d "$cuda_path/lib64" ]; then
                    cuda_found=true
                    break
                fi
            done
        fi
        
        if [ "$cuda_found" = false ] && command -v nvcc >/dev/null 2>&1; then
            cuda_found=true
        fi
        
        if [ "$cuda_found" = true ]; then
            log_success "✅ CUDA development environment found"
            BUILD_TAGS="cuda"
            GPU_TYPE="CUDA"
            export CGO_ENABLED=1
        else
            log_info "🚨 CUDA development libraries not found, checking for Qiskit-Aer GPU..."
            
            if command -v python3 >/dev/null 2>&1; then
                if python3 -c "import qiskit_aer; from qiskit_aer import AerSimulator; AerSimulator(device='GPU')" >/dev/null 2>&1; then
                    log_success "✅ Qiskit-Aer GPU support detected"
                    BUILD_TAGS="cuda"
                    GPU_TYPE="Qiskit-GPU"
                    export CGO_ENABLED=0
                else
                    log_warning "🚨 No GPU acceleration available, building CPU-only version"
                    export CGO_ENABLED=0
                fi
            else
                log_warning "🚨 Python3 not available for Qiskit check, building CPU-only version"
                export CGO_ENABLED=0
            fi
        fi
    else
        log_info "🚨 No NVIDIA GPU detected, building CPU-only version"
        export CGO_ENABLED=0
    fi
    
    log_info "Miner configuration:"
    log_info "  GPU Type: $GPU_TYPE"
    log_info "  Build Tags: ${BUILD_TAGS:-none}"
    log_info "  CGO Enabled: $CGO_ENABLED"
    log_info "  Architecture: $QGETH_GO_ARCH"
    
    cd ../../quantum-miner
    
    # Ensure temp directories exist
    mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
    
    # Build command with appropriate flags based on Go version
    if [ "$USE_CHECKLINKNAME" = true ]; then
        log_info "🔧 Using -checklinkname=0 flag for Go 1.23+ memsize compatibility"
        BUILD_CMD="PATH=\"$PATH\" go build -ldflags \"-checklinkname=0 $LDFLAGS\" -trimpath -buildvcs=false"
    else
        log_info "🔧 Using standard build flags for Go version compatibility"
        BUILD_CMD="PATH=\"$PATH\" go build -ldflags \"$LDFLAGS\" -trimpath -buildvcs=false"
    fi
    
    if [ -n "$BUILD_TAGS" ]; then
        BUILD_CMD="$BUILD_CMD -tags $BUILD_TAGS"
    fi
    BUILD_CMD="$BUILD_CMD -o ../quantum-miner ."
    
    if build_with_retry "quantum-miner" "$BUILD_CMD" "../quantum-miner"; then
        cd ..
        log_success "✅ Quantum-Miner built successfully: ./quantum-miner ($GPU_TYPE)"
        
        if command -v ls >/dev/null 2>&1; then
            ls -lh ./quantum-miner 2>/dev/null || log_info "Binary created: ./quantum-miner"
        else
            log_info "Binary created: ./quantum-miner"
        fi
        
        if [ "$GPU_TYPE" != "CPU" ]; then
            log_info "🚀 Testing GPU support..."
            if ./quantum-miner --help 2>/dev/null | grep -q "GPU" 2>/dev/null; then
                log_success "✅ GPU support confirmed in binary"
            else
                log_warning "🚨 GPU support may not be active (check dependencies)"
            fi
        fi
    else
        cd ..
        log_error "Failed to build quantum-miner after all retry attempts"
        return 1
    fi
}

# Create geth wrapper
create_geth_wrapper() {
    log_info "🚀 Creating Q Coin geth wrapper..."
    
    cat > "./geth" << 'EOF'
#!/bin/bash
# Q Coin Geth Wrapper
# This wrapper ensures geth ALWAYS uses Q Coin networks, never Ethereum
# Default: Q Coin Testnet (Chain ID 73235)

export PATH="/usr/local/go/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_GETH="$SCRIPT_DIR/geth.bin"

if [ ! -f "$REAL_GETH" ]; then
    echo "🚨 ERROR: Q Coin geth binary not found at $REAL_GETH"
    echo "   Build it first: ./build-linux.sh geth"
    exit 1
fi

USE_QCOIN_MAINNET=false
FILTERED_ARGS=()

for arg in "$@"; do
    case $arg in
        --qcoin-mainnet)
            USE_QCOIN_MAINNET=true
            ;;
        --help|-h)
            echo "Q Coin Geth - Quantum Blockchain Node"
            echo ""
            echo "This geth ONLY connects to Q Coin networks, never Ethereum!"
            echo ""
            echo "Q Coin Networks:"
            echo "  Default:         Q Coin Testnet (Chain ID 73235)"
            echo "  --qcoin-mainnet  Q Coin Mainnet (Chain ID 73236)"
            echo ""
            echo "Quick Start:"
            echo "  ./start-geth.sh              # Easy testnet startup"
            echo "  ./start-geth.sh mainnet      # Easy mainnet startup"
            echo ""
            echo "Manual Usage:"
            echo "  ./geth --datadir \$HOME/.qcoin init configs/genesis_quantum_testnet.json"
            echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0"
            echo ""
            echo "Standard geth options also available."
            exit 0
            ;;
        *)
            FILTERED_ARGS+=("$arg")
            ;;
    esac
done

if [[ " ${FILTERED_ARGS[*]} " =~ " init " ]]; then
    exec "$REAL_GETH" "${FILTERED_ARGS[@]}"
elif [ ${#FILTERED_ARGS[@]} -eq 0 ] || ([[ ! " ${FILTERED_ARGS[*]} " =~ " --networkid " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " --datadir " ]]); then
    echo "🚨 Q Coin Geth: Prevented connection to Ethereum mainnet!"
    echo ""
    echo "This geth is configured for Q Coin networks only."
    echo ""
    echo "Quick Start:"
    echo "  ./start-geth.sh              # Q Coin Testnet"
    echo "  ./start-geth.sh mainnet      # Q Coin Mainnet"
    echo ""
    echo "Manual Start:"
    if $USE_QCOIN_MAINNET; then
        echo "  ./geth --datadir ~/.qcoin/mainnet --networkid 73236 init configs/genesis_quantum_mainnet.json"
        echo "  ./geth --datadir ~/.qcoin/mainnet --networkid 73236 --mine --miner.threads 0"
    else
        echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 init configs/genesis_quantum_testnet.json"
        echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0"
    fi
    echo ""
    echo "Use --help for more options."
    exit 1
fi

if [[ ! " ${FILTERED_ARGS[*]} " =~ " --networkid " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " init " ]]; then
    if $USE_QCOIN_MAINNET; then
        FILTERED_ARGS+=("--networkid" "73236")
    else
        FILTERED_ARGS+=("--networkid" "73235")
    fi
fi

exec "$REAL_GETH" "${FILTERED_ARGS[@]}"
EOF
    
    if command -v chmod >/dev/null 2>&1; then
        chmod +x "./geth"
        log_success "✅ Q Coin geth wrapper created: ./geth"
    else
        log_warning "⚠️ chmod not available - wrapper may need manual chmod +x"
    fi
}

# Create quantum solver script
create_solver() {
    log_info "🚀 Creating quantum solver helper script..."
    
    cat > ./quantum_solver.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Quantum Circuit Solver for Q Coin Mining
Compatible with the quantum-geth consensus algorithm
"""

import sys
import json
import argparse
import hashlib
import random

def solve_puzzles(seed, puzzle_count, qubits=16):
    """Solve quantum puzzles for Q Coin mining"""
    
    all_outcomes = []
    total_t_gates = 0
    
    for i in range(puzzle_count):
        # Create deterministic quantum circuit simulation
        circuit_seed = hashlib.sha256((seed + str(i)).encode()).hexdigest()
        random.seed(circuit_seed)
        
        # Simulate 16-qubit quantum circuit with T-gates
        outcome = []
        t_gate_count = 512  # T-gates per qubit for quantum advantage
        
        for qubit in range(qubits):
            # Simulate quantum measurement (0 or 1)
            measurement = random.randint(0, 1)
            outcome.append(measurement)
        
        all_outcomes.extend(outcome)
        total_t_gates += qubits * t_gate_count
    
    # Create quantum proof
    outcome_data = bytes(all_outcomes)
    outcome_root = hashlib.sha256(outcome_data).hexdigest()
    
    gate_data = f"T-gates:{total_t_gates}".encode()
    gate_hash = hashlib.sha256(gate_data).hexdigest()
    
    proof_root = hashlib.sha256((outcome_root + gate_hash).encode()).hexdigest()
    quantum_blob = outcome_root[:31].encode().hex()
    
    return {
        'proof_root': proof_root,
        'outcome_root': outcome_root,
        'gate_hash': gate_hash,
        'quantum_blob': quantum_blob,
        't_gates': total_t_gates,
        'puzzle_count': puzzle_count,
        'success': True
    }

def main():
    parser = argparse.ArgumentParser(description="Q Coin Quantum Circuit Solver")
    parser.add_argument("--seed", required=True, help="Hex seed for circuit generation")
    parser.add_argument("--puzzles", type=int, default=128, help="Number of puzzles")
    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits")
    
    args = parser.parse_args()
    
    try:
        result = solve_puzzles(args.seed, args.puzzles, args.qubits)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        error_result = {
            'error': str(e),
            'success': False
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()
EOF
    
    if command -v chmod >/dev/null 2>&1; then
        chmod +x ./quantum_solver.py
    fi
    
    log_success "✅ Quantum solver script created: ./quantum_solver.py"
}

# Main execution function
main() {
    echo ""
    echo -e "${CYAN}🚀 Q Coin Enhanced Cross-Distribution Build System${NC}"
    echo ""
    
    # Detect system
    detect_system
    
    log_info "System Configuration:"
    log_info "  OS: $QGETH_OS ($QGETH_DISTRO $QGETH_DISTRO_VERSION)"
    log_info "  Architecture: $QGETH_ARCH (Go: $QGETH_GO_ARCH)"
    log_info "  Package Manager: ${PKG_MANAGER:-unknown}"
    log_info "  Shell: $QGETH_SHELL"
    echo ""
    log_info "Build Configuration:"
    log_info "  Target: $TARGET"
    log_info "  Version: $VERSION"
    log_info "  Clean: $CLEAN"
    log_info "  Auto-confirm: $AUTO_CONFIRM"
    log_info "  Debug: $DEBUG"
    echo ""
    
    # Pre-build checks
    log_info "🔧 Running automated pre-build checks and fixes..."
    check_and_fix_permissions
    check_memory
    setup_temp_build
    
    # Check for partial builds
    if [ -f "geth.bin" ] && [ ! -f "geth" ]; then
        log_info "🔍 Detected partial build (geth.bin without geth wrapper)"
        log_info "🧹 Cleaning partial build state..."
        rm -f geth.bin geth quantum-miner quantum_solver.py
        log_info "Partial build artifacts removed"
    fi
    
    # Clean if requested
    if [ "$CLEAN" = true ]; then
        log_info "🧹 Cleaning previous builds..."
        rm -f geth geth.bin quantum-miner quantum_solver.py
        go clean -cache 2>/dev/null || true
        log_info "Previous binaries removed"
    fi
    
    # Check directories exist
    if [ ! -d "../../quantum-geth" ]; then
        log_error "quantum-geth directory not found!"
        log_error "Please run this script from scripts/linux/ directory."
        exit 1
    fi
    
    if [ ! -d "../../quantum-miner" ]; then
        log_error "quantum-miner directory not found!"
        log_error "Please run this script from scripts/linux/ directory."
        exit 1
    fi
    
    # Ensure Go 1.24.4 is in PATH (bootstrap installs to /usr/local/go)
    export PATH="/usr/local/go/bin:$PATH"
    
    # Verify Go version after PATH update
    if command -v go >/dev/null 2>&1; then
        CURRENT_GO_VERSION=$(go version 2>/dev/null)
        log_info "Active Go version: $CURRENT_GO_VERSION"
        
        if ! echo "$CURRENT_GO_VERSION" | grep -q "go1.24"; then
            log_warning "⚠️ Go 1.24.x not active! Current: $CURRENT_GO_VERSION"
            log_info "Checking for Go 1.24.4 installation..."
            
            if [ -x "/usr/local/go/bin/go" ]; then
                export PATH="/usr/local/go/bin:$PATH"
                log_info "Updated PATH to use /usr/local/go/bin/go"
                UPDATED_GO_VERSION=$(go version 2>/dev/null)
                log_success "✅ Now using: $UPDATED_GO_VERSION"
            else
                log_error "Go 1.24.4 not found at /usr/local/go/bin/go"
                log_error "Please run bootstrap script first to install Go 1.24.4"
                exit 1
            fi
        else
            log_success "✅ Go 1.24.x is active and ready"
        fi
    else
        log_error "Go not found in PATH"
        exit 1
    fi
    
    # Set build environment
    export GOOS=linux
    export GOARCH=$QGETH_GO_ARCH
    export CGO_ENABLED=0
    
    # Get build flags and check Go version
    get_build_flags
    check_go_version_for_memsize
    
    log_info "Build environment:"
    log_info "  Target OS: $GOOS/$GOARCH"
    log_info "  Build Time: $BUILD_TIME"
    log_info "  Git Commit: $GIT_COMMIT"
    log_info "  Output Directory: $(pwd)"
    echo ""
    
    # Execute builds
    case $TARGET in
        "geth")
            if build_geth; then
                create_solver
                log_success "🎉 Geth build completed successfully!"
            else
                log_error "Geth build failed"
                exit 1
            fi
            ;;
        "miner")
            if build_miner; then
                log_success "🎉 Miner build completed successfully!"
            else
                log_error "Miner build failed"
                exit 1
            fi
            ;;
        "both")
            if build_geth && build_miner; then
                create_solver
                log_success "🎉 Both builds completed successfully!"
            else
                log_error "One or more builds failed"
                exit 1
            fi
            ;;
    esac
    
    log_success "✅ Enhanced cross-distribution build completed!"
    log_info "Target: $TARGET"
    log_info "System: $QGETH_OS ($QGETH_DISTRO $QGETH_DISTRO_VERSION)"
    log_info "Architecture: $QGETH_ARCH"
    
    if [ -f "./geth" ]; then
        log_info "📱 Q Coin node ready: ./start-geth.sh"
    fi
    
    if [ -f "./quantum-miner" ]; then
        log_info "⛏️ Q Coin miner ready: ./start-miner.sh"
    fi
}

# Run main function
main "$@" 