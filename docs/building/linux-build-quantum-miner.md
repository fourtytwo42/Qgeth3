# Linux Quantum Miner Build Guide

Complete guide for building Q Coin quantum-miner from source on Linux systems with GPU/CPU acceleration support.

## 📋 Prerequisites

### System Requirements
- **OS:** Ubuntu 20.04+, Debian 11+, CentOS 8+, or compatible Linux distribution
- **CPU:** 4+ cores (8+ recommended for competitive mining)
- **RAM:** 4GB minimum (8GB+ recommended for GPU mining)
- **Storage:** 3GB free space for source code and build artifacts
- **GPU:** NVIDIA GPU with CUDA support (optional, for maximum performance)

### Required Software
- **Go 1.21+** (mandatory for building)
- **Python 3.8+** (for quantum algorithms and GPU acceleration)
- **Git** (for source code management)
- **gcc/make** (C compiler and build tools)
- **CUDA Toolkit** (optional, for native CUDA acceleration)

## 🛠️ Installing Dependencies

### Basic Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y golang-go python3 python3-pip git build-essential pkg-config curl wget

# CentOS/RHEL/Fedora
sudo dnf update -y
sudo dnf install -y golang python3 python3-pip git gcc make curl wget
sudo dnf groupinstall -y "Development Tools"

# Verify installations
go version    # Should be 1.21+
python3 --version  # Should be 3.8+
```

### Python Quantum Dependencies
```bash
# Core quantum computing libraries
pip3 install qiskit qiskit-aer numpy

# For advanced GPU acceleration (optional)
pip3 install qiskit-aer-gpu

# Additional optimization libraries
pip3 install scipy numba

# Verify quantum libraries
python3 -c "import qiskit; print('Qiskit OK')"
python3 -c "from qiskit_aer import AerSimulator; print('Aer OK')"
```

### NVIDIA GPU Setup (Optional)
```bash
# Check if NVIDIA GPU available
nvidia-smi

# Install CUDA toolkit for native acceleration
# Ubuntu/Debian
sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev

# CentOS/RHEL/Fedora
sudo dnf install -y cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi

# Test GPU quantum acceleration
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

### Manual Go Installation (Latest)
```bash
# If system Go is too old
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
go version
```

## 📥 Getting the Source Code

### Clone Repository
```bash
# Clone Q Coin repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Verify source structure
ls -la
# Should see: quantum-geth/, tools/, scripts/, etc.

# Check miner source location
ls -la quantum-geth/tools/solver/
# Should see: main.go, *.go files
```

### Source Code Structure
```
Qgeth3/
├── quantum-geth/
│   └── tools/
│       └── solver/           # Quantum miner source code
│           ├── main.go       # Main miner executable
│           ├── quantum.go    # Quantum algorithm implementation
│           ├── gpu.go        # GPU acceleration code
│           └── ...
├── scripts/linux/           # Linux build scripts
└── ...
```

## 🔨 Building Quantum Miner

### Using Build Script (Recommended)
```bash
# Make scripts executable
chmod +x scripts/linux/*.sh

# Build miner with automatic GPU detection
./scripts/linux/build-linux.sh miner

# Expected outputs based on capabilities:
# ✅ Quantum-Miner built successfully: ./quantum-miner (CUDA)      # Native CUDA
# ✅ Quantum-Miner built successfully: ./quantum-miner (Qiskit-GPU) # Python GPU
# ✅ Quantum-Miner built successfully: ./quantum-miner (CPU)       # CPU fallback
```

### Manual Build Process
```bash
# Navigate to miner source
cd quantum-geth/tools/solver

# Set build environment
export CGO_ENABLED=1  # Required for Python integration
export GOOS=linux
export GOARCH=amd64

# Set build metadata
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT=$(git rev-parse HEAD)

# Build with quantum optimizations
go build \
  -ldflags="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME" \
  -o ../../../quantum-miner \
  .

# Return to root directory
cd ../../..

# Verify build
ls -la quantum-miner
file quantum-miner  # Should show: ELF 64-bit LSB executable
```

### GPU Detection and Optimization
The build system automatically detects available acceleration:

```bash
# GPU detection during build
echo "=== GPU Detection ==="

# Check for NVIDIA CUDA
if command -v nvcc &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected - Native GPU acceleration available"
    BUILD_TAG="cuda"
fi

# Check for Qiskit GPU support
if python3 -c "from qiskit_aer import AerSimulator; AerSimulator(device='GPU')" &> /dev/null; then
    echo "✅ Qiskit GPU detected - Python GPU acceleration available"
    BUILD_TAG="qiskit_gpu"
fi

# Fallback to CPU
if [[ -z "$BUILD_TAG" ]]; then
    echo "⚠️  No GPU detected - CPU mode only"
    BUILD_TAG="cpu"
fi
```

## ⚙️ Build Configuration

### GPU Acceleration Modes

#### Native CUDA Mode (Best Performance)
```bash
# Requirements: CUDA toolkit + NVIDIA GPU
# Expected performance: ~3.0-5.0 puzzles/sec (RTX 3080)

# Check CUDA availability
nvcc --version
nvidia-smi

# Build with CUDA tags
cd quantum-geth/tools/solver
go build -tags cuda -o ../../../quantum-miner .
```

#### Qiskit GPU Mode (Good Performance)
```bash
# Requirements: Python + qiskit-aer-gpu
# Expected performance: ~2.0-3.5 puzzles/sec (RTX 3080)

# Install GPU-enabled Qiskit
pip3 install qiskit-aer-gpu

# Verify GPU support
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"

# Build with Qiskit GPU support
cd quantum-geth/tools/solver
go build -tags qiskit_gpu -o ../../../quantum-miner .
```

#### CPU Mode (Fallback)
```bash
# Requirements: Python + qiskit
# Expected performance: ~0.1-0.5 puzzles/sec (8-core CPU)

# Build CPU-only version
cd quantum-geth/tools/solver
go build -tags cpu -o ../../../quantum-miner .
```

### Build Tags and Features
```bash
# Available build tags
-tags cuda          # Native CUDA acceleration
-tags qiskit_gpu    # Python GPU acceleration via Qiskit
-tags qiskit_cpu    # Python CPU acceleration
-tags cpu           # Pure Go CPU implementation
-tags debug         # Debug logging enabled
-tags profile       # Performance profiling enabled

# Combined tags example
go build -tags "cuda,debug" -o quantum-miner .
```

### Optimization Levels
```bash
# Performance optimized (default)
go build -ldflags="-s -w" -o quantum-miner .

# Debug build (with symbols)
go build -gcflags="-N -l" -tags debug -o quantum-miner.debug .

# Profiling build
go build -tags profile -o quantum-miner.profile .

# Size optimized
go build -ldflags="-s -w" -trimpath -o quantum-miner .
```

## ✅ Verification

### Test Build
```bash
# Check binary exists and is executable
ls -la quantum-miner
chmod +x quantum-miner

# Test basic functionality
./quantum-miner --help

# Expected output should include:
# Usage: quantum-miner [options]
# Options:
#   --gpu              Enable GPU acceleration
#   --threads N        Number of CPU threads
#   --coinbase ADDR    Mining reward address
#   --testnet          Use testnet
```

### GPU Acceleration Test
```bash
# Test GPU capabilities
./quantum-miner --gpu --test --verbose

# Expected outputs based on build:
# ✅ CUDA GPU detected: GeForce RTX 3080 (10240 MB)
# ✅ Native CUDA acceleration enabled
# ⚡ GPU Batch Complete: 1024 puzzles in 0.342s (2991.2 puzzles/sec)

# OR for Qiskit GPU:
# ✅ Qiskit GPU backend available
# ⚡ GPU Batch Complete: 512 puzzles in 0.256s (2000.0 puzzles/sec)

# OR for CPU fallback:
# ⚠️  No GPU acceleration available, using CPU
# ⚡ CPU Batch Complete: 64 puzzles in 0.128s (500.0 puzzles/sec)
```

### Performance Benchmarking
```bash
# Run mining benchmark
./quantum-miner --benchmark --duration 30s

# CPU benchmark
./quantum-miner --cpu --threads $(nproc) --benchmark --duration 30s

# GPU benchmark (if available)
./quantum-miner --gpu --benchmark --duration 30s

# Expected benchmark results:
# Platform        | Puzzles/sec | Power Usage
# ----------------|-------------|------------
# RTX 4090 (CUDA) | 4.5-6.0    | ~350W
# RTX 3080 (CUDA) | 3.0-4.5    | ~320W
# RX 6800 XT (CPU)| 0.4-0.7    | ~300W
# 16-core CPU     | 0.8-1.2    | ~150W
# 8-core CPU      | 0.3-0.6    | ~95W
```

### Quantum Algorithm Verification
```bash
# Test quantum computation correctness
./quantum-miner --verify --test-vectors

# This runs the miner against known test vectors to ensure:
# - Quantum algorithm implementation is correct
# - Hash computations match expected values
# - GPU/CPU results are identical
# - No arithmetic errors in acceleration code
```

## 📂 Build Artifacts

### Generated Files
```bash
# Main executable
quantum-miner               # Main miner binary (5-15MB typically)

# Optional debug/profiling binaries
quantum-miner.debug        # Debug version with symbols
quantum-miner.profile      # Profiling version

# Temporary build files (cleaned automatically)
build-temp-*/              # Temporary build directories
```

### Installation
```bash
# Install to system PATH (optional)
sudo cp quantum-miner /usr/local/bin/qminer
sudo chmod +x /usr/local/bin/qminer

# Create symlink
sudo ln -sf $(pwd)/quantum-miner /usr/local/bin/qminer

# Test system installation
qminer --version
qminer --help
```

## 🔧 Advanced Build Options

### Multi-Architecture Builds
```bash
# Build for different architectures
export GOARCH=amd64    # x86_64 (default)
# export GOARCH=arm64  # ARM64 (Raspberry Pi 4, Apple M1)
# export GOARCH=386    # x86_32

go build -o quantum-miner-$GOARCH .
```

### Custom CUDA Configuration
```bash
# Specify CUDA compute capability
export CUDA_ARCH="75"  # RTX 20xx series
# export CUDA_ARCH="86"  # RTX 30xx series
# export CUDA_ARCH="89"  # RTX 40xx series

# Build with specific CUDA optimization
cd quantum-geth/tools/solver
go build -tags cuda -ldflags="-X main.cudaArch=$CUDA_ARCH" -o ../../../quantum-miner .
```

### Python Integration Tuning
```bash
# Set Python path explicitly
export PYTHONPATH="/usr/local/lib/python3.10/site-packages"

# Use specific Python version
export PYTHON_BINARY="/usr/bin/python3.10"

# Build with custom Python configuration
go build -ldflags="-X main.pythonPath=$PYTHONPATH" -o quantum-miner .
```

### Memory and Performance Tuning
```bash
# For systems with limited RAM
export GOMEMLIMIT=2GiB
go build -o quantum-miner .

# For high-performance systems
export GOMAXPROCS=$(nproc)
go build -p $(nproc) -o quantum-miner .

# Enable hardware-specific optimizations
export GOAMD64=v3  # For modern AMD64 CPUs
go build -o quantum-miner .
```

## 🚀 Post-Build Optimization

### System Tuning for Mining
```bash
# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase process priority
sudo sysctl -w kernel.sched_rt_runtime_us=-1

# Optimize GPU for mining (NVIDIA)
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -pl 300  # Set power limit (adjust for your card)

# Increase file descriptor limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf
```

### Environment Configuration
```bash
# Create mining-optimized environment script
cat > setup-mining-env.sh << 'EOF'
#!/bin/bash
# Q Coin Mining Environment Setup

export CGO_ENABLED=1
export GOMEMLIMIT=4GiB
export GOMAXPROCS=$(nproc)

# CUDA optimization
export CUDA_CACHE_PATH=/tmp/cuda-cache
export CUDA_CACHE_MAXSIZE=2147483648

# Python optimization
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=1

# Mining configuration
export QMINER_GPU_MEMORY_FRACTION=0.9
export QMINER_CPU_THREADS=$(nproc)

echo "Mining environment configured!"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not available')"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
EOF

chmod +x setup-mining-env.sh
```

## 📈 Performance Analysis

### Mining Performance Monitoring
```bash
# Monitor miner performance
./quantum-miner --stats --interval 10s

# System resource monitoring while mining
watch -n 5 'echo "=== CPU ===" && top -n1 -b | head -5 && echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader && echo "=== Network ===" && ss -tuln | grep 8545'

# Performance profiling
./quantum-miner.profile --profile-cpu --profile-mem --duration 60s
```

### Optimization Recommendations
```bash
# Optimal thread configuration per hardware
# Single RTX 4090: --gpu (no CPU threads needed)
# RTX 3080 + 12-core CPU: --gpu --cpu-threads 4
# CPU-only 16 cores: --cpu-threads 14 (leave 2 for system)
# CPU-only 8 cores: --cpu-threads 6

# Memory usage optimization
# 8GB RAM: --batch-size 256
# 16GB RAM: --batch-size 512 (default)
# 32GB+ RAM: --batch-size 1024
```

## 🔍 Troubleshooting

For build issues, see the [Linux Quantum Miner Build Troubleshooting Guide](troubleshooting-linux-build-quantum-miner.md).

Common quick fixes:
```bash
# Update Python dependencies
pip3 install --upgrade qiskit qiskit-aer numpy

# Rebuild with clean environment
go clean -cache && go clean -modcache
./scripts/linux/build-linux.sh miner --clean

# Test GPU separately
python3 -c "from qiskit_aer import AerSimulator; print('GPU OK' if AerSimulator(device='GPU') else 'GPU Failed')"
```

## ✅ Build Checklist

### Pre-Build
- [ ] Go 1.21+ installed (`go version`)
- [ ] Python 3.8+ with quantum libraries (`python3 -c "import qiskit"`)
- [ ] Build tools installed (`gcc --version`)
- [ ] GPU drivers installed (if using GPU)
- [ ] Source code cloned (`ls quantum-geth/tools/solver/`)

### Build Process
- [ ] CGO_ENABLED=1 set for Python integration
- [ ] Build completes without errors
- [ ] Binary generated (`ls -la quantum-miner`)
- [ ] Binary shows help when executed

### Post-Build
- [ ] Miner shows correct acceleration mode (`./quantum-miner --help`)
- [ ] GPU detection working (if available)
- [ ] Benchmark shows reasonable performance
- [ ] Test vectors pass verification

**Successfully built quantum miner is ready for competitive Q Coin mining!** ⚡ 