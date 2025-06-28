# Linux Q Geth Build Guide

Complete guide for building Q Coin quantum-geth from source on Linux systems.

## 📋 Prerequisites

### System Requirements
- **OS:** Ubuntu 20.04+, Debian 11+, CentOS 8+, or compatible Linux distribution
- **CPU:** 2+ cores (4+ recommended for faster builds)
- **RAM:** 2GB minimum (4GB+ recommended for builds)
- **Storage:** 5GB free space for source code and build artifacts
- **Network:** Internet connection for downloading dependencies

### Required Software
- **Go 1.21+** (mandatory for building)
- **Git** (for source code management)
- **gcc/make** (C compiler and build tools)
- **pkg-config** (for library dependency management)

## 🛠️ Installing Dependencies

### Ubuntu/Debian
```bash
# Update package lists
sudo apt update

# Install Go compiler
sudo apt install -y golang-go

# Install build tools
sudo apt install -y git build-essential pkg-config curl wget

# Verify Go installation
go version  # Should show 1.21 or later
```

### CentOS/RHEL/Fedora
```bash
# Update system
sudo dnf update -y

# Install Go compiler
sudo dnf install -y golang

# Install build tools
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git pkg-config curl wget

# Verify installation
go version
```

### Manual Go Installation (Latest Version)
If your distribution has an older Go version, install manually:

```bash
# Remove old Go installation
sudo rm -rf /usr/local/go

# Download latest Go (check https://golang.org/dl/ for current version)
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz

# Extract and install
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
rm go1.21.5.linux-amd64.tar.gz

# Add to PATH permanently
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version
```

## 📥 Getting the Source Code

### Clone Repository
```bash
# Clone the Q Coin repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Verify source structure
ls -la
# Should see: quantum-geth/, scripts/, configs/, etc.

# Check quantum-geth source
ls -la quantum-geth/
# Should see: cmd/, core/, consensus/, etc.
```

### Source Code Structure
```
Qgeth3/
├── quantum-geth/           # Main geth source code
│   ├── cmd/geth/          # Geth main executable
│   ├── core/              # Blockchain core logic
│   ├── consensus/qmpow/   # Quantum consensus implementation
│   ├── eth/               # Ethereum protocol implementation
│   └── ...
├── scripts/linux/         # Linux build scripts
├── configs/               # Network configurations
└── ...
```

## 🔨 Building Q Geth

### Using Build Script (Recommended)
```bash
# Make scripts executable
chmod +x scripts/linux/*.sh

# Build geth with automatic configuration
./scripts/linux/build-linux.sh geth

# Expected output:
# Building quantum-geth...
# ENFORCING: CGO_ENABLED=0 for geth build (quantum field compatibility)
# quantum-geth built successfully (CGO_ENABLED=0)
```

### Manual Build Process
```bash
# Navigate to quantum-geth source
cd quantum-geth

# Set build environment for quantum compatibility
export CGO_ENABLED=0
export GOOS=linux
export GOARCH=amd64

# Set build metadata
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT=$(git rev-parse HEAD)

# Build geth with quantum consensus
go build \
  -ldflags="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME" \
  -o ../geth.bin \
  ./cmd/geth

# Return to root directory
cd ..

# Verify build
ls -la geth.bin
file geth.bin  # Should show: ELF 64-bit LSB executable
```

### Build with Clean Environment
```bash
# Clean previous builds
./scripts/linux/build-linux.sh geth --clean

# This will:
# 1. Remove old binaries
# 2. Clean Go module cache
# 3. Rebuild from scratch
# 4. Ensure fresh dependencies
```

## ⚙️ Build Configuration

### Quantum Field Compatibility
Q Geth requires `CGO_ENABLED=0` for quantum field serialization:

```bash
# Why CGO_ENABLED=0 is required:
# - Ensures consistent quantum field marshaling
# - Prevents C library dependencies
# - Enables static binary compilation
# - Guarantees cross-platform compatibility

export CGO_ENABLED=0  # Mandatory for Q Geth builds
```

### Build Optimization
```bash
# Debug build (larger, with symbols)
go build -gcflags="-N -l" -o geth.bin ./cmd/geth

# Release build (optimized, smaller)
go build -ldflags="-s -w" -o geth.bin ./cmd/geth

# Static build (no dependencies)
CGO_ENABLED=0 go build -a -ldflags="-s -w" -o geth.bin ./cmd/geth
```

### Cross-Compilation
```bash
# Build for different architectures
export GOOS=linux
export GOARCH=amd64    # x86_64
# export GOARCH=arm64  # ARM64
# export GOARCH=386    # x86_32

go build -o geth.bin ./cmd/geth
```

## ✅ Verification

### Test Build
```bash
# Check binary exists and is executable
ls -la geth.bin
chmod +x geth.bin

# Test basic functionality
./geth.bin version

# Expected output similar to:
# Geth
# Version: 1.13.5-stable
# Git Commit: 916d6a44c9b9b89efdc31b62a78d26a6b84bb9c1
# Git Commit Date: 20231128
# Architecture: amd64
# Go Version: go1.21.5
# Operating System: linux
```

### Quantum Consensus Verification
```bash
# Check if quantum consensus is available
./geth.bin help | grep -i quantum

# Check quantum-specific commands
./geth.bin console --help | grep -E "(qmpow|quantum)"

# Test initialization with quantum genesis
./geth.bin --datadir /tmp/test-qgeth init configs/genesis_quantum_testnet.json
ls -la /tmp/test-qgeth/geth/
rm -rf /tmp/test-qgeth  # Cleanup test
```

### Performance Test
```bash
# Quick startup test
timeout 10s ./geth.bin --datadir /tmp/test --dev console || true

# Memory usage check
./geth.bin --help > /dev/null
echo "Binary size: $(du -h geth.bin | cut -f1)"
echo "Dependencies: $(ldd geth.bin 2>/dev/null | wc -l || echo 'Static binary')"
```

## 📂 Build Artifacts

### Generated Files
After successful build, you'll have:

```bash
# Main executable
geth.bin                    # Q Geth binary (typically 15-25MB)

# Temporary build files (cleaned automatically)
build-temp-*/              # Temporary build directories
go.mod.bak                 # Go module backup
go.sum.bak                 # Go sum backup
```

### Installation
```bash
# Install to system PATH (optional)
sudo cp geth.bin /usr/local/bin/qgeth
sudo chmod +x /usr/local/bin/qgeth

# Or create symlink
sudo ln -sf $(pwd)/geth.bin /usr/local/bin/qgeth

# Test system installation
qgeth version
```

## 🔧 Build Customization

### Custom Build Tags
```bash
# Build with specific features
go build -tags "netgo,osusergo" -o geth.bin ./cmd/geth

# Available tags:
# - netgo: Pure Go networking
# - osusergo: Pure Go user/group lookups
# - noemulator: Disable emulator mode
```

### Debug Builds
```bash
# Build with debug symbols
go build -gcflags="-N -l" -o geth.debug ./cmd/geth

# Build with race detection
go build -race -o geth.race ./cmd/geth

# Build with verbose output
go build -v -o geth.bin ./cmd/geth
```

### Memory-Optimized Builds
```bash
# For low-memory systems
export GOMEMLIMIT=1GiB
go build -o geth.bin ./cmd/geth

# Use build cache
export GOCACHE=$HOME/.cache/go-build
go build -o geth.bin ./cmd/geth
```

## 🚀 Advanced Build Options

### Using Build Temp Directory
```bash
# Set custom temp directory for builds
export QGETH_BUILD_TEMP=/tmp/qgeth-build
mkdir -p $QGETH_BUILD_TEMP

./scripts/linux/build-linux.sh geth

# This uses the custom temp directory for:
# - Go cache
# - Go temp files
# - Build artifacts
```

### Parallel Builds
```bash
# Use multiple CPU cores for faster builds
export GOMAXPROCS=$(nproc)
go build -p $(nproc) -o geth.bin ./cmd/geth

# Build with make-style parallelism
make -j$(nproc) -C quantum-geth geth
```

### Reproducible Builds
```bash
# Ensure reproducible builds
export CGO_ENABLED=0
export GOOS=linux
export GOARCH=amd64
export SOURCE_DATE_EPOCH=1234567890

go build -trimpath -ldflags="-buildid=" -o geth.bin ./cmd/geth
```

## 📝 Build Environment

### Required Environment Variables
```bash
# Mandatory for Q Geth
export CGO_ENABLED=0

# Recommended
export GOOS=linux
export GOARCH=amd64
export GO111MODULE=on

# Optional optimizations
export GOMAXPROCS=$(nproc)
export GOCACHE=$HOME/.cache/go-build
export GOPATH=$HOME/go
```

### Build Script Environment
The `build-linux.sh` script automatically sets:

```bash
# Quantum compatibility
CGO_ENABLED=0

# Build metadata
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT=$(git rev-parse HEAD)

# Linker flags
LDFLAGS="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME"

# Temp directory management
BUILD_TEMP_DIR="./build-temp-$(date +%s)"
```

## 🔍 Troubleshooting

For build issues, see the [Linux Geth Build Troubleshooting Guide](troubleshooting-linux-build-geth.md).

Common quick fixes:
```bash
# Update Go
sudo rm -rf /usr/local/go && wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz

# Clear caches
go clean -cache && go clean -modcache

# Rebuild clean
./scripts/linux/build-linux.sh geth --clean
```

## ✅ Build Checklist

### Pre-Build
- [ ] Go 1.21+ installed and working (`go version`)
- [ ] Build tools installed (`gcc --version`, `make --version`)
- [ ] Source code cloned (`git status`)
- [ ] Scripts executable (`chmod +x scripts/linux/*.sh`)

### Build Process
- [ ] CGO_ENABLED=0 set for quantum compatibility
- [ ] Build completes without errors
- [ ] Binary generated (`ls -la geth.bin`)
- [ ] Binary is executable and shows version

### Post-Build
- [ ] Quantum consensus available (`./geth.bin help | grep quantum`)
- [ ] Can initialize with quantum genesis
- [ ] Binary size reasonable (15-25MB)
- [ ] No unexpected dependencies (`ldd geth.bin`)

**Successfully built Q Geth binary is ready for quantum blockchain operations!** 🎉 