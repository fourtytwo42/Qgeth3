# Linux Q Geth Build Troubleshooting

Comprehensive troubleshooting guide for Q Coin quantum-geth build issues on Linux systems.

## 🚨 Common Build Errors

### Go Compiler Issues

#### Error: `bash: go: command not found`
**Problem:** Go compiler not installed or not in PATH.

**Solutions:**
```bash
# Check if Go is installed
which go
echo $PATH

# Install Go (Ubuntu/Debian)
sudo apt update
sudo apt install -y golang-go

# Install Go (CentOS/RHEL/Fedora)
sudo dnf install -y golang

# Manual Go installation (latest version)
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

#### Error: `go: go.mod file not found in current directory or any parent directory`
**Problem:** Running Go commands outside the quantum-geth directory.

**Solutions:**
```bash
# Navigate to the correct directory
cd quantum-geth
pwd  # Should show: /path/to/Qgeth3/quantum-geth

# Verify go.mod exists
ls -la go.mod

# If go.mod is missing, initialize (rare case)
go mod init quantum-geth
go mod tidy
```

#### Error: `go version go1.x.x: directive requires go 1.21 or later`
**Problem:** Go version too old for quantum-geth requirements.

**Solutions:**
```bash
# Check current Go version
go version

# Remove old Go installation
sudo rm -rf /usr/local/go

# Install latest Go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
rm go1.21.5.linux-amd64.tar.gz

# Update PATH
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify new version
go version
```

### CGO Compatibility Issues

#### Error: Build fails with CGO errors despite CGO_ENABLED=0
**Problem:** CGO configuration not properly set.

**Solutions:**
```bash
# Explicitly disable CGO
export CGO_ENABLED=0

# Verify setting
echo $CGO_ENABLED

# Force static build
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o geth.bin ./cmd/geth

# Check for C dependencies (should be none)
ldd geth.bin  # Should show "not a dynamic executable"
```

#### Error: `cgo: C compiler "gcc" not found`
**Problem:** C compiler missing (shouldn't occur with CGO_ENABLED=0).

**Solutions:**
```bash
# Install build tools (even though not needed for static builds)
# Ubuntu/Debian
sudo apt install -y build-essential

# CentOS/RHEL/Fedora
sudo dnf groupinstall -y "Development Tools"

# Force CGO disabled build
export CGO_ENABLED=0
./scripts/linux/build-linux.sh geth
```

### Memory and Space Issues

#### Error: `virtual memory exhausted: Cannot allocate memory` or `killed`
**Problem:** Insufficient RAM during compilation.

**Solutions:**
```bash
# Check available memory
free -h

# Create/increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Limit Go's memory usage
export GOMEMLIMIT=1GiB
go build -o geth.bin ./cmd/geth

# Build with reduced parallelism
go build -p 1 -o geth.bin ./cmd/geth
```

#### Error: `No space left on device`
**Problem:** Insufficient disk space in build directory.

**Solutions:**
```bash
# Check disk space
df -h .
df -h /tmp

# Clean Go caches
go clean -cache
go clean -modcache

# Use custom temp directory
export QGETH_BUILD_TEMP=/home/$USER/qgeth-build
mkdir -p $QGETH_BUILD_TEMP
./scripts/linux/build-linux.sh geth

# Clean up old build artifacts
rm -rf build-temp-*
rm -f *.bin *.exe quantum-miner
```

### Network and Dependency Issues

#### Error: `fatal: could not read Username/Password for 'https://github.com'`
**Problem:** Git authentication issues when fetching dependencies.

**Solutions:**
```bash
# Check Git configuration
git config --global user.name
git config --global user.email

# Set Git credentials (if needed)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use HTTPS instead of SSH (if SSH issues)
git config --global url."https://github.com/".insteadOf git@github.com:

# Clear Go module cache and retry
go clean -modcache
go mod download
```

#### Error: `Get "https://proxy.golang.org/...": context deadline exceeded`
**Problem:** Network timeout downloading Go modules.

**Solutions:**
```bash
# Increase timeout
export GOPROXY=direct
export GOSUMDB=off

# Use different proxy
export GOPROXY=https://goproxy.cn,direct

# Download dependencies separately
go mod download -x

# Build offline after downloading
go build -mod=readonly -o geth.bin ./cmd/geth
```

### Build Script Issues

#### Error: `./scripts/linux/build-linux.sh: Permission denied`
**Problem:** Script not executable.

**Solutions:**
```bash
# Make scripts executable
chmod +x scripts/linux/*.sh

# Verify permissions
ls -la scripts/linux/build-linux.sh

# Run directly if needed
bash scripts/linux/build-linux.sh geth
```

#### Error: `quantum-geth directory not found!`
**Problem:** Running script from wrong directory.

**Solutions:**
```bash
# Check current directory
pwd
ls -la

# Should be in Qgeth3 root directory
cd /path/to/Qgeth3

# Verify quantum-geth exists
ls -la quantum-geth/

# Run from correct location
./scripts/linux/build-linux.sh geth
```

### Linker and Binary Issues

#### Error: `undefined: main.gitCommit` or similar linker errors
**Problem:** Linker flags not properly formatted.

**Solutions:**
```bash
# Check Git repository
git status
git log --oneline -n 1

# Manual build with explicit flags
cd quantum-geth
export CGO_ENABLED=0
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT=$(git rev-parse HEAD)

go build \
  -ldflags="-s -w -X main.gitCommit=$GIT_COMMIT -X main.buildTime=$BUILD_TIME" \
  -o ../geth.bin \
  ./cmd/geth
```

#### Error: Binary built but crashes immediately
**Problem:** Incompatible build environment or corruption.

**Solutions:**
```bash
# Check binary
file geth.bin
ldd geth.bin

# Test basic functionality
./geth.bin version

# Rebuild clean
go clean -cache
./scripts/linux/build-linux.sh geth --clean

# Build with debug symbols
go build -gcflags="-N -l" -o geth.debug ./cmd/geth
gdb ./geth.debug
```

## 🔧 Environment Issues

### PATH and Shell Configuration

#### Error: Environment variables not persistent
**Problem:** Variables not saved to shell profile.

**Solutions:**
```bash
# Add to shell profile permanently
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
echo 'export CGO_ENABLED=0' >> ~/.bashrc

# For zsh users
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.zshrc

# Source the profile
source ~/.bashrc  # or ~/.zshrc

# Verify settings
echo $PATH
echo $GOPATH
echo $CGO_ENABLED
```

### File System and Permissions

#### Error: `Permission denied` writing to build directory
**Problem:** Insufficient permissions in build location.

**Solutions:**
```bash
# Check ownership
ls -la .
ls -la geth.bin

# Fix ownership
sudo chown -R $USER:$USER .

# Use user-writable build location
export QGETH_BUILD_TEMP=$HOME/qgeth-build
mkdir -p $QGETH_BUILD_TEMP
./scripts/linux/build-linux.sh geth

# Build in home directory
cd ~
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
./scripts/linux/build-linux.sh geth
```

### System Dependencies

#### Error: `pkg-config: command not found`
**Problem:** Missing development tools.

**Solutions:**
```bash
# Install pkg-config (Ubuntu/Debian)
sudo apt install -y pkg-config build-essential

# Install pkg-config (CentOS/RHEL/Fedora)
sudo dnf install -y pkgconfig gcc make

# Verify installation
pkg-config --version
gcc --version
make --version
```

## 🐛 Advanced Debugging

### Verbose Build Information

```bash
# Enable verbose Go output
go build -v -x -o geth.bin ./cmd/geth

# Show all compiler commands
go build -work -o geth.bin ./cmd/geth

# Keep temporary files for inspection
go build -work -o geth.bin ./cmd/geth
# Check the printed work directory
```

### Build Environment Diagnosis

```bash
# Complete environment check
echo "=== Go Environment ==="
go version
go env

echo "=== Build Environment ==="
echo "CGO_ENABLED: $CGO_ENABLED"
echo "GOOS: $GOOS"
echo "GOARCH: $GOARCH"
echo "GOPATH: $GOPATH"
echo "GOCACHE: $GOCACHE"

echo "=== System Information ==="
uname -a
cat /etc/os-release
free -h
df -h .

echo "=== Dependencies ==="
which go gcc make git
gcc --version
make --version
git --version
```

### Dependency Analysis

```bash
# Check Go module dependencies
go list -m all
go mod why -m golang.org/x/crypto

# Download all dependencies
go mod download -x

# Verify checksums
go mod verify

# Clean and re-download
go clean -modcache
go mod download
```

### Binary Analysis

```bash
# Analyze built binary
file geth.bin
size geth.bin
objdump -x geth.bin | head -20

# Check for dynamic dependencies (should be none)
ldd geth.bin

# Symbol table analysis
nm geth.bin | grep main

# Strings analysis
strings geth.bin | grep -E "(version|commit|quantum)"
```

## 🚀 Performance Optimization

### Build Speed Improvements

```bash
# Use build cache
export GOCACHE=$HOME/.cache/go-build
mkdir -p $GOCACHE

# Parallel builds
export GOMAXPROCS=$(nproc)
go build -p $(nproc) -o geth.bin ./cmd/geth

# Pre-download dependencies
go mod download

# Use local proxy
export GOPROXY=file:///$HOME/goproxy,https://proxy.golang.org,direct
```

### Memory Optimization

```bash
# Limit memory usage
export GOMEMLIMIT=2GiB

# Reduce garbage collection pressure
export GOGC=100

# Use swap if needed
sudo swapon --show
```

## ✅ Build Health Check

### Pre-Build Verification

```bash
#!/bin/bash
echo "=== Q Geth Linux Build Health Check ==="

# Check Go installation
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | cut -d' ' -f3)
    echo "✅ Go installed: $GO_VERSION"
    if [[ "$GO_VERSION" < "go1.21" ]]; then
        echo "❌ Go version too old, need 1.21+"
        exit 1
    fi
else
    echo "❌ Go not found"
    exit 1
fi

# Check build tools
for tool in gcc make git; do
    if command -v $tool &> /dev/null; then
        echo "✅ $tool available"
    else
        echo "❌ $tool not found"
        exit 1
    fi
done

# Check source code
if [[ -d "quantum-geth" && -f "quantum-geth/go.mod" ]]; then
    echo "✅ Source code present"
else
    echo "❌ Source code missing"
    exit 1
fi

# Check environment
if [[ "$CGO_ENABLED" == "0" ]]; then
    echo "✅ CGO properly disabled"
else
    echo "⚠️  CGO not disabled, setting now"
    export CGO_ENABLED=0
fi

echo "✅ Build environment ready!"
```

### Post-Build Verification

```bash
#!/bin/bash
echo "=== Q Geth Build Verification ==="

if [[ -f "geth.bin" ]]; then
    echo "✅ Binary exists"
    
    # Check size (should be reasonable)
    SIZE=$(stat -c%s geth.bin)
    SIZE_MB=$((SIZE / 1024 / 1024))
    echo "📦 Binary size: ${SIZE_MB}MB"
    
    if [[ $SIZE_MB -lt 10 || $SIZE_MB -gt 50 ]]; then
        echo "⚠️  Unusual binary size"
    fi
    
    # Check if executable
    if [[ -x "geth.bin" ]]; then
        echo "✅ Binary is executable"
    else
        echo "❌ Binary not executable"
        chmod +x geth.bin
    fi
    
    # Test basic functionality
    if ./geth.bin version &> /dev/null; then
        echo "✅ Binary executes successfully"
        ./geth.bin version | head -5
    else
        echo "❌ Binary execution failed"
        exit 1
    fi
    
    # Check for quantum features
    if ./geth.bin help 2>&1 | grep -q quantum; then
        echo "✅ Quantum features detected"
    else
        echo "⚠️  Quantum features not found in help"
    fi
    
else
    echo "❌ Binary not found"
    exit 1
fi

echo "✅ Build verification complete!"
```

## 📞 Getting Help

### Community Support
- **GitHub Issues:** [Report build issues](https://github.com/fourtytwo42/Qgeth3/issues)
- **Documentation:** Check [main installation guide](../getting-started/quick-start.md)

### Self-Help Resources
- Run the health check scripts above
- Check [build guide](linux-build-geth.md) for correct process
- Review [node operation troubleshooting](../node-operation/troubleshooting-linux-geth.md)

### Last Resort Debug
```bash
# Create comprehensive debug log
{
    echo "=== Debug Information ==="
    date
    uname -a
    go version
    go env
    echo "PWD: $(pwd)"
    echo "USER: $USER"
    ls -la
    echo "=== Build Attempt ==="
    ./scripts/linux/build-linux.sh geth 2>&1
} | tee debug-build.log

# Share debug-build.log when seeking help
```

**Most Linux build issues are resolved by ensuring proper Go installation and CGO_ENABLED=0 setting!** 🛠️ 