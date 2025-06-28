# Linux Quantum Miner Build Troubleshooting

Comprehensive troubleshooting guide for Q Coin quantum-miner build issues on Linux systems.

## 🚨 Common Build Errors

### Go and CGO Issues

#### Error: `cgo: C compiler "gcc" not found`
**Problem:** C compiler required for Python integration but missing.

**Solutions:**
```bash
# Install build tools (Ubuntu/Debian)
sudo apt update
sudo apt install -y build-essential gcc g++ make pkg-config

# Install build tools (CentOS/RHEL/Fedora)
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc gcc-c++ make pkgconfig

# Verify installation
gcc --version
g++ --version
make --version
```

#### Error: `CGO_ENABLED=0 but CGO required for quantum miner`
**Problem:** Quantum miner requires CGO=1 for Python integration but CGO is disabled.

**Solutions:**
```bash
# Enable CGO for miner build
export CGO_ENABLED=1

# Verify setting
echo $CGO_ENABLED

# Build miner with CGO enabled
cd quantum-geth/tools/solver
CGO_ENABLED=1 go build -o ../../../quantum-miner .

# Note: This is different from geth which requires CGO_ENABLED=0
```

#### Error: `undefined reference to Python symbols`
**Problem:** Python development headers not installed.

**Solutions:**
```bash
# Install Python development headers (Ubuntu/Debian)
sudo apt install -y python3-dev python3-distutils

# Install Python development headers (CentOS/RHEL/Fedora)
sudo dnf install -y python3-devel

# Verify Python development files
python3-config --includes
python3-config --ldflags
```

### Python Integration Issues

#### Error: `ModuleNotFoundError: No module named 'qiskit'`
**Problem:** Required Python quantum libraries not installed.

**Solutions:**
```bash
# Install quantum dependencies
pip3 install qiskit qiskit-aer numpy scipy

# For GPU support
pip3 install qiskit-aer-gpu

# Verify installation
python3 -c "import qiskit; print('Qiskit OK')"
python3 -c "from qiskit_aer import AerSimulator; print('Aer OK')"

# Check installed packages
pip3 list | grep qiskit
```

#### Error: `ImportError: libpython3.x.so.1.0: cannot open shared object file`
**Problem:** Python shared libraries not found during runtime.

**Solutions:**
```bash
# Find Python library location
python3-config --ldflags
find /usr -name "libpython3*.so*" 2>/dev/null

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Or install python3-dev package
sudo apt install -y python3-dev  # Ubuntu/Debian
sudo dnf install -y python3-devel  # CentOS/RHEL/Fedora

# Rebuild with proper linking
cd quantum-geth/tools/solver
go build -o ../../../quantum-miner .
```

#### Error: `Python.h: No such file or directory`
**Problem:** Python headers missing for CGO compilation.

**Solutions:**
```bash
# Locate Python include directory
python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])"

# Install development headers
sudo apt install -y python3-dev  # Ubuntu/Debian
sudo dnf install -y python3-devel  # CentOS/RHEL/Fedora

# Set CGO flags explicitly
export CGO_CFLAGS="-I$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
export CGO_LDFLAGS="$(python3-config --ldflags)"

# Build with explicit flags
go build -o quantum-miner .
```

### GPU Acceleration Issues

#### Error: `CUDA not found` during build
**Problem:** CUDA toolkit not installed or not in PATH.

**Solutions:**
```bash
# Check if CUDA is installed
nvcc --version
nvidia-smi

# Install CUDA toolkit (Ubuntu)
sudo apt update
sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev

# Install CUDA toolkit (CentOS/RHEL/Fedora)
sudo dnf install -y cuda-toolkit

# Manual CUDA installation
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Error: `ImportError: No module named 'cupy'`
**Problem:** CuPy not installed for GPU acceleration.

**Solutions:**
```bash
# Install CuPy for your CUDA version
pip3 install cupy-cuda11x  # For CUDA 11.x
# pip3 install cupy-cuda12x  # For CUDA 12.x

# Check CUDA version first
nvcc --version
nvidia-smi

# Verify CuPy installation
python3 -c "import cupy; print('CuPy version:', cupy.__version__)"
python3 -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

#### Error: `qiskit_aer.AerError: GPU device not found`
**Problem:** Qiskit GPU backend not properly configured.

**Solutions:**
```bash
# Install GPU-enabled Qiskit Aer
pip3 install qiskit-aer-gpu

# Verify GPU backend
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"

# Check available backends
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"

# If GPU not available, build without GPU tags
cd quantum-geth/tools/solver
go build -tags cpu -o ../../../quantum-miner .
```

### Build Script Issues

#### Error: `./scripts/linux/build-linux.sh: miner: command not found`
**Problem:** Build script not recognizing miner argument.

**Solutions:**
```bash
# Check script exists and is executable
ls -la scripts/linux/build-linux.sh
chmod +x scripts/linux/build-linux.sh

# Use correct argument
./scripts/linux/build-linux.sh miner

# Manual build if script fails
cd quantum-geth/tools/solver
export CGO_ENABLED=1
go build -o ../../../quantum-miner .
```

#### Error: `quantum-geth/tools/solver directory not found`
**Problem:** Incorrect source code structure or location.

**Solutions:**
```bash
# Verify source structure
pwd
ls -la quantum-geth/tools/
ls -la quantum-geth/tools/solver/

# If missing, check repository integrity
git status
git pull origin main

# Verify solver source files
ls -la quantum-geth/tools/solver/*.go
```

### Memory and Performance Issues

#### Error: `runtime: out of memory` during build
**Problem:** Insufficient memory for compilation.

**Solutions:**
```bash
# Check available memory
free -h

# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Limit build parallelism
export GOMAXPROCS=2
go build -p 2 -o quantum-miner .

# Use build cache to reduce memory usage
export GOCACHE=$HOME/.cache/go-build
mkdir -p $GOCACHE
```

#### Error: Build succeeds but miner performance is poor
**Problem:** Built without optimal GPU support.

**Solutions:**
```bash
# Check what acceleration was built
./quantum-miner --version
./quantum-miner --help | grep -i gpu

# Verify GPU support
nvidia-smi
python3 -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'No GPU')"

# Rebuild with explicit GPU tags
cd quantum-geth/tools/solver
go build -tags "cupy_gpu,debug" -o ../../../quantum-miner .

# Test GPU performance
./quantum-miner --gpu --benchmark --duration 10s
```

## 🔧 Environment Issues

### Python Environment Problems

#### Error: Multiple Python versions causing conflicts
**Problem:** System has multiple Python installations.

**Solutions:**
```bash
# Check Python versions
which python3
python3 --version
ls -la /usr/bin/python*

# Use specific Python version
export PYTHON_BINARY=/usr/bin/python3.10
export PYTHONPATH=$($PYTHON_BINARY -c "import site; print(site.getsitepackages()[0])")

# Install packages for specific version
python3.10 -m pip install qiskit qiskit-aer numpy

# Build with specific Python
CGO_CFLAGS="-I$(python3.10 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
CGO_LDFLAGS="$(python3.10-config --ldflags)" \
go build -o quantum-miner .
```

#### Error: `pip: command not found`
**Problem:** pip not installed or not in PATH.

**Solutions:**
```bash
# Install pip (Ubuntu/Debian)
sudo apt install -y python3-pip

# Install pip (CentOS/RHEL/Fedora)
sudo dnf install -y python3-pip

# Manual pip installation
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user

# Add pip to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Library Dependency Issues

#### Error: `ImportError: libffi.so.6: cannot open shared object file`
**Problem:** Missing system libraries required by Python packages.

**Solutions:**
```bash
# Install missing libraries (Ubuntu/Debian)
sudo apt install -y libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev

# Install missing libraries (CentOS/RHEL/Fedora)
sudo dnf install -y libffi-devel openssl-devel bzip2-devel readline-devel sqlite-devel

# Update library cache
sudo ldconfig

# Reinstall Python packages
pip3 uninstall qiskit qiskit-aer
pip3 install qiskit qiskit-aer
```

## 🐛 Advanced Debugging

### Build Debugging

```bash
# Enable verbose Go build output
go build -v -x -o quantum-miner .

# Debug CGO compilation
CGO_ENABLED=1 go build -work -x -o quantum-miner .

# Check CGO flags
go env CGO_CFLAGS
go env CGO_LDFLAGS

# Manual CGO test
cat > test_cgo.go << 'EOF'
package main

/*
#include <Python.h>
*/
import "C"

func main() {
    C.Py_Initialize()
    C.Py_Finalize()
}
EOF

go run test_cgo.go  # Should compile without errors
rm test_cgo.go
```

### Runtime Debugging

```bash
# Debug miner startup
./quantum-miner --debug --verbose --dry-run

# Check Python integration
./quantum-miner --test-python

# Debug GPU detection
./quantum-miner --debug-gpu

# Trace system calls
strace -e trace=file ./quantum-miner --help

# Check library dependencies
ldd quantum-miner
```

### Performance Analysis

```bash
# Profile miner build
go build -cpuprofile=miner.prof -o quantum-miner .
go tool pprof miner.prof

# Memory profiling
go build -memprofile=mem.prof -o quantum-miner .
go tool pprof mem.prof

# Benchmark different builds
time go build -tags cpu -o quantum-miner-cpu .
time go build -tags cupy_gpu -o quantum-miner-gpu .

# Compare performance
./quantum-miner-cpu --benchmark --duration 30s
./quantum-miner-gpu --benchmark --duration 30s
```

## 🚀 Performance Troubleshooting

### GPU Performance Issues

```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor GPU during mining
watch -n 1 nvidia-smi

# Check thermal throttling
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,clocks.max.graphics --format=csv -l 1

# Optimize GPU settings
sudo nvidia-smi -pm 1  # Persistence mode
sudo nvidia-smi -pl 300  # Power limit (adjust for your card)

# Check GPU memory usage
./quantum-miner --gpu --verbose | grep -i memory
```

### CPU Performance Issues

```bash
# Check CPU usage
htop
top -H

# Monitor CPU frequency
watch -n 1 'cat /proc/cpuinfo | grep "cpu MHz"'

# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check thermal throttling
sensors
dmesg | grep -i thermal
```

### Memory Issues

```bash
# Monitor memory usage during mining
watch -n 1 'free -h && echo "=== Miner ===" && ps aux | grep quantum-miner'

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./quantum-miner --test

# Optimize for low memory
export GOMEMLIMIT=2GiB
./quantum-miner --cpu --threads 2 --batch-size 128
```

## ✅ Build Health Check

### Pre-Build Health Check

```bash
#!/bin/bash
echo "=== Quantum Miner Linux Build Health Check ==="

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

# Check CGO support
if [[ "$CGO_ENABLED" == "1" ]]; then
    echo "✅ CGO enabled for Python integration"
else
    echo "⚠️  CGO not enabled, setting now"
    export CGO_ENABLED=1
fi

# Check Python and dependencies
if command -v python3 &> /dev/null; then
    echo "✅ Python3 available"
    if python3 -c "import qiskit" &> /dev/null; then
        echo "✅ Qiskit installed"
    else
        echo "❌ Qiskit not installed"
        exit 1
    fi
else
    echo "❌ Python3 not found"
    exit 1
fi

# Check build tools
for tool in gcc g++ make; do
    if command -v $tool &> /dev/null; then
        echo "✅ $tool available"
    else
        echo "❌ $tool not found"
        exit 1
    fi
done

# Check GPU support (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    if python3 -c "import cupy; cupy.cuda.is_available()" &> /dev/null; then
        echo "✅ CuPy GPU support available"
    else
        echo "⚠️  CuPy GPU support not available"
    fi
else
    echo "⚠️  No NVIDIA GPU detected, CPU-only mode"
fi

echo "✅ Quantum miner build environment ready!"
```

### Post-Build Verification

```bash
#!/bin/bash
echo "=== Quantum Miner Build Verification ==="

if [[ -f "quantum-miner" ]]; then
    echo "✅ Binary exists"
    
    # Check if executable
    if [[ -x "quantum-miner" ]]; then
        echo "✅ Binary is executable"
    else
        echo "❌ Binary not executable"
        chmod +x quantum-miner
    fi
    
    # Test basic functionality
    if ./quantum-miner --help &> /dev/null; then
        echo "✅ Binary executes successfully"
    else
        echo "❌ Binary execution failed"
        exit 1
    fi
    
    # Check quantum features
    if ./quantum-miner --help 2>&1 | grep -q "gpu\|quantum"; then
        echo "✅ Quantum features detected"
    else
        echo "⚠️  Quantum features not found"
    fi
    
    # Test GPU support if available
    if ./quantum-miner --gpu --test &> /dev/null; then
        echo "✅ GPU acceleration working"
    else
        echo "⚠️  GPU acceleration not available"
    fi
    
else
    echo "❌ Binary not found"
    exit 1
fi

echo "✅ Quantum miner build verification complete!"
```

## 📞 Getting Help

### Community Support
- **GitHub Issues:** [Report build issues](https://github.com/fourtytwo42/Qgeth3/issues)
- **Documentation:** Check [build guide](linux-build-quantum-miner.md)

### Self-Help Resources
- Run the health check scripts above
- Review [mining troubleshooting](../mining/troubleshooting-linux-mining.md)
- Check [general Linux troubleshooting](../node-operation/troubleshooting-linux-geth.md)

### Debug Information Collection

```bash
# Create comprehensive debug log
{
    echo "=== Debug Information ==="
    date
    uname -a
    echo "Go: $(go version)"
    echo "Python: $(python3 --version)"
    echo "CGO_ENABLED: $CGO_ENABLED"
    echo "=== GPU Information ==="
    nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"
    echo "=== Python Packages ==="
    pip3 list | grep -E "(qiskit|cupy|numpy)"
    echo "=== Build Attempt ==="
    cd quantum-geth/tools/solver
    CGO_ENABLED=1 go build -v -x -o ../../../quantum-miner . 2>&1
} | tee quantum-miner-debug.log

# Share quantum-miner-debug.log when seeking help
```

**Most quantum miner build issues are resolved by ensuring proper Python dependencies and CGO configuration!** ⚡ 