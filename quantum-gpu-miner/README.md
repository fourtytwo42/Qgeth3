# Quantum-GPU-Miner

A high-performance quantum mining application for Quantum-Geth blockchain with GPU acceleration support via CUDA and Qiskit.

## 🚀 Features

- **✅ Quantum Circuit Mining**: Real 16-qubit quantum circuits with 8192 T-gates per puzzle
- **✅ GPU Acceleration**: Qiskit-based quantum simulation with GPU support
- **✅ CUDA Support**: Native CUDA kernels for quantum state vector simulation (90% complete)
- **✅ Hybrid Backend**: Automatic fallback between CUDA, Qiskit, and CPU simulation
- **✅ Batch Processing**: Efficient batch simulation of 48 puzzles per block
- **✅ Blockchain Integration**: Full integration with Quantum-Geth network
- **✅ Cross-Platform**: Windows and Linux support

## 📊 Performance Results

| Mode | Puzzles/sec | Block Success Rate | Status |
|------|-------------|-------------------|---------|
| **GPU (Qiskit)** | **0.45** | **35.5%** | ✅ **Working** |
| CPU-Only | 0.36 | 37.5% | ✅ Working |
| CUDA (Native) | TBD | TBD | 🔧 90% Complete |

## 🛠️ System Requirements

### CPU-Only Mode
- Go 1.19 or later
- Windows 10/11 or Linux
- 4GB+ RAM
- Active Quantum-Geth node

### GPU-Accelerated Mode
- All CPU requirements plus:
- **Python 3.8+** with pip
- **Qiskit** quantum computing framework
- **Optional**: NVIDIA GPU with CUDA Toolkit 12.x for native CUDA support

## 📦 Installation

### 1. Install Python Dependencies (Required for GPU mode)

```bash
# Install Python dependencies
pip install qiskit qiskit-aer numpy

# For GPU acceleration (optional)
pip install qiskit-aer-gpu
```

### 2. Install CUDA Toolkit (Optional - for native CUDA support)

Download and install [CUDA Toolkit 12.9](https://developer.nvidia.com/cuda-downloads) from NVIDIA.

### 3. Install Visual Studio Build Tools (Windows only - for CUDA compilation)

Download and install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with C++ support.

## 🔨 Building

### Quick Build (CPU + Qiskit GPU)

**Windows:**
```powershell
# CPU-only build
go build -o quantum-gpu-miner-cpu.exe .

# GPU-accelerated build (Qiskit)
go build -o quantum-gpu-miner.exe .
```

**Linux:**
```bash
# CPU-only build  
go build -o quantum-gpu-miner-cpu .

# GPU-accelerated build (Qiskit)
go build -o quantum-gpu-miner .
```

### Advanced Build with CUDA (Experimental)

**Windows:**
```powershell
# Use the build script for CUDA support
.\build-simple.ps1 gpu
```

**Linux:**
```bash
# Use the Makefile for CUDA support
make gpu
```

## 🎮 Usage

### Quick Start with PowerShell Scripts (Recommended)

**GPU-Accelerated Mining (Best Performance):**
```powershell
# Windows - GPU mining with Qiskit acceleration
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress

# With custom options
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -Threads 2 -NodeURL http://localhost:8545 -GpuId 0

# Show help
.\run-gpu-miner.ps1 -Help
```

**CPU-Only Mining:**
```powershell
# Windows - CPU mining
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress

# With custom options  
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress -Threads 4 -NodeURL http://localhost:8545

# Show help
.\run-cpu-miner.ps1 -Help
```

### Manual Command Line Usage

**GPU-Accelerated Mining:**
```bash
# Windows
.\quantum-gpu-miner.exe -gpu -threads 1 -coinbase 0xYourAddress -node http://localhost:8545

# Linux  
./quantum-gpu-miner -gpu -threads 1 -coinbase 0xYourAddress -node http://localhost:8545
```

**CPU-Only Mining:**
```bash
# Windows
.\quantum-gpu-miner-cpu.exe -threads 2 -coinbase 0xYourAddress -node http://localhost:8545

# Linux
./quantum-gpu-miner-cpu -threads 2 -coinbase 0xYourAddress -node http://localhost:8545
```

### Command Line Options

```
Usage: quantum-gpu-miner [options]

Options:
  -coinbase string    Coinbase address for mining rewards (required)
  -node string        Quantum-Geth node URL (default: http://localhost:8545)
  -threads int        Number of mining threads (default: 1)
  -gpu               Enable GPU acceleration (uses device 0)
  -gpu-id int        GPU device ID (default: 0)
  -help              Show this help message
```

### PowerShell Script Features

**✅ Enhanced User Experience:**
- Comprehensive help with `-Help` flag
- Input validation for coinbase addresses
- Detailed configuration display
- Error handling with troubleshooting tips
- Performance comparisons and recommendations

**✅ Smart Defaults:**
- Automatic executable detection
- Sensible default parameters
- Clear error messages for missing dependencies

**✅ Easy Setup:**
- No need to remember complex command line arguments
- Built-in examples and usage instructions
- Automatic build guidance if executables are missing

## 🔬 Architecture

### Quantum Circuit Specifications
- **Qubits**: 16 per puzzle
- **T-Gates**: 8192 per puzzle  
- **Puzzles**: 48 per block
- **Gate Types**: Hadamard (H), T-gate, CNOT

### Backend Selection
1. **CUDA**: Native GPU kernels (when compiled with CUDA support)
2. **Qiskit**: Python-based quantum simulation with GPU acceleration
3. **CPU**: Classical simulation fallback

### Batch Processing
- Processes all 48 puzzles in a single Python call
- Reduces overhead from 48 process spawns to 1
- Achieves ~30x performance improvement over sequential processing

## 🧪 Testing

### Test Qiskit Backend
```bash
python pkg/quantum/qiskit_gpu.py test
```

### Test Batch Simulation
```bash
python pkg/quantum/qiskit_gpu.py batch_simulate test_hash 12345 16 8192 48
```

### Benchmark Performance
```bash
python pkg/quantum/qiskit_gpu.py benchmark 0 10
```

## 📈 Mining Statistics

The miner provides real-time statistics:
- **Puzzle Rate**: Quantum puzzles solved per second
- **Block Success Rate**: Percentage of blocks that meet difficulty target
- **Accepted/Rejected**: Network submission results
- **Stale/Duplicate**: Mining efficiency metrics

## 🔧 Troubleshooting

### Common Issues

**"Python was not found" Error:**
- Install Python 3.8+ from python.org (not Microsoft Store)
- Ensure Python is in your system PATH

**"Qiskit backend not available" Error:**
```bash
pip install qiskit qiskit-aer numpy
```

**CUDA Compilation Errors:**
- Install CUDA Toolkit 12.9
- Install Visual Studio Build Tools 2022 (Windows)
- Use `-tags cuda` flag for Go build

**Low Mining Performance:**
- Ensure quantum-geth node is running locally
- Check network connectivity
- Verify coinbase address format

### Performance Optimization

**For GPU Mining:**
- Use `-gpu` flag for Qiskit acceleration
- Ensure adequate GPU memory (4GB+ recommended)
- Close other GPU-intensive applications

**For CPU Mining:**
- Adjust `-threads` based on CPU cores
- Monitor CPU temperature and throttling
- Consider power management settings

## 🏗️ Development Status

### ✅ Completed Features
- [x] Quantum circuit simulation (CPU)
- [x] Qiskit GPU backend integration
- [x] Batch processing optimization
- [x] Blockchain network integration
- [x] Cross-platform building
- [x] Performance monitoring
- [x] Error handling and fallbacks

### 🔧 In Progress
- [ ] Native CUDA backend completion (90% done)
- [ ] Windows MSVC/CGO compatibility
- [ ] Multi-GPU support
- [ ] Advanced quantum optimization

### 🚀 Future Enhancements
- [ ] Distributed mining pools
- [ ] Quantum algorithm optimization
- [ ] Real quantum hardware integration
- [ ] Advanced mining strategies

## 📄 License

This project is part of the Quantum-Geth ecosystem. See the main repository for license information.

## 🤝 Contributing

Contributions are welcome! Please see the main Quantum-Geth repository for contribution guidelines.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review quantum-geth node logs
3. Verify system requirements
4. Submit issues to the main repository

---

**Happy Quantum Mining!** ⚛️🚀
