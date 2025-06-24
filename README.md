# Qgeth3 - Quantum Blockchain Mining Platform

A complete quantum blockchain platform featuring **Quantum-Geth** (quantum-enhanced Ethereum client) and **high-performance quantum miners** with both CPU and GPU acceleration support.

## 🚀 Quick Start

### 1. Start Quantum-Geth Node
```powershell
# Start the quantum blockchain node
.\start-geth.ps1
```

### 2. Choose Your Mining Method

**🎮 GPU Mining (Recommended - Best Performance)**
```powershell
# High-performance GPU mining with Qiskit acceleration
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress

# Show all GPU mining options
.\run-gpu-miner.ps1 -Help
```

**💻 CPU Mining (No Dependencies)**
```powershell
# CPU-only mining (no additional setup required)
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress

# Show all CPU mining options  
.\run-cpu-miner.ps1 -Help
```

## 📊 Performance Comparison

| Mining Method | Performance | Dependencies | Best For |
|---------------|-------------|--------------|----------|
| **GPU (Qiskit)** | **0.45 puzzles/sec** | Python 3.8+, Qiskit | Maximum performance |
| **CPU** | 0.36 puzzles/sec | None | Easy setup |

## 🏗️ Project Structure

```
Qgeth3/
├── quantum-geth/          # Quantum-enhanced Ethereum client
├── quantum-miner/         # Original CPU miner
├── quantum-gpu-miner/     # Advanced GPU/CPU miner with Qiskit
├── scripts/               # Blockchain management scripts
├── run-gpu-miner.ps1     # GPU mining launcher (root)
├── run-cpu-miner.ps1     # CPU mining launcher (root)
└── start-geth.ps1        # Blockchain node launcher
```

## ⚛️ Quantum Mining Features

**✅ Real Quantum Circuits:**
- 16-qubit quantum circuits per puzzle
- 8192 T-gates per puzzle for quantum complexity
- 48 quantum puzzles per block

**✅ Advanced Acceleration:**
- Qiskit-based GPU quantum simulation
- Batch processing optimization (48 puzzles in one call)
- Automatic fallback to CPU if GPU unavailable

**✅ Bitcoin-Style Mining:**
- Proof-of-Work consensus with quantum difficulty
- Dynamic difficulty adjustment
- Real blockchain integration

## 🛠️ Setup Instructions

### Prerequisites
- **Windows 10/11** (PowerShell scripts)
- **Go 1.21+** (for building from source)
- **Python 3.8+** (for GPU mining only)

### GPU Mining Setup (Optional)
```powershell
# Install Python dependencies for GPU acceleration
pip install qiskit qiskit-aer numpy
```

### Build from Source (Optional)
```powershell
# Build GPU miner
cd quantum-gpu-miner
go build -o quantum-gpu-miner.exe .

# Build CPU miner  
cd quantum-miner
go build -o quantum-miner.exe .
```

## 🎮 Detailed Usage

### GPU Mining (quantum-gpu-miner)

**Features:**
- Qiskit GPU acceleration
- Batch quantum simulation
- Real-time performance monitoring
- Automatic backend selection

**Location:** `quantum-gpu-miner/` folder
**Executables:** `quantum-gpu-miner.exe` (GPU), `quantum-gpu-miner-cpu.exe` (CPU fallback)

```powershell
# From root directory
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -Threads 2

# From quantum-gpu-miner directory
cd quantum-gpu-miner
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress  # CPU version
```

### CPU Mining (quantum-miner)

**Features:**
- Pure CPU quantum simulation
- Multi-threaded mining
- No external dependencies
- Lightweight and fast

**Location:** `quantum-miner/` folder  
**Executable:** `quantum-miner.exe`

```powershell
# From root directory
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress -Threads 4

# Direct execution
.\quantum-miner.exe -coinbase 0xYourAddress -threads 4 -node http://localhost:8545
```

## 🔧 Advanced Configuration

### Custom Node Connection
```powershell
# Connect to remote node
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -NodeURL http://192.168.1.100:8545

# Multiple GPU devices
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -GpuId 1
```

### Multi-threaded Mining
```powershell
# CPU mining with 8 threads
.\run-cpu-miner.ps1 -Coinbase 0xYourAddress -Threads 8

# GPU mining with 4 parallel circuits
.\run-gpu-miner.ps1 -Coinbase 0xYourAddress -Threads 4
```

## 📈 Mining Statistics

Both miners provide real-time statistics:
- **Puzzle Rate:** Quantum puzzles solved per second
- **Block Success Rate:** Percentage of successful block submissions
- **Hash Rate:** Equivalent traditional mining hash rate
- **Quantum Metrics:** Circuit execution time and success rates

## 🔍 Troubleshooting

### GPU Mining Issues
```powershell
# Check Python installation
python --version

# Install/update Qiskit
pip install --upgrade qiskit qiskit-aer numpy

# Test GPU miner help
.\run-gpu-miner.ps1 -Help
```

### CPU Mining Issues
```powershell
# Check if quantum-geth is running
curl http://localhost:8545

# Test CPU miner help
.\run-cpu-miner.ps1 -Help
```

### Common Solutions
1. **"Executable not found":** Run build scripts or use pre-built binaries
2. **"Connection refused":** Start quantum-geth with `.\start-geth.ps1`
3. **"Invalid coinbase":** Use proper Ethereum address format (0x...)
4. **"Python not found":** Install Python 3.8+ for GPU mining

## 🏆 Performance Tips

1. **Use GPU Mining:** 25% better performance than CPU-only
2. **Optimize Threads:** Start with 1-2 threads, increase based on system
3. **Monitor Resources:** Watch CPU/GPU usage to find optimal settings
4. **Network Latency:** Run miner on same machine as quantum-geth for best results

## 📚 Documentation

- **Quantum-GPU-Miner:** See `quantum-gpu-miner/README.md` for detailed GPU mining guide
- **Quantum-Miner:** See `quantum-miner/README.md` for CPU mining specifics  
- **Quantum-Geth:** See `quantum-geth/README.md` for blockchain node documentation
- **Scripts:** See `scripts/README.md` for blockchain management tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both CPU and GPU miners
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the individual component licenses for details.

---

## 🎯 Getting Started Checklist

- [ ] Install Python 3.8+ (for GPU mining)
- [ ] Run `pip install qiskit qiskit-aer numpy` (for GPU mining)
- [ ] Start quantum-geth: `.\start-geth.ps1`
- [ ] Get your coinbase address ready
- [ ] Choose mining method: GPU (`.\run-gpu-miner.ps1`) or CPU (`.\run-cpu-miner.ps1`)
- [ ] Start mining with your coinbase address!

**Happy Quantum Mining! ⚛️💎** 