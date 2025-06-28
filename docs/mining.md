# Mining Guide

Complete guide to quantum mining with Q Coin, including CPU and GPU acceleration options.

## ⚛️ Quantum Mining Overview

### Real Quantum Computing Features
- **16-qubit quantum circuits** per puzzle
- **20 T-gates per puzzle** for quantum complexity
- **128 quantum puzzles per block**
- **Bitcoin-style Proof-of-Work** with quantum difficulty
- **Dynamic difficulty adjustment** (ASERT-Q algorithm)
- **Real blockchain integration** with halving rewards

### Mining Acceleration Options
- **Linux**: Native CUDA + Qiskit-Aer GPU support
- **Windows**: CuPy GPU acceleration
- **Both**: CPU fallback with optimization
- **Batch processing**: 128 puzzles in one GPU call
- **Automatic fallback**: GPU → CPU if GPU unavailable

## 📊 Performance Comparison

| Platform | Method | Performance | Dependencies | Best For |
|----------|--------|-------------|--------------|----------|
| **Linux** | **Native CUDA** | **~3.2 puzzles/sec** | CUDA toolkit, C++ dev tools | Maximum performance |
| **Linux** | **Qiskit-Aer GPU** | **~2.4 puzzles/sec** | Python, qiskit-aer | Good performance, easier setup |
| **Windows** | **CuPy GPU** | **~2.0 puzzles/sec** | Python, CuPy | Windows GPU acceleration |
| **Both** | **CPU** | ~0.3-0.8 puzzles/sec | None | Easy setup, any system |

### Expected VPS Performance

**CPU Mining:**
- 1 vCPU: ~0.1-0.2 puzzles/sec
- 2 vCPU: ~0.2-0.4 puzzles/sec  
- 4 vCPU: ~0.4-0.8 puzzles/sec
- 8 vCPU: ~0.6-1.2 puzzles/sec

**GPU Mining (Linux VPS):**
- GTX 1080 Ti: ~1.3 puzzles/sec (CUDA) / ~0.9 puzzles/sec (Qiskit)
- RTX 3080: ~2.1 puzzles/sec (CUDA) / ~1.6 puzzles/sec (Qiskit)
- RTX 4090: ~3.2 puzzles/sec (CUDA) / ~2.4 puzzles/sec (Qiskit)

## 🐧 Linux Mining Setup

### GPU Mining Installation
```bash
# For maximum performance (Native CUDA)
sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev

# For easier setup (Qiskit-Aer GPU)
pip3 install qiskit qiskit-aer numpy

# Verify GPU support
nvidia-smi
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

### Building the Miner
```bash
# Build miner with auto-detection
./scripts/linux/build-linux.sh miner

# Possible outputs:
# ✅ Quantum-Miner built successfully: ./quantum-miner (CUDA)      # Native CUDA
# ✅ Quantum-Miner built successfully: ./quantum-miner (Qiskit-GPU) # Python GPU
# ✅ Quantum-Miner built successfully: ./quantum-miner (CPU)       # CPU fallback
```

### Linux Mining Commands
```bash
# Smart miner (auto-detects GPU/CPU)
./scripts/linux/start-miner.sh --testnet --verbose

# Force GPU mining
./scripts/linux/start-miner.sh --gpu --threads 8 --coinbase 0xYourAddress

# Force CPU mining
./scripts/linux/start-miner.sh --cpu --threads 16 --coinbase 0xYourAddress

# Connect to remote node
./scripts/linux/start-miner.sh --node http://192.168.1.100:8545 --mainnet

# Direct miner usage
./quantum-miner -threads 4 -coinbase 0xYourAddress -node http://localhost:8545 -log
```

## 🪟 Windows Mining Setup

### GPU Mining Installation
```powershell
# Install CUDA toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Install Python dependencies
pip install qiskit qiskit-aer numpy cupy-cuda11x

# Verify GPU support
python -c "import cupy; print('GPU OK' if cupy.cuda.is_available() else 'No GPU')"
```

### Building the Miner
```powershell
# Build miner releases
.\scripts\windows\build-release.ps1 miner

# The build system automatically detects Visual Studio and GPU capabilities
```

### Windows Mining Commands
```powershell
# GPU mining
.\scripts\windows\start-miner.ps1 -GPU -Threads 8 -Coinbase 0xYourAddress

# CPU mining
.\scripts\windows\start-miner.ps1 -CPU -Threads 4 -Coinbase 0xYourAddress

# Connect to remote node
.\scripts\windows\start-miner.ps1 -Node "http://192.168.1.100:8545" -Coinbase 0xYourAddress
```

## 🎮 Mining Usage Examples

### Network Selection
```bash
# Q Coin Testnet (Chain ID 73235) - Default
./scripts/linux/start-miner.sh --testnet

# Q Coin Mainnet (Chain ID 73236)
./scripts/linux/start-miner.sh --mainnet

# Custom node connection
./scripts/linux/start-miner.sh --node http://localhost:8545
```

### Basic Mining
```bash
# Start testnet node
./scripts/linux/start-geth.sh testnet

# Start smart miner (auto-detects best acceleration)
./scripts/linux/start-miner.sh --testnet --verbose

# Expected output:
# ✅ GPU mining available - Using GPU mode
# ⚡ Linux GPU Batch Complete: 128 puzzles in 0.234s (547.0 puzzles/sec)
```

### Advanced Mining Options
```bash
# Custom coinbase address
./scripts/linux/start-miner.sh --testnet --coinbase 0x1234567890123456789012345678901234567890

# Specific thread count
./scripts/linux/start-miner.sh --threads 8 --verbose

# Mining with built-in node
./scripts/linux/start-geth.sh testnet --mining

# Multi-GPU setup
CUDA_VISIBLE_DEVICES=0 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545
CUDA_VISIBLE_DEVICES=1 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545
```

## 🔧 Mining Optimization

### GPU Mining Optimization
```bash
# Set GPU performance mode (Linux)
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit to 300W

# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export QISKIT_IN_PARALLEL=TRUE
export OPENBLAS_NUM_THREADS=1

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### CPU Mining Optimization
```bash
# Set CPU performance mode (Linux)
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimal thread count (usually = CPU cores)
nproc  # Shows number of CPU cores
./scripts/linux/start-miner.sh --cpu --threads $(nproc)

# Monitor CPU usage
htop
```

### Memory Optimization
```bash
# For systems with limited RAM
export GOMEMLIMIT=1GiB

# Monitor memory usage
free -h
ps aux | grep quantum-miner
```

## 📈 Mining Statistics & Monitoring

### Real-Time Mining Stats
Both miners provide comprehensive statistics:
- **Puzzle Rate:** Quantum puzzles solved per second
- **Block Success Rate:** Percentage of successful block submissions
- **Hash Rate:** Equivalent traditional mining hash rate
- **Quantum Metrics:** Circuit execution time and success rates
- **GPU Utilization:** GPU memory and compute usage (if applicable)

### Monitoring Commands
```bash
# Monitor mining performance
./scripts/linux/start-miner.sh --verbose | grep "puzzles/sec"

# Monitor system resources
htop
iotop
nvidia-smi -l 1  # For GPU systems

# Check node sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545

# Check mining statistics
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_mining","params":[],"id":1}' \
  http://localhost:8545
```

### Performance Logging
```bash
# Save mining performance to log
./scripts/linux/start-miner.sh --verbose 2>&1 | tee mining-performance.log

# Filter for performance metrics
grep "puzzles/sec" mining-performance.log

# Monitor in real-time
tail -f mining-performance.log | grep -E "(puzzles/sec|Block found|Error)"
```

## 🌐 Remote Mining Setup

### Connect to Remote Node
```bash
# Mine against remote Q Coin node
./scripts/linux/start-miner.sh --node http://192.168.1.100:8545 --verbose

# Mine against mainnet node
./scripts/linux/start-miner.sh --node http://mainnet.qcoin.network:8545 --mainnet

# Custom configuration
./quantum-miner \
  -node http://your-vps-ip:8545 \
  -coinbase 0xYourMiningAddress \
  -threads 8 \
  -log
```

### Mining Pool Setup
```bash
# Connect to mining pool (when available)
./scripts/linux/start-miner.sh --pool stratum+tcp://pool.qcoin.network:4444 --worker your-worker-name

# Pool mining with specific GPU
CUDA_VISIBLE_DEVICES=0 ./quantum-miner \
  -pool stratum+tcp://pool.qcoin.network:4444 \
  -worker worker1 \
  -threads 4
```

## 🚨 Mining Troubleshooting

### GPU Issues
```bash
# GPU not detected
nvidia-smi
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev

# CUDA version conflicts
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Python GPU dependencies
pip3 install --upgrade qiskit qiskit-aer
python3 -c "import qiskit_aer; print('GPU OK')"
```

### CPU Issues
```bash
# High CPU usage but low performance
# Check for thermal throttling
cat /proc/cpuinfo | grep MHz

# Optimize thread count
./scripts/linux/start-miner.sh --cpu --threads $(($(nproc) - 1))

# Check for competing processes
top
ps aux | grep -E "(geth|miner)"
```

### Network Issues
```bash
# Node not responding
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Connection timeouts
# Check firewall settings
sudo ufw status
sudo ufw allow 8545/tcp

# Check if node is syncing
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545
```

### Build Issues
```bash
# Miner build fails
./scripts/linux/build-linux.sh miner --clean

# Missing dependencies
sudo apt install -y golang-go python3 python3-pip git build-essential

# GPU build fails
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
export PATH=/usr/local/cuda/bin:$PATH
```

## 🏆 Mining Best Practices

### Hardware Optimization
1. **Use GPU Mining:** 3-10x better performance than CPU-only
2. **Dedicated Mining Rigs:** Separate mining from node operations
3. **Multiple GPUs:** Run multiple miner instances with different CUDA devices
4. **Cooling:** Ensure adequate cooling for sustained performance
5. **Power Supply:** Ensure adequate PSU for high-end GPUs

### Software Optimization
1. **Keep Updated:** Rebuild regularly for latest optimizations
2. **Monitor Resources:** Watch CPU/GPU usage to find optimal settings
3. **Thread Tuning:** Start with CPU core count, adjust based on performance
4. **Network Optimization:** Use local node or fast connection to remote node
5. **Environment Variables:** Set optimal environment for GPU frameworks

### Operational Best Practices
1. **24/7 Operation:** Set up systemd services for automatic restarts
2. **Monitoring:** Implement alerts for miner failures or performance drops
3. **Backup Plans:** Have fallback mining configurations
4. **Security:** Use dedicated mining addresses, not main wallets
5. **Documentation:** Keep mining configuration documented

## 🎯 Mining Performance Tips

### Maximize GPU Performance
```bash
# Use native CUDA over Python GPU when possible
./scripts/linux/build-linux.sh miner  # Builds with best available GPU support

# Optimize GPU memory
nvidia-smi -q -d MEMORY
export CUDA_MEMORY_POOL_SIZE=0.8  # Use 80% of GPU memory

# Multiple GPU setup
CUDA_VISIBLE_DEVICES=0 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545 &
CUDA_VISIBLE_DEVICES=1 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545 &
```

### Optimize for Different Scenarios
```bash
# High-end gaming PC (single powerful GPU)
./scripts/linux/start-miner.sh --gpu --threads 8 --verbose

# Multi-core server (CPU mining)
./scripts/linux/start-miner.sh --cpu --threads $(($(nproc) * 2))

# Low-power system (efficient mining)
./scripts/linux/start-miner.sh --cpu --threads 2

# VPS mining (conservative)
./scripts/linux/start-miner.sh --cpu --threads $(nproc) --node http://localhost:8545
```

## 🔗 Next Steps

### Mining Setup Complete
After successful mining setup:

1. **[Performance Monitoring](advanced-configuration.md#performance-monitoring)** - Monitor and optimize performance
2. **[VPS Deployment](vps-deployment.md)** - Scale to multiple VPS instances
3. **[Troubleshooting](troubleshooting.md)** - Fix common mining issues
4. **[Advanced Configuration](advanced-configuration.md)** - Fine-tune for optimal performance

### Join the Network
- **Testnet:** Start with Q Coin Testnet for testing and learning
- **Mainnet:** Move to Q Coin Mainnet for production mining
- **Community:** Join Discord/Telegram for mining tips and support

## ✅ Mining Checklist

### Pre-Mining Setup
- [ ] Node synced and running
- [ ] Miner built with optimal GPU/CPU support
- [ ] Mining address generated
- [ ] Network connection stable
- [ ] System resources monitored

### Mining Operation
- [ ] Miner starting successfully
- [ ] Puzzle rate meeting expectations
- [ ] GPU/CPU utilization optimal
- [ ] No errors in mining logs
- [ ] Block submissions successful

### Ongoing Monitoring
- [ ] Performance metrics tracked
- [ ] System temperature monitored
- [ ] Mining profitability calculated
- [ ] Software kept up to date
- [ ] Backup mining configuration ready

**🎉 Happy Quantum Mining! ⚛️💎** 