# Linux Mining Guide

Complete guide to quantum mining on Linux with Q Coin, including GPU acceleration and CPU fallback options.

## ⚛️ Quantum Mining Overview

### Real Quantum Computing Features
- **16-qubit quantum circuits** per puzzle
- **20 T-gates per puzzle** for quantum complexity
- **128 quantum puzzles per block**
- **Bitcoin-style Proof-of-Work** with quantum difficulty
- **Dynamic difficulty adjustment** (ASERT-Q algorithm)
- **Real blockchain integration** with halving rewards

### Linux GPU Acceleration Options

Q Coin Linux supports **two GPU acceleration methods**:

#### 1. **Native CUDA** (Highest Performance)
- Direct CUDA C++ implementation
- Maximum performance for NVIDIA GPUs
- Requires CUDA development toolkit
- **Performance**: ~3.2 puzzles/sec (RTX 4090)

#### 2. **Qiskit-Aer GPU** (Python-based)
- Uses Qiskit-Aer GPU backend
- Good performance, easier setup
- Linux-only feature (Windows uses CuPy)
- **Performance**: ~2.4 puzzles/sec (RTX 4090)

#### 3. **CPU Fallback**
- Universal compatibility
- Optimized for multi-core systems
- **Performance**: ~0.3-0.8 puzzles/sec

## 🔧 Prerequisites

### System Requirements
- **Linux OS** (Ubuntu 20.04+ recommended)
- **NVIDIA GPU** with CUDA support (for GPU mining)
- **8GB+ RAM** (16GB+ recommended)
- **Python 3.8+**

### For Native CUDA Mining (Maximum Performance)
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Install development tools
sudo apt install build-essential pkg-config
```

### For Qiskit-Aer GPU Mining (Easier Setup)
```bash
# Install Python dependencies
pip3 install qiskit qiskit-aer numpy

# Verify GPU support
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

### For CPU-Only Mining
```bash
# Basic dependencies only
sudo apt update
sudo apt install golang-go python3 python3-pip git build-essential
```

## 🏗️ Building the Miner

### Auto-Detection Build
```bash
# Auto-detect and build with best available GPU support
./scripts/linux/build-linux.sh miner

# Force CPU-only build
GPU_MODE=disable ./scripts/linux/build-linux.sh miner
```

### Build Output Examples

**CUDA Build:**
```
🔍 NVIDIA GPU detected, checking CUDA availability...
✅ CUDA development environment found
🏗️  Build Configuration:
  GPU Type: CUDA
  Build Tags: cuda
  CGO Enabled: 1
✅ Quantum-Miner built successfully: ./quantum-miner (CUDA)
```

**Qiskit-GPU Build:**
```
🔍 NVIDIA GPU detected, checking CUDA availability...
⚠️  CUDA development libraries not found, checking for Qiskit-Aer GPU...
✅ Qiskit-Aer GPU support detected
🏗️  Build Configuration:
  GPU Type: Qiskit-GPU
  Build Tags: cuda
  CGO Enabled: 0
✅ Quantum-Miner built successfully: ./quantum-miner (Qiskit-GPU)
```

**CPU Build:**
```
🔍 No NVIDIA GPU detected, building CPU-only version
🏗️  Build Configuration:
  GPU Type: CPU
  Build Tags: none
  CGO Enabled: 0
✅ Quantum-Miner built successfully: ./quantum-miner (CPU)
```

## 🚀 Mining Usage

### Quick Start
```bash
# Auto-detect everything and start mining
./scripts/linux/start-miner.sh

# Mine with specific settings
./scripts/linux/start-miner.sh --threads 8 --coinbase 0xYourAddress --verbose

# Force GPU mining (fail if not available)
./scripts/linux/start-miner.sh --gpu

# Force CPU mining
./scripts/linux/start-miner.sh --cpu
```

### Network Selection
```bash
# Q Coin Testnet (default)
./scripts/linux/start-miner.sh --testnet

# Q Coin Mainnet
./scripts/linux/start-miner.sh --mainnet

# Custom node connection
./scripts/linux/start-miner.sh --node http://192.168.1.100:8545
```

### Advanced Options
```bash
# Full option example
./scripts/linux/start-miner.sh \
  --threads 16 \
  --coinbase 0x1234567890123456789012345678901234567890 \
  --node http://localhost:8545 \
  --gpu \
  --verbose

# Built-in mining with node
./scripts/linux/start-geth.sh testnet --mining

# Direct miner usage
./quantum-miner -threads 4 -coinbase 0xYourAddress -node http://localhost:8545 -log
```

## 📊 Performance Comparison

| Method | RTX 4090 | RTX 3080 | GTX 1080 Ti | Setup Difficulty |
|--------|----------|----------|-------------|------------------|
| **Native CUDA** | ~3.2 puzzles/sec | ~2.1 puzzles/sec | ~1.3 puzzles/sec | Hard |
| **Qiskit-Aer GPU** | ~2.4 puzzles/sec | ~1.6 puzzles/sec | ~0.9 puzzles/sec | Medium |
| **CPU (24 cores)** | ~0.4 puzzles/sec | ~0.3 puzzles/sec | ~0.2 puzzles/sec | Easy |

## 🔧 Optimization Tips

### GPU Optimization
```bash
# Set GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit (adjust for your GPU)

# CPU governor for performance
sudo cpupower frequency-set -g performance
```

### Environment Variables
```bash
# For Qiskit-Aer GPU
export QISKIT_IN_PARALLEL=TRUE
export OPENBLAS_NUM_THREADS=1

# For CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0

# Memory optimization
export GOMEMLIMIT=2GiB
```

### Multi-GPU Setup
```bash
# Run multiple miners for multiple GPUs
CUDA_VISIBLE_DEVICES=0 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545 &
CUDA_VISIBLE_DEVICES=1 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545 &
CUDA_VISIBLE_DEVICES=2 ./quantum-miner -threads 4 -coinbase 0xAddr -node http://localhost:8545 &
```

## 🎮 Real-Time Monitoring

### GPU Monitoring
```bash
# Monitor GPU usage while mining
watch -n 1 nvidia-smi

# Monitor GPU temperature and power
nvidia-smi -l 1 --query-gpu=timestamp,name,temperature.gpu,power.draw --format=csv
```

### Performance Monitoring
```bash
# Check miner logs
./scripts/linux/start-miner.sh --verbose | grep "puzzles/sec"

# Monitor system resources
htop
iotop
```

### Mining Statistics
```bash
# Monitor mining performance with timestamps
./scripts/linux/start-miner.sh --verbose 2>&1 | tee mining-performance.log

# Filter for performance metrics
grep "puzzles/sec" mining-performance.log

# Real-time performance tracking
tail -f mining-performance.log | grep -E "(puzzles/sec|Block found|Error)"
```

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Qiskit-Aer GPU
python3 -c "import qiskit_aer; from qiskit_aer import AerSimulator; AerSimulator(device='GPU')"
```

### Build Issues

**CUDA Build Fails:**
```bash
# Install missing dependencies
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
sudo apt install libcudart-dev libcublas-dev

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

**Qiskit-Aer Issues:**
```bash
# Reinstall with GPU support
pip3 uninstall qiskit-aer
pip3 install qiskit-aer-gpu

# Check installation
python3 -c "import qiskit_aer; print(qiskit_aer.__version__)"
```

### Runtime Issues

**GPU Memory Errors:**
```bash
# Reduce batch size or threads
./scripts/linux/start-miner.sh --threads 4

# Set memory limits
export CUDA_MEMORY_POOL_SIZE=0.5  # Use 50% of GPU memory
```

**Permission Errors:**
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login again
```

**Network Issues:**
```bash
# Check if node is responding
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Check node sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545
```

## 🏆 Best Practices

### Hardware Optimization
1. **Dedicated Mining Rigs**: Separate mining from node operations
2. **Multiple GPUs**: Run multiple miner instances with different CUDA devices
3. **Adequate Cooling**: Ensure proper ventilation for sustained performance
4. **Power Supply**: Ensure adequate PSU for high-end GPUs

### Software Optimization
1. **Keep Updated**: Rebuild regularly with `git pull && ./scripts/linux/build-linux.sh miner`
2. **Monitor Resources**: Use `htop` and `nvidia-smi` to find optimal settings
3. **Thread Tuning**: Start with CPU core count, adjust based on performance
4. **Environment Variables**: Set optimal values for GPU frameworks

### Operational Best Practices
1. **24/7 Operation**: Set up systemd services for automatic restarts
2. **Monitoring**: Implement alerts for miner failures or performance drops
3. **Security**: Use dedicated mining addresses, not main wallets
4. **Backup Plans**: Have fallback mining configurations

## 🎯 Systemd Service Setup

### Create Mining Service
```bash
# Create systemd service for automatic mining
sudo tee /etc/systemd/system/qcoin-miner.service << EOF
[Unit]
Description=Q Coin Quantum Miner
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$HOME/Qgeth3
ExecStart=$HOME/Qgeth3/scripts/linux/start-miner.sh --testnet --verbose
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable qcoin-miner.service
sudo systemctl start qcoin-miner.service

# Check status
sudo systemctl status qcoin-miner.service
```

### Monitor Service
```bash
# View service logs
sudo journalctl -u qcoin-miner.service -f

# Service management
sudo systemctl stop qcoin-miner.service
sudo systemctl restart qcoin-miner.service
sudo systemctl disable qcoin-miner.service
```

## 🔗 Next Steps

### Mining Setup Complete
After successful mining setup:

1. **Monitor Performance**: Use the monitoring commands above
2. **Scale Operations**: Consider multiple miners or VPS deployment
3. **Join Community**: Connect with other miners for tips and support
4. **Stay Updated**: Keep software updated for latest optimizations

### Performance Tuning
- **Thread Count**: Experiment with different thread counts
- **GPU Settings**: Adjust GPU power limits and memory usage
- **Network Optimization**: Ensure fast connection to blockchain node
- **System Tuning**: Optimize Linux kernel settings for mining

## ✅ Linux Mining Checklist

### Pre-Mining Setup
- [ ] NVIDIA drivers installed (for GPU mining)
- [ ] CUDA toolkit installed (for native CUDA)
- [ ] Python dependencies installed (for Qiskit-Aer)
- [ ] Miner built with desired acceleration
- [ ] Node running and synced

### Mining Operation
- [ ] Miner starting successfully
- [ ] GPU utilization optimal (if using GPU)
- [ ] Thread count optimized
- [ ] Mining address configured
- [ ] Performance meeting expectations

### Ongoing Monitoring
- [ ] GPU temperatures monitored
- [ ] Performance metrics tracked
- [ ] System resources monitored
- [ ] Mining logs reviewed regularly
- [ ] Software kept up to date

**🎉 Happy Linux Quantum Mining! 🐧⚛️💎** 