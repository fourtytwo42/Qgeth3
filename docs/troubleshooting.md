# Troubleshooting Guide

Solutions for common Q Coin issues, organized by category with step-by-step fixes.

## 🔧 Quick Diagnostics

### System Health Check
```bash
# Check Q Geth status
./quick-start.sh status

# Check if node is running
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Check system resources
free -h
df -h
htop
```

### Log Analysis
```bash
# Linux: Check geth logs
tail -f ~/.qcoin/logs/geth.log

# VPS: Check service logs
qgeth-service logs geth

# Windows: Check PowerShell output
# Logs are displayed in the console when running scripts
```

## 🔨 Build & Installation Issues

### Go Build Errors

#### Go Not Found
```bash
# Symptoms: "go: command not found"
# Solution: Install or fix Go PATH
export PATH=$PATH:/usr/local/go/bin
go version

# If still not found, install Go:
sudo apt install golang-go  # Ubuntu/Debian
sudo dnf install golang     # CentOS/RHEL
```

#### Go Version Too Old
```bash
# Symptoms: "go version go1.xx.x: minimum supported version is go1.21"
# Solution: Install newer Go manually
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

#### Memory Issues During Build
```bash
# Symptoms: Build killed or "out of memory"
# Solution: Add swap space
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
```

#### Linker Errors
```bash
# Symptoms: "stat ./build-temp-X/gotmp: no such file or directory"
# Solution: Already fixed in latest version, update your scripts
git pull origin main
./scripts/linux/build-linux.sh geth --clean
```

### GPU Build Issues

#### CUDA Not Found (Linux)
```bash
# Symptoms: "nvcc: command not found" or "CUDA not available"
# Solution: Install CUDA development tools
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
nvcc --version
nvidia-smi
```

#### Python GPU Dependencies
```bash
# Symptoms: "ModuleNotFoundError: No module named 'qiskit_aer'"
# Solution: Install Python dependencies
pip3 install --upgrade qiskit qiskit-aer numpy

# For GPU support
pip3 install qiskit-aer-gpu  # Linux
pip install cupy-cuda11x     # Windows

# Test installation
python3 -c "import qiskit_aer; print('OK')"
```

#### Visual Studio Not Found (Windows)
```powershell
# Symptoms: "vcvarsall.bat not found" 
# Solution: Install Visual Studio 2022 Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Install with C++ components:
# - MSVC v143 - VS 2022 C++ x64/x86 build tools
# - Windows 10/11 SDK
# - CMake tools for Visual Studio
```

## 🌐 Network & Connectivity Issues

### Node Not Starting

#### Port Already in Use
```bash
# Symptoms: "bind: address already in use"
# Solution: Kill existing processes
sudo lsof -i :8545  # Find processes using port 8545
sudo kill -9 <PID>  # Kill the process

# Or use different ports
./scripts/linux/start-geth.sh testnet --http.port 8546
```

#### Permission Denied
```bash
# Symptoms: "permission denied" when binding to ports
# Solution: Use non-privileged ports or fix permissions
# Non-privileged ports (>1024) don't need sudo
./scripts/linux/start-geth.sh testnet --http.port 8545 --port 30303

# Or run with sudo (not recommended for regular use)
sudo ./scripts/linux/start-geth.sh testnet
```

### Sync Issues

#### Peers Not Connecting
```bash
# Check peer count
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545

# Check if firewall is blocking P2P
sudo ufw status
sudo ufw allow 30303/tcp
sudo ufw allow 30303/udp

# Manual peer addition (if needed)
# Use geth console to add peers manually
```

#### Sync Stuck or Slow
```bash
# Check sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545

# If sync is stuck, restart with fresh database
./scripts/linux/start-geth.sh testnet --syncmode full

# For complete reset (removes all blockchain data)
rm -rf ~/.qcoin/testnet/geth/chaindata
./scripts/linux/start-geth.sh testnet
```

## 🏗️ VPS & Auto-Service Issues

### Bootstrap Script Issues

#### 404 Error When Downloading
```bash
# Symptoms: "bash: line 1: 404:: command not found"
# Cause: Script not available on GitHub (old URL or network issue)
# Solution: Use correct URL and check network
curl -I https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh

# If 404, clone manually:
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
sudo ./scripts/deployment/bootstrap-qgeth.sh -y
```

#### Memory Prompts in Non-Interactive Mode
```bash
# Symptoms: Script hangs on memory prompts despite -y flag
# Solution: Ensure using latest version with non-interactive fixes
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y

# Or update existing installation
cd Qgeth3
git pull origin main
sudo ./scripts/deployment/bootstrap-qgeth.sh -y
```

#### Lock File Issues
```bash
# Symptoms: "Auto-service installation already in progress"
# Solution: Remove stale lock file
sudo rm -f /tmp/qgeth-auto-service.lock

# Then try again
sudo ./scripts/deployment/bootstrap-qgeth.sh -y
```

### Service Management Issues

#### Commands Not Found
```bash
# Symptoms: "qgeth-service: command not found"
# Solution: Reload shell or use full path
source ~/.bashrc

# Or use full path
/usr/local/bin/qgeth-service status

# Or use systemctl directly
sudo systemctl status qgeth-node.service
```

#### Service Fails to Start
```bash
# Check service status
qgeth-service status
sudo systemctl status qgeth-node.service

# Check logs for errors
qgeth-service logs geth
sudo journalctl -u qgeth-node.service -f

# Common fixes:
# 1. Rebuild binary
cd /opt/qgeth/Qgeth3
sudo ./scripts/linux/build-linux.sh geth --clean

# 2. Fix permissions
sudo chown -R qgeth:qgeth /opt/qgeth

# 3. Restart daemon
sudo systemctl daemon-reload
sudo systemctl restart qgeth-node.service
```

### Auto-Update Issues

#### GitHub Monitor Not Working
```bash
# Check monitor status
qgeth-service logs github

# Common issues:
# 1. Network connectivity
ping -c 4 github.com

# 2. Git pull failures
cd /opt/qgeth/Qgeth3
git status
git pull origin main

# 3. Service not enabled
sudo systemctl enable qgeth-github-monitor.service
sudo systemctl start qgeth-github-monitor.service
```

#### Build Failures During Update
```bash
# Check update logs
qgeth-service logs update

# Manual recovery
cd /opt/qgeth/Qgeth3
sudo ./scripts/linux/build-linux.sh geth --clean

# Restore from backup if needed
ls -la /opt/qgeth/backup/
sudo rm -rf /opt/qgeth/Qgeth3
sudo cp -r /opt/qgeth/backup/Qgeth3_YYYYMMDD_HHMMSS /opt/qgeth/Qgeth3
qgeth-service restart
```

## ⛏️ Mining Issues

### Miner Not Starting

#### Miner Binary Not Found
```bash
# Symptoms: "./quantum-miner: No such file or directory"
# Solution: Build the miner
./scripts/linux/build-linux.sh miner

# Check if binary exists
ls -la quantum-miner
file quantum-miner  # Should show it's executable
```

#### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Install GPU drivers if needed
sudo apt install nvidia-driver-470  # Or latest version

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify GPU mining dependencies
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

### Mining Performance Issues

#### Low Hash Rate
```bash
# Check system resources
htop
nvidia-smi  # For GPU systems

# Optimize thread count
# Start with CPU core count
./scripts/linux/start-miner.sh --cpu --threads $(nproc)

# For GPU, try different thread counts
./scripts/linux/start-miner.sh --gpu --threads 4
./scripts/linux/start-miner.sh --gpu --threads 8
```

#### GPU Memory Errors
```bash
# Symptoms: "CUDA out of memory" or similar
# Solution: Reduce batch size or threads
export CUDA_MEMORY_POOL_SIZE=0.5  # Use 50% of GPU memory

# Or use CPU mining as fallback
./scripts/linux/start-miner.sh --cpu --threads 4
```

#### Connection Timeouts
```bash
# Check if node is responding
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Use different node
./scripts/linux/start-miner.sh --node http://192.168.1.100:8545

# Check firewall
sudo ufw allow 8545/tcp
```

## 💾 Data & Storage Issues

### Disk Space Problems

#### Out of Disk Space
```bash
# Check disk usage
df -h
du -h ~/.qcoin  # Check blockchain data size

# Clean up space
# 1. Remove old backups
sudo rm -rf /opt/qgeth/backup/Qgeth3_OLD*

# 2. Clean package cache
sudo apt autoremove
sudo apt autoclean

# 3. Rotate logs
sudo truncate -s 0 /opt/qgeth/logs/*.log
```

#### Database Corruption
```bash
# Symptoms: "database corruption" or "bad block"
# Solution: Reset blockchain database
rm -rf ~/.qcoin/testnet/geth/chaindata
./scripts/linux/start-geth.sh testnet

# For VPS installations
sudo systemctl stop qgeth-node.service
sudo rm -rf /opt/qgeth/Qgeth3/qdata
sudo systemctl start qgeth-node.service
```

### Permission Issues

#### Mixed Ownership
```bash
# Symptoms: "permission denied" when accessing files
# Solution: Fix ownership
sudo chown -R $(whoami):$(whoami) ~/Qgeth3
sudo chown -R qgeth:qgeth /opt/qgeth  # For VPS installations

# Check permissions
ls -la ~/.qcoin
ls -la /opt/qgeth
```

## 🔐 Security & Firewall Issues

### Firewall Blocking Connections

#### RPC Not Accessible
```bash
# Check if firewall is blocking HTTP RPC
sudo ufw status

# Allow RPC access
sudo ufw allow 8545/tcp

# Test RPC access
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545
```

#### P2P Not Working
```bash
# Allow P2P networking
sudo ufw allow 30303/tcp
sudo ufw allow 30303/udp

# Check if P2P is working
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545
```

## 🚨 Emergency Recovery Procedures

### Complete System Recovery

#### VPS Complete Reinstall
```bash
# If everything is broken, start fresh
sudo systemctl stop qgeth-node.service qgeth-github-monitor.service qgeth-updater.service
sudo systemctl disable qgeth-node.service qgeth-github-monitor.service qgeth-updater.service
sudo rm -rf /opt/qgeth

# Reinstall from scratch
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y
```

#### Backup Recovery
```bash
# Restore from backup (VPS)
qgeth-service stop
ls -la /opt/qgeth/backup/
sudo rm -rf /opt/qgeth/Qgeth3
sudo cp -r /opt/qgeth/backup/Qgeth3_YYYYMMDD_HHMMSS /opt/qgeth/Qgeth3
qgeth-service start

# Restore user installation
rm -rf ~/Qgeth3
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
./quick-start.sh build
```

### Service Recovery

#### Reset All Services
```bash
# Stop all services
sudo systemctl stop qgeth-node.service qgeth-github-monitor.service qgeth-updater.service

# Reset failed state
sudo systemctl reset-failed

# Reload daemon
sudo systemctl daemon-reload

# Restart services
qgeth-service restart

# Check status
qgeth-service status
```

#### Reset Crash Counter
```bash
# If services are in crash loop
qgeth-service reset-crashes

# Check crash count
cat /opt/qgeth/logs/crash_count.txt

# Manual reset
echo "0" | sudo tee /opt/qgeth/logs/crash_count.txt
```

## 🔍 Advanced Debugging

### Debug Logging

#### Enable Verbose Logging
```bash
# For geth debugging
./scripts/linux/start-geth.sh testnet --verbosity 4

# For miner debugging
./scripts/linux/start-miner.sh --testnet --verbose

# For service debugging
qgeth-service logs all | grep -E "(ERROR|WARN|DEBUG)"
```

#### Trace Network Issues
```bash
# Test network connectivity
ping -c 4 8.8.8.8
traceroute github.com

# Check DNS resolution
nslookup github.com

# Test GitHub API access
curl -I https://api.github.com/repos/fourtytwo42/Qgeth3/commits/main
```

### System Diagnostics

#### Memory Analysis
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck ./quantum-miner -threads 1 -test
```

#### Performance Profiling
```bash
# Profile CPU usage
perf record -g ./quantum-miner -threads 4 -test
perf report

# Monitor file I/O
iotop -ao

# Network monitoring
netstat -tulpn | grep -E "(8545|30303)"
```

## 📚 Getting Help

### Information to Collect
When seeking help, provide:
1. **Operating System**: `uname -a`
2. **Go Version**: `go version`
3. **Git Commit**: `git rev-parse HEAD`
4. **System Resources**: `free -h && df -h`
5. **Error Messages**: Full error output
6. **Log Files**: Relevant log excerpts

### Log Collection Script
```bash
# Quick system info collection
echo "=== System Info ===" > debug-info.txt
uname -a >> debug-info.txt
go version >> debug-info.txt
git rev-parse HEAD >> debug-info.txt
free -h >> debug-info.txt
df -h >> debug-info.txt

echo "=== Recent Logs ===" >> debug-info.txt
tail -50 ~/.qcoin/logs/geth.log >> debug-info.txt 2>/dev/null || echo "No local logs" >> debug-info.txt
qgeth-service logs geth | tail -50 >> debug-info.txt 2>/dev/null || echo "No service logs" >> debug-info.txt

cat debug-info.txt
```

## ✅ Troubleshooting Checklist

### Basic Checks
- [ ] System meets minimum requirements
- [ ] Latest version of Q Geth installed
- [ ] Required dependencies installed
- [ ] Adequate disk space available
- [ ] Network connectivity working

### Build Issues
- [ ] Go version 1.21+ installed
- [ ] Build tools installed (gcc, make)
- [ ] GPU drivers installed (for GPU mining)
- [ ] Python dependencies installed (for GPU mining)
- [ ] No permission errors during build

### Runtime Issues
- [ ] Ports 8545, 30303 available and not blocked
- [ ] Firewall configured correctly
- [ ] Sufficient memory for operation
- [ ] No conflicting processes running
- [ ] Configuration files accessible

### Service Issues (VPS)
- [ ] Services enabled and starting on boot
- [ ] Correct user permissions set
- [ ] Log files being written
- [ ] GitHub monitoring working
- [ ] Auto-update system functional

**If issues persist after following this guide, collect debug information and seek help from the community!** 