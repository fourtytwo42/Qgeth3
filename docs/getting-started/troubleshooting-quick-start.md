# Quick Start Troubleshooting

Solutions for common issues when getting started with Q Coin.

## 🔧 Quick Diagnostics

### Basic System Check
```bash
# Check if you're in the right directory
pwd
ls -la | grep -E "(quantum-geth|scripts|configs)"

# Check system requirements
uname -a
go version  # Should be 1.21+
python3 --version  # Should be 3.8+
free -h  # Should have 4GB+ RAM
df -h   # Should have 10GB+ free space

# Quick status check
./quick-start.sh status
```

## 📁 Directory and File Issues

### Wrong Directory
```bash
# Symptoms: "quick-start.sh: No such file or directory"
# Solution: Navigate to correct directory

# Find Qgeth3 directory
find ~ -name "Qgeth3" -type d 2>/dev/null
find /home -name "Qgeth3" -type d 2>/dev/null

# Navigate to project directory
cd ~/Qgeth3  # Most common location
# or
cd /path/to/your/Qgeth3

# Verify you're in the right place
ls -la | grep -E "(quick-start.sh|quantum-geth|scripts)"
```

### Missing Files
```bash
# Check if all required files exist
ls -la quick-start.sh
ls -la quantum-geth/
ls -la scripts/
ls -la configs/

# If files are missing, re-clone repository
git status
git pull origin main

# If repository is corrupted, fresh clone
cd ..
rm -rf Qgeth3
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
```

### Permission Issues
```bash
# Symptoms: "Permission denied" when running quick-start.sh
# Solution: Fix file permissions

# Make quick-start.sh executable
chmod +x quick-start.sh

# Fix all script permissions
find scripts/ -name "*.sh" -exec chmod +x {} \;

# Fix ownership if needed
sudo chown -R $(whoami):$(whoami) .

# Verify permissions
ls -la quick-start.sh  # Should show -rwxr-xr-x
```

## 🚀 Quick-Start Script Issues

### Script Won't Run
```bash
# Check shell compatibility
echo $SHELL
/bin/bash quick-start.sh  # Try explicitly with bash

# Check for script errors
bash -x quick-start.sh build  # Debug mode

# Verify line endings (Windows to Linux)
file quick-start.sh  # Should show "with LF line terminators"
# If it shows CRLF, fix with:
dos2unix quick-start.sh
```

### Build Command Fails
```bash
# Try building components individually
./quick-start.sh build geth
./quick-start.sh build miner

# Clean build
./quick-start.sh clean
./quick-start.sh build

# Check build logs
ls -la build-*.log
tail -50 build-*.log
```

### Start Command Issues
```bash
# Check what's preventing startup
./quick-start.sh status

# Look for port conflicts
sudo lsof -i :8545
sudo lsof -i :30303

# Kill conflicting processes
pkill -f geth
pkill -f quantum-miner

# Try starting with explicit network
./quick-start.sh start testnet
./quick-start.sh start devnet
```

## 🔧 Build Issues

### Go Not Found
```bash
# Check Go installation
which go
go version

# Install Go if missing
sudo apt update
sudo apt install golang-go

# Or install latest Go manually
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

### Build Tools Missing
```bash
# Install required build tools
sudo apt update
sudo apt install build-essential git

# For CentOS/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install git

# Verify installation
gcc --version
make --version
git --version
```

### Memory Issues
```bash
# Check available memory
free -h

# Add swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Build with memory limit
export GOMEMLIMIT=1GiB
./quick-start.sh build
```

## 🌐 Network Issues

### Port Already in Use
```bash
# Find what's using port 8545
sudo lsof -i :8545
sudo netstat -tulpn | grep :8545

# Kill the process
sudo kill -9 <PID>

# Or use different port
export GETH_HTTP_PORT=8546
./quick-start.sh start
```

### Firewall Blocking
```bash
# Check firewall status
sudo ufw status

# Allow required ports
sudo ufw allow 8545/tcp  # HTTP RPC
sudo ufw allow 30303/tcp # P2P
sudo ufw allow 30303/udp # P2P

# Test connectivity
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545
```

### DNS/Connectivity Issues
```bash
# Test internet connectivity
ping -c 4 google.com
ping -c 4 github.com

# Test DNS resolution
nslookup github.com

# If behind proxy, configure Git
git config --global http.proxy http://proxy:port
git config --global https.proxy https://proxy:port
```

## 💾 Storage Issues

### Insufficient Disk Space
```bash
# Check disk usage
df -h
du -sh ~/.qcoin  # Check blockchain data

# Clean up space
sudo apt autoremove
sudo apt autoclean
go clean -cache

# Remove old blockchain data if needed
rm -rf ~/.qcoin/testnet/geth/chaindata
rm -rf ~/.qcoin/devnet/geth/chaindata
```

### Disk I/O Errors
```bash
# Check disk health
sudo dmesg | grep -i error
sudo fsck /dev/sda1  # Replace with your disk

# Check for read-only filesystem
mount | grep ro

# Remount as read-write if needed
sudo mount -o remount,rw /
```

## 🎮 GPU Issues (Quick Start Mining)

### No GPU Detected
```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia
nvidia-smi

# Install drivers if needed
sudo apt update
sudo apt install nvidia-driver-525

# Reboot after driver installation
sudo reboot
```

### Python Dependencies Missing
```bash
# Install Python packages for mining
pip3 install qiskit qiskit-aer numpy

# For GPU acceleration
pip3 install qiskit-aer-gpu

# Test installation
python3 -c "import qiskit_aer; print('Qiskit-Aer OK')"
```

## 🔄 Service Issues

### Auto-Start Problems
```bash
# Check if systemd services exist
systemctl --user status qgeth.service
systemctl --user status qminer.service

# Create service files if missing
./quick-start.sh install-service

# Enable auto-start
systemctl --user enable qgeth.service
systemctl --user enable qminer.service
```

### Log Issues
```bash
# Check log directories exist
ls -la ~/.qcoin/logs/

# Create log directory if missing
mkdir -p ~/.qcoin/logs

# Check log permissions
ls -la ~/.qcoin/logs/
chmod 755 ~/.qcoin/logs
```

## 🧪 Testing Issues

### Test Commands Fail
```bash
# Run basic connectivity test
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Test with timeout
timeout 10 curl http://localhost:8545

# Check if geth is actually running
ps aux | grep geth
```

### Performance Tests
```bash
# Test system performance
./quick-start.sh test

# CPU benchmark
sysbench --test=cpu --cpu-max-prime=20000 run

# Memory test
sysbench --test=memory --memory-total-size=1G run

# Disk I/O test
dd if=/dev/zero of=testfile bs=1M count=100
rm testfile
```

## 🚨 Emergency Recovery

### Complete Reset
```bash
# Stop everything
./quick-start.sh stop

# Kill any remaining processes
pkill -f geth
pkill -f quantum-miner

# Clean build artifacts
./quick-start.sh clean

# Remove data directories
rm -rf ~/.qcoin

# Fresh start
./quick-start.sh build
./quick-start.sh start
```

### Configuration Reset
```bash
# Backup current config
cp -r configs/ configs.backup/

# Reset to defaults
git checkout configs/

# Or download fresh configs
curl -o configs/genesis_quantum_testnet.json \
  https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/configs/genesis_quantum_testnet.json
```

## 📊 Monitoring and Diagnostics

### Status Monitoring
```bash
# Continuous status monitoring
watch -n 5 './quick-start.sh status'

# Check blockchain sync
./quick-start.sh sync-status

# Monitor system resources
htop
iotop -ao
```

### Log Analysis
```bash
# Real-time log monitoring
tail -f ~/.qcoin/logs/geth.log
tail -f ~/.qcoin/logs/miner.log

# Search for errors
grep -i error ~/.qcoin/logs/*.log
grep -i warn ~/.qcoin/logs/*.log

# Check last startup
tail -100 ~/.qcoin/logs/geth.log | grep -E "(starting|error|fatal)"
```

## 🔍 Debug Information Collection

### System Information
```bash
# Collect debug information
echo "=== Quick Start Debug Info ===" > debug-info.txt
echo "Date: $(date)" >> debug-info.txt
echo "Directory: $(pwd)" >> debug-info.txt
echo "User: $(whoami)" >> debug-info.txt
echo "" >> debug-info.txt

echo "System:" >> debug-info.txt
uname -a >> debug-info.txt
echo "" >> debug-info.txt

echo "Go Version:" >> debug-info.txt
go version >> debug-info.txt
echo "" >> debug-info.txt

echo "Python Version:" >> debug-info.txt
python3 --version >> debug-info.txt
echo "" >> debug-info.txt

echo "Memory:" >> debug-info.txt
free -h >> debug-info.txt
echo "" >> debug-info.txt

echo "Disk Space:" >> debug-info.txt
df -h >> debug-info.txt
echo "" >> debug-info.txt

echo "File Permissions:" >> debug-info.txt
ls -la quick-start.sh >> debug-info.txt
ls -la scripts/ >> debug-info.txt
echo "" >> debug-info.txt

cat debug-info.txt
```

### Quick-Start Specific Diagnostics
```bash
# Test each quick-start function
echo "Testing quick-start functions..." > qs-test.txt

echo "Build test:" >> qs-test.txt
timeout 30 ./quick-start.sh build --test >> qs-test.txt 2>&1

echo "Status test:" >> qs-test.txt
./quick-start.sh status >> qs-test.txt 2>&1

echo "Network test:" >> qs-test.txt
./quick-start.sh test-network >> qs-test.txt 2>&1

cat qs-test.txt
```

## 📚 Getting Help

### Information to Provide
When seeking help with quick-start issues:

1. **Operating System**: `uname -a`
2. **Go Version**: `go version`
3. **Python Version**: `python3 --version`
4. **Project Directory**: `pwd && ls -la`
5. **Error Messages**: Full command output
6. **System Resources**: `free -h && df -h`

### Common Quick Fixes
```bash
# The "universal" fix for most issues
git pull origin main
chmod +x quick-start.sh scripts/**/*.sh
./quick-start.sh clean
./quick-start.sh build
./quick-start.sh start

# If that doesn't work, nuclear option:
pkill -f geth
pkill -f quantum-miner
rm -rf ~/.qcoin
git pull origin main
./quick-start.sh build
./quick-start.sh start
```

## ✅ Quick Start Checklist

### Pre-Requirements
- [ ] Linux system with 4GB+ RAM
- [ ] 10GB+ free disk space  
- [ ] Internet connectivity
- [ ] Go 1.21+ installed
- [ ] Git installed

### File Setup
- [ ] In correct Qgeth3 directory
- [ ] quick-start.sh is executable
- [ ] All scripts have execute permissions
- [ ] No file permission issues

### Network Setup
- [ ] Ports 8545, 30303 not in use
- [ ] Firewall allows required ports
- [ ] No proxy blocking connections
- [ ] DNS resolution working

### Build Requirements
- [ ] Build tools installed (gcc, make)
- [ ] Sufficient memory/swap available
- [ ] No disk space issues
- [ ] Go environment variables set

**Most quick-start issues are resolved by ensuring proper permissions and having all dependencies installed!** 