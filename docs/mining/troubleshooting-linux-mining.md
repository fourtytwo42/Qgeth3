# Linux Mining Troubleshooting

Solutions for Linux mining issues with Q Coin quantum blockchain.

## 🔧 Quick Mining Diagnostics

### Check Mining Status
```bash
# Check if miner binary exists
ls -la quantum-miner
file quantum-miner  # Should show it's executable

# Check GPU availability
nvidia-smi

# Check system resources
htop
free -h

# Test quantum computing libraries
python3 -c "import qiskit_aer; print('Qiskit-Aer installed')"
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"
```

## 🎮 GPU Detection Issues

### NVIDIA GPU Not Detected
```bash
# Symptoms: "No NVIDIA GPU detected", nvidia-smi fails
# Solution: Install/fix NVIDIA drivers

# Check if GPU is physically detected
lspci | grep -i nvidia

# Check current driver status
nvidia-smi
cat /proc/driver/nvidia/version

# Install NVIDIA drivers (Ubuntu 20.04+)
sudo apt update
sudo apt install nvidia-driver-525  # Or latest version

# Alternative: Use graphics-drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-525

# For Ubuntu 18.04 or older systems
sudo apt install nvidia-driver-470

# Reboot after installation
sudo reboot

# Verify after reboot
nvidia-smi
nvidia-settings  # If GUI available
```

### CUDA Not Available
```bash
# Symptoms: "CUDA not available", nvcc command not found
# Solution: Install CUDA toolkit

# Method 1: Package manager (recommended)
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev

# Method 2: Official CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi

# Test CUDA with simple program
echo '#include <stdio.h>
__global__ void hello() { printf("Hello from GPU!\\n"); }
int main() { hello<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }' > test.cu
nvcc test.cu -o test
./test
rm test test.cu
```

### Multiple GPU Issues
```bash
# Check all GPUs
nvidia-smi -L

# Check GPU memory and utilization
nvidia-smi -q -d MEMORY,UTILIZATION

# Set specific GPU for mining (if multiple)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs

# Test specific GPU
python3 -c "
import cupy
cupy.cuda.Device(0).use()  # Use GPU 0
print('GPU 0 available:', cupy.cuda.is_available())
"

# If GPU not working, check power management
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -pl 300  # Set power limit (watts)
```

## 🧮 Quantum Computing Library Issues

### Qiskit Installation Problems
```bash
# Symptoms: "ModuleNotFoundError: No module named 'qiskit'"
# Solution: Install or reinstall Qiskit

# Check Python version (need 3.8+)
python3 --version

# Update pip first
pip3 install --upgrade pip

# Install Qiskit with Aer backend
pip3 install qiskit qiskit-aer numpy

# For GPU acceleration (Linux)
pip3 install qiskit-aer-gpu

# If installation fails, try with --user flag
pip3 install --user qiskit qiskit-aer

# Verify installation
python3 -c "import qiskit; print(qiskit.__version__)"
python3 -c "import qiskit_aer; print('Aer available')"
```

### GPU Acceleration Not Working
```bash
# Symptoms: Miner falls back to CPU, slow performance
# Solution: Fix GPU libraries

# Check if Qiskit can see GPU
python3 -c "
from qiskit_aer import AerSimulator
sim = AerSimulator(device='GPU')
print('GPU simulator available:', sim.available_devices())
"

# Install CUDA-specific libraries
pip3 install cupy-cuda11x  # For CUDA 11.x
pip3 install cupy-cuda12x  # For CUDA 12.x

# Check CuPy installation
python3 -c "
import cupy
print('CuPy version:', cupy.__version__)
print('CUDA available:', cupy.cuda.is_available())
print('GPU count:', cupy.cuda.runtime.getDeviceCount())
"

# If still not working, reinstall with specific CUDA version
pip3 uninstall qiskit-aer-gpu cupy
pip3 install qiskit-aer-gpu cupy-cuda11x

# Alternative: Use Native CUDA instead
# Ensure quantum-miner was built with CUDA support
./scripts/linux/build-linux.sh miner --cuda
```

### Memory Issues with GPU
```bash
# Symptoms: "CUDA out of memory", "GPU memory allocation failed"
# Solution: Reduce memory usage

# Check GPU memory usage
nvidia-smi

# Set memory fraction for mining
export CUDA_MEMORY_POOL_SIZE=0.8  # Use 80% of GPU memory

# Reduce threads if using GPU
./scripts/linux/start-miner.sh --gpu --threads 2

# Use GPU memory management
export CUPY_MEMPOOL_SIZE=2048  # 2GB memory pool

# Alternative: Use CPU mining
./scripts/linux/start-miner.sh --cpu --threads $(nproc)

# Monitor memory during mining
watch -n 1 nvidia-smi
```

## ⚡ Performance Issues

### Low Hash Rate
```bash
# Check system performance
htop
iostat 1
nvidia-smi dmon  # Monitor GPU in real-time

# Optimize CPU mining threads
# Start with CPU core count
./scripts/linux/start-miner.sh --cpu --threads $(nproc)

# Test different thread counts
for i in {1..8}; do
  echo "Testing $i threads..."
  timeout 30s ./scripts/linux/start-miner.sh --cpu --threads $i --test
done

# Optimize GPU mining
./scripts/linux/start-miner.sh --gpu --threads 4  # Start with 4
./scripts/linux/start-miner.sh --gpu --threads 8  # Try 8
./scripts/linux/start-miner.sh --gpu --threads 2  # Try fewer for stability

# Check CPU governor (performance mode)
sudo cpupower frequency-set -g performance

# Disable CPU power saving
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### System Overheating
```bash
# Check temperatures
sensors  # Install: sudo apt install lm-sensors
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits

# Monitor during mining
watch -n 2 'sensors | grep -E "(Core|GPU)"; nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader'

# Reduce GPU power if overheating
sudo nvidia-smi -pl 250  # Limit to 250W

# Improve cooling
# Check fans
sudo pwmconfig  # Configure fan curves

# Alternative: Reduce mining intensity
./scripts/linux/start-miner.sh --gpu --threads 2 --intensity 50
```

### Network Connectivity Issues
```bash
# Check if geth node is running
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Test different node endpoints
./scripts/linux/start-miner.sh --node http://127.0.0.1:8545
./scripts/linux/start-miner.sh --node http://localhost:8545

# Check firewall
sudo ufw status
sudo ufw allow 8545/tcp

# Test network with telnet
telnet localhost 8545

# If using remote node
ping your.node.ip.address
./scripts/linux/start-miner.sh --node http://your.node.ip.address:8545
```

## 🔧 Mining Binary Issues

### Miner Binary Not Found
```bash
# Symptoms: "./quantum-miner: No such file or directory"
# Solution: Build the miner

# Check if binary exists
ls -la quantum-miner

# Build the miner
./scripts/linux/build-linux.sh miner

# If build fails, try with CUDA
./scripts/linux/build-linux.sh miner --cuda

# Check binary after build
ls -la quantum-miner
file quantum-miner  # Should show ELF executable

# Make executable if needed
chmod +x quantum-miner

# Test binary
./quantum-miner --help
```

### Miner Crashes on Start
```bash
# Check for missing libraries
ldd quantum-miner

# Run with debugging
./quantum-miner --cpu --threads 1 --debug

# Check system logs
dmesg | tail -20
journalctl -f | grep quantum-miner

# Try minimal configuration
./quantum-miner --cpu --threads 1 --node http://localhost:8545

# If segmentation fault, rebuild with debug symbols
./scripts/linux/build-linux.sh miner --debug
gdb ./quantum-miner
# Inside gdb: run --cpu --threads 1
# If it crashes: bt (for backtrace)
```

### Permission Issues
```bash
# Symptoms: "Permission denied" when starting miner
# Solution: Fix permissions

# Check file permissions
ls -la quantum-miner

# Make executable
chmod +x quantum-miner

# Check if user is in required groups
groups
sudo usermod -a -G video $(whoami)

# Reboot or re-login for group changes
sudo reboot

# For GPU access
sudo chmod 666 /dev/nvidia*
```

## 🌐 Network and Connectivity

### Connection Timeouts
```bash
# Check network connectivity to node
ping -c 4 localhost
curl -I http://localhost:8545

# Increase timeout values
./scripts/linux/start-miner.sh --timeout 30

# Check if geth is responding
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  http://localhost:8545

# Use different ports if needed
./scripts/linux/start-miner.sh --node http://localhost:8546

# Check firewall blocking
sudo netstat -tulpn | grep 8545
sudo ufw allow from 127.0.0.1 to any port 8545
```

### Mining Pool Issues
```bash
# If using mining pool instead of solo mining
# Test pool connectivity
telnet pool.address.com 4444

# Check pool status
curl http://pool.address.com:8080/stats

# Use different stratum settings
./scripts/linux/start-miner.sh --pool stratum+tcp://pool.address.com:4444 --wallet your.wallet.address
```

## 🔄 Service Management Issues

### Systemd Service Problems
```bash
# Check service status
sudo systemctl status quantum-miner.service

# View service logs
sudo journalctl -u quantum-miner.service -f

# Restart service
sudo systemctl restart quantum-miner.service

# Enable auto-start
sudo systemctl enable quantum-miner.service

# Edit service configuration
sudo systemctl edit quantum-miner.service

# Reload after changes
sudo systemctl daemon-reload
sudo systemctl restart quantum-miner.service
```

### Screen/Tmux Session Issues
```bash
# List active sessions
screen -ls
tmux list-sessions

# Attach to mining session
screen -r mining
tmux attach -t mining

# If session is dead/detached incorrectly
screen -wipe
tmux kill-server

# Start new mining session
screen -S mining -dm bash -c './scripts/linux/start-miner.sh --gpu'
tmux new-session -d -s mining './scripts/linux/start-miner.sh --gpu'
```

## 🚨 Emergency Mining Recovery

### Complete Mining Reset
```bash
# Stop all mining processes
pkill -f quantum-miner
screen -wipe
tmux kill-server

# Rebuild miner binary
rm -f quantum-miner
./scripts/linux/build-linux.sh miner --clean

# Reset GPU state
sudo nvidia-smi --gpu-reset

# Clear any stale locks
rm -f /tmp/quantum-miner.lock

# Restart mining
./scripts/linux/start-miner.sh --gpu --threads 4
```

### GPU Recovery Procedures
```bash
# Reset GPU if hung
sudo nvidia-smi --gpu-reset

# Reload NVIDIA drivers
sudo rmmod nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm

# Reset persistence mode
sudo nvidia-smi -pm 0
sudo nvidia-smi -pm 1

# Check for hardware issues
nvidia-smi -q | grep -E "(Temperature|Memory|Power)"
```

## 📊 Performance Monitoring

### Mining Statistics
```bash
# Monitor hash rate
tail -f ~/.qcoin/logs/mining.log | grep -E "(hashrate|accepted|rejected)"

# System resource monitoring
htop
iotop -ao
nvidia-smi dmon

# Network monitoring
netstat -i
vnstat -l  # Install: sudo apt install vnstat

# Temperature monitoring during mining
watch -n 2 'sensors; nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv,noheader'
```

### Optimization Testing
```bash
# Benchmark different configurations
for threads in 2 4 6 8; do
  echo "Testing GPU with $threads threads..."
  timeout 60s ./scripts/linux/start-miner.sh --gpu --threads $threads --benchmark
done

# CPU vs GPU comparison
echo "CPU Mining (5 minutes):"
timeout 300s ./scripts/linux/start-miner.sh --cpu --threads $(nproc) --benchmark

echo "GPU Mining (5 minutes):"
timeout 300s ./scripts/linux/start-miner.sh --gpu --threads 4 --benchmark
```

## 📚 Getting Mining Help

### Mining Diagnostic Info
```bash
# Collect mining diagnostic information
echo "=== Mining Diagnostics ===" > mining-debug.txt
echo "GPU Info:" >> mining-debug.txt
nvidia-smi >> mining-debug.txt 2>&1
echo -e "\nCUDA Version:" >> mining-debug.txt
nvcc --version >> mining-debug.txt 2>&1
echo -e "\nPython Libraries:" >> mining-debug.txt
python3 -c "import qiskit; print('Qiskit:', qiskit.__version__)" >> mining-debug.txt 2>&1
python3 -c "import qiskit_aer; print('Qiskit-Aer available')" >> mining-debug.txt 2>&1
python3 -c "import cupy; print('CuPy:', cupy.__version__)" >> mining-debug.txt 2>&1
echo -e "\nMiner Binary:" >> mining-debug.txt
ls -la quantum-miner >> mining-debug.txt 2>&1
echo -e "\nSystem Resources:" >> mining-debug.txt
free -h >> mining-debug.txt
echo -e "\nCPU Info:" >> mining-debug.txt
lscpu | grep -E "(Model name|CPU.*:|Core.*:|Thread.*:)" >> mining-debug.txt

cat mining-debug.txt
```

## ✅ Linux Mining Checklist

### Hardware
- [ ] NVIDIA GPU detected and drivers installed
- [ ] CUDA toolkit installed and working
- [ ] Adequate GPU memory (4GB minimum)
- [ ] Proper cooling and power supply
- [ ] Stable system temperature

### Software
- [ ] Python 3.8+ with Qiskit and dependencies
- [ ] quantum-miner binary built and executable
- [ ] Q Geth node running and accessible
- [ ] No firewall blocking mining ports
- [ ] Systemd service configured (optional)

### Performance
- [ ] Optimal thread count determined
- [ ] GPU acceleration working
- [ ] System resources not overloaded
- [ ] Network connectivity stable
- [ ] Mining statistics being logged

**For persistent mining issues on Linux, collect the diagnostic information above and seek help!** 