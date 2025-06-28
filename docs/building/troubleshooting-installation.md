# Installation Troubleshooting

Solutions for common installation and build issues with Q Coin.

## 🔧 Quick Installation Diagnostics

### Check Installation Status
```bash
# Check Go installation
go version  # Should be 1.21+

# Check Git installation  
git --version

# Check Python installation (for GPU mining)
python3 --version  # Should be 3.8+

# Check system resources
free -h
df -h
```

## 🔨 Go Build Issues

### Go Not Found
```bash
# Symptoms: "go: command not found"
# Solution: Install or fix Go PATH

# Ubuntu/Debian
sudo apt update
sudo apt install golang-go

# CentOS/RHEL/Fedora
sudo dnf install golang

# Manual installation (latest version)
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz

# Fix PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version
```

### Go Version Too Old
```bash
# Symptoms: "go version go1.xx.x: minimum supported version is go1.21"
# Solution: Upgrade Go manually

# Remove old version
sudo apt remove golang-go  # If installed via package manager
sudo rm -rf /usr/local/go

# Install latest version
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz

# Update PATH and reload
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
go version
```

### Memory Issues During Build
```bash
# Symptoms: Build killed, "out of memory", or process terminated
# Solution: Add swap space

# Check current memory
free -h

# Add 3GB swap file
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
swapon --show

# Alternative: Use smaller build with memory limit
export GOMEMLIMIT=1GiB
./scripts/linux/build-linux.sh geth
```

### Linker Errors
```bash
# Symptoms: "stat ./build-temp-X/gotmp: no such file or directory"
# Solution: Clean build and use latest scripts

# Update to latest version
git pull origin main

# Clean previous build artifacts
./scripts/linux/build-linux.sh geth --clean

# If still failing, check temp directory permissions
ls -la ./build-temp*
sudo chown -R $(whoami):$(whoami) ./build-temp*

# Manual cleanup
rm -rf ./build-temp-*
rm -f geth geth.bin quantum-miner
```

### Build Tool Dependencies
```bash
# Symptoms: "gcc: command not found", "make: command not found"
# Solution: Install build tools

# Ubuntu/Debian
sudo apt update
sudo apt install build-essential pkg-config git

# CentOS/RHEL 8+
sudo dnf groupinstall "Development Tools"
sudo dnf install git pkgconfig

# Verify installation
gcc --version
make --version
git --version
```

## 🎮 GPU Installation Issues

### NVIDIA Drivers (Linux)
```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Check current driver
nvidia-smi

# Install NVIDIA drivers (Ubuntu)
sudo apt update
sudo apt install nvidia-driver-525  # Or latest version

# Alternative: Use graphics-drivers PPA for latest
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-525

# Reboot after installation
sudo reboot

# Verify after reboot
nvidia-smi
```

### CUDA Toolkit (Linux)
```bash
# Symptoms: "nvcc: command not found", "CUDA not available"
# Solution: Install CUDA development tools

# Method 1: Package manager (easier)
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev

# Method 2: Official CUDA installer (latest)
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

### Python GPU Dependencies
```bash
# Symptoms: "ModuleNotFoundError: No module named 'qiskit_aer'"
# Solution: Install Python GPU dependencies

# Basic Python dependencies
sudo apt install python3 python3-pip

# Install Qiskit and GPU libraries
pip3 install --upgrade pip
pip3 install qiskit qiskit-aer numpy

# For GPU support (Linux)
pip3 install qiskit-aer-gpu

# For Windows GPU support
pip install cupy-cuda11x

# Test GPU installation
python3 -c "import qiskit_aer; from qiskit_aer import AerSimulator; print(AerSimulator(device='GPU'))"

# If import fails, check CUDA compatibility
python3 -c "import cupy; print('CUDA available:', cupy.cuda.is_available())"
```

### Visual Studio Build Tools (Windows)
```powershell
# Symptoms: "vcvarsall.bat not found", "Microsoft Visual C++ 14.0 is required"
# Solution: Install Visual Studio Build Tools

# Download and install from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Required components:
# - MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
# - Windows 10/11 SDK (latest version)
# - CMake tools for Visual Studio

# Verify installation
where cl  # Should show path to MSVC compiler

# If not found, check Visual Studio installation
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vs_installer.exe"

# Alternative: Install Visual Studio Community (includes build tools)
# https://visualstudio.microsoft.com/vs/community/
```

## 🌐 Network & Dependency Issues

### Git Clone Issues
```bash
# Symptoms: "Permission denied", "Could not resolve host"
# Solution: Check network and Git configuration

# Test network connectivity
ping -c 4 github.com

# Check DNS resolution
nslookup github.com

# If behind corporate firewall, configure Git proxy
git config --global http.proxy http://proxy.company.com:8080
git config --global https.proxy https://proxy.company.com:8080

# Alternative: Use HTTPS instead of SSH
git clone https://github.com/fourtytwo42/Qgeth3.git

# If permission denied, check SSH keys
ssh -T git@github.com
```

### Package Manager Issues
```bash
# Ubuntu/Debian: Package not found
sudo apt update
sudo apt install software-properties-common

# Add universe repository if needed
sudo add-apt-repository universe

# CentOS/RHEL: Enable EPEL repository
sudo dnf install epel-release

# If package conflicts
sudo apt --fix-broken install
sudo dpkg --configure -a
```

### Python Version Issues
```bash
# Symptoms: "Python 3.8+ required", module compatibility issues
# Solution: Install correct Python version

# Check current version
python3 --version

# Ubuntu 20.04+: Install Python 3.9
sudo apt install python3.9 python3.9-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Alternative: Use pyenv for version management
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.9.16
pyenv global 3.9.16

# Verify
python3 --version
pip3 --version
```

## 🔧 Build Environment Issues

### Permissions Problems
```bash
# Symptoms: "Permission denied" during build
# Solution: Fix file permissions

# Fix ownership of project directory
sudo chown -R $(whoami):$(whoami) ~/Qgeth3

# Fix permissions on scripts
chmod +x scripts/linux/*.sh
chmod +x scripts/deployment/*.sh

# If building in /opt or system directory
sudo chown -R $(whoami):$(whoami) /opt/qgeth

# Check if user is in required groups
groups
sudo usermod -a -G video $(whoami)  # For GPU access
sudo usermod -a -G docker $(whoami)  # If using Docker
```

### Disk Space Issues
```bash
# Symptoms: "No space left on device"
# Solution: Free up disk space

# Check disk usage
df -h
du -sh /tmp  # Check temp directory

# Clean package cache
sudo apt autoremove
sudo apt autoclean

# Clean Go cache
go clean -cache
go clean -modcache

# Remove old build artifacts
rm -rf ./build-temp-*
rm -f geth geth.bin quantum-miner

# Clear system logs if needed
sudo journalctl --vacuum-time=7d
```

### Environment Variable Issues
```bash
# Symptoms: Build fails with "GOPATH not set" or similar
# Solution: Set required environment variables

# Set Go environment
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc

# Set build variables
echo 'export CGO_ENABLED=0' >> ~/.bashrc  # For Linux builds
echo 'export GOOS=linux' >> ~/.bashrc
echo 'export GOARCH=amd64' >> ~/.bashrc

# Reload environment
source ~/.bashrc

# Verify settings
go env
echo $GOPATH
echo $PATH
```

## 🚨 Emergency Installation Recovery

### Complete Clean Installation
```bash
# Remove all Q Geth installations
rm -rf ~/Qgeth3
sudo rm -rf /opt/qgeth

# Clean Go cache and modules
go clean -cache
go clean -modcache

# Start fresh installation
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Build with clean environment
./scripts/linux/build-linux.sh geth --clean
```

### Dependency Recovery
```bash
# Reset package manager state (Ubuntu)
sudo apt update --fix-missing
sudo apt install -f

# Reinstall critical dependencies
sudo apt install --reinstall golang-go python3 python3-pip git build-essential

# Clear pip cache
pip3 cache purge

# Reinstall Python dependencies
pip3 uninstall qiskit qiskit-aer cupy-cuda11x
pip3 install qiskit qiskit-aer numpy
```

## 📚 Getting Installation Help

### Information to Collect
When reporting installation issues:

1. **Operating System**: `uname -a`
2. **Go Version**: `go version`
3. **Python Version**: `python3 --version`
4. **Available Memory**: `free -h`
5. **Disk Space**: `df -h`
6. **GPU Info**: `nvidia-smi` (if applicable)
7. **Error Messages**: Full build output

### Installation Test Script
```bash
# Quick installation verification
echo "=== Installation Check ===" > install-check.txt
echo "OS: $(uname -a)" >> install-check.txt
echo "Go: $(go version 2>&1)" >> install-check.txt
echo "Python: $(python3 --version 2>&1)" >> install-check.txt
echo "Git: $(git --version 2>&1)" >> install-check.txt
echo "Memory: $(free -h | grep Mem)" >> install-check.txt
echo "Disk: $(df -h | grep -E '^/dev')" >> install-check.txt
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU')" >> install-check.txt

cat install-check.txt
```

## ✅ Installation Checklist

### Pre-Installation
- [ ] Operating system supported (Linux/Windows)
- [ ] Sufficient disk space (10GB minimum)
- [ ] Adequate RAM (4GB minimum, 8GB recommended)
- [ ] Network connectivity working

### Dependencies
- [ ] Go 1.21+ installed and in PATH
- [ ] Git installed and configured
- [ ] Python 3.8+ installed (for GPU mining)
- [ ] Build tools installed (gcc, make)
- [ ] NVIDIA drivers installed (for GPU)
- [ ] CUDA toolkit installed (for GPU)

### Build Environment
- [ ] Project directory has correct permissions
- [ ] Scripts are executable
- [ ] Environment variables set correctly
- [ ] Sufficient swap space available
- [ ] No conflicting software installed

**If installation continues to fail, collect the diagnostic information above and seek help!** 