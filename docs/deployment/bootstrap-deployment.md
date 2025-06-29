# Q Geth Universal Bootstrap Deployment Guide

Complete guide for operating and managing your Q Geth node after installation by the universal bootstrap script.

## 🚀 Universal System Service Features

The bootstrap script creates a **truly universal** system service that works on ALL Linux distributions:

### 🎯 Multi-Init System Support
- **Systemd**: Ubuntu, Debian, CentOS, RHEL, openSUSE
- **OpenRC**: Alpine Linux, Gentoo
- **SysV Init**: Traditional Unix systems, older distributions
- **Upstart**: Older Ubuntu versions

### ⚠️ Distribution Compatibility

**Fully Supported (Bootstrap Script + System Service):**
- Ubuntu 20.04+ (LTS recommended)
- Debian 10+ (Stable/Testing)
- CentOS 7+, RHEL 7+
- openSUSE Leap/Tumbleweed
- Alpine Linux 3.14+
- Arch Linux (current)

**Manual Installation Only:**
- **Fedora**: Due to systemd service execution complexities and security policy differences, Fedora requires manual installation. The quantum blockchain builds and runs perfectly on Fedora, but the automated bootstrap service creation fails.

**Why Fedora Bootstrap Fails:**
- Fedora's enhanced systemd security policies cause service execution failures (exit code 203)
- Complex interactions between Fedora's SELinux policies and systemd sandboxing
- Fedora's rapidly changing package management and system configuration
- Different default PATH and environment variable handling in systemd services

### 🔒 Enterprise Security Features
- **Sandboxed Execution**: NoNewPrivileges, PrivateTmp, ProtectSystem
- **Resource Limits**: 65536 file handles, 4096 processes, memory protection
- **Privilege Restrictions**: Runs as user, not root
- **Secure File Access**: Read-only system directories, isolated temp space

### 🏗️ Production-Grade Reliability
- **Persistent Services**: Survive reboots and system restarts
- **Automatic Recovery**: Restart on failure with exponential backoff (10s delay)
- **Resource Management**: Proper cleanup on service stop
- **Professional Logging**: Integration with system logging (journalctl, /var/log)

## 🎯 Quick Reference

After running the bootstrap script, your system has:
- **Q Geth Node**: Running on testnet with persistent system service
- **Universal Service**: Auto-detected init system (systemd/OpenRC/SysV/Upstart)
- **Enterprise Security**: Sandboxed execution, resource limits, privilege restrictions
- **API Endpoints**: HTTP RPC (8545), WebSocket (8546)
- **Network**: P2P networking on port 30303
- **Service Management**: Universal scripts work with any Linux distribution
- **Persistent Operation**: Survives reboots, auto-restarts on failure

## 🚀 Bootstrap Installation

### Prerequisites

**For Debian/Ubuntu minimal installations**, you may need to install basic packages first:
```bash
# Update package list
apt update

# Install required packages for bootstrap script
apt install -y curl sudo

# For minimal Debian installations, you may also need:
apt install -y wget ca-certificates
```

**Note**: Most standard Ubuntu installations already include these packages, but minimal Debian containers/installations may not have `curl` and `sudo` by default.

**Go Version Handling**: The bootstrap script automatically handles Go version conflicts that can occur on Debian systems. If the package manager provides an outdated Go version (< 1.21), the script will:
- Remove conflicting package manager Go installations
- Install the latest Go manually to `/usr/local/go/`
- Configure PATH priorities to ensure the correct version is used
- Clean module caches to prevent version conflicts

### One-Command Installation
```bash
# Non-interactive installation (recommended for automation)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y

# 🐳 Docker installation (perfect for Fedora and cross-platform!)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash -s -- --docker

# Interactive installation (prompts for confirmation)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash

# Docker + non-interactive mode
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash -s -- --docker -y
```

### What Bootstrap Sets Up
- **Universal Compatibility**: Auto-detects init system (systemd/OpenRC/SysV/Upstart)
- **Dependencies**: Go 1.24.4, build tools, git, firewall configuration
- **Memory**: Creates swap file if needed (minimum 4GB total memory)
- **Project**: Clones repository to `~/qgeth/Qgeth3/` (user directory installation)
- **Build**: Compiles Q Geth with automated error recovery and retry logic
- **Persistent Service**: Creates appropriate service for your init system
- **Security Hardening**: Sandboxed execution, resource limits, privilege restrictions
- **Management Scripts**: Universal scripts that work with any Linux distribution
- **Firewall**: Configures system firewall with required ports (8545, 8546, 30303)
- **Auto-Start**: Service configured to start on boot and restart on failure

## 🔧 Fedora Manual Installation

**Important**: Fedora is not supported by the bootstrap script due to systemd execution complexities. However, Q Geth builds and runs perfectly on Fedora with manual installation.

### Prerequisites
```bash
# Update system packages
sudo dnf update -y

# Install required dependencies
sudo dnf install -y git gcc gcc-c++ make curl wget

# Install Go 1.24.4 (required for quantum consensus)
# Remove any existing Go from package manager
sudo dnf remove golang -y

# Download and install Go 1.24.4
cd /tmp
wget https://golang.org/dl/go1.24.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz

# Configure Go environment
echo 'export PATH=/usr/local/go/bin:$PATH' | sudo tee /etc/profile.d/go.sh
chmod +x /etc/profile.d/go.sh
source /etc/profile.d/go.sh

# Verify Go installation
go version  # Should show go1.24.4
```

### Manual Q Geth Installation
```bash
# Create installation directory
mkdir -p ~/qgeth
cd ~/qgeth

# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Make scripts executable
find . -name "*.sh" -type f -exec chmod +x {} \;

# Build Q Geth
cd scripts/linux
./build-linux.sh geth

# Verify build
ls -la ../../geth.bin  # Should exist
ls -la ../../geth      # Should exist
```

### Manual Service Setup (Optional)
```bash
# Create simple systemd service (minimal configuration)
sudo tee /etc/systemd/system/qgeth-manual.service > /dev/null << EOF
[Unit]
Description=Q Geth Quantum Blockchain Node (Manual)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/qgeth/Qgeth3/scripts/linux
ExecStart=/bin/bash -c 'cd $HOME/qgeth/Qgeth3/scripts/linux && ./start-geth.sh testnet'
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable qgeth-manual.service
sudo systemctl start qgeth-manual.service

# Check service status
sudo systemctl status qgeth-manual.service
```

### Manual Start (Recommended for Fedora)
```bash
# Start Q Geth manually (most reliable on Fedora)
cd ~/qgeth/Qgeth3/scripts/linux
./start-geth.sh testnet

# The node will run in foreground - use screen or tmux for background:
# Install screen: sudo dnf install screen -y
# Start in screen: screen -S qgeth
# Run: ./start-geth.sh testnet
# Detach: Ctrl+A, D
# Reattach: screen -r qgeth
```

### Fedora Firewall Configuration
```bash
# Configure firewalld (Fedora's default firewall)
sudo firewall-cmd --permanent --add-port=8545/tcp  # HTTP RPC
sudo firewall-cmd --permanent --add-port=8546/tcp  # WebSocket
sudo firewall-cmd --permanent --add-port=30303/tcp # P2P
sudo firewall-cmd --permanent --add-port=30303/udp # P2P
sudo firewall-cmd --reload

# Verify firewall rules
sudo firewall-cmd --list-ports
```

### Fedora-Specific Troubleshooting
```bash
# If SELinux causes issues, check and adjust:
sudo setsebool -P httpd_can_network_connect 1
sudo semanage port -a -t http_port_t -p tcp 8545
sudo semanage port -a -t http_port_t -p tcp 8546

# Check SELinux status
sestatus

# View SELinux denials (if any)
sudo sealert -a /var/log/audit/audit.log
```

## 🔧 Universal Service Management

The bootstrap installer creates management scripts that work with **ANY** Linux distribution and init system:

### Universal Service Control
```bash
# Use these scripts - they work on ALL Linux distributions!
~/qgeth/start-qgeth.sh      # Start Q Geth service
~/qgeth/stop-qgeth.sh       # Stop Q Geth service
~/qgeth/restart-qgeth.sh    # Restart Q Geth service
~/qgeth/status-qgeth.sh     # Check service status

# Examples:
~/qgeth/status-qgeth.sh     # Shows: "Q Geth service is running" or "stopped"
~/qgeth/restart-qgeth.sh    # Restarts service regardless of init system
```

### Init System Specific Commands (Advanced)
If you prefer to use native init system commands:

**Systemd** (Ubuntu, Fedora, Debian, CentOS):
```bash
sudo systemctl start qgeth.service
sudo systemctl stop qgeth.service
sudo systemctl status qgeth.service
sudo systemctl enable qgeth.service
```

**OpenRC** (Alpine, Gentoo):
```bash
sudo rc-service qgeth start
sudo rc-service qgeth stop
sudo rc-service qgeth status
sudo rc-update add qgeth default
```

**SysV Init** (Traditional systems):
```bash
sudo service qgeth start
sudo service qgeth stop
sudo service qgeth status
sudo chkconfig qgeth on
```

**Upstart** (Older Ubuntu):
```bash
sudo start qgeth
sudo stop qgeth
sudo status qgeth
```

### Auto-Start Configuration
```bash
# Services are automatically configured to start on boot
# To disable auto-start (advanced users):
~/qgeth/disable-autostart.sh

# To re-enable auto-start:
~/qgeth/enable-autostart.sh
```

## 📊 Monitoring Your Node

### Universal Log Viewing
```bash
# Use universal log viewer (works with any init system)
~/qgeth/logs-qgeth.sh       # View recent logs
~/qgeth/logs-qgeth.sh -f    # Follow live logs
~/qgeth/logs-qgeth.sh -n 100 # View last 100 lines

# The script automatically detects your init system and uses:
# - journalctl for systemd
# - /var/log files for SysV/OpenRC
# - Upstart log files for Upstart
```

### Init System Specific Log Commands (Advanced)
**Systemd** (Ubuntu, Fedora, Debian):
```bash
sudo journalctl -u qgeth.service -f
sudo journalctl -u qgeth.service --since "1 hour ago"
sudo journalctl -u qgeth.service --no-pager -l | tail -50
```

**OpenRC/SysV/Upstart** (Alpine, Gentoo, Traditional):
```bash
sudo tail -f /var/log/qgeth.log
sudo tail -f ~/qgeth/logs/qgeth.log
```

### Check Node Status
```bash
# Check if node is syncing
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545

# Get current block number
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545

# Check peer connections
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545

# Get node info
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"admin_nodeInfo","params":[],"id":1}' \
  http://localhost:8545

# Check client version
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545
```

### Check System Resources
```bash
# Memory and CPU usage
htop
free -h

# Disk usage
df -h
du -h /opt/qgeth/

# Network connections
ss -tuln | grep -E ':(8545|8546|30303)'

# Check geth process
ps aux | grep geth
```

## 🌐 Network Configuration

### Firewall Status
```bash
# Check UFW firewall status
sudo ufw status

# View detailed firewall rules
sudo ufw status verbose

# Check if required ports are open
sudo netstat -tuln | grep -E ':(8545|8546|30303)'
```

### Required Ports
| Port | Protocol | Purpose | Access |
|------|----------|---------|--------|
| 22 | TCP | SSH Access | Admin only |
| 8545 | TCP | HTTP RPC API | Public/Private |
| 8546 | TCP | WebSocket API | Public/Private |
| 30303 | TCP/UDP | P2P Networking | Public |

### Network Troubleshooting
```bash
# Test external connectivity
curl -I https://github.com
ping -c 4 8.8.8.8

# Test API accessibility
curl http://localhost:8545
curl http://your-vps-ip:8545

# Check if services are listening
sudo ss -tuln | grep -E ':(8545|8546|30303)'

# Test WebSocket
wscat -c ws://localhost:8546
```

## 🖥️ Geth Console Access

### Attach to Running Node
```bash
# Connect via IPC (recommended)
cd /opt/qgeth/Qgeth3
./geth attach ipc:~/.qcoin/testnet/geth.ipc

# Connect via HTTP (if remote)
./geth attach http://localhost:8545

# Connect via WebSocket
./geth attach ws://localhost:8546
```

### Common Console Commands
```javascript
// Check account balance
eth.getBalance("0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A")

// Get latest block
eth.getBlock("latest")

// Check sync status
eth.syncing

// List connected peers
admin.peers

// Node information
admin.nodeInfo

// Network information
net.version
net.peerCount

// Exit console
exit
```

### Mining Interface Commands
```javascript
// Check mining status
miner.start(1)  // Start mining with 1 thread
miner.stop()    // Stop mining

// Set mining address
miner.setEtherbase("0xYourAddressHere")

// Check current mining address
miner.coinbase

// Check hash rate
eth.hashrate

// Check if mining
eth.mining
```

## 🔄 Manual Updates

**Important**: Q Geth does not auto-update. All updates must be performed manually for security and stability. This gives you full control over when and how updates are applied.

### Check for Updates
```bash
# Check current version
cd /opt/qgeth/Qgeth3
git log --oneline -5

# Check for remote updates
git fetch origin
git log --oneline HEAD..origin/main

# View what would be updated
git diff HEAD origin/main --name-only
```

### Apply Updates
```bash
# Stop the service
sudo systemctl stop qgeth.service

# Update code
cd /opt/qgeth/Qgeth3
sudo git pull origin main

# Make scripts executable
sudo find . -name "*.sh" -type f -exec chmod +x {} \;

# Rebuild if needed (for code changes)
cd scripts/linux
sudo ./build-linux.sh geth

# Restart service
sudo systemctl start qgeth.service

# Verify service is running
sudo systemctl status qgeth.service
```

### Manual Update Script
```bash
# Create update script
sudo tee /opt/qgeth/update-qgeth.sh > /dev/null << 'EOF'
#!/bin/bash
echo "🔄 Updating Q Geth..."

# Stop service
sudo systemctl stop qgeth.service

# Update repository
cd /opt/qgeth/Qgeth3
sudo git pull origin main

# Make scripts executable
sudo find . -name "*.sh" -type f -exec chmod +x {} \;

# Rebuild geth
cd scripts/linux
sudo ./build-linux.sh geth

# Restart service
sudo systemctl start qgeth.service

echo "✅ Q Geth updated successfully!"
sudo systemctl status qgeth.service
EOF

sudo chmod +x /opt/qgeth/update-qgeth.sh

# Usage:
sudo /opt/qgeth/update-qgeth.sh
```

## 🛠️ Configuration Files

### Important Directories
```bash
~/qgeth/Qgeth3/              # Main project directory (user installation)
~/qgeth/logs/                # All log files
~/.qcoin/testnet/            # Blockchain data directory
~/qgeth/                     # Management scripts (start, stop, status, logs)

# Service files (init system dependent):
/etc/systemd/system/         # Systemd service files
/etc/init.d/                 # SysV init scripts  
/etc/conf.d/                 # OpenRC configuration
/etc/init/                   # Upstart configuration
```

### Service Configuration

**Universal Service Management**: The bootstrap automatically creates the appropriate service configuration for your init system.

**Systemd** (Ubuntu, Fedora, Debian):
```bash
# View service configuration
sudo cat /etc/systemd/system/qgeth.service

# Edit service configuration (if needed)
sudo nano /etc/systemd/system/qgeth.service

# Reload configuration after changes
sudo systemctl daemon-reload
sudo systemctl restart qgeth.service
```

**OpenRC** (Alpine, Gentoo):
```bash
# View service configuration
sudo cat /etc/init.d/qgeth

# Edit service configuration
sudo nano /etc/init.d/qgeth
sudo nano /etc/conf.d/qgeth
```

**SysV Init** (Traditional systems):
```bash
# View service configuration
sudo cat /etc/init.d/qgeth

# Edit service configuration
sudo nano /etc/init.d/qgeth
```

**Upstart** (Older Ubuntu):
```bash
# View service configuration
sudo cat /etc/init/qgeth.conf

# Edit service configuration
sudo nano /etc/init/qgeth.conf
```

### Geth Configuration
```bash
# Current network: Q Coin Testnet (Chain ID 73235)
# Data directory: ~/.qcoin/testnet/
# Genesis file: /opt/qgeth/Qgeth3/configs/genesis_quantum_testnet.json

# View current geth arguments
ps aux | grep geth

# Check start script configuration
cat /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh
```

## 🔌 API Usage Examples

### Basic API Calls
```bash
# Get client version
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Get network ID
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' \
  http://localhost:8545

# Get gas price
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_gasPrice","params":[],"id":1}' \
  http://localhost:8545

# Get account list
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
  http://localhost:8545
```

### Mining API Calls
```bash
# Check mining status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_mining","params":[],"id":1}' \
  http://localhost:8545

# Get hash rate
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_hashrate","params":[],"id":1}' \
  http://localhost:8545

# Set mining address
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"miner_setEtherbase","params":["0xYourAddress"],"id":1}' \
  http://localhost:8545
```

### WebSocket API
```bash
# Test WebSocket connection
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     --header "Sec-WebSocket-Version: 13" \
     http://localhost:8546/
```

## 🗄️ Data Management

### Backup Blockchain Data
```bash
# Stop the node
sudo systemctl stop qgeth.service

# Create backup
sudo tar -czf qgeth-backup-$(date +%Y%m%d).tar.gz ~/.qcoin/

# Restart the node
sudo systemctl start qgeth.service

# Verify backup
ls -lh qgeth-backup-*.tar.gz
```

### Reset Blockchain Data
```bash
# Stop the node
sudo systemctl stop qgeth.service

# Remove blockchain data (keeps configuration)
rm -rf ~/.qcoin/testnet/geth/

# Restart the node (will re-sync from genesis)
sudo systemctl start qgeth.service

# Monitor re-sync progress
sudo journalctl -u qgeth.service -f
```

### View Data Directory Size
```bash
# Check blockchain data size
du -sh ~/.qcoin/testnet/

# Check log file sizes
du -sh /opt/qgeth/logs/

# Check total Q Geth installation size
du -sh /opt/qgeth/

# Check specific directories
du -sh ~/.qcoin/testnet/geth/chaindata/
du -sh ~/.qcoin/testnet/geth/nodes/
```

## 🚨 Basic Troubleshooting

### Service Not Starting
```bash
# Check service status and logs
sudo systemctl status qgeth.service
sudo journalctl -u qgeth.service --no-pager -l | tail -20

# Check if geth binary exists
ls -la /opt/qgeth/Qgeth3/geth*

# Check directory permissions
ls -la /opt/qgeth/Qgeth3/
```

### API Not Responding
```bash
# Check if geth is listening on correct ports
sudo ss -tuln | grep 8545

# Test local API connection
curl http://localhost:8545

# Check firewall rules
sudo ufw status | grep 8545

# Check service logs for errors
sudo journalctl -u qgeth.service -f
```

### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Add more swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 🔧 Recovery Procedures

### Restart Service
```bash
# Simple restart
sudo systemctl restart qgeth.service

# Full reload and restart
sudo systemctl daemon-reload
sudo systemctl restart qgeth.service

# Restart the entire system
sudo reboot
```

### Rebuild Geth Binary
```bash
# Stop service
sudo systemctl stop qgeth.service

# Manual rebuild
cd /opt/qgeth/Qgeth3/scripts/linux
sudo ./build-linux.sh geth

# Restart service
sudo systemctl start qgeth.service

# Verify rebuild
sudo systemctl status qgeth.service
```

### Complete Reinstall
```bash
# Stop and remove service
sudo systemctl stop qgeth.service
sudo systemctl disable qgeth.service
sudo rm /etc/systemd/system/qgeth.service

# Remove installation
sudo rm -rf /opt/qgeth

# Remove blockchain data (optional)
sudo rm -rf ~/.qcoin

# Run bootstrap script again
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y
```

## 🔒 Security Management

### SSH Security
```bash
# Change SSH port (optional)
sudo nano /etc/ssh/sshd_config
# Change: Port 22 → Port 2222
sudo systemctl restart ssh

# Disable password authentication (use SSH keys only)
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh

# Allow new SSH port in firewall
sudo ufw allow 2222/tcp
sudo ufw delete allow 22/tcp
```

### Firewall Management
```bash
# View current firewall rules
sudo ufw status numbered

# Allow specific IP for RPC access
sudo ufw allow from YOUR_IP_ADDRESS to any port 8545

# Block unwanted connections
sudo ufw deny from BAD_IP_ADDRESS

# Reset firewall (if needed)
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8545/tcp
sudo ufw allow 8546/tcp
sudo ufw allow 30303
sudo ufw enable
```

### System Updates
```bash
# Update system packages (not Q Geth)
sudo apt update && sudo apt upgrade -y

# Check for security updates
sudo apt list --upgradable

# Configure automatic system security updates (OS packages only)
# Note: This is for OS security patches, NOT Q Geth updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

## 📈 Performance Optimization

### Resource Monitoring
```bash
# Monitor resource usage in real-time
htop
iotop
nload

# Check system load
uptime
cat /proc/loadavg

# Monitor Q Geth process specifically
top -p $(pgrep geth)
```

### Storage Optimization
```bash
# Enable log rotation for Q Geth logs
sudo tee /etc/logrotate.d/qgeth > /dev/null << 'EOF'
/opt/qgeth/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    notifempty
    copytruncate
    su geth geth
}
EOF

# Test log rotation
sudo logrotate -d /etc/logrotate.d/qgeth

# Clean old blockchain data (if needed)
cd ~/.qcoin/testnet/geth/
sudo rm -rf chaindata/ancient/  # Removes very old blocks
```

### Network Optimization
```bash
# Increase network buffers
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Increase file descriptor limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf

# Reboot to apply limits
sudo reboot
```

## 💡 Advanced Usage

### Switch Networks
```bash
# Stop current service
sudo systemctl stop qgeth.service

# Edit start script to use mainnet
sudo nano /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh
# Or edit service to pass mainnet parameter

# Alternative: Edit service file
sudo nano /etc/systemd/system/qgeth.service
# Change: testnet → mainnet in ExecStart line

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl start qgeth.service
```

### Custom RPC Configuration
```bash
# Edit geth startup script for custom RPC modules
sudo nano /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh

# Add custom RPC modules to GETH_ARGS
# --http.api "eth,net,web3,personal,admin,txpool,miner,debug,qmpow"

# Restart service to apply changes
sudo systemctl restart qgeth.service
```

### External Mining Setup
```bash
# Q Geth node is pre-configured for external mining
# Mining interface enabled with 0 threads (no CPU mining)

# Start external quantum-miner
cd /opt/qgeth/Qgeth3
./quantum-miner -rpc-url http://localhost:8545 -address YOUR_ADDRESS -threads 4

# Or use the convenience script
./start-linux-miner.sh 4 YOUR_ADDRESS
```

## 📋 Maintenance Checklist

### Daily
- [ ] Check service status: `sudo systemctl status qgeth.service`
- [ ] Monitor resource usage: `htop`, `free -h`
- [ ] Check sync status via API
- [ ] Verify peer connections

### Weekly
- [ ] Review logs for errors: `sudo journalctl -u qgeth.service | grep -i error`
- [ ] Check disk space: `df -h`
- [ ] Verify API endpoints are responding
- [ ] Check firewall status: `sudo ufw status`

### Monthly
- [ ] Update system packages: `sudo apt update && sudo apt upgrade`
- [ ] Check for Q Geth updates: `cd /opt/qgeth/Qgeth3 && git fetch`
- [ ] Review security logs: `sudo grep UFW /var/log/syslog`
- [ ] Backup blockchain data if needed
- [ ] Test recovery procedures

### Quarterly
- [ ] Update Q Geth: Run manual update process
- [ ] Review and optimize performance settings
- [ ] Check log rotation configuration
- [ ] Review SSH and firewall security

## 🎯 Next Steps

Your Q Coin node is now operational! Consider:

1. **[Mining Setup](../mining/linux-mining.md)** - Start mining Q Coin
2. **[Troubleshooting Guide](troubleshooting-bootstrap-deployment.md)** - Fix common issues
3. **[Node Operation](../node-operation/linux-geth.md)** - Advanced node management
4. **[Development API](../development/)** - Build applications on Q Coin

Your node is running and ready for production use! 🚀 