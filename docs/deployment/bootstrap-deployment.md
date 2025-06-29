# Q Geth Bootstrap Deployment Guide

Complete guide for operating and managing your Q Geth node after installation by the bootstrap script.

## 🎯 Quick Reference

After running the bootstrap script, your VPS has:
- **Q Geth Node**: Running on testnet with single streamlined service
- **System Service**: `qgeth.service` (simplified architecture)  
- **API Endpoints**: HTTP RPC (8545), WebSocket (8546)
- **Network**: P2P networking on port 30303
- **Manual Updates**: Use git pull for updates when needed

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

### One-Command Installation
```bash
# Non-interactive installation (recommended for automation)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y

# Interactive installation (prompts for confirmation)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash
```

### What Bootstrap Sets Up
- **Dependencies**: Go, build tools, git, firewall
- **Memory**: Creates swap file if needed (minimum 4GB total memory)
- **Project**: Clones repository to `/opt/qgeth/Qgeth3/`
- **Build**: Compiles Q Geth with automated error recovery
- **Service**: Creates and starts `qgeth.service`
- **Firewall**: Configures UFW with required ports
- **User**: Creates `geth` user for service execution

## 🔧 Service Management

### Check Service Status
```bash
# Check Q Geth service status
sudo systemctl status qgeth.service

# Quick status check
sudo systemctl is-active qgeth.service

# Check if service is enabled
sudo systemctl is-enabled qgeth.service
```

### Start/Stop/Restart Service
```bash
# Control the blockchain node
sudo systemctl start qgeth.service
sudo systemctl stop qgeth.service
sudo systemctl restart qgeth.service

# Check service after restart
sudo systemctl status qgeth.service
```

### Enable/Disable Auto-Start
```bash
# Enable service to start on boot (default)
sudo systemctl enable qgeth.service

# Disable auto-start
sudo systemctl disable qgeth.service

# Check if enabled
sudo systemctl is-enabled qgeth.service
```

## 📊 Monitoring Your Node

### View Live Logs
```bash
# Follow geth service logs (systemd journal)
sudo journalctl -u qgeth.service -f

# Follow geth log file
sudo tail -f /opt/qgeth/logs/geth.log

# View recent service activity
sudo journalctl -u qgeth.service --no-pager -l | tail -50

# View logs from specific time
sudo journalctl -u qgeth.service --since "1 hour ago"
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

### Update Automation Script
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
/opt/qgeth/Qgeth3/           # Main project directory
/opt/qgeth/logs/             # All log files
~/.qcoin/testnet/            # Blockchain data directory
/etc/systemd/system/         # Service configuration files
```

### Service Configuration
```bash
# View geth service configuration
sudo cat /etc/systemd/system/qgeth.service

# Edit service configuration (if needed)
sudo nano /etc/systemd/system/qgeth.service

# Reload configuration after changes
sudo systemctl daemon-reload
sudo systemctl restart qgeth.service
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

# Configure automatic security updates
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