# Q Geth VPS Operation Guide

Complete guide for operating and managing your Q Geth node after installation by the bootstrap script.

## 🎯 Quick Reference

After running the bootstrap script, your VPS has:
- **Q Geth Node**: Running on testnet, auto-updating from GitHub
- **System Services**: `qgeth-node.service` and `qgeth-monitor.service`
- **API Endpoints**: HTTP RPC (8545), WebSocket (8546)
- **Network**: P2P networking on port 30303
- **Auto-Updates**: Monitors GitHub every 5 minutes

## 🔧 Service Management

### Check Service Status
```bash
# Check all Q Geth services
sudo systemctl status qgeth-node.service
sudo systemctl status qgeth-monitor.service

# Quick status overview
sudo systemctl is-active qgeth-node.service qgeth-monitor.service
```

### Start/Stop/Restart Services
```bash
# Control the blockchain node
sudo systemctl start qgeth-node.service
sudo systemctl stop qgeth-node.service
sudo systemctl restart qgeth-node.service

# Control the auto-updater
sudo systemctl start qgeth-monitor.service
sudo systemctl stop qgeth-monitor.service
sudo systemctl restart qgeth-monitor.service

# Restart both services
sudo systemctl restart qgeth-node.service qgeth-monitor.service
```

### Enable/Disable Auto-Start
```bash
# Enable services to start on boot (default)
sudo systemctl enable qgeth-node.service qgeth-monitor.service

# Disable auto-start
sudo systemctl disable qgeth-node.service qgeth-monitor.service
```

## 📊 Monitoring Your Node

### View Live Logs
```bash
# Follow geth node logs
sudo tail -f /opt/qgeth/logs/geth-node.log

# Follow auto-updater logs
sudo tail -f /opt/qgeth/logs/github-monitor.log

# View systemd service logs
sudo journalctl -u qgeth-node.service -f
sudo journalctl -u qgeth-monitor.service -f
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
```

## 🖥️ Geth Console Access

### Attach to Running Node
```bash
# Connect via IPC (recommended)
cd /opt/qgeth/Qgeth3
./geth attach ipc:~/.qcoin/testnet/geth.ipc

# Connect via HTTP (if remote)
./geth attach http://localhost:8545
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
```

## 🔄 Auto-Update System

### Monitor Auto-Updates
```bash
# Check when auto-updater last ran
sudo systemctl show qgeth-monitor.service --property=ActiveEnterTimestamp

# View auto-update activity
sudo tail -20 /opt/qgeth/logs/github-monitor.log

# Check current git commit
cd /opt/qgeth/Qgeth3
git log --oneline -5
```

### Manual Update Trigger
```bash
# Force immediate update check
sudo systemctl restart qgeth-monitor.service

# View update progress
sudo tail -f /opt/qgeth/logs/github-monitor.log
```

### Auto-Update Configuration
The auto-updater:
- Checks GitHub every 5 minutes
- Pulls new commits automatically
- Rebuilds geth binary if code changed
- Restarts services after successful updates
- Logs all activity to `/opt/qgeth/logs/github-monitor.log`

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
sudo cat /etc/systemd/system/qgeth-node.service

# View auto-updater configuration
sudo cat /etc/systemd/system/qgeth-monitor.service

# Reload configuration after changes
sudo systemctl daemon-reload
```

### Geth Configuration
```bash
# Current network: Q Coin Testnet (Chain ID 73235)
# Data directory: ~/.qcoin/testnet/
# Genesis file: /opt/qgeth/Qgeth3/configs/genesis_quantum_testnet.json

# View current geth arguments
ps aux | grep geth
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
sudo systemctl stop qgeth-node.service

# Create backup
sudo tar -czf qgeth-backup-$(date +%Y%m%d).tar.gz ~/.qcoin/

# Restart the node
sudo systemctl start qgeth-node.service
```

### Reset Blockchain Data
```bash
# Stop the node
sudo systemctl stop qgeth-node.service

# Remove blockchain data (keeps configuration)
rm -rf ~/.qcoin/testnet/geth/

# Restart the node (will re-sync from genesis)
sudo systemctl start qgeth-node.service
```

### View Data Directory Size
```bash
# Check blockchain data size
du -sh ~/.qcoin/testnet/

# Check log file sizes
du -sh /opt/qgeth/logs/

# Check total Q Geth installation size
du -sh /opt/qgeth/
```

## 🚨 Troubleshooting

### Common Issues

#### Node Not Starting
```bash
# Check service status and logs
sudo systemctl status qgeth-node.service
sudo journalctl -u qgeth-node.service --no-pager

# Check if geth binary exists
ls -la /opt/qgeth/Qgeth3/geth*

# Check directory permissions
ls -la /opt/qgeth/Qgeth3/
```

#### API Not Responding
```bash
# Check if geth is listening on correct ports
sudo ss -tuln | grep 8545

# Test local API connection
curl http://localhost:8545

# Check firewall rules
sudo ufw status | grep 8545
```

#### Out of Sync
```bash
# Check sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545

# Check peer count
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545

# Restart node to reconnect to peers
sudo systemctl restart qgeth-node.service
```

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Add more swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Recovery Procedures

#### Restart Everything
```bash
# Restart all Q Geth services
sudo systemctl restart qgeth-node.service qgeth-monitor.service

# Restart the entire system
sudo reboot
```

#### Rebuild Geth Binary
```bash
# Stop services
sudo systemctl stop qgeth-node.service qgeth-monitor.service

# Manual rebuild
cd /opt/qgeth/Qgeth3/scripts/linux
sudo ./build-linux.sh geth

# Restart services
sudo systemctl start qgeth-node.service qgeth-monitor.service
```

#### Complete Reinstall
```bash
# Stop and remove services
sudo systemctl stop qgeth-node.service qgeth-monitor.service
sudo systemctl disable qgeth-node.service qgeth-monitor.service
sudo rm /etc/systemd/system/qgeth-*.service

# Remove installation
sudo rm -rf /opt/qgeth

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
```

### Firewall Management
```bash
# View current firewall rules
sudo ufw status numbered

# Allow additional IPs (if needed)
sudo ufw allow from YOUR_IP_ADDRESS to any port 8545

# Block unwanted connections
sudo ufw deny from BAD_IP_ADDRESS
```

### System Updates
```bash
# Update system packages (not Q Geth)
sudo apt update && sudo apt upgrade -y

# Check for security updates
sudo unattended-upgrades --dry-run

# Configure automatic security updates
sudo dpkg-reconfigure -plow unattended-upgrades
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
```

### Storage Optimization
```bash
# Enable log rotation for Q Geth logs
sudo nano /etc/logrotate.d/qgeth
# Add log rotation configuration

# Clean old blockchain data (if needed)
cd ~/.qcoin/testnet/geth/
rm -rf chaindata/ancient/  # Removes very old blocks
```

## 💡 Advanced Usage

### Switch Networks
```bash
# Stop current service
sudo systemctl stop qgeth-node.service

# Edit service to use mainnet
sudo nano /etc/systemd/system/qgeth-node.service
# Change: testnet → mainnet

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl start qgeth-node.service
```

### Custom RPC Configuration
```bash
# Edit geth startup script
sudo nano /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh

# Add custom RPC modules
# --http.api "eth,net,web3,personal,admin,txpool,miner,debug"

# Restart service
sudo systemctl restart qgeth-node.service
```

### Mining Configuration
```bash
# Enable mining in geth service
sudo nano /etc/systemd/system/qgeth-node.service
# Add: --mining flag to start-geth.sh command

# Or use external quantum-miner
cd /opt/qgeth/Qgeth3
./quantum-miner -rpc-url http://localhost:8545 -address YOUR_ADDRESS
```

## 📋 Maintenance Checklist

### Daily
- [ ] Check service status: `sudo systemctl status qgeth-node.service`
- [ ] Monitor resource usage: `htop`, `free -h`
- [ ] Check sync status via API

### Weekly
- [ ] Review logs for errors: `sudo tail -100 /opt/qgeth/logs/geth-node.log`
- [ ] Check disk space: `df -h`
- [ ] Verify auto-updater is working: `sudo tail -20 /opt/qgeth/logs/github-monitor.log`

### Monthly
- [ ] Update system packages: `sudo apt update && sudo apt upgrade`
- [ ] Review firewall logs: `sudo grep UFW /var/log/syslog`
- [ ] Backup blockchain data if needed
- [ ] Check for security updates

## 🎯 Next Steps

Your Q Coin VPS node is now operational! Consider:

1. **[Mining Setup](../mining/linux-mining.md)** - Start mining Q Coin
2. **[Troubleshooting Guide](troubleshooting-vps-deployment.md)** - Fix common issues
3. **[API Reference](../development/api-reference.md)** - Learn the full API
4. **[Network Monitoring](monitoring.md)** - Set up advanced monitoring

Your node is now running, auto-updating, and ready for production use! 🚀 