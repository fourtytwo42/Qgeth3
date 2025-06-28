# VPS Deployment Guide

Deploy Q Coin nodes to production VPS environments with automated setup and monitoring.

## 🌐 VPS Provider Recommendations

### Recommended VPS Configurations

| Provider | vCPU | RAM | Storage | Est. Performance | Monthly Cost |
|----------|------|-----|---------|------------------|--------------|
| DigitalOcean | 2 | 2GB | 50GB | ~0.3 puzzles/sec | $12 |
| Vultr | 2 | 4GB | 80GB | ~0.4 puzzles/sec | $12 |
| Linode | 4 | 8GB | 160GB | ~0.6 puzzles/sec | $48 |
| AWS EC2 | 4 | 16GB | 100GB | ~0.8 puzzles/sec | $50 |
| **GPU VPS** | 4 | 16GB | 100GB + GPU | **2.0+ puzzles/sec** | $100+ |

### VPS Selection Criteria
- **Minimum:** 2 vCPU, 2GB RAM for basic node operation
- **Recommended:** 4 vCPU, 4GB+ RAM for optimal performance
- **Network:** Good connectivity to major internet backbones
- **Uptime:** 99.9%+ uptime guarantee
- **Support:** 24/7 support for production environments

## 🚀 One-Command VPS Setup

### Ultimate Bootstrap Installation
```bash
# Interactive mode (asks for confirmations)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash

# Non-interactive mode (perfect for automation)
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y

# Alternative with wget
wget -qO- https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y
```

**What this single command does:**
- ✅ Installs all dependencies (git, curl, golang, build tools)
- ✅ Downloads the Q Geth repository automatically
- ✅ Prepares VPS (memory checks, swap creation with 50MB tolerance, firewall)
- ✅ Builds and configures Q Geth with auto-updating service
- ✅ Sets up crash recovery and GitHub monitoring
- ✅ Handles 4095MB vs 4096MB memory requirements automatically
- ✅ **Safe to run multiple times** - detects existing installations gracefully
- ✅ **Handles partial builds** - cleans interrupted compilations automatically
- ✅ **Lock file protection** - prevents simultaneous installations

## 🛠️ Manual VPS Setup

### Initial VPS Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential dependencies
sudo apt install -y golang-go python3 python3-pip git build-essential curl wget

# Install quantum mining dependencies
pip3 install qiskit qiskit-aer numpy

# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
chmod +x scripts/deployment/*.sh scripts/linux/*.sh
```

### Low-Memory VPS Optimization
**Important for VPS with <4GB RAM:**
```bash
# Check memory
free -h

# Add swap space (required for compilation on low-memory VPS)
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h  # Should show 3.0G swap available
```

**Why swap is needed:**
- Go compilation uses significant memory during linking phase
- 1-2GB VPS runs out of memory during `quantum-geth` build
- 3GB swap provides enough virtual memory to complete compilation
- After build completes, swap usage returns to minimal levels

### Build and Deploy
```bash
# For VPS deployment (node only, optimized)
sudo ./scripts/deployment/auto-geth-service.sh

# Or manual build
./scripts/linux/build-linux.sh geth
```

## ⚙️ Auto-Updating Service Setup

### Complete Auto-Service Features

The `auto-geth-service.sh` script provides production-ready deployment:

```bash
cd Qgeth3
sudo ./scripts/deployment/auto-geth-service.sh
```

**Service Features:**
1. **VPS Preparation** - Memory checks, swap creation, dependency installation
2. **Firewall Configuration** - UFW setup with required ports:
   - Port 22 (SSH) - Remote access
   - Port 8545 (HTTP RPC) - Geth API access
   - Port 30303 (P2P TCP/UDP) - Blockchain networking
   - Port 8546 (WebSocket) - WebSocket API access
3. **Memory Optimization** - Creates swap if needed (<4GB RAM)
4. **Systemd Services** - Creates 3 production services:
   - `qgeth-node.service` - Main geth service with crash recovery
   - `qgeth-github-monitor.service` - Monitors GitHub for updates
   - `qgeth-updater.service` - Handles updates when triggered
5. **Auto-Start** - Services start automatically on boot

### Service Management Commands

```bash
# Service control
qgeth-service start         # Start both geth and GitHub monitor
qgeth-service stop          # Stop all services
qgeth-service restart       # Restart all services
qgeth-service status        # Show status of all services

# Monitoring
qgeth-service logs geth     # Follow geth logs
qgeth-service logs github   # Follow GitHub monitor logs
qgeth-service logs update   # Follow update logs
qgeth-service logs all      # Follow all logs

# Maintenance
qgeth-service update        # Trigger manual update
qgeth-service reset-crashes # Reset crash counter
qgeth-service version       # Show geth version
```

## 🔄 Auto-Update System

### GitHub Monitoring
- **Monitor Frequency:** Every 5 minutes
- **Target Repository:** `fourtytwo42/Qgeth3` main branch
- **Update Trigger:** New commits detected
- **Update Process:** Stop → Pull → Build → Restart
- **Backup System:** Keeps last 5 versions for rollback

### Crash Recovery
- **Auto-Restart:** Geth restarts automatically if it crashes
- **Retry Delay:** 5-minute delay between restart attempts
- **Infinite Retries:** Until manually stopped
- **Smart Recovery:** After 3+ crashes, triggers auto-update
- **Crash Tracking:** Logs crashes in `/opt/qgeth/logs/crash_count.txt`

## 🔥 VPS Security & Optimization

### Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw allow 22/tcp      # SSH access
sudo ufw allow 8545/tcp    # HTTP RPC API
sudo ufw allow 30303/tcp   # P2P TCP
sudo ufw allow 30303/udp   # P2P UDP
sudo ufw allow 8546/tcp    # WebSocket API
sudo ufw --force enable

# Check firewall status
sudo ufw status
```

### Performance Optimization
```bash
# Set CPU performance mode
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file limits for networking
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf

# Network optimizations
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Security Hardening
```bash
# Disable root SSH login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# Enable SSH key authentication only
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Restart SSH service
sudo systemctl restart ssh

# Install fail2ban for brute force protection
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
```

## 📊 VPS Monitoring & Maintenance

### Performance Monitoring
```bash
# Check system resources
htop
iotop
free -h
df -h

# Monitor geth sync status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545

# Check peer connections
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545
```

### Log Management
```bash
# Follow live logs
qgeth-service logs geth | tail -f

# Check log files directly
tail -f /opt/qgeth/logs/geth-output.log
tail -f /opt/qgeth/logs/github-monitor.log

# Check disk usage of logs
du -h /opt/qgeth/logs/
```

### Backup Management
```bash
# View available backups
ls -la /opt/qgeth/backup/

# Manual backup creation
sudo systemctl stop qgeth-node.service
sudo cp -r /opt/qgeth/Qgeth3 /opt/qgeth/backup/manual-backup-$(date +%Y%m%d_%H%M%S)
sudo systemctl start qgeth-node.service
```

## 🏗️ Production Configuration

### File Locations
```
/opt/qgeth/
├── Qgeth3/                    # Main project directory
├── logs/
│   ├── geth-runner.log        # Geth service logs
│   ├── geth-output.log        # Geth stdout
│   ├── geth-error.log         # Geth stderr
│   ├── github-monitor.log     # GitHub monitoring
│   ├── update.log             # Update process logs
│   └── crash_count.txt        # Crash counter
├── backup/
│   ├── Qgeth3_20240101_120000/
│   └── ...                    # Auto-backups (keeps last 5)
└── scripts/
    ├── github-monitor.sh      # GitHub monitoring script
    ├── update-geth.sh         # Update handler
    └── run-geth.sh            # Geth runner with crash recovery
```

### Default Service Configuration
- **Network:** `testnet` (Q Coin Testnet)
- **API Access:** `--http.corsdomain "*" --http.api "eth,net,web3,personal,txpool"`
- **GitHub Check:** Every 5 minutes
- **Crash Retry:** Every 5 minutes
- **Log Rotation:** Automatic
- **Backup Retention:** Last 5 versions

### Environment Variables
```bash
# VPS-specific environment (set by auto-service)
QGETH_BUILD_TEMP="/opt/qgeth/build-temp"
GETH_NETWORK="testnet"
GETH_ARGS="--http.corsdomain \"*\" --http.api \"eth,net,web3,personal,txpool\""
```

## 🚨 VPS Troubleshooting

### Common VPS Issues

#### Service Not Starting
```bash
# Check service status
qgeth-service status
sudo systemctl status qgeth-node.service

# Check logs for errors
qgeth-service logs geth
sudo journalctl -u qgeth-node.service -f
```

#### Memory Issues
```bash
# Check memory usage
free -h

# Check swap usage
swapon --show

# Add more swap if needed
sudo fallocate -l 2G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
```

#### Network Connectivity
```bash
# Test internet connectivity
ping -c 4 8.8.8.8

# Test GitHub access
curl -I https://github.com

# Check if geth API is responding
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545
```

#### Build Failures
```bash
# Check build logs
cat /opt/qgeth/logs/update.log

# Manual rebuild
cd /opt/qgeth/Qgeth3
sudo ./scripts/linux/build-linux.sh geth --clean

# Check disk space
df -h
```

### Recovery Procedures

#### Restore from Backup
```bash
# Stop services
qgeth-service stop

# List available backups
ls -la /opt/qgeth/backup/

# Restore from backup
sudo rm -rf /opt/qgeth/Qgeth3
sudo cp -r /opt/qgeth/backup/Qgeth3_YYYYMMDD_HHMMSS /opt/qgeth/Qgeth3

# Restart services
qgeth-service start
```

#### Clean Reinstall
```bash
# Stop and disable services
qgeth-service stop
sudo systemctl disable qgeth-node.service qgeth-github-monitor.service qgeth-updater.service

# Remove installation
sudo rm -rf /opt/qgeth

# Run bootstrap again
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y
```

## 🎯 VPS Best Practices

### Regular Maintenance
- **Weekly:** Check logs and system resources
- **Monthly:** Review backup retention and cleanup old logs
- **Updates:** Let auto-update handle repository updates
- **Security:** Keep VPS system packages updated

### Monitoring Recommendations
- Set up external monitoring for HTTP API availability
- Monitor disk space and set alerts for >80% usage
- Track CPU and memory usage trends
- Set up email notifications for service failures

### Cost Optimization
- Use VPS pricing calculators to find best value
- Consider reserved instances for long-term deployments
- Monitor bandwidth usage if provider charges for it
- Use auto-scaling if provider supports it

## 🔗 Next Steps

After VPS deployment:

1. **[Auto-Service Guide](auto-service.md)** - Detailed service management
2. **[Mining Guide](mining.md)** - Set up remote mining
3. **[Troubleshooting](troubleshooting.md)** - Fix common issues
4. **[Advanced Configuration](advanced-configuration.md)** - Optimize performance

## ✅ VPS Deployment Checklist

### Pre-Deployment
- [ ] VPS provider selected with adequate resources
- [ ] SSH access configured with key authentication
- [ ] Firewall ports planned (22, 8545, 30303, 8546)
- [ ] DNS/domain setup (if using custom domain)

### Deployment
- [ ] Bootstrap script executed successfully
- [ ] Auto-service installed and running
- [ ] Firewall configured and enabled
- [ ] Services start automatically on boot
- [ ] GitHub monitoring active

### Post-Deployment
- [ ] Node syncing with network
- [ ] API endpoints responding correctly
- [ ] Auto-update system tested
- [ ] Monitoring and alerting configured
- [ ] Backup system verified

### Production Ready ✅
Your Q Coin VPS node is now running in production with automatic updates and crash recovery! 