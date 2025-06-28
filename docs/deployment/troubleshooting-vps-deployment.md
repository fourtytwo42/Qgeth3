# VPS Deployment Troubleshooting

Solutions for production VPS deployment issues with Q Coin auto-service system.

## 🔧 Quick VPS Diagnostics

### Bootstrap Status Check
```bash
# Check if auto-service is installed
which qgeth-service
/usr/local/bin/qgeth-service status

# Check systemd services
sudo systemctl status qgeth-node.service
sudo systemctl status qgeth-github-monitor.service
sudo systemctl status qgeth-updater.service

# Check installation directory
ls -la /opt/qgeth/
ps aux | grep -E "(geth|quantum)"
```

## 🚀 Bootstrap Script Issues

### Download/Access Issues
```bash
# Symptoms: "bash: line 1: 404: command not found"
# Solution: Check URL and network connectivity

# Verify script URL is accessible
curl -I https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh

# If 404 error, check repository status
curl -I https://github.com/fourtytwo42/Qgeth3

# Test network connectivity
ping -c 4 github.com
ping -c 4 8.8.8.8

# If blocked, try alternative download
wget https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh
chmod +x bootstrap-qgeth.sh
sudo ./bootstrap-qgeth.sh -y
```

### Permission Denied Errors
```bash
# Symptoms: "Permission denied" during bootstrap
# Solution: Ensure running as root or with sudo

# Check current user
whoami
id

# Run with sudo
sudo bash <(curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh) -y

# Alternative: Download and run manually
curl -O https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh
sudo bash bootstrap-qgeth.sh -y
```

### Lock File Issues
```bash
# Symptoms: "Auto-service installation already in progress"
# Solution: Remove stale lock file

# Check for lock file
ls -la /tmp/qgeth-auto-service.lock

# Remove lock file
sudo rm -f /tmp/qgeth-auto-service.lock

# Check for running bootstrap processes
ps aux | grep bootstrap
sudo pkill -f bootstrap-qgeth.sh

# Try bootstrap again
sudo bash <(curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh) -y
```

### Non-Interactive Mode Issues
```bash
# Symptoms: Script hangs on prompts despite -y flag
# Solution: Ensure latest version and proper environment

# Check if script supports -y flag
curl -s https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | grep -A 10 -B 10 "\-y"

# Set non-interactive environment
export DEBIAN_FRONTEND=noninteractive
sudo -E bash <(curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh) -y

# Alternative: Download latest version
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh -o bootstrap.sh
sudo bash bootstrap.sh -y
```

## 🔧 Build Issues on VPS

### Memory Issues
```bash
# Check available memory and swap
free -h
swapon --show

# If insufficient memory, add swap
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Retry build with more memory
cd /opt/qgeth/Qgeth3
sudo ./scripts/linux/build-linux.sh geth --clean
```

### Go Version Issues
```bash
# Check Go version (need 1.21+)
go version

# Install latest Go if needed
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee -a /etc/profile
source /etc/profile

# Verify Go installation
go version
which go
```

### Permission/Ownership Issues
```bash
# Check directory ownership
ls -la /opt/qgeth/

# Fix ownership if needed
sudo chown -R qgeth:qgeth /opt/qgeth

# If qgeth user doesn't exist
sudo useradd -r -s /bin/bash qgeth
sudo chown -R qgeth:qgeth /opt/qgeth

# Check build permissions
sudo -u qgeth ls -la /opt/qgeth/Qgeth3/
sudo -u qgeth touch /opt/qgeth/test.txt
```

### Build Tool Dependencies
```bash
# Install required build tools
sudo apt update
sudo apt install -y build-essential git pkg-config

# For CentOS/RHEL
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git pkgconfig

# Verify tools are installed
gcc --version
make --version
git --version
```

## 🔄 Service Management Issues

### Services Not Starting
```bash
# Check service status
sudo systemctl status qgeth-node.service
sudo systemctl status qgeth-github-monitor.service
sudo systemctl status qgeth-updater.service

# Check for failed services
sudo systemctl --failed

# View service logs
sudo journalctl -u qgeth-node.service -f
sudo journalctl -u qgeth-github-monitor.service -f

# Restart services
sudo systemctl restart qgeth-node.service
sudo systemctl restart qgeth-github-monitor.service
```

### qgeth-service Command Not Found
```bash
# Check if command exists
which qgeth-service
ls -la /usr/local/bin/qgeth-service

# If missing, recreate symlink
sudo ln -sf /opt/qgeth/Qgeth3/scripts/deployment/auto-geth-service.sh /usr/local/bin/qgeth-service
sudo chmod +x /usr/local/bin/qgeth-service

# Reload shell
source ~/.bashrc

# Test command
qgeth-service status
```

### Systemd Service File Issues
```bash
# Check if service files exist
ls -la /etc/systemd/system/qgeth*.service

# View service file contents
sudo cat /etc/systemd/system/qgeth-node.service

# If corrupted, regenerate services
cd /opt/qgeth/Qgeth3
sudo ./scripts/deployment/auto-geth-service.sh install

# Reload systemd daemon
sudo systemctl daemon-reload
sudo systemctl enable qgeth-node.service
sudo systemctl start qgeth-node.service
```

## 🌐 Network and Firewall Issues

### UFW Firewall Configuration
```bash
# Check firewall status
sudo ufw status verbose

# If firewall is blocking, configure rules
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8545/tcp  # HTTP RPC
sudo ufw allow 30303/tcp # P2P TCP
sudo ufw allow 30303/udp # P2P UDP

# Enable firewall if not active
sudo ufw --force enable

# Test connectivity
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545
```

### SSH Access Issues
```bash
# Check SSH service
sudo systemctl status ssh

# Check SSH configuration
sudo sshd -T | grep -E "(Port|PermitRootLogin|PasswordAuthentication)"

# If SSH locked out, use VPS console
# Enable password auth temporarily:
sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Fix SSH keys
cat ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### Port Conflicts
```bash
# Check what's using required ports
sudo lsof -i :8545
sudo lsof -i :30303
sudo netstat -tulpn | grep -E "(8545|30303)"

# Kill conflicting processes
sudo pkill -f geth
sudo pkill -f quantum-miner

# Start services with different ports if needed
sudo systemctl edit qgeth-node.service
# Add:
# [Service]
# Environment="GETH_HTTP_PORT=8546"
# Environment="GETH_P2P_PORT=30304"
```

## 📊 Auto-Update System Issues

### GitHub Monitor Not Working
```bash
# Check monitor service
sudo systemctl status qgeth-github-monitor.service
qgeth-service logs github

# Test GitHub connectivity
ping -c 4 github.com
curl -I https://api.github.com/repos/fourtytwo42/Qgeth3/commits/main

# Check Git configuration
cd /opt/qgeth/Qgeth3
git status
git remote -v
git log --oneline -5

# Manual update test
cd /opt/qgeth/Qgeth3
sudo -u qgeth git pull origin main
```

### Update Process Failing
```bash
# Check update logs
qgeth-service logs update
sudo journalctl -u qgeth-updater.service -f

# Test manual update
qgeth-service update

# Check for update script issues
ls -la /opt/qgeth/Qgeth3/scripts/deployment/
sudo -u qgeth /opt/qgeth/Qgeth3/scripts/deployment/auto-geth-service.sh update

# If Git pull fails, check permissions
sudo chown -R qgeth:qgeth /opt/qgeth/Qgeth3/.git
```

### Backup System Issues
```bash
# Check backup directory
ls -la /opt/qgeth/backup/

# Check backup space
df -h /opt/qgeth/backup/

# Manual backup test
qgeth-service backup

# Clean old backups if space is low
sudo find /opt/qgeth/backup/ -name "Qgeth3_*" -mtime +7 -exec rm -rf {} \;

# Check backup logs
grep -i backup /opt/qgeth/logs/*.log
```

## 💾 Storage and Data Issues

### Disk Space Problems
```bash
# Check disk usage
df -h
du -sh /opt/qgeth/
du -sh /root/.qcoin/

# Clean up space
# 1. Remove old backups
sudo rm -rf /opt/qgeth/backup/Qgeth3_OLD*

# 2. Clean package cache
sudo apt autoremove -y
sudo apt autoclean

# 3. Clean blockchain data (if safe to do)
sudo systemctl stop qgeth-node.service
sudo rm -rf /root/.qcoin/*/geth/chaindata
sudo systemctl start qgeth-node.service

# 4. Rotate logs
sudo logrotate -f /etc/logrotate.conf
```

### Database Corruption
```bash
# Symptoms: "database corruption", "bad block", sync issues
# Solution: Reset blockchain database

# Stop services
qgeth-service stop

# Backup current data
sudo cp -r /root/.qcoin/testnet /root/.qcoin/testnet.backup

# Remove corrupted data
sudo rm -rf /root/.qcoin/*/geth/chaindata

# Restart services (will re-sync from genesis)
qgeth-service start

# Monitor sync progress
qgeth-service logs geth | grep -E "(sync|block|chain)"
```

### Log File Issues
```bash
# Check log file sizes
du -sh /opt/qgeth/logs/*
ls -la /opt/qgeth/logs/

# Rotate large logs
sudo logrotate -f /etc/logrotate.d/qgeth

# If logrotate config missing, create it
sudo tee /etc/logrotate.d/qgeth > /dev/null <<EOF
/opt/qgeth/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    notifempty
    copytruncate
    su qgeth qgeth
}
EOF

# Test log rotation
sudo logrotate -d /etc/logrotate.d/qgeth
```

## 🔐 Security Issues

### User and Permission Problems
```bash
# Check qgeth user exists
id qgeth
sudo passwd -S qgeth

# Create qgeth user if missing
sudo useradd -r -m -s /bin/bash qgeth
sudo usermod -aG sudo qgeth  # If sudo access needed

# Fix ownership recursively
sudo chown -R qgeth:qgeth /opt/qgeth
sudo chown -R qgeth:qgeth /root/.qcoin  # Or move to /home/qgeth/.qcoin

# Check service file user
grep -E "User|Group" /etc/systemd/system/qgeth-node.service
```

### SSH Security
```bash
# Harden SSH configuration
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Recommended SSH settings
sudo tee -a /etc/ssh/sshd_config > /dev/null <<EOF
# Q Geth VPS Security Settings
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
EOF

# Restart SSH
sudo systemctl restart ssh

# Test SSH access before closing current session
ssh -o ConnectTimeout=5 user@your-vps-ip
```

### Firewall Security
```bash
# Check current firewall rules
sudo ufw status numbered

# Remove unnecessary rules
sudo ufw delete <rule_number>

# Set strict default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Only allow necessary services
sudo ufw allow ssh
sudo ufw allow 8545/tcp  # Only if external RPC access needed
sudo ufw limit ssh       # Rate limit SSH connections
```

## 🚨 Emergency Recovery

### Complete VPS Reset
```bash
# Stop all services
sudo systemctl stop qgeth-node.service qgeth-github-monitor.service qgeth-updater.service

# Disable services
sudo systemctl disable qgeth-node.service qgeth-github-monitor.service qgeth-updater.service

# Remove installation
sudo rm -rf /opt/qgeth
sudo rm -f /usr/local/bin/qgeth-service
sudo rm -f /etc/systemd/system/qgeth*.service

# Remove blockchain data
sudo rm -rf /root/.qcoin

# Fresh installation
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y
```

### Service Recovery
```bash
# Reset systemd state
sudo systemctl daemon-reload
sudo systemctl reset-failed

# Recreate services
cd /opt/qgeth/Qgeth3
sudo ./scripts/deployment/auto-geth-service.sh install

# Start services one by one
sudo systemctl start qgeth-node.service
sleep 10
sudo systemctl start qgeth-github-monitor.service
sleep 5
sudo systemctl start qgeth-updater.service

# Check all services
qgeth-service status
```

### Backup Recovery
```bash
# List available backups
ls -la /opt/qgeth/backup/

# Stop services
qgeth-service stop

# Restore from backup
sudo rm -rf /opt/qgeth/Qgeth3
sudo cp -r /opt/qgeth/backup/Qgeth3_YYYYMMDD_HHMMSS /opt/qgeth/Qgeth3
sudo chown -R qgeth:qgeth /opt/qgeth/Qgeth3

# Restart services
qgeth-service start
qgeth-service status
```

## 📊 Monitoring and Maintenance

### System Monitoring
```bash
# Check system resources
htop
iotop -ao
df -h
free -h

# Monitor services
watch -n 5 'systemctl status qgeth-node.service --no-pager -l'

# Network monitoring
netstat -tulpn | grep -E "(8545|30303)"
ss -tulpn | grep -E "(8545|30303)"

# Check blockchain sync status
qgeth-service logs geth | tail -20
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545
```

### Maintenance Scripts
```bash
# Create automated maintenance script
sudo tee /opt/qgeth/maintenance.sh > /dev/null <<'EOF'
#!/bin/bash
# Q Geth VPS Maintenance Script

echo "=== Q Geth VPS Maintenance $(date) ==="

# Check disk space
df -h | grep -E "(Filesystem|/dev/)"

# Check service status
systemctl is-active qgeth-node.service
systemctl is-active qgeth-github-monitor.service

# Clean old logs
find /opt/qgeth/logs/ -name "*.log" -mtime +30 -delete

# Update system packages
apt update && apt upgrade -y

echo "Maintenance completed at $(date)"
EOF

sudo chmod +x /opt/qgeth/maintenance.sh

# Add to crontab for weekly runs
(sudo crontab -l 2>/dev/null; echo "0 2 * * 0 /opt/qgeth/maintenance.sh >> /opt/qgeth/logs/maintenance.log 2>&1") | sudo crontab -
```

## 📚 Getting VPS Help

### Information to Collect
When reporting VPS deployment issues:

1. **VPS Details**: Provider, OS version, specs
2. **Service Status**: `qgeth-service status`
3. **System Resources**: `free -h && df -h`
4. **Service Logs**: `qgeth-service logs all | tail -100`
5. **Network Configuration**: `sudo ufw status && netstat -tulpn`
6. **Error Messages**: Full command output

### VPS Diagnostic Script
```bash
# Comprehensive VPS diagnostic collection
sudo tee /tmp/vps-diag.sh > /dev/null <<'EOF'
#!/bin/bash
echo "=== Q Geth VPS Diagnostics ===" > vps-diagnostics.txt
echo "Date: $(date)" >> vps-diagnostics.txt
echo "Hostname: $(hostname)" >> vps-diagnostics.txt
echo "" >> vps-diagnostics.txt

echo "=== System Info ===" >> vps-diagnostics.txt
uname -a >> vps-diagnostics.txt
lsb_release -a >> vps-diagnostics.txt 2>&1
echo "" >> vps-diagnostics.txt

echo "=== Resources ===" >> vps-diagnostics.txt
free -h >> vps-diagnostics.txt
df -h >> vps-diagnostics.txt
echo "" >> vps-diagnostics.txt

echo "=== Services ===" >> vps-diagnostics.txt
systemctl status qgeth-node.service --no-pager >> vps-diagnostics.txt 2>&1
systemctl status qgeth-github-monitor.service --no-pager >> vps-diagnostics.txt 2>&1
echo "" >> vps-diagnostics.txt

echo "=== Network ===" >> vps-diagnostics.txt
ufw status >> vps-diagnostics.txt 2>&1
netstat -tulpn | grep -E "(8545|30303)" >> vps-diagnostics.txt 2>&1
echo "" >> vps-diagnostics.txt

echo "=== Installation ===" >> vps-diagnostics.txt
ls -la /opt/qgeth/ >> vps-diagnostics.txt 2>&1
which qgeth-service >> vps-diagnostics.txt 2>&1

cat vps-diagnostics.txt
EOF

chmod +x /tmp/vps-diag.sh
/tmp/vps-diag.sh
```

## ✅ VPS Deployment Checklist

### Pre-Deployment
- [ ] VPS meets minimum requirements (2GB RAM, 20GB disk)
- [ ] SSH access configured with key authentication
- [ ] Root or sudo access available
- [ ] Network connectivity working
- [ ] Firewall allows SSH

### Installation
- [ ] Bootstrap script runs without errors
- [ ] All services created and enabled
- [ ] qgeth-service command available
- [ ] Build completes successfully
- [ ] Services start automatically

### Post-Deployment
- [ ] Services running and stable
- [ ] Blockchain syncing properly
- [ ] GitHub monitoring active
- [ ] Auto-update system functional
- [ ] Firewall properly configured
- [ ] Backup system working

**VPS deployment issues are usually related to permissions, memory, or network configuration!** 