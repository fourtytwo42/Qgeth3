# Bootstrap Deployment Troubleshooting

Solutions for bootstrap deployment issues with Q Coin quantum blockchain across all environments.

## 🔧 Quick Bootstrap Diagnostics

### Installation Status Check
```bash
# Check if geth is installed and running
ps aux | grep -E "(geth|quantum)"
pgrep -f geth

# Check installation directory
ls -la /opt/qgeth/
ls -la /opt/qgeth/Qgeth3/releases/

# Check if geth binary exists
ls -la /opt/qgeth/Qgeth3/geth.bin

# Check blockchain data
ls -la ~/.qcoin/
```

### Service Status Check
```bash
# Check systemd services
sudo systemctl status qgeth.service

# Check running processes
ps aux | grep geth
netstat -tulpn | grep -E "(8545|8546|30303)"

# Check firewall status
sudo ufw status verbose
```

## 🚀 Bootstrap Script Issues

### Missing Prerequisites (Debian/Ubuntu)
```bash
# Symptoms: "curl: command not found" or "sudo: command not found"
# Solution: Install basic packages first

# For Debian/Ubuntu minimal installations
apt update
apt install -y curl sudo

# For very minimal installations, you may also need:
apt install -y wget ca-certificates

# Then run the bootstrap script
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y

# Alternative: Download and run manually
wget https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh
bash bootstrap-qgeth.sh -y
```

### Go Version Conflicts (Debian/Ubuntu)
```bash
# Symptoms: Build failures with "missing go.sum entry" or "package slices is not in GOROOT"
# This usually indicates Go version conflicts between package manager and manual installations

# The bootstrap script now handles this automatically, but for manual fixes:

# 1. Check current Go version and location
go version
which go

# 2. If using old version (< 1.21), remove package manager Go
sudo apt remove --purge golang-go golang-1.* -y
sudo apt autoremove -y

# 3. Install latest Go manually
sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.6.linux-amd64.tar.gz

# 4. Fix PATH priorities
export PATH="/usr/local/go/bin:$PATH"
echo 'export PATH="/usr/local/go/bin:$PATH"' >> ~/.bashrc
sudo bash -c 'echo "PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> /etc/environment'

# 5. Clean module cache
go clean -cache -modcache -testcache
rm -rf ~/.cache/go-build

# 6. Verify correct version is active
go version  # Should show go1.21.6 or later
which go    # Should show /usr/local/go/bin/go

# 7. Retry bootstrap
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y
```

### Download/Access Issues
```bash
# Symptoms: "bash: line 1: 404: command not found"
# Solution: Check URL and network connectivity

# Verify script URL is accessible
curl -I https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh

# If 404 error, check repository status
curl -I https://github.com/fourtytwo42/Qgeth3

# Test network connectivity
ping -c 4 github.com
ping -c 4 8.8.8.8

# If blocked, try alternative download
wget https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh
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
sudo bash <(curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh) -y

# Alternative: Download and run manually
curl -O https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh
sudo bash bootstrap-qgeth.sh -y
```

### Lock File Issues
```bash
# Symptoms: "Installation already in progress"
# Solution: Remove stale lock file

# Check for lock file
ls -la /tmp/qgeth-bootstrap.lock

# Remove lock file
sudo rm -f /tmp/qgeth-bootstrap.lock

# Check for running bootstrap processes
ps aux | grep bootstrap
sudo pkill -f bootstrap-qgeth.sh

# Try bootstrap again
sudo bash <(curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh) -y
```

### Non-Interactive Mode Issues
```bash
# Symptoms: Script hangs on prompts despite -y flag
# Solution: Ensure latest version and proper environment

# Check if script supports -y flag
curl -s https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | grep -A 10 -B 10 "\-y"

# Set non-interactive environment
export DEBIAN_FRONTEND=noninteractive
sudo -E bash <(curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh) -y

# Alternative: Download latest version
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh -o bootstrap.sh
sudo bash bootstrap.sh -y
```

## 🔧 Build Issues

### Memory Issues During Build
```bash
# Check available memory and swap
free -h
swapon --show

# If insufficient memory, add swap (bootstrap should do this automatically)
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

# Install latest Go if bootstrap didn't work
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

# Fix ownership if needed (bootstrap creates 'geth' user)
sudo chown -R geth:geth /opt/qgeth

# If geth user doesn't exist
sudo useradd -r -s /bin/bash geth
sudo chown -R geth:geth /opt/qgeth

# Check build permissions
sudo -u geth ls -la /opt/qgeth/Qgeth3/
sudo -u geth touch /opt/qgeth/test.txt
```

### Build Tool Dependencies
```bash
# Install required build tools (bootstrap should handle this)
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

### Build Temp Directory Issues
```bash
# Check if build temp is configured
echo $QGETH_BUILD_TEMP
ls -la /opt/qgeth/temp/

# If temp directory issues, clear and recreate
sudo rm -rf /opt/qgeth/temp/*
sudo mkdir -p /opt/qgeth/temp
sudo chown geth:geth /opt/qgeth/temp

# Retry build
cd /opt/qgeth/Qgeth3
sudo -u geth ./scripts/linux/build-linux.sh geth
```

## 🔄 Service Management Issues

### Geth Service Not Starting
```bash
# Check service status
sudo systemctl status qgeth.service

# Check for failed services
sudo systemctl --failed

# View service logs
sudo journalctl -u qgeth.service -f
sudo journalctl -u qgeth.service --no-pager -l

# Restart service
sudo systemctl restart qgeth.service

# If service file missing, recreate
sudo tee /etc/systemd/system/qgeth.service > /dev/null << 'EOF'
[Unit]
Description=Q Geth Quantum Blockchain Node
After=network.target

[Service]
Type=simple
User=geth
Group=geth
WorkingDirectory=/opt/qgeth/Qgeth3
ExecStart=/opt/qgeth/Qgeth3/scripts/linux/start-geth.sh testnet
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable qgeth.service
sudo systemctl start qgeth.service
```

### Geth Binary Not Found
```bash
# Check if geth binary exists
ls -la /opt/qgeth/Qgeth3/geth.bin

# If missing, rebuild
cd /opt/qgeth/Qgeth3
sudo -u geth ./scripts/linux/build-linux.sh geth

# Check if build was successful
ls -la /opt/qgeth/Qgeth3/geth.bin
file /opt/qgeth/Qgeth3/geth.bin

# Test geth binary
sudo -u geth /opt/qgeth/Qgeth3/geth.bin version
```

### Start Script Issues
```bash
# Check if start script exists and is executable
ls -la /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh

# Make executable if needed
sudo chmod +x /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh

# Test start script manually
cd /opt/qgeth/Qgeth3
sudo -u geth ./scripts/linux/start-geth.sh testnet

# Check for script errors
bash -x /opt/qgeth/Qgeth3/scripts/linux/start-geth.sh testnet
```

## 🌐 Network and Firewall Issues

### UFW Firewall Configuration
```bash
# Check firewall status
sudo ufw status verbose

# Configure required ports (bootstrap should handle this)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8545/tcp    # HTTP RPC
sudo ufw allow 8546/tcp    # WebSocket RPC
sudo ufw allow 30303/tcp   # P2P TCP
sudo ufw allow 30303/udp   # P2P UDP

# Enable firewall if not active
sudo ufw --force enable

# Test HTTP RPC connectivity
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Test WebSocket RPC
wscat -c ws://localhost:8546 -x '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'
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
sudo lsof -i :8546
sudo lsof -i :30303
sudo netstat -tulpn | grep -E "(8545|8546|30303)"

# Kill conflicting processes
sudo pkill -f geth
sudo pkill -f quantum-miner

# Check if geth is already running
ps aux | grep geth
sudo systemctl stop qgeth.service

# Start with different ports if needed
cd /opt/qgeth/Qgeth3
sudo -u geth ./geth.bin --testnet --http --http.port 8547 --ws --ws.port 8548
```

## 💾 Storage and Data Issues

### Disk Space Problems
```bash
# Check disk usage
df -h
du -sh /opt/qgeth/
du -sh ~/.qcoin/

# Clean up space
# 1. Clean build artifacts
sudo rm -rf /opt/qgeth/temp/*
sudo rm -rf /opt/qgeth/Qgeth3/quantum-geth/build/

# 2. Clean package cache
sudo apt autoremove -y
sudo apt autoclean

# 3. Clean blockchain data (if safe to do)
sudo systemctl stop qgeth.service
sudo rm -rf ~/.qcoin/*/geth/chaindata
sudo systemctl start qgeth.service

# 4. Rotate logs
sudo logrotate -f /etc/logrotate.conf
```

### Database Corruption
```bash
# Symptoms: "database corruption", "bad block", sync issues
# Solution: Reset blockchain database

# Stop service
sudo systemctl stop qgeth.service

# Backup current data
sudo cp -r ~/.qcoin/testnet ~/.qcoin/testnet.backup

# Remove corrupted data
sudo rm -rf ~/.qcoin/*/geth/chaindata

# Restart service (will re-sync from genesis)
sudo systemctl start qgeth.service

# Monitor sync progress
sudo journalctl -u qgeth.service -f | grep -E "(sync|block|chain)"
```

### Blockchain Data Directory Issues
```bash
# Check blockchain data location
ls -la ~/.qcoin/
ls -la ~/.qcoin/testnet/geth/

# If data directory owned by wrong user
sudo chown -R geth:geth ~/.qcoin/

# Check disk space for blockchain data
du -sh ~/.qcoin/testnet/geth/chaindata

# If data directory corrupted, reset
sudo systemctl stop qgeth.service
sudo rm -rf ~/.qcoin/testnet/geth/chaindata
sudo systemctl start qgeth.service
```

## 🔐 Security Issues

### User and Permission Problems
```bash
# Check geth user exists
id geth
sudo passwd -S geth

# Create geth user if missing
sudo useradd -r -m -s /bin/bash geth

# Fix ownership recursively
sudo chown -R geth:geth /opt/qgeth
sudo chown -R geth:geth ~/.qcoin

# Check service file user
grep -E "User|Group" /etc/systemd/system/qgeth.service
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
sudo ufw allow 8546/tcp  # Only if external WebSocket access needed
sudo ufw limit ssh       # Rate limit SSH connections
```

## 🚨 Emergency Recovery

### Complete System Reset
```bash
# Stop service
sudo systemctl stop qgeth.service

# Disable service
sudo systemctl disable qgeth.service

# Remove installation
sudo rm -rf /opt/qgeth
sudo rm -f /etc/systemd/system/qgeth.service

# Remove blockchain data
sudo rm -rf ~/.qcoin

# Remove user
sudo userdel -r geth

# Fresh installation
curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y
```

### Service Recovery
```bash
# Reset systemd state
sudo systemctl daemon-reload
sudo systemctl reset-failed

# Recreate service manually if needed
sudo tee /etc/systemd/system/qgeth.service > /dev/null << 'EOF'
[Unit]
Description=Q Geth Quantum Blockchain Node
After=network.target

[Service]
Type=simple
User=geth
Group=geth
WorkingDirectory=/opt/qgeth/Qgeth3
ExecStart=/opt/qgeth/Qgeth3/scripts/linux/start-geth.sh testnet
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable qgeth.service
sudo systemctl start qgeth.service

# Check service status
sudo systemctl status qgeth.service
```

## 📊 Monitoring and Diagnostics

### Real-time Monitoring
```bash
# Monitor service logs
sudo journalctl -u qgeth.service -f

# Monitor system resources
htop
iotop -ao
df -h
free -h

# Monitor network connections
watch -n 5 'netstat -tulpn | grep -E "(8545|8546|30303)"'

# Monitor blockchain sync status
watch -n 10 'curl -s -X POST -H "Content-Type: application/json" --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_syncing\",\"params\":[],\"id\":1}" http://localhost:8545'
```

### Performance Diagnostics
```bash
# Check geth performance
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545

# Check peer connections
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545

# Check mining status
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_mining","params":[],"id":1}' \
  http://localhost:8545

# Check gas price
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_gasPrice","params":[],"id":1}' \
  http://localhost:8545
```

### Bootstrap Environment Health Check Script
```bash
# Create comprehensive health check script
sudo tee /opt/qgeth/health-check.sh > /dev/null <<'EOF'
#!/bin/bash
# Q Geth Bootstrap Health Check Script

echo "=== Q Geth Bootstrap Health Check $(date) ==="

# Check service status
echo "--- Service Status ---"
systemctl is-active qgeth.service
if [ $? -eq 0 ]; then
    echo "✅ Geth service is running"
else
    echo "❌ Geth service is not running"
    systemctl status qgeth.service --no-pager -l
fi

# Check disk space
echo "--- Disk Space ---"
df -h | grep -E "(Filesystem|/dev/)"
df -h / | awk 'NR==2 {if ($5+0 > 90) print "⚠️  Disk usage high: " $5; else print "✅ Disk usage OK: " $5}'

# Check memory
echo "--- Memory Usage ---"
free -h
free | awk 'NR==2{printf "Memory Usage: %.2f%%\n", $3*100/$2}'

# Check geth connectivity
echo "--- Geth Connectivity ---"
if curl -s -X POST -H "Content-Type: application/json" \
   --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
   http://localhost:8545 > /dev/null; then
    echo "✅ HTTP RPC responding"
else
    echo "❌ HTTP RPC not responding"
fi

# Check blockchain sync
echo "--- Blockchain Status ---"
SYNC=$(curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545 | jq -r '.result')

if [ "$SYNC" = "false" ]; then
    echo "✅ Blockchain is synced"
else
    echo "🔄 Blockchain is syncing..."
fi

# Check peer connections
PEERS=$(curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545 | jq -r '.result')

PEER_COUNT=$((16#${PEERS#0x}))
echo "Connected peers: $PEER_COUNT"
if [ $PEER_COUNT -gt 0 ]; then
    echo "✅ Connected to peers"
else
    echo "⚠️  No peer connections"
fi

echo "=== Health Check Complete ==="
EOF

sudo chmod +x /opt/qgeth/health-check.sh

# Run health check
sudo /opt/qgeth/health-check.sh

# Add to crontab for regular checks
(sudo crontab -l 2>/dev/null; echo "*/30 * * * * /opt/qgeth/health-check.sh >> /opt/qgeth/health.log 2>&1") | sudo crontab -
```

## 📚 Getting Help

### Information to Collect
When reporting bootstrap deployment issues, provide:

1. **System Details**: OS version, specs (`lsb_release -a`, `free -h`, `df -h`)
2. **Service Status**: `sudo systemctl status qgeth.service`
3. **Service Logs**: `sudo journalctl -u qgeth.service --no-pager -l | tail -50`
4. **Network Status**: `sudo ufw status && netstat -tulpn | grep -E "(8545|8546|30303)"`
5. **Geth Status**: Run the health check script above
6. **Error Messages**: Full command output and error messages

### Comprehensive Diagnostic Collection
```bash
# Create diagnostic collection script
sudo tee /tmp/bootstrap-diag.sh > /dev/null <<'EOF'
#!/bin/bash
DIAG_FILE="bootstrap-diagnostics-$(date +%Y%m%d-%H%M%S).txt"

echo "=== Q Geth Bootstrap Diagnostics ===" > $DIAG_FILE
echo "Date: $(date)" >> $DIAG_FILE
echo "Hostname: $(hostname)" >> $DIAG_FILE
echo "" >> $DIAG_FILE

echo "=== System Info ===" >> $DIAG_FILE
uname -a >> $DIAG_FILE
lsb_release -a >> $DIAG_FILE 2>&1
echo "" >> $DIAG_FILE

echo "=== Resources ===" >> $DIAG_FILE
free -h >> $DIAG_FILE
df -h >> $DIAG_FILE
echo "" >> $DIAG_FILE

echo "=== Service Status ===" >> $DIAG_FILE
systemctl status qgeth.service --no-pager >> $DIAG_FILE 2>&1
echo "" >> $DIAG_FILE

echo "=== Service Logs (last 50 lines) ===" >> $DIAG_FILE
journalctl -u qgeth.service --no-pager -l | tail -50 >> $DIAG_FILE 2>&1
echo "" >> $DIAG_FILE

echo "=== Network Status ===" >> $DIAG_FILE
ufw status >> $DIAG_FILE 2>&1
netstat -tulpn | grep -E "(8545|8546|30303)" >> $DIAG_FILE 2>&1
echo "" >> $DIAG_FILE

echo "=== Installation Status ===" >> $DIAG_FILE
ls -la /opt/qgeth/ >> $DIAG_FILE 2>&1
ls -la /opt/qgeth/Qgeth3/geth.bin >> $DIAG_FILE 2>&1
ls -la ~/.qcoin/ >> $DIAG_FILE 2>&1
echo "" >> $DIAG_FILE

echo "=== Geth Version ===" >> $DIAG_FILE
/opt/qgeth/Qgeth3/geth.bin version >> $DIAG_FILE 2>&1
echo "" >> $DIAG_FILE

echo "=== RPC Test ===" >> $DIAG_FILE
curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545 >> $DIAG_FILE 2>&1

echo "Diagnostics saved to: $DIAG_FILE"
cat $DIAG_FILE
EOF

chmod +x /tmp/bootstrap-diag.sh
/tmp/bootstrap-diag.sh
```

## ✅ Bootstrap Deployment Checklist

### Pre-Deployment
- [ ] System meets minimum requirements (2GB RAM, 20GB disk, Ubuntu 20.04+)
- [ ] SSH access configured with key authentication
- [ ] Root or sudo access available
- [ ] Network connectivity working
- [ ] Firewall allows SSH (port 22)

### During Installation
- [ ] Bootstrap script downloads successfully
- [ ] Bootstrap runs with `-y` flag for non-interactive mode
- [ ] Go 1.21+ installed automatically
- [ ] Build completes without memory issues
- [ ] Systemd service created and enabled
- [ ] Firewall configured automatically

### Post-Deployment Verification
- [ ] `sudo systemctl status qgeth.service` shows active (running)
- [ ] HTTP RPC responds on port 8545
- [ ] WebSocket RPC responds on port 8546
- [ ] P2P networking active on port 30303
- [ ] Blockchain syncing properly
- [ ] Health check script reports all green

### Common Issues and Quick Fixes
- **Service won't start**: Check `sudo journalctl -u qgeth.service -f` for errors
- **Build fails**: Ensure 3GB+ total memory (RAM + swap)
- **RPC not responding**: Check firewall with `sudo ufw status`
- **Sync issues**: Reset blockchain data and re-sync from genesis
- **Permission errors**: Fix ownership with `sudo chown -R geth:geth /opt/qgeth`

**Most bootstrap deployment issues are caused by insufficient memory, firewall configuration, or permission problems!** 