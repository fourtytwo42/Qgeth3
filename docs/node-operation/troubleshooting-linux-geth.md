# Linux Geth Troubleshooting

Solutions for Q Coin geth node issues on Linux systems.

## 🔧 Quick Geth Diagnostics

### Node Status Check
```bash
# Check if geth is running
ps aux | grep geth.bin
pgrep -f geth.bin

# Check geth process details
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | grep geth

# Quick health check via RPC
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Check geth logs
tail -f ~/.qcoin/logs/geth.log
journalctl -u qgeth.service -f  # If using systemd
```

## 🚀 Node Startup Issues

### Geth Binary Not Found
```bash
# Symptoms: "command not found", "No such file or directory"
# Solution: Check binary location and permissions

# Verify binary exists and is executable
ls -la geth.bin
file geth.bin
ldd geth.bin  # Check library dependencies

# Make executable if needed
chmod +x geth.bin

# Check if binary is built for correct architecture
file geth.bin  # Should show ELF 64-bit for x86_64 systems

# Rebuild if necessary
./scripts/linux/build-linux.sh geth --clean
```

### Genesis Block Initialization Issues
```bash
# Symptoms: "genesis file not found", "invalid genesis"
# Solution: Verify genesis file and initialization

# Check genesis file exists
ls -la configs/genesis_quantum_testnet.json
cat configs/genesis_quantum_testnet.json | jq .  # Validate JSON

# Initialize datadir with genesis (if needed)
./geth.bin --datadir ~/.qcoin/testnet init configs/genesis_quantum_testnet.json

# Check if genesis was properly initialized
ls -la ~/.qcoin/testnet/geth/chaindata/

# If genesis mismatch, remove and reinitialize
rm -rf ~/.qcoin/testnet/geth/chaindata
./geth.bin --datadir ~/.qcoin/testnet init configs/genesis_quantum_testnet.json
```

### Data Directory Issues
```bash
# Check data directory permissions
ls -la ~/.qcoin/
ls -la ~/.qcoin/testnet/

# Fix ownership if needed
sudo chown -R $(whoami):$(whoami) ~/.qcoin/

# Check disk space
df -h ~/.qcoin/
du -sh ~/.qcoin/testnet/

# If permission denied on IPC
chmod 755 ~/.qcoin/testnet/
ls -la ~/.qcoin/testnet/geth.ipc
```

## 🌐 Networking Issues

### Port Binding Failures
```bash
# Symptoms: "bind: address already in use"
# Solution: Find and resolve port conflicts

# Check what's using geth ports
sudo lsof -i :8545
sudo lsof -i :30303
sudo netstat -tulpn | grep -E "(8545|30303)"

# Kill conflicting processes
sudo kill -9 <PID>
sudo pkill -f geth

# Use different ports if needed
./geth.bin --datadir ~/.qcoin/testnet \
  --http.port 8546 \
  --port 30304

# Check if ports are now available
nc -zv localhost 8545
```

### Firewall Blocking
```bash
# Check firewall status
sudo ufw status verbose
sudo iptables -L

# Allow geth ports
sudo ufw allow 30303/tcp
sudo ufw allow 30303/udp
sudo ufw allow 8545/tcp  # If external access needed

# Test local connectivity
telnet localhost 8545
nc -zv localhost 30303

# Test from another machine (if firewall allows)
telnet YOUR_SERVER_IP 8545
```

### P2P Connectivity Issues
```bash
# Check peer count
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "net.peerCount"

# Check listening status
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "net.listening"

# Check node info
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "admin.nodeInfo"

# If no peers, check network connectivity
ping -c 4 8.8.8.8
dig github.com

# Check NAT settings
./geth.bin --datadir ~/.qcoin/testnet --nat "upnp"
./geth.bin --datadir ~/.qcoin/testnet --nat "extip:YOUR_EXTERNAL_IP"

# Add bootstrap nodes manually
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec 'admin.addPeer("enode://NODEID@IP:PORT")'
```

## 🔄 Sync Issues

### Blockchain Sync Problems
```bash
# Check sync status
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.syncing"

# Check current block vs latest
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.blockNumber"

# Check peer sync info
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "admin.peers"

# If sync is stuck, restart with fresh database
systemctl --user stop qgeth.service  # If using service
rm -rf ~/.qcoin/testnet/geth/chaindata
./geth.bin --datadir ~/.qcoin/testnet init configs/genesis_quantum_testnet.json
systemctl --user start qgeth.service
```

### Database Corruption
```bash
# Symptoms: "database corruption", "bad block", "leveldb error"
# Solution: Rebuild database

# Stop geth
pkill -f geth.bin
systemctl --user stop qgeth.service

# Check database integrity
./geth.bin --datadir ~/.qcoin/testnet --check-db

# Backup corrupted data (optional)
cp -r ~/.qcoin/testnet ~/.qcoin/testnet.backup

# Remove corrupted database
rm -rf ~/.qcoin/testnet/geth/chaindata

# Reinitialize and restart
./geth.bin --datadir ~/.qcoin/testnet init configs/genesis_quantum_testnet.json
./scripts/linux/start-geth.sh testnet
```

### Sync Performance Issues
```bash
# Monitor sync progress
watch -n 5 './geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.blockNumber"'

# Check system resources during sync
htop
iotop -ao
iostat 1

# Optimize cache settings
./geth.bin --datadir ~/.qcoin/testnet --cache 2048

# Use faster sync mode (if available)
./geth.bin --datadir ~/.qcoin/testnet --syncmode "fast"

# Check disk I/O performance
dd if=/dev/zero of=test.tmp bs=1M count=1000
rm test.tmp
```

## 💬 Console and IPC Issues

### Console Access Problems
```bash
# Symptoms: "connection refused", "no such file"
# Solution: Check IPC socket

# Verify IPC socket exists
ls -la ~/.qcoin/testnet/geth.ipc
file ~/.qcoin/testnet/geth.ipc  # Should be socket

# Check socket permissions
stat ~/.qcoin/testnet/geth.ipc

# If socket missing, check if geth is running
ps aux | grep geth.bin

# Try HTTP attachment instead
./geth.bin attach http://localhost:8545

# If HTTP also fails, check geth logs
tail -50 ~/.qcoin/logs/geth.log
```

### Console Commands Failing
```bash
# In geth console, check basic functionality
> web3.version
> eth.accounts
> net.peerCount

# If commands fail, check API availability
> web3.admin  # Should show admin functions
> personal    # Should show personal functions

# Common fixes:
# 1. Restart geth with proper APIs enabled
./geth.bin --datadir ~/.qcoin/testnet \
  --http.api "eth,net,web3,personal,admin,txpool"

# 2. Check if account is unlocked
> personal.listAccounts
> personal.unlockAccount(eth.accounts[0], "password")
```

### Account Management Issues
```bash
# Account creation problems
./geth.bin account new --datadir ~/.qcoin/testnet

# If account creation fails
ls -la ~/.qcoin/testnet/keystore/
chmod 755 ~/.qcoin/testnet/keystore/

# List existing accounts
./geth.bin account list --datadir ~/.qcoin/testnet

# Import account issues
./geth.bin account import private_key.txt --datadir ~/.qcoin/testnet

# Check keystore file format
cat ~/.qcoin/testnet/keystore/UTC--*
```

## ⛏️ Mining Integration Issues

### Mining Not Starting
```bash
# Check if mining is enabled
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.mining"

# Check coinbase address
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.coinbase"

# Set coinbase and start mining
> miner.setEtherbase(eth.accounts[0])
> miner.start(1)

# Check mining status
> eth.hashrate
> eth.mining
```

### External Miner Connection
```bash
# Test if geth RPC is accessible for mining
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_getWork","params":[],"id":1}' \
  http://localhost:8545

# If connection refused, check geth startup
./geth.bin --datadir ~/.qcoin/testnet \
  --http --http.addr "127.0.0.1" --http.port 8545 \
  --http.api "eth,net,web3,personal,txpool" \
  --allow-insecure-unlock

# Test miner connectivity
./quantum-miner --node http://localhost:8545 --test

# Check miner logs for connection issues
./quantum-miner --node http://localhost:8545 --verbose
```

## 📊 Performance Issues

### High CPU Usage
```bash
# Monitor geth CPU usage
top -p $(pgrep geth.bin)
htop | grep geth

# Check verbosity level (lower = less CPU)
./geth.bin --verbosity 2  # Reduce from default 3

# Optimize cache settings
./geth.bin --cache 1024 --trie-cache-gens 120

# Check for excessive logging
tail -f ~/.qcoin/logs/geth.log | wc -l

# Monitor peer connections
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "admin.peers.length"
```

### High Memory Usage
```bash
# Monitor memory usage
ps -eo pid,ppid,cmd,%mem,%cpu | grep geth
free -h

# Reduce cache if needed
./geth.bin --cache 512

# Check for memory leaks
valgrind --tool=memcheck --track-origins=yes ./geth.bin --datadir ~/.qcoin/testnet

# Monitor memory over time
while true; do
  ps -p $(pgrep geth.bin) -o pid,ppid,cmd,%mem,%cpu
  sleep 10
done
```

### Disk I/O Issues
```bash
# Monitor disk usage
iotop -ao
iostat 1

# Check disk space
df -h ~/.qcoin/
du -sh ~/.qcoin/testnet/geth/chaindata/

# Optimize database settings
./geth.bin --datadir ~/.qcoin/testnet --cache.database 75

# Move data to faster disk (SSD)
systemctl --user stop qgeth.service
mv ~/.qcoin /mnt/ssd/.qcoin
ln -s /mnt/ssd/.qcoin ~/.qcoin
systemctl --user start qgeth.service
```

## 🔧 Configuration Issues

### Invalid Command Line Options
```bash
# Check available options
./geth.bin help

# Verify option syntax
./geth.bin --help | grep -A 2 -B 2 "your-option"

# Common option fixes:
./geth.bin --datadir ~/.qcoin/testnet  # Not --datadir=~/.qcoin/testnet
./geth.bin --http.api "eth,net,web3"   # Use quotes for multiple APIs
./geth.bin --bootnodes "enode://..."  # Use quotes for node IDs
```

### TOML Configuration Issues
```bash
# Validate TOML configuration
./geth.bin dumpconfig > default.toml
diff default.toml ~/.qcoin/config/geth.toml

# Test configuration file
./geth.bin --config ~/.qcoin/config/geth.toml --help

# Check for syntax errors
python3 -c "import toml; toml.load('~/.qcoin/config/geth.toml')"
```

## 🚨 Emergency Recovery

### Complete Node Reset
```bash
# Stop all geth processes
pkill -f geth.bin
systemctl --user stop qgeth.service

# Backup important data
cp -r ~/.qcoin/testnet/keystore ~/keystore-backup
cp ~/.qcoin/config/geth.toml ~/geth.toml.backup

# Remove all blockchain data
rm -rf ~/.qcoin/testnet/geth/

# Reinitialize
./geth.bin --datadir ~/.qcoin/testnet init configs/genesis_quantum_testnet.json

# Restore keystore
cp -r ~/keystore-backup/* ~/.qcoin/testnet/keystore/

# Restart node
./scripts/linux/start-geth.sh testnet
```

### Log Analysis
```bash
# Analyze geth logs for errors
grep -i error ~/.qcoin/logs/geth.log | tail -20
grep -i fatal ~/.qcoin/logs/geth.log
grep -i panic ~/.qcoin/logs/geth.log

# Check for specific issues
grep -i "database" ~/.qcoin/logs/geth.log
grep -i "network" ~/.qcoin/logs/geth.log
grep -i "sync" ~/.qcoin/logs/geth.log

# Monitor logs in real-time
tail -f ~/.qcoin/logs/geth.log | grep -E "(ERROR|WARN|FATAL)"
```

## 📚 Advanced Debugging

### Debug Logging
```bash
# Enable debug logging
./geth.bin --datadir ~/.qcoin/testnet --verbosity 5

# Log specific modules
./geth.bin --datadir ~/.qcoin/testnet --vmodule "p2p=4,rpc=3"

# Structured JSON logging
./geth.bin --datadir ~/.qcoin/testnet --log.json

# Log to file with rotation
./geth.bin --datadir ~/.qcoin/testnet 2>&1 | rotatelogs ~/.qcoin/logs/geth.log.%Y%m%d 86400
```

### Network Debugging
```bash
# Test P2P connectivity
nc -zv PEER_IP 30303

# Check routing
traceroute PEER_IP

# Monitor network traffic
sudo tcpdump -i any port 30303
sudo netstat -an | grep 30303

# Debug RPC calls
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"admin_nodeInfo","params":[],"id":1}' \
  http://localhost:8545
```

### Performance Profiling
```bash
# CPU profiling
./geth.bin --datadir ~/.qcoin/testnet --pprof --pprof.addr "127.0.0.1" --pprof.port 6060

# View profiles in browser
go tool pprof http://localhost:6060/debug/pprof/profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Memory profiling
go tool pprof http://localhost:6060/debug/pprof/allocs
```

## 📋 Diagnostic Information Collection

### System Diagnostics
```bash
# Collect comprehensive system info
cat > linux-geth-diag.sh << 'EOF'
#!/bin/bash
echo "=== Linux Geth Diagnostics ===" > geth-diagnostics.txt
echo "Date: $(date)" >> geth-diagnostics.txt
echo "Hostname: $(hostname)" >> geth-diagnostics.txt
echo "" >> geth-diagnostics.txt

echo "=== System Info ===" >> geth-diagnostics.txt
uname -a >> geth-diagnostics.txt
lsb_release -a >> geth-diagnostics.txt 2>&1
echo "" >> geth-diagnostics.txt

echo "=== Geth Process ===" >> geth-diagnostics.txt
ps aux | grep geth >> geth-diagnostics.txt
echo "" >> geth-diagnostics.txt

echo "=== Network ===" >> geth-diagnostics.txt
netstat -tulpn | grep -E "(8545|30303)" >> geth-diagnostics.txt
echo "" >> geth-diagnostics.txt

echo "=== Disk Space ===" >> geth-diagnostics.txt
df -h >> geth-diagnostics.txt
du -sh ~/.qcoin/* >> geth-diagnostics.txt 2>&1
echo "" >> geth-diagnostics.txt

echo "=== Recent Logs ===" >> geth-diagnostics.txt
tail -50 ~/.qcoin/logs/geth.log >> geth-diagnostics.txt 2>&1

cat geth-diagnostics.txt
EOF

chmod +x linux-geth-diag.sh
./linux-geth-diag.sh
```

## ✅ Linux Geth Checklist

### Pre-Startup
- [ ] Geth binary exists and is executable
- [ ] Genesis file properly configured
- [ ] Data directory has correct permissions
- [ ] Required ports are available
- [ ] Adequate disk space available

### Runtime
- [ ] Node starts without errors
- [ ] P2P networking functional
- [ ] Blockchain syncing properly
- [ ] RPC API accessible
- [ ] Console attachment working

### Performance
- [ ] System resources not overloaded
- [ ] Peer connections stable
- [ ] Sync progress reasonable
- [ ] No database corruption
- [ ] Log files rotating properly

**For most Linux geth issues, checking logs and verifying basic connectivity resolves the problem!** 