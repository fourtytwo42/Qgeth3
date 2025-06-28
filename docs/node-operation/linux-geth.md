# Linux Q Geth Node Guide

Complete guide for running Q Coin quantum blockchain nodes on Linux systems.

## 📋 Requirements

### System Requirements
- **OS**: Ubuntu 20.04+, Debian 11+, CentOS 8+, or compatible Linux
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 20GB minimum (SSD recommended for better performance)
- **Network**: Stable internet connection with 1Mbps+ bandwidth

### Software Dependencies
- **Go**: Version 1.21 or later
- **Git**: For source code management
- **Build Tools**: gcc, make, pkg-config

## 🚀 Installation

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Build Q Geth
./scripts/linux/build-linux.sh geth

# Start the node (testnet)
./scripts/linux/start-geth.sh testnet
```

### Manual Build Process
```bash
# Navigate to quantum-geth directory
cd quantum-geth

# Set build environment
export CGO_ENABLED=0
export GOOS=linux
export GOARCH=amd64

# Build geth binary
go build -ldflags="-s -w -X main.gitCommit=$(git rev-parse HEAD) -X main.buildTime=$(date -u '+%Y-%m-%d_%H:%M:%S')" -o ../geth.bin ./cmd/geth

# Verify build
../geth.bin version
```

### Installation Verification
```bash
# Check if geth binary exists and is executable
ls -la geth.bin
file geth.bin

# Test basic functionality
./geth.bin help
./geth.bin version

# Check quantum consensus is available
./geth.bin help | grep -i quantum
```

## ⚙️ Configuration

### Network Selection
Q Coin supports three networks:

#### Testnet (Recommended for testing)
```bash
# Start testnet node
./scripts/linux/start-geth.sh testnet

# Custom testnet configuration
./geth.bin --datadir ~/.qcoin/testnet \
  --networkid 1337 \
  --genesis configs/genesis_quantum_testnet.json \
  --http --http.addr "0.0.0.0" --http.port 8545 \
  --http.corsdomain "*" --http.api "eth,net,web3,personal,txpool" \
  --ws --ws.addr "0.0.0.0" --ws.port 8546 \
  --ws.origins "*" --ws.api "eth,net,web3" \
  --port 30303 --nat "any" \
  --allow-insecure-unlock \
  console
```

#### Devnet (Development)
```bash
# Start development network
./scripts/linux/start-geth.sh devnet

# Custom devnet with mining
./geth.bin --datadir ~/.qcoin/devnet \
  --networkid 1338 \
  --genesis configs/genesis_quantum_dev.json \
  --http --http.addr "127.0.0.1" --http.port 8545 \
  --mine --miner.etherbase "0xYourCoinbaseAddress" \
  console
```

#### Mainnet (Production)
```bash
# Start mainnet node (when available)
./scripts/linux/start-geth.sh mainnet

# Mainnet with conservative settings
./geth.bin --datadir ~/.qcoin/mainnet \
  --networkid 1339 \
  --genesis configs/genesis_quantum_mainnet.json \
  --http --http.addr "127.0.0.1" --http.port 8545 \
  --http.api "eth,net,web3" \
  --port 30303 \
  --cache 1024 --maxpeers 50 \
  console
```

### Data Directory Structure
```
~/.qcoin/
├── testnet/
│   ├── geth/
│   │   ├── chaindata/      # Blockchain database
│   │   ├── lightchaindata/ # Light client data
│   │   ├── nodes/          # Node discovery data
│   │   └── LOCK           # Database lock file
│   ├── keystore/          # Account keystore files
│   └── geth.ipc          # IPC communication socket
├── devnet/               # Development network data
└── mainnet/             # Mainnet data (when available)
```

### Configuration Files
```bash
# Create custom configuration
mkdir -p ~/.qcoin/config
cat > ~/.qcoin/config/geth.toml << EOF
[Eth]
NetworkId = 1337
DatabaseHandles = 1024
DatabaseCache = 1024
TrieCleanCache = 256
TrieCleanCacheJournal = "triecache"
TrieCleanCacheRejournal = 3600000000000
TrieDirtyCache = 256
TrieTimeout = 3600000000000
EnablePreimageRecording = false

[Node]
DataDir = "/home/$USER/.qcoin/testnet"
IPCPath = "geth.ipc"
HTTPHost = "127.0.0.1"
HTTPPort = 8545
HTTPCors = ["*"]
HTTPVirtualHosts = ["localhost"]
HTTPModules = ["eth", "net", "web3", "personal", "txpool"]
WSHost = "127.0.0.1"
WSPort = 8546
WSOrigins = ["*"]
WSModules = ["eth", "net", "web3"]

[Node.P2P]
MaxPeers = 50
NoDiscovery = false
BootstrapNodes = []
StaticNodes = []
TrustedNodes = []
ListenAddr = ":30303"
EnableMsgEvents = false
EOF

# Use custom configuration
./geth.bin --config ~/.qcoin/config/geth.toml
```

## 🌐 Networking

### Port Configuration
| Service | Default Port | Protocol | Purpose |
|---------|-------------|----------|---------|
| P2P | 30303 | TCP/UDP | Peer-to-peer networking |
| HTTP RPC | 8545 | TCP | JSON-RPC API |
| WebSocket | 8546 | TCP | WebSocket API |
| IPC | N/A | Unix Socket | Local IPC communication |

### Firewall Setup
```bash
# Ubuntu/Debian with UFW
sudo ufw allow 30303/tcp  # P2P TCP
sudo ufw allow 30303/udp  # P2P UDP
sudo ufw allow 8545/tcp   # HTTP RPC (if external access needed)
sudo ufw allow 8546/tcp   # WebSocket (if external access needed)

# CentOS/RHEL with firewalld
sudo firewall-cmd --permanent --add-port=30303/tcp
sudo firewall-cmd --permanent --add-port=30303/udp
sudo firewall-cmd --permanent --add-port=8545/tcp  # Optional
sudo firewall-cmd --reload

# Check open ports
sudo netstat -tulpn | grep -E "(8545|8546|30303)"
```

### NAT and Router Configuration
```bash
# For nodes behind NAT/router, configure port forwarding:
# Router Admin Panel -> Port Forwarding:
# - Internal IP: Your machine's LAN IP
# - External Port: 30303
# - Internal Port: 30303
# - Protocol: TCP and UDP

# Test external connectivity
./geth.bin --datadir ~/.qcoin/testnet --nat "extip:YOUR_EXTERNAL_IP"

# Use UPnP for automatic port mapping
./geth.bin --datadir ~/.qcoin/testnet --nat "upnp"
```

### Peer Discovery and Connectivity
```bash
# Check peer count
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "net.peerCount"

# List connected peers
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "admin.peers"

# Add static peers
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec 'admin.addPeer("enode://NODEID@IP:PORT")'

# Check node info
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "admin.nodeInfo"
```

## 🔧 Operation

### Starting and Stopping
```bash
# Start in background
nohup ./scripts/linux/start-geth.sh testnet > geth.log 2>&1 &

# Start with custom options
./geth.bin --datadir ~/.qcoin/testnet \
  --networkid 1337 \
  --http --http.port 8545 \
  --port 30303 \
  --verbosity 3 \
  console

# Stop gracefully
# In console: exit
# Or send SIGTERM: pkill -TERM geth.bin
```

### Console Access
```bash
# Attach to running node (IPC - fastest)
./geth.bin attach ~/.qcoin/testnet/geth.ipc

# Attach via HTTP (local)
./geth.bin attach http://localhost:8545

# Attach to remote geth node via HTTP (manage remote nodes)
./geth.bin attach http://YOUR_VPS_IP:8545

# Examples:
# ./geth.bin attach http://134.199.202.42:8545
# ./geth.bin attach http://192.168.1.100:8545

# JavaScript console examples
> eth.accounts
> eth.blockNumber
> net.peerCount
> personal.newAccount("password")
> eth.sendTransaction({from: eth.accounts[0], to: "0x...", value: web3.toWei(1, "ether")})
```

**Remote Console Management:**
The HTTP attach method allows you to manage remote Q Geth nodes from your local machine. This is particularly useful for:
- Managing VPS-deployed nodes
- Monitoring multiple nodes from one location
- Remote administration and debugging
- Testing network connectivity between nodes

**Security Note:** Ensure the remote node is configured with `--http.addr "0.0.0.0"` and proper firewall rules for external HTTP RPC access.

### Account Management
```bash
# Create new account
./geth.bin account new --datadir ~/.qcoin/testnet

# List accounts
./geth.bin account list --datadir ~/.qcoin/testnet

# Import private key
./geth.bin account import private_key.txt --datadir ~/.qcoin/testnet

# In console - unlock account
> personal.unlockAccount(eth.accounts[0], "password", 0)

# Check balance
> eth.getBalance(eth.accounts[0])
```

### Blockchain Operations
```bash
# Check sync status
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.syncing"

# Get latest block
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.getBlock('latest')"

# Check chain ID
./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.chainId()"

# Export/import blockchain data
./geth.bin export blockchain.rlp --datadir ~/.qcoin/testnet
./geth.bin import blockchain.rlp --datadir ~/.qcoin/testnet
```

## ⛏️ Mining Integration

### Solo Mining Setup
```bash
# Start geth with mining enabled
./geth.bin --datadir ~/.qcoin/testnet \
  --networkid 1337 \
  --mine --miner.etherbase "0xYourCoinbaseAddress" \
  --miner.threads 1 \
  console

# Enable mining in console
> miner.setEtherbase(eth.accounts[0])
> miner.start(1)  # 1 thread
> miner.stop()
```

### External Miner Connection
```bash
# Start geth as mining pool backend
./geth.bin --datadir ~/.qcoin/testnet \
  --http --http.addr "127.0.0.1" --http.port 8545 \
  --http.api "eth,net,web3,personal,txpool" \
  --allow-insecure-unlock

# Connect quantum-miner to geth
./quantum-miner --node http://localhost:8545 \
  --coinbase "0xYourAddress" \
  --threads 4
```

## 📊 Monitoring and Logging

### Logging Configuration
```bash
# Start with specific log level
./geth.bin --datadir ~/.qcoin/testnet --verbosity 4  # 0=silent, 5=debug

# Log to file
./geth.bin --datadir ~/.qcoin/testnet 2>&1 | tee geth.log

# Structured logging
./geth.bin --datadir ~/.qcoin/testnet --log.json 2>&1 | tee geth.json

# Rotate logs with logrotate
sudo tee /etc/logrotate.d/qgeth > /dev/null <<EOF
/home/$USER/*.log {
    daily
    missingok
    rotate 14
    compress
    notifempty
    copytruncate
    su $USER $USER
}
EOF
```

### Metrics and Monitoring
```bash
# Enable metrics collection
./geth.bin --datadir ~/.qcoin/testnet \
  --metrics --metrics.addr "127.0.0.1" --metrics.port 6060

# View metrics in browser
curl http://localhost:6060/debug/metrics

# Performance monitoring
htop  # System resources
iotop -ao  # Disk I/O
netstat -i  # Network interfaces

# Monitor blockchain sync
watch -n 5 './geth.bin attach ~/.qcoin/testnet/geth.ipc --exec "eth.blockNumber"'
```

### Health Checks
```bash
# RPC health check
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
  http://localhost:8545

# Sync status check
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_syncing","params":[],"id":1}' \
  http://localhost:8545

# Peer count check
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://localhost:8545
```

## 🔄 Service Management

### Systemd Service Setup
```bash
# Create systemd service file
sudo tee /etc/systemd/system/qgeth.service > /dev/null <<EOF
[Unit]
Description=Q Geth Node
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=/home/$USER/Qgeth3
Environment=HOME=/home/$USER
ExecStart=/home/$USER/Qgeth3/scripts/linux/start-geth.sh testnet
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=10
TimeoutStopSec=60
KillMode=mixed
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable qgeth.service
sudo systemctl start qgeth.service

# Check service status
sudo systemctl status qgeth.service
sudo journalctl -u qgeth.service -f
```

### User Service (Rootless)
```bash
# Create user service directory
mkdir -p ~/.config/systemd/user

# Create user service file
tee ~/.config/systemd/user/qgeth.service > /dev/null <<EOF
[Unit]
Description=Q Geth Node (User Service)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/Qgeth3
Environment=HOME=%h
ExecStart=%h/Qgeth3/scripts/linux/start-geth.sh testnet
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF

# Enable and start user service
systemctl --user daemon-reload
systemctl --user enable qgeth.service
systemctl --user start qgeth.service

# Enable user services at boot
sudo loginctl enable-linger $USER
```

## 🔒 Security

### Network Security
```bash
# Bind to localhost only (default)
./geth.bin --http.addr "127.0.0.1" --ws.addr "127.0.0.1"

# Restrict API access
./geth.bin --http.api "eth,net,web3"  # Remove personal, admin

# Use allowlist for CORS
./geth.bin --http.corsdomain "https://your-dapp.com"

# Disable dangerous APIs
./geth.bin --http.api "eth,net,web3"  # No debug, admin, personal
```

### Account Security
```bash
# Use hardware wallet or external signer
./geth.bin --external-signer "http://localhost:8550"

# Backup keystore files
cp -r ~/.qcoin/testnet/keystore ~/keystore-backup-$(date +%Y%m%d)

# Use strong passwords
# Generate secure password: openssl rand -base64 32

# Lock accounts after use
./geth.bin attach --exec "personal.lockAccount(eth.accounts[0])"
```

### File Permissions
```bash
# Secure data directory
chmod 700 ~/.qcoin
chmod 600 ~/.qcoin/*/keystore/*

# Secure configuration files
chmod 600 ~/.qcoin/config/geth.toml

# Regular security audit
find ~/.qcoin -type f -perm /o+rwx -ls  # Find world-writable files
```

## 🛠️ Maintenance

### Regular Maintenance
```bash
# Update Q Geth
cd Qgeth3
git pull origin main
./scripts/linux/build-linux.sh geth
sudo systemctl restart qgeth.service

# Clean up disk space
# Remove old logs
find ~/.qcoin -name "*.log" -mtime +30 -delete

# Compact database (requires stopping node)
systemctl --user stop qgeth.service
./geth.bin --datadir ~/.qcoin/testnet removedb
# Restart and resync from genesis
systemctl --user start qgeth.service

# Backup blockchain data
tar -czf blockchain-backup-$(date +%Y%m%d).tar.gz ~/.qcoin/testnet
```

### Performance Optimization
```bash
# SSD optimization
# Enable TRIM if using SSD
sudo systemctl enable fstrim.timer

# Tune database cache
./geth.bin --cache 2048  # Increase cache size

# Use fast sync (if available)
./geth.bin --syncmode "fast"

# Optimize network settings
echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default = 262144' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 📚 Advanced Usage

### Multiple Network Setup
```bash
# Run multiple networks simultaneously
./geth.bin --datadir ~/.qcoin/testnet --port 30303 --http.port 8545 &
./geth.bin --datadir ~/.qcoin/devnet --port 30304 --http.port 8546 &

# Use different IPC paths
./geth.bin --datadir ~/.qcoin/testnet --ipcpath ~/.qcoin/testnet.ipc
./geth.bin --datadir ~/.qcoin/devnet --ipcpath ~/.qcoin/devnet.ipc
```

### Custom Genesis Block
```bash
# Create custom genesis
cat > custom_genesis.json << EOF
{
  "config": {
    "chainId": 1340,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "qmpow": {}
  },
  "alloc": {
    "0xYourAddress": {
      "balance": "0x200000000000000000000"
    }
  },
  "coinbase": "0x0000000000000000000000000000000000000000",
  "difficulty": "0x20000",
  "extraData": "",
  "gasLimit": "0x2fefd8",
  "nonce": "0x0000000000000042",
  "mixhash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
  "timestamp": "0x00"
}
EOF

# Initialize custom network
./geth.bin --datadir ~/.qcoin/custom init custom_genesis.json
./geth.bin --datadir ~/.qcoin/custom --networkid 1340 console
```

### API Integration Examples
```bash
# Python integration
pip install web3
python3 << EOF
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
print(f"Connected: {w3.isConnected()}")
print(f"Block number: {w3.eth.block_number}")
print(f"Chain ID: {w3.eth.chain_id}")
EOF

# JavaScript integration (Node.js)
npm install web3
node << EOF
const Web3 = require('web3');
const web3 = new Web3('http://localhost:8545');
web3.eth.getBlockNumber().then(console.log);
EOF
```

## 📖 Reference

### Command Line Options
```bash
# View all options
./geth.bin help

# Common options
--datadir          # Data directory path
--networkid        # Network identifier
--http             # Enable HTTP-RPC server
--http.addr        # HTTP-RPC server listening interface
--http.port        # HTTP-RPC server listening port
--http.api         # API's offered over the HTTP-RPC interface
--ws               # Enable WS-RPC server
--port             # Network listening port
--bootnodes        # Comma separated enode URLs for P2P discovery bootstrap
--verbosity        # Logging verbosity: 0=silent, 1=error, 2=warn, 3=info, 4=debug, 5=detail
```

### Useful Scripts
```bash
# Quick node status check
cat > check_node.sh << 'EOF'
#!/bin/bash
echo "=== Q Geth Node Status ==="
echo "Process: $(pgrep -f geth.bin | wc -l) running"
echo "Block: $(./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec 'eth.blockNumber' 2>/dev/null || echo 'N/A')"
echo "Peers: $(./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec 'net.peerCount' 2>/dev/null || echo 'N/A')"
echo "Syncing: $(./geth.bin attach ~/.qcoin/testnet/geth.ipc --exec 'eth.syncing' 2>/dev/null || echo 'N/A')"
EOF
chmod +x check_node.sh
```

For troubleshooting Linux geth issues, see [Linux Geth Troubleshooting](troubleshooting-linux-geth.md).

---

**Happy quantum blockchain exploring with Q Geth on Linux! 🐧⚡** 