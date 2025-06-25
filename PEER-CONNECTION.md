# 🔗 Q Coin Peer Connection Guide

This guide shows how to connect Q Coin nodes to each other for blockchain synchronization and peer-to-peer networking.

## 📋 Overview

To connect two Q Coin nodes, you need:
1. **Local Node**: Your running Q Coin node
2. **Remote Node**: Another Q Coin node you want to connect to
3. **Enode Information**: Connection details from the remote node

## 🛠️ Tools Provided

### Windows (PowerShell)
- `get-node-info.ps1` - Get your node's connection info
- `connect-peer.ps1` - Connect to a remote peer

### Linux (Bash)
- `get-node-info.sh` - Get your node's connection info  
- `connect-peer.sh` - Connect to a remote peer

## 🚀 Quick Start

### Step 1: Get Local Node Info

**Windows:**
```powershell
.\get-node-info.ps1
```

**Linux:**
```bash
./get-node-info.sh
```

This will show your node's enode information like:
```
enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.1.100:30303
```

### Step 2: Share Enode with Remote User

Send your enode string to the person running the remote node. They will use this to connect to you.

### Step 3: Connect to Remote Peer

When you receive an enode from a remote node, connect to it:

**Windows:**
```powershell
.\connect-peer.ps1 "enode://[remote-enode-here]"
```

**Linux:**
```bash
./connect-peer.sh "enode://[remote-enode-here]"
```

## 📡 Network Requirements

### Firewall Configuration
Make sure the following ports are open:

| Network | Port | Protocol | Direction |
|---------|------|----------|-----------|
| Dev Network | 30305 | TCP/UDP | Inbound/Outbound |
| Testnet | 30303 | TCP/UDP | Inbound/Outbound |
| Mainnet | 30303 | TCP/UDP | Inbound/Outbound |

### NAT/Router Setup
If behind a router, forward the appropriate port to your machine:
- **Dev**: Forward port 30305 → your_local_ip:30305
- **Testnet/Mainnet**: Forward port 30303 → your_local_ip:30303

## 🌐 Network Types

### Q Coin Dev Network (Chain ID 73234)
- **Purpose**: Development and testing
- **Port**: 30305
- **Data Directory**: `qdata/`
- **Start Command**: `./dev-quick-start.sh` (Linux) or `.\dev-quick-start.ps1` (Windows)

### Q Coin Testnet (Chain ID 73235)
- **Purpose**: Public testing network
- **Port**: 30303
- **Data Directory**: `%APPDATA%\Qcoin\` (Windows) or `~/.qcoin/` (Linux)
- **Start Command**: `./start-linux-geth.sh` (Linux) or `.\qcoin-geth.ps1` (Windows)

### Q Coin Mainnet (Chain ID 73236)
- **Purpose**: Production network
- **Port**: 30303
- **Data Directory**: `%APPDATA%\Qcoin\mainnet\` (Windows) or `~/.qcoin/mainnet/` (Linux)
- **Start Command**: `./start-linux-geth.sh --mainnet` (Linux) or `.\qcoin-geth.ps1 -mainnet` (Windows)

## 🔍 Troubleshooting

### "No Q Coin node detected"
**Problem**: The scripts can't find your running node.

**Solutions**:
1. Make sure your Q Coin node is running
2. Check that RPC is enabled (should be by default)
3. Verify the RPC port (default: 8545)

### "Invalid enode format"
**Problem**: The enode string is malformed.

**Solutions**:
1. Make sure you copied the complete enode string
2. Enode should start with `enode://` and include IP:port
3. Get fresh enode info from the remote node

### "Peer added but connection not established"
**Problem**: Peer was added to node table but connection failed.

**Solutions**:
1. **Firewall**: Check that the required ports are open
2. **NAT**: Ensure proper port forwarding if behind a router
3. **Network**: Verify both nodes are on the same Q Coin network (same Chain ID)
4. **Time**: Connections can take time - wait a few minutes

### "Connection refused"
**Problem**: Cannot reach the remote node.

**Solutions**:
1. **IP Address**: Replace `127.0.0.1` in enode with actual public/local IP
2. **Port**: Verify the port is correct and open
3. **Network**: Ensure both nodes can reach each other (ping test)

## 📊 Monitoring Connections

### Check Current Peers
**Windows:**
```powershell
.\get-node-info.ps1
```

**Linux:**
```bash
./get-node-info.sh
```

### Manual RPC Queries
You can also check peers manually:

```bash
# Get peer count
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":1}' \
  http://127.0.0.1:8545

# Get peer list
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"admin_peers","params":[],"id":1}' \
  http://127.0.0.1:8545
```

## 🔐 Security Notes

1. **Network Isolation**: Each Q Coin network (Dev/Testnet/Mainnet) is isolated
2. **Enode Sharing**: Enode strings are safe to share - they contain no private keys
3. **Firewall**: Only open the specific ports you need
4. **Updates**: Keep your Q Coin node software updated

## 💡 Best Practices

1. **Static IP**: Use static IP addresses for stable connections
2. **Multiple Peers**: Connect to multiple peers for better network resilience
3. **Network Matching**: Ensure all nodes are on the same network (same Chain ID)
4. **Monitoring**: Regularly check peer connections and sync status
5. **Backup**: Keep backups of your blockchain data directory

## 🆘 Getting Help

If you're having trouble connecting nodes:

1. **Check Logs**: Look at your geth console output for connection errors
2. **Network Test**: Use `ping` and `telnet` to test basic connectivity
3. **Documentation**: Refer to this guide and the main README
4. **Community**: Ask for help in Q Coin community channels

## 📝 Example Connection Flow

Here's a complete example of connecting two nodes:

### Node A (Local)
```bash
# 1. Start your Q Coin node
./dev-quick-start.sh

# 2. Get your enode info (in another terminal)
./get-node-info.sh
# Output: enode://abc123...@192.168.1.100:30305
```

### Node B (Remote)
```bash
# 1. Start Q Coin node on same network
./dev-quick-start.sh

# 2. Connect to Node A
./connect-peer.sh "enode://abc123...@192.168.1.100:30305"
```

### Verification
Both nodes should now show 1+ connected peers:
```bash
./get-node-info.sh
# Should show: Peers: 1 connected
```

---

🎉 **Success!** Your Q Coin nodes are now connected and will sync blockchain data automatically. 