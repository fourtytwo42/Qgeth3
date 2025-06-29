# Q Geth Docker Deployment Guide

Complete guide for running Q Geth quantum blockchain nodes using Docker containers on **Linux**, **Windows**, and **macOS**.

## 🐳 Why Docker?

### Universal Compatibility
- **Cross-Platform**: Works identically on Linux, Windows, and macOS
- **No Dependencies**: No need to install Go, build tools, or manage system packages
- **Isolated Environment**: Clean, reproducible deployments
- **Easy Scaling**: Run multiple nodes with different configurations

### Perfect for Fedora Users
Since Fedora is not supported by the bootstrap script due to systemd complexities, **Docker provides the perfect solution** for Fedora users to run Q Geth without any compatibility issues.

### Container Features
- **Multi-Stage Build**: Optimized container size (~50MB runtime image)
- **Security**: Runs as non-root user with minimal attack surface
- **Health Checks**: Built-in container health monitoring
- **Persistent Data**: Blockchain data survives container restarts
- **Resource Control**: Built-in memory and CPU limits

## 🚀 Quick Start

### Prerequisites
- **Docker**: Install from [docker.com](https://docs.docker.com/get-docker/)
- **Docker Compose**: Usually included with Docker Desktop

### 🔧 Smart Docker Scripts (Recommended)

**Linux:**
```bash
# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Make script executable
chmod +x scripts/linux/start-geth-docker.sh

# Start Q Geth testnet node
./scripts/linux/start-geth-docker.sh

# Start with mining enabled
./scripts/linux/start-geth-docker.sh --mining

# Start development node
./scripts/linux/start-geth-docker.sh devnet

# View help
./scripts/linux/start-geth-docker.sh --help
```

**Windows PowerShell:**
```powershell
# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Start Q Geth testnet node
.\scripts\windows\start-geth-docker.ps1

# Start with mining enabled
.\scripts\windows\start-geth-docker.ps1 -Mining

# Start development node
.\scripts\windows\start-geth-docker.ps1 devnet

# View help
.\scripts\windows\start-geth-docker.ps1 -Help
```

**Smart Script Features:**
- ✅ **Auto-build**: Builds containers if missing
- ✅ **Health checks**: Monitors container status
- ✅ **API testing**: Verifies connectivity
- ✅ **Management**: Easy status, logs, stop commands
- ✅ **Docker validation**: Checks Docker installation

### ⚙️ Manual Docker-Compose (Advanced)
```bash
# Clone repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Start Q Geth testnet node
docker-compose up -d qgeth-testnet

# Check status
docker-compose ps
docker-compose logs -f qgeth-testnet
```

### Quick Test
```bash
# Test API connectivity
curl http://localhost:8545 \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'

# Expected response:
# {"jsonrpc":"2.0","id":1,"result":"CoreGeth/v1.12.21-unstable/linux-amd64/go1.24.4"}
```

## 📋 Available Configurations

### Default Testnet Node
```bash
# Basic Q Geth node for network participation
docker-compose up -d qgeth-testnet

# Accessible at:
# - HTTP RPC: http://localhost:8545
# - WebSocket: ws://localhost:8546
# - P2P: port 30303
```

### Mining Node
```bash
# Q Geth node with quantum mining enabled
docker-compose --profile mining up -d qgeth-miner

# Accessible at:
# - HTTP RPC: http://localhost:8547
# - WebSocket: ws://localhost:8548
# - P2P: port 30304
```

### Development Node
```bash
# Isolated development node with debug APIs
docker-compose --profile dev up -d qgeth-dev

# Accessible at:
# - HTTP RPC: http://localhost:8549
# - WebSocket: ws://localhost:8550
# - P2P: port 30305
# - Features: Debug APIs, verbose logging, no peer discovery
```

### Multiple Nodes
```bash
# Run all configurations simultaneously
docker-compose --profile mining --profile dev up -d

# Or specific combinations
docker-compose up -d qgeth-testnet qgeth-miner
```

## 🛠️ Platform-Specific Instructions

### Linux (Including Fedora)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose (if not included)
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy Q Geth
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
docker-compose up -d qgeth-testnet
```

### Windows (Docker Desktop)
```powershell
# Install Docker Desktop from https://docs.docker.com/desktop/install/windows/
# Enable WSL2 backend for better performance

# Open PowerShell and deploy
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
docker-compose up -d qgeth-testnet

# Windows-specific: Use Docker Desktop GUI for monitoring
```

### Windows (Command Line only)
```cmd
REM For Windows Server or headless environments
REM Install Docker Engine following Microsoft documentation

git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
docker-compose up -d qgeth-testnet
```

### macOS
```bash
# Install Docker Desktop from https://docs.docker.com/desktop/install/mac/
# Or use Homebrew:
brew install --cask docker

# Deploy Q Geth
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3
docker-compose up -d qgeth-testnet
```

## 🔧 Configuration & Customization

### Environment Variables
Create a `.env` file for custom configurations:

```bash
# .env file
QGETH_NETWORK_ID=73235
QGETH_MINER_ADDRESS=0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A
QGETH_VERBOSITY=3
QGETH_MAX_PEERS=25
QGETH_HTTP_PORT=8545
QGETH_WS_PORT=8546
QGETH_P2P_PORT=30303
```

### Custom Mining Address
```bash
# Method 1: Environment variable
export MINER_ETHERBASE=0xYourAddressHere
docker-compose --profile mining up -d qgeth-miner

# Method 2: Edit docker-compose.yml
# Update the --miner.etherbase parameter in the qgeth-miner service
```

### Resource Limits
Add resource constraints to `docker-compose.yml`:

```yaml
services:
  qgeth-testnet:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Persistent Storage
```bash
# View persistent volumes
docker volume ls

# Backup blockchain data
docker run --rm \
  -v qgeth3_qgeth_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/qgeth-backup.tar.gz -C /data .

# Restore blockchain data
docker run --rm \
  -v qgeth3_qgeth_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/qgeth-backup.tar.gz -C /data
```

## 📊 Monitoring & Management

### Container Status
```bash
# View running containers
docker-compose ps

# View resource usage
docker stats

# View logs
docker-compose logs -f qgeth-testnet
docker-compose logs -f --tail 100 qgeth-testnet
```

### API Testing
```bash
# Test HTTP RPC API
curl -X POST http://localhost:8545 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}'

# Test WebSocket (using wscat)
npm install -g wscat
echo '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' | wscat -c ws://localhost:8546

# Test mining status
curl -X POST http://localhost:8545 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_mining","params":[],"id":1}'

# Get current block number
curl -X POST http://localhost:8545 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

### Health Checks
```bash
# Container health status
docker-compose ps

# Manual health check
docker exec qgeth-testnet wget --no-verbose --tries=1 --spider http://localhost:8545

# View health check logs
docker inspect qgeth-testnet | jq '.[0].State.Health'
```

## 🔄 Updates & Maintenance

### Update to Latest Version
```bash
# Stop containers
docker-compose down

# Update repository
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Start with new version
docker-compose up -d qgeth-testnet
```

### Clean Rebuild
```bash
# Stop and remove everything
docker-compose down -v --remove-orphans

# Remove old images
docker rmi qgeth:latest

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d qgeth-testnet
```

### Container Maintenance
```bash
# View container logs
docker-compose logs qgeth-testnet

# Execute commands in running container
docker-compose exec qgeth-testnet /bin/sh

# View container filesystem
docker-compose exec qgeth-testnet ls -la /opt/qgeth/

# Monitor resource usage
docker stats qgeth-testnet
```

## 🌐 Network Configurations

### Port Mapping Summary
| Service | HTTP RPC | WebSocket | P2P TCP | P2P UDP |
|---------|----------|-----------|---------|---------|
| qgeth-testnet | 8545 | 8546 | 30303 | 30303 |
| qgeth-miner | 8547 | 8548 | 30304 | 30304 |
| qgeth-dev | 8549 | 8550 | 30305 | 30305 |

### External Access
```bash
# Allow external connections (modify docker-compose.yml)
# Change "127.0.0.1:8545:8545" to "0.0.0.0:8545:8545"

# For cloud deployments, ensure firewall rules allow:
# - TCP 8545 (HTTP RPC)
# - TCP 8546 (WebSocket)
# - TCP/UDP 30303 (P2P)
```

### Custom Networks
```yaml
# Add to docker-compose.yml for custom network configuration
networks:
  qgeth-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
          gateway: 172.20.0.1
```

## 🔒 Security Considerations

### Container Security
- **Non-root User**: Containers run as user `qgeth` (UID 1000)
- **Minimal Image**: Based on Alpine Linux for reduced attack surface
- **No Privileged Access**: Containers run without elevated privileges
- **Network Isolation**: Containers use isolated Docker networks

### Production Deployment
```bash
# Use specific image tags instead of 'latest'
# Add resource limits
# Configure log rotation
# Use Docker secrets for sensitive data
# Regular security updates

# Example production docker-compose.yml excerpt:
services:
  qgeth-testnet:
    image: qgeth:v1.0.0  # Use specific version
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
```

### Firewall Configuration
```bash
# Linux firewall (ufw)
sudo ufw allow 8545/tcp   # HTTP RPC
sudo ufw allow 8546/tcp   # WebSocket  
sudo ufw allow 30303     # P2P

# Windows Firewall
# Use Windows Defender Firewall GUI or PowerShell:
# New-NetFirewallRule -DisplayName "Q Geth RPC" -Direction Inbound -Protocol TCP -LocalPort 8545
```

## 🚨 Troubleshooting

### Common Issues

**Container Won't Start**
```bash
# Check Docker daemon
docker --version
docker info

# Check image build
docker-compose build qgeth-testnet

# Check logs
docker-compose logs qgeth-testnet
```

**Port Conflicts**
```bash
# Check what's using the port
netstat -tulpn | grep 8545
lsof -i :8545

# Use different ports in docker-compose.yml
ports:
  - "8555:8545"  # Use 8555 instead of 8545
```

**API Not Responding**
```bash
# Check container health
docker-compose ps

# Test internal connectivity
docker-compose exec qgeth-testnet wget -qO- http://localhost:8545

# Check firewall/network settings
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats

# Increase resource limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

### Platform-Specific Issues

**Windows Docker Desktop**
- Ensure WSL2 backend is enabled for better performance
- Check Windows Defender Firewall settings
- Verify Docker Desktop has necessary permissions

**Linux SELinux (RHEL/CentOS/Fedora)**
```bash
# If SELinux causes issues:
sudo setsebool -P container_manage_cgroup 1
sudo semanage fcontext -a -t container_file_t "/path/to/qgeth/data(/.*)?"
sudo restorecon -R /path/to/qgeth/data
```

**macOS Performance**
- Ensure Docker Desktop has sufficient resources allocated
- Consider using Docker Machine for better performance on older Macs

## 📈 Performance Optimization

### Resource Allocation
```yaml
# Optimized docker-compose.yml for production
services:
  qgeth-testnet:
    deploy:
      resources:
        limits:
          memory: 4G      # Adjust based on available RAM
          cpus: '2.0'     # Adjust based on available CPUs
        reservations:
          memory: 2G
          cpus: '1.0'
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
```

### Storage Optimization
```bash
# Use SSD storage for better I/O performance
# Mount host SSD directory as volume:

volumes:
  - /fast/ssd/path:/opt/qgeth/data  # Linux
  - C:\fast\ssd\path:/opt/qgeth/data  # Windows
```

### Network Optimization
```bash
# For high-performance networking, use host networking (Linux only)
network_mode: "host"  # Only in docker-compose.yml on Linux

# Note: This bypasses Docker's network isolation
```

## 🎯 Use Cases

### Development Environment
```bash
# Quick development setup
docker-compose --profile dev up -d qgeth-dev

# Clean environment for each development session
docker-compose down -v && docker-compose --profile dev up -d qgeth-dev
```

### Testing Environment
```bash
# Run multiple isolated networks for testing
docker-compose up -d qgeth-testnet qgeth-dev

# Test different configurations simultaneously
```

### Production Deployment
```bash
# Production-ready deployment with monitoring
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Educational/Research
```bash
# Perfect for educational institutions
# Easy setup for students and researchers
# No complex dependency management
# Cross-platform compatibility
```

---

## 📞 Support

For Docker-specific issues:
- Check [Docker documentation](https://docs.docker.com/)
- Review container logs: `docker-compose logs`
- Verify system requirements and permissions

For Q Geth-specific issues:
- See main project troubleshooting guides
- Check API connectivity and blockchain sync status

---

**🐳 Professional quantum blockchain deployment made simple with Docker!**

**Ready to run Q Geth anywhere? Get started with `docker-compose up -d qgeth-testnet`!** 