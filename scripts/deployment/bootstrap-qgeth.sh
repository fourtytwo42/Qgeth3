#!/bin/bash
# Q Geth Simplified Bootstrap Script
# Single-command VPS setup for Q Geth auto-updating service
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y

set -e

# Parse command line arguments
AUTO_CONFIRM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            AUTO_CONFIRM=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[BOOTSTRAP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script with sudo"
    echo ""
    echo "Usage:"
    echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash"
    echo ""
    echo "For non-interactive mode:"
    echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y"
    exit 1
fi

# Configuration
GITHUB_REPO="fourtytwo42/Qgeth3"
GITHUB_BRANCH="main"
INSTALL_DIR="/opt/qgeth"
PROJECT_DIR="$INSTALL_DIR/Qgeth3"
LOGS_DIR="$INSTALL_DIR/logs"

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

print_step "🚀 Q Geth Simplified Bootstrap"
echo ""
echo "This single script will:"
echo "  ✅ Clean up any existing installations"
echo "  ✅ Prepare VPS (memory, swap, dependencies)"
echo "  ✅ Clone Q Geth repository to /opt/qgeth/"
echo "  ✅ Build Q Geth with automated error recovery"
echo "  ✅ Create auto-updating systemd services"
echo "  ✅ Configure firewall for Q Geth operations"
echo ""

if [ "$AUTO_CONFIRM" != true ]; then
    echo -n "Proceed with installation? (y/N): "
    read -r RESPONSE
    if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# ===========================================
# STEP 1: CLEANUP EXISTING INSTALLATIONS
# ===========================================
print_step "🧹 Step 1: Cleanup existing installations"

# Stop and remove systemd services
print_step "Stopping Q Geth services..."
systemctl stop qgeth-node.service 2>/dev/null || true
systemctl stop qgeth-monitor.service 2>/dev/null || true
systemctl disable qgeth-node.service 2>/dev/null || true
systemctl disable qgeth-monitor.service 2>/dev/null || true
rm -f /etc/systemd/system/qgeth-*.service
systemctl daemon-reload

# Kill any remaining geth processes
print_step "Terminating geth processes..."
pkill -f "geth" 2>/dev/null || true
sleep 2
pkill -9 -f "geth" 2>/dev/null || true

# Remove installation directories and lock files
print_step "Removing installation directories..."
rm -rf "$INSTALL_DIR" 2>/dev/null || true
rm -f /tmp/qgeth-*.lock 2>/dev/null || true
rm -rf /tmp/qgeth-build* 2>/dev/null || true

print_success "✅ Cleanup completed"

# ===========================================
# STEP 2: VPS PREPARATION
# ===========================================
print_step "🔧 Step 2: VPS Preparation"

# Install dependencies
print_step "Installing dependencies..."
if command -v apt >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt update -qq
    DEBIAN_FRONTEND=noninteractive apt install -y git curl golang-go build-essential systemd ufw jq python3 python3-pip unzip
elif command -v yum >/dev/null 2>&1; then
    yum install -y git curl golang gcc systemd firewalld jq python3 python3-pip unzip
else
    print_error "Unsupported package manager. Please install dependencies manually."
    exit 1
fi

# Memory and swap check
print_step "Checking memory and swap..."
REQUIRED_MB=4096  # 4GB minimum total
TOTAL_MB=0
SWAP_TOTAL=0
CURRENT_TOTAL=0

if [ -f /proc/meminfo ]; then
    MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MB=$((MEM_TOTAL / 1024))
    
    if [ -f /proc/swaps ]; then
        SWAP_KB=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
        SWAP_TOTAL=$((SWAP_KB / 1024))
    fi
    
    CURRENT_TOTAL=$((TOTAL_MB + SWAP_TOTAL))
    
    echo "RAM: ${TOTAL_MB}MB, Swap: ${SWAP_TOTAL}MB, Total: ${CURRENT_TOTAL}MB"
    echo "Required: ${REQUIRED_MB}MB"
    
    # Create swap if needed (with 50MB tolerance)
    if [ $CURRENT_TOTAL -lt $((REQUIRED_MB - 50)) ]; then
        NEEDED_SWAP=$((REQUIRED_MB - TOTAL_MB))
        if [ $NEEDED_SWAP -gt $SWAP_TOTAL ]; then
            NEEDED_SWAP=$((NEEDED_SWAP - SWAP_TOTAL))
            print_step "Creating ${NEEDED_SWAP}MB swap file..."
            
            # Remove existing swap file if present
            if [ -f /swapfile ]; then
                swapoff /swapfile 2>/dev/null || true
                rm -f /swapfile
            fi
            
            # Create new swap
            fallocate -l "${NEEDED_SWAP}M" /swapfile || dd if=/dev/zero of=/swapfile bs=1024 count=$((NEEDED_SWAP * 1024))
            chmod 600 /swapfile
            mkswap /swapfile
            swapon /swapfile
            
            # Add to fstab for persistence
            if ! grep -q "/swapfile" /etc/fstab; then
                echo "/swapfile none swap sw 0 0" >> /etc/fstab
            fi
            
            print_success "✅ Swap file created"
        fi
    else
        print_success "✅ Sufficient memory available"
    fi
fi

print_success "✅ VPS preparation completed"

# ===========================================
# STEP 3: FIREWALL CONFIGURATION
# ===========================================
print_step "🔥 Step 3: Firewall configuration"

if command -v ufw >/dev/null 2>&1; then
    # Configure UFW
    print_step "Configuring UFW firewall..."
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow essential ports
    ufw allow 22/tcp comment 'SSH'
    ufw allow 8545/tcp comment 'Q Geth RPC API'
    ufw allow 8546/tcp comment 'Q Geth WebSocket API'
    ufw allow 30303/tcp comment 'Q Geth P2P TCP'
    ufw allow 30303/udp comment 'Q Geth P2P UDP'
    
    ufw --force enable
    print_success "✅ UFW firewall configured"
else
    print_warning "UFW not available, skipping firewall configuration"
fi

# ===========================================
# STEP 4: REPOSITORY SETUP
# ===========================================
print_step "📦 Step 4: Repository setup"

# Create directories
mkdir -p "$INSTALL_DIR" "$LOGS_DIR"
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$INSTALL_DIR"

# Clone repository
print_step "Cloning Q Geth repository..."
cd "$INSTALL_DIR"
if sudo -u "$ACTUAL_USER" git clone "https://github.com/$GITHUB_REPO.git"; then
    print_success "✅ Repository cloned successfully"
else
    print_error "Failed to clone repository"
    exit 1
fi

# Make scripts executable
cd "$PROJECT_DIR"
find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

print_success "✅ Repository setup completed"

# ===========================================
# STEP 5: BUILD Q GETH
# ===========================================
print_step "🔨 Step 5: Building Q Geth"

# Set up build environment
export QGETH_BUILD_TEMP="$INSTALL_DIR/build-temp"
mkdir -p "$QGETH_BUILD_TEMP"
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$QGETH_BUILD_TEMP"

# Build with automated error recovery
cd "$PROJECT_DIR/scripts/linux"
print_step "Building with automated error recovery..."

BUILD_SUCCESS=false
BUILD_ATTEMPTS=0
MAX_BUILD_ATTEMPTS=3

while [ $BUILD_ATTEMPTS -lt $MAX_BUILD_ATTEMPTS ] && [ "$BUILD_SUCCESS" = false ]; do
    BUILD_ATTEMPTS=$((BUILD_ATTEMPTS + 1))
    print_step "🚀 Build attempt $BUILD_ATTEMPTS/$MAX_BUILD_ATTEMPTS"
    
    if [ "$AUTO_CONFIRM" = true ]; then
        if sudo -u "$ACTUAL_USER" env QGETH_BUILD_TEMP="$QGETH_BUILD_TEMP" ./build-linux.sh geth -y; then
            BUILD_SUCCESS=true
        fi
    else
        if sudo -u "$ACTUAL_USER" env QGETH_BUILD_TEMP="$QGETH_BUILD_TEMP" ./build-linux.sh geth; then
            BUILD_SUCCESS=true
        fi
    fi
    
    if [ "$BUILD_SUCCESS" = false ] && [ $BUILD_ATTEMPTS -lt $MAX_BUILD_ATTEMPTS ]; then
        print_warning "Build attempt $BUILD_ATTEMPTS failed, applying recovery..."
        
        # Clean and retry
        cd "$PROJECT_DIR"
        rm -f geth geth.bin 2>/dev/null || true
        chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"
        sudo -u "$ACTUAL_USER" go clean -cache -modcache -testcache 2>/dev/null || true
        
        cd "$PROJECT_DIR/quantum-geth"
        sudo -u "$ACTUAL_USER" go mod tidy 2>/dev/null || true
        sudo -u "$ACTUAL_USER" go mod download 2>/dev/null || true
        
        cd "$PROJECT_DIR/scripts/linux"
        sleep 5
    fi
done

if [ "$BUILD_SUCCESS" = false ]; then
    print_error "Build failed after $MAX_BUILD_ATTEMPTS attempts"
    exit 1
fi

print_success "✅ Q Geth built successfully"

# ===========================================
# STEP 6: CREATE SYSTEMD SERVICES
# ===========================================
print_step "⚙️ Step 6: Creating systemd services"

# Create Q Geth node service
cat > /etc/systemd/system/qgeth-node.service << EOF
[Unit]
Description=Q Geth Node Service
After=network.target
Wants=network.target

[Service]
Type=exec
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin
ExecStart=$PROJECT_DIR/scripts/linux/start-geth.sh testnet
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=always
RestartSec=10
StandardOutput=append:$LOGS_DIR/geth-node.log
StandardError=append:$LOGS_DIR/geth-node.log

[Install]
WantedBy=multi-user.target
EOF

# Create GitHub monitor service
cat > /etc/systemd/system/qgeth-monitor.service << EOF
[Unit]
Description=Q Geth GitHub Monitor
After=network.target
Wants=network.target

[Service]
Type=exec
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin
ExecStart=/bin/bash -c 'while true; do \\
    REMOTE_COMMIT=\$(curl -s https://api.github.com/repos/$GITHUB_REPO/commits/$GITHUB_BRANCH | jq -r .sha 2>/dev/null); \\
    LOCAL_COMMIT=\$(git rev-parse HEAD 2>/dev/null); \\
    if [ "\$REMOTE_COMMIT" != "\$LOCAL_COMMIT" ] && [ "\$REMOTE_COMMIT" != "null" ] && [ -n "\$REMOTE_COMMIT" ]; then \\
        echo "[$(date)] New commit detected: \$REMOTE_COMMIT"; \\
        systemctl stop qgeth-node.service; \\
        git fetch origin && git reset --hard origin/$GITHUB_BRANCH; \\
        find . -name "*.sh" -type f -exec chmod +x {} \\; 2>/dev/null; \\
        cd scripts/linux && sudo -u $ACTUAL_USER env QGETH_BUILD_TEMP=$QGETH_BUILD_TEMP ./build-linux.sh geth -y; \\
        cd $PROJECT_DIR && systemctl start qgeth-node.service; \\
        echo "[$(date)] Update completed successfully"; \\
    else \\
        echo "[$(date)] No updates available"; \\
    fi; \\
    sleep 300; \\
done'
Restart=always
RestartSec=30
StandardOutput=append:$LOGS_DIR/github-monitor.log
StandardError=append:$LOGS_DIR/github-monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
systemctl daemon-reload
systemctl enable qgeth-node.service
systemctl enable qgeth-monitor.service

print_success "✅ Systemd services created and enabled"

# ===========================================
# STEP 7: START SERVICES
# ===========================================
print_step "🚀 Step 7: Starting services"

systemctl start qgeth-node.service
sleep 5
systemctl start qgeth-monitor.service

print_success "✅ Services started successfully"

# ===========================================
# FINAL STATUS AND INFORMATION
# ===========================================
echo ""
echo "========================================"
echo "🎉 Q Geth Bootstrap Completed Successfully!"
echo "========================================"
echo ""
echo "📋 Installation Summary:"
echo "  • Install Directory: $INSTALL_DIR"
echo "  • Project Directory: $PROJECT_DIR"
echo "  • Log Directory: $LOGS_DIR"
echo "  • User: $ACTUAL_USER"
echo ""
echo "⚙️ Services Created:"
echo "  • qgeth-node.service     - Q Geth blockchain node"
echo "  • qgeth-monitor.service  - GitHub auto-updater"
echo ""
echo "🔗 Network Access:"
echo "  • HTTP RPC API:  http://localhost:8545"
echo "  • WebSocket API: ws://localhost:8546"
echo "  • P2P Network:   port 30303"
echo ""
echo "📊 Service Management:"
echo "  sudo systemctl status qgeth-node.service"
echo "  sudo systemctl restart qgeth-node.service"
echo "  sudo systemctl logs -f qgeth-node.service"
echo ""
echo "📁 Log Files:"
echo "  tail -f $LOGS_DIR/geth-node.log"
echo "  tail -f $LOGS_DIR/github-monitor.log"
echo ""
echo "🔄 Auto-Update Features:"
echo "  • Monitors GitHub every 5 minutes"
echo "  • Automatically rebuilds on new commits"
echo "  • Restarts services after successful updates"
echo ""
echo "✅ Q Geth is now running and will auto-update!"
echo ""

# Final verification
print_step "🔍 Final verification"
sleep 3

if systemctl is-active --quiet qgeth-node.service; then
    print_success "✅ Q Geth node is running"
else
    print_warning "⚠️ Q Geth node is not running - check logs"
fi

if systemctl is-active --quiet qgeth-monitor.service; then
    print_success "✅ GitHub monitor is running"
else
    print_warning "⚠️ GitHub monitor is not running - check logs"
fi

echo ""
echo "Installation completed! 🚀" 