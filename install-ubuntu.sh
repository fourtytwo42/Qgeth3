#!/bin/bash
# Q Geth Ubuntu Installation Script
# Run this after: git clone https://github.com/fourtytwo42/Qgeth3.git
# Usage: cd Qgeth3 && sudo ./install-ubuntu.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (use sudo)"
    echo "Usage: sudo ./install-ubuntu.sh"
    exit 1
fi

# Get the actual user (not root)
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER="$SUDO_USER"
    USER_HOME="/home/$SUDO_USER"
else
    # If running directly as root (not via sudo)
    ACTUAL_USER="root"
    USER_HOME="/root"
    log_warning "Running directly as root. Service will run as root user."
fi

# Project directory
PROJECT_DIR="$(pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts/linux"

log_info "Starting Q Geth installation for user: $ACTUAL_USER"
log_info "Project directory: $PROJECT_DIR"

# Step 1: Install dependencies
log_info "Installing system dependencies..."
apt update
apt install -y git curl build-essential wget

log_success "System dependencies installed"

# Step 2: Install Go 1.24.4
log_info "Installing Go 1.24.4..."

# Remove any existing Go
apt remove golang-go -y 2>/dev/null || true
rm -rf /usr/local/go

# Download and install Go 1.24.4
cd /tmp
log_info "Downloading Go 1.24.4..."
wget -q https://golang.org/dl/go1.24.4.linux-amd64.tar.gz

log_info "Installing Go 1.24.4 to /usr/local/go..."
tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz
rm go1.24.4.linux-amd64.tar.gz

# Configure Go PATH for the user
log_info "Configuring Go environment..."
if ! grep -q "/usr/local/go/bin" "$USER_HOME/.bashrc"; then
    echo 'export PATH="/usr/local/go/bin:$PATH"' >> "$USER_HOME/.bashrc"
    echo 'export GOPATH="$HOME/go"' >> "$USER_HOME/.bashrc"
    echo 'export GOROOT="/usr/local/go"' >> "$USER_HOME/.bashrc"
fi

# Set Go environment for current session
export PATH="/usr/local/go/bin:$PATH"
export GOPATH="$USER_HOME/go"
export GOROOT="/usr/local/go"

# Verify Go installation
if /usr/local/go/bin/go version | grep -q "go1.24.4"; then
    log_success "Go 1.24.4 installed successfully"
else
    log_error "Go installation failed"
    exit 1
fi

# Step 3: Build Q Geth
log_info "Building Q Geth..."
cd "$PROJECT_DIR"

# Make scripts executable
chmod +x scripts/linux/*.sh

# Build as the actual user (not root)
log_info "Compiling Q Geth binaries..."
cd "$SCRIPTS_DIR"

# Run build as the actual user
sudo -u "$ACTUAL_USER" -H bash -c "
    export PATH='/usr/local/go/bin:\$PATH'
    export GOPATH='$USER_HOME/go'
    export GOROOT='/usr/local/go'
    cd '$SCRIPTS_DIR'
    ./build-linux.sh
"

# Verify build
if [ -f "$PROJECT_DIR/geth.bin" ] && [ -f "$PROJECT_DIR/geth" ]; then
    log_success "Q Geth built successfully"
else
    log_error "Build failed - binaries not found"
    exit 1
fi

# Step 4: Create systemd service
log_info "Creating systemd service..."

# Create systemd service file
tee /etc/systemd/system/qgeth-planck.service > /dev/null << EOF
[Unit]
Description=Q Geth Planck Network Node
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$SCRIPTS_DIR
Environment=PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=$USER_HOME
Environment=GOPATH=$USER_HOME/go
Environment=GOROOT=/usr/local/go
ExecStart=/bin/bash $SCRIPTS_DIR/start-geth.sh planck
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

log_success "Systemd service created"

# Step 5: Enable and start service
log_info "Enabling and starting Q Geth service..."

# Reload systemd
systemctl daemon-reload

# Enable service to start at boot
systemctl enable qgeth-planck.service

# Start service now
systemctl start qgeth-planck.service

# Wait a moment for service to start
sleep 3

# Check service status
if systemctl is-active --quiet qgeth-planck.service; then
    log_success "Q Geth service started successfully"
else
    log_warning "Service may not have started properly"
    log_info "Check status with: sudo systemctl status qgeth-planck.service"
fi

# Final setup
log_info "Setting correct ownership..."
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"
if [ -f "$USER_HOME/.bashrc" ]; then
    chown "$ACTUAL_USER:$ACTUAL_USER" "$USER_HOME/.bashrc"
fi

# Installation complete
echo ""
echo "========================================"
log_success "Q Geth Installation Complete!"
echo "========================================"
echo ""
echo "📋 Installation Summary:"
echo "  ✅ Go 1.24.4 installed to /usr/local/go"
echo "  ✅ Q Geth built successfully"
echo "  ✅ Systemd service created and started"
echo "  ✅ Service enabled for auto-start at boot"
echo ""
echo "🌐 Network Information:"
echo "  Network: Planck Network (Chain ID 73237)"
echo "  HTTP RPC: http://localhost:8545"
echo "  WebSocket: ws://localhost:8546"
echo "  P2P Port: 30307"
echo ""
echo "🔧 Service Management:"
echo "  Start:   sudo systemctl start qgeth-planck.service"
echo "  Stop:    sudo systemctl stop qgeth-planck.service"
echo "  Restart: sudo systemctl restart qgeth-planck.service"
echo "  Status:  sudo systemctl status qgeth-planck.service"
echo "  Logs:    sudo journalctl -u qgeth-planck.service -f"
echo ""
echo "📊 Test Commands:"
echo '  curl -X POST -H "Content-Type: application/json" \'
echo '    --data '"'"'{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'"'"' \'
echo '    http://localhost:8545'
echo ""
echo "💡 Next Steps:"
echo "  1. Restart your terminal to get Go in PATH"
echo "  2. Check service status: sudo systemctl status qgeth-planck.service"
echo "  3. View logs: sudo journalctl -u qgeth-planck.service -f"
echo ""
log_success "Installation complete! Q Geth is running on Planck Network." 