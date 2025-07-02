#!/bin/bash
# Q Geth Ubuntu Bootstrap Script
# One-command setup for Q Geth on Ubuntu
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-ubuntu.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[BOOTSTRAP]${NC} $1"; }
log_success() { echo -e "${GREEN}[BOOTSTRAP]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[BOOTSTRAP]${NC} $1"; }
log_error() { echo -e "${RED}[BOOTSTRAP]${NC} $1"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This bootstrap script must be run as root"
    echo "Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-ubuntu.sh | sudo bash"
    exit 1
fi

log_info "Starting Q Geth Ubuntu Bootstrap..."
log_info "This will clone the repository and run the installation script"

# Determine installation directory
if [ -n "$SUDO_USER" ]; then
    INSTALL_DIR="/home/$SUDO_USER"
    USER_NAME="$SUDO_USER"
else
    INSTALL_DIR="/root"
    USER_NAME="root"
fi

log_info "Installing to: $INSTALL_DIR/Qgeth3"
log_info "Target user: $USER_NAME"

# Step 1: Install git if not present
if ! command -v git &> /dev/null; then
    log_info "Installing git..."
    apt update -qq
    apt install -y git
    log_success "Git installed"
fi

# Step 2: Remove existing installation if present
if [ -d "$INSTALL_DIR/Qgeth3" ]; then
    log_warning "Existing installation found, removing..."
    rm -rf "$INSTALL_DIR/Qgeth3"
    log_success "Existing installation removed"
fi

# Step 3: Clone repository
log_info "Cloning Q Geth repository..."
cd "$INSTALL_DIR"
git clone https://github.com/fourtytwo42/Qgeth3.git
log_success "Repository cloned successfully"

# Step 4: Enter directory and make script executable
log_info "Preparing installation script..."
cd "$INSTALL_DIR/Qgeth3"
chmod +x install-ubuntu.sh
log_success "Installation script prepared"

# Step 5: Run the installation script
log_info "Starting Q Geth installation..."
echo "========================================"
echo "  Running install-ubuntu.sh..."
echo "========================================"
./install-ubuntu.sh

# Bootstrap complete
echo ""
echo "========================================"
log_success "Q Geth Bootstrap Complete!"  
echo "========================================"
echo ""
echo "🎉 Installation Summary:"
echo "  ✅ Repository cloned to: $INSTALL_DIR/Qgeth3"
echo "  ✅ Q Geth installed and configured"
echo "  ✅ Systemd service created and started"
echo "  ✅ Ready for use!"
echo ""
echo "📁 Installation Location:"
echo "  Directory: $INSTALL_DIR/Qgeth3"
echo "  Scripts: $INSTALL_DIR/Qgeth3/scripts/linux/"
echo ""
echo "🚀 Next Steps:"
echo "  1. Check service status: sudo systemctl status qgeth-planck.service"
echo "  2. View logs: sudo journalctl -u qgeth-planck.service -f"
echo "  3. Test connection: curl -X POST -H \"Content-Type: application/json\" --data '{\"jsonrpc\":\"2.0\",\"method\":\"web3_clientVersion\",\"params\":[],\"id\":1}' http://localhost:8545"
echo ""
log_success "Q Geth is now running on Planck Network!" 