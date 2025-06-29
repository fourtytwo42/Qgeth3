#!/usr/bin/env bash
# Q Geth Simple Bootstrap Script
# Installs Q Geth to ~/qgeth/Qgeth3
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash

set -e

# Configuration
GITHUB_REPO="fourtytwo42/Qgeth3"

# Determine correct user home directory even when run with sudo
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    # Script run with sudo, use original user's home
    USER_HOME=$(eval echo "~$SUDO_USER")
    ACTUAL_USER="$SUDO_USER"
else
    # Script run normally or as root
    USER_HOME="$HOME"
    ACTUAL_USER="$USER"
fi

INSTALL_DIR="$USER_HOME/qgeth"
PROJECT_DIR="$INSTALL_DIR/Qgeth3"
AUTO_CONFIRM=false

# Parse simple flags
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        --help|-h)
            echo "Q Geth Simple Bootstrap Script"
            echo ""
            echo "Installs Q Geth to ~/qgeth/Qgeth3"
            echo ""
            echo "Usage:"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash -s -- -y"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Auto-confirm all prompts"
            echo "  --help       Show this help"
            echo ""
            exit 0
            ;;
    esac
done

# Colors
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect system and package manager
detect_system() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        log_error "Cannot detect operating system"
        exit 1
    fi
    
    # Detect package manager
    if command -v apt >/dev/null 2>&1; then
        PKG_MANAGER="apt"
        PKG_UPDATE="apt update"
        PKG_INSTALL="apt install -y"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
        PKG_UPDATE="dnf check-update || true"
        PKG_INSTALL="dnf install -y"
    elif command -v yum >/dev/null 2>&1; then
        PKG_MANAGER="yum"
        PKG_UPDATE="yum check-update || true"
        PKG_INSTALL="yum install -y"
    elif command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
        PKG_UPDATE="pacman -Sy"
        PKG_INSTALL="pacman -S --noconfirm"
    else
        log_error "No supported package manager found (apt/dnf/yum/pacman)"
        exit 1
    fi
    
    log_info "Detected: $OS (using $PKG_MANAGER)"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Check if running as root for package installation
    if [ "$EUID" -ne 0 ]; then
        log_info "Installing dependencies (may require sudo password)..."
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # Update package lists
    $SUDO $PKG_UPDATE
    
    # Install base dependencies
    case $PKG_MANAGER in
        apt)
            $SUDO $PKG_INSTALL git curl build-essential wget
            ;;
        dnf|yum)
            $SUDO $PKG_INSTALL git curl gcc gcc-c++ make wget
            ;;
        pacman)
            $SUDO $PKG_INSTALL git curl base-devel wget
            ;;
    esac
    
    # Install Go 1.24.4 specifically (required for quantum consensus compatibility)
    install_go_1_24
    
    log_success "Dependencies installed"
}

# Install Go 1.24.4 specifically for quantum blockchain consensus
install_go_1_24() {
    log_info "Installing Go 1.24.4 for quantum consensus compatibility..."
    
    # Check if Go 1.24.4 is already installed
    if command -v go >/dev/null 2>&1; then
        GO_VERSION_FULL=$(go version 2>/dev/null)
        GO_VERSION_EXACT=$(echo "$GO_VERSION_FULL" | grep -o 'go1\.24\.[0-9]*' | head -1)
        
        log_info "Found existing Go: $GO_VERSION_FULL"
        log_info "Extracted version: $GO_VERSION_EXACT"
        
        if [ "$GO_VERSION_EXACT" = "go1.24.4" ]; then
            log_success "✅ Go 1.24.4 already installed - quantum consensus compatible!"
            return 0
        else
            log_warning "⚠️ Go $GO_VERSION_EXACT found, but need Go 1.24.4 for quantum consensus"
            log_info "Upgrading to Go 1.24.4..."
        fi
    else
        log_info "No Go installation found, installing Go 1.24.4..."
    fi
    
    # Determine architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            GO_ARCH="amd64"
            ;;
        aarch64|arm64)
            GO_ARCH="arm64"
            ;;
        armv7l)
            GO_ARCH="armv6l"
            ;;
        i686)
            GO_ARCH="386"
            ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    GO_VERSION="1.24.4"
    GO_TARBALL="go${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
    GO_URL="https://golang.org/dl/${GO_TARBALL}"
    
    log_info "Downloading Go $GO_VERSION for $GO_ARCH..."
    
    # Download Go
    cd /tmp
    if ! wget -q "$GO_URL"; then
        log_error "Failed to download Go $GO_VERSION"
        exit 1
    fi
    
    # Verify download
    if [ ! -f "$GO_TARBALL" ]; then
        log_error "Go tarball not found after download"
        exit 1
    fi
    
    # Remove existing Go installation
    $SUDO rm -rf /usr/local/go
    
    # Install Go 1.24
    log_info "Installing Go $GO_VERSION to /usr/local/go..."
    $SUDO tar -C /usr/local -xzf "$GO_TARBALL"
    
    # Clean up
    rm -f "$GO_TARBALL"
    
    # Add Go to PATH for current session
    export PATH="/usr/local/go/bin:$PATH"
    
    # Add Go to system PATH
    if [ ! -f /etc/profile.d/go.sh ]; then
        $SUDO tee /etc/profile.d/go.sh > /dev/null << 'EOF'
#!/bin/bash
# Go 1.24.4 for Quantum Blockchain Consensus
export PATH="/usr/local/go/bin:$PATH"
export GOPATH="$HOME/go"
export GOROOT="/usr/local/go"
EOF
        $SUDO chmod +x /etc/profile.d/go.sh
    fi
    
    # Verify installation
    if command -v go >/dev/null 2>&1; then
        GO_INSTALLED_VERSION=$(go version 2>/dev/null)
        log_success "✅ Go installed successfully: $GO_INSTALLED_VERSION"
        
        # Verify it's Go 1.24.4
        if echo "$GO_INSTALLED_VERSION" | grep -q "go1.24.4"; then
            log_success "✅ Go 1.24.4 confirmed for quantum consensus compatibility"
        else
            log_warning "⚠️ Go version verification: got $GO_INSTALLED_VERSION, expected go1.24.4"
        fi
    else
        log_error "Go installation verification failed"
        exit 1
    fi
    
    log_info "Go 1.24.4 installation complete"
}

# Main installation
main() {
    echo ""
    echo -e "${CYAN}🚀 Q Geth Simple Bootstrap${NC}"
    echo ""
    echo "This will install Q Geth to: $PROJECT_DIR"
    echo ""
    
    # Confirm installation
    if [ "$AUTO_CONFIRM" != true ]; then
        echo -n "Continue? (y/N): "
        read -r RESPONSE
        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Installation cancelled."
            exit 0
        fi
    fi
    
    # Check for existing installation
    if [ -d "$PROJECT_DIR" ]; then
        log_warning "Existing installation found"
        if [ "$AUTO_CONFIRM" != true ]; then
            echo -n "Remove and reinstall? (y/N): "
            read -r RESPONSE
            if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                rm -rf "$PROJECT_DIR"
                log_info "Removed existing installation"
            else
                log_error "Installation cancelled"
                exit 1
            fi
        else
            rm -rf "$PROJECT_DIR"
            log_info "Auto-removed existing installation"
        fi
    fi
    
    # Detect system
    detect_system
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    log_info "Creating directories..."
    mkdir -p "$INSTALL_DIR"
    
    # Clone repository
    log_info "Cloning Q Geth repository..."
    cd "$INSTALL_DIR"
    git clone "https://github.com/$GITHUB_REPO.git"
    
    # Make scripts executable
    cd "$PROJECT_DIR"
    find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
    
    # Build Q Geth
    log_info "Building Q Geth..."
    cd "$PROJECT_DIR/scripts/linux"
    
    # Use consolidated build script
    ./build-linux.sh geth ${AUTO_CONFIRM:+-y}
    
    # Create simple service management scripts
    log_info "Creating management scripts..."
    
    cat > "$INSTALL_DIR/start-qgeth.sh" << EOF
#!/bin/bash
# Start Q Geth Node for user: $ACTUAL_USER
QGETH_DIR="$INSTALL_DIR"
cd "\$QGETH_DIR/Qgeth3/scripts/linux"
nohup ./start-geth.sh testnet > "\$QGETH_DIR/geth.log" 2>&1 &
echo \$! > "\$QGETH_DIR/qgeth.pid"
echo "Q Geth started in background"
echo "PID: \$(cat \"\$QGETH_DIR/qgeth.pid\")"
echo "Logs: tail -f \"\$QGETH_DIR/geth.log\""
EOF
    
    cat > "$INSTALL_DIR/stop-qgeth.sh" << EOF
#!/bin/bash
# Stop Q Geth Node for user: $ACTUAL_USER
QGETH_DIR="$INSTALL_DIR"
if [ -f "\$QGETH_DIR/qgeth.pid" ]; then
    PID=\$(cat "\$QGETH_DIR/qgeth.pid")
    if kill -0 \$PID 2>/dev/null; then
        kill \$PID
        echo "Q Geth stopped (PID: \$PID)"
        rm -f "\$QGETH_DIR/qgeth.pid"
    else
        echo "Q Geth not running"
        rm -f "\$QGETH_DIR/qgeth.pid"
    fi
else
    echo "Q Geth PID file not found"
fi
EOF
    
    cat > "$INSTALL_DIR/status-qgeth.sh" << EOF
#!/bin/bash
# Check Q Geth Status for user: $ACTUAL_USER
QGETH_DIR="$INSTALL_DIR"
if [ -f "\$QGETH_DIR/qgeth.pid" ]; then
    PID=\$(cat "\$QGETH_DIR/qgeth.pid")
    if kill -0 \$PID 2>/dev/null; then
        echo "Q Geth is running (PID: \$PID)"
        echo "Logs: tail -f \"\$QGETH_DIR/geth.log\""
    else
        echo "Q Geth not running (stale PID file)"
        rm -f "\$QGETH_DIR/qgeth.pid"
    fi
else
    echo "Q Geth is not running"
fi
EOF
    
    chmod +x "$INSTALL_DIR"/*.sh
    
    # Fix ownership if installed with sudo
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        log_info "Fixing ownership for user $SUDO_USER..."
        chown -R "$SUDO_USER:$SUDO_USER" "$INSTALL_DIR" 2>/dev/null || true
        log_success "✅ Ownership set to $SUDO_USER"
    fi
    
    # Success message
    echo ""
    echo "========================================"
    echo -e "${GREEN}🎉 Q Geth Installation Complete!${NC}"
    echo "========================================"
    echo ""
    echo -e "${BLUE}📋 Installation Summary:${NC}"
    echo "  Directory: $PROJECT_DIR"
    echo "  User: $ACTUAL_USER"
    echo "  Home: $USER_HOME"
    echo "  Management: $INSTALL_DIR/*.sh"
    echo ""
    echo -e "${BLUE}🔧 Quick Commands (for user $ACTUAL_USER):${NC}"
    echo "  Start:  $INSTALL_DIR/start-qgeth.sh"
    echo "  Stop:   $INSTALL_DIR/stop-qgeth.sh"
    echo "  Status: $INSTALL_DIR/status-qgeth.sh"
    echo "  Logs:   tail -f $INSTALL_DIR/geth.log"
    echo ""
    echo -e "${BLUE}🎯 Next Steps:${NC}"
    echo "  1. Start Q Geth: $INSTALL_DIR/start-qgeth.sh"
    echo "  2. Check status: $INSTALL_DIR/status-qgeth.sh"
    echo "  3. Start mining: cd $PROJECT_DIR/scripts/linux && ./start-miner.sh"
    echo "  4. RPC API: http://localhost:8545"
    echo ""
    echo -e "${GREEN}Ready to go! 🚀${NC}"
}

# Run main function
main "$@" 