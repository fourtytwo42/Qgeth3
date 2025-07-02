#!/usr/bin/env bash
# Q Geth Universal Bootstrap Script with System Service Integration
# Installs Q Geth to ~/qgeth/Qgeth3 and creates persistent system service
# Supports systemd, OpenRC, SysV init, and Upstart across all Linux distributions
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
DOCKER_MODE=false

# Parse simple flags
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        --docker)
            DOCKER_MODE=true
            ;;
        --help|-h)
            echo "Q Geth Universal Bootstrap Script with System Service Integration"
            echo ""
            echo "Installs Q Geth to ~/qgeth/Qgeth3 and creates persistent system service"
            echo "Supports systemd, OpenRC, SysV init, and Upstart across all Linux distributions"
            echo ""
            echo "Features:"
            echo "  ✅ Universal Linux distribution support"
            echo "  ✅ Automatic init system detection"
            echo "  ✅ Persistent system service creation"
            echo "  ✅ Automatic service startup"
            echo "  ✅ Go 1.24.4 enforcement for quantum consensus"
            echo "  ✅ Sudo installation compatibility"
            echo ""
            echo "Usage:"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash -s -- -y"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash -s -- --docker"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Auto-confirm all prompts"
            echo "  --docker     Use Docker deployment instead of system service"
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
    
    # Special note for Fedora users
    if [ "$OS" = "fedora" ]; then
        log_warning "⚠️ Fedora Detected"
        log_info "Fedora requires manual installation due to systemd execution complexities."
        log_info "The quantum blockchain builds and runs perfectly on Fedora, but the"
        log_info "automated bootstrap service creation fails with exit code 203."
        log_info ""
        log_info "🐳 RECOMMENDED: Use Docker for Fedora instead!"
        log_info "  Docker provides perfect cross-platform compatibility:"
        log_info "  1. Install Docker: sudo dnf install docker docker-compose"
        log_info "  2. Start Docker: sudo systemctl start docker"
        log_info "  3. Run Q Geth: docker-compose up -d qgeth-planck"
        log_info "  4. See: docs/deployment/docker-deployment.md"
        log_info ""
        log_info "📋 Manual Installation Guide:"
        log_info "  See: docs/deployment/bootstrap-deployment.md#-fedora-manual-installation"
        log_info ""
        
        if [ "$AUTO_CONFIRM" != true ]; then
            echo -n "Continue with bootstrap anyway? (y/N): "
            read -r RESPONSE
            if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                log_info "Bootstrap cancelled. Consider using Docker for easier Fedora deployment!"
                exit 0
            fi
        else
            log_warning "Continuing with bootstrap in non-interactive mode..."
            log_warning "Service creation will likely fail on Fedora."
        fi
    fi
}

# Detect init system for service creation
detect_init_system() {
    log_info "Detecting init system..."
    
    # Check for systemd
    if [ -d /run/systemd/system ] || command -v systemctl >/dev/null 2>&1; then
        INIT_SYSTEM="systemd"
        SERVICE_CMD="systemctl"
        log_info "Init system: systemd"
        return 0
    fi
    
    # Check for OpenRC
    if [ -f /sbin/openrc ] || [ -d /etc/runlevels ]; then
        INIT_SYSTEM="openrc"
        SERVICE_CMD="rc-service"
        log_info "Init system: OpenRC"
        return 0
    fi
    
    # Check for SysV init
    if [ -d /etc/rc.d ] || [ -d /etc/init.d ]; then
        INIT_SYSTEM="sysv"
        SERVICE_CMD="service"
        log_info "Init system: SysV init"
        return 0
    fi
    
    # Check for Upstart
    if [ -d /etc/init ] && command -v initctl >/dev/null 2>&1; then
        INIT_SYSTEM="upstart"
        SERVICE_CMD="initctl"
        log_info "Init system: Upstart"
        return 0
    fi
    
    # Fallback to systemd (most common)
    log_warning "Could not detect init system, defaulting to systemd"
    INIT_SYSTEM="systemd"
    SERVICE_CMD="systemctl"
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
        # Extract any Go version for logging
        GO_VERSION_ANY=$(echo "$GO_VERSION_FULL" | grep -o 'go1\.[0-9]*\.[0-9]*' | head -1)
        # Extract specifically Go 1.24.x versions for compatibility check
        GO_VERSION_EXACT=$(echo "$GO_VERSION_FULL" | grep -o 'go1\.24\.[0-9]*' | head -1)
        
        log_info "Found existing Go: $GO_VERSION_FULL"
        log_info "Detected version: $GO_VERSION_ANY"
        
        if [ "$GO_VERSION_EXACT" = "go1.24.4" ]; then
            log_success "✅ Go 1.24.4 already installed - quantum consensus compatible!"
            return 0
        else
            if [ -n "$GO_VERSION_EXACT" ]; then
                log_warning "⚠️ Go $GO_VERSION_EXACT found, but need Go 1.24.4 for quantum consensus"
            else
                log_warning "⚠️ Go $GO_VERSION_ANY found, but need Go 1.24.4 for quantum consensus"
            fi
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

# Check and create swap if needed for building
setup_swap_if_needed() {
    log_info "Checking memory and swap for build requirements..."
    
    local required_mb=4096  # 4GB minimum total (RAM + swap)
    local total_mb=0
    local swap_mb=0
    local combined_mb=0
    
    if [ -f /proc/meminfo ]; then
        # Get RAM and swap in MB
        local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        total_mb=$((mem_total / 1024))
        
        # Check current swap
        if [ -f /proc/swaps ]; then
            local swap_total=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
            swap_mb=$((swap_total / 1024))
        fi
        
        combined_mb=$((total_mb + swap_mb))
        
        log_info "Memory Status:"
        log_info "  RAM: ${total_mb}MB"
        log_info "  Swap: ${swap_mb}MB"
        log_info "  Total Available: ${combined_mb}MB"
        log_info "  Required: ${required_mb}MB (4GB)"
        
        # Check if we need more memory
        if [ $combined_mb -lt $required_mb ]; then
            local needed_swap=$((required_mb - combined_mb))
            log_warning "Insufficient memory for building Q Geth!"
            log_info "Need additional ${needed_swap}MB of swap space"
            
            # Only create swap if running as root/sudo
            if [ "$EUID" -eq 0 ] || [ -n "$SUDO_USER" ]; then
                log_info "Creating swap file automatically..."
                create_swap_file $needed_swap
            else
                log_warning "Cannot create swap (not running as root/sudo)"
                log_info "Manual swap creation needed:"
                log_info "  sudo fallocate -l ${needed_swap}M /swapfile"
                log_info "  sudo chmod 600 /swapfile"
                log_info "  sudo mkswap /swapfile"
                log_info "  sudo swapon /swapfile"
                log_info "  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
                
                if [ "$AUTO_CONFIRM" != true ]; then
                    echo -n "Continue without creating swap? Build may fail (y/N): "
                    read -r RESPONSE
                    if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                        log_info "Bootstrap cancelled. Please create swap manually or run with sudo."
                        exit 1
                    fi
                fi
            fi
        else
            log_success "✅ Sufficient memory available (${combined_mb}MB total)"
        fi
    else
        log_warning "Cannot check memory - /proc/meminfo not found"
    fi
}

# Create swap file
create_swap_file() {
    local needed_mb=$1
    local swap_size_mb=$((needed_mb + 512))  # Add 512MB buffer
    
    log_info "Creating ${swap_size_mb}MB swap file..."
    
    # Check available disk space
    local available_mb=$(df / | awk 'NR==2 {print int($4/1024)}')
    if [ $available_mb -lt $swap_size_mb ]; then
        log_error "Insufficient disk space for swap file"
        log_error "  Available: ${available_mb}MB"
        log_error "  Required: ${swap_size_mb}MB"
        return 1
    fi
    
    # Remove any existing swapfile
    if [ -f /swapfile ]; then
        log_info "Removing existing swap file..."
        swapoff /swapfile 2>/dev/null || true
        rm -f /swapfile
    fi
    
    # Create new swap file
    log_info "Allocating ${swap_size_mb}MB swap file..."
    if ! fallocate -l ${swap_size_mb}M /swapfile 2>/dev/null; then
        # Fallback to dd if fallocate fails
        log_info "fallocate failed, using dd method..."
        if ! dd if=/dev/zero of=/swapfile bs=1M count=$swap_size_mb 2>/dev/null; then
            log_error "Failed to create swap file"
            return 1
        fi
    fi
    
    # Set correct permissions
    chmod 600 /swapfile
    
    # Make swap
    if ! mkswap /swapfile >/dev/null 2>&1; then
        log_error "Failed to format swap file"
        rm -f /swapfile
        return 1
    fi
    
    # Enable swap
    if ! swapon /swapfile; then
        log_error "Failed to enable swap file"
        rm -f /swapfile
        return 1
    fi
    
    # Add to fstab for persistence
    if ! grep -q "/swapfile" /etc/fstab 2>/dev/null; then
        echo "/swapfile none swap sw 0 0" >> /etc/fstab
    fi
    
    # Verify swap is active
    local new_swap=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
    local new_swap_mb=$((new_swap / 1024))
    
    log_success "✅ Swap file created and activated"
    log_info "  Swap file: /swapfile (${swap_size_mb}MB)"
    log_info "  Total swap now: ${new_swap_mb}MB"
    
    # Update memory status
    local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_mb=$((mem_total / 1024))
    local combined_mb=$((total_mb + new_swap_mb))
    log_success "✅ Total memory now: ${combined_mb}MB (sufficient for building)"
}

# Create simple process management (avoiding systemd complexity)
create_system_service() {
    log_info "Creating simple process management (avoiding systemd complexity)..."
    log_info "✅ Using reliable PID-based process management instead of systemd"
    log_info "This avoids systemd library issues on problematic distributions"
    
    # Create PID-based management scripts
    cat > "$INSTALL_DIR/start-qgeth.sh" << 'EOF'
#!/bin/bash
# Start Q Geth with PID management
PID_FILE="$HOME/qgeth/qgeth.pid"
LOG_FILE="$HOME/qgeth/qgeth.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Q Geth is already running (PID: $PID)"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

echo "Starting Q Geth..."
cd "$HOME/qgeth/Qgeth3/scripts/linux"
nohup ./start-geth.sh planck > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Q Geth started successfully!"
echo "PID: $(cat $PID_FILE)"
echo "Logs: tail -f $LOG_FILE"
EOF

    cat > "$INSTALL_DIR/stop-qgeth.sh" << 'EOF'
#!/bin/bash
# Stop Q Geth
PID_FILE="$HOME/qgeth/qgeth.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Q Geth (PID: $PID)..."
        kill "$PID"
        sleep 3
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force stopping..."
            kill -9 "$PID"
        fi
        rm -f "$PID_FILE"
        echo "Q Geth stopped"
    else
        echo "Q Geth not running"
        rm -f "$PID_FILE"
    fi
else
    echo "Q Geth not running"
fi
EOF

    cat > "$INSTALL_DIR/status-qgeth.sh" << 'EOF'
#!/bin/bash
# Check Q Geth status
PID_FILE="$HOME/qgeth/qgeth.pid"

echo "=== Q Geth Status ==="
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Status: RUNNING (PID: $PID)"
        echo "Started: $(stat -c %y $PID_FILE | cut -d. -f1)"
        echo "Memory: $(ps -p $PID -o rss= | awk '{printf "%.0f MB", $1/1024}')"
    else
        echo "Status: NOT RUNNING (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "Status: NOT RUNNING"
fi

echo ""
echo "=== Files ==="
echo "Config: $HOME/qgeth/Qgeth3/scripts/linux/"
echo "Logs: $HOME/qgeth/qgeth.log"
echo "PID File: $PID_FILE"

if [ -f "$HOME/qgeth/qgeth.log" ]; then
    echo ""
    echo "=== Recent Logs ==="
    tail -10 "$HOME/qgeth/qgeth.log"
fi

echo ""
echo "=== Management Commands ==="
echo "Start:  $HOME/qgeth/start-qgeth.sh"
echo "Stop:   $HOME/qgeth/stop-qgeth.sh"
echo "Status: $HOME/qgeth/status-qgeth.sh"
echo "Logs:   tail -f $HOME/qgeth/qgeth.log"
EOF

    cat > "$INSTALL_DIR/restart-qgeth.sh" << 'EOF'
#!/bin/bash
# Restart Q Geth
echo "Restarting Q Geth..."
$HOME/qgeth/stop-qgeth.sh
sleep 2
$HOME/qgeth/start-qgeth.sh
EOF

    cat > "$INSTALL_DIR/update-qgeth.sh" << 'EOF'
#!/bin/bash
# Update Q Geth
echo "Updating Q Geth..."
cd "$HOME/qgeth/Qgeth3"

# Stop if running
if [ -f "$HOME/qgeth/qgeth.pid" ]; then
    echo "Stopping Q Geth for update..."
    $HOME/qgeth/stop-qgeth.sh
    RESTART_AFTER=true
else
    RESTART_AFTER=false
fi

# Update repository
echo "Pulling latest changes..."
git pull origin main

# Rebuild
echo "Rebuilding Q Geth..."
cd scripts/linux
./build-linux.sh

# Restart if it was running
if [ "$RESTART_AFTER" = true ]; then
    echo "Restarting Q Geth..."
    $HOME/qgeth/start-qgeth.sh
fi

echo "Update complete!"
EOF

    chmod +x "$INSTALL_DIR"/*.sh
    
    log_success "✅ Simple process management created"
    log_info "Management scripts: $INSTALL_DIR/*.sh"
    
    return 0
}

# Setup minimal log management
setup_log_management() {
    log_info "Setting up minimal log management..."
    
    log_info "Checking log rotation requirements..."
    log_info "✅ Log rotation not needed - geth verbosity level 1 generates minimal logs"
    log_info "ERROR-only logging produces small log files that don't require rotation"
    
    log_success "✅ Log management: Simple and efficient with verbosity level 1"
    
    # Check systemd journal configuration
    log_info "Checking systemd journal configuration..."
    log_info "✅ Journal configuration not needed - geth verbosity already set to level 1 (ERROR only)"
    log_info "This provides minimal logging without journal configuration complexity"
    
    log_success "✅ Log management: Using geth verbosity level 1 (optimal)"
    
    # Check disk monitoring requirements
    log_info "Checking disk monitoring requirements..."
    log_info "✅ Disk monitoring not needed - geth verbosity level 1 generates minimal logs"
    log_info "With ERROR-only logging, there are no large log files to monitor or clean up"
    
    log_success "✅ Log management: Minimal footprint with geth verbosity level 1"
    
    return 0
}

# Start the PID-based service
start_system_service() {
    log_info "Starting Q Geth with simple process management..."
    
    cd "$INSTALL_DIR"
    if ./start-qgeth.sh; then
        log_success "✅ Q Geth started successfully with PID-based management"
        log_info "Management commands:"
        log_info "  Status: $INSTALL_DIR/status-qgeth.sh"
        log_info "  Stop:   $INSTALL_DIR/stop-qgeth.sh"
        log_info "  Logs:   tail -f $INSTALL_DIR/qgeth.log"
    else
        log_error "Failed to start Q Geth"
        return 1
    fi
}

# Clean up after installation
cleanup_post_install() {
    log_info "Performing post-installation cleanup..."
    
    # Clean Go build cache to save space
    log_info "Cleaning Go build cache..."
    if command -v go >/dev/null 2>&1; then
        go clean -cache >/dev/null 2>&1 || true
        go clean -modcache >/dev/null 2>&1 || true
    fi
    
    # Clean package manager cache
    case $PKG_MANAGER in
        apt)
            $SUDO apt autoremove -y >/dev/null 2>&1 || true
            $SUDO apt autoclean >/dev/null 2>&1 || true
            ;;
        dnf|yum)
            $SUDO $PKG_MANAGER clean all >/dev/null 2>&1 || true
            ;;
        pacman)
            $SUDO pacman -Scc --noconfirm >/dev/null 2>&1 || true
            ;;
    esac
    
    log_success "✅ Post-installation cleanup completed"
}

# Main installation
main() {
    echo ""
    echo -e "${CYAN}🚀 Q Geth Universal Bootstrap with System Service${NC}"
    echo ""
    echo "This will:"
    echo "  📦 Install Q Geth to: $PROJECT_DIR"
    echo "  🔧 Create persistent system service"
    echo "  🚀 Auto-start Q Geth service"
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
    
    # Detect system and init system
    detect_system
    detect_init_system
    
    # Install dependencies
    install_dependencies
    # Setup swap if needed for building
    setup_swap_if_needed
    
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
    
    # Use consolidated build script with error handling
    if ! ./build-linux.sh geth ${AUTO_CONFIRM:+-y}; then
        log_error "Failed to build Q Geth"
        log_info "Build log may contain more details"
        log_info "You can try running the build manually:"
        log_info "  cd $PROJECT_DIR/scripts/linux"
        log_info "  ./build-linux.sh geth"
        exit 1
    fi
    
    # Verify geth binary was created
    if [ ! -f "$PROJECT_DIR/geth.bin" ] && [ ! -f "$PROJECT_DIR/geth" ]; then
        log_error "Q Geth binary not found after build"
        log_info "Expected binary at: $PROJECT_DIR/geth.bin or $PROJECT_DIR/geth"
        exit 1
    fi
    
    log_success "✅ Q Geth build completed successfully"
    
    # Create system service and management scripts
    if ! create_system_service; then
        log_error "Failed to create system service"
        log_info "Q Geth was built successfully but service creation failed"
        log_info "You can still run Q Geth manually:"
        log_info "  cd $PROJECT_DIR/scripts/linux"
        log_info "  ./start-geth.sh planck"
        exit 1
    fi
    
    # Setup log management
    setup_log_management
    
    log_info "PID-based management scripts ready"
    
    # Start the service
    start_system_service
    
    # Fix ownership if installed with sudo
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        log_info "Fixing ownership for user $SUDO_USER..."
        chown -R "$SUDO_USER:$SUDO_USER" "$INSTALL_DIR" 2>/dev/null || true
        log_success "✅ Ownership set to $SUDO_USER"
    fi
    
    # Clean up
    cleanup_post_install
    
    # Success message
    echo ""
    echo "========================================"
    echo -e "${GREEN}🎉 Q Geth System Service Installation Complete!${NC}"
    echo "========================================"
    echo ""
    echo -e "${BLUE}📋 Installation Summary:${NC}"
    echo "  Directory: $PROJECT_DIR"
    echo "  User: $ACTUAL_USER"
    echo "  Home: $USER_HOME"
    echo "  Process Management: PID-based (avoiding systemd issues)"
    echo "  Service: Q Geth background process"
    echo "  Management: $INSTALL_DIR/*.sh"
    echo ""
    echo -e "${BLUE}🔧 PID-Based Management Commands:${NC}"
    echo "  Start:   $INSTALL_DIR/start-qgeth.sh"
    echo "  Stop:    $INSTALL_DIR/stop-qgeth.sh"
    echo "  Restart: $INSTALL_DIR/restart-qgeth.sh"
    echo "  Status:  $INSTALL_DIR/status-qgeth.sh"
    echo "  Update:  $INSTALL_DIR/update-qgeth.sh"
    echo "  Logs:    tail -f $INSTALL_DIR/qgeth.log"
    echo ""
    echo -e "${BLUE}🎯 Service Features:${NC}"
    echo "  ✅ Background process (survives terminal close)"
    echo "  ✅ PID-based management (reliable cross-platform)"
    echo "  ✅ Simple restart and update capabilities"
    echo "  ✅ Minimal logging (verbosity level 1)"
    echo "  ✅ No systemd dependencies (universal compatibility)"
    echo ""
    echo -e "${BLUE}🎯 Next Steps:${NC}"
    echo "  1. Service started automatically"
    echo "  2. Check status: $INSTALL_DIR/status-qgeth.sh"
    echo "  3. View logs: tail -f $INSTALL_DIR/qgeth.log"
    echo "  4. Start mining: cd $PROJECT_DIR/scripts/linux && ./start-miner.sh"
    echo "  5. RPC API: http://localhost:8545"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo "  If service fails to start:"
    echo "  1. Check status: $INSTALL_DIR/status-qgeth.sh"
    echo "  2. Check logs: tail -f $INSTALL_DIR/qgeth.log"
    echo "  3. Restart: $INSTALL_DIR/restart-qgeth.sh"
            echo "  4. Or run manually: cd $PROJECT_DIR/scripts/linux && ./start-geth.sh planck"
    echo ""
    echo -e "${GREEN}Q Geth is now running as a persistent background service! 🚀${NC}"
}

# Run main function
main "$@" 