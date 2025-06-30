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
        log_info "  3. Run Q Geth: docker-compose up -d qgeth-testnet"
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
            
            # Ignore small differences under 10MB (threshold check)
            if [ $needed_swap -lt 10 ]; then
                log_success "✅ Memory difference is only ${needed_swap}MB (under 10MB threshold)"
                log_success "✅ Sufficient memory available (${combined_mb}MB total)"
            else
                log_warning "Insufficient memory for building Q Geth!"
                log_info "Need additional ${needed_swap}MB of swap space"
                
                # Only create swap if running as root/sudo
                if [ "$EUID" -eq 0 ] || [ -n "$SUDO_USER" ]; then
                    log_info "Creating swap file automatically..."
                    create_swap_file $needed_swap $swap_mb
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
    local existing_swap_mb=${2:-0}  # Second parameter for existing swap size, default to 0
    
    # Calculate new swap size: existing swap + needed swap + 512MB buffer
    local swap_size_mb=$((existing_swap_mb + needed_mb + 512))
    
    log_info "Creating ${swap_size_mb}MB swap file..."
    log_info "  Existing swap: ${existing_swap_mb}MB"
    log_info "  Additional needed: ${needed_mb}MB"
    log_info "  Buffer: 512MB"
    log_info "  Total new swap: ${swap_size_mb}MB"
    
    # Check available disk space
    local available_mb=$(df / | awk 'NR==2 {print int($4/1024)}')
    if [ $available_mb -lt $swap_size_mb ]; then
        log_error "Insufficient disk space for swap file"
        log_error "  Available: ${available_mb}MB"
        log_error "  Required: ${swap_size_mb}MB"
        return 1
    fi
    
    # Remove any existing swapfile (we'll replace it with larger one)
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

# Create system service based on detected init system
create_system_service() {
    log_info "Creating Q Geth system service ($INIT_SYSTEM)..."
    
    # Determine if we need sudo for service creation
    SUDO_CMD=""
    if [ "$EUID" -ne 0 ]; then
        SUDO_CMD="sudo"
    fi
    
    case $INIT_SYSTEM in
        "systemd")
            create_systemd_service
            ;;
        "openrc")
            create_openrc_service
            ;;
        "sysv")
            create_sysv_service
            ;;
        "upstart")
            create_upstart_service
            ;;
        *)
            log_error "Unsupported init system: $INIT_SYSTEM"
            return 1
            ;;
    esac
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."
    
    # Ensure start script is executable 
    if [ -f "$PROJECT_DIR/scripts/linux/start-geth.sh" ]; then
        chmod +x "$PROJECT_DIR/scripts/linux/start-geth.sh"
    else
        log_error "Start script not found: $PROJECT_DIR/scripts/linux/start-geth.sh"
        return 1
    fi
    
    # Create service wrapper script that handles process cleanup
    cat > "$PROJECT_DIR/scripts/linux/systemd-start-geth.sh" << 'WRAPPER_EOF'
#!/bin/bash
# Systemd Service Wrapper for Q Geth
# Handles automatic process cleanup before starting

echo "[$(date)] Q Geth systemd service starting..."

# Clean up any existing geth processes (but not this startup script)
echo "[$(date)] Cleaning up existing geth processes..."
pkill -f "geth.bin" >/dev/null 2>&1 || true
pkill -f "geth --" >/dev/null 2>&1 || true

# Clean up any stale geth processes by name (not command line)
pgrep -x "geth" >/dev/null 2>&1 && pkill -x "geth" >/dev/null 2>&1 || true

# Wait a moment for cleanup
sleep 2

# Start Q Geth
echo "[$(date)] Starting Q Geth..."
exec ./start-geth.sh testnet
WRAPPER_EOF
    
    chmod +x "$PROJECT_DIR/scripts/linux/systemd-start-geth.sh"
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs" 2>/dev/null || true
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        chown -R "$SUDO_USER:$SUDO_USER" "$PROJECT_DIR/logs" 2>/dev/null || true
    fi
    
    $SUDO_CMD tee /etc/systemd/system/qgeth.service > /dev/null << EOF
[Unit]
Description=Q Geth Quantum Blockchain Node
Documentation=https://github.com/fourtytwo42/Qgeth3
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR/scripts/linux
ExecStart=/bin/bash $PROJECT_DIR/scripts/linux/systemd-start-geth.sh
Restart=on-failure
RestartSec=10s
TimeoutStartSec=60s
TimeoutStopSec=30s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qgeth
SyslogLevel=info
SyslogLevelPrefix=true

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$USER_HOME
ReadWritePaths=/tmp
ReadWritePaths=/var/tmp
ReadWritePaths=$PROJECT_DIR/logs

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Environment
Environment=HOME=$USER_HOME
Environment=PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

    # Validate service file was created correctly
    if [ ! -f "/etc/systemd/system/qgeth.service" ]; then
        log_error "Failed to create systemd service file"
        return 1
    fi
    
    # Check if ExecStart path exists and is executable
    local exec_start_script="$PROJECT_DIR/scripts/linux/systemd-start-geth.sh"
    if [ ! -f "$exec_start_script" ]; then
        log_error "ExecStart script missing: $exec_start_script"
        return 1
    fi
    
    if [ ! -x "$exec_start_script" ]; then
        log_warning "ExecStart script not executable, fixing..."
        chmod +x "$exec_start_script"
        if [ ! -x "$exec_start_script" ]; then
            log_error "Cannot make ExecStart script executable: $exec_start_script"
            return 1
        fi
    fi
    
    # Reload systemd and enable service
    if ! $SUDO_CMD systemctl daemon-reload; then
        log_error "Failed to reload systemd"
        return 1
    fi
    
    if ! $SUDO_CMD systemctl enable qgeth.service; then
        log_error "Failed to enable systemd service"
        return 1
    fi
    
    log_success "✅ Systemd service created and enabled"
    return 0
}

# Create OpenRC service
create_openrc_service() {
    log_info "Creating OpenRC service..."
    
    $SUDO_CMD tee /etc/init.d/qgeth > /dev/null << EOF
#!/sbin/openrc-run
# Q Geth Quantum Blockchain Node

name="Q Geth"
description="Quantum Blockchain Node"

user="$ACTUAL_USER"
group="$ACTUAL_USER"
directory="$PROJECT_DIR/scripts/linux"
command="$PROJECT_DIR/scripts/linux/start-geth.sh"
command_args="testnet"
command_background="yes"
pidfile="/var/run/qgeth.pid"
output_log="/var/log/qgeth.log"
error_log="/var/log/qgeth.error.log"

depend() {
    need net
    after firewall
}

start_pre() {
    checkpath --directory --owner \$user:\$group --mode 0755 \$(dirname \$pidfile)
    checkpath --file --owner \$user:\$group --mode 0644 \$output_log \$error_log
}
EOF
    
    $SUDO_CMD chmod +x /etc/init.d/qgeth
    
    if ! $SUDO_CMD rc-update add qgeth default; then
        log_error "Failed to enable OpenRC service"
        return 1
    fi
    
    log_success "✅ OpenRC service created and enabled"
    return 0
}

# Create SysV init script
create_sysv_service() {
    log_info "Creating SysV init script..."
    
    $SUDO_CMD tee /etc/init.d/qgeth > /dev/null << EOF
#!/bin/bash
# Q Geth        Quantum Blockchain Node
# chkconfig: 35 80 20
# description: Q Geth Quantum Blockchain Node
#

. /etc/rc.d/init.d/functions

USER="$ACTUAL_USER"
DAEMON="Q Geth"
ROOT_DIR="$PROJECT_DIR/scripts/linux"

DAEMON_PATH="\$ROOT_DIR/start-geth.sh"
DAEMON_ARGS="testnet"
PIDFILE="/var/run/qgeth.pid"
LOCKFILE="/var/lock/subsys/qgeth"

start() {
    printf "%-50s" "Starting \$DAEMON: "
    if [ -f \$PIDFILE ] && kill -0 \$(cat \$PIDFILE); then
        printf '%s\n' "Already running [\$(cat \$PIDFILE)]"
        return 1
    fi
    daemon --user "\$USER" --pidfile="\$PIDFILE" "\$DAEMON_PATH \$DAEMON_ARGS" && echo_success || echo_failure
    RETVAL=\$?
    echo
    [ \$RETVAL -eq 0 ] && touch \$LOCKFILE
    return \$RETVAL
}

stop() {
    printf "%-50s" "Shutting down \$DAEMON: "
    pid=\$(ps -aefw | grep "\$DAEMON" | grep -v " grep " | awk '{print \$2}')
    kill -9 \$pid > /dev/null 2>&1
    [ \$? -eq 0 ] && echo_success || echo_failure
    RETVAL=\$?
    echo
    [ \$RETVAL -eq 0 ] && rm -f \$LOCKFILE \$PIDFILE
    return \$RETVAL
}

restart() {
    stop
    start
}

status() {
    if [ -f \$PIDFILE ] && kill -0 \$(cat \$PIDFILE); then
        echo "\$DAEMON is running [\$(cat \$PIDFILE)]"
    else
        echo "\$DAEMON is stopped"
        RETVAL=1
    fi
}

case "\$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: {\$0} {start|stop|status|restart}"
        exit 1
        ;;
esac

exit \$RETVAL
EOF
    
    $SUDO_CMD chmod +x /etc/init.d/qgeth
    
    # Enable service for different runlevels
    SERVICE_CREATED=true
    for runlevel in 3 4 5; do
        if ! $SUDO_CMD ln -sf /etc/init.d/qgeth "/etc/rc${runlevel}.d/S80qgeth" 2>/dev/null; then
            log_warning "Failed to create start link for runlevel $runlevel"
            SERVICE_CREATED=false
        fi
    done
    for runlevel in 0 1 2 6; do
        if ! $SUDO_CMD ln -sf /etc/init.d/qgeth "/etc/rc${runlevel}.d/K20qgeth" 2>/dev/null; then
            log_warning "Failed to create stop link for runlevel $runlevel"
        fi
    done
    
    if [ "$SERVICE_CREATED" = true ]; then
        log_success "✅ SysV init script created and enabled"
        return 0
    else
        log_error "Failed to fully enable SysV service"
        return 1
    fi
}

# Create Upstart service
create_upstart_service() {
    log_info "Creating Upstart service..."
    
    $SUDO_CMD tee /etc/init/qgeth.conf > /dev/null << EOF
# Q Geth Quantum Blockchain Node

description "Quantum Blockchain Node"
author "Q Geth Team"

start on runlevel [2345]
stop on runlevel [!2345]

setuid $ACTUAL_USER
setgid $ACTUAL_USER

chdir $PROJECT_DIR/scripts/linux

respawn
respawn limit 10 5

exec $PROJECT_DIR/scripts/linux/start-geth.sh testnet

pre-start script
    echo "[\$(date)] Starting Q Geth" >> /var/log/qgeth.log
end script

pre-stop script
    echo "[\$(date)] Stopping Q Geth" >> /var/log/qgeth.log
end script
EOF
    
    log_success "✅ Upstart service created"
    return 0
}

# Install Docker and Docker Compose
install_docker() {
    log_info "Installing Docker and Docker Compose..."
    
    # Check if running as root for package installation
    if [ "$EUID" -ne 0 ]; then
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # Check if Docker is already installed
    if command -v docker >/dev/null 2>&1; then
        DOCKER_VERSION=$(docker --version 2>/dev/null)
        log_info "Found existing Docker: $DOCKER_VERSION"
        
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            log_success "✅ Docker is already installed and running"
        else
            log_info "Docker is installed but daemon not running, starting..."
            $SUDO systemctl start docker
            $SUDO systemctl enable docker
            log_success "✅ Docker daemon started and enabled"
        fi
    else
        log_info "Installing Docker..."
        
        case $PKG_MANAGER in
            apt)
                # Install Docker using official script (most reliable)
                curl -fsSL https://get.docker.com -o get-docker.sh
                $SUDO sh get-docker.sh
                rm get-docker.sh
                
                # Start and enable Docker
                $SUDO systemctl start docker
                $SUDO systemctl enable docker
                ;;
            dnf)
                $SUDO $PKG_INSTALL docker docker-compose
                $SUDO systemctl start docker
                $SUDO systemctl enable docker
                ;;
            yum)
                $SUDO $PKG_INSTALL docker docker-compose
                $SUDO systemctl start docker
                $SUDO systemctl enable docker
                ;;
            pacman)
                $SUDO $PKG_INSTALL docker docker-compose
                $SUDO systemctl start docker
                $SUDO systemctl enable docker
                ;;
        esac
        
        # Add user to docker group if not root
        if [ "$ACTUAL_USER" != "root" ]; then
            $SUDO usermod -aG docker "$ACTUAL_USER"
            log_info "Added user $ACTUAL_USER to docker group"
        fi
        
        log_success "✅ Docker installed successfully"
    fi
    
    # Check Docker Compose
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_VERSION=$(docker-compose --version 2>/dev/null)
        log_success "✅ Docker Compose found: $COMPOSE_VERSION"
    else
        log_info "Installing Docker Compose..."
        
        # Install Docker Compose
        COMPOSE_VERSION="v2.24.0"
        ARCH=$(uname -m)
        case $ARCH in
            x86_64)
                COMPOSE_ARCH="x86_64"
                ;;
            aarch64|arm64)
                COMPOSE_ARCH="aarch64"
                ;;
            *)
                log_error "Unsupported architecture for Docker Compose: $ARCH"
                exit 1
                ;;
        esac
        
        $SUDO curl -L "https://github.com/docker/compose/releases/download/$COMPOSE_VERSION/docker-compose-$(uname -s)-$COMPOSE_ARCH" \
            -o /usr/local/bin/docker-compose
        $SUDO chmod +x /usr/local/bin/docker-compose
        
        if command -v docker-compose >/dev/null 2>&1; then
            log_success "✅ Docker Compose installed successfully"
        else
            log_error "Failed to install Docker Compose"
            exit 1
        fi
    fi
    
    # Verify Docker is working
    if docker info >/dev/null 2>&1; then
        log_success "✅ Docker is working correctly"
    else
        log_error "Docker installation completed but daemon is not responding"
        log_info "You may need to:"
        log_info "  1. Restart your shell session"
        log_info "  2. Log out and back in (for group membership)"
        log_info "  3. Run: newgrp docker"
        exit 1
    fi
}

# Deploy Q Geth with Docker
deploy_docker() {
    log_info "Deploying Q Geth with Docker..."
    
    cd "$PROJECT_DIR"
    
    # Make Docker scripts executable
    chmod +x scripts/linux/start-geth-docker.sh 2>/dev/null || true
    
    # Build Docker containers
    log_info "Building Q Geth Docker containers..."
    if ! docker-compose build; then
        log_error "Failed to build Docker containers"
        exit 1
    fi
    
    log_success "✅ Docker containers built successfully"
    
    # Start Q Geth container
    log_info "Starting Q Geth Docker container..."
    if ! docker-compose up -d qgeth-testnet; then
        log_error "Failed to start Q Geth Docker container"
        exit 1
    fi
    
    # Wait for container to start
    log_info "Waiting for container to start..."
    sleep 5
    
    # Check container status
    if docker-compose ps | grep -q "qgeth-testnet.*Up"; then
        log_success "✅ Q Geth Docker container started successfully"
    else
        log_warning "⚠️ Container may not be healthy, checking logs..."
        docker-compose logs qgeth-testnet
        return 1
    fi
    
    # Test API connectivity
    log_info "Testing API connectivity..."
    sleep 3
    if curl -s http://localhost:8545 >/dev/null 2>&1; then
        log_success "✅ Q Geth API is accessible at http://localhost:8545"
    else
        log_info "API not yet ready (normal during startup)"
    fi
}

# Create Docker management scripts
create_docker_scripts() {
    log_info "Creating Docker management scripts..."
    
    cat > "$INSTALL_DIR/start-qgeth-docker.sh" << 'EOF'
#!/bin/bash
# Start Q Geth Docker Container
echo "Starting Q Geth Docker container..."
cd ~/qgeth/Qgeth3

if ! docker-compose up -d qgeth-testnet; then
    echo "❌ Failed to start Q Geth Docker container"
    exit 1
fi

echo "✅ Q Geth Docker container started"
echo "🌐 HTTP RPC: http://localhost:8545"
echo "🌐 WebSocket: ws://localhost:8546"
echo ""
echo "💡 Management Commands:"
echo "  Status: docker-compose ps"
echo "  Logs:   docker-compose logs -f qgeth-testnet"
echo "  Stop:   docker-compose stop qgeth-testnet"
EOF
    
    cat > "$INSTALL_DIR/stop-qgeth-docker.sh" << 'EOF'
#!/bin/bash
# Stop Q Geth Docker Container
echo "Stopping Q Geth Docker container..."
cd ~/qgeth/Qgeth3

docker-compose stop qgeth-testnet
echo "✅ Q Geth Docker container stopped"
EOF
    
    cat > "$INSTALL_DIR/restart-qgeth-docker.sh" << 'EOF'
#!/bin/bash
# Restart Q Geth Docker Container
echo "Restarting Q Geth Docker container..."
cd ~/qgeth/Qgeth3

docker-compose restart qgeth-testnet
echo "✅ Q Geth Docker container restarted"
EOF
    
    cat > "$INSTALL_DIR/status-qgeth-docker.sh" << 'EOF'
#!/bin/bash
# Check Q Geth Docker Container Status
echo "Q Geth Docker container status:"
echo ""
cd ~/qgeth/Qgeth3

docker-compose ps
echo ""

# Test API connectivity
echo "🔗 Testing API connectivity..."
if curl -s http://localhost:8545 >/dev/null 2>&1; then
    echo "✅ HTTP RPC API: http://localhost:8545 (accessible)"
else
    echo "❌ HTTP RPC API: http://localhost:8545 (not accessible)"
fi

if curl -s http://localhost:8546 >/dev/null 2>&1; then
    echo "✅ WebSocket API: ws://localhost:8546 (accessible)"
else
    echo "❌ WebSocket API: ws://localhost:8546 (not accessible)"
fi

echo ""
echo "📋 Docker Management Commands:"
echo "  Start:   docker-compose up -d qgeth-testnet"
echo "  Stop:    docker-compose stop qgeth-testnet"
echo "  Restart: docker-compose restart qgeth-testnet"
echo "  Logs:    docker-compose logs -f qgeth-testnet"
echo "  Status:  docker-compose ps"
echo "  Remove:  docker-compose down"
EOF
    
    cat > "$INSTALL_DIR/logs-qgeth-docker.sh" << 'EOF'
#!/bin/bash
# View Q Geth Docker Container Logs
# Usage: ./logs-qgeth-docker.sh [-f] [-n NUMBER]

FOLLOW_MODE=false
LINE_COUNT=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW_MODE=true
            shift
            ;;
        -n|--lines)
            LINE_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-f|--follow] [-n|--lines NUMBER]"
            echo "  -f, --follow    Follow log output (live mode)"
            echo "  -n, --lines     Number of lines to show (default: 50)"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

echo "Q Geth Docker container logs:"
echo ""
cd ~/qgeth/Qgeth3

if [ "$FOLLOW_MODE" = true ]; then
    echo "Following Q Geth Docker logs (press Ctrl+C to exit)..."
    docker-compose logs -f qgeth-testnet
else
    echo "Last $LINE_COUNT lines from Q Geth Docker:"
    docker-compose logs --tail="$LINE_COUNT" qgeth-testnet
fi
EOF

    cat > "$INSTALL_DIR/mining-qgeth-docker.sh" << 'EOF'
#!/bin/bash
# Start Q Geth Docker Container with Mining
echo "Starting Q Geth Docker container with mining enabled..."
cd ~/qgeth/Qgeth3

# Stop any existing containers
docker-compose down >/dev/null 2>&1

# Start mining container
if ! docker-compose --profile mining up -d qgeth-miner; then
    echo "❌ Failed to start Q Geth mining Docker container"
    exit 1
fi

echo "✅ Q Geth mining Docker container started"
echo "⚡ Mining: Enabled"
echo "🌐 HTTP RPC: http://localhost:8547"
echo "🌐 WebSocket: ws://localhost:8548"
echo ""
echo "💡 Management Commands:"
echo "  Status: docker-compose ps"
echo "  Logs:   docker-compose logs -f qgeth-miner"
echo "  Stop:   docker-compose stop qgeth-miner"
EOF
    
    chmod +x "$INSTALL_DIR"/*-qgeth-docker.sh
    log_success "✅ Docker management scripts created"
}

# Start and enable the system service
start_system_service() {
    log_info "Starting Q Geth system service..."
    
    # Determine if we need sudo for service management
    SUDO_CMD=""
    if [ "$EUID" -ne 0 ]; then
        SUDO_CMD="sudo"
    fi
    
    case $INIT_SYSTEM in
        "systemd")
            $SUDO_CMD systemctl start qgeth.service
            if $SUDO_CMD systemctl is-active --quiet qgeth.service; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start"
                log_info "Troubleshooting commands:"
                log_info "  Check status: sudo systemctl status qgeth.service"
                log_info "  View logs: sudo journalctl -xeu qgeth.service -f"
                log_info "  Restart service: sudo systemctl restart qgeth.service"
                log_info "  Or run manually: cd $PROJECT_DIR/scripts/linux && ./start-geth.sh testnet"
            fi
            ;;
        "openrc")
            $SUDO_CMD rc-service qgeth start
            if $SUDO_CMD rc-service qgeth status >/dev/null 2>&1; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start"
                log_info "Check status: sudo rc-service qgeth status"
                log_info "Or run manually: cd $PROJECT_DIR/scripts/linux && ./start-geth.sh testnet"
            fi
            ;;
        "sysv")
            $SUDO_CMD service qgeth start
            if $SUDO_CMD service qgeth status >/dev/null 2>&1; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start"
                log_info "Check status: sudo service qgeth status"
                log_info "Or run manually: cd $PROJECT_DIR/scripts/linux && ./start-geth.sh testnet"
            fi
            ;;
        "upstart")
            $SUDO_CMD initctl start qgeth
            if $SUDO_CMD initctl status qgeth | grep -q "start/running"; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start"
                log_info "Check status: sudo initctl status qgeth"
                log_info "Or run manually: cd $PROJECT_DIR/scripts/linux && ./start-geth.sh testnet"
            fi
            ;;
        *)
            log_error "Cannot start service for init system: $INIT_SYSTEM"
            return 1
            ;;
    esac
}

# ENHANCED LOG MANAGEMENT AND CLEANUP FUNCTIONS

# Setup log rotation for Q Geth
setup_log_rotation() {
    log_info "Setting up log rotation for Q Geth..."
    
    # Determine if we need sudo for log rotation setup
    SUDO_CMD=""
    if [ "$EUID" -ne 0 ]; then
        SUDO_CMD="sudo"
    fi
    
    # Create logrotate config for Q Geth
    $SUDO_CMD tee /etc/logrotate.d/qgeth > /dev/null << 'EOF'
# Q Geth log rotation configuration

# Q Geth user directory logs
/home/*/qgeth/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $(stat -c %U:%G %)
    maxsize 100M
    su $(stat -c %U) $(stat -c %G)
}

# Q Geth data directory logs
/home/*/.qcoin/*/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $(stat -c %U:%G %)
    maxsize 100M
    su $(stat -c %U) $(stat -c %G)
}
EOF
    
    log_success "✅ Log rotation configured for Q Geth"
}

# Setup systemd journal limits to prevent log explosion
setup_systemd_journal_limits() {
    log_info "Configuring systemd journal limits..."
    
    # Determine if we need sudo
    SUDO_CMD=""
    if [ "$EUID" -ne 0 ]; then
        SUDO_CMD="sudo"
    fi
    
    # Configure journald limits
    $SUDO_CMD mkdir -p /etc/systemd/journald.conf.d
    
    $SUDO_CMD tee /etc/systemd/journald.conf.d/qgeth-limits.conf > /dev/null << 'EOF'
# Q Geth systemd journal limits to prevent disk space issues
[Journal]
SystemMaxUse=500M
SystemMaxFileSize=50M
MaxRetentionSec=1week
MaxFileSec=1day
ForwardToSyslog=no
Compress=yes
EOF
    
    # Apply the changes
    if command -v systemctl >/dev/null 2>&1; then
        $SUDO_CMD systemctl restart systemd-journald
        log_success "✅ Systemd journal limits configured and applied"
    else
        log_success "✅ Systemd journal limits configured (restart required)"
    fi
}

# Create disk monitoring and cleanup script
create_disk_monitor() {
    log_info "Creating disk monitoring and cleanup system..."
    
    # Use project directory for monitoring
    MONITOR_DIR="$PROJECT_DIR"
    
    # Create disk monitor script
    cat > "$MONITOR_DIR/disk-monitor.sh" << 'EOF'
#!/bin/bash
# Q Geth Disk Space Monitor and Emergency Cleanup
# Automatically cleans up logs and temp files when disk usage is high

LOG_TAG="qgeth-disk-monitor"
CRITICAL_THRESHOLD=90
WARNING_THRESHOLD=80

# Function to log messages
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$LOG_TAG] $1" | logger -t "$LOG_TAG"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$LOG_TAG] $1"
}

# Get disk usage percentage
get_disk_usage() {
    df / | awk 'NR==2 {print $5}' | sed 's/%//'
}

# Emergency cleanup function
emergency_cleanup() {
    local usage=$1
    log_msg "CRITICAL: Disk usage at ${usage}% - performing emergency cleanup"
    
    # 1. Clean systemd journal logs (keep last 3 days)
    if command -v journalctl >/dev/null 2>&1; then
        journalctl --vacuum-time=3d >/dev/null 2>&1
        log_msg "Cleaned systemd journal logs"
    fi
    
    # 2. Clean Q Geth logs older than 3 days
    find /home/*/qgeth/logs -name "*.log" -mtime +3 -delete 2>/dev/null || true
    find /home/*/.qcoin/*/logs -name "*.log" -mtime +3 -delete 2>/dev/null || true
    log_msg "Cleaned old Q Geth logs"
    
    # 3. Truncate large active log files (>100MB)
    find /home/*/qgeth/logs -name "*.log" -size +100M -exec truncate -s 50M {} \; 2>/dev/null || true
    find /home/*/.qcoin -name "*.log" -size +100M -exec truncate -s 50M {} \; 2>/dev/null || true
    log_msg "Truncated large log files"
    
    # 4. Clean Go build cache and temp files
    if command -v go >/dev/null 2>&1; then
        go clean -cache -modcache -testcache >/dev/null 2>&1 || true
        log_msg "Cleaned Go build cache"
    fi
    
    # 5. Clean system temp files
    find /tmp -name "*qgeth*" -mtime +1 -delete 2>/dev/null || true
    find /tmp -name "*quantum*" -mtime +1 -delete 2>/dev/null || true
    log_msg "Cleaned temp files"
    
    # 6. Clean apt cache if available
    if command -v apt >/dev/null 2>&1; then
        apt clean >/dev/null 2>&1 || true
        log_msg "Cleaned package cache"
    fi
    
    # 7. Force log rotation
    if command -v logrotate >/dev/null 2>&1; then
        logrotate -f /etc/logrotate.d/qgeth >/dev/null 2>&1 || true
        log_msg "Forced log rotation"
    fi
}

# Warning cleanup function  
warning_cleanup() {
    local usage=$1
    log_msg "WARNING: Disk usage at ${usage}% - performing routine cleanup"
    
    # 1. Clean journal logs (keep last week)
    if command -v journalctl >/dev/null 2>&1; then
        journalctl --vacuum-time=1week >/dev/null 2>&1
    fi
    
    # 2. Clean old logs (>7 days)
    find /home/*/qgeth/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    find /home/*/.qcoin/*/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # 3. Clean old temp files
    find /tmp -name "*qgeth*" -mtime +7 -delete 2>/dev/null || true
    find /tmp -name "*quantum*" -mtime +7 -delete 2>/dev/null || true
    
    log_msg "Routine cleanup completed"
}

# Main monitoring logic
main() {
    local usage=$(get_disk_usage)
    
    if [ "$usage" -ge "$CRITICAL_THRESHOLD" ]; then
        emergency_cleanup "$usage"
        
        # Check if cleanup helped
        local new_usage=$(get_disk_usage)
        local saved=$((usage - new_usage))
        log_msg "Emergency cleanup completed: ${usage}% -> ${new_usage}% (saved ${saved}%)"
        
        # If still critical, send alert
        if [ "$new_usage" -ge "$CRITICAL_THRESHOLD" ]; then
            log_msg "ALERT: Disk usage still critical at ${new_usage}% after cleanup!"
        fi
        
    elif [ "$usage" -ge "$WARNING_THRESHOLD" ]; then
        warning_cleanup "$usage"
        
        local new_usage=$(get_disk_usage)
        local saved=$((usage - new_usage))
        log_msg "Warning cleanup completed: ${usage}% -> ${new_usage}% (saved ${saved}%)"
        
    else
        # Normal operation - just log status
        log_msg "Disk usage normal: ${usage}%"
    fi
}

# Run the monitor
main "$@"
EOF

    chmod +x "$MONITOR_DIR/disk-monitor.sh"
    
    # Create systemd timer for disk monitoring (if systemd is available)
    if [ "$INIT_SYSTEM" = "systemd" ]; then
        log_info "Creating systemd timer for disk monitoring..."
        
        # Determine if we need sudo
        SUDO_CMD=""
        if [ "$EUID" -ne 0 ]; then
            SUDO_CMD="sudo"
        fi
        
        # Create service file
        $SUDO_CMD tee /etc/systemd/system/qgeth-disk-monitor.service > /dev/null << EOF
[Unit]
Description=Q Geth Disk Space Monitor
After=multi-user.target

[Service]
Type=oneshot
ExecStart=$MONITOR_DIR/disk-monitor.sh
User=root
StandardOutput=journal
StandardError=journal
EOF

        # Create timer file  
        $SUDO_CMD tee /etc/systemd/system/qgeth-disk-monitor.timer > /dev/null << EOF
[Unit]
Description=Q Geth Disk Monitor Timer
Requires=qgeth-disk-monitor.service

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF

        # Enable and start timer
        $SUDO_CMD systemctl daemon-reload
        $SUDO_CMD systemctl enable qgeth-disk-monitor.timer
        $SUDO_CMD systemctl start qgeth-disk-monitor.timer
        
        log_success "✅ Disk monitoring timer created and started"
    else
        # For non-systemd systems, add to crontab
        log_info "Adding disk monitor to crontab..."
        
        # Add hourly disk monitoring to root crontab
        (crontab -l 2>/dev/null || true; echo "0 * * * * $MONITOR_DIR/disk-monitor.sh") | crontab -
        
        log_success "✅ Disk monitoring added to crontab"
    fi
    
    log_success "✅ Disk monitoring system created"
}

# Initial cleanup function to remove unnecessary files after installation
initial_cleanup() {
    log_info "Performing initial cleanup after installation..."
    
    # Clean Go build cache and temp files  
    if command -v go >/dev/null 2>&1; then
        log_info "Cleaning Go build cache..."
        go clean -cache -modcache -testcache 2>/dev/null || true
        
        # Remove downloaded Go modules cache if not needed
        if [ -d "/root/go/pkg/mod/cache" ]; then
            rm -rf "/root/go/pkg/mod/cache" 2>/dev/null || true
        fi
        if [ -d "$HOME/go/pkg/mod/cache" ]; then
            rm -rf "$HOME/go/pkg/mod/cache" 2>/dev/null || true
        fi
    fi
    
    # Remove duplicate project directories if they exist
    if [ -d "/root/Qgeth3" ] && [ "/root/Qgeth3" != "$PROJECT_DIR" ]; then
        log_info "Removing duplicate project directory: /root/Qgeth3"
        rm -rf "/root/Qgeth3" 2>/dev/null || true
    fi
    
    if [ -d "/root/qgeth" ] && [ "/root/qgeth" != "$INSTALL_DIR" ]; then
        log_info "Removing old project directory: /root/qgeth"
        rm -rf "/root/qgeth" 2>/dev/null || true
    fi
    
    if [ -d "$HOME/Qgeth3" ] && [ "$HOME/Qgeth3" != "$PROJECT_DIR" ]; then
        log_info "Removing duplicate project directory in home"
        rm -rf "$HOME/Qgeth3" 2>/dev/null || true
    fi
    
    # Clean up any temporary build files
    find "$PROJECT_DIR" -name "build-temp-*" -type d -exec rm -rf {} \; 2>/dev/null || true
    find /tmp -name "*qgeth*" -mtime +0 -delete 2>/dev/null || true
    find /tmp -name "*quantum*" -mtime +0 -delete 2>/dev/null || true
    
    # Clean apt cache to save space
    if command -v apt >/dev/null 2>&1 && [ "$EUID" -eq 0 ]; then
        apt clean 2>/dev/null || true
        apt autoremove -y 2>/dev/null || true
    fi
    
    log_success "✅ Initial cleanup completed"
}

# Main installation
main() {
    echo ""
    if [ "$DOCKER_MODE" = true ]; then
        echo -e "${CYAN}🐳 Q Geth Universal Bootstrap with Docker${NC}"
        echo ""
        echo "This will:"
        echo "  📦 Install Q Geth to: $PROJECT_DIR"
        echo "  🐳 Install Docker and Docker Compose"
        echo "  🚀 Deploy Q Geth as Docker container"
        echo ""
    else
        echo -e "${CYAN}🚀 Q Geth Universal Bootstrap with System Service${NC}"
        echo ""
        echo "This will:"
        echo "  📦 Install Q Geth to: $PROJECT_DIR"
        echo "  🔧 Create persistent system service"
        echo "  🚀 Auto-start Q Geth service"
        echo ""
    fi
    
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
    if [ "$DOCKER_MODE" != true ]; then
        detect_init_system
    fi
    
    # Install dependencies
    if [ "$DOCKER_MODE" = true ]; then
        # For Docker mode, we need basic dependencies and Docker
        install_dependencies
        install_docker
    else
        # For system service mode, install dependencies normally
        install_dependencies
        # Setup swap if needed for building
        setup_swap_if_needed
    fi
    
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
    
    if [ "$DOCKER_MODE" = true ]; then
        # Docker deployment path
        log_info "Deploying Q Geth with Docker..."
        
        # Deploy with Docker
        if ! deploy_docker; then
            log_error "Failed to deploy Q Geth with Docker"
            log_info "You can try running Docker manually:"
            log_info "  cd $PROJECT_DIR"
            log_info "  docker-compose up -d qgeth-testnet"
            exit 1
        fi
        
        # Create Docker management scripts
        create_docker_scripts
        
        log_success "✅ Q Geth Docker deployment completed successfully"
    else
        # Traditional build and service creation path
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
            log_info "  ./start-geth.sh testnet"
            exit 1
        fi
        
        # Setup log management and monitoring
        log_info "Setting up log management and monitoring..."
        setup_log_rotation
        setup_systemd_journal_limits  
        create_disk_monitor
    fi
    
    if [ "$DOCKER_MODE" != true ]; then
        log_info "Creating universal service management scripts..."
        
        # Create universal service management scripts
    cat > "$INSTALL_DIR/start-qgeth.sh" << EOF
#!/bin/bash
# Start Q Geth System Service for user: $ACTUAL_USER
echo "Starting Q Geth system service ($INIT_SYSTEM)..."

case "$INIT_SYSTEM" in
    "systemd")
        sudo systemctl start qgeth.service
        if sudo systemctl is-active --quiet qgeth.service; then
            echo "✅ Q Geth service started successfully"
        else
            echo "❌ Failed to start Q Geth service"
            echo "Check logs: sudo journalctl -u qgeth.service -f"
            exit 1
        fi
        ;;
    "openrc")
        sudo rc-service qgeth start
        echo "Q Geth service start command executed"
        ;;
    "sysv")
        sudo service qgeth start
        echo "Q Geth service start command executed"
        ;;
    "upstart")
        sudo initctl start qgeth
        echo "Q Geth service start command executed"
        ;;
    *)
        echo "❌ Unsupported init system: $INIT_SYSTEM"
        exit 1
        ;;
esac
EOF
    
    cat > "$INSTALL_DIR/stop-qgeth.sh" << EOF
#!/bin/bash
# Stop Q Geth System Service for user: $ACTUAL_USER
echo "Stopping Q Geth system service ($INIT_SYSTEM)..."

case "$INIT_SYSTEM" in
    "systemd")
        sudo systemctl stop qgeth.service
        echo "✅ Q Geth service stopped"
        ;;
    "openrc")
        sudo rc-service qgeth stop
        echo "Q Geth service stop command executed"
        ;;
    "sysv")
        sudo service qgeth stop
        echo "Q Geth service stop command executed"
        ;;
    "upstart")
        sudo initctl stop qgeth
        echo "Q Geth service stop command executed"
        ;;
    *)
        echo "❌ Unsupported init system: $INIT_SYSTEM"
        exit 1
        ;;
esac
EOF
    
    cat > "$INSTALL_DIR/status-qgeth.sh" << EOF
#!/bin/bash
# Check Q Geth System Service Status for user: $ACTUAL_USER
echo "Q Geth system service status ($INIT_SYSTEM):"
echo ""

case "$INIT_SYSTEM" in
    "systemd")
        sudo systemctl status qgeth.service --no-pager -l
        echo ""
        echo "📋 Service Management Commands:"
        echo "  Start:   sudo systemctl start qgeth.service"
        echo "  Stop:    sudo systemctl stop qgeth.service"
        echo "  Restart: sudo systemctl restart qgeth.service"
        echo "  Logs:    sudo journalctl -u qgeth.service -f"
        echo "  Enable:  sudo systemctl enable qgeth.service"
        echo "  Disable: sudo systemctl disable qgeth.service"
        ;;
    "openrc")
        sudo rc-service qgeth status
        echo ""
        echo "📋 Service Management Commands:"
        echo "  Start:   sudo rc-service qgeth start"
        echo "  Stop:    sudo rc-service qgeth stop"
        echo "  Restart: sudo rc-service qgeth restart"
        echo "  Logs:    tail -f /var/log/qgeth.log"
        ;;
    "sysv")
        sudo service qgeth status
        echo ""
        echo "📋 Service Management Commands:"
        echo "  Start:   sudo service qgeth start"
        echo "  Stop:    sudo service qgeth stop"
        echo "  Restart: sudo service qgeth restart"
        echo "  Logs:    tail -f /var/log/qgeth.log"
        ;;
    "upstart")
        sudo initctl status qgeth
        echo ""
        echo "📋 Service Management Commands:"
        echo "  Start:   sudo initctl start qgeth"
        echo "  Stop:    sudo initctl stop qgeth"
        echo "  Restart: sudo initctl restart qgeth"
        echo "  Logs:    tail -f /var/log/qgeth.log"
        ;;
    *)
        echo "❌ Unsupported init system: $INIT_SYSTEM"
        exit 1
        ;;
esac
EOF

    cat > "$INSTALL_DIR/restart-qgeth.sh" << EOF
#!/bin/bash
# Restart Q Geth System Service for user: $ACTUAL_USER
echo "Restarting Q Geth system service ($INIT_SYSTEM)..."

case "$INIT_SYSTEM" in
    "systemd")
        sudo systemctl restart qgeth.service
        if sudo systemctl is-active --quiet qgeth.service; then
            echo "✅ Q Geth service restarted successfully"
        else
            echo "❌ Failed to restart Q Geth service"
            exit 1
        fi
        ;;
    "openrc")
        sudo rc-service qgeth restart
        echo "Q Geth service restart command executed"
        ;;
    "sysv")
        sudo service qgeth restart
        echo "Q Geth service restart command executed"
        ;;
    "upstart")
        sudo initctl restart qgeth
        echo "Q Geth service restart command executed"
        ;;
    *)
        echo "❌ Unsupported init system: $INIT_SYSTEM"
        exit 1
        ;;
esac
EOF

    cat > "$INSTALL_DIR/logs-qgeth.sh" << EOF
#!/bin/bash
# View Q Geth System Service Logs for user: $ACTUAL_USER
# Usage: ./logs-qgeth.sh [-f] [-n NUMBER]

FOLLOW_MODE=false
LINE_COUNT=50

# Parse arguments
while [[ \$# -gt 0 ]]; do
    case \$1 in
        -f|--follow)
            FOLLOW_MODE=true
            shift
            ;;
        -n|--lines)
            LINE_COUNT="\$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: \$0 [-f|--follow] [-n|--lines NUMBER]"
            echo "  -f, --follow    Follow log output (live mode)"
            echo "  -n, --lines     Number of lines to show (default: 50)"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: \$1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

echo "Q Geth system service logs ($INIT_SYSTEM):"
echo ""

case "$INIT_SYSTEM" in
    "systemd")
        if [ "\$FOLLOW_MODE" = true ]; then
            echo "Following Q Geth service logs (press Ctrl+C to exit)..."
            sudo journalctl -u qgeth.service -f
        else
            echo "Last \$LINE_COUNT lines from Q Geth service:"
            sudo journalctl -u qgeth.service --no-pager -l -n "\$LINE_COUNT"
        fi
        ;;
    "openrc"|"sysv"|"upstart")
        # Try multiple log locations
        LOG_FILES=(
            "/var/log/qgeth.log"
            "$INSTALL_DIR/logs/qgeth.log"
            "$USER_HOME/.qcoin/testnet/geth.log"
        )
        
        FOUND_LOG=false
        for LOG_FILE in "\${LOG_FILES[@]}"; do
            if [ -f "\$LOG_FILE" ]; then
                echo "Using log file: \$LOG_FILE"
                if [ "\$FOLLOW_MODE" = true ]; then
                    echo "Following Q Geth logs (press Ctrl+C to exit)..."
                    tail -f "\$LOG_FILE"
                else
                    echo "Last \$LINE_COUNT lines from Q Geth:"
                    tail -n "\$LINE_COUNT" "\$LOG_FILE"
                fi
                FOUND_LOG=true
                break
            fi
        done
        
        if [ "\$FOUND_LOG" = false ]; then
            echo "❌ No log files found. Tried:"
            for LOG_FILE in "\${LOG_FILES[@]}"; do
                echo "  - \$LOG_FILE"
            done
            echo ""
            echo "💡 Try checking service status:"
            echo "  $INSTALL_DIR/status-qgeth.sh"
        fi
        ;;
    *)
        echo "❌ Unsupported init system: $INIT_SYSTEM"
        exit 1
        ;;
esac
EOF
    
        chmod +x "$INSTALL_DIR"/*.sh
        
        # Start the service
        start_system_service
    fi
    
    # Perform initial cleanup to save disk space
    initial_cleanup
    
    # Fix ownership if installed with sudo
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        log_info "Fixing ownership for user $SUDO_USER..."
        chown -R "$SUDO_USER:$SUDO_USER" "$INSTALL_DIR" 2>/dev/null || true
        log_success "✅ Ownership set to $SUDO_USER"
    fi
    
    # Success message
    echo ""
    echo "========================================"
    if [ "$DOCKER_MODE" = true ]; then
        echo -e "${GREEN}🎉 Q Geth Docker Installation Complete!${NC}"
    else
        echo -e "${GREEN}🎉 Q Geth System Service Installation Complete!${NC}"
    fi
    echo "========================================"
    echo ""
    echo -e "${BLUE}📋 Installation Summary:${NC}"
    echo "  Directory: $PROJECT_DIR"
    echo "  User: $ACTUAL_USER"
    echo "  Home: $USER_HOME"
    if [ "$DOCKER_MODE" = true ]; then
        echo "  Deployment: Docker containers"
        echo "  Management: Docker Compose + scripts"
    else
        echo "  Init System: $INIT_SYSTEM"
        echo "  Service: qgeth ($INIT_SYSTEM service)"
        echo "  Management: $INSTALL_DIR/*.sh"
    fi
    echo ""
    if [ "$DOCKER_MODE" = true ]; then
        echo -e "${BLUE}🐳 Docker Management Commands:${NC}"
        echo "  Start:   docker-compose up -d qgeth-testnet"
        echo "  Stop:    docker-compose stop qgeth-testnet"
        echo "  Restart: docker-compose restart qgeth-testnet"
        echo "  Status:  docker-compose ps"
        echo "  Logs:    docker-compose logs -f qgeth-testnet"
        echo "  Remove:  docker-compose down"
        echo ""
        echo -e "${BLUE}🔧 Docker Management Scripts:${NC}"
        echo "  Start:    $INSTALL_DIR/start-qgeth-docker.sh"
        echo "  Stop:     $INSTALL_DIR/stop-qgeth-docker.sh"
        echo "  Restart:  $INSTALL_DIR/restart-qgeth-docker.sh"
        echo "  Status:   $INSTALL_DIR/status-qgeth-docker.sh"
        echo "  Logs:     $INSTALL_DIR/logs-qgeth-docker.sh [-f] [-n 100]"
        echo "  Mining:   $INSTALL_DIR/mining-qgeth-docker.sh"
    else
        echo -e "${BLUE}🔧 System Service Commands:${NC}"
        case $INIT_SYSTEM in
            "systemd")
                echo "  Start:   sudo systemctl start qgeth.service"
                echo "  Stop:    sudo systemctl stop qgeth.service"
                echo "  Restart: sudo systemctl restart qgeth.service"
                echo "  Status:  sudo systemctl status qgeth.service"
                echo "  Logs:    sudo journalctl -u qgeth.service -f"
                echo "  Enable:  sudo systemctl enable qgeth.service"
                echo "  Disable: sudo systemctl disable qgeth.service"
                ;;
            "openrc")
                echo "  Start:   sudo rc-service qgeth start"
                echo "  Stop:    sudo rc-service qgeth stop"
                echo "  Restart: sudo rc-service qgeth restart" 
                echo "  Status:  sudo rc-service qgeth status"
                echo "  Logs:    tail -f /var/log/qgeth.log"
                ;;
            "sysv")
                echo "  Start:   sudo service qgeth start"
                echo "  Stop:    sudo service qgeth stop"
                echo "  Restart: sudo service qgeth restart"
                echo "  Status:  sudo service qgeth status"
                echo "  Logs:    tail -f /var/log/qgeth.log"
                ;;
            "upstart")
                echo "  Start:   sudo initctl start qgeth"
                echo "  Stop:    sudo initctl stop qgeth"
                echo "  Restart: sudo initctl restart qgeth"
                echo "  Status:  sudo initctl status qgeth"
                echo "  Logs:    tail -f /var/log/qgeth.log"
                ;;
        esac
        echo ""
        echo -e "${BLUE}🔧 Universal Management Scripts:${NC}"
        echo "  Start:   $INSTALL_DIR/start-qgeth.sh"
        echo "  Stop:    $INSTALL_DIR/stop-qgeth.sh"
        echo "  Restart: $INSTALL_DIR/restart-qgeth.sh"
        echo "  Status:  $INSTALL_DIR/status-qgeth.sh"
        echo "  Logs:    $INSTALL_DIR/logs-qgeth.sh [-f] [-n 100]"
    fi
}

# Run main function
main "$@" 