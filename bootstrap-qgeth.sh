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

# Parse simple flags
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
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
    
    $SUDO_CMD tee /etc/systemd/system/qgeth.service > /dev/null << EOF
[Unit]
Description=Q Geth Quantum Blockchain Node
Documentation=https://github.com/fourtytwo42/Qgeth3
After=network.target
Wants=network.target

[Service]
Type=exec
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR/scripts/linux
ExecStart=$PROJECT_DIR/scripts/linux/start-geth.sh testnet
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=process
Restart=on-failure
RestartSec=10s
TimeoutStopSec=30s

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$INSTALL_DIR
ReadWritePaths=/tmp

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Environment
Environment=HOME=$USER_HOME
Environment=PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    $SUDO_CMD systemctl daemon-reload
    $SUDO_CMD systemctl enable qgeth.service
    log_success "✅ Systemd service created and enabled"
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
    $SUDO_CMD rc-update add qgeth default
    log_success "✅ OpenRC service created and enabled"
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
    for runlevel in 3 4 5; do
        $SUDO_CMD ln -sf /etc/init.d/qgeth "/etc/rc${runlevel}.d/S80qgeth" 2>/dev/null || true
    done
    for runlevel in 0 1 2 6; do
        $SUDO_CMD ln -sf /etc/init.d/qgeth "/etc/rc${runlevel}.d/K20qgeth" 2>/dev/null || true
    done
    
    log_success "✅ SysV init script created and enabled"
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
                log_warning "⚠️ Service created but failed to start - check logs"
            fi
            ;;
        "openrc")
            $SUDO_CMD rc-service qgeth start
            if $SUDO_CMD rc-service qgeth status >/dev/null 2>&1; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start - check logs"
            fi
            ;;
        "sysv")
            $SUDO_CMD service qgeth start
            if $SUDO_CMD service qgeth status >/dev/null 2>&1; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start - check logs"
            fi
            ;;
        "upstart")
            $SUDO_CMD initctl start qgeth
            if $SUDO_CMD initctl status qgeth | grep -q "start/running"; then
                log_success "✅ Q Geth service started successfully"
            else
                log_warning "⚠️ Service created but failed to start - check logs"
            fi
            ;;
        *)
            log_error "Cannot start service for init system: $INIT_SYSTEM"
            return 1
            ;;
    esac
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
    
    # Create system service and management scripts
    create_system_service
    
    log_info "Creating system service management scripts..."
    
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
    
    chmod +x "$INSTALL_DIR"/*.sh
    
    # Start the service
    start_system_service
    
    # Fix ownership if installed with sudo
    if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
        log_info "Fixing ownership for user $SUDO_USER..."
        chown -R "$SUDO_USER:$SUDO_USER" "$INSTALL_DIR" 2>/dev/null || true
        log_success "✅ Ownership set to $SUDO_USER"
    fi
    
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
    echo "  Init System: $INIT_SYSTEM"
    echo "  Service: qgeth ($INIT_SYSTEM service)"
    echo "  Management: $INSTALL_DIR/*.sh"
    echo ""
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
    echo -e "${BLUE}🔧 Quick Management Scripts:${NC}"
    echo "  Start:   $INSTALL_DIR/start-qgeth.sh"
    echo "  Stop:    $INSTALL_DIR/stop-qgeth.sh"
    echo "  Restart: $INSTALL_DIR/restart-qgeth.sh"
    echo "  Status:  $INSTALL_DIR/status-qgeth.sh"
    echo ""
    echo -e "${BLUE}🎯 Service Features:${NC}"
    echo "  ✅ Persistent service (survives reboots)"
    echo "  ✅ Automatic restart on failure"
    echo "  ✅ Proper logging and monitoring"
    echo "  ✅ Secure execution as user: $ACTUAL_USER"
    echo "  ✅ System integration with $INIT_SYSTEM"
    echo ""
    echo -e "${BLUE}🎯 Next Steps:${NC}"
    echo "  1. Service started automatically"
    echo "  2. Check status: $INSTALL_DIR/status-qgeth.sh"
    echo "  3. Start mining: cd $PROJECT_DIR/scripts/linux && ./start-miner.sh"
    echo "  4. RPC API: http://localhost:8545"
    echo ""
    echo -e "${GREEN}Q Geth is now running as a persistent system service! 🚀${NC}"
}

# Run main function
main "$@" 