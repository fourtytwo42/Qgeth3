#!/usr/bin/env bash
# Q Geth Enhanced Cross-Distribution Bootstrap Script
# Universal Linux/Unix deployment with automatic system detection
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth-enhanced.sh | sudo bash
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth-enhanced.sh | sudo bash -s -- -y

set -e

# Parse command line arguments
AUTO_CONFIRM=false
DEBUG=false

# Better argument parsing that works with curl pipes
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        --debug)
            DEBUG=true
            ;;
    esac
done

# Auto-detect non-interactive environment (like curl pipe)
if [ ! -t 0 ] || [ ! -t 1 ]; then
    AUTO_CONFIRM=true
fi

# Configuration
GITHUB_REPO="fourtytwo42/Qgeth3"
GITHUB_BRANCH="main"
INSTALL_DIR="/opt/qgeth"
PROJECT_DIR="$INSTALL_DIR/Qgeth3"
LOGS_DIR="$INSTALL_DIR/logs"

# Download and source system detection library
DETECT_SCRIPT_URL="https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/linux/detect-system.sh"
TEMP_DETECT_SCRIPT="/tmp/qgeth-detect-system-$$.sh"

echo "📡 Downloading system detection library..."
if command -v curl >/dev/null 2>&1; then
    curl -sSL "$DETECT_SCRIPT_URL" > "$TEMP_DETECT_SCRIPT"
elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$DETECT_SCRIPT_URL" > "$TEMP_DETECT_SCRIPT"
else
    echo "❌ Neither curl nor wget is available. Please install one of them."
    exit 1
fi

# Source the detection library
chmod +x "$TEMP_DETECT_SCRIPT"
source "$TEMP_DETECT_SCRIPT"

# Clean up temp file
rm -f "$TEMP_DETECT_SCRIPT"

# Check if running with appropriate privileges
if ! check_root; then
    log_error "Please run this script with sudo"
    echo ""
    echo "Usage:"
    echo "  curl -sSL https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/deployment/bootstrap-qgeth-enhanced.sh | sudo bash"
    echo ""
    echo "For non-interactive mode:"
    echo "  curl -sSL https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/deployment/bootstrap-qgeth-enhanced.sh | sudo bash -s -- -y"
    exit 1
fi

# Get actual user information
ACTUAL_USER=$(get_actual_user)
ACTUAL_HOME=$(get_actual_home)

log_info "🚀 Q Geth Enhanced Cross-Distribution Bootstrap"
echo ""
echo "Detected System:"
echo "  OS: $QGETH_OS ($QGETH_DISTRO $QGETH_DISTRO_VERSION)"
echo "  Package Manager: $QGETH_PKG_MANAGER"
echo "  Init System: $QGETH_INIT_SYSTEM"
echo "  Firewall: $QGETH_FIREWALL"
echo "  Architecture: $QGETH_ARCH"
echo "  User: $ACTUAL_USER"
echo ""
echo "This script will:"
echo "  ✅ Clean up any existing installations"
echo "  ✅ Install dependencies using $QGETH_PKG_MANAGER"
echo "  ✅ Prepare system (memory, swap, dependencies)"
echo "  ✅ Clone Q Geth repository to $INSTALL_DIR"
echo "  ✅ Build Q Geth with automated error recovery"
echo "  ✅ Create persistent service using $QGETH_INIT_SYSTEM"
echo "  ✅ Configure firewall using $QGETH_FIREWALL"
echo ""

# Interactive confirmation
if [ "$AUTO_CONFIRM" != true ]; then
    if [ -t 0 ] && [ -t 1 ]; then
        echo -n "Proceed with installation? (y/N): "
        read -r RESPONSE </dev/tty
        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Installation cancelled."
            exit 0
        fi
    else
        log_warning "Non-interactive environment detected, proceeding automatically..."
        AUTO_CONFIRM=true
    fi
else
    log_info "Auto-confirm mode enabled, proceeding..."
fi

# ===========================================
# STEP 1: CLEANUP EXISTING INSTALLATIONS
# ===========================================
log_info "🧹 Step 1: Cleanup existing installations"

# Stop services using detected init system
if [ "$QGETH_INIT_SYSTEM" != "unknown" ]; then
    log_info "Stopping Q Geth services using $QGETH_INIT_SYSTEM..."
    for service in qgeth qgeth-node qgeth-monitor; do
        $QGETH_SERVICE_STOP $service 2>/dev/null || true
        $QGETH_SERVICE_DISABLE $service 2>/dev/null || true
    done
    
    # Remove service files based on init system
    case "$QGETH_INIT_SYSTEM" in
        systemd)
            rm -f /etc/systemd/system/qgeth.service
            rm -f /etc/systemd/system/qgeth-node.service
            rm -f /etc/systemd/system/qgeth-monitor.service
            $QGETH_SERVICE_RELOAD
            ;;
        openrc)
            rm -f /etc/init.d/qgeth
            rm -f /etc/init.d/qgeth-node
            rm -f /etc/init.d/qgeth-monitor
            ;;
        sysv)
            rm -f /etc/init.d/qgeth
            rm -f /etc/init.d/qgeth-node
            rm -f /etc/init.d/qgeth-monitor
            ;;
    esac
fi

# Kill any remaining geth processes
log_info "Terminating geth processes..."
pkill -f "geth" 2>/dev/null || true
sleep 2
pkill -9 -f "geth" 2>/dev/null || true

# Remove installation directories and lock files
log_info "Removing installation directories..."
rm -rf "$INSTALL_DIR" 2>/dev/null || true
rm -f /tmp/qgeth-*.lock 2>/dev/null || true
rm -rf /tmp/qgeth-build* 2>/dev/null || true

log_success "✅ Cleanup completed"

# ===========================================
# STEP 2: DEPENDENCY INSTALLATION
# ===========================================
log_info "🔧 Step 2: Installing dependencies using $QGETH_PKG_MANAGER"

# Update package lists
log_info "Updating package lists..."
eval "$QGETH_PKG_UPDATE" || true

# Get distribution-specific package names
GIT_PACKAGES=$(get_package_names "git")
CURL_PACKAGES=$(get_package_names "curl")
BUILD_PACKAGES=$(get_package_names "build-tools")
GOLANG_PACKAGES=$(get_package_names "golang")
PYTHON_PACKAGES=$(get_package_names "python")

# Install essential packages
log_info "Installing essential packages..."
ESSENTIAL_PACKAGES="$GIT_PACKAGES $CURL_PACKAGES $BUILD_PACKAGES $PYTHON_PACKAGES"

# Add distribution-specific packages
case "$QGETH_PKG_MANAGER" in
    apt)
        ESSENTIAL_PACKAGES="$ESSENTIAL_PACKAGES systemd ufw jq unzip ca-certificates software-properties-common apt-transport-https gnupg lsb-release"
        ;;
    dnf|yum)
        ESSENTIAL_PACKAGES="$ESSENTIAL_PACKAGES systemd firewalld jq unzip"
        ;;
    pacman)
        ESSENTIAL_PACKAGES="$ESSENTIAL_PACKAGES systemd jq unzip"
        ;;
    zypper)
        ESSENTIAL_PACKAGES="$ESSENTIAL_PACKAGES systemd jq unzip"
        ;;
    apk)
        ESSENTIAL_PACKAGES="$ESSENTIAL_PACKAGES openrc jq unzip ca-certificates"
        ;;
esac

eval "$QGETH_PKG_INSTALL $ESSENTIAL_PACKAGES"

# Go installation with version compatibility
log_info "Installing Go (checking version compatibility)..."

# Check if Go is already installed and version
GO_VERSION=""
if command -v go >/dev/null 2>&1; then
    GO_VERSION=$(go version 2>/dev/null | grep -o 'go[0-9]*\.[0-9]*' | head -1)
    log_info "Found existing Go: $GO_VERSION"
fi

# Install Go if not present or version is too old
GO_MIN_VERSION="1.21"
NEED_GO_INSTALL=true

if [ ! -z "$GO_VERSION" ]; then
    # Extract version numbers for comparison
    GO_VER_MAJOR=$(echo "$GO_VERSION" | sed 's/go//' | cut -d. -f1)
    GO_VER_MINOR=$(echo "$GO_VERSION" | sed 's/go//' | cut -d. -f2)
    
    if [ "$GO_VER_MAJOR" -gt 1 ] || ([ "$GO_VER_MAJOR" -eq 1 ] && [ "$GO_VER_MINOR" -ge 21 ]); then
        log_success "✅ Go $GO_VERSION is compatible (>= $GO_MIN_VERSION)"
        NEED_GO_INSTALL=false
    else
        log_warning "Go $GO_VERSION is too old, need >= $GO_MIN_VERSION"
    fi
fi

if [ "$NEED_GO_INSTALL" = true ]; then
    log_info "Installing Go $GO_MIN_VERSION or later..."
    
    # Try package manager first
    if eval "$QGETH_PKG_INSTALL $GOLANG_PACKAGES"; then
        # Check if installed version is sufficient
        if command -v go >/dev/null 2>&1; then
            NEW_GO_VERSION=$(go version 2>/dev/null | grep -o 'go[0-9]*\.[0-9]*' | head -1 || echo "")
            if [ ! -z "$NEW_GO_VERSION" ]; then
                NEW_GO_VER_MAJOR=$(echo "$NEW_GO_VERSION" | sed 's/go//' | cut -d. -f1)
                NEW_GO_VER_MINOR=$(echo "$NEW_GO_VERSION" | sed 's/go//' | cut -d. -f2)
                
                if [ "$NEW_GO_VER_MAJOR" -gt 1 ] || ([ "$NEW_GO_VER_MAJOR" -eq 1 ] && [ "$NEW_GO_VER_MINOR" -ge 21 ]); then
                    log_success "✅ Go $NEW_GO_VERSION installed successfully"
                else
                    NEED_MANUAL_GO=true
                fi
            else
                NEED_MANUAL_GO=true
            fi
        else
            NEED_MANUAL_GO=true
        fi
    else
        NEED_MANUAL_GO=true
    fi
    
    # Manual Go installation if package manager version is insufficient
    if [ "$NEED_MANUAL_GO" = true ]; then
        log_info "Installing latest Go manually..."
        
        # Remove conflicting package manager Go installations
        case "$QGETH_PKG_MANAGER" in
            apt)
                eval "$QGETH_PKG_REMOVE golang-go golang-1.*" 2>/dev/null || true
                ;;
            dnf|yum)
                eval "$QGETH_PKG_REMOVE golang" 2>/dev/null || true
                ;;
            pacman)
                eval "$QGETH_PKG_REMOVE go" 2>/dev/null || true
                ;;
        esac
        
        eval "$QGETH_PKG_AUTOREMOVE" 2>/dev/null || true
        
        # Remove old manual Go installations
        rm -rf /usr/local/go
        
        # Download and install latest Go
        GO_LATEST_VERSION="1.21.6"  # Known working version
        GO_TARBALL="go${GO_LATEST_VERSION}.linux-${QGETH_GO_ARCH}.tar.gz"
        
        cd /tmp
        if curl -L "https://go.dev/dl/${GO_TARBALL}" -o "$GO_TARBALL" || wget "https://go.dev/dl/${GO_TARBALL}"; then
            tar -C /usr/local -xzf "$GO_TARBALL"
            
            # Set up PATH for different shells and systems
            export PATH="/usr/local/go/bin:$PATH"
            
            # Persistent PATH setup for different distributions
            case "$QGETH_DISTRO" in
                ubuntu|debian)
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/profile
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/bash.bashrc
                    ;;
                fedora|centos|rhel)
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/profile
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/bashrc
                    ;;
                arch)
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/profile
                    ;;
                alpine)
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/profile
                    ;;
                *)
                    echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/profile
                    ;;
            esac
            
            # Set for systemd services
            echo 'PATH="/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' >> /etc/environment
            
            # Verify installation
            if /usr/local/go/bin/go version; then
                log_success "✅ Go $GO_LATEST_VERSION installed manually"
            else
                log_error "Failed to install Go manually"
                exit 1
            fi
        else
            log_error "Failed to download Go $GO_LATEST_VERSION"
            exit 1
        fi
    fi
fi

# Final Go verification
if command -v go >/dev/null 2>&1; then
    GO_FINAL_VERSION=$(go version)
    log_success "✅ Go verified: $GO_FINAL_VERSION"
else
    log_error "Go installation failed"
    exit 1
fi

log_success "✅ Dependencies installed successfully"

# ===========================================
# STEP 3: SYSTEM PREPARATION
# ===========================================
log_info "🔧 Step 3: System preparation"

# Memory and swap check
log_info "Checking memory and swap..."
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
    
    log_info "RAM: ${TOTAL_MB}MB, Swap: ${SWAP_TOTAL}MB, Total: ${CURRENT_TOTAL}MB"
    log_info "Required: ${REQUIRED_MB}MB"
    
    # Create swap if needed
    if [ $CURRENT_TOTAL -lt $((REQUIRED_MB - 50)) ]; then
        NEEDED_SWAP=$((REQUIRED_MB - TOTAL_MB))
        if [ $NEEDED_SWAP -gt $SWAP_TOTAL ]; then
            NEEDED_SWAP=$((NEEDED_SWAP - SWAP_TOTAL))
            log_info "Creating ${NEEDED_SWAP}MB swap file..."
            
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
            
            log_success "✅ Swap file created"
        fi
    else
        log_success "✅ Sufficient memory available"
    fi
fi

log_success "✅ System preparation completed"

# ===========================================
# STEP 4: FIREWALL CONFIGURATION
# ===========================================
log_info "🔥 Step 4: Firewall configuration using $QGETH_FIREWALL"

case "$QGETH_FIREWALL" in
    ufw)
        log_info "Configuring UFW firewall..."
        ufw --force reset
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow essential ports
        ufw allow 22/tcp comment 'SSH'
        ufw allow 8545/tcp comment 'Q Geth RPC API'
        ufw allow 8546/tcp comment 'Q Geth WebSocket API'
        ufw allow 30303/tcp comment 'Q Geth P2P TCP'
        ufw allow 30303/udp comment 'Q Geth P2P UDP'
        
        $QGETH_FIREWALL_ENABLE
        log_success "✅ UFW firewall configured"
        ;;
    firewalld)
        log_info "Configuring firewalld..."
        $QGETH_FIREWALL_ENABLE
        
        # Allow essential ports
        firewall-cmd --permanent --add-port=22/tcp  # SSH
        firewall-cmd --permanent --add-port=8545/tcp  # RPC API
        firewall-cmd --permanent --add-port=8546/tcp  # WebSocket API
        firewall-cmd --permanent --add-port=30303/tcp  # P2P TCP
        firewall-cmd --permanent --add-port=30303/udp  # P2P UDP
        firewall-cmd --reload
        
        log_success "✅ firewalld configured"
        ;;
    iptables)
        log_info "Configuring iptables..."
        
        # Basic iptables rules
        iptables -F
        iptables -P INPUT DROP
        iptables -P FORWARD DROP
        iptables -P OUTPUT ACCEPT
        
        # Allow loopback
        iptables -A INPUT -i lo -j ACCEPT
        
        # Allow established connections
        iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
        
        # Allow essential ports
        iptables -A INPUT -p tcp --dport 22 -j ACCEPT  # SSH
        iptables -A INPUT -p tcp --dport 8545 -j ACCEPT  # RPC API
        iptables -A INPUT -p tcp --dport 8546 -j ACCEPT  # WebSocket API
        iptables -A INPUT -p tcp --dport 30303 -j ACCEPT  # P2P TCP
        iptables -A INPUT -p udp --dport 30303 -j ACCEPT  # P2P UDP
        
        # Save rules based on distribution
        case "$QGETH_DISTRO" in
            ubuntu|debian)
                iptables-save > /etc/iptables/rules.v4 2>/dev/null || true
                ;;
            centos|rhel|fedora)
                iptables-save > /etc/sysconfig/iptables 2>/dev/null || true
                ;;
        esac
        
        log_success "✅ iptables configured"
        ;;
    none)
        log_warning "No firewall detected - manual configuration may be needed"
        ;;
esac

# ===========================================
# STEP 5: REPOSITORY SETUP
# ===========================================
log_info "📦 Step 5: Repository setup"

# Create directories with proper ownership
create_dir_with_ownership "$INSTALL_DIR"
create_dir_with_ownership "$LOGS_DIR"

# Clone repository
log_info "Cloning Q Geth repository..."
cd "$INSTALL_DIR"
if sudo -u "$ACTUAL_USER" git clone "https://github.com/$GITHUB_REPO.git"; then
    log_success "✅ Repository cloned successfully"
else
    log_error "Failed to clone repository"
    exit 1
fi

# Make scripts executable
cd "$PROJECT_DIR"
find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

log_success "✅ Repository setup completed"

# ===========================================
# STEP 6: BUILD Q GETH
# ===========================================
log_info "🔨 Step 6: Building Q Geth"

# Set up build environment
export QGETH_BUILD_TEMP="$INSTALL_DIR/build-temp"
create_dir_with_ownership "$QGETH_BUILD_TEMP"

# Build with automated error recovery
cd "$PROJECT_DIR/scripts/linux"
log_info "Building with automated error recovery..."

# Ensure proper ownership before build
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"

BUILD_SUCCESS=false
BUILD_ATTEMPTS=0
MAX_BUILD_ATTEMPTS=3

while [ $BUILD_ATTEMPTS -lt $MAX_BUILD_ATTEMPTS ] && [ "$BUILD_SUCCESS" = false ]; do
    BUILD_ATTEMPTS=$((BUILD_ATTEMPTS + 1))
    log_info "🚀 Build attempt $BUILD_ATTEMPTS/$MAX_BUILD_ATTEMPTS"
    
    if [ "$AUTO_CONFIRM" = true ]; then
        BUILD_CMD="sudo -u \"$ACTUAL_USER\" env PATH=\"/usr/local/go/bin:$PATH\" QGETH_BUILD_TEMP=\"$QGETH_BUILD_TEMP\" ./build-linux.sh geth -y"
    else
        BUILD_CMD="sudo -u \"$ACTUAL_USER\" env PATH=\"/usr/local/go/bin:$PATH\" QGETH_BUILD_TEMP=\"$QGETH_BUILD_TEMP\" ./build-linux.sh geth"
    fi
    
    if eval "$BUILD_CMD"; then
        BUILD_SUCCESS=true
    fi
    
    if [ "$BUILD_SUCCESS" = false ] && [ $BUILD_ATTEMPTS -lt $MAX_BUILD_ATTEMPTS ]; then
        log_warning "Build attempt $BUILD_ATTEMPTS failed, applying recovery..."
        
        # Clean and retry
        cd "$PROJECT_DIR"
        rm -f geth geth.bin quantum_solver.py 2>/dev/null || true
        chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"
        sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" go clean -cache -modcache -testcache 2>/dev/null || true
        
        cd "$PROJECT_DIR/scripts/linux"
        sleep 5
    fi
done

if [ "$BUILD_SUCCESS" = false ]; then
    log_error "Build failed after $MAX_BUILD_ATTEMPTS attempts"
    exit 1
fi

# Fix ownership of created files
cd "$PROJECT_DIR"
chown "$ACTUAL_USER:$ACTUAL_USER" geth geth.bin quantum_solver.py 2>/dev/null || true
chmod +x geth geth.bin quantum_solver.py 2>/dev/null || true

log_success "✅ Q Geth built successfully"

# ===========================================
# STEP 7: CREATE SERVICE
# ===========================================
log_info "⚙️ Step 7: Creating service using $QGETH_INIT_SYSTEM"

case "$QGETH_INIT_SYSTEM" in
    systemd)
        # Create systemd service
        cat > /etc/systemd/system/qgeth.service << EOF
[Unit]
Description=Q Geth Quantum Blockchain Node
After=network.target
Wants=network.target

[Service]
Type=exec
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR/scripts/linux
Environment=PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$PROJECT_DIR/scripts/linux/start-geth.sh testnet
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=always
RestartSec=10
StandardOutput=append:$LOGS_DIR/geth.log
StandardError=append:$LOGS_DIR/geth.log

[Install]
WantedBy=multi-user.target
EOF
        
        $QGETH_SERVICE_RELOAD
        $QGETH_SERVICE_ENABLE qgeth.service
        ;;
        
    openrc)
        # Create OpenRC service
        cat > /etc/init.d/qgeth << EOF
#!/sbin/openrc-run

name="Q Geth Quantum Blockchain Node"
description="Q Geth Quantum Blockchain Node"

user="$ACTUAL_USER"
group="$ACTUAL_USER"
command="$PROJECT_DIR/scripts/linux/start-geth.sh"
command_args="testnet"
command_background="yes"
pidfile="/var/run/qgeth.pid"
output_log="$LOGS_DIR/geth.log"
error_log="$LOGS_DIR/geth.log"

depend() {
    need net
    after firewall
}
EOF
        chmod +x /etc/init.d/qgeth
        $QGETH_SERVICE_ENABLE qgeth default
        ;;
        
    sysv)
        # Create SysV init script
        cat > /etc/init.d/qgeth << 'EOF'
#!/bin/bash
# Q Geth service script for SysV

. /etc/rc.d/init.d/functions

USER="$ACTUAL_USER"
DAEMON="qgeth"
ROOT_DIR="$PROJECT_DIR"

SERVER="$ROOT_DIR/scripts/linux/start-geth.sh"
LOCK_FILE="/var/lock/subsys/qgeth"

start() {
    echo -n $"Starting $DAEMON: "
    daemon --user "$USER" --pidfile="$LOCK_FILE" "$SERVER testnet"
    RETVAL=$?
    echo
    [ $RETVAL -eq 0 ] && touch $LOCK_FILE
    return $RETVAL
}

stop() {
    echo -n $"Shutting down $DAEMON: "
    pid=`ps -aefw | grep "$DAEMON" | grep -v " grep " | awk '{print $2}'`
    kill -9 $pid > /dev/null 2>&1
    [ $? -eq 0 ] && echo_success || echo_failure
    echo
    [ $? -eq 0 ] && rm -f $LOCK_FILE
}

restart() {
    stop
    start
}

status() {
    if [ -f $LOCK_FILE ]; then
        echo "$DAEMON is running."
    else
        echo "$DAEMON is stopped."
    fi
}

case "$1" in
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
        echo "Usage: {start|stop|status|restart}"
        exit 1
        ;;
esac

exit $?
EOF
        chmod +x /etc/init.d/qgeth
        $QGETH_SERVICE_ENABLE qgeth
        ;;
        
    *)
        log_warning "Unknown init system - service creation skipped"
        log_info "Manual service creation required"
        ;;
esac

log_success "✅ Service created and enabled"

# ===========================================
# STEP 8: START SERVICE
# ===========================================
if [ "$QGETH_INIT_SYSTEM" != "unknown" ]; then
    log_info "🚀 Step 8: Starting service"
    
    $QGETH_SERVICE_START qgeth
    
    log_success "✅ Service started successfully"
else
    log_warning "Manual service start required"
fi

# ===========================================
# FINAL STATUS AND INFORMATION
# ===========================================
echo ""
echo "========================================"
echo "🎉 Q Geth Enhanced Bootstrap Completed!"
echo "========================================"
echo ""
echo "📋 Installation Summary:"
echo "  • OS: $QGETH_OS ($QGETH_DISTRO $QGETH_DISTRO_VERSION)"
echo "  • Package Manager: $QGETH_PKG_MANAGER"
echo "  • Init System: $QGETH_INIT_SYSTEM"
echo "  • Firewall: $QGETH_FIREWALL"
echo "  • Architecture: $QGETH_ARCH"
echo "  • Install Directory: $INSTALL_DIR"
echo "  • Project Directory: $PROJECT_DIR"
echo "  • Logs Directory: $LOGS_DIR"
echo "  • User: $ACTUAL_USER"
echo ""
echo "🔧 Service Management:"
case "$QGETH_INIT_SYSTEM" in
    systemd)
        echo "  • Status: systemctl status qgeth"
        echo "  • Logs:   journalctl -u qgeth -f"
        echo "  • Start:  systemctl start qgeth"
        echo "  • Stop:   systemctl stop qgeth"
        echo "  • Restart: systemctl restart qgeth"
        ;;
    openrc)
        echo "  • Status: rc-service qgeth status"
        echo "  • Logs:   tail -f $LOGS_DIR/geth.log"
        echo "  • Start:  rc-service qgeth start"
        echo "  • Stop:   rc-service qgeth stop"
        echo "  • Restart: rc-service qgeth restart"
        ;;
    sysv)
        echo "  • Status: service qgeth status"
        echo "  • Logs:   tail -f $LOGS_DIR/geth.log"
        echo "  • Start:  service qgeth start"
        echo "  • Stop:   service qgeth stop"
        echo "  • Restart: service qgeth restart"
        ;;
    *)
        echo "  • Manual management required"
        echo "  • Script: $PROJECT_DIR/scripts/linux/start-geth.sh"
        ;;
esac
echo ""
echo "🌐 Network Access:"
echo "  • HTTP RPC:  http://$(hostname -I | awk '{print $1}'):8545"
echo "  • WebSocket: ws://$(hostname -I | awk '{print $1}'):8546"
echo "  • P2P Port:  30303"
echo ""
echo "🎯 Next Steps:"
echo "  1. Monitor service status and logs"
echo "  2. Set up mining: cd $PROJECT_DIR && ./scripts/linux/start-miner.sh"
echo "  3. Connect external tools to RPC endpoints"
echo "  4. Review firewall settings for your environment"
echo ""
echo "📚 Documentation: https://github.com/$GITHUB_REPO"
echo ""
log_success "Q Geth is now running and ready for use! 🚀" 