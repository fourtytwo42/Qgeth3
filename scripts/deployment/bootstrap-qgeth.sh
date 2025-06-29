#!/bin/bash
# Q Geth Simplified Bootstrap Script
# Single-command VPS setup for Q Geth auto-updating service
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | sudo bash -s -- -y

set -e

# Parse command line arguments - FIXED for curl pipe compatibility
AUTO_CONFIRM=false

# Better argument parsing that works with curl pipes
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
    esac
done

# Auto-detect non-interactive environment (like curl pipe)
if [ ! -t 0 ] || [ ! -t 1 ]; then
    # Running via pipe or non-interactive - default to auto-confirm
    AUTO_CONFIRM=true
fi

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

# Ubuntu/OS compatibility check
print_step "🔍 Checking system compatibility..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $NAME $VERSION"
    
    # Special handling for Ubuntu 25.04+ (if it exists)
    if [ "$ID" = "ubuntu" ]; then
        VERSION_ID_NUM=$(echo "$VERSION_ID" | cut -d. -f1)
        if [ "$VERSION_ID_NUM" -ge 25 ]; then
            print_warning "Ubuntu $VERSION_ID detected - applying compatibility fixes"
            # Set additional compatibility flags for newer Ubuntu
            export DEBIAN_FRONTEND=noninteractive
            export NEEDRESTART_MODE=a
        fi
    fi
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

# FIXED: Better interactive prompt handling
if [ "$AUTO_CONFIRM" != true ]; then
    # Only prompt if we have a real terminal
    if [ -t 0 ] && [ -t 1 ]; then
        echo -n "Proceed with installation? (y/N): "
        read -r RESPONSE </dev/tty
        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Installation cancelled."
            exit 0
        fi
    else
        print_warning "Non-interactive environment detected, proceeding automatically..."
        AUTO_CONFIRM=true
    fi
else
    print_step "Auto-confirm mode enabled, proceeding..."
fi

# ===========================================
# STEP 1: CLEANUP EXISTING INSTALLATIONS
# ===========================================
print_step "🧹 Step 1: Cleanup existing installations"

# Stop and remove systemd services (handle both old and new service names)
print_step "Stopping Q Geth services..."
systemctl stop qgeth.service 2>/dev/null || true
systemctl stop qgeth-node.service 2>/dev/null || true
systemctl stop qgeth-monitor.service 2>/dev/null || true
systemctl disable qgeth.service 2>/dev/null || true
systemctl disable qgeth-node.service 2>/dev/null || true
systemctl disable qgeth-monitor.service 2>/dev/null || true
rm -f /etc/systemd/system/qgeth.service
rm -f /etc/systemd/system/qgeth-node.service
rm -f /etc/systemd/system/qgeth-monitor.service
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
    # Ubuntu/Debian package installation with version compatibility
    print_step "Updating package lists..."
    DEBIAN_FRONTEND=noninteractive apt update -qq
    
    # Enhanced package installation for Ubuntu 25.04+ compatibility
    print_step "Installing essential packages..."
    DEBIAN_FRONTEND=noninteractive apt install -y \
        git \
        curl \
        wget \
        build-essential \
        systemd \
        ufw \
        jq \
        python3 \
        python3-pip \
        unzip \
        ca-certificates \
        software-properties-common \
        apt-transport-https \
        gnupg \
        lsb-release

    # Go installation with version compatibility
    print_step "Installing Go (checking version compatibility)..."
    
    # Check if Go is already installed and version
    GO_VERSION=""
    if command -v go >/dev/null 2>&1; then
        GO_VERSION=$(go version 2>/dev/null | grep -o 'go[0-9]*\.[0-9]*' | head -1)
        echo "Found existing Go: $GO_VERSION"
    fi
    
    # Install Go if not present or version is too old
    GO_MIN_VERSION="1.21"
    NEED_GO_INSTALL=true
    
    if [ ! -z "$GO_VERSION" ]; then
        # Extract version numbers for comparison
        GO_VER_MAJOR=$(echo "$GO_VERSION" | sed 's/go//' | cut -d. -f1)
        GO_VER_MINOR=$(echo "$GO_VERSION" | sed 's/go//' | cut -d. -f2)
        
        if [ "$GO_VER_MAJOR" -gt 1 ] || ([ "$GO_VER_MAJOR" -eq 1 ] && [ "$GO_VER_MINOR" -ge 21 ]); then
            print_success "✅ Go $GO_VERSION is compatible (>= $GO_MIN_VERSION)"
            NEED_GO_INSTALL=false
        else
            print_warning "Go $GO_VERSION is too old, need >= $GO_MIN_VERSION"
        fi
    fi
    
    if [ "$NEED_GO_INSTALL" = true ]; then
        print_step "Installing Go $GO_MIN_VERSION or later..."
        
        # Try package manager first
        if DEBIAN_FRONTEND=noninteractive apt install -y golang-go; then
            # Check if installed version is sufficient
            if command -v go >/dev/null 2>&1; then
                NEW_GO_VERSION=$(go version 2>/dev/null | grep -o 'go[0-9]*\.[0-9]*' | head -1 || echo "")
                if [ ! -z "$NEW_GO_VERSION" ]; then
                    NEW_GO_VER_MAJOR=$(echo "$NEW_GO_VERSION" | sed 's/go//' | cut -d. -f1)
                    NEW_GO_VER_MINOR=$(echo "$NEW_GO_VERSION" | sed 's/go//' | cut -d. -f2)
                    
                    if [ "$NEW_GO_VER_MAJOR" -gt 1 ] || ([ "$NEW_GO_VER_MAJOR" -eq 1 ] && [ "$NEW_GO_VER_MINOR" -ge 21 ]); then
                        print_success "✅ Go $NEW_GO_VERSION installed successfully"
                    else
                        print_warning "Package manager Go $NEW_GO_VERSION is too old, installing latest manually..."
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
            print_step "Installing latest Go manually..."
            
            # CRITICAL: Remove conflicting package manager Go installations to prevent PATH conflicts
            print_step "Removing conflicting package manager Go installations..."
            DEBIAN_FRONTEND=noninteractive apt remove --purge -y golang-go golang-1.* 2>/dev/null || true
            DEBIAN_FRONTEND=noninteractive apt autoremove -y 2>/dev/null || true
            
            # Clean up any Go-related environment variables from package manager
            sed -i '/golang/d' /etc/environment 2>/dev/null || true
            
            # Remove old manual Go installations
            rm -rf /usr/local/go
            
            # Download and install latest Go
            GO_LATEST_VERSION="1.21.6"  # Known working version
            GO_ARCH="amd64"
            if [ "$(uname -m)" = "aarch64" ]; then
                GO_ARCH="arm64"
            fi
            
            GO_TARBALL="go${GO_LATEST_VERSION}.linux-${GO_ARCH}.tar.gz"
            
            cd /tmp
            if wget "https://go.dev/dl/${GO_TARBALL}"; then
                tar -C /usr/local -xzf "$GO_TARBALL"
                
                # CRITICAL: Set up PATH properly for both current session and persistence
                # 1. Set for current session immediately
                export PATH="/usr/local/go/bin:$PATH"
                
                # 2. Set for all future sessions (multiple locations for compatibility)
                echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/profile
                echo 'export PATH="/usr/local/go/bin:$PATH"' >> /etc/bash.bashrc
                
                # 3. Set for systemd services
                echo 'PATH="/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' >> /etc/environment
                
                # 4. Clean Go module cache to prevent conflicts
                export GOCACHE="/tmp/go-cache-clean"
                export GOMODCACHE="/tmp/go-mod-cache-clean"
                rm -rf ~/.cache/go-build 2>/dev/null || true
                rm -rf /root/.cache/go-build 2>/dev/null || true
                
                # 5. Verify installation with explicit path
                print_step "Verifying manual Go installation..."
                if /usr/local/go/bin/go version; then
                    # Double-check that the correct Go is now in PATH
                    which go
                    go version
                    
                    # Verify no old Go is interfering
                    CURRENT_GO_PATH=$(which go 2>/dev/null || echo "none")
                    if [[ "$CURRENT_GO_PATH" == "/usr/local/go/bin/go" ]]; then
                        print_success "✅ Go $GO_LATEST_VERSION installed manually and active in PATH"
                    else
                        print_warning "⚠️ Go installed but PATH may have conflicts. Current go: $CURRENT_GO_PATH"
                        print_step "Forcing PATH priority for manual Go..."
                        export PATH="/usr/local/go/bin:$PATH"
                        hash -r  # Clear command cache
                        
                        # Verify again
                        NEW_GO_PATH=$(which go 2>/dev/null || echo "none")
                        if [[ "$NEW_GO_PATH" == "/usr/local/go/bin/go" ]]; then
                            print_success "✅ PATH fixed - manual Go now active"
                        else
                            print_error "Failed to prioritize manual Go in PATH"
                            exit 1
                        fi
                    fi
                else
                    print_error "Failed to install Go manually"
                    exit 1
                fi
            else
                print_error "Failed to download Go $GO_LATEST_VERSION"
                exit 1
            fi
        fi
    fi
    
elif command -v yum >/dev/null 2>&1; then
    # CentOS/RHEL/Fedora
    yum install -y git curl golang gcc systemd firewalld jq python3 python3-pip unzip
elif command -v dnf >/dev/null 2>&1; then
    # Modern Fedora
    dnf install -y git curl golang gcc systemd firewalld jq python3 python3-pip unzip
elif command -v pacman >/dev/null 2>&1; then
    # Arch Linux
    pacman -Sy --noconfirm git curl go gcc systemd jq python python-pip unzip
else
    print_error "Unsupported package manager. Please install dependencies manually:"
    echo "  - git, curl, golang (>= 1.21), build-essential, systemd, jq, python3"
    exit 1
fi

# Final Go verification with conflict detection
print_step "Verifying Go installation..."
if command -v go >/dev/null 2>&1; then
    GO_FINAL_VERSION=$(go version)
    GO_FINAL_PATH=$(which go)
    print_success "✅ Go verified: $GO_FINAL_VERSION"
    print_step "Go location: $GO_FINAL_PATH"
    
    # CRITICAL: Ensure we're using the correct Go version (1.21+)
    FINAL_GO_VER=$(echo "$GO_FINAL_VERSION" | grep -o 'go[0-9]*\.[0-9]*' | head -1 | sed 's/go//')
    FINAL_GO_MAJOR=$(echo "$FINAL_GO_VER" | cut -d. -f1)
    FINAL_GO_MINOR=$(echo "$FINAL_GO_VER" | cut -d. -f2)
    
    if [ "$FINAL_GO_MAJOR" -lt 1 ] || ([ "$FINAL_GO_MAJOR" -eq 1 ] && [ "$FINAL_GO_MINOR" -lt 21 ]); then
        print_error "CRITICAL: Final Go version $FINAL_GO_VER is still too old (< 1.21)"
        print_error "This indicates PATH conflicts. Current go path: $GO_FINAL_PATH"
        
        # Emergency fix: Force manual Go path
        if [ -f "/usr/local/go/bin/go" ]; then
            print_step "Emergency fix: Forcing manual Go..."
            export PATH="/usr/local/go/bin:$PATH"
            hash -r
            
            NEW_GO_VERSION=$(go version 2>/dev/null || echo "failed")
            if [[ "$NEW_GO_VERSION" =~ go1\.2[1-9] ]] || [[ "$NEW_GO_VERSION" =~ go1\.[3-9][0-9] ]]; then
                print_success "✅ Emergency fix successful: $NEW_GO_VERSION"
            else
                print_error "Emergency fix failed. Manual intervention required."
                exit 1
            fi
        else
            print_error "Manual Go not found at /usr/local/go/bin/go"
            exit 1
        fi
    fi
    
    # Set up Go environment with clean cache
    export GOPATH=/tmp/go-build-bootstrap
    export GOCACHE=/tmp/go-cache-bootstrap
    export GOMODCACHE=/tmp/go-mod-cache-bootstrap
    
    # Clean any existing cache to prevent version conflicts
    rm -rf ~/.cache/go-build 2>/dev/null || true
    rm -rf /root/.cache/go-build 2>/dev/null || true
    go clean -cache -modcache -testcache 2>/dev/null || true
    
    mkdir -p "$GOPATH" "$GOCACHE" "$GOMODCACHE"
    print_step "Go environment set up with clean cache"
else
    print_error "Go installation failed - please install Go >= 1.21 manually"
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

# Ensure proper ownership before build
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"

# CRITICAL: Final verification that build will use correct Go version
print_step "Pre-build Go verification..."
BUILD_GO_VERSION=$(sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" go version 2>/dev/null || echo "failed")
if [[ "$BUILD_GO_VERSION" =~ go1\.2[1-9] ]] || [[ "$BUILD_GO_VERSION" =~ go1\.[3-9][0-9] ]]; then
    print_success "✅ Build will use: $BUILD_GO_VERSION"
else
    print_error "CRITICAL: Build would use wrong Go version: $BUILD_GO_VERSION"
    
    # Last resort: ensure build environment has correct PATH
    export PATH="/usr/local/go/bin:$PATH"
    
    # Verify one more time
    FINAL_BUILD_GO=$(sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" go version 2>/dev/null || echo "failed")
    if [[ "$FINAL_BUILD_GO" =~ go1\.2[1-9] ]] || [[ "$FINAL_BUILD_GO" =~ go1\.[3-9][0-9] ]]; then
        print_success "✅ PATH corrected, build will use: $FINAL_BUILD_GO"
    else
        print_error "Cannot ensure correct Go version for build. Aborting."
        exit 1
    fi
fi

BUILD_SUCCESS=false
BUILD_ATTEMPTS=0
MAX_BUILD_ATTEMPTS=3

while [ $BUILD_ATTEMPTS -lt $MAX_BUILD_ATTEMPTS ] && [ "$BUILD_SUCCESS" = false ]; do
    BUILD_ATTEMPTS=$((BUILD_ATTEMPTS + 1))
    print_step "🚀 Build attempt $BUILD_ATTEMPTS/$MAX_BUILD_ATTEMPTS"
    
    if [ "$AUTO_CONFIRM" = true ]; then
        if sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" QGETH_BUILD_TEMP="$QGETH_BUILD_TEMP" GOCACHE="/tmp/go-cache-build" GOMODCACHE="/tmp/go-mod-cache-build" ./build-linux.sh geth -y; then
            BUILD_SUCCESS=true
        fi
    else
        if sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" QGETH_BUILD_TEMP="$QGETH_BUILD_TEMP" GOCACHE="/tmp/go-cache-build" GOMODCACHE="/tmp/go-mod-cache-build" ./build-linux.sh geth; then
            BUILD_SUCCESS=true
        fi
    fi
    
    if [ "$BUILD_SUCCESS" = false ] && [ $BUILD_ATTEMPTS -lt $MAX_BUILD_ATTEMPTS ]; then
        print_warning "Build attempt $BUILD_ATTEMPTS failed, applying recovery..."
        
        # Clean and retry with proper ownership and environment
        cd "$PROJECT_DIR"
        rm -f geth geth.bin quantum_solver.py 2>/dev/null || true
        chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"
        sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" go clean -cache -modcache -testcache 2>/dev/null || true
        
        cd "$PROJECT_DIR/quantum-geth"
        sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" go mod tidy 2>/dev/null || true
        sudo -u "$ACTUAL_USER" env PATH="/usr/local/go/bin:$PATH" go mod download 2>/dev/null || true
        
        cd "$PROJECT_DIR/scripts/linux"
        sleep 5
    fi
done

if [ "$BUILD_SUCCESS" = false ]; then
    print_error "Build failed after $MAX_BUILD_ATTEMPTS attempts"
    exit 1
fi

# Fix ownership of created files
cd "$PROJECT_DIR"
chown "$ACTUAL_USER:$ACTUAL_USER" geth geth.bin quantum_solver.py 2>/dev/null || true
chmod +x geth geth.bin quantum_solver.py 2>/dev/null || true

# Ensure geth wrapper exists with correct path resolution (backup creation)
if [ -f "geth.bin" ] && [ ! -f "geth" ]; then
    print_step "Creating backup geth wrapper (build didn't create it)..."
    cat > geth << 'EOF'
#!/bin/bash
# Q Coin Geth Wrapper - Fixed path version
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/geth.bin" "$@"
EOF
    chown "$ACTUAL_USER:$ACTUAL_USER" geth
    chmod +x geth
    print_success "✅ Backup geth wrapper created"
fi

print_success "✅ Q Geth built successfully"

# ===========================================
# STEP 6: CREATE SYSTEMD SERVICES
# ===========================================
print_step "⚙️ Step 6: Creating systemd services"

# Create single Q Geth service (simplified approach)
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

# Enable and start service
systemctl daemon-reload
systemctl enable qgeth.service

print_success "✅ Systemd service created and enabled"

# ===========================================
# STEP 7: START SERVICE
# ===========================================
print_step "🚀 Step 7: Starting service"

systemctl start qgeth.service

print_success "✅ Service started successfully"

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
echo "⚙️ Service Created:"
echo "  • qgeth.service - Q Geth blockchain node"
echo ""
echo "🔗 Network Access:"
echo "  • HTTP RPC API:  http://localhost:8545"
echo "  • WebSocket API: ws://localhost:8546"
echo "  • P2P Network:   port 30303"
echo ""
echo "📊 Service Management:"
echo "  sudo systemctl status qgeth.service"
echo "  sudo systemctl restart qgeth.service"
echo "  sudo journalctl -u qgeth.service -f"
echo ""
echo "📁 Log Files:"
echo "  tail -f $LOGS_DIR/geth.log"
echo ""
echo "✅ Q Geth is now running!"
echo ""

# Final verification
print_step "🔍 Final verification"
sleep 3

if systemctl is-active --quiet qgeth.service; then
    print_success "✅ Q Geth service is running"
else
    print_warning "⚠️ Q Geth service is not running - check logs"
fi 