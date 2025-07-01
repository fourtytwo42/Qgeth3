#!/usr/bin/env bash
# Q Geth Simple Bootstrap Script - No Sudo Required
# Installs Q Geth to ~/qgeth/ with all dependencies in user space
# Safe for all Linux distributions, no root privileges needed
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash

set -e

# Configuration
GITHUB_REPO="fourtytwo42/Qgeth3"
INSTALL_DIR="$HOME/qgeth"
PROJECT_DIR="$INSTALL_DIR/Qgeth3"
GO_VERSION="1.24.4"
GO_DIR="$INSTALL_DIR/go"
AUTO_CONFIRM=false

# Parse simple flags
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        --help|-h)
            echo "Q Geth Simple Bootstrap Script - No Sudo Required"
            echo ""
            echo "Installs Q Geth to ~/qgeth/ with all dependencies in user space"
            echo "Safe for all Linux distributions, no root privileges needed"
            echo ""
            echo "Features:"
            echo "  ✅ No sudo required - everything in user space"
            echo "  ✅ Go 1.24.4 installed to ~/qgeth/go/"
            echo "  ✅ Safe for Ubuntu 24.10 and all distributions"
            echo "  ✅ Simple PID-based process management"
            echo "  ✅ Works in restricted environments"
            echo ""
            echo "Usage:"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash"
            echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | bash -s -- -y"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Auto-confirm all prompts"
            echo "  --help       Show this help"
            echo ""
            echo "Requirements:"
            echo "  - curl or wget"
            echo "  - tar"
            echo "  - Basic build tools (gcc, make) - script will check and guide if missing"
            echo ""
            echo "After installation:"
            echo "  cd ~/qgeth/Qgeth3"
            echo "  ./start-qgeth.sh"
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_tools=()
    local optional_tools=()
    
    # Essential tools
    if ! command_exists curl && ! command_exists wget; then
        missing_tools+=("curl or wget")
    fi
    
    if ! command_exists tar; then
        missing_tools+=("tar")
    fi
    
    # Build tools (we'll guide user if missing)
    if ! command_exists gcc; then
        optional_tools+=("gcc")
    fi
    
    if ! command_exists make; then
        optional_tools+=("make")
    fi
    
    if ! command_exists git; then
        optional_tools+=("git")
    fi
    
    # Check for missing essential tools
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing essential tools: ${missing_tools[*]}"
        log_info "Please install these tools first:"
        log_info "  Ubuntu/Debian: sudo apt install curl tar"
        log_info "  CentOS/RHEL:   sudo yum install curl tar"
        log_info "  Fedora:        sudo dnf install curl tar"
        log_info "  Arch:          sudo pacman -S curl tar"
        exit 1
    fi
    
    # Check for optional build tools
    if [ ${#optional_tools[@]} -gt 0 ]; then
        log_warning "Missing build tools: ${optional_tools[*]}"
        log_info "These will be needed for building Q Geth:"
        log_info "  Ubuntu/Debian: sudo apt install git build-essential"
        log_info "  CentOS/RHEL:   sudo yum install git gcc gcc-c++ make"
        log_info "  Fedora:        sudo dnf install git gcc gcc-c++ make"
        log_info "  Arch:          sudo pacman -S git base-devel"
        log_info ""
        
        if [ "$AUTO_CONFIRM" != true ]; then
            echo -n "Install these build tools manually, then continue? (y/N): "
            read -r RESPONSE
            if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                log_info "Bootstrap cancelled. Please install build tools first."
                exit 1
            fi
        else
            log_info "Continuing - you'll need to install build tools manually..."
        fi
    fi
    
    log_success "✅ System requirements check complete"
}

# Check memory (no swap creation, just warn)
check_memory() {
    log_info "Checking available memory..."
    
    if [ -f /proc/meminfo ]; then
        local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        local mem_mb=$((mem_total / 1024))
        
        # Check current swap
        local swap_mb=0
        if [ -f /proc/swaps ]; then
            local swap_total=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
            swap_mb=$((swap_total / 1024))
        fi
        
        local total_mb=$((mem_mb + swap_mb))
        
        log_info "Memory Status:"
        log_info "  RAM: ${mem_mb}MB"
        log_info "  Swap: ${swap_mb}MB"  
        log_info "  Total: ${total_mb}MB"
        
        if [ $total_mb -lt 3072 ]; then  # 3GB threshold
            log_warning "⚠️ Low memory detected (${total_mb}MB total)"
            log_info "Q Geth build may fail with less than 3GB total memory"
            log_info "Consider adding swap if build fails:"
            log_info "  sudo fallocate -l 2G /swapfile"
            log_info "  sudo chmod 600 /swapfile"
            log_info "  sudo mkswap /swapfile"
            log_info "  sudo swapon /swapfile"
            log_info ""
            
            if [ "$AUTO_CONFIRM" != true ]; then
                echo -n "Continue anyway? (y/N): "
                read -r RESPONSE
                if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                    log_info "Bootstrap cancelled."
                    exit 1
                fi
            fi
        else
            log_success "✅ Sufficient memory available (${total_mb}MB)"
        fi
    else
        log_info "Cannot check memory - continuing anyway"
    fi
}

# Install Go 1.24.4 to user directory
install_go() {
    log_info "Installing Go $GO_VERSION to $GO_DIR..."
    
    # Check if Go 1.24.4 is already installed in our directory
    if [ -f "$GO_DIR/bin/go" ]; then
        local go_version_output=$("$GO_DIR/bin/go" version 2>/dev/null || echo "")
        if echo "$go_version_output" | grep -q "go1.24.4"; then
            log_success "✅ Go 1.24.4 already installed in $GO_DIR"
            return 0
        else
            log_info "Different Go version found, updating to 1.24.4..."
            rm -rf "$GO_DIR"
        fi
    fi
    
    # Determine architecture
    local arch=$(uname -m)
    case $arch in
        x86_64)
            local go_arch="amd64"
            ;;
        aarch64|arm64)
            local go_arch="arm64"
            ;;
        armv7l)
            local go_arch="armv6l"
            ;;
        i686)
            local go_arch="386"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    local go_tarball="go${GO_VERSION}.linux-${go_arch}.tar.gz"
    local go_url="https://golang.org/dl/${go_tarball}"
    
    log_info "Downloading Go $GO_VERSION for $go_arch..."
    
    # Create temp directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    # Download Go
    if command_exists curl; then
        curl -fsSL "$go_url" -o "$go_tarball"
    elif command_exists wget; then
        wget -q "$go_url"
    else
        log_error "Neither curl nor wget available"
        exit 1
    fi
    
    # Verify download
    if [ ! -f "$go_tarball" ]; then
        log_error "Go tarball not found after download"
        exit 1
    fi
    
    # Create Go directory
    mkdir -p "$GO_DIR"
    
    # Extract Go to our directory (strip the 'go' folder from tar)
    log_info "Installing Go $GO_VERSION to $GO_DIR..."
    tar -C "$GO_DIR" --strip-components=1 -xzf "$go_tarball"
    
    # Clean up
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    # Verify installation
    if [ -f "$GO_DIR/bin/go" ]; then
        local installed_version=$("$GO_DIR/bin/go" version 2>/dev/null)
        log_success "✅ Go installed: $installed_version"
    else
        log_error "Go installation failed"
        exit 1
    fi
}

# Setup Go PATH for current session and future sessions
setup_go_path() {
    log_info "Setting up Go environment..."
    
    # Add to current session
    export PATH="$GO_DIR/bin:$PATH"
    export GOPATH="$INSTALL_DIR/go-workspace"
    export GOROOT="$GO_DIR"
    
    # Add to user's shell profile
    local shell_profile=""
    if [ -n "$BASH_VERSION" ]; then
        shell_profile="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        shell_profile="$HOME/.zshrc"
    else
        # Default to .profile for compatibility
        shell_profile="$HOME/.profile"
    fi
    
    # Check if already added
    if ! grep -q "# Q Geth Go Environment" "$shell_profile" 2>/dev/null; then
        log_info "Adding Go to $shell_profile..."
        cat >> "$shell_profile" << EOF

# Q Geth Go Environment
export PATH="$GO_DIR/bin:\$PATH"
export GOPATH="$INSTALL_DIR/go-workspace"
export GOROOT="$GO_DIR"
EOF
        log_success "✅ Go environment added to $shell_profile"
        log_info "Run 'source $shell_profile' to update current session, or restart terminal"
    else
        log_info "Go environment already configured in $shell_profile"
    fi
}

# Clone repository
clone_repository() {
    log_info "Cloning Q Geth repository..."
    
    # Create install directory
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Remove existing directory if it exists
    if [ -d "$PROJECT_DIR" ]; then
        log_info "Removing existing Q Geth installation..."
        rm -rf "$PROJECT_DIR"
    fi
    
    # Clone repository
    if command_exists git; then
        git clone "https://github.com/$GITHUB_REPO.git"
        log_success "✅ Repository cloned to $PROJECT_DIR"
    else
        log_info "Git not available, downloading archive..."
        local archive_url="https://github.com/$GITHUB_REPO/archive/refs/heads/main.tar.gz"
        
        if command_exists curl; then
            curl -fsSL "$archive_url" | tar -xz
        elif command_exists wget; then
            wget -qO- "$archive_url" | tar -xz
        fi
        
        # Rename extracted directory
        mv "Qgeth3-main" "Qgeth3" 2>/dev/null || true
        log_success "✅ Repository downloaded to $PROJECT_DIR"
    fi
}

# Build Q Geth
build_qgeth() {
    log_info "Building Q Geth..."
    
    cd "$PROJECT_DIR"
    
    # Use our Go installation
    export PATH="$GO_DIR/bin:$PATH"
    export GOPATH="$INSTALL_DIR/go-workspace"
    export GOROOT="$GO_DIR"
    
    # Verify Go is working
    if ! command_exists go; then
        log_error "Go not found in PATH"
        exit 1
    fi
    
    local go_version=$(go version)
    log_info "Using: $go_version"
    
    # Build using the Linux build script
    if [ -f "scripts/linux/build-linux.sh" ]; then
        log_info "Running build script..."
        cd scripts/linux
        chmod +x build-linux.sh
        
        # Set environment variable to use our temp space
        export QGETH_BUILD_TEMP="$INSTALL_DIR/build-temp"
        
        ./build-linux.sh
        cd ../..
    else
        log_error "Build script not found"
        exit 1
    fi
    
    # Verify build
    if [ -f "geth" ] && [ -f "quantum-miner" ]; then
        log_success "✅ Q Geth built successfully"
    else
        log_error "Build failed - binaries not found"
        exit 1
    fi
}

# Create management scripts
create_management_scripts() {
    log_info "Creating management scripts..."
    
    cd "$PROJECT_DIR"
    
    # Create start script
    cat > start-qgeth.sh << 'EOF'
#!/bin/bash
# Q Geth Startup Script
cd "$(dirname "$0")"

# Add our Go to PATH
export PATH="$(dirname "$0")/../../go/bin:$PATH"

# Run the existing start script
if [ -f "scripts/linux/start-geth.sh" ]; then
    cd scripts/linux
    ./start-geth.sh "$@"
else
    echo "Error: start-geth.sh not found"
    exit 1
fi
EOF

    # Create stop script
    cat > stop-qgeth.sh << 'EOF'
#!/bin/bash
# Q Geth Stop Script
PID_FILE="$HOME/qgeth/Qgeth3/qdata/geth.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Q Geth (PID: $PID)..."
        kill "$PID"
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "Q Geth stopped successfully"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        # Force kill if necessary
        echo "Force stopping Q Geth..."
        kill -9 "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
    else
        echo "Q Geth not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "Q Geth not running"
fi
EOF

    # Create status script
    cat > status-qgeth.sh << 'EOF'
#!/bin/bash
# Q Geth Status Script
PID_FILE="$HOME/qgeth/Qgeth3/qdata/geth.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Q Geth is running (PID: $PID)"
        
        # Show process info
        echo "Process info:"
        ps -p "$PID" -o pid,ppid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null || echo "  Process details unavailable"
        
        # Show RPC status
        echo ""
        echo "RPC Status:"
        if curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' http://localhost:8545 >/dev/null 2>&1; then
            echo "  HTTP RPC: Available on http://localhost:8545"
        else
            echo "  HTTP RPC: Not responding"
        fi
        
        if curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' http://localhost:8546 >/dev/null 2>&1; then
            echo "  WebSocket: Available on ws://localhost:8546"
        else
            echo "  WebSocket: Not responding"
        fi
        
        exit 0
    else
        echo "Q Geth not running (stale PID file)"
        rm -f "$PID_FILE"
        exit 1
    fi
else
    echo "Q Geth not running"
    exit 1
fi
EOF

    # Create restart script
    cat > restart-qgeth.sh << 'EOF'
#!/bin/bash
# Q Geth Restart Script
cd "$(dirname "$0")"

echo "Restarting Q Geth..."
./stop-qgeth.sh
sleep 2
./start-qgeth.sh "$@"
EOF

    # Make scripts executable
    chmod +x start-qgeth.sh stop-qgeth.sh status-qgeth.sh restart-qgeth.sh
    
    log_success "✅ Management scripts created"
}

# Main installation function
main() {
    echo -e "${CYAN}🚀 Q Geth Simple Bootstrap - No Sudo Required${NC}"
    echo ""
    echo "This script will:"
    echo "  📦 Install Go 1.24.4 to ~/qgeth/go/"
    echo "  🔽 Clone Q Geth to ~/qgeth/Qgeth3/"
    echo "  🔨 Build Q Geth binaries"
    echo "  📜 Create management scripts"
    echo "  ✨ No root privileges required!"
    echo ""
    
    if [ "$AUTO_CONFIRM" != true ]; then
        echo -n "Continue with installation? (y/N): "
        read -r RESPONSE
        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            log_info "Installation cancelled"
            exit 0
        fi
    fi
    
    echo ""
    log_info "Starting Q Geth installation..."
    
    # Run installation steps
    check_requirements
    check_memory
    install_go
    setup_go_path
    clone_repository
    build_qgeth
    create_management_scripts
    
    echo ""
    echo -e "${GREEN}🎉 Q Geth Installation Complete!${NC}"
    echo ""
    echo "Installation Summary:"
    echo "  📍 Location: $PROJECT_DIR"
    echo "  🔧 Go: $GO_DIR"
    echo "  ⚡ Management: Simple PID-based scripts"
    echo ""
    echo "Quick Start:"
    echo "  cd ~/qgeth/Qgeth3"
    echo "  ./start-qgeth.sh         # Start Q Geth"
    echo "  ./status-qgeth.sh        # Check status"
    echo "  ./stop-qgeth.sh          # Stop Q Geth"
    echo "  ./restart-qgeth.sh       # Restart Q Geth"
    echo ""
    echo "Network Access:"
    echo "  HTTP RPC:    http://localhost:8545"
    echo "  WebSocket:   ws://localhost:8546"
    echo "  P2P Network: 30303"
    echo ""
    echo "Next Steps:"
    echo "  1. cd ~/qgeth/Qgeth3"
    echo "  2. ./start-qgeth.sh"
    echo "  3. Check status with ./status-qgeth.sh"
    echo ""
    echo "Documentation: https://github.com/$GITHUB_REPO"
    echo ""
    log_success "✅ Ready to start your quantum blockchain journey!"
}

# Run main function
main "$@" 