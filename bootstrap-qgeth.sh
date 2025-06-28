#!/bin/bash
# Q Geth Bootstrap Script
# One-command setup for Q Geth auto-updating service
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y

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
    echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash"
    echo ""
    echo "Or download and run:"
    echo "  wget https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh"
    echo "  chmod +x bootstrap-qgeth.sh"
    echo "  sudo ./bootstrap-qgeth.sh"
    exit 1
fi

print_step "🚀 Q Geth One-Command Bootstrap"
echo ""
echo "This will:"
echo "  ✅ Detect and stop any existing Q Geth installations"
echo "  ✅ Install required dependencies (git, curl)"
echo "  ✅ Clone the Q Geth repository"
echo "  ✅ Run the complete auto-service setup"
echo "  ✅ Configure VPS with 4GB memory, firewall, and services"
echo ""

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

# Comprehensive cleanup of existing installations
print_step "🔍 Detecting existing Q Geth installations"

EXISTING_INSTALLATION=false

# Check for systemd services
if systemctl list-units --full -all | grep -E "(qgeth|geth)" | grep -v "grep"; then
    print_warning "Found existing Q Geth systemd services"
    EXISTING_INSTALLATION=true
fi

# Check for running geth processes
if pgrep -f "geth" >/dev/null 2>&1; then
    print_warning "Found running geth processes"
    EXISTING_INSTALLATION=true
fi

# Check for existing installation directories
if [ -d "/opt/qgeth" ] || [ -d "$ACTUAL_HOME/Qgeth3" ]; then
    print_warning "Found existing Q Geth installation directories"
    EXISTING_INSTALLATION=true
fi

# Check for Q Geth management command
if command -v qgeth-service >/dev/null 2>&1; then
    print_warning "Found existing qgeth-service command"
    EXISTING_INSTALLATION=true
fi

if [ "$EXISTING_INSTALLATION" = true ]; then
    print_step "🧹 Performing comprehensive cleanup of existing installations"
    
    # Stop all Q Geth related systemd services
    print_step "Stopping Q Geth systemd services..."
    systemctl stop qgeth-node.service 2>/dev/null || true
    systemctl stop qgeth-github-monitor.service 2>/dev/null || true  
    systemctl stop qgeth-updater.service 2>/dev/null || true
    systemctl disable qgeth-node.service 2>/dev/null || true
    systemctl disable qgeth-github-monitor.service 2>/dev/null || true
    systemctl disable qgeth-updater.service 2>/dev/null || true
    
    # Remove systemd service files
    rm -f /etc/systemd/system/qgeth-*.service
    systemctl daemon-reload
    print_success "✅ Systemd services cleaned up"
    
    # Kill any remaining geth processes
    print_step "Terminating running geth processes..."
    pkill -f "geth" 2>/dev/null || true
    sleep 3
    pkill -9 -f "geth" 2>/dev/null || true
    print_success "✅ Geth processes terminated"
    
    # Remove installation directories
    print_step "Removing installation directories..."
    rm -rf /opt/qgeth 2>/dev/null || true
    rm -f /usr/local/bin/qgeth-service 2>/dev/null || true
    print_success "✅ Installation directories removed"
    
    # Clean up lock files
    print_step "Cleaning up lock files and temp directories..."
    rm -f /tmp/github-monitor.lock 2>/dev/null || true
    rm -f /tmp/update-geth.lock 2>/dev/null || true
    rm -rf /tmp/qgeth-build* 2>/dev/null || true
    print_success "✅ Lock files and temp directories cleaned"
    
    print_success "🧹 Complete cleanup finished"
    echo ""
fi

print_step "🔧 Installing basic dependencies"
if command -v apt >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt update -qq
    DEBIAN_FRONTEND=noninteractive apt install -y git curl golang-go build-essential jq python3 python3-pip
elif command -v yum >/dev/null 2>&1; then
    yum install -y git curl golang gcc jq python3 python3-pip
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y git curl golang gcc jq python3 python3-pip
else
    print_warning "Unknown package manager. Please install dependencies manually."
fi

print_success "✅ Basic dependencies installed"

# Determine working directory
WORK_DIR="$ACTUAL_HOME"
if [ "$ACTUAL_USER" = "root" ]; then
    WORK_DIR="/root"
fi

print_step "📥 Cloning Q Geth repository"
cd "$WORK_DIR"

# Remove existing directory if it exists
if [ -d "Qgeth3" ]; then
    print_step "Removing existing Qgeth3 directory..."
    rm -rf Qgeth3
fi

# Clone the repository
if sudo -u "$ACTUAL_USER" git clone https://github.com/fourtytwo42/Qgeth3.git; then
    print_success "✅ Repository cloned successfully"
else
    print_error "Failed to clone repository"
    print_step "Trying with direct download..."
    
    # Fallback: download as zip
    if command -v wget >/dev/null 2>&1; then
        wget -O Qgeth3.zip https://github.com/fourtytwo42/Qgeth3/archive/main.zip
        unzip -q Qgeth3.zip
        mv Qgeth3-main Qgeth3
        rm Qgeth3.zip
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o Qgeth3.zip https://github.com/fourtytwo42/Qgeth3/archive/main.zip
        unzip -q Qgeth3.zip
        mv Qgeth3-main Qgeth3
        rm Qgeth3.zip
    else
        print_error "Cannot download repository. Please install git, wget, or curl."
        exit 1
    fi
    
    chown -R "$ACTUAL_USER:$ACTUAL_USER" Qgeth3
    print_success "✅ Repository downloaded as zip"
fi

# Change to the project directory
cd Qgeth3

print_step "🔧 Making auto-service script executable"
chmod +x auto-geth-service.sh

print_step "🚀 Starting Q Geth auto-service setup"
echo ""
echo "========================================"
echo "  Starting Auto-Service Installation"
echo "========================================"
echo ""

# Run the auto-service setup
if [ "$AUTO_CONFIRM" = true ]; then
    exec ./auto-geth-service.sh -y
else
    exec ./auto-geth-service.sh
fi 