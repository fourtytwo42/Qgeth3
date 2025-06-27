#!/bin/bash
# Q Geth Bootstrap Script
# One-command setup for Q Geth auto-updating service
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash

set -e

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
echo "  ✅ Install required dependencies (git, curl)"
echo "  ✅ Clone the Q Geth repository"
echo "  ✅ Run the complete auto-service setup"
echo "  ✅ Configure VPS with 4GB memory, firewall, and services"
echo ""

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

print_step "🔧 Installing basic dependencies"
if command -v apt >/dev/null 2>&1; then
    apt update -qq
    apt install -y git curl
elif command -v yum >/dev/null 2>&1; then
    yum install -y git curl
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y git curl
else
    print_warning "Unknown package manager. Please install 'git' and 'curl' manually."
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
exec ./auto-geth-service.sh 