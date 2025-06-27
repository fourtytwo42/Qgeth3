#!/bin/bash
# VPS Preparation Script for Q Geth
# Prepares low-memory VPS environments for building
# Checks memory, sets up swap space, and installs dependencies
# Usage: sudo ./prepare-vps.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
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
    echo "Usage: sudo ./prepare-vps.sh"
    exit 1
fi

print_step "ðŸš€ Preparing VPS for Q Geth Build"
echo ""

# System info
print_step "ðŸ“Š System Information"
if command -v lsb_release >/dev/null 2>&1; then
    echo "OS: $(lsb_release -d | cut -f2)"
elif [ -f /etc/os-release ]; then
    echo "OS: $(grep PRETTY_NAME /etc/os-release | cut -d '"' -f2)"
fi

if command -v uname >/dev/null 2>&1; then
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
fi

echo ""

# Memory check
print_step "ðŸ’¾ Memory Analysis"
REQUIRED_MB=3072  # 3GB minimum
AVAILABLE_MB=0
TOTAL_MB=0

if [ -f /proc/meminfo ]; then
    MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_AVAILABLE=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    
    TOTAL_MB=$((MEM_TOTAL / 1024))
    
    if [ -n "$MEM_AVAILABLE" ]; then
        AVAILABLE_MB=$((MEM_AVAILABLE / 1024))
    else
        AVAILABLE_MB=$TOTAL_MB
    fi
    
    echo "Total RAM: ${TOTAL_MB}MB"
    echo "Available RAM: ${AVAILABLE_MB}MB"
    echo "Required RAM: ${REQUIRED_MB}MB"
    echo ""
    
    if [ $AVAILABLE_MB -lt $REQUIRED_MB ]; then
        print_warning "Insufficient memory for building!"
        echo "  Available: ${AVAILABLE_MB}MB"
        echo "  Required: ${REQUIRED_MB}MB"
        echo "  Deficit: $((REQUIRED_MB - AVAILABLE_MB))MB"
        echo ""
        
        # Check existing swap
        SWAP_TOTAL=0
        if [ -f /proc/swaps ]; then
            SWAP_KB=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
            SWAP_TOTAL=$((SWAP_KB / 1024))
        fi
        
        echo "Current swap space: ${SWAP_TOTAL}MB"
        
        # Calculate needed swap
        NEEDED_SWAP=$((REQUIRED_MB - AVAILABLE_MB - SWAP_TOTAL))
        if [ $NEEDED_SWAP -lt 1024 ]; then
            NEEDED_SWAP=2048  # Minimum 2GB swap
        fi
        
        echo "Recommended swap: ${NEEDED_SWAP}MB"
        echo ""
        
        echo -n "Create swap file to enable building? (y/N): "
        read -r RESPONSE
        if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            create_swap $NEEDED_SWAP
        else
            print_warning "Swap file not created. Build may fail due to insufficient memory."
        fi
    else
        print_success "Memory check passed - sufficient RAM available"
    fi
else
    print_error "Cannot check memory - /proc/meminfo not found"
fi

echo ""

# Storage check
print_step "ðŸ’½ Storage Analysis"
AVAILABLE_GB=0
if command -v df >/dev/null 2>&1; then
    AVAILABLE_KB=$(df . | awk 'NR==2 {print $4}')
    AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
    echo "Available storage: ${AVAILABLE_GB}GB"
    
    if [ $AVAILABLE_GB -lt 5 ]; then
        print_warning "Low storage space detected!"
        echo "  Available: ${AVAILABLE_GB}GB"
        echo "  Recommended: 5GB+ for build artifacts"
        echo ""
    else
        print_success "Storage check passed"
    fi
else
    print_warning "Cannot check storage space"
fi

echo ""

# Dependencies check
print_step "ðŸ“¦ Dependencies Check"
MISSING_DEPS=()

# Check for essential tools
if ! command -v git >/dev/null 2>&1; then
    MISSING_DEPS+=("git")
fi

if ! command -v curl >/dev/null 2>&1; then
    MISSING_DEPS+=("curl")
fi

if ! command -v go >/dev/null 2>&1; then
    MISSING_DEPS+=("golang-go")
fi

if ! command -v make >/dev/null 2>&1; then
    MISSING_DEPS+=("build-essential")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_warning "Missing dependencies detected"
    echo "Missing: ${MISSING_DEPS[*]}"
    echo ""
    echo -n "Install missing dependencies? (y/N): "
    read -r RESPONSE
    if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        install_dependencies
    fi
else
    print_success "All dependencies are installed"
fi

echo ""

# Final recommendations
print_step "ðŸŽ¯ Final Recommendations"
echo ""
echo "Build Environment Tips:"
echo "  1. Close unnecessary applications during build"
echo "  2. Monitor build progress with 'htop' or 'top'"
echo "  3. Use 'ionice -c3' for lower I/O priority if needed"
echo "  4. Consider building during off-peak hours"
echo ""

if [ $AVAILABLE_MB -ge $REQUIRED_MB ]; then
    print_success "âœ… VPS is ready for building Q Geth!"
    echo ""
    echo "Next steps:"
    echo "  ./build-linux.sh            # Build both geth and miner"
    echo "  ./build-linux.sh geth       # Build geth only"
    echo "  ./build-linux.sh miner      # Build miner only"
else
    print_warning "âš ï¸  VPS may have issues building due to low memory"
    echo ""
    echo "Consider:"
    echo "  - Upgrading to a VPS with at least 4GB RAM"
    echo "  - Running this script again to add swap space"
    echo "  - Building only one component at a time"
fi

echo ""
print_step "ðŸ” System Summary"
echo "RAM: ${AVAILABLE_MB}MB available / ${TOTAL_MB}MB total"
echo "Swap: ${SWAP_TOTAL}MB"
echo "Storage: ${AVAILABLE_GB}GB available"
echo "Status: $([ $AVAILABLE_MB -ge $REQUIRED_MB ] && echo "âœ… Ready" || echo "âš ï¸  May need optimization")"

create_swap() {
    local size_mb=$1
    local swap_file="/swapfile"
    
    print_step "ðŸ”§ Creating ${size_mb}MB swap file..."
    
    # Check if swap file already exists
    if [ -f "$swap_file" ]; then
        print_warning "Swap file already exists at $swap_file"
        echo -n "Replace existing swap file? (y/N): "
        read -r RESPONSE
        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            return 0
        fi
        
        # Turn off existing swap
        swapoff "$swap_file" 2>/dev/null || true
        rm -f "$swap_file"
    fi
    
    # Create swap file
    echo "Creating swap file..."
    if command -v fallocate >/dev/null 2>&1; then
        fallocate -l "${size_mb}M" "$swap_file"
    else
        dd if=/dev/zero of="$swap_file" bs=1024 count=$((size_mb * 1024))
    fi
    
    # Set permissions
    chmod 600 "$swap_file"
    
    # Make swap
    echo "Setting up swap..."
    mkswap "$swap_file"
    swapon "$swap_file"
    
    # Add to fstab for persistence
    if ! grep -q "$swap_file" /etc/fstab; then
        echo "$swap_file none swap sw 0 0" >> /etc/fstab
        echo "Added swap to /etc/fstab for persistence"
    fi
    
    # Verify swap
    if swapon --show | grep -q "$swap_file"; then
        print_success "Swap file created and activated successfully"
        
        # Update global variable
        SWAP_KB=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
        SWAP_TOTAL=$((SWAP_KB / 1024))
    else
        print_error "Failed to activate swap file"
        return 1
    fi
}

install_dependencies() {
    print_step "ðŸ“¦ Installing dependencies..."
    
    # Update package list
    echo "Updating package list..."
    apt update -qq
    
    # Install packages
    echo "Installing packages: ${MISSING_DEPS[*]}"
    apt install -y "${MISSING_DEPS[@]}"
    
    # Verify Go installation
    if command -v go >/dev/null 2>&1; then
        GO_VERSION=$(go version | awk '{print $3}')
        echo "Go installed: $GO_VERSION"
        
        # Check if Go version is recent enough
        GO_MAJOR=$(echo "$GO_VERSION" | sed 's/go//' | cut -d. -f1)
        GO_MINOR=$(echo "$GO_VERSION" | sed 's/go//' | cut -d. -f2)
        
        if [ "$GO_MAJOR" -lt 1 ] || ([ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -lt 18 ]); then
            print_warning "Go version may be too old (need 1.18+)"
            echo "Consider installing a newer Go version from https://golang.org/dl/"
        fi
    fi
    
    print_success "Dependencies installed successfully"
} 