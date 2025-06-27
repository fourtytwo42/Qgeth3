#!/bin/bash
# VPS Preparation Script for Q Geth
# Prepares low-memory VPS environments for building
# Checks memory, sets up swap space, and installs dependencies
# Usage: sudo ./prepare-vps.sh [-y|--yes]

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

# Function to create swap file
create_swap() {
    local size_mb=$1
    local swap_file="/swapfile"
    
    print_step "🔧 Creating ${size_mb}MB swap file..."
    
    # Check if swap file already exists
    if [ -f "$swap_file" ]; then
        print_warning "Swap file already exists at $swap_file"
        if [ "$AUTO_CONFIRM" = true ]; then
            print_step "Auto-confirming: Replacing existing swap file"
        else
            echo -n "Replace existing swap file? (y/N): "
            read -r RESPONSE
            if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                return 0
            fi
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

# Function to install dependencies
install_dependencies() {
    print_step "📦 Installing dependencies..."
    
    # Update package list
    echo "Updating package list..."
    DEBIAN_FRONTEND=noninteractive apt update -qq
    
    # Install packages
    echo "Installing packages: ${MISSING_DEPS[*]}"
    DEBIAN_FRONTEND=noninteractive apt install -y "${MISSING_DEPS[@]}"
    
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

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script with sudo"
    echo "Usage: sudo ./prepare-vps.sh"
    exit 1
fi

print_step "🚀 Preparing VPS for Q Geth Build"
echo ""

# System info
print_step "📊 System Information"
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
print_step "💾 Memory Analysis"
REQUIRED_MB=4096  # 4GB minimum total (RAM + swap)
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
    echo "Required Total Memory: ${REQUIRED_MB}MB (4GB)"
    echo ""
    
    # Check existing swap
    SWAP_TOTAL=0
    if [ -f /proc/swaps ]; then
        SWAP_KB=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
        SWAP_TOTAL=$((SWAP_KB / 1024))
    fi
    
    CURRENT_TOTAL=$((TOTAL_MB + SWAP_TOTAL))
    echo "Current swap space: ${SWAP_TOTAL}MB"
    echo "Current total memory: ${CURRENT_TOTAL}MB (RAM + swap)"
    echo ""
    
    # Add tolerance margin - 50MB difference is acceptable
    local tolerance_mb=50
    local effective_required=$((REQUIRED_MB - tolerance_mb))
    
    if [ $CURRENT_TOTAL -lt $effective_required ]; then
        local deficit=$((REQUIRED_MB - CURRENT_TOTAL))
        print_warning "Insufficient total memory for building!"
        echo "  Current total: ${CURRENT_TOTAL}MB"
        echo "  Required total: ${REQUIRED_MB}MB"
        echo "  Deficit: ${deficit}MB"
        echo ""
        
        # Calculate exact swap needed to reach 4GB total
        NEEDED_SWAP=$((REQUIRED_MB - TOTAL_MB))
        if [ $NEEDED_SWAP -lt 0 ]; then
            NEEDED_SWAP=0
        fi
        
        # If we already have some swap, subtract it
        if [ $SWAP_TOTAL -gt 0 ]; then
            NEEDED_SWAP=$((NEEDED_SWAP - SWAP_TOTAL))
        fi
        
        # Ensure we don't create negative swap
        if [ $NEEDED_SWAP -lt 0 ]; then
            NEEDED_SWAP=0
        fi
        
        if [ $NEEDED_SWAP -gt 0 ]; then
            echo "Need to create: ${NEEDED_SWAP}MB additional swap"
            echo "This will give total: $((TOTAL_MB + SWAP_TOTAL + NEEDED_SWAP))MB"
            echo ""
            
            if [ "$AUTO_CONFIRM" = true ]; then
                print_step "Auto-confirming: Creating swap file to reach 4GB total memory"
                create_swap $NEEDED_SWAP
            else
                echo -n "Create swap file to reach 4GB total memory? (y/N): "
                read -r RESPONSE
                if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                    create_swap $NEEDED_SWAP
                else
                    print_warning "Swap file not created. Build may fail due to insufficient memory."
                fi
            fi
        else
            print_success "Total memory check passed"
        fi
    elif [ $CURRENT_TOTAL -lt $REQUIRED_MB ]; then
        local deficit=$((REQUIRED_MB - CURRENT_TOTAL))
        print_step "💡 Memory within tolerance range"
        echo "  Current total: ${CURRENT_TOTAL}MB"
        echo "  Required total: ${REQUIRED_MB}MB" 
        echo "  Deficit: ${deficit}MB (within ${tolerance_mb}MB tolerance)"
        echo "  ✅ Proceeding with build - memory difference is acceptable"
        print_success "Total memory check passed (within tolerance)"
    else
        print_success "Memory check passed - sufficient total memory available"
    fi
else
    print_error "Cannot check memory - /proc/meminfo not found"
fi

echo ""

# Temporary directory setup
print_step "📁 Temporary Directory Setup"
TEMP_DIR="/tmp/qgeth-build"
TEMP_AVAILABLE_GB=0

# Check /tmp space
if command -v df >/dev/null 2>&1; then
    TEMP_AVAILABLE_KB=$(df /tmp | awk 'NR==2 {print $4}')
    TEMP_AVAILABLE_GB=$((TEMP_AVAILABLE_KB / 1024 / 1024))
    echo "Available /tmp space: ${TEMP_AVAILABLE_GB}GB"
    
    if [ $TEMP_AVAILABLE_GB -lt 2 ]; then
        print_warning "Low /tmp space detected!"
        echo "  Available: ${TEMP_AVAILABLE_GB}GB"
        echo "  Recommended: 2GB+ for build temp files"
        echo ""
        
        # Try to create temp build directory in current location
        BUILD_TEMP_DIR="./build-temp"
        echo "Will use local build temp directory: $BUILD_TEMP_DIR"
        mkdir -p "$BUILD_TEMP_DIR"
        TEMP_DIR="$BUILD_TEMP_DIR"
    else
        echo "Creating build temp directory: $TEMP_DIR"
        mkdir -p "$TEMP_DIR"
        # Set proper permissions
        chmod 755 "$TEMP_DIR"
        print_success "Temporary directory setup complete"
    fi
    
    echo "Build temp directory: $TEMP_DIR"
    echo ""
else
    print_warning "Cannot check /tmp space"
    mkdir -p "$TEMP_DIR"
fi

echo ""

# Storage check
print_step "💽 Storage Analysis"
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
print_step "📦 Dependencies Check"
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
    if [ "$AUTO_CONFIRM" = true ]; then
        print_step "Auto-confirming: Installing missing dependencies"
        install_dependencies
    else
        echo -n "Install missing dependencies? (y/N): "
        read -r RESPONSE
        if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            install_dependencies
        fi
    fi
else
    print_success "All dependencies are installed"
fi

echo ""

# Final recommendations
print_step "🎯 Final Recommendations"
echo ""
echo "Build Environment Tips:"
echo "  1. Close unnecessary applications during build"
echo "  2. Monitor build progress with 'htop' or 'top'"
echo "  3. Use 'ionice -c3' for lower I/O priority if needed"
echo "  4. Consider building during off-peak hours"
echo ""

# Export temp directory for build scripts
if [ -n "$TEMP_DIR" ]; then
    export QGETH_BUILD_TEMP="$TEMP_DIR"
    echo "export QGETH_BUILD_TEMP=\"$TEMP_DIR\"" >> ~/.bashrc
    echo "Exported QGETH_BUILD_TEMP environment variable"
    echo ""
fi

# Recalculate final totals
FINAL_SWAP_TOTAL=0
if [ -f /proc/swaps ]; then
    SWAP_KB=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
    FINAL_SWAP_TOTAL=$((SWAP_KB / 1024))
fi
FINAL_TOTAL=$((TOTAL_MB + FINAL_SWAP_TOTAL))

# Apply same tolerance logic for final check
local final_tolerance_mb=50
local final_effective_required=$((REQUIRED_MB - final_tolerance_mb))

if [ $FINAL_TOTAL -ge $final_effective_required ]; then
    print_success "✅ VPS is ready for building Q Geth!"
    if [ $FINAL_TOTAL -lt $REQUIRED_MB ]; then
        local final_deficit=$((REQUIRED_MB - FINAL_TOTAL))
        echo "  (${final_deficit}MB under target, but within ${final_tolerance_mb}MB tolerance)"
    fi
    echo ""
    echo "Next steps:"
    echo "  ./build-linux.sh            # Build both geth and miner"
    echo "  ./build-linux.sh geth       # Build geth only"
    echo "  ./build-linux.sh miner      # Build miner only"
    echo ""
    echo "Build will use:"
    echo "  - Total Memory: ${FINAL_TOTAL}MB (${TOTAL_MB}MB RAM + ${FINAL_SWAP_TOTAL}MB swap)"
    echo "  - Temp Directory: ${TEMP_DIR:-/tmp}"
else
    print_warning "⚠️  VPS may have issues building due to low memory"
    echo ""
    echo "Current total: ${FINAL_TOTAL}MB (need ${REQUIRED_MB}MB)"
    echo ""
    echo "Consider:"
    echo "  - Running this script again to add more swap space"
    echo "  - Upgrading to a VPS with more RAM"
    echo "  - Building only one component at a time"
fi

echo ""
print_step "📋 System Summary"
echo "RAM: ${TOTAL_MB}MB total"
echo "Swap: ${FINAL_SWAP_TOTAL}MB"
echo "Total Memory: ${FINAL_TOTAL}MB"
echo "Required: ${REQUIRED_MB}MB (4GB)"
echo "Storage: ${AVAILABLE_GB}GB available"
echo "Temp Dir: ${TEMP_DIR:-/tmp}"
echo "Status: $([ $FINAL_TOTAL -ge $final_effective_required ] && echo "✅ Ready for 4GB builds" || echo "⚠️  Need more memory")" 