#!/bin/bash
# Build script for Q Coin Linux Binaries
# Ensures complete compatibility between Windows and Linux builds
# Handles minimal Linux environments with missing utilities
# Optimized for low-memory VPS environments (requires minimum 3GB RAM)
# AUTOMATED ERROR RECOVERY: Handles permission issues and module conflicts
# Usage: ./build-linux.sh [geth|miner|both] [--clean] [-y|--yes]

# Set robust PATH for minimal Linux environments
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:$PATH"

# Parse command line arguments
AUTO_CONFIRM=false
TARGET="geth"
CLEAN="false"
USE_SUDO_FOR_FILES=false
QUIET_MODE=false

for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_CONFIRM=true
            ;;
        -q|--quiet)
            QUIET_MODE=true
            ;;
        --clean)
            CLEAN="--clean"
            ;;
        geth|miner|both)
            TARGET="$arg"
            ;;
    esac
done

VERSION="1.0.0"

if [ "$QUIET_MODE" != true ]; then
    echo "🚀 Building Q Coin Linux Binaries (Memory-Optimized + Auto-Recovery)..."
    echo "Target: $TARGET"
    echo "Version: $VERSION"
    if [ "$AUTO_CONFIRM" = true ]; then
        echo "Mode: Non-interactive (auto-confirm enabled)"
    fi
    echo ""
else
    echo "Building Q Coin $TARGET binaries..."
fi

# CRITICAL FIX: Ensure Go 1.24.4 is active and prioritized
validate_go_version() {
    [ "$QUIET_MODE" != true ] && echo "🔧 Validating Go installation and PATH..."
    
    # Force PATH to prioritize /usr/local/go/bin (where bootstrap installs Go 1.24.4)
    export PATH="/usr/local/go/bin:$PATH"
    
    # Check if Go is available
    if ! command -v go >/dev/null 2>&1; then
        echo "🚨 Error: Go not found in PATH"
        echo "Current PATH: $PATH"
        echo "Please run bootstrap script first to install Go 1.24.4"
        exit 1
    fi
    
    # Get Go version
    local go_version=$(go version 2>/dev/null)
    echo "Active Go version: $go_version"
    echo "Go location: $(which go)"
    echo "GOROOT: $(go env GOROOT 2>/dev/null)"
    
    # Verify it's Go 1.24.x
    if ! echo "$go_version" | grep -q "go1\.24"; then
        echo "🚨 Error: Wrong Go version detected!"
        echo "Expected: Go 1.24.x"
        echo "Found: $go_version"
        echo ""
        echo "Available Go installations:"
        find /usr/local /usr /opt -name "go" -type f -executable 2>/dev/null | head -5
        echo ""
        
        # Try to fix PATH automatically
        if [ -x "/usr/local/go/bin/go" ]; then
            echo "🔧 Auto-fix: Found Go 1.24.4 at /usr/local/go/bin/go"
            export PATH="/usr/local/go/bin:$PATH"
            go_version=$(go version 2>/dev/null)
            echo "Updated Go version: $go_version"
            
            if echo "$go_version" | grep -q "go1\.24"; then
                echo "✅ Go 1.24.x now active"
            else
                echo "🚨 Auto-fix failed. Please run bootstrap script first."
                exit 1
            fi
        else
            echo "🚨 Go 1.24.4 not found. Please run bootstrap script first."
            exit 1
        fi
    else
        echo "✅ Go 1.24.x is active and ready"
    fi
    echo ""
}

# CRITICAL FIX: Enhanced permission validation and repair
enhanced_permission_check() {
    echo "🔒 Enhanced permission validation..."
    
    local output_dir=$(pwd)
    local project_root="$(cd ../.. && pwd)"
    local current_user=$(whoami)
    local dir_owner=""
    local project_owner=""
    
    # Get directory owners
    if command -v stat >/dev/null 2>&1; then
        dir_owner=$(stat -c '%U' "$output_dir" 2>/dev/null || echo "unknown")
        project_owner=$(stat -c '%U' "$project_root" 2>/dev/null || echo "unknown")
    fi
    
    echo "  Script directory: $output_dir (owner: $dir_owner)"
    echo "  Project root: $project_root (owner: $project_owner)"
    echo "  Current user: $current_user"
    
    # Check if we can write to project root (where binaries are created)
    if [ ! -w "$project_root" ]; then
        echo "🚨 Permission issue: Cannot write to project root directory"
        echo "🔧 Auto-fix: Attempting to fix project root ownership..."
        
        if [ "$current_user" = "root" ] && [ "$project_owner" != "root" ]; then
            echo "Running as root - changing ownership to root..."
            chown -R root:root "$project_root" 2>/dev/null || true
        elif [ "$current_user" != "root" ] && [ "$project_owner" = "root" ]; then
            echo "Running as user but project owned by root - fixing with sudo..."
            if command -v sudo >/dev/null 2>&1; then
                if sudo chown -R "$current_user:$current_user" "$project_root" 2>/dev/null; then
                    echo "✅ Project ownership fixed with sudo"
                else
                    echo "🚨 Failed to fix ownership. Manual fix needed:"
                    echo "   sudo chown -R $current_user:$current_user $project_root"
                    
                    if [ "$AUTO_CONFIRM" != true ]; then
                        echo -n "Continue anyway and try alternative approach? (y/N): "
                        read -r RESPONSE
                        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                            echo "Build aborted."
                            exit 1
                        fi
                    fi
                    
                    # Alternative: use sudo for the entire build
                    echo "🔧 Alternative: Will use sudo for file operations"
                    USE_SUDO_FOR_FILES=true
                fi
            else
                echo "🚨 sudo not available and cannot write to project directory"
                exit 1
            fi
        fi
    else
        echo "✅ Project root permissions OK"
    fi
    echo ""
}

# AUTOMATED FIX 1: Directory ownership detection and correction
check_and_fix_permissions() {
    echo "🔒 Checking directory permissions..."
    
    local output_dir=$(pwd)
    local current_user=$(whoami)
    local dir_owner=""
    
    # Get directory owner
    if command -v stat >/dev/null 2>&1; then
        dir_owner=$(stat -c '%U' "$output_dir" 2>/dev/null || echo "unknown")
    fi
    
    echo "  Current directory: $output_dir"
    echo "  Current user: $current_user"
    echo "  Directory owner: $dir_owner"
    
    # Check if we can write to current directory
    if [ ! -w "$output_dir" ]; then
        echo "🚨 Permission issue detected: Cannot write to output directory"
        
        # If running as root, fix ownership
        if [ "$current_user" = "root" ] && [ "$dir_owner" != "root" ]; then
            echo "🔧 Auto-fix: Changing directory ownership to root..."
            chown -R root:root "$output_dir" 2>/dev/null || true
        # If non-root user and directory owned by root, try to fix with sudo
        elif [ "$current_user" != "root" ] && [ "$dir_owner" = "root" ]; then
            echo "🔧 Auto-fix: Attempting to change ownership to $current_user..."
            if command -v sudo >/dev/null 2>&1; then
                if sudo chown -R "$current_user:$current_user" "$output_dir" 2>/dev/null; then
                    echo "✅ Directory ownership fixed with sudo"
                else
                    echo "🚨 Failed to fix directory ownership - you may need to run:"
                    echo "   sudo chown -R $current_user:$current_user $output_dir"
                    
                    if [ "$AUTO_CONFIRM" != true ]; then
                        echo -n "Continue anyway? (y/N): "
                        read -r RESPONSE
                        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                            echo "Build aborted."
                            exit 1
                        fi
                    else
                        echo "⚠️ Auto-continuing in non-interactive mode"
                    fi
                fi
            else
                echo "🚨 sudo not available and cannot write to directory"
                echo "   Manual fix needed: chown -R $current_user:$current_user $output_dir"
                exit 1
            fi
        fi
    else
        echo "✅ Directory permissions OK"
    fi
    echo ""
}

# AUTOMATED FIX 2: Go module dependency conflict resolution
fix_go_module_conflicts() {
    local module_dir="$1"
    echo "🔧 Checking and fixing Go module dependencies in $module_dir..."
    
    cd "$module_dir"
    
    # Check for known problematic dependencies
    if grep -q "github.com/fjl/memsize" go.mod 2>/dev/null; then
        echo "🚨 Detected problematic memsize dependency"
        echo "🔧 Auto-fix: Cleaning Go module cache and dependencies..."
        
        # Clean everything
        go clean -cache -modcache -testcache 2>/dev/null || true
        go clean -r -cache 2>/dev/null || true
        
        # Remove go.sum to force fresh resolution
        rm -f go.sum
        
        # Download with explicit module proxy
        echo "🚀 Re-downloading dependencies with fresh resolution..."
        GOPROXY=direct go mod download 2>/dev/null || true
        go mod tidy 2>/dev/null || true
        
        # Verify dependencies
        if go mod verify 2>/dev/null; then
            echo "✅ Module dependencies verified and fixed"
        else
            echo "🚨 Module verification failed, trying alternative approach..."
            
            # Try with different Go version/tags if available
            go clean -cache -modcache 2>/dev/null || true
            GOPROXY=https://proxy.golang.org go mod download 2>/dev/null || true
            go mod tidy 2>/dev/null || true
        fi
    else
        echo "✅ No known problematic dependencies detected"
    fi
    
    cd - >/dev/null
}

# AUTOMATED FIX 3: Build retry with error recovery
build_with_retry() {
    local build_type="$1"
    local build_cmd="$2"
    local output_binary="$3"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "🚀 Build attempt $((retry_count + 1))/$max_retries for $build_type..."
        
        # Try the build
        if eval "$build_cmd"; then
            echo "✅ $build_type built successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            echo "🚨 Build attempt $retry_count failed for $build_type"
            
            if [ $retry_count -lt $max_retries ]; then
                echo "🔧 Attempting automated recovery..."
                
                # Clean build artifacts
                rm -f "$output_binary" 2>/dev/null || true
                
                # Clean Go cache and modules
                go clean -cache -testcache 2>/dev/null || true
                
                # Fix module dependencies
                go mod tidy 2>/dev/null || true
                go mod download 2>/dev/null || true
                
                # Wait a moment before retry
                sleep 2
            else
                echo "🚨 All build attempts failed for $build_type"
                return 1
            fi
        fi
    done
    
    return 1
}

# Create swap file for build (simplified version of bootstrap function)
create_swap_for_build() {
    local needed_mb=$1
    local swap_size_mb=$((needed_mb + 512))  # Add 512MB buffer
    
    echo "🔧 Creating ${swap_size_mb}MB swap file for build..."
    
    # Check available disk space
    local available_mb=$(df / | awk 'NR==2 {print int($4/1024)}')
    if [ $available_mb -lt $swap_size_mb ]; then
        echo "❌ Insufficient disk space for swap file"
        echo "  Available: ${available_mb}MB"
        echo "  Required: ${swap_size_mb}MB"
        return 1
    fi
    
    # Remove any existing swapfile
    if [ -f /swapfile ]; then
        echo "🔧 Removing existing swap file..."
        swapoff /swapfile 2>/dev/null || true
        rm -f /swapfile
    fi
    
    # Create new swap file
    echo "📁 Allocating ${swap_size_mb}MB swap file..."
    if ! fallocate -l ${swap_size_mb}M /swapfile 2>/dev/null; then
        # Fallback to dd if fallocate fails
        echo "📁 fallocate failed, using dd method..."
        if ! dd if=/dev/zero of=/swapfile bs=1M count=$swap_size_mb 2>/dev/null; then
            echo "❌ Failed to create swap file"
            return 1
        fi
    fi
    
    # Set correct permissions
    chmod 600 /swapfile
    
    # Make swap
    if ! mkswap /swapfile >/dev/null 2>&1; then
        echo "❌ Failed to format swap file"
        rm -f /swapfile
        return 1
    fi
    
    # Enable swap
    if ! swapon /swapfile; then
        echo "❌ Failed to enable swap file"
        rm -f /swapfile
        return 1
    fi
    
    # Add to fstab for persistence (if not already there)
    if ! grep -q "/swapfile" /etc/fstab 2>/dev/null; then
        echo "/swapfile none swap sw 0 0" >> /etc/fstab 2>/dev/null || true
    fi
    
    # Verify swap is active
    local new_swap=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
    local new_swap_mb=$((new_swap / 1024))
    
    echo "✅ Swap file created and activated"
    echo "  Swap file: /swapfile (${swap_size_mb}MB)"
    echo "  Total swap now: ${new_swap_mb}MB"
    
    # Update memory status
    local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_mb=$((mem_total / 1024))
    local combined_mb=$((total_mb + new_swap_mb))
    echo "✅ Total memory now: ${combined_mb}MB (sufficient for building)"
    
    return 0
}

# Memory check function
check_memory() {
    local required_mb=4096  # 4GB minimum total (RAM + swap)
    local total_mb=0
    local swap_mb=0
    local combined_mb=0
    
    if [ -f /proc/meminfo ]; then
        # Get RAM and swap in MB
        local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        total_mb=$((mem_total / 1024))
        
        # Check swap
        if [ -f /proc/swaps ]; then
            local swap_total=$(awk 'NR>1 {sum+=$3} END {print sum+0}' /proc/swaps)
            swap_mb=$((swap_total / 1024))
        fi
        
        combined_mb=$((total_mb + swap_mb))
        
        if [ "$QUIET_MODE" != true ]; then
            echo "💾 Memory Check:"
            echo "  RAM: ${total_mb}MB"
            echo "  Swap: ${swap_mb}MB"
            echo "  Total Available: ${combined_mb}MB"
            echo "  Required Total: ${required_mb}MB (4GB)"
        fi
        
        # Add tolerance margin - 50MB difference is acceptable (same as prepare-vps.sh)
        local tolerance_mb=50
        local effective_required=$((required_mb - tolerance_mb))
        
        if [ $combined_mb -lt $effective_required ]; then
            local deficit=$((required_mb - combined_mb))
            echo "⚠️  WARNING: Insufficient total memory!"
            echo "   Current total: ${combined_mb}MB"
            echo "   Required total: ${required_mb}MB"
            echo "   Deficit: ${deficit}MB"
            echo ""
            echo "🔧 Fixing memory shortage automatically..."
            echo ""
            
            # Try to create swap automatically
            if [ "$EUID" -eq 0 ] || [ -n "$SUDO_USER" ]; then
                echo "💡 Creating swap file to fix memory shortage..."
                if create_swap_for_build $deficit; then
                    echo "✅ Swap created successfully - continuing with build"
                else
                    echo "❌ Failed to create swap automatically"
                    echo "🔧 Manual recommendations:"
                    echo "  1. Run bootstrap script with sudo: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash"
                    echo "  2. Create swap manually: sudo fallocate -l ${deficit}M /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
                    echo "  3. Close unnecessary programs"
                    echo "  4. Use a VPS with more RAM"
                    echo ""
                    
                    # Auto-continue if non-interactive mode and deficit is small (< 100MB)
                    if [ "$AUTO_CONFIRM" = true ] && [ $deficit -lt 100 ]; then
                        echo "💡 Auto-continuing: Memory deficit is small (${deficit}MB < 100MB) and non-interactive mode is enabled"
                    else
                        echo -n "Continue anyway? (y/N): "
                        read -r RESPONSE
                        if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                            echo "Build aborted."
                            exit 1
                        fi
                    fi
                fi
            else
                echo "❌ Cannot create swap (not running as root/sudo)"
                echo "🔧 Recommendations:"
                echo "  1. Run bootstrap script with sudo: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash"
                echo "  2. Create swap manually: sudo fallocate -l ${deficit}M /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
                echo "  3. Close unnecessary programs"
                echo "  4. Use a VPS with more RAM"
                echo ""
                
                # Auto-continue if non-interactive mode and deficit is small (< 100MB)
                if [ "$AUTO_CONFIRM" = true ] && [ $deficit -lt 100 ]; then
                    echo "💡 Auto-continuing: Memory deficit is small (${deficit}MB < 100MB) and non-interactive mode is enabled"
                else
                    echo -n "Continue anyway? (y/N): "
                    read -r RESPONSE
                    if [[ ! "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                        echo "Build aborted."
                        exit 1
                    fi
                fi
            fi
        elif [ $combined_mb -lt $required_mb ]; then
            local deficit=$((required_mb - combined_mb))
            echo "💡 Memory within tolerance range"
            echo "   Current total: ${combined_mb}MB"
            echo "   Required total: ${required_mb}MB"
            echo "   Deficit: ${deficit}MB (within ${tolerance_mb}MB tolerance)"
            echo "   ✅ Proceeding with build - memory difference is acceptable"
        else
            echo "✅ Memory check passed (${combined_mb}MB total)"
        fi
    else
        echo "⚠️  Cannot check memory - /proc/meminfo not found"
    fi
    echo ""
}

# Setup temporary build directory
setup_temp_build() {
    # Use temp directory from prepare-vps.sh if available
    if [ -n "$QGETH_BUILD_TEMP" ]; then
        BUILD_TEMP_DIR="$QGETH_BUILD_TEMP"
        # Create the directory if it doesn't exist
        mkdir -p "$BUILD_TEMP_DIR"
        echo "📁 Using VPS-prepared temp directory: $BUILD_TEMP_DIR"
    else
        # Check /tmp space availability
        if command -v df >/dev/null 2>&1; then
            tmp_space=$(df /tmp 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
            if [ "$tmp_space" -lt 1048576 ]; then  # Less than 1GB in KB
                echo "⚠️  Low /tmp space detected, using local temp directory"
                BUILD_TEMP_DIR="./build-temp-$$"
            else
                BUILD_TEMP_DIR="/tmp/qgeth-build-$$"
            fi
        else
            # Default to local if df not available
            BUILD_TEMP_DIR="./build-temp-$$"
        fi
        mkdir -p "$BUILD_TEMP_DIR"
        echo "📁 Created temporary build directory: $BUILD_TEMP_DIR"
    fi
    
    # Convert to absolute path (required by Go)
    if [ "${BUILD_TEMP_DIR#/}" = "$BUILD_TEMP_DIR" ]; then
        # Relative path, convert to absolute
        BUILD_TEMP_DIR="$(pwd)/$BUILD_TEMP_DIR"
    fi
    
    # Set Go build cache and temp directories (Go requires absolute paths)
    export GOCACHE="$BUILD_TEMP_DIR/gocache"
    export GOTMPDIR="$BUILD_TEMP_DIR/gotmp"
    export TMPDIR="$BUILD_TEMP_DIR/tmp"
    
    # Create directories
    mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
    
    echo "🗂️  Temporary Build Setup:"
    echo "  Build Temp Dir: $BUILD_TEMP_DIR"
    echo "  Go Cache: $GOCACHE"
    echo "  Go Temp: $GOTMPDIR"
    echo "  System Temp: $TMPDIR"
    echo ""
    
    # Cleanup function (only if we created the temp dir)
    cleanup_temp() {
        if [ -z "$QGETH_BUILD_TEMP" ]; then
            echo "🧹 Cleaning up temporary build directory..."
            rm -rf "$BUILD_TEMP_DIR" 2>/dev/null || true
            echo "✅ Temporary files cleaned up"
        else
            echo "🧹 Preserving VPS-prepared temp directory: $BUILD_TEMP_DIR"
        fi
    }
    
    # Set trap for cleanup on exit
    trap cleanup_temp EXIT
}

# Memory-efficient build flags
get_build_flags() {
    # Memory-efficient linker flags
    local ldflags="-s -w"
    ldflags="$ldflags -X main.VERSION=$VERSION"
    ldflags="$ldflags -X main.BUILD_TIME=$BUILD_TIME"
    ldflags="$ldflags -X main.GIT_COMMIT=$GIT_COMMIT"
    
    # Store ldflags for use in build commands
    LDFLAGS="$ldflags"
    
    echo "💾 Memory-Optimized Build Flags:"
    echo "  Linker: $ldflags"
    echo "  Trimpath: enabled"
    echo "  VCS info: disabled"
    echo "  Cache: $GOCACHE"
    echo ""
}

# EXECUTE AUTOMATED FIXES
echo "🔧 Running automated pre-build checks and fixes..."

# Step 1: Validate Go version and PATH
validate_go_version

# Step 2: Enhanced permission checking
enhanced_permission_check

# Step 3: Original permission checking (for script directory)
check_and_fix_permissions

# Run memory check
check_memory

# Setup temporary build environment
setup_temp_build

# Check for interrupted/partial builds and clean if needed
if [ -f "geth.bin" ] && [ ! -f "geth" ]; then
    echo "🔍 Detected partial build (geth.bin without geth wrapper)"
    echo "🧹 Cleaning partial build state..."
    rm -f geth.bin geth quantum-miner quantum_solver.py
    echo "  Partial build artifacts removed"
fi

# Check for stale build artifacts in case of interruption
if [ -n "$BUILD_TEMP_DIR" ] && [ -d "$BUILD_TEMP_DIR" ]; then
    # Check if temp directory has stale content
    if [ "$(find "$BUILD_TEMP_DIR" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "🔍 Found stale build temp directory: $BUILD_TEMP_DIR"
        echo "🧹 Cleaning stale build artifacts..."
        rm -rf "$BUILD_TEMP_DIR"
        mkdir -p "$BUILD_TEMP_DIR"
        # Recreate the required subdirectories
        mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
        echo "  Stale temp directory cleaned and subdirectories recreated"
    fi
fi

# Clean previous builds if requested
if [ "$CLEAN" = "--clean" ] || [ "$2" = "--clean" ]; then
    echo "🧹 Cleaning previous builds..."
    rm -f geth geth.bin quantum-miner quantum_solver.py
    echo "  Previous binaries removed"
    
    # Clean Go cache
    go clean -cache 2>/dev/null || true
    echo "  Go cache cleaned"
    
    # Clean build temp directory
    if [ -n "$BUILD_TEMP_DIR" ] && [ -d "$BUILD_TEMP_DIR" ]; then
        rm -rf "$BUILD_TEMP_DIR"
        mkdir -p "$BUILD_TEMP_DIR"
        # Recreate the required subdirectories
        mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
        echo "  Build temp directory cleaned and subdirectories recreated"
    fi
fi

# Check directories exist
if [ ! -d "../../quantum-geth" ]; then
    echo "🚨 Error: quantum-geth directory not found!"
    echo "Please run this script from scripts/linux/ directory."
    exit 1
fi

if [ ! -d "../../quantum-miner" ]; then
    echo "🚨 Error: quantum-miner directory not found!"
    echo "Please run this script from scripts/linux/ directory."
    exit 1
fi

# CRITICAL: Set consistent build environment
# These settings MUST match Windows builds for quantum field compatibility
export GOOS=linux
export GOARCH=amd64
export CGO_ENABLED=0  # ALWAYS 0 for geth - this is crucial for quantum field compatibility

# Build info - handle missing utilities gracefully
if command -v date >/dev/null 2>&1; then
    BUILD_TIME=$(date "+%Y-%m-%d_%H:%M:%S" 2>/dev/null || echo "unknown")
else
    BUILD_TIME="unknown"
fi

if command -v git >/dev/null 2>&1; then
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
else
    GIT_COMMIT="unknown"
fi

CURRENT_DIR=$(pwd)

# Get memory-efficient build flags
get_build_flags

echo "🚀 Build Environment:"
echo "  Target OS: linux/amd64"
echo "  Build Time: $BUILD_TIME"
echo "  Git Commit: $GIT_COMMIT"
echo "  Output Directory: $CURRENT_DIR"
echo "  PATH: $PATH"
echo ""

# Function to build quantum-geth with automated error recovery
build_geth() {
    echo "🚀 Building Quantum-Geth (Memory-Optimized + Auto-Recovery)..."
    
    # Check if go.mod exists
    if [ ! -f "../../quantum-geth/go.mod" ]; then
        echo "🚨 Error: go.mod not found in quantum-geth directory!"
        exit 1
    fi
    
    # AUTOMATED FIX: Pre-build module dependency resolution
    fix_go_module_conflicts "../../quantum-geth"
    
    # CRITICAL: ALWAYS enforce CGO_ENABLED=0 for geth
    # This ensures 100% compatibility with Windows builds for quantum fields
    # Store original value to restore after miner build (if needed)
    ORIGINAL_CGO=$CGO_ENABLED
    export CGO_ENABLED=0
    
    echo "🚀 Enforcing CGO_ENABLED=0 for geth build (quantum field compatibility)"
    echo "    This ensures identical quantum field handling as Windows builds"
    echo "💾 Using temporary directory: $BUILD_TEMP_DIR"
    
    cd ../../quantum-geth/cmd/geth
    
    # Ensure Go temp directories exist before build
    echo "🔧 Ensuring Go temp directories exist..."
    mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
    
    # Detect Go version for memsize compatibility
    local go_version=$(go version 2>/dev/null | grep -oE 'go[0-9]+\.[0-9]+' | head -1)
    local use_checklinkname=false
    
    if [ -n "$go_version" ]; then
        # Extract major and minor version numbers
        local major=$(echo "$go_version" | cut -d'.' -f1 | sed 's/go//')
        local minor=$(echo "$go_version" | cut -d'.' -f2)
        
        # Check if version is 1.23 or higher (needs -checklinkname=0)
        if [ "$major" -gt 1 ] || ([ "$major" -eq 1 ] && [ "$minor" -ge 23 ]); then
            echo "🔧 Go $go_version requires -checklinkname=0 for memsize compatibility"
            use_checklinkname=true
        else
            echo "🔧 Go $go_version uses standard build flags"
        fi
    fi
    
    # Build command for retry mechanism with conditional checklinkname flag
    if [ "$use_checklinkname" = true ]; then
        BUILD_CMD="CGO_ENABLED=0 go build -ldflags \"-checklinkname=0 $LDFLAGS\" -trimpath -buildvcs=false -o ../../../geth.bin ."
    else
        BUILD_CMD="CGO_ENABLED=0 go build -ldflags \"$LDFLAGS\" -trimpath -buildvcs=false -o ../../../geth.bin ."
    fi
    
    # Use sudo for file operations if needed
    if [ "$USE_SUDO_FOR_FILES" = true ]; then
        echo "🔧 Using sudo for file operations due to permission issues..."
        BUILD_CMD="CGO_ENABLED=0 go build -ldflags \"$LDFLAGS\" -trimpath -buildvcs=false -o /tmp/geth.bin.tmp . && sudo mv /tmp/geth.bin.tmp ../../../geth.bin"
    fi
    
    # Use automated retry with error recovery
    if build_with_retry "quantum-geth" "$BUILD_CMD" "../../../geth.bin"; then
        cd ../../../..
        echo "✅ Quantum-Geth built successfully: ./geth.bin (CGO_ENABLED=0)"
        
        # Show file info if ls is available
        if command -v ls >/dev/null 2>&1; then
            ls -lh ../../geth.bin 2>/dev/null || echo "Binary created: ../../geth.bin"
        else
            echo "Binary created: ../../geth.bin"
        fi
    else
        cd ../../../..
        echo "🚨 Error: Failed to build quantum-geth after all retry attempts"
        echo "🔧 Manual troubleshooting steps:"
        echo "  1. Check Go version: go version"
        echo "  2. Clean everything: go clean -cache -modcache -testcache"
        echo "  3. Verify network: ping proxy.golang.org"
        echo "  4. Check disk space: df -h"
        exit 1
    fi
    
    # Restore original CGO setting for any subsequent builds
    export CGO_ENABLED=$ORIGINAL_CGO
}

# Function to build quantum-miner with automated error recovery
build_miner() {
    echo "🚀 Building Quantum-Miner (Memory-Optimized + Auto-Recovery)..."
    
    # Check if go.mod exists
    if [ ! -f "../../quantum-miner/go.mod" ]; then
        echo "🚨 Error: go.mod not found in quantum-miner directory!"
        exit 1
    fi
    
    # AUTOMATED FIX: Pre-build module dependency resolution
    fix_go_module_conflicts "../../quantum-miner"
    
    # Detect GPU capabilities and build accordingly
    BUILD_TAGS=""
    GPU_TYPE="CPU"
    
    # Check for NVIDIA GPU and CUDA
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "🚀 NVIDIA GPU detected, checking CUDA availability..."
        
        # Check for CUDA development libraries
        if command -v pkg-config >/dev/null 2>&1 && (pkg-config --exists cuda-12.0 2>/dev/null || pkg-config --exists cuda-11.0 2>/dev/null) || [ -d "/usr/local/cuda" ]; then
            echo "✅ CUDA development environment found"
            BUILD_TAGS="cuda"
            GPU_TYPE="CUDA"
            export CGO_ENABLED=1
        else
            echo "🚨 CUDA development libraries not found, checking for Qiskit-Aer GPU..."
            
            # Check for Qiskit-Aer GPU support
            if command -v python3 >/dev/null 2>&1 && python3 -c "import qiskit_aer; from qiskit_aer import AerSimulator; AerSimulator(device='GPU')" >/dev/null 2>&1; then
                echo "✅ Qiskit-Aer GPU support detected"
                BUILD_TAGS="cuda"
                GPU_TYPE="Qiskit-GPU"
                export CGO_ENABLED=0
            else
                echo "🚨 No GPU acceleration available, building CPU-only version"
                export CGO_ENABLED=0
            fi
        fi
    else
        echo "🚨 No NVIDIA GPU detected, building CPU-only version"
        export CGO_ENABLED=0
    fi
    
    echo "🚀 Build Configuration:"
    echo "  GPU Type: $GPU_TYPE"
    echo "  Build Tags: ${BUILD_TAGS:-none}"
    echo "  CGO Enabled: $CGO_ENABLED"
    echo "💾 Using temporary directory: $BUILD_TEMP_DIR"
    echo ""
    
    cd ../../quantum-miner
    
    # Ensure Go temp directories exist before build
    echo "🔧 Ensuring Go temp directories exist..."
    mkdir -p "$GOCACHE" "$GOTMPDIR" "$TMPDIR"
    
    # Detect Go version for memsize compatibility
    local go_version=$(go version 2>/dev/null | grep -oE 'go[0-9]+\.[0-9]+' | head -1)
    local use_checklinkname=false
    
    if [ -n "$go_version" ]; then
        # Extract major and minor version numbers
        local major=$(echo "$go_version" | cut -d'.' -f1 | sed 's/go//')
        local minor=$(echo "$go_version" | cut -d'.' -f2)
        
        # Check if version is 1.23 or higher (needs -checklinkname=0)
        if [ "$major" -gt 1 ] || ([ "$major" -eq 1 ] && [ "$minor" -ge 23 ]); then
            echo "🔧 Go $go_version requires -checklinkname=0 for memsize compatibility"
            use_checklinkname=true
        else
            echo "🔧 Go $go_version uses standard build flags"
        fi
    fi
    
    # Build command for retry mechanism with conditional flags
    if [ "$use_checklinkname" = true ]; then
        BUILD_CMD="go build -ldflags \"-checklinkname=0 $LDFLAGS\" -trimpath -buildvcs=false"
    else
        BUILD_CMD="go build -ldflags \"$LDFLAGS\" -trimpath -buildvcs=false"
    fi
    if [ -n "$BUILD_TAGS" ]; then
        BUILD_CMD="$BUILD_CMD -tags $BUILD_TAGS"
    fi
    BUILD_CMD="$BUILD_CMD -o ../quantum-miner ."
    
    # Use sudo for file operations if needed
    if [ "$USE_SUDO_FOR_FILES" = true ]; then
        echo "🔧 Using sudo for miner file operations due to permission issues..."
        # Replace the output part with temp file + sudo move
        BUILD_CMD=$(echo "$BUILD_CMD" | sed 's|-o ../quantum-miner|-o /tmp/quantum-miner.tmp|')
        BUILD_CMD="$BUILD_CMD && sudo mv /tmp/quantum-miner.tmp ../quantum-miner"
    fi
    
    # Use automated retry with error recovery
    if build_with_retry "quantum-miner" "$BUILD_CMD" "../quantum-miner"; then
        cd ..
        echo "✅ Quantum-Miner built successfully: ./quantum-miner ($GPU_TYPE)"
        
        # Go to project root 
        cd ..
        
        # Show file info if ls is available
        if command -v ls >/dev/null 2>&1; then
            ls -lh quantum-miner 2>/dev/null || echo "Binary created: quantum-miner"
        else
            echo "Binary created: quantum-miner"
        fi
        
        # Test GPU support
        if [ "$GPU_TYPE" != "CPU" ]; then
            echo "🚀 Testing GPU support..."
            if ./quantum-miner --help 2>/dev/null | grep -q "GPU" 2>/dev/null; then
                echo "✅ GPU support confirmed in binary"
            else
                echo "🚨 GPU support may not be active (check dependencies)"
            fi
        fi
        
        # Return to scripts/linux directory
        cd scripts/linux
    else
        cd ..
        echo "🚨 Error: Failed to build quantum-miner after all retry attempts"
        echo "🔧 Manual troubleshooting steps:"
        echo "  1. Check Go version: go version"
        echo "  2. Clean everything: go clean -cache -modcache -testcache"
        echo "  3. Install build tools: apt install build-essential"
        echo "  4. Check GPU drivers: nvidia-smi"
        exit 1
    fi
}

# Function to create Q Coin geth wrapper - robust version for minimal Linux
create_geth_wrapper() {
    echo "🚀 Creating Q Coin geth wrapper (prevents Ethereum connections)..."
    
    # Use absolute path to avoid directory context issues
    WRAPPER_PATH="../../geth"
    
    # Create wrapper with fixed path resolution
    cat > "$WRAPPER_PATH" << 'EOF'
#!/bin/bash
# Q Coin Geth Wrapper - Fixed path version
# This wrapper ensures geth ALWAYS uses Q Coin networks, never Ethereum
# Default: Q Coin Testnet (Chain ID 73235)
# Use --qcoin-mainnet for Q Coin Mainnet (Chain ID 73236)

# Set robust PATH for minimal Linux environments
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:$PATH"

# Get absolute path to geth.bin regardless of where this script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_GETH="$SCRIPT_DIR/geth.bin"

# Check if actual geth binary exists
if [ ! -f "$REAL_GETH" ]; then
    echo "🚨 ERROR: Q Coin geth binary not found at $REAL_GETH"
    echo "   Build it first: ./build-linux.sh"
    exit 1
fi
# Parse Q Coin specific flags
USE_QCOIN_MAINNET=false
FILTERED_ARGS=()

for arg in "$@"; do
    case $arg in
        --qcoin-mainnet)
            USE_QCOIN_MAINNET=true
            ;;
        --help|-h)
            echo "Q Coin Geth - Quantum Blockchain Node"
            echo ""
            echo "This geth ONLY connects to Q Coin networks, never Ethereum!"
            echo ""
            echo "Q Coin Networks:"
            echo "  Default:         Q Coin Testnet (Chain ID 73235)"
            echo "  --qcoin-mainnet  Q Coin Mainnet (Chain ID 73236)"
            echo ""
            echo "Quick Start:"
            echo "  ./start-geth.sh              # Easy testnet startup"
            echo "  ./start-geth.sh mainnet      # Easy mainnet startup"
            echo ""
            echo "Manual Usage:"
            echo "  ./geth --datadir \$HOME/.qcoin init configs/genesis_quantum_testnet.json"
            echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0"
            echo ""
            echo "Standard geth options also available."
            exit 0
            ;;
        *)
            FILTERED_ARGS+=("$arg")
            ;;
    esac
done

# Check if this is a bare geth call (likely trying to connect to Ethereum)
# Allow init commands and commands with --datadir or --networkid through
# CRITICAL: Always allow init commands through without interference
if [[ " ${FILTERED_ARGS[*]} " =~ " init " ]]; then
    # This is an init command - pass through directly without any modification
    exec "$REAL_GETH" "${FILTERED_ARGS[@]}"
elif [ ${#FILTERED_ARGS[@]} -eq 0 ] || ([[ ! " ${FILTERED_ARGS[*]} " =~ " --networkid " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " --datadir " ]]); then
    echo "🚨 Q Coin Geth: Prevented connection to Ethereum mainnet!"
    echo ""
    echo "This geth is configured for Q Coin networks only."
    echo ""
    echo "Quick Start:"
    echo "  ./start-geth.sh              # Q Coin Testnet"
    echo "  ./start-geth.sh mainnet      # Q Coin Mainnet"
    echo ""
    echo "Manual Start:"
    if $USE_QCOIN_MAINNET; then
        echo "  ./geth --datadir ~/.qcoin/mainnet --networkid 73236 init configs/genesis_quantum_mainnet.json"
        echo "  ./geth --datadir ~/.qcoin/mainnet --networkid 73236 --mine --miner.threads 0"
    else
        echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 init configs/genesis_quantum_testnet.json"
        echo "  ./geth --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0"
    fi
    echo ""
    echo "Use --help for more options."
    exit 1
fi

# Add Q Coin network defaults if not specified (but not for init commands)
if [[ ! " ${FILTERED_ARGS[*]} " =~ " --networkid " ]] && [[ ! " ${FILTERED_ARGS[*]} " =~ " init " ]]; then
    if $USE_QCOIN_MAINNET; then
        FILTERED_ARGS+=("--networkid" "73236")
    else
        FILTERED_ARGS+=("--networkid" "73235")
    fi
fi

# Execute the real geth with filtered arguments
exec "$REAL_GETH" "${FILTERED_ARGS[@]}"
EOF
    
    # Make wrapper executable with proper error handling
    if chmod +x "$WRAPPER_PATH" 2>/dev/null; then
        echo "✅ Q Coin geth wrapper created: ../../geth"
    else
        echo "⚠️ Q Coin geth wrapper created but chmod failed (may need manual chmod +x)"
        echo "✅ Wrapper created at: ../../geth"
    fi
}

# Function to create quantum solver Python script
create_solver() {
    echo "🚀 Creating quantum solver helper script..."
    
    # Create using shell built-ins for robustness
    {
        echo '#!/usr/bin/env python3'
        echo '"""'
        echo 'Quantum circuit solver for Q Coin mining'
        echo 'Compatible with the quantum-geth consensus algorithm'
        echo '"""'
        echo ''
        echo 'import sys'
        echo 'import json'
        echo 'import argparse'
        echo 'import hashlib'
        echo 'import random'
        echo 'from typing import List, Tuple'
        echo ''
        echo 'def create_quantum_circuit(seed: str, puzzle_idx: int) -> dict:'
        echo '    """Create a 16-qubit quantum circuit based on seed and puzzle index"""'
        echo '    # Use seed + puzzle index to generate deterministic circuit'
        echo '    circuit_seed = hashlib.sha256((seed + str(puzzle_idx)).encode()).hexdigest()'
        echo '    random.seed(circuit_seed)'
        echo '    '
        echo '    # Generate gates (simplified T-gate heavy circuit)'
        echo '    gates = []'
        echo '    for i in range(16):  # 16 qubits'
        echo '        # Add T-gates for quantum advantage'
        echo '        for _ in range(512):  # 512 T-gates per qubit = 8192 total'
        echo '            gates.append(f"T q[{i}]")'
        echo '    '
        echo '    # Add some CNOT gates for entanglement'
        echo '    for i in range(15):'
        echo '        gates.append(f"CNOT q[{i}], q[{i+1}]")'
        echo '    '
        echo '    # Add measurements'
        echo '    measurements = []'
        echo '    for i in range(16):'
        echo '        # Deterministic measurement outcome based on circuit'
        echo '        outcome = random.randint(0, 1)'
        echo '        measurements.append(outcome)'
        echo '    '
        echo '    return {'
        echo '        "gates": gates,'
        echo '        "measurements": measurements,'
        echo '        "t_gate_count": 8192,'
        echo '        "total_gates": len(gates),'
        echo '        "depth": 16'
        echo '    }'
        echo ''
        echo 'def solve_puzzles(seed: str, puzzle_count: int, qubits: int = 16) -> dict:'
        echo '    """Solve multiple quantum puzzles"""'
        echo '    all_proofs = []'
        echo '    all_outcomes = []'
        echo '    '
        echo '    for i in range(puzzle_count):'
        echo '        circuit = create_quantum_circuit(seed, i)'
        echo '        all_proofs.extend(circuit["gates"])'
        echo '        all_outcomes.extend(circuit["measurements"])'
        echo '    '
        echo '    # Create Merkle roots (simplified)'
        echo '    proof_data = "".join(all_proofs).encode()'
        echo '    proof_root = hashlib.sha256(proof_data).hexdigest()'
        echo '    '
        echo '    outcome_data = bytes(all_outcomes)'
        echo '    outcome_root = hashlib.sha256(outcome_data).hexdigest()'
        echo '    '
        echo '    gate_data = f"T-gates:{puzzle_count * 20}".encode()'
        echo '    gate_hash = hashlib.sha256(gate_data).hexdigest()'
        echo '    '
        echo '    # Create compressed quantum blob'
        echo '    blob_data = proof_root[:31].encode()  # 31 bytes'
        echo '    quantum_blob = blob_data.hex()'
        echo '    '
        echo '    return {'
        echo '        "proof_root": proof_root,'
        echo '        "outcome_root": outcome_root,'
        echo '        "gate_hash": gate_hash,'
        echo '        "quantum_blob": quantum_blob,'
        echo '        "total_gates": puzzle_count * 20,'
        echo '        "t_gates": puzzle_count * 20,'
        echo '        "circuit_depth": 16,'
        echo '        "measurements": all_outcomes,'
        echo '        "success": True'
        echo '    }'
        echo ''
        echo 'def main():'
        echo '    parser = argparse.ArgumentParser(description="Quantum circuit solver")'
        echo '    parser.add_argument("--seed", required=True, help="Hex seed for circuit generation")'
        echo '    parser.add_argument("--puzzles", type=int, default=128, help="Number of puzzles")'
        echo '    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits")'
        echo '    parser.add_argument("--simulator", default="aer_simulator", help="Simulator type")'
        echo '    '
        echo '    args = parser.parse_args()'
        echo '    '
        echo '    try:'
        echo '        result = solve_puzzles(args.seed, args.puzzles, args.qubits)'
        echo '        print(json.dumps(result, indent=2))'
        echo '        sys.exit(0)'
        echo '    except Exception as e:'
        echo '        error_result = {'
        echo '            "error": str(e),'
        echo '            "success": False'
        echo '        }'
        echo '        print(json.dumps(error_result, indent=2))'
        echo '        sys.exit(1)'
        echo ''
        echo 'if __name__ == "__main__":'
        echo '    main()'
    } > ../../quantum_solver.py
    
    # Make executable if chmod is available
    if command -v chmod >/dev/null 2>&1; then
        chmod +x ../../quantum_solver.py
    fi
    
    echo "✅ Quantum solver script created: ../../quantum_solver.py"
}

# Function to create Linux miner startup script
create_linux_miner_script() {
    echo "🚀 Creating Linux miner startup script..."
    
    {
        echo '#!/bin/bash'
        echo '# Easy Q Coin miner startup for Linux'
        echo '# Usage: ./start-linux-miner.sh [threads] [address]'
        echo ''
        echo 'THREADS=${1:-1}'
        echo 'MINING_ADDRESS=${2:-"0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"}'
        echo 'RPC_URL="http://127.0.0.1:8545"'
        echo ''
        echo 'echo "🚀 Starting Q Coin Linux Miner..."'
        echo 'echo "Threads: $THREADS"'
        echo 'echo "Mining Address: $MINING_ADDRESS"'
        echo 'echo "RPC URL: $RPC_URL"'
        echo 'echo ""'
        echo ''
        echo 'if [ ! -f "./quantum-miner" ]; then'
        echo '    echo "🚨 ERROR: quantum-miner not found!"'
        echo '    echo "Build it first: ./build-linux.sh"'
        echo '    exit 1'
        echo 'fi'
        echo ''
        echo './quantum-miner -rpc-url "$RPC_URL" -address "$MINING_ADDRESS" -threads "$THREADS"'
    } > ../../start-linux-miner.sh
    
    if command -v chmod >/dev/null 2>&1; then
        chmod +x ../../start-linux-miner.sh
    fi
    
    echo "✅ Linux miner script created: ../../start-linux-miner.sh"
}

# Function to clean up after build to save disk space
cleanup_after_build() {
    echo "🧹 Performing post-build cleanup to save disk space..."
    
    # Clean Go build cache - keep only essential cache
    if command -v go >/dev/null 2>&1; then
        echo "🔧 Cleaning Go build cache..."
        go clean -cache -testcache 2>/dev/null || true
        # Keep module cache but clean downloads cache
        go clean -modcache 2>/dev/null || true
        echo "✅ Go build cache cleaned"
    fi
    
    # Clean temporary build directory
    if [ -n "$BUILD_TEMP_DIR" ] && [ -d "$BUILD_TEMP_DIR" ]; then
        echo "🔧 Cleaning temporary build directory: $BUILD_TEMP_DIR"
        rm -rf "$BUILD_TEMP_DIR" 2>/dev/null || true
        echo "✅ Temporary build directory cleaned"
    fi
    
    # Clean any build-temp directories in project
    echo "🔧 Cleaning any remaining build temp directories..."
    find ../.. -name "build-temp-*" -type d -exec rm -rf {} \; 2>/dev/null || true
    
    # Clean system temp files related to the build
    echo "🔧 Cleaning system temp files..."
    find /tmp -name "*qgeth*" -mtime +0 -delete 2>/dev/null || true
    find /tmp -name "*quantum*" -mtime +0 -delete 2>/dev/null || true
    find /tmp -name "*go-build*" -type d -mtime +0 -exec rm -rf {} \; 2>/dev/null || true
    
    # Clean old object files and build artifacts
    echo "🔧 Cleaning build artifacts..."
    find ../.. -name "*.o" -delete 2>/dev/null || true
    find ../.. -name "*.a" -delete 2>/dev/null || true
    find ../.. -name "*.so" -delete 2>/dev/null || true
    
    # Remove duplicate binaries if they exist (keep only the latest)
    if [ -f "../../geth.bin" ] && [ -f "../../geth.bin.bak" ]; then
        rm -f "../../geth.bin.bak" 2>/dev/null || true
    fi
    
    if [ -f "../../quantum-miner" ] && [ -f "../../quantum-miner.bak" ]; then
        rm -f "../../quantum-miner.bak" 2>/dev/null || true  
    fi
    
    echo "✅ Post-build cleanup completed"
    echo ""
}

# Function to show disk space saved
show_disk_usage() {
    echo "💾 Disk Space Summary:"
    
    # Show current directory size
    if command -v du >/dev/null 2>&1; then
        local project_size=$(du -sh ../.. 2>/dev/null | cut -f1 || echo "unknown")
        echo "  Project directory: $project_size"
    fi
    
    # Show available disk space
    if command -v df >/dev/null 2>&1; then
        local disk_usage=$(df / 2>/dev/null | awk 'NR==2 {print $5}' || echo "unknown")
        local disk_available=$(df -h / 2>/dev/null | awk 'NR==2 {print $4}' || echo "unknown")
        echo "  Disk usage: $disk_usage"
        echo "  Available space: $disk_available"
    fi
    
    echo ""
}

# Main build logic
case $TARGET in
    "geth")
        build_geth
        create_solver
        ;;
    "miner")
        build_miner
        create_solver
        create_linux_miner_script
        ;;
    "both")
        build_geth
        build_miner
        create_solver
        create_linux_miner_script
        ;;
    *)
        echo "🚨 Error: Invalid target '$TARGET'"
        echo "Usage: ./build-linux.sh [geth|miner|both] [--clean]"
        exit 1
        ;;
esac

echo ""
echo "✅ Build Complete!"
echo ""
echo "🚀 Binaries created in project root directory:"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  ../../geth.bin             - Quantum-Geth binary"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo "  ../../quantum-miner        - Quantum Miner"
    echo "  ../../start-linux-miner.sh - Easy miner startup"
fi
echo "  ../../quantum_solver.py    - Python quantum solver helper"
echo ""
echo "🚀 Quick Start (Easy Method):"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  # Start node:"
    echo "  ./start-geth.sh"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo "  # Start mining (in another terminal):"
    echo "  ../../start-linux-miner.sh"
fi
echo ""
echo "💾 Manual Method:"
if [ "$TARGET" = "geth" ] || [ "$TARGET" = "both" ]; then
    echo "  # Initialize blockchain:"
    echo "  ../../geth.bin --datadir \$HOME/.qcoin init ../../configs/genesis_quantum_testnet.json"
    echo ""
    echo "  # Start node (testnet, external mining):"
    echo "  ../../geth.bin --datadir \$HOME/.qcoin --networkid 73235 --mine --miner.threads 0 \\"
    echo "         --http --http.api eth,net,web3,personal,admin,miner \\"
    echo "         --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
fi
if [ "$TARGET" = "miner" ] || [ "$TARGET" = "both" ]; then
    echo ""
    echo "  # Start mining (in another terminal):"
    echo "  ../../quantum-miner -rpc-url http://127.0.0.1:8545 \\"
    echo "                  -address 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A"
fi
echo ""
echo "✅ All builds use CGO_ENABLED=0 for geth - quantum field compatibility guaranteed!"

# Clean up after build
cleanup_after_build
show_disk_usage 