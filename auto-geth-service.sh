#!/bin/bash
# Auto-Geth Service Setup Script
# Complete VPS setup with auto-updating Q Geth service
# Usage: sudo ./auto-geth-service.sh [-y|--yes]

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

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script with sudo"
    echo "Usage: sudo ./auto-geth-service.sh"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

print_step "🚀 Setting up Auto-Updating Q Geth Service"
echo "User: $ACTUAL_USER"
echo "Home: $ACTUAL_HOME"
echo ""

# Make all shell scripts in current directory executable
print_step "🔧 Making all scripts executable"
chmod +x *.sh 2>/dev/null || true
print_success "✅ All shell scripts are now executable"
echo ""

# Configuration
GITHUB_REPO="fourtytwo42/Qgeth3"
GITHUB_BRANCH="main"
INSTALL_DIR="/opt/qgeth"
PROJECT_DIR="$INSTALL_DIR/Qgeth3"
LOGS_DIR="$INSTALL_DIR/logs"
BACKUP_DIR="$INSTALL_DIR/backup"
SCRIPTS_DIR="$INSTALL_DIR/scripts"

# Geth configuration
GETH_NETWORK="testnet"
GETH_ARGS="--http.corsdomain \"*\" --http.api \"eth,net,web3,personal,txpool\""
CHECK_INTERVAL=300  # Check GitHub every 5 minutes
CRASH_RETRY_DELAY=300  # Wait 5 minutes after crash before retry
MAX_RETRIES=999999  # Essentially infinite retries

print_step "📋 Configuration:"
echo "  GitHub: $GITHUB_REPO"
echo "  Branch: $GITHUB_BRANCH"
echo "  Install Directory: $INSTALL_DIR"
echo "  Network: $GETH_NETWORK"
echo "  Extra Args: $GETH_ARGS"
echo ""

# Step 1: Prepare VPS
print_step "💾 Step 1: Preparing VPS Environment"
if [ -f "./prepare-vps.sh" ]; then
    print_step "Running VPS preparation script..."
    if [ "$AUTO_CONFIRM" = true ]; then
        ./prepare-vps.sh -y
    else
        ./prepare-vps.sh
    fi
else
    print_warning "prepare-vps.sh not found in current directory"
    print_step "Creating basic VPS preparation..."
    
    # Basic memory check and swap creation
    if [ -f /proc/meminfo ]; then
        MEM_AVAILABLE=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        AVAILABLE_MB=$((MEM_AVAILABLE / 1024))
        
        if [ $AVAILABLE_MB -lt 3072 ]; then
            print_warning "Low memory detected (${AVAILABLE_MB}MB), creating swap..."
            if [ ! -f /swapfile ]; then
                fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1024 count=2097152
                chmod 600 /swapfile
                mkswap /swapfile
                swapon /swapfile
                if ! grep -q "/swapfile" /etc/fstab; then
                    echo "/swapfile none swap sw 0 0" >> /etc/fstab
                fi
                print_success "2GB swap created"
            fi
        fi
    fi
    
    # Install dependencies
    print_step "Installing dependencies..."
    DEBIAN_FRONTEND=noninteractive apt update -qq
    DEBIAN_FRONTEND=noninteractive apt install -y git curl golang-go build-essential systemd jq python3 python3-pip
    print_success "Dependencies installed"
fi

# Step 2: Configure firewall
print_step "🔥 Step 2: Configuring UFW firewall"

# Install UFW if not present
if ! command -v ufw >/dev/null 2>&1; then
    print_step "Installing UFW firewall..."
    DEBIAN_FRONTEND=noninteractive apt install -y ufw
fi

print_step "Configuring firewall rules..."

# Reset UFW to default state
ufw --force reset

# Set default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (port 22) - CRITICAL: This must be first!
ufw allow 22/tcp comment 'SSH'
print_success "✅ SSH access allowed (port 22)"

# Allow Q Geth RPC API (port 8545)
ufw allow 8545/tcp comment 'Q Geth RPC API'
print_success "✅ Q Geth RPC API allowed (port 8545)"

# Allow Q Geth P2P networking (port 30303)
ufw allow 30303/tcp comment 'Q Geth P2P TCP'
ufw allow 30303/udp comment 'Q Geth P2P UDP'
print_success "✅ Q Geth P2P networking allowed (port 30303)"

# Allow Q Geth WebSocket API (port 8546) - optional but useful
ufw allow 8546/tcp comment 'Q Geth WebSocket API'
print_success "✅ Q Geth WebSocket API allowed (port 8546)"

# Enable UFW
print_step "Enabling UFW firewall..."
ufw --force enable

# Show firewall status
print_step "Firewall configuration:"
ufw status numbered

print_success "🔥 Firewall configured and enabled"
echo ""

# Step 3: Create directory structure
print_step "📁 Step 3: Creating directory structure"
mkdir -p "$INSTALL_DIR" "$LOGS_DIR" "$BACKUP_DIR" "$SCRIPTS_DIR"
chown -R $ACTUAL_USER:$ACTUAL_USER "$INSTALL_DIR"

# Step 4: Clone/setup project
print_step "📦 Step 4: Setting up project"
if [ ! -d "$PROJECT_DIR" ]; then
    print_step "Cloning repository..."
    cd "$INSTALL_DIR"
    sudo -u $ACTUAL_USER git clone "https://github.com/$GITHUB_REPO.git"
    if [ $? -ne 0 ]; then
        print_error "Failed to clone repository"
        exit 1
    fi
else
    print_step "Project directory exists, updating..."
    cd "$PROJECT_DIR"
    sudo -u $ACTUAL_USER git fetch origin
    sudo -u $ACTUAL_USER git reset --hard "origin/$GITHUB_BRANCH"
fi

cd "$PROJECT_DIR"
chmod +x *.sh

# Step 5: Initial build
print_step "🔨 Step 5: Initial build"
sudo -u $ACTUAL_USER ./build-linux.sh geth
if [ $? -ne 0 ]; then
    print_error "Initial build failed"
    exit 1
fi

# Step 6: Create GitHub monitor script
print_step "🔍 Step 6: Creating GitHub monitor script"
cat > "$SCRIPTS_DIR/github-monitor.sh" << 'EOF'
#!/bin/bash
# GitHub Monitor for Auto-Updating Q Geth Service

# Configuration from environment or defaults
GITHUB_REPO="${GITHUB_REPO:-fourtytwo42/Qgeth3}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
GITHUB_API_URL="https://api.github.com/repos/${GITHUB_REPO}/commits/${GITHUB_BRANCH}"
PROJECT_DIR="${PROJECT_DIR:-/opt/qgeth/Qgeth3}"
LOGS_DIR="${LOGS_DIR:-/opt/qgeth/logs}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"

GITHUB_LOG="$LOGS_DIR/github-monitor.log"
LAST_COMMIT_FILE="$LOGS_DIR/last_commit.txt"
LOCK_FILE="/tmp/github-monitor.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$GITHUB_LOG"
}

cleanup() {
    rm -f "$LOCK_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Check if already running
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        log "GitHub monitor already running (PID: $PID)"
        exit 1
    else
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"

log "🔍 GitHub monitor started - watching $GITHUB_REPO:$GITHUB_BRANCH"

while true; do
    # Fetch latest commit hash
    LATEST_COMMIT=$(curl -s --connect-timeout 10 --max-time 30 "$GITHUB_API_URL" | jq -r '.sha' 2>/dev/null)
    
    if [ "$LATEST_COMMIT" = "null" ] || [ -z "$LATEST_COMMIT" ]; then
        log "⚠️  Failed to fetch latest commit from GitHub"
        sleep "$CHECK_INTERVAL"
        continue
    fi
    
    # Check if this is the first run
    if [ ! -f "$LAST_COMMIT_FILE" ]; then
        echo "$LATEST_COMMIT" > "$LAST_COMMIT_FILE"
        log "📝 Initial commit recorded: $LATEST_COMMIT"
        sleep "$CHECK_INTERVAL"
        continue
    fi
    
    # Read last known commit
    LAST_COMMIT=$(cat "$LAST_COMMIT_FILE" 2>/dev/null)
    
    if [ "$LATEST_COMMIT" != "$LAST_COMMIT" ]; then
        log "🚨 NEW COMMIT DETECTED!"
        log "   Previous: $LAST_COMMIT"
        log "   Latest:   $LATEST_COMMIT"
        
        # Update the commit file
        echo "$LATEST_COMMIT" > "$LAST_COMMIT_FILE"
        
        # Trigger update
        log "🔄 Triggering geth update..."
        systemctl restart qgeth-updater.service
        
        # Wait longer after triggering update
        sleep 60
    else
        log "✅ No changes detected ($LATEST_COMMIT)"
    fi
    
    sleep "$CHECK_INTERVAL"
done
EOF

# Step 7: Create update script
print_step "🔨 Step 7: Creating update script"
cat > "$SCRIPTS_DIR/update-geth.sh" << 'EOF'
#!/bin/bash
# Q Geth Update Script - Handles pulling, building, and restarting

# Configuration from environment or defaults
PROJECT_DIR="${PROJECT_DIR:-/opt/qgeth/Qgeth3}"
BACKUP_DIR="${BACKUP_DIR:-/opt/qgeth/backup}"
LOGS_DIR="${LOGS_DIR:-/opt/qgeth/logs}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
ACTUAL_USER="${ACTUAL_USER:-$(logname 2>/dev/null || echo 'root')}"

UPDATE_LOG="$LOGS_DIR/update.log"
LOCK_FILE="/tmp/update-geth.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$UPDATE_LOG"
}

cleanup() {
    rm -f "$LOCK_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Check if update already running
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        log "Update already in progress (PID: $PID)"
        exit 1
    else
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"

log "🔄 Starting geth update process..."

# Stop the geth service
log "🛑 Stopping geth service..."
systemctl stop qgeth-node.service || true
sleep 10

# Backup current version
log "💾 Creating backup..."
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
if [ -d "$PROJECT_DIR" ]; then
    cp -r "$PROJECT_DIR" "$BACKUP_DIR/Qgeth3_$TIMESTAMP" 2>/dev/null || true
    log "✅ Backup created: Qgeth3_$TIMESTAMP"
fi

# Pull latest changes
log "📥 Pulling latest changes from GitHub..."
cd "$PROJECT_DIR" || {
    log "❌ Cannot access project directory: $PROJECT_DIR"
    cleanup
}

if sudo -u "$ACTUAL_USER" git fetch origin && sudo -u "$ACTUAL_USER" git reset --hard "origin/$GITHUB_BRANCH"; then
    log "✅ Successfully pulled latest changes"
else
    log "❌ Failed to pull updates"
    cleanup
fi

# Build the new version with memory optimization
log "🔨 Building new version..."
if sudo -u "$ACTUAL_USER" ./build-linux.sh geth; then
    log "✅ Build completed successfully"
else
    log "❌ Build failed! Restoring backup..."
    
    # Restore from backup
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/Qgeth3_* 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        rm -rf "$PROJECT_DIR"
        cp -r "$LATEST_BACKUP" "$PROJECT_DIR"
        log "✅ Restored from backup: $(basename "$LATEST_BACKUP")"
    else
        log "❌ No backup available!"
    fi
    
    cleanup
fi

# Test the build
if [ ! -f "$PROJECT_DIR/geth" ] || [ ! -x "$PROJECT_DIR/geth" ]; then
    log "❌ Build verification failed - geth binary not found or not executable"
    cleanup
fi

log "✅ Build verification passed"

# Start the service
log "🚀 Starting geth service..."
systemctl start qgeth-node.service

# Wait and verify it started
sleep 15
if systemctl is-active --quiet qgeth-node.service; then
    log "✅ Geth service started successfully!"
    
    # Get version info if possible
    if [ -x "$PROJECT_DIR/geth" ]; then
        VERSION_INFO=$("$PROJECT_DIR/geth" version 2>/dev/null | head -1 || echo "Version unknown")
        log "📋 Running version: $VERSION_INFO"
    fi
else
    log "❌ Failed to start geth service after update"
fi

# Cleanup old backups (keep last 5)
log "🧹 Cleaning up old backups..."
cd "$BACKUP_DIR" 2>/dev/null && ls -t Qgeth3_* 2>/dev/null | tail -n +6 | xargs rm -rf 2>/dev/null || true
log "✅ Backup cleanup completed"

cleanup
EOF

# Step 8: Create geth runner script with crash recovery
print_step "🏃 Step 8: Creating geth runner script"
cat > "$SCRIPTS_DIR/run-geth.sh" << 'EOF'
#!/bin/bash
# Q Geth Runner with Crash Recovery

# Configuration from environment or defaults
PROJECT_DIR="${PROJECT_DIR:-/opt/qgeth/Qgeth3}"
LOGS_DIR="${LOGS_DIR:-/opt/qgeth/logs}"
GETH_NETWORK="${GETH_NETWORK:-testnet}"
GETH_ARGS="${GETH_ARGS:---http.corsdomain \"*\" --http.api \"eth,net,web3,personal,txpool\"}"
CRASH_RETRY_DELAY="${CRASH_RETRY_DELAY:-300}"
MAX_RETRIES="${MAX_RETRIES:-999999}"

GETH_LOG="$LOGS_DIR/geth-runner.log"
CRASH_COUNT_FILE="$LOGS_DIR/crash_count.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$GETH_LOG"
}

cd "$PROJECT_DIR" || {
    log "❌ Cannot change to project directory: $PROJECT_DIR"
    exit 1
}

# Initialize crash counter
if [ ! -f "$CRASH_COUNT_FILE" ]; then
    echo "0" > "$CRASH_COUNT_FILE"
fi

RETRY_COUNT=$(cat "$CRASH_COUNT_FILE" 2>/dev/null || echo "0")

while true; do
    log "🚀 Starting Q Geth (attempt $((RETRY_COUNT + 1)))"
    
    # Check if geth binary exists
    if [ ! -f "./geth" ] || [ ! -x "./geth" ]; then
        log "❌ Geth binary not found or not executable"
        log "🔄 Triggering update to fix missing binary..."
        systemctl restart qgeth-updater.service
        sleep "$CRASH_RETRY_DELAY"
        continue
    fi
    
    # Start geth with specified arguments
    log "📋 Starting with: ./start-geth.sh $GETH_NETWORK $GETH_ARGS"
    eval "./start-geth.sh $GETH_NETWORK $GETH_ARGS"
    EXIT_CODE=$?
    
    log "⚠️  Geth exited with code: $EXIT_CODE"
    
    # If exit code is 0, it was intentional shutdown
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ Geth shutdown gracefully"
        echo "0" > "$CRASH_COUNT_FILE"  # Reset crash counter
        break
    fi
    
    # Increment retry counter
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "$RETRY_COUNT" > "$CRASH_COUNT_FILE"
    
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        log "❌ Max retries reached ($MAX_RETRIES). Giving up."
        exit 1
    fi
    
    # Check if this is a recurring crash (more than 3 crashes)
    if [ $RETRY_COUNT -gt 3 ]; then
        log "🔄 Multiple crashes detected ($RETRY_COUNT). Triggering update to fix issues..."
        systemctl restart qgeth-updater.service &
        sleep 30  # Give update some time to start
    fi
    
    log "💤 Waiting $CRASH_RETRY_DELAY seconds before retry (attempt $((RETRY_COUNT + 1)))..."
    sleep $CRASH_RETRY_DELAY
done
EOF

# Step 9: Make scripts executable
print_step "🔧 Step 9: Setting permissions"
chmod +x "$SCRIPTS_DIR"/*.sh
chown -R $ACTUAL_USER:$ACTUAL_USER "$INSTALL_DIR"

# Step 10: Create systemd services
print_step "🎯 Step 10: Creating systemd services"

# Main geth service
cat > /etc/systemd/system/qgeth-node.service << EOF
[Unit]
Description=Q Geth Quantum Blockchain Node
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$SCRIPTS_DIR/run-geth.sh
ExecStop=/bin/kill -TERM \$MAINPID
Restart=no
StandardOutput=append:$LOGS_DIR/geth-output.log
StandardError=append:$LOGS_DIR/geth-error.log

# Environment variables
Environment=PROJECT_DIR=$PROJECT_DIR
Environment=LOGS_DIR=$LOGS_DIR
Environment=GETH_NETWORK=$GETH_NETWORK
Environment=GETH_ARGS=$GETH_ARGS
Environment=CRASH_RETRY_DELAY=$CRASH_RETRY_DELAY
Environment=MAX_RETRIES=$MAX_RETRIES
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

# GitHub monitor service
cat > /etc/systemd/system/qgeth-github-monitor.service << EOF
[Unit]
Description=Q Geth GitHub Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$SCRIPTS_DIR/github-monitor.sh
Restart=always
RestartSec=30
StandardOutput=append:$LOGS_DIR/github-monitor.log
StandardError=append:$LOGS_DIR/github-monitor.log

# Environment variables
Environment=GITHUB_REPO=$GITHUB_REPO
Environment=GITHUB_BRANCH=$GITHUB_BRANCH
Environment=PROJECT_DIR=$PROJECT_DIR
Environment=LOGS_DIR=$LOGS_DIR
Environment=CHECK_INTERVAL=$CHECK_INTERVAL

[Install]
WantedBy=multi-user.target
EOF

# Update service (triggered by GitHub monitor)
cat > /etc/systemd/system/qgeth-updater.service << EOF
[Unit]
Description=Q Geth Updater Service
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$SCRIPTS_DIR/update-geth.sh
StandardOutput=append:$LOGS_DIR/update.log
StandardError=append:$LOGS_DIR/update.log

# Environment variables
Environment=PROJECT_DIR=$PROJECT_DIR
Environment=BACKUP_DIR=$BACKUP_DIR
Environment=LOGS_DIR=$LOGS_DIR
Environment=GITHUB_BRANCH=$GITHUB_BRANCH
Environment=ACTUAL_USER=$ACTUAL_USER

[Install]
WantedBy=multi-user.target
EOF

# Step 11: Create management script
print_step "🛠️  Step 11: Creating management script"
cat > /usr/local/bin/qgeth-service << 'EOF'
#!/bin/bash
# Q Geth Auto-Service Management Script

LOGS_DIR="/opt/qgeth/logs"

case "$1" in
    start)
        echo "🚀 Starting Q Geth auto-service..."
        sudo systemctl start qgeth-node.service
        sudo systemctl start qgeth-github-monitor.service
        echo "✅ Services started"
        ;;
    stop)
        echo "🛑 Stopping Q Geth auto-service..."
        sudo systemctl stop qgeth-node.service
        sudo systemctl stop qgeth-github-monitor.service
        echo "✅ Services stopped"
        ;;
    restart)
        echo "🔄 Restarting Q Geth auto-service..."
        sudo systemctl restart qgeth-node.service
        sudo systemctl restart qgeth-github-monitor.service
        echo "✅ Services restarted"
        ;;
    status)
        echo "📊 Q Geth Node Status:"
        sudo systemctl status qgeth-node.service --no-pager
        echo ""
        echo "📊 GitHub Monitor Status:"
        sudo systemctl status qgeth-github-monitor.service --no-pager
        echo ""
        echo "📊 Last Update Status:"
        sudo systemctl status qgeth-updater.service --no-pager
        ;;
    logs)
        case "$2" in
            geth)
                echo "📋 Following geth logs (Ctrl+C to exit)..."
                tail -f "$LOGS_DIR/geth-runner.log" "$LOGS_DIR/geth-output.log" 2>/dev/null
                ;;
            github)
                echo "📋 Following GitHub monitor logs (Ctrl+C to exit)..."
                tail -f "$LOGS_DIR/github-monitor.log" 2>/dev/null
                ;;
            update)
                echo "📋 Following update logs (Ctrl+C to exit)..."
                tail -f "$LOGS_DIR/update.log" 2>/dev/null
                ;;
            all)
                echo "📋 Following all logs (Ctrl+C to exit)..."
                tail -f "$LOGS_DIR"/*.log 2>/dev/null
                ;;
            *)
                echo "Available log types: geth, github, update, all"
                echo "Usage: qgeth-service logs [geth|github|update|all]"
                ;;
        esac
        ;;
    update)
        echo "🔄 Triggering manual update..."
        sudo systemctl restart qgeth-updater.service
        echo "✅ Update triggered (check logs with: qgeth-service logs update)"
        ;;
    reset-crashes)
        echo "🔄 Resetting crash counter..."
        echo "0" > "$LOGS_DIR/crash_count.txt"
        echo "✅ Crash counter reset"
        ;;
    version)
        echo "📋 Q Geth Service Version Info:"
        if [ -f "/opt/qgeth/Qgeth3/geth" ]; then
            /opt/qgeth/Qgeth3/geth version 2>/dev/null | head -3 || echo "Version info unavailable"
        else
            echo "Geth binary not found"
        fi
        ;;
    *)
        echo "Q Geth Auto-Service Management"
        echo ""
        echo "Usage: qgeth-service [command]"
        echo ""
        echo "Commands:"
        echo "  start         - Start auto-service (geth + github monitor)"
        echo "  stop          - Stop auto-service"  
        echo "  restart       - Restart auto-service"
        echo "  status        - Show service status"
        echo "  logs          - View logs (geth|github|update|all)"
        echo "  update        - Trigger manual update"
        echo "  reset-crashes - Reset crash counter"
        echo "  version       - Show geth version"
        echo ""
        echo "Examples:"
        echo "  qgeth-service start"
        echo "  qgeth-service logs geth"
        echo "  qgeth-service status"
        echo ""
        echo "Auto-Service Features:"
        echo "  ✅ Monitors GitHub for updates every 5 minutes"
        echo "  ✅ Auto-rebuilds and restarts on new commits"
        echo "  ✅ Crash recovery with 5-minute retry delay"
        echo "  ✅ Persistent service (starts on boot)"
        echo "  ✅ Memory-optimized builds for low-RAM VPS"
        ;;
esac
EOF

chmod +x /usr/local/bin/qgeth-service

# Step 12: Enable and start services
print_step "⚙️  Step 12: Enabling services"
systemctl daemon-reload
systemctl enable qgeth-node.service
systemctl enable qgeth-github-monitor.service

# Final success message
print_success "🎉 Q Geth Auto-Service Setup Complete!"
echo ""
echo "📋 What was set up:"
echo "  ✅ VPS prepared with memory optimization"
echo "  ✅ UFW firewall configured (SSH, RPC, P2P, WebSocket)"
echo "  ✅ Q Geth cloned and built successfully"
echo "  ✅ GitHub monitoring every 5 minutes"
echo "  ✅ Auto-update on new commits"
echo "  ✅ Crash recovery with 5-minute retry"
echo "  ✅ Persistent systemd services"
echo "  ✅ Management commands available"
echo ""
echo "🚀 Management commands:"
echo "  qgeth-service start       # Start the auto-service"
echo "  qgeth-service status      # Check service status"
echo "  qgeth-service logs geth   # View geth logs"
echo "  qgeth-service logs github # View GitHub monitor logs"
echo "  qgeth-service update      # Trigger manual update"
echo ""
echo "📂 Important directories:"
echo "  Project: $PROJECT_DIR"
echo "  Logs: $LOGS_DIR"
echo "  Backups: $BACKUP_DIR"
echo ""
echo "🔧 Service behavior:"
echo "  • Runs: testnet $GETH_ARGS"
echo "  • Checks GitHub every 5 minutes for updates"
echo "  • Auto-restarts on new commits to main branch"
echo "  • Retries every 5 minutes if geth crashes"
echo "  • Keeps backups of last 5 versions"
echo "  • Memory-optimized builds for VPS environments"
echo ""

# Ask if user wants to start services now
echo -n "Start the auto-service now? (y/N): "
read -r RESPONSE
if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    print_step "🚀 Starting services..."
    systemctl start qgeth-github-monitor.service
    sleep 2
    systemctl start qgeth-node.service
    sleep 5
    
    if systemctl is-active --quiet qgeth-node.service && systemctl is-active --quiet qgeth-github-monitor.service; then
        print_success "🎉 Auto-service is running!"
        echo ""
        echo "📊 Current status:"
        systemctl status qgeth-node.service --no-pager -l
        echo ""
        echo "🔍 To monitor:"
        echo "  qgeth-service logs geth     # Watch geth logs"
        echo "  qgeth-service status        # Check all services"
    else
        print_warning "Services may have issues starting. Check status:"
        echo "  qgeth-service status"
        echo "  qgeth-service logs all"
    fi
else
    echo ""
    echo "Services not started. Start them when ready:"
    echo "  qgeth-service start"
fi

print_success "✅ Setup complete! Your VPS now has a fully automated Q Geth service."
