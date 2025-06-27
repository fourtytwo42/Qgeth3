#!/bin/bash
# Q Geth Auto-Updating Node Setup Script
# Run this script to set up automatic node management with GitHub monitoring

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
    echo "Usage: sudo ./setup-auto-node.sh"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

print_step "🚀 Setting up Q Geth Auto-Updating Node System"
echo "User: $ACTUAL_USER"
echo "Project Directory: $(pwd)"
echo ""

# Install dependencies
print_step "📦 Installing dependencies..."
apt update -qq
apt install -y git jq curl build-essential systemd

# Create directory structure
print_step "📁 Creating directory structure..."
mkdir -p /opt/qgeth
mkdir -p /opt/qgeth/logs
mkdir -p /opt/qgeth/scripts
mkdir -p /opt/qgeth/backup

# Get current directory (should be the Qgeth3 project root)
PROJECT_DIR=$(pwd)
GITHUB_REPO=$(git remote get-url origin | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git.*/\1/' | sed 's/.*github.com[:/]\([^/]*\/[^/]*\).*/\1/')

if [ -z "$GITHUB_REPO" ]; then
    print_warning "Could not detect GitHub repository from git remote"
    echo -n "Please enter your GitHub repository (format: owner/repo): "
    read GITHUB_REPO
fi

print_step "📋 Configuration detected:"
echo "  GitHub Repository: $GITHUB_REPO"
echo "  Project Directory: $PROJECT_DIR"

# Create configuration file
print_step "⚙️  Creating configuration..."
cat > /opt/qgeth/config.env << EOF
#!/bin/bash
# Q Geth Node Configuration

# GitHub Repository
GITHUB_REPO="$GITHUB_REPO"
GITHUB_BRANCH="main"
GITHUB_API_URL="https://api.github.com/repos/\${GITHUB_REPO}/commits/\${GITHUB_BRANCH}"

# Local Paths
QGETH_DIR="/opt/qgeth/Qgeth3"
BACKUP_DIR="/opt/qgeth/backup"
LOG_DIR="/opt/qgeth/logs"
SCRIPTS_DIR="/opt/qgeth/scripts"

# Node Configuration
GETH_NETWORK="devnet"
GETH_EXTRA_ARGS="--http.corsdomain '*' --http.api 'eth,net,web3,personal,txpool,miner,qmpow'"

# Update Settings
CHECK_INTERVAL=300  # Check GitHub every 5 minutes
RETRY_DELAY=300     # Wait 5 minutes before retry after crash
MAX_RETRIES=999999  # Essentially unlimited retries

# Logging
LOG_FILE="\${LOG_DIR}/qgeth-node.log"
UPDATE_LOG="\${LOG_DIR}/update.log"
GITHUB_LOG="\${LOG_DIR}/github-monitor.log"
EOF

# Create GitHub Monitor Script
print_step "🔍 Creating GitHub monitor script..."
cat > /opt/qgeth/scripts/github-monitor.sh << 'EOF'
#!/bin/bash
# GitHub Monitor - Watches for changes to main branch

source /opt/qgeth/config.env

LAST_COMMIT_FILE="${LOG_DIR}/last_commit.txt"
LOCK_FILE="${LOG_DIR}/github-monitor.lock"

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

# Create lock file
echo $$ > "$LOCK_FILE"

log "🔍 GitHub monitor started - watching $GITHUB_REPO:$GITHUB_BRANCH"

while true; do
    # Fetch latest commit hash
    LATEST_COMMIT=$(curl -s "$GITHUB_API_URL" | jq -r '.sha' 2>/dev/null)
    
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
        log "🔄 Triggering node update..."
        ${SCRIPTS_DIR}/update-node.sh &
        
        # Wait a bit before next check to avoid rapid updates
        sleep 60
    else
        log "✅ No changes detected ($LATEST_COMMIT)"
    fi
    
    sleep "$CHECK_INTERVAL"
done
EOF

# Create Update Script
print_step "🔨 Creating update script..."
cat > /opt/qgeth/scripts/update-node.sh << 'EOF'
#!/bin/bash
# Node Update Script - Handles pulling, building, and restarting

source /opt/qgeth/config.env

UPDATE_LOCK_FILE="${LOG_DIR}/update.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$UPDATE_LOG"
}

cleanup() {
    rm -f "$UPDATE_LOCK_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Check if update already running
if [ -f "$UPDATE_LOCK_FILE" ]; then
    PID=$(cat "$UPDATE_LOCK_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        log "Update already in progress (PID: $PID)"
        exit 1
    else
        rm -f "$UPDATE_LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$UPDATE_LOCK_FILE"

log "🔄 Starting node update process..."

# Stop the geth service
log "🛑 Stopping geth service..."
sudo systemctl stop qgeth-node

# Wait for graceful shutdown
sleep 10

# Backup current version
log "💾 Creating backup..."
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
if [ -d "$QGETH_DIR" ]; then
    cp -r "$QGETH_DIR" "${BACKUP_DIR}/Qgeth3_${TIMESTAMP}"
    log "✅ Backup created: Qgeth3_${TIMESTAMP}"
fi

# Pull latest changes
log "📥 Pulling latest changes from GitHub..."
cd /opt/qgeth

if [ ! -d "$QGETH_DIR" ]; then
    log "📦 Cloning repository..."
    git clone https://github.com/${GITHUB_REPO}.git
    if [ $? -ne 0 ]; then
        log "❌ Failed to clone repository"
        cleanup
    fi
else
    cd "$QGETH_DIR"
    log "🔄 Fetching updates..."
    git fetch origin
    git reset --hard origin/$GITHUB_BRANCH
    if [ $? -ne 0 ]; then
        log "❌ Failed to pull updates"
        cleanup
    fi
fi

cd "$QGETH_DIR"

# Memory check for build
log "💾 Checking system memory for build..."
REQUIRED_MB=3072  # 3GB minimum
AVAILABLE_MB=0

if [ -f /proc/meminfo ]; then
    MEM_TOTAL=\$(grep MemTotal /proc/meminfo | awk '{print \$2}')
    MEM_AVAILABLE=\$(grep MemAvailable /proc/meminfo | awk '{print \$2}')
    
    if [ -n "\$MEM_AVAILABLE" ]; then
        AVAILABLE_MB=\$((MEM_AVAILABLE / 1024))
    elif [ -n "\$MEM_TOTAL" ]; then
        AVAILABLE_MB=\$((MEM_TOTAL / 1024))
    fi
    
    log "Available RAM: \${AVAILABLE_MB}MB (Required: \${REQUIRED_MB}MB)"
    
    if [ \$AVAILABLE_MB -lt \$REQUIRED_MB ]; then
        log "⚠️  Low memory detected for building!"
        log "   Available: \${AVAILABLE_MB}MB, Required: \${REQUIRED_MB}MB"
        
        # Create swap file if it doesn't exist
        if [ ! -f /swapfile ]; then
            log "🔧 Adding 2GB swap space to help with build..."
            sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1024 count=2097152
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            
            # Add to fstab for persistence
            if ! grep -q "/swapfile" /etc/fstab; then
                echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
            fi
            
            log "✅ Swap space added successfully"
        else
            log "🔧 Ensuring swap file is active..."
            sudo swapon /swapfile 2>/dev/null || true
        fi
    else
        log "✅ Memory check passed"
    fi
else
    log "⚠️  Cannot check memory - /proc/meminfo not found"
fi

# Build the new version with memory optimization
log "🔨 Building new version with memory optimization..."

# Set memory-optimized environment
export TMPDIR="/tmp/qgeth-update-\$\$"
mkdir -p "\$TMPDIR"

# Cleanup function for temp directory
cleanup_build_temp() {
    rm -rf "\$TMPDIR" 2>/dev/null || true
}
trap cleanup_build_temp EXIT

log "Using temporary build directory: \$TMPDIR"
chmod +x build-linux.sh
./build-linux.sh geth

if [ $? -ne 0 ]; then
    log "❌ Build failed! Restoring backup..."
    
    # Restore from backup
    LATEST_BACKUP=$(ls -t ${BACKUP_DIR}/Qgeth3_* 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        rm -rf "$QGETH_DIR"
        cp -r "$LATEST_BACKUP" "$QGETH_DIR"
        log "✅ Restored from backup: $(basename $LATEST_BACKUP)"
    else
        log "❌ No backup available!"
    fi
    
    cleanup
fi

# Test the build
if [ ! -f "$QGETH_DIR/geth" ] || [ ! -x "$QGETH_DIR/geth" ]; then
    log "❌ Build verification failed - geth binary not found or not executable"
    cleanup
fi

log "✅ Build successful!"

# Start the service
log "🚀 Starting geth service..."
sudo systemctl start qgeth-node

# Wait and verify it started
sleep 15
if sudo systemctl is-active --quiet qgeth-node; then
    log "✅ Node update completed successfully!"
else
    log "❌ Failed to start node after update"
fi

# Cleanup old backups (keep last 5)
log "🧹 Cleaning up old backups..."
cd "$BACKUP_DIR"
ls -t Qgeth3_* 2>/dev/null | tail -n +6 | xargs rm -rf 2>/dev/null
log "✅ Backup cleanup completed"

cleanup
EOF

# Create Node Runner Script
print_step "🏃 Creating node runner script..."
cat > /opt/qgeth/scripts/run-node.sh << 'EOF'
#!/bin/bash
# Node Runner - Handles geth execution with retry logic

source /opt/qgeth/config.env

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$QGETH_DIR" || {
    log "❌ Cannot change to Qgeth directory: $QGETH_DIR"
    exit 1
}

RETRY_COUNT=0

while true; do
    log "🚀 Starting Q Geth Node (attempt $((RETRY_COUNT + 1)))"
    
    # Check if geth binary exists
    if [ ! -f "./geth" ] || [ ! -x "./geth" ]; then
        log "❌ Geth binary not found or not executable"
        exit 1
    fi
    
    # Start geth
    ./start-geth.sh $GETH_NETWORK $GETH_EXTRA_ARGS
    EXIT_CODE=$?
    
    log "⚠️  Geth exited with code: $EXIT_CODE"
    
    # If exit code is 0, it was intentional shutdown
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ Geth shutdown gracefully"
        break
    fi
    
    # Increment retry counter
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        log "❌ Max retries reached ($MAX_RETRIES). Giving up."
        exit 1
    fi
    
    log "💤 Waiting ${RETRY_DELAY} seconds before retry..."
    sleep $RETRY_DELAY
done
EOF

# Create systemd service for the node
print_step "🎯 Creating systemd services..."
cat > /etc/systemd/system/qgeth-node.service << EOF
[Unit]
Description=Q Geth Quantum Blockchain Node
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=/opt/qgeth/Qgeth3
ExecStart=/opt/qgeth/scripts/run-node.sh
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=30
StandardOutput=append:/opt/qgeth/logs/qgeth-node.log
StandardError=append:/opt/qgeth/logs/qgeth-node.log

# Environment
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for GitHub monitor
cat > /etc/systemd/system/qgeth-github-monitor.service << EOF
[Unit]
Description=Q Geth GitHub Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=/opt/qgeth
ExecStart=/opt/qgeth/scripts/github-monitor.sh
Restart=always
RestartSec=30
StandardOutput=append:/opt/qgeth/logs/github-monitor.log
StandardError=append:/opt/qgeth/logs/github-monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Make scripts executable
print_step "🔧 Setting permissions..."
chmod +x /opt/qgeth/scripts/*.sh
chown -R $ACTUAL_USER:$ACTUAL_USER /opt/qgeth

# Copy current project to /opt/qgeth/Qgeth3
print_step "📦 Installing current project..."
cp -r "$PROJECT_DIR" /opt/qgeth/Qgeth3
chown -R $ACTUAL_USER:$ACTUAL_USER /opt/qgeth/Qgeth3

# Memory check for build
print_step "💾 Checking system memory for build..."
REQUIRED_MB=3072  # 3GB minimum
AVAILABLE_MB=0

if [ -f /proc/meminfo ]; then
    MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_AVAILABLE=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    
    if [ -n "$MEM_AVAILABLE" ]; then
        AVAILABLE_MB=$((MEM_AVAILABLE / 1024))
    elif [ -n "$MEM_TOTAL" ]; then
        AVAILABLE_MB=$((MEM_TOTAL / 1024))
    fi
    
    print_step "Available RAM: ${AVAILABLE_MB}MB (Required: ${REQUIRED_MB}MB)"
    
    if [ $AVAILABLE_MB -lt $REQUIRED_MB ]; then
        print_warning "Low memory detected for building!"
        echo "  Available: ${AVAILABLE_MB}MB"
        echo "  Required: ${REQUIRED_MB}MB"
        echo ""
        echo "🔧 Adding 2GB swap space to help with build..."
        
        # Create swap file if it doesn't exist
        if [ ! -f /swapfile ]; then
            fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1024 count=2097152
            chmod 600 /swapfile
            mkswap /swapfile
            swapon /swapfile
            
            # Add to fstab for persistence
            if ! grep -q "/swapfile" /etc/fstab; then
                echo "/swapfile none swap sw 0 0" >> /etc/fstab
            fi
            
            print_success "Swap space added successfully"
        else
            print_step "Swap file already exists, ensuring it's active..."
            swapon /swapfile 2>/dev/null || true
        fi
    else
        print_success "Memory check passed"
    fi
else
    print_warning "Cannot check memory - /proc/meminfo not found"
fi

# Build the project with memory optimization
print_step "🔨 Building Qgeth with memory optimization..."
cd /opt/qgeth/Qgeth3
chmod +x build-linux.sh

# Set memory-optimized environment
export TMPDIR="/tmp/qgeth-setup-$$"
mkdir -p "$TMPDIR"

# Cleanup function for temp directory
cleanup_build_temp() {
    rm -rf "$TMPDIR" 2>/dev/null || true
}
trap cleanup_build_temp EXIT

print_step "Using temporary build directory: $TMPDIR"
sudo -u $ACTUAL_USER ./build-linux.sh geth

# Reload systemd and enable services
print_step "⚙️  Configuring systemd services..."
systemctl daemon-reload
systemctl enable qgeth-node
systemctl enable qgeth-github-monitor

# Create management script
print_step "🛠️  Creating management script..."
cat > /usr/local/bin/qgeth-manage << 'EOF'
#!/bin/bash
# Q Geth Node Management Script

case "$1" in
    start)
        echo "🚀 Starting Q Geth services..."
        sudo systemctl start qgeth-node
        sudo systemctl start qgeth-github-monitor
        echo "✅ Services started"
        ;;
    stop)
        echo "🛑 Stopping Q Geth services..."
        sudo systemctl stop qgeth-node
        sudo systemctl stop qgeth-github-monitor
        echo "✅ Services stopped"
        ;;
    restart)
        echo "🔄 Restarting Q Geth services..."
        sudo systemctl restart qgeth-node
        sudo systemctl restart qgeth-github-monitor
        echo "✅ Services restarted"
        ;;
    status)
        echo "📊 Q Geth Node Status:"
        sudo systemctl status qgeth-node --no-pager
        echo ""
        echo "📊 GitHub Monitor Status:"
        sudo systemctl status qgeth-github-monitor --no-pager
        ;;
    logs)
        case "$2" in
            node)
                tail -f /opt/qgeth/logs/qgeth-node.log
                ;;
            github)
                tail -f /opt/qgeth/logs/github-monitor.log
                ;;
            update)
                tail -f /opt/qgeth/logs/update.log
                ;;
            *)
                echo "Available log types: node, github, update"
                echo "Usage: qgeth-manage logs [node|github|update]"
                ;;
        esac
        ;;
    update)
        echo "🔄 Triggering manual update..."
        sudo /opt/qgeth/scripts/update-node.sh
        ;;
    *)
        echo "Q Geth Node Management"
        echo ""
        echo "Usage: qgeth-manage [command]"
        echo ""
        echo "Commands:"
        echo "  start    - Start both services"
        echo "  stop     - Stop both services"  
        echo "  restart  - Restart both services"
        echo "  status   - Show service status"
        echo "  logs     - View logs (node|github|update)"
        echo "  update   - Trigger manual update"
        echo ""
        echo "Examples:"
        echo "  qgeth-manage start"
        echo "  qgeth-manage logs node"
        echo "  qgeth-manage status"
        ;;
esac
EOF

chmod +x /usr/local/bin/qgeth-manage

# Completion message
print_success "🎉 Q Geth Auto-Updating Node System installed successfully!"
echo ""
echo "📋 Management commands:"
echo "  qgeth-manage start    - Start services"
echo "  qgeth-manage stop     - Stop services"
echo "  qgeth-manage status   - Check status"
echo "  qgeth-manage logs node - View node logs"
echo ""
echo "🚀 To start the services now:"
echo "  qgeth-manage start"
echo ""
echo "📂 Files created:"
echo "  - Configuration: /opt/qgeth/config.env"
echo "  - Scripts: /opt/qgeth/scripts/"
echo "  - Logs: /opt/qgeth/logs/"
echo "  - Services: qgeth-node, qgeth-github-monitor"
echo ""
echo "🔍 The system will:"
echo "  ✅ Auto-start on boot"
echo "  ✅ Monitor GitHub for changes every 5 minutes"
echo "  ✅ Auto-update and restart on new commits"
echo "  ✅ Retry every 5 minutes if crashed"
echo "  ✅ Keep backups of last 5 versions"

# Ask if user wants to start services now
echo ""
echo -n "Start the services now? (y/N): "
read -r RESPONSE
if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    print_step "🚀 Starting services..."
    systemctl start qgeth-node
    systemctl start qgeth-github-monitor
    print_success "Services started! Use 'qgeth-manage status' to check status."
else
    echo "Services not started. Use 'qgeth-manage start' when ready."
fi 