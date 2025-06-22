#!/bin/bash

# Quantum-Geth Mining Monitor
# Linux/macOS version of monitor-mining.ps1

# Default configuration
DATADIR="qdata_quantum"
REFRESH_INTERVAL=3
LOG_LINES=50
SHOW_HELP=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --datadir DIR       Set data directory (default: qdata_quantum)"
    echo "  --interval SEC      Refresh interval in seconds (default: 3)"
    echo "  --lines NUM         Number of log lines to show (default: 50)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --interval 5"
    echo "  $0 --lines 100"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datadir)
            DATADIR="$2"
            shift 2
            ;;
        --interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --lines)
            LOG_LINES="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate numeric parameters
if ! [[ "$REFRESH_INTERVAL" =~ ^[0-9]+$ ]] || [ "$REFRESH_INTERVAL" -lt 1 ]; then
    echo -e "${RED}Error: Refresh interval must be a positive integer${NC}"
    exit 1
fi

if ! [[ "$LOG_LINES" =~ ^[0-9]+$ ]] || [ "$LOG_LINES" -lt 1 ]; then
    echo -e "${RED}Error: Log lines must be a positive integer${NC}"
    exit 1
fi

# Function to get blockchain info
get_blockchain_info() {
    local GETH_BIN=""
    
    # Find geth binary
    if [ -f "./quantum-geth/build/bin/geth" ]; then
        GETH_BIN="./quantum-geth/build/bin/geth"
    elif [ -f "./geth" ]; then
        GETH_BIN="./geth"
    elif [ -f "./geth.exe" ]; then
        GETH_BIN="./geth.exe"
    else
        echo "N/A (geth binary not found)"
        return 1
    fi
    
    # Try to get blockchain info via IPC
    local info
    if info=$($GETH_BIN --datadir "$DATADIR" --exec "eth.blockNumber" attach 2>/dev/null); then
        echo "$info"
    else
        echo "N/A (unable to connect)"
    fi
}

# Function to get mining stats from logs
get_mining_stats() {
    local log_file="$DATADIR/geth/geth.log"
    
    if [ ! -f "$log_file" ]; then
        echo "No log file found"
        return 1
    fi
    
    # Extract recent mining info
    local recent_logs
    recent_logs=$(tail -n "$LOG_LINES" "$log_file" 2>/dev/null | grep -E "(Successfully sealed new block|Quantum target check|Block mining|qnonce=)" | tail -n 10)
    
    echo "$recent_logs"
}

# Function to calculate mining rate
calculate_mining_rate() {
    local log_file="$DATADIR/geth/geth.log"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return 1
    fi
    
    # Get timestamps of last 5 successful blocks
    local block_times
    block_times=$(grep "Successfully sealed new block" "$log_file" | tail -n 5 | grep -o '\[[0-9][0-9]-[0-9][0-9]|[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9][0-9]\]' | sed 's/\[//;s/\]//')
    
    if [ -z "$block_times" ]; then
        echo "N/A (no recent blocks)"
        return 1
    fi
    
    # Simple rate calculation (blocks per minute)
    local block_count
    block_count=$(echo "$block_times" | wc -l)
    
    if [ "$block_count" -ge 2 ]; then
        echo "${block_count} blocks in recent history"
    else
        echo "N/A (insufficient data)"
    fi
}

# Function to get current difficulty
get_current_difficulty() {
    local log_file="$DATADIR/geth/geth.log"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return 1
    fi
    
    # Extract most recent difficulty from logs
    local difficulty
    difficulty=$(grep "difficulty=" "$log_file" | tail -n 1 | grep -o 'difficulty=[0-9]*' | cut -d= -f2)
    
    if [ -n "$difficulty" ]; then
        echo "$difficulty"
    else
        echo "N/A"
    fi
}

# Function to get process info
get_process_info() {
    local geth_pids
    geth_pids=$(pgrep -f "geth.*mine" 2>/dev/null)
    
    if [ -n "$geth_pids" ]; then
        local pid_count
        pid_count=$(echo "$geth_pids" | wc -l)
        echo "${pid_count} mining process(es) running"
        
        # Get memory usage for first process
        local first_pid
        first_pid=$(echo "$geth_pids" | head -n 1)
        
        if command -v ps >/dev/null 2>&1; then
            local mem_info
            if mem_info=$(ps -p "$first_pid" -o rss= 2>/dev/null); then
                local mem_mb
                mem_mb=$((mem_info / 1024))
                echo "Memory usage: ${mem_mb}MB (PID: $first_pid)"
            fi
        fi
    else
        echo "No mining processes found"
    fi
}

# Function to extract latest nonce progression
get_nonce_progression() {
    local log_file="$DATADIR/geth/geth.log"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return 1
    fi
    
    # Extract recent qnonce values
    local nonces
    nonces=$(grep "qnonce=" "$log_file" | tail -n 5 | grep -o 'qnonce=[0-9]*' | cut -d= -f2 | tr '\n' '→' | sed 's/→$//')
    
    if [ -n "$nonces" ]; then
        echo "$nonces"
    else
        echo "N/A"
    fi
}

# Function to display dashboard
display_dashboard() {
    clear
    
    echo -e "${BOLD}${CYAN}🔬 QUANTUM-GETH MINING DASHBOARD${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # System information
    echo -e "${BOLD}${YELLOW}⚡ System Status${NC}"
    echo -e "${CYAN}├─${NC} Timestamp: $(date)"
    echo -e "${CYAN}├─${NC} Data Directory: $DATADIR"
    echo -e "${CYAN}├─${NC} Refresh Interval: ${REFRESH_INTERVAL}s"
    
    # Process information
    echo -e "${CYAN}└─${NC} Process Info: $(get_process_info)"
    echo ""
    
    # Blockchain information
    echo -e "${BOLD}${GREEN}⛓️  Blockchain Status${NC}"
    local block_number
    block_number=$(get_blockchain_info)
    echo -e "${CYAN}├─${NC} Block Height: $block_number"
    
    local difficulty
    difficulty=$(get_current_difficulty)
    echo -e "${CYAN}├─${NC} Current Difficulty: $difficulty"
    
    local mining_rate
    mining_rate=$(calculate_mining_rate)
    echo -e "${CYAN}└─${NC} Mining Rate: $mining_rate"
    echo ""
    
    # Quantum mining details
    echo -e "${BOLD}${MAGENTA}🔮 Quantum Mining Status${NC}"
    local nonce_prog
    nonce_prog=$(get_nonce_progression)
    echo -e "${CYAN}├─${NC} QNonce Progression: $nonce_prog"
    echo -e "${CYAN}├─${NC} Security Level: 1,152-bit (48 puzzles × 12 qubits)"
    echo -e "${CYAN}├─${NC} Gate Complexity: 4,096 T-gates per puzzle"
    echo -e "${CYAN}└─${NC} Mining Algorithm: Bitcoin-style Quantum PoW"
    echo ""
    
    # Recent mining activity
    echo -e "${BOLD}${BLUE}📊 Recent Mining Activity${NC}"
    echo -e "${CYAN}──────────────────────────────────────────────────────────────${NC}"
    
    local mining_stats
    mining_stats=$(get_mining_stats)
    
    if [ -n "$mining_stats" ]; then
        echo "$mining_stats" | while IFS= read -r line; do
            # Color code different types of log entries
            if echo "$line" | grep -q "Successfully sealed"; then
                echo -e "${GREEN}✓ $line${NC}"
            elif echo "$line" | grep -q "Quantum target check"; then
                echo -e "${YELLOW}🎯 $line${NC}"
            elif echo "$line" | grep -q "qnonce="; then
                echo -e "${BLUE}🔄 $line${NC}"
            else
                echo -e "${NC}  $line${NC}"
            fi
        done
    else
        echo -e "${YELLOW}No recent mining activity found${NC}"
        echo -e "${CYAN}Check if mining is running with:${NC} ./scripts/start-mining.sh"
    fi
    
    echo ""
    echo -e "${CYAN}──────────────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit monitoring${NC}"
}

# Main monitoring loop
echo -e "${BOLD}${CYAN}Starting Quantum-Geth Mining Monitor...${NC}"
echo -e "${YELLOW}Monitoring data directory: $DATADIR${NC}"
echo -e "${YELLOW}Refresh interval: ${REFRESH_INTERVAL} seconds${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop monitoring${NC}"
sleep 2

# Trap Ctrl+C to clean exit
trap 'echo -e "\n${YELLOW}Monitoring stopped.${NC}"; exit 0' INT

# Main monitoring loop
while true; do
    display_dashboard
    sleep "$REFRESH_INTERVAL"
done 