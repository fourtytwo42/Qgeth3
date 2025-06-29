#!/usr/bin/env bash
# Q Coin System Detection Library
# Provides cross-distribution compatibility functions
# Source this file in other scripts: source ./detect-system.sh

# Colors for output
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    # No colors in non-terminal environments
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    PURPLE=''
    CYAN=''
    NC=''
fi

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [ "${DEBUG:-}" = "true" ]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Detect operating system and distribution
detect_os() {
    QGETH_OS=""
    QGETH_DISTRO=""
    QGETH_DISTRO_VERSION=""
    QGETH_DISTRO_CODENAME=""
    
    # Detect kernel type
    case "$(uname -s)" in
        Linux*)
            QGETH_OS="linux"
            ;;
        Darwin*)
            QGETH_OS="macos"
            log_error "macOS not supported by Linux scripts"
            return 1
            ;;
        FreeBSD*)
            QGETH_OS="freebsd"
            log_warning "FreeBSD support experimental"
            ;;
        *)
            QGETH_OS="unknown"
            log_error "Unsupported operating system: $(uname -s)"
            return 1
            ;;
    esac
    
    # Detect Linux distribution
    if [ "$QGETH_OS" = "linux" ] || [ "$QGETH_OS" = "freebsd" ]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            QGETH_DISTRO="$ID"
            QGETH_DISTRO_VERSION="$VERSION_ID"
            QGETH_DISTRO_CODENAME="${VERSION_CODENAME:-}"
        elif [ -f /etc/lsb-release ]; then
            . /etc/lsb-release
            QGETH_DISTRO="$(echo "$DISTRIB_ID" | tr '[:upper:]' '[:lower:]')"
            QGETH_DISTRO_VERSION="$DISTRIB_RELEASE"
            QGETH_DISTRO_CODENAME="$DISTRIB_CODENAME"
        elif [ -f /etc/redhat-release ]; then
            if grep -q "CentOS" /etc/redhat-release; then
                QGETH_DISTRO="centos"
            elif grep -q "Red Hat" /etc/redhat-release; then
                QGETH_DISTRO="rhel"
            elif grep -q "Fedora" /etc/redhat-release; then
                QGETH_DISTRO="fedora"
            fi
            QGETH_DISTRO_VERSION=$(grep -o '[0-9]\+' /etc/redhat-release | head -1)
        elif [ -f /etc/arch-release ]; then
            QGETH_DISTRO="arch"
        elif [ -f /etc/gentoo-release ]; then
            QGETH_DISTRO="gentoo"
        elif [ -f /etc/alpine-release ]; then
            QGETH_DISTRO="alpine"
            QGETH_DISTRO_VERSION=$(cat /etc/alpine-release)
        else
            QGETH_DISTRO="unknown"
        fi
    fi
    
    log_debug "Detected OS: $QGETH_OS"
    log_debug "Detected Distribution: $QGETH_DISTRO $QGETH_DISTRO_VERSION"
    
    export QGETH_OS QGETH_DISTRO QGETH_DISTRO_VERSION QGETH_DISTRO_CODENAME
}

# Detect package manager and set commands
detect_package_manager() {
    QGETH_PKG_MANAGER=""
    QGETH_PKG_UPDATE=""
    QGETH_PKG_INSTALL=""
    QGETH_PKG_SEARCH=""
    QGETH_PKG_REMOVE=""
    QGETH_PKG_AUTOREMOVE=""
    
    # Try to detect based on available commands
    if command -v apt >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="apt"
        QGETH_PKG_UPDATE="apt update"
        QGETH_PKG_INSTALL="DEBIAN_FRONTEND=noninteractive apt install -y"
        QGETH_PKG_SEARCH="apt search"
        QGETH_PKG_REMOVE="apt remove -y"
        QGETH_PKG_AUTOREMOVE="apt autoremove -y"
    elif command -v dnf >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="dnf"
        QGETH_PKG_UPDATE="dnf check-update"
        QGETH_PKG_INSTALL="dnf install -y"
        QGETH_PKG_SEARCH="dnf search"
        QGETH_PKG_REMOVE="dnf remove -y"
        QGETH_PKG_AUTOREMOVE="dnf autoremove -y"
    elif command -v yum >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="yum"
        QGETH_PKG_UPDATE="yum check-update"
        QGETH_PKG_INSTALL="yum install -y"
        QGETH_PKG_SEARCH="yum search"
        QGETH_PKG_REMOVE="yum remove -y"
        QGETH_PKG_AUTOREMOVE="yum autoremove -y"
    elif command -v pacman >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="pacman"
        QGETH_PKG_UPDATE="pacman -Sy"
        QGETH_PKG_INSTALL="pacman -S --noconfirm"
        QGETH_PKG_SEARCH="pacman -Ss"
        QGETH_PKG_REMOVE="pacman -R --noconfirm"
        QGETH_PKG_AUTOREMOVE="pacman -Rs --noconfirm"
    elif command -v zypper >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="zypper"
        QGETH_PKG_UPDATE="zypper refresh"
        QGETH_PKG_INSTALL="zypper install -y"
        QGETH_PKG_SEARCH="zypper search"
        QGETH_PKG_REMOVE="zypper remove -y"
        QGETH_PKG_AUTOREMOVE="zypper remove -y --clean-deps"
    elif command -v emerge >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="emerge"
        QGETH_PKG_UPDATE="emerge --sync"
        QGETH_PKG_INSTALL="emerge"
        QGETH_PKG_SEARCH="emerge --search"
        QGETH_PKG_REMOVE="emerge --unmerge"
        QGETH_PKG_AUTOREMOVE="emerge --depclean"
    elif command -v apk >/dev/null 2>&1; then
        QGETH_PKG_MANAGER="apk"
        QGETH_PKG_UPDATE="apk update"
        QGETH_PKG_INSTALL="apk add"
        QGETH_PKG_SEARCH="apk search"
        QGETH_PKG_REMOVE="apk del"
        QGETH_PKG_AUTOREMOVE="apk del --purge"
    else
        QGETH_PKG_MANAGER="unknown"
        log_error "No supported package manager found"
        return 1
    fi
    
    log_debug "Detected package manager: $QGETH_PKG_MANAGER"
    
    export QGETH_PKG_MANAGER QGETH_PKG_UPDATE QGETH_PKG_INSTALL QGETH_PKG_SEARCH QGETH_PKG_REMOVE QGETH_PKG_AUTOREMOVE
}

# Detect init system
detect_init_system() {
    QGETH_INIT_SYSTEM=""
    QGETH_SERVICE_ENABLE=""
    QGETH_SERVICE_DISABLE=""
    QGETH_SERVICE_START=""
    QGETH_SERVICE_STOP=""
    QGETH_SERVICE_RESTART=""
    QGETH_SERVICE_STATUS=""
    QGETH_SERVICE_RELOAD=""
    
    if command -v systemctl >/dev/null 2>&1 && systemctl --version >/dev/null 2>&1; then
        QGETH_INIT_SYSTEM="systemd"
        QGETH_SERVICE_ENABLE="systemctl enable"
        QGETH_SERVICE_DISABLE="systemctl disable"
        QGETH_SERVICE_START="systemctl start"
        QGETH_SERVICE_STOP="systemctl stop"
        QGETH_SERVICE_RESTART="systemctl restart"
        QGETH_SERVICE_STATUS="systemctl status"
        QGETH_SERVICE_RELOAD="systemctl daemon-reload"
    elif command -v rc-service >/dev/null 2>&1; then
        QGETH_INIT_SYSTEM="openrc"
        QGETH_SERVICE_ENABLE="rc-update add"
        QGETH_SERVICE_DISABLE="rc-update del"
        QGETH_SERVICE_START="rc-service start"
        QGETH_SERVICE_STOP="rc-service stop"
        QGETH_SERVICE_RESTART="rc-service restart"
        QGETH_SERVICE_STATUS="rc-service status"
        QGETH_SERVICE_RELOAD="rc-service reload"
    elif [ -d /etc/init.d ] && command -v service >/dev/null 2>&1; then
        QGETH_INIT_SYSTEM="sysv"
        QGETH_SERVICE_ENABLE="update-rc.d enable"
        QGETH_SERVICE_DISABLE="update-rc.d disable"
        QGETH_SERVICE_START="service start"
        QGETH_SERVICE_STOP="service stop"
        QGETH_SERVICE_RESTART="service restart"
        QGETH_SERVICE_STATUS="service status"
        QGETH_SERVICE_RELOAD="service reload"
    else
        QGETH_INIT_SYSTEM="unknown"
        log_warning "Unknown init system - manual service management required"
    fi
    
    log_debug "Detected init system: $QGETH_INIT_SYSTEM"
    
    export QGETH_INIT_SYSTEM QGETH_SERVICE_ENABLE QGETH_SERVICE_DISABLE QGETH_SERVICE_START QGETH_SERVICE_STOP QGETH_SERVICE_RESTART QGETH_SERVICE_STATUS QGETH_SERVICE_RELOAD
}

# Detect firewall system
detect_firewall() {
    QGETH_FIREWALL=""
    QGETH_FIREWALL_ENABLE=""
    QGETH_FIREWALL_DISABLE=""
    QGETH_FIREWALL_ALLOW=""
    QGETH_FIREWALL_DENY=""
    QGETH_FIREWALL_STATUS=""
    
    if command -v ufw >/dev/null 2>&1; then
        QGETH_FIREWALL="ufw"
        QGETH_FIREWALL_ENABLE="ufw --force enable"
        QGETH_FIREWALL_DISABLE="ufw --force disable"
        QGETH_FIREWALL_ALLOW="ufw allow"
        QGETH_FIREWALL_DENY="ufw deny"
        QGETH_FIREWALL_STATUS="ufw status"
    elif command -v firewall-cmd >/dev/null 2>&1; then
        QGETH_FIREWALL="firewalld"
        QGETH_FIREWALL_ENABLE="systemctl enable firewalld && systemctl start firewalld"
        QGETH_FIREWALL_DISABLE="systemctl stop firewalld && systemctl disable firewalld"
        QGETH_FIREWALL_ALLOW="firewall-cmd --permanent --add-port"
        QGETH_FIREWALL_DENY="firewall-cmd --permanent --remove-port"
        QGETH_FIREWALL_STATUS="firewall-cmd --state"
    elif command -v iptables >/dev/null 2>&1; then
        QGETH_FIREWALL="iptables"
        QGETH_FIREWALL_ENABLE="echo 'iptables rules active'"
        QGETH_FIREWALL_DISABLE="iptables -F"
        QGETH_FIREWALL_ALLOW="iptables -A INPUT -p tcp --dport"
        QGETH_FIREWALL_DENY="iptables -D INPUT -p tcp --dport"
        QGETH_FIREWALL_STATUS="iptables -L"
    else
        QGETH_FIREWALL="none"
        log_warning "No supported firewall detected"
    fi
    
    log_debug "Detected firewall: $QGETH_FIREWALL"
    
    export QGETH_FIREWALL QGETH_FIREWALL_ENABLE QGETH_FIREWALL_DISABLE QGETH_FIREWALL_ALLOW QGETH_FIREWALL_DENY QGETH_FIREWALL_STATUS
}

# Detect architecture
detect_architecture() {
    QGETH_ARCH=$(uname -m)
    case "$QGETH_ARCH" in
        x86_64|amd64)
            QGETH_ARCH="amd64"
            QGETH_GO_ARCH="amd64"
            ;;
        i386|i686)
            QGETH_ARCH="386"
            QGETH_GO_ARCH="386"
            ;;
        aarch64|arm64)
            QGETH_ARCH="arm64"
            QGETH_GO_ARCH="arm64"
            ;;
        armv7l|armv6l)
            QGETH_ARCH="arm"
            QGETH_GO_ARCH="arm"
            ;;
        *)
            log_warning "Unsupported architecture: $QGETH_ARCH"
            QGETH_GO_ARCH="$QGETH_ARCH"
            ;;
    esac
    
    log_debug "Detected architecture: $QGETH_ARCH (Go: $QGETH_GO_ARCH)"
    
    export QGETH_ARCH QGETH_GO_ARCH
}

# Get package names for different distributions
get_package_names() {
    local package_type="$1"
    
    case "$package_type" in
        "git")
            case "$QGETH_PKG_MANAGER" in
                apt) echo "git" ;;
                dnf|yum) echo "git" ;;
                pacman) echo "git" ;;
                zypper) echo "git" ;;
                emerge) echo "dev-vcs/git" ;;
                apk) echo "git" ;;
                *) echo "git" ;;
            esac
            ;;
        "curl")
            case "$QGETH_PKG_MANAGER" in
                apt) echo "curl" ;;
                dnf|yum) echo "curl" ;;
                pacman) echo "curl" ;;
                zypper) echo "curl" ;;
                emerge) echo "net-misc/curl" ;;
                apk) echo "curl" ;;
                *) echo "curl" ;;
            esac
            ;;
        "build-tools")
            case "$QGETH_PKG_MANAGER" in
                apt) echo "build-essential" ;;
                dnf|yum) echo "gcc gcc-c++ make" ;;
                pacman) echo "base-devel" ;;
                zypper) echo "patterns-devel-base-devel_basis" ;;
                emerge) echo "sys-devel/gcc sys-devel/make" ;;
                apk) echo "build-base" ;;
                *) echo "gcc make" ;;
            esac
            ;;
        "golang")
            case "$QGETH_PKG_MANAGER" in
                apt) echo "golang-go" ;;
                dnf|yum) echo "golang" ;;
                pacman) echo "go" ;;
                zypper) echo "go" ;;
                emerge) echo "dev-lang/go" ;;
                apk) echo "go" ;;
                *) echo "golang" ;;
            esac
            ;;
        "python")
            case "$QGETH_PKG_MANAGER" in
                apt) echo "python3 python3-pip" ;;
                dnf|yum) echo "python3 python3-pip" ;;
                pacman) echo "python python-pip" ;;
                zypper) echo "python3 python3-pip" ;;
                emerge) echo "dev-lang/python dev-python/pip" ;;
                apk) echo "python3 py3-pip" ;;
                *) echo "python3 python3-pip" ;;
            esac
            ;;
        *)
            echo "$package_type"
            ;;
    esac
}

# Find shell interpreter
detect_shell() {
    QGETH_SHELL=""
    
    # Try different locations for bash
    for shell_path in /bin/bash /usr/bin/bash /usr/local/bin/bash; do
        if [ -x "$shell_path" ]; then
            QGETH_SHELL="$shell_path"
            break
        fi
    done
    
    # Fallback to sh if bash not found
    if [ -z "$QGETH_SHELL" ]; then
        QGETH_SHELL="/bin/sh"
        log_warning "Bash not found, using sh compatibility mode"
    fi
    
    log_debug "Shell interpreter: $QGETH_SHELL"
    export QGETH_SHELL
}

# Check if we have root privileges
check_root() {
    if [ "$EUID" -ne 0 ]; then
        return 1
    fi
    return 0
}

# Get actual user when running with sudo
get_actual_user() {
    if [ -n "$SUDO_USER" ]; then
        echo "$SUDO_USER"
    else
        echo "$USER"
    fi
}

# Get actual user home directory
get_actual_home() {
    local actual_user=$(get_actual_user)
    eval echo "~$actual_user"
}

# Create directory with proper ownership
create_dir_with_ownership() {
    local dir_path="$1"
    local actual_user=$(get_actual_user)
    
    mkdir -p "$dir_path"
    if check_root; then
        chown -R "$actual_user:$actual_user" "$dir_path"
    fi
}

# Main detection function - call this to detect everything
detect_system() {
    log_info "Detecting system configuration..."
    
    detect_os || return 1
    detect_package_manager || return 1
    detect_init_system
    detect_firewall
    detect_architecture
    detect_shell
    
    log_success "System detection completed"
    log_info "OS: $QGETH_OS ($QGETH_DISTRO $QGETH_DISTRO_VERSION)"
    log_info "Package Manager: $QGETH_PKG_MANAGER"
    log_info "Init System: $QGETH_INIT_SYSTEM"
    log_info "Firewall: $QGETH_FIREWALL"
    log_info "Architecture: $QGETH_ARCH"
    log_info "Shell: $QGETH_SHELL"
    
    return 0
}

# Auto-detect when sourced
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script is being executed directly
    detect_system
else
    # Script is being sourced
    detect_system >/dev/null 2>&1
fi 