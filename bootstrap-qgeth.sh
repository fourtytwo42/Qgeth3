#!/bin/bash
# Q Geth Bootstrap Script Redirect
# This script redirects to the new simplified bootstrap
# Usage: curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[BOOTSTRAP]${NC} $1"
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
    echo "For non-interactive mode:"
    echo "  curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/bootstrap-qgeth.sh | sudo bash -s -- -y"
    exit 1
fi

print_step "🚀 Q Geth Bootstrap (Simplified Version)"
echo ""
echo "Redirecting to the new simplified bootstrap script..."
echo ""

# Download and execute the new simplified bootstrap
if command -v curl >/dev/null 2>&1; then
    curl -sSL https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | bash -s -- "$@"
elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://raw.githubusercontent.com/fourtytwo42/Qgeth3/main/scripts/deployment/bootstrap-qgeth.sh | bash -s -- "$@"
else
    print_error "Neither curl nor wget is available. Please install one of them."
    exit 1
fi 