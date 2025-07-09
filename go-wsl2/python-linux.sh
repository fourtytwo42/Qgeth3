#!/bin/bash

# WSL2 Python wrapper script for Quantum GPU Mining
# This script executes Python commands inside WSL2 with GPU acceleration

# Check if we're inside WSL2
if [ -z "$WSL_DISTRO_NAME" ]; then
    echo "Error: This script must be run inside WSL2" >&2
    exit 1
fi

# Set CUDA environment variables for GPU access
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_PATH=/tmp/cuda_cache
export NUMBA_CACHE_DIR=/tmp/numba_cache

# Create cache directories if they don't exist
mkdir -p /tmp/cuda_cache /tmp/numba_cache

# Default Python executable
PYTHON_EXEC="/usr/bin/python3"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed in WSL2" >&2
    echo "Please install Python3 in WSL2: sudo apt update && sudo apt install python3 python3-pip" >&2
    exit 1
fi

# Use the specified Python executable or default
if [ -n "$PYTHON_EXEC" ]; then
    if [ -f "$PYTHON_EXEC" ]; then
        PYTHON_EXEC="$PYTHON_EXEC"
    else
        echo "Warning: Specified Python executable not found: $PYTHON_EXEC" >&2
        echo "Using default: /usr/bin/python3" >&2
        PYTHON_EXEC="/usr/bin/python3"
    fi
fi

# Check if Qiskit is installed
if ! $PYTHON_EXEC -c "import qiskit" &> /dev/null; then
    echo "Warning: Qiskit not found in WSL2 Python environment" >&2
    echo "Installing Qiskit and dependencies..." >&2
    
    # Install Qiskit and GPU dependencies
    $PYTHON_EXEC -m pip install --user qiskit qiskit-aer cuquantum-python cupy-cuda12x numpy
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Qiskit in WSL2" >&2
        exit 1
    fi
    
    echo "✅ Qiskit installed successfully in WSL2" >&2
fi

# Execute the Python command with all arguments
exec $PYTHON_EXEC "$@" 