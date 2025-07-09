@echo off
REM Setup WSL2 environment for Quantum GPU Mining

echo Setting up WSL2 environment for Quantum GPU Mining...
echo.

REM Check if WSL2 is available
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL2 is not available on this system
    echo Please install WSL2 from Microsoft Store or enable Windows Subsystem for Linux
    pause
    exit /b 1
)

REM Check if a WSL2 distribution is installed
wsl --list --quiet | findstr /r "." >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: No WSL2 distribution found
    echo Please install Ubuntu or another Linux distribution from Microsoft Store
    pause
    exit /b 1
)

echo ✅ WSL2 is available

REM Copy the Python wrapper script to WSL2
echo Copying Python wrapper script to WSL2...
wsl mkdir -p /tmp/qgeth-wsl2
wsl cp /mnt/c/Users/%USERNAME%/OneDrive/Documents/GitHub/Qgeth3/go-wsl2/python-linux.sh /tmp/qgeth-wsl2/
wsl chmod +x /tmp/qgeth-wsl2/python-linux.sh

REM Install Python and dependencies in WSL2
echo Installing Python and dependencies in WSL2...
wsl sudo apt update
wsl sudo apt install -y python3 python3-pip python3-venv

REM Install Qiskit and GPU dependencies
echo Installing Qiskit and GPU dependencies...
wsl python3 -m pip install --user qiskit qiskit-aer cuquantum-python cupy-cuda12x numpy

REM Test GPU acceleration
echo Testing GPU acceleration...
wsl python3 -c "import qiskit; print('✅ Qiskit available in WSL2')"

REM Set up environment variables for the session
echo Setting up environment variables...
set WSL2_MODE=true
set PYTHON_EXEC=wsl /tmp/qgeth-wsl2/python-linux.sh

echo.
echo ✅ WSL2 setup complete!
echo.
echo To use WSL2 GPU mining, run:
echo quantum-miner.exe -gpu -coinbase 0xYourAddress
echo.
echo Environment variables have been set for this session:
echo WSL2_MODE=%WSL2_MODE%
echo PYTHON_EXEC=%PYTHON_EXEC%
echo.

pause 