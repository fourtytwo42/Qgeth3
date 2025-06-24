@echo off
REM Quantum-Miner GPU Launcher (Windows Batch)
REM Usage: start-miner-gpu.bat <coinbase> [threads] [gpu_id] [node_url]

set COINBASE=%1
set THREADS=%2
set GPU_ID=%3
set NODE_URL=%4
if "%THREADS%"=="" set THREADS=1
if "%GPU_ID%"=="" set GPU_ID=0
if "%NODE_URL%"=="" set NODE_URL=http://localhost:8545

if "%COINBASE%"=="" (
    echo ERROR: Coinbase address required!
    echo Usage: start-miner-gpu.bat 0xYourAddress [threads] [gpu_id] [node_url]
    pause
    exit /b 1
)

echo Starting Quantum-Miner (GPU Mode)...
echo Coinbase: %COINBASE%
echo Threads: %THREADS%
echo GPU ID: %GPU_ID%
echo Node URL: %NODE_URL%
echo.

quantum-miner.exe -gpu -coinbase "%COINBASE%" -threads %THREADS% -gpu-id %GPU_ID% -node "%NODE_URL%"

pause
