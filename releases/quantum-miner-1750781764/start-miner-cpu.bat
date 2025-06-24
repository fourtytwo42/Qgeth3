@echo off
REM Quantum-Miner CPU Launcher (Windows Batch)
REM Usage: start-miner-cpu.bat <coinbase> [threads] [node_url]

set COINBASE=%1
set THREADS=%2
set NODE_URL=%3
if "%THREADS%"=="" set THREADS=1
if "%NODE_URL%"=="" set NODE_URL=http://localhost:8545

if "%COINBASE%"=="" (
    echo ERROR: Coinbase address required!
    echo Usage: start-miner-cpu.bat 0xYourAddress [threads] [node_url]
    pause
    exit /b 1
)

echo Starting Quantum-Miner (CPU Mode)...
echo Coinbase: %COINBASE%
echo Threads: %THREADS%
echo Node URL: %NODE_URL%
echo.

quantum-miner.exe -coinbase "%COINBASE%" -threads %THREADS% -node "%NODE_URL%"

pause
