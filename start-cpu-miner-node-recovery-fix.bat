@echo off
title Quantum Miner - Node Recovery Test
echo.
echo ========================================
echo   Q Geth Quantum Miner - Node Recovery Test
echo ========================================
echo.
echo Testing new features:
echo - Solution rate limiting (500ms minimum between submissions)
echo - Quantum-geth node recovery (detects "no mining work available yet")
echo - Intelligent work refresh after solutions
echo.

REM Build the quantum miner
echo Building quantum miner...
cd quantum-miner
go build -o quantum-miner.exe .
if errorlevel 1 (
    echo ERROR: Failed to build quantum miner
    pause
    exit /b 1
)

REM Run the quantum miner with logging
echo.
echo Starting quantum miner with node recovery features...
echo.

quantum-miner.exe ^
    -coinbase 0x742d35Cc6634C0532925a3b8D186aaD9a5C9B9b5 ^
    -url http://127.0.0.1:8545 ^
    -threads 4 ^
    -log quantum-miner-node-recovery.log

echo.
echo Quantum miner stopped.
pause 