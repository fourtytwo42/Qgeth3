@echo off
echo Q Coin CPU Miner - Stale Work Fix Version
echo Ultra-aggressive work change detection
echo.

echo Testing connection to http://localhost:8545...
echo Starting CPU mining: 16 threads (optimized)
echo.

.\quantum-miner-stale-fix.exe -coinbase 0x0000000000000000000000000000000000000001 -node http://localhost:8545 -threads 16 -log true 