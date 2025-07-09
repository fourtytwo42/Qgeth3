@echo off
echo Q Coin CPU Miner - Fixed Version
echo No panic errors, improved performance
echo.

echo Testing connection to http://localhost:8545...
echo Starting CPU mining: 32 threads
echo.

.\quantum-miner-fixed.exe -coinbase 0x0000000000000000000000000000000000000001 -node http://localhost:8545 -threads 32 -log true 