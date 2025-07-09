@echo off
echo Q Coin CPU Miner - Post-Solution Fix Version
echo Intelligent post-solution work switching with exponential backoff
echo.

echo Testing connection to http://localhost:8545...
echo Starting CPU mining: 16 threads (optimized with post-solution fix)
echo.

.\quantum-miner-post-solution-fix.exe -coinbase 0x0000000000000000000000000000000000000001 -node http://localhost:8545 -threads 16 -log true 