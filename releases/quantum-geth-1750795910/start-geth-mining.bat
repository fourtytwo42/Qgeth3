@echo off
REM Quantum-Geth Mining Node Launcher (Windows Batch)
REM Usage: start-geth-mining.bat [threads] [datadir]

set THREADS=%1
set DATADIR=%2
if "%THREADS%"=="" set THREADS=1
if "%DATADIR%"=="" set DATADIR=qdata

echo Starting Quantum-Geth Mining Node...
echo Data Directory: %DATADIR%
echo Network ID: 1337
echo Mining: ENABLED with %THREADS% threads
echo.

geth.exe --datadir "%DATADIR%" --networkid 1337 --nodiscover --allow-insecure-unlock --http --http.addr "0.0.0.0" --http.port 8545 --http.corsdomain "*" --http.api "eth,net,web3,personal,miner,qmpow,admin" --ws --ws.addr "0.0.0.0" --ws.port 8546 --ws.origins "*" --ws.api "eth,net,web3,personal,miner,qmpow,admin" --mine --miner.threads %THREADS% --miner.etherbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A

pause
