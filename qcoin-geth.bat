@echo off
REM Q Coin Geth Wrapper for Windows
REM This wrapper ensures geth ALWAYS uses Q Coin networks, never Ethereum
REM Default: Q Coin Testnet (Chain ID 73235)
REM Use -qcoin-mainnet for Q Coin Mainnet (Chain ID 73236)

setlocal EnableDelayedExpansion

REM Find the actual geth executable
set REAL_GETH=
if exist "geth.exe" (
    set REAL_GETH=geth.exe
) else if exist "releases\quantum-geth-*\geth.exe" (
    for /f %%i in ('dir /b /od "releases\quantum-geth-*\geth.exe" 2^>nul') do set REAL_GETH=releases\%%i
) else (
    echo ❌ ERROR: Q Coin geth binary not found!
    echo    Build it first: .\build-release.ps1
    exit /b 1
)

REM Parse arguments
set USE_QCOIN_MAINNET=false
set FILTERED_ARGS=

:parse_args
if "%1"=="" goto done_parsing
if "%1"=="-qcoin-mainnet" (
    set USE_QCOIN_MAINNET=true
    shift
    goto parse_args
)
if "%1"=="--help" goto show_help
if "%1"=="-h" goto show_help

set FILTERED_ARGS=!FILTERED_ARGS! %1
shift
goto parse_args

:show_help
echo Q Coin Geth - Quantum Blockchain Node
echo.
echo This geth ONLY connects to Q Coin networks, never Ethereum!
echo.
echo Q Coin Networks:
echo   Default:           Q Coin Testnet (Chain ID 73235)
echo   -qcoin-mainnet     Q Coin Mainnet (Chain ID 73236)
echo.
echo Quick Start:
echo   .\qcoin-geth.ps1           # Easy testnet startup
echo   .\qcoin-geth.ps1 -mainnet  # Easy mainnet startup
echo.
echo Manual Usage:
echo   .\qcoin-geth.bat --datadir qdata init genesis_quantum_testnet.json
echo   .\qcoin-geth.bat --datadir qdata --networkid 73235 --mine --miner.threads 0
echo.
echo Standard geth options also available.
exit /b 0

:done_parsing

REM Check if this is a bare geth call (likely trying to connect to Ethereum)
echo !FILTERED_ARGS! | findstr /C:"--networkid" >nul
if errorlevel 1 (
    echo !FILTERED_ARGS! | findstr /C:"--datadir" >nul
    if errorlevel 1 (
        if "!FILTERED_ARGS!"=="" (
            echo 🚫 Q Coin Geth: Prevented connection to Ethereum mainnet!
            echo.
            echo This geth is configured for Q Coin networks only.
            echo.
            echo Quick Start:
            echo   .\qcoin-geth.ps1           # Q Coin Testnet
            echo   .\qcoin-geth.ps1 -mainnet  # Q Coin Mainnet
            echo.
            echo Manual Start:
            if "!USE_QCOIN_MAINNET!"=="true" (
                echo   .\qcoin-geth.bat --datadir %%APPDATA%%\Qcoin\mainnet --networkid 73236 init genesis_quantum_mainnet.json
                echo   .\qcoin-geth.bat --datadir %%APPDATA%%\Qcoin\mainnet --networkid 73236 --mine --miner.threads 0
            ) else (
                echo   .\qcoin-geth.bat --datadir qdata --networkid 73235 init genesis_quantum_testnet.json
                echo   .\qcoin-geth.bat --datadir qdata --networkid 73235 --mine --miner.threads 0
            )
            echo.
            echo Use --help for more options.
            exit /b 1
        )
    )
)

REM Add Q Coin network defaults if not specified
echo !FILTERED_ARGS! | findstr /C:"--networkid" >nul
if errorlevel 1 (
    if "!USE_QCOIN_MAINNET!"=="true" (
        set FILTERED_ARGS=!FILTERED_ARGS! --networkid 73236
    ) else (
        set FILTERED_ARGS=!FILTERED_ARGS! --networkid 73235
    )
)

REM Execute the real geth with filtered arguments
"!REAL_GETH!" !FILTERED_ARGS! 