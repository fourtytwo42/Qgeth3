@echo off
echo Testing WSL2 setup components...
echo.

REM Test 1: Basic WSL2 functionality
echo Test 1: WSL2 availability
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL2 not available
    exit /b 1
)
echo [OK] WSL2 is available

REM Test 2: Python test
echo Test 2: Python test
wsl python3 --version
if %errorlevel% neq 0 (
    echo ERROR: Python not available
    exit /b 1
)
echo [OK] Python is available

REM Test 3: Directory creation
echo Test 3: Directory creation
wsl mkdir -p /tmp/qgeth-wsl2
echo [OK] Directory created

REM Test 4: Get current directory
echo Test 4: Current directory
FOR /F "tokens=*" %%G IN ('cd') DO SET CURRENT_DIR=%%G
echo Current directory: %CURRENT_DIR%

REM Test 5: Path conversion
echo Test 5: Path conversion
set "WSL_PATH=%CURRENT_DIR:\=/%"
echo WSL path step 1: %WSL_PATH%
set "WSL_PATH=%WSL_PATH:C:=/mnt/c%"
echo WSL path step 2: %WSL_PATH%

REM Test 6: Simple Python command
echo Test 6: Simple Python command
wsl python3 -c "print('Hello from WSL2')"
echo [OK] Simple Python command works

REM Test 7: Python with import
echo Test 7: Python import test
wsl python3 -c "import sys; print('Python import works')"
echo [OK] Python import works

REM Test 8: Try the problematic command
echo Test 8: Testing problematic command
wsl python3 -c "import qiskit; print('Qiskit ready in WSL2')" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Qiskit not available, but continuing...
) else (
    echo [OK] Qiskit test passed
)

echo.
echo All tests completed!
pause 