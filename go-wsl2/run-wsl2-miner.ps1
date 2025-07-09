# WSL2 Quantum GPU Miner Launcher
param(
    [string]$Coinbase = "",
    [string]$NodeURL = "http://localhost:8545",
    [int]$Threads = 2,
    [int]$GpuId = 0,
    [switch]$Help
)

if ($Help) {
    Write-Host "WSL2 Quantum GPU Miner Launcher" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Description:" -ForegroundColor Yellow
    Write-Host "  Launches quantum miner with WSL2 GPU acceleration for maximum performance" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run-wsl2-miner.ps1 -Coinbase <address> [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -Coinbase <address>    Coinbase address for mining rewards (required)" -ForegroundColor White
    Write-Host "  -NodeURL <url>         Quantum-Geth node URL (default: http://localhost:8545)" -ForegroundColor White
    Write-Host "  -Threads <number>      Number of mining threads (default: 2)" -ForegroundColor White
    Write-Host "  -GpuId <number>        GPU device ID (default: 0)" -ForegroundColor White
    Write-Host "  -Help                  Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run-wsl2-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor Green
    Write-Host "  .\run-wsl2-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -Threads 4" -ForegroundColor Green
    Write-Host ""
    Write-Host "Requirements:" -ForegroundColor Yellow
    Write-Host "  * WSL2 installed with Linux distribution" -ForegroundColor White
    Write-Host "  * NVIDIA GPU with CUDA support" -ForegroundColor White
    Write-Host "  * Run setup-wsl2.bat first to install dependencies" -ForegroundColor White
    Write-Host ""
    exit 0
}

Write-Host "WSL2 Quantum GPU Miner" -ForegroundColor Cyan
Write-Host "Linux GPU acceleration via WSL2" -ForegroundColor Magenta
Write-Host ""

if ($Coinbase -eq "") {
    Write-Host "ERROR: Coinbase address required!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage examples:" -ForegroundColor Yellow
    Write-Host "  .\run-wsl2-miner.ps1 -Coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A" -ForegroundColor White
    Write-Host "  .\run-wsl2-miner.ps1 -Help" -ForegroundColor White
    exit 1
}

if ($Coinbase -notmatch "^0x[0-9a-fA-F]{40}$") {
    Write-Host "ERROR: Invalid coinbase address format!" -ForegroundColor Red
    Write-Host "Expected format: 0x followed by 40 hex characters" -ForegroundColor Yellow
    exit 1
}

# Check if WSL2 is available
$wslCheck = wsl --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: WSL2 is not available on this system" -ForegroundColor Red
    Write-Host "Please install WSL2 from Microsoft Store or enable Windows Subsystem for Linux" -ForegroundColor Yellow
    exit 1
}

# Check if quantum miner executable exists
$MinerExecutable = "..\quantum-miner-wsl2.exe"
if (-not (Test-Path $MinerExecutable)) {
    Write-Host "ERROR: Quantum miner WSL2 executable not found: $MinerExecutable" -ForegroundColor Red
    Write-Host "Please build the quantum miner first with WSL2 support." -ForegroundColor Yellow
    exit 1
}

# Set up WSL2 environment variables
$env:WSL2_MODE = "true"
$env:PYTHON_EXEC = "wsl /tmp/qgeth-wsl2/python-linux.sh"

Write-Host "WSL2 GPU Mining Configuration:" -ForegroundColor Green
Write-Host "   Coinbase: $Coinbase" -ForegroundColor White
Write-Host "   Node URL: $NodeURL" -ForegroundColor White
Write-Host "   GPU Device: $GpuId" -ForegroundColor White
Write-Host "   Threads: $Threads" -ForegroundColor White
Write-Host "   WSL2 Mode: $($env:WSL2_MODE)" -ForegroundColor White
Write-Host "   Python Exec: $($env:PYTHON_EXEC)" -ForegroundColor White
Write-Host ""

# Test WSL2 Python setup
Write-Host "Testing WSL2 Python setup..." -ForegroundColor Yellow
$testResult = wsl /tmp/qgeth-wsl2/python-linux.sh -c "import qiskit; print('✅ Qiskit available')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: WSL2 Python setup test failed" -ForegroundColor Yellow
    Write-Host "Run setup-wsl2.bat first to install dependencies" -ForegroundColor Yellow
    Write-Host "Continuing anyway..." -ForegroundColor Yellow
} else {
    Write-Host $testResult -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting WSL2 quantum miner..." -ForegroundColor Blue
Write-Host "Note: First run may take longer while initializing GPU backend" -ForegroundColor Yellow
Write-Host ""

# Build command arguments
$MinerArgs = @(
    "-gpu",
    "-coinbase", $Coinbase,
    "-node", $NodeURL,
    "-threads", $Threads,
    "-gpu-id", $GpuId
)

try {
    # Run the WSL2 quantum miner executable
    & $MinerExecutable @MinerArgs
} catch {
    Write-Host "ERROR: Failed to start WSL2 miner: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Run setup-wsl2.bat to install WSL2 dependencies" -ForegroundColor White
    Write-Host "  2. Ensure WSL2 is installed and Ubuntu distribution is available" -ForegroundColor White
    Write-Host "  3. Check that GPU drivers support WSL2" -ForegroundColor White
    Write-Host "  4. Verify quantum-geth node is running at $NodeURL" -ForegroundColor White
    Write-Host "  5. Check coinbase address format" -ForegroundColor White
    exit 1
} 