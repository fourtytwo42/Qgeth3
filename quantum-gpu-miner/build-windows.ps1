# Build script for Quantum GPU Miner (Windows)
# Builds the quantum-gpu-miner executable with GPU support

Write-Host "🚀 Building Quantum GPU Miner for Windows..." -ForegroundColor Green

# Set build parameters
$env:GOOS="windows"
$env:GOARCH="amd64"
$env:CGO_ENABLED="1"  # Enable CGO for potential CUDA support

# Build flags for optimization
$buildFlags = @(
    "-ldflags", "-s -w -H windowsgui",
    "-trimpath",
    "-o", "quantum-gpu-miner.exe"
)

Write-Host "📋 Build Configuration:" -ForegroundColor Cyan
Write-Host "   Target: $env:GOOS/$env:GOARCH" -ForegroundColor White
Write-Host "   CGO: $env:CGO_ENABLED" -ForegroundColor White
Write-Host "   Output: quantum-gpu-miner.exe" -ForegroundColor White
Write-Host ""

# Run the build
Write-Host "🔨 Compiling..." -ForegroundColor Yellow
try {
    go build @buildFlags
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Build successful!" -ForegroundColor Green
        
        # Check file size
        $fileInfo = Get-Item "quantum-gpu-miner.exe"
        $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
        Write-Host "📦 Output size: $sizeMB MB" -ForegroundColor Cyan
        
        Write-Host ""
        Write-Host "🎮 GPU Mining Features:" -ForegroundColor Magenta
        Write-Host "   • CPU mining (default)" -ForegroundColor White
        Write-Host "   • GPU mining with -gpu flag" -ForegroundColor White
        Write-Host "   • CUDA/Qiskit acceleration support" -ForegroundColor White
        Write-Host "   • 10x faster quantum circuit execution" -ForegroundColor White
        Write-Host ""
        Write-Host "Usage examples:" -ForegroundColor Yellow
        Write-Host "   .\quantum-gpu-miner.exe -coinbase 0xYourAddress" -ForegroundColor White
        Write-Host "   .\quantum-gpu-miner.exe -coinbase 0xYourAddress -gpu" -ForegroundColor White
        Write-Host "   .\quantum-gpu-miner.exe -coinbase 0xYourAddress -gpu -gpu-id 1" -ForegroundColor White
    }
    else {
        Write-Host "❌ Build failed!" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ Build error: $_" -ForegroundColor Red
    exit 1
}
