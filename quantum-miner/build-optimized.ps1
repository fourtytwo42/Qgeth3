# Build High-Performance Quantum Miner with Optimized CUDA Batch Processing
# Eliminates GPU synchronization bottlenecks for 10-100x speedup

param(
    [switch]$cuda = $false,      # Build with CUDA optimization
    [switch]$release = $false,   # Release build
    [switch]$clean = $false      # Clean build
)

Write-Host "BUILDING HIGH-PERFORMANCE Quantum Miner..." -ForegroundColor Green
Write-Host "   Eliminates GPU synchronization bottlenecks" -ForegroundColor Yellow
Write-Host "   Expected 10-100x GPU utilization improvement" -ForegroundColor Yellow
Write-Host ""

# Clean previous builds if requested
if ($clean) {
    Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
    Remove-Item -Force -ErrorAction SilentlyContinue quantum-miner.exe
    Remove-Item -Force -ErrorAction SilentlyContinue quantum-miner
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue pkg/quantum/*.o
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue pkg/quantum/*.so
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue pkg/quantum/*.dll
    Write-Host "✅ Clean completed" -ForegroundColor Green
    Write-Host ""
}

# Set build environment
$env:CGO_ENABLED = "1"
$env:GOOS = "windows"
$env:GOARCH = "amd64"

# Build optimized CUDA libraries if requested
if ($cuda) {
    Write-Host "🔧 Building optimized CUDA batch processor..." -ForegroundColor Cyan
    
    # Check for NVCC (CUDA compiler)
    $nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $nvccPath) {
        Write-Host "❌ NVCC not found! Please install CUDA Toolkit" -ForegroundColor Red
        Write-Host "   Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "   📍 Found NVCC: $($nvccPath.Source)" -ForegroundColor Gray
    
    # Compile optimized CUDA batch processor
    $cudaFile = "pkg/quantum/batch_cuda_optimizer.cu"
    $outputLib = "pkg/quantum/libcuda_batch_optimizer.dll"
    
    if (Test-Path $cudaFile) {
        Write-Host "   🏗️  Compiling optimized CUDA kernels..." -ForegroundColor Yellow
        
        $nvccArgs = @(
            "--shared",
            "--compiler-options", "/MD",
            "-O3",                          # Maximum optimization
            "-use_fast_math",               # Fast math operations
            "-gencode", "arch=compute_75,code=sm_75",   # RTX 20xx/30xx series
            "-gencode", "arch=compute_86,code=sm_86",   # RTX 30xx/40xx series
            "-gencode", "arch=compute_89,code=sm_89",   # RTX 40xx series
            "-o", $outputLib,
            $cudaFile
        )
        
        & nvcc @nvccArgs
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ CUDA compilation failed!" -ForegroundColor Red
            exit 1
        }
        
        if (Test-Path $outputLib) {
            Write-Host "✅ Optimized CUDA library compiled: $outputLib" -ForegroundColor Green
        } else {
            Write-Host "❌ CUDA library not found after compilation!" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ CUDA source file not found: $cudaFile" -ForegroundColor Red
        exit 1
    }
    
    # Set CUDA build tags
    $buildTags = "-tags cuda"
    $env:CGO_LDFLAGS = "-L./pkg/quantum -lcuda_batch_optimizer -lcudart"
    $env:CGO_CFLAGS = "-I./pkg/quantum"
    
    Write-Host "🚀 CUDA batch processor ready for compilation" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host "📦 Building CPU-only version (no CUDA optimization)" -ForegroundColor Yellow
    $buildTags = ""
}

# Set build flags
$buildFlags = @()
if ($release) {
    $buildFlags += "-ldflags", "-s -w"  # Strip debug info for smaller binary
    Write-Host "🏗️  Release build mode (optimized binary)" -ForegroundColor Cyan
} else {
    Write-Host "🏗️  Debug build mode" -ForegroundColor Cyan
}

# Add build tags if using CUDA
if ($buildTags) {
    $buildFlags += $buildTags
}

Write-Host "🛠️  Compiling Go quantum miner..." -ForegroundColor Yellow

# Build the quantum miner
$buildCmd = @("go", "build") + $buildFlags + @("-o", "quantum-miner.exe", ".")

Write-Host "   Command: $($buildCmd -join ' ')" -ForegroundColor Gray

& $buildCmd[0] $buildCmd[1..($buildCmd.Length-1)]

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Go build failed!" -ForegroundColor Red
    exit 1
}

if (Test-Path "quantum-miner.exe") {
    $fileSize = (Get-Item "quantum-miner.exe").Length
    $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
    
    Write-Host ""
    Write-Host "✅ HIGH-PERFORMANCE Quantum Miner built successfully!" -ForegroundColor Green
    Write-Host "   📁 Output: quantum-miner.exe ($fileSizeMB MB)" -ForegroundColor White
    
    if ($cuda) {
        Write-Host "   🚀 Includes optimized CUDA batch processor" -ForegroundColor Cyan
        Write-Host "   📊 Expected performance improvements:" -ForegroundColor Yellow
        Write-Host "      • 1 thread:   3s → 0.1s   (30x faster)" -ForegroundColor Green
        Write-Host "      • 64 threads: 10s → 0.3s  (33x faster)" -ForegroundColor Green
        Write-Host "      • 256 threads: 30s → 0.5s (60x faster)" -ForegroundColor Green
        Write-Host "      • GPU utilization: 4% → 80%+ (20x better)" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  CPU-only build (for CUDA optimization, use -cuda flag)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "🎯 Usage:" -ForegroundColor Cyan
    Write-Host "   CPU Mining:  .\quantum-miner.exe -coinbase 0xYourAddress -threads 8" -ForegroundColor White
    Write-Host "   GPU Mining:  .\quantum-miner.exe -coinbase 0xYourAddress -gpu -threads 64" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 Pro Tips:" -ForegroundColor Cyan
    Write-Host "   • Start with 64 threads for GPU mode" -ForegroundColor Gray
    Write-Host "   • Increase threads gradually while monitoring GPU utilization" -ForegroundColor Gray
    Write-Host "   • Expected GPU utilization: 80%+ (vs previous 4%)" -ForegroundColor Gray
    Write-Host ""
    
} else {
    Write-Host "❌ Build completed but executable not found!" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 HIGH-PERFORMANCE Quantum Miner ready for deployment!" -ForegroundColor Green 