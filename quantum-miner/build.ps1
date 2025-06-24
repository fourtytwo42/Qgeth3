#!/usr/bin/env pwsh
# Quantum-GPU-Miner Build Script for Windows
# Usage: .\build.ps1 [cpu|gpu|both] [clean]

param(
    [string]$Mode = "cpu",
    [switch]$Clean,
    [switch]$Help
)

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue
$White = [System.ConsoleColor]::White

function Write-ColorOutput($Message, $Color) {
    $host.UI.RawUI.ForegroundColor = $Color
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $White
}

function Show-Help {
    Write-ColorOutput "Quantum-GPU-Miner Build Script" $Blue
    Write-Output ""
    Write-Output "Usage: .\build.ps1 [mode] [options]"
    Write-Output ""
    Write-Output "Modes:"
    Write-Output "  cpu    - Build CPU-only version (default)"
    Write-Output "  gpu    - Build GPU-accelerated version (requires CUDA)"
    Write-Output "  both   - Build both CPU and GPU versions"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -Clean - Clean build artifacts before building"
    Write-Output "  -Help  - Show this help message"
    Write-Output ""
    Write-Output "Examples:"
    Write-Output "  .\build.ps1 cpu"
    Write-Output "  .\build.ps1 gpu -Clean"
    Write-Output "  .\build.ps1 both"
    Write-Output ""
    Write-Output "Requirements:"
    Write-Output "  CPU Mode: Go 1.19+, Python 3.8+, Qiskit"
    Write-Output "  GPU Mode: CUDA Toolkit 12.0+, Visual Studio Build Tools"
}

function Test-Prerequisites {
    Write-ColorOutput "Checking prerequisites..." $Blue
    
    # Check Go
    $goVersion = go version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Error: Go not found. Please install Go 1.19+" $Red
        return $false
    }
    Write-ColorOutput "Success: Go: $goVersion" $Green
    
    # Check Python
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Error: Python not found. Please install Python 3.8+" $Red
        return $false
    }
    Write-ColorOutput "Success: Python: $pythonVersion" $Green
    
    # Check Qiskit
    try {
        $qiskitCheck = python -c "import qiskit; print('Qiskit installed')" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Warning: Qiskit not found. Installing..." $Yellow
            pip install qiskit qiskit-aer numpy cupy-cuda12x
            if ($LASTEXITCODE -ne 0) {
                Write-ColorOutput "Error: Failed to install Qiskit dependencies" $Red
                return $false
            }
        } else {
            Write-ColorOutput "Success: Qiskit is available" $Green
        }
    } catch {
        Write-ColorOutput "Warning: Could not check Qiskit. Attempting install..." $Yellow
        pip install qiskit qiskit-aer numpy cupy-cuda12x
    }
    
    return $true
}

function Test-CUDAPrerequisites {
    Write-ColorOutput "Checking CUDA prerequisites..." $Blue
    
    # Check NVCC
    $nvccVersion = nvcc --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Error: NVCC not found. Please install CUDA Toolkit 12.0+" $Red
        return $false
    }
    Write-ColorOutput "Success: NVCC found" $Green
    
    # Check Visual Studio Build Tools
    $vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vsPath)) {
        $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        if (-not (Test-Path $vsPath)) {
            Write-ColorOutput "Error: Visual Studio Build Tools not found" $Red
            Write-ColorOutput "   Please install Visual Studio Build Tools 2022" $Red
            return $false
        }
    }
    Write-ColorOutput "Success: Visual Studio Build Tools found" $Green
    
    return $true
}

function Clean-BuildArtifacts {
    Write-ColorOutput "Cleaning build artifacts..." $Blue
    
    # Remove executables
    Remove-Item -Path "quantum-miner.exe" -Force -ErrorAction SilentlyContinue
    
    # Remove CUDA libraries
    Remove-Item -Path "pkg\quantum\*.dll" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "pkg\quantum\*.so" -Force -ErrorAction SilentlyContinue
    
    # Clean Go cache
    go clean -cache
    
    Write-ColorOutput "Success: Build artifacts cleaned" $Green
}

function Build-CPUVersion {
    Write-ColorOutput "Building quantum-gpu-miner (CPU/GPU capable)..." $Blue
    
    $env:CGO_ENABLED = "1"
    go build -ldflags "-s -w" -o quantum-miner.exe .
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "Success: Quantum miner built successfully: quantum-miner.exe" $Green
        Write-ColorOutput "Note: This executable supports both CPU and GPU modes" $Blue
        
        # Test the build
        $version = .\quantum-gpu-miner.exe --version
        Write-ColorOutput "Version: $version" $Blue
        return $true
    } else {
        Write-ColorOutput "Error: Build failed" $Red
        return $false
    }
}

function Build-CUDALibrary {
    Write-ColorOutput "Building CUDA library..." $Blue
    
    # Initialize Visual Studio environment
    $vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vsPath)) {
        $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    }
    
    Push-Location "pkg\quantum"
    
    # Compile CUDA library using cmd with VS environment
    $cudaCommand = "`"$vsPath`" `&`& nvcc -shared -O3 -o libquantum_cuda.dll quantum_cuda.cu"
    $result = cmd /c $cudaCommand
    
    Pop-Location
    
    if (Test-Path "pkg\quantum\libquantum_cuda.dll") {
        Write-ColorOutput "Success: CUDA library built successfully" $Green
        return $true
    } else {
        Write-ColorOutput "Error: CUDA library build failed" $Red
        Write-ColorOutput "   Output: $result" $Red
        return $false
    }
}

function Build-GPUVersion {
    Write-ColorOutput "Building quantum-gpu-miner with CUDA support..." $Blue
    
    # Build CUDA library first
    if (-not (Build-CUDALibrary)) {
        return $false
    }
    
    # Build Go application with CUDA tags
    $env:CGO_ENABLED = "1"
    go build -tags cuda -ldflags "-s -w" -o quantum-miner.exe .
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "Success: Quantum miner with CUDA built successfully: quantum-miner.exe" $Green
        Write-ColorOutput "Note: This executable supports both CPU and GPU modes" $Blue
        
        # Test the build
        $version = .\quantum-gpu-miner.exe --version
        Write-ColorOutput "Version: $version" $Blue
        return $true
    } else {
        Write-ColorOutput "Error: GPU build failed" $Red
        return $false
    }
}

function Main {
    Write-ColorOutput "Quantum-GPU-Miner Build Script" $Blue
    Write-Output ""
    
    if ($Help) {
        Show-Help
        return
    }
    
    if ($Clean) {
        Clean-BuildArtifacts
    }
    
    # Check basic prerequisites
    if (-not (Test-Prerequisites)) {
        Write-ColorOutput "Error: Prerequisites not met" $Red
        exit 1
    }
    
    $success = $true
    
    switch ($Mode.ToLower()) {
        "cpu" {
            $success = Build-CPUVersion
        }
        "gpu" {
            if (-not (Test-CUDAPrerequisites)) {
                Write-ColorOutput "Error: CUDA prerequisites not met" $Red
                exit 1
            }
            $success = Build-GPUVersion
        }
        "both" {
            $success = Build-CPUVersion
            if ($success) {
                if (Test-CUDAPrerequisites) {
                    $success = Build-GPUVersion
                } else {
                    Write-ColorOutput "Warning: Skipping GPU build - CUDA prerequisites not met" $Yellow
                }
            }
        }
        default {
            Write-ColorOutput "Error: Invalid mode: $Mode" $Red
            Show-Help
            exit 1
        }
    }
    
    Write-Output ""
    if ($success) {
        Write-ColorOutput "Success: Build completed successfully!" $Green
        Write-Output ""
        Write-ColorOutput "Usage Examples:" $Blue
        Write-Output "  .\quantum-gpu-miner-cpu.exe -node http://localhost:8545 -threads 4"
        if (Test-Path "quantum-gpu-miner-gpu.exe") {
            Write-Output "  .\quantum-gpu-miner-gpu.exe -gpu -node http://localhost:8545 -threads 2"
        }
    } else {
        Write-ColorOutput "Error: Build failed!" $Red
        exit 1
    }
}

# Run main function
Main 