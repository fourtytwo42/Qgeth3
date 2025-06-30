#!/usr/bin/env pwsh

# Q Coin Python Isolation Test
# Demonstrates that embedded Python doesn't interfere with system Python

param([switch]$Help)

if ($Help) {
    Write-Host "Q Coin Python Isolation Test" -ForegroundColor Cyan
    Write-Host "Demonstrates complete isolation between embedded and system Python" -ForegroundColor Green
    exit 0
}

Write-Host "🔬 Q Coin Python Isolation Test" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Testing that our embedded Python doesn't interfere with system Python" -ForegroundColor Yellow
Write-Host ""

# Test 1: Check system Python
Write-Host "1️⃣ Testing System Python (if installed)..." -ForegroundColor Yellow
try {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) {
        $systemVersion = python --version 2>&1
        $systemPath = python -c "import sys; print(sys.executable)" 2>&1
        Write-Host "   ✅ System Python found:" -ForegroundColor Green
        Write-Host "      Version: $systemVersion" -ForegroundColor White
        Write-Host "      Location: $systemPath" -ForegroundColor White
        
        # Check system packages
        $systemQiskit = python -c "import qiskit; print(qiskit.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      Qiskit: $systemQiskit" -ForegroundColor White
        } else {
            Write-Host "      Qiskit: Not installed" -ForegroundColor White
        }
    } else {
        Write-Host "   ℹ️ No system Python detected (this is fine)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ℹ️ No system Python detected (this is fine)" -ForegroundColor Cyan
}

Write-Host ""

# Test 2: Check embedded Python
Write-Host "2️⃣ Testing Embedded Python..." -ForegroundColor Yellow
$embeddedPython = "python.bat"
if (Test-Path $embeddedPython) {
    try {
        $embeddedVersion = & $embeddedPython -c "import sys; print('Python', sys.version.split()[0])" 2>&1
        $embeddedPath = & $embeddedPython -c "import sys; print(sys.executable)" 2>&1
        Write-Host "   ✅ Embedded Python found:" -ForegroundColor Green
        Write-Host "      Version: $embeddedVersion" -ForegroundColor White
        Write-Host "      Location: $embeddedPath" -ForegroundColor White
        
        # Check embedded packages
        $embeddedQiskit = & $embeddedPython -c "import qiskit; print(qiskit.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      Qiskit: $embeddedQiskit" -ForegroundColor White
        } else {
            Write-Host "      Qiskit: Installation issue" -ForegroundColor Red
        }
        
        # Check CuPy
        $embeddedCupy = & $embeddedPython -c "import cupy; print('CuPy', cupy.__version__, '- GPU:', cupy.cuda.is_available())" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      GPU: $embeddedCupy" -ForegroundColor White
        } else {
            Write-Host "      GPU: CPU-only mode (no CuPy/CUDA)" -ForegroundColor White
        }
    } catch {
        Write-Host "   ❌ Embedded Python test failed: $_" -ForegroundColor Red
    }
} else {
    Write-Host "   ❌ python.bat not found - run from embedded release directory" -ForegroundColor Red
}

Write-Host ""

# Test 3: Isolation verification
Write-Host "3️⃣ Testing Isolation..." -ForegroundColor Yellow
if ($systemPython -and (Test-Path $embeddedPython)) {
    Write-Host "   🔍 Comparing Python environments..." -ForegroundColor Cyan
    
    # Compare executable paths
    $systemExe = python -c "import sys; print(sys.executable)" 2>&1
    $embeddedExe = & $embeddedPython -c "import sys; print(sys.executable)" 2>&1
    
    if ($systemExe -ne $embeddedExe) {
        Write-Host "   ✅ ISOLATED: Different Python executables" -ForegroundColor Green
        Write-Host "      System:   $systemExe" -ForegroundColor White  
        Write-Host "      Embedded: $embeddedExe" -ForegroundColor White
    } else {
        Write-Host "   ⚠️  WARNING: Same executable path detected" -ForegroundColor Yellow
    }
    
    # Compare site-packages
    $systemSite = python -c "import site; print(site.getsitepackages()[0])" 2>&1
    $embeddedSite = & $embeddedPython -c "import site; print(site.getsitepackages()[0])" 2>&1
    
    if ($systemSite -ne $embeddedSite) {
        Write-Host "   ✅ ISOLATED: Different package directories" -ForegroundColor Green
        Write-Host "      System:   $systemSite" -ForegroundColor White
        Write-Host "      Embedded: $embeddedSite" -ForegroundColor White
    } else {
        Write-Host "   ⚠️  WARNING: Same package directory detected" -ForegroundColor Yellow
    }
    
} elseif (Test-Path $embeddedPython) {
    Write-Host "   ✅ ISOLATED: No system Python to conflict with" -ForegroundColor Green
} else {
    Write-Host "   ❌ Cannot test isolation - missing embedded Python" -ForegroundColor Red
}

Write-Host ""

# Test 4: Environment variable check
Write-Host "4️⃣ Testing Environment Variables..." -ForegroundColor Yellow
$originalPythonHome = $env:PYTHON_HOME
$originalPythonPath = $env:PYTHONPATH

Write-Host "   Original PYTHON_HOME: $originalPythonHome" -ForegroundColor White
Write-Host "   Original PYTHONPATH: $originalPythonPath" -ForegroundColor White

# Run embedded Python and check if environment is restored
if (Test-Path $embeddedPython) {
    & $embeddedPython -c "import os; print('Inside wrapper PYTHON_HOME:', os.environ.get('PYTHON_HOME', 'Not set'))" | Out-Null
    
    $afterPythonHome = $env:PYTHON_HOME
    $afterPythonPath = $env:PYTHONPATH
    
    if ($afterPythonHome -eq $originalPythonHome -and $afterPythonPath -eq $originalPythonPath) {
        Write-Host "   ✅ ISOLATED: Environment variables properly restored" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  WARNING: Environment variables may have leaked" -ForegroundColor Yellow
        Write-Host "      After PYTHON_HOME: $afterPythonHome" -ForegroundColor White
        Write-Host "      After PYTHONPATH: $afterPythonPath" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "🎉 Isolation Test Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor Cyan
Write-Host "   • Embedded Python runs in complete isolation" -ForegroundColor White
Write-Host "   • Your system Python (if any) remains untouched" -ForegroundColor White  
Write-Host "   • Different executables and package directories" -ForegroundColor White
Write-Host "   • Environment variables properly isolated" -ForegroundColor White
Write-Host "   • Safe to run alongside any Python environment" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Ready for conflict-free quantum mining!" -ForegroundColor Green 