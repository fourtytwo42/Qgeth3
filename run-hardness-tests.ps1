#!/usr/bin/env powershell
# QPoW Hardness and Security Test Suite Runner
# Comprehensive testing of all attack vectors and security assumptions

param(
    [string]$TestCategory = "all",
    [switch]$Verbose,
    [switch]$FailFast,
    [string]$OutputFormat = "console"
)

# Test configuration
$TestRoot = "tests/hardness"
$TestResults = @()
$StartTime = Get-Date

Write-Host "QPoW Hardness and Security Test Suite" -ForegroundColor Cyan
Write-Host "Testing quantum proof-of-work security assumptions" -ForegroundColor Yellow
Write-Host "Test Category: $TestCategory" -ForegroundColor Green
Write-Host ""

# Test categories
$TestCategories = @{
    "puzzle" = @{
        "name" = "Quantum Puzzle Execution"
        "tests" = @("PZ-01", "PZ-02", "PZ-03")
        "description" = "Deterministic execution, branch-serial enforcement, nonce wraparound"
    }
    "gatehash" = @{
        "name" = "Canonical-Compile and GateHash"
        "tests" = @("GH-01", "GH-02", "GH-03")
        "description" = "QASM compilation, Z-mask application, collision resistance"
    }
    "merkle" = @{
        "name" = "OutcomeRoot and BranchNibbles"
        "tests" = @("MR-01", "MR-02", "MR-03")
        "description" = "Merkle tree consistency, nibble extraction, proof verification"
    }
    "proofs" = @{
        "name" = "Proof Generation and Verification"
        "tests" = @("PR-01", "PR-02", "PR-03", "PR-04")
        "description" = "Mahadev traces, CAPSS SNARKs, Nova aggregation, ProofRoot binding"
    }
    "quality" = @{
        "name" = "ProofQuality and Target Test"
        "tests" = @("PQ-01", "PQ-02", "PQ-03", "PQ-04")
        "description" = "Quality computation, boundary conditions, difficulty validation"
    }
}

function Write-TestHeader($category, $info) {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor DarkGray
    Write-Host $info.name -ForegroundColor Cyan
    Write-Host $info.description -ForegroundColor Gray
    Write-Host "============================================================================" -ForegroundColor DarkGray
}

function Run-Test($testId, $testName, $category) {
    $testStart = Get-Date
    
    try {
        # Run the specific test
        $testScript = "$TestRoot/$category/$testId.go"
        
        if (Test-Path $testScript) {
            Write-Host "Testing $testId : $testName" -NoNewline
            
            # Run Go test
            $result = & go test -run $testId "$TestRoot/$category" -v 2>&1
            $exitCode = $LASTEXITCODE
            
            $duration = (Get-Date) - $testStart
            
            if ($exitCode -eq 0) {
                Write-Host " PASS" -ForegroundColor Green
                Write-Host "   Duration: $($duration.TotalMilliseconds)ms" -ForegroundColor Gray
                
                $script:TestResults += @{
                    TestId = $testId
                    Name = $testName
                    Category = $category
                    Status = "PASS"
                    Duration = $duration.TotalMilliseconds
                    Output = $result
                }
                
                return $true
            } else {
                Write-Host " FAIL" -ForegroundColor Red
                Write-Host "   Duration: $($duration.TotalMilliseconds)ms" -ForegroundColor Gray
                if ($Verbose) {
                    Write-Host "   Error: $result" -ForegroundColor Red
                }
                
                $script:TestResults += @{
                    TestId = $testId
                    Name = $testName
                    Category = $category
                    Status = "FAIL"
                    Duration = $duration.TotalMilliseconds
                    Output = $result
                }
                
                if ($FailFast) {
                    throw "Test $testId failed, stopping due to -FailFast"
                }
                
                return $false
            }
        } else {
            Write-Host "Testing $testId : $testName" -NoNewline
            Write-Host " SKIP (Not implemented)" -ForegroundColor Yellow
            
            $script:TestResults += @{
                TestId = $testId
                Name = $testName
                Category = $category
                Status = "SKIP"
                Duration = 0
                Output = "Test not implemented"
            }
            
            return $true
        }
    } catch {
        Write-Host " ERROR" -ForegroundColor Magenta
        Write-Host "   Exception: $($_.Exception.Message)" -ForegroundColor Red
        
        $script:TestResults += @{
            TestId = $testId
            Name = $testName
            Category = $category
            Status = "ERROR"
            Duration = 0
            Output = $_.Exception.Message
        }
        
        return $false
    }
}

function Show-TestSummary {
    $endTime = Get-Date
    $totalDuration = $endTime - $StartTime
    
    $totalTests = $TestResults.Count
    $passedTests = ($TestResults | Where-Object { $_.Status -eq "PASS" }).Count
    $failedTests = ($TestResults | Where-Object { $_.Status -eq "FAIL" }).Count
    $skippedTests = ($TestResults | Where-Object { $_.Status -eq "SKIP" }).Count
    $errorTests = ($TestResults | Where-Object { $_.Status -eq "ERROR" }).Count
    
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor DarkGray
    Write-Host "QPoW Hardness and Security Test Results" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor DarkGray
    
    Write-Host "Total Tests:   $totalTests" -ForegroundColor White
    Write-Host "Passed:        $passedTests" -ForegroundColor Green
    Write-Host "Failed:        $failedTests" -ForegroundColor Red
    Write-Host "Skipped:       $skippedTests" -ForegroundColor Yellow
    Write-Host "Errors:        $errorTests" -ForegroundColor Magenta
    Write-Host "Total Time:    $($totalDuration.TotalSeconds)s" -ForegroundColor Gray
    
    $successRate = if ($totalTests -gt 0) { [math]::Round(($passedTests / $totalTests) * 100, 1) } else { 0 }
    $successColor = if ($successRate -eq 100) { "Green" } elseif ($successRate -ge 80) { "Yellow" } else { "Red" }
    Write-Host "Success Rate:  $successRate%" -ForegroundColor $successColor
    
    # Category breakdown
    Write-Host ""
    Write-Host "Category Breakdown:" -ForegroundColor Cyan
    
    foreach ($category in $TestCategories.Keys) {
        $categoryResults = $TestResults | Where-Object { $_.Category -eq $category }
        $categoryPassed = ($categoryResults | Where-Object { $_.Status -eq "PASS" }).Count
        $categoryTotal = $categoryResults.Count
        
        if ($categoryTotal -gt 0) {
            $categoryRate = [math]::Round(($categoryPassed / $categoryTotal) * 100, 1)
            $statusColor = if ($categoryRate -eq 100) { "Green" } elseif ($categoryRate -ge 80) { "Yellow" } else { "Red" }
            Write-Host "   $category : $categoryPassed/$categoryTotal ($categoryRate%)" -ForegroundColor $statusColor
        }
    }
    
    # Failed tests detail
    if ($failedTests -gt 0) {
        Write-Host ""
        Write-Host "Failed Tests:" -ForegroundColor Red
        $TestResults | Where-Object { $_.Status -eq "FAIL" } | ForEach-Object {
            Write-Host "   $($_.TestId): $($_.Name)" -ForegroundColor Red
            if ($Verbose) {
                Write-Host "      $($_.Output)" -ForegroundColor DarkRed
            }
        }
    }
    
    Write-Host ""
    
    # Exit with appropriate code
    if ($failedTests -gt 0 -or $errorTests -gt 0) {
        Write-Host "Some tests failed. QPoW security assumptions may be violated!" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "All tests passed! QPoW security assumptions are solid." -ForegroundColor Green
        exit 0
    }
}

# Ensure test directories exist
if (!(Test-Path $TestRoot)) {
    New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
}

# Main test execution
try {
    if ($TestCategory -eq "all") {
        foreach ($category in $TestCategories.Keys) {
            $categoryInfo = $TestCategories[$category]
            Write-TestHeader $category $categoryInfo
            
            foreach ($testId in $categoryInfo.tests) {
                $testName = "Test implementation for $testId"
                Run-Test $testId $testName $category | Out-Null
            }
        }
    } else {
        if ($TestCategories.ContainsKey($TestCategory)) {
            $categoryInfo = $TestCategories[$TestCategory]
            Write-TestHeader $TestCategory $categoryInfo
            
            foreach ($testId in $categoryInfo.tests) {
                $testName = "Test implementation for $testId"
                Run-Test $testId $testName $TestCategory | Out-Null
            }
        } else {
            Write-Host "Unknown test category: $TestCategory" -ForegroundColor Red
            Write-Host "Available categories: $($TestCategories.Keys -join ', ')" -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "Test suite execution failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Show-TestSummary
} 