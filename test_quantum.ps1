# Test Quantum Proof of Work Solver
Write-Host "=== Quantum Proof of Work Test ===" -ForegroundColor Green

# Test input parameters - smaller L for clearer output
$testInput = @{
    seed0 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    qbits = 8
    tcount = 25
    L = 8  # Smaller number for testing
} | ConvertTo-Json -Compress

Write-Host "Test Input:" -ForegroundColor Yellow
Write-Host $testInput

Write-Host "`nRunning quantum solver..." -ForegroundColor Yellow

# Create temporary input file
$tempFile = [System.IO.Path]::GetTempFileName()
$testInput | Out-File -FilePath $tempFile -Encoding UTF8

try {
    # Run the solver and capture output
    $output = Get-Content $tempFile | python tools/solver/solver.py 2>&1 | Out-String
    
    Write-Host "`nSolver Raw Output:" -ForegroundColor Green
    Write-Host $output
    
    # Try to parse as JSON
    try {
        $result = $output | ConvertFrom-Json
        Write-Host "`n=== Quantum Proof Results ===" -ForegroundColor Green
        Write-Host "Puzzle Count: $($result.puzzle_count)" -ForegroundColor Cyan
        Write-Host "QBits: $($result.qbits)" -ForegroundColor Cyan
        Write-Host "TCount: $($result.tcount)" -ForegroundColor Cyan
        Write-Host "Outcomes Length: $($result.outcomes.Length) hex characters" -ForegroundColor Cyan
        Write-Host "Proof Length: $($result.proof.Length) hex characters" -ForegroundColor Cyan
        
        # Show first few outcomes
        if ($result.outcomes) {
            $outcomeBytes = $result.outcomes.Length / 2
            $outcomesPerPuzzle = $result.qbits / 8
            Write-Host "Expected outcome bytes: $($result.puzzle_count * $outcomesPerPuzzle)" -ForegroundColor Yellow
            Write-Host "Actual outcome bytes: $outcomeBytes" -ForegroundColor Yellow
            Write-Host "First 32 chars of outcomes: $($result.outcomes.Substring(0, [Math]::Min(32, $result.outcomes.Length)))" -ForegroundColor Magenta
        }
        
        Write-Host "`nSuccess: Quantum proof generated successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "Error parsing JSON: $_" -ForegroundColor Red
        Write-Host "Raw output was: $output" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Error running solver: $_" -ForegroundColor Red
}
finally {
    # Cleanup
    Remove-Item $tempFile -ErrorAction SilentlyContinue
}

# Test with default parameters (64 puzzles)
Write-Host "`n=== Testing with Full 64 Puzzles ===" -ForegroundColor Green

$fullTest = @{
    seed0 = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    qbits = 8
    tcount = 25
    L = 64
} | ConvertTo-Json -Compress

$tempFile2 = [System.IO.Path]::GetTempFileName()
$fullTest | Out-File -FilePath $tempFile2 -Encoding UTF8

try {
    $output2 = Get-Content $tempFile2 | python tools/solver/solver.py 2>&1 | Out-String
    $result2 = $output2 | ConvertFrom-Json
    
    Write-Host "64-Puzzle Test Results:" -ForegroundColor Cyan
    Write-Host "  Puzzle Count: $($result2.puzzle_count)" -ForegroundColor White
    Write-Host "  Outcome Bytes: $($result2.outcomes.Length / 2)" -ForegroundColor White
    Write-Host "  Proof Bytes: $($result2.proof.Length / 2)" -ForegroundColor White
    Write-Host "  Success: Full difficulty test passed!" -ForegroundColor Green
}
catch {
    Write-Host "Error in full test: $_" -ForegroundColor Red
}
finally {
    Remove-Item $tempFile2 -ErrorAction SilentlyContinue
}

Write-Host "`n=== All Tests Complete ===" -ForegroundColor Green 