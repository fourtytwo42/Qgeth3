# Simple Bitcoin-Style Quantum Mining Test
# Fixed 48 puzzles for 1,152-bit security with nonce-based difficulty

Write-Host "*** Simple Bitcoin-Style Quantum Mining Test ***" -ForegroundColor Cyan
Write-Host "Security Level: 1,152-bit (48 puzzles x 24 bits each)" -ForegroundColor Green
Write-Host "Style: Nonce-based difficulty (like Bitcoin)" -ForegroundColor Green

# Clean up previous data
Write-Host "Cleaning up previous blockchain data..."
if (Test-Path "qdata") {
    Remove-Item -Recurse -Force "qdata"
}

# Initialize with Bitcoin-style genesis
Write-Host "Initializing Bitcoin-style quantum blockchain..."
& .\quantum-geth\build\bin\geth.exe --datadir qdata init genesis_qmpow_bitcoin_style.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "Genesis initialization failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Genesis initialized successfully" -ForegroundColor Green

# Display configuration
Write-Host "Starting Bitcoin-style quantum mining (dev mode)..." -ForegroundColor Yellow
Write-Host "Fixed Parameters:" -ForegroundColor Cyan
Write-Host "   * Puzzles: 48 (fixed, never changes)" -ForegroundColor White
Write-Host "   * QBits: 12 (4,096 quantum states per puzzle)" -ForegroundColor White
Write-Host "   * T-Gates: 4,096 (maximum complexity)" -ForegroundColor White
Write-Host "   * Security: 1,152-bit equivalent" -ForegroundColor White
Write-Host "   * Difficulty: Adjusts through nonce target (Bitcoin-style)" -ForegroundColor White

# Start geth with mining
Write-Host "Starting geth with Bitcoin-style quantum mining..." -ForegroundColor Green

# Use one of the pre-funded addresses as etherbase
$etherbase = "0x965e15c0d7fa23fe70d760b380ae60b204f289f2"

$gethArgs = @(
    "--datadir", "qdata",
    "--networkid", "1337",
    "--mine",
    "--miner.threads", "1",
    "--miner.etherbase", $etherbase,
    "--http",
    "--http.api", "eth,net,web3,miner",
    "--http.corsdomain", "*",
    "--nodiscover",
    "--maxpeers", "0",
    "--verbosity", "4",
    "console"
)

Write-Host "Command: geth $($gethArgs -join ' ')" -ForegroundColor Gray
Write-Host ""
Write-Host "Expected behavior:" -ForegroundColor Yellow
Write-Host "- Should show QMPoW consensus engine initialization" -ForegroundColor White
Write-Host "- Mining should start automatically" -ForegroundColor White
Write-Host "- Each block should show 48 fixed puzzles being solved" -ForegroundColor White
Write-Host "- Difficulty should adjust through nonce target, not puzzle count" -ForegroundColor White
Write-Host ""

& .\quantum-geth\build\bin\geth.exe @gethArgs 