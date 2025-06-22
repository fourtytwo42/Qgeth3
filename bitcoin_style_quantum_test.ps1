# Bitcoin-Style Quantum Mining Test
# Fixed 48 puzzles for 1,152-bit security with nonce-based difficulty

Write-Host "🚀 Bitcoin-Style Quantum Mining Test" -ForegroundColor Cyan
Write-Host "📊 Security Level: 1,152-bit (48 puzzles × 24 bits each)" -ForegroundColor Green
Write-Host "⚡ Style: Nonce-based difficulty (like Bitcoin)" -ForegroundColor Green

# Clean up previous data
Write-Host "🧹 Cleaning up previous blockchain data..."
if (Test-Path "qdata") {
    Remove-Item -Recurse -Force "qdata"
}

# Initialize with Bitcoin-style genesis
Write-Host "🌱 Initializing Bitcoin-style quantum blockchain..."
& .\quantum-geth\build\bin\geth.exe --datadir qdata init genesis_qmpow_bitcoin_style.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Genesis initialization failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Genesis initialized successfully" -ForegroundColor Green

# Start mining with Bitcoin-style parameters
Write-Host "⛏️ Starting Bitcoin-style quantum mining..."
Write-Host "🎯 Fixed Parameters:" -ForegroundColor Yellow
Write-Host "   • Puzzles: 48 (fixed, never changes)" -ForegroundColor Yellow
Write-Host "   • QBits: 12 (4,096 quantum states per puzzle)" -ForegroundColor Yellow
Write-Host "   • T-Gates: 4,096 (maximum complexity)" -ForegroundColor Yellow
Write-Host "   • Security: 1,152-bit equivalent" -ForegroundColor Yellow
Write-Host "   • Difficulty: Adjusts through nonce target (Bitcoin-style)" -ForegroundColor Yellow

# Create mining account file
$miningAccount = Get-Content "qdata\mining-account.txt" -Raw
$miningAccount = $miningAccount.Trim()

Write-Host "💰 Mining to account: $miningAccount" -ForegroundColor Green

# Start geth with Bitcoin-style mining
$gethArgs = @(
    "--datadir", "qdata",
    "--networkid", "1337",
    "--mine",
    "--miner.etherbase", $miningAccount,
    "--miner.threads", "1",
    "--unlock", $miningAccount,
    "--password", "qdata\password.txt",
    "--allow-insecure-unlock",
    "--http",
    "--http.api", "eth,net,web3,personal,miner,qmpow",
    "--http.corsdomain", "*",
    "--nodiscover",
    "--maxpeers", "0",
    "--verbosity", "3",
    "--console"
)

Write-Host "🔧 Starting geth with Bitcoin-style quantum mining..." -ForegroundColor Cyan
Write-Host "📝 Command: geth $($gethArgs -join ' ')" -ForegroundColor Gray

& .\quantum-geth\build\bin\geth.exe @gethArgs 