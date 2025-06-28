# Enhanced Geth startup script for better peer synchronization
# This helps resolve sync timeout issues when connecting to advanced peers

Write-Host "🚀 Starting Qgeth3 with enhanced sync settings..." -ForegroundColor Green

# Kill any existing geth processes
Get-Process geth -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Start geth with better sync configuration
.\geth.exe `
    --datadir "qdata" `
    --genesis "genesis_quantum_dev.json" `
    --networkid 73234 `
    --http `
    --http.addr "0.0.0.0" `
    --http.port 8545 `
    --http.api "eth,net,web3,personal,admin,txpool,miner,qmpow" `
    --http.corsdomain "*" `
    --ws `
    --ws.addr "0.0.0.0" `
    --ws.port 8546 `
    --ws.api "eth,net,web3,personal,admin,txpool,miner,qmpow" `
    --ws.origins "*" `
    --port 30305 `
    --discovery.port 30305 `
    --nat "extip:69.243.132.233" `
    --mine `
    --miner.threads -1 `
    --miner.etherbase "0x0000000000000000000000000000000000000001" `
    --unlock "0x0000000000000000000000000000000000000001" `
    --password /dev/null `
    --allow-insecure-unlock `
    --syncmode "full" `
    --gcmode "archive" `
    --cache 2048 `
    --maxpeers 50 `
    --netrestrict "0.0.0.0/0" `
    --bootnodes "enode://4600db1dae9e2e9285542a5fb4e231adc17967c175e3d332e7258d3f53cea552b01ac844415de9b5b9262daa28377b0e55a0bc66c2a49974325e381247b0c253@143.110.231.183:30305" `
    --verbosity 3 `
    --rpc.evmtimeout 30s `
    2>&1 | Tee-Object -FilePath "geth-sync.log"

Write-Host "📋 Geth started. Check geth-sync.log for detailed logs." -ForegroundColor Green
Write-Host "🌐 Connect to console: .\geth.exe attach http://localhost:8545" -ForegroundColor Yellow 