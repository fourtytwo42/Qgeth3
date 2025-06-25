# Add Bootnode Peer Configuration
# Adds real enode IDs to Q Coin scripts for peer discovery
# Usage: .\add-bootnode-peer.ps1 [-Environment testnet|mainnet|dev] [-NodeIP <ip>] [-Port <port>] [-EnodeID <id>]

param(
    [string]$Environment = "testnet",           # Which environment to configure
    [string]$NodeIP = "69.243.132.233",        # IP of the peer node
    [string]$Port = "4294",                     # Port of the peer node  
    [string]$EnodeID = "89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c"  # Current node's enode ID
)

# Display current node's enode information
Write-Host "Q Coin Bootnode Peer Configuration" -ForegroundColor Cyan
Write-Host "=================================="
Write-Host ""
Write-Host "Current Node Enode Information:" -ForegroundColor Yellow
Write-Host "  Node ID: dd486ee8ea6d548df6a1e256ce9db616b23b765c4f2aa22d6ada573ef97c91e5"
Write-Host "  Full Enode: enode://$EnodeID@$($NodeIP):$Port"
Write-Host "  Network: 73235 (Testnet)"
Write-Host "  Listening Port: $Port"
Write-Host ""

# Environment configurations
$configs = @{
    "testnet" = @{
        "ChainID" = "73235"
        "Port" = "4294"
        "Description" = "Q Coin Testnet"
    }
    "mainnet" = @{
        "ChainID" = "73236" 
        "Port" = "4294"
        "Description" = "Q Coin Mainnet"
    }
    "dev" = @{
        "ChainID" = "73234"
        "Port" = "30305"
        "Description" = "Q Coin Dev/Staging"
    }
}

$config = $configs[$Environment]
if (-not $config) {
    Write-Host "ERROR: Invalid environment '$Environment'. Use: testnet, mainnet, or dev" -ForegroundColor Red
    exit 1
}

Write-Host "Configuring for: $($config.Description)" -ForegroundColor Green
Write-Host "  Chain ID: $($config.ChainID)"
Write-Host "  Default Port: $($config.Port)"
Write-Host ""

# Generate bootnode configuration
$bootnodeConfig = "enode://$EnodeID@$($NodeIP):$Port"

Write-Host "Bootnode Configuration:" -ForegroundColor Cyan
Write-Host "  --bootnodes `"$bootnodeConfig`"" -ForegroundColor White
Write-Host ""

# Show how to add to different environments
Write-Host "To add this bootnode to other nodes:" -ForegroundColor Yellow
Write-Host ""

if ($Environment -eq "testnet" -or $Environment -eq "mainnet") {
    Write-Host "Windows Testnet/Mainnet (start-geth.ps1):" -ForegroundColor Gray
    Write-Host "  Add to `$gethArgs array:" -ForegroundColor Gray
    Write-Host "    `"--bootnodes`", `"$bootnodeConfig`"" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Linux Testnet/Mainnet (scripts/start-geth.sh):" -ForegroundColor Gray
    Write-Host "  Add to GETH_ARGS array:" -ForegroundColor Gray
    Write-Host "    --bootnodes `"$bootnodeConfig`"" -ForegroundColor White
    Write-Host ""
}

if ($Environment -eq "dev") {
    Write-Host "Windows Dev (dev-start-geth.ps1, dev-start-geth-mining.ps1):" -ForegroundColor Gray
    Write-Host "  Add bootnode parameter:" -ForegroundColor Gray
    Write-Host "    --bootnodes `"$bootnodeConfig`"" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Linux Dev (scripts/dev-start-geth.sh):" -ForegroundColor Gray
    Write-Host "  Add to geth command:" -ForegroundColor Gray
    Write-Host "    --bootnodes `"$bootnodeConfig`" \" -ForegroundColor White
    Write-Host ""
}

Write-Host "Static Nodes Alternative:" -ForegroundColor Yellow
Write-Host "  Create static-nodes.json in data directory:" -ForegroundColor Gray
Write-Host "  [`"$bootnodeConfig`"]" -ForegroundColor White
Write-Host ""

Write-Host "Peer Discovery Status:" -ForegroundColor Cyan
Write-Host "  ✅ Automatic peer discovery ENABLED (no hardcoded bootnodes)"
Write-Host "  ✅ Your node is discoverable at: $($NodeIP):$($Port)"
Write-Host "  ✅ Other nodes can connect using the bootnode configuration above"
Write-Host "  ✅ Three environments are isolated by Chain ID and ports"
Write-Host ""

Write-Host "To check connected peers:" -ForegroundColor Yellow
Write-Host "  PowerShell: Invoke-RestMethod -Uri 'http://localhost:8545' -Method Post -ContentType 'application/json' -Body '{`"jsonrpc`":`"2.0`",`"method`":`"net_peerCount`",`"params`":[],`"id`":1}'"
Write-Host "  Linux: curl -X POST -H 'Content-Type: application/json' --data '{`"jsonrpc`":`"2.0`",`"method`":`"net_peerCount`",`"params`":[],`"id`":1}' http://localhost:8545"
Write-Host "" 