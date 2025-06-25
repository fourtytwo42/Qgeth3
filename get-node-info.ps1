# Get Q Coin Node Information
Write-Host "Q COIN NODE INFO" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan

$RPC_URL = "http://127.0.0.1:8545"

function Invoke-RPC {
    param([string]$Method, [array]$Params = @())
    $body = @{
        jsonrpc = "2.0"
        method = $Method
        params = $Params
        id = 1
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri $RPC_URL -Method POST -Body $body -ContentType "application/json" -TimeoutSec 5
        return $response.result
    }
    catch {
        return $null
    }
}

$chainId = Invoke-RPC -Method "eth_chainId"
if (-not $chainId) {
    Write-Host "No Q Coin node detected on port 8545" -ForegroundColor Red
    Write-Host "Start a Q Coin node first" -ForegroundColor Yellow
    exit 1
}

$chainIdDecimal = [Convert]::ToInt64($chainId, 16)
$networkName = switch ($chainIdDecimal) {
    73234 { "Q Coin Dev Network" }
    73235 { "Q Coin Testnet" }
    73236 { "Q Coin Mainnet" }
    default { "Unknown Q Coin Network" }
}

Write-Host "Network: $networkName (Chain ID: $chainIdDecimal)" -ForegroundColor Green

$enode = Invoke-RPC -Method "admin_nodeInfo"
if (-not $enode -or -not $enode.enode) {
    Write-Host "Failed to retrieve node information" -ForegroundColor Red
    exit 1
}

$peerCount = Invoke-RPC -Method "net_peerCount"
$peerCountDecimal = if ($peerCount) { [Convert]::ToInt32($peerCount, 16) } else { 0 }

Write-Host ""
Write-Host "NODE CONNECTION INFO" -ForegroundColor Green
Write-Host "===================" -ForegroundColor Green
Write-Host "Node ID: $($enode.id)" -ForegroundColor White
Write-Host "Peers: $peerCountDecimal connected" -ForegroundColor White
Write-Host ""
Write-Host "ENODE (for remote connections):" -ForegroundColor Cyan
Write-Host "$($enode.enode)" -ForegroundColor Yellow
Write-Host ""
Write-Host "To connect a remote node, use:" -ForegroundColor Yellow
Write-Host ".\connect-peer.ps1 'your-enode-here'" -ForegroundColor White
