#!/usr/bin/env pwsh
# Connect to Remote Q Coin Peer
# This script adds a peer to your running Q Coin node

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Enode,
    
    [Parameter(Mandatory=$false)]
    [string]$RpcUrl = "http://127.0.0.1:8545",
    
    [switch]$Help
)

if ($Help) {
    Write-Host "🔗 Q COIN PEER CONNECTION" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Connects your Q Coin node to a remote peer" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor White
    Write-Host "  .\connect-peer.ps1 <enode>" -ForegroundColor White
    Write-Host ""
    Write-Host "Example:" -ForegroundColor White
    Write-Host "  .\connect-peer.ps1 'enode://89df9647...@192.168.1.100:30303'" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -RpcUrl    RPC endpoint (default: http://127.0.0.1:8545)" -ForegroundColor White
    Write-Host "  -Help      Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Get enode info from remote node:" -ForegroundColor Yellow
    Write-Host "  .\get-node-info.ps1" -ForegroundColor White
    Write-Host ""
    exit 0
}

Write-Host "🔗 Q COIN PEER CONNECTION" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Validate enode format
if ($Enode -notmatch "^enode://[a-fA-F0-9]{128}@[\d\.]+:\d+$") {
    Write-Host "❌ Invalid enode format!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected format:" -ForegroundColor Yellow
    Write-Host "  enode://[128-char-hex]@[ip]:[port]" -ForegroundColor White
    Write-Host ""
    Write-Host "Example:" -ForegroundColor Yellow
    Write-Host "  enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@192.168.1.100:30303" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Extract connection details
if ($Enode -match "enode://([^@]+)@([^:]+):(\d+)") {
    $nodeId = $matches[1]
    $ip = $matches[2]
    $port = $matches[3]
    
    Write-Host "🎯 Target Peer:" -ForegroundColor Yellow
    Write-Host "   Node ID: $nodeId" -ForegroundColor White
    Write-Host "   IP:      $ip" -ForegroundColor White
    Write-Host "   Port:    $port" -ForegroundColor White
    Write-Host ""
}

# Function to make RPC call
function Invoke-RPC {
    param(
        [string]$Url,
        [string]$Method,
        [array]$Params = @()
    )
    
    $body = @{
        jsonrpc = "2.0"
        method = $Method
        params = $Params
        id = 1
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri $Url -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
        return $response
    }
    catch {
        Write-Host "❌ RPC call failed: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Check if local node is running
Write-Host "🔍 Checking local Q Coin node..." -ForegroundColor Yellow
$chainId = Invoke-RPC -Url $RpcUrl -Method "eth_chainId"
if (-not $chainId -or -not $chainId.result) {
    Write-Host "❌ No Q Coin node detected at $RpcUrl" -ForegroundColor Red
    Write-Host ""
    Write-Host "Start a Q Coin node first:" -ForegroundColor Yellow
    Write-Host "  Dev Network:  .\dev-quick-start.ps1" -ForegroundColor White
    Write-Host "  Testnet:      .\qcoin-geth.ps1" -ForegroundColor White
    Write-Host "  Mainnet:      .\qcoin-geth.ps1 -mainnet" -ForegroundColor White
    exit 1
}

$chainIdDecimal = [Convert]::ToInt64($chainId.result, 16)
$networkName = switch ($chainIdDecimal) {
    73234 { "Q Coin Dev Network" }
    73235 { "Q Coin Testnet" }
    73236 { "Q Coin Mainnet" }
    default { "Unknown Q Coin Network" }
}

Write-Host "✅ Local node detected: $networkName (Chain ID: $chainIdDecimal)" -ForegroundColor Green
Write-Host ""

# Get current peer count
$peerCountBefore = Invoke-RPC -Url $RpcUrl -Method "net_peerCount"
$peersBefore = if ($peerCountBefore.result) { [Convert]::ToInt32($peerCountBefore.result, 16) } else { 0 }

Write-Host "📊 Current peers: $peersBefore" -ForegroundColor White
Write-Host ""

# Attempt to add peer
Write-Host "🔗 Attempting to connect to peer..." -ForegroundColor Yellow
$addResult = Invoke-RPC -Url $RpcUrl -Method "admin_addPeer" -Params @($Enode)

if ($addResult -and $addResult.result -eq $true) {
    Write-Host "✅ Peer connection initiated successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Wait a moment for connection to establish
    Write-Host "⏳ Waiting for connection to establish..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    # Check new peer count
    $peerCountAfter = Invoke-RPC -Url $RpcUrl -Method "net_peerCount"
    $peersAfter = if ($peerCountAfter.result) { [Convert]::ToInt32($peerCountAfter.result, 16) } else { 0 }
    
    Write-Host "📊 Peer count after connection: $peersAfter" -ForegroundColor White
    
    if ($peersAfter -gt $peersBefore) {
        Write-Host "🎉 New peer connected successfully!" -ForegroundColor Green
        
        # Show current peers
        $peers = Invoke-RPC -Url $RpcUrl -Method "admin_peers"
        if ($peers.result) {
            Write-Host ""
            Write-Host "👥 Current Peers:" -ForegroundColor Cyan
            foreach ($peer in $peers.result) {
                Write-Host "   • $($peer.name)" -ForegroundColor White
                Write-Host "     $($peer.network.remoteAddress)" -ForegroundColor Gray
            }
        }
    }
    else {
        Write-Host "⚠️  Peer added but connection not yet established" -ForegroundColor Yellow
        Write-Host "   This is normal - connections may take time to establish" -ForegroundColor Gray
    }
}
elseif ($addResult -and $addResult.result -eq $false) {
    Write-Host "⚠️  Peer already known or connection failed" -ForegroundColor Yellow
    Write-Host "   The peer may already be in the node table" -ForegroundColor Gray
}
else {
    Write-Host "❌ Failed to add peer" -ForegroundColor Red
    if ($addResult.error) {
        Write-Host "   Error: $($addResult.error.message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "💡 TIP: Use .\get-node-info.ps1 to check connection status" -ForegroundColor Blue
Write-Host "" 