#!/usr/bin/env pwsh
# Get Q Coin Node Information for Peer Connections
# This script retrieves your local node's enode info so other nodes can connect

Write-Host "🔍 Q COIN NODE INFO RETRIEVAL" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$RPC_URL = "http://127.0.0.1:8545"
$DEV_RPC_URL = "http://127.0.0.1:8545"  # Dev network uses same port

Write-Host "📡 Checking for running Q Coin nodes..." -ForegroundColor Yellow

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
        $response = Invoke-RestMethod -Uri $Url -Method POST -Body $body -ContentType "application/json" -TimeoutSec 5
        return $response.result
    }
    catch {
        return $null
    }
}

# Check for dev network (Chain ID 73234)
Write-Host "🔍 Checking Dev Network (Chain ID 73234)..." -ForegroundColor White
$chainId = Invoke-RPC -Url $DEV_RPC_URL -Method "eth_chainId"
if ($chainId) {
    $chainIdDecimal = [Convert]::ToInt64($chainId, 16)
    Write-Host "   Found network with Chain ID: $chainIdDecimal" -ForegroundColor Green
    
    if ($chainIdDecimal -eq 73234) {
        Write-Host "   ✅ Q Coin Dev Network detected!" -ForegroundColor Green
        $networkName = "Q Coin Dev Network"
        $rpcUrl = $DEV_RPC_URL
    }
    elseif ($chainIdDecimal -eq 73235) {
        Write-Host "   ✅ Q Coin Testnet detected!" -ForegroundColor Green
        $networkName = "Q Coin Testnet"
        $rpcUrl = $DEV_RPC_URL
    }
    elseif ($chainIdDecimal -eq 73236) {
        Write-Host "   ✅ Q Coin Mainnet detected!" -ForegroundColor Green
        $networkName = "Q Coin Mainnet"
        $rpcUrl = $DEV_RPC_URL
    }
    else {
        Write-Host "   ⚠️  Unknown network (Chain ID: $chainIdDecimal)" -ForegroundColor Yellow
        $networkName = "Unknown Q Coin Network"
        $rpcUrl = $DEV_RPC_URL
    }
}
else {
    Write-Host "   ❌ No Q Coin node detected on port 8545" -ForegroundColor Red
    Write-Host ""
    Write-Host "🚀 Start a Q Coin node first:" -ForegroundColor Yellow
    Write-Host "   Dev Network:  .\dev-quick-start.ps1" -ForegroundColor White
    Write-Host "   Testnet:      .\qcoin-geth.ps1" -ForegroundColor White
    Write-Host "   Mainnet:      .\qcoin-geth.ps1 -mainnet" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "🌐 Network: $networkName" -ForegroundColor Cyan
Write-Host ""

# Get node info
Write-Host "📋 Retrieving node information..." -ForegroundColor Yellow

# Get enode info
$enode = Invoke-RPC -Url $rpcUrl -Method "admin_nodeInfo"
if ($enode -and $enode.enode) {
    Write-Host "✅ Node enode retrieved!" -ForegroundColor Green
    
    # Get peer count
    $peerCount = Invoke-RPC -Url $rpcUrl -Method "net_peerCount"
    $peerCountDecimal = if ($peerCount) { [Convert]::ToInt32($peerCount, 16) } else { 0 }
    
    # Get listening status
    $listening = Invoke-RPC -Url $rpcUrl -Method "net_listening"
    
    # Display results
    Write-Host ""
    Write-Host "🔗 NODE CONNECTION INFO" -ForegroundColor Green
    Write-Host "========================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Network:     $networkName" -ForegroundColor White
    Write-Host "Chain ID:    $chainIdDecimal" -ForegroundColor White
    Write-Host "Node ID:     $($enode.id)" -ForegroundColor White
    Write-Host "Listening:   $listening" -ForegroundColor White
    Write-Host "Peers:       $peerCountDecimal connected" -ForegroundColor White
    Write-Host ""
    Write-Host "🌐 ENODE (for remote connections):" -ForegroundColor Cyan
    Write-Host "$($enode.enode)" -ForegroundColor Yellow
    Write-Host ""
    
    # Extract IP and port for convenience
    if ($enode.enode -match "enode://([^@]+)@([^:]+):(\d+)") {
        $nodeId = $matches[1]
        $ip = $matches[2]
        $port = $matches[3]
        
        Write-Host "📡 Connection Details:" -ForegroundColor Cyan
        Write-Host "   Node ID: $nodeId" -ForegroundColor White
        Write-Host "   IP:      $ip" -ForegroundColor White
        Write-Host "   Port:    $port" -ForegroundColor White
        Write-Host ""
        
        # Show commands for remote connection
        Write-Host "🔧 REMOTE CONNECTION COMMANDS" -ForegroundColor Green
        Write-Host "==============================" -ForegroundColor Green
        Write-Host ""
        Write-Host "To connect a remote node to this one, use:" -ForegroundColor Yellow
        Write-Host ""
        
        # Replace localhost/127.0.0.1 with actual IP if needed
        $publicEnode = $enode.enode
        if ($ip -eq "127.0.0.1" -or $ip -eq "localhost") {
            Write-Host "⚠️  Note: Replace 127.0.0.1 with your actual IP address!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Find your IP with: ipconfig (Windows) or ip addr (Linux)" -ForegroundColor Yellow
            Write-Host ""
        }
        
        Write-Host "Windows (add to bootnode):" -ForegroundColor Cyan
        Write-Host "   --bootnodes `"$publicEnode`"" -ForegroundColor White
        Write-Host ""
        Write-Host "Linux (add to bootnode):" -ForegroundColor Cyan
        Write-Host "   --bootnodes '$publicEnode'" -ForegroundColor White
        Write-Host ""
        Write-Host "Or add peer manually via console:" -ForegroundColor Cyan
        Write-Host "   admin.addPeer('$publicEnode')" -ForegroundColor White
        Write-Host ""
    }
    
    # Show current peers if any
    if ($peerCountDecimal -gt 0) {
        Write-Host "👥 CURRENT PEERS" -ForegroundColor Green
        Write-Host "=================" -ForegroundColor Green
        
        $peers = Invoke-RPC -Url $rpcUrl -Method "admin_peers"
        if ($peers) {
            foreach ($peer in $peers) {
                Write-Host ""
                Write-Host "Peer: $($peer.name)" -ForegroundColor White
                Write-Host "   Enode: $($peer.enode)" -ForegroundColor Gray
                Write-Host "   Network: $($peer.network.remoteAddress)" -ForegroundColor Gray
            }
        }
    }
    
}
else {
    Write-Host "❌ Failed to retrieve node information" -ForegroundColor Red
    Write-Host "   Make sure your Q Coin node is running with RPC enabled" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "💡 TIP: Save this enode info to connect other Q Coin nodes!" -ForegroundColor Blue
Write-Host "" 