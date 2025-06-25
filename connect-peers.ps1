# Q Coin Peer Connection Tool
# Usage: ./connect-peers.ps1 [enode] [options]

param(
    [Parameter(Position=0)]
    [string]$Enode = "",
    
    [string]$GethRpc = "http://localhost:8545",
    [switch]$List,
    [switch]$Help
)

if ($Help) {
    Write-Host "Q Coin Peer Connection Tool" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: ./connect-peers.ps1 [enode] [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Arguments:" -ForegroundColor Yellow
    Write-Host "  enode             - Enode URL to connect to"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -gethRpc <url>    - Geth RPC endpoint (default: http://localhost:8545)"
    Write-Host "  -list             - List current peers"
    Write-Host "  -help             - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ./connect-peers.ps1 -list                                    # List current peers"
    Write-Host "  ./connect-peers.ps1 'enode://abc123@192.168.1.100:30305'    # Connect to peer"
    Write-Host ""
    Write-Host "Default Bootnode:" -ForegroundColor Magenta
    Write-Host "  enode://0bc243936ebc13ebf57895dff1321695064ae4b0ac0c1e047d52d695c396b64c52847f852a9738f0d079af4ba109dfceafd1cf0924587b151765834caf13e5fd@69.243.132.233:30305"
    exit 0
}

function Test-GethConnection {
    param([string]$rpcUrl)
    
    try {
        $response = Invoke-RestMethod -Uri $rpcUrl -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"net_version","params":[],"id":1}' -TimeoutSec 5 -ErrorAction Stop
        return $response.result -ne $null
    } catch {
        return $false
    }
}

function Get-PeerInfo {
    param([string]$rpcUrl)
    
    try {
        $response = Invoke-RestMethod -Uri $rpcUrl -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"jsonrpc":"2.0","method":"admin_peers","params":[],"id":1}' -ErrorAction Stop
        return $response.result
    } catch {
        Write-Host "❌ Failed to get peer information: $_" -ForegroundColor Red
        return $null
    }
}

function Add-Peer {
    param([string]$rpcUrl, [string]$enode)
    
    $body = @{
        jsonrpc = "2.0"
        method = "admin_addPeer"
        params = @($enode)
        id = 1
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri $rpcUrl -Method POST -Headers @{"Content-Type"="application/json"} -Body $body -ErrorAction Stop
        return $response.result
    } catch {
        Write-Host "❌ Failed to add peer: $_" -ForegroundColor Red
        return $false
    }
}

Write-Host "🔗 Q Coin Peer Connection Tool" -ForegroundColor Cyan
Write-Host ""

# Test Geth connection
Write-Host "📡 Testing connection to Geth RPC..." -ForegroundColor Yellow
if (-not (Test-GethConnection $GethRpc)) {
    Write-Host "❌ Cannot connect to Geth RPC at $GethRpc" -ForegroundColor Red
    Write-Host "   Make sure Geth is running and RPC is enabled!" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Connected to Geth RPC" -ForegroundColor Green
Write-Host ""

# List peers if requested
if ($List -or $Enode -eq "") {
    Write-Host "📋 Current Peers:" -ForegroundColor Cyan
    
    $peers = Get-PeerInfo $GethRpc
    if ($peers -and $peers.Count -gt 0) {
        for ($i = 0; $i -lt $peers.Count; $i++) {
            $peer = $peers[$i]
            Write-Host "  [$($i + 1)] $($peer.network.remoteAddress)" -ForegroundColor White
            Write-Host "      Enode: $($peer.enode)" -ForegroundColor Gray
            Write-Host "      Name: $($peer.name)" -ForegroundColor Gray
            if ($peer.network.inbound) {
                Write-Host "      Direction: Inbound" -ForegroundColor Green
            } else {
                Write-Host "      Direction: Outbound" -ForegroundColor Yellow
            }
            Write-Host ""
        }
    } else {
        Write-Host "  No peers connected" -ForegroundColor Gray
    }
    
    if ($Enode -eq "") {
        exit 0
    }
}

# Connect to peer if enode provided
if ($Enode -ne "") {
    Write-Host "🔗 Connecting to peer..." -ForegroundColor Yellow
    Write-Host "   Enode: $Enode" -ForegroundColor Gray
    Write-Host ""
    
    # Validate enode format
    if ($Enode -notmatch "^enode://[0-9a-fA-F]{128}@[\d\.:]+:\d+$") {
        Write-Host "❌ Invalid enode format!" -ForegroundColor Red
        Write-Host "   Expected format: enode://pubkey@ip:port" -ForegroundColor Yellow
        exit 1
    }
    
    $result = Add-Peer $GethRpc $Enode
    if ($result) {
        Write-Host "✅ Peer connection request sent successfully!" -ForegroundColor Green
        Write-Host "   Note: It may take a few moments to establish the connection." -ForegroundColor Yellow
    } else {
        Write-Host "❌ Failed to send peer connection request!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "🔍 Checking connection status in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep 5
    
    $peers = Get-PeerInfo $GethRpc
    $peerIp = ($Enode -split "@")[1].Split(":")[0]
    $connected = $false
    
    if ($peers) {
        foreach ($peer in $peers) {
            if ($peer.network.remoteAddress -match $peerIp) {
                Write-Host "✅ Successfully connected to peer!" -ForegroundColor Green
                Write-Host "   Remote Address: $($peer.network.remoteAddress)" -ForegroundColor White
                Write-Host "   Name: $($peer.name)" -ForegroundColor White
                $connected = $true
                break
            }
        }
    }
    
    if (-not $connected) {
        Write-Host "⚠️  Peer not yet connected. This is normal and may take more time." -ForegroundColor Yellow
        Write-Host "   Use './connect-peers.ps1 -list' to check connection status later." -ForegroundColor Yellow
    }
} 