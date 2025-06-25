#!/usr/bin/env pwsh

Write-Host "Testing RPC methods..."

# Test rpc_modules
$body = @{
    jsonrpc = "2.0"
    method = "rpc_modules"
    params = @()
    id = 1
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $body
    Write-Host "Available RPC modules:"
    $response.result | ConvertTo-Json -Depth 3
} catch {
    Write-Host "Error: $_"
}

# Test eth_getWork
$body2 = @{
    jsonrpc = "2.0"
    method = "eth_getWork"
    params = @()
    id = 2
} | ConvertTo-Json

try {
    $response2 = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $body2
    Write-Host "eth_getWork response:"
    $response2 | ConvertTo-Json -Depth 3
} catch {
    Write-Host "eth_getWork Error: $_"
}

# Test qmpow_getWork
$body3 = @{
    jsonrpc = "2.0"
    method = "qmpow_getWork"
    params = @()
    id = 3
} | ConvertTo-Json

try {
    $response3 = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $body3
    Write-Host "qmpow_getWork response:"
    $response3 | ConvertTo-Json -Depth 3
} catch {
    Write-Host "qmpow_getWork Error: $_"
}

# Test admin_nodeInfo to see what consensus engine is running
$body4 = @{
    jsonrpc = "2.0"
    method = "admin_nodeInfo"
    params = @()
    id = 4
} | ConvertTo-Json

try {
    $response4 = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $body4
    Write-Host "Node info (consensus engine):"
    $response4.result | ConvertTo-Json -Depth 3
} catch {
    Write-Host "admin_nodeInfo Error: $_"
}

# Test miner_start to see if mining can be started
$body5 = @{
    jsonrpc = "2.0"
    method = "miner_start"
    params = @(1)
    id = 5
} | ConvertTo-Json

try {
    $response5 = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $body5
    Write-Host "miner_start response:"
    $response5 | ConvertTo-Json -Depth 3
} catch {
    Write-Host "miner_start Error: $_"
}

# Test eth_mining to see if mining is active
$body6 = @{
    jsonrpc = "2.0"
    method = "eth_mining"
    params = @()
    id = 6
} | ConvertTo-Json

try {
    $response6 = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $body6
    Write-Host "eth_mining response:"
    $response6 | ConvertTo-Json -Depth 3
} catch {
    Write-Host "eth_mining Error: $_"
}

# Try alternative QMPoW method names that might exist
$alternativeMethods = @("qmpow_getQuantumParams", "qmpow_getHashrate", "qmpow_getThreads", "eth_submitWork", "qmpow_submitWork")

foreach ($method in $alternativeMethods) {
    $bodyAlt = @{
        jsonrpc = "2.0"
        method = $method
        params = @()
        id = 99
    } | ConvertTo-Json

    try {
        $responseAlt = Invoke-RestMethod -Uri "http://localhost:8545" -Method Post -ContentType "application/json" -Body $bodyAlt
        Write-Host "$method response:"
        $responseAlt | ConvertTo-Json -Depth 3
    } catch {
        Write-Host "$method Error: $_"
    }
} 