# Stop the running geth instance
Write-Host "Stopping the quantum blockchain..."
Stop-Process -Name geth -ErrorAction SilentlyContinue
Write-Host "Blockchain stopped!" 