# Direct Quantum Blockchain Test

# Stop any running geth instances
Stop-Process -Name geth -ErrorAction SilentlyContinue

# Build geth
Set-Location quantum-geth
go build -o ../geth.exe ./cmd/geth
Set-Location ..

# Initialize blockchain
./geth.exe --datadir ./qdata init ./quantum-geth/eth/configs/genesis_qmpow.json

# Run geth with quantum mining
./geth.exe --datadir ./qdata --networkid 9248 --mine --miner.threads 1 --miner.etherbase 0x965e15c0d7fa23fe70d760b380ae60b204f289f2 --password ./qdata/password.txt --unlock 0x965e15c0d7fa23fe70d760b380ae60b204f289f2 --allow-insecure-unlock --qmpow.powmode 1 --qmpow.testmode --verbosity 4 console 