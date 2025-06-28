package main

import (
	"context"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/accounts/keystore"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rpc"
	
	// Import quantum-geth components for embedded node
	"github.com/fourtytwo42/Qgeth3/quantum-wallet-embedded/node"
	"github.com/fourtytwo42/Qgeth3/quantum-wallet-embedded/eth"
	"github.com/fourtytwo42/Qgeth3/quantum-wallet-embedded/eth/ethconfig"
	"github.com/fourtytwo42/Qgeth3/quantum-wallet-embedded/core"
	"github.com/fourtytwo42/Qgeth3/quantum-wallet-embedded/params"
)

// WalletService handles all quantum blockchain operations with embedded geth node
type WalletService struct {
	mu          sync.RWMutex
	keystore    *keystore.KeyStore
	client      *ethclient.Client
	rpcClient   *rpc.Client
	accounts    []accounts.Account
	dataDir     string
	networkInfo *NetworkInfo
	isConnected bool
	isMining    bool
	
	// Embedded geth node components
	node        *node.Node
	ethereum    *eth.Ethereum
	config      *ethconfig.Config
	nodeConfig  *node.Config
}

// Account represents a wallet account
type Account struct {
	Address  string  `json:"address"`
	Balance  string  `json:"balance"`
	Name     string  `json:"name"`
	IsLocked bool    `json:"isLocked"`
}

// Transaction represents a blockchain transaction
type Transaction struct {
	Hash        string `json:"hash"`
	From        string `json:"from"`
	To          string `json:"to"`
	Value       string `json:"value"`
	Gas         uint64 `json:"gas"`
	GasPrice    string `json:"gasPrice"`
	Status      string `json:"status"`
	BlockNumber uint64 `json:"blockNumber"`
	Timestamp   int64  `json:"timestamp"`
	Type        string `json:"type"`
}

// NetworkInfo contains network status information
type NetworkInfo struct {
	ChainID      string `json:"chainId"`
	NetworkName  string `json:"networkName"`
	BlockNumber  uint64 `json:"blockNumber"`
	PeerCount    uint64 `json:"peerCount"`
	Syncing      bool   `json:"syncing"`
	GasPrice     string `json:"gasPrice"`
	Difficulty   string `json:"difficulty"`
	HashRate     string `json:"hashRate"`
}

// MiningInfo contains mining status and configuration
type MiningInfo struct {
	IsMining     bool   `json:"isMining"`
	HashRate     string `json:"hashRate"`
	MinerAddress string `json:"minerAddress"`
	BlocksMined  uint64 `json:"blocksMined"`
}

// WalletConfig contains wallet configuration
type WalletConfig struct {
	DataDir        string `json:"dataDir"`
	Network        string `json:"network"`
	AutoMining     bool   `json:"autoMining"`
	MinerAddress   string `json:"minerAddress"`
	ExternalMiner  bool   `json:"externalMiner"`
	MinerEndpoint  string `json:"minerEndpoint"`
	HTTPPort       int    `json:"httpPort"`
	WSPort         int    `json:"wsPort"`
	EnableLogging  bool   `json:"enableLogging"`
	LogLevel       string `json:"logLevel"`
}

// NewWalletService creates a new wallet service with embedded geth node
func NewWalletService() *WalletService {
	homeDir, _ := os.UserHomeDir()
	dataDir := filepath.Join(homeDir, ".quantum-wallet")
	
	return &WalletService{
		dataDir: dataDir,
		networkInfo: &NetworkInfo{
			NetworkName: "Quantum Testnet",
		},
	}
}

// Initialize sets up the wallet service and starts embedded geth node
func (w *WalletService) Initialize() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	log.Info("Initializing Quantum Wallet Service...")

	// Create data directory
	if err := os.MkdirAll(w.dataDir, 0755); err != nil {
		log.Error("Failed to create data directory", "error", err)
		return fmt.Errorf("failed to create data directory: %v", err)
	}
	log.Info("Data directory created", "path", w.dataDir)

	// Initialize keystore
	keystoreDir := filepath.Join(w.dataDir, "keystore")
	w.keystore = keystore.NewKeyStore(keystoreDir, keystore.StandardScryptN, keystore.StandardScryptP)
	log.Info("Keystore initialized", "path", keystoreDir)

	// Load existing accounts
	if err := w.loadAccounts(); err != nil {
		log.Warn("Failed to load accounts", "error", err)
	} else {
		log.Info("Accounts loaded", "count", len(w.accounts))
	}

	// Start embedded Q Geth node
	log.Info("Starting embedded Q Geth node...")
	if err := w.startEmbeddedNode(); err != nil {
		log.Error("Failed to start embedded Q Geth node", "error", err)
		return fmt.Errorf("failed to start embedded geth node: %v", err)
	}

	log.Info("Quantum Wallet Service initialization completed")
	return nil
}

// startEmbeddedNode starts the embedded Q Geth node
func (w *WalletService) startEmbeddedNode() error {
	log.Info("Configuring embedded Q Geth node...")

	// Configure the embedded node for Quantum Testnet
	w.nodeConfig = &node.Config{
		Name:             "QuantumWallet",
		UserIdent:        "quantum-wallet-embedded",
		Version:          "1.0.0",
		DataDir:          filepath.Join(w.dataDir, "chaindata"),
		IPCPath:          "geth.ipc",
		HTTPHost:         "127.0.0.1",
		HTTPPort:         8545,
		HTTPModules:      []string{"eth", "net", "web3", "personal", "admin", "txpool", "miner", "qmpow"},
		HTTPVirtualHosts: []string{"localhost"},
		HTTPCors:         []string{"*"},
		WSHost:           "127.0.0.1",
		WSPort:           8546,
		WSModules:        []string{"eth", "net", "web3", "personal", "admin", "txpool", "miner", "qmpow"},
		WSOrigins:        []string{"*"},
		P2P: node.P2PConfig{
			MaxPeers: 25,
			NoDiscovery: false,
			DiscoveryV5: true,
			Name: "quantum-wallet-embedded",
		},
	}

	// Create the node
	var err error
	w.node, err = node.New(w.nodeConfig)
	if err != nil {
		return fmt.Errorf("failed to create node: %v", err)
	}

	// Configure Ethereum service for Quantum Testnet
	w.config = &ethconfig.Config{
		Genesis:   w.getQuantumTestnetGenesis(),
		NetworkId: 73235, // Quantum Testnet
		SyncMode:  ethconfig.FullSync,
		DatabaseCache: 512,
		TrieCleanCache: 256,
		TrieDirtyCache: 256,
		SnapshotCache: 102,
		Miner: ethconfig.Miner{
			Etherbase:  common.Address{},
			ExtraData:  []byte("Quantum Wallet Embedded"),
			GasFloor:   8000000,
			GasCeil:    8000000,
			GasPrice:   big.NewInt(1000000000),
			Recommit:   time.Second,
		},
		TxPool: ethconfig.DefaultTxPoolConfig(),
	}

	// Register the Ethereum service
	ethService, err := eth.New(w.node, w.config)
	if err != nil {
		return fmt.Errorf("failed to create Ethereum service: %v", err)
	}
	w.ethereum = ethService

	// Start the node
	if err := w.node.Start(); err != nil {
		return fmt.Errorf("failed to start node: %v", err)
	}

	// Connect to the embedded node via RPC
	rpcClient := w.node.Attach()
	w.rpcClient = rpcClient
	w.client = ethclient.NewClient(rpcClient)
	w.isConnected = true

	// Start network info updates
	go w.updateNetworkInfo()

	log.Info("Embedded Q Geth node started successfully")
	return nil
}

// getQuantumTestnetGenesis returns the genesis configuration for Quantum Testnet
func (w *WalletService) getQuantumTestnetGenesis() *core.Genesis {
	// Load the quantum testnet genesis from the configs directory
	return &core.Genesis{
		Config: &params.ChainConfig{
			ChainID:             big.NewInt(73235),
			HomesteadBlock:      big.NewInt(0),
			EIP150Block:         big.NewInt(0),
			EIP155Block:         big.NewInt(0),
			EIP158Block:         big.NewInt(0),
			ByzantiumBlock:      big.NewInt(0),
			ConstantinopleBlock: big.NewInt(0),
			PetersburgBlock:     big.NewInt(0),
			IstanbulBlock:       big.NewInt(0),
			BerlinBlock:         big.NewInt(0),
			LondonBlock:         big.NewInt(0),
			TerminalTotalDifficulty: nil, // No Proof-of-Stake for quantum blockchain
		},
		Alloc:      core.GenesisAlloc{},
		Coinbase:   common.Address{},
		Difficulty: big.NewInt(160),
		GasLimit:   8000000,
		Timestamp:  1640995200, // Jan 1, 2022
	}
}

// ConnectToNode connects to an external Q Geth node
func (w *WalletService) ConnectToNode(endpoint string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	log.Info("Attempting to connect to Q Geth node", "endpoint", endpoint)

	// Close existing connection
	if w.client != nil {
		w.client.Close()
		log.Info("Closed existing client connection")
	}
	if w.rpcClient != nil {
		w.rpcClient.Close()
		log.Info("Closed existing RPC client connection")
	}

	// Connect to the new endpoint
	log.Info("Dialing Q Geth node...")
	client, err := ethclient.Dial(endpoint)
	if err != nil {
		log.Error("Failed to dial Q Geth node", "endpoint", endpoint, "error", err)
		return fmt.Errorf("failed to connect to node at %s: %v", endpoint, err)
	}

	w.client = client
	w.rpcClient = client.Client()
	w.isConnected = true

	log.Info("Successfully connected to Q Geth node", "endpoint", endpoint)

	// Start network info updates
	log.Info("Starting network info update goroutine")
	go w.updateNetworkInfo()

	log.Info("Q Geth node connection completed successfully")
	return nil
}

// Embedded node methods disabled for now - using external node connection instead
// TODO: Re-implement when Q Geth API is stable

// Shutdown gracefully shuts down the wallet service and embedded node
func (w *WalletService) Shutdown() {
	w.mu.Lock()
	defer w.mu.Unlock()

	log.Info("Shutting down Quantum Wallet service...")

	if w.client != nil {
		w.client.Close()
		log.Info("Closed client connection")
	}
	if w.rpcClient != nil {
		w.rpcClient.Close()
		log.Info("Closed RPC client connection")
	}
	if w.node != nil {
		if err := w.node.Close(); err != nil {
			log.Error("Error closing embedded node", "error", err)
		} else {
			log.Info("Embedded Q Geth node shut down successfully")
		}
	}
	
	log.Info("Quantum Wallet service shut down completed")
}

// updateNetworkInfo periodically updates network information
func (w *WalletService) updateNetworkInfo() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if !w.isConnected || w.client == nil {
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		
		// Get chain ID
		if chainID, err := w.client.ChainID(ctx); err == nil {
			w.networkInfo.ChainID = chainID.String()
			
			// Determine network name based on chain ID
			switch chainID.Int64() {
			case 73235:
				w.networkInfo.NetworkName = "Quantum Testnet"
			case 73236:
				w.networkInfo.NetworkName = "Quantum Mainnet"
			default:
				w.networkInfo.NetworkName = fmt.Sprintf("Quantum Network (Chain ID: %d)", chainID.Int64())
			}
		}

		// Get latest block
		if block, err := w.client.BlockByNumber(ctx, nil); err == nil {
			w.networkInfo.BlockNumber = block.NumberU64()
			w.networkInfo.Difficulty = block.Difficulty().String()
		}

		// Get peer count
		if peerCount, err := w.client.PeerCount(ctx); err == nil {
			w.networkInfo.PeerCount = uint64(peerCount)
		}

		// Get sync status
		if syncProgress, err := w.client.SyncProgress(ctx); err == nil {
			w.networkInfo.Syncing = syncProgress != nil
		}

		// Get gas price
		if gasPrice, err := w.client.SuggestGasPrice(ctx); err == nil {
			w.networkInfo.GasPrice = gasPrice.String()
		}

		cancel()
	}
}

// loadAccounts loads existing accounts from keystore
func (w *WalletService) loadAccounts() error {
	w.accounts = w.keystore.Accounts()
	return nil
}

// CreateAccount creates a new account with the given passphrase
func (w *WalletService) CreateAccount(passphrase string) (*Account, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	account, err := w.keystore.NewAccount(passphrase)
	if err != nil {
		return nil, fmt.Errorf("failed to create account: %v", err)
	}

	w.accounts = append(w.accounts, account)

	return &Account{
		Address:  account.Address.Hex(),
		Balance:  "0",
		Name:     fmt.Sprintf("Account %d", len(w.accounts)),
		IsLocked: true,
	}, nil
}

// GetAccounts returns all wallet accounts with balances
func (w *WalletService) GetAccounts() ([]*Account, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	var accounts []*Account
	for i, acc := range w.accounts {
		account := &Account{
			Address:  acc.Address.Hex(),
			Balance:  "0",
			Name:     fmt.Sprintf("Account %d", i+1),
			IsLocked: true,
		}

		// Get balance if connected
		if w.isConnected && w.client != nil {
			if balance, err := w.client.BalanceAt(context.Background(), acc.Address, nil); err == nil {
				account.Balance = balance.String()
			}
		}

		accounts = append(accounts, account)
	}

	return accounts, nil
}

// GetBalance returns the balance for a specific account address
func (w *WalletService) GetBalance(address string) (string, error) {
	if !w.isConnected || w.client == nil {
		return "0", fmt.Errorf("not connected to quantum network")
	}

	addr := common.HexToAddress(address)
	balance, err := w.client.BalanceAt(context.Background(), addr, nil)
	if err != nil {
		return "0", fmt.Errorf("failed to get balance: %v", err)
	}

	// Convert from Wei to Ether (Q tokens)
	ethBalance := new(big.Float).Quo(new(big.Float).SetInt(balance), big.NewFloat(1e18))
	return ethBalance.String(), nil
}

// GetNetworkInfo returns current network information
func (w *WalletService) GetNetworkInfo() *NetworkInfo {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.networkInfo
}

// IsConnected returns connection status
func (w *WalletService) IsConnected() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.isConnected
}

// SendTransaction sends a transaction
func (w *WalletService) SendTransaction(from, to, value, passphrase string) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isConnected {
		return "", fmt.Errorf("not connected to quantum network")
	}

	// Find the account
	fromAddr := common.HexToAddress(from)
	var account accounts.Account
	found := false
	for _, acc := range w.accounts {
		if acc.Address == fromAddr {
			account = acc
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("account not found")
	}

	// Unlock account
	if err := w.keystore.Unlock(account, passphrase); err != nil {
		return "", fmt.Errorf("failed to unlock account: %v", err)
	}
	defer w.keystore.Lock(account.Address)

	// Parse value
	amount, ok := new(big.Int).SetString(value, 10)
	if !ok {
		return "", fmt.Errorf("invalid value amount")
	}

	// Get nonce
	nonce, err := w.client.PendingNonceAt(context.Background(), fromAddr)
	if err != nil {
		return "", fmt.Errorf("failed to get nonce: %v", err)
	}

	// Get gas price
	gasPrice, err := w.client.SuggestGasPrice(context.Background())
	if err != nil {
		return "", fmt.Errorf("failed to get gas price: %v", err)
	}

	// Create transaction
	toAddr := common.HexToAddress(to)
	tx := types.NewTransaction(nonce, toAddr, amount, 21000, gasPrice, nil)

	// Sign transaction
	chainID, err := w.client.ChainID(context.Background())
	if err != nil {
		return "", fmt.Errorf("failed to get chain ID: %v", err)
	}

	signedTx, err := w.keystore.SignTx(account, tx, chainID)
	if err != nil {
		return "", fmt.Errorf("failed to sign transaction: %v", err)
	}

	// Send transaction
	if err := w.client.SendTransaction(context.Background(), signedTx); err != nil {
		return "", fmt.Errorf("failed to send transaction: %v", err)
	}

	return signedTx.Hash().Hex(), nil
}

// GetTransactionHistory returns transaction history for an account
func (w *WalletService) GetTransactionHistory(address string) ([]*Transaction, error) {
	if !w.isConnected {
		return nil, fmt.Errorf("not connected to quantum network")
	}

	// This is a simplified version - in a real implementation,
	// you'd need to index transactions or use an external indexer
	var transactions []*Transaction

	// Get recent blocks and scan for transactions
	latestBlock, err := w.client.BlockByNumber(context.Background(), nil)
	if err != nil {
		return nil, err
	}

	addr := common.HexToAddress(address)
	
	// Scan last 100 blocks for transactions involving this address
	for i := int64(0); i < 100 && latestBlock.NumberU64() >= uint64(i); i++ {
		blockNum := new(big.Int).SetUint64(latestBlock.NumberU64() - uint64(i))
		block, err := w.client.BlockByNumber(context.Background(), blockNum)
		if err != nil {
			continue
		}

		for _, tx := range block.Transactions() {
			// Convert string ChainID back to *big.Int for transaction sender calculation
			chainID, ok := new(big.Int).SetString(w.networkInfo.ChainID, 10)
			if !ok {
				continue
			}
			msg, err := types.Sender(types.NewEIP155Signer(chainID), tx)
			if err != nil {
				continue
			}

			// Check if this transaction involves our address
			if msg == addr || (tx.To() != nil && *tx.To() == addr) {
				transaction := &Transaction{
					Hash:        tx.Hash().Hex(),
					From:        msg.Hex(),
					To:          "",
					Value:       tx.Value().String(),
					Gas:         tx.Gas(),
					GasPrice:    tx.GasPrice().String(),
					Status:      "confirmed",
					BlockNumber: block.NumberU64(),
					Timestamp:   int64(block.Time()),
					Type:        "transfer",
				}

				if tx.To() != nil {
					transaction.To = tx.To().Hex()
				}

				transactions = append(transactions, transaction)
			}
		}
	}

	return transactions, nil
}

// StartMining starts the mining process
func (w *WalletService) StartMining(minerAddress string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isConnected {
		return fmt.Errorf("not connected to quantum network")
	}

	// Set miner address
	if minerAddress != "" {
		var call bool
		if err := w.rpcClient.Call(&call, "miner_setEtherbase", common.HexToAddress(minerAddress)); err != nil {
			return fmt.Errorf("failed to set miner address: %v", err)
		}
	}

	// Start mining
	var result bool
	if err := w.rpcClient.Call(&result, "miner_start", 1); err != nil {
		return fmt.Errorf("failed to start mining: %v", err)
	}

	w.isMining = true
	return nil
}

// StopMining stops the mining process
func (w *WalletService) StopMining() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.isConnected {
		return fmt.Errorf("not connected to quantum network")
	}

	var result bool
	if err := w.rpcClient.Call(&result, "miner_stop"); err != nil {
		return fmt.Errorf("failed to stop mining: %v", err)
	}

	w.isMining = false
	return nil
}

// GetMiningInfo returns current mining information
func (w *WalletService) GetMiningInfo() *MiningInfo {
	w.mu.RLock()
	defer w.mu.RUnlock()

	info := &MiningInfo{
		IsMining: w.isMining,
		HashRate: "0",
	}

	if w.isConnected && w.rpcClient != nil {
		// Get hash rate
		var hashRate string
		if err := w.rpcClient.Call(&hashRate, "eth_hashrate"); err == nil {
			if rate, ok := new(big.Int).SetString(strings.TrimPrefix(hashRate, "0x"), 16); ok {
				info.HashRate = rate.String()
			}
		}

		// Get coinbase (miner address)
		var coinbase string
		if err := w.rpcClient.Call(&coinbase, "eth_coinbase"); err == nil {
			info.MinerAddress = coinbase
		}
	}

	return info
}

// ExecuteConsoleCommand executes a geth console command
func (w *WalletService) ExecuteConsoleCommand(command string) (string, error) {
	if !w.isConnected {
		return "", fmt.Errorf("not connected to quantum network")
	}

	// Parse and execute the command
	// This is a simplified version - you'd want to implement a full JavaScript console
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return "", fmt.Errorf("empty command")
	}

	switch parts[0] {
	case "eth.accounts":
		accounts, err := w.GetAccounts()
		if err != nil {
			return "", err
		}
		result := "["
		for i, acc := range accounts {
			if i > 0 {
				result += ", "
			}
			result += fmt.Sprintf(`"%s"`, acc.Address)
		}
		result += "]"
		return result, nil

	case "eth.blockNumber":
		if w.networkInfo != nil {
			return fmt.Sprintf("%d", w.networkInfo.BlockNumber), nil
		}
		return "0", nil

	case "net.peerCount":
		if w.networkInfo != nil {
			return fmt.Sprintf("%d", w.networkInfo.PeerCount), nil
		}
		return "0", nil

	default:
		return "", fmt.Errorf("command not supported: %s", parts[0])
	}
} 