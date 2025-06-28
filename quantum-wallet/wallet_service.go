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
)

// WalletService handles all quantum blockchain operations
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
	ChainID      *big.Int `json:"chainId"`
	NetworkName  string   `json:"networkName"`
	BlockNumber  uint64   `json:"blockNumber"`
	PeerCount    uint64   `json:"peerCount"`
	Syncing      bool     `json:"syncing"`
	GasPrice     *big.Int `json:"gasPrice"`
	Difficulty   *big.Int `json:"difficulty"`
	HashRate     *big.Int `json:"hashRate"`
}

// MiningInfo contains mining status and configuration
type MiningInfo struct {
	IsMining     bool     `json:"isMining"`
	HashRate     *big.Int `json:"hashRate"`
	MinerAddress string   `json:"minerAddress"`
	BlocksMined  uint64   `json:"blocksMined"`
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

// NewWalletService creates a new wallet service
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

// Initialize sets up the wallet service
func (w *WalletService) Initialize() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Create data directory
	if err := os.MkdirAll(w.dataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	// Initialize keystore
	keystoreDir := filepath.Join(w.dataDir, "keystore")
	w.keystore = keystore.NewKeyStore(keystoreDir, keystore.StandardScryptN, keystore.StandardScryptP)

	// Load existing accounts
	if err := w.loadAccounts(); err != nil {
		log.Warn("Failed to load accounts", "error", err)
	}

	// Try to connect to local quantum node
	w.connectToNode()

	return nil
}

// Shutdown gracefully shuts down the wallet service
func (w *WalletService) Shutdown() {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.client != nil {
		w.client.Close()
	}
	if w.rpcClient != nil {
		w.rpcClient.Close()
	}
}

// connectToNode attempts to connect to a quantum geth node
func (w *WalletService) connectToNode() {
	// Try multiple connection methods
	endpoints := []string{
		"http://localhost:8545",  // HTTP RPC
		"ws://localhost:8546",    // WebSocket
		filepath.Join(w.dataDir, "geth.ipc"), // IPC
	}

	for _, endpoint := range endpoints {
		if client, err := ethclient.Dial(endpoint); err == nil {
			w.client = client
			w.rpcClient = client.Client()
			w.isConnected = true
			log.Info("Connected to quantum node", "endpoint", endpoint)
			
			// Update network info
			go w.updateNetworkInfo()
			return
		}
	}
	
	log.Warn("Could not connect to quantum node, running in offline mode")
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
			w.networkInfo.ChainID = chainID
			
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
			w.networkInfo.Difficulty = block.Difficulty()
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
			w.networkInfo.GasPrice = gasPrice
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
			msg, err := types.Sender(types.NewEIP155Signer(w.networkInfo.ChainID), tx)
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
		HashRate: big.NewInt(0),
	}

	if w.isConnected && w.rpcClient != nil {
		// Get hash rate
		var hashRate string
		if err := w.rpcClient.Call(&hashRate, "eth_hashrate"); err == nil {
			if rate, ok := new(big.Int).SetString(strings.TrimPrefix(hashRate, "0x"), 16); ok {
				info.HashRate = rate
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