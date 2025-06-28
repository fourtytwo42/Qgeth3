package main

import (
	"fmt"
)

// API methods exposed to the frontend - these must match the frontend WalletContext expectations

// CreateAccount creates a new wallet account
func (a *App) CreateAccount(passphrase string) (string, error) {
	account, err := a.walletService.CreateAccount(passphrase)
	if err != nil {
		return "", err
	}
	return account.Address, nil
}

// GetAccounts returns all wallet accounts
func (a *App) GetAccounts() ([]*Account, error) {
	return a.walletService.GetAccounts()
}

// GetBalance returns the balance for a specific account
func (a *App) GetBalance(address string) (string, error) {
	balance, err := a.walletService.GetBalance(address)
	if err != nil {
		return "0", err
	}
	return balance, nil
}

// SendTransaction sends a transaction
func (a *App) SendTransaction(from, to, amount, passphrase string) (string, error) {
	return a.walletService.SendTransaction(from, to, amount, passphrase)
}

// GetTransactions returns transaction history for an account
func (a *App) GetTransactions(address string) ([]*Transaction, error) {
	return a.walletService.GetTransactionHistory(address)
}

// StartMining starts mining process
func (a *App) StartMining() error {
	// Use first account as miner address if available
	accounts, err := a.walletService.GetAccounts()
	if err != nil || len(accounts) == 0 {
		return fmt.Errorf("no accounts available for mining")
	}
	return a.walletService.StartMining(accounts[0].Address)
}

// StopMining stops mining process
func (a *App) StopMining() error {
	return a.walletService.StopMining()
}

// ConnectToNode connects to a Q Geth node
func (a *App) ConnectToNode(endpoint string) error {
	return a.walletService.ConnectToNode(endpoint)
}

// GetNetworkInfo returns current network information
func (a *App) GetNetworkInfo() (*NetworkInfo, error) {
	networkInfo := a.walletService.GetNetworkInfo()
	if networkInfo == nil {
		return nil, fmt.Errorf("not connected to network")
	}
	return networkInfo, nil
}

// ExecuteConsoleCommand executes a geth console command
func (a *App) ExecuteConsoleCommand(command string) (string, error) {
	return a.walletService.ExecuteConsoleCommand(command)
}

// Additional API methods for full wallet functionality

// GetWalletInfo returns general wallet information
func (a *App) GetWalletInfo() map[string]interface{} {
	return map[string]interface{}{
		"version":   "1.0.0",
		"name":      "Quantum Wallet",
		"connected": a.walletService.IsConnected(),
		"network":   a.walletService.GetNetworkInfo(),
	}
}

// GetMiningInfo returns mining information
func (a *App) GetMiningInfo() *MiningInfo {
	return a.walletService.GetMiningInfo()
}

// GetMinerSetupInstructions returns instructions for external miner setup
func (a *App) GetMinerSetupInstructions() map[string]interface{} {
	return map[string]interface{}{
		"externalMinerSetup": map[string]interface{}{
			"endpoint": "http://localhost:8545",
			"instructions": []string{
				"1. Download the quantum-miner from the releases page",
				"2. Extract the quantum-miner to a folder",
				"3. Open a terminal/command prompt in that folder", 
				"4. Run: ./quantum-miner --rpc-endpoint http://localhost:8545",
				"5. The miner will automatically connect to your Q Geth node",
				"6. Set your miner address in the wallet settings",
			},
			"requirements": []string{
				"Q Geth node must be running with HTTP RPC enabled",
				"Port 8545 must be accessible to the miner",
				"Miner address must be set in wallet settings",
			},
		},
		"builtInMining": map[string]interface{}{
			"description": "Enable built-in mining for testing purposes",
			"warning": "Built-in mining is less efficient than external quantum-miner",
			"instructions": []string{
				"1. Set your miner address in settings",
				"2. Click 'Start Mining' in the Mining tab",
				"3. Monitor mining progress in the dashboard",
			},
		},
	}
}

// ImportAccount imports an account from private key
func (a *App) ImportAccount(privateKey, passphrase string) (*Account, error) {
	// This would need to be implemented in the wallet service
	return nil, fmt.Errorf("import account not yet implemented")
}

// ExportAccount exports an account's private key
func (a *App) ExportAccount(address, passphrase string) (string, error) {
	// This would need to be implemented in the wallet service  
	return "", fmt.Errorf("export account not yet implemented")
}

// GetWalletConfig returns current wallet configuration
func (a *App) GetWalletConfig() *WalletConfig {
	// Return default config for now
	return &WalletConfig{
		Network:       "testnet",
		AutoMining:    false,
		ExternalMiner: true,
		HTTPPort:      8545,
		WSPort:        8546,
		EnableLogging: true,
		LogLevel:      "info",
	}
}

// SaveWalletConfig saves wallet configuration
func (a *App) SaveWalletConfig(config *WalletConfig) error {
	// This would save the config to a file
	return fmt.Errorf("save config not yet implemented")
}

// GetVersion returns the wallet version
func (a *App) GetVersion() string {
	return "1.0.0"
} 