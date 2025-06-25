package config

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
)

// Config represents the complete miner configuration
type Config struct {
	Mode    string        `json:"mode"` // "pool" or "solo"
	Mining  MiningConfig  `json:"mining"`
	Pool    PoolConfig    `json:"pool"`
	Solo    SoloConfig    `json:"solo"`
	Quantum QuantumConfig `json:"quantum"`
	Logging LoggingConfig `json:"logging"`
}

// MiningConfig contains mining-specific settings
type MiningConfig struct {
	Threads     int `json:"threads"`      // Number of mining threads
	Intensity   int `json:"intensity"`    // Mining intensity 1-10
	Timeout     int `json:"timeout"`      // Timeout for quantum computation (seconds)
	RetryCount  int `json:"retry_count"`  // Number of retries for failed work
	StatsReport int `json:"stats_report"` // Statistics reporting interval (seconds)
}

// PoolConfig contains pool mining settings
type PoolConfig struct {
	URL       string `json:"url"`        // Pool URL (stratum+tcp://...)
	Worker    string `json:"worker"`     // Worker name
	Password  string `json:"password"`   // Worker password
	KeepAlive int    `json:"keep_alive"` // Keep-alive interval (seconds)
	Reconnect int    `json:"reconnect"`  // Reconnection attempts
	ExtraData string `json:"extra_data"` // Extra data to include in blocks
}

// SoloConfig contains solo mining settings
type SoloConfig struct {
	NodeURL   string `json:"node_url"`   // Geth node RPC URL
	Coinbase  string `json:"coinbase"`   // Coinbase address
	GasLimit  uint64 `json:"gas_limit"`  // Gas limit for blocks
	GasPrice  uint64 `json:"gas_price"`  // Gas price (wei)
	ExtraData string `json:"extra_data"` // Extra data to include in blocks
}

// QuantumConfig contains quantum backend settings
type QuantumConfig struct {
	Backend   string `json:"backend"`   // "qiskit" or "cirq"
	Simulator string `json:"simulator"` // Simulator type
	Qubits    int    `json:"qubits"`    // Number of qubits (fixed at 16)
	Puzzles   int    `json:"puzzles"`   // Number of puzzles per block (fixed at 48)
	Python    string `json:"python"`    // Python executable path
}

// LoggingConfig contains logging settings
type LoggingConfig struct {
	Level    string `json:"level"`     // "debug", "info", "warn", "error"
	File     string `json:"file"`      // Log file path (empty for stdout)
	Rotate   bool   `json:"rotate"`    // Enable log rotation
	MaxSize  int    `json:"max_size"`  // Max log file size (MB)
	MaxFiles int    `json:"max_files"` // Max number of log files
}

// Default returns a default configuration
func Default() *Config {
	return &Config{
		Mode: "solo",
		Mining: MiningConfig{
			Threads:     runtime.NumCPU(),
			Intensity:   1,
			Timeout:     30,
			RetryCount:  3,
			StatsReport: 30,
		},
		Pool: PoolConfig{
			URL:       "",
			Worker:    "quantum-miner",
			Password:  "x",
			KeepAlive: 30,
			Reconnect: 5,
			ExtraData: "quantum-geth-v0.9",
		},
		Solo: SoloConfig{
			NodeURL:   "http://localhost:8545",
			Coinbase:  "",
			GasLimit:  8000000,
			GasPrice:  1000000000, // 1 gwei
			ExtraData: "quantum-geth-v0.9",
		},
		Quantum: QuantumConfig{
			Backend:   "qiskit-aer-statevector",
			Simulator: "aer_simulator",
			Qubits:    16,
			Puzzles:   48,
			Python:    "python",
		},
		Logging: LoggingConfig{
			Level:    "info",
			File:     "",
			Rotate:   false,
			MaxSize:  100,
			MaxFiles: 5,
		},
	}
}

// Load loads configuration from a JSON file
func Load(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &cfg, nil
}

// Save saves the configuration to a JSON file
func (c *Config) Save(filename string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
	// Validate mode
	if c.Mode != "pool" && c.Mode != "solo" {
		return fmt.Errorf("invalid mode: %s (must be 'pool' or 'solo')", c.Mode)
	}

	// Validate mining config
	if c.Mining.Threads < 1 {
		return fmt.Errorf("threads must be at least 1")
	}
	if c.Mining.Intensity < 1 || c.Mining.Intensity > 10 {
		return fmt.Errorf("intensity must be between 1 and 10")
	}
	if c.Mining.Timeout < 1 {
		return fmt.Errorf("timeout must be at least 1 second")
	}

	// Validate pool config
	if c.Mode == "pool" {
		if c.Pool.URL == "" {
			return fmt.Errorf("pool URL is required for pool mining")
		}
		if !strings.HasPrefix(c.Pool.URL, "stratum+tcp://") {
			return fmt.Errorf("pool URL must start with 'stratum+tcp://'")
		}
		if c.Pool.Worker == "" {
			return fmt.Errorf("worker name is required for pool mining")
		}
	}

	// Validate solo config
	if c.Mode == "solo" {
		if c.Solo.NodeURL == "" {
			return fmt.Errorf("node URL is required for solo mining")
		}
		if !strings.HasPrefix(c.Solo.NodeURL, "http://") && !strings.HasPrefix(c.Solo.NodeURL, "https://") {
			return fmt.Errorf("node URL must start with 'http://' or 'https://'")
		}
		if c.Solo.Coinbase == "" {
			return fmt.Errorf("coinbase address is required for solo mining")
		}
		if !strings.HasPrefix(c.Solo.Coinbase, "0x") || len(c.Solo.Coinbase) != 42 {
			return fmt.Errorf("coinbase must be a valid Ethereum address")
		}
	}

	// Validate quantum config
	if c.Quantum.Qubits != 16 {
		return fmt.Errorf("qubits must be 16 for quantum-geth compatibility")
	}
	if c.Quantum.Puzzles != 48 {
		return fmt.Errorf("puzzles must be 48 for quantum-geth compatibility")
	}

	return nil
}

// Print prints the configuration in a human-readable format
func (c *Config) Print() {
	fmt.Println("📋 Current Configuration:")
	fmt.Printf("  Mode: %s\n", c.Mode)
	fmt.Printf("  Mining Threads: %d\n", c.Mining.Threads)
	fmt.Printf("  Mining Intensity: %d\n", c.Mining.Intensity)

	if c.Mode == "pool" {
		fmt.Printf("  Pool URL: %s\n", c.Pool.URL)
		fmt.Printf("  Worker: %s\n", c.Pool.Worker)
	} else {
		fmt.Printf("  Node URL: %s\n", c.Solo.NodeURL)
		fmt.Printf("  Coinbase: %s\n", c.Solo.Coinbase)
	}

	fmt.Printf("  Quantum Backend: %s\n", c.Quantum.Backend)
	fmt.Printf("  Qubits: %d\n", c.Quantum.Qubits)
	fmt.Printf("  Puzzles: %d\n", c.Quantum.Puzzles)
}
