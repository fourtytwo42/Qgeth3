package miner

import (
	"time"
)

// Miner represents a quantum miner instance
type Miner interface {
	// Start begins the mining process
	Start() error

	// Stop gracefully stops the mining process
	Stop()

	// GetStats returns current mining statistics
	GetStats() *Stats

	// IsRunning returns true if the miner is currently running
	IsRunning() bool
}

// Stats represents mining statistics
type Stats struct {
	// Mining performance
	Hashrate    float64 `json:"hashrate"`     // Hashes per second
	QuantumRate float64 `json:"quantum_rate"` // Quantum circuits per second
	Accepted    uint64  `json:"accepted"`     // Accepted shares/blocks
	Rejected    uint64  `json:"rejected"`     // Rejected shares/blocks
	Stale       uint64  `json:"stale"`        // Stale shares
	Invalid     uint64  `json:"invalid"`      // Invalid shares

	// Connection status
	Connected bool          `json:"connected"`  // Connection status
	Uptime    time.Duration `json:"uptime"`     // Miner uptime
	LastShare time.Time     `json:"last_share"` // Time of last share
	LastBlock time.Time     `json:"last_block"` // Time of last block found

	// Pool/Solo specific
	Difficulty  float64 `json:"difficulty"`   // Current difficulty
	BlockHeight uint64  `json:"block_height"` // Current block height
	NetworkHash float64 `json:"network_hash"` // Network hashrate

	// Hardware stats
	Temperature float64 `json:"temperature"` // Hardware temperature (if available)
	PowerUsage  float64 `json:"power_usage"` // Power usage (if available)

	// Quantum specific
	QuantumErrors uint64 `json:"quantum_errors"` // Quantum computation errors
	CircuitDepth  int    `json:"circuit_depth"`  // Average circuit depth
	GateCount     int    `json:"gate_count"`     // Average gate count per circuit
}

// Work represents a unit of mining work
type Work struct {
	// Block template data
	BlockNumber uint64 `json:"block_number"`
	ParentHash  string `json:"parent_hash"`
	Timestamp   uint64 `json:"timestamp"`
	Difficulty  string `json:"difficulty"`
	GasLimit    uint64 `json:"gas_limit"`
	Coinbase    string `json:"coinbase"`
	ExtraData   string `json:"extra_data"`

	// Quantum mining specific
	QuantumSeed string `json:"quantum_seed"` // Seed for quantum circuit generation
	PuzzleCount int    `json:"puzzle_count"` // Number of quantum puzzles
	Target      string `json:"target"`       // Mining target

	// Timing
	StartTime time.Time `json:"start_time"` // When work started
	Deadline  time.Time `json:"deadline"`   // Work deadline
}

// Solution represents a mining solution
type Solution struct {
	// Basic solution data
	Nonce   uint64 `json:"nonce"`
	QNonce  uint64 `json:"qnonce"`
	MixHash string `json:"mix_hash"`

	// Quantum proof data
	ProofRoot   string `json:"proof_root"`   // Merkle root of quantum proofs
	OutcomeRoot string `json:"outcome_root"` // Merkle root of measurement outcomes
	GateHash    string `json:"gate_hash"`    // Hash of gate sequence
	QuantumBlob string `json:"quantum_blob"` // Compressed quantum data

	// Metadata
	ComputeTime  time.Duration `json:"compute_time"`  // Time to compute solution
	CircuitStats CircuitStats  `json:"circuit_stats"` // Circuit statistics
}

// CircuitStats represents quantum circuit statistics
type CircuitStats struct {
	TotalGates   int     `json:"total_gates"`  // Total number of gates
	TGates       int     `json:"t_gates"`      // Number of T-gates
	Depth        int     `json:"depth"`        // Circuit depth
	Qubits       int     `json:"qubits"`       // Number of qubits used
	Measurements int     `json:"measurements"` // Number of measurements
	Fidelity     float64 `json:"fidelity"`     // Circuit fidelity (if available)
}

// NewStats creates a new Stats instance
func NewStats() *Stats {
	return &Stats{
		Hashrate:      0.0,
		QuantumRate:   0.0,
		Accepted:      0,
		Rejected:      0,
		Stale:         0,
		Invalid:       0,
		Connected:     false,
		Uptime:        0,
		LastShare:     time.Time{},
		LastBlock:     time.Time{},
		Difficulty:    0.0,
		BlockHeight:   0,
		NetworkHash:   0.0,
		Temperature:   0.0,
		PowerUsage:    0.0,
		QuantumErrors: 0,
		CircuitDepth:  0,
		GateCount:     0,
	}
}

// AcceptanceRate returns the share acceptance rate as a percentage
func (s *Stats) AcceptanceRate() float64 {
	total := s.Accepted + s.Rejected + s.Stale + s.Invalid
	if total == 0 {
		return 0.0
	}
	return float64(s.Accepted) / float64(total) * 100.0
}

// RejectionRate returns the share rejection rate as a percentage
func (s *Stats) RejectionRate() float64 {
	total := s.Accepted + s.Rejected + s.Stale + s.Invalid
	if total == 0 {
		return 0.0
	}
	return float64(s.Rejected+s.Stale+s.Invalid) / float64(total) * 100.0
}

// TotalShares returns the total number of shares submitted
func (s *Stats) TotalShares() uint64 {
	return s.Accepted + s.Rejected + s.Stale + s.Invalid
}
