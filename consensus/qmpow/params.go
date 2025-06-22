// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the Quantum-Geth v0.9-rc3-hw0 quantum proof-of-work consensus engine.
package qmpow

import (
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// Quantum-Geth v0.9-rc3-hw0 Parameters
const (
	// Epochic glide schedule
	DefaultStartingQBits uint16 = 12   // Start n = 12 at epoch 0
	DefaultTCount        uint32 = 4096 // Constant 4,096 T-gates per puzzle
	DefaultLNet          uint16 = 48   // Fixed 48 puzzles providing 1,152-bit security
	DefaultEpochLen      uint64 = 100  // Difficulty retarget period in blocks
	TargetBlockTime      uint64 = 12   // Target block time in seconds

	// Glide parameters
	GlideInterval = 12500 // Add +1.0 qubit every 12,500 blocks (≈ 2 days)
	EpochInterval = 50000 // Epoch = ⌊Height / 50,000⌋

	// Mining complexity parameters (for simulation)
	BaseComplexityMs      = 100 // Base time per puzzle in milliseconds
	ComplexityScaleFactor = 1.5 // How much complexity increases per puzzle
)

// QMPoWParams represents the configuration parameters for Quantum-Geth v0.9-rc3-hw0
type QMPoWParams struct {
	Epoch    uint32 // Current epoch
	QBits    uint16 // Qubits per puzzle (from glide schedule)
	TCount   uint32 // T gates per puzzle (constant 4096)
	LNet     uint16 // Puzzles per block (constant 48)
	EpochLen uint64 // Blocks per epoch for difficulty adjustment
}

// DefaultParams returns the default quantum proof of work parameters for a given height
func DefaultParams(height uint64) QMPoWParams {
	return QMPoWParams{
		Epoch:    uint32(height / EpochInterval),
		QBits:    CalculateQBitsForHeight(height),
		TCount:   DefaultTCount,
		LNet:     DefaultLNet,
		EpochLen: DefaultEpochLen,
	}
}

// CalculateQBitsForHeight implements the epochic n-qubit glide schedule
// Start n = 12 at epoch 0, add +1.0 qubit every 12,500 blocks (≈ 2 days)
func CalculateQBitsForHeight(height uint64) uint16 {
	// Calculate how many glide intervals have passed
	glideSteps := height / GlideInterval

	// Start at 12 qubits, add 1 per glide step
	qbits := DefaultStartingQBits + uint16(glideSteps)

	log.Debug("🧮 Calculating QBits for height",
		"height", height,
		"glideSteps", glideSteps,
		"qbits", qbits)

	return qbits
}

// ParamsForHeight returns quantum parameters for a specific block height
func (q *QMPoW) ParamsForHeight(height uint64) QMPoWParams {
	return DefaultParams(height)
}

// EstimateNextDifficulty estimates the difficulty for the next block
// In v0.9-rc3-hw0, difficulty is handled via nonce targeting, not puzzle count changes
func (q *QMPoW) EstimateNextDifficulty(chain consensus.ChainHeaderReader, header *types.Header) uint16 {
	// Always return fixed puzzle count for v0.9-rc3-hw0
	log.Info("🔗 Fixed puzzle difficulty (v0.9-rc3-hw0)",
		"blockNumber", header.Number.Uint64(),
		"puzzles", DefaultLNet,
		"security", "1,152-bit")

	return DefaultLNet
}

// EstimateBlockTime estimates how long it will take to mine a block with given difficulty
func EstimateBlockTime(lnet uint16) float64 {
	// Base time per puzzle increases with complexity
	// This is a simplified model - real quantum circuits would have more complex timing
	baseTime := float64(BaseComplexityMs) / 1000.0 // Convert to seconds

	// Each additional puzzle adds exponentially more complexity
	totalTime := 0.0
	for i := uint16(0); i < lnet; i++ {
		puzzleTime := baseTime * (1.0 + float64(i)*ComplexityScaleFactor/100.0)
		totalTime += puzzleTime
	}

	return totalTime
}
