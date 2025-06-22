// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the quantum micro-puzzle proof-of-work consensus engine.
package qmpow

import (
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// Default quantum proof of work parameters
const (
	DefaultQBits    uint8  = 12   // 4,096 quantum states per puzzle
	DefaultTCount   uint16 = 4096 // Maximum T-gate complexity for high security
	DefaultLNet     uint16 = 48   // Static puzzle count for 1,152-bit security (Bitcoin-style)
	DefaultEpochLen uint64 = 100  // difficulty retarget period in blocks (every ~20 minutes at 12s blocks)
	TargetBlockTime uint64 = 12   // target block time in seconds

	// Difficulty adjustment parameters - now controls nonce difficulty, not puzzle count
	DifficultyAdjustmentThreshold = 0.15 // 15% threshold for adjustment (more tolerant)
	DifficultyStepSmall           = 1    // Small adjustment step
	DifficultyStepLarge           = 4    // Large adjustment step for big deviations
	MinLNet                       = 48   // Fixed puzzle count - no longer variable
	MaxLNet                       = 48   // Fixed puzzle count - Bitcoin-style static work

	// Mining complexity parameters
	BaseComplexityMs      = 100 // base time per puzzle in milliseconds
	ComplexityScaleFactor = 1.5 // how much complexity increases per puzzle
)

// QMPoWParams represents the configuration parameters for quantum proof of work
type QMPoWParams struct {
	QBits    uint8  // qubits per puzzle
	TCount   uint16 // T gates per puzzle
	LNet     uint16 // puzzles per block (current difficulty)
	EpochLen uint64 // blocks per epoch for difficulty adjustment
}

// DefaultParams returns the default quantum proof of work parameters
func DefaultParams() *QMPoWParams {
	return &QMPoWParams{
		QBits:    DefaultQBits,
		TCount:   DefaultTCount,
		LNet:     DefaultLNet,
		EpochLen: DefaultEpochLen,
	}
}

// ParamsForHeight returns the quantum proof of work parameters for a given block height
func (q *QMPoW) ParamsForHeight(height uint64) *QMPoWParams {
	if height == 0 {
		return DefaultParams()
	}

	// Calculate difficulty based on block timing
	params := DefaultParams()
	params.LNet = q.calculateDifficultyForHeight(height)

	return params
}

// calculateDifficultyForHeight implements the difficulty adjustment algorithm
// NOTE: With Bitcoin-style mining, puzzle count is fixed at 48
// Difficulty now adjusts through nonce target, not puzzle count
func (q *QMPoW) calculateDifficultyForHeight(height uint64) uint16 {
	// Always return fixed puzzle count - difficulty adjusts through nonce target
	return DefaultLNet
}

// RetargetDifficulty adjusts the difficulty based on block timing
// NOTE: With Bitcoin-style mining, this now affects nonce difficulty target
// The puzzle count remains fixed at 48 for consistent 1,152-bit security
func RetargetDifficulty(currentTime, parentTime uint64, currentLNet uint16) uint16 {
	// Always return fixed puzzle count
	// Real difficulty adjustment happens in nonce target calculation
	log.Info("🎯 Bitcoin-style difficulty (fixed puzzles)",
		"puzzles", DefaultLNet,
		"security", "1,152-bit",
		"style", "nonce-based")

	return DefaultLNet
}

// EstimateNextDifficulty estimates what the next block's difficulty should be
// NOTE: With Bitcoin-style mining, puzzle count is always fixed
func (q *QMPoW) EstimateNextDifficulty(chain consensus.ChainHeaderReader, header *types.Header) uint16 {
	// Always return fixed puzzle count for Bitcoin-style mining
	log.Info("🔗 Fixed puzzle difficulty (Bitcoin-style)",
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
