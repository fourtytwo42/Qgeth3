// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the quantum micro-puzzle proof-of-work consensus engine.
package qmpow

import (
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
)

// Default quantum proof of work parameters
const (
	DefaultQBits    uint8  = 6  // qubits per micro-puzzle (lowered for testing)
	DefaultTCount   uint16 = 15 // T-gate count per puzzle (lowered for testing)
	DefaultLNet     uint16 = 20 // puzzles per block (network difficulty) (lowered for testing)
	DefaultEpochLen uint64 = 50 // difficulty retarget period in blocks
	TargetBlockTime uint64 = 12 // target block time in seconds

	// Difficulty adjustment parameters
	DifficultyAdjustmentThreshold = 0.1 // 10% threshold for adjustment
	DifficultyStep                = 2   // L_net adjustment step (lowered)
	MinLNet                       = 4   // minimum puzzles per block (lowered)
	MaxLNet                       = 64  // maximum puzzles per block (lowered)
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

	// Get the current epoch
	epoch := height / DefaultEpochLen
	if epoch == 0 {
		return DefaultParams()
	}

	// Calculate difficulty based on previous epoch timing
	params := DefaultParams()
	params.LNet = q.calculateDifficulty(height)

	return params
}

// calculateDifficulty implements the difficulty adjustment algorithm
func (q *QMPoW) calculateDifficulty(height uint64) uint16 {
	if height < DefaultEpochLen {
		return DefaultLNet
	}

	// Get the header from the start of current epoch
	epochStart := (height / DefaultEpochLen) * DefaultEpochLen
	if epochStart == 0 {
		epochStart = 1
	}

	// Get the header from the start of previous epoch
	prevEpochStart := epochStart - DefaultEpochLen
	if prevEpochStart == 0 {
		return DefaultLNet
	}

	// This is a simplified version - in the real implementation we would
	// get the actual headers from the chain to calculate timing
	// For now, return the default difficulty
	return DefaultLNet
}

// RetargetDifficulty adjusts the difficulty based on block timing
func RetargetDifficulty(currentTime, parentTime uint64, currentLNet uint16) uint16 {
	deltaTime := currentTime - parentTime
	target := TargetBlockTime

	// If block time is too fast, increase difficulty
	if float64(deltaTime) < float64(target)*(1.0-DifficultyAdjustmentThreshold) {
		newLNet := currentLNet + DifficultyStep
		if newLNet > MaxLNet {
			return MaxLNet
		}
		return newLNet
	}

	// If block time is too slow, decrease difficulty
	if float64(deltaTime) > float64(target)*(1.0+DifficultyAdjustmentThreshold) {
		if currentLNet <= DifficultyStep {
			return MinLNet
		}
		newLNet := currentLNet - DifficultyStep
		if newLNet < MinLNet {
			return MinLNet
		}
		return newLNet
	}

	// Block time is within acceptable range, keep difficulty
	return currentLNet
}

// EstimateNextDifficulty estimates what the next block's difficulty should be
func (q *QMPoW) EstimateNextDifficulty(chain consensus.ChainHeaderReader, header *types.Header) uint16 {
	if header.Number.Uint64() == 0 {
		return DefaultLNet
	}

	parent := chain.GetHeader(header.ParentHash, header.Number.Uint64()-1)
	if parent == nil {
		return DefaultLNet
	}

	// Get current L_net from parent
	currentLNet := DefaultLNet
	if parent.LUsed != nil {
		currentLNet = *parent.LUsed
	}

	return RetargetDifficulty(header.Time, parent.Time, currentLNet)
}
