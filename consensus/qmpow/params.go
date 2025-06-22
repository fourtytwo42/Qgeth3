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
	DefaultQBits    uint8  = 8   // qubits per micro-puzzle (restored to proper value)
	DefaultTCount   uint16 = 25  // T-gate count per puzzle (restored to proper value)
	DefaultLNet     uint16 = 32  // puzzles per block (network difficulty) (moderate starting value)
	DefaultEpochLen uint64 = 100 // difficulty retarget period in blocks (every ~20 minutes at 12s blocks)
	TargetBlockTime uint64 = 12  // target block time in seconds

	// Difficulty adjustment parameters
	DifficultyAdjustmentThreshold = 0.15 // 15% threshold for adjustment (more tolerant)
	DifficultyStepSmall           = 1    // Small adjustment step
	DifficultyStepLarge           = 4    // Large adjustment step for big deviations
	MinLNet                       = 8    // minimum puzzles per block
	MaxLNet                       = 256  // maximum puzzles per block

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
func (q *QMPoW) calculateDifficultyForHeight(height uint64) uint16 {
	if height < 10 { // Allow first 10 blocks to use default difficulty
		return DefaultLNet
	}

	// For the first epoch, use gradual adjustment
	if height < DefaultEpochLen {
		// Gradual increase for the first epoch to find the right difficulty
		return DefaultLNet + uint16(height/10)
	}

	// This is a simplified version - in the real implementation we would
	// get the actual headers from the chain to calculate timing
	// For now, return a reasonable difficulty
	return DefaultLNet
}

// RetargetDifficulty adjusts the difficulty based on block timing
func RetargetDifficulty(currentTime, parentTime uint64, currentLNet uint16) uint16 {
	if parentTime == 0 {
		return currentLNet
	}

	deltaTime := currentTime - parentTime
	target := TargetBlockTime

	log.Info("🎯 Difficulty adjustment calculation",
		"deltaTime", deltaTime,
		"target", target,
		"currentLNet", currentLNet,
		"ratio", float64(deltaTime)/float64(target))

	// Calculate the deviation from target
	ratio := float64(deltaTime) / float64(target)

	// Determine adjustment step based on deviation size
	var step uint16
	if ratio < 0.5 || ratio > 2.0 {
		// Large deviation - use large step
		step = DifficultyStepLarge
	} else {
		// Small deviation - use small step
		step = DifficultyStepSmall
	}

	// If block time is too fast, increase difficulty
	if ratio < (1.0 - DifficultyAdjustmentThreshold) {
		newLNet := currentLNet + step
		if newLNet > MaxLNet {
			newLNet = MaxLNet
		}
		log.Info("⬆️ Increasing difficulty", "from", currentLNet, "to", newLNet, "reason", "blocks too fast")
		return newLNet
	}

	// If block time is too slow, decrease difficulty
	if ratio > (1.0 + DifficultyAdjustmentThreshold) {
		var newLNet uint16
		if currentLNet <= step {
			newLNet = MinLNet
		} else {
			newLNet = currentLNet - step
			if newLNet < MinLNet {
				newLNet = MinLNet
			}
		}
		log.Info("⬇️ Decreasing difficulty", "from", currentLNet, "to", newLNet, "reason", "blocks too slow")
		return newLNet
	}

	// Block time is within acceptable range, keep difficulty
	log.Info("➡️ Maintaining difficulty", "lnet", currentLNet, "reason", "timing acceptable")
	return currentLNet
}

// EstimateNextDifficulty estimates what the next block's difficulty should be
func (q *QMPoW) EstimateNextDifficulty(chain consensus.ChainHeaderReader, header *types.Header) uint16 {
	if header.Number.Uint64() == 0 {
		return DefaultLNet
	}

	parent := chain.GetHeader(header.ParentHash, header.Number.Uint64()-1)
	if parent == nil {
		// If we can't find parent, use default + small increment
		log.Info("🔗 Parent not found during difficulty calc, using default+1", "blockNumber", header.Number.Uint64())
		return DefaultLNet + 1
	}

	// Get current L_net from parent
	currentLNet := DefaultLNet
	if parent.LUsed != nil {
		currentLNet = *parent.LUsed
	}

	// Use the retarget algorithm
	nextLNet := RetargetDifficulty(header.Time, parent.Time, currentLNet)

	log.Info("🎯 Next difficulty estimated",
		"blockNumber", header.Number.Uint64(),
		"parentLNet", currentLNet,
		"nextLNet", nextLNet,
		"parentTime", parent.Time,
		"currentTime", header.Time)

	return nextLNet
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
