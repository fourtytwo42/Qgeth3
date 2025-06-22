// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Bitcoin-Style Quantum Mining Implementation
// The world's first Bitcoin-style quantum proof-of-work consensus engine

package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"math/big"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// Bitcoin-Style Quantum Mining Constants
const (
	// Target adjustment parameters (like Bitcoin)
	TargetAdjustmentFactor = 4   // Maximum adjustment per retarget (like Bitcoin)
	RetargetBlocks         = 100 // Retarget every 100 blocks (~20 minutes at 12s blocks)

	// Bitcoin-style mining parameters
	MaxNonceAttempts    = 0xFFFFFFFF // 4 billion attempts like Bitcoin
	ProgressLogInterval = 1000       // Log progress every 1000 attempts
)

// Quantum proof quality thresholds (using big.Int to avoid overflow)
var (
	MaxQuantumTarget = new(big.Int).SetUint64(0xFFFFFFFFFFFFFFFF) // Maximum target (easiest difficulty)
	MinQuantumTarget = big.NewInt(1)                              // Minimum target (hardest difficulty)
)

// CalculateQuantumProofQuality computes the "quality" of a quantum proof
// Higher quality = better proof (like lower hash in Bitcoin)
// This is the core innovation: quantum proof quality replaces hash value
func CalculateQuantumProofQuality(outcomes []byte, proof []byte, qnonce uint64) *big.Int {
	// Bitcoin-style hash-based quality calculation
	h := sha256.New()

	// Include quantum outcomes (the heart of the proof)
	h.Write(outcomes)

	// Include quantum proof data
	h.Write(proof)

	// Include qnonce in quality calculation (Bitcoin-style nonce)
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, qnonce)
	h.Write(nonceBytes)

	// Add quantum-specific entropy
	h.Write([]byte("QUANTUM_BITCOIN_MINING"))

	hash := h.Sum(nil)

	// Convert hash to big integer
	quality := new(big.Int).SetBytes(hash)

	// Invert for Bitcoin-style comparison (lower hash = higher quality)
	// In Bitcoin: hash < target means success
	// In Quantum: quality > target means success (same logic, inverted)
	maxHash := new(big.Int).Lsh(big.NewInt(1), 256) // 2^256
	quality.Sub(maxHash, quality)

	return quality
}

// CalculateQuantumTarget converts difficulty to quantum target
// Like Bitcoin: higher difficulty = lower target = harder to mine
func CalculateQuantumTarget(difficulty *big.Int) *big.Int {
	if difficulty.Cmp(big.NewInt(0)) <= 0 {
		return new(big.Int).Set(MaxQuantumTarget)
	}

	// Bitcoin-style target calculation: target = max_target / difficulty
	target := new(big.Int).Div(MaxQuantumTarget, difficulty)

	// Ensure target stays within bounds
	if target.Cmp(MinQuantumTarget) < 0 {
		target.Set(MinQuantumTarget)
	}
	if target.Cmp(MaxQuantumTarget) > 0 {
		target.Set(MaxQuantumTarget)
	}

	return target
}

// CheckQuantumProofTarget verifies if quantum proof meets target
// Returns true if proof quality > target (like Bitcoin hash < target)
func CheckQuantumProofTarget(outcomes []byte, proof []byte, qnonce uint64, target *big.Int) bool {
	quality := CalculateQuantumProofQuality(outcomes, proof, qnonce)

	// Log the quality check for debugging
	log.Debug("🎯 Quantum proof quality check",
		"qnonce", qnonce,
		"quality", quality.String(),
		"target", target.String(),
		"success", quality.Cmp(target) > 0)

	return quality.Cmp(target) > 0
}

// CalculateNextDifficulty implements Bitcoin-style difficulty adjustment
// This is exactly like Bitcoin's difficulty adjustment algorithm
func CalculateNextDifficulty(currentDifficulty *big.Int, actualTime, targetTime uint64) *big.Int {
	log.Info("🔗 Bitcoin-style difficulty adjustment",
		"currentDifficulty", currentDifficulty,
		"actualTime", actualTime,
		"targetTime", targetTime)

	// Bitcoin's difficulty adjustment formula
	newDifficulty := new(big.Int).Set(currentDifficulty)

	// Calculate the ratio of actual time to target time
	if actualTime == 0 {
		actualTime = 1 // Prevent division by zero
	}

	// newDifficulty = currentDifficulty * targetTime / actualTime
	newDifficulty.Mul(newDifficulty, big.NewInt(int64(targetTime)))
	newDifficulty.Div(newDifficulty, big.NewInt(int64(actualTime)))

	// Limit adjustment to 4x up or down (like Bitcoin)
	maxAdjustment := new(big.Int).Mul(currentDifficulty, big.NewInt(TargetAdjustmentFactor))
	minAdjustment := new(big.Int).Div(currentDifficulty, big.NewInt(TargetAdjustmentFactor))

	if newDifficulty.Cmp(maxAdjustment) > 0 {
		newDifficulty.Set(maxAdjustment)
		log.Info("🔼 Difficulty capped at maximum adjustment", "newDifficulty", newDifficulty)
	}
	if newDifficulty.Cmp(minAdjustment) < 0 {
		newDifficulty.Set(minAdjustment)
		log.Info("🔽 Difficulty capped at minimum adjustment", "newDifficulty", newDifficulty)
	}

	// Ensure minimum difficulty
	if newDifficulty.Cmp(big.NewInt(1)) < 0 {
		newDifficulty.SetUint64(1)
	}

	log.Info("✅ Bitcoin-style difficulty adjusted",
		"oldDifficulty", currentDifficulty,
		"newDifficulty", newDifficulty,
		"ratio", float64(actualTime)/float64(targetTime))

	return newDifficulty
}

// ShouldRetargetDifficulty checks if it's time for difficulty adjustment
func ShouldRetargetDifficulty(blockNumber uint64) bool {
	return blockNumber > 0 && blockNumber%RetargetBlocks == 0
}

// GetRetargetPeriodStart returns the block number where the current retarget period started
func GetRetargetPeriodStart(blockNumber uint64) uint64 {
	if blockNumber < RetargetBlocks {
		return 0
	}
	return (blockNumber / RetargetBlocks) * RetargetBlocks
}

// EstimateHashrate calculates the network hashrate in puzzles per second
func EstimateHashrate(difficulty *big.Int, blockTime uint64) float64 {
	if blockTime == 0 {
		return 0
	}

	// Convert difficulty to float for calculation
	diffFloat := new(big.Float).SetInt(difficulty)

	// Hashrate = difficulty * puzzles_per_block / block_time
	puzzlesPerBlock := float64(DefaultLNet) // 48 puzzles
	hashrate := new(big.Float).Mul(diffFloat, big.NewFloat(puzzlesPerBlock))
	hashrate.Quo(hashrate, big.NewFloat(float64(blockTime)))

	result, _ := hashrate.Float64()
	return result
}

// ValidateQuantumProof validates a complete quantum proof for Bitcoin-style mining
func ValidateQuantumProof(header *types.Header) error {
	// Check that quantum fields are present
	if header.QBits == nil || header.TCount == nil || header.LUsed == nil {
		return ErrMissingQuantumFields
	}

	// Check that outcomes and proof are present
	if len(header.QOutcome) == 0 || len(header.QProof) == 0 {
		return ErrInvalidQuantumProof
	}

	// Calculate target from difficulty
	target := CalculateQuantumTarget(header.Difficulty)

	// Verify proof meets target
	if !CheckQuantumProofTarget(header.QOutcome, header.QProof, 0, target) {
		return ErrInvalidQuantumProof
	}

	log.Debug("✅ Bitcoin-style quantum proof validated",
		"blockNumber", header.Number.Uint64(),
		"difficulty", header.Difficulty)

	return nil
}
