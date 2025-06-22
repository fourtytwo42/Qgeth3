// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Bitcoin-Style Quantum Mining Implementation
// The world's first Bitcoin-style quantum proof-of-work consensus engine

package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/big"
	"strings"

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

	// High-precision difficulty constants
	DifficultyPrecision = 1000000000 // 1e9 - 9 decimal places for fractional difficulty
)

// FormatDifficulty formats a fixed-point difficulty value for display
func FormatDifficulty(difficulty *big.Int) string {
	precision := big.NewInt(DifficultyPrecision)

	// Extract integer and fractional parts
	wholePart := new(big.Int).Div(difficulty, precision)
	fractionalPart := new(big.Int).Mod(difficulty, precision)

	// Format as decimal with up to 9 decimal places
	fractionalStr := fmt.Sprintf("%09d", fractionalPart.Uint64())
	// Trim trailing zeros
	fractionalStr = strings.TrimRight(fractionalStr, "0")

	if fractionalStr == "" {
		return wholePart.String()
	}
	return fmt.Sprintf("%s.%s", wholePart.String(), fractionalStr)
}

// Quantum mining target constants (Bitcoin-style)
// Using smaller maximum target for better nonce-level difficulty scaling
var (
	MaxQuantumTarget = new(big.Int).Lsh(big.NewInt(1), 240) // 2^240 maximum target (more reasonable scale)
	MinQuantumTarget = big.NewInt(1)                        // Minimum target (hardest difficulty)
)

// CalculateQuantumProofQuality computes the "quality" of a quantum proof
// This implements Bitcoin-style proof-of-work where lower quality values are better
func CalculateQuantumProofQuality(outcomes []byte, proof []byte, qnonce uint64) *big.Int {
	// Enhanced Bitcoin-style hash-based quality calculation
	// Multiple rounds of hashing for better nonce sensitivity
	h := sha256.New()

	// First, hash the nonce alone to create base entropy
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, qnonce)
	h.Write(nonceBytes)
	h.Write([]byte("QUANTUM_NONCE_SEED"))
	nonceSeed := h.Sum(nil)

	// Reset hasher and combine nonce seed with quantum data
	h.Reset()
	h.Write(nonceSeed)
	h.Write(outcomes)
	h.Write(proof)

	// Add nonce again for extra sensitivity
	h.Write(nonceBytes)

	// Multiple rounds of hashing for better distribution
	for i := 0; i < 3; i++ {
		h.Write([]byte(fmt.Sprintf("QUANTUM_ROUND_%d", i)))
		intermediate := h.Sum(nil)
		h.Reset()
		h.Write(intermediate)
		h.Write(nonceBytes) // Nonce in every round
	}

	// Final hash with entropy marker
	h.Write([]byte("QUANTUM_BITCOIN_FINAL"))
	hash := h.Sum(nil)

	// Convert hash to big integer
	quality := new(big.Int).SetBytes(hash)

	// FIXED: Use modulo to keep values positive and within target range
	// Instead of subtraction which can go negative, use modulo against max target
	maxHash := new(big.Int).Set(MaxQuantumTarget) // Use MaxQuantumTarget directly
	quality.Mod(quality, maxHash)                 // This ensures 0 <= quality < MaxQuantumTarget

	return quality
}

// CalculateQuantumTarget converts fractional difficulty to quantum target
// Like Bitcoin: higher difficulty = lower target = harder to mine
func CalculateQuantumTarget(difficulty *big.Int) *big.Int {
	if difficulty.Cmp(big.NewInt(0)) <= 0 {
		return new(big.Int).Set(MaxQuantumTarget)
	}

	// Bitcoin-style target calculation: target = max_target / difficulty
	// For fractional difficulties, we scale appropriately
	target := new(big.Int).Set(MaxQuantumTarget)
	target.Div(target, difficulty)

	// For very low difficulties, ensure target doesn't overflow
	if target.Cmp(MaxQuantumTarget) > 0 {
		target.Set(MaxQuantumTarget)
		log.Debug("🎯 Target clamped to maximum",
			"difficulty", difficulty.String(),
			"target", target.String())
	}

	// Allow very low targets for high difficulty (no minimum enforcement)
	log.Debug("🎯 Target calculated",
		"difficulty", difficulty.String(),
		"target", target.String(),
		"targetHex", fmt.Sprintf("0x%064x", target))

	return target
}

// CheckQuantumProofTarget verifies if quantum proof meets target
// Returns true if proof quality < target (Bitcoin-style: lower is better)
func CheckQuantumProofTarget(outcomes []byte, proof []byte, qnonce uint64, target *big.Int) bool {
	quality := CalculateQuantumProofQuality(outcomes, proof, qnonce)

	// Bitcoin-style comparison: success when quality < target
	success := quality.Cmp(target) < 0

	// Log the quality check for debugging
	log.Debug("🎯 Quantum proof quality check",
		"qnonce", qnonce,
		"quality", quality.String(),
		"target", target.String(),
		"success", success)

	return success
}

// CalculateNextDifficulty implements Bitcoin-style difficulty adjustment with high precision
// Uses fixed-point arithmetic with 9 decimal places for fractional difficulty
func CalculateNextDifficulty(currentDifficulty *big.Int, actualTime, targetTime uint64) *big.Int {
	log.Info("🔗 Bitcoin-style difficulty adjustment",
		"currentDifficulty", FormatDifficulty(currentDifficulty),
		"actualTime", actualTime,
		"targetTime", targetTime)

	// Calculate the ratio of actual time to target time
	if actualTime == 0 {
		actualTime = 1 // Prevent division by zero
	}

	// High-precision calculation: newDifficulty = currentDifficulty * targetTime / actualTime
	// Using fixed-point arithmetic to preserve fractional precision
	newDifficulty := new(big.Int).Set(currentDifficulty)
	newDifficulty.Mul(newDifficulty, big.NewInt(int64(targetTime)))
	newDifficulty.Mul(newDifficulty, big.NewInt(DifficultyPrecision))
	newDifficulty.Div(newDifficulty, big.NewInt(int64(actualTime)))
	newDifficulty.Div(newDifficulty, big.NewInt(DifficultyPrecision))

	// Limit adjustment to 4x up or down (like Bitcoin)
	maxAdjustment := new(big.Int).Mul(currentDifficulty, big.NewInt(TargetAdjustmentFactor))
	minAdjustment := new(big.Int).Div(currentDifficulty, big.NewInt(TargetAdjustmentFactor))

	if newDifficulty.Cmp(maxAdjustment) > 0 {
		newDifficulty.Set(maxAdjustment)
		log.Info("🔼 Difficulty capped at maximum adjustment", "newDifficulty", FormatDifficulty(newDifficulty))
	}
	if newDifficulty.Cmp(minAdjustment) < 0 {
		newDifficulty.Set(minAdjustment)
		log.Info("🔽 Difficulty capped at minimum adjustment", "newDifficulty", FormatDifficulty(newDifficulty))
	}

	// Allow very low difficulty for testing - comment out minimum enforcement
	// minDiff := big.NewInt(DifficultyPrecision)
	// if newDifficulty.Cmp(minDiff) < 0 {
	//	newDifficulty.Set(minDiff)
	// }

	ratio := float64(targetTime) / float64(actualTime)
	log.Info("✅ Bitcoin-style difficulty adjusted",
		"oldDifficulty", FormatDifficulty(currentDifficulty),
		"newDifficulty", FormatDifficulty(newDifficulty),
		"ratio", ratio)

	return newDifficulty
}

// ShouldRetargetDifficulty checks if it's time for difficulty adjustment
func ShouldRetargetDifficulty(blockNumber uint64) bool {
	return blockNumber > 0 && blockNumber%RetargetBlocks == 0
}

// GetRetargetPeriodStart returns the block number where the current retarget period started
func GetRetargetPeriodStart(blockNumber uint64) uint64 {
	if blockNumber < RetargetBlocks {
		return 1 // Start from block 1, not genesis block 0
	}
	// For retarget block, we want the START of the period, not the retarget block itself
	// Block 100 should look back to block 1 (first retarget: 1-100)
	// Block 200 should look back to block 101 (second retarget: 101-200)
	periodStart := ((blockNumber-1)/RetargetBlocks)*RetargetBlocks + 1

	// Ensure we never return block 0 (genesis)
	if periodStart == 0 {
		periodStart = 1
	}

	return periodStart
}

// EstimateHashrate calculates the network hashrate in puzzles per second
func EstimateHashrate(difficulty *big.Int, blockTime uint64) float64 {
	if blockTime == 0 {
		return 0
	}

	// Convert fractional difficulty to float for calculation
	diffFloat := new(big.Float).SetInt(difficulty)
	precisionFloat := new(big.Float).SetInt64(DifficultyPrecision)
	diffFloat.Quo(diffFloat, precisionFloat) // Convert from fixed-point to decimal

	// Hashrate = difficulty * puzzles_per_block / block_time
	puzzlesPerBlock := float64(DefaultLNet) // 48 puzzles
	hashrate := new(big.Float).Mul(diffFloat, big.NewFloat(puzzlesPerBlock))
	hashrate.Quo(hashrate, big.NewFloat(float64(blockTime)))

	result, _ := hashrate.Float64()
	return result
}

// ValidateQuantumProofBitcoinStyle validates a complete quantum proof using Bitcoin-style validation
func ValidateQuantumProofBitcoinStyle(header *types.Header) error {
	// Check that v0.9-rc3-hw0 quantum fields are present
	if header.QBits == nil || header.TCount == nil || header.LNet == nil {
		return ErrMissingQuantumFields
	}

	// Check that v0.9-rc3-hw0 proof fields are present
	if header.OutcomeRoot == nil || header.GateHash == nil || header.ProofRoot == nil {
		return ErrInvalidQuantumProof
	}

	// Calculate target from difficulty
	target := CalculateQuantumTarget(header.Difficulty)

	// For v0.9-rc3-hw0, we check proof quality based on the quantum blob fields
	h := sha256.New()
	h.Write(header.OutcomeRoot.Bytes())
	h.Write(header.GateHash.Bytes())
	h.Write(header.ProofRoot.Bytes())
	if len(header.BranchNibbles) > 0 {
		h.Write(header.BranchNibbles)
	}

	proofHash := h.Sum(nil)
	quality := new(big.Int).SetBytes(proofHash)

	// Verify proof meets target (Bitcoin-style: quality < target for success)
	if quality.Cmp(target) >= 0 {
		return ErrInvalidQuantumProof
	}

	log.Debug("✅ v0.9-rc3-hw0 quantum proof validated",
		"blockNumber", header.Number.Uint64(),
		"difficulty", header.Difficulty,
		"quality", quality.String(),
		"target", target.String())

	return nil
}
