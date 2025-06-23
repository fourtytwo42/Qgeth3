// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Bitcoin-Style Quantum Mining Implementation
// The world's first Bitcoin-style quantum proof-of-work consensus engine

package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
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

	// Bitcoin-style mining parameters optimized for quantum computation speeds
	QuantumTargetBits = 248        // Target space size (bits)
	TargetTimeSeconds = 12         // Target block time like Ethereum
	MaxNonceAttempts  = 0xFFFFFFFF // 4 billion attempts like Bitcoin

	// Ultra-fine difficulty precision for smooth adjustments
	DifficultyPrecision = 1000000 // 1e6 (6 decimal places) for precise fractional difficulties
	MinimumDifficulty   = 500     // 0.0005 in fixed-point (ultra-easy startup based on testing)

	// Ultra-easy startup mode constants - calibrated from real quantum mining data
	// 0.0000005 = mines every block, 0.0000006 = nearly impossible (only 20% harder!)
	StartupModeBlocks       = 20     // First 20 blocks use ultra-easy mode
	StartupTargetMultiplier = 100000 // 100000x easier targets for startup (0.000000005 effective difficulty)
	TransitionBlocks        = 100    // Gradual transition over 100 blocks for ultra-smooth scaling
)

// Advanced difficulty formatting with decimal precision
func FormatDifficulty(difficulty *big.Int) string {
	if difficulty == nil {
		return "0"
	}

	// Convert fixed-point to decimal representation
	precisionBig := big.NewInt(DifficultyPrecision)
	whole := new(big.Int).Div(difficulty, precisionBig)
	remainder := new(big.Int).Mod(difficulty, precisionBig)

	// Format with up to 6 decimal places, removing trailing zeros
	decimal := fmt.Sprintf("%06d", remainder.Uint64())
	decimal = strings.TrimRight(decimal, "0")
	if decimal == "" {
		return whole.String()
	}
	return fmt.Sprintf("%s.%s", whole.String(), decimal)
}

var (
	MaxQuantumTarget = new(big.Int).Lsh(big.NewInt(1), QuantumTargetBits) // 2^248 maximum target
	MinQuantumTarget = big.NewInt(1)                                      // Minimum target (hardest difficulty)
)

// GetStartupMaxTarget returns an ultra-easy target for startup blocks
func GetStartupMaxTarget(blockNumber uint64) *big.Int {
	if blockNumber <= StartupModeBlocks {
		// Ultra-easy startup mode: multiply max target by 1000
		startupTarget := new(big.Int).Set(MaxQuantumTarget)
		startupTarget.Mul(startupTarget, big.NewInt(StartupTargetMultiplier))

		log.Info("🚀 Using ultra-easy startup mode",
			"blockNumber", blockNumber,
			"startupMultiplier", StartupTargetMultiplier,
			"blocksRemaining", StartupModeBlocks-blockNumber+1)

		return startupTarget
	}

	// Gradual transition from ultra-easy to normal over next blocks
	if blockNumber <= StartupModeBlocks+TransitionBlocks {
		transitionProgress := int64(blockNumber - StartupModeBlocks)
		maxTransition := int64(TransitionBlocks)

		// Exponential transition from 100000x to 1x (ultra-smooth difficulty curve)
		// Given the extreme sensitivity (0.0000005 vs 0.0000006), we need very gradual scaling
		progressRatio := float64(transitionProgress) / float64(maxTransition)

		// Exponential decay to handle the massive multiplier range smoothly
		// multiplier = 100000 * (1/100000)^progressRatio = 100000 * e^(-ln(100000) * progressRatio)
		exponent := -11.5129 * progressRatio // ln(100000) ≈ 11.5129
		multiplierFloat := float64(StartupTargetMultiplier) * math.Exp(exponent)

		multiplier := int64(multiplierFloat)
		if multiplier < 1 {
			multiplier = 1
		}

		transitionTarget := new(big.Int).Set(MaxQuantumTarget)
		transitionTarget.Mul(transitionTarget, big.NewInt(multiplier))

		log.Info("🔄 Transitioning from startup mode",
			"blockNumber", blockNumber,
			"multiplier", multiplier,
			"progress", fmt.Sprintf("%.1f%%", progressRatio*100),
			"effectiveDifficulty", fmt.Sprintf("%.9f", 0.0005/float64(multiplier)))

		return transitionTarget
	}

	// Normal mode
	return MaxQuantumTarget
}

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

	// Use modulo to keep values positive and within target range
	maxHash := new(big.Int).Set(MaxQuantumTarget)
	quality.Mod(quality, maxHash) // This ensures 0 <= quality < MaxQuantumTarget

	return quality
}

// CalculateQuantumTarget converts fractional difficulty to quantum target using ASERT-Q
// Like Bitcoin: higher difficulty = lower target = harder to mine
// Uses real-time block timing to adjust target granularly
func CalculateQuantumTarget(difficulty *big.Int, blockNumber uint64, actualBlockTime int64) *big.Int {
	if difficulty.Cmp(big.NewInt(0)) <= 0 {
		difficulty = big.NewInt(MinimumDifficulty)
	}

	// Get base max target
	maxTarget := new(big.Int).Set(MaxQuantumTarget)

	// Bitcoin-style base target calculation: baseTarget = max_target / difficulty
	baseTarget := new(big.Int).Set(maxTarget)
	baseTarget.Div(baseTarget, difficulty)

	// Apply ASERT-Q multiplier for granular adjustment
	multiplier := CalculateASERTMultiplier(actualBlockTime, blockNumber)

	// Final target = baseTarget * multiplier
	target := new(big.Int).Mul(baseTarget, big.NewInt(multiplier))

	// Ensure target doesn't exceed maximum
	if target.Cmp(maxTarget) > 0 {
		target.Set(maxTarget)
	}

	log.Debug("🎯 ASERT-Q Target calculated",
		"difficulty", FormatDifficulty(difficulty),
		"baseTarget", baseTarget.String(),
		"multiplier", multiplier,
		"actualBlockTime", actualBlockTime,
		"finalTarget", target.String(),
		"targetHex", fmt.Sprintf("0x%064x", target),
		"blockNumber", blockNumber)

	return target
}

// CalculateASERTMultiplier computes the ASERT-Q multiplier for granular difficulty adjustment
// Uses exponential adjustment similar to Bitcoin's ASERT but optimized for quantum mining
func CalculateASERTMultiplier(actualBlockTime int64, blockNumber uint64) int64 {
	const (
		TargetBlockTime = 12   // seconds
		ASERTLambda     = 0.12 // adjustment rate (12% per block maximum)
		MinMultiplier   = 1
		MaxMultiplier   = 1000 // Maximum 1000x easier than base difficulty
	)

	// For first few blocks, use a startup multiplier to ensure smooth start
	if blockNumber <= 10 {
		return 100 // 100x easier for first 10 blocks
	}

	// If no timing data available, use moderate multiplier
	if actualBlockTime <= 0 {
		return 10
	}

	// Calculate time deviation from target
	timeDiff := actualBlockTime - TargetBlockTime

	// ASERT-Q formula: multiplier = e^(lambda * timeDiff / TargetBlockTime)
	// If blocks are slow (positive timeDiff): multiplier > 1 (easier)
	// If blocks are fast (negative timeDiff): multiplier < 1 (harder)
	exponent := ASERTLambda * float64(timeDiff) / float64(TargetBlockTime)
	multiplierFloat := math.Exp(exponent)

	// Convert to integer and clamp
	multiplier := int64(math.Max(float64(MinMultiplier), math.Min(float64(MaxMultiplier), multiplierFloat)))

	log.Info("🎯 ASERT-Q Multiplier calculated",
		"actualBlockTime", actualBlockTime,
		"targetBlockTime", TargetBlockTime,
		"timeDiff", timeDiff,
		"exponent", fmt.Sprintf("%.4f", exponent),
		"multiplier", multiplier,
		"blockNumber", blockNumber)

	return multiplier
}

// CheckQuantumProofTarget verifies if quantum proof meets target
// Returns true if proof quality < target (Bitcoin-style: lower is better)
func CheckQuantumProofTarget(outcomes []byte, proof []byte, qnonce uint64, target *big.Int) bool {
	quality := CalculateQuantumProofQuality(outcomes, proof, qnonce)

	// Bitcoin-style comparison: success when quality < target
	success := quality.Cmp(target) < 0

	// Enhanced logging with quantum-specific context
	log.Debug("🎯 Quantum target check DETAILED",
		"qnonce", qnonce,
		"quality", quality.String(),
		"target", target.String(),
		"success", success,
		"qualityHex", fmt.Sprintf("0x%064x", quality),
		"targetHex", fmt.Sprintf("0x%064x", target))

	return success
}

// CalculateNextDifficulty implements Bitcoin-style difficulty adjustment with quantum optimization
// Uses 6-decimal-place fixed-point arithmetic for ultra-fine granularity
func CalculateNextDifficulty(currentDifficulty *big.Int, actualTime, targetTime uint64) *big.Int {
	log.Info("🔗 Quantum-optimized difficulty adjustment",
		"currentDifficulty", FormatDifficulty(currentDifficulty),
		"actualTime", actualTime,
		"targetTime", targetTime)

	// Calculate the ratio of actual time to target time
	if actualTime == 0 {
		actualTime = 1 // Prevent division by zero
	}

	// Calculate new difficulty using fixed-point arithmetic
	// newDifficulty = currentDifficulty * targetTime / actualTime
	newDifficulty := new(big.Int).Set(currentDifficulty)
	newDifficulty.Mul(newDifficulty, big.NewInt(int64(targetTime)))
	newDifficulty.Div(newDifficulty, big.NewInt(int64(actualTime)))

	// Apply adjustment factor limits (max 4x change like Bitcoin)
	maxIncrease := new(big.Int).Set(currentDifficulty)
	maxIncrease.Mul(maxIncrease, big.NewInt(TargetAdjustmentFactor))

	maxDecrease := new(big.Int).Set(currentDifficulty)
	maxDecrease.Div(maxDecrease, big.NewInt(TargetAdjustmentFactor))

	if newDifficulty.Cmp(maxIncrease) > 0 {
		newDifficulty.Set(maxIncrease)
		log.Info("🔒 Difficulty increase clamped", "factor", TargetAdjustmentFactor)
	} else if newDifficulty.Cmp(maxDecrease) < 0 {
		newDifficulty.Set(maxDecrease)
		log.Info("🔒 Difficulty decrease clamped", "factor", TargetAdjustmentFactor)
	}

	// Enforce minimum difficulty for quantum mining
	minDiff := big.NewInt(MinimumDifficulty)
	if newDifficulty.Cmp(minDiff) < 0 {
		newDifficulty.Set(minDiff)
		log.Info("🔒 Difficulty clamped to quantum minimum", "minDifficulty", FormatDifficulty(minDiff))
	}

	// Calculate the ratio for logging
	ratio := float64(targetTime) / float64(actualTime)

	log.Info("✅ Quantum-optimized difficulty adjusted",
		"oldDifficulty", FormatDifficulty(currentDifficulty),
		"newDifficulty", FormatDifficulty(newDifficulty),
		"ratio", fmt.Sprintf("%.4f", ratio),
		"actualMinutes", float64(actualTime)/60,
		"targetMinutes", float64(targetTime)/60)

	return newDifficulty
}

// EstimateQuantumHashrate estimates the effective hashrate for quantum mining
// Converts quantum puzzle rate to equivalent traditional hashrate units
func EstimateQuantumHashrate(difficulty *big.Int, blockTime uint64) float64 {
	if blockTime == 0 {
		return 0
	}

	// Convert fixed-point difficulty to float
	diffFloat := new(big.Float).SetInt(difficulty)
	precisionFloat := new(big.Float).SetInt64(DifficultyPrecision)
	diffFloat.Quo(diffFloat, precisionFloat)

	// Estimate hashrate as difficulty / block_time_seconds
	blockTimeFloat, _ := diffFloat.Float64()
	hashrate := blockTimeFloat / float64(blockTime)

	return hashrate
}

// ShouldRetargetDifficulty checks if we're at a retarget block
func ShouldRetargetDifficulty(blockNumber uint64) bool {
	return blockNumber > 0 && blockNumber%RetargetBlocks == 0
}

// GetRetargetPeriodStart returns the starting block number for the current retarget period
func GetRetargetPeriodStart(blockNumber uint64) uint64 {
	if blockNumber < RetargetBlocks {
		return 0
	}
	periods := blockNumber / RetargetBlocks
	return periods * RetargetBlocks
}

// EstimateHashrate provides legacy compatibility for hashrate estimation
func EstimateHashrate(difficulty *big.Int, blockTime uint64) float64 {
	return EstimateQuantumHashrate(difficulty, blockTime)
}

// ValidateQuantumProofBitcoinStyle validates quantum proof using Bitcoin-style rules
func ValidateQuantumProofBitcoinStyle(header *types.Header) error {
	// Implementation would go here for full proof validation
	// This is a placeholder for the complete validation logic
	return nil
}

// ConvertLegacyDifficulty converts old integer difficulties to new quantum-optimized format
// This handles the transition from the old system to the new ultra-granular system
func ConvertLegacyDifficulty(oldDifficulty *big.Int) *big.Int {
	// Handle zero difficulty - should never happen but protect against it
	if oldDifficulty.Cmp(big.NewInt(0)) == 0 {
		log.Warn("🚨 Zero difficulty detected! Using minimum quantum difficulty",
			"minDifficulty", MinimumDifficulty)
		return big.NewInt(MinimumDifficulty)
	}

	// If difficulty is suspiciously high (> 1000), assume it's already in fixed-point
	thousandThreshold := big.NewInt(1000)
	if oldDifficulty.Cmp(thousandThreshold) > 0 {
		log.Info("🔄 Detected fixed-point difficulty, converting to quantum scale",
			"oldValue", oldDifficulty.String(),
			"oldFormatted", FormatDifficulty(oldDifficulty))

		// Convert from old 1e9 precision to new 1e6 precision
		// This shrinks the scale by 1000x making mining much more feasible
		oldPrecision := big.NewInt(1000000000)          // Old 1e9 precision
		newPrecision := big.NewInt(DifficultyPrecision) // New 1e6 precision

		// Convert: new_value = old_value * new_precision / old_precision
		converted := new(big.Int).Set(oldDifficulty)
		converted.Mul(converted, newPrecision)
		converted.Div(converted, oldPrecision)

		// Ensure minimum
		minDiff := big.NewInt(MinimumDifficulty)
		if converted.Cmp(minDiff) < 0 {
			converted.Set(minDiff)
		}

		log.Info("🎯 Converted to quantum-optimized difficulty",
			"newValue", converted.String(),
			"newFormatted", FormatDifficulty(converted),
			"reductionFactor", "1000x")

		return converted
	}

	// For small difficulties (like 0.0000005 → 500 in fixed-point), they should already be correct
	// The genesis creation converts 0.0000005 → 500, so just return as-is
	log.Info("🔄 Using pre-converted quantum difficulty",
		"value", oldDifficulty.String(),
		"formatted", FormatDifficulty(oldDifficulty))

	// Ensure minimum
	minDiff := big.NewInt(MinimumDifficulty)
	if oldDifficulty.Cmp(minDiff) < 0 {
		log.Warn("🚨 Difficulty below minimum! Clamping to minimum",
			"input", oldDifficulty.String(),
			"minimum", MinimumDifficulty)
		return big.NewInt(MinimumDifficulty)
	}

	return new(big.Int).Set(oldDifficulty)
}
