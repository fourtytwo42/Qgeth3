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
	TargetAdjustmentFactor = 2   // Maximum adjustment per retarget (max 1x increase/decrease)
	RetargetBlocks         = 50  // Retarget every 50 blocks (~10 minutes at 12s blocks)

	// Bitcoin-style mining parameters optimized for quantum computation speeds
	QuantumTargetBits = 248        // Target space size (bits)
	MaxNonceAttempts  = 0xFFFFFFFF // 4 billion attempts like Bitcoin

	// Ultra-fine difficulty precision for smooth adjustments
	DifficultyPrecision = 1000000 // 1e6 (6 decimal places) for precise fractional difficulties
	MinimumDifficulty   = 500     // 0.0005 in fixed-point (ultra-easy startup based on testing)

	// ASERT parameters for smooth per-block adjustments
	ASERTFixedPoint = 16 // Q16 fixed-point arithmetic
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
	// Quantum-optimized target range for better mining balance
	// Using 2^256 - 1 (full 256-bit range) and scaling appropriately
	MaxQuantumTarget = new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1)) // 2^256 - 1 (max 256-bit)
	MinQuantumTarget = new(big.Int).Lsh(big.NewInt(1), 200)                                  // 2^200 (minimum reasonable target)
	ASERTRadix       = new(big.Int).Lsh(big.NewInt(1), ASERTFixedPoint)                      // 2^16 for Q16 arithmetic
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

	// Convert hash to big integer (full 256-bit range)
	quality := new(big.Int).SetBytes(hash)

	// Hash already gives us a 256-bit value, which is perfect for our target range
	// No need to mod since we want the full distribution
	return quality
}

// DifficultyToTarget converts difficulty to Bitcoin-style target
// Like Bitcoin: target = max_target / difficulty
// But adjusted for quantum mining's unique characteristics
func DifficultyToTarget(difficulty *big.Int) *big.Int {
	if difficulty.Cmp(big.NewInt(0)) <= 0 {
		difficulty = big.NewInt(MinimumDifficulty)
	}

	// Quantum-optimized target calculation
	// For quantum mining, we need much larger targets than traditional Bitcoin
	// Use a quantum-specific scaling factor to make targets achievable
	quantumScalingFactor := big.NewInt(100000000) // 100M scaling for quantum mining

	// target = (max_target / difficulty) * scaling_factor
	target := new(big.Int).Set(MaxQuantumTarget)
	target.Div(target, difficulty)
	target.Mul(target, quantumScalingFactor)

	// Ensure target doesn't exceed maximum or go below minimum
	if target.Cmp(MaxQuantumTarget) > 0 {
		target.Set(MaxQuantumTarget)
	}
	if target.Cmp(MinQuantumTarget) < 0 {
		target.Set(MinQuantumTarget)
	}

	return target
}

// CalculateQuantumTarget converts fractional difficulty to quantum target using ASERT
// Like Bitcoin: higher difficulty = lower target = harder to mine
// Uses real-time block timing to adjust target smoothly per block
func CalculateQuantumTarget(difficulty *big.Int, blockNumber uint64, actualBlockTime int64) *big.Int {
	if difficulty.Cmp(big.NewInt(0)) <= 0 {
		difficulty = big.NewInt(MinimumDifficulty)
	}

	// Get base target from difficulty
	baseTarget := DifficultyToTarget(difficulty)

	// Apply ASERT adjustment for smooth per-block difficulty changes
	adjustedTarget := ApplyASERTAdjustment(baseTarget, actualBlockTime, blockNumber)

	log.Debug("🎯 ASERT-Q Target calculated",
		"difficulty", FormatDifficulty(difficulty),
		"baseTarget", baseTarget.String(),
		"actualBlockTime", actualBlockTime,
		"finalTarget", adjustedTarget.String(),
		"targetHex", fmt.Sprintf("0x%064x", adjustedTarget),
		"blockNumber", blockNumber)

	return adjustedTarget
}

// ApplyASERTAdjustment applies Bitcoin Cash style ASERT adjustment to target
// Formula: newTarget = oldTarget * 2^((actualTime - targetTime) / halfLife)
func ApplyASERTAdjustment(baseTarget *big.Int, actualBlockTime int64, blockNumber uint64) *big.Int {
	// For first few blocks, don't adjust (let it stabilize)
	if blockNumber <= 5 {
		log.Info("🚀 Early block - no ASERT adjustment", "blockNumber", blockNumber)
		return baseTarget
	}

	// If no timing data, return base target
	if actualBlockTime <= 0 {
		actualBlockTime = int64(TargetBlockTime) // Use target time as default
	}

	// Calculate time difference
	timeDiff := actualBlockTime - int64(TargetBlockTime)

	// ASERT formula: exponent = timeDiff / halfLife
	// We use Q16 fixed-point: exponent_q16 = (timeDiff << 16) / halfLife
	exponentQ16 := (int64(timeDiff) << ASERTFixedPoint) / int64(ASERTHalfLife)

	// Apply 2^exponent using fixed-point arithmetic
	adjustedTarget := ApplyPowerOfTwo(baseTarget, exponentQ16)

	// Clamp to valid range
	if adjustedTarget.Cmp(MaxQuantumTarget) > 0 {
		adjustedTarget.Set(MaxQuantumTarget)
	}
	if adjustedTarget.Cmp(MinQuantumTarget) < 0 {
		adjustedTarget.Set(MinQuantumTarget)
	}

	direction := "STABLE"
	if timeDiff > 1 {
		direction = "EASIER"
	} else if timeDiff < -1 {
		direction = "HARDER"
	}

	log.Info("🎯 ASERT-Q adjustment applied",
		"actualBlockTime", actualBlockTime,
		"targetBlockTime", TargetBlockTime,
		"timeDiff", timeDiff,
		"direction", direction,
		"blockNumber", blockNumber)

	return adjustedTarget
}

// ApplyPowerOfTwo applies 2^(exponent_q16) to a big.Int using Q16 fixed-point
// This implements smooth exponential adjustment like Bitcoin Cash ASERT
func ApplyPowerOfTwo(value *big.Int, exponentQ16 int64) *big.Int {
	if exponentQ16 == 0 {
		return new(big.Int).Set(value)
	}

	// Handle negative exponents (making target smaller/harder)
	isNegative := exponentQ16 < 0
	if isNegative {
		exponentQ16 = -exponentQ16
	}

	// Extract integer and fractional parts
	integerPart := exponentQ16 >> ASERTFixedPoint
	fractionalPart := exponentQ16 & ((1 << ASERTFixedPoint) - 1)

	result := new(big.Int).Set(value)

	// Apply integer part: multiply by 2^integerPart
	if integerPart > 0 {
		// Clamp integer part to prevent overflow
		if integerPart > 10 {
			integerPart = 10 // Max 1024x change
		}
		shift := big.NewInt(1)
		shift.Lsh(shift, uint(integerPart))
		result.Mul(result, shift)
	}

	// Apply fractional part using approximation
	// 2^(x/65536) ≈ 1 + x*ln(2)/65536 for small x
	if fractionalPart > 0 {
		// ln(2) in Q16: 0.693147 * 65536 ≈ 45426
		fracAdjustment := (fractionalPart * 45426) >> ASERTFixedPoint
		adjustment := new(big.Int).SetInt64(fracAdjustment)
		adjustment.Mul(adjustment, result)
		adjustment.Div(adjustment, ASERTRadix)
		result.Add(result, adjustment)
	}

	// Handle negative exponent (division)
	if isNegative && result.Cmp(big.NewInt(1)) > 0 {
		// For negative exponent: result = value / (2^|exponent|)
		// We computed 2^|exponent| above, so divide original value by result
		divisor := new(big.Int).Set(result)
		result.Set(value)
		result.Div(result, divisor)
		// Prevent target from becoming too small
		if result.Cmp(MinQuantumTarget) < 0 {
			result.Set(MinQuantumTarget)
		}
	}

	return result
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

	// For small difficulties, convert to fixed-point
	converted := new(big.Int).Set(oldDifficulty)
	converted.Mul(converted, big.NewInt(DifficultyPrecision))

	// Ensure minimum
	minDiff := big.NewInt(MinimumDifficulty)
	if converted.Cmp(minDiff) < 0 {
		converted.Set(minDiff)
	}

	log.Info("🔄 Converted legacy difficulty to quantum-optimized format",
		"oldDifficulty", oldDifficulty.String(),
		"newDifficulty", FormatDifficulty(converted))

	return converted
}
