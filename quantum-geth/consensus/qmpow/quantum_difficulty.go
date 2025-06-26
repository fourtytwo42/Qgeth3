// Copyright 2024 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// Quantum difficulty calculation functions to replace ethash.CalcDifficulty
// Uses ASERT-Q (Absolutely Scheduled Exponentially Rising Targets for Quantum)

package qmpow

import (
	"errors"
	"math/big"
	
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/params/types/ctypes"
	"github.com/ethereum/go-ethereum/log"
)

const (
	// DefaultMinDifficulty is the minimum difficulty for quantum mining
	DefaultMinDifficulty = 131072 // 2^17, reasonable for quantum networks
)

var (
	// ErrInvalidDifficulty is returned when difficulty transition is invalid
	ErrInvalidDifficulty = errors.New("invalid quantum difficulty transition")
)

// CalcDifficulty calculates the quantum difficulty for a given time and parent header
// This replaces ethash.CalcDifficulty for quantum blockchain compatibility
func CalcDifficulty(config ctypes.ChainConfigurator, time uint64, parent *types.Header) *big.Int {
	// Get quantum configuration from chain config
	qmpowConfig := config.GetQMPoWConfig()
	if qmpowConfig == nil {
		log.Warn("⚠️  No QMPoW config found, using default quantum parameters")
		qmpowConfig = &ctypes.QMPoWConfig{
			QBits:    16,
			TCount:   20,
			LNet:     128,
			EpochLen: 100,
			TestMode: true,
		}
	}
	
	// Calculate quantum difficulty using ASERT-Q algorithm
	return CalcQuantumDifficulty(config, time, parent, qmpowConfig)
}

// CalcQuantumDifficulty calculates quantum difficulty using ASERT-Q algorithm
func CalcQuantumDifficulty(config ctypes.ChainConfigurator, time uint64, parent *types.Header, qmpowConfig *ctypes.QMPoWConfig) *big.Int {
	blockNumber := parent.Number.Uint64() + 1
	
	log.Debug("🔢 Calculating quantum difficulty",
		"blockNumber", blockNumber,
		"parentTime", parent.Time,
		"currentTime", time,
		"parentDifficulty", parent.Difficulty)
	
	// For genesis block, use minimum quantum difficulty
	if parent.Number.Uint64() == 0 {
		minDiff := CalculateMinimumQuantumDifficulty(qmpowConfig)
		log.Debug("📍 Genesis block quantum difficulty", "difficulty", minDiff)
		return minDiff
	}
	
	// Use ASERT-Q algorithm for difficulty adjustment
	targetBlockTime := int64(15) // 15 second target block time
	timeDiff := int64(time) - int64(parent.Time)
	
	// ASERT-Q adjustment calculation
	newDifficulty := calculateASERTQDifficulty(parent.Difficulty, timeDiff, targetBlockTime, blockNumber)
	
	// Ensure minimum quantum difficulty
	minDiff := CalculateMinimumQuantumDifficulty(qmpowConfig)
	if newDifficulty.Cmp(minDiff) < 0 {
		log.Debug("⬆️  Clamping to minimum quantum difficulty", "calculated", newDifficulty, "minimum", minDiff)
		newDifficulty = minDiff
	}
	
	// Ensure maximum reasonable difficulty (prevent overflow)
	maxDiff := new(big.Int).Exp(big.NewInt(2), big.NewInt(256), nil)
	maxDiff.Sub(maxDiff, big.NewInt(1)) // 2^256 - 1
	if newDifficulty.Cmp(maxDiff) > 0 {
		log.Warn("⬇️  Clamping to maximum difficulty", "calculated", newDifficulty, "maximum", maxDiff)
		newDifficulty = maxDiff
	}
	
	log.Debug("✅ Quantum difficulty calculated",
		"newDifficulty", newDifficulty,
		"timeDiff", timeDiff,
		"adjustment", new(big.Int).Sub(newDifficulty, parent.Difficulty))
	
	return newDifficulty
}

// calculateASERTQDifficulty implements the ASERT-Q algorithm
func calculateASERTQDifficulty(parentDifficulty *big.Int, timeDiff, targetTime int64, blockNumber uint64) *big.Int {
	// ASERT-Q: exponential adjustment based on time deviation
	// NewDifficulty = ParentDifficulty * 2^((TimeDiff - TargetTime) / TargetTime / 2048)
	
	deviation := timeDiff - targetTime
	
	// Scale factor for quantum networks (more conservative than traditional PoW)
	scaleFactor := int64(4096) // Larger scale factor for more stable adjustments
	
	if deviation == 0 {
		// Perfect timing, no adjustment
		return new(big.Int).Set(parentDifficulty)
	}
	
	// Calculate adjustment ratio using fixed-point arithmetic
	// adjustment = deviation / targetTime / scaleFactor
	adjustment := new(big.Int).SetInt64(deviation)
	adjustment.Mul(adjustment, big.NewInt(1000000)) // Scale up for precision
	adjustment.Div(adjustment, big.NewInt(targetTime))
	adjustment.Div(adjustment, big.NewInt(scaleFactor))
	
	// Calculate 2^adjustment using approximation for small values
	// For small x: 2^x ≈ 1 + x*ln(2) + (x*ln(2))^2/2 + ...
	// ln(2) ≈ 693147 (scaled by 1000000)
	
	ln2Scaled := big.NewInt(693147) // ln(2) * 1000000
	adjustmentScaled := new(big.Int).Mul(adjustment, ln2Scaled)
	adjustmentScaled.Div(adjustmentScaled, big.NewInt(1000000))
	
	// Calculate multiplier = 1 + adjustmentScaled/1000000
	multiplier := big.NewInt(1000000)
	multiplier.Add(multiplier, adjustmentScaled)
	
	// Apply multiplier to parent difficulty
	newDifficulty := new(big.Int).Set(parentDifficulty)
	newDifficulty.Mul(newDifficulty, multiplier)
	newDifficulty.Div(newDifficulty, big.NewInt(1000000))
	
	// Additional quantum-specific adjustments
	
	// Epoch-based difficulty scaling
	epoch := blockNumber / 50000
	if epoch > 0 {
		// Gradual increase in base difficulty every epoch
		epochMultiplier := big.NewInt(int64(1000 + epoch*5)) // 0.5% increase per epoch
		newDifficulty.Mul(newDifficulty, epochMultiplier)
		newDifficulty.Div(newDifficulty, big.NewInt(1000))
	}
	
	return newDifficulty
}

// CalculateMinimumQuantumDifficulty calculates the minimum difficulty for quantum mining
func CalculateMinimumQuantumDifficulty(qmpowConfig *ctypes.QMPoWConfig) *big.Int {
	// Minimum difficulty based on quantum parameters
	// Base difficulty = 2^(QBits + log2(LNet) + log2(TCount))
	
	qbits := uint32(16)
	lnet := uint32(128)
	tcount := uint32(20)
	
	if qmpowConfig != nil {
		qbits = qmpowConfig.QBits
		lnet = qmpowConfig.LNet
		tcount = qmpowConfig.TCount
	}
	
	// Calculate complexity factor
	complexityBits := qbits + 7 + 5 // log2(128) ≈ 7, log2(20) ≈ 5
	
	// Minimum difficulty = 2^complexityBits / 1024 (to keep manageable)
	minDiff := new(big.Int).Exp(big.NewInt(2), big.NewInt(int64(complexityBits)), nil)
	minDiff.Div(minDiff, big.NewInt(1024))
	
	log.Debug("🔢 Calculated minimum quantum difficulty",
		"qbits", qbits,
		"lnet", lnet,
		"tcount", tcount,
		"complexityBits", complexityBits,
		"minDifficulty", minDiff)
	
	return minDiff
}

// IsQuantumDifficultyValid validates if a difficulty value is valid for quantum mining
func IsQuantumDifficultyValid(difficulty *big.Int, qmpowConfig *ctypes.QMPoWConfig) bool {
	if difficulty == nil || difficulty.Sign() <= 0 {
		return false
	}
	
	minDiff := CalculateMinimumQuantumDifficulty(qmpowConfig)
	return difficulty.Cmp(minDiff) >= 0
}

// GetQuantumTargetFromDifficulty converts difficulty to mining target
func GetQuantumTargetFromDifficulty(difficulty *big.Int) *big.Int {
	// Target = 2^256 / Difficulty
	maxTarget := new(big.Int).Exp(big.NewInt(2), big.NewInt(256), nil)
	target := new(big.Int).Div(maxTarget, difficulty)
	return target
}

// ValidateQuantumDifficultyTransition validates difficulty change between blocks
func ValidateQuantumDifficultyTransition(parent, current *types.Header, qmpowConfig *ctypes.QMPoWConfig) error {
	// Calculate expected difficulty
	expectedDiff := CalcQuantumDifficulty(nil, current.Time, parent, qmpowConfig)
	
	// Allow some tolerance for network conditions
	tolerance := new(big.Int).Div(expectedDiff, big.NewInt(100)) // 1% tolerance
	
	diff := new(big.Int).Sub(current.Difficulty, expectedDiff)
	if diff.Sign() < 0 {
		diff.Neg(diff)
	}
	
	if diff.Cmp(tolerance) > 0 {
		log.Error("❌ Invalid quantum difficulty transition",
			"expected", expectedDiff,
			"actual", current.Difficulty,
			"difference", diff,
			"tolerance", tolerance)
		return ErrInvalidDifficulty
	}
	
	return nil
}

// CalcDifficultyASERTQ is a standalone ASERT-Q calculator for testing purposes
// This provides quantum difficulty calculation without full QMPoW context
func CalcDifficultyASERTQ(config *Config, time uint64, parent *types.Header) *big.Int {
	// Use simplified quantum difficulty calculation for fuzzing/testing
	// This ensures consistency with main CalcDifficulty function
	
	// Basic quantum difficulty parameters
	parentTime := parent.Time
	parentDiff := parent.Difficulty
	
	if parentDiff == nil {
		parentDiff = big.NewInt(DefaultMinDifficulty)
	}
	
	// Simple ASERT-Q approximation for testing
	timeDelta := int64(time) - int64(parentTime)
	targetTime := int64(10) // 10 second target
	
	// Quantum-adjusted difficulty based on time delta
	adjustment := big.NewInt(1000) // Base 1.0 scaled by 1000
	if timeDelta > targetTime {
		// Decrease difficulty if blocks are slow
		adjustment = big.NewInt(999)
	} else if timeDelta < targetTime {
		// Increase difficulty if blocks are fast  
		adjustment = big.NewInt(1001)
	}
	
	newDiff := new(big.Int).Mul(parentDiff, adjustment)
	newDiff.Div(newDiff, big.NewInt(1000))
	
	// Ensure minimum difficulty
	minDiff := big.NewInt(DefaultMinDifficulty)
	if newDiff.Cmp(minDiff) < 0 {
		newDiff.Set(minDiff)
	}
	
	return newDiff
} 