// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"fmt"
	"math"
	"math/big"
	"time"

	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// ASERT-Q Difficulty Adjustment Algorithm Constants
const (
	// Target block time: 12 seconds
	ASERTTargetBlockTime = 12 * time.Second

	// ASERT adjustment half-life: 150 seconds (12.5 blocks)
	ASERTHalfLifeTime = 150 * time.Second

	// Maximum difficulty adjustment per block: ±10%
	MaxAdjustmentFactor = 1.1
	MinAdjustmentFactor = 1.0 / MaxAdjustmentFactor

	// Genesis difficulty shock protection: λ=0.2 for initial 5000 blocks
	GenesisProtectionBlocks = 5000
	GenesisLambda           = 0.2

	// Initial difficulty target (high value = low difficulty for testing)
	InitialDifficultyTarget = uint64(0x1000000000000000) // 2^60
)

// ASERTQDifficulty implements the ASERT-Q difficulty adjustment algorithm
type ASERTQDifficulty struct {
	targetBlockTime time.Duration
	halfLife        time.Duration
	maxAdjustment   float64
	minAdjustment   float64
}

// DifficultyAdjustment represents a difficulty adjustment calculation
type DifficultyAdjustment struct {
	PreviousTarget   *big.Int      // Previous difficulty target
	NewTarget        *big.Int      // New difficulty target
	ActualTime       time.Duration // Actual time since previous block
	ExpectedTime     time.Duration // Expected time (target block time)
	TimeDelta        time.Duration // Difference (actual - expected)
	AdjustmentFactor float64       // Calculated adjustment factor
	ClampedFactor    float64       // Factor after clamping
	BlockHeight      uint64        // Block height for this adjustment
	IsGenesis        bool          // Whether genesis protection applies
}

// NewASERTQDifficulty creates a new ASERT-Q difficulty calculator
func NewASERTQDifficulty() *ASERTQDifficulty {
	return &ASERTQDifficulty{
		targetBlockTime: ASERTTargetBlockTime,
		halfLife:        ASERTHalfLifeTime,
		maxAdjustment:   MaxAdjustmentFactor,
		minAdjustment:   MinAdjustmentFactor,
	}
}

// CalculateNextDifficulty implements the ASERT-Q algorithm:
// newTarget = oldTarget × 2^((t_actual - 12s) / 150s)
// with ±10% per-block clamping and genesis protection
func (asert *ASERTQDifficulty) CalculateNextDifficulty(
	chain consensus.ChainHeaderReader,
	header *types.Header,
) *DifficultyAdjustment {

	// Get parent header for timing calculation
	parent := chain.GetHeaderByHash(header.ParentHash)
	if parent == nil {
		// Genesis block - use initial difficulty
		initialTarget := new(big.Int).SetUint64(InitialDifficultyTarget)
		return &DifficultyAdjustment{
			PreviousTarget:   initialTarget,
			NewTarget:        initialTarget,
			ActualTime:       0,
			ExpectedTime:     asert.targetBlockTime,
			TimeDelta:        0,
			AdjustmentFactor: 1.0,
			ClampedFactor:    1.0,
			BlockHeight:      header.Number.Uint64(),
			IsGenesis:        true,
		}
	}

	// Calculate actual time since parent block
	actualTime := time.Duration(header.Time-parent.Time) * time.Second
	expectedTime := asert.targetBlockTime
	timeDelta := actualTime - expectedTime

	// Get previous difficulty target
	var previousTarget *big.Int
	if parent.Difficulty != nil {
		previousTarget = new(big.Int).Set(parent.Difficulty)
	} else {
		previousTarget = new(big.Int).SetUint64(InitialDifficultyTarget)
	}

	blockHeight := header.Number.Uint64()

	// Apply genesis protection for first 5000 blocks
	var adjustmentFactor float64
	isGenesis := blockHeight <= GenesisProtectionBlocks

	if isGenesis {
		// Genesis protection: use λ=0.2 damping factor
		normalFactor := asert.calculateRawAdjustmentFactor(timeDelta)
		adjustmentFactor = 1.0 + GenesisLambda*(normalFactor-1.0)
	} else {
		// Normal ASERT-Q calculation
		adjustmentFactor = asert.calculateRawAdjustmentFactor(timeDelta)
	}

	// Apply ±10% per-block clamping
	clampedFactor := asert.clampAdjustmentFactor(adjustmentFactor)

	// Calculate new target: newTarget = oldTarget × clampedFactor
	newTarget := new(big.Int).Set(previousTarget)
	if clampedFactor != 1.0 {
		// Convert to float for calculation
		targetFloat := new(big.Float).SetInt(previousTarget)
		factorFloat := big.NewFloat(clampedFactor)
		newTargetFloat := new(big.Float).Mul(targetFloat, factorFloat)

		// Convert back to integer
		newTarget, _ = newTargetFloat.Int(nil)
	}

	adjustment := &DifficultyAdjustment{
		PreviousTarget:   previousTarget,
		NewTarget:        newTarget,
		ActualTime:       actualTime,
		ExpectedTime:     expectedTime,
		TimeDelta:        timeDelta,
		AdjustmentFactor: adjustmentFactor,
		ClampedFactor:    clampedFactor,
		BlockHeight:      blockHeight,
		IsGenesis:        isGenesis,
	}

	log.Debug("🎯 ASERT-Q difficulty adjustment",
		"block", blockHeight,
		"actual_time", actualTime,
		"expected_time", expectedTime,
		"time_delta", timeDelta,
		"raw_factor", adjustmentFactor,
		"clamped_factor", clampedFactor,
		"previous_target", previousTarget.String(),
		"new_target", newTarget.String(),
		"genesis_protection", isGenesis)

	return adjustment
}

// calculateRawAdjustmentFactor computes 2^((t_actual - 12s) / 150s)
func (asert *ASERTQDifficulty) calculateRawAdjustmentFactor(timeDelta time.Duration) float64 {
	// Convert time delta to seconds
	timeDeltaSeconds := timeDelta.Seconds()
	halfLifeSeconds := asert.halfLife.Seconds()

	// Calculate exponent: (t_actual - target) / half_life
	exponent := timeDeltaSeconds / halfLifeSeconds

	// Return 2^exponent
	return math.Pow(2.0, exponent)
}

// clampAdjustmentFactor applies ±10% per-block clamping
func (asert *ASERTQDifficulty) clampAdjustmentFactor(factor float64) float64 {
	if factor > asert.maxAdjustment {
		return asert.maxAdjustment
	}
	if factor < asert.minAdjustment {
		return asert.minAdjustment
	}
	return factor
}

// ValidateBlockTime validates that a block time is reasonable
func (asert *ASERTQDifficulty) ValidateBlockTime(
	header *types.Header,
	parent *types.Header,
) error {

	if parent == nil {
		// Genesis block - no validation needed
		return nil
	}

	actualTime := time.Duration(header.Time-parent.Time) * time.Second

	// Block time must be positive
	if actualTime <= 0 {
		return fmt.Errorf("invalid block time: %v (must be positive)", actualTime)
	}

	// Block time must not be too far in the future (max 2 hours)
	maxBlockTime := 2 * time.Hour
	if actualTime > maxBlockTime {
		return fmt.Errorf("block time too large: %v (max %v)", actualTime, maxBlockTime)
	}

	return nil
}

// GetTargetBlockTime returns the target block time
func (asert *ASERTQDifficulty) GetTargetBlockTime() time.Duration {
	return asert.targetBlockTime
}

// GetHalfLife returns the ASERT half-life
func (asert *ASERTQDifficulty) GetHalfLife() time.Duration {
	return asert.halfLife
}

// EstimateHashrate estimates network hashrate from difficulty and block time
func (asert *ASERTQDifficulty) EstimateHashrate(
	difficulty *big.Int,
	actualBlockTime time.Duration,
) *big.Float {

	if difficulty == nil || actualBlockTime <= 0 {
		return big.NewFloat(0)
	}

	// Hashrate = Difficulty / BlockTime
	difficultyFloat := new(big.Float).SetInt(difficulty)
	blockTimeFloat := big.NewFloat(actualBlockTime.Seconds())

	hashrate := new(big.Float).Quo(difficultyFloat, blockTimeFloat)
	return hashrate
}

// DifficultyStats represents difficulty adjustment statistics
type DifficultyStats struct {
	CurrentTarget      *big.Int      // Current difficulty target
	PreviousTarget     *big.Int      // Previous difficulty target
	TargetBlockTime    time.Duration // Target block time (12s)
	ActualBlockTime    time.Duration // Actual time of last block
	AdjustmentFactor   float64       // Last adjustment factor
	EstimatedHashrate  *big.Float    // Estimated network hashrate
	BlockHeight        uint64        // Current block height
	GenesisProtection  bool          // Whether genesis protection is active
	TimeSinceLastBlock time.Duration // Time since last block
}

// GetDifficultyStats returns comprehensive difficulty statistics
func (asert *ASERTQDifficulty) GetDifficultyStats(
	chain consensus.ChainHeaderReader,
	header *types.Header,
) *DifficultyStats {

	adjustment := asert.CalculateNextDifficulty(chain, header)
	hashrate := asert.EstimateHashrate(adjustment.NewTarget, adjustment.ActualTime)

	return &DifficultyStats{
		CurrentTarget:      adjustment.NewTarget,
		PreviousTarget:     adjustment.PreviousTarget,
		TargetBlockTime:    asert.targetBlockTime,
		ActualBlockTime:    adjustment.ActualTime,
		AdjustmentFactor:   adjustment.ClampedFactor,
		EstimatedHashrate:  hashrate,
		BlockHeight:        adjustment.BlockHeight,
		GenesisProtection:  adjustment.IsGenesis,
		TimeSinceLastBlock: adjustment.ActualTime,
	}
}

// DifficultyHistory represents historical difficulty data
type DifficultyHistory struct {
	Entries []DifficultyHistoryEntry
	Window  int // Number of blocks to track
}

// DifficultyHistoryEntry represents one block's difficulty data
type DifficultyHistoryEntry struct {
	BlockHeight      uint64        // Block height
	Timestamp        uint64        // Block timestamp
	Difficulty       *big.Int      // Block difficulty
	BlockTime        time.Duration // Time since previous block
	AdjustmentFactor float64       // Difficulty adjustment factor
}

// NewDifficultyHistory creates a new difficulty history tracker
func NewDifficultyHistory(window int) *DifficultyHistory {
	return &DifficultyHistory{
		Entries: make([]DifficultyHistoryEntry, 0, window),
		Window:  window,
	}
}

// AddEntry adds a new difficulty history entry
func (dh *DifficultyHistory) AddEntry(entry DifficultyHistoryEntry) {
	dh.Entries = append(dh.Entries, entry)

	// Keep only the last 'window' entries
	if len(dh.Entries) > dh.Window {
		dh.Entries = dh.Entries[len(dh.Entries)-dh.Window:]
	}
}

// GetAverageBlockTime returns average block time over the history window
func (dh *DifficultyHistory) GetAverageBlockTime() time.Duration {
	if len(dh.Entries) < 2 {
		return 0
	}

	totalTime := time.Duration(0)
	count := 0

	for _, entry := range dh.Entries {
		if entry.BlockTime > 0 {
			totalTime += entry.BlockTime
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return totalTime / time.Duration(count)
}

// GetDifficultyVariance returns the variance in difficulty adjustments
func (dh *DifficultyHistory) GetDifficultyVariance() float64 {
	if len(dh.Entries) < 2 {
		return 0
	}

	// Calculate mean adjustment factor
	sum := 0.0
	count := 0
	for _, entry := range dh.Entries {
		if entry.AdjustmentFactor > 0 {
			sum += entry.AdjustmentFactor
			count++
		}
	}

	if count == 0 {
		return 0
	}

	mean := sum / float64(count)

	// Calculate variance
	variance := 0.0
	for _, entry := range dh.Entries {
		if entry.AdjustmentFactor > 0 {
			diff := entry.AdjustmentFactor - mean
			variance += diff * diff
		}
	}

	return variance / float64(count)
}

// ASERTQValidator validates ASERT-Q difficulty adjustments
type ASERTQValidator struct {
	asert *ASERTQDifficulty
}

// NewASERTQValidator creates a new ASERT-Q validator
func NewASERTQValidator() *ASERTQValidator {
	return &ASERTQValidator{
		asert: NewASERTQDifficulty(),
	}
}

// ValidateDifficulty validates that a block's difficulty follows ASERT-Q rules
func (validator *ASERTQValidator) ValidateDifficulty(
	chain consensus.ChainHeaderReader,
	header *types.Header,
) error {

	// Calculate expected difficulty
	expectedAdjustment := validator.asert.CalculateNextDifficulty(chain, header)
	expectedDifficulty := expectedAdjustment.NewTarget

	// Compare with actual difficulty
	actualDifficulty := header.Difficulty
	if actualDifficulty == nil {
		return fmt.Errorf("header has nil difficulty")
	}

	if expectedDifficulty.Cmp(actualDifficulty) != 0 {
		return fmt.Errorf("invalid difficulty: expected %s, got %s",
			expectedDifficulty.String(), actualDifficulty.String())
	}

	// Validate block timing
	parent := chain.GetHeaderByHash(header.ParentHash)
	if err := validator.asert.ValidateBlockTime(header, parent); err != nil {
		return fmt.Errorf("invalid block time: %v", err)
	}

	return nil
}

// GetExpectedDifficulty returns the expected difficulty for a block
func (validator *ASERTQValidator) GetExpectedDifficulty(
	chain consensus.ChainHeaderReader,
	header *types.Header,
) *big.Int {

	adjustment := validator.asert.CalculateNextDifficulty(chain, header)
	return adjustment.NewTarget
}
