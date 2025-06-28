package qmpow

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
)

// MockChainReader implements consensus.ChainHeaderReader for testing
type MockChainReader struct {
	headers map[common.Hash]*types.Header
}

func NewMockChainReader() *MockChainReader {
	return &MockChainReader{
		headers: make(map[common.Hash]*types.Header),
	}
}

func (mcr *MockChainReader) GetHeaderByHash(hash common.Hash) *types.Header {
	return mcr.headers[hash]
}

func (mcr *MockChainReader) AddHeader(header *types.Header) {
	mcr.headers[header.Hash()] = header
}

// TestASERTQDifficultyCreation tests basic ASERT-Q difficulty calculator creation
func TestASERTQDifficultyCreation(t *testing.T) {
	asert := NewASERTQDifficulty()

	if asert == nil {
		t.Fatal("Failed to create ASERT-Q difficulty calculator")
	}

	if asert.GetTargetBlockTime() != ASERTTargetBlockTime {
		t.Errorf("Expected target block time %v, got %v",
			ASERTTargetBlockTime, asert.GetTargetBlockTime())
	}

	if asert.GetHalfLife() != ASERTHalfLifeTime {
		t.Errorf("Expected half-life %v, got %v",
			ASERTHalfLifeTime, asert.GetHalfLife())
	}

	t.Logf("✅ ASERT-Q difficulty calculator created successfully")
	t.Logf("   - Target block time: %v", asert.GetTargetBlockTime())
	t.Logf("   - Half-life: %v", asert.GetHalfLife())
}

// TestGenesisBlockDifficulty tests difficulty calculation for genesis block
func TestGenesisBlockDifficulty(t *testing.T) {
	asert := NewASERTQDifficulty()
	chain := NewMockChainReader()

	// Create genesis block header
	genesisHeader := &types.Header{
		Number:     big.NewInt(0),
		Time:       uint64(time.Now().Unix()),
		ParentHash: common.Hash{}, // Empty parent hash for genesis
		Difficulty: big.NewInt(0),
	}

	adjustment := asert.CalculateNextDifficulty(chain, genesisHeader)

	if adjustment == nil {
		t.Fatal("Expected difficulty adjustment, got nil")
	}

	if !adjustment.IsGenesis {
		t.Error("Expected genesis flag to be true")
	}

	if adjustment.AdjustmentFactor != 1.0 {
		t.Errorf("Expected adjustment factor 1.0, got %f", adjustment.AdjustmentFactor)
	}

	expectedTarget := new(big.Int).SetUint64(InitialDifficultyTarget)
	if adjustment.NewTarget.Cmp(expectedTarget) != 0 {
		t.Errorf("Expected initial target %s, got %s",
			expectedTarget.String(), adjustment.NewTarget.String())
	}

	t.Logf("✅ Genesis block difficulty test passed")
	t.Logf("   - Initial target: %s", adjustment.NewTarget.String())
	t.Logf("   - Genesis protection: %v", adjustment.IsGenesis)
}

// TestNormalDifficultyAdjustment tests normal ASERT-Q difficulty adjustment
func TestNormalDifficultyAdjustment(t *testing.T) {
	asert := NewASERTQDifficulty()
	chain := NewMockChainReader()

	// Create parent block
	parentHeader := &types.Header{
		Number:     big.NewInt(6000), // Beyond genesis protection
		Time:       1000,
		Difficulty: big.NewInt(1000000),
		Hash:       func() common.Hash { return common.HexToHash("0x1234") },
	}
	chain.AddHeader(parentHeader)

	// Create current block - 20 seconds after parent (8 seconds late)
	currentHeader := &types.Header{
		Number:     big.NewInt(6001),
		Time:       1020, // 20 seconds later
		ParentHash: parentHeader.Hash(),
		Difficulty: big.NewInt(0),
	}

	adjustment := asert.CalculateNextDifficulty(chain, currentHeader)

	if adjustment.IsGenesis {
		t.Error("Expected genesis protection to be false for block 6001")
	}

	if adjustment.ActualTime != 20*time.Second {
		t.Errorf("Expected actual time 20s, got %v", adjustment.ActualTime)
	}

	if adjustment.TimeDelta != 8*time.Second {
		t.Errorf("Expected time delta 8s, got %v", adjustment.TimeDelta)
	}

	// Since block was late, difficulty should decrease (target should increase)
	if adjustment.NewTarget.Cmp(adjustment.PreviousTarget) <= 0 {
		t.Error("Expected target to increase (difficulty to decrease) for late block")
	}

	t.Logf("✅ Normal difficulty adjustment test passed")
	t.Logf("   - Actual time: %v", adjustment.ActualTime)
	t.Logf("   - Time delta: %v", adjustment.TimeDelta)
	t.Logf("   - Adjustment factor: %.4f", adjustment.AdjustmentFactor)
	t.Logf("   - Clamped factor: %.4f", adjustment.ClampedFactor)
}

// TestGenesisProtection tests genesis protection for first 5000 blocks
func TestGenesisProtection(t *testing.T) {
	asert := NewASERTQDifficulty()
	chain := NewMockChainReader()

	// Test block within genesis protection range
	parentHeader := &types.Header{
		Number:     big.NewInt(2000),
		Time:       1000,
		Difficulty: big.NewInt(1000000),
		Hash:       func() common.Hash { return common.HexToHash("0x2000") },
	}
	chain.AddHeader(parentHeader)

	currentHeader := &types.Header{
		Number:     big.NewInt(2001),
		Time:       1030, // 30 seconds later (18 seconds late)
		ParentHash: parentHeader.Hash(),
	}

	adjustment := asert.CalculateNextDifficulty(chain, currentHeader)

	if !adjustment.IsGenesis {
		t.Error("Expected genesis protection to be true for block 2001")
	}

	// Calculate what the adjustment would be without genesis protection
	normalFactor := asert.calculateRawAdjustmentFactor(adjustment.TimeDelta)
	expectedFactor := 1.0 + GenesisLambda*(normalFactor-1.0)

	if abs(adjustment.AdjustmentFactor-expectedFactor) > 0.001 {
		t.Errorf("Expected genesis-protected factor %.4f, got %.4f",
			expectedFactor, adjustment.AdjustmentFactor)
	}

	t.Logf("✅ Genesis protection test passed")
	t.Logf("   - Block height: %d", adjustment.BlockHeight)
	t.Logf("   - Genesis protection: %v", adjustment.IsGenesis)
	t.Logf("   - Normal factor: %.4f", normalFactor)
	t.Logf("   - Protected factor: %.4f", adjustment.AdjustmentFactor)
}

// TestClampingMechanism tests ±10% per-block clamping
func TestClampingMechanism(t *testing.T) {
	asert := NewASERTQDifficulty()

	// Test upward clamping
	largeFactor := 2.0 // Would increase difficulty by 100%
	clampedUp := asert.clampAdjustmentFactor(largeFactor)
	if clampedUp != MaxAdjustmentFactor {
		t.Errorf("Expected upward clamp to %f, got %f", MaxAdjustmentFactor, clampedUp)
	}

	// Test downward clamping
	smallFactor := 0.5 // Would decrease difficulty by 50%
	clampedDown := asert.clampAdjustmentFactor(smallFactor)
	if clampedDown != MinAdjustmentFactor {
		t.Errorf("Expected downward clamp to %f, got %f", MinAdjustmentFactor, clampedDown)
	}

	// Test no clamping needed
	normalFactor := 1.05 // 5% increase
	notClamped := asert.clampAdjustmentFactor(normalFactor)
	if notClamped != normalFactor {
		t.Errorf("Expected no clamping for factor %f, got %f", normalFactor, notClamped)
	}

	t.Logf("✅ Clamping mechanism test passed")
	t.Logf("   - Large factor %.2f clamped to %.2f", largeFactor, clampedUp)
	t.Logf("   - Small factor %.2f clamped to %.2f", smallFactor, clampedDown)
	t.Logf("   - Normal factor %.2f not clamped", normalFactor)
}

// TestRawAdjustmentFactor tests the core ASERT formula: 2^((t_actual - 12s) / 150s)
func TestRawAdjustmentFactor(t *testing.T) {
	asert := NewASERTQDifficulty()

	testCases := []struct {
		name        string
		timeDelta   time.Duration
		expectedMin float64
		expectedMax float64
	}{
		{"On time", 0, 0.99, 1.01},
		{"6s late", 6 * time.Second, 1.02, 1.04},
		{"12s late", 12 * time.Second, 1.05, 1.07},
		{"6s early", -6 * time.Second, 0.96, 0.98},
		{"12s early", -12 * time.Second, 0.93, 0.95},
		{"75s late (half-life)", 75 * time.Second, 1.41, 1.42},        // Should be ~√2
		{"150s late (full half-life)", 150 * time.Second, 1.99, 2.01}, // Should be ~2
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			factor := asert.calculateRawAdjustmentFactor(tc.timeDelta)

			if factor < tc.expectedMin || factor > tc.expectedMax {
				t.Errorf("Factor %.4f outside expected range [%.4f, %.4f]",
					factor, tc.expectedMin, tc.expectedMax)
			}

			t.Logf("   %s: delta=%v, factor=%.4f", tc.name, tc.timeDelta, factor)
		})
	}

	t.Logf("✅ Raw adjustment factor tests passed")
}

// TestBlockTimeValidation tests block time validation
func TestBlockTimeValidation(t *testing.T) {
	asert := NewASERTQDifficulty()

	// Valid block time
	parent := &types.Header{Time: 1000}
	current := &types.Header{Time: 1012} // 12 seconds later

	err := asert.ValidateBlockTime(current, parent)
	if err != nil {
		t.Errorf("Expected valid block time, got error: %v", err)
	}

	// Invalid: negative time
	invalidCurrent := &types.Header{Time: 999} // Before parent
	err = asert.ValidateBlockTime(invalidCurrent, parent)
	if err == nil {
		t.Error("Expected error for negative block time")
	}

	// Invalid: too large time
	largeCurrent := &types.Header{Time: 1000 + 3*60*60} // 3 hours later
	err = asert.ValidateBlockTime(largeCurrent, parent)
	if err == nil {
		t.Error("Expected error for too large block time")
	}

	// Genesis block (no parent)
	err = asert.ValidateBlockTime(current, nil)
	if err != nil {
		t.Errorf("Expected genesis block to be valid, got error: %v", err)
	}

	t.Logf("✅ Block time validation tests passed")
}

// TestHashrateEstimation tests network hashrate estimation
func TestHashrateEstimation(t *testing.T) {
	asert := NewASERTQDifficulty()

	difficulty := big.NewInt(1000000)
	blockTime := 12 * time.Second

	hashrate := asert.EstimateHashrate(difficulty, blockTime)

	expectedHashrate := big.NewFloat(1000000.0 / 12.0) // ~83333.33
	if hashrate.Cmp(expectedHashrate) != 0 {
		t.Errorf("Expected hashrate %s, got %s",
			expectedHashrate.String(), hashrate.String())
	}

	// Test edge cases
	zeroHashrate := asert.EstimateHashrate(nil, blockTime)
	if zeroHashrate.Cmp(big.NewFloat(0)) != 0 {
		t.Error("Expected zero hashrate for nil difficulty")
	}

	zeroHashrate2 := asert.EstimateHashrate(difficulty, 0)
	if zeroHashrate2.Cmp(big.NewFloat(0)) != 0 {
		t.Error("Expected zero hashrate for zero block time")
	}

	t.Logf("✅ Hashrate estimation tests passed")
	t.Logf("   - Difficulty: %s", difficulty.String())
	t.Logf("   - Block time: %v", blockTime)
	t.Logf("   - Estimated hashrate: %s", hashrate.String())
}

// TestDifficultyStats tests comprehensive difficulty statistics
func TestDifficultyStats(t *testing.T) {
	asert := NewASERTQDifficulty()
	chain := NewMockChainReader()

	// Create parent block
	parentHeader := &types.Header{
		Number:     big.NewInt(7000),
		Time:       2000,
		Difficulty: big.NewInt(5000000),
		Hash:       func() common.Hash { return common.HexToHash("0x7000") },
	}
	chain.AddHeader(parentHeader)

	// Create current block
	currentHeader := &types.Header{
		Number:     big.NewInt(7001),
		Time:       2015, // 15 seconds later
		ParentHash: parentHeader.Hash(),
	}

	stats := asert.GetDifficultyStats(chain, currentHeader)

	if stats.BlockHeight != 7001 {
		t.Errorf("Expected block height 7001, got %d", stats.BlockHeight)
	}

	if stats.TargetBlockTime != ASERTTargetBlockTime {
		t.Errorf("Expected target block time %v, got %v",
			ASERTTargetBlockTime, stats.TargetBlockTime)
	}

	if stats.ActualBlockTime != 15*time.Second {
		t.Errorf("Expected actual block time 15s, got %v", stats.ActualBlockTime)
	}

	if stats.GenesisProtection {
		t.Error("Expected genesis protection to be false for block 7001")
	}

	t.Logf("✅ Difficulty stats test passed")
	t.Logf("   - Block height: %d", stats.BlockHeight)
	t.Logf("   - Target time: %v", stats.TargetBlockTime)
	t.Logf("   - Actual time: %v", stats.ActualBlockTime)
	t.Logf("   - Adjustment factor: %.4f", stats.AdjustmentFactor)
	t.Logf("   - Genesis protection: %v", stats.GenesisProtection)
}

// TestDifficultyHistory tests difficulty history tracking
func TestDifficultyHistory(t *testing.T) {
	history := NewDifficultyHistory(10)

	if len(history.Entries) != 0 {
		t.Error("Expected empty history initially")
	}

	// Add some entries
	for i := 0; i < 15; i++ {
		entry := DifficultyHistoryEntry{
			BlockHeight:      uint64(i + 1),
			Timestamp:        uint64(1000 + i*12),
			Difficulty:       big.NewInt(int64(1000000 + i*1000)),
			BlockTime:        time.Duration(12+i%3) * time.Second,
			AdjustmentFactor: 1.0 + float64(i%5-2)*0.01,
		}
		history.AddEntry(entry)
	}

	// Should only keep last 10 entries
	if len(history.Entries) != 10 {
		t.Errorf("Expected 10 entries, got %d", len(history.Entries))
	}

	// Check that we kept the last 10
	if history.Entries[0].BlockHeight != 6 {
		t.Errorf("Expected first entry to be block 6, got %d", history.Entries[0].BlockHeight)
	}

	avgBlockTime := history.GetAverageBlockTime()
	if avgBlockTime == 0 {
		t.Error("Expected non-zero average block time")
	}

	variance := history.GetDifficultyVariance()
	if variance < 0 {
		t.Error("Expected non-negative variance")
	}

	t.Logf("✅ Difficulty history test passed")
	t.Logf("   - Entries: %d", len(history.Entries))
	t.Logf("   - Average block time: %v", avgBlockTime)
	t.Logf("   - Difficulty variance: %.6f", variance)
}

// TestASERTQValidator tests difficulty validation
func TestASERTQValidator(t *testing.T) {
	validator := NewASERTQValidator()
	chain := NewMockChainReader()

	// Create parent block
	parentHeader := &types.Header{
		Number:     big.NewInt(8000),
		Time:       3000,
		Difficulty: big.NewInt(2000000),
		Hash:       func() common.Hash { return common.HexToHash("0x8000") },
	}
	chain.AddHeader(parentHeader)

	// Calculate expected difficulty
	tempHeader := &types.Header{
		Number:     big.NewInt(8001),
		Time:       3012, // 12 seconds later
		ParentHash: parentHeader.Hash(),
	}

	expectedDifficulty := validator.GetExpectedDifficulty(chain, tempHeader)

	// Create header with correct difficulty
	validHeader := &types.Header{
		Number:     big.NewInt(8001),
		Time:       3012,
		ParentHash: parentHeader.Hash(),
		Difficulty: expectedDifficulty,
	}

	err := validator.ValidateDifficulty(chain, validHeader)
	if err != nil {
		t.Errorf("Expected valid difficulty, got error: %v", err)
	}

	// Create header with incorrect difficulty
	invalidHeader := &types.Header{
		Number:     big.NewInt(8001),
		Time:       3012,
		ParentHash: parentHeader.Hash(),
		Difficulty: big.NewInt(999999), // Wrong difficulty
	}

	err = validator.ValidateDifficulty(chain, invalidHeader)
	if err == nil {
		t.Error("Expected error for invalid difficulty")
	}

	t.Logf("✅ ASERT-Q validator test passed")
	t.Logf("   - Expected difficulty: %s", expectedDifficulty.String())
	t.Logf("   - Validation passed for correct difficulty")
	t.Logf("   - Validation failed for incorrect difficulty")
}

// TestSequentialBlocks tests difficulty adjustment over multiple blocks
func TestSequentialBlocks(t *testing.T) {
	asert := NewASERTQDifficulty()
	chain := NewMockChainReader()

	// Start with genesis
	currentHeader := &types.Header{
		Number:     big.NewInt(0),
		Time:       1000,
		Difficulty: big.NewInt(InitialDifficultyTarget),
		Hash:       func() common.Hash { return common.HexToHash("0x0000") },
	}
	chain.AddHeader(currentHeader)

	// Mine 20 blocks with varying times
	blockTimes := []int{10, 15, 8, 20, 12, 11, 14, 9, 18, 12, 13, 7, 16, 12, 10, 19, 11, 12, 8, 15}

	for i, blockTime := range blockTimes {
		parentHash := currentHeader.Hash()

		nextHeader := &types.Header{
			Number:     big.NewInt(int64(i + 1)),
			Time:       currentHeader.Time + uint64(blockTime),
			ParentHash: parentHash,
		}

		adjustment := asert.CalculateNextDifficulty(chain, nextHeader)
		nextHeader.Difficulty = adjustment.NewTarget
		nextHeader.Hash = func() common.Hash {
			return common.HexToHash(fmt.Sprintf("0x%04x", i+1))
		}

		chain.AddHeader(nextHeader)
		currentHeader = nextHeader

		t.Logf("Block %d: time=%ds, factor=%.4f, target=%s",
			i+1, blockTime, adjustment.ClampedFactor, adjustment.NewTarget.String())
	}

	t.Logf("✅ Sequential blocks test passed")
	t.Logf("   - Mined %d blocks with varying times", len(blockTimes))
	t.Logf("   - Final difficulty: %s", currentHeader.Difficulty.String())
}

// Helper function for floating point comparison
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
