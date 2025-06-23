// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"math/big"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
)

func TestQGCToWei(t *testing.T) {
	tests := []struct {
		qgc      float64
		expected string // Expected wei as string
	}{
		{0.0, "0"},
		{1.0, "1000000000000000000"},   // 1 QGC = 10^18 wei
		{50.0, "50000000000000000000"}, // 50 QGC = 50 * 10^18 wei
		{0.5, "500000000000000000"},    // 0.5 QGC = 5 * 10^17 wei
		{25.0, "25000000000000000000"}, // 25 QGC = 25 * 10^18 wei
		{12.5, "12500000000000000000"}, // 12.5 QGC = 12.5 * 10^18 wei
	}

	for _, test := range tests {
		result := QGCToWei(test.qgc)
		expected := new(big.Int)
		expected.SetString(test.expected, 10)

		if result.Cmp(expected) != 0 {
			t.Errorf("QGCToWei(%f): got %s, expected %s", test.qgc, result.String(), expected.String())
		}
	}
}

func TestWeiToQGC(t *testing.T) {
	tests := []struct {
		wei      string  // Wei as string
		expected float64 // Expected QGC
	}{
		{"0", 0.0},
		{"1000000000000000000", 1.0},   // 10^18 wei = 1 QGC
		{"50000000000000000000", 50.0}, // 50 * 10^18 wei = 50 QGC
		{"500000000000000000", 0.5},    // 5 * 10^17 wei = 0.5 QGC
		{"25000000000000000000", 25.0}, // 25 * 10^18 wei = 25 QGC
		{"12500000000000000000", 12.5}, // 12.5 * 10^18 wei = 12.5 QGC
	}

	for _, test := range tests {
		wei := new(big.Int)
		wei.SetString(test.wei, 10)
		result := WeiToQGC(wei)

		if result != test.expected {
			t.Errorf("WeiToQGC(%s): got %f, expected %f", test.wei, result, test.expected)
		}
	}
}

func TestCalculateBlockSubsidy(t *testing.T) {
	tests := []struct {
		epoch    uint32
		expected float64
	}{
		{0, 50.0},         // Epoch 0: 50 QGC
		{1, 25.0},         // Epoch 1: 25 QGC (50/2)
		{2, 12.5},         // Epoch 2: 12.5 QGC (25/2)
		{3, 6.25},         // Epoch 3: 6.25 QGC (12.5/2)
		{4, 3.125},        // Epoch 4: 3.125 QGC (6.25/2)
		{10, 50.0 / 1024}, // Epoch 10: 50 / 2^10
	}

	for _, test := range tests {
		result := CalculateBlockSubsidy(test.epoch)
		if result != test.expected {
			t.Errorf("CalculateBlockSubsidy(%d): got %f, expected %f", test.epoch, result, test.expected)
		}
	}
}

func TestGetCurrentSubsidy(t *testing.T) {
	tests := []struct {
		blockNumber uint64
		expected    float64
	}{
		{0, 50.0},       // Block 0: Epoch 0, 50 QGC
		{1, 50.0},       // Block 1: Epoch 0, 50 QGC
		{599999, 50.0},  // Block 599999: Epoch 0, 50 QGC
		{600000, 25.0},  // Block 600000: Epoch 1, 25 QGC
		{600001, 25.0},  // Block 600001: Epoch 1, 25 QGC
		{1200000, 12.5}, // Block 1200000: Epoch 2, 12.5 QGC
	}

	for _, test := range tests {
		result := GetCurrentSubsidy(test.blockNumber)
		if result != test.expected {
			t.Errorf("GetCurrentSubsidy(%d): got %f, expected %f", test.blockNumber, result, test.expected)
		}
	}
}

func TestGetNextHalvingBlock(t *testing.T) {
	tests := []struct {
		blockNumber uint64
		expected    uint64
	}{
		{0, 600000},
		{1, 600000},
		{599999, 600000},
		{600000, 1200000},
		{600001, 1200000},
		{1199999, 1200000},
		{1200000, 1800000},
	}

	for _, test := range tests {
		result := GetNextHalvingBlock(test.blockNumber)
		if result != test.expected {
			t.Errorf("GetNextHalvingBlock(%d): got %d, expected %d", test.blockNumber, result, test.expected)
		}
	}
}

func TestGetBlocksUntilHalving(t *testing.T) {
	tests := []struct {
		blockNumber uint64
		expected    uint64
	}{
		{0, 600000},
		{1, 599999},
		{299999, 300001},
		{599999, 1},
		{600000, 600000}, // At halving block, next halving is 600k blocks away
		{600001, 599999},
	}

	for _, test := range tests {
		result := GetBlocksUntilHalving(test.blockNumber)
		if result != test.expected {
			t.Errorf("GetBlocksUntilHalving(%d): got %d, expected %d", test.blockNumber, result, test.expected)
		}
	}
}

func TestEstimateTimeUntilHalving(t *testing.T) {
	tests := []struct {
		blockNumber     uint64
		expectedSeconds uint64
	}{
		{0, 600000 * 12},      // 600k blocks * 12 seconds
		{1, 599999 * 12},      // 599999 blocks * 12 seconds
		{599999, 1 * 12},      // 1 block * 12 seconds
		{600000, 600000 * 12}, // Next halving is 600k blocks away
	}

	for _, test := range tests {
		result := EstimateTimeUntilHalving(test.blockNumber)
		expected := time.Duration(test.expectedSeconds) * time.Second
		if result != expected {
			t.Errorf("EstimateTimeUntilHalving(%d): got %v, expected %v", test.blockNumber, result, expected)
		}
	}
}

func TestNewHalvingFeeModel(t *testing.T) {
	chainConfig := &types.ChainConfig{}
	model := NewHalvingFeeModel(chainConfig)

	if model == nil {
		t.Fatal("Failed to create halving fee model")
	}

	if model.chainConfig != chainConfig {
		t.Error("Chain config not set correctly")
	}

	stats := model.GetHalvingStats()
	if stats.CurrentSubsidy != InitialSubsidyQGC {
		t.Errorf("Initial subsidy should be %f, got %f", InitialSubsidyQGC, stats.CurrentSubsidy)
	}

	if stats.NextHalvingBlock != HalvingEpochSize {
		t.Errorf("Next halving block should be %d, got %d", HalvingEpochSize, stats.NextHalvingBlock)
	}
}

func TestCalculateBlockReward(t *testing.T) {
	chainConfig := &types.ChainConfig{}
	model := NewHalvingFeeModel(chainConfig)

	// Create test header
	header := &types.Header{
		Number:   big.NewInt(1),
		Coinbase: common.HexToAddress("0x1234567890123456789012345678901234567890"),
		BaseFee:  big.NewInt(1000000000), // 1 gwei
	}

	// Create test transactions
	txs := []*types.Transaction{
		createTestTransaction(21000, big.NewInt(2000000000)), // 2 gwei gas price, 21k gas
		createTestTransaction(50000, big.NewInt(1500000000)), // 1.5 gwei gas price, 50k gas
	}

	// Create test receipts
	receipts := []*types.Receipt{
		{GasUsed: 21000},
		{GasUsed: 50000},
	}

	reward := model.CalculateBlockReward(header, txs, receipts)

	if reward == nil {
		t.Fatal("Reward calculation failed")
	}

	if reward.BlockNumber != 1 {
		t.Errorf("Block number should be 1, got %d", reward.BlockNumber)
	}

	if reward.Epoch != 0 {
		t.Errorf("Epoch should be 0, got %d", reward.Epoch)
	}

	if reward.SubsidyQGC != 50.0 {
		t.Errorf("Subsidy should be 50.0 QGC, got %f", reward.SubsidyQGC)
	}

	// Check that transaction fees are calculated
	if reward.TransactionFees.Sign() <= 0 {
		t.Error("Transaction fees should be positive")
	}

	// Check that total reward = subsidy + fees
	expectedTotal := new(big.Int).Add(reward.SubsidyWei, reward.TransactionFees)
	if reward.TotalRewardWei.Cmp(expectedTotal) != 0 {
		t.Errorf("Total reward mismatch: got %s, expected %s",
			reward.TotalRewardWei.String(), expectedTotal.String())
	}
}

func TestHalvingBlockDetection(t *testing.T) {
	chainConfig := &types.ChainConfig{}
	model := NewHalvingFeeModel(chainConfig)

	// Test non-halving blocks
	for _, blockNum := range []uint64{1, 100, 599999} {
		header := &types.Header{Number: big.NewInt(int64(blockNum))}
		reward := model.CalculateBlockReward(header, nil, nil)

		if reward.IsHalvingBlock {
			t.Errorf("Block %d should not be a halving block", blockNum)
		}
	}

	// Test halving blocks
	for _, blockNum := range []uint64{600000, 1200000, 1800000} {
		header := &types.Header{Number: big.NewInt(int64(blockNum))}
		reward := model.CalculateBlockReward(header, nil, nil)

		if !reward.IsHalvingBlock {
			t.Errorf("Block %d should be a halving block", blockNum)
		}
	}
}

func TestHalvingStatsTracking(t *testing.T) {
	chainConfig := &types.ChainConfig{}
	model := NewHalvingFeeModel(chainConfig)

	// Process several blocks
	for i := uint64(1); i <= 3; i++ {
		header := &types.Header{
			Number:   big.NewInt(int64(i)),
			Coinbase: common.HexToAddress("0x1234567890123456789012345678901234567890"),
		}

		reward := model.CalculateBlockReward(header, nil, nil)
		model.updateStats(reward)
	}

	stats := model.GetHalvingStats()

	if stats.TotalBlocksProcessed != 3 {
		t.Errorf("Should have processed 3 blocks, got %d", stats.TotalBlocksProcessed)
	}

	if stats.TotalSubsidyPaid.Sign() <= 0 {
		t.Error("Total subsidy paid should be positive")
	}

	if stats.CurrentEpoch != 0 {
		t.Errorf("Current epoch should be 0, got %d", stats.CurrentEpoch)
	}
}

func TestQMPoWGetHalvingInfo(t *testing.T) {
	config := Config{PowMode: ModeTest, TestMode: true}
	qmpow := New(config)

	// Test for different block numbers
	testCases := []struct {
		blockNumber     uint64
		expectedEpoch   uint32
		expectedSubsidy float64
	}{
		{0, 0, 50.0},
		{599999, 0, 50.0},
		{600000, 1, 25.0},
		{1200000, 2, 12.5},
	}

	for _, tc := range testCases {
		info := qmpow.GetHalvingInfo(tc.blockNumber)

		if info["currentEpoch"].(uint32) != tc.expectedEpoch {
			t.Errorf("Block %d: expected epoch %d, got %v",
				tc.blockNumber, tc.expectedEpoch, info["currentEpoch"])
		}

		if info["currentSubsidyQGC"].(float64) != tc.expectedSubsidy {
			t.Errorf("Block %d: expected subsidy %f, got %v",
				tc.blockNumber, tc.expectedSubsidy, info["currentSubsidyQGC"])
		}

		// Check that halving info contains expected fields
		expectedFields := []string{
			"currentEpoch", "currentSubsidyQGC", "currentSubsidyWei",
			"nextHalvingBlock", "blocksUntilHalving", "timeUntilHalving",
			"halvingEpochSize",
		}

		for _, field := range expectedFields {
			if _, exists := info[field]; !exists {
				t.Errorf("Missing field %s in halving info", field)
			}
		}
	}
}

// Helper function to create test transactions
func createTestTransaction(gasLimit uint64, gasPrice *big.Int) *types.Transaction {
	return types.NewTransaction(
		0, // nonce
		common.HexToAddress("0x0000000000000000000000000000000000000001"), // to
		big.NewInt(0), // value
		gasLimit,      // gas limit
		gasPrice,      // gas price
		nil,           // data
	)
}
