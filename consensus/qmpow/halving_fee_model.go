// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements Halving & Fee Model Implementation
// Task 13: Halving & Fee Model Implementation - v0.9–BareBones+Halving Specification
package qmpow

import (
	"math/big"
	"time"

	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/holiman/uint256"
)

// Halving & Fee Model Constants (Section 11)
const (
	// Block subsidy parameters
	InitialSubsidyQGC = 50.0   // 50 QGCoins per block at epoch 0
	HalvingEpochSize  = 600000 // 600,000 blocks per epoch (≈ 6 months at 12s blocks)

	// Conversion factors
	QGCToWeiConstant = 1e18 // 1 QGCoin = 10^18 wei (same as ETH)

	// Minimum subsidy (essentially zero)
	MinimumSubsidy = 1e-8 // Minimum meaningful subsidy
)

// HalvingFeeModel handles the complete reward calculation for Quantum-Geth
type HalvingFeeModel struct {
	chainConfig consensus.ChainHeaderReader
	stats       HalvingStats
}

// HalvingStats tracks halving and fee statistics
type HalvingStats struct {
	TotalBlocksProcessed uint64    // Total blocks processed
	TotalSubsidyPaid     *big.Int  // Total subsidy paid out (in wei)
	TotalFeesPaid        *big.Int  // Total transaction fees paid out (in wei)
	CurrentEpoch         uint32    // Current halving epoch
	CurrentSubsidy       float64   // Current block subsidy in QGC
	LastHalvingBlock     uint64    // Block number of last halving
	NextHalvingBlock     uint64    // Block number of next halving
	LastProcessedTime    time.Time // Last processing timestamp
}

// BlockReward represents the complete reward calculation for a block
type BlockReward struct {
	BlockNumber     uint64    // Block number
	Epoch           uint32    // Halving epoch
	SubsidyQGC      float64   // Block subsidy in QGCoins
	SubsidyWei      *big.Int  // Block subsidy in wei
	TransactionFees *big.Int  // Total transaction fees in wei
	TotalRewardWei  *big.Int  // Total reward (subsidy + fees) in wei
	IsHalvingBlock  bool      // Whether this block triggers a halving
	CalculatedAt    time.Time // When this reward was calculated
}

// NewHalvingFeeModel creates a new halving and fee model
func NewHalvingFeeModel(chainConfig consensus.ChainHeaderReader) *HalvingFeeModel {
	return &HalvingFeeModel{
		chainConfig: chainConfig,
		stats: HalvingStats{
			TotalSubsidyPaid: big.NewInt(0),
			TotalFeesPaid:    big.NewInt(0),
			CurrentSubsidy:   InitialSubsidyQGC,
			NextHalvingBlock: HalvingEpochSize,
		},
	}
}

// CalculateBlockReward calculates the complete block reward according to Section 11
func (hfm *HalvingFeeModel) CalculateBlockReward(
	header *types.Header,
	txs []*types.Transaction,
	receipts []*types.Receipt,
) *BlockReward {

	blockNumber := header.Number.Uint64()

	// Calculate current epoch and subsidy
	epoch := uint32(blockNumber / HalvingEpochSize)
	subsidyQGC := CalculateBlockSubsidy(epoch)
	subsidyWei := QGCToWei(subsidyQGC)

	// Calculate total transaction fees
	transactionFees := hfm.calculateTransactionFees(txs, receipts, header.BaseFee)

	// Total reward = subsidy + fees
	totalReward := new(big.Int).Add(subsidyWei, transactionFees)

	// Check if this is a halving block
	isHalvingBlock := blockNumber > 0 && blockNumber%HalvingEpochSize == 0

	log.Info("💰 Block reward calculated",
		"blockNumber", blockNumber,
		"epoch", epoch,
		"subsidyQGC", subsidyQGC,
		"subsidyWei", subsidyWei,
		"transactionFees", transactionFees,
		"totalReward", totalReward,
		"isHalvingBlock", isHalvingBlock)

	return &BlockReward{
		BlockNumber:     blockNumber,
		Epoch:           epoch,
		SubsidyQGC:      subsidyQGC,
		SubsidyWei:      subsidyWei,
		TransactionFees: transactionFees,
		TotalRewardWei:  totalReward,
		IsHalvingBlock:  isHalvingBlock,
		CalculatedAt:    time.Now(),
	}
}

// ApplyBlockReward applies the calculated reward to the state
func (hfm *HalvingFeeModel) ApplyBlockReward(
	state *state.StateDB,
	header *types.Header,
	reward *BlockReward,
) {

	// Award total reward to coinbase
	state.AddBalance(header.Coinbase, uint256.MustFromBig(reward.TotalRewardWei))

	// Update statistics
	hfm.updateStats(reward)

	log.Info("💸 Block reward applied",
		"blockNumber", reward.BlockNumber,
		"coinbase", header.Coinbase.Hex(),
		"totalReward", reward.TotalRewardWei,
		"subsidyQGC", reward.SubsidyQGC,
		"fees", reward.TransactionFees)

	// Log halving events
	if reward.IsHalvingBlock {
		prevSubsidy := CalculateBlockSubsidy(reward.Epoch - 1)
		log.Warn("🎉 HALVING EVENT!",
			"blockNumber", reward.BlockNumber,
			"epoch", reward.Epoch,
			"previousSubsidy", prevSubsidy,
			"newSubsidy", reward.SubsidyQGC,
			"reductionFactor", "2x")
	}
}

// calculateTransactionFees calculates total transaction fees for a block
func (hfm *HalvingFeeModel) calculateTransactionFees(
	txs []*types.Transaction,
	receipts []*types.Receipt,
	baseFee *big.Int,
) *big.Int {

	totalFees := big.NewInt(0)

	for i, tx := range txs {
		if i >= len(receipts) {
			continue
		}

		receipt := receipts[i]
		gasUsed := new(big.Int).SetUint64(receipt.GasUsed)

		// Calculate effective gas price
		var effectiveGasPrice *big.Int
		if baseFee != nil && tx.Type() == types.DynamicFeeTxType {
			// EIP-1559 transaction
			tip, _ := tx.EffectiveGasTip(baseFee)
			effectiveGasPrice = new(big.Int).Add(baseFee, tip)
		} else {
			// Legacy transaction
			effectiveGasPrice = tx.GasPrice()
		}

		// Fee = gasUsed * effectiveGasPrice
		txFee := new(big.Int).Mul(gasUsed, effectiveGasPrice)
		totalFees.Add(totalFees, txFee)
	}

	return totalFees
}

// updateStats updates internal statistics
func (hfm *HalvingFeeModel) updateStats(reward *BlockReward) {
	hfm.stats.TotalBlocksProcessed++
	hfm.stats.TotalSubsidyPaid.Add(hfm.stats.TotalSubsidyPaid, reward.SubsidyWei)
	hfm.stats.TotalFeesPaid.Add(hfm.stats.TotalFeesPaid, reward.TransactionFees)
	hfm.stats.CurrentEpoch = reward.Epoch
	hfm.stats.CurrentSubsidy = reward.SubsidyQGC
	hfm.stats.LastProcessedTime = time.Now()

	if reward.IsHalvingBlock {
		hfm.stats.LastHalvingBlock = reward.BlockNumber
		hfm.stats.NextHalvingBlock = reward.BlockNumber + HalvingEpochSize
	}
}

// GetHalvingStats returns current halving statistics
func (hfm *HalvingFeeModel) GetHalvingStats() HalvingStats {
	return hfm.stats
}

// Utility functions

// QGCToWei converts QGCoins to wei
func QGCToWei(qgc float64) *big.Int {
	if qgc < MinimumSubsidy {
		return big.NewInt(0)
	}

	// Convert float QGC to wei (1 QGC = 10^18 wei)
	qgcBig := new(big.Float).SetFloat64(qgc)
	weiBig := new(big.Float).Mul(qgcBig, big.NewFloat(QGCToWeiConstant))

	result, _ := weiBig.Int(nil)
	return result
}

// WeiToQGC converts wei to QGCoins
func WeiToQGC(wei *big.Int) float64 {
	if wei == nil || wei.Sign() == 0 {
		return 0.0
	}

	weiBig := new(big.Float).SetInt(wei)
	qgcBig := new(big.Float).Quo(weiBig, big.NewFloat(QGCToWeiConstant))

	result, _ := qgcBig.Float64()
	return result
}

// GetCurrentSubsidy returns the current block subsidy for a given block number
func GetCurrentSubsidy(blockNumber uint64) float64 {
	epoch := uint32(blockNumber / HalvingEpochSize)
	return CalculateBlockSubsidy(epoch)
}

// GetNextHalvingBlock returns the block number of the next halving
func GetNextHalvingBlock(blockNumber uint64) uint64 {
	epoch := blockNumber / HalvingEpochSize
	return (epoch + 1) * HalvingEpochSize
}

// GetBlocksUntilHalving returns the number of blocks until the next halving
func GetBlocksUntilHalving(blockNumber uint64) uint64 {
	nextHalving := GetNextHalvingBlock(blockNumber)
	if nextHalving <= blockNumber {
		return 0
	}
	return nextHalving - blockNumber
}

// EstimateTimeUntilHalving estimates time until next halving
func EstimateTimeUntilHalving(blockNumber uint64) time.Duration {
	blocksRemaining := GetBlocksUntilHalving(blockNumber)
	secondsRemaining := blocksRemaining * TargetBlockTime
	return time.Duration(secondsRemaining) * time.Second
}

// Enhanced QMPoW Finalize method with halving support
func (q *QMPoW) FinalizeWithHalving(
	chain consensus.ChainHeaderReader,
	header *types.Header,
	state *state.StateDB,
	txs []*types.Transaction,
	uncles []*types.Header,
	withdrawals []*types.Withdrawal,
	receipts []*types.Receipt,
) {

	// Create halving fee model if not exists
	if q.halvingModel == nil {
		q.halvingModel = NewHalvingFeeModel(chain)
	}

	// Calculate block reward according to halving schedule
	reward := q.halvingModel.CalculateBlockReward(header, txs, receipts)

	// Apply the reward to the state
	q.halvingModel.ApplyBlockReward(state, header, reward)

	// Handle uncle rewards (if any) - simplified for quantum consensus
	for _, uncle := range uncles {
		// Uncle reward is 1/32 of block subsidy (following Ethereum tradition)
		uncleReward := new(big.Int).Div(reward.SubsidyWei, big.NewInt(32))
		state.AddBalance(uncle.Coinbase, uint256.MustFromBig(uncleReward))

		log.Info("👥 Uncle reward applied",
			"uncleHash", uncle.Hash().Hex(),
			"uncleCoinbase", uncle.Coinbase.Hex(),
			"reward", uncleReward)
	}

	log.Info("✅ Block finalization with halving completed",
		"blockNumber", header.Number.Uint64(),
		"epoch", reward.Epoch,
		"totalReward", reward.TotalRewardWei,
		"uncleCount", len(uncles))
}

// Enhanced QMPoW struct to include halving model
// This would be added to the QMPoW struct in qmpow.go:
// halvingModel *HalvingFeeModel

// GetHalvingInfo returns information about the current halving state
func (q *QMPoW) GetHalvingInfo(blockNumber uint64) map[string]interface{} {
	epoch := uint32(blockNumber / HalvingEpochSize)
	currentSubsidy := CalculateBlockSubsidy(epoch)
	nextHalving := GetNextHalvingBlock(blockNumber)
	blocksUntilHalving := GetBlocksUntilHalving(blockNumber)
	timeUntilHalving := EstimateTimeUntilHalving(blockNumber)

	var stats HalvingStats
	if q.halvingModel != nil {
		stats = q.halvingModel.GetHalvingStats()
	}

	return map[string]interface{}{
		"currentEpoch":         epoch,
		"currentSubsidyQGC":    currentSubsidy,
		"currentSubsidyWei":    QGCToWei(currentSubsidy),
		"nextHalvingBlock":     nextHalving,
		"blocksUntilHalving":   blocksUntilHalving,
		"timeUntilHalving":     timeUntilHalving.String(),
		"halvingEpochSize":     HalvingEpochSize,
		"totalBlocksProcessed": stats.TotalBlocksProcessed,
		"totalSubsidyPaid":     stats.TotalSubsidyPaid,
		"totalFeesPaid":        stats.TotalFeesPaid,
	}
}
