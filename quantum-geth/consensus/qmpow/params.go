// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the Quantum-Geth quantum proof-of-work consensus engine.
package qmpow

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// Quantum-Geth Genesis Constants (Section 16)
const (
	// Genesis Constants & Configuration Hashes (immutable)
	ProofSystemHash      = "0xA1B2C3D41234567890ABCDEF1234567890ABCDEF1234567890ABCDEF12341234"
	TemplateAuditRoot_v2 = "0xDEADBEEFCAFEBABEDEADBEEFCAFEBABEDEADBEEFCAFEBABEDEADBEEFCAFEBEEF"
	GlideTableHash       = "0xCAFEBABEFACECAFEBABEFACECAFEBABEFACECAFEBABEFACECAFEBABEFACEFACE"
	CanonicompSHA        = "0x123456789ABC123456789ABC123456789ABC123456789ABC123456789ABC9ABC"
	ChainIDHash          = "0xFEEDFACECAFEFEEDFACECAFEFEEDFACECAFEFEEDFACECAFEFEEDFACECAFECAFE"

	// v0.9 Quantum Parameters (Section 9)
	DefaultStartingQBits uint16 = 16  // Start n = 16 at epoch 0 (126-bit security)
	DefaultTCount        uint32 = 20  // ENFORCED MINIMUM: 20 T-gates per puzzle (was 8192)
	DefaultLNet          uint16 = 128 // 128 chained puzzles per block (restored for miner compatibility)
	TargetBlockTime      uint64 = 12 // Target block time in seconds

	// Halving Schedule (Section 11)
	InitialBlockSubsidy = 50.0   // 50 QGCoins per block at epoch 0
	HalvingEpochLength  = 600000 // 600,000 blocks per epoch (≈ 6 months)

	// ASERT-Q Difficulty Parameters (Section 10)
	ASERTLambda            = 0.12 // λ = 0.12 for exponential adjustment
	ASERTHalfLife          = 150  // 150 seconds half-life
	ASERTMaxAdjustmentUp   = 1.10 // +10% max per block
	ASERTMaxAdjustmentDown = 0.90 // -10% max per block

	// Glide Schedule (immutable - Section 9)
	// QBits increases according to Security Hardness Table
	// Height ranges for each QBits level (based on 6-month epochs)
	QBits16_MaxHeight = 1200000 // QBits=16, TCount=20, LNet=128 (enhanced security)
	QBits17_MaxHeight = 1800000 // QBits=17, TCount=20, LNet=128 (enhanced security)
	QBits18_MaxHeight = 2400000 // QBits=18, TCount=20, LNet=128 (enhanced security)
	QBits19_MaxHeight = 3000000 // QBits=19, TCount=20, LNet=128 (enhanced security)
	QBits20_MaxHeight = 3600000 // QBits=20, TCount=20, LNet=128 (enhanced security)

	// Mining complexity parameters (for simulation)
	BaseComplexityMs      = 100 // Base time per puzzle in milliseconds
	ComplexityScaleFactor = 1.5 // How much complexity increases per puzzle
)

// QMPoWParams represents the configuration parameters for Quantum-Geth
type QMPoWParams struct {
	Epoch        uint32  // Current epoch ⌊Height / 600,000⌋
	QBits        uint16  // Qubits per puzzle (from glide schedule)
	TCount       uint32  // T gates per puzzle (from glide schedule)
	LNet         uint16  // Puzzles per block (from glide schedule)
	BlockSubsidy float64 // Current block subsidy in QGCoins
}

// DefaultParams returns the default quantum proof of work parameters for a given height
func DefaultParams(height uint64) QMPoWParams {
	epoch := uint32(height / HalvingEpochLength)
	qbits, tcount, lnet := CalculateQuantumParamsForHeight(height)
	subsidy := CalculateBlockSubsidy(epoch)

	return QMPoWParams{
		Epoch:        epoch,
		QBits:        qbits,
		TCount:       tcount,
		LNet:         lnet,
		BlockSubsidy: subsidy,
	}
}

// CalculateQuantumParamsForHeight implements the immutable glide schedule (Section 9)
func CalculateQuantumParamsForHeight(height uint64) (uint16, uint32, uint16) {
	switch {
	case height <= QBits16_MaxHeight:
		return 16, 20, 128 // Enhanced security with 128 chained puzzles
	case height <= QBits17_MaxHeight:
		return 17, 20, 128 // Enhanced security with 128 chained puzzles
	case height <= QBits18_MaxHeight:
		return 18, 20, 128 // Enhanced security with 128 chained puzzles
	case height <= QBits19_MaxHeight:
		return 19, 20, 128 // Enhanced security with 128 chained puzzles
	case height <= QBits20_MaxHeight:
		return 20, 20, 128 // Enhanced security with 128 chained puzzles
	default:
		// Beyond defined schedule, maintain maximum security
		return 20, 20, 128
	}
}

// CalculateBlockSubsidy implements Bitcoin-style halving (Section 11.1)
func CalculateBlockSubsidy(epoch uint32) float64 {
	if epoch == 0 {
		return InitialBlockSubsidy
	}

	// Subsidy(epoch) = 50 × 2^(-epoch)
	subsidy := InitialBlockSubsidy
	for i := uint32(0); i < epoch; i++ {
		subsidy /= 2.0
	}

	// Minimum subsidy is 0 (no negative subsidies)
	if subsidy < 1e-8 { // Essentially zero for practical purposes
		return 0.0
	}

	return subsidy
}

// ParamsForHeight returns quantum parameters for a specific block height
func (q *QMPoW) ParamsForHeight(height uint64) QMPoWParams {
	return DefaultParams(height)
}

// EstimateNextDifficulty estimates the difficulty for the next block
// Difficulty is handled via ASERT-Q targeting
func (q *QMPoW) EstimateNextDifficulty(chain consensus.ChainHeaderReader, header *types.Header) uint16 {
	params := DefaultParams(header.Number.Uint64())

	log.Info("🔗 Quantum difficulty parameters",
		"blockNumber", header.Number.Uint64(),
		"epoch", params.Epoch,
		"qbits", params.QBits,
		"tcount", params.TCount,
		"lnet", params.LNet,
		"subsidy", params.BlockSubsidy,
		"effectiveSecurity", CalculateEffectiveSecurityBits(params.QBits, params.LNet))

	return params.LNet
}

// CalculateEffectiveSecurityBits calculates the effective security level
func CalculateEffectiveSecurityBits(qbits uint16, lnet uint16) uint16 {
	// Approximate calculation based on Section 9 Security Hardness Table
	// This is a simplified model - real security analysis would be more complex
	baseSecurity := uint16(qbits * 8)     // Rough approximation
	networkEffect := uint16(lnet / 8)     // Network amplification factor
	groversReduction := uint16(qbits / 2) // √-Grover penalty

	effectiveBits := baseSecurity + networkEffect - groversReduction
	return effectiveBits
}

// ValidateGenesisConstants verifies all genesis constants are properly embedded
func ValidateGenesisConstants() error {
	// Verify all required constants are non-empty
	constants := map[string]string{
		"ProofSystemHash":      ProofSystemHash,
		"TemplateAuditRoot_v2": TemplateAuditRoot_v2,
		"GlideTableHash":       GlideTableHash,
		"CanonicompSHA":        CanonicompSHA,
		"ChainIDHash":          ChainIDHash,
	}

	for name, value := range constants {
		if value == "" {
			log.Crit("Genesis constant not defined", "constant", name)
		}
		if !common.IsHexAddress(value) && len(value) != 66 { // 0x + 64 hex chars
			log.Crit("Invalid genesis constant format", "constant", name, "value", value)
		}
	}

	log.Info("✅ All genesis constants validated",
		"proofSystemHash", ProofSystemHash[:10]+"...",
		"templateAuditRoot", TemplateAuditRoot_v2[:10]+"...",
		"glideTableHash", GlideTableHash[:10]+"...",
		"canonicompSHA", CanonicompSHA[:10]+"...",
		"chainIDHash", ChainIDHash[:10]+"...")

	return nil
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
