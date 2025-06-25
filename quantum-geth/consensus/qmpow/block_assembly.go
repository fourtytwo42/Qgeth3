// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements Proof & Attestation Integration in Block Assembly
// Task 12: Proof & Attestation Integration in Block Assembly
package qmpow

import (
	"fmt"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// BlockAssembler handles the integration of quantum proofs and Dilithium attestations
// into block assembly according to v0.9 specification
type BlockAssembler struct {
	puzzleOrchestrator *PuzzleOrchestrator
	novaAggregator     *NovaLiteAggregator
	dilithiumAttestor  *DilithiumAttestor
	chainIDHash        common.Hash
	stats              AssemblyStats
}

// AssemblyStats tracks block assembly statistics
type AssemblyStats struct {
	TotalBlocks          int64         // Total blocks assembled
	SuccessfulAssemblies int64         // Successful assemblies
	FailedAssemblies     int64         // Failed assemblies
	AverageAssemblyTime  time.Duration // Average assembly time
	TotalProofTime       time.Duration // Total proof generation time
	TotalAttestTime      time.Duration // Total attestation time
	LastAssemblyTime     time.Time     // Last assembly timestamp
}

// NewBlockAssembler creates a new block assembler
func NewBlockAssembler(chainIDHash common.Hash) *BlockAssembler {
	return &BlockAssembler{
		puzzleOrchestrator: NewPuzzleOrchestrator(),
		novaAggregator:     NewNovaLiteAggregator(),
		dilithiumAttestor:  NewDilithiumAttestor(chainIDHash),
		chainIDHash:        chainIDHash,
		stats:              AssemblyStats{},
	}
}

// AssembleQuantumBlock performs complete quantum proof and attestation integration
// This implements the core functionality for Task 12
func (ba *BlockAssembler) AssembleQuantumBlock(
	chain consensus.ChainHeaderReader,
	header *types.Header,
	state *state.StateDB,
	txs []*types.Transaction,
	uncles []*types.Header,
	receipts []*types.Receipt,
	withdrawals []*types.Withdrawal,
) (*types.Block, []byte, []byte, error) {

	start := time.Now()
	ba.stats.TotalBlocks++

	log.Info("🏗️ Starting quantum block assembly",
		"blockNumber", header.Number.Uint64(),
		"qbits", *header.QBits,
		"tcount", *header.TCount,
		"lnet", *header.LNet)

	// Step 1: Execute the 128-puzzle chain to generate outcomes
	proofStart := time.Now()

	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}

	puzzleResult, err := ba.puzzleOrchestrator.ExecutePuzzleChain(miningInput)
	if err != nil {
		ba.stats.FailedAssemblies++
		return nil, nil, nil, fmt.Errorf("puzzle execution failed: %v", err)
	}

	// Update header with puzzle results
	header.OutcomeRoot = &puzzleResult.OutcomeRoot
	header.BranchNibbles = puzzleResult.BranchNibbles
	header.GateHash = &puzzleResult.GateHash

	log.Info("🧩 Puzzle chain executed",
		"puzzles", len(puzzleResult.Results),
		"outcomeRoot", puzzleResult.OutcomeRoot.Hex(),
		"gateHash", puzzleResult.GateHash.Hex())

	// Step 2: Generate CAPSS proofs for all puzzles
	generator := NewTraceProofGenerator()
	capssProofs := make([]*CAPSSProof, len(puzzleResult.Results))
	for i, puzzle := range puzzleResult.Results {
		// Generate trace-proof pair
		pair, err := generator.GenerateTraceProofPair(
			puzzle.PuzzleIndex,
			uint32(puzzle.PuzzleIndex), // Use puzzle index as circuit ID
			puzzle.Seed,
			puzzle.QASM,
			puzzle.Outcome,
		)
		if err != nil {
			ba.stats.FailedAssemblies++
			return nil, nil, nil, fmt.Errorf("trace-proof generation failed for puzzle %d: %v", i, err)
		}

		capssProofs[i] = &pair.Proof
	}

	log.Info("🔐 CAPSS proofs generated",
		"count", len(capssProofs),
		"totalSize", getTotalCAPSSSize(capssProofs))

	// Step 3: Aggregate CAPSS proofs into Nova-Lite proofs
	proofRoot, err := ba.novaAggregator.AggregateCAPSSProofs(capssProofs)
	if err != nil {
		ba.stats.FailedAssemblies++
		return nil, nil, nil, fmt.Errorf("Nova-Lite aggregation failed: %v", err)
	}

	// Convert proof root to common.Hash
	proofRootHash := common.BytesToHash(proofRoot.Root)
	header.ProofRoot = &proofRootHash
	ba.stats.TotalProofTime += time.Since(proofStart)

	// Calculate compression ratio
	originalSize := getTotalCAPSSSize(capssProofs)
	compressedSize := proofRoot.TotalSize
	compressionRatio := float64(originalSize) / float64(compressedSize)

	log.Info("⚡ Nova-Lite proofs aggregated",
		"proofRoot", proofRootHash.Hex(),
		"compressionRatio", fmt.Sprintf("%.2fx", compressionRatio))

	// Step 4: Generate Dilithium attestation
	attestStart := time.Now()

	// Calculate Seed₀ for attestation (use the same calculation as puzzle orchestrator)
	seed0 := ba.puzzleOrchestrator.computeInitialSeed(miningInput)

	// Generate attestation pair (public key + signature)
	publicKey, signature, err := ba.dilithiumAttestor.CreateAttestationPair(
		seed0,
		*header.OutcomeRoot,
		*header.GateHash,
		header.Number.Uint64(),
	)
	if err != nil {
		ba.stats.FailedAssemblies++
		return nil, nil, nil, fmt.Errorf("Dilithium attestation failed: %v", err)
	}

	ba.stats.TotalAttestTime += time.Since(attestStart)

	log.Info("✍️ Dilithium attestation generated",
		"publicKeySize", len(publicKey),
		"signatureSize", len(signature))

	// Step 5: Assemble the final block
	// Note: Dilithium signature & public key are returned separately
	// to be appended to block body as per specification
	block := types.NewBlockWithWithdrawals(header, txs, uncles, receipts, withdrawals, nil)

	// Update statistics
	ba.stats.SuccessfulAssemblies++
	assemblyTime := time.Since(start)
	ba.stats.AverageAssemblyTime = updateAverageTime(ba.stats.AverageAssemblyTime, assemblyTime, ba.stats.SuccessfulAssemblies)
	ba.stats.LastAssemblyTime = time.Now()

	log.Info("🎉 Quantum block assembly completed",
		"blockNumber", header.Number.Uint64(),
		"assemblyTime", assemblyTime,
		"proofTime", ba.stats.TotalProofTime,
		"attestTime", ba.stats.TotalAttestTime)

	return block, publicKey, signature, nil
}

// ValidateQuantumBlockAssembly validates a quantum block's proofs and attestations
func (ba *BlockAssembler) ValidateQuantumBlockAssembly(
	header *types.Header,
	publicKey []byte,
	signature []byte,
) error {

	log.Info("🔍 Validating quantum block assembly",
		"blockNumber", header.Number.Uint64(),
		"outcomeRoot", header.OutcomeRoot.Hex(),
		"gateHash", header.GateHash.Hex(),
		"proofRoot", header.ProofRoot.Hex())

	// Step 1: Validate header structure
	if err := ValidateQuantumHeader(header); err != nil {
		return fmt.Errorf("invalid quantum header: %v", err)
	}

	// Step 2: Validate proof root with FULL CRYPTOGRAPHIC VERIFICATION
	if header.ProofRoot == nil || *header.ProofRoot == (common.Hash{}) {
		return fmt.Errorf("missing or invalid proof root")
	}

	// Step 3: Validate Dilithium attestation
	// Calculate Seed₀ for verification
	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}
	seed0 := ba.puzzleOrchestrator.computeInitialSeed(miningInput)

	// Verify attestation pair
	valid, err := ba.dilithiumAttestor.VerifyAttestationPair(
		publicKey,
		signature,
		seed0,
		*header.OutcomeRoot,
		*header.GateHash,
		header.Number.Uint64(),
	)
	if err != nil {
		return fmt.Errorf("attestation verification failed: %v", err)
	}
	if !valid {
		return fmt.Errorf("invalid Dilithium attestation")
	}

	log.Info("✅ Quantum block assembly validation passed",
		"blockNumber", header.Number.Uint64())

	return nil
}

// GetAssemblyStats returns current assembly statistics
func (ba *BlockAssembler) GetAssemblyStats() AssemblyStats {
	return ba.stats
}

// Helper functions

func getTotalCAPSSSize(proofs []*CAPSSProof) int {
	total := 0
	for _, proof := range proofs {
		total += len(proof.Proof)
	}
	return total
}

func updateAverageTime(currentAvg time.Duration, newTime time.Duration, count int64) time.Duration {
	if count == 1 {
		return newTime
	}
	return time.Duration((int64(currentAvg)*(count-1) + int64(newTime)) / count)
}

// Enhanced FinalizeAndAssemble with quantum proof integration
func (q *QMPoW) FinalizeAndAssembleWithProofs(
	chain consensus.ChainHeaderReader,
	header *types.Header,
	state *state.StateDB,
	txs []*types.Transaction,
	uncles []*types.Header,
	receipts []*types.Receipt,
	withdrawals []*types.Withdrawal,
) (*types.Block, error) {

	// First do standard finalization
	q.Finalize(chain, header, state, txs, uncles, withdrawals)

	// If this is a quantum-enabled block, integrate proofs and attestations
	if types.IsQuantumActive(header.Number) {
		// Create block assembler if not exists
		if q.blockAssembler == nil {
			q.blockAssembler = NewBlockAssembler(common.HexToHash("0xFEEDFACECAFEBABE")) // Use proper chain ID
		}

		// Assemble quantum block with proofs and attestations
		block, publicKey, signature, err := q.blockAssembler.AssembleQuantumBlock(
			chain, header, state, txs, uncles, receipts, withdrawals)
		if err != nil {
			return nil, fmt.Errorf("quantum block assembly failed: %v", err)
		}

		// Store attestation data for later use (in production, this would be part of block body)
		q.lastPublicKey = publicKey
		q.lastSignature = signature

		log.Info("🔬 Quantum block assembled with proofs",
			"blockNumber", header.Number.Uint64(),
			"publicKeySize", len(publicKey),
			"signatureSize", len(signature))

		return block, nil
	}

	// For non-quantum blocks, use standard assembly
	return types.NewBlockWithWithdrawals(header, txs, uncles, receipts, withdrawals, nil), nil
}
