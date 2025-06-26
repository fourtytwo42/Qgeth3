// Copyright 2024 The Quantum-Geth Authors
// This file is part of the Quantum-Geth library.
//
// The Quantum-Geth library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Quantum Proof-of-Work Real Qiskit Integration
// This implements genuine quantum circuit execution using Qiskit-Aer

package qmpow

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// QiskitResult represents the result from the Qiskit quantum solver
type QiskitResult struct {
	Outcomes         string    `json:"outcomes"`
	BranchNibbles    string    `json:"branch_nibbles"`
	GateHash         string    `json:"gate_hash"`
	ProofRoot        string    `json:"proof_root"`
	PuzzleCount      int       `json:"puzzle_count"`
	QBits            int       `json:"qbits"`
	TCount           int       `json:"tcount"`
	TotalTime        float64   `json:"total_time"`
	AvgTimePerPuzzle float64   `json:"avg_time_per_puzzle"`
	ExecutionTimes   []float64 `json:"execution_times"`
	Backend          string    `json:"backend"`
	ShotsPerCircuit  int       `json:"shots_per_circuit"`
}

// initializeQuantumFields initializes all quantum proof-of-work fields
func (q *QMPoW) initializeQuantumFields(header *types.Header) {
	// Calculate epoch and quantum parameters based on block height
	blockHeight := header.Number.Uint64()
	params := DefaultParams(blockHeight)
	qbits, tcount, lnet := CalculateQuantumParamsForHeight(blockHeight)

	// Set quantum parameters
	header.Epoch = &params.Epoch
	header.QBits = &qbits
	header.TCount = &tcount
	header.LNet = &lnet

	// Initialize nonce
	qnonce := uint64(0)
	header.QNonce64 = &qnonce

	// Initialize arrays
	header.ExtraNonce32 = make([]byte, ExtraNonce32Size)
	header.BranchNibbles = make([]byte, BranchNibblesSize)

	// Set attestation mode
	attestMode := uint8(AttestModeDilithium)
	header.AttestMode = &attestMode

	// Initialize hash fields (will be filled by quantum computation)
	// CRITICAL: Always create new hash instances to avoid RLP encoding issues
	// Each field must have its own memory address for proper RLP handling
	outcomeHash := common.Hash{}
	gateHash := common.Hash{}
	proofHash := common.Hash{}
	header.OutcomeRoot = &outcomeHash
	header.GateHash = &gateHash
	header.ProofRoot = &proofHash

	// Ensure ExtraNonce32 and BranchNibbles are properly sized
	if len(header.ExtraNonce32) != ExtraNonce32Size {
		header.ExtraNonce32 = make([]byte, ExtraNonce32Size)
	}
	if len(header.BranchNibbles) != BranchNibblesSize {
		header.BranchNibbles = make([]byte, BranchNibblesSize)
	}

	// Initialize EIP optional fields to prevent RLP encoding issues
	// This is critical for proper RLP encoding/decoding
	if header.WithdrawalsHash == nil {
		emptyWithdrawalsHash := common.HexToHash("0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421")
		header.WithdrawalsHash = &emptyWithdrawalsHash
	}
	if header.BlobGasUsed == nil {
		var zero uint64 = 0
		header.BlobGasUsed = &zero
	}
	if header.ExcessBlobGas == nil {
		var zero uint64 = 0
		header.ExcessBlobGas = &zero
	}
	if header.BaseFee == nil {
		header.BaseFee = big.NewInt(0)
	}
	if header.ParentBeaconRoot == nil {
		emptyHash := common.Hash{}
		header.ParentBeaconRoot = &emptyHash
	}

	// Marshal quantum fields into QBlob for proper RLP encoding
	header.MarshalQuantumBlob()

	log.Debug("🔬 Quantum fields initialized",
		"blockNumber", blockHeight,
		"epoch", params.Epoch,
		"qbits", qbits,
		"puzzles", lnet)
}

// callQiskitSolver executes the Python Qiskit solver for real quantum computation
func (q *QMPoW) callQiskitSolver(header *types.Header) (*QiskitResult, error) {
	// Generate seed chain for Qiskit
	seeds := CalculateSeedChain(header)
	seed0 := seeds[0].Hex()[2:] // Remove "0x" prefix for Python

	// Prepare input for Qiskit solver
	input := map[string]interface{}{
		"seed0":  seed0,
		"qbits":  int(*header.QBits),
		"tcount": int(*header.TCount),
		"lnet":   int(*header.LNet),
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	// Find the Qiskit solver script
	solverPath := filepath.Join("quantum-geth", "tools", "solver", "qiskit_solver.py")

	// Execute the Qiskit solver
	cmd := exec.Command("python", solverPath)
	cmd.Stdin = strings.NewReader(string(inputJSON))

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("qiskit solver execution failed: %w", err)
	}

	// Parse the result
	var result QiskitResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse qiskit result: %w", err)
	}

	return &result, nil
}

// SolveQuantumPuzzles implements the complete quantum puzzle solving using real Qiskit
func (q *QMPoW) SolveQuantumPuzzles(header *types.Header) error {
	// Use real Qiskit quantum computation
	result, err := q.callQiskitSolver(header)
	if err != nil {
		return fmt.Errorf("qiskit solver failed: %w", err)
	}

	// Parse outcomes from Qiskit result
	outcomesBytes, err := hex.DecodeString(result.Outcomes)
	if err != nil {
		return fmt.Errorf("failed to decode outcomes: %w", err)
	}

	// Parse branch nibbles
	branchNibblesBytes, err := hex.DecodeString(result.BranchNibbles)
	if err != nil {
		return fmt.Errorf("failed to decode branch nibbles: %w", err)
	}

	// Parse gate hash
	gateHashBytes, err := hex.DecodeString(result.GateHash)
	if err != nil {
		return fmt.Errorf("failed to decode gate hash: %w", err)
	}

	// Parse proof root
	proofRootBytes, err := hex.DecodeString(result.ProofRoot)
	if err != nil {
		return fmt.Errorf("failed to decode proof root: %w", err)
	}

	// Set header fields from Qiskit results
	// Calculate outcome root from all outcomes
	outcomeRoot := CalculateOutcomeRootFromBytes(outcomesBytes, int(*header.LNet), int(*header.QBits))
	header.OutcomeRoot = &outcomeRoot

	// Handle branch nibbles - Python solver returns one nibble per puzzle (lnet nibbles)
	// but Go expects BranchNibblesSize (64) bytes. We need to pad if necessary.
	if len(branchNibblesBytes) > BranchNibblesSize {
		// Truncate if too long
		copy(header.BranchNibbles, branchNibblesBytes[:BranchNibblesSize])
		log.Debug("🔬 Truncated branch nibbles", "got", len(branchNibblesBytes), "used", BranchNibblesSize)
	} else {
		// Copy what we have and pad with zeros if necessary
		copy(header.BranchNibbles, branchNibblesBytes)
		if len(branchNibblesBytes) < BranchNibblesSize {
			// Clear remaining bytes (they were already initialized to zero in Prepare)
			for i := len(branchNibblesBytes); i < BranchNibblesSize; i++ {
				header.BranchNibbles[i] = 0
			}
			log.Debug("🔬 Padded branch nibbles with zeros", 
				"got", len(branchNibblesBytes), 
				"expected", BranchNibblesSize,
				"padded", BranchNibblesSize-len(branchNibblesBytes))
		}
	}

	gateHash := common.BytesToHash(gateHashBytes)
	header.GateHash = &gateHash

	proofRoot := common.BytesToHash(proofRootBytes)
	header.ProofRoot = &proofRoot

	// Marshal updated quantum fields into QBlob for proper RLP encoding
	header.MarshalQuantumBlob()

	log.Debug("🔬 Real quantum computation completed",
		"qnonce", *header.QNonce64,
		"totalTime", fmt.Sprintf("%.3fs", result.TotalTime),
		"avgTimePerPuzzle", fmt.Sprintf("%.3fs", result.AvgTimePerPuzzle),
		"backend", result.Backend)

	return nil
}

// CalculateOutcomeRootFromBytes calculates the outcome root from concatenated outcome bytes
func CalculateOutcomeRootFromBytes(outcomesBytes []byte, lnet int, qbits int) common.Hash {
	// Split the concatenated outcomes back into individual outcomes
	outcomeLen := (qbits + 7) / 8
	outcomes := make([][]byte, lnet)

	for i := 0; i < lnet; i++ {
		start := i * outcomeLen
		end := start + outcomeLen
		if end > len(outcomesBytes) {
			// Pad with zeros if not enough data
			outcome := make([]byte, outcomeLen)
			copy(outcome, outcomesBytes[start:])
			outcomes[i] = outcome
		} else {
			outcomes[i] = outcomesBytes[start:end]
		}
	}

	// Use existing function to calculate root
	return CalculateOutcomeRoot(outcomes)
}

// checkQuantumTarget verifies if the quantum proof meets the target
func (q *QMPoW) checkQuantumTarget(header *types.Header) bool {
	// Use difficulty directly (no complex conversions)
	difficulty := header.Difficulty

	// Use Bitcoin-style quality calculation
	quality := CalculateQuantumProofQuality(
		header.OutcomeRoot.Bytes(),
		append(header.GateHash.Bytes(), header.ProofRoot.Bytes()...),
		*header.QNonce64)

	// Simple target calculation: target = maxTarget / difficulty
	target := DifficultyToTarget(difficulty)

	// Check if quality meets target (Bitcoin-style: lower quality = better proof)
	success := quality.Cmp(target) < 0

	log.Debug("🎯 Quantum target check",
		"qnonce", *header.QNonce64,
		"difficulty", FormatDifficulty(difficulty),
		"quality", quality.String(),
		"target", target.String(),
		"success", success)

	if success {
		log.Info("🎉 Quantum target met!",
			"qnonce", *header.QNonce64,
			"difficulty", FormatDifficulty(difficulty),
			"quality", quality.String(),
			"target", target.String())
	}

	return success
}

// ValidateQuantumHeader validates a quantum header structure
func ValidateQuantumHeader(header *types.Header) error {
	// Verify all required fields are present
	if header.Epoch == nil {
		return fmt.Errorf("missing Epoch field")
	}
	if header.QBits == nil {
		return fmt.Errorf("missing QBits field")
	}
	if header.TCount == nil {
		return fmt.Errorf("missing TCount field")
	}
	if header.LNet == nil {
		return fmt.Errorf("missing LNet field")
	}
	if header.QNonce64 == nil {
		return fmt.Errorf("missing QNonce64 field")
	}
	if header.OutcomeRoot == nil {
		return fmt.Errorf("missing OutcomeRoot field")
	}
	if header.GateHash == nil {
		return fmt.Errorf("missing GateHash field")
	}
	if header.ProofRoot == nil {
		return fmt.Errorf("missing ProofRoot field")
	}
	if header.AttestMode == nil {
		return fmt.Errorf("missing AttestMode field")
	}

	// Verify field sizes
	if len(header.ExtraNonce32) != ExtraNonce32Size {
		return fmt.Errorf("invalid ExtraNonce32 size: got %d, expected %d",
			len(header.ExtraNonce32), ExtraNonce32Size)
	}

	if len(header.BranchNibbles) != BranchNibblesSize {
		return fmt.Errorf("invalid BranchNibbles size: got %d, expected %d",
			len(header.BranchNibbles), BranchNibblesSize)
	}

	// Verify parameter values
	expectedEpoch := uint32(header.Number.Uint64() / EpochBlocks)
	if *header.Epoch != expectedEpoch {
		return fmt.Errorf("invalid epoch: got %d, expected %d", *header.Epoch, expectedEpoch)
	}

	expectedQBits, _, _ := CalculateQuantumParamsForHeight(header.Number.Uint64())
	if *header.QBits != expectedQBits {
		return fmt.Errorf("invalid qbits: got %d, expected %d", *header.QBits, expectedQBits)
	}

	_, expectedTCount, expectedLNet := CalculateQuantumParamsForHeight(header.Number.Uint64())
	if *header.TCount != expectedTCount {
		return fmt.Errorf("invalid tcount: got %d, expected %d", *header.TCount, expectedTCount)
	}

	if *header.LNet != expectedLNet {
		return fmt.Errorf("invalid lnet: got %d, expected %d", *header.LNet, expectedLNet)
	}

	if *header.AttestMode != AttestModeDilithium {
		return fmt.Errorf("invalid attest mode: got %d, expected %d", *header.AttestMode, AttestModeDilithium)
	}

	return nil
}

// PrepareQuantum prepares a header for quantum mining
func (q *QMPoW) PrepareQuantum(chain consensus.ChainHeaderReader, header *types.Header) error {
	// Initialize quantum fields
	q.initializeQuantumFields(header)
	return nil
}
