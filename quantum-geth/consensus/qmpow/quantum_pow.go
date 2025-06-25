// Copyright 2024 The Quantum-Geth Authors
// This file is part of the Quantum-Geth library.
//
// The Quantum-Geth library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// FIXED QUANTUM PARAMETERS - These are now ENFORCED and cannot be reduced by miners
const (
	FixedQBits  = 16  // ENFORCED: Exactly 16 qubits per puzzle
	FixedTCount = 20  // ENFORCED: Minimum 20 T-gates per puzzle (was 8192)
	FixedLNet   = 128 // ENFORCED: Exactly 128 chained puzzles per block for maximum entropy
)

// QuantumConstants defines the constants for quantum proof-of-work
type QuantumConstants struct {
	EpochBlocks      uint64 // 50,000 blocks per epoch
	GlideBlocks      uint64 // 12,500 blocks per qubit increase
	StartingQBits    uint16 // 12 qubits at start
	FixedTCount      uint32 // 4,096 T-gates constant
	FixedLNet        uint16 // 128 chained puzzles constant
	MaxNonceAttempts uint64 // 4 billion nonce attempts
}

// QuantumSpec holds the quantum proof-of-work specification constants
var QuantumSpec = QuantumConstants{
	EpochBlocks:      50000,
	GlideBlocks:      12500,
	StartingQBits:    12,
	FixedTCount:      4096,
	FixedLNet:        128,
	MaxNonceAttempts: 4000000000,
}

// CalculateSeedChain generates the seed chain for quantum puzzles
// Seed₀ = SHA256(ParentHash || TxRoot || ExtraNonce || QNonce)
// Seedᵢ = SHA256(Seedᵢ₋₁ || Outcomeᵢ₋₁)
func CalculateSeedChain(header *types.Header) []common.Hash {
	seeds := make([]common.Hash, QuantumSpec.FixedLNet+1)

	// Calculate Seed₀
	h := sha256.New()
	h.Write(header.ParentHash.Bytes())
	h.Write(header.TxHash.Bytes())
	h.Write(header.ExtraNonce32)

	// Convert QNonce64 to bytes
	qnonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(qnonceBytes, *header.QNonce64)
	h.Write(qnonceBytes)

	seeds[0] = common.BytesToHash(h.Sum(nil))

	// For development, generate deterministic but realistic seeds
	for i := uint16(1); i <= QuantumSpec.FixedLNet; i++ {
		h := sha256.New()
		h.Write(seeds[i-1].Bytes())

		// Generate deterministic "outcome" for seed chain
		// In real implementation, this would be the actual quantum measurement outcome
		mockOutcome := make([]byte, (*header.QBits+7)/8)
		for j := range mockOutcome {
			mockOutcome[j] = byte((uint64(i)*uint64(j) + *header.QNonce64) % 256)
		}
		h.Write(mockOutcome)

		seeds[i] = common.BytesToHash(h.Sum(nil))
	}

	return seeds
}

// GenerateBranchNibbles creates the branch nibbles for template selection
// Each puzzle uses high nibble of its outcome for template selection
func GenerateBranchNibbles(header *types.Header, outcomes [][]byte) error {
	if len(outcomes) != int(QuantumSpec.FixedLNet) {
		return fmt.Errorf("invalid outcomes count: got %d, expected %d", len(outcomes), QuantumSpec.FixedLNet)
	}

	header.BranchNibbles = make([]byte, BranchNibblesSize)

	for i := 0; i < int(QuantumSpec.FixedLNet); i++ {
		if len(outcomes[i]) == 0 {
			return fmt.Errorf("empty outcome for puzzle %d", i)
		}

		// Extract high nibble from outcome
		lastByte := outcomes[i][len(outcomes[i])-1]
		highNibble := (lastByte >> 4) & 0x0F
		header.BranchNibbles[i] = highNibble
	}

	return nil
}

// CalculateOutcomeRoot computes the Merkle root of 128 × QBits bits
func CalculateOutcomeRoot(outcomes [][]byte) common.Hash {
	// Simple implementation - in production would use proper Merkle tree
	h := sha256.New()
	h.Write([]byte("QUANTUM_OUTCOME_ROOT"))

	for _, outcome := range outcomes {
		h.Write(outcome)
	}

	return common.BytesToHash(h.Sum(nil))
}

// CalculateGateHash computes SHA-256 of concatenated canonical gate streams
func CalculateGateHash(seeds []common.Hash, qbits uint16, tcount uint32) common.Hash {
	h := sha256.New()
	h.Write([]byte("QUANTUM_GATE_HASH"))

	// For each puzzle, generate deterministic gate stream representation
	for i, seed := range seeds[:QuantumSpec.FixedLNet] {
		// Simulate canonical compiler output
		gateStream := generateCanonicalGateStream(seed, qbits, tcount, uint16(i))
		h.Write(gateStream)
	}

	return common.BytesToHash(h.Sum(nil))
}

// generateCanonicalGateStream simulates the canonical compiler output
func generateCanonicalGateStream(seed common.Hash, qbits uint16, tcount uint32, puzzleIndex uint16) []byte {
	// Deterministic but realistic gate stream generation
	h := sha256.New()
	h.Write(seed.Bytes())
	h.Write([]byte{byte(qbits), byte(qbits >> 8)})
	h.Write([]byte{byte(tcount), byte(tcount >> 8), byte(tcount >> 16), byte(tcount >> 24)})
	h.Write([]byte{byte(puzzleIndex), byte(puzzleIndex >> 8)})

	// Generate deterministic gate sequence
	streamHash := h.Sum(nil)

	// Simulate gate stream (simplified)
	gateStream := make([]byte, 64) // 64-byte gate stream representation
	for i := range gateStream {
		gateStream[i] = streamHash[i%32] ^ byte(i)
	}

	return gateStream
}

// CalculateProofRoot computes Merkle root of three Tier-B Nova proofs
func CalculateProofRoot(outcomes [][]byte, gateHash common.Hash) common.Hash {
	h := sha256.New()
	h.Write([]byte("QUANTUM_PROOF_ROOT"))
	h.Write(gateHash.Bytes())

	// Simulate three Tier-B Nova proof batches
	for batchIdx := 0; batchIdx < 3; batchIdx++ {
		batchHash := sha256.New()
		batchHash.Write([]byte{byte(batchIdx)})

		// Each batch covers 16 puzzles (128/8 = 16)
		startPuzzle := batchIdx * 16
		endPuzzle := startPuzzle + 16
		if endPuzzle > len(outcomes) {
			endPuzzle = len(outcomes)
		}

		for i := startPuzzle; i < endPuzzle; i++ {
			batchHash.Write(outcomes[i])
		}

		h.Write(batchHash.Sum(nil))
	}

	return common.BytesToHash(h.Sum(nil))
}

// SolveQuantumPuzzlesOriginal implements quantum puzzle solving (simulation for development)
func (q *QMPoW) SolveQuantumPuzzlesOriginal(header *types.Header) error {
	log.Info("🔬 Solving quantum puzzles",
		"blockNumber", header.Number.Uint64(),
		"epoch", *header.Epoch,
		"qbits", *header.QBits,
		"puzzles", *header.LNet)

	start := time.Now()

	// Generate seed chain
	seeds := CalculateSeedChain(header)

	// Solve puzzles (simulation for development)
	outcomes := make([][]byte, QuantumSpec.FixedLNet)
	outcomeLen := int((*header.QBits + 7) / 8)

	for i := uint16(0); i < QuantumSpec.FixedLNet; i++ {
		outcome := make([]byte, outcomeLen)

		// Generate deterministic but nonce-sensitive quantum outcomes
		seed := seeds[i]
		h := sha256.New()
		h.Write(seed.Bytes())

		// Add nonce-specific entropy for each puzzle
		nonceBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(nonceBytes, *header.QNonce64)
		h.Write(nonceBytes)
		h.Write([]byte{byte(i), byte(i >> 8)}) // Puzzle index

		puzzleHash := h.Sum(nil)
		for j := 0; j < outcomeLen; j++ {
			outcome[j] = puzzleHash[j%32] ^ byte(j)
		}

		outcomes[i] = outcome

		// Simulate quantum computation time
		time.Sleep(time.Duration(10+rand.Intn(20)) * time.Millisecond)
	}

	// Generate branch nibbles
	if err := GenerateBranchNibbles(header, outcomes); err != nil {
		return fmt.Errorf("failed to generate branch nibbles: %v", err)
	}

	// Calculate outcome root
	outcomeRoot := CalculateOutcomeRoot(outcomes)
	header.OutcomeRoot = &outcomeRoot

	// Calculate gate hash
	gateHash := CalculateGateHash(seeds, *header.QBits, *header.TCount)
	header.GateHash = &gateHash

	// Calculate proof root
	proofRoot := CalculateProofRoot(outcomes, gateHash)
	header.ProofRoot = &proofRoot

	solvingTime := time.Since(start)

	log.Info("✅ Quantum puzzles solved",
		"blockNumber", header.Number.Uint64(),
		"solvingTime", solvingTime,
		"outcomeRoot", outcomeRoot.Hex(),
		"gateHash", gateHash.Hex(),
		"proofRoot", proofRoot.Hex())

	return nil
}

// ValidateQuantumProof validates a complete quantum proof
func ValidateQuantumProof(header *types.Header) error {
	// Verify all required fields are present
	if header.OutcomeRoot == nil {
		return fmt.Errorf("missing OutcomeRoot")
	}
	if header.GateHash == nil {
		return fmt.Errorf("missing GateHash")
	}
	if header.ProofRoot == nil {
		return fmt.Errorf("missing ProofRoot")
	}
	if len(header.BranchNibbles) != BranchNibblesSize {
		return fmt.Errorf("invalid BranchNibbles size: got %d, expected %d",
			len(header.BranchNibbles), BranchNibblesSize)
	}

	// Additional validation would go here
	// - Verify quantum circuit compilation
	// - Validate measurement outcomes
	// - Check proof consistency

	return nil
}

// fillExtraNonce32 fills the ExtraNonce32 field with random data
func (q *QMPoW) fillExtraNonce32(header *types.Header, qnonce uint64) {
	if len(header.ExtraNonce32) != ExtraNonce32Size {
		header.ExtraNonce32 = make([]byte, ExtraNonce32Size)
	}

	// Fill with deterministic but varied data based on qnonce
	h := sha256.New()
	h.Write(header.ParentHash.Bytes())
	h.Write(header.TxHash.Bytes())

	qnonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(qnonceBytes, qnonce)
	h.Write(qnonceBytes)

	extraData := h.Sum(nil)
	copy(header.ExtraNonce32, extraData[:ExtraNonce32Size])
}

// checkQuantumTargetOriginal checks if the quantum proof meets the difficulty target
func (q *QMPoW) checkQuantumTargetOriginal(header *types.Header) bool {
	// Calculate quantum proof quality from all quantum fields
	h := sha256.New()
	h.Write(header.OutcomeRoot.Bytes())
	h.Write(header.GateHash.Bytes())
	h.Write(header.ProofRoot.Bytes())
	h.Write(header.BranchNibbles)

	// Include nonce in quality calculation
	nonceBytes := make([]byte, 8)
	for i := 0; i < 8; i++ {
		nonceBytes[i] = byte(*header.QNonce64 >> (8 * (7 - i)))
	}
	h.Write(nonceBytes)

	proofHash := h.Sum(nil)

	// Simple target check - in production would use proper difficulty calculation
	target := make([]byte, 32)
	target[0] = 0x00
	target[1] = 0x00
	target[2] = 0x0F // Adjust difficulty as needed

	for i := 0; i < 3; i++ {
		if proofHash[i] < target[i] {
			return true
		} else if proofHash[i] > target[i] {
			return false
		}
	}

	return false
}
