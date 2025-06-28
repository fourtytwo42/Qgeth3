// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package proof implements quantum proof aggregation and verification.
package proof

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/ethereum/go-ethereum/common"
)

// Proof represents a single quantum micro-puzzle proof
type Proof struct {
	PuzzleIndex uint16 // index of the puzzle in the chain
	Outcome     []byte // measurement outcome (q bits)
	Witness     []byte // Mahadev witness/proof
}

// AggregateProof represents the combined proof for all puzzles in a block
type AggregateProof struct {
	NumPuzzles uint16   // number of puzzles (L_net)
	QBits      uint8    // qubits per puzzle
	TCount     uint16   // T-gates per puzzle
	Outcomes   []byte   // concatenated outcomes
	ProofData  []byte   // aggregated proof blob
	MerkleRoot [32]byte // Merkle root of individual proofs
}

// CreateAggregate combines individual puzzle proofs into an aggregate proof
func CreateAggregate(proofs []Proof, qbits uint8, tcount uint16) (*AggregateProof, error) {
	if len(proofs) == 0 {
		return nil, fmt.Errorf("no proofs provided")
	}

	numPuzzles := uint16(len(proofs))

	// Calculate expected outcome length per puzzle (bits -> bytes)
	expectedOutcomeLen := int(qbits+7) / 8 // ceiling division for bit to byte conversion

	// Concatenate all outcomes
	outcomes := make([]byte, 0, numPuzzles*uint16(expectedOutcomeLen))
	proofHashes := make([][32]byte, numPuzzles)

	for i, proof := range proofs {
		if len(proof.Outcome) != expectedOutcomeLen {
			return nil, fmt.Errorf("proof %d: invalid outcome length %d, expected %d",
				i, len(proof.Outcome), expectedOutcomeLen)
		}

		outcomes = append(outcomes, proof.Outcome...)

		// Create hash of individual proof for Merkle tree
		h := sha256.New()
		binary.Write(h, binary.LittleEndian, proof.PuzzleIndex)
		h.Write(proof.Outcome)
		h.Write(proof.Witness)
		copy(proofHashes[i][:], h.Sum(nil))
	}

	// Build simple Merkle tree of proof hashes
	merkleRoot := buildMerkleTree(proofHashes)

	// For now, use a simple concatenation of witness data as aggregate proof
	// In a real implementation, this would use proper Mahadev aggregation
	proofData := make([]byte, 0)
	for _, proof := range proofs {
		proofData = append(proofData, proof.Witness...)
	}

	return &AggregateProof{
		NumPuzzles: numPuzzles,
		QBits:      qbits,
		TCount:     tcount,
		Outcomes:   outcomes,
		ProofData:  proofData,
		MerkleRoot: merkleRoot,
	}, nil
}

// VerifyAggregate verifies an aggregate quantum proof
func VerifyAggregate(proof *AggregateProof, seed []byte, parentHash common.Hash) error {
	if proof == nil {
		return fmt.Errorf("nil proof")
	}

	// Check outcome length consistency
	expectedOutcomeLen := int(proof.QBits+7) / 8
	totalExpectedLen := int(proof.NumPuzzles) * expectedOutcomeLen
	if len(proof.Outcomes) != totalExpectedLen {
		return fmt.Errorf("invalid total outcome length: got %d, expected %d",
			len(proof.Outcomes), totalExpectedLen)
	}

	// For testnet/development, we'll use a simplified verification
	// that just checks the proof structure is valid

	// Extract individual outcomes
	outcomes := make([][]byte, proof.NumPuzzles)
	for i := uint16(0); i < proof.NumPuzzles; i++ {
		start := int(i) * expectedOutcomeLen
		end := start + expectedOutcomeLen
		outcomes[i] = proof.Outcomes[start:end]
	}

	// Verify the chain structure (each puzzle depends on previous outcome)
	currentSeed := seed
	for i := uint16(0); i < proof.NumPuzzles; i++ {
		// Verify this outcome is consistent with the seed
		if err := verifyPuzzleOutcome(currentSeed, outcomes[i], proof.QBits, proof.TCount); err != nil {
			return fmt.Errorf("puzzle %d verification failed: %v", i, err)
		}

		// Generate seed for next puzzle
		if i < proof.NumPuzzles-1 {
			h := sha256.New()
			h.Write(currentSeed)
			h.Write(outcomes[i])
			currentSeed = h.Sum(nil)
		}
	}

	return nil
}

// verifyPuzzleOutcome verifies a single puzzle outcome (simplified for development)
func verifyPuzzleOutcome(seed []byte, outcome []byte, qbits uint8, tcount uint16) error {
	// In a real implementation, this would verify the Mahadev proof
	// For development, we'll just do basic sanity checks

	if len(outcome) != int(qbits+7)/8 {
		return fmt.Errorf("invalid outcome length")
	}

	// Use the seed to generate expected outcome pattern (deterministic for testing)
	h := sha256.New()
	h.Write(seed)
	binary.Write(h, binary.LittleEndian, qbits)
	binary.Write(h, binary.LittleEndian, tcount)
	expected := h.Sum(nil)

	// Compare with actual outcome (masked to qbits)
	for i := 0; i < len(outcome); i++ {
		mask := byte(0xFF)
		if i == len(outcome)-1 && qbits%8 != 0 {
			// Mask the last byte to only include valid bits
			mask = byte(0xFF) >> (8 - qbits%8)
		}

		if (outcome[i] & mask) != (expected[i] & mask) {
			return fmt.Errorf("outcome mismatch at byte %d", i)
		}
	}

	return nil
}

// buildMerkleTree builds a simple binary Merkle tree from proof hashes
func buildMerkleTree(hashes [][32]byte) [32]byte {
	if len(hashes) == 0 {
		return [32]byte{}
	}

	if len(hashes) == 1 {
		return hashes[0]
	}

	// Build tree level by level
	current := make([][32]byte, len(hashes))
	copy(current, hashes)

	for len(current) > 1 {
		next := make([][32]byte, 0, (len(current)+1)/2)

		for i := 0; i < len(current); i += 2 {
			if i+1 < len(current) {
				// Hash pair
				h := sha256.New()
				h.Write(current[i][:])
				h.Write(current[i+1][:])
				var combined [32]byte
				copy(combined[:], h.Sum(nil))
				next = append(next, combined)
			} else {
				// Odd number, promote single hash
				next = append(next, current[i])
			}
		}

		current = next
	}

	return current[0]
}

// ExtractOutcomes extracts individual puzzle outcomes from aggregate proof
func (ap *AggregateProof) ExtractOutcomes() [][]byte {
	expectedOutcomeLen := int(ap.QBits+7) / 8
	outcomes := make([][]byte, ap.NumPuzzles)

	for i := uint16(0); i < ap.NumPuzzles; i++ {
		start := int(i) * expectedOutcomeLen
		end := start + expectedOutcomeLen
		outcomes[i] = make([]byte, expectedOutcomeLen)
		copy(outcomes[i], ap.Outcomes[start:end])
	}

	return outcomes
}

// Serialize converts the aggregate proof to bytes for storage/transmission
func (ap *AggregateProof) Serialize() []byte {
	// Format: NumPuzzles(2) + QBits(1) + TCount(2) + OutcomeLen(4) + Outcomes + ProofLen(4) + ProofData + MerkleRoot(32)
	buf := make([]byte, 0, 2+1+2+4+len(ap.Outcomes)+4+len(ap.ProofData)+32)

	// Header
	binary.LittleEndian.PutUint16(buf[len(buf):len(buf)+2], ap.NumPuzzles)
	buf = buf[:len(buf)+2]
	buf = append(buf, ap.QBits)
	binary.LittleEndian.PutUint16(buf[len(buf):len(buf)+2], ap.TCount)
	buf = buf[:len(buf)+2]

	// Outcomes
	binary.LittleEndian.PutUint32(buf[len(buf):len(buf)+4], uint32(len(ap.Outcomes)))
	buf = buf[:len(buf)+4]
	buf = append(buf, ap.Outcomes...)

	// Proof data
	binary.LittleEndian.PutUint32(buf[len(buf):len(buf)+4], uint32(len(ap.ProofData)))
	buf = buf[:len(buf)+4]
	buf = append(buf, ap.ProofData...)

	// Merkle root
	buf = append(buf, ap.MerkleRoot[:]...)

	return buf
}

// DeserializeAggregateProof reconstructs an aggregate proof from bytes
func DeserializeAggregateProof(data []byte) (*AggregateProof, error) {
	if len(data) < 2+1+2+4+4+32 {
		return nil, fmt.Errorf("insufficient data length")
	}

	offset := 0

	// Read header
	numPuzzles := binary.LittleEndian.Uint16(data[offset:])
	offset += 2
	qbits := data[offset]
	offset += 1
	tcount := binary.LittleEndian.Uint16(data[offset:])
	offset += 2

	// Read outcomes
	outcomeLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if offset+int(outcomeLen) > len(data) {
		return nil, fmt.Errorf("invalid outcome length")
	}
	outcomes := make([]byte, outcomeLen)
	copy(outcomes, data[offset:offset+int(outcomeLen)])
	offset += int(outcomeLen)

	// Read proof data
	proofLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if offset+int(proofLen) > len(data) {
		return nil, fmt.Errorf("invalid proof length")
	}
	proofData := make([]byte, proofLen)
	copy(proofData, data[offset:offset+int(proofLen)])
	offset += int(proofLen)

	// Read Merkle root
	if offset+32 > len(data) {
		return nil, fmt.Errorf("insufficient data for Merkle root")
	}
	var merkleRoot [32]byte
	copy(merkleRoot[:], data[offset:offset+32])

	return &AggregateProof{
		NumPuzzles: numPuzzles,
		QBits:      qbits,
		TCount:     tcount,
		Outcomes:   outcomes,
		ProofData:  proofData,
		MerkleRoot: merkleRoot,
	}, nil
}
