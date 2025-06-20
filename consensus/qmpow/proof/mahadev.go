// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package proof implements Mahadev-style quantum proof verification.
package proof

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// MahadevProof represents a Mahadev-style quantum proof
type MahadevProof struct {
	CircuitSeed []byte // seed used to generate the circuit
	QBits       uint8  // number of qubits
	TCount      uint16 // number of T-gates
	Outcome     []byte // measurement outcome
	Witness     []byte // classical witness/proof
}

// GenerateMahadevProof creates a Mahadev proof for a quantum circuit (simplified for development)
func GenerateMahadevProof(seed []byte, qbits uint8, tcount uint16) (*MahadevProof, error) {
	if qbits == 0 || qbits > 64 {
		return nil, fmt.Errorf("invalid qubit count: %d", qbits)
	}

	if tcount == 0 || tcount > 1000 {
		return nil, fmt.Errorf("invalid T-gate count: %d", tcount)
	}

	// For development, generate deterministic outcome from seed
	h := sha256.New()
	h.Write(seed)
	binary.Write(h, binary.LittleEndian, qbits)
	binary.Write(h, binary.LittleEndian, tcount)
	h.Write([]byte("quantum_outcome"))

	digest := h.Sum(nil)

	// Extract outcome bits (convert to bytes)
	outcomeBytes := int(qbits+7) / 8
	outcome := make([]byte, outcomeBytes)
	copy(outcome, digest[:outcomeBytes])

	// Mask the last byte if qbits is not a multiple of 8
	if qbits%8 != 0 {
		lastByteMask := byte(0xFF) >> (8 - qbits%8)
		outcome[outcomeBytes-1] &= lastByteMask
	}

	// Generate a simple witness (in real implementation, this would be the Mahadev witness)
	witnessH := sha256.New()
	witnessH.Write(seed)
	witnessH.Write(outcome)
	binary.Write(witnessH, binary.LittleEndian, qbits)
	binary.Write(witnessH, binary.LittleEndian, tcount)
	witnessH.Write([]byte("mahadev_witness"))

	// Create a fixed-size witness
	witness := witnessH.Sum(nil)

	return &MahadevProof{
		CircuitSeed: seed,
		QBits:       qbits,
		TCount:      tcount,
		Outcome:     outcome,
		Witness:     witness,
	}, nil
}

// VerifyMahadevProof verifies a Mahadev proof (simplified for development)
func VerifyMahadevProof(proof *MahadevProof) error {
	if proof == nil {
		return fmt.Errorf("nil proof")
	}

	// Regenerate expected outcome from the same seed
	expectedProof, err := GenerateMahadevProof(proof.CircuitSeed, proof.QBits, proof.TCount)
	if err != nil {
		return fmt.Errorf("failed to generate expected proof: %v", err)
	}

	// Compare outcomes
	if len(proof.Outcome) != len(expectedProof.Outcome) {
		return fmt.Errorf("outcome length mismatch: got %d, expected %d",
			len(proof.Outcome), len(expectedProof.Outcome))
	}

	for i := range proof.Outcome {
		if proof.Outcome[i] != expectedProof.Outcome[i] {
			return fmt.Errorf("outcome mismatch at byte %d: got 0x%02x, expected 0x%02x",
				i, proof.Outcome[i], expectedProof.Outcome[i])
		}
	}

	// Compare witnesses
	if len(proof.Witness) != len(expectedProof.Witness) {
		return fmt.Errorf("witness length mismatch: got %d, expected %d",
			len(proof.Witness), len(expectedProof.Witness))
	}

	for i := range proof.Witness {
		if proof.Witness[i] != expectedProof.Witness[i] {
			return fmt.Errorf("witness mismatch at byte %d", i)
		}
	}

	return nil
}

// QubitMeasurement represents the result of measuring a single qubit
type QubitMeasurement struct {
	QubitIndex uint8 // which qubit was measured
	Outcome    bool  // measurement result (0 or 1)
}

// ExtractQubitMeasurements extracts individual qubit measurements from the outcome
func (mp *MahadevProof) ExtractQubitMeasurements() []QubitMeasurement {
	measurements := make([]QubitMeasurement, mp.QBits)

	for i := uint8(0); i < mp.QBits; i++ {
		byteIndex := i / 8
		bitIndex := i % 8

		bit := false
		if byteIndex < uint8(len(mp.Outcome)) {
			bit = (mp.Outcome[byteIndex] & (1 << bitIndex)) != 0
		}

		measurements[i] = QubitMeasurement{
			QubitIndex: i,
			Outcome:    bit,
		}
	}

	return measurements
}

// ComputeCircuitComplexity estimates the computational complexity of the quantum circuit
func (mp *MahadevProof) ComputeCircuitComplexity() uint64 {
	// Simple complexity metric: 2^qbits * tcount
	// In reality, this would be much more sophisticated
	if mp.QBits > 30 { // Prevent overflow
		return ^uint64(0) // Max value
	}

	stateSpaceSize := uint64(1) << mp.QBits
	return stateSpaceSize * uint64(mp.TCount)
}

// IsValidQuantumState checks if the outcome represents a valid quantum measurement
func (mp *MahadevProof) IsValidQuantumState() bool {
	// Check that outcome length matches qubit count
	expectedBytes := int(mp.QBits+7) / 8
	if len(mp.Outcome) != expectedBytes {
		return false
	}

	// Check that unused bits in the last byte are zero
	if mp.QBits%8 != 0 {
		lastByte := mp.Outcome[len(mp.Outcome)-1]
		unusedBits := 8 - (mp.QBits % 8)
		mask := byte(0xFF) << (8 - unusedBits)
		if (lastByte & mask) != 0 {
			return false
		}
	}

	return true
}

// Serialize converts the Mahadev proof to bytes
func (mp *MahadevProof) Serialize() []byte {
	// Format: SeedLen(4) + Seed + QBits(1) + TCount(2) + OutcomeLen(4) + Outcome + WitnessLen(4) + Witness
	buf := make([]byte, 0, 4+len(mp.CircuitSeed)+1+2+4+len(mp.Outcome)+4+len(mp.Witness))

	// Seed
	binary.LittleEndian.PutUint32(buf[len(buf):len(buf)+4], uint32(len(mp.CircuitSeed)))
	buf = buf[:len(buf)+4]
	buf = append(buf, mp.CircuitSeed...)

	// Circuit parameters
	buf = append(buf, mp.QBits)
	binary.LittleEndian.PutUint16(buf[len(buf):len(buf)+2], mp.TCount)
	buf = buf[:len(buf)+2]

	// Outcome
	binary.LittleEndian.PutUint32(buf[len(buf):len(buf)+4], uint32(len(mp.Outcome)))
	buf = buf[:len(buf)+4]
	buf = append(buf, mp.Outcome...)

	// Witness
	binary.LittleEndian.PutUint32(buf[len(buf):len(buf)+4], uint32(len(mp.Witness)))
	buf = buf[:len(buf)+4]
	buf = append(buf, mp.Witness...)

	return buf
}

// DeserializeMahadevProof reconstructs a Mahadev proof from bytes
func DeserializeMahadevProof(data []byte) (*MahadevProof, error) {
	if len(data) < 4+1+2+4+4 {
		return nil, fmt.Errorf("insufficient data length")
	}

	offset := 0

	// Read seed
	seedLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if offset+int(seedLen) > len(data) {
		return nil, fmt.Errorf("invalid seed length")
	}
	seed := make([]byte, seedLen)
	copy(seed, data[offset:offset+int(seedLen)])
	offset += int(seedLen)

	// Read circuit parameters
	qbits := data[offset]
	offset += 1
	tcount := binary.LittleEndian.Uint16(data[offset:])
	offset += 2

	// Read outcome
	outcomeLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if offset+int(outcomeLen) > len(data) {
		return nil, fmt.Errorf("invalid outcome length")
	}
	outcome := make([]byte, outcomeLen)
	copy(outcome, data[offset:offset+int(outcomeLen)])
	offset += int(outcomeLen)

	// Read witness
	witnessLen := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if offset+int(witnessLen) > len(data) {
		return nil, fmt.Errorf("invalid witness length")
	}
	witness := make([]byte, witnessLen)
	copy(witness, data[offset:offset+int(witnessLen)])

	return &MahadevProof{
		CircuitSeed: seed,
		QBits:       qbits,
		TCount:      tcount,
		Outcome:     outcome,
		Witness:     witness,
	}, nil
}

// BatchVerifyMahadevProofs verifies multiple Mahadev proofs efficiently
func BatchVerifyMahadevProofs(proofs []*MahadevProof) error {
	for i, proof := range proofs {
		if err := VerifyMahadevProof(proof); err != nil {
			return fmt.Errorf("proof %d verification failed: %v", i, err)
		}
	}
	return nil
}
