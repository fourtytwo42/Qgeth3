// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"crypto/rand"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestNewSNARKVerifier(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	if verifier == nil {
		t.Fatal("NewSNARKVerifier returned nil")
	}
	
	if !verifier.IsAvailable() {
		t.Error("SNARK verifier should be available after initialization")
	}
	
	if verifier.GetName() != "SNARKVerifier_v1.0_gnark" {
		t.Errorf("Expected verifier name 'SNARKVerifier_v1.0_gnark', got '%s'", verifier.GetName())
	}
	
	// Check that CAPSS keys were initialized
	if verifier.capssKeys == nil {
		t.Error("CAPSS verification keys should be initialized")
	}
	
	if verifier.capssKeys.QuantumGateVK == nil {
		t.Error("Quantum gate verification key should be initialized")
	}
	
	if verifier.capssKeys.MeasurementVK == nil {
		t.Error("Measurement verification key should be initialized")
	}
	
	if verifier.capssKeys.EntanglementVK == nil {
		t.Error("Entanglement verification key should be initialized")
	}
	
	if verifier.capssKeys.CircuitDepthVK == nil {
		t.Error("Circuit depth verification key should be initialized")
	}
	
	t.Logf("SNARK verifier initialized successfully with %d verification keys", len(verifier.keys))
}

func TestVerificationKeyGeneration(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Test generating a verification key
	vk, err := verifier.generateStandardVerificationKey("test_circuit", 100)
	if err != nil {
		t.Fatalf("Failed to generate verification key: %v", err)
	}
	
	if vk.Size != 100 {
		t.Errorf("Expected key size 100, got %d", vk.Size)
	}
	
	if len(vk.IC) != 10 {
		t.Errorf("Expected 10 IC elements, got %d", len(vk.IC))
	}
	
	// Check that circuit hash is correct
	expectedHash := [32]byte{}
	if vk.CircuitHash == expectedHash {
		t.Error("Circuit hash should not be all zeros")
	}
	
	// Check that verification key elements are properly initialized
	// Since we generate non-zero elements, they should not be the identity element
	zeroG1 := vk.Alpha.IsInfinity()
	if zeroG1 {
		t.Error("Alpha element should not be the identity (infinity) element")
	}
	
	zeroG2 := vk.Beta.IsInfinity()
	if zeroG2 {
		t.Error("Beta element should not be the identity (infinity) element")
	}
	
	t.Logf("Generated verification key with %d constraints and circuit hash %x", vk.Size, vk.CircuitHash[:8])
}

func TestSNARKProofParsing(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Test with too short proof
	shortProof := make([]byte, 100)
	_, err := verifier.parseSNARKProof(shortProof)
	if err == nil {
		t.Error("Should reject proof that's too short")
	}
	
	// Test with minimum valid size but invalid content
	minProof := make([]byte, 256)
	rand.Read(minProof)
	_, err = verifier.parseSNARKProof(minProof)
	if err == nil {
		t.Error("Should reject proof with invalid elliptic curve points")
	}
	
	t.Logf("SNARK proof parsing correctly rejects invalid proofs")
}

func TestPublicInputsParsing(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Test with invalid length (not multiple of 32)
	invalidInputs := make([]byte, 31)
	_, err := verifier.parsePublicInputs(invalidInputs)
	if err == nil {
		t.Error("Should reject inputs with invalid length")
	}
	
	// Test with valid length
	validInputs := make([]byte, 64) // 2 field elements
	rand.Read(validInputs)
	inputs, err := verifier.parsePublicInputs(validInputs)
	if err != nil {
		t.Errorf("Should accept valid inputs: %v", err)
	}
	
	if len(inputs) != 2 {
		t.Errorf("Expected 2 field elements, got %d", len(inputs))
	}
	
	t.Logf("Successfully parsed %d public inputs", len(inputs))
}

func TestCAPSSProofVerification(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Create a mock CAPSS proof for testing
	proof := &CAPSSProof{
		TraceID:      1000,
		Proof:        make([]byte, 2200), // Original CAPSS proof size
		PublicInputs: make([]byte, 64),   // 2 field elements
		ProofHash:    make([]byte, 32),
	}
	
	// Fill with non-zero data to pass basic checks
	for i := range proof.Proof {
		proof.Proof[i] = byte((i + 1) % 256)
	}
	for i := range proof.PublicInputs {
		proof.PublicInputs[i] = byte((i + 100) % 256)
	}
	
	// Test verification (should fail due to invalid SNARK structure)
	valid, err := verifier.VerifyCAPSSProof(proof)
	if err == nil {
		t.Error("Expected verification to fail with invalid SNARK proof structure")
	}
	
	if valid {
		t.Error("Invalid proof should not be marked as valid")
	}
	
	t.Logf("CAPSS proof verification correctly rejects invalid proofs")
}

func TestSNARKVerifierStatistics(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Get initial stats
	stats := verifier.GetStats()
	if stats.TotalVerifications != 0 {
		t.Error("Initial verification count should be 0")
	}
	
	// Try some verification attempts (they will fail but should update stats)
	proof := &CAPSSProof{
		TraceID:      1001,
		Proof:        make([]byte, 500),
		PublicInputs: make([]byte, 64),
		ProofHash:    make([]byte, 32),
	}
	
	// Fill with data
	for i := range proof.Proof {
		proof.Proof[i] = byte(i % 256)
	}
	for i := range proof.PublicInputs {
		proof.PublicInputs[i] = byte(i % 256)
	}
	
	verifier.VerifyCAPSSProof(proof)
	
	// Check stats updated
	stats = verifier.GetStats()
	if stats.TotalVerifications == 0 {
		t.Error("Verification count should have increased")
	}
	
	if stats.FailedVerifications == 0 {
		t.Error("Failed verification count should have increased")
	}
	
	t.Logf("Statistics tracking works: %d total, %d failed", stats.TotalVerifications, stats.FailedVerifications)
}

func TestVerificationKeyConsistency(t *testing.T) {
	// Test that verification keys are generated consistently
	verifier1 := NewSNARKVerifier()
	verifier2 := NewSNARKVerifier()
	
	// Since we use deterministic generation based on circuit type,
	// keys should be identical
	vk1 := verifier1.capssKeys.QuantumGateVK
	vk2 := verifier2.capssKeys.QuantumGateVK
	
	if vk1.CircuitHash != vk2.CircuitHash {
		t.Error("Verification keys should be generated consistently")
	}
	
	if vk1.Size != vk2.Size {
		t.Error("Verification key sizes should be consistent")
	}
	
	t.Logf("Verification key generation is deterministic and consistent")
}

func TestGroth16VerificationStructuralValidation(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Create mock SNARK proof elements
	proof := &SNARKProof{}
	// Initialize with zero elements (will fail IsOnCurve checks)
	
	publicInputs := make([]fr.Element, 9) // Should match expected inputs for quantum gate VK
	vk := verifier.capssKeys.QuantumGateVK
	
	// Test verification with invalid proof elements
	valid, err := verifier.verifyGroth16Proof(proof, publicInputs, vk)
	if err == nil {
		t.Error("Should reject proof with elements not on curve")
	}
	
	if valid {
		t.Error("Invalid proof should not be marked as valid")
	}
	
	t.Logf("Groth16 structural validation correctly rejects invalid proofs")
}

func TestDeterministicRNG(t *testing.T) {
	seed := []byte("test_seed_for_deterministic_rng")
	
	rng1 := &deterministicRNG{seed: seed, counter: 0}
	rng2 := &deterministicRNG{seed: seed, counter: 0}
	
	data1 := make([]byte, 64)
	data2 := make([]byte, 64)
	
	rng1.Read(data1)
	rng2.Read(data2)
	
	// Should be identical
	for i := range data1 {
		if data1[i] != data2[i] {
			t.Error("Deterministic RNG should produce identical output for same seed")
			break
		}
	}
	
	t.Logf("Deterministic RNG produces consistent output")
}

func TestConcurrentVerification(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	// Test concurrent access to verifier
	done := make(chan bool, 10)
	
	for i := 0; i < 10; i++ {
		go func(id int) {
			proof := &CAPSSProof{
				TraceID:      uint32(2000 + id),
				Proof:        make([]byte, 300),
				PublicInputs: make([]byte, 64),
				ProofHash:    make([]byte, 32),
			}
			
			// Fill with different data per goroutine
			for j := range proof.Proof {
				proof.Proof[j] = byte((j + id) % 256)
			}
			
			verifier.VerifyCAPSSProof(proof)
			done <- true
		}(i)
	}
	
	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		select {
		case <-done:
			// Success
		case <-time.After(5 * time.Second):
			t.Fatal("Timeout waiting for concurrent verification")
		}
	}
	
	stats := verifier.GetStats()
	if stats.TotalVerifications < 10 {
		t.Errorf("Expected at least 10 verifications, got %d", stats.TotalVerifications)
	}
	
	t.Logf("Concurrent verification handled successfully: %d total verifications", stats.TotalVerifications)
}

func TestVerificationKeySelection(t *testing.T) {
	verifier := NewSNARKVerifier()
	
	proof := &CAPSSProof{
		TraceID: 3000,
	}
	
	vk, err := verifier.selectVerificationKey(proof)
	if err != nil {
		t.Fatalf("Failed to select verification key: %v", err)
	}
	
	if vk == nil {
		t.Error("Selected verification key should not be nil")
	}
	
	// Should select quantum gate VK by default
	if vk != verifier.capssKeys.QuantumGateVK {
		t.Error("Should select quantum gate VK as default")
	}
	
	t.Logf("Verification key selection works correctly")
}

func BenchmarkSNARKVerificationSetup(b *testing.B) {
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		verifier := NewSNARKVerifier()
		if !verifier.IsAvailable() {
			b.Fatal("Verifier should be available")
		}
	}
}

func BenchmarkCAPSSProofVerification(b *testing.B) {
	verifier := NewSNARKVerifier()
	
	proof := &CAPSSProof{
		TraceID:      4000,
		Proof:        make([]byte, 300),
		PublicInputs: make([]byte, 64),
		ProofHash:    make([]byte, 32),
	}
	
	// Fill with data
	for i := range proof.Proof {
		proof.Proof[i] = byte(i % 256)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		verifier.VerifyCAPSSProof(proof)
	}
} 