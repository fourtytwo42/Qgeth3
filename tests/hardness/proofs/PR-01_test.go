// PR-01: Mahadev Trace Integrity Test
// Tests generation of 48 Mahadev transcripts for quantum circuits and verification
// Attack Vector: Faulty trapdoor usage
// Expected Outcome: Verifier accepts all 48 valid proofs, rejects invalid ones

package proofs

import (
	"crypto/rand"
	"crypto/sha256"
	"testing"
	"time"
)

// MahadevTranscript represents a Mahadev proof transcript
type MahadevTranscript struct {
	CircuitID    int      `json:"circuit_id"`
	TrapdoorKey  []byte   `json:"trapdoor_key"`
	Commitment   []byte   `json:"commitment"`
	Challenge    []byte   `json:"challenge"`
	Response     []byte   `json:"response"`
	Witness      []byte   `json:"witness"`
	ProofData    []byte   `json:"proof_data"`
	IsValid      bool     `json:"is_valid"`
}

// Test PR-01: Mahadev Trace Integrity
func TestPR01_MahadevTraceIntegrity(t *testing.T) {
	puzzleCount := 48
	qubits := 4 // Toy circuit for testing
	
	t.Logf("🔍 PR-01: Testing Mahadev trace integrity for %d puzzles", puzzleCount)
	t.Logf("   Circuit: %d qubits (toy circuit for testing)", qubits)
	
	// Generate 48 Mahadev transcripts
	transcripts := make([]*MahadevTranscript, puzzleCount)
	validCount := 0
	
	for i := 0; i < puzzleCount; i++ {
		t.Logf("   Generating Mahadev transcript %d/%d", i+1, puzzleCount)
		
		transcript, err := generateMahadevTranscript(i, qubits)
		if err != nil {
			t.Errorf("❌ Failed to generate transcript %d: %v", i, err)
			continue
		}
		
		transcripts[i] = transcript
		
		// Verify each transcript
		isValid := verifyMahadevTranscript(transcript)
		transcript.IsValid = isValid
		
		if isValid {
			validCount++
			t.Logf("   ✅ Transcript %d: Valid", i)
		} else {
			t.Errorf("   ❌ Transcript %d: Invalid", i)
		}
	}
	
	// All transcripts should be valid
	if validCount != puzzleCount {
		t.Errorf("❌ PR-01 FAILED: Only %d/%d transcripts are valid", validCount, puzzleCount)
	} else {
		t.Logf("✅ PR-01 PASSED: All %d transcripts are valid", puzzleCount)
	}
	
	// Test attack vector: Faulty trapdoor usage
	t.Log("   Testing attack vector: Faulty trapdoor usage")
	testFaultyTrapdoorAttack(t, transcripts[0])
}

// generateMahadevTranscript generates a Mahadev proof transcript for a quantum circuit
func generateMahadevTranscript(circuitID, qubits int) (*MahadevTranscript, error) {
	// Generate trapdoor key (in real implementation, this would be more complex)
	trapdoorKey := make([]byte, 32)
	if _, err := rand.Read(trapdoorKey); err != nil {
		return nil, err
	}
	
	// Generate commitment based on circuit and trapdoor
	commitment := generateCommitment(circuitID, qubits, trapdoorKey)
	
	// Generate challenge (Fiat-Shamir transform)
	challenge := generateChallenge(commitment)
	
	// Generate response using trapdoor
	response := generateResponse(challenge, trapdoorKey)
	
	// Generate witness (quantum execution trace)
	witness := generateQuantumWitness(circuitID, qubits)
	
	// Create proof data
	proofData := createProofData(commitment, challenge, response, witness)
	
	return &MahadevTranscript{
		CircuitID:   circuitID,
		TrapdoorKey: trapdoorKey,
		Commitment:  commitment,
		Challenge:   challenge,
		Response:    response,
		Witness:     witness,
		ProofData:   proofData,
	}, nil
}

// verifyMahadevTranscript verifies a Mahadev proof transcript
func verifyMahadevTranscript(transcript *MahadevTranscript) bool {
	// Verify commitment consistency
	expectedCommitment := generateCommitment(transcript.CircuitID, 4, transcript.TrapdoorKey)
	if !bytesEqual(transcript.Commitment, expectedCommitment) {
		return false
	}
	
	// Verify challenge consistency (Fiat-Shamir)
	expectedChallenge := generateChallenge(transcript.Commitment)
	if !bytesEqual(transcript.Challenge, expectedChallenge) {
		return false
	}
	
	// Verify response consistency
	expectedResponse := generateResponse(transcript.Challenge, transcript.TrapdoorKey)
	if !bytesEqual(transcript.Response, expectedResponse) {
		return false
	}
	
	// Verify witness integrity
	if !verifyQuantumWitness(transcript.CircuitID, transcript.Witness) {
		return false
	}
	
	// Verify proof data integrity
	expectedProofData := createProofData(transcript.Commitment, transcript.Challenge, 
		transcript.Response, transcript.Witness)
	if !bytesEqual(transcript.ProofData, expectedProofData) {
		return false
	}
	
	return true
}

// testFaultyTrapdoorAttack tests the security against faulty trapdoor usage
func testFaultyTrapdoorAttack(t *testing.T, originalTranscript *MahadevTranscript) {
	t.Log("   🔴 ATTACK: Testing faulty trapdoor usage")
	
	// Attack 1: Wrong trapdoor key
	faultyTranscript1 := *originalTranscript
	faultyKey := make([]byte, 32)
	for i := range faultyKey {
		faultyKey[i] = 0xFF // All ones
	}
	faultyTranscript1.TrapdoorKey = faultyKey
	
	if verifyMahadevTranscript(&faultyTranscript1) {
		t.Error("   ❌ SECURITY VULNERABILITY: Wrong trapdoor key accepted")
	} else {
		t.Log("   ✅ Wrong trapdoor key rejected correctly")
	}
	
	// Attack 2: Tampered commitment
	faultyTranscript2 := *originalTranscript
	faultyTranscript2.Commitment[0] ^= 0x01 // Flip one bit
	
	if verifyMahadevTranscript(&faultyTranscript2) {
		t.Error("   ❌ SECURITY VULNERABILITY: Tampered commitment accepted")
	} else {
		t.Log("   ✅ Tampered commitment rejected correctly")
	}
	
	// Attack 3: Invalid response
	faultyTranscript3 := *originalTranscript
	faultyTranscript3.Response[0] ^= 0x01 // Flip one bit
	
	if verifyMahadevTranscript(&faultyTranscript3) {
		t.Error("   ❌ SECURITY VULNERABILITY: Invalid response accepted")
	} else {
		t.Log("   ✅ Invalid response rejected correctly")
	}
}

// Helper functions for Mahadev proof simulation
func generateCommitment(circuitID, qubits int, trapdoorKey []byte) []byte {
	hasher := sha256.New()
	hasher.Write([]byte("commitment"))
	hasher.Write([]byte{byte(circuitID)})
	hasher.Write([]byte{byte(qubits)})
	hasher.Write(trapdoorKey)
	return hasher.Sum(nil)
}

func generateChallenge(commitment []byte) []byte {
	hasher := sha256.New()
	hasher.Write([]byte("challenge"))
	hasher.Write(commitment)
	return hasher.Sum(nil)
}

func generateResponse(challenge, trapdoorKey []byte) []byte {
	hasher := sha256.New()
	hasher.Write([]byte("response"))
	hasher.Write(challenge)
	hasher.Write(trapdoorKey)
	return hasher.Sum(nil)
}

func generateQuantumWitness(circuitID, qubits int) []byte {
	hasher := sha256.New()
	hasher.Write([]byte("witness"))
	hasher.Write([]byte{byte(circuitID)})
	hasher.Write([]byte{byte(qubits)})
	return hasher.Sum(nil)
}

func verifyQuantumWitness(circuitID int, witness []byte) bool {
	expected := generateQuantumWitness(circuitID, 4)
	return bytesEqual(witness, expected)
}

func createProofData(commitment, challenge, response, witness []byte) []byte {
	hasher := sha256.New()
	hasher.Write([]byte("proofdata"))
	hasher.Write(commitment)
	hasher.Write(challenge)
	hasher.Write(response)
	hasher.Write(witness)
	return hasher.Sum(nil)
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Test batch verification performance
func TestPR01_BatchVerificationPerformance(t *testing.T) {
	puzzleCount := 48
	
	t.Log("🔍 PR-01 Performance: Testing batch verification performance")
	
	// Generate all transcripts
	transcripts := make([]*MahadevTranscript, puzzleCount)
	for i := 0; i < puzzleCount; i++ {
		transcript, err := generateMahadevTranscript(i, 4)
		if err != nil {
			t.Fatalf("Failed to generate transcript %d: %v", i, err)
		}
		transcripts[i] = transcript
	}
	
	// Measure individual verification time
	start := time.Now()
	for _, transcript := range transcripts {
		if !verifyMahadevTranscript(transcript) {
			t.Error("Individual verification failed")
		}
	}
	individualTime := time.Since(start)
	
	t.Logf("   Individual verification: %v (%v per proof)", 
		individualTime, individualTime/time.Duration(puzzleCount))
	
	// In a real implementation, you might have batch verification optimizations
	t.Log("✅ Performance test completed")
} 