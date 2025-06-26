// PZ-01: Deterministic Seed→Outcome Mapping Test
// Tests that given a fixed seed and branch templates, the 128 puzzles
// execute in sequence and produce deterministic outcomes

package puzzle

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// Test PZ-01: Deterministic Seed→Outcome Mapping
func TestPZ01_DeterministicSeedOutcomeMapping(t *testing.T) {
	// Golden reference values for deterministic testing
	goldenSeed := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	
	// Expected outcomes for the first 8 puzzles (out of 128)
	// These would be computed from actual quantum circuit execution
	expectedOutcomes := []uint16{
		0x1234, 0x5678, 0x9abc, 0xdef0,
		0x2468, 0xace1, 0x3579, 0xbdf0,
	}
	
	t.Logf("🔍 PZ-01: Testing deterministic seed→outcome mapping")
	t.Logf("   Seed: %s", goldenSeed)
	t.Logf("   Testing %d puzzle outcomes", len(expectedOutcomes))
	
	// Simulate quantum puzzle execution for deterministic testing
	actualOutcomes := simulateQuantumPuzzleExecution(goldenSeed, len(expectedOutcomes))
	
	// Verify outcomes match golden values
	if len(actualOutcomes) != len(expectedOutcomes) {
		t.Fatalf("❌ Outcome count mismatch: expected %d, got %d", 
			len(expectedOutcomes), len(actualOutcomes))
	}
	
	allMatch := true
	for i := 0; i < len(expectedOutcomes); i++ {
		if actualOutcomes[i] != expectedOutcomes[i] {
			t.Errorf("❌ Puzzle %d outcome mismatch: expected 0x%04x, got 0x%04x", 
				i, expectedOutcomes[i], actualOutcomes[i])
			allMatch = false
		} else {
			t.Logf("✅ Puzzle %d: outcome 0x%04x matches golden value", i, actualOutcomes[i])
		}
	}
	
	if !allMatch {
		t.Fatal("❌ PZ-01 FAILED: Outcomes do not match golden values")
	}
	
	t.Log("✅ PZ-01 PASSED: All outcomes match golden values")
}

// simulateQuantumPuzzleExecution simulates the quantum puzzle execution
// In the real implementation, this would call the actual quantum simulator
func simulateQuantumPuzzleExecution(seedHex string, puzzleCount int) []uint16 {
	outcomes := make([]uint16, puzzleCount)
	
	// Convert seed to bytes
	seedBytes, err := hex.DecodeString(seedHex)
	if err != nil {
		panic("Invalid seed hex")
	}
	
	// For deterministic testing, we simulate the quantum execution
	// using SHA256 to generate reproducible "quantum" outcomes
	hasher := sha256.New()
	hasher.Write(seedBytes)
	
	for i := 0; i < puzzleCount; i++ {
		// Each puzzle uses the previous outcome as input (branch-serial)
		hasher.Write([]byte("puzzle_"))
		hasher.Write([]byte{byte(i)})
		
		if i > 0 {
			// Branch-serial: inject previous outcome
			prevBytes := []byte{byte(outcomes[i-1] >> 8), byte(outcomes[i-1] & 0xFF)}
			hasher.Write(prevBytes)
		}
		
		hash := hasher.Sum(nil)
		// Extract 16-bit outcome from hash
		outcomes[i] = uint16(hash[0])<<8 | uint16(hash[1])
	}
	
	return outcomes
}

// Test that repeated execution with same seed produces identical outcomes
func TestPZ01_RepeatabilityCheck(t *testing.T) {
	seed := "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
	puzzleCount := 8
	
	t.Log("🔍 PZ-01 Repeatability: Testing execution consistency")
	
	// Execute twice
	outcomes1 := simulateQuantumPuzzleExecution(seed, puzzleCount)
	outcomes2 := simulateQuantumPuzzleExecution(seed, puzzleCount)
	
	// Verify identical results
	for i := 0; i < puzzleCount; i++ {
		if outcomes1[i] != outcomes2[i] {
			t.Errorf("❌ Repeatability failure at puzzle %d: first=0x%04x, second=0x%04x", 
				i, outcomes1[i], outcomes2[i])
		}
	}
	
	t.Log("✅ PZ-01 Repeatability: Execution is deterministic")
}

// Test that different seeds produce different outcomes
func TestPZ01_SeedSensitivity(t *testing.T) {
	seed1 := "1111111111111111111111111111111111111111111111111111111111111111"
	seed2 := "1111111111111111111111111111111111111111111111111111111111111112" // Last bit different
	puzzleCount := 8
	
	t.Log("🔍 PZ-01 Sensitivity: Testing seed sensitivity")
	
	outcomes1 := simulateQuantumPuzzleExecution(seed1, puzzleCount)
	outcomes2 := simulateQuantumPuzzleExecution(seed2, puzzleCount)
	
	// Verify at least one outcome is different
	identical := true
	for i := 0; i < puzzleCount; i++ {
		if outcomes1[i] != outcomes2[i] {
			identical = false
			t.Logf("✅ Puzzle %d differs: seed1=0x%04x, seed2=0x%04x", 
				i, outcomes1[i], outcomes2[i])
			break
		}
	}
	
	if identical {
		t.Error("❌ Seed sensitivity failure: different seeds produced identical outcomes")
	} else {
		t.Log("✅ PZ-01 Sensitivity: Different seeds produce different outcomes")
	}
} 