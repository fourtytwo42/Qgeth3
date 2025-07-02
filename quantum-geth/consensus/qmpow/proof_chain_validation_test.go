package qmpow

import (
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
)

func TestProofChainValidator_Creation(t *testing.T) {
	validator := NewProofChainValidator()
	
	if validator == nil {
		t.Fatal("Expected validator to be created, got nil")
	}
	
	if validator.capssVerifier == nil {
		t.Error("Expected CAPSS verifier to be initialized")
	}
	
	if validator.novaVerifier == nil {
		t.Error("Expected Nova-Lite verifier to be initialized")
	}
	
	if validator.finalVerifier == nil {
		t.Error("Expected Final Nova verifier to be initialized")
	}
	
	stats := validator.GetProofChainStats()
	if stats.TotalValidations != 0 {
		t.Error("Expected initial stats to be zero")
	}
}

func TestProofChainValidator_ExtractProofChain(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Create mock embedded proof data
	embeddedData := &EmbeddedProofData{
		Magic:   0xDEADBEEF,
		Version: 1,
		FinalNovaProof: createMockFinalNovaProofData(),
		Checksum: 0x12345678,
	}
	
	proofChain, err := validator.ExtractProofChainFromFinalProof(embeddedData, chainID)
	if err != nil {
		t.Fatalf("Failed to extract proof chain: %v", err)
	}
	
	if proofChain == nil {
		t.Fatal("Expected proof chain to be created, got nil")
	}
	
	if proofChain.ChainID != chainID {
		t.Errorf("Expected chain ID %s, got %s", chainID.Hex(), proofChain.ChainID.Hex())
	}
	
	if proofChain.FinalNovaProof == nil {
		t.Error("Expected Final Nova proof to be present")
	}
	
	if proofChain.FinalNovaProof.AggregatedProofs != 128 {
		t.Errorf("Expected 128 aggregated proofs, got %d", proofChain.FinalNovaProof.AggregatedProofs)
	}
}

func TestProofChainValidator_ValidateValidChain(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Create a valid proof chain structure
	proofChain := &ProofChainStructure{
		FinalNovaProof: &FinalNovaProof{
			ProofData:        []byte("mock_final_nova_proof_data"),
			PublicInputs:     []byte("mock_public_inputs"),
			ProofSize:        26, // Length of mock proof data
			AggregatedProofs: 128,
		},
		ChainID:      chainID,
		CreationTime: time.Now(),
	}
	
	// Calculate proof root
	proofChain.ProofRoot = validator.calculateProofChainRoot(proofChain)
	
	valid, err := validator.ValidateProofChain(proofChain)
	if err != nil {
		t.Fatalf("Validation failed with error: %v", err)
	}
	
	if !valid {
		t.Error("Expected valid proof chain to pass validation")
	}
	
	stats := validator.GetProofChainStats()
	if stats.TotalValidations != 1 {
		t.Errorf("Expected 1 total validation, got %d", stats.TotalValidations)
	}
	
	if stats.SuccessfulChains != 1 {
		t.Errorf("Expected 1 successful validation, got %d", stats.SuccessfulChains)
	}
}

func TestProofChainValidator_RejectInvalidAggregation(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Create proof chain with wrong aggregation count
	proofChain := &ProofChainStructure{
		FinalNovaProof: &FinalNovaProof{
			ProofData:        []byte("mock_proof_data"),
			PublicInputs:     []byte("mock_inputs"),
			ProofSize:        15,
			AggregatedProofs: 64, // Wrong count - should be 128
		},
		ChainID:      chainID,
		CreationTime: time.Now(),
	}
	
	valid, err := validator.ValidateProofChain(proofChain)
	if err == nil {
		t.Error("Expected validation to fail with wrong aggregation count")
	}
	
	if valid {
		t.Error("Expected invalid proof chain to fail validation")
	}
	
	stats := validator.GetProofChainStats()
	if stats.FailedChains != 1 {
		t.Errorf("Expected 1 failed validation, got %d", stats.FailedChains)
	}
}

func TestProofChainValidator_RejectMissingFinalProof(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Create proof chain without Final Nova proof
	proofChain := &ProofChainStructure{
		FinalNovaProof: nil, // Missing proof
		ChainID:        chainID,
		CreationTime:   time.Now(),
	}
	
	valid, err := validator.ValidateProofChain(proofChain)
	if err == nil {
		t.Error("Expected validation to fail with missing Final Nova proof")
	}
	
	if valid {
		t.Error("Expected invalid proof chain to fail validation")
	}
}

func TestProofChainValidator_RejectEmptyProofData(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Create proof chain with empty proof data
	proofChain := &ProofChainStructure{
		FinalNovaProof: &FinalNovaProof{
			ProofData:        []byte{}, // Empty proof data
			PublicInputs:     []byte("mock_inputs"),
			ProofSize:        0,
			AggregatedProofs: 128,
		},
		ChainID:      chainID,
		CreationTime: time.Now(),
	}
	
	valid, err := validator.ValidateProofChain(proofChain)
	if err == nil {
		t.Error("Expected validation to fail with empty proof data")
	}
	
	if valid {
		t.Error("Expected invalid proof chain to fail validation")
	}
}

func TestProofChainValidator_RejectOversizedProof(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Create proof chain with oversized proof (>6KB)
	oversizedProof := make([]byte, 7*1024) // 7KB proof
	proofChain := &ProofChainStructure{
		FinalNovaProof: &FinalNovaProof{
			ProofData:        oversizedProof,
			PublicInputs:     []byte("mock_inputs"),
			ProofSize:        uint32(len(oversizedProof)),
			AggregatedProofs: 128,
		},
		ChainID:      chainID,
		CreationTime: time.Now(),
	}
	
	valid, err := validator.ValidateProofChain(proofChain)
	if err == nil {
		t.Error("Expected validation to fail with oversized proof")
	}
	
	if valid {
		t.Error("Expected invalid proof chain to fail validation")
	}
}

func TestProofChainValidator_Stats(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	// Test multiple validations
	for i := 0; i < 3; i++ {
		proofChain := &ProofChainStructure{
			FinalNovaProof: &FinalNovaProof{
				ProofData:        []byte("mock_proof_data"),
				PublicInputs:     []byte("mock_inputs"),
				ProofSize:        15,
				AggregatedProofs: 128,
			},
			ChainID:      chainID,
			CreationTime: time.Now(),
		}
		
		valid, err := validator.ValidateProofChain(proofChain)
		if err != nil || !valid {
			t.Fatalf("Validation %d failed", i)
		}
	}
	
	stats := validator.GetProofChainStats()
	if stats.TotalValidations != 3 {
		t.Errorf("Expected 3 total validations, got %d", stats.TotalValidations)
	}
	
	if stats.SuccessfulChains != 3 {
		t.Errorf("Expected 3 successful validations, got %d", stats.SuccessfulChains)
	}
	
	// Average validation time can be zero for very fast operations
	// Just verify it's not negative
	if stats.AverageValidateTime < 0 {
		t.Error("Expected non-negative average validation time")
	}
	
	// Reset stats
	validator.ResetProofChainStats()
	stats = validator.GetProofChainStats()
	if stats.TotalValidations != 0 {
		t.Error("Expected stats to be reset to zero")
	}
}

func TestProofChainValidator_ProofRootConsistency(t *testing.T) {
	validator := NewProofChainValidator()
	chainID := common.HexToHash("0x1234567890abcdef")
	
	proofChain := &ProofChainStructure{
		FinalNovaProof: &FinalNovaProof{
			ProofData:        []byte("consistent_proof_data"),
			PublicInputs:     []byte("consistent_inputs"),
			ProofSize:        21,
			AggregatedProofs: 128,
		},
		ChainID:      chainID,
		CreationTime: time.Now(),
	}
	
	// Calculate proof root
	expectedRoot := validator.calculateProofChainRoot(proofChain)
	proofChain.ProofRoot = expectedRoot
	
	valid, err := validator.ValidateProofChain(proofChain)
	if err != nil {
		t.Fatalf("Validation failed: %v", err)
	}
	
	if !valid {
		t.Error("Expected consistent proof root to pass validation")
	}
	
	// Test with wrong proof root - Create a completely different proof chain for wrong root
	wrongProofChain := &ProofChainStructure{
		FinalNovaProof: &FinalNovaProof{
			ProofData:        []byte("different_proof_data"), // Different data will produce different root
			PublicInputs:     []byte("different_inputs"),
			ProofSize:        18,
			AggregatedProofs: 128,
		},
		ChainID:      chainID,
		CreationTime: time.Now(),
		ProofRoot:    expectedRoot, // Use the old root with new data - this should fail
	}
	
	valid, err = validator.ValidateProofChain(wrongProofChain)
	if err == nil {
		t.Error("Expected validation to fail with wrong proof root")
	}
	
	if valid {
		t.Error("Expected inconsistent proof root to fail validation")
	}
}

// Helper function to create mock Final Nova proof data
func createMockFinalNovaProofData() []byte {
	// Create mock binary data that represents a Final Nova proof
	// Format: proofSize(4) + publicInputsSize(4) + aggregatedProofs(4) + proofData + publicInputs
	proofData := []byte("mock_final_nova_proof_data")
	publicInputs := []byte("mock_public_inputs")
	
	result := make([]byte, 12+len(proofData)+len(publicInputs))
	
	// Write header
	copy(result[0:4], []byte{26, 0, 0, 0})   // proofSize = 26
	copy(result[4:8], []byte{18, 0, 0, 0})   // publicInputsSize = 18  
	copy(result[8:12], []byte{128, 0, 0, 0}) // aggregatedProofs = 128
	
	// Write data
	copy(result[12:12+len(proofData)], proofData)
	copy(result[12+len(proofData):], publicInputs)
	
	return result
} 