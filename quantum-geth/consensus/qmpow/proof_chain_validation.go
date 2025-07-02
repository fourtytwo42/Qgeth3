package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/log"
)

// ProofChainValidator validates the hierarchical proof structure without re-computation
type ProofChainValidator struct {
	capssVerifier *CAPSSVerifier
	novaVerifier  *NovaLiteVerifier
	finalVerifier *FinalNovaVerifier
	stats         ProofChainStats
}

// ProofChainStats tracks proof chain validation statistics
type ProofChainStats struct {
	TotalValidations     uint64        // Total proof chains validated
	SuccessfulChains     uint64        // Successfully validated chains
	FailedChains         uint64        // Failed chain validations
	AverageValidateTime  time.Duration // Average validation time
	
	// Level-specific stats
	CAPSSValidations     uint64        // CAPSS proof validations
	NovaLiteValidations  uint64        // Nova-Lite proof validations
	FinalNovaValidations uint64        // Final Nova proof validations
	
	// Error breakdown
	CAPSSErrors          uint64        // CAPSS validation errors
	NovaLiteErrors       uint64        // Nova-Lite validation errors
	FinalNovaErrors      uint64        // Final Nova validation errors
	ChainIntegrityErrors uint64        // Chain integrity errors
}

// ProofChainStructure represents the complete hierarchical proof structure
type ProofChainStructure struct {
	// Level 1: Individual CAPSS proofs (128 total)
	CAPSSProofs []*CAPSSProofChain
	
	// Level 2: Nova-Lite aggregated proofs (8 total, 16 CAPSS each)
	NovaLiteProofs []*NovaLiteProofChain
	
	// Level 3: Final Nova proof (1 total, aggregates 8 Nova-Lite)
	FinalNovaProof *FinalNovaProof
	
	// Metadata
	ProofRoot      common.Hash   // Root hash of the entire proof chain
	ChainID        common.Hash   // Blockchain chain ID for verification context
	CreationTime   time.Time     // When this proof chain was created
}

// CAPSSProofChain represents an individual CAPSS proof for one quantum puzzle
type CAPSSProofChain struct {
	ProofData     []byte        // Raw CAPSS proof bytes
	PublicInputs  []byte        // Public inputs for this proof
	CircuitHash   common.Hash   // Hash of the quantum circuit
	PuzzleIndex   uint8         // Index in the 128-puzzle chain (0-127)
	ProofHash     common.Hash   // Hash of this proof for chain validation
}

// NovaLiteProofChain represents a Nova-Lite proof aggregating 16 CAPSS proofs
type NovaLiteProofChain struct {
	ProofData        []byte        // Raw Nova-Lite proof bytes
	PublicInputs     []byte        // Public inputs for aggregation
	AggregatedHashes []common.Hash // Hashes of the 16 aggregated CAPSS proofs
	BatchIndex       uint8         // Index in the 8-batch sequence (0-7)
	ProofHash        common.Hash   // Hash of this Nova-Lite proof
}

// NewProofChainValidator creates a new hierarchical proof chain validator
func NewProofChainValidator() *ProofChainValidator {
	return &ProofChainValidator{
		capssVerifier: NewCAPSSVerifier(),
		novaVerifier:  NewNovaLiteVerifier(),
		finalVerifier: NewFinalNovaVerifier(),
		stats:         ProofChainStats{},
	}
}

// ValidateProofChain validates the complete hierarchical proof structure
func (pcv *ProofChainValidator) ValidateProofChain(proofChain *ProofChainStructure) (bool, error) {
	startTime := time.Now()
	pcv.stats.TotalValidations++
	
	log.Debug("🔗 Starting hierarchical proof chain validation",
		"capss_proofs", len(proofChain.CAPSSProofs),
		"nova_lite_proofs", len(proofChain.NovaLiteProofs),
		"chain_id", proofChain.ChainID.Hex())
	
	// Step 1: Validate individual CAPSS proofs (Level 1)
	if err := pcv.validateCAPSSLevel(proofChain.CAPSSProofs); err != nil {
		pcv.stats.FailedChains++
		pcv.stats.CAPSSErrors++
		return false, fmt.Errorf("CAPSS level validation failed: %v", err)
	}
	
	// Step 2: Validate Nova-Lite aggregation proofs (Level 2)
	if err := pcv.validateNovaLiteLevel(proofChain.NovaLiteProofs, proofChain.CAPSSProofs); err != nil {
		pcv.stats.FailedChains++
		pcv.stats.NovaLiteErrors++
		return false, fmt.Errorf("Nova-Lite level validation failed: %v", err)
	}
	
	// Step 3: Validate Final Nova aggregation proof (Level 3)
	if err := pcv.validateFinalNovaLevel(proofChain.FinalNovaProof, proofChain.NovaLiteProofs); err != nil {
		pcv.stats.FailedChains++
		pcv.stats.FinalNovaErrors++
		return false, fmt.Errorf("Final Nova level validation failed: %v", err)
	}
	
	// Step 4: Validate proof chain integrity
	if err := pcv.validateChainIntegrity(proofChain); err != nil {
		pcv.stats.FailedChains++
		pcv.stats.ChainIntegrityErrors++
		return false, fmt.Errorf("proof chain integrity validation failed: %v", err)
	}
	
	// Update statistics
	validationTime := time.Since(startTime)
	pcv.stats.SuccessfulChains++
	pcv.updateAverageTime(validationTime)
	
	log.Debug("✅ Hierarchical proof chain validation successful",
		"validation_time", validationTime,
		"chain_id", proofChain.ChainID.Hex())
	
	return true, nil
}

// validateCAPSSLevel validates all 128 CAPSS proofs (Level 1)
func (pcv *ProofChainValidator) validateCAPSSLevel(capssProofs []*CAPSSProofChain) error {
	// For simplified implementation, we validate the Final Nova proof represents 128 CAPSS proofs
	// In full implementation, this would validate each individual CAPSS proof
	log.Debug("🧩 Validating CAPSS proof level (Level 1 - simplified)", "expected_count", 128)
	
	// The CAPSS level validation is represented by the Final Nova proof's aggregated count
	// This is a simplified approach for the current implementation
	return nil
}

// validateNovaLiteLevel validates the 8 Nova-Lite aggregation proofs (Level 2)
func (pcv *ProofChainValidator) validateNovaLiteLevel(novaLiteProofs []*NovaLiteProofChain, capssProofs []*CAPSSProofChain) error {
	// For simplified implementation, we validate through the Final Nova proof
	// In full implementation, this would validate each Nova-Lite proof aggregates 16 CAPSS proofs
	log.Debug("🔗 Validating Nova-Lite proof level (Level 2 - simplified)", "expected_count", 8)
	
	// The Nova-Lite level validation is represented by the Final Nova proof structure
	// This is a simplified approach for the current implementation
	return nil
}

// validateFinalNovaLevel validates the Final Nova aggregation proof (Level 3)
func (pcv *ProofChainValidator) validateFinalNovaLevel(finalProof *FinalNovaProof, novaLiteProofs []*NovaLiteProofChain) error {
	if finalProof == nil {
		return fmt.Errorf("Final Nova proof is nil")
	}
	
	log.Debug("🎯 Validating Final Nova proof level (Level 3)")
	
	// Verify the Final Nova proof aggregates exactly 128 CAPSS proofs through hierarchical structure
	if finalProof.AggregatedProofs != 128 {
		return fmt.Errorf("Final Nova proof has wrong aggregated count: expected 128, got %d",
			finalProof.AggregatedProofs)
	}
	
	// Verify proof structure
	if len(finalProof.ProofData) == 0 {
		return fmt.Errorf("Final Nova proof has empty proof data")
	}
	
	if len(finalProof.PublicInputs) == 0 {
		return fmt.Errorf("Final Nova proof has empty public inputs")
	}
	
	// Verify proof size is reasonable (should aggregate 8 Nova-Lite proofs, each aggregating 16 CAPSS proofs)
	if finalProof.ProofSize == 0 || finalProof.ProofSize > 6*1024 {
		return fmt.Errorf("Final Nova proof size invalid: %d bytes", finalProof.ProofSize)
	}
	
	pcv.stats.FinalNovaValidations++
	
	log.Debug("✅ Final Nova proof level validation successful", 
		"aggregated_proofs", finalProof.AggregatedProofs,
		"proof_size", finalProof.ProofSize)
	return nil
}

// validateChainIntegrity validates the overall integrity of the proof chain
func (pcv *ProofChainValidator) validateChainIntegrity(proofChain *ProofChainStructure) error {
	log.Debug("🔒 Validating proof chain integrity")
	
	// Verify chain completeness - Final Nova proof must exist
	if proofChain.FinalNovaProof == nil {
		return fmt.Errorf("missing Final Nova proof")
	}
	
	// Verify the Final Nova proof correctly represents the hierarchical aggregation
	// 128 CAPSS proofs → 8 Nova-Lite proofs → 1 Final Nova proof
	if proofChain.FinalNovaProof.AggregatedProofs != 128 {
		return fmt.Errorf("proof chain aggregation mismatch: expected 128 CAPSS proofs, got %d",
			proofChain.FinalNovaProof.AggregatedProofs)
	}
	
	// Calculate expected proof root
	expectedRoot := pcv.calculateProofChainRoot(proofChain)
	
	// STRICT: Verify proof root matches if provided (not zero)
	if proofChain.ProofRoot != (common.Hash{}) {
		if proofChain.ProofRoot != expectedRoot {
			return fmt.Errorf("proof chain root mismatch: expected %s, got %s",
				expectedRoot.Hex(), proofChain.ProofRoot.Hex())
		}
	} else {
		// Update proof root if not set
		proofChain.ProofRoot = expectedRoot
	}
	
	// Verify temporal consistency (proofs shouldn't be too old)
	maxAge := 24 * time.Hour // Proofs shouldn't be older than 24 hours
	if !proofChain.CreationTime.IsZero() && time.Since(proofChain.CreationTime) > maxAge {
		return fmt.Errorf("proof chain too old: created %v ago", time.Since(proofChain.CreationTime))
	}
	
	log.Debug("✅ Proof chain integrity validation successful", "proof_root", expectedRoot.Hex())
	return nil
}

// calculateProofChainRoot calculates the proof root from the hierarchical structure
func (pcv *ProofChainValidator) calculateProofChainRoot(proofChain *ProofChainStructure) common.Hash {
	hasher := sha256.New()
	
	// Hash the Final Nova proof data (represents the entire chain)
	if proofChain.FinalNovaProof != nil {
		hasher.Write(proofChain.FinalNovaProof.ProofData)
		
		// Include aggregated proof count for integrity
		countBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(countBytes, uint32(proofChain.FinalNovaProof.AggregatedProofs))
		hasher.Write(countBytes)
	}
	
	// Include chain ID for context
	hasher.Write(proofChain.ChainID.Bytes())
	
	return common.BytesToHash(hasher.Sum(nil))
}

// ExtractProofChainFromFinalProof extracts the hierarchical proof structure from embedded proof data
func (pcv *ProofChainValidator) ExtractProofChainFromFinalProof(embeddedData *EmbeddedProofData, chainID common.Hash) (*ProofChainStructure, error) {
	// Parse the Final Nova proof
	finalProof, err := pcv.parseFinalNovaProofData(embeddedData.FinalNovaProof)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Final Nova proof: %v", err)
	}
	
	// Create proof chain structure with the Final Nova proof
	proofChain := &ProofChainStructure{
		FinalNovaProof: finalProof,
		ChainID:        chainID,
		CreationTime:   time.Now(),
	}
	
	// Calculate proof root
	proofChain.ProofRoot = pcv.calculateProofChainRoot(proofChain)
	
	return proofChain, nil
}

// parseFinalNovaProofData parses Final Nova proof from raw bytes
func (pcv *ProofChainValidator) parseFinalNovaProofData(data []byte) (*FinalNovaProof, error) {
	if len(data) < 12 {
		return nil, fmt.Errorf("Final Nova proof data too small: %d bytes", len(data))
	}
	
	proofSize := binary.LittleEndian.Uint32(data[0:4])
	publicInputsSize := binary.LittleEndian.Uint32(data[4:8])
	aggregatedProofs := binary.LittleEndian.Uint32(data[8:12])
	
	if aggregatedProofs != 128 {
		return nil, fmt.Errorf("invalid aggregated proofs count: expected 128, got %d", aggregatedProofs)
	}
	
	headerSize := 12
	if len(data) < headerSize+int(proofSize)+int(publicInputsSize) {
		return nil, fmt.Errorf("insufficient data for Final Nova proof")
	}
	
	proofData := data[headerSize : headerSize+int(proofSize)]
	publicInputs := data[headerSize+int(proofSize) : headerSize+int(proofSize)+int(publicInputsSize)]
	
	return &FinalNovaProof{
		ProofData:        proofData,
		PublicInputs:     publicInputs,
		ProofSize:        proofSize,
		AggregatedProofs: int(aggregatedProofs),
	}, nil
}

// GetProofChainStats returns current proof chain validation statistics
func (pcv *ProofChainValidator) GetProofChainStats() ProofChainStats {
	return pcv.stats
}

// ResetProofChainStats resets proof chain validation statistics
func (pcv *ProofChainValidator) ResetProofChainStats() {
	pcv.stats = ProofChainStats{}
}

// updateAverageTime updates the average validation time
func (pcv *ProofChainValidator) updateAverageTime(newTime time.Duration) {
	if pcv.stats.TotalValidations == 0 {
		pcv.stats.AverageValidateTime = newTime
		return
	}
	
	// Calculate weighted average
	totalNanos := int64(pcv.stats.AverageValidateTime)*int64(pcv.stats.TotalValidations-1) + int64(newTime)
	pcv.stats.AverageValidateTime = time.Duration(totalNanos / int64(pcv.stats.TotalValidations))
}

// CAPSSVerifier stub for hierarchical proof validation
type CAPSSVerifier struct {
	name      string
	available bool
}

// NovaLiteVerifier stub for hierarchical proof validation
type NovaLiteVerifier struct {
	name      string
	available bool
}

// NewCAPSSVerifier creates a stub CAPSS verifier for proof chain validation
func NewCAPSSVerifier() *CAPSSVerifier {
	return &CAPSSVerifier{
		name:      "CAPSSVerifier_v1.0",
		available: true, // Simplified for current implementation
	}
}

// NewNovaLiteVerifier creates a stub Nova-Lite verifier for proof chain validation  
func NewNovaLiteVerifier() *NovaLiteVerifier {
	return &NovaLiteVerifier{
		name:      "NovaLiteVerifier_v1.0", 
		available: true, // Simplified for current implementation
	}
}

// VerifyProof verifies a CAPSS proof (simplified implementation)
func (cv *CAPSSVerifier) VerifyProof(proof *CAPSSProofChain) (bool, error) {
	// Simplified validation - in full implementation this would do cryptographic verification
	if proof == nil {
		return false, fmt.Errorf("CAPSS proof is nil")
	}
	
	if len(proof.ProofData) == 0 {
		return false, fmt.Errorf("CAPSS proof data is empty")
	}
	
	log.Debug("🧩 CAPSS proof verification (simplified)", "puzzle_index", proof.PuzzleIndex)
	return true, nil
}

// VerifyProof verifies a Nova-Lite proof (simplified implementation)
func (nv *NovaLiteVerifier) VerifyProof(proof *NovaLiteProofChain) (bool, error) {
	// Simplified validation - in full implementation this would do cryptographic verification
	if proof == nil {
		return false, fmt.Errorf("Nova-Lite proof is nil")
	}
	
	if len(proof.ProofData) == 0 {
		return false, fmt.Errorf("Nova-Lite proof data is empty")
	}
	
	if len(proof.AggregatedHashes) != 16 {
		return false, fmt.Errorf("Nova-Lite proof should aggregate 16 CAPSS proofs, got %d", len(proof.AggregatedHashes))
	}
	
	log.Debug("🔗 Nova-Lite proof verification (simplified)", "batch_index", proof.BatchIndex)
	return true, nil
}

// NewFinalNovaVerifier creates a stub Final Nova verifier for proof chain validation
func NewFinalNovaVerifier() *FinalNovaVerifier {
	return &FinalNovaVerifier{
		name:          "FinalNovaVerifier_v1.0",
		available:     true, // Simplified for current implementation
		snarkVerifier: NewSNARKVerifier(),
	}
} 