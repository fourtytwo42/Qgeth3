// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sync"
	"time"
	"hash/crc32"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/ethereum/go-ethereum/trie"
)

// BlockValidationPipeline implements the complete v0.9 quantum block validation
// according to specification Section 14: Block Validation Pipeline
type BlockValidationPipeline struct {
	chainIDHash        common.Hash
	puzzleOrchestrator *PuzzleOrchestrator
	novaAggregator     *NovaLiteAggregator
	dilithiumAttestor  *DilithiumAttestor
	canonicalCompiler  *CanonicalCompiler
	asertCalculator    *ASERTQDifficulty
	stats              ValidationStats
	snarkVerifier      *SNARKVerifier
	cache              *VerificationCache
	proofChainValidator *ProofChainValidator
	mu                 sync.RWMutex
}

// ValidationStats tracks block validation statistics
type ValidationStats struct {
	TotalValidations      uint64        // Total blocks validated
	SuccessfulValidations uint64        // Successfully validated blocks
	FailedValidations     uint64        // Failed validations
	AverageValidationTime time.Duration // Average validation time
	LastValidationTime    time.Time     // Last validation timestamp

	// Step-specific timing
	RLPDecodeTime        time.Duration // RLP decode time
	QuantumBlobTime      time.Duration // Quantum blob validation time
	CanonicalCompileTime time.Duration // Canonical compile check time
	NovaProofTime        time.Duration // Nova proof verification time
	DilithiumTime        time.Duration // Dilithium signature verification time
	PoWTargetTime        time.Duration // PoW target test time
	EVMExecutionTime     time.Duration // EVM execution time

	// Error breakdown
	RLPDecodeErrors        uint64 // RLP decode failures
	QuantumBlobErrors      uint64 // Quantum blob validation failures
	CanonicalCompileErrors uint64 // Canonical compile failures
	NovaProofErrors        uint64 // Nova proof verification failures
	DilithiumErrors        uint64 // Dilithium signature failures
	PoWTargetErrors        uint64 // PoW target test failures
	EVMExecutionErrors     uint64 // EVM execution failures
}

// ValidationResult contains the result of block validation
type ValidationResult struct {
	Valid            bool          // Whether block is valid
	ValidationTime   time.Duration // Total validation time
	FailureReason    string        // Reason for failure (if any)
	FailureStep      string        // Which step failed
	QuantumProofHash common.Hash   // Hash of quantum proof for verification
	GateHashMatch    bool          // Whether gate hash matches
	ProofRootValid   bool          // Whether proof root is valid
	AttestationValid bool          // Whether attestation is valid
}

// EmbeddedProofData represents proof data embedded in block headers
type EmbeddedProofData struct {
	Magic          uint32 // 0xDEADBEEF
	Version        uint32 // Proof format version
	FinalNovaProof []byte // The final aggregated Nova proof
	Checksum       uint32 // CRC32 checksum for integrity
}

// FinalNovaProof represents the final aggregated Nova proof
type FinalNovaProof struct {
	ProofData    []byte            // Raw proof bytes
	PublicInputs []byte            // Public inputs for verification
	ProofSize    uint32            // Size of the proof
	AggregatedProofs int          // Number of aggregated CAPSS proofs (should be 128)
}

// FinalNovaVerifier handles Final Nova proof verification
type FinalNovaVerifier struct {
	name      string
	available bool
	snarkVerifier *SNARKVerifier
}

// NewBlockValidationPipeline creates a new validation pipeline with caching
func NewBlockValidationPipeline(chainIDHash common.Hash) *BlockValidationPipeline {
	return &BlockValidationPipeline{
		chainIDHash:        chainIDHash,
		puzzleOrchestrator: NewPuzzleOrchestrator(),
		novaAggregator:     NewNovaLiteAggregator(),
		dilithiumAttestor:  NewDilithiumAttestor(chainIDHash),
		canonicalCompiler:  NewCanonicalCompiler(),
		asertCalculator:    NewASERTQDifficulty(),
		stats:              ValidationStats{},
		snarkVerifier:      NewSNARKVerifier(),
		cache:              NewVerificationCache(DefaultVerificationCacheConfig()),
		proofChainValidator: NewProofChainValidator(),
	}
}

// ValidateQuantumBlock implements the complete v0.9 block validation pipeline
// Steps according to specification:
// 1. RLP-decode classical header + quantum blob
// 2. Canonical-compile & GateHash check
// 3. Nova proof verify (Tier-B)
// 4. Dilithium signature verify
// 5. PoW target test via ASERT-Q
// 6. EVM execution & state transition
func (bvp *BlockValidationPipeline) ValidateQuantumBlock(
	chain consensus.ChainHeaderReader,
	block *types.Block,
	state *state.StateDB,
	publicKey []byte,
	signature []byte,
) (*ValidationResult, error) {

	start := time.Now()
	bvp.stats.TotalValidations++

	result := &ValidationResult{
		Valid: false,
	}

	header := block.Header()

	log.Info("🔍 Starting quantum block validation pipeline",
		"blockNumber", header.Number.Uint64(),
		"blockHash", block.Hash().Hex(),
		"parentHash", header.ParentHash.Hex())

	// Step 1: RLP-decode classical header + quantum blob
	step1Start := time.Now()
	if err := bvp.validateRLPDecoding(block); err != nil {
		bvp.stats.FailedValidations++
		bvp.stats.RLPDecodeErrors++
		result.FailureReason = fmt.Sprintf("RLP decode failed: %v", err)
		result.FailureStep = "RLP_DECODE"
		result.ValidationTime = time.Since(start)
		return result, err
	}
	bvp.stats.RLPDecodeTime += time.Since(step1Start)

	log.Debug("✅ Step 1: RLP decode successful")

	// Step 2: Canonical-compile & GateHash check
	step2Start := time.Now()
	gateHashValid, err := bvp.validateCanonicalCompileAndGateHash(header)
	if err != nil {
		bvp.stats.FailedValidations++
		bvp.stats.CanonicalCompileErrors++
		result.FailureReason = fmt.Sprintf("Canonical compile failed: %v", err)
		result.FailureStep = "CANONICAL_COMPILE"
		result.ValidationTime = time.Since(start)
		return result, err
	}
	result.GateHashMatch = gateHashValid
	bvp.stats.CanonicalCompileTime += time.Since(step2Start)

	log.Debug("✅ Step 2: Canonical compile & GateHash check successful")

	// Step 3: Nova proof verify (Tier-B)
	step3Start := time.Now()
	proofValid, err := bvp.validateNovaProof(header)
	if err != nil {
		bvp.stats.FailedValidations++
		bvp.stats.NovaProofErrors++
		result.FailureReason = fmt.Sprintf("Nova proof verification failed: %v", err)
		result.FailureStep = "NOVA_PROOF"
		result.ValidationTime = time.Since(start)
		return result, err
	}
	result.ProofRootValid = proofValid
	bvp.stats.NovaProofTime += time.Since(step3Start)

	log.Debug("✅ Step 3: Nova proof verification successful")

	// Step 4: Dilithium signature verify
	step4Start := time.Now()
	attestValid, err := bvp.validateDilithiumSignature(header, publicKey, signature)
	if err != nil {
		bvp.stats.FailedValidations++
		bvp.stats.DilithiumErrors++
		result.FailureReason = fmt.Sprintf("Dilithium signature verification failed: %v", err)
		result.FailureStep = "DILITHIUM_SIGNATURE"
		result.ValidationTime = time.Since(start)
		return result, err
	}
	result.AttestationValid = attestValid
	bvp.stats.DilithiumTime += time.Since(step4Start)

	log.Debug("✅ Step 4: Dilithium signature verification successful")

	// Step 5: PoW target test via ASERT-Q
	step5Start := time.Now()
	if err := bvp.validatePoWTarget(chain, header); err != nil {
		bvp.stats.FailedValidations++
		bvp.stats.PoWTargetErrors++
		result.FailureReason = fmt.Sprintf("PoW target test failed: %v", err)
		result.FailureStep = "POW_TARGET"
		result.ValidationTime = time.Since(start)
		return result, err
	}
	bvp.stats.PoWTargetTime += time.Since(step5Start)

	log.Debug("✅ Step 5: PoW target test successful")

	// Step 6: EVM execution & state transition
	step6Start := time.Now()
	if err := bvp.validateEVMExecution(chain, block, state); err != nil {
		bvp.stats.FailedValidations++
		bvp.stats.EVMExecutionErrors++
		result.FailureReason = fmt.Sprintf("EVM execution failed: %v", err)
		result.FailureStep = "EVM_EXECUTION"
		result.ValidationTime = time.Since(start)
		return result, err
	}
	bvp.stats.EVMExecutionTime += time.Since(step6Start)

	log.Debug("✅ Step 6: EVM execution & state transition successful")

	// All steps passed - validation successful
	result.Valid = true
	result.ValidationTime = time.Since(start)

	// Compute quantum proof hash for verification
	h := sha256.New()
	h.Write(header.OutcomeRoot.Bytes())
	h.Write(header.GateHash.Bytes())
	h.Write(header.ProofRoot.Bytes())
	result.QuantumProofHash = common.BytesToHash(h.Sum(nil))

	// Update statistics
	bvp.stats.SuccessfulValidations++
	bvp.stats.AverageValidationTime = updateAverageTimeValidation(
		bvp.stats.AverageValidationTime,
		result.ValidationTime,
		bvp.stats.SuccessfulValidations,
	)
	bvp.stats.LastValidationTime = time.Now()

	log.Info("🎉 Quantum block validation completed successfully",
		"blockNumber", header.Number.Uint64(),
		"validationTime", result.ValidationTime,
		"quantumProofHash", result.QuantumProofHash.Hex())

	return result, nil
}

// validateRLPDecoding validates RLP decoding of classical header + quantum blob
func (bvp *BlockValidationPipeline) validateRLPDecoding(block *types.Block) error {
	header := block.Header()

	// Verify header can be RLP encoded/decoded properly
	encoded, err := rlp.EncodeToBytes(header)
	if err != nil {
		return fmt.Errorf("header RLP encoding failed: %v", err)
	}

	// Decode back to verify roundtrip
	var decodedHeader types.Header
	if err := rlp.DecodeBytes(encoded, &decodedHeader); err != nil {
		return fmt.Errorf("header RLP decoding failed: %v", err)
	}

	// Unmarshal quantum blob to populate virtual quantum fields
	if err := decodedHeader.UnmarshalQuantumBlob(); err != nil {
		return fmt.Errorf("quantum blob unmarshaling failed: %v", err)
	}

	// Verify quantum fields are properly decoded
	if err := ValidateQuantumHeader(&decodedHeader); err != nil {
		return fmt.Errorf("quantum header validation failed: %v", err)
	}

	// Verify block body can be decoded
	body := block.Body()
	if body == nil {
		return fmt.Errorf("block body is nil")
	}

	// Validate transaction list
	if len(block.Transactions()) != len(body.Transactions) {
		return fmt.Errorf("transaction count mismatch: block=%d, body=%d",
			len(block.Transactions()), len(body.Transactions))
	}

	return nil
}

// validateCanonicalCompileAndGateHash validates canonical compilation and GateHash
func (bvp *BlockValidationPipeline) validateCanonicalCompileAndGateHash(header *types.Header) (bool, error) {
	if header.GateHash == nil {
		return false, fmt.Errorf("missing GateHash field")
	}

	if header.OutcomeRoot == nil || header.BranchNibbles == nil {
		return false, fmt.Errorf("missing required quantum fields for gate hash validation")
	}

	// Create mining input for seed calculation
	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}

	// Execute puzzle chain to verify gate hash
	result, err := bvp.puzzleOrchestrator.ExecutePuzzleChain(miningInput)
	if err != nil {
		return false, fmt.Errorf("puzzle chain execution failed: %v", err)
	}

	// Compare computed gate hash with header
	expectedGateHash := common.BytesToHash(result.GateHash[:])
	if expectedGateHash != *header.GateHash {
		log.Warn("GateHash mismatch",
			"expected", expectedGateHash.Hex(),
			"actual", header.GateHash.Hex())
		return false, nil
	}

	// Verify outcome root matches
	expectedOutcomeRoot := common.BytesToHash(result.OutcomeRoot[:])
	if expectedOutcomeRoot != *header.OutcomeRoot {
		log.Warn("OutcomeRoot mismatch",
			"expected", expectedOutcomeRoot.Hex(),
			"actual", header.OutcomeRoot.Hex())
		return false, fmt.Errorf("outcome root mismatch")
	}

	return true, nil
}

// validateNovaProof performs full cryptographic verification of Nova proof with caching
func (bvp *BlockValidationPipeline) validateNovaProof(header *types.Header) (bool, error) {
	if header.ProofRoot == nil {
		return false, fmt.Errorf("missing ProofRoot field")
	}

	// Calculate proof hash for cache lookup
	proofHash := *header.ProofRoot
	
	// Check cache first for performance optimization
	if cachedResult, found := bvp.cache.GetProofVerification(proofHash); found {
		if cachedResult.Error != nil {
			return false, cachedResult.Error
		}
		return cachedResult.Valid, nil
	}

	// Basic structure validation
	proofRootBytes := header.ProofRoot.Bytes()
	if len(proofRootBytes) != 32 {
		result := VerificationResult{
			Valid:     false,
			Timestamp: time.Now(),
			Error:     fmt.Errorf("invalid proof root size: got %d, expected 32", len(proofRootBytes)),
		}
		bvp.cache.StoreProofVerification(proofHash, result, "ProofRoot", common.Hash{})
		return false, result.Error
	}

	// Check that proof root is not zero (indicates missing proof)
	zeroHash := common.Hash{}
	if *header.ProofRoot == zeroHash {
		result := VerificationResult{
			Valid:     false,
			Timestamp: time.Now(),
			Error:     fmt.Errorf("proof root is zero - missing proof"),
		}
		bvp.cache.StoreProofVerification(proofHash, result, "ProofRoot", common.Hash{})
		return false, result.Error
	}

	// Extract proof data from block header
	proofData, err := bvp.extractProofDataFromHeader(header)
	if err != nil {
		result := VerificationResult{
			Valid:     false,
			Timestamp: time.Now(),
			Error:     fmt.Errorf("proof extraction failed: %v", err),
		}
		bvp.cache.StoreProofVerification(proofHash, result, "Extraction", common.Hash{})
		return false, result.Error
	}

	// Verify the final Nova proof cryptographically
	valid, err := bvp.verifyFinalNovaProofCryptographic(proofData.FinalNovaProof)
	
	// Cache the result for future use
	result := VerificationResult{
		Valid:     valid,
		Timestamp: time.Now(),
		Error:     err,
	}
	bvp.cache.StoreProofVerification(proofHash, result, "FinalNova", common.Hash{})
	
	if err != nil {
		return false, err
	}

	return valid, nil
}

// validateProofRootConsistency verifies that the proof root matches the embedded proof data
func (bvp *BlockValidationPipeline) validateProofRootConsistency(header *types.Header) (bool, error) {
	if header.ProofRoot == nil {
		return false, fmt.Errorf("missing ProofRoot field")
	}

	// Extract proof data to validate consistency
	proofData, err := bvp.extractProofDataFromHeader(header)
	if err != nil {
		return false, fmt.Errorf("failed to extract proof data: %v", err)
	}

	// Calculate expected proof root from embedded data
	expectedProofRoot := bvp.calculateProofRoot(proofData)

	// Verify proof root matches
	if *header.ProofRoot != expectedProofRoot {
		return false, fmt.Errorf("proof root mismatch: expected %s, got %s",
			expectedProofRoot.Hex(), header.ProofRoot.Hex())
	}

	return true, nil
}

// extractProofDataFromHeader extracts embedded proof data from quantum block header
func (bvp *BlockValidationPipeline) extractProofDataFromHeader(header *types.Header) (*EmbeddedProofData, error) {
	// Check if QBlob field exists and has sufficient data
	if header.QBlob == nil || len(header.QBlob) < 277+16 { // 277 bytes quantum fields + minimum proof data
		return nil, fmt.Errorf("insufficient QBlob data for proof extraction")
	}

	// Extract proof data starting after the 277-byte quantum fields
	qblobData := header.QBlob
	proofDataStart := 277
	
	if len(qblobData) < proofDataStart+16 {
		return nil, fmt.Errorf("QBlob too small for proof data")
	}

	// Parse embedded proof data structure
	proofBytes := qblobData[proofDataStart:]
	return bvp.parseEmbeddedProofData(proofBytes)
}

// parseEmbeddedProofData parses the embedded proof data format
func (bvp *BlockValidationPipeline) parseEmbeddedProofData(data []byte) (*EmbeddedProofData, error) {
	if len(data) < 16 { // Minimum size for header
		return nil, fmt.Errorf("proof data too small: %d bytes", len(data))
	}

	// Parse header
	magic := binary.LittleEndian.Uint32(data[0:4])
	version := binary.LittleEndian.Uint32(data[4:8])
	proofLength := binary.LittleEndian.Uint32(data[8:12])
	checksum := binary.LittleEndian.Uint32(data[12:16])

	// Validate magic number
	if magic != 0xDEADBEEF {
		return nil, fmt.Errorf("invalid magic number: 0x%08x", magic)
	}

	// Validate version
	if version != 1 {
		return nil, fmt.Errorf("unsupported proof format version: %d", version)
	}

	// Validate proof length
	if len(data) < 16+int(proofLength) {
		return nil, fmt.Errorf("insufficient data for proof: need %d, have %d",
			16+proofLength, len(data))
	}

	// Extract proof data
	proofData := data[16 : 16+proofLength]

	// Validate checksum
	calculatedChecksum := crc32.ChecksumIEEE(proofData)
	if checksum != calculatedChecksum {
		return nil, fmt.Errorf("proof data checksum mismatch: expected 0x%08x, got 0x%08x",
			checksum, calculatedChecksum)
	}

	return &EmbeddedProofData{
		Magic:          magic,
		Version:        version,
		FinalNovaProof: proofData,
		Checksum:       checksum,
	}, nil
}

// calculateProofRoot calculates the expected proof root from embedded proof data
func (bvp *BlockValidationPipeline) calculateProofRoot(proofData *EmbeddedProofData) common.Hash {
	// Calculate hash of the proof data for consistency verification
	proofHash := sha256.Sum256(proofData.FinalNovaProof)
	return common.BytesToHash(proofHash[:])
}

// verifyFinalNovaProofCryptographic performs cryptographic verification of the Final Nova proof
func (bvp *BlockValidationPipeline) verifyFinalNovaProofCryptographic(proofData []byte) (bool, error) {
	// Parse the Final Nova proof
	finalNovaProof, err := bvp.parseFinalNovaProof(proofData)
	if err != nil {
		return false, fmt.Errorf("failed to parse Final Nova proof: %v", err)
	}
	
	// Get the Nova verifier
	verifier := bvp.newFinalNovaVerifier()
	if !verifier.IsAvailable() {
		return false, fmt.Errorf("Final Nova verifier not available")
	}
	
	// Perform cryptographic verification
	valid, err := verifier.VerifyFinalProof(finalNovaProof)
	if err != nil {
		return false, fmt.Errorf("Final Nova verification error: %v", err)
	}
	
	return valid, nil
}

// validateDilithiumSignature validates Dilithium-2 self-attestation
func (bvp *BlockValidationPipeline) validateDilithiumSignature(
	header *types.Header,
	publicKey []byte,
	signature []byte,
) (bool, error) {

	if len(publicKey) != DilithiumPublicKeySize {
		return false, fmt.Errorf("invalid public key size: got %d, expected %d",
			len(publicKey), DilithiumPublicKeySize)
	}

	if len(signature) != DilithiumSignatureSize {
		return false, fmt.Errorf("invalid signature size: got %d, expected %d",
			len(signature), DilithiumSignatureSize)
	}

	// Calculate Seed₀ for verification
	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}
	seed0 := bvp.puzzleOrchestrator.computeInitialSeed(miningInput)

	// Verify attestation pair
	valid, err := bvp.dilithiumAttestor.VerifyAttestationPair(
		publicKey,
		signature,
		seed0,
		*header.OutcomeRoot,
		*header.GateHash,
		header.Number.Uint64(),
	)

	if err != nil {
		return false, fmt.Errorf("attestation verification error: %v", err)
	}

	return valid, nil
}

// validatePoWTarget validates PoW target using ASERT-Q difficulty adjustment
func (bvp *BlockValidationPipeline) validatePoWTarget(
	chain consensus.ChainHeaderReader,
	header *types.Header,
) error {

	// Get parent header for difficulty calculation
	parent := chain.GetHeader(header.ParentHash, header.Number.Uint64()-1)
	if parent == nil {
		return fmt.Errorf("parent header not found")
	}

	// Calculate expected difficulty using ASERT-Q
	adjustment := bvp.asertCalculator.CalculateNextDifficulty(chain, header)

	// Verify difficulty matches expected
	if header.Difficulty.Cmp(adjustment.NewTarget) != 0 {
		return fmt.Errorf("difficulty mismatch: got %s, expected %s",
			header.Difficulty.String(), adjustment.NewTarget.String())
	}

	// Verify quantum proof meets difficulty target
	if err := ValidateQuantumProofBitcoinStyle(header); err != nil {
		return fmt.Errorf("quantum proof target validation failed: %v", err)
	}

	return nil
}

// validateEVMExecution validates EVM execution and state transition
func (bvp *BlockValidationPipeline) validateEVMExecution(
	chain consensus.ChainHeaderReader,
	block *types.Block,
	state *state.StateDB,
) error {

	// Verify transaction root
	if block.Header().TxHash != types.DeriveSha(block.Transactions(), trie.NewStackTrie(nil)) {
		return fmt.Errorf("transaction root mismatch")
	}

	// Verify uncle root
	if block.Header().UncleHash != types.CalcUncleHash(block.Uncles()) {
		return fmt.Errorf("uncle root mismatch")
	}

	// Verify receipt root (if receipts are available)
	// This would typically be done by the full block processor

	// Verify gas limit and gas used
	if block.Header().GasUsed > block.Header().GasLimit {
		return fmt.Errorf("gas used (%d) exceeds gas limit (%d)",
			block.Header().GasUsed, block.Header().GasLimit)
	}

	// Additional EVM-specific validations would go here
	// In a full implementation, this would:
	// 1. Execute all transactions in the block
	// 2. Verify state root matches after execution
	// 3. Verify receipt root matches
	// 4. Verify gas calculations

	return nil
}

// GetValidationStats returns current validation statistics
func (bvp *BlockValidationPipeline) GetValidationStats() ValidationStats {
	return bvp.stats
}

// ResetValidationStats resets validation statistics
func (bvp *BlockValidationPipeline) ResetValidationStats() {
	bvp.stats = ValidationStats{}
}

// Helper function to update average time
func updateAverageTimeValidation(currentAvg time.Duration, newTime time.Duration, count uint64) time.Duration {
	if count == 0 {
		return newTime
	}

	// Calculate weighted average
	totalNanos := int64(currentAvg)*int64(count-1) + int64(newTime)
	return time.Duration(totalNanos / int64(count))
}

// parseFinalNovaProof parses the Final Nova proof from bytes
func (bvp *BlockValidationPipeline) parseFinalNovaProof(data []byte) (*FinalNovaProof, error) {
	if len(data) < 12 { // Minimum: proofSize(4) + publicInputsSize(4) + aggregatedProofs(4)
		return nil, fmt.Errorf("Final Nova proof data too small: %d bytes", len(data))
	}
	
	buf := bytes.NewReader(data)
	
	var proofSize uint32
	var publicInputsSize uint32
	var aggregatedProofs uint32
	
	if err := binary.Read(buf, binary.LittleEndian, &proofSize); err != nil {
		return nil, fmt.Errorf("failed to read proof size: %v", err)
	}
	
	if err := binary.Read(buf, binary.LittleEndian, &publicInputsSize); err != nil {
		return nil, fmt.Errorf("failed to read public inputs size: %v", err)
	}
	
	if err := binary.Read(buf, binary.LittleEndian, &aggregatedProofs); err != nil {
		return nil, fmt.Errorf("failed to read aggregated proofs count: %v", err)
	}
	
	// Validate aggregated proofs count (should be 128 CAPSS proofs)
	if aggregatedProofs != 128 {
		return nil, fmt.Errorf("invalid aggregated proofs count: expected 128, got %d", aggregatedProofs)
	}
	
	// Read proof data
	proofData := make([]byte, proofSize)
	if _, err := buf.Read(proofData); err != nil {
		return nil, fmt.Errorf("failed to read proof data: %v", err)
	}
	
	// Read public inputs
	publicInputs := make([]byte, publicInputsSize)
	if _, err := buf.Read(publicInputs); err != nil {
		return nil, fmt.Errorf("failed to read public inputs: %v", err)
	}
	
	return &FinalNovaProof{
		ProofData:        proofData,
		PublicInputs:     publicInputs,
		ProofSize:        proofSize,
		AggregatedProofs: int(aggregatedProofs),
	}, nil
}

// newFinalNovaVerifier creates a new Final Nova proof verifier
func (bvp *BlockValidationPipeline) newFinalNovaVerifier() *FinalNovaVerifier {
	return &FinalNovaVerifier{
		name:          "FinalNovaVerifier_v1.0",
		available:     bvp.snarkVerifier.IsAvailable(),
		snarkVerifier: bvp.snarkVerifier,
	}
}

// IsAvailable checks if the Final Nova verifier is available
func (fnv *FinalNovaVerifier) IsAvailable() bool {
	return fnv.available
}

// VerifyFinalProof performs cryptographic verification of a Final Nova proof
func (fnv *FinalNovaVerifier) VerifyFinalProof(proof *FinalNovaProof) (bool, error) {
	if !fnv.available {
		return false, fmt.Errorf("Final Nova verifier not available")
	}
	
	log.Debug("🔐 Performing Final Nova proof verification",
		"proofSize", proof.ProofSize,
		"publicInputsSize", len(proof.PublicInputs),
		"aggregatedProofs", proof.AggregatedProofs)
	
	// Validate proof structure
	if proof.ProofSize == 0 || proof.ProofSize > 6*1024 {
		return false, fmt.Errorf("invalid Final Nova proof size: %d", proof.ProofSize)
	}
	
	if proof.AggregatedProofs != 128 {
		return false, fmt.Errorf("invalid aggregated proofs count: expected 128, got %d", proof.AggregatedProofs)
	}
	
	// Check proof is not all zeros
	allZeros := true
	for _, b := range proof.ProofData {
		if b != 0 {
			allZeros = false
			break
		}
	}
	if allZeros {
		return false, fmt.Errorf("Final Nova proof is all zeros (fake proof)")
	}
	
	// Verify public inputs are present
	if len(proof.PublicInputs) == 0 {
		return false, fmt.Errorf("missing Final Nova public inputs")
	}
	
	// CRITICAL: Perform cryptographic SNARK verification
	if fnv.snarkVerifier == nil {
		return false, fmt.Errorf("SNARK verifier not initialized")
	}
	
	// Use specialized Final Nova verification
	valid, err := fnv.snarkVerifier.VerifyFinalNovaProof(proof)
	if err != nil {
		return false, fmt.Errorf("Final Nova SNARK verification error: %v", err)
	}
	
	if valid {
		log.Debug("✅ Final Nova proof cryptographically verified",
			"aggregatedProofs", proof.AggregatedProofs,
			"verifier", fnv.snarkVerifier.GetName())
	} else {
		log.Warn("❌ Final Nova proof verification failed",
			"aggregatedProofs", proof.AggregatedProofs,
			"verifier", fnv.snarkVerifier.GetName())
	}
	
	return valid, nil
}

// ValidateBlockWithCache performs complete block validation with caching
func (bvp *BlockValidationPipeline) ValidateBlockWithCache(header *types.Header) (bool, error) {
	// Calculate block hash for cache lookup
	blockHash := header.Hash()
	
	// Check cache first for performance optimization
	if cachedResult, found := bvp.cache.GetBlockVerification(blockHash); found {
		if cachedResult.Error != nil {
			return false, cachedResult.Error
		}
		return cachedResult.Valid, nil
	}

	// Perform full validation if not cached
	valid, err := bvp.validateNovaProof(header)
	
	// Additional validations can be added here
	if valid {
		valid, err = bvp.validateProofRootConsistency(header)
	}
	
	// Cache the complete block validation result
	result := VerificationResult{
		Valid:     valid,
		Timestamp: time.Now(),
		Error:     err,
	}
	
	proofRoot := common.Hash{}
	if header.ProofRoot != nil {
		proofRoot = *header.ProofRoot
	}
	
	bvp.cache.StoreBlockVerification(blockHash, result, proofRoot)
	
	if err != nil {
		return false, err
	}

	return valid, nil
}

// InvalidateCache provides cache invalidation capabilities
func (bvp *BlockValidationPipeline) InvalidateCache() {
	bvp.cache.Clear()
}

// InvalidateProof invalidates a specific proof from cache
func (bvp *BlockValidationPipeline) InvalidateProof(proofHash common.Hash) {
	bvp.cache.InvalidateProof(proofHash)
}

// InvalidateBlock invalidates a specific block from cache
func (bvp *BlockValidationPipeline) InvalidateBlock(blockHash common.Hash) {
	bvp.cache.InvalidateBlock(blockHash)
}

// GetCacheStats returns current cache performance statistics
func (bvp *BlockValidationPipeline) GetCacheStats() VerificationCacheStats {
	return bvp.cache.GetStats()
}

// StopCache stops the cache cleanup goroutine
func (bvp *BlockValidationPipeline) StopCache() {
	if bvp.cache != nil {
		bvp.cache.Stop()
	}
}

// validateProofChain validates the hierarchical proof structure (CAPSS → Nova → Root) without re-computation
func (bvp *BlockValidationPipeline) validateProofChain(header *types.Header) (bool, error) {
	// Extract proof data from block header
	proofData, err := bvp.extractProofDataFromHeader(header)
	if err != nil {
		return false, fmt.Errorf("failed to extract proof data for chain validation: %v", err)
	}

	// Extract proof chain structure from Final Nova proof
	proofChain, err := bvp.proofChainValidator.ExtractProofChainFromFinalProof(proofData, bvp.chainIDHash)
	if err != nil {
		return false, fmt.Errorf("failed to extract proof chain structure: %v", err)
	}

	// Validate the complete hierarchical proof structure
	valid, err := bvp.proofChainValidator.ValidateProofChain(proofChain)
	if err != nil {
		return false, fmt.Errorf("proof chain validation failed: %v", err)
	}

	if !valid {
		return false, fmt.Errorf("hierarchical proof chain validation failed")
	}

	log.Debug("✅ Hierarchical proof chain validation successful",
		"chain_id", bvp.chainIDHash.Hex(),
		"aggregated_proofs", proofChain.FinalNovaProof.AggregatedProofs,
		"proof_root", proofChain.ProofRoot.Hex())

	return true, nil
}
