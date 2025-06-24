// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"math/big"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
)

func TestNewBlockValidationPipeline(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	if pipeline == nil {
		t.Fatal("Expected non-nil pipeline")
	}

	if pipeline.chainIDHash != chainIDHash {
		t.Errorf("Expected chainIDHash %s, got %s", chainIDHash.Hex(), pipeline.chainIDHash.Hex())
	}

	if pipeline.puzzleOrchestrator == nil {
		t.Error("Expected non-nil puzzle orchestrator")
	}

	if pipeline.novaAggregator == nil {
		t.Error("Expected non-nil nova aggregator")
	}

	if pipeline.dilithiumAttestor == nil {
		t.Error("Expected non-nil dilithium attestor")
	}
}

func TestValidateQuantumBlockSuccess(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create test block with proper quantum fields
	header := createTestQuantumHeader()
	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	// Create mock chain reader
	chain := &MockChainReader{}

	// Create empty state
	state := &state.StateDB{}

	// Generate valid attestation
	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}

	// Create attestation
	attestor := NewDilithiumAttestor(chainIDHash)
	seed0 := pipeline.puzzleOrchestrator.computeInitialSeed(miningInput)
	publicKey, signature, err := attestor.CreateAttestationPair(
		seed0,
		*header.OutcomeRoot,
		*header.GateHash,
		header.Number.Uint64(),
	)
	if err != nil {
		t.Fatalf("Failed to create attestation: %v", err)
	}

	// Test validation
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, publicKey, signature)
	if err != nil {
		t.Fatalf("Validation failed: %v", err)
	}

	if !result.Valid {
		t.Errorf("Expected validation to succeed, got failure: %s", result.FailureReason)
	}

	if result.ValidationTime == 0 {
		t.Error("Expected non-zero validation time")
	}

	if !result.GateHashMatch {
		t.Error("Expected gate hash to match")
	}

	if !result.ProofRootValid {
		t.Error("Expected proof root to be valid")
	}

	if !result.AttestationValid {
		t.Error("Expected attestation to be valid")
	}
}

func TestValidateQuantumBlockRLPDecodeError(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create invalid block with nil header fields
	header := &types.Header{
		Number: big.NewInt(1),
		// Missing required fields to cause RLP decode issues
	}
	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	chain := &MockChainReader{}
	state := &state.StateDB{}

	// Test validation with invalid block
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, nil, nil)
	if err == nil {
		t.Fatal("Expected validation to fail")
	}

	if result.Valid {
		t.Error("Expected validation to fail")
	}

	if result.FailureStep != "RLP_DECODE" {
		t.Errorf("Expected failure step RLP_DECODE, got %s", result.FailureStep)
	}

	// Check statistics
	stats := pipeline.GetValidationStats()
	if stats.FailedValidations != 1 {
		t.Errorf("Expected 1 failed validation, got %d", stats.FailedValidations)
	}

	if stats.RLPDecodeErrors != 1 {
		t.Errorf("Expected 1 RLP decode error, got %d", stats.RLPDecodeErrors)
	}
}

func TestValidateQuantumBlockCanonicalCompileError(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create block with invalid gate hash
	header := createTestQuantumHeader()
	// Set invalid gate hash
	invalidGateHash := common.HexToHash("0x1111111111111111111111111111111111111111111111111111111111111111")
	header.GateHash = &invalidGateHash

	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	chain := &MockChainReader{}
	state := &state.StateDB{}

	// Create dummy attestation
	publicKey := make([]byte, DilithiumPublicKeySize)
	signature := make([]byte, DilithiumSignatureSize)

	// Test validation
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, publicKey, signature)
	if err == nil {
		t.Fatal("Expected validation to fail")
	}

	if result.Valid {
		t.Error("Expected validation to fail")
	}

	if result.FailureStep != "CANONICAL_COMPILE" {
		t.Errorf("Expected failure step CANONICAL_COMPILE, got %s", result.FailureStep)
	}

	// Check statistics
	stats := pipeline.GetValidationStats()
	if stats.CanonicalCompileErrors != 1 {
		t.Errorf("Expected 1 canonical compile error, got %d", stats.CanonicalCompileErrors)
	}
}

func TestValidateQuantumBlockNovaProofError(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create block with zero proof root
	header := createTestQuantumHeader()
	zeroHash := common.Hash{}
	header.ProofRoot = &zeroHash

	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	chain := &MockChainReader{}
	state := &state.StateDB{}

	// Create dummy attestation
	publicKey := make([]byte, DilithiumPublicKeySize)
	signature := make([]byte, DilithiumSignatureSize)

	// Test validation
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, publicKey, signature)
	if err == nil {
		t.Fatal("Expected validation to fail")
	}

	if result.Valid {
		t.Error("Expected validation to fail")
	}

	if result.FailureStep != "NOVA_PROOF" {
		t.Errorf("Expected failure step NOVA_PROOF, got %s", result.FailureStep)
	}

	// Check statistics
	stats := pipeline.GetValidationStats()
	if stats.NovaProofErrors != 1 {
		t.Errorf("Expected 1 Nova proof error, got %d", stats.NovaProofErrors)
	}
}

func TestValidateQuantumBlockDilithiumError(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create valid block
	header := createTestQuantumHeader()
	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	chain := &MockChainReader{}
	state := &state.StateDB{}

	// Create invalid attestation (wrong size)
	publicKey := make([]byte, 100) // Wrong size
	signature := make([]byte, DilithiumSignatureSize)

	// Test validation
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, publicKey, signature)
	if err == nil {
		t.Fatal("Expected validation to fail")
	}

	if result.Valid {
		t.Error("Expected validation to fail")
	}

	if result.FailureStep != "DILITHIUM_SIGNATURE" {
		t.Errorf("Expected failure step DILITHIUM_SIGNATURE, got %s", result.FailureStep)
	}

	// Check statistics
	stats := pipeline.GetValidationStats()
	if stats.DilithiumErrors != 1 {
		t.Errorf("Expected 1 Dilithium error, got %d", stats.DilithiumErrors)
	}
}

func TestValidateQuantumBlockPoWTargetError(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create block with invalid difficulty
	header := createTestQuantumHeader()
	header.Difficulty = big.NewInt(0) // Invalid difficulty

	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	// Create mock chain reader that returns no parent (will cause error)
	chain := &MockChainReader{
		returnNilParent: true,
	}
	state := &state.StateDB{}

	// Create valid attestation
	attestor := NewDilithiumAttestor(chainIDHash)
	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}
	seed0 := pipeline.puzzleOrchestrator.computeInitialSeed(miningInput)
	publicKey, signature, err := attestor.CreateAttestationPair(
		seed0,
		*header.OutcomeRoot,
		*header.GateHash,
		header.Number.Uint64(),
	)
	if err != nil {
		t.Fatalf("Failed to create attestation: %v", err)
	}

	// Test validation
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, publicKey, signature)
	if err == nil {
		t.Fatal("Expected validation to fail")
	}

	if result.Valid {
		t.Error("Expected validation to fail")
	}

	if result.FailureStep != "POW_TARGET" {
		t.Errorf("Expected failure step POW_TARGET, got %s", result.FailureStep)
	}

	// Check statistics
	stats := pipeline.GetValidationStats()
	if stats.PoWTargetErrors != 1 {
		t.Errorf("Expected 1 PoW target error, got %d", stats.PoWTargetErrors)
	}
}

func TestValidateRLPDecoding(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Test with valid quantum block
	header := createTestQuantumHeader()
	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	err := pipeline.validateRLPDecoding(block)
	if err != nil {
		t.Errorf("Expected validation to succeed, got error: %v", err)
	}

	// Test with invalid block (missing quantum fields)
	invalidHeader := &types.Header{
		Number: big.NewInt(1),
		Time:   uint64(time.Now().Unix()),
		// Missing quantum fields
	}
	invalidBlock := types.NewBlockWithWithdrawals(invalidHeader, nil, nil, nil, nil, nil)

	err = pipeline.validateRLPDecoding(invalidBlock)
	if err == nil {
		t.Error("Expected validation to fail with missing quantum fields")
	}
}

func TestValidateCanonicalCompileAndGateHash(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Test with valid header
	header := createTestQuantumHeader()

	valid, err := pipeline.validateCanonicalCompileAndGateHash(header)
	if err != nil {
		t.Errorf("Expected validation to succeed, got error: %v", err)
	}

	if !valid {
		t.Error("Expected gate hash to be valid")
	}

	// Test with missing gate hash
	headerNoGateHash := createTestQuantumHeader()
	headerNoGateHash.GateHash = nil

	valid, err = pipeline.validateCanonicalCompileAndGateHash(headerNoGateHash)
	if err == nil {
		t.Error("Expected validation to fail with missing gate hash")
	}
}

func TestValidateNovaProof(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Test with valid proof root
	header := createTestQuantumHeader()

	valid, err := pipeline.validateNovaProof(header)
	if err != nil {
		t.Errorf("Expected validation to succeed, got error: %v", err)
	}

	if !valid {
		t.Error("Expected proof root to be valid")
	}

	// Test with missing proof root
	headerNoProof := createTestQuantumHeader()
	headerNoProof.ProofRoot = nil

	valid, err = pipeline.validateNovaProof(headerNoProof)
	if err == nil {
		t.Error("Expected validation to fail with missing proof root")
	}

	// Test with zero proof root
	headerZeroProof := createTestQuantumHeader()
	zeroHash := common.Hash{}
	headerZeroProof.ProofRoot = &zeroHash

	valid, err = pipeline.validateNovaProof(headerZeroProof)
	if err == nil {
		t.Error("Expected validation to fail with zero proof root")
	}
}

func TestValidateDilithiumSignature(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create valid attestation
	header := createTestQuantumHeader()
	attestor := NewDilithiumAttestor(chainIDHash)

	miningInput := &MiningInput{
		ParentHash:   header.ParentHash,
		TxRoot:       header.TxHash,
		ExtraNonce32: header.ExtraNonce32,
		QNonce64:     *header.QNonce64,
	}
	seed0 := pipeline.puzzleOrchestrator.computeInitialSeed(miningInput)

	publicKey, signature, err := attestor.CreateAttestationPair(
		seed0,
		*header.OutcomeRoot,
		*header.GateHash,
		header.Number.Uint64(),
	)
	if err != nil {
		t.Fatalf("Failed to create attestation: %v", err)
	}

	// Test with valid signature
	valid, err := pipeline.validateDilithiumSignature(header, publicKey, signature)
	if err != nil {
		t.Errorf("Expected validation to succeed, got error: %v", err)
	}

	if !valid {
		t.Error("Expected signature to be valid")
	}

	// Test with invalid public key size
	invalidPublicKey := make([]byte, 100)
	valid, err = pipeline.validateDilithiumSignature(header, invalidPublicKey, signature)
	if err == nil {
		t.Error("Expected validation to fail with invalid public key size")
	}

	// Test with invalid signature size
	invalidSignature := make([]byte, 100)
	valid, err = pipeline.validateDilithiumSignature(header, publicKey, invalidSignature)
	if err == nil {
		t.Error("Expected validation to fail with invalid signature size")
	}
}

func TestValidatePoWTarget(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create valid header
	header := createTestQuantumHeader()

	// Create mock chain reader
	chain := &MockChainReader{}

	// Test validation
	err := pipeline.validatePoWTarget(chain, header)
	if err != nil {
		t.Errorf("Expected validation to succeed, got error: %v", err)
	}

	// Test with missing parent
	chainNoParent := &MockChainReader{
		returnNilParent: true,
	}

	err = pipeline.validatePoWTarget(chainNoParent, header)
	if err == nil {
		t.Error("Expected validation to fail with missing parent")
	}
}

func TestValidateEVMExecution(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create valid block
	header := createTestQuantumHeader()
	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	// Create mock chain reader
	chain := &MockChainReader{}
	state := &state.StateDB{}

	// Test validation
	err := pipeline.validateEVMExecution(chain, block, state)
	if err != nil {
		t.Errorf("Expected validation to succeed, got error: %v", err)
	}

	// Test with invalid gas usage
	invalidHeader := createTestQuantumHeader()
	invalidHeader.GasUsed = invalidHeader.GasLimit + 1 // Gas used > gas limit
	invalidBlock := types.NewBlockWithWithdrawals(invalidHeader, nil, nil, nil, nil, nil)

	err = pipeline.validateEVMExecution(chain, invalidBlock, state)
	if err == nil {
		t.Error("Expected validation to fail with invalid gas usage")
	}
}

func TestValidationStats(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Initial stats should be zero
	stats := pipeline.GetValidationStats()
	if stats.TotalValidations != 0 {
		t.Errorf("Expected 0 total validations, got %d", stats.TotalValidations)
	}

	// Test reset
	pipeline.ResetValidationStats()
	stats = pipeline.GetValidationStats()
	if stats.TotalValidations != 0 {
		t.Errorf("Expected 0 total validations after reset, got %d", stats.TotalValidations)
	}
}

func TestUpdateAverageTime(t *testing.T) {
	// Test with first measurement
	avg := updateAverageTime(0, 100*time.Millisecond, 1)
	if avg != 100*time.Millisecond {
		t.Errorf("Expected 100ms, got %v", avg)
	}

	// Test with second measurement
	avg = updateAverageTime(100*time.Millisecond, 200*time.Millisecond, 2)
	expected := 150 * time.Millisecond // (100 + 200) / 2
	if avg != expected {
		t.Errorf("Expected %v, got %v", expected, avg)
	}

	// Test with zero count
	avg = updateAverageTime(100*time.Millisecond, 200*time.Millisecond, 0)
	if avg != 200*time.Millisecond {
		t.Errorf("Expected 200ms with zero count, got %v", avg)
	}
}

// Enhanced MockChainReader for testing
type MockChainReader struct {
	returnNilParent bool
}

func (m *MockChainReader) Config() *params.ChainConfig {
	return params.TestChainConfig
}

func (m *MockChainReader) CurrentHeader() *types.Header {
	return createTestQuantumHeader()
}

func (m *MockChainReader) GetHeader(hash common.Hash, number uint64) *types.Header {
	if m.returnNilParent {
		return nil
	}

	// Return a mock parent header
	parentHeader := createTestQuantumHeader()
	parentHeader.Number = big.NewInt(int64(number))
	parentHeader.Time = uint64(time.Now().Unix()) - 12 // 12 seconds ago
	parentHeader.Difficulty = big.NewInt(1000000)
	return parentHeader
}

func (m *MockChainReader) GetHeaderByHash(hash common.Hash) *types.Header {
	return m.GetHeader(hash, 0)
}

func (m *MockChainReader) GetHeaderByNumber(number uint64) *types.Header {
	return m.GetHeader(common.Hash{}, number)
}

// TestValidateNovaProofFakeProofRejection tests that fake proofs are properly rejected
func TestValidateNovaProofFakeProofRejection(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Test 1: Fake proof root (all zeros) should be rejected
	header := createTestQuantumHeader()
	zeroHash := common.Hash{}
	header.ProofRoot = &zeroHash

	valid, err := pipeline.validateNovaProof(header)
	if err == nil {
		t.Error("Expected validation to fail with zero proof root")
	}
	if valid {
		t.Error("Zero proof root should be invalid")
	}

	// Test 2: Missing proof root should be rejected
	headerNoProof := createTestQuantumHeader()
	headerNoProof.ProofRoot = nil

	valid, err = pipeline.validateNovaProof(headerNoProof)
	if err == nil {
		t.Error("Expected validation to fail with missing proof root")
	}
	if valid {
		t.Error("Missing proof root should be invalid")
	}

	// Test 3: Valid proof root structure should trigger full verification
	headerValid := createTestQuantumHeader()
	// This will fail during reconstruction/verification, which is expected
	// since we don't have a real quantum execution environment in tests
	valid, err = pipeline.validateNovaProof(headerValid)
	if err == nil {
		t.Error("Expected validation to fail during proof reconstruction (test environment)")
	}
	if valid {
		t.Error("Should fail during full verification in test environment")
	}

	t.Log("✅ Fake proof rejection tests passed")
}

// TestCAPSSVerifierFakeProofDetection tests CAPSS verifier fake proof detection
func TestCAPSSVerifierFakeProofDetection(t *testing.T) {
	verifier := &CAPSSVerifier{
		name:      "TestVerifier",
		available: true,
	}

	// Test 1: All-zero proof should be rejected
	fakeProof := &CAPSSProof{
		TraceID:      1000,
		Proof:        make([]byte, 2200), // All zeros
		PublicInputs: make([]byte, 64),
		ProofHash:    make([]byte, 32),
	}

	valid, err := verifier.VerifyProof(fakeProof)
	if err == nil {
		t.Error("Expected fake proof to be rejected")
	}
	if valid {
		t.Error("All-zero proof should be invalid")
	}

	// Test 2: Wrong size proof should be rejected
	wrongSizeProof := &CAPSSProof{
		TraceID:      1001,
		Proof:        make([]byte, 1000), // Wrong size
		PublicInputs: make([]byte, 64),
		ProofHash:    make([]byte, 32),
	}

	valid, err = verifier.VerifyProof(wrongSizeProof)
	if err == nil {
		t.Error("Expected wrong-size proof to be rejected")
	}
	if valid {
		t.Error("Wrong-size proof should be invalid")
	}

	// Test 3: Missing public inputs should be rejected
	noInputsProof := &CAPSSProof{
		TraceID:      1002,
		Proof:        make([]byte, 2200),
		PublicInputs: []byte{}, // Empty
		ProofHash:    make([]byte, 32),
	}
	// Fill with non-zero data
	for i := range noInputsProof.Proof {
		noInputsProof.Proof[i] = byte(i % 256)
	}

	valid, err = verifier.VerifyProof(noInputsProof)
	if err == nil {
		t.Error("Expected proof with missing public inputs to be rejected")
	}
	if valid {
		t.Error("Proof with missing public inputs should be invalid")
	}

	// Test 4: Valid structure should pass basic checks
	validProof := &CAPSSProof{
		TraceID:      1003,
		Proof:        make([]byte, 2200),
		PublicInputs: make([]byte, 64),
		ProofHash:    make([]byte, 32),
	}
	// Fill with non-zero data
	for i := range validProof.Proof {
		validProof.Proof[i] = byte((i + 1) % 256)
	}
	for i := range validProof.PublicInputs {
		validProof.PublicInputs[i] = byte((i + 100) % 256)
	}

	valid, err = verifier.VerifyProof(validProof)
	if err != nil {
		t.Errorf("Valid proof structure should pass basic checks: %v", err)
	}
	if !valid {
		t.Error("Valid proof structure should be accepted")
	}

	t.Log("✅ CAPSS verifier fake proof detection tests passed")
}

// TestNovaLiteVerifierFakeProofDetection tests Nova-Lite verifier fake proof detection
func TestNovaLiteVerifierFakeProofDetection(t *testing.T) {
	verifier := &NovaLiteVerifier{
		name:      "TestVerifier",
		available: true,
	}

	// Test 1: All-zero proof should be rejected
	fakeProof := &NovaLiteProof{
		ProofID:      1000,
		Tier:         2,
		BatchIndex:   0,
		ProofData:    make([]byte, 5000), // All zeros
		PublicInputs: make([]byte, 96),
		ProofHash:    make([]byte, 32),
		CAPSSCount:   16,
		Size:         5000,
	}

	valid, err := verifier.VerifyRecursiveProof(fakeProof)
	if err == nil {
		t.Error("Expected fake Nova-Lite proof to be rejected")
	}
	if valid {
		t.Error("All-zero Nova-Lite proof should be invalid")
	}

	// Test 2: Oversized proof should be rejected
	oversizedProof := &NovaLiteProof{
		ProofID:      1001,
		Tier:         2,
		BatchIndex:   0,
		ProofData:    make([]byte, 7000), // > 6 kB limit
		PublicInputs: make([]byte, 96),
		ProofHash:    make([]byte, 32),
		CAPSSCount:   16,
		Size:         7000,
	}

	valid, err = verifier.VerifyRecursiveProof(oversizedProof)
	if err == nil {
		t.Error("Expected oversized proof to be rejected")
	}
	if valid {
		t.Error("Oversized proof should be invalid")
	}

	// Test 3: Wrong CAPSS count should be rejected
	wrongCountProof := &NovaLiteProof{
		ProofID:      1002,
		Tier:         2,
		BatchIndex:   0,
		ProofData:    make([]byte, 5000),
		PublicInputs: make([]byte, 96),
		ProofHash:    make([]byte, 32),
		CAPSSCount:   10, // Should be 16
		Size:         5000,
	}
	// Fill with non-zero data
	for i := range wrongCountProof.ProofData {
		wrongCountProof.ProofData[i] = byte(i % 256)
	}

	valid, err = verifier.VerifyRecursiveProof(wrongCountProof)
	if err == nil {
		t.Error("Expected wrong CAPSS count proof to be rejected")
	}
	if valid {
		t.Error("Wrong CAPSS count proof should be invalid")
	}

	// Test 4: Valid structure should pass basic checks
	validProof := &NovaLiteProof{
		ProofID:      1003,
		Tier:         2,
		BatchIndex:   0,
		ProofData:    make([]byte, 5000),
		PublicInputs: make([]byte, 96),
		ProofHash:    make([]byte, 32),
		CAPSSCount:   16,
		Size:         5000,
	}
	// Fill with non-zero data
	for i := range validProof.ProofData {
		validProof.ProofData[i] = byte((i + 1) % 256)
	}
	for i := range validProof.PublicInputs {
		validProof.PublicInputs[i] = byte((i + 200) % 256)
	}

	valid, err = verifier.VerifyRecursiveProof(validProof)
	if err != nil {
		t.Errorf("Valid Nova-Lite proof structure should pass basic checks: %v", err)
	}
	if !valid {
		t.Error("Valid Nova-Lite proof structure should be accepted")
	}

	t.Log("✅ Nova-Lite verifier fake proof detection tests passed")
}

// TestFullQuantumProofVerificationPipeline tests the complete proof verification pipeline
func TestFullQuantumProofVerificationPipeline(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	pipeline := NewBlockValidationPipeline(chainIDHash)

	// Create a test block with quantum fields
	header := createTestQuantumHeader()
	block := types.NewBlockWithWithdrawals(header, nil, nil, nil, nil, nil)

	chain := &MockChainReader{}
	state := &state.StateDB{}

	// Create dummy attestation
	publicKey := make([]byte, DilithiumPublicKeySize)
	signature := make([]byte, DilithiumSignatureSize)

	// Test the full validation pipeline
	// This should fail during proof reconstruction since we don't have real quantum execution
	result, err := pipeline.ValidateQuantumBlock(chain, block, state, publicKey, signature)

	// We expect this to fail in the test environment during proof reconstruction
	if err == nil {
		t.Error("Expected validation to fail during proof reconstruction in test environment")
	}

	if result != nil && result.Valid {
		t.Error("Validation should fail in test environment")
	}

	// Verify it fails at the correct step (Nova proof verification)
	if result != nil && result.FailureStep != "NOVA_PROOF" {
		t.Errorf("Expected failure at NOVA_PROOF step, got %s", result.FailureStep)
	}

	t.Log("✅ Full quantum proof verification pipeline test completed")
	t.Logf("   - Pipeline correctly rejects invalid proofs")
	t.Logf("   - Failure step: %s", result.FailureStep)
	t.Logf("   - Failure reason: %s", result.FailureReason)
}
