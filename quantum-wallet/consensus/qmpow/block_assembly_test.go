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

func TestNewBlockAssembler(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	assembler := NewBlockAssembler(chainIDHash)

	if assembler == nil {
		t.Fatal("Failed to create block assembler")
	}

	if assembler.chainIDHash != chainIDHash {
		t.Errorf("Chain ID hash mismatch: got %v, expected %v", assembler.chainIDHash, chainIDHash)
	}

	if assembler.puzzleOrchestrator == nil {
		t.Error("Puzzle orchestrator not initialized")
	}

	if assembler.novaAggregator == nil {
		t.Error("Nova aggregator not initialized")
	}

	if assembler.dilithiumAttestor == nil {
		t.Error("Dilithium attestor not initialized")
	}

	stats := assembler.GetAssemblyStats()
	if stats.TotalBlocks != 0 {
		t.Errorf("Expected 0 total blocks, got %d", stats.TotalBlocks)
	}
}

func TestAssembleQuantumBlock(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	assembler := NewBlockAssembler(chainIDHash)

	// Create test header with quantum fields
	header := createTestQuantumHeader()

	// Create mock chain reader
	chain := &MockChainReader{}

	// Create empty state
	state := &state.StateDB{}

	// Test block assembly
	block, publicKey, signature, err := assembler.AssembleQuantumBlock(
		chain, header, state, nil, nil, nil, nil)

	if err != nil {
		t.Fatalf("Block assembly failed: %v", err)
	}

	if block == nil {
		t.Fatal("Block is nil")
	}

	if len(publicKey) != DilithiumPublicKeySize {
		t.Errorf("Invalid public key size: got %d, expected %d", len(publicKey), DilithiumPublicKeySize)
	}

	if len(signature) != DilithiumSignatureSize {
		t.Errorf("Invalid signature size: got %d, expected %d", len(signature), DilithiumSignatureSize)
	}

	// Verify header was updated with quantum fields
	if header.OutcomeRoot == nil {
		t.Error("OutcomeRoot not set")
	}

	if header.GateHash == nil {
		t.Error("GateHash not set")
	}

	if header.ProofRoot == nil {
		t.Error("ProofRoot not set")
	}

	// Validate BranchNibbles (128 bytes for 128 puzzles)
	if len(header.BranchNibbles) != BranchNibblesSize {
		t.Errorf("Invalid BranchNibbles size: got %d, expected %d", len(header.BranchNibbles), BranchNibblesSize)
	}

	// Check statistics
	stats := assembler.GetAssemblyStats()
	if stats.TotalBlocks != 1 {
		t.Errorf("Expected 1 total block, got %d", stats.TotalBlocks)
	}

	if stats.SuccessfulAssemblies != 1 {
		t.Errorf("Expected 1 successful assembly, got %d", stats.SuccessfulAssemblies)
	}
}

func TestValidateQuantumBlockAssembly(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	assembler := NewBlockAssembler(chainIDHash)

	// Create test header with quantum fields
	header := createTestQuantumHeader()

	// Create mock chain reader
	chain := &MockChainReader{}

	// Create empty state
	state := &state.StateDB{}

	// Assemble a block first
	_, publicKey, signature, err := assembler.AssembleQuantumBlock(
		chain, header, state, nil, nil, nil, nil)

	if err != nil {
		t.Fatalf("Block assembly failed: %v", err)
	}

	// Test validation
	err = assembler.ValidateQuantumBlockAssembly(header, publicKey, signature)
	if err != nil {
		t.Errorf("Block validation failed: %v", err)
	}
}

func TestValidateQuantumBlockAssemblyInvalidSignature(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	assembler := NewBlockAssembler(chainIDHash)

	// Create test header with quantum fields
	header := createTestQuantumHeader()

	// Create mock chain reader
	chain := &MockChainReader{}

	// Create empty state
	state := &state.StateDB{}

	// Assemble a block first
	_, publicKey, _, err := assembler.AssembleQuantumBlock(
		chain, header, state, nil, nil, nil, nil)

	if err != nil {
		t.Fatalf("Block assembly failed: %v", err)
	}

	// Create invalid signature
	invalidSignature := make([]byte, DilithiumSignatureSize)
	for i := range invalidSignature {
		invalidSignature[i] = 0xFF // All ones instead of proper signature
	}

	// Test validation with invalid signature
	err = assembler.ValidateQuantumBlockAssembly(header, publicKey, invalidSignature)
	if err == nil {
		t.Error("Expected validation to fail with invalid signature")
	}
}

func TestAssemblyStatsTracking(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	assembler := NewBlockAssembler(chainIDHash)

	// Initial stats should be zero
	stats := assembler.GetAssemblyStats()
	if stats.TotalBlocks != 0 {
		t.Errorf("Expected 0 total blocks, got %d", stats.TotalBlocks)
	}

	// Create test header and assemble multiple blocks
	for i := 0; i < 3; i++ {
		header := createTestQuantumHeader()
		header.Number = big.NewInt(int64(i + 1))

		chain := &MockChainReader{}
		state := &state.StateDB{}

		_, _, _, err := assembler.AssembleQuantumBlock(
			chain, header, state, nil, nil, nil, nil)

		if err != nil {
			t.Fatalf("Block assembly %d failed: %v", i+1, err)
		}
	}

	// Check final stats
	stats = assembler.GetAssemblyStats()
	if stats.TotalBlocks != 3 {
		t.Errorf("Expected 3 total blocks, got %d", stats.TotalBlocks)
	}

	if stats.SuccessfulAssemblies != 3 {
		t.Errorf("Expected 3 successful assemblies, got %d", stats.SuccessfulAssemblies)
	}

	if stats.FailedAssemblies != 0 {
		t.Errorf("Expected 0 failed assemblies, got %d", stats.FailedAssemblies)
	}

	if stats.AverageAssemblyTime == 0 {
		t.Error("Average assembly time should be non-zero")
	}
}

func TestGetTotalCAPSSSize(t *testing.T) {
	// Create test CAPSS proofs
	proofs := []*CAPSSProof{
		{Size: 2200},
		{Size: 2200},
		{Size: 2200},
	}

	total := getTotalCAPSSSize(proofs)
	expected := 6600

	if total != expected {
		t.Errorf("Expected total size %d, got %d", expected, total)
	}
}

func TestUpdateAverageTime(t *testing.T) {
	// Test first time
	avg := updateAverageTime(0, 100*time.Millisecond, 1)
	if avg != 100*time.Millisecond {
		t.Errorf("Expected 100ms, got %v", avg)
	}

	// Test second time
	avg = updateAverageTime(100*time.Millisecond, 200*time.Millisecond, 2)
	expected := 150 * time.Millisecond
	if avg != expected {
		t.Errorf("Expected %v, got %v", expected, avg)
	}

	// Test third time
	avg = updateAverageTime(150*time.Millisecond, 300*time.Millisecond, 3)
	expected = 200 * time.Millisecond
	if avg != expected {
		t.Errorf("Expected %v, got %v", expected, avg)
	}
}

func TestQMPoWFinalizeAndAssembleWithProofs(t *testing.T) {
	// Create QMPoW instance
	config := Config{
		PowMode:  ModeTest,
		TestMode: true,
	}
	qmpow := New(config)

	// Create test header
	header := createTestQuantumHeader()

	// Create mock chain reader
	chain := &MockChainReader{}

	// Create empty state
	state := &state.StateDB{}

	// Test enhanced finalize and assemble
	block, err := qmpow.FinalizeAndAssembleWithProofs(
		chain, header, state, nil, nil, nil, nil)

	if err != nil {
		t.Fatalf("Enhanced finalize and assemble failed: %v", err)
	}

	if block == nil {
		t.Fatal("Block is nil")
	}

	// Check that attestation data was stored
	if qmpow.lastPublicKey == nil {
		t.Error("Public key not stored")
	}

	if qmpow.lastSignature == nil {
		t.Error("Signature not stored")
	}

	if len(qmpow.lastPublicKey) != DilithiumPublicKeySize {
		t.Errorf("Invalid stored public key size: got %d, expected %d",
			len(qmpow.lastPublicKey), DilithiumPublicKeySize)
	}

	if len(qmpow.lastSignature) != DilithiumSignatureSize {
		t.Errorf("Invalid stored signature size: got %d, expected %d",
			len(qmpow.lastSignature), DilithiumSignatureSize)
	}
}

func TestQMPoWFinalizeAndAssembleNonQuantum(t *testing.T) {
	// Create QMPoW instance
	config := Config{
		PowMode:  ModeTest,
		TestMode: true,
	}
	qmpow := New(config)

	// Create non-quantum header (block 0 but force non-quantum)
	header := &types.Header{
		Number:     big.NewInt(0),
		Difficulty: big.NewInt(1000),
		Time:       uint64(time.Now().Unix()),
		GasLimit:   8000000,
	}

	// Temporarily disable quantum for this test
	originalIsQuantumActive := types.IsQuantumActive
	types.IsQuantumActive = func(num *big.Int) bool { return false }
	defer func() { types.IsQuantumActive = originalIsQuantumActive }()

	// Create mock chain reader
	chain := &MockChainReader{}

	// Create empty state
	state := &state.StateDB{}

	// Test standard finalize and assemble
	block, err := qmpow.FinalizeAndAssembleWithProofs(
		chain, header, state, nil, nil, nil, nil)

	if err != nil {
		t.Fatalf("Standard finalize and assemble failed: %v", err)
	}

	if block == nil {
		t.Fatal("Block is nil")
	}

	// Check that no attestation data was stored
	if qmpow.lastPublicKey != nil {
		t.Error("Public key should not be stored for non-quantum block")
	}

	if qmpow.lastSignature != nil {
		t.Error("Signature should not be stored for non-quantum block")
	}
}

// Helper functions for testing

func createTestQuantumHeader() *types.Header {
	epoch := uint32(0)
	qbits := uint16(16)
	tcount := uint32(20)
	lnet := uint16(128)
	qnonce := uint64(0)
	attestMode := uint8(0)

	return &types.Header{
		Number:        big.NewInt(1),
		Difficulty:    big.NewInt(1000),
		Time:          uint64(time.Now().Unix()),
		GasLimit:      8000000,
		ParentHash:    common.HexToHash("0x1234567890abcdef"),
		TxHash:        common.HexToHash("0xabcdef1234567890"),
		Epoch:         &epoch,
		QBits:         &qbits,
		TCount:        &tcount,
		LNet:          &lnet,
		QNonce64:      &qnonce,
		AttestMode:    &attestMode,
		ExtraNonce32:  make([]byte, 32),
		BranchNibbles: make([]byte, 128),
	}
}

// MockChainReader for testing
type MockChainReader struct{}

func (m *MockChainReader) Config() *types.ChainConfig {
	return &types.ChainConfig{}
}

func (m *MockChainReader) CurrentHeader() *types.Header {
	return &types.Header{}
}

func (m *MockChainReader) GetHeader(hash common.Hash, number uint64) *types.Header {
	return &types.Header{}
}

func (m *MockChainReader) GetHeaderByNumber(number uint64) *types.Header {
	return &types.Header{}
}

func (m *MockChainReader) GetHeaderByHash(hash common.Hash) *types.Header {
	return &types.Header{}
}
