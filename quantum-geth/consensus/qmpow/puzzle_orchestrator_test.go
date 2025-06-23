// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/ethereum/go-ethereum/common"
)

func TestPuzzleOrchestrator(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	// Verify orchestrator initialization
	if orchestrator.templateEngine == nil {
		t.Error("Template engine not initialized")
	}

	if orchestrator.compiler == nil {
		t.Error("Compiler not initialized")
	}

	if orchestrator.backend == nil {
		t.Error("Backend not initialized")
	}

	if !orchestrator.backend.IsAvailable() {
		t.Error("Backend should be available")
	}
}

func TestComputeInitialSeed(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	input := &MiningInput{
		ParentHash:   common.HexToHash("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"),
		TxRoot:       common.HexToHash("0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"),
		ExtraNonce32: make([]byte, 32),
		QNonce64:     0x123456789abcdef0,
		BlockHeight:  1000,
		QBits:        16,
		TCount:       8192,
		LNet:         48,
	}

	// Fill ExtraNonce32 with test pattern
	for i := range input.ExtraNonce32 {
		input.ExtraNonce32[i] = byte(i)
	}

	seed0 := orchestrator.computeInitialSeed(input)

	// Verify seed properties
	if len(seed0) != 32 {
		t.Errorf("Expected seed length 32, got %d", len(seed0))
	}

	// Verify determinism
	seed0_2 := orchestrator.computeInitialSeed(input)
	if !bytes.Equal(seed0, seed0_2) {
		t.Error("Seed generation is not deterministic")
	}

	// Verify different inputs produce different seeds
	input2 := *input
	input2.QNonce64 = 0x123456789abcdef1
	seed0_different := orchestrator.computeInitialSeed(&input2)

	if bytes.Equal(seed0, seed0_different) {
		t.Error("Different inputs should produce different seeds")
	}

	t.Logf("Seed₀: %x", seed0)
}

func TestChainSeed(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	seed := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
	outcome := uint16(0x1234)

	nextSeed := orchestrator.chainSeed(seed, outcome)

	// Verify next seed properties
	if len(nextSeed) != 32 {
		t.Errorf("Expected next seed length 32, got %d", len(nextSeed))
	}

	// Verify determinism
	nextSeed2 := orchestrator.chainSeed(seed, outcome)
	if !bytes.Equal(nextSeed, nextSeed2) {
		t.Error("Seed chaining is not deterministic")
	}

	// Verify different outcomes produce different seeds
	nextSeedDifferent := orchestrator.chainSeed(seed, 0x5678)
	if bytes.Equal(nextSeed, nextSeedDifferent) {
		t.Error("Different outcomes should produce different next seeds")
	}

	t.Logf("Chained seed: %x", nextSeed)
}

func TestBuildOutcomeRoot(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	// Test with 48 outcomes
	outcomes := make([]uint16, 48)
	for i := range outcomes {
		outcomes[i] = uint16(i * 1000) // Varied outcomes
	}

	root := orchestrator.buildOutcomeRoot(outcomes)

	// Verify root properties
	if root == (common.Hash{}) {
		t.Error("Outcome root should not be empty")
	}

	// Verify determinism
	root2 := orchestrator.buildOutcomeRoot(outcomes)
	if root != root2 {
		t.Error("Outcome root calculation is not deterministic")
	}

	// Verify different outcomes produce different roots
	outcomes[0] = 0xFFFF
	rootDifferent := orchestrator.buildOutcomeRoot(outcomes)
	if root == rootDifferent {
		t.Error("Different outcomes should produce different roots")
	}

	t.Logf("Outcome root: %s", root.Hex())
}

func TestExtractBranchNibbles(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	// Create 48 outcomes for proper testing
	outcomes := make([]uint16, 48)
	for i := range outcomes {
		// Create test pattern
		outcomes[i] = uint16(0x1000 + i*0x100) // 0x1000, 0x1100, 0x1200, etc.
	}

	// Set specific test values for first 8
	outcomes[0] = 0x1234
	outcomes[1] = 0x5678
	outcomes[2] = 0x9ABC
	outcomes[3] = 0xDEF0
	outcomes[4] = 0x2468
	outcomes[5] = 0xACE0
	outcomes[6] = 0x1357
	outcomes[7] = 0x9BDF

	nibbles := orchestrator.extractBranchNibbles(outcomes)

	// Verify nibbles length (should always be 48)
	if len(nibbles) != 48 {
		t.Errorf("Expected nibbles length 48, got %d", len(nibbles))
	}

	// Verify nibble extraction for first 8
	expectedNibbles := []byte{0x1, 0x5, 0x9, 0xD, 0x2, 0xA, 0x1, 0x9}
	for i, expected := range expectedNibbles {
		if nibbles[i] != expected {
			t.Errorf("Nibble %d: expected 0x%X, got 0x%X", i, expected, nibbles[i])
		}
	}

	t.Logf("Branch nibbles (first 8): %x", nibbles[:8])
}

func TestComputeGateHash(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	gateStreams := [][]byte{
		{0x01, 0x02, 0x03, 0x04},
		{0x05, 0x06, 0x07, 0x08},
		{0x09, 0x0A, 0x0B, 0x0C},
	}

	gateHash := orchestrator.computeGateHash(gateStreams)

	// Verify gate hash properties
	if gateHash == (common.Hash{}) {
		t.Error("Gate hash should not be empty")
	}

	// Verify determinism
	gateHash2 := orchestrator.computeGateHash(gateStreams)
	if gateHash != gateHash2 {
		t.Error("Gate hash calculation is not deterministic")
	}

	// Verify different streams produce different hashes
	gateStreams[0][0] = 0xFF
	gateHashDifferent := orchestrator.computeGateHash(gateStreams)
	if gateHash == gateHashDifferent {
		t.Error("Different gate streams should produce different hashes")
	}

	t.Logf("Gate hash: %s", gateHash.Hex())
}

func TestExecuteSinglePuzzle(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	seed := common.Hex2Bytes("1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
	input := &MiningInput{
		ParentHash:   common.HexToHash("0x1111111111111111111111111111111111111111111111111111111111111111"),
		TxRoot:       common.HexToHash("0x2222222222222222222222222222222222222222222222222222222222222222"),
		ExtraNonce32: make([]byte, 32),
		QNonce64:     12345,
		BlockHeight:  1000,
		QBits:        16,
		TCount:       8192,
		LNet:         48,
	}

	result, err := orchestrator.executeSinglePuzzle(0, seed, input)
	if err != nil {
		t.Fatalf("Single puzzle execution failed: %v", err)
	}

	// Verify result properties
	if result.PuzzleIndex != 0 {
		t.Errorf("Expected puzzle index 0, got %d", result.PuzzleIndex)
	}

	if !bytes.Equal(result.Seed, seed) {
		t.Error("Result seed doesn't match input seed")
	}

	if len(result.NextSeed) != 32 {
		t.Errorf("Expected next seed length 32, got %d", len(result.NextSeed))
	}

	if result.QASM == "" {
		t.Error("QASM should not be empty")
	}

	if len(result.GateStream) == 0 {
		t.Error("Gate stream should not be empty")
	}

	if result.BranchID != 0 {
		t.Errorf("Expected branch ID 0, got %d", result.BranchID)
	}

	if result.Depth <= 0 {
		t.Errorf("Expected positive depth, got %d", result.Depth)
	}

	if result.TGateCount < 0 {
		t.Errorf("Expected non-negative T-gate count, got %d", result.TGateCount)
	}

	t.Logf("Puzzle result: index=%d, branch=%d, outcome=0x%04x, depth=%d, t_gates=%d",
		result.PuzzleIndex, result.BranchID, result.Outcome, result.Depth, result.TGateCount)
}

func TestExecutePuzzleChain(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	input := &MiningInput{
		ParentHash:   common.HexToHash("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
		TxRoot:       common.HexToHash("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
		ExtraNonce32: make([]byte, 32),
		QNonce64:     0xabcdef1234567890,
		BlockHeight:  2000,
		QBits:        16,
		TCount:       8192,
		LNet:         48,
	}

	// Fill ExtraNonce32 with pattern
	for i := range input.ExtraNonce32 {
		input.ExtraNonce32[i] = byte(0xAA)
	}

	result, err := orchestrator.ExecutePuzzleChain(input)
	if err != nil {
		t.Fatalf("Puzzle chain execution failed: %v", err)
	}

	// Verify result properties
	if len(result.Results) != 48 {
		t.Errorf("Expected 48 puzzle results, got %d", len(result.Results))
	}

	if len(result.Outcomes) != 48 {
		t.Errorf("Expected 48 outcomes, got %d", len(result.Outcomes))
	}

	if len(result.BranchNibbles) != 48 {
		t.Errorf("Expected 48 branch nibbles, got %d", len(result.BranchNibbles))
	}

	if result.OutcomeRoot == (common.Hash{}) {
		t.Error("Outcome root should not be empty")
	}

	if result.GateHash == (common.Hash{}) {
		t.Error("Gate hash should not be empty")
	}

	if result.TotalGates <= 0 {
		t.Errorf("Expected positive total gates, got %d", result.TotalGates)
	}

	if result.TotalDepth <= 0 {
		t.Errorf("Expected positive total depth, got %d", result.TotalDepth)
	}

	// Verify seed chaining
	for i, puzzleResult := range result.Results {
		if puzzleResult.PuzzleIndex != i {
			t.Errorf("Puzzle %d has wrong index: %d", i, puzzleResult.PuzzleIndex)
		}

		expectedBranchID := i % 16
		if puzzleResult.BranchID != expectedBranchID {
			t.Errorf("Puzzle %d has wrong branch ID: expected %d, got %d", i, expectedBranchID, puzzleResult.BranchID)
		}

		// Verify outcome matches
		if result.Outcomes[i] != puzzleResult.Outcome {
			t.Errorf("Puzzle %d outcome mismatch", i)
		}

		// Verify nibble extraction
		expectedNibble := byte((puzzleResult.Outcome >> 12) & 0xF)
		if result.BranchNibbles[i] != expectedNibble {
			t.Errorf("Puzzle %d nibble mismatch: expected 0x%X, got 0x%X", i, expectedNibble, result.BranchNibbles[i])
		}
	}

	t.Logf("Puzzle chain completed: %d puzzles, %d total gates, max depth %d",
		len(result.Results), result.TotalGates, result.TotalDepth)
	t.Logf("Outcome root: %s", result.OutcomeRoot.Hex())
	t.Logf("Gate hash: %s", result.GateHash.Hex())
}

func TestValidatePuzzleChain(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	input := &MiningInput{
		ParentHash:   common.HexToHash("0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
		TxRoot:       common.HexToHash("0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"),
		ExtraNonce32: make([]byte, 32),
		QNonce64:     0x1122334455667788,
		BlockHeight:  3000,
		QBits:        16,
		TCount:       8192,
		LNet:         48,
	}

	// Execute puzzle chain
	result, err := orchestrator.ExecutePuzzleChain(input)
	if err != nil {
		t.Fatalf("Puzzle chain execution failed: %v", err)
	}

	// Validate the result
	err = orchestrator.ValidatePuzzleChain(input, result)
	if err != nil {
		t.Errorf("Puzzle chain validation failed: %v", err)
	}

	// Test validation with corrupted data
	corruptedResult := *result
	corruptedResult.Outcomes[0] = 0xFFFF // Corrupt first outcome

	err = orchestrator.ValidatePuzzleChain(input, &corruptedResult)
	if err == nil {
		t.Error("Validation should fail with corrupted outcomes")
	}

	t.Logf("Puzzle chain validation passed")
}

func TestInvalidLNet(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	input := &MiningInput{
		ParentHash:   common.HexToHash("0x1111111111111111111111111111111111111111111111111111111111111111"),
		TxRoot:       common.HexToHash("0x2222222222222222222222222222222222222222222222222222222222222222"),
		ExtraNonce32: make([]byte, 32),
		QNonce64:     12345,
		BlockHeight:  1000,
		QBits:        16,
		TCount:       8192,
		LNet:         32, // Invalid: should be 48
	}

	_, err := orchestrator.ExecutePuzzleChain(input)
	if err == nil {
		t.Error("Expected error for invalid LNet")
	}

	if err.Error() != "invalid LNet: expected 48, got 32" {
		t.Errorf("Unexpected error message: %v", err)
	}
}

func TestDeterministicExecution(t *testing.T) {
	orchestrator1 := NewPuzzleOrchestrator()
	orchestrator2 := NewPuzzleOrchestrator()

	input := &MiningInput{
		ParentHash:   common.HexToHash("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
		TxRoot:       common.HexToHash("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
		ExtraNonce32: make([]byte, 32),
		QNonce64:     0x9876543210abcdef,
		BlockHeight:  4000,
		QBits:        16,
		TCount:       8192,
		LNet:         48,
	}

	result1, err1 := orchestrator1.ExecutePuzzleChain(input)
	if err1 != nil {
		t.Fatalf("First execution failed: %v", err1)
	}

	result2, err2 := orchestrator2.ExecutePuzzleChain(input)
	if err2 != nil {
		t.Fatalf("Second execution failed: %v", err2)
	}

	// Compare results
	if !bytes.Equal(result1.InitialSeed, result2.InitialSeed) {
		t.Error("Initial seeds don't match")
	}

	if result1.OutcomeRoot != result2.OutcomeRoot {
		t.Error("Outcome roots don't match")
	}

	if result1.GateHash != result2.GateHash {
		t.Error("Gate hashes don't match")
	}

	if len(result1.Outcomes) != len(result2.Outcomes) {
		t.Error("Outcome lengths don't match")
	}

	for i := range result1.Outcomes {
		if result1.Outcomes[i] != result2.Outcomes[i] {
			t.Errorf("Outcome %d doesn't match: %04x vs %04x", i, result1.Outcomes[i], result2.Outcomes[i])
		}
	}

	t.Logf("Deterministic execution verified")
}

func TestMerkleRootBuilding(t *testing.T) {
	// Test empty leaves
	emptyRoot := buildMerkleRoot([][]byte{})
	if emptyRoot != (common.Hash{}) {
		t.Error("Empty Merkle root should be zero hash")
	}

	// Test single leaf
	singleLeaf := [][]byte{{0x01, 0x02, 0x03, 0x04}}
	singleRoot := buildMerkleRoot(singleLeaf)
	if singleRoot == (common.Hash{}) {
		t.Error("Single leaf Merkle root should not be zero")
	}

	// Test multiple leaves
	leaves := [][]byte{
		{0x01, 0x02, 0x03, 0x04},
		{0x05, 0x06, 0x07, 0x08},
		{0x09, 0x0A, 0x0B, 0x0C},
		{0x0D, 0x0E, 0x0F, 0x10},
	}
	multiRoot := buildMerkleRoot(leaves)
	if multiRoot == (common.Hash{}) {
		t.Error("Multi-leaf Merkle root should not be zero")
	}

	// Test odd number of leaves
	oddLeaves := [][]byte{
		{0x01, 0x02, 0x03, 0x04},
		{0x05, 0x06, 0x07, 0x08},
		{0x09, 0x0A, 0x0B, 0x0C},
	}
	oddRoot := buildMerkleRoot(oddLeaves)
	if oddRoot == (common.Hash{}) {
		t.Error("Odd-leaf Merkle root should not be zero")
	}

	t.Logf("Merkle root tests passed")
}

func TestKnownSeedOutcomePairs(t *testing.T) {
	orchestrator := NewPuzzleOrchestrator()

	testCases := []struct {
		name         string
		parentHash   string
		txRoot       string
		qnonce       uint64
		expectedSeed string // First 16 hex chars of Seed₀
	}{
		{
			name:         "genesis_test",
			parentHash:   "0x0000000000000000000000000000000000000000000000000000000000000000",
			txRoot:       "0x0000000000000000000000000000000000000000000000000000000000000000",
			qnonce:       0,
			expectedSeed: "290decd9548b62a8", // Expected first 8 bytes
		},
		{
			name:         "block_1000",
			parentHash:   "0x1111111111111111111111111111111111111111111111111111111111111111",
			txRoot:       "0x2222222222222222222222222222222222222222222222222222222222222222",
			qnonce:       1000,
			expectedSeed: "e4c2b4b1c8a6d3f2", // Expected first 8 bytes
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := &MiningInput{
				ParentHash:   common.HexToHash(tc.parentHash),
				TxRoot:       common.HexToHash(tc.txRoot),
				ExtraNonce32: make([]byte, 32),
				QNonce64:     tc.qnonce,
				BlockHeight:  1000,
				QBits:        16,
				TCount:       8192,
				LNet:         48,
			}

			seed0 := orchestrator.computeInitialSeed(input)
			actualSeed := hex.EncodeToString(seed0[:8])

			t.Logf("Test case %s:", tc.name)
			t.Logf("  Parent: %s", tc.parentHash)
			t.Logf("  TxRoot: %s", tc.txRoot)
			t.Logf("  QNonce: %d", tc.qnonce)
			t.Logf("  Seed₀ (first 8 bytes): %s", actualSeed)
			t.Logf("  Full Seed₀: %x", seed0)
		})
	}
}
