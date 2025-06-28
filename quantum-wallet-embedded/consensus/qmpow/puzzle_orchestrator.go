// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/log"
)

// PuzzleOrchestrator manages the sequential execution of 128 quantum puzzles
// for the Quantum-Geth mining process
type PuzzleOrchestrator struct {
	templateEngine *BranchTemplateEngine
	compiler       *CanonicalCompiler
	backend        QuantumBackend // Will be defined in next task
}

// PuzzleResult represents the result of executing a single quantum puzzle
type PuzzleResult struct {
	PuzzleIndex int    // Puzzle index (0-127)
	Seed        []byte // Input seed for this puzzle
	NextSeed    []byte // Chained seed for next puzzle
	QASM        string // Instantiated QASM circuit
	GateStream  []byte // Canonical compiled gate stream
	Outcome     uint16 // 16-bit quantum execution outcome
	BranchID    int    // Template branch ID used
	Depth       int    // Circuit depth
	TGateCount  int    // T-gate count
	ExecutionMS int64  // Execution time in milliseconds
}

// PuzzleChainResult represents the complete result of executing all 128 puzzles
type PuzzleChainResult struct {
	InitialSeed   []byte         // Seed₀
	Results       []PuzzleResult // Results for puzzles 0-127
	Outcomes      []uint16       // All 128 outcomes
	OutcomeRoot   common.Hash    // Merkle root of outcomes
	BranchNibbles []byte         // Full bytes of outcomes (128 bytes)
	GateHash      common.Hash    // SHA256 of concatenated gate streams
	TotalGates    int            // Total gates across all puzzles
	TotalDepth    int            // Maximum depth across puzzles
	ExecutionMS   int64          // Total execution time
}

// MiningInput represents the input parameters for puzzle orchestration
type MiningInput struct {
	ParentHash   common.Hash // Parent block hash
	TxRoot       common.Hash // Transaction root
	ExtraNonce32 []byte      // 32-byte extra nonce
	QNonce64     uint64      // 64-bit quantum nonce
	BlockHeight  uint64      // Current block height
	QBits        uint16      // Qubits per puzzle
	TCount       uint32      // T-gates per puzzle
	LNet         uint16      // Number of chained puzzles (MUST be exactly 128)
}

// NewPuzzleOrchestrator creates a new puzzle orchestrator
func NewPuzzleOrchestrator() *PuzzleOrchestrator {
	return &PuzzleOrchestrator{
		templateEngine: NewBranchTemplateEngine(),
		compiler:       NewCanonicalCompiler(),
		backend:        NewSimulatorBackend(), // Mock backend for now
	}
}

// ExecutePuzzleChain executes the complete 128-puzzle chain according to v0.9 spec
func (po *PuzzleOrchestrator) ExecutePuzzleChain(input *MiningInput) (*PuzzleChainResult, error) {
	if input.LNet != 128 {
		return nil, fmt.Errorf("SECURITY VIOLATION: LNet must be exactly 128 chained puzzles, got %d", input.LNet)
	}

	if input.TCount < 20 {
		return nil, fmt.Errorf("SECURITY VIOLATION: TCount must be at least 20 T-gates per puzzle, got %d", input.TCount)
	}

	log.Info("🧩 Starting puzzle chain execution",
		"parent", input.ParentHash.Hex()[:10],
		"qnonce", input.QNonce64,
		"height", input.BlockHeight,
		"qbits", input.QBits,
		"tcount", input.TCount)

	// Step 1: Compute Seed₀
	seed0 := po.computeInitialSeed(input)

	// Step 2: Execute 128 sequential puzzles
	results := make([]PuzzleResult, 128)
	outcomes := make([]uint16, 128)
	gateStreams := make([][]byte, 128)
	currentSeed := seed0

	for i := 0; i < 128; i++ {
		result, err := po.executeSinglePuzzle(i, currentSeed, input)
		if err != nil {
			return nil, fmt.Errorf("puzzle %d failed: %v", i, err)
		}

		results[i] = *result
		outcomes[i] = result.Outcome
		gateStreams[i] = result.GateStream
		currentSeed = result.NextSeed

		if i%16 == 15 {
			log.Debug("🔗 Puzzle chain progress",
				"completed", i+1,
				"latest_outcome", fmt.Sprintf("0x%04x", result.Outcome),
				"total_gates", result.TGateCount)
		}
	}

	// Step 3: Build OutcomeRoot (Merkle tree of outcomes)
	outcomeRoot := po.buildOutcomeRoot(outcomes)

	// Step 4: Extract BranchNibbles (high 4 bits of each outcome)
	branchNibbles := po.extractBranchNibbles(outcomes)

	// Step 5: Compute GateHash (SHA256 of concatenated gate streams)
	gateHash := po.computeGateHash(gateStreams)

	// Step 6: Compute statistics
	totalGates := 0
	maxDepth := 0
	totalExecTime := int64(0)

	for _, result := range results {
		totalGates += result.TGateCount
		if result.Depth > maxDepth {
			maxDepth = result.Depth
		}
		totalExecTime += result.ExecutionMS
	}

	chainResult := &PuzzleChainResult{
		InitialSeed:   seed0,
		Results:       results,
		Outcomes:      outcomes,
		OutcomeRoot:   outcomeRoot,
		BranchNibbles: branchNibbles,
		GateHash:      gateHash,
		TotalGates:    totalGates,
		TotalDepth:    maxDepth,
		ExecutionMS:   totalExecTime,
	}

	log.Info("✅ Puzzle chain completed",
		"outcome_root", outcomeRoot.Hex()[:10],
		"gate_hash", gateHash.Hex()[:10],
		"total_gates", totalGates,
		"max_depth", maxDepth,
		"exec_time_ms", totalExecTime)

	return chainResult, nil
}

// computeInitialSeed computes Seed₀ = SHA256(ParentHash‖TxRoot‖ExtraNonce32‖QNonce64)
func (po *PuzzleOrchestrator) computeInitialSeed(input *MiningInput) []byte {
	hasher := sha256.New()

	// ParentHash (32 bytes)
	hasher.Write(input.ParentHash.Bytes())

	// TxRoot (32 bytes)
	hasher.Write(input.TxRoot.Bytes())

	// ExtraNonce32 (32 bytes)
	hasher.Write(input.ExtraNonce32)

	// QNonce64 (8 bytes, big-endian)
	qnonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(qnonceBytes, input.QNonce64)
	hasher.Write(qnonceBytes)

	seed0 := hasher.Sum(nil)

	log.Debug("🌱 Generated Seed₀",
		"seed", fmt.Sprintf("%x", seed0),
		"parent", input.ParentHash.Hex()[:10],
		"qnonce", input.QNonce64)

	return seed0
}

// executeSinglePuzzle executes a single quantum puzzle in the chain
func (po *PuzzleOrchestrator) executeSinglePuzzle(index int, seed []byte, input *MiningInput) (*PuzzleResult, error) {
	startTime := getCurrentTimeMS()

	// Step 1: Select branch template (cycle through 16 templates)
	branchID := index % 16

	// Step 2: Instantiate branch template
	instantiation, err := po.templateEngine.InstantiateBranch(branchID, seed)
	if err != nil {
		return nil, fmt.Errorf("template instantiation failed: %v", err)
	}

	// Step 3: Canonical compile
	compileResult, err := po.compiler.CanonicalCompile(seed, instantiation.QASM)
	if err != nil {
		return nil, fmt.Errorf("canonical compilation failed: %v", err)
	}

	// Step 4: Execute on quantum backend
	outcome, err := po.backend.Execute(instantiation.QASM, seed)
	if err != nil {
		return nil, fmt.Errorf("quantum execution failed: %v", err)
	}

	// Step 5: Chain to next seed: Seed_{i+1} = SHA256(Seed_i ‖ Outcome_i)
	nextSeed := po.chainSeed(seed, outcome)

	executionTime := getCurrentTimeMS() - startTime

	result := &PuzzleResult{
		PuzzleIndex: index,
		Seed:        seed,
		NextSeed:    nextSeed,
		QASM:        instantiation.QASM,
		GateStream:  compileResult.GateStream,
		Outcome:     outcome,
		BranchID:    branchID,
		Depth:       compileResult.DAG.Depth,
		TGateCount:  compileResult.DAG.TGateCount,
		ExecutionMS: executionTime,
	}

	log.Debug("🎯 Puzzle executed",
		"index", index,
		"branch", branchID,
		"outcome", fmt.Sprintf("0x%04x", outcome),
		"depth", result.Depth,
		"t_gates", result.TGateCount,
		"exec_ms", executionTime)

	return result, nil
}

// chainSeed computes Seed_{i+1} = SHA256(Seed_i ‖ Outcome_i)
func (po *PuzzleOrchestrator) chainSeed(currentSeed []byte, outcome uint16) []byte {
	hasher := sha256.New()

	// Current seed
	hasher.Write(currentSeed)

	// Outcome as 2 bytes (big-endian)
	outcomeBytes := make([]byte, 2)
	binary.BigEndian.PutUint16(outcomeBytes, outcome)
	hasher.Write(outcomeBytes)

	return hasher.Sum(nil)
}

// buildOutcomeRoot builds Merkle root of 128 outcomes
func (po *PuzzleOrchestrator) buildOutcomeRoot(outcomes []uint16) common.Hash {
	// Convert outcomes to 32-byte leaves for Merkle tree
	leaves := make([][]byte, len(outcomes))
	for i, outcome := range outcomes {
		leaf := make([]byte, 32)
		binary.BigEndian.PutUint16(leaf[30:], outcome) // Place outcome in last 2 bytes
		leaves[i] = leaf
	}

	// Build Merkle tree
	return buildMerkleRoot(leaves)
}

// extractBranchNibbles extracts full low byte of each outcome (128 bytes)
func (po *PuzzleOrchestrator) extractBranchNibbles(outcomes []uint16) []byte {
	bytes := make([]byte, 128)
	for i, outcome := range outcomes {
		// Extract full low byte of the 16-bit outcome for maximum entropy
		bytes[i] = byte(outcome & 0xFF)
	}
	return bytes
}

// computeGateHash computes SHA256 of concatenated gate streams
func (po *PuzzleOrchestrator) computeGateHash(gateStreams [][]byte) common.Hash {
	hasher := sha256.New()

	for _, stream := range gateStreams {
		hasher.Write(stream)
	}

	hash := hasher.Sum(nil)
	return common.BytesToHash(hash)
}

// buildMerkleRoot builds a Merkle root from leaves
func buildMerkleRoot(leaves [][]byte) common.Hash {
	if len(leaves) == 0 {
		return common.Hash{}
	}

	// Copy leaves to avoid modifying input
	nodes := make([][]byte, len(leaves))
	copy(nodes, leaves)

	// Build tree bottom-up
	for len(nodes) > 1 {
		nextLevel := make([][]byte, 0, (len(nodes)+1)/2)

		for i := 0; i < len(nodes); i += 2 {
			hasher := sha256.New()
			hasher.Write(nodes[i])

			if i+1 < len(nodes) {
				hasher.Write(nodes[i+1])
			} else {
				// Odd number of nodes - duplicate last node
				hasher.Write(nodes[i])
			}

			nextLevel = append(nextLevel, hasher.Sum(nil))
		}

		nodes = nextLevel
	}

	return common.BytesToHash(nodes[0])
}

// getCurrentTimeMS returns current time in milliseconds
func getCurrentTimeMS() int64 {
	return crypto.Keccak256Hash([]byte("mock_time")).Big().Int64() % 1000 // Mock implementation
}

// ValidatePuzzleChain validates a puzzle chain result
func (po *PuzzleOrchestrator) ValidatePuzzleChain(input *MiningInput, result *PuzzleChainResult) error {
	// Validate initial seed
	expectedSeed0 := po.computeInitialSeed(input)
	if !slicesEqual(result.InitialSeed, expectedSeed0) {
		return fmt.Errorf("invalid initial seed")
	}

	// Validate seed chaining
	currentSeed := result.InitialSeed
	for i, puzzleResult := range result.Results {
		if !slicesEqual(puzzleResult.Seed, currentSeed) {
			return fmt.Errorf("invalid seed at puzzle %d", i)
		}

		expectedNextSeed := po.chainSeed(currentSeed, puzzleResult.Outcome)
		if !slicesEqual(puzzleResult.NextSeed, expectedNextSeed) {
			return fmt.Errorf("invalid next seed at puzzle %d", i)
		}

		currentSeed = puzzleResult.NextSeed
	}

	// Validate OutcomeRoot
	expectedRoot := po.buildOutcomeRoot(result.Outcomes)
	if result.OutcomeRoot != expectedRoot {
		return fmt.Errorf("invalid outcome root")
	}

	// Validate BranchNibbles
	expectedNibbles := po.extractBranchNibbles(result.Outcomes)
	if !slicesEqual(result.BranchNibbles, expectedNibbles) {
		return fmt.Errorf("invalid branch nibbles")
	}

	// Validate GateHash
	gateStreams := make([][]byte, len(result.Results))
	for i, r := range result.Results {
		gateStreams[i] = r.GateStream
	}
	expectedGateHash := po.computeGateHash(gateStreams)
	if result.GateHash != expectedGateHash {
		return fmt.Errorf("invalid gate hash")
	}

	return nil
}

// slicesEqual compares two byte slices
func slicesEqual(a, b []byte) bool {
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
