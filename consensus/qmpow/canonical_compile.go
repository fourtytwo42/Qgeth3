// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements canonical compilation for Quantum-Geth v0.9–BareBones+Halving
// Canonical compilation ensures deterministic QASM-lite → gate stream transformation

package qmpow

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"

	"github.com/ethereum/go-ethereum/log"
)

// CanonicalCompiler implements deterministic QASM-lite compilation
type CanonicalCompiler struct {
	OptLevel int // Always 0 for canonical compilation
}

// CompileResult represents the result of canonical compilation
type CompileResult struct {
	GateStream []byte      // Canonical byte stream of gates
	DAG        *QuantumDAG // Directed Acyclic Graph representation
	Metadata   CompileMetadata
}

// QuantumDAG represents the quantum circuit as a directed acyclic graph
type QuantumDAG struct {
	Gates      []DAGGate
	QubitsUsed int
	Depth      int
	TGateCount int
}

// DAGGate represents a gate in the DAG with dependencies
type DAGGate struct {
	ID           int
	Type         string
	Qubits       []int
	ZMask        int
	Dependencies []int // Gate IDs this gate depends on
	Layer        int   // Execution layer (for depth calculation)
}

// CompileMetadata contains compilation metadata
type CompileMetadata struct {
	OriginalGates int
	CompiledGates int
	Depth         int
	TGateCount    int
	OptLevel      int
	Seed          []byte
}

// NewCanonicalCompiler creates a new canonical compiler
func NewCanonicalCompiler() *CanonicalCompiler {
	return &CanonicalCompiler{
		OptLevel: 0, // Fixed at 0 for canonical compilation
	}
}

// CanonicalCompile performs deterministic compilation of QASM-lite with Z-mask application
func (c *CanonicalCompiler) CanonicalCompile(seed []byte, qasm string) (*CompileResult, error) {
	log.Debug("🔧 Starting canonical compilation",
		"seed", hex.EncodeToString(seed)[:16]+"...",
		"qasm_length", len(qasm))

	// Step 1: Parse QASM-lite
	parser := NewQASMLiteParser(qasm)
	program, err := parser.Parse()
	if err != nil {
		return nil, fmt.Errorf("QASM parse error: %v", err)
	}

	// Step 2: Apply Pauli-Z mask from SHA256(seed)
	maskedProgram := ApplyZMask(program, seed)

	// Step 3: Build DAG (opt_level=0 - no optimization)
	dag, err := c.buildDAG(maskedProgram)
	if err != nil {
		return nil, fmt.Errorf("DAG build error: %v", err)
	}

	// Step 4: Serialize gate list byte-exactly
	gateStream := c.serializeDAG(dag)

	result := &CompileResult{
		GateStream: gateStream,
		DAG:        dag,
		Metadata: CompileMetadata{
			OriginalGates: len(program.Gates),
			CompiledGates: len(dag.Gates),
			Depth:         dag.Depth,
			TGateCount:    dag.TGateCount,
			OptLevel:      c.OptLevel,
			Seed:          seed,
		},
	}

	log.Debug("✅ Canonical compilation completed",
		"original_gates", result.Metadata.OriginalGates,
		"compiled_gates", result.Metadata.CompiledGates,
		"depth", result.Metadata.Depth,
		"t_gates", result.Metadata.TGateCount,
		"stream_bytes", len(result.GateStream))

	return result, nil
}

// buildDAG constructs a DAG from the masked QASM program with opt_level=0
func (c *CanonicalCompiler) buildDAG(program *QASMLiteProgram) (*QuantumDAG, error) {
	dag := &QuantumDAG{
		Gates:      make([]DAGGate, 0, len(program.Gates)),
		QubitsUsed: program.QubitsUsed,
	}

	// Track last gate on each qubit for dependency calculation
	lastGateOnQubit := make(map[int]int) // qubit -> gate_id

	// Convert gates to DAG gates with dependencies (opt_level=0 means no reordering)
	for i, gate := range program.Gates {
		dagGate := DAGGate{
			ID:           i,
			Type:         gate.Type,
			Qubits:       make([]int, len(gate.Qubits)),
			ZMask:        gate.ZMask,
			Dependencies: make([]int, 0),
		}

		copy(dagGate.Qubits, gate.Qubits)

		// Calculate dependencies based on qubit usage
		for _, qubit := range gate.Qubits {
			if lastGateID, exists := lastGateOnQubit[qubit]; exists {
				dagGate.Dependencies = append(dagGate.Dependencies, lastGateID)
			}
			lastGateOnQubit[qubit] = i
		}

		// Remove duplicate dependencies and sort for deterministic output
		dagGate.Dependencies = removeDuplicatesAndSort(dagGate.Dependencies)

		dag.Gates = append(dag.Gates, dagGate)

		// Count T-gates
		if gate.Type == "T" {
			dag.TGateCount++
		}
	}

	// Calculate circuit depth (critical path)
	dag.Depth = c.calculateDepth(dag)

	// Assign layers for visualization/analysis
	c.assignLayers(dag)

	return dag, nil
}

// calculateDepth calculates the circuit depth (critical path length)
func (c *CanonicalCompiler) calculateDepth(dag *QuantumDAG) int {
	gateDepths := make(map[int]int)
	visiting := make(map[int]bool) // Track gates currently being visited to detect cycles

	// Calculate depth for each gate
	var calculateGateDepth func(gateID int) int
	calculateGateDepth = func(gateID int) int {
		// Check if we already calculated depth for this gate
		if depth, exists := gateDepths[gateID]; exists {
			return depth
		}

		// Check for circular dependency
		if visiting[gateID] {
			// Circular dependency detected - assign depth 1 to break cycle
			gateDepths[gateID] = 1
			return 1
		}

		// Check bounds
		if gateID < 0 || gateID >= len(dag.Gates) {
			return 0
		}

		// Mark as visiting
		visiting[gateID] = true

		gate := dag.Gates[gateID]
		maxDepth := 0

		// Find maximum depth of dependencies
		for _, depID := range gate.Dependencies {
			depDepth := calculateGateDepth(depID)
			if depDepth > maxDepth {
				maxDepth = depDepth
			}
		}

		// Unmark as visiting
		visiting[gateID] = false

		// Calculate and store depth
		depth := maxDepth + 1
		gateDepths[gateID] = depth
		return depth
	}

	// Calculate depth for all gates
	maxDepth := 0
	for i := range dag.Gates {
		depth := calculateGateDepth(i)
		if depth > maxDepth {
			maxDepth = depth
		}
	}

	return maxDepth
}

// assignLayers assigns execution layers to gates for analysis
func (c *CanonicalCompiler) assignLayers(dag *QuantumDAG) {
	gateDepths := make(map[int]int)
	visiting := make(map[int]bool) // Track gates currently being visited to detect cycles

	// Calculate depth for each gate (reuse logic from calculateDepth)
	var calculateGateDepth func(gateID int) int
	calculateGateDepth = func(gateID int) int {
		// Check if we already calculated depth for this gate
		if depth, exists := gateDepths[gateID]; exists {
			return depth
		}

		// Check for circular dependency
		if visiting[gateID] {
			// Circular dependency detected - assign depth 0 to break cycle
			gateDepths[gateID] = 0
			return 0
		}

		// Check bounds
		if gateID < 0 || gateID >= len(dag.Gates) {
			return 0
		}

		// Mark as visiting
		visiting[gateID] = true

		gate := dag.Gates[gateID]
		maxDepth := 0

		for _, depID := range gate.Dependencies {
			depDepth := calculateGateDepth(depID)
			if depDepth > maxDepth {
				maxDepth = depDepth
			}
		}

		// Unmark as visiting
		visiting[gateID] = false

		// Calculate and store depth (for layers, we use maxDepth directly)
		gateDepths[gateID] = maxDepth
		return maxDepth
	}

	// Assign layers
	for i := range dag.Gates {
		dag.Gates[i].Layer = calculateGateDepth(i)
	}
}

// serializeDAG serializes the DAG to a canonical byte stream
func (c *CanonicalCompiler) serializeDAG(dag *QuantumDAG) []byte {
	var result []byte

	// Header: qubits, gates, depth, t-count (12 bytes)
	result = append(result,
		byte(dag.QubitsUsed>>8), byte(dag.QubitsUsed&0xFF), // qubits (2 bytes)
		byte(len(dag.Gates)>>24), byte(len(dag.Gates)>>16), // gates count (4 bytes)
		byte(len(dag.Gates)>>8), byte(len(dag.Gates)&0xFF),
		byte(dag.Depth>>8), byte(dag.Depth&0xFF), // depth (2 bytes)
		byte(dag.TGateCount>>24), byte(dag.TGateCount>>16), // t-count (4 bytes)
		byte(dag.TGateCount>>8), byte(dag.TGateCount&0xFF))

	// Serialize gates in original order (opt_level=0 preserves order)
	for _, gate := range dag.Gates {
		result = append(result, c.serializeDAGGate(gate)...)
	}

	return result
}

// serializeDAGGate serializes a single DAG gate
func (c *CanonicalCompiler) serializeDAGGate(gate DAGGate) []byte {
	var result []byte

	// Gate type (1 byte)
	var gateTypeByte byte
	switch gate.Type {
	case "H":
		gateTypeByte = 0x01
	case "S":
		gateTypeByte = 0x02
	case "T":
		gateTypeByte = 0x03
	case "CX":
		gateTypeByte = 0x04
	case "Z":
		gateTypeByte = 0x05
	}
	result = append(result, gateTypeByte)

	// Gate ID (4 bytes)
	result = append(result,
		byte(gate.ID>>24), byte(gate.ID>>16),
		byte(gate.ID>>8), byte(gate.ID&0xFF))

	// Layer (2 bytes)
	result = append(result, byte(gate.Layer>>8), byte(gate.Layer&0xFF))

	// Qubit count (1 byte)
	result = append(result, byte(len(gate.Qubits)))

	// Qubits (2 bytes each)
	for _, qubit := range gate.Qubits {
		result = append(result, byte(qubit>>8), byte(qubit&0xFF))
	}

	// Z-mask for Z gates (4 bytes)
	if gate.Type == "Z" {
		result = append(result,
			byte(gate.ZMask>>24), byte(gate.ZMask>>16),
			byte(gate.ZMask>>8), byte(gate.ZMask&0xFF))
	}

	// Dependencies count (2 bytes)
	result = append(result, byte(len(gate.Dependencies)>>8), byte(len(gate.Dependencies)&0xFF))

	// Dependencies (4 bytes each)
	for _, depID := range gate.Dependencies {
		result = append(result,
			byte(depID>>24), byte(depID>>16),
			byte(depID>>8), byte(depID&0xFF))
	}

	return result
}

// ValidateCompilation validates a compilation result against expected properties
func (c *CanonicalCompiler) ValidateCompilation(result *CompileResult, expectedDepth, expectedTGates int) error {
	if result.Metadata.OptLevel != 0 {
		return fmt.Errorf("invalid opt level: expected 0, got %d", result.Metadata.OptLevel)
	}

	if result.DAG.Depth != expectedDepth {
		return fmt.Errorf("depth mismatch: expected %d, got %d", expectedDepth, result.DAG.Depth)
	}

	if result.DAG.TGateCount != expectedTGates {
		return fmt.Errorf("T-gate count mismatch: expected %d, got %d", expectedTGates, result.DAG.TGateCount)
	}

	if len(result.GateStream) == 0 {
		return fmt.Errorf("empty gate stream")
	}

	return nil
}

// removeDuplicatesAndSort removes duplicates and sorts integers
func removeDuplicatesAndSort(slice []int) []int {
	if len(slice) == 0 {
		return slice
	}

	// Remove duplicates using map
	seen := make(map[int]bool)
	result := make([]int, 0, len(slice))

	for _, item := range slice {
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}

	// Sort for deterministic output
	sort.Ints(result)
	return result
}

// GetGateHash computes SHA256 hash of the gate stream for GateHash field
func GetGateHash(gateStream []byte) []byte {
	hash := sha256.Sum256(gateStream)
	return hash[:]
}
