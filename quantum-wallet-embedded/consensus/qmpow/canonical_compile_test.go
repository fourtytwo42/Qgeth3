// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestCanonicalCompiler(t *testing.T) {
	compiler := NewCanonicalCompiler()

	if compiler.OptLevel != 0 {
		t.Errorf("Expected opt level 0, got %d", compiler.OptLevel)
	}
}

func TestCanonicalCompile(t *testing.T) {
	tests := []struct {
		name           string
		qasm           string
		seed           []byte
		expectedDepth  int
		expectedTGates int
		expectError    bool
	}{
		{
			name: "simple_h_gate",
			qasm: `qreg q[2];
H q[0];`,
			seed:           make([]byte, 32), // Zero seed
			expectedDepth:  1,
			expectedTGates: 0,
			expectError:    false,
		},
		{
			name: "sequential_gates",
			qasm: `qreg q[2];
H q[0];
S q[0];
T q[0];`,
			seed:           make([]byte, 32), // Zero seed
			expectedDepth:  3,
			expectedTGates: 1,
			expectError:    false,
		},
		{
			name: "parallel_gates",
			qasm: `qreg q[3];
H q[0];
H q[1];
H q[2];`,
			seed:           make([]byte, 32), // Zero seed
			expectedDepth:  1,
			expectedTGates: 0,
			expectError:    false,
		},
		{
			name: "cx_gates",
			qasm: `qreg q[3];
H q[0];
CX q[0],q[1];
CX q[1],q[2];`,
			seed:           make([]byte, 32), // Zero seed
			expectedDepth:  3,
			expectedTGates: 0,
			expectError:    false,
		},
		{
			name: "complex_circuit",
			qasm: `qreg q[4];
H q[0];
H q[1];
CX q[0],q[1];
T q[0];
S q[1];
CX q[0],q[2];
T q[2];`,
			seed:           make([]byte, 32), // Zero seed
			expectedDepth:  5,                // Corrected expected depth
			expectedTGates: 2,
			expectError:    false,
		},
		{
			name: "invalid_qasm",
			qasm: `qreg q[2];
INVALID q[0];`,
			seed:        make([]byte, 32),
			expectError: true,
		},
	}

	compiler := NewCanonicalCompiler()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := compiler.CanonicalCompile(tt.seed, tt.qasm)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Validate metadata
			if result.Metadata.OptLevel != 0 {
				t.Errorf("Expected opt level 0, got %d", result.Metadata.OptLevel)
			}

			if result.DAG.Depth != tt.expectedDepth {
				t.Errorf("Expected depth %d, got %d", tt.expectedDepth, result.DAG.Depth)
			}

			if result.DAG.TGateCount != tt.expectedTGates {
				t.Errorf("Expected %d T-gates, got %d", tt.expectedTGates, result.DAG.TGateCount)
			}

			// Validate gate stream
			if len(result.GateStream) == 0 {
				t.Error("Gate stream is empty")
			}

			// Validate DAG structure
			if len(result.DAG.Gates) == 0 {
				t.Error("DAG has no gates")
			}

			t.Logf("Compiled circuit: %d gates, depth %d, T-gates %d, stream %d bytes",
				len(result.DAG.Gates), result.DAG.Depth, result.DAG.TGateCount, len(result.GateStream))
		})
	}
}

func TestDAGDependencies(t *testing.T) {
	qasm := `qreg q[3];
H q[0];
CX q[0],q[1];
T q[1];
CX q[1],q[2];`

	compiler := NewCanonicalCompiler()
	result, err := compiler.CanonicalCompile(make([]byte, 32), qasm)
	if err != nil {
		t.Fatalf("Compilation error: %v", err)
	}

	// Verify dependencies
	gates := result.DAG.Gates

	// Gate 0 (H q[0]) should have no dependencies
	if len(gates[0].Dependencies) != 0 {
		t.Errorf("Gate 0 should have no dependencies, got %v", gates[0].Dependencies)
	}

	// Gate 1 (CX q[0],q[1]) should depend on gate 0
	if len(gates[1].Dependencies) != 1 || gates[1].Dependencies[0] != 0 {
		t.Errorf("Gate 1 should depend on gate 0, got %v", gates[1].Dependencies)
	}

	// Gate 2 (T q[1]) should depend on gate 1
	if len(gates[2].Dependencies) != 1 || gates[2].Dependencies[0] != 1 {
		t.Errorf("Gate 2 should depend on gate 1, got %v", gates[2].Dependencies)
	}

	// Gate 3 (CX q[1],q[2]) should depend on gate 2
	if len(gates[3].Dependencies) != 1 || gates[3].Dependencies[0] != 2 {
		t.Errorf("Gate 3 should depend on gate 2, got %v", gates[3].Dependencies)
	}
}

func TestZMaskApplication(t *testing.T) {
	qasm := `qreg q[2];
H q[0];
S q[1];`

	compiler := NewCanonicalCompiler()

	// Test with zero seed (no Z-mask)
	zeroSeed := make([]byte, 32)
	result1, err := compiler.CanonicalCompile(zeroSeed, qasm)
	if err != nil {
		t.Fatalf("Compilation error with zero seed: %v", err)
	}

	// Test with non-zero seed (Z-mask applied)
	nonZeroSeed := sha256.Sum256([]byte("test_seed"))
	result2, err := compiler.CanonicalCompile(nonZeroSeed[:], qasm)
	if err != nil {
		t.Fatalf("Compilation error with non-zero seed: %v", err)
	}

	// Results should be different due to Z-mask application
	if bytes.Equal(result1.GateStream, result2.GateStream) {
		t.Error("Gate streams should be different with different seeds")
	}

	// Non-zero seed should have more gates (due to Z-mask gates)
	if len(result2.DAG.Gates) <= len(result1.DAG.Gates) {
		t.Errorf("Expected more gates with Z-mask, got %d vs %d",
			len(result2.DAG.Gates), len(result1.DAG.Gates))
	}

	t.Logf("Zero seed: %d gates, Non-zero seed: %d gates",
		len(result1.DAG.Gates), len(result2.DAG.Gates))
}

func TestDeterministicCompilation(t *testing.T) {
	qasm := `qreg q[3];
H q[0];
CX q[0],q[1];
T q[1];
S q[2];`

	seed := sha256.Sum256([]byte("deterministic_test"))
	compiler := NewCanonicalCompiler()

	// Compile the same circuit multiple times
	result1, err := compiler.CanonicalCompile(seed[:], qasm)
	if err != nil {
		t.Fatalf("First compilation error: %v", err)
	}

	result2, err := compiler.CanonicalCompile(seed[:], qasm)
	if err != nil {
		t.Fatalf("Second compilation error: %v", err)
	}

	// Results should be identical
	if !bytes.Equal(result1.GateStream, result2.GateStream) {
		t.Error("Gate streams should be identical for deterministic compilation")
		t.Logf("Stream 1: %s", hex.EncodeToString(result1.GateStream))
		t.Logf("Stream 2: %s", hex.EncodeToString(result2.GateStream))
	}

	if result1.DAG.Depth != result2.DAG.Depth {
		t.Errorf("Depths should be identical: %d vs %d", result1.DAG.Depth, result2.DAG.Depth)
	}

	if result1.DAG.TGateCount != result2.DAG.TGateCount {
		t.Errorf("T-gate counts should be identical: %d vs %d", result1.DAG.TGateCount, result2.DAG.TGateCount)
	}
}

func TestGateStreamSerialization(t *testing.T) {
	qasm := `qreg q[2];
H q[0];
CX q[0],q[1];
Z[255] q[1];`

	compiler := NewCanonicalCompiler()
	result, err := compiler.CanonicalCompile(make([]byte, 32), qasm)
	if err != nil {
		t.Fatalf("Compilation error: %v", err)
	}

	stream := result.GateStream

	// Verify header structure (12 bytes)
	if len(stream) < 12 {
		t.Fatalf("Gate stream too short: %d bytes", len(stream))
	}

	// Extract header fields
	qubits := (int(stream[0]) << 8) | int(stream[1])
	gateCount := (int(stream[2]) << 24) | (int(stream[3]) << 16) | (int(stream[4]) << 8) | int(stream[5])
	depth := (int(stream[6]) << 8) | int(stream[7])
	tCount := (int(stream[8]) << 24) | (int(stream[9]) << 16) | (int(stream[10]) << 8) | int(stream[11])

	if qubits != result.DAG.QubitsUsed {
		t.Errorf("Header qubits mismatch: %d vs %d", qubits, result.DAG.QubitsUsed)
	}

	if gateCount != len(result.DAG.Gates) {
		t.Errorf("Header gate count mismatch: %d vs %d", gateCount, len(result.DAG.Gates))
	}

	if depth != result.DAG.Depth {
		t.Errorf("Header depth mismatch: %d vs %d", depth, result.DAG.Depth)
	}

	if tCount != result.DAG.TGateCount {
		t.Errorf("Header T-count mismatch: %d vs %d", tCount, result.DAG.TGateCount)
	}

	t.Logf("Gate stream: %d bytes, header: qubits=%d, gates=%d, depth=%d, t-count=%d",
		len(stream), qubits, gateCount, depth, tCount)
}

func TestValidateCompilation(t *testing.T) {
	qasm := `qreg q[2];
H q[0];
T q[0];
S q[1];`

	compiler := NewCanonicalCompiler()
	result, err := compiler.CanonicalCompile(make([]byte, 32), qasm)
	if err != nil {
		t.Fatalf("Compilation error: %v", err)
	}

	// Valid validation
	err = compiler.ValidateCompilation(result, result.DAG.Depth, result.DAG.TGateCount)
	if err != nil {
		t.Errorf("Valid compilation should pass validation: %v", err)
	}

	// Invalid depth validation
	err = compiler.ValidateCompilation(result, 999, result.DAG.TGateCount)
	if err == nil {
		t.Error("Invalid depth should fail validation")
	}

	// Invalid T-gate count validation
	err = compiler.ValidateCompilation(result, result.DAG.Depth, 999)
	if err == nil {
		t.Error("Invalid T-gate count should fail validation")
	}
}

func TestGetGateHash(t *testing.T) {
	qasm := `qreg q[2];
H q[0];
CX q[0],q[1];`

	compiler := NewCanonicalCompiler()
	result, err := compiler.CanonicalCompile(make([]byte, 32), qasm)
	if err != nil {
		t.Fatalf("Compilation error: %v", err)
	}

	// Test gate hash computation
	gateHash := GetGateHash(result.GateStream)
	if len(gateHash) != 32 {
		t.Errorf("Gate hash should be 32 bytes, got %d", len(gateHash))
	}

	// Hash should be deterministic
	gateHash2 := GetGateHash(result.GateStream)
	if !bytes.Equal(gateHash, gateHash2) {
		t.Error("Gate hash should be deterministic")
	}

	t.Logf("Gate hash: %s", hex.EncodeToString(gateHash))
}

func TestKnownSeedQASMPairs(t *testing.T) {
	// Test cases with known seed/QASM→stream pairs for regression testing
	testCases := []struct {
		name         string
		seed         string
		qasm         string
		expectedHash string // Expected hash of gate stream
	}{
		{
			name: "basic_h_gate",
			seed: "0000000000000000000000000000000000000000000000000000000000000000",
			qasm: `qreg q[1];
H q[0];`,
			expectedHash: "", // Will be computed in test
		},
		{
			name: "cx_circuit",
			seed: "0000000000000000000000000000000000000000000000000000000000000000",
			qasm: `qreg q[2];
H q[0];
CX q[0],q[1];`,
			expectedHash: "", // Will be computed in test
		},
	}

	compiler := NewCanonicalCompiler()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			seed, err := hex.DecodeString(tc.seed)
			if err != nil {
				t.Fatalf("Invalid seed hex: %v", err)
			}

			result, err := compiler.CanonicalCompile(seed, tc.qasm)
			if err != nil {
				t.Fatalf("Compilation error: %v", err)
			}

			gateHash := GetGateHash(result.GateStream)
			hashHex := hex.EncodeToString(gateHash)

			t.Logf("Test case %s:", tc.name)
			t.Logf("  Seed: %s", tc.seed)
			t.Logf("  Gates: %d", len(result.DAG.Gates))
			t.Logf("  Depth: %d", result.DAG.Depth)
			t.Logf("  T-gates: %d", result.DAG.TGateCount)
			t.Logf("  Stream bytes: %d", len(result.GateStream))
			t.Logf("  Gate hash: %s", hashHex)

			// Store expected hash for future regression testing
			// In a real implementation, you would compare against known good values
		})
	}
}

func BenchmarkCanonicalCompile(b *testing.B) {
	qasm := `qreg q[4];
H q[0];
H q[1];
CX q[0],q[1];
T q[0];
S q[1];
CX q[1],q[2];
T q[2];
CX q[2],q[3];
H q[3];`

	seed := sha256.Sum256([]byte("benchmark_seed"))
	compiler := NewCanonicalCompiler()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := compiler.CanonicalCompile(seed[:], qasm)
		if err != nil {
			b.Fatalf("Compilation error: %v", err)
		}
	}
}

func BenchmarkGateStreamSerialization(b *testing.B) {
	qasm := `qreg q[4];
H q[0];
CX q[0],q[1];
T q[1];
S q[2];
CX q[2],q[3];`

	compiler := NewCanonicalCompiler()
	result, err := compiler.CanonicalCompile(make([]byte, 32), qasm)
	if err != nil {
		b.Fatalf("Compilation error: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = compiler.serializeDAG(result.DAG)
	}
}
