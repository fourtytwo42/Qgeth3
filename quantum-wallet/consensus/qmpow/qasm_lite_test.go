// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestQASMLiteParser(t *testing.T) {
	tests := []struct {
		name         string
		source       string
		expectError  bool
		expectGates  int
		expectQubits int
	}{
		{
			name: "simple_h_gate",
			source: `qreg q[2];
H q[0];`,
			expectError:  false,
			expectGates:  1,
			expectQubits: 2,
		},
		{
			name: "multiple_single_gates",
			source: `qreg q[4];
H q[0];
S q[1];
T q[2];`,
			expectError:  false,
			expectGates:  3,
			expectQubits: 4,
		},
		{
			name: "cx_gate",
			source: `qreg q[3];
CX q[0],q[1];`,
			expectError:  false,
			expectGates:  1,
			expectQubits: 3,
		},
		{
			name: "z_mask_gate",
			source: `qreg q[2];
Z[255] q[0];`,
			expectError:  false,
			expectGates:  1,
			expectQubits: 2,
		},
		{
			name: "complex_program",
			source: `// Quantum circuit with comments
qreg q[16];
H q[0];
S q[1];
T q[2];
CX q[0],q[1];
Z[128] q[3];
// More gates
H q[4];`,
			expectError:  false,
			expectGates:  6,
			expectQubits: 16,
		},
		{
			name: "missing_semicolon",
			source: `qreg q[2];
H q[0]`,
			expectError: true,
		},
		{
			name: "invalid_gate",
			source: `qreg q[2];
X q[0];`,
			expectError: true,
		},
		{
			name: "invalid_qubit_ref",
			source: `qreg q[2];
H q[];`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewQASMLiteParser(tt.source)
			program, err := parser.Parse()

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

			if len(program.Gates) != tt.expectGates {
				t.Errorf("Expected %d gates, got %d", tt.expectGates, len(program.Gates))
			}

			if program.QubitsUsed != tt.expectQubits {
				t.Errorf("Expected %d qubits, got %d", tt.expectQubits, program.QubitsUsed)
			}
		})
	}
}

func TestSerializeProgram(t *testing.T) {
	source := `qreg q[4];
H q[0];
S q[1];
T q[2];
CX q[0],q[1];
Z[255] q[3];`

	parser := NewQASMLiteParser(source)
	program, err := parser.Parse()
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}

	// Test serialization
	serialized := SerializeProgram(program)
	if len(serialized) == 0 {
		t.Error("Serialization produced empty result")
	}

	// Test that serialization is deterministic
	serialized2 := SerializeProgram(program)
	if !bytes.Equal(serialized, serialized2) {
		t.Error("Serialization is not deterministic")
	}

	t.Logf("Serialized program: %d bytes", len(serialized))
	t.Logf("Hex: %s", hex.EncodeToString(serialized))
}

func TestApplyZMask(t *testing.T) {
	source := `qreg q[4];
H q[0];
S q[1];
T q[2];
CX q[0],q[1];`

	parser := NewQASMLiteParser(source)
	program, err := parser.Parse()
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}

	// Apply Z-mask with a test seed
	seed := sha256.Sum256([]byte("test_seed"))
	maskedProgram := ApplyZMask(program, seed[:])

	// Check that Z-mask gates were added
	zMaskCount := 0
	for _, gate := range maskedProgram.Gates {
		if gate.Type == "Z" {
			zMaskCount++
		}
	}

	if zMaskCount == 0 {
		t.Error("No Z-mask gates were applied")
	}

	t.Logf("Applied %d Z-mask gates", zMaskCount)
}

func TestValidateTestVectors(t *testing.T) {
	err := ValidateTestVectors()
	if err != nil {
		t.Errorf("Test vector validation failed: %v", err)
	}
}

func TestGenerateTestVectors(t *testing.T) {
	vectors := GenerateTestVectors()
	if len(vectors) == 0 {
		t.Error("No test vectors generated")
	}

	for i, vector := range vectors {
		t.Logf("Test Vector %d: %s", i+1, vector.Name)
		t.Logf("  Seed: %s", hex.EncodeToString(vector.Seed))
		t.Logf("  QASM: %s", vector.QASM)
		t.Logf("  Expected bytes: %d", len(vector.ExpectedBytes))

		// Parse and serialize each test vector
		parser := NewQASMLiteParser(vector.QASM)
		program, err := parser.Parse()
		if err != nil {
			t.Errorf("Test vector %d parse error: %v", i+1, err)
			continue
		}

		// Apply Z-mask if seed is provided
		if len(vector.Seed) > 0 {
			program = ApplyZMask(program, vector.Seed)
		}

		serialized := SerializeProgram(program)
		if !bytes.Equal(serialized, vector.ExpectedBytes) {
			t.Errorf("Test vector %d serialization mismatch", i+1)
			t.Logf("  Expected: %s", hex.EncodeToString(vector.ExpectedBytes))
			t.Logf("  Got:      %s", hex.EncodeToString(serialized))
		}
	}
}

func TestGateTypes(t *testing.T) {
	tests := []struct {
		gate     string
		qubits   []int
		zmask    int
		expected string
	}{
		{"H", []int{0}, 0, "H"},
		{"S", []int{1}, 0, "S"},
		{"T", []int{2}, 0, "T"},
		{"CX", []int{0, 1}, 0, "CX"},
		{"Z", []int{3}, 255, "Z"},
	}

	for _, tt := range tests {
		t.Run(tt.gate, func(t *testing.T) {
			gate := QASMLiteGate{
				Type:   tt.gate,
				Qubits: tt.qubits,
				ZMask:  tt.zmask,
				Line:   1,
			}

			if gate.Type != tt.expected {
				t.Errorf("Expected gate type %s, got %s", tt.expected, gate.Type)
			}

			if len(gate.Qubits) != len(tt.qubits) {
				t.Errorf("Expected %d qubits, got %d", len(tt.qubits), len(gate.Qubits))
			}

			for i, qubit := range tt.qubits {
				if gate.Qubits[i] != qubit {
					t.Errorf("Expected qubit %d at index %d, got %d", qubit, i, gate.Qubits[i])
				}
			}

			if gate.ZMask != tt.zmask {
				t.Errorf("Expected Z-mask %d, got %d", tt.zmask, gate.ZMask)
			}
		})
	}
}

func TestBNFGrammarCompliance(t *testing.T) {
	// Test all supported BNF grammar constructs
	source := `// BNF Grammar compliance test
qreg q[16];

// Single qubit gates
H q[0];
S q[1];
T q[2];

// Two qubit gate
CX q[3],q[4];

// Z-mask gates with different values
Z[0] q[5];
Z[255] q[6];
Z[128] q[7];

// Comments should be ignored
// Another comment

// More gates
H q[8];
CX q[9],q[10];`

	parser := NewQASMLiteParser(source)
	program, err := parser.Parse()
	if err != nil {
		t.Fatalf("BNF compliance test failed: %v", err)
	}

	expectedGates := 9 // H, S, T, CX, Z[0], Z[255], Z[128], H, CX
	if len(program.Gates) != expectedGates {
		t.Errorf("Expected %d gates for BNF compliance, got %d", expectedGates, len(program.Gates))
	}

	// Verify gate types
	expectedTypes := []string{"H", "S", "T", "CX", "Z", "Z", "Z", "H", "CX"}
	for i, gate := range program.Gates {
		if gate.Type != expectedTypes[i] {
			t.Errorf("Gate %d: expected type %s, got %s", i, expectedTypes[i], gate.Type)
		}
	}

	t.Logf("BNF Grammar compliance test passed with %d gates", len(program.Gates))
}

func BenchmarkQASMLiteParsing(b *testing.B) {
	source := `qreg q[16];
H q[0];
S q[1];
T q[2];
CX q[0],q[1];
Z[255] q[3];
H q[4];
S q[5];
T q[6];
CX q[2],q[3];
Z[128] q[7];`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parser := NewQASMLiteParser(source)
		_, err := parser.Parse()
		if err != nil {
			b.Fatalf("Parse error: %v", err)
		}
	}
}

func BenchmarkProgramSerialization(b *testing.B) {
	source := `qreg q[16];
H q[0];
S q[1];
T q[2];
CX q[0],q[1];
Z[255] q[3];`

	parser := NewQASMLiteParser(source)
	program, err := parser.Parse()
	if err != nil {
		b.Fatalf("Parse error: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SerializeProgram(program)
	}
}
