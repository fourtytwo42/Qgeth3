package qmpow

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/ethereum/go-ethereum/common"
)

// TestQuantumCircuitCanonicalizer_CanonicalizeCircuit tests the main canonicalization function
func TestQuantumCircuitCanonicalizer_CanonicalizeCircuit(t *testing.T) {
	canonicalizer := NewQuantumCircuitCanonicalizer()

	tests := []struct {
		name          string
		rawCircuit    []byte
		qubits        int
		expectValid   bool
		expectError   string
		description   string
	}{
		{
			name:          "Valid simple circuit",
			rawCircuit:    createSimpleCircuit(t),
			qubits:        2,
			expectValid:   true,
			expectError:   "",
			description:   "Simple circuit with H and CNOT gates should canonicalize successfully",
		},
		{
			name:          "Circuit with T-gates",
			rawCircuit:    createTGateCircuit(t),
			qubits:        2,
			expectValid:   true,
			expectError:   "",
			description:   "Circuit with T-gates should canonicalize and count T-gates correctly",
		},
		{
			name:          "Circuit with rotations",
			rawCircuit:    createRotationCircuit(t),
			qubits:        3,
			expectValid:   true,
			expectError:   "",
			description:   "Circuit with RY and RZ rotations should canonicalize successfully",
		},
		{
			name:          "Empty circuit data",
			rawCircuit:    []byte{},
			qubits:        2,
			expectValid:   false,
			expectError:   "insufficient circuit data",
			description:   "Empty circuit data should be rejected",
		},
		{
			name:          "Invalid qubit index",
			rawCircuit:    createInvalidQubitCircuit(t),
			qubits:        2,
			expectValid:   false,
			expectError:   "invalid qubit index",
			description:   "Circuit with invalid qubit indices should be rejected",
		},
		{
			name:          "Complex optimization circuit",
			rawCircuit:    createOptimizationCircuit(t),
			qubits:        3,
			expectValid:   true,
			expectError:   "",
			description:   "Circuit with canceling gates should be optimized during canonicalization",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			circuit, err := canonicalizer.CanonicalizeCircuit(tt.rawCircuit, tt.qubits)

			if tt.expectValid {
				if err != nil {
					t.Errorf("Expected successful canonicalization but got error: %v", err)
				}
				if circuit == nil {
					t.Errorf("Expected valid circuit but got nil")
				}
				if circuit != nil {
					// Verify circuit properties
					if circuit.Qubits != tt.qubits {
						t.Errorf("Expected %d qubits, got %d", tt.qubits, circuit.Qubits)
					}
					if circuit.CircuitHash == (common.Hash{}) {
						t.Errorf("Expected non-zero circuit hash")
					}
					if circuit.QASMString == "" {
						t.Errorf("Expected non-empty QASM string")
					}
				}
			} else {
				if err == nil {
					t.Errorf("Expected error but canonicalization succeeded")
				}
				if tt.expectError != "" && err != nil {
					if !containsSubstring(err.Error(), tt.expectError) {
						t.Errorf("Expected error containing '%s' but got: %v", tt.expectError, err)
					}
				}
			}

			t.Logf("Test case: %s - %s", tt.name, tt.description)
			if circuit != nil {
				t.Logf("Result: gates=%d, t_gates=%d, cx_gates=%d, depth=%d",
					circuit.TotalGates, circuit.TGateCount, circuit.CXGateCount, circuit.Depth)
			}
		})
	}
}

// TestQuantumCircuitCanonicalizer_StandardGateSet tests the standard gate set definition
func TestQuantumCircuitCanonicalizer_StandardGateSet(t *testing.T) {
	gateSet := StandardGateSet()

	expectedGates := map[StandardQuantumGate]bool{
		GateI: true, GateX: true, GateY: true, GateZ: true, GateH: true,
		GateS: true, GateT: true, GateSdg: true, GateTdg: true,
		GateCX: true, GateCZ: true, GateRX: true, GateRY: true, GateRZ: true,
	}

	if len(gateSet) != len(expectedGates) {
		t.Errorf("Expected %d gates in standard set, got %d", len(expectedGates), len(gateSet))
	}

	for _, gate := range gateSet {
		if !expectedGates[gate] {
			t.Errorf("Unexpected gate in standard set: %s", gate)
		}
	}

	t.Logf("Standard gate set verified: %d gates", len(gateSet))
}

// TestQuantumCircuitCanonicalizer_DeterministicHashing tests deterministic circuit hashing
func TestQuantumCircuitCanonicalizer_DeterministicHashing(t *testing.T) {
	canonicalizer := NewQuantumCircuitCanonicalizer()

	rawCircuit := createStandardCircuit(t)

	// Canonicalize the same circuit multiple times
	var hashes []common.Hash
	for i := 0; i < 5; i++ {
		circuit, err := canonicalizer.CanonicalizeCircuit(rawCircuit, 2)
		if err != nil {
			t.Fatalf("Canonicalization %d failed: %v", i, err)
		}
		hashes = append(hashes, circuit.CircuitHash)
	}

	// All hashes should be identical (deterministic)
	for i := 1; i < len(hashes); i++ {
		if hashes[i] != hashes[0] {
			t.Errorf("Circuit hash %d differs from hash 0: %s vs %s", 
				i, hashes[i].Hex(), hashes[0].Hex())
		}
	}

	t.Logf("Deterministic hashing test passed: all 5 hashes identical: %s", 
		hashes[0].Hex()[:10]+"...")
}

// Helper functions for creating test circuits

func createSimpleCircuit(t *testing.T) []byte {
	// H q[0]; CNOT q[0],q[1];
	return []byte{0, 0, 0, 1, 0, 1} // H on q0, CNOT q0->q1
}

func createTGateCircuit(t *testing.T) []byte {
	// T q[0]; T q[1];
	return []byte{2, 0, 0, 2, 1, 0} // T on q0, T on q1
}

func createRotationCircuit(t *testing.T) []byte {
	// RY(π/4) q[0]; RZ(π/2) q[1];
	angle1 := math.Pi / 4
	angle2 := math.Pi / 2
	
	circuit := []byte{4, 0, 0} // RY on q0
	angle1Bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(angle1Bytes, math.Float64bits(angle1))
	circuit = append(circuit, angle1Bytes...)
	
	circuit = append(circuit, []byte{5, 1, 0}...) // RZ on q1
	angle2Bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(angle2Bytes, math.Float64bits(angle2))
	circuit = append(circuit, angle2Bytes...)
	
	return circuit
}

func createInvalidQubitCircuit(t *testing.T) []byte {
	// H q[5]; (invalid for 2-qubit system) - need minimum 3 bytes + padding
	return []byte{0, 5, 0, 0} // Add padding to meet minimum length requirement
}

func createOptimizationCircuit(t *testing.T) []byte {
	// H q[0]; H q[0]; (canceling gates) - these should be optimized away
	// But need to ensure we have valid qubits for a 3-qubit system
	return []byte{0, 0, 0, 0, 0, 0} // H-H on q0 (should cancel)
}

func createStandardCircuit(t *testing.T) []byte {
	// H q[0]; CNOT q[0],q[1]; T q[0];
	return []byte{0, 0, 0, 1, 0, 1, 2, 0, 0}
}

// Helper function to check if string contains substring
func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
