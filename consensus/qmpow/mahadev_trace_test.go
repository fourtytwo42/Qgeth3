package qmpow

import (
	"testing"
)

func TestMahadevWitness_Basic(t *testing.T) {
	witness := NewMahadevWitness()

	if witness.GetName() != "MahadevWitness_v1.0" {
		t.Errorf("Expected name 'MahadevWitness_v1.0', got '%s'", witness.GetName())
	}

	if !witness.IsAvailable() {
		t.Error("Mahadev witness should be available")
	}

	stats := witness.GetStats()
	if stats.TotalTraces != 0 {
		t.Errorf("Expected 0 initial traces, got %d", stats.TotalTraces)
	}
}

func TestMahadevWitness_GenerateTrace(t *testing.T) {
	witness := NewMahadevWitness()

	qasm := "H q[0]; T q[1]; CX q[0],q[1];"
	seed := []byte("test_seed_mahadev_123456789012345")
	circuitID := uint32(42)
	outcome := uint16(0x1234)

	trace, err := witness.GenerateTrace(circuitID, seed, qasm, outcome)
	if err != nil {
		t.Fatalf("GenerateTrace failed: %v", err)
	}

	if trace.CircuitID != circuitID {
		t.Errorf("Expected circuit ID %d, got %d", circuitID, trace.CircuitID)
	}

	if trace.Outcome != outcome {
		t.Errorf("Expected outcome 0x%04x, got 0x%04x", outcome, trace.Outcome)
	}

	if len(trace.Transcript) == 0 {
		t.Error("Transcript should not be empty")
	}

	stats := witness.GetStats()
	if stats.TotalTraces != 1 {
		t.Errorf("Expected 1 total trace, got %d", stats.TotalTraces)
	}

	t.Logf("Trace generated: size=%d bytes", trace.Size)
}

func TestCAPSSProver_Basic(t *testing.T) {
	prover := NewCAPSSProver()

	if prover.GetName() != "CAPSSProver_v1.0" {
		t.Errorf("Expected name 'CAPSSProver_v1.0', got '%s'", prover.GetName())
	}

	if !prover.IsAvailable() {
		t.Error("CAPSS prover should be available")
	}

	stats := prover.GetStats()
	if stats.TotalProofs != 0 {
		t.Errorf("Expected 0 initial proofs, got %d", stats.TotalProofs)
	}
}

func TestCAPSSProver_GenerateProof(t *testing.T) {
	witness := NewMahadevWitness()
	prover := NewCAPSSProver()

	seed := []byte("capss_test_seed_1234567890123456")
	trace, err := witness.GenerateTrace(200, seed, "H q[0];", 0x5678)
	if err != nil {
		t.Fatalf("GenerateTrace failed: %v", err)
	}

	proof, err := prover.GenerateProof(trace)
	if err != nil {
		t.Fatalf("GenerateProof failed: %v", err)
	}

	if proof.TraceID != trace.CircuitID {
		t.Errorf("Expected trace ID %d, got %d", trace.CircuitID, proof.TraceID)
	}

	if len(proof.Proof) != 2200 {
		t.Errorf("Expected proof size 2200 bytes, got %d", len(proof.Proof))
	}

	t.Logf("CAPSS proof generated: size=%d bytes", len(proof.Proof))
}

func TestCAPSSProver_VerifyProof(t *testing.T) {
	witness := NewMahadevWitness()
	prover := NewCAPSSProver()

	seed := []byte("verify_test_seed_123456789012345")
	trace, err := witness.GenerateTrace(300, seed, "H q[0]; T q[1];", 0x9ABC)
	if err != nil {
		t.Fatalf("GenerateTrace failed: %v", err)
	}

	proof, err := prover.GenerateProof(trace)
	if err != nil {
		t.Fatalf("GenerateProof failed: %v", err)
	}

	valid, err := prover.VerifyProof(proof, trace)
	if err != nil {
		t.Fatalf("VerifyProof failed: %v", err)
	}

	if !valid {
		t.Error("Proof should be valid")
	}

	t.Logf("CAPSS proof verified in %d microseconds", proof.VerifyTime)
}
