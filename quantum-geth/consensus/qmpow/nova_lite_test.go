// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"fmt"
	"testing"
	"time"
)

// TestNovaLiteAggregatorCreation tests basic aggregator creation
func TestNovaLiteAggregatorCreation(t *testing.T) {
	aggregator := NewNovaLiteAggregator()

	if aggregator == nil {
		t.Fatal("Failed to create Nova-Lite aggregator")
	}

	if aggregator.GetName() != "NovaLiteAggregator_v1.0" {
		t.Errorf("Expected name 'NovaLiteAggregator_v1.0', got '%s'", aggregator.GetName())
	}

	if !aggregator.IsAvailable() {
		t.Error("Aggregator should be available")
	}

	stats := aggregator.GetStats()
	if stats.TotalAggregations != 0 {
		t.Errorf("Expected 0 total aggregations, got %d", stats.TotalAggregations)
	}

	t.Logf("✅ Nova-Lite aggregator created successfully: %s", aggregator.GetName())
}

// TestCAPSSProofGeneration tests generation of CAPSS proofs for testing
func TestCAPSSProofGeneration(t *testing.T) {
	prover := NewCAPSSProver()
	witness := NewMahadevWitness()

	// Generate a test trace
	seed := []byte("test_quantum_circuit_seed_data_32b")
	qasm := "qreg q[4]; creg c[4]; h q; measure q -> c;"
	trace, err := witness.GenerateTrace(123, seed, qasm, 0x1234)
	if err != nil {
		t.Fatalf("Failed to generate trace: %v", err)
	}

	// Generate CAPSS proof from trace
	proof, err := prover.GenerateProof(trace)
	if err != nil {
		t.Fatalf("Failed to generate CAPSS proof: %v", err)
	}

	// Validate proof properties
	if len(proof.Proof) != 2200 {
		t.Errorf("Expected proof size 2200 bytes, got %d", len(proof.Proof))
	}

	if proof.TraceID != trace.CircuitID {
		t.Errorf("Proof TraceID mismatch: expected %d, got %d", trace.CircuitID, proof.TraceID)
	}

	t.Logf("✅ CAPSS proof generated: %d bytes, trace %d", len(proof.Proof), proof.TraceID)
}

// TestNovaLiteAggregation tests the core aggregation functionality
func TestNovaLiteAggregation(t *testing.T) {
	aggregator := NewNovaLiteAggregator()
	prover := NewCAPSSProver()
	witness := NewMahadevWitness()

	// Generate 48 CAPSS proofs for testing
	capssProofs := make([]*CAPSSProof, 48)
	for i := 0; i < 48; i++ {
		seed := []byte(fmt.Sprintf("quantum_puzzle_%d_seed_32bytes", i))
		if len(seed) < 32 {
			seed = append(seed, make([]byte, 32-len(seed))...)
		}
		qasm := fmt.Sprintf("qreg q[4]; creg c[4]; h q[0]; cx q[0],q[%d]; measure q -> c;", i%4)
		trace, err := witness.GenerateTrace(uint32(i+100), seed[:32], qasm, uint16(i+0x1000))
		if err != nil {
			t.Fatalf("Failed to generate trace %d: %v", i, err)
		}

		proof, err := prover.GenerateProof(trace)
		if err != nil {
			t.Fatalf("Failed to generate CAPSS proof %d: %v", i, err)
		}

		capssProofs[i] = proof
	}

	// Perform Nova-Lite aggregation
	startTime := time.Now()
	proofRoot, err := aggregator.AggregateCAPSSProofs(capssProofs)
	aggregationTime := time.Since(startTime)

	if err != nil {
		t.Fatalf("Nova-Lite aggregation failed: %v", err)
	}

	// Validate aggregation results
	if proofRoot == nil {
		t.Fatal("ProofRoot is nil")
	}

	if len(proofRoot.NovaProofs) != 3 {
		t.Errorf("Expected 3 Nova-Lite proofs, got %d", len(proofRoot.NovaProofs))
	}

	if len(proofRoot.Root) != 32 {
		t.Errorf("Expected 32-byte Merkle root, got %d bytes", len(proofRoot.Root))
	}

	// Validate individual Nova-Lite proofs
	totalSize := 0
	for i, novaProof := range proofRoot.NovaProofs {
		if novaProof.Tier != 2 {
			t.Errorf("Nova proof %d: expected tier 2, got %d", i, novaProof.Tier)
		}

		if novaProof.BatchIndex != i {
			t.Errorf("Nova proof %d: expected batch index %d, got %d", i, i, novaProof.BatchIndex)
		}

		if len(novaProof.ProofData) == 0 {
			t.Errorf("Nova proof %d has empty proof data", i)
		}

		if len(novaProof.ProofData) > 6144 { // 6 kB limit
			t.Errorf("Nova proof %d exceeds 6 kB limit: %d bytes", i, len(novaProof.ProofData))
		}

		if novaProof.CAPSSCount != 16 {
			t.Errorf("Nova proof %d: expected 16 CAPSS proofs, got %d", i, novaProof.CAPSSCount)
		}

		totalSize += novaProof.Size
	}

	if proofRoot.TotalSize != totalSize {
		t.Errorf("ProofRoot total size mismatch: expected %d, got %d", totalSize, proofRoot.TotalSize)
	}

	// Check aggregation statistics
	stats := aggregator.GetStats()
	if stats.TotalAggregations != 1 {
		t.Errorf("Expected 1 total aggregation, got %d", stats.TotalAggregations)
	}

	if stats.SuccessfulAggs != 1 {
		t.Errorf("Expected 1 successful aggregation, got %d", stats.SuccessfulAggs)
	}

	if stats.TotalCAPSSProcessed != 48 {
		t.Errorf("Expected 48 CAPSS proofs processed, got %d", stats.TotalCAPSSProcessed)
	}

	t.Logf("✅ Nova-Lite aggregation successful:")
	t.Logf("   - 3 Nova-Lite proofs generated")
	t.Logf("   - Total size: %d bytes", proofRoot.TotalSize)
	t.Logf("   - Aggregation time: %v", aggregationTime)
	t.Logf("   - Average proof size: %.1f bytes", stats.AverageProofSize)
	t.Logf("   - Merkle root: %x", proofRoot.Root[:8])
}

// TestProofBatchCreation tests the batch creation functionality
func TestProofBatchCreation(t *testing.T) {
	aggregator := NewNovaLiteAggregator()
	prover := NewCAPSSProver()
	witness := NewMahadevWitness()

	// Generate 48 CAPSS proofs
	capssProofs := make([]*CAPSSProof, 48)
	for i := 0; i < 48; i++ {
		seed := []byte(fmt.Sprintf("batch_test_seed_%d_32bytes", i))
		if len(seed) < 32 {
			seed = append(seed, make([]byte, 32-len(seed))...)
		}
		qasm := fmt.Sprintf("qreg q[4]; creg c[4]; ry(0.%d) q[0]; measure q -> c;", i%10)
		trace, err := witness.GenerateTrace(uint32(i+200), seed[:32], qasm, uint16(i+0x2000))
		if err != nil {
			t.Fatalf("Failed to generate trace %d: %v", i, err)
		}

		proof, err := prover.GenerateProof(trace)
		if err != nil {
			t.Fatalf("Failed to generate CAPSS proof %d: %v", i, err)
		}

		capssProofs[i] = proof
	}

	// Test batch creation
	batches, err := aggregator.createProofBatches(capssProofs)
	if err != nil {
		t.Fatalf("Failed to create proof batches: %v", err)
	}

	if len(batches) != 3 {
		t.Errorf("Expected 3 batches, got %d", len(batches))
	}

	// Validate each batch
	for i, batch := range batches {
		if batch.BatchID != i {
			t.Errorf("Batch %d: expected ID %d, got %d", i, i, batch.BatchID)
		}

		if len(batch.CAPSSProofs) != 16 {
			t.Errorf("Batch %d: expected 16 CAPSS proofs, got %d", i, len(batch.CAPSSProofs))
		}

		if len(batch.MerkleRoot) != 32 {
			t.Errorf("Batch %d: expected 32-byte Merkle root, got %d bytes", i, len(batch.MerkleRoot))
		}

		if len(batch.ProofHashes) != 16 {
			t.Errorf("Batch %d: expected 16 proof hashes, got %d", i, len(batch.ProofHashes))
		}

		if batch.Size == 0 {
			t.Errorf("Batch %d has zero size", i)
		}

		t.Logf("✅ Batch %d: %d proofs, %d bytes, root %x",
			i, len(batch.CAPSSProofs), batch.Size, batch.MerkleRoot[:8])
	}
}

// TestNovaLiteProofGeneration tests individual Nova-Lite proof generation
func TestNovaLiteProofGeneration(t *testing.T) {
	aggregator := NewNovaLiteAggregator()
	prover := NewCAPSSProver()
	witness := NewMahadevWitness()

	// Create a single batch for testing
	capssProofs := make([]*CAPSSProof, 16)
	for i := 0; i < 16; i++ {
		seed := []byte(fmt.Sprintf("nova_test_seed_%d_32bytes", i))
		if len(seed) < 32 {
			seed = append(seed, make([]byte, 32-len(seed))...)
		}
		qasm := fmt.Sprintf("qreg q[4]; creg c[4]; rz(0.%d) q[0]; measure q -> c;", i%10)
		trace, err := witness.GenerateTrace(uint32(i+300), seed[:32], qasm, uint16(i+0x3000))
		if err != nil {
			t.Fatalf("Failed to generate trace %d: %v", i, err)
		}

		proof, err := prover.GenerateProof(trace)
		if err != nil {
			t.Fatalf("Failed to generate CAPSS proof %d: %v", i, err)
		}

		capssProofs[i] = proof
	}

	// Create batch
	batches, err := aggregator.createProofBatches(append(capssProofs, make([]*CAPSSProof, 32)...))
	if err != nil {
		t.Fatalf("Failed to create batches: %v", err)
	}

	// Test Nova-Lite proof generation for first batch
	batch := batches[0]
	novaProof, err := aggregator.generateNovaLiteProof(batch, 12345)
	if err != nil {
		t.Fatalf("Failed to generate Nova-Lite proof: %v", err)
	}

	// Validate Nova-Lite proof
	if novaProof.ProofID != 12345 {
		t.Errorf("Expected proof ID 12345, got %d", novaProof.ProofID)
	}

	if novaProof.Tier != 2 {
		t.Errorf("Expected tier 2, got %d", novaProof.Tier)
	}

	if novaProof.BatchIndex != 0 {
		t.Errorf("Expected batch index 0, got %d", novaProof.BatchIndex)
	}

	if len(novaProof.ProofData) == 0 {
		t.Error("Nova-Lite proof data is empty")
	}

	if len(novaProof.ProofData) > 6144 {
		t.Errorf("Nova-Lite proof exceeds 6 kB limit: %d bytes", len(novaProof.ProofData))
	}

	if len(novaProof.PublicInputs) == 0 {
		t.Error("Nova-Lite public inputs are empty")
	}

	if len(novaProof.ProofHash) != 32 {
		t.Errorf("Expected 32-byte proof hash, got %d bytes", len(novaProof.ProofHash))
	}

	if novaProof.CAPSSCount != 16 {
		t.Errorf("Expected 16 CAPSS proofs, got %d", novaProof.CAPSSCount)
	}

	if novaProof.Size != len(novaProof.ProofData) {
		t.Errorf("Size mismatch: expected %d, got %d", len(novaProof.ProofData), novaProof.Size)
	}

	t.Logf("✅ Nova-Lite proof generated successfully:")
	t.Logf("   - Proof ID: %d", novaProof.ProofID)
	t.Logf("   - Size: %d bytes", novaProof.Size)
	t.Logf("   - CAPSS count: %d", novaProof.CAPSSCount)
	t.Logf("   - Proof hash: %x", novaProof.ProofHash[:8])
}

// TestProofRootComputation tests Merkle root computation
func TestProofRootComputation(t *testing.T) {
	aggregator := NewNovaLiteAggregator()

	// Create 3 mock Nova-Lite proofs
	novaProofs := make([]*NovaLiteProof, 3)
	for i := 0; i < 3; i++ {
		novaProofs[i] = &NovaLiteProof{
			ProofID:      uint32(i + 1000),
			Tier:         2,
			BatchIndex:   i,
			ProofData:    make([]byte, 5000+i*200), // Variable sizes
			PublicInputs: make([]byte, 64),
			ProofHash:    make([]byte, 32),
			CAPSSCount:   16,
			GeneratedAt:  time.Now(),
			Size:         5000 + i*200,
		}

		// Fill with deterministic data
		for j := range novaProofs[i].ProofData {
			novaProofs[i].ProofData[j] = byte((i*1000 + j) % 256)
		}
		for j := range novaProofs[i].ProofHash {
			novaProofs[i].ProofHash[j] = byte((i*100 + j) % 256)
		}
	}

	// Compute proof root
	proofRoot, err := aggregator.computeProofRoot(novaProofs)
	if err != nil {
		t.Fatalf("Failed to compute proof root: %v", err)
	}

	// Validate proof root
	if len(proofRoot.Root) != 32 {
		t.Errorf("Expected 32-byte root, got %d bytes", len(proofRoot.Root))
	}

	if len(proofRoot.NovaProofs) != 3 {
		t.Errorf("Expected 3 Nova proofs, got %d", len(proofRoot.NovaProofs))
	}

	expectedTotalSize := 5000 + 5200 + 5400 // Sum of individual sizes
	if proofRoot.TotalSize != expectedTotalSize {
		t.Errorf("Expected total size %d, got %d", expectedTotalSize, proofRoot.TotalSize)
	}

	// Test determinism - compute again and verify same root
	proofRoot2, err := aggregator.computeProofRoot(novaProofs)
	if err != nil {
		t.Fatalf("Failed to compute proof root second time: %v", err)
	}

	if string(proofRoot.Root) != string(proofRoot2.Root) {
		t.Error("Proof root computation is not deterministic")
	}

	t.Logf("✅ Proof root computed successfully:")
	t.Logf("   - Root: %x", proofRoot.Root[:8])
	t.Logf("   - Total size: %d bytes", proofRoot.TotalSize)
	t.Logf("   - Nova proofs: %d", len(proofRoot.NovaProofs))
}

// TestErrorHandling tests various error conditions
func TestErrorHandling(t *testing.T) {
	aggregator := NewNovaLiteAggregator()

	// Test with wrong number of CAPSS proofs
	_, err := aggregator.AggregateCAPSSProofs(make([]*CAPSSProof, 47))
	if err == nil {
		t.Error("Expected error for 47 CAPSS proofs, got nil")
	}

	_, err = aggregator.AggregateCAPSSProofs(make([]*CAPSSProof, 49))
	if err == nil {
		t.Error("Expected error for 49 CAPSS proofs, got nil")
	}

	// Test with nil proofs
	_, err = aggregator.AggregateCAPSSProofs(nil)
	if err == nil {
		t.Error("Expected error for nil CAPSS proofs, got nil")
	}

	// Test proof root with wrong number of Nova proofs
	_, err = aggregator.computeProofRoot(make([]*NovaLiteProof, 2))
	if err == nil {
		t.Error("Expected error for 2 Nova proofs, got nil")
	}

	_, err = aggregator.computeProofRoot(make([]*NovaLiteProof, 4))
	if err == nil {
		t.Error("Expected error for 4 Nova proofs, got nil")
	}

	t.Logf("✅ Error handling tests passed")
}

// TestStreamingAPI tests the streaming functionality
func TestStreamingAPI(t *testing.T) {
	aggregator := NewNovaLiteAggregator()
	streamingAPI := NewStreamingAPI(aggregator)

	// Create a mock proof root
	novaProofs := make([]*NovaLiteProof, 3)
	for i := 0; i < 3; i++ {
		novaProofs[i] = &NovaLiteProof{
			ProofID:      uint32(i + 2000),
			Tier:         2,
			BatchIndex:   i,
			ProofData:    make([]byte, 4000), // 4 kB each
			PublicInputs: make([]byte, 64),
			ProofHash:    make([]byte, 32),
			CAPSSCount:   16,
			Size:         4000,
		}
	}

	proofRoot := &ProofRoot{
		Root:        make([]byte, 32),
		NovaProofs:  novaProofs,
		TotalSize:   12000,
		GeneratedAt: time.Now(),
	}

	// Test streaming with 1 kB chunks
	chunks, err := streamingAPI.StreamProofRoot(proofRoot, 1024)
	if err != nil {
		t.Fatalf("Streaming failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("No chunks returned from streaming")
	}

	// Verify total size
	totalBytes := 0
	for _, chunk := range chunks {
		totalBytes += len(chunk)
	}

	// Should include root (32 bytes) + 3 proofs (4000 each) = 12032 bytes
	expectedBytes := 32 + 12000
	if totalBytes != expectedBytes {
		t.Errorf("Expected %d bytes streamed, got %d", expectedBytes, totalBytes)
	}

	// Check streaming stats
	stats := streamingAPI.GetStreamingStats()
	if stats.TotalStreams != 1 {
		t.Errorf("Expected 1 total stream, got %d", stats.TotalStreams)
	}

	if stats.SuccessfulStreams != 1 {
		t.Errorf("Expected 1 successful stream, got %d", stats.SuccessfulStreams)
	}

	if stats.BytesStreamed != int64(totalBytes) {
		t.Errorf("Expected %d bytes streamed, got %d", totalBytes, stats.BytesStreamed)
	}

	t.Logf("✅ Streaming API test passed:")
	t.Logf("   - Chunks: %d", len(chunks))
	t.Logf("   - Total bytes: %d", totalBytes)
	t.Logf("   - Successful streams: %d", stats.SuccessfulStreams)
}

// TestCompressionRatio tests compression ratio estimation
func TestCompressionRatio(t *testing.T) {
	// Create mock Nova-Lite proofs
	novaProofs := make([]*NovaLiteProof, 3)
	for i := 0; i < 3; i++ {
		novaProofs[i] = &NovaLiteProof{
			Size: 5500 + i*100, // Variable sizes around 5.5 kB
		}
	}

	// Test compression ratio calculation
	ratio := EstimateCompressionRatio(48, novaProofs)

	// Expected calculation:
	// CAPSS total: 48 * 2200 = 105,600 bytes
	// Nova total: 5500 + 5600 + 5700 = 16,800 bytes
	// Ratio: 105,600 / 16,800 = 6.286
	expectedRatio := 105600.0 / 16800.0

	if ratio < expectedRatio*0.95 || ratio > expectedRatio*1.05 {
		t.Errorf("Expected compression ratio ~%.2f, got %.2f", expectedRatio, ratio)
	}

	t.Logf("✅ Compression ratio test passed:")
	t.Logf("   - 48 CAPSS proofs: %d bytes", 48*2200)
	t.Logf("   - 3 Nova proofs: %d bytes", 16800)
	t.Logf("   - Compression ratio: %.2fx", ratio)
}

// TestDeterminism tests that aggregation is deterministic
func TestDeterminism(t *testing.T) {
	prover := NewCAPSSProver()
	witness := NewMahadevWitness()

	// Generate the same set of CAPSS proofs twice
	capssProofs1 := make([]*CAPSSProof, 48)
	capssProofs2 := make([]*CAPSSProof, 48)

	for i := 0; i < 48; i++ {
		seed := []byte(fmt.Sprintf("determinism_test_%d_32bytes", i))
		if len(seed) < 32 {
			seed = append(seed, make([]byte, 32-len(seed))...)
		}
		qasm := fmt.Sprintf("qreg q[4]; creg c[4]; h q[0]; cx q[0],q[%d]; measure q -> c;", i%4)

		trace1, err := witness.GenerateTrace(uint32(i+400), seed[:32], qasm, uint16(i+0x4000))
		if err != nil {
			t.Fatalf("Failed to generate trace1 %d: %v", i, err)
		}

		trace2, err := witness.GenerateTrace(uint32(i+400), seed[:32], qasm, uint16(i+0x4000))
		if err != nil {
			t.Fatalf("Failed to generate trace2 %d: %v", i, err)
		}

		proof1, err := prover.GenerateProof(trace1)
		if err != nil {
			t.Fatalf("Failed to generate proof1 %d: %v", i, err)
		}

		proof2, err := prover.GenerateProof(trace2)
		if err != nil {
			t.Fatalf("Failed to generate proof2 %d: %v", i, err)
		}

		capssProofs1[i] = proof1
		capssProofs2[i] = proof2
	}

	// Aggregate both sets
	aggregator1 := NewNovaLiteAggregator()
	aggregator2 := NewNovaLiteAggregator()

	proofRoot1, err := aggregator1.AggregateCAPSSProofs(capssProofs1)
	if err != nil {
		t.Fatalf("First aggregation failed: %v", err)
	}

	proofRoot2, err := aggregator2.AggregateCAPSSProofs(capssProofs2)
	if err != nil {
		t.Fatalf("Second aggregation failed: %v", err)
	}

	// Compare results
	if string(proofRoot1.Root) != string(proofRoot2.Root) {
		t.Error("Aggregation is not deterministic - different roots")
	}

	if len(proofRoot1.NovaProofs) != len(proofRoot2.NovaProofs) {
		t.Error("Aggregation is not deterministic - different proof counts")
	}

	for i := 0; i < len(proofRoot1.NovaProofs); i++ {
		if string(proofRoot1.NovaProofs[i].ProofData) != string(proofRoot2.NovaProofs[i].ProofData) {
			t.Errorf("Nova proof %d is not deterministic", i)
		}
	}

	t.Logf("✅ Determinism test passed - identical results:")
	t.Logf("   - Root 1: %x", proofRoot1.Root[:8])
	t.Logf("   - Root 2: %x", proofRoot2.Root[:8])
}

// TestIntegrationWithPuzzleOrchestrator tests integration with the puzzle orchestrator
func TestIntegrationWithPuzzleOrchestrator(t *testing.T) {
	// This test would integrate with the puzzle orchestrator from Task 5
	// For now, we'll test the interface compatibility

	aggregator := NewNovaLiteAggregator()

	// Verify the aggregator can be used in the mining pipeline
	if !aggregator.IsAvailable() {
		t.Error("Aggregator should be available for mining pipeline")
	}

	stats := aggregator.GetStats()
	if stats.TotalAggregations < 0 {
		t.Error("Stats should be initialized properly")
	}

	t.Logf("✅ Integration compatibility verified")
	t.Logf("   - Aggregator available: %v", aggregator.IsAvailable())
	t.Logf("   - Name: %s", aggregator.GetName())
}
