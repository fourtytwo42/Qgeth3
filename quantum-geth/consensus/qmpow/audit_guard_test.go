// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
)

func TestNewAuditGuardRail(t *testing.T) {
	guard := NewAuditGuardRail()

	if guard == nil {
		t.Fatal("Expected non-nil audit guard rail")
	}

	if guard.verificationStatus != AuditPending {
		t.Errorf("Expected status AuditPending, got %v", guard.verificationStatus)
	}

	// Verify embedded hashes are set
	hashes := guard.GetEmbeddedHashes()
	if len(hashes) != 5 {
		t.Errorf("Expected 5 embedded hashes, got %d", len(hashes))
	}

	expectedHashes := []string{
		"ProofSystemHash",
		"TemplateAuditRoot",
		"GlideTableHash",
		"CanonicompSHA",
		"ChainIDHash",
	}

	for _, hashName := range expectedHashes {
		if hash, exists := hashes[hashName]; !exists {
			t.Errorf("Missing embedded hash: %s", hashName)
		} else if hash == (common.Hash{}) {
			t.Errorf("Embedded hash %s is zero", hashName)
		}
	}

	t.Log("✅ Audit guard rail created successfully with embedded hashes")
}

func TestVerifyAuditRoots(t *testing.T) {
	guard := NewAuditGuardRail()

	result, err := guard.VerifyAuditRoots()
	if err != nil {
		t.Fatalf("Audit verification failed: %v", err)
	}

	if result == nil {
		t.Fatal("Expected non-nil audit result")
	}

	if !result.OverallValid {
		t.Errorf("Expected overall audit to pass, got %v", result.OverallValid)
		t.Errorf("Error message: %s", result.ErrorMessage)
	}

	// Verify individual components
	if !result.ProofSystemValid {
		t.Error("ProofSystemHash verification failed")
	}
	if !result.TemplateAuditValid {
		t.Error("TemplateAuditRoot verification failed")
	}
	if !result.GlideTableValid {
		t.Error("GlideTableHash verification failed")
	}
	if !result.CanonicompValid {
		t.Error("CanonicompSHA verification failed")
	}
	if !result.ChainIDValid {
		t.Error("ChainIDHash verification failed")
	}

	// Verify timing
	if result.VerificationTime <= 0 {
		t.Error("Expected positive verification time")
	}

	// Verify status changed
	if guard.GetVerificationStatus() != AuditPassed {
		t.Errorf("Expected status AuditPassed, got %v", guard.GetVerificationStatus())
	}

	t.Logf("✅ Audit verification passed in %v", result.VerificationTime)
}

func TestVerifyProofSystemHash(t *testing.T) {
	guard := NewAuditGuardRail()

	// Test valid hash
	valid := guard.verifyProofSystemHash()
	if !valid {
		t.Error("ProofSystemHash verification should pass")
	}

	// Test with zero hash
	guard.proofSystemHash = common.Hash{}
	valid = guard.verifyProofSystemHash()
	if valid {
		t.Error("Zero ProofSystemHash should fail verification")
	}

	t.Log("✅ ProofSystemHash verification tests passed")
}

func TestVerifyTemplateAuditRoot(t *testing.T) {
	guard := NewAuditGuardRail()

	// Test valid root
	valid := guard.verifyTemplateAuditRoot()
	if !valid {
		t.Error("TemplateAuditRoot verification should pass")
	}

	// Test with zero root
	guard.templateAuditRoot = common.Hash{}
	valid = guard.verifyTemplateAuditRoot()
	if valid {
		t.Error("Zero TemplateAuditRoot should fail verification")
	}

	t.Log("✅ TemplateAuditRoot verification tests passed")
}

func TestVerifyTemplateConsistency(t *testing.T) {
	guard := NewAuditGuardRail()

	valid := guard.verifyTemplateConsistency()
	if !valid {
		t.Error("Template consistency check should pass")
	}

	t.Log("✅ Template consistency verification passed")
}

func TestVerifyGlideTableHash(t *testing.T) {
	guard := NewAuditGuardRail()

	// Test valid hash
	valid := guard.verifyGlideTableHash()
	if !valid {
		t.Error("GlideTableHash verification should pass")
	}

	// Test with zero hash
	guard.glideTableHash = common.Hash{}
	valid = guard.verifyGlideTableHash()
	if valid {
		t.Error("Zero GlideTableHash should fail verification")
	}

	t.Log("✅ GlideTableHash verification tests passed")
}

func TestVerifyGlideTableConsistency(t *testing.T) {
	guard := NewAuditGuardRail()

	valid := guard.verifyGlideTableConsistency()
	if !valid {
		t.Error("Glide table consistency check should pass")
	}

	t.Log("✅ Glide table consistency verification passed")
}

func TestVerifyCanonicompSHA(t *testing.T) {
	guard := NewAuditGuardRail()

	// Test valid SHA
	valid := guard.verifyCanonicompSHA()
	if !valid {
		t.Error("CanonicompSHA verification should pass")
	}

	// Test with zero SHA
	guard.canonicompSHA = common.Hash{}
	valid = guard.verifyCanonicompSHA()
	if valid {
		t.Error("Zero CanonicompSHA should fail verification")
	}

	t.Log("✅ CanonicompSHA verification tests passed")
}

func TestVerifyChainIDHash(t *testing.T) {
	guard := NewAuditGuardRail()

	// Test valid hash
	valid := guard.verifyChainIDHash()
	if !valid {
		t.Error("ChainIDHash verification should pass")
	}

	// Test with zero hash
	guard.chainIDHash = common.Hash{}
	valid = guard.verifyChainIDHash()
	if valid {
		t.Error("Zero ChainIDHash should fail verification")
	}

	t.Log("✅ ChainIDHash verification tests passed")
}

func TestIsOperationAllowed(t *testing.T) {
	guard := NewAuditGuardRail()

	// Initially should be false (pending)
	if guard.IsOperationAllowed() {
		t.Error("Operations should not be allowed before verification")
	}

	// After successful verification, should be true
	_, err := guard.VerifyAuditRoots()
	if err != nil {
		t.Fatalf("Audit verification failed: %v", err)
	}

	if !guard.IsOperationAllowed() {
		t.Error("Operations should be allowed after successful verification")
	}

	// After failed verification, should be false
	guard.verificationStatus = AuditFailed
	if guard.IsOperationAllowed() {
		t.Error("Operations should not be allowed after failed verification")
	}

	t.Log("✅ Operation allowance tests passed")
}

func TestForceVerification(t *testing.T) {
	guard := NewAuditGuardRail()

	// Set status to passed
	guard.verificationStatus = AuditPassed

	// Force verification should reset to pending and re-verify
	result, err := guard.ForceVerification()
	if err != nil {
		t.Fatalf("Force verification failed: %v", err)
	}

	if result == nil {
		t.Fatal("Expected non-nil result from force verification")
	}

	if !result.OverallValid {
		t.Error("Force verification should pass")
	}

	if guard.GetVerificationStatus() != AuditPassed {
		t.Error("Status should be AuditPassed after force verification")
	}

	t.Log("✅ Force verification tests passed")
}

func TestGetAuditStats(t *testing.T) {
	guard := NewAuditGuardRail()

	// Initial stats
	stats := guard.GetAuditStats()
	if stats.TotalVerifications != 0 {
		t.Errorf("Expected 0 total verifications, got %d", stats.TotalVerifications)
	}

	// After verification
	_, err := guard.VerifyAuditRoots()
	if err != nil {
		t.Fatalf("Audit verification failed: %v", err)
	}

	stats = guard.GetAuditStats()
	if stats.TotalVerifications != 1 {
		t.Errorf("Expected 1 total verification, got %d", stats.TotalVerifications)
	}

	if stats.PassedVerifications != 1 {
		t.Errorf("Expected 1 passed verification, got %d", stats.PassedVerifications)
	}

	if stats.FailedVerifications != 0 {
		t.Errorf("Expected 0 failed verifications, got %d", stats.FailedVerifications)
	}

	if stats.AverageVerifyTime <= 0 {
		t.Error("Expected positive average verification time")
	}

	t.Logf("✅ Audit stats: %d total, %d passed, avg time %v",
		stats.TotalVerifications, stats.PassedVerifications, stats.AverageVerifyTime)
}

func TestAuditResultStructure(t *testing.T) {
	guard := NewAuditGuardRail()

	result, err := guard.VerifyAuditRoots()
	if err != nil {
		t.Fatalf("Audit verification failed: %v", err)
	}

	// Verify result structure
	if result.Timestamp.IsZero() {
		t.Error("Expected non-zero timestamp")
	}

	if result.VerificationTime <= 0 {
		t.Error("Expected positive verification time")
	}

	if result.ErrorMessage != "" && result.OverallValid {
		t.Error("Should not have error message for successful verification")
	}

	// Verify boolean fields are set
	expectedFields := []bool{
		result.ProofSystemValid,
		result.TemplateAuditValid,
		result.GlideTableValid,
		result.CanonicompValid,
		result.ChainIDValid,
		result.OverallValid,
	}

	for i, field := range expectedFields {
		if !field {
			t.Errorf("Expected field %d to be true", i)
		}
	}

	t.Log("✅ Audit result structure validation passed")
}

func TestMultipleVerifications(t *testing.T) {
	guard := NewAuditGuardRail()

	// Run multiple verifications
	for i := 0; i < 5; i++ {
		result, err := guard.VerifyAuditRoots()
		if err != nil {
			t.Fatalf("Verification %d failed: %v", i, err)
		}

		if !result.OverallValid {
			t.Errorf("Verification %d should pass", i)
		}
	}

	// Check stats
	stats := guard.GetAuditStats()
	if stats.TotalVerifications != 5 {
		t.Errorf("Expected 5 total verifications, got %d", stats.TotalVerifications)
	}

	if stats.PassedVerifications != 5 {
		t.Errorf("Expected 5 passed verifications, got %d", stats.PassedVerifications)
	}

	t.Log("✅ Multiple verifications test passed")
}

func TestFailureScenarios(t *testing.T) {
	guard := NewAuditGuardRail()

	// Test with corrupted proof system hash
	originalHash := guard.proofSystemHash
	guard.proofSystemHash = common.HexToHash("0xdeadbeef")

	result, err := guard.VerifyAuditRoots()
	if err == nil {
		t.Error("Expected verification to fail with corrupted hash")
	}

	if result == nil {
		t.Fatal("Expected non-nil result even for failed verification")
	}

	if result.OverallValid {
		t.Error("Expected overall verification to fail")
	}

	if result.ProofSystemValid {
		t.Error("Expected ProofSystemHash verification to fail")
	}

	if guard.GetVerificationStatus() != AuditFailed {
		t.Error("Expected status to be AuditFailed")
	}

	// Restore original hash
	guard.proofSystemHash = originalHash

	// Test successful verification after failure
	result, err = guard.VerifyAuditRoots()
	if err != nil {
		t.Fatalf("Verification should pass after restoring hash: %v", err)
	}

	if !result.OverallValid {
		t.Error("Expected verification to pass after restoration")
	}

	t.Log("✅ Failure scenario tests passed")
}

func TestConcurrentVerifications(t *testing.T) {
	guard := NewAuditGuardRail()

	// Run concurrent verifications
	done := make(chan bool, 3)

	for i := 0; i < 3; i++ {
		go func(id int) {
			result, err := guard.VerifyAuditRoots()
			if err != nil {
				t.Errorf("Concurrent verification %d failed: %v", id, err)
			}
			if result != nil && !result.OverallValid {
				t.Errorf("Concurrent verification %d should pass", id)
			}
			done <- true
		}(i)
	}

	// Wait for all to complete
	for i := 0; i < 3; i++ {
		select {
		case <-done:
			// Success
		case <-time.After(5 * time.Second):
			t.Fatal("Concurrent verification timed out")
		}
	}

	// Check final stats
	stats := guard.GetAuditStats()
	if stats.TotalVerifications < 3 {
		t.Errorf("Expected at least 3 verifications, got %d", stats.TotalVerifications)
	}

	t.Log("✅ Concurrent verification tests passed")
}

// Benchmark tests
func BenchmarkVerifyAuditRoots(b *testing.B) {
	guard := NewAuditGuardRail()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := guard.VerifyAuditRoots()
		if err != nil {
			b.Fatalf("Verification failed: %v", err)
		}
	}
}

func BenchmarkVerifyProofSystemHash(b *testing.B) {
	guard := NewAuditGuardRail()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		guard.verifyProofSystemHash()
	}
}

func BenchmarkVerifyTemplateConsistency(b *testing.B) {
	guard := NewAuditGuardRail()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		guard.verifyTemplateConsistency()
	}
}
