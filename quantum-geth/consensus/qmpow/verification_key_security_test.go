package qmpow

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestVerificationKeySecuritySystem_Creation(t *testing.T) {
	// Create temporary directory for testing
	tempDir, err := os.MkdirTemp("", "vkey_security_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test configuration
	config := &VKeySecurityConfig{
		StorageDir:                 tempDir,
		FilePermissions:            0600,
		DefaultKeyExpiry:           24 * time.Hour,
		RotationInterval:           1 * time.Hour,
		AutoRotationEnabled:        false, // Disable for testing
		EncryptionKeySize:          32,
		RequireHardwareAttestation: false,
		IntegrityCheckInterval:     1 * time.Hour,
		MultiSigEnabled:            false, // Disable for basic testing
		MultiSigThreshold:          2,
		BackupEnabled:              false, // Disable for basic testing
		BackupInterval:             1 * time.Hour,
		ThresholdShares:            5,
		MinRecoveryShares:          3,
		CompromiseDetectionEnabled: false, // Disable for basic testing
		UsageAnomalyThreshold:      5.0,
		MaxFailureRate:             0.05,
	}

	// Create verification key security system
	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Verify initialization
	if vkss.storageDir != tempDir {
		t.Errorf("Expected storage dir %s, got %s", tempDir, vkss.storageDir)
	}

	if vkss.config.EncryptionKeySize != 32 {
		t.Errorf("Expected encryption key size 32, got %d", vkss.config.EncryptionKeySize)
	}

	// Verify directories were created
	if _, err := os.Stat(tempDir); os.IsNotExist(err) {
		t.Errorf("Storage directory was not created")
	}

	t.Logf("✅ Verification Key Security System created successfully")
}

func TestVerificationKeySecuritySystem_KeyGeneration(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "vkey_generation_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := &VKeySecurityConfig{
		StorageDir:                 tempDir,
		FilePermissions:            0600,
		DefaultKeyExpiry:           24 * time.Hour,
		EncryptionKeySize:          32,
		MultiSigEnabled:            false,
		BackupEnabled:              false,
		AutoRotationEnabled:        false,
		CompromiseDetectionEnabled: false,
	}

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Generate test verification key data
	testKeyData := make([]byte, 256)
	if _, err := rand.Read(testKeyData); err != nil {
		t.Fatalf("Failed to generate test key data: %v", err)
	}

	// Create hardware attestation (optional)
	attestation := &KeyAttestation{
		HardwareID:       "test_hardware_001",
		AttestationData:  []byte("test_attestation_data"),
		Timestamp:        time.Now(),
		AttestationType:  "TEST",
		AttestationProof: []byte("test_proof"),
	}

	// Test key generation
	secureKey, err := vkss.GenerateSecureVerificationKey(
		VKeyTypeCAPSS,
		"test_circuit_family",
		testKeyData,
		attestation,
	)
	if err != nil {
		t.Fatalf("Failed to generate secure verification key: %v", err)
	}

	// Verify key properties
	if secureKey.KeyType != VKeyTypeCAPSS {
		t.Errorf("Expected key type %s, got %s", VKeyTypeCAPSS, secureKey.KeyType)
	}

	if secureKey.CircuitFamily != "test_circuit_family" {
		t.Errorf("Expected circuit family 'test_circuit_family', got '%s'", secureKey.CircuitFamily)
	}

	if secureKey.Version != 1 {
		t.Errorf("Expected version 1, got %d", secureKey.Version)
	}

	if len(secureKey.KeyData) == 0 {
		t.Errorf("Expected non-empty encrypted key data")
	}

	if len(secureKey.Nonce) == 0 {
		t.Errorf("Expected non-empty nonce")
	}

	if len(secureKey.AuthTag) == 0 {
		t.Errorf("Expected non-empty auth tag")
	}

	if len(secureKey.Signature) == 0 {
		t.Errorf("Expected non-empty signature")
	}

	if secureKey.Attestation == nil {
		t.Errorf("Expected attestation to be preserved")
	}

	if secureKey.Attestation.HardwareID != "test_hardware_001" {
		t.Errorf("Expected hardware ID 'test_hardware_001', got '%s'", secureKey.Attestation.HardwareID)
	}

	// Verify key files were created
	keyFilePath := filepath.Join(tempDir, secureKey.ID+".key")
	if _, err := os.Stat(keyFilePath); os.IsNotExist(err) {
		t.Errorf("Key file was not created: %s", keyFilePath)
	}

	encKeyPath := filepath.Join(tempDir, secureKey.ID+".enc")
	if _, err := os.Stat(encKeyPath); os.IsNotExist(err) {
		t.Errorf("Encryption key file was not created: %s", encKeyPath)
	}

	// Verify statistics
	stats := vkss.GetVerificationKeySecurityStats()
	if stats.TotalKeys != 1 {
		t.Errorf("Expected 1 total key, got %d", stats.TotalKeys)
	}

	if stats.ActiveKeys != 1 {
		t.Errorf("Expected 1 active key, got %d", stats.ActiveKeys)
	}

	t.Logf("✅ Secure verification key generated successfully: %s", secureKey.ID)
}

func TestVerificationKeySecuritySystem_KeyRetrieval(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "vkey_retrieval_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := &VKeySecurityConfig{
		StorageDir:                 tempDir,
		FilePermissions:            0600,
		DefaultKeyExpiry:           24 * time.Hour,
		EncryptionKeySize:          32,
		MultiSigEnabled:            false,
		BackupEnabled:              false,
		AutoRotationEnabled:        false,
		CompromiseDetectionEnabled: false,
	}

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Generate test key data
	originalKeyData := make([]byte, 256)
	if _, err := rand.Read(originalKeyData); err != nil {
		t.Fatalf("Failed to generate test key data: %v", err)
	}

	// Generate a secure verification key
	secureKey, err := vkss.GenerateSecureVerificationKey(
		VKeyTypeNovaLite,
		"nova_circuit_family",
		originalKeyData,
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to generate secure verification key: %v", err)
	}

	// Test key retrieval
	retrievedKey, decryptedData, err := vkss.GetSecureVerificationKey(secureKey.ID)
	if err != nil {
		t.Fatalf("Failed to retrieve verification key: %v", err)
	}

	// Verify retrieved key matches original
	if retrievedKey.ID != secureKey.ID {
		t.Errorf("Expected key ID %s, got %s", secureKey.ID, retrievedKey.ID)
	}

	if retrievedKey.KeyType != VKeyTypeNovaLite {
		t.Errorf("Expected key type %s, got %s", VKeyTypeNovaLite, retrievedKey.KeyType)
	}

	// Verify decrypted data matches original
	if !bytes.Equal(decryptedData, originalKeyData) {
		t.Errorf("Decrypted key data does not match original")
	}

	// Verify usage statistics were updated
	if retrievedKey.UsageCount != 1 {
		t.Errorf("Expected usage count 1, got %d", retrievedKey.UsageCount)
	}

	if retrievedKey.LastUsed.IsZero() {
		t.Errorf("Expected LastUsed to be updated")
	}

	// Test retrieval of non-existent key
	_, _, err = vkss.GetSecureVerificationKey("non_existent_key")
	if err == nil {
		t.Errorf("Expected error when retrieving non-existent key")
	}

	t.Logf("✅ Verification key retrieved and decrypted successfully")
}

func TestVerificationKeySecuritySystem_KeyTypes(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "vkey_types_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultVKeySecurityConfig()
	config.StorageDir = tempDir
	config.MultiSigEnabled = false
	config.BackupEnabled = false
	config.AutoRotationEnabled = false
	config.CompromiseDetectionEnabled = false

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Test all key types
	keyTypes := []VKeyType{
		VKeyTypeCAPSS,
		VKeyTypeNovaLite,
		VKeyTypeFinalNova,
		VKeyTypeQuantumGates,
		VKeyTypeMeasurement,
		VKeyTypeEntanglement,
		VKeyTypeCircuitDepth,
		VKeyTypeCanonical,
	}

	testKeyData := make([]byte, 128)
	if _, err := rand.Read(testKeyData); err != nil {
		t.Fatalf("Failed to generate test key data: %v", err)
	}

	generatedKeys := make([]*SecureVKey, 0, len(keyTypes))

	// Generate keys of each type
	for _, keyType := range keyTypes {
		circuitFamily := string(keyType) + "_family"
		
		secureKey, err := vkss.GenerateSecureVerificationKey(
			keyType,
			circuitFamily,
			testKeyData,
			nil,
		)
		if err != nil {
			t.Fatalf("Failed to generate %s key: %v", keyType, err)
		}

		if secureKey.KeyType != keyType {
			t.Errorf("Expected key type %s, got %s", keyType, secureKey.KeyType)
		}

		if secureKey.CircuitFamily != circuitFamily {
			t.Errorf("Expected circuit family %s, got %s", circuitFamily, secureKey.CircuitFamily)
		}

		generatedKeys = append(generatedKeys, secureKey)
		t.Logf("Generated %s key: %s", keyType, secureKey.ID)
	}

	// Verify all keys can be retrieved
	for _, originalKey := range generatedKeys {
		retrievedKey, decryptedData, err := vkss.GetSecureVerificationKey(originalKey.ID)
		if err != nil {
			t.Fatalf("Failed to retrieve %s key: %v", originalKey.KeyType, err)
		}

		if retrievedKey.KeyType != originalKey.KeyType {
			t.Errorf("Retrieved key type mismatch: expected %s, got %s", 
				originalKey.KeyType, retrievedKey.KeyType)
		}

		if !bytes.Equal(decryptedData, testKeyData) {
			t.Errorf("Decrypted data mismatch for %s key", originalKey.KeyType)
		}
	}

	// Verify statistics
	stats := vkss.GetVerificationKeySecurityStats()
	if stats.TotalKeys != uint64(len(keyTypes)) {
		t.Errorf("Expected %d total keys, got %d", len(keyTypes), stats.TotalKeys)
	}

	if stats.ActiveKeys != uint64(len(keyTypes)) {
		t.Errorf("Expected %d active keys, got %d", len(keyTypes), stats.ActiveKeys)
	}

	t.Logf("✅ All %d key types generated and retrieved successfully", len(keyTypes))
}

func TestVerificationKeySecuritySystem_KeyExpiry(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "vkey_expiry_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := &VKeySecurityConfig{
		StorageDir:                 tempDir,
		FilePermissions:            0600,
		DefaultKeyExpiry:           100 * time.Millisecond, // Very short expiry for testing
		EncryptionKeySize:          32,
		MultiSigEnabled:            false,
		BackupEnabled:              false,
		AutoRotationEnabled:        false,
		CompromiseDetectionEnabled: false,
	}

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Generate test key
	testKeyData := make([]byte, 64)
	if _, err := rand.Read(testKeyData); err != nil {
		t.Fatalf("Failed to generate test key data: %v", err)
	}

	secureKey, err := vkss.GenerateSecureVerificationKey(
		VKeyTypeCAPSS,
		"expiry_test",
		testKeyData,
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to generate secure verification key: %v", err)
	}

	// Verify key is initially valid
	_, _, err = vkss.GetSecureVerificationKey(secureKey.ID)
	if err != nil {
		t.Fatalf("Expected key to be valid initially: %v", err)
	}

	// Wait for key to expire
	time.Sleep(200 * time.Millisecond)

	// Verify key is now expired
	_, _, err = vkss.GetSecureVerificationKey(secureKey.ID)
	if err == nil {
		t.Errorf("Expected error when retrieving expired key")
	}

	if err.Error() != "verification key expired" {
		t.Errorf("Expected 'verification key expired' error, got: %v", err)
	}

	t.Logf("✅ Key expiry mechanism working correctly")
}

func TestVerificationKeySecuritySystem_DefaultConfig(t *testing.T) {
	config := DefaultVKeySecurityConfig()

	// Verify default values
	if config.StorageDir != "./verification_keys" {
		t.Errorf("Expected default storage dir './verification_keys', got '%s'", config.StorageDir)
	}

	if config.FilePermissions != 0600 {
		t.Errorf("Expected default file permissions 0600, got %o", config.FilePermissions)
	}

	if config.DefaultKeyExpiry != 365*24*time.Hour {
		t.Errorf("Expected default key expiry 1 year, got %v", config.DefaultKeyExpiry)
	}

	if config.RotationInterval != 90*24*time.Hour {
		t.Errorf("Expected default rotation interval 90 days, got %v", config.RotationInterval)
	}

	if !config.AutoRotationEnabled {
		t.Errorf("Expected auto rotation to be enabled by default")
	}

	if config.EncryptionKeySize != 32 {
		t.Errorf("Expected default encryption key size 32, got %d", config.EncryptionKeySize)
	}

	if !config.MultiSigEnabled {
		t.Errorf("Expected multi-signature to be enabled by default")
	}

	if config.MultiSigThreshold != 2 {
		t.Errorf("Expected default multi-sig threshold 2, got %d", config.MultiSigThreshold)
	}

	if !config.BackupEnabled {
		t.Errorf("Expected backup to be enabled by default")
	}

	if config.BackupInterval != 7*24*time.Hour {
		t.Errorf("Expected default backup interval 7 days, got %v", config.BackupInterval)
	}

	if config.ThresholdShares != 5 {
		t.Errorf("Expected default threshold shares 5, got %d", config.ThresholdShares)
	}

	if config.MinRecoveryShares != 3 {
		t.Errorf("Expected default min recovery shares 3, got %d", config.MinRecoveryShares)
	}

	if !config.CompromiseDetectionEnabled {
		t.Errorf("Expected compromise detection to be enabled by default")
	}

	if config.UsageAnomalyThreshold != 5.0 {
		t.Errorf("Expected default usage anomaly threshold 5.0, got %f", config.UsageAnomalyThreshold)
	}

	if config.MaxFailureRate != 0.05 {
		t.Errorf("Expected default max failure rate 0.05, got %f", config.MaxFailureRate)
	}

	t.Logf("✅ Default configuration values verified")
}

func TestVerificationKeySecuritySystem_Encryption(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "vkey_encryption_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultVKeySecurityConfig()
	config.StorageDir = tempDir
	config.MultiSigEnabled = false
	config.BackupEnabled = false
	config.AutoRotationEnabled = false
	config.CompromiseDetectionEnabled = false

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Test with different key data sizes
	testSizes := []int{32, 64, 128, 256, 512, 1024}
	
	for _, size := range testSizes {
		// Generate test key data
		testKeyData := make([]byte, size)
		if _, err := rand.Read(testKeyData); err != nil {
			t.Fatalf("Failed to generate test key data of size %d: %v", size, err)
		}

		// Generate key
		secureKey, err := vkss.GenerateSecureVerificationKey(
			VKeyTypeCAPSS,
			"encryption_test",
			testKeyData,
			nil,
		)
		if err != nil {
			t.Fatalf("Failed to generate key for size %d: %v", size, err)
		}

		// Retrieve and verify
		_, decryptedData, err := vkss.GetSecureVerificationKey(secureKey.ID)
		if err != nil {
			t.Fatalf("Failed to retrieve key for size %d: %v", size, err)
		}

		if !bytes.Equal(decryptedData, testKeyData) {
			t.Errorf("Encryption/decryption failed for size %d", size)
		}

		// Verify encrypted data is different from original
		if bytes.Equal(secureKey.KeyData, testKeyData) {
			t.Errorf("Key data was not encrypted for size %d", size)
		}

		t.Logf("✅ Encryption/decryption successful for size %d bytes", size)
	}
}

func TestVerificationKeySecuritySystem_ConcurrentAccess(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "vkey_concurrent_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultVKeySecurityConfig()
	config.StorageDir = tempDir
	config.MultiSigEnabled = false
	config.BackupEnabled = false
	config.AutoRotationEnabled = false
	config.CompromiseDetectionEnabled = false

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		t.Fatalf("Failed to create verification key security system: %v", err)
	}

	// Test concurrent key generation
	const numGoroutines = 10
	const keysPerGoroutine = 5

	done := make(chan bool, numGoroutines)
	errors := make(chan error, numGoroutines*keysPerGoroutine)

	for i := 0; i < numGoroutines; i++ {
		go func(goroutineID int) {
			defer func() { done <- true }()

			for j := 0; j < keysPerGoroutine; j++ {
				// Generate test key data
				testKeyData := make([]byte, 64)
				if _, err := rand.Read(testKeyData); err != nil {
					errors <- err
					return
				}

				// Generate key
				circuitFamily := fmt.Sprintf("concurrent_test_%d_%d", goroutineID, j)
				_, err := vkss.GenerateSecureVerificationKey(
					VKeyTypeCAPSS,
					circuitFamily,
					testKeyData,
					nil,
				)
				if err != nil {
					errors <- err
					return
				}
			}
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Check for errors
	close(errors)
	for err := range errors {
		t.Errorf("Concurrent access error: %v", err)
	}

	// Verify total keys generated
	stats := vkss.GetVerificationKeySecurityStats()
	expectedKeys := uint64(numGoroutines * keysPerGoroutine)
	if stats.TotalKeys != expectedKeys {
		t.Errorf("Expected %d total keys, got %d", expectedKeys, stats.TotalKeys)
	}

	t.Logf("✅ Concurrent access test completed: %d keys generated by %d goroutines", 
		stats.TotalKeys, numGoroutines)
}

// Benchmark tests

func BenchmarkVerificationKeyGeneration(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "vkey_bench")
	if err != nil {
		b.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultVKeySecurityConfig()
	config.StorageDir = tempDir
	config.MultiSigEnabled = false
	config.BackupEnabled = false
	config.AutoRotationEnabled = false
	config.CompromiseDetectionEnabled = false

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		b.Fatalf("Failed to create verification key security system: %v", err)
	}

	testKeyData := make([]byte, 256)
	if _, err := rand.Read(testKeyData); err != nil {
		b.Fatalf("Failed to generate test key data: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := vkss.GenerateSecureVerificationKey(
			VKeyTypeCAPSS,
			"benchmark_test",
			testKeyData,
			nil,
		)
		if err != nil {
			b.Fatalf("Failed to generate key: %v", err)
		}
	}
}

func BenchmarkVerificationKeyRetrieval(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "vkey_bench")
	if err != nil {
		b.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultVKeySecurityConfig()
	config.StorageDir = tempDir
	config.MultiSigEnabled = false
	config.BackupEnabled = false
	config.AutoRotationEnabled = false
	config.CompromiseDetectionEnabled = false

	vkss, err := NewVerificationKeySecuritySystem(config)
	if err != nil {
		b.Fatalf("Failed to create verification key security system: %v", err)
	}

	testKeyData := make([]byte, 256)
	if _, err := rand.Read(testKeyData); err != nil {
		b.Fatalf("Failed to generate test key data: %v", err)
	}

	// Generate a key for benchmarking retrieval
	secureKey, err := vkss.GenerateSecureVerificationKey(
		VKeyTypeCAPSS,
		"benchmark_test",
		testKeyData,
		nil,
	)
	if err != nil {
		b.Fatalf("Failed to generate key: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := vkss.GetSecureVerificationKey(secureKey.ID)
		if err != nil {
			b.Fatalf("Failed to retrieve key: %v", err)
		}
	}
} 