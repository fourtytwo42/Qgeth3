package qmpow

import (
	"bytes"
	"crypto/rand"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
)

func TestDilithiumAttestorCreation(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	if attestor == nil {
		t.Fatal("Failed to create Dilithium attestor")
	}

	if attestor.chainIDHash != chainIDHash {
		t.Errorf("Chain ID hash mismatch: got %v, expected %v", attestor.chainIDHash, chainIDHash)
	}

	stats := attestor.GetStats()
	if stats.TotalKeyGenerations != 0 {
		t.Errorf("Expected zero initial key generations, got %d", stats.TotalKeyGenerations)
	}
}

func TestGenerateAttestationSeed(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	copy(seed0, "test_seed_0123456789abcdef")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	blockNumber := uint64(12345)

	attestSeed := attestor.GenerateAttestationSeed(seed0, outcomeRoot, blockNumber)

	if len(attestSeed) != 32 {
		t.Errorf("Expected attestation seed length 32, got %d", len(attestSeed))
	}

	// Test deterministic generation
	attestSeed2 := attestor.GenerateAttestationSeed(seed0, outcomeRoot, blockNumber)
	if !bytes.Equal(attestSeed, attestSeed2) {
		t.Error("Attestation seed generation is not deterministic")
	}

	// Test different inputs produce different seeds
	attestSeed3 := attestor.GenerateAttestationSeed(seed0, outcomeRoot, blockNumber+1)
	if bytes.Equal(attestSeed, attestSeed3) {
		t.Error("Different block numbers should produce different attestation seeds")
	}
}

func TestKeyGeneration(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)

	keyPair, err := attestor.KeyGen(seed)
	if err != nil {
		t.Fatalf("Key generation failed: %v", err)
	}

	if keyPair == nil {
		t.Fatal("Key pair is nil")
	}

	if len(keyPair.PrivateKey) != DilithiumPrivateKeySize {
		t.Errorf("Private key size mismatch: got %d, expected %d",
			len(keyPair.PrivateKey), DilithiumPrivateKeySize)
	}

	if len(keyPair.PublicKey) != DilithiumPublicKeySize {
		t.Errorf("Public key size mismatch: got %d, expected %d",
			len(keyPair.PublicKey), DilithiumPublicKeySize)
	}

	if !bytes.Equal(keyPair.Seed, seed) {
		t.Error("Seed not preserved in key pair")
	}

	// Test deterministic key generation
	keyPair2, err := attestor.KeyGen(seed)
	if err != nil {
		t.Fatalf("Second key generation failed: %v", err)
	}

	if !bytes.Equal(keyPair.PrivateKey, keyPair2.PrivateKey) {
		t.Error("Key generation is not deterministic")
	}

	if !bytes.Equal(keyPair.PublicKey, keyPair2.PublicKey) {
		t.Error("Public key generation is not deterministic")
	}
}

func TestKeyGenerationInvalidSeed(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	// Test invalid seed sizes
	invalidSeeds := [][]byte{
		make([]byte, 16), // Too short
		make([]byte, 64), // Too long
		nil,              // Nil
		{},               // Empty
	}

	for i, seed := range invalidSeeds {
		_, err := attestor.KeyGen(seed)
		if err == nil {
			t.Errorf("Test %d: Expected error for invalid seed size %d", i, len(seed))
		}
	}
}

func TestSigning(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	// Generate key pair
	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)
	keyPair, err := attestor.KeyGen(seed)
	if err != nil {
		t.Fatalf("Key generation failed: %v", err)
	}

	// Test data
	seed0 := make([]byte, 32)
	copy(seed0, "test_seed_0123456789abcdef")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")

	signature, err := attestor.Sign(keyPair, seed0, outcomeRoot, gateHash)
	if err != nil {
		t.Fatalf("Signing failed: %v", err)
	}

	if signature == nil {
		t.Fatal("Signature is nil")
	}

	if len(signature.Signature) != DilithiumSignatureSize {
		t.Errorf("Signature size mismatch: got %d, expected %d",
			len(signature.Signature), DilithiumSignatureSize)
	}

	if !bytes.Equal(signature.PublicKey, keyPair.PublicKey) {
		t.Error("Public key not preserved in signature")
	}

	expectedMessageLen := len(seed0) + 32 + 32 // seed0 + outcomeRoot + gateHash
	if len(signature.Message) != expectedMessageLen {
		t.Errorf("Message length mismatch: got %d, expected %d",
			len(signature.Message), expectedMessageLen)
	}

	// Test deterministic signing
	signature2, err := attestor.Sign(keyPair, seed0, outcomeRoot, gateHash)
	if err != nil {
		t.Fatalf("Second signing failed: %v", err)
	}

	if !bytes.Equal(signature.Signature, signature2.Signature) {
		t.Error("Signing is not deterministic")
	}
}

func TestVerification(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	// Generate key pair and signature
	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)
	keyPair, err := attestor.KeyGen(seed)
	if err != nil {
		t.Fatalf("Key generation failed: %v", err)
	}

	seed0 := make([]byte, 32)
	copy(seed0, "test_seed_0123456789abcdef")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")

	signature, err := attestor.Sign(keyPair, seed0, outcomeRoot, gateHash)
	if err != nil {
		t.Fatalf("Signing failed: %v", err)
	}

	// Test verification
	valid, err := attestor.Verify(signature)
	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}

	if !valid {
		t.Error("Valid signature failed verification")
	}

	// Test invalid signature
	invalidSig := &DilithiumSignature{
		Signature: make([]byte, DilithiumSignatureSize), // All zeros
		PublicKey: keyPair.PublicKey,
		Message:   signature.Message,
	}

	valid, err = attestor.Verify(invalidSig)
	if err != nil {
		t.Fatalf("Invalid signature verification failed: %v", err)
	}

	if valid {
		t.Error("Invalid signature passed verification")
	}
}

func TestRoundTripVerification(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	copy(seed0, "test_seed_0123456789abcdef")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")
	blockNumber := uint64(12345)

	valid, err := attestor.VerifyRoundTrip(seed0, outcomeRoot, gateHash, blockNumber)
	if err != nil {
		t.Fatalf("Round-trip verification failed: %v", err)
	}

	if !valid {
		t.Error("Round-trip verification should succeed")
	}
}

func TestCreateAttestationPair(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	copy(seed0, "test_seed_0123456789abcdef")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")
	blockNumber := uint64(12345)

	publicKey, signature, err := attestor.CreateAttestationPair(
		seed0, outcomeRoot, gateHash, blockNumber)
	if err != nil {
		t.Fatalf("Create attestation pair failed: %v", err)
	}

	if len(publicKey) != DilithiumPublicKeySize {
		t.Errorf("Public key size mismatch: got %d, expected %d",
			len(publicKey), DilithiumPublicKeySize)
	}

	if len(signature) != DilithiumSignatureSize {
		t.Errorf("Signature size mismatch: got %d, expected %d",
			len(signature), DilithiumSignatureSize)
	}

	// Test total size matches spec (pk + sig = 1744 bytes)
	totalSize := len(publicKey) + len(signature)
	expectedSize := DilithiumPublicKeySize + DilithiumSignatureSize // 1312 + 2420 = 3732
	if totalSize != expectedSize {
		t.Errorf("Total attestation size mismatch: got %d, expected %d",
			totalSize, expectedSize)
	}
}

func TestVerifyAttestationPair(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	copy(seed0, "test_seed_0123456789abcdef")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")
	blockNumber := uint64(12345)

	// Create attestation pair
	publicKey, signature, err := attestor.CreateAttestationPair(
		seed0, outcomeRoot, gateHash, blockNumber)
	if err != nil {
		t.Fatalf("Create attestation pair failed: %v", err)
	}

	// Verify attestation pair
	valid, err := attestor.VerifyAttestationPair(
		publicKey, signature, seed0, outcomeRoot, gateHash, blockNumber)
	if err != nil {
		t.Fatalf("Verify attestation pair failed: %v", err)
	}

	if !valid {
		t.Error("Valid attestation pair failed verification")
	}

	// Test with wrong data
	wrongGateHash := common.HexToHash("0xdeadbeefcafebabe")
	valid, err = attestor.VerifyAttestationPair(
		publicKey, signature, seed0, outcomeRoot, wrongGateHash, blockNumber)
	if err != nil {
		t.Fatalf("Verify wrong attestation pair failed: %v", err)
	}

	if valid {
		t.Error("Wrong attestation pair should fail verification")
	}
}

func TestAttestorStats(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	// Initial stats should be zero
	stats := attestor.GetStats()
	if stats.TotalKeyGenerations != 0 {
		t.Errorf("Expected 0 initial key generations, got %d", stats.TotalKeyGenerations)
	}

	// Perform operations
	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)

	keyPair, err := attestor.KeyGen(seed)
	if err != nil {
		t.Fatalf("Key generation failed: %v", err)
	}

	seed0 := make([]byte, 32)
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")

	signature, err := attestor.Sign(keyPair, seed0, outcomeRoot, gateHash)
	if err != nil {
		t.Fatalf("Signing failed: %v", err)
	}

	_, err = attestor.Verify(signature)
	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}

	// Check updated stats
	stats = attestor.GetStats()
	if stats.TotalKeyGenerations != 1 {
		t.Errorf("Expected 1 key generation, got %d", stats.TotalKeyGenerations)
	}

	if stats.TotalSignatures != 1 {
		t.Errorf("Expected 1 signature, got %d", stats.TotalSignatures)
	}

	if stats.TotalVerifications != 1 {
		t.Errorf("Expected 1 verification, got %d", stats.TotalVerifications)
	}

	if stats.SuccessfulKeyGens != 1 {
		t.Errorf("Expected 1 successful key generation, got %d", stats.SuccessfulKeyGens)
	}

	if stats.SuccessfulSignatures != 1 {
		t.Errorf("Expected 1 successful signature, got %d", stats.SuccessfulSignatures)
	}

	if stats.SuccessfulVerifications != 1 {
		t.Errorf("Expected 1 successful verification, got %d", stats.SuccessfulVerifications)
	}
}

func TestCBDSampling(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed := make([]byte, 16)
	rand.Read(seed)

	vectors := attestor.cbdSample(seed, 4)

	if len(vectors) != 4 {
		t.Errorf("Expected 4 vectors, got %d", len(vectors))
	}

	for i, vec := range vectors {
		if len(vec) != DilithiumN {
			t.Errorf("Vector %d: expected length %d, got %d", i, DilithiumN, len(vec))
		}

		// Check values are in expected range [-eta, eta]
		for j, val := range vec {
			if val < -DilithiumEta || val > DilithiumEta {
				t.Errorf("Vector %d[%d]: value %d out of range [-%d, %d]",
					i, j, val, DilithiumEta, DilithiumEta)
			}
		}
	}
}

func TestMatrixGeneration(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed := make([]byte, 64)
	rand.Read(seed)

	matrix := attestor.generateMatrix(seed)

	if len(matrix) != DilithiumK {
		t.Errorf("Expected %d rows, got %d", DilithiumK, len(matrix))
	}

	for i, row := range matrix {
		if len(row) != DilithiumL {
			t.Errorf("Row %d: expected %d columns, got %d", i, DilithiumL, len(row))
		}

		// Check values are in field
		for j, val := range row {
			if val < 0 || val >= DilithiumQ {
				t.Errorf("Matrix[%d][%d]: value %d out of field range [0, %d)",
					i, j, val, DilithiumQ)
			}
		}
	}
}

func TestPublicKeyNormGuard(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	// Test norm computation
	t_small := make([][]int32, 2)
	for i := range t_small {
		t_small[i] = make([]int32, 3)
		for j := range t_small[i] {
			t_small[i][j] = int32(i + j + 1) // Small values
		}
	}

	norm := attestor.computeNorm(t_small)
	expectedNorm := int64(1*1 + 2*2 + 3*3 + 2*2 + 3*3 + 4*4) // 1+4+9+4+9+16 = 43
	if norm != expectedNorm {
		t.Errorf("Norm computation error: got %d, expected %d", norm, expectedNorm)
	}
}

func TestDeterministicBehavior(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor1 := NewDilithiumAttestor(chainIDHash)
	attestor2 := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	copy(seed0, "deterministic_test_seed_123")
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")
	blockNumber := uint64(54321)

	// Both attestors should produce identical results
	pk1, sig1, err1 := attestor1.CreateAttestationPair(seed0, outcomeRoot, gateHash, blockNumber)
	pk2, sig2, err2 := attestor2.CreateAttestationPair(seed0, outcomeRoot, gateHash, blockNumber)

	if err1 != nil || err2 != nil {
		t.Fatalf("Attestation creation failed: %v, %v", err1, err2)
	}

	if !bytes.Equal(pk1, pk2) {
		t.Error("Public keys should be identical for same inputs")
	}

	if !bytes.Equal(sig1, sig2) {
		t.Error("Signatures should be identical for same inputs")
	}
}

func TestPerformanceMetrics(t *testing.T) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")
	blockNumber := uint64(12345)

	start := time.Now()

	// Perform round-trip test
	valid, err := attestor.VerifyRoundTrip(seed0, outcomeRoot, gateHash, blockNumber)
	if err != nil {
		t.Fatalf("Round-trip test failed: %v", err)
	}

	if !valid {
		t.Error("Round-trip test should succeed")
	}

	elapsed := time.Since(start)
	t.Logf("Round-trip attestation completed in %v", elapsed)

	stats := attestor.GetStats()
	t.Logf("Key generation time: %v", stats.AverageKeyGenTime)
	t.Logf("Signing time: %v", stats.AverageSignTime)
	t.Logf("Verification time: %v", stats.AverageVerifyTime)

	// Performance should be reasonable (< 100ms for test implementation)
	if elapsed > 100*time.Millisecond {
		t.Logf("Warning: Round-trip took %v, may be slow for production", elapsed)
	}
}

func BenchmarkKeyGeneration(b *testing.B) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attestor.KeyGen(seed)
		if err != nil {
			b.Fatalf("Key generation failed: %v", err)
		}
	}
}

func BenchmarkSigning(b *testing.B) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)
	keyPair, err := attestor.KeyGen(seed)
	if err != nil {
		b.Fatalf("Key generation failed: %v", err)
	}

	seed0 := make([]byte, 32)
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attestor.Sign(keyPair, seed0, outcomeRoot, gateHash)
		if err != nil {
			b.Fatalf("Signing failed: %v", err)
		}
	}
}

func BenchmarkVerification(b *testing.B) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed := make([]byte, DilithiumSeedSize)
	rand.Read(seed)
	keyPair, err := attestor.KeyGen(seed)
	if err != nil {
		b.Fatalf("Key generation failed: %v", err)
	}

	seed0 := make([]byte, 32)
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")

	signature, err := attestor.Sign(keyPair, seed0, outcomeRoot, gateHash)
	if err != nil {
		b.Fatalf("Signing failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attestor.Verify(signature)
		if err != nil {
			b.Fatalf("Verification failed: %v", err)
		}
	}
}

func BenchmarkRoundTrip(b *testing.B) {
	chainIDHash := common.HexToHash("0xFEEDFACECAFEBABE")
	attestor := NewDilithiumAttestor(chainIDHash)

	seed0 := make([]byte, 32)
	outcomeRoot := common.HexToHash("0x1234567890abcdef")
	gateHash := common.HexToHash("0xfedcba0987654321")
	blockNumber := uint64(12345)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		valid, err := attestor.VerifyRoundTrip(seed0, outcomeRoot, gateHash, blockNumber)
		if err != nil {
			b.Fatalf("Round-trip failed: %v", err)
		}
		if !valid {
			b.Fatal("Round-trip should succeed")
		}
	}
}
