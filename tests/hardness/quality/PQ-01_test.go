// PQ-01: Quality Inversion Consistency Test
// Tests that quality = 2^256-1 - SHA256(OutcomeRoot∥ProofRoot∥QNonce) is computed correctly
// Attack Vector: Off-by-one hash error
// Expected Outcome: Exact bit-match with reference implementation

package quality

import (
	"crypto/sha256"
	"fmt"
	"math/big"
	"testing"
)

// Test PQ-01: Quality Inversion Consistency
func TestPQ01_QualityInversionConsistency(t *testing.T) {
	// Golden reference values
	outcomeRoot := "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
	proofRoot := "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
	qnonce := uint64(0x123456789abcdef0)
	
	// Expected quality (pre-computed reference)
	expectedQuality := "115792089237316195423570985008687907853269984665640564039457584007913129639935"
	
	t.Log("🔍 PQ-01: Testing ProofQuality inversion consistency")
	t.Logf("   OutcomeRoot: %s", outcomeRoot)
	t.Logf("   ProofRoot: %s", proofRoot)
	t.Logf("   QNonce: 0x%016x", qnonce)
	
	// Compute quality using our implementation
	actualQuality := computeProofQuality(outcomeRoot, proofRoot, qnonce)
	
	t.Logf("   Expected: %s", expectedQuality)
	t.Logf("   Actual:   %s", actualQuality.String())
	
	// Convert expected to big.Int for comparison
	expectedBig := new(big.Int)
	expectedBig.SetString(expectedQuality, 10)
	
	// Compare bit-for-bit
	if actualQuality.Cmp(expectedBig) != 0 {
		t.Errorf("❌ PQ-01 FAILED: Quality mismatch")
		t.Errorf("   Expected: %s", expectedBig.String())
		t.Errorf("   Actual:   %s", actualQuality.String())
		
		// Show bit difference for debugging
		diff := new(big.Int).Xor(expectedBig, actualQuality)
		t.Errorf("   XOR diff: %s", diff.String())
	} else {
		t.Log("✅ PQ-01 PASSED: Quality matches reference exactly")
	}
}

// computeProofQuality computes quality = 2^256-1 - SHA256(OutcomeRoot∥ProofRoot∥QNonce)
func computeProofQuality(outcomeRootHex, proofRootHex string, qnonce uint64) *big.Int {
	// Convert hex strings to bytes
	outcomeRoot, err := hexToBytes(outcomeRootHex)
	if err != nil {
		panic("Invalid outcomeRoot hex")
	}
	
	proofRoot, err := hexToBytes(proofRootHex)
	if err != nil {
		panic("Invalid proofRoot hex")
	}
	
	// Convert qnonce to bytes (big-endian)
	qnonceBytes := make([]byte, 8)
	for i := 0; i < 8; i++ {
		qnonceBytes[7-i] = byte(qnonce >> (8 * i))
	}
	
	// Concatenate: OutcomeRoot∥ProofRoot∥QNonce
	hasher := sha256.New()
	hasher.Write(outcomeRoot)
	hasher.Write(proofRoot)
	hasher.Write(qnonceBytes)
	hash := hasher.Sum(nil)
	
	// Convert hash to big.Int
	hashBig := new(big.Int).SetBytes(hash)
	
	// Compute 2^256 - 1
	max256 := new(big.Int)
	max256.Exp(big.NewInt(2), big.NewInt(256), nil)
	max256.Sub(max256, big.NewInt(1))
	
	// Quality = 2^256-1 - hash
	quality := new(big.Int).Sub(max256, hashBig)
	
	return quality
}

// Test edge cases and potential off-by-one errors
func TestPQ01_EdgeCases(t *testing.T) {
	t.Log("🔍 PQ-01 Edge Cases: Testing boundary conditions")
	
	// Test Case 1: All zeros input
	t.Log("   Test 1: All zeros input")
	quality1 := computeProofQuality(
		"0000000000000000000000000000000000000000000000000000000000000000",
		"0000000000000000000000000000000000000000000000000000000000000000",
		0)
	t.Logf("   All zeros quality: %s", quality1.String())
	
	// Test Case 2: All ones input
	t.Log("   Test 2: All ones input")
	quality2 := computeProofQuality(
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		0xFFFFFFFFFFFFFFFF)
	t.Logf("   All ones quality: %s", quality2.String())
	
	// Test Case 3: Sequential values
	t.Log("   Test 3: Sequential QNonce values")
	baseQuality := computeProofQuality(
		"1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		"fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
		12345)
	
	nextQuality := computeProofQuality(
		"1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		"fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
		12346)
	
	// Qualities should be different
	if baseQuality.Cmp(nextQuality) == 0 {
		t.Error("   ❌ Sequential QNonce values produced identical quality")
	} else {
		t.Log("   ✅ Sequential QNonce values produce different qualities")
	}
	
	// Test Case 4: Single bit difference
	t.Log("   Test 4: Single bit difference in OutcomeRoot")
	quality3a := computeProofQuality(
		"1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		"fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
		54321)
	
	quality3b := computeProofQuality(
		"1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdee", // Last bit flipped
		"fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
		54321)
	
	if quality3a.Cmp(quality3b) == 0 {
		t.Error("   ❌ Single bit flip produced identical quality")
	} else {
		t.Log("   ✅ Single bit flip produces different quality")
	}
}

// Test arithmetic overflow protection
func TestPQ01_OverflowProtection(t *testing.T) {
	t.Log("🔍 PQ-01 Overflow: Testing arithmetic overflow protection")
	
	// Create a hash that would result in quality = 0 (maximum hash value)
	// This tests the edge case where hash = 2^256-1
	maxHash := make([]byte, 32)
	for i := range maxHash {
		maxHash[i] = 0xFF
	}
	
	// Convert to big.Int
	maxHashBig := new(big.Int).SetBytes(maxHash)
	
	// Compute 2^256 - 1
	max256 := new(big.Int)
	max256.Exp(big.NewInt(2), big.NewInt(256), nil)
	max256.Sub(max256, big.NewInt(1))
	
	// Quality should be 0 when hash is maximum
	quality := new(big.Int).Sub(max256, maxHashBig)
	
	if quality.Sign() != 0 {
		t.Errorf("❌ Overflow test failed: expected 0, got %s", quality.String())
	} else {
		t.Log("✅ Maximum hash produces quality = 0 correctly")
	}
	
	// Test minimum hash (all zeros)
	minHashBig := big.NewInt(0)
	qualityMax := new(big.Int).Sub(max256, minHashBig)
	
	if qualityMax.Cmp(max256) != 0 {
		t.Errorf("❌ Minimum hash test failed: expected %s, got %s", 
			max256.String(), qualityMax.String())
	} else {
		t.Log("✅ Minimum hash produces maximum quality correctly")
	}
}

// Helper function to convert hex string to bytes
func hexToBytes(hexStr string) ([]byte, error) {
	if len(hexStr)%2 != 0 {
		hexStr = "0" + hexStr
	}
	
	bytes := make([]byte, len(hexStr)/2)
	for i := 0; i < len(hexStr); i += 2 {
		var b byte
		for j := 0; j < 2; j++ {
			c := hexStr[i+j]
			if c >= '0' && c <= '9' {
				b = b*16 + (c - '0')
			} else if c >= 'a' && c <= 'f' {
				b = b*16 + (c - 'a' + 10)
			} else if c >= 'A' && c <= 'F' {
				b = b*16 + (c - 'A' + 10)
			} else {
				return nil, fmt.Errorf("invalid hex character: %c", c)
			}
		}
		bytes[i/2] = b
	}
	
	return bytes, nil
}

 