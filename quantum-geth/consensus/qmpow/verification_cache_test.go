package qmpow

import (
	"fmt"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
)

func TestVerificationCache_ProofCaching(t *testing.T) {
	// Create cache with short TTL for testing
	config := VerificationCacheConfig{
		MaxProofEntries: 100,
		MaxBlockEntries: 100,
		ProofTTL:        100 * time.Millisecond,
		BlockTTL:        200 * time.Millisecond,
		CleanupInterval: 50 * time.Millisecond,
	}
	cache := NewVerificationCache(config)
	defer cache.Stop()

	// Test proof verification caching
	proofHash := common.HexToHash("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
	circuitHash := common.HexToHash("0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890")
	
	// Cache miss test
	_, found := cache.GetProofVerification(proofHash)
	if found {
		t.Error("Expected cache miss for new proof hash")
	}

	// Store proof verification result
	result := VerificationResult{
		Valid:     true,
		Timestamp: time.Now(),
		Error:     nil,
	}
	cache.StoreProofVerification(proofHash, result, "CAPSS", circuitHash)

	// Cache hit test
	cached, found := cache.GetProofVerification(proofHash)
	if !found {
		t.Error("Expected cache hit for stored proof hash")
	}
	if !cached.Valid {
		t.Error("Expected cached result to be valid")
	}
	if cached.Error != nil {
		t.Errorf("Expected no error, got: %v", cached.Error)
	}

	// Test TTL expiration
	time.Sleep(150 * time.Millisecond) // Wait for proof TTL to expire
	_, found = cache.GetProofVerification(proofHash)
	if found {
		t.Error("Expected cache miss after TTL expiration")
	}
}

func TestVerificationCache_BlockCaching(t *testing.T) {
	config := DefaultVerificationCacheConfig()
	config.BlockTTL = 100 * time.Millisecond
	cache := NewVerificationCache(config)
	defer cache.Stop()

	// Test block verification caching
	blockHash := common.HexToHash("0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321")
	proofRoot := common.HexToHash("0x1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff")

	// Cache miss test
	_, found := cache.GetBlockVerification(blockHash)
	if found {
		t.Error("Expected cache miss for new block hash")
	}

	// Store block verification result
	result := VerificationResult{
		Valid:     true,
		Timestamp: time.Now(),
		Error:     nil,
	}
	cache.StoreBlockVerification(blockHash, result, proofRoot)

	// Cache hit test
	cached, found := cache.GetBlockVerification(blockHash)
	if !found {
		t.Error("Expected cache hit for stored block hash")
	}
	if !cached.Valid {
		t.Error("Expected cached result to be valid")
	}

	// Test TTL expiration
	time.Sleep(150 * time.Millisecond) // Wait for block TTL to expire
	_, found = cache.GetBlockVerification(blockHash)
	if found {
		t.Error("Expected cache miss after TTL expiration")
	}
}

func TestVerificationCache_LRUEviction(t *testing.T) {
	// Create cache with small capacity
	config := VerificationCacheConfig{
		MaxProofEntries: 2,
		MaxBlockEntries: 2,
		ProofTTL:        1 * time.Hour,
		BlockTTL:        1 * time.Hour,
		CleanupInterval: 1 * time.Minute,
	}
	cache := NewVerificationCache(config)
	defer cache.Stop()

	// Fill proof cache beyond capacity
	result := VerificationResult{Valid: true, Timestamp: time.Now()}
	circuitHash := common.Hash{}

	hash1 := common.HexToHash("0x0001")
	hash2 := common.HexToHash("0x0002")
	hash3 := common.HexToHash("0x0003")

	cache.StoreProofVerification(hash1, result, "CAPSS", circuitHash)
	cache.StoreProofVerification(hash2, result, "CAPSS", circuitHash)
	cache.StoreProofVerification(hash3, result, "CAPSS", circuitHash) // Should evict hash1

	// Check that oldest entry was evicted
	_, found1 := cache.GetProofVerification(hash1)
	_, found2 := cache.GetProofVerification(hash2)
	_, found3 := cache.GetProofVerification(hash3)

	if found1 {
		t.Error("Expected oldest entry (hash1) to be evicted")
	}
	if !found2 {
		t.Error("Expected hash2 to still be cached")
	}
	if !found3 {
		t.Error("Expected hash3 to be cached")
	}
}

func TestVerificationCache_Stats(t *testing.T) {
	cache := NewVerificationCache(DefaultVerificationCacheConfig())
	defer cache.Stop()

	initialStats := cache.GetStats()
	if initialStats.ProofHits != 0 || initialStats.ProofMisses != 0 {
		t.Error("Expected initial stats to be zero")
	}

	proofHash := common.HexToHash("0xaaaa")
	result := VerificationResult{Valid: true, Timestamp: time.Now()}

	// Generate miss
	cache.GetProofVerification(proofHash)

	// Store and generate hit
	cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})
	cache.GetProofVerification(proofHash)

	stats := cache.GetStats()
	if stats.ProofMisses != 1 {
		t.Errorf("Expected 1 proof miss, got %d", stats.ProofMisses)
	}
	if stats.ProofHits != 1 {
		t.Errorf("Expected 1 proof hit, got %d", stats.ProofHits)
	}
	if stats.TotalEntries != 1 {
		t.Errorf("Expected 1 total entry, got %d", stats.TotalEntries)
	}
}

func TestVerificationCache_Invalidation(t *testing.T) {
	cache := NewVerificationCache(DefaultVerificationCacheConfig())
	defer cache.Stop()

	proofHash := common.HexToHash("0xbbbb")
	blockHash := common.HexToHash("0xcccc")
	result := VerificationResult{Valid: true, Timestamp: time.Now()}

	// Store entries
	cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})
	cache.StoreBlockVerification(blockHash, result, common.Hash{})

	// Verify they exist
	_, found1 := cache.GetProofVerification(proofHash)
	_, found2 := cache.GetBlockVerification(blockHash)
	if !found1 || !found2 {
		t.Error("Expected entries to be cached")
	}

	// Test individual invalidation
	cache.InvalidateProof(proofHash)
	_, found1 = cache.GetProofVerification(proofHash)
	if found1 {
		t.Error("Expected proof to be invalidated")
	}

	cache.InvalidateBlock(blockHash)
	_, found2 = cache.GetBlockVerification(blockHash)
	if found2 {
		t.Error("Expected block to be invalidated")
	}

	// Test clear all
	cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})
	cache.StoreBlockVerification(blockHash, result, common.Hash{})
	
	cache.Clear()
	
	_, found1 = cache.GetProofVerification(proofHash)
	_, found2 = cache.GetBlockVerification(blockHash)
	if found1 || found2 {
		t.Error("Expected all entries to be cleared")
	}
}

func TestVerificationCache_ErrorCaching(t *testing.T) {
	cache := NewVerificationCache(DefaultVerificationCacheConfig())
	defer cache.Stop()

	proofHash := common.HexToHash("0xdddd")
	
	// Store error result
	result := VerificationResult{
		Valid:     false,
		Timestamp: time.Now(),
		Error:     fmt.Errorf("verification failed"),
	}
	cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})

	// Retrieve and verify error is cached
	cached, found := cache.GetProofVerification(proofHash)
	if !found {
		t.Error("Expected error result to be cached")
	}
	if cached.Valid {
		t.Error("Expected cached result to be invalid")
	}
	if cached.Error == nil {
		t.Error("Expected cached error to be preserved")
	}
	if cached.Error.Error() != "verification failed" {
		t.Errorf("Expected error message 'verification failed', got '%v'", cached.Error)
	}
}

func TestVerificationCache_Concurrent(t *testing.T) {
	cache := NewVerificationCache(DefaultVerificationCacheConfig())
	defer cache.Stop()

	// Test concurrent access
	done := make(chan bool, 10)
	
	for i := 0; i < 10; i++ {
		go func(id int) {
			proofHash := common.HexToHash(fmt.Sprintf("0x%04x", id))
			result := VerificationResult{Valid: true, Timestamp: time.Now()}
			
			// Store
			cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})
			
			// Retrieve
			_, found := cache.GetProofVerification(proofHash)
			if !found {
				t.Errorf("Goroutine %d: Expected to find cached result", id)
			}
			
			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}
}

func BenchmarkVerificationCache_Get(b *testing.B) {
	cache := NewVerificationCache(DefaultVerificationCacheConfig())
	defer cache.Stop()

	proofHash := common.HexToHash("0xbench")
	result := VerificationResult{Valid: true, Timestamp: time.Now()}
	cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.GetProofVerification(proofHash)
	}
}

func BenchmarkVerificationCache_Store(b *testing.B) {
	cache := NewVerificationCache(DefaultVerificationCacheConfig())
	defer cache.Stop()

	result := VerificationResult{Valid: true, Timestamp: time.Now()}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proofHash := common.HexToHash(fmt.Sprintf("0x%08x", i))
		cache.StoreProofVerification(proofHash, result, "CAPSS", common.Hash{})
	}
} 