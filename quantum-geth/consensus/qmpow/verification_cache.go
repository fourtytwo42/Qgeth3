package qmpow

import (
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/log"
)

// VerificationCacheConfig contains configuration for the verification cache
type VerificationCacheConfig struct {
	MaxProofEntries int           // Maximum number of proof verification results to cache
	MaxBlockEntries int           // Maximum number of block verification results to cache
	ProofTTL        time.Duration // Time-to-live for proof verification results
	BlockTTL        time.Duration // Time-to-live for block verification results
	CleanupInterval time.Duration // How often to clean up expired entries
}

// DefaultVerificationCacheConfig returns sensible default configuration
func DefaultVerificationCacheConfig() VerificationCacheConfig {
	return VerificationCacheConfig{
		MaxProofEntries: 10000,        // Cache up to 10,000 proof verifications
		MaxBlockEntries: 1000,         // Cache up to 1,000 block verifications
		ProofTTL:        1 * time.Hour, // Proof verifications valid for 1 hour
		BlockTTL:        24 * time.Hour, // Block verifications valid for 24 hours
		CleanupInterval: 10 * time.Minute, // Clean up every 10 minutes
	}
}

// VerificationResult represents a cached verification result
type VerificationResult struct {
	Valid     bool      // Whether the verification passed
	Timestamp time.Time // When this result was cached
	Error     error     // Any error that occurred during verification
}

// ProofCacheEntry represents a cached proof verification result
type ProofCacheEntry struct {
	ProofHash   common.Hash        // Hash of the proof data
	Result      VerificationResult // Verification result
	ProofType   string            // Type of proof (CAPSS, Nova, Final)
	CircuitHash common.Hash        // Hash of the circuit used
}

// BlockCacheEntry represents a cached block verification result
type BlockCacheEntry struct {
	BlockHash common.Hash        // Hash of the block
	Result    VerificationResult // Verification result
	ProofRoot common.Hash        // Proof root from block header
}

// VerificationCache implements an LRU cache with TTL for verification results
type VerificationCache struct {
	config VerificationCacheConfig
	
	// Proof verification cache
	proofCache  map[common.Hash]*ProofCacheEntry
	proofLRU    []common.Hash // LRU order for proof cache
	
	// Block verification cache
	blockCache  map[common.Hash]*BlockCacheEntry
	blockLRU    []common.Hash // LRU order for block cache
	
	// Statistics
	stats VerificationCacheStats
	
	// Synchronization
	mu     sync.RWMutex
	stopCh chan struct{}
}

// VerificationCacheStats tracks cache performance metrics
type VerificationCacheStats struct {
	ProofHits         uint64 // Number of proof cache hits
	ProofMisses       uint64 // Number of proof cache misses
	ProofEvictions    uint64 // Number of proof cache evictions
	BlockHits         uint64 // Number of block cache hits
	BlockMisses       uint64 // Number of block cache misses
	BlockEvictions    uint64 // Number of block cache evictions
	CleanupOperations uint64 // Number of cleanup operations performed
	TotalEntries      uint64 // Total entries currently cached
}

// NewVerificationCache creates a new verification cache
func NewVerificationCache(config VerificationCacheConfig) *VerificationCache {
	cache := &VerificationCache{
		config:     config,
		proofCache: make(map[common.Hash]*ProofCacheEntry),
		proofLRU:   make([]common.Hash, 0, config.MaxProofEntries),
		blockCache: make(map[common.Hash]*BlockCacheEntry),
		blockLRU:   make([]common.Hash, 0, config.MaxBlockEntries),
		stopCh:     make(chan struct{}),
	}
	
	// Start cleanup goroutine
	go cache.cleanupLoop()
	
	return cache
}

// GetProofVerification retrieves a cached proof verification result
func (vc *VerificationCache) GetProofVerification(proofHash common.Hash) (VerificationResult, bool) {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	
	entry, exists := vc.proofCache[proofHash]
	if !exists {
		vc.stats.ProofMisses++
		return VerificationResult{}, false
	}
	
	// Check if entry has expired
	if time.Since(entry.Result.Timestamp) > vc.config.ProofTTL {
		vc.stats.ProofMisses++
		// Note: We don't delete here to avoid write lock, cleanup will handle it
		return VerificationResult{}, false
	}
	
	vc.stats.ProofHits++
	
	// Update LRU order (move to front)
	vc.updateProofLRU(proofHash)
	
	return entry.Result, true
}

// StoreProofVerification stores a proof verification result in the cache
func (vc *VerificationCache) StoreProofVerification(proofHash common.Hash, result VerificationResult, proofType string, circuitHash common.Hash) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	// Create new entry
	entry := &ProofCacheEntry{
		ProofHash:   proofHash,
		Result:      result,
		ProofType:   proofType,
		CircuitHash: circuitHash,
	}
	
	// Store in cache
	vc.proofCache[proofHash] = entry
	
	// Update LRU order (add to front)
	vc.addToProofLRU(proofHash)
	
	// Evict oldest entries if cache is full
	for len(vc.proofCache) > vc.config.MaxProofEntries {
		vc.evictOldestProof()
	}
	
	vc.updateTotalEntries()
}

// GetBlockVerification retrieves a cached block verification result
func (vc *VerificationCache) GetBlockVerification(blockHash common.Hash) (VerificationResult, bool) {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	
	entry, exists := vc.blockCache[blockHash]
	if !exists {
		vc.stats.BlockMisses++
		return VerificationResult{}, false
	}
	
	// Check if entry has expired
	if time.Since(entry.Result.Timestamp) > vc.config.BlockTTL {
		vc.stats.BlockMisses++
		return VerificationResult{}, false
	}
	
	vc.stats.BlockHits++
	
	// Update LRU order (move to front)
	vc.updateBlockLRU(blockHash)
	
	return entry.Result, true
}

// StoreBlockVerification stores a block verification result in the cache
func (vc *VerificationCache) StoreBlockVerification(blockHash common.Hash, result VerificationResult, proofRoot common.Hash) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	// Create new entry
	entry := &BlockCacheEntry{
		BlockHash: blockHash,
		Result:    result,
		ProofRoot: proofRoot,
	}
	
	// Store in cache
	vc.blockCache[blockHash] = entry
	
	// Update LRU order (add to front)
	vc.addToBlockLRU(blockHash)
	
	// Evict oldest entries if cache is full
	for len(vc.blockCache) > vc.config.MaxBlockEntries {
		vc.evictOldestBlock()
	}
	
	vc.updateTotalEntries()
}

// InvalidateProof removes a specific proof from the cache
func (vc *VerificationCache) InvalidateProof(proofHash common.Hash) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	if _, exists := vc.proofCache[proofHash]; exists {
		delete(vc.proofCache, proofHash)
		vc.removeFromProofLRU(proofHash)
		vc.updateTotalEntries()
	}
}

// InvalidateBlock removes a specific block from the cache
func (vc *VerificationCache) InvalidateBlock(blockHash common.Hash) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	if _, exists := vc.blockCache[blockHash]; exists {
		delete(vc.blockCache, blockHash)
		vc.removeFromBlockLRU(blockHash)
		vc.updateTotalEntries()
	}
}

// Clear removes all entries from the cache
func (vc *VerificationCache) Clear() {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	vc.proofCache = make(map[common.Hash]*ProofCacheEntry)
	vc.proofLRU = vc.proofLRU[:0]
	vc.blockCache = make(map[common.Hash]*BlockCacheEntry)
	vc.blockLRU = vc.blockLRU[:0]
	vc.updateTotalEntries()
	
	log.Info("🗑️ Verification cache cleared")
}

// GetStats returns current cache statistics
func (vc *VerificationCache) GetStats() VerificationCacheStats {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	
	stats := vc.stats
	stats.TotalEntries = uint64(len(vc.proofCache) + len(vc.blockCache))
	return stats
}

// Stop stops the cache cleanup goroutine
func (vc *VerificationCache) Stop() {
	close(vc.stopCh)
}

// LRU management functions
func (vc *VerificationCache) updateProofLRU(hash common.Hash) {
	// Remove from current position
	vc.removeFromProofLRU(hash)
	// Add to front
	vc.addToProofLRU(hash)
}

func (vc *VerificationCache) addToProofLRU(hash common.Hash) {
	vc.proofLRU = append([]common.Hash{hash}, vc.proofLRU...)
}

func (vc *VerificationCache) removeFromProofLRU(hash common.Hash) {
	for i, h := range vc.proofLRU {
		if h == hash {
			vc.proofLRU = append(vc.proofLRU[:i], vc.proofLRU[i+1:]...)
			break
		}
	}
}

func (vc *VerificationCache) evictOldestProof() {
	if len(vc.proofLRU) == 0 {
		return
	}
	
	oldest := vc.proofLRU[len(vc.proofLRU)-1]
	delete(vc.proofCache, oldest)
	vc.proofLRU = vc.proofLRU[:len(vc.proofLRU)-1]
	vc.stats.ProofEvictions++
}

func (vc *VerificationCache) updateBlockLRU(hash common.Hash) {
	// Remove from current position
	vc.removeFromBlockLRU(hash)
	// Add to front
	vc.addToBlockLRU(hash)
}

func (vc *VerificationCache) addToBlockLRU(hash common.Hash) {
	vc.blockLRU = append([]common.Hash{hash}, vc.blockLRU...)
}

func (vc *VerificationCache) removeFromBlockLRU(hash common.Hash) {
	for i, h := range vc.blockLRU {
		if h == hash {
			vc.blockLRU = append(vc.blockLRU[:i], vc.blockLRU[i+1:]...)
			break
		}
	}
}

func (vc *VerificationCache) evictOldestBlock() {
	if len(vc.blockLRU) == 0 {
		return
	}
	
	oldest := vc.blockLRU[len(vc.blockLRU)-1]
	delete(vc.blockCache, oldest)
	vc.blockLRU = vc.blockLRU[:len(vc.blockLRU)-1]
	vc.stats.BlockEvictions++
}

func (vc *VerificationCache) updateTotalEntries() {
	vc.stats.TotalEntries = uint64(len(vc.proofCache) + len(vc.blockCache))
}

// cleanupLoop periodically removes expired entries
func (vc *VerificationCache) cleanupLoop() {
	ticker := time.NewTicker(vc.config.CleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			vc.cleanup()
		case <-vc.stopCh:
			return
		}
	}
}

// cleanup removes expired entries from the cache
func (vc *VerificationCache) cleanup() {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	
	now := time.Now()
	proofExpiredCount := 0
	blockExpiredCount := 0
	
	// Clean up expired proof entries
	for hash, entry := range vc.proofCache {
		if now.Sub(entry.Result.Timestamp) > vc.config.ProofTTL {
			delete(vc.proofCache, hash)
			vc.removeFromProofLRU(hash)
			proofExpiredCount++
		}
	}
	
	// Clean up expired block entries
	for hash, entry := range vc.blockCache {
		if now.Sub(entry.Result.Timestamp) > vc.config.BlockTTL {
			delete(vc.blockCache, hash)
			vc.removeFromBlockLRU(hash)
			blockExpiredCount++
		}
	}
	
	if proofExpiredCount > 0 || blockExpiredCount > 0 {
		log.Debug("🧹 Cache cleanup completed",
			"proof_expired", proofExpiredCount,
			"block_expired", blockExpiredCount,
			"proof_entries", len(vc.proofCache),
			"block_entries", len(vc.blockCache))
	}
	
	vc.stats.CleanupOperations++
	vc.updateTotalEntries()
} 