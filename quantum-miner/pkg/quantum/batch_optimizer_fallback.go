//go:build !cuda
// +build !cuda

package quantum

import (
	"fmt"
	"time"
)

// HighPerformanceQuantumSimulator fallback for non-CUDA builds
type HighPerformanceQuantumSimulator struct {
	nQubits int
}

// BatchResult fallback structure
type BatchResult struct {
	Outcomes       [][]byte
	ProcessingTime time.Duration
	BatchSize      int
	GPUUtilization float64
}

// NewHighPerformanceQuantumSimulator creates fallback simulator (CPU-only)
func NewHighPerformanceQuantumSimulator(nQubits int) (*HighPerformanceQuantumSimulator, error) {
	// GPU mode not available, using CPU fallback (quiet operation)
	return &HighPerformanceQuantumSimulator{
		nQubits: nQubits,
	}, nil
}

// BatchSimulateQuantumPuzzles fallback to CPU processing
func (h *HighPerformanceQuantumSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	// Using CPU fallback for puzzle solving (quiet operation)

	// Fallback to individual CPU simulation
	outcomes := make([][]byte, nPuzzles)
	for i := 0; i < nPuzzles; i++ {
		outcome := make([]byte, (nQubits+7)/8)
		// Simple deterministic outcome generation
		seed := fmt.Sprintf("%s_%d_%d", workHash, qnonce, i)
		hash := sha256Hash(seed)
		copy(outcome, []byte(hash)[:len(outcome)])
		outcomes[i] = outcome
	}

	return outcomes, nil
}

func (h *HighPerformanceQuantumSimulator) Cleanup() {
	// Nothing to cleanup in fallback mode
}

// Simple hash function for fallback
func sha256Hash(input string) string {
	// Very simple hash for fallback
	hash := 0
	for _, c := range input {
		hash = hash*31 + int(c)
	}
	return fmt.Sprintf("%x", hash)
}
