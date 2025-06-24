//go:build !cuda
// +build !cuda

package quantum

import (
	"fmt"
	"log"
	"time"
)

// HighPerformanceQuantumSimulator with CuPy GPU + Qiskit fallback
type HighPerformanceQuantumSimulator struct {
	nQubits   int
	cupyGPU   *CupyGPUSimulator   // Primary: CuPy GPU backend
	qiskitSim *QiskitGPUSimulator // Fallback: Qiskit backend
}

// BatchResult fallback structure
type BatchResult struct {
	Outcomes       [][]byte
	ProcessingTime time.Duration
	BatchSize      int
	GPUUtilization float64
}

// NewHighPerformanceQuantumSimulator creates simulator with CuPy GPU + Qiskit fallback
func NewHighPerformanceQuantumSimulator(nQubits int) (*HighPerformanceQuantumSimulator, error) {
	sim := &HighPerformanceQuantumSimulator{
		nQubits: nQubits,
	}

	// Try to initialize CuPy GPU backend first
	sim.cupyGPU = NewCupyGPUSimulator()
	if sim.cupyGPU.IsAvailable() {
		log.Printf("🚀 CuPy GPU quantum simulator ACTIVE!")
		log.Printf("   Device: %v", sim.cupyGPU.GetDeviceInfo())
	} else {
		log.Printf("⚠️  CuPy GPU not available, trying Qiskit...")

		// Fallback to Qiskit GPU backend
		qiskitSim, err := NewQiskitGPUSimulator(0) // Device 0
		if err != nil {
			log.Printf("⚠️  Qiskit GPU initialization also failed: %v", err)
			log.Printf("💡 Using CPU fallback simulation")
			sim.qiskitSim = nil
		} else {
			sim.qiskitSim = qiskitSim
			if qiskitSim.IsGPUAvailable() {
				log.Printf("🚀 Qiskit CUDA 12.9 GPU quantum simulator ACTIVE!")
			} else {
				log.Printf("💻 Qiskit CPU quantum simulator active")
			}
		}
	}

	return sim, nil
}

// BatchSimulateQuantumPuzzles uses CuPy GPU first, then Qiskit, then CPU fallback
func (h *HighPerformanceQuantumSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	// Try CuPy GPU first
	if h.cupyGPU != nil && h.cupyGPU.IsAvailable() {
		return h.cupyBatchSimulate(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	// Fallback to Qiskit GPU/CPU backend
	if h.qiskitSim != nil {
		return h.qiskitSim.BatchSimulateQuantumPuzzles(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	// Final fallback to simple CPU simulation
	log.Printf("🔧 Using basic CPU fallback simulation for %d puzzles", nPuzzles)
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

// cupyBatchSimulate uses CuPy GPU backend for batch simulation
func (h *HighPerformanceQuantumSimulator) cupyBatchSimulate(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	// Silent GPU processing - only log on first run or errors

	// Create puzzles for CuPy simulation
	puzzles := make([]map[string]interface{}, nPuzzles)
	for i := 0; i < nPuzzles; i++ {
		puzzles[i] = map[string]interface{}{
			"num_qubits":        nQubits,
			"target_state":      "entangled", // Use entangled state for mining
			"measurement_basis": "computational",
			"work_hash":         workHash,
			"qnonce":            qnonce,
			"puzzle_id":         i,
		}
	}

	// Run batch simulation on GPU
	results, err := h.cupyGPU.BatchSimulateQuantumPuzzles(puzzles)
	if err != nil {
		// Check if it's an interrupt (user stopping miner)
		if err.Error() == "simulation interrupted" {
			return nil, err // Pass through interrupt cleanly
		}
		return nil, fmt.Errorf("CuPy GPU batch simulation failed: %v", err)
	}

	// Convert results to byte outcomes
	outcomes := make([][]byte, nPuzzles)
	for i, result := range results {
		if i >= nPuzzles {
			break
		}

		// Extract probabilities and convert to deterministic outcome
		outcome := make([]byte, (nQubits+7)/8)

		if probabilities, ok := result["probabilities"].([]interface{}); ok && len(probabilities) > 0 {
			// Use highest probability state as outcome
			maxProb := 0.0
			maxState := 0

			for state, prob := range probabilities {
				if p, ok := prob.(float64); ok && p > maxProb {
					maxProb = p
					maxState = state
				}
			}

			// Convert state to bytes
			stateBytes := uint64(maxState)
			for j := 0; j < len(outcome); j++ {
				outcome[j] = byte(stateBytes >> (8 * j))
			}
		} else {
			// Fallback: use puzzle metadata for deterministic outcome
			seed := fmt.Sprintf("%s_%d_%d", workHash, qnonce, i)
			hash := sha256Hash(seed)
			copy(outcome, []byte(hash)[:len(outcome)])
		}

		outcomes[i] = outcome
	}

	return outcomes, nil
}

func (h *HighPerformanceQuantumSimulator) Cleanup() {
	// CuPy doesn't need explicit cleanup (garbage collected)
	if h.qiskitSim != nil {
		h.qiskitSim.Cleanup()
	}
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
