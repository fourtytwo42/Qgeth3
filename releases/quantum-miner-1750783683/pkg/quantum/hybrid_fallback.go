//go:build !cuda
// +build !cuda

package quantum

import (
	"fmt"
)

// HybridQuantumSimulator combines CUDA and Qiskit backends for optimal performance (fallback)
type HybridQuantumSimulator struct {
	qiskitAvailable bool
	deviceID        int
	qiskitBackend   *QiskitGPUSimulator
}

// NewHybridQuantumSimulator creates a new hybrid simulator without CUDA support
func NewHybridQuantumSimulator(deviceID int) (*HybridQuantumSimulator, error) {
	simulator := &HybridQuantumSimulator{
		deviceID: deviceID,
	}

	fmt.Printf("⚠️  CUDA backend not available (not compiled with CUDA support)\n")

	// Test Qiskit availability
	qiskitBackend, err := NewQiskitGPUSimulator(deviceID)
	if err == nil {
		simulator.qiskitAvailable = true
		simulator.qiskitBackend = qiskitBackend
		if qiskitBackend.IsGPUAvailable() {
			fmt.Printf("✅ Qiskit GPU backend available\n")
		} else {
			fmt.Printf("✅ Qiskit CPU backend available\n")
		}
	} else {
		fmt.Printf("⚠️  Qiskit backend not available: %v\n", err)
	}

	if !simulator.qiskitAvailable {
		return nil, fmt.Errorf("no quantum backends are available")
	}

	return simulator, nil
}

// SimulateQuantumPuzzleHybrid uses only Qiskit backend in fallback mode
func (h *HybridQuantumSimulator) SimulateQuantumPuzzleHybrid(puzzleIndex int, workHash string,
	qnonce uint64, nQubits, nGates int) ([]byte, error) {

	if h.qiskitAvailable {
		// Use the single puzzle simulation method (we'll need to add this)
		outcomes, err := h.qiskitBackend.BatchSimulateQuantumPuzzles(workHash, qnonce, nQubits, nGates, 1)
		if err != nil {
			return nil, err
		}
		return outcomes[0], nil
	}

	return nil, fmt.Errorf("no suitable quantum backend available")
}

// BenchmarkHybrid benchmarks only Qiskit backend in fallback mode
func (h *HybridQuantumSimulator) BenchmarkHybrid() (map[string]interface{}, error) {
	results := make(map[string]interface{})

	if h.qiskitAvailable {
		// For now, just return basic info since we don't have the old Benchmark method
		results["qiskit"] = map[string]interface{}{
			"available": true,
			"gpu_used":  h.qiskitBackend.IsGPUAvailable(),
		}
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no benchmarks completed successfully")
	}

	return results, nil
}

// BatchSimulateQuantumPuzzles simulates multiple quantum puzzles efficiently
func (h *HybridQuantumSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	if h.qiskitAvailable {
		return h.qiskitBackend.BatchSimulateQuantumPuzzles(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	return nil, fmt.Errorf("no suitable quantum backend available")
}
