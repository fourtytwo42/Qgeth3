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
	qiskitBackend   *QiskitBackend
}

// NewHybridQuantumSimulator creates a new hybrid simulator without CUDA support
func NewHybridQuantumSimulator(deviceID int) (*HybridQuantumSimulator, error) {
	simulator := &HybridQuantumSimulator{
		deviceID: deviceID,
	}

	fmt.Printf("⚠️  CUDA backend not available (not compiled with CUDA support)\n")

	// Test Qiskit availability
	qiskitBackend, err := NewQiskitBackend(deviceID)
	if err == nil {
		simulator.qiskitAvailable = true
		simulator.qiskitBackend = qiskitBackend
		fmt.Printf("✅ Qiskit GPU backend available\n")
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
		return h.qiskitBackend.SimulateQuantumPuzzle(puzzleIndex, workHash, qnonce, nQubits, nGates)
	}

	return nil, fmt.Errorf("no suitable quantum backend available")
}

// BenchmarkHybrid benchmarks only Qiskit backend in fallback mode
func (h *HybridQuantumSimulator) BenchmarkHybrid() (map[string]interface{}, error) {
	results := make(map[string]interface{})

	if h.qiskitAvailable {
		qiskitResult, err := h.qiskitBackend.Benchmark(10)
		if err != nil {
			results["qiskit_error"] = err.Error()
		} else {
			results["qiskit"] = qiskitResult
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
