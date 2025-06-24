//go:build cuda
// +build cuda

package quantum

import (
	"fmt"
)

// HybridQuantumSimulator combines CUDA and Qiskit backends for optimal performance
type HybridQuantumSimulator struct {
	cudaAvailable   bool
	qiskitAvailable bool
	deviceID        int
	qiskitBackend   *QiskitBackend
}

// NewHybridQuantumSimulator creates a new hybrid simulator with CUDA support
func NewHybridQuantumSimulator(deviceID int) (*HybridQuantumSimulator, error) {
	simulator := &HybridQuantumSimulator{
		deviceID: deviceID,
	}

	// Test CUDA availability
	deviceInfo, err := GetCUDADeviceInfo(deviceID)
	if err == nil && deviceInfo != nil {
		simulator.cudaAvailable = true
		fmt.Printf("✅ CUDA backend available: %s\n", deviceInfo.Name)
	} else {
		fmt.Printf("⚠️  CUDA backend not available: %v\n", err)
		simulator.cudaAvailable = false
	}

	// Test Qiskit availability
	qiskitBackend, err := NewQiskitBackend(deviceID)
	if err == nil {
		simulator.qiskitAvailable = true
		simulator.qiskitBackend = qiskitBackend
		fmt.Printf("✅ Qiskit GPU backend available\n")
	} else {
		fmt.Printf("⚠️  Qiskit backend not available: %v\n", err)
	}

	if !simulator.cudaAvailable && !simulator.qiskitAvailable {
		return nil, fmt.Errorf("neither CUDA nor Qiskit backends are available")
	}

	return simulator, nil
}

// SimulateQuantumPuzzleHybrid chooses the best backend for simulation
func (h *HybridQuantumSimulator) SimulateQuantumPuzzleHybrid(puzzleIndex int, workHash string,
	qnonce uint64, nQubits, nGates int) ([]byte, error) {

	// For small circuits (< 12 qubits), prefer CUDA for speed
	// For larger circuits, prefer Qiskit for better memory management
	if nQubits < 12 && h.cudaAvailable {
		return SimulateQuantumPuzzleGPU(puzzleIndex, workHash, qnonce, nQubits, nGates, h.deviceID)
	} else if h.qiskitAvailable {
		return h.qiskitBackend.SimulateQuantumPuzzle(puzzleIndex, workHash, qnonce, nQubits, nGates)
	} else if h.cudaAvailable {
		return SimulateQuantumPuzzleGPU(puzzleIndex, workHash, qnonce, nQubits, nGates, h.deviceID)
	}

	return nil, fmt.Errorf("no suitable quantum backend available")
}

// BenchmarkHybrid benchmarks both backends if available
func (h *HybridQuantumSimulator) BenchmarkHybrid() (map[string]interface{}, error) {
	results := make(map[string]interface{})

	if h.cudaAvailable {
		cudaResult, err := BenchmarkGPUPerformance(h.deviceID)
		if err != nil {
			results["cuda_error"] = err.Error()
		} else {
			results["cuda"] = cudaResult
		}
	}

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

	// For batch operations, prefer Qiskit for now (CUDA batch not implemented yet)
	if h.qiskitAvailable {
		return h.qiskitBackend.BatchSimulateQuantumPuzzles(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	// Fallback to individual CUDA simulations
	if h.cudaAvailable {
		outcomes := make([][]byte, nPuzzles)
		for i := 0; i < nPuzzles; i++ {
			outcome, err := SimulateQuantumPuzzleGPU(i, workHash, qnonce, nQubits, nGates, h.deviceID)
			if err != nil {
				return nil, fmt.Errorf("CUDA simulation failed for puzzle %d: %w", i, err)
			}
			outcomes[i] = outcome
		}
		return outcomes, nil
	}

	return nil, fmt.Errorf("no suitable quantum backend available")
}
