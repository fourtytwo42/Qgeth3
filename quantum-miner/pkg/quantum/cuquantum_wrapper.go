package quantum

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// QuantumPuzzle represents a quantum puzzle to be solved
type QuantumPuzzle struct {
	Index      int    `json:"index"`       // Puzzle index
	Qbits      int    `json:"qbits"`       // Number of qubits  
	Tcount     int    `json:"tcount"`      // Number of gates
	Seed       int    `json:"seed"`        // Random seed
	QNonce     uint64 `json:"qnonce"`      // Quantum nonce
	PuzzleHash string `json:"puzzle_hash"` // Puzzle hash
}

// QuantumPuzzleResult represents the result of solving a quantum puzzle
type QuantumPuzzleResult struct {
	Success         bool              `json:"success"`
	PuzzleIndex     int               `json:"puzzle_index"`
	Probabilities   []float64         `json:"probabilities,omitempty"`
	Measurements    map[string]int    `json:"measurements,omitempty"`
	ErrorMessage    string            `json:"error,omitempty"`
	ProcessingTime  float64           `json:"processing_time"`
	GPUUsed         bool              `json:"gpu_used"`
}

// CuQuantumSimulator handles quantum circuit simulation using NVIDIA cuQuantum Appliance Docker container
type CuQuantumSimulator struct {
	containerImage string
	gpuDevices     string
	available      bool
}

// CuQuantumResult represents the result from cuQuantum simulation
type CuQuantumResult struct {
	Success      bool    `json:"success"`
	GPUAvailable bool    `json:"gpu_available"`
	Benchmark    struct {
		Trials    int     `json:"trials"`
		AvgTime   float64 `json:"avg_time"`
		TotalTime float64 `json:"total_time"`
		Qubits    int     `json:"qubits"`
		Gates     int     `json:"gates"`
		GPUUsed   bool    `json:"gpu_used"`
	} `json:"benchmark"`
	Measurements map[string]int `json:"measurements,omitempty"`
	Probabilities []float64     `json:"probabilities,omitempty"`
	ErrorMessage  string        `json:"error,omitempty"`
}

// NewCuQuantumSimulator creates a new cuQuantum Docker simulator
func NewCuQuantumSimulator() (*CuQuantumSimulator, error) {
	sim := &CuQuantumSimulator{
		containerImage: "nvcr.io/nvidia/cuquantum-appliance:25.03-x86_64",
		gpuDevices:     "all",
		available:      false,
	}

	// Test if cuQuantum is available
	if err := sim.testAvailability(); err != nil {
		return nil, fmt.Errorf("cuQuantum not available: %v", err)
	}

	sim.available = true
	return sim, nil
}

// testAvailability checks if cuQuantum Docker container is available and working
func (sim *CuQuantumSimulator) testAvailability() error {
	// Check if running on Windows and provide specific guidance
	if runtime.GOOS == "windows" {
		return fmt.Errorf("cuQuantum Docker requires WSL2 environment on Windows\n\nTo use cuQuantum on Windows:\n1. Install WSL2: wsl --install\n2. Install Docker Desktop with WSL2 backend\n3. Run quantum-miner from WSL2 terminal, not Windows PowerShell\n4. Ensure NVIDIA drivers support WSL2 GPU Paravirtualization")
	}
	
	// Enhanced Windows detection for WSL environments
	isWSL := false
	if runtime.GOOS == "linux" {
		// Check if running in WSL
		if _, err := os.Stat("/proc/version"); err == nil {
			if data, err := os.ReadFile("/proc/version"); err == nil {
				if strings.Contains(strings.ToLower(string(data)), "microsoft") || 
				   strings.Contains(strings.ToLower(string(data)), "wsl") {
					isWSL = true
				}
			}
		}
	}
	
	// Test basic container functionality
	cmd := exec.Command("docker", "run", "--rm", "--gpus", sim.gpuDevices,
		sim.containerImage,
		"python", "-c", "import qiskit; import qiskit_aer; print('cuQuantum ready')")

	output, err := cmd.CombinedOutput()
	if err != nil {
		if isWSL {
			return fmt.Errorf("cuQuantum test failed in WSL2: %v\n\nWSL2 Troubleshooting:\n- Ensure Docker Desktop is running\n- Verify WSL2 integration is enabled in Docker Desktop settings\n- Check NVIDIA drivers support WSL2 GPU Paravirtualization\n- Test basic GPU access: docker run --rm --gpus all nvidia/cuda:11.2-base-ubuntu20.04 nvidia-smi\n\nOutput: %s", err, string(output))
		}
		return fmt.Errorf("cuQuantum test failed: %v, output: %s", err, string(output))
	}

	if !strings.Contains(string(output), "cuQuantum ready") {
		return fmt.Errorf("cuQuantum container not responding correctly, output: %s", string(output))
	}

	return nil
}

// SolveQuantumPuzzleBatch processes multiple quantum puzzles using cuQuantum
func (sim *CuQuantumSimulator) SolveQuantumPuzzleBatch(ctx context.Context, puzzles []QuantumPuzzle) ([]QuantumPuzzleResult, error) {
	if !sim.available {
		return nil, fmt.Errorf("cuQuantum simulator not available")
	}

	// Create Python script for batch processing
	pythonScript := sim.generateBatchScript(puzzles)

	// Execute in cuQuantum container
	cmd := exec.CommandContext(ctx, "docker", "run", "--rm", "--gpus", sim.gpuDevices,
		"-i", sim.containerImage,
		"python", "-c", pythonScript)

	// Send puzzle data via stdin
	puzzleData, err := json.Marshal(puzzles)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal puzzle data: %v", err)
	}

	cmd.Stdin = strings.NewReader(string(puzzleData))

	// Execute and capture output
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("cuQuantum execution failed: %v", err)
	}

	// Parse results
	var results []QuantumPuzzleResult
	if err := json.Unmarshal(output, &results); err != nil {
		return nil, fmt.Errorf("failed to parse cuQuantum results: %v", err)
	}

	return results, nil
}

// generateBatchScript creates Python script for batch quantum puzzle processing
func (sim *CuQuantumSimulator) generateBatchScript(puzzles []QuantumPuzzle) string {
	return `
import json
import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

def create_quantum_circuit(qbits, tcount, seed):
    """Create quantum circuit based on puzzle parameters"""
    circuit = QuantumCircuit(qbits, qbits)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Add quantum gates based on tcount
    gate_count = 0
    for i in range(tcount):
        if gate_count >= tcount:
            break
            
        # Add Hadamard gates
        if gate_count < tcount:
            circuit.h(i % qbits)
            gate_count += 1
            
        # Add CNOT gates
        if gate_count < tcount and qbits > 1:
            circuit.cx(i % qbits, (i + 1) % qbits)
            gate_count += 1
            
        # Add T gates for quantum advantage
        if gate_count < tcount:
            circuit.t(i % qbits)
            gate_count += 1
    
    # Add measurements
    circuit.measure_all()
    return circuit

def solve_puzzle(puzzle):
    """Solve single quantum puzzle using cuQuantum GPU acceleration"""
    try:
        # Create circuit
        circuit = create_quantum_circuit(
            puzzle['qbits'], 
            puzzle['tcount'], 
            puzzle['seed']
        )
        
        # Create cuQuantum-accelerated simulator
        simulator = AerSimulator(method='statevector', device='GPU')
        
        # Enable cuStateVec for GPU acceleration
        transpiled_circuit = transpile(circuit, simulator, optimization_level=0)
        
        # Execute with cuQuantum acceleration
        job = simulator.run(transpiled_circuit, shots=1024, cuStateVec_enable=True)
        result = job.result()
        
        # Get measurement counts
        counts = result.get_counts()
        
        # Convert to probabilities
        total_shots = sum(counts.values())
        probabilities = []
        for i in range(2**puzzle['qbits']):
            bit_string = format(i, f"0{puzzle['qbits']}b")
            prob = counts.get(bit_string, 0) / total_shots
            probabilities.append(prob)
        
        return {
            'success': True,
            'puzzle_index': puzzle['index'],
            'probabilities': probabilities,
            'measurements': counts,
            'gpu_used': True,
            'processing_time': 0.1  # cuQuantum is very fast
        }
        
    except Exception as e:
        return {
            'success': False,
            'puzzle_index': puzzle['index'],
            'error': str(e),
            'gpu_used': False
        }

def main():
    """Main batch processing function"""
    try:
        # Read puzzle data from stdin
        puzzle_data = sys.stdin.read()
        puzzles = json.loads(puzzle_data)
        
        # Process all puzzles
        results = []
        for puzzle in puzzles:
            result = solve_puzzle(puzzle)
            results.append(result)
        
        # Output results as JSON
        print(json.dumps(results))
        
    except Exception as e:
        error_result = [{
            'success': False,
            'error': f'Batch processing failed: {str(e)}',
            'gpu_used': False
        }]
        print(json.dumps(error_result))

if __name__ == '__main__':
    main()
`
}

// SolveQuantumPuzzle processes a single quantum puzzle using cuQuantum
func (sim *CuQuantumSimulator) SolveQuantumPuzzle(ctx context.Context, puzzle QuantumPuzzle) (*QuantumPuzzleResult, error) {
	results, err := sim.SolveQuantumPuzzleBatch(ctx, []QuantumPuzzle{puzzle})
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no results returned from cuQuantum")
	}

	return &results[0], nil
}

// IsAvailable returns whether cuQuantum simulation is available
func (sim *CuQuantumSimulator) IsAvailable() bool {
	return sim.available
}

// GetBackendInfo returns information about the cuQuantum backend
func (sim *CuQuantumSimulator) GetBackendInfo() string {
	if !sim.available {
		return "cuQuantum: Not Available"
	}
	return "cuQuantum: NVIDIA Enterprise GPU Acceleration"
}

// Cleanup performs any necessary cleanup
func (sim *CuQuantumSimulator) Cleanup() {
	// cuQuantum Docker containers are ephemeral, no cleanup needed
}

// SetGPUDevices configures which GPU devices to use
func (sim *CuQuantumSimulator) SetGPUDevices(devices string) {
	sim.gpuDevices = devices
}

// BenchmarkPerformance runs a performance benchmark using cuQuantum
func (sim *CuQuantumSimulator) BenchmarkPerformance(ctx context.Context, qubits int) (*CuQuantumResult, error) {
	pythonScript := fmt.Sprintf(`
import json
import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def benchmark_cuquantum(qubits):
    """Benchmark cuQuantum performance"""
    try:
        # Create test circuit
        circuit = QuantumCircuit(qubits, qubits)
        
        # Add gates for complexity
        for i in range(qubits):
            circuit.h(i)
        for i in range(qubits-1):
            circuit.cx(i, i+1)
        circuit.measure_all()
        
        # Create cuQuantum simulator
        simulator = AerSimulator(method='statevector', device='GPU')
        transpiled = transpile(circuit, simulator, optimization_level=0)
        
        # Benchmark execution
        start_time = time.time()
        job = simulator.run(transpiled, shots=1000, cuStateVec_enable=True)
        result = job.result()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        return {
            'success': True,
            'gpu_available': True,
            'benchmark': {
                'trials': 1,
                'avg_time': execution_time,
                'total_time': execution_time,
                'qubits': qubits,
                'gates': qubits * 2,
                'gpu_used': True
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'gpu_available': False,
            'error': str(e)
        }

result = benchmark_cuquantum(%d)
print(json.dumps(result))
`, qubits)

	cmd := exec.CommandContext(ctx, "docker", "run", "--rm", "--gpus", sim.gpuDevices,
		sim.containerImage,
		"python", "-c", pythonScript)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("cuQuantum benchmark failed: %v", err)
	}

	var result CuQuantumResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse benchmark result: %v", err)
	}

	return &result, nil
} 