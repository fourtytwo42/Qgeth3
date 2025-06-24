//go:build cuda
// +build cuda

package quantum

/*
#cgo windows CFLAGS: -I${SRCDIR}
#cgo windows LDFLAGS: -L${SRCDIR} -lquantum_cuda -lcudart -lcublas
#cgo linux CFLAGS: -I${SRCDIR}
#cgo linux LDFLAGS: -L${SRCDIR} -lquantum_cuda -lcudart -lcublas

// Forward declarations for CUDA functions
typedef struct { double x, y; } cuDoubleComplex;

cuDoubleComplex* cuda_alloc_quantum_state(int n_qubits);
void cuda_free_quantum_state(cuDoubleComplex* d_state);
int cuda_init_quantum_state(cuDoubleComplex* d_state, int n_qubits);
int cuda_apply_hadamard(cuDoubleComplex* d_state, int n_qubits, int target);
int cuda_apply_t_gate(cuDoubleComplex* d_state, int n_qubits, int target);
int cuda_apply_cnot(cuDoubleComplex* d_state, int n_qubits, int control, int target);
int cuda_measure_state(cuDoubleComplex* d_state, double* h_probabilities, int n_qubits);
int cuda_get_device_info(int device_id, char* name, int* major, int* minor, size_t* memory);
int cuda_set_device(int device_id);
*/
import "C"
import (
	"fmt"
	"math/rand"
	"time"
	"unsafe"
)

// QuantumStateGPU represents a quantum state vector on GPU
type QuantumStateGPU struct {
	nQubits     int
	deviceState unsafe.Pointer // Pointer to cuDoubleComplex array on GPU
	deviceID    int
}

// CUDADeviceInfo holds information about a CUDA device
type CUDADeviceInfo struct {
	Name        string
	Major       int
	Minor       int
	TotalMemory uint64
	DeviceID    int
}

// InitCUDA initializes CUDA and sets the device
func InitCUDA(deviceID int) error {
	result := C.cuda_set_device(C.int(deviceID))
	if result == 0 {
		return fmt.Errorf("failed to set CUDA device %d", deviceID)
	}
	return nil
}

// GetCUDADeviceInfo retrieves information about a CUDA device
func GetCUDADeviceInfo(deviceID int) (*CUDADeviceInfo, error) {
	var name [256]C.char
	var major, minor C.int
	var memory C.size_t

	result := C.cuda_get_device_info(C.int(deviceID), &name[0], &major, &minor, &memory)
	if result == 0 {
		return nil, fmt.Errorf("failed to get info for CUDA device %d", deviceID)
	}

	return &CUDADeviceInfo{
		Name:        C.GoString(&name[0]),
		Major:       int(major),
		Minor:       int(minor),
		TotalMemory: uint64(memory),
		DeviceID:    deviceID,
	}, nil
}

// NewQuantumStateGPU creates a new quantum state on GPU
func NewQuantumStateGPU(nQubits int, deviceID int) (*QuantumStateGPU, error) {
	if nQubits > 16 {
		return nil, fmt.Errorf("maximum 16 qubits supported, got %d", nQubits)
	}

	// Set CUDA device
	if err := InitCUDA(deviceID); err != nil {
		return nil, fmt.Errorf("CUDA init failed: %w", err)
	}

	// Allocate GPU memory
	deviceState := C.cuda_alloc_quantum_state(C.int(nQubits))
	if deviceState == nil {
		return nil, fmt.Errorf("failed to allocate GPU memory for %d qubits", nQubits)
	}

	state := &QuantumStateGPU{
		nQubits:     nQubits,
		deviceState: unsafe.Pointer(deviceState),
		deviceID:    deviceID,
	}

	// Initialize to |0⟩^n state
	if err := state.Initialize(); err != nil {
		state.Free()
		return nil, fmt.Errorf("failed to initialize quantum state: %w", err)
	}

	return state, nil
}

// Free releases GPU memory
func (q *QuantumStateGPU) Free() {
	if q.deviceState != nil {
		C.cuda_free_quantum_state((*C.cuDoubleComplex)(q.deviceState))
		q.deviceState = nil
	}
}

// Initialize sets the quantum state to |0⟩^n
func (q *QuantumStateGPU) Initialize() error {
	result := C.cuda_init_quantum_state((*C.cuDoubleComplex)(q.deviceState), C.int(q.nQubits))
	if result == 0 {
		return fmt.Errorf("failed to initialize quantum state on GPU")
	}
	return nil
}

// ApplyHadamard applies a Hadamard gate to the specified qubit
func (q *QuantumStateGPU) ApplyHadamard(target int) error {
	if target < 0 || target >= q.nQubits {
		return fmt.Errorf("invalid qubit index %d for %d-qubit system", target, q.nQubits)
	}

	result := C.cuda_apply_hadamard((*C.cuDoubleComplex)(q.deviceState), C.int(q.nQubits), C.int(target))
	if result == 0 {
		return fmt.Errorf("failed to apply Hadamard gate to qubit %d", target)
	}
	return nil
}

// ApplyTGate applies a T gate to the specified qubit
func (q *QuantumStateGPU) ApplyTGate(target int) error {
	if target < 0 || target >= q.nQubits {
		return fmt.Errorf("invalid qubit index %d for %d-qubit system", target, q.nQubits)
	}

	result := C.cuda_apply_t_gate((*C.cuDoubleComplex)(q.deviceState), C.int(q.nQubits), C.int(target))
	if result == 0 {
		return fmt.Errorf("failed to apply T gate to qubit %d", target)
	}
	return nil
}

// ApplyCNOT applies a CNOT gate with the specified control and target qubits
func (q *QuantumStateGPU) ApplyCNOT(control, target int) error {
	if control < 0 || control >= q.nQubits {
		return fmt.Errorf("invalid control qubit index %d for %d-qubit system", control, q.nQubits)
	}
	if target < 0 || target >= q.nQubits {
		return fmt.Errorf("invalid target qubit index %d for %d-qubit system", target, q.nQubits)
	}
	if control == target {
		return fmt.Errorf("control and target qubits must be different")
	}

	result := C.cuda_apply_cnot((*C.cuDoubleComplex)(q.deviceState), C.int(q.nQubits), C.int(control), C.int(target))
	if result == 0 {
		return fmt.Errorf("failed to apply CNOT gate (control: %d, target: %d)", control, target)
	}
	return nil
}

// Measure performs a measurement and returns the outcome probabilities
func (q *QuantumStateGPU) Measure() ([]float64, error) {
	totalStates := 1 << q.nQubits
	probabilities := make([]float64, totalStates)

	result := C.cuda_measure_state(
		(*C.cuDoubleComplex)(q.deviceState),
		(*C.double)(&probabilities[0]),
		C.int(q.nQubits),
	)

	if result == 0 {
		return nil, fmt.Errorf("failed to measure quantum state")
	}

	return probabilities, nil
}

// SampleMeasurement samples a measurement outcome based on probabilities
func (q *QuantumStateGPU) SampleMeasurement() (int, error) {
	probabilities, err := q.Measure()
	if err != nil {
		return 0, err
	}

	// Sample based on probabilities
	r := rand.Float64()
	cumulative := 0.0

	for i, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			return i, nil
		}
	}

	// Fallback to last state (should not happen with proper normalization)
	return len(probabilities) - 1, nil
}

// ExecuteQuantumCircuit executes a quantum circuit defined by gate sequence
func (q *QuantumStateGPU) ExecuteQuantumCircuit(gates []QuantumGate) error {
	// Reset to |0⟩^n state
	if err := q.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize circuit: %w", err)
	}

	// Apply gates in sequence
	for i, gate := range gates {
		var err error

		switch gate.Type {
		case "H", "hadamard":
			err = q.ApplyHadamard(gate.Target)
		case "T", "t_gate":
			err = q.ApplyTGate(gate.Target)
		case "CNOT", "cnot":
			if gate.Control == -1 {
				return fmt.Errorf("CNOT gate at position %d missing control qubit", i)
			}
			err = q.ApplyCNOT(gate.Control, gate.Target)
		default:
			return fmt.Errorf("unsupported gate type '%s' at position %d", gate.Type, i)
		}

		if err != nil {
			return fmt.Errorf("failed to apply gate %d (%s): %w", i, gate.Type, err)
		}
	}

	return nil
}

// QuantumGate represents a quantum gate operation
type QuantumGate struct {
	Type    string // "H", "T", "CNOT"
	Target  int    // Target qubit
	Control int    // Control qubit (for CNOT, -1 if not applicable)
}

// GenerateRandomQuantumCircuit generates a random quantum circuit for mining
func GenerateRandomQuantumCircuit(nQubits, nGates int, seed int64) []QuantumGate {
	rng := rand.New(rand.NewSource(seed))
	gates := make([]QuantumGate, nGates)

	// Gate types with probabilities: H (40%), T (40%), CNOT (20%)
	gateTypes := []string{"H", "T", "CNOT"}
	gateProbs := []float64{0.4, 0.8, 1.0} // Cumulative probabilities

	for i := 0; i < nGates; i++ {
		r := rng.Float64()
		gateType := "H"

		for j, prob := range gateProbs {
			if r <= prob {
				gateType = gateTypes[j]
				break
			}
		}

		target := rng.Intn(nQubits)
		control := -1

		if gateType == "CNOT" {
			// Choose different control and target
			control = rng.Intn(nQubits)
			for control == target {
				control = rng.Intn(nQubits)
			}
		}

		gates[i] = QuantumGate{
			Type:    gateType,
			Target:  target,
			Control: control,
		}
	}

	return gates
}

// SimulateQuantumPuzzleGPU solves a quantum puzzle using GPU acceleration
func SimulateQuantumPuzzleGPU(puzzleIndex int, workHash string, qnonce uint64,
	nQubits, nGates int, deviceID int) ([]byte, error) {

	// Create quantum state on GPU
	state, err := NewQuantumStateGPU(nQubits, deviceID)
	if err != nil {
		return nil, fmt.Errorf("failed to create GPU quantum state: %w", err)
	}
	defer state.Free()

	// Generate deterministic circuit based on puzzle parameters
	seed := int64(puzzleIndex) ^ int64(qnonce) ^ hashStringToInt64(workHash)
	circuit := GenerateRandomQuantumCircuit(nQubits, nGates, seed)

	// Execute the quantum circuit on GPU
	if err := state.ExecuteQuantumCircuit(circuit); err != nil {
		return nil, fmt.Errorf("failed to execute quantum circuit: %w", err)
	}

	// Sample measurement outcome
	outcome, err := state.SampleMeasurement()
	if err != nil {
		return nil, fmt.Errorf("failed to measure quantum state: %w", err)
	}

	// Convert outcome to bytes (16 qubits = 2 bytes)
	outcomeBytes := make([]byte, (nQubits+7)/8)
	for i := 0; i < len(outcomeBytes); i++ {
		outcomeBytes[i] = byte(outcome >> (i * 8))
	}

	return outcomeBytes, nil
}

// Helper function to convert string to int64 for seeding
func hashStringToInt64(s string) int64 {
	var hash int64
	for i, c := range s {
		hash = hash*31 + int64(c) + int64(i)
	}
	return hash
}

// BenchmarkGPUPerformance tests GPU quantum simulation performance
func BenchmarkGPUPerformance(deviceID int) (*GPUBenchmarkResult, error) {
	const nQubits = 16
	const nGates = 100
	const nTrials = 10

	deviceInfo, err := GetCUDADeviceInfo(deviceID)
	if err != nil {
		return nil, fmt.Errorf("failed to get device info: %w", err)
	}

	var totalTime time.Duration
	var successful int

	for trial := 0; trial < nTrials; trial++ {
		start := time.Now()

		state, err := NewQuantumStateGPU(nQubits, deviceID)
		if err != nil {
			continue
		}

		circuit := GenerateRandomQuantumCircuit(nQubits, nGates, int64(trial))

		if err := state.ExecuteQuantumCircuit(circuit); err != nil {
			state.Free()
			continue
		}

		_, err = state.SampleMeasurement()
		state.Free()

		if err == nil {
			totalTime += time.Since(start)
			successful++
		}
	}

	if successful == 0 {
		return nil, fmt.Errorf("all GPU benchmark trials failed")
	}

	avgTime := totalTime / time.Duration(successful)
	puzzlesPerSec := float64(time.Second) / float64(avgTime)

	return &GPUBenchmarkResult{
		DeviceInfo:       *deviceInfo,
		AvgPuzzleTime:    avgTime,
		PuzzlesPerSecond: puzzlesPerSec,
		SuccessfulTrials: successful,
		TotalTrials:      nTrials,
		Qubits:           nQubits,
		Gates:            nGates,
	}, nil
}

// GPUBenchmarkResult holds GPU performance benchmark results
type GPUBenchmarkResult struct {
	DeviceInfo       CUDADeviceInfo
	AvgPuzzleTime    time.Duration
	PuzzlesPerSecond float64
	SuccessfulTrials int
	TotalTrials      int
	Qubits           int
	Gates            int
}
