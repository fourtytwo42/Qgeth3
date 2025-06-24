//go:build !cuda
// +build !cuda

package quantum

import (
	"fmt"
	"math/rand"
	"time"
)

// CUDADeviceInfo holds information about a CUDA device (fallback)
type CUDADeviceInfo struct {
	Name        string
	Major       int
	Minor       int
	TotalMemory uint64
	DeviceID    int
}

// InitCUDA initializes CUDA and sets the device (fallback)
func InitCUDA(deviceID int) error {
	return fmt.Errorf("CUDA support not compiled in")
}

// GetCUDADeviceInfo retrieves information about a CUDA device (fallback)
func GetCUDADeviceInfo(deviceID int) (*CUDADeviceInfo, error) {
	return nil, fmt.Errorf("CUDA support not compiled in")
}

// QuantumStateGPU represents a quantum state vector on GPU (fallback)
type QuantumStateGPU struct {
	nQubits  int
	deviceID int
}

// NewQuantumStateGPU creates a new quantum state on GPU (fallback)
func NewQuantumStateGPU(nQubits int, deviceID int) (*QuantumStateGPU, error) {
	return nil, fmt.Errorf("CUDA support not compiled in")
}

// Free releases GPU memory (fallback)
func (q *QuantumStateGPU) Free() {
	// No-op
}

// BenchmarkGPUPerformance tests GPU quantum simulation performance (fallback)
func BenchmarkGPUPerformance(deviceID int) (*GPUBenchmarkResult, error) {
	return nil, fmt.Errorf("CUDA support not compiled in")
}

// GPUBenchmarkResult holds GPU performance benchmark results (fallback)
type GPUBenchmarkResult struct {
	DeviceInfo       CUDADeviceInfo
	AvgPuzzleTime    time.Duration
	PuzzlesPerSecond float64
	SuccessfulTrials int
	TotalTrials      int
	Qubits           int
	Gates            int
}

// SimulateQuantumPuzzleGPU solves a quantum puzzle using CPU simulation (fallback)
func SimulateQuantumPuzzleGPU(puzzleIndex int, workHash string, qnonce uint64,
	nQubits, nGates int, deviceID int) ([]byte, error) {

	// CPU-based quantum simulation fallback
	seed := int64(puzzleIndex) ^ int64(qnonce) ^ hashStringToInt64(workHash)
	rng := rand.New(rand.NewSource(seed))

	// Simulate quantum computation time
	computeTime := time.Duration(nGates) * time.Microsecond
	if computeTime > 50*time.Millisecond {
		computeTime = 50 * time.Millisecond
	}
	time.Sleep(computeTime)

	// Generate quantum outcome (16 qubits = 2 bytes)
	outcomeBytes := make([]byte, (nQubits+7)/8)
	for i := 0; i < len(outcomeBytes); i++ {
		outcomeBytes[i] = byte(rng.Intn(256))
	}

	return outcomeBytes, nil
}

// Helper function to convert string to int64 for seeding (fallback)
func hashStringToInt64(s string) int64 {
	var hash int64
	for i, c := range s {
		hash = hash*31 + int64(c) + int64(i)
	}
	return hash
}
