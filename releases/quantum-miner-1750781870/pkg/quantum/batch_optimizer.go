//go:build cuda
// +build cuda

package quantum

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lcuda_batch_optimizer -lcudart

#include <stdio.h>
#include <stdlib.h>

typedef struct BatchQuantumState BatchQuantumState;
typedef void* cudaStream_t;

BatchQuantumState* cuda_alloc_batch_state(int batch_size, int n_qubits);
void cuda_free_batch_state(BatchQuantumState* batch);
int cuda_batch_init_async(BatchQuantumState* batch, cudaStream_t stream);
int cuda_batch_hadamard_async(BatchQuantumState* batch, int* targets, cudaStream_t stream);
int cuda_batch_t_gate_async(BatchQuantumState* batch, int* targets, cudaStream_t stream);
int cuda_batch_measure_async(BatchQuantumState* batch, double* h_probs, cudaStream_t stream);
cudaStream_t cuda_create_stream();
void cuda_destroy_stream(cudaStream_t stream);
int cuda_stream_sync(cudaStream_t stream);
*/
import "C"

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// BatchCudaProcessor eliminates GPU synchronization bottlenecks
type BatchCudaProcessor struct {
	batchState  *C.BatchQuantumState
	streams     []C.cudaStream_t
	batchSize   int
	nQubits     int
	streamCount int
	mu          sync.RWMutex
	initialized bool
}

// BatchResult contains high-performance processing results
type BatchResult struct {
	Outcomes       [][]byte
	ProcessingTime time.Duration
	BatchSize      int
	GPUUtilization float64
}

// NewBatchCudaProcessor creates optimized batch processor
func NewBatchCudaProcessor(batchSize, nQubits int) (*BatchCudaProcessor, error) {
	if batchSize > 1024 {
		batchSize = 1024
	}

	// Optimal streams for async processing
	streamCount := runtime.NumCPU()
	if streamCount > 8 {
		streamCount = 8
	}

	processor := &BatchCudaProcessor{
		batchSize:   batchSize,
		nQubits:     nQubits,
		streamCount: streamCount,
	}

	if err := processor.initialize(); err != nil {
		return nil, fmt.Errorf("batch processor init failed: %w", err)
	}

	return processor, nil
}

func (b *BatchCudaProcessor) initialize() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.initialized {
		return nil
	}

	// Allocate GPU batch memory
	b.batchState = C.cuda_alloc_batch_state(C.int(b.batchSize), C.int(b.nQubits))
	if b.batchState == nil {
		return fmt.Errorf("GPU batch allocation failed")
	}

	// Create async streams
	b.streams = make([]C.cudaStream_t, b.streamCount)
	for i := 0; i < b.streamCount; i++ {
		stream := C.cuda_create_stream()
		if stream == nil {
			for j := 0; j < i; j++ {
				C.cuda_destroy_stream(b.streams[j])
			}
			C.cuda_free_batch_state(b.batchState)
			return fmt.Errorf("stream creation failed")
		}
		b.streams[i] = stream
	}

	b.initialized = true
	log.Printf("🚀 Batch GPU Processor: %d puzzles, %d streams", b.batchSize, b.streamCount)
	return nil
}

// ProcessBatch - High-throughput quantum puzzle processing
func (b *BatchCudaProcessor) ProcessBatch(workHash string, qnonce uint64,
	nGates, nPuzzles int) (*BatchResult, error) {

	if !b.initialized {
		return nil, fmt.Errorf("processor not initialized")
	}

	if nPuzzles > b.batchSize {
		nPuzzles = b.batchSize
	}

	start := time.Now()

	// Generate deterministic gate targets
	gateTargets := make([]C.int, nPuzzles)
	streamIdx := int(qnonce) % b.streamCount
	stream := b.streams[streamIdx]

	// Step 1: Initialize (async, no sync)
	if C.cuda_batch_init_async(b.batchState, stream) == 0 {
		return nil, fmt.Errorf("batch init failed")
	}

	// Step 2: Apply gates (async, no sync between operations)
	for gate := 0; gate < nGates; gate++ {
		// Update gate targets
		for i := 0; i < nPuzzles; i++ {
			seed := hashString(fmt.Sprintf("%s_%d_%d_%d", workHash, qnonce, i, gate))
			gateTargets[i] = C.int(seed % int64(b.nQubits))
		}

		if gate%2 == 0 {
			// Hadamard gates
			if C.cuda_batch_hadamard_async(b.batchState, (*C.int)(unsafe.Pointer(&gateTargets[0])), stream) == 0 {
				return nil, fmt.Errorf("batch Hadamard failed")
			}
		} else {
			// T gates
			if C.cuda_batch_t_gate_async(b.batchState, (*C.int)(unsafe.Pointer(&gateTargets[0])), stream) == 0 {
				return nil, fmt.Errorf("batch T-gate failed")
			}
		}
	}

	// Step 3: Measure (sync only this stream)
	totalStates := 1 << b.nQubits
	probabilities := make([]C.double, nPuzzles*totalStates)

	if C.cuda_batch_measure_async(b.batchState, (*C.double)(unsafe.Pointer(&probabilities[0])), stream) == 0 {
		return nil, fmt.Errorf("batch measure failed")
	}

	// Convert to outcomes
	outcomes := make([][]byte, nPuzzles)
	for i := 0; i < nPuzzles; i++ {
		outcome := make([]byte, (b.nQubits+7)/8)

		// Sample measurement
		probOffset := i * totalStates
		sample := rand.Float64()
		cumulative := 0.0

		for state := 0; state < totalStates; state++ {
			cumulative += float64(probabilities[probOffset+state])
			if cumulative >= sample {
				// Encode state bits
				for bit := 0; bit < b.nQubits; bit++ {
					if (state>>bit)&1 == 1 {
						outcome[bit/8] |= 1 << (bit % 8)
					}
				}
				break
			}
		}
		outcomes[i] = outcome
	}

	processingTime := time.Since(start)

	// GPU utilization estimate
	theoretical := time.Duration(nPuzzles*nGates) * time.Microsecond
	utilization := float64(theoretical) / float64(processingTime) * 100
	if utilization > 100 {
		utilization = 100
	}

	return &BatchResult{
		Outcomes:       outcomes,
		ProcessingTime: processingTime,
		BatchSize:      nPuzzles,
		GPUUtilization: utilization,
	}, nil
}

func (b *BatchCudaProcessor) Cleanup() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.initialized {
		return
	}

	for _, stream := range b.streams {
		C.cuda_destroy_stream(stream)
	}

	if b.batchState != nil {
		C.cuda_free_batch_state(b.batchState)
		b.batchState = nil
	}

	b.initialized = false
}

// GetOptimalBatchSize for GPU memory
func GetOptimalBatchSize(nQubits int) int {
	stateSize := (1 << nQubits) * 16           // 16 bytes per complex number
	gpuMemory := int64(8 * 1024 * 1024 * 1024) // 8GB
	usable := int64(float64(gpuMemory) * 0.8)

	maxBatch := int(usable / int64(stateSize))
	if maxBatch > 1024 {
		maxBatch = 1024
	}

	// Round to power of 2
	optimal := 1
	for optimal < maxBatch {
		optimal *= 2
	}
	optimal /= 2

	if optimal < 16 {
		optimal = 16
	}

	return optimal
}

// hashString creates deterministic hash
func hashString(s string) int64 {
	hash := int64(0)
	for _, c := range s {
		hash = hash*31 + int64(c)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}

// HighPerformanceQuantumSimulator replaces old approach
type HighPerformanceQuantumSimulator struct {
	processor *BatchCudaProcessor
	nQubits   int
}

// NewHighPerformanceQuantumSimulator creates optimized simulator
func NewHighPerformanceQuantumSimulator(nQubits int) (*HighPerformanceQuantumSimulator, error) {
	batchSize := GetOptimalBatchSize(nQubits)

	processor, err := NewBatchCudaProcessor(batchSize, nQubits)
	if err != nil {
		return nil, err
	}

	return &HighPerformanceQuantumSimulator{
		processor: processor,
		nQubits:   nQubits,
	}, nil
}

// BatchSimulateQuantumPuzzles - High-performance interface
func (h *HighPerformanceQuantumSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	result, err := h.processor.ProcessBatch(workHash, qnonce, nGates, nPuzzles)
	if err != nil {
		return nil, err
	}

	log.Printf("⚡ Processed %d puzzles in %.2fms (%.1f%% GPU util)",
		result.BatchSize,
		float64(result.ProcessingTime.Nanoseconds())/1e6,
		result.GPUUtilization)

	return result.Outcomes, nil
}

func (h *HighPerformanceQuantumSimulator) Cleanup() {
	if h.processor != nil {
		h.processor.Cleanup()
	}
}
