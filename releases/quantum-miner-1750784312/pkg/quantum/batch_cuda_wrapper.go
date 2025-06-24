//go:build cuda
// +build cuda

package quantum

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lcuda_batch_optimizer -lcudart -lcurand

#include <stdio.h>
#include <stdlib.h>

// Forward declarations for optimized batch CUDA functions
typedef struct BatchQuantumState BatchQuantumState;
typedef void* cudaStream_t;

// Batch processing functions
BatchQuantumState* cuda_alloc_batch_state(int batch_size, int n_qubits);
void cuda_free_batch_state(BatchQuantumState* batch);

int cuda_batch_init_async(BatchQuantumState* batch, cudaStream_t stream);
int cuda_batch_hadamard_async(BatchQuantumState* batch, int* targets, cudaStream_t stream);
int cuda_batch_t_gate_async(BatchQuantumState* batch, int* targets, cudaStream_t stream);
int cuda_batch_measure_async(BatchQuantumState* batch, double* h_probs, cudaStream_t stream);

// Stream management
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

// BatchCudaProcessor - High-performance batch quantum processor
type BatchCudaProcessor struct {
	batchState  *C.BatchQuantumState
	streams     []C.cudaStream_t
	batchSize   int
	nQubits     int
	deviceID    int
	streamCount int
	mu          sync.RWMutex
	initialized bool
}

// BatchProcessingResult contains results from batch processing
type BatchProcessingResult struct {
	Outcomes        [][]byte
	ProcessingTime  time.Duration
	BatchSize       int
	GPU_Utilization float64
}

// NewBatchCudaProcessor creates an optimized batch processor
func NewBatchCudaProcessor(batchSize, nQubits, deviceID int) (*BatchCudaProcessor, error) {
	if batchSize > 1024 {
		batchSize = 1024 // Hardware limit
	}

	// Optimal stream count based on hardware
	streamCount := runtime.NumCPU()
	if streamCount > 16 {
		streamCount = 16
	}

	processor := &BatchCudaProcessor{
		batchSize:   batchSize,
		nQubits:     nQubits,
		deviceID:    deviceID,
		streamCount: streamCount,
	}

	if err := processor.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize batch processor: %w", err)
	}

	return processor, nil
}

// Initialize the batch processor
func (b *BatchCudaProcessor) initialize() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.initialized {
		return nil
	}

	// Allocate batch state on GPU
	b.batchState = C.cuda_alloc_batch_state(C.int(b.batchSize), C.int(b.nQubits))
	if b.batchState == nil {
		return fmt.Errorf("failed to allocate batch state on GPU")
	}

	// Create CUDA streams for async processing
	b.streams = make([]C.cudaStream_t, b.streamCount)
	for i := 0; i < b.streamCount; i++ {
		stream := C.cuda_create_stream()
		if stream == nil {
			// Clean up previously created streams
			for j := 0; j < i; j++ {
				C.cuda_destroy_stream(b.streams[j])
			}
			C.cuda_free_batch_state(b.batchState)
			return fmt.Errorf("failed to create CUDA stream %d", i)
		}
		b.streams[i] = stream
	}

	b.initialized = true
	log.Printf("🚀 Batch CUDA Processor initialized: %d puzzles, %d streams, %d qubits",
		b.batchSize, b.streamCount, b.nQubits)

	return nil
}

// ProcessQuantumPuzzlesBatch - High-throughput batch processing
func (b *BatchCudaProcessor) ProcessQuantumPuzzlesBatch(workHash string, qnonce uint64,
	nGates, nPuzzles int) (*BatchProcessingResult, error) {

	if !b.initialized {
		return nil, fmt.Errorf("batch processor not initialized")
	}

	if nPuzzles > b.batchSize {
		nPuzzles = b.batchSize
	}

	start := time.Now()

	// Generate gate sequences for all puzzles (deterministic)
	gateTargets := make([]C.int, nPuzzles)
	for i := 0; i < nPuzzles; i++ {
		seed := fmt.Sprintf("%s_%d_%d", workHash, qnonce, i)
		rng := rand.New(rand.NewSource(hashToSeed(seed)))
		gateTargets[i] = C.int(rng.Intn(b.nQubits))
	}

	// Select stream for this batch
	streamIdx := int(qnonce) % b.streamCount
	stream := b.streams[streamIdx]

	// Step 1: Initialize all quantum states (async, no sync)
	if C.cuda_batch_init_async(b.batchState, stream) == 0 {
		return nil, fmt.Errorf("failed to initialize batch quantum states")
	}

	// Step 2: Apply quantum gates (async, no sync between gates)
	hadamardGates := nGates / 3
	tGates := nGates - hadamardGates

	// Apply Hadamard gates
	for gate := 0; gate < hadamardGates; gate++ {
		// Update targets for each gate
		for i := 0; i < nPuzzles; i++ {
			seed := fmt.Sprintf("%s_%d_%d_%d_H", workHash, qnonce, i, gate)
			rng := rand.New(rand.NewSource(hashToSeed(seed)))
			gateTargets[i] = C.int(rng.Intn(b.nQubits))
		}

		if C.cuda_batch_hadamard_async(b.batchState, (*C.int)(unsafe.Pointer(&gateTargets[0])), stream) == 0 {
			return nil, fmt.Errorf("failed to apply batch Hadamard gate %d", gate)
		}
	}

	// Apply T gates
	for gate := 0; gate < tGates; gate++ {
		// Update targets for each gate
		for i := 0; i < nPuzzles; i++ {
			seed := fmt.Sprintf("%s_%d_%d_%d_T", workHash, qnonce, i, gate)
			rng := rand.New(rand.NewSource(hashToSeed(seed)))
			gateTargets[i] = C.int(rng.Intn(b.nQubits))
		}

		if C.cuda_batch_t_gate_async(b.batchState, (*C.int)(unsafe.Pointer(&gateTargets[0])), stream) == 0 {
			return nil, fmt.Errorf("failed to apply batch T gate %d", gate)
		}
	}

	// Step 3: Measure all states (sync only this stream, not entire device)
	totalStates := 1 << b.nQubits
	probabilities := make([]C.double, nPuzzles*totalStates)

	if C.cuda_batch_measure_async(b.batchState, (*C.double)(unsafe.Pointer(&probabilities[0])), stream) == 0 {
		return nil, fmt.Errorf("failed to measure batch quantum states")
	}

	// Convert probabilities to outcomes
	outcomes := make([][]byte, nPuzzles)
	for i := 0; i < nPuzzles; i++ {
		outcome := make([]byte, (b.nQubits+7)/8)

		// Sample from probability distribution
		probOffset := i * totalStates
		cumulative := 0.0
		sample := rand.Float64()

		for state := 0; state < totalStates; state++ {
			cumulative += float64(probabilities[probOffset+state])
			if cumulative >= sample {
				// Encode state into outcome bytes
				for bit := 0; bit < b.nQubits; bit++ {
					if (state>>bit)&1 == 1 {
						byteIdx := bit / 8
						bitIdx := bit % 8
						outcome[byteIdx] |= 1 << bitIdx
					}
				}
				break
			}
		}

		outcomes[i] = outcome
	}

	processingTime := time.Since(start)

	// Calculate GPU utilization estimate
	theoretical_min := time.Duration(nPuzzles*nGates) * time.Microsecond
	utilization := float64(theoretical_min) / float64(processingTime) * 100
	if utilization > 100 {
		utilization = 100
	}

	return &BatchProcessingResult{
		Outcomes:        outcomes,
		ProcessingTime:  processingTime,
		BatchSize:       nPuzzles,
		GPU_Utilization: utilization,
	}, nil
}

// Cleanup releases GPU resources
func (b *BatchCudaProcessor) Cleanup() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.initialized {
		return
	}

	// Destroy streams
	for _, stream := range b.streams {
		C.cuda_destroy_stream(stream)
	}

	// Free batch state
	if b.batchState != nil {
		C.cuda_free_batch_state(b.batchState)
		b.batchState = nil
	}

	b.initialized = false
	log.Printf("🧹 Batch CUDA Processor cleaned up")
}

// GetOptimalBatchSize calculates optimal batch size based on available GPU memory
func GetOptimalBatchSize(nQubits int) (int, error) {
	// Each puzzle needs 2^nQubits complex numbers (16 bytes each)
	stateSize := (1 << nQubits) * 16 // bytes per puzzle

	// Assume we want to use 80% of GPU memory for batch processing
	totalGPUMemory := int64(8 * 1024 * 1024 * 1024) // 8GB typical
	usableMemory := int64(float64(totalGPUMemory) * 0.8)

	maxBatchSize := int(usableMemory / int64(stateSize))

	// Cap at hardware limit
	if maxBatchSize > 1024 {
		maxBatchSize = 1024
	}

	// Ensure power of 2 for optimal memory alignment
	optimalBatch := 1
	for optimalBatch < maxBatchSize {
		optimalBatch *= 2
	}
	optimalBatch /= 2 // Back off one power of 2 for safety

	if optimalBatch < 1 {
		optimalBatch = 1
	}

	return optimalBatch, nil
}

// hashToSeed converts a string to a deterministic seed
func hashToSeed(s string) int64 {
	hash := int64(0)
	for _, c := range s {
		hash = hash*31 + int64(c)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}

// BatchQuantumSimulator replaces the old individual puzzle approach
type BatchQuantumSimulator struct {
	processor *BatchCudaProcessor
	deviceID  int
	nQubits   int
}

// NewBatchQuantumSimulator creates a new batch-optimized simulator
func NewBatchQuantumSimulator(deviceID, nQubits int) (*BatchQuantumSimulator, error) {
	optimalBatch, err := GetOptimalBatchSize(nQubits)
	if err != nil {
		return nil, err
	}

	processor, err := NewBatchCudaProcessor(optimalBatch, nQubits, deviceID)
	if err != nil {
		return nil, err
	}

	return &BatchQuantumSimulator{
		processor: processor,
		deviceID:  deviceID,
		nQubits:   nQubits,
	}, nil
}

// BatchSimulateQuantumPuzzles implements the HybridQuantumSimulator interface
func (b *BatchQuantumSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	result, err := b.processor.ProcessQuantumPuzzlesBatch(workHash, qnonce, nGates, nPuzzles)
	if err != nil {
		return nil, err
	}

	log.Printf("⚡ Batch processed %d puzzles in %.2fms (%.1f%% GPU utilization)",
		result.BatchSize,
		float64(result.ProcessingTime.Nanoseconds())/1e6,
		result.GPU_Utilization)

	return result.Outcomes, nil
}

// Cleanup releases resources
func (b *BatchQuantumSimulator) Cleanup() {
	if b.processor != nil {
		b.processor.Cleanup()
	}
}
