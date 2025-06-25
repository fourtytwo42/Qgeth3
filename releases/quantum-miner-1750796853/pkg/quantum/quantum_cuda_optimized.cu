#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel for batch quantum state vector simulation
// Processes multiple puzzles simultaneously with minimal synchronization

#define MAX_QUBITS 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1024
#define MAX_STREAMS 16

// Optimized batch quantum state structure
typedef struct {
    cuDoubleComplex* states;     // Batch of quantum states
    int* puzzle_offsets;         // Offset for each puzzle in batch
    int* gate_counts;           // Gates processed per puzzle
    int batch_size;             // Number of puzzles in batch
    int n_qubits;              // Qubits per puzzle
    int total_states;          // States per puzzle (2^n_qubits)
} BatchQuantumState;

// Complex number operations (optimized)
__device__ __forceinline__ cuDoubleComplex complex_mult_opt(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        __fma_rn(a.x, b.x, -a.y * b.y),
        __fma_rn(a.x, b.y, a.y * b.x)
    );
}

__device__ __forceinline__ cuDoubleComplex complex_add_opt(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

// Batch Hadamard gate kernel - processes multiple puzzles simultaneously
__global__ void batch_apply_hadamard_gates(cuDoubleComplex* batch_states, 
                                          int* puzzle_offsets, 
                                          int* target_qubits,
                                          int batch_size, 
                                          int n_qubits) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int puzzle_idx = global_idx / (1 << (n_qubits - 1));
    int state_idx = global_idx % (1 << (n_qubits - 1));
    
    if (puzzle_idx >= batch_size) return;
    
    int total_states = 1 << n_qubits;
    int puzzle_offset = puzzle_offsets[puzzle_idx];
    int target = target_qubits[puzzle_idx];
    int bit_mask = 1 << target;
    
    if (state_idx < total_states / 2) {
        int i0 = (state_idx & ~bit_mask) + puzzle_offset;
        int i1 = (state_idx | bit_mask) + puzzle_offset;
        
        if (i0 < i1) {
            cuDoubleComplex amp0 = batch_states[i0];
            cuDoubleComplex amp1 = batch_states[i1];
            
            // Hadamard matrix: (1/√2) * [[1, 1], [1, -1]]
            double inv_sqrt2 = 0.7071067811865476; // Precomputed 1/√2
            
            batch_states[i0] = make_cuDoubleComplex(
                inv_sqrt2 * (amp0.x + amp1.x),
                inv_sqrt2 * (amp0.y + amp1.y)
            );
            batch_states[i1] = make_cuDoubleComplex(
                inv_sqrt2 * (amp0.x - amp1.x),
                inv_sqrt2 * (amp0.y - amp1.y)
            );
        }
    }
}

// Batch T gate kernel - optimized for multiple puzzles
__global__ void batch_apply_t_gates(cuDoubleComplex* batch_states, 
                                   int* puzzle_offsets, 
                                   int* target_qubits,
                                   int batch_size, 
                                   int n_qubits) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int puzzle_idx = global_idx / (1 << n_qubits);
    int state_idx = global_idx % (1 << n_qubits);
    
    if (puzzle_idx >= batch_size) return;
    
    int puzzle_offset = puzzle_offsets[puzzle_idx];
    int target = target_qubits[puzzle_idx];
    int bit_mask = 1 << target;
    
    if (state_idx & bit_mask) {  // If target qubit is |1⟩
        int global_state_idx = puzzle_offset + state_idx;
        
        // T gate: multiply by e^(iπ/4) = (1+i)/√2
        cuDoubleComplex t_phase = make_cuDoubleComplex(0.7071067811865476, 0.7071067811865476);
        batch_states[global_state_idx] = complex_mult_opt(batch_states[global_state_idx], t_phase);
    }
}

// Batch CNOT gate kernel
__global__ void batch_apply_cnot_gates(cuDoubleComplex* batch_states, 
                                      int* puzzle_offsets, 
                                      int* control_qubits,
                                      int* target_qubits,
                                      int batch_size, 
                                      int n_qubits) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int puzzle_idx = global_idx / (1 << (n_qubits - 1));
    int state_idx = global_idx % (1 << (n_qubits - 1));
    
    if (puzzle_idx >= batch_size) return;
    
    int puzzle_offset = puzzle_offsets[puzzle_idx];
    int control = control_qubits[puzzle_idx];
    int target = target_qubits[puzzle_idx];
    int control_mask = 1 << control;
    int target_mask = 1 << target;
    
    // Only process states where control qubit is |1⟩
    int base_idx = (state_idx & ~target_mask) | control_mask;
    
    if (base_idx < (1 << n_qubits)) {
        int i0 = puzzle_offset + (base_idx & ~target_mask);
        int i1 = puzzle_offset + (base_idx | target_mask);
        
        if (i0 != i1) {
            // Swap amplitudes when control is |1⟩
            cuDoubleComplex temp = batch_states[i0];
            batch_states[i0] = batch_states[i1];
            batch_states[i1] = temp;
        }
    }
}

// Batch measurement kernel - measures all puzzles simultaneously
__global__ void batch_measure_quantum_states(cuDoubleComplex* batch_states, 
                                            double* batch_probabilities,
                                            int* puzzle_offsets,
                                            int batch_size, 
                                            int n_qubits) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int puzzle_idx = global_idx / (1 << n_qubits);
    int state_idx = global_idx % (1 << n_qubits);
    
    if (puzzle_idx >= batch_size) return;
    
    int global_state_idx = puzzle_offsets[puzzle_idx] + state_idx;
    int global_prob_idx = puzzle_idx * (1 << n_qubits) + state_idx;
    
    cuDoubleComplex amp = batch_states[global_state_idx];
    batch_probabilities[global_prob_idx] = amp.x * amp.x + amp.y * amp.y;
}

// Initialize batch quantum states to |0⟩^n
__global__ void batch_initialize_quantum_states(cuDoubleComplex* batch_states, 
                                               int* puzzle_offsets,
                                               int batch_size, 
                                               int n_qubits) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int puzzle_idx = global_idx / (1 << n_qubits);
    int state_idx = global_idx % (1 << n_qubits);
    
    if (puzzle_idx >= batch_size) return;
    
    int global_state_idx = puzzle_offsets[puzzle_idx] + state_idx;
    
    if (state_idx == 0) {
        batch_states[global_state_idx] = make_cuDoubleComplex(1.0, 0.0);  // |00...0⟩ = 1
    } else {
        batch_states[global_state_idx] = make_cuDoubleComplex(0.0, 0.0);  // All other states = 0
    }
}

// C interface functions for optimized batch processing
extern "C" {
    
    // Allocate GPU memory for batch processing
    BatchQuantumState* cuda_alloc_batch_quantum_state(int batch_size, int n_qubits) {
        BatchQuantumState* batch = (BatchQuantumState*)malloc(sizeof(BatchQuantumState));
        if (!batch) return NULL;
        
        int total_states = 1 << n_qubits;
        size_t states_size = batch_size * total_states * sizeof(cuDoubleComplex);
        size_t offsets_size = batch_size * sizeof(int);
        size_t counts_size = batch_size * sizeof(int);
        
        // Allocate GPU memory
        cudaError_t error = cudaMalloc(&batch->states, states_size);
        if (error != cudaSuccess) {
            free(batch);
            return NULL;
        }
        
        error = cudaMalloc(&batch->puzzle_offsets, offsets_size);
        if (error != cudaSuccess) {
            cudaFree(batch->states);
            free(batch);
            return NULL;
        }
        
        error = cudaMalloc(&batch->gate_counts, counts_size);
        if (error != cudaSuccess) {
            cudaFree(batch->states);
            cudaFree(batch->puzzle_offsets);
            free(batch);
            return NULL;
        }
        
        // Initialize puzzle offsets on CPU then copy to GPU
        int* h_offsets = (int*)malloc(offsets_size);
        for (int i = 0; i < batch_size; i++) {
            h_offsets[i] = i * total_states;
        }
        
        cudaMemcpy(batch->puzzle_offsets, h_offsets, offsets_size, cudaMemcpyHostToDevice);
        free(h_offsets);
        
        batch->batch_size = batch_size;
        batch->n_qubits = n_qubits;
        batch->total_states = total_states;
        
        return batch;
    }
    
    // Free batch GPU memory
    void cuda_free_batch_quantum_state(BatchQuantumState* batch) {
        if (batch) {
            if (batch->states) cudaFree(batch->states);
            if (batch->puzzle_offsets) cudaFree(batch->puzzle_offsets);
            if (batch->gate_counts) cudaFree(batch->gate_counts);
            free(batch);
        }
    }
    
    // Initialize batch quantum states (async, no sync)
    int cuda_batch_init_quantum_states(BatchQuantumState* batch, cudaStream_t stream) {
        int total_threads = batch->batch_size * batch->total_states;
        dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_initialize_quantum_states<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, batch->batch_size, batch->n_qubits);
        
        cudaError_t error = cudaGetLastError();
        return (error == cudaSuccess) ? 1 : 0;
    }
    
    // Apply batch Hadamard gates (async, no sync)
    int cuda_batch_apply_hadamard(BatchQuantumState* batch, int* target_qubits, cudaStream_t stream) {
        int total_threads = batch->batch_size * (batch->total_states / 2);
        dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_apply_hadamard_gates<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, target_qubits, 
            batch->batch_size, batch->n_qubits);
        
        cudaError_t error = cudaGetLastError();
        return (error == cudaSuccess) ? 1 : 0;
    }
    
    // Apply batch T gates (async, no sync)
    int cuda_batch_apply_t_gate(BatchQuantumState* batch, int* target_qubits, cudaStream_t stream) {
        int total_threads = batch->batch_size * batch->total_states;
        dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_apply_t_gates<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, target_qubits, 
            batch->batch_size, batch->n_qubits);
        
        cudaError_t error = cudaGetLastError();
        return (error == cudaSuccess) ? 1 : 0;
    }
    
    // Apply batch CNOT gates (async, no sync)
    int cuda_batch_apply_cnot(BatchQuantumState* batch, int* control_qubits, 
                             int* target_qubits, cudaStream_t stream) {
        int total_threads = batch->batch_size * (batch->total_states / 2);
        dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_apply_cnot_gates<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, control_qubits, target_qubits,
            batch->batch_size, batch->n_qubits);
        
        cudaError_t error = cudaGetLastError();
        return (error == cudaSuccess) ? 1 : 0;
    }
    
    // Measure batch quantum states (async, minimal sync)
    int cuda_batch_measure_states(BatchQuantumState* batch, double* h_probabilities, cudaStream_t stream) {
        size_t prob_size = batch->batch_size * batch->total_states * sizeof(double);
        
        // Allocate GPU memory for probabilities
        double* d_probabilities;
        cudaError_t error = cudaMalloc(&d_probabilities, prob_size);
        if (error != cudaSuccess) return 0;
        
        int total_threads = batch->batch_size * batch->total_states;
        dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_measure_quantum_states<<<grid, block, 0, stream>>>(
            batch->states, d_probabilities, batch->puzzle_offsets,
            batch->batch_size, batch->n_qubits);
        
        // Async copy back to host
        cudaMemcpyAsync(h_probabilities, d_probabilities, prob_size, 
                       cudaMemcpyDeviceToHost, stream);
        
        // Only sync this stream (not entire device)
        cudaStreamSynchronize(stream);
        
        cudaFree(d_probabilities);
        return 1;
    }
    
    // Create CUDA stream for async operations
    cudaStream_t cuda_create_stream() {
        cudaStream_t stream;
        cudaError_t error = cudaStreamCreate(&stream);
        return (error == cudaSuccess) ? stream : NULL;
    }
    
    // Destroy CUDA stream
    void cuda_destroy_stream(cudaStream_t stream) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    
    // Synchronize specific stream (not entire device)
    int cuda_stream_synchronize(cudaStream_t stream) {
        cudaError_t error = cudaStreamSynchronize(stream);
        return (error == cudaSuccess) ? 1 : 0;
    }
    
    // Get GPU memory info
    int cuda_get_memory_info(size_t* free_mem, size_t* total_mem) {
        cudaError_t error = cudaMemGetInfo(free_mem, total_mem);
        return (error == cudaSuccess) ? 1 : 0;
    }
} 