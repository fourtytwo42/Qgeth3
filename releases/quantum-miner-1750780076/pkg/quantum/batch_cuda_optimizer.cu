#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// High-Performance Batch Quantum Processing System
// Eliminates synchronization bottlenecks for massive GPU utilization

#define MAX_QUBITS 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1024

// Optimized batch quantum state structure
typedef struct {
    cuDoubleComplex* states;
    int* puzzle_offsets;
    int batch_size;
    int n_qubits;
    int total_states;
} BatchQuantumState;

// Ultra-fast complex operations
__device__ __forceinline__ cuDoubleComplex cmult_opt(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        __fma_rn(a.x, b.x, -a.y * b.y),
        __fma_rn(a.x, b.y, a.y * b.x)
    );
}

// Batch Hadamard kernel - processes multiple puzzles in parallel
__global__ void batch_hadamard(cuDoubleComplex* states, int* offsets, 
                              int* targets, int batch_size, int n_qubits) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pid = gid / (1 << (n_qubits - 1));
    int sid = gid % (1 << (n_qubits - 1));
    
    if (pid >= batch_size) return;
    
    int offset = offsets[pid];
    int target = targets[pid];
    int mask = 1 << target;
    
    int i0 = offset + (sid & ~mask);
    int i1 = offset + (sid | mask);
    
    if (i0 < i1) {
        cuDoubleComplex a0 = states[i0];
        cuDoubleComplex a1 = states[i1];
        
        double s = 0.7071067811865476; // 1/√2
        states[i0] = make_cuDoubleComplex(s * (a0.x + a1.x), s * (a0.y + a1.y));
        states[i1] = make_cuDoubleComplex(s * (a0.x - a1.x), s * (a0.y - a1.y));
    }
}

// Batch T-gate kernel
__global__ void batch_t_gate(cuDoubleComplex* states, int* offsets,
                            int* targets, int batch_size, int n_qubits) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pid = gid / (1 << n_qubits);
    int sid = gid % (1 << n_qubits);
    
    if (pid >= batch_size) return;
    
    int target = targets[pid];
    if (sid & (1 << target)) {
        int idx = offsets[pid] + sid;
        cuDoubleComplex t = make_cuDoubleComplex(0.7071067811865476, 0.7071067811865476);
        states[idx] = cmult_opt(states[idx], t);
    }
}

// Batch measurement kernel
__global__ void batch_measure(cuDoubleComplex* states, double* probs,
                             int* offsets, int batch_size, int n_qubits) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pid = gid / (1 << n_qubits);
    int sid = gid % (1 << n_qubits);
    
    if (pid >= batch_size) return;
    
    int state_idx = offsets[pid] + sid;
    int prob_idx = pid * (1 << n_qubits) + sid;
    
    cuDoubleComplex amp = states[state_idx];
    probs[prob_idx] = amp.x * amp.x + amp.y * amp.y;
}

// Initialize all states to |0⟩
__global__ void batch_init(cuDoubleComplex* states, int* offsets,
                          int batch_size, int n_qubits) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pid = gid / (1 << n_qubits);
    int sid = gid % (1 << n_qubits);
    
    if (pid >= batch_size) return;
    
    int idx = offsets[pid] + sid;
    states[idx] = make_cuDoubleComplex((sid == 0) ? 1.0 : 0.0, 0.0);
}

extern "C" {
    
    // Allocate batch state
    BatchQuantumState* cuda_alloc_batch_state(int batch_size, int n_qubits) {
        BatchQuantumState* batch = (BatchQuantumState*)malloc(sizeof(BatchQuantumState));
        if (!batch) return NULL;
        
        int total_states = 1 << n_qubits;
        size_t states_size = batch_size * total_states * sizeof(cuDoubleComplex);
        size_t offsets_size = batch_size * sizeof(int);
        
        if (cudaMalloc(&batch->states, states_size) != cudaSuccess) {
            free(batch);
            return NULL;
        }
        
        if (cudaMalloc(&batch->puzzle_offsets, offsets_size) != cudaSuccess) {
            cudaFree(batch->states);
            free(batch);
            return NULL;
        }
        
        // Setup offsets
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
    
    // Free batch state
    void cuda_free_batch_state(BatchQuantumState* batch) {
        if (batch) {
            if (batch->states) cudaFree(batch->states);
            if (batch->puzzle_offsets) cudaFree(batch->puzzle_offsets);
            free(batch);
        }
    }
    
    // Initialize batch (async)
    int cuda_batch_init_async(BatchQuantumState* batch, cudaStream_t stream) {
        int threads = batch->batch_size * batch->total_states;
        dim3 grid((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_init<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, batch->batch_size, batch->n_qubits);
        
        return (cudaGetLastError() == cudaSuccess) ? 1 : 0;
    }
    
    // Batch Hadamard (async)
    int cuda_batch_hadamard_async(BatchQuantumState* batch, int* targets, cudaStream_t stream) {
        int threads = batch->batch_size * (batch->total_states / 2);
        dim3 grid((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_hadamard<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, targets, 
            batch->batch_size, batch->n_qubits);
        
        return (cudaGetLastError() == cudaSuccess) ? 1 : 0;
    }
    
    // Batch T-gate (async)
    int cuda_batch_t_gate_async(BatchQuantumState* batch, int* targets, cudaStream_t stream) {
        int threads = batch->batch_size * batch->total_states;
        dim3 grid((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_t_gate<<<grid, block, 0, stream>>>(
            batch->states, batch->puzzle_offsets, targets,
            batch->batch_size, batch->n_qubits);
        
        return (cudaGetLastError() == cudaSuccess) ? 1 : 0;
    }
    
    // Batch measure (with stream sync only)
    int cuda_batch_measure_async(BatchQuantumState* batch, double* h_probs, cudaStream_t stream) {
        size_t prob_size = batch->batch_size * batch->total_states * sizeof(double);
        
        double* d_probs;
        if (cudaMalloc(&d_probs, prob_size) != cudaSuccess) return 0;
        
        int threads = batch->batch_size * batch->total_states;
        dim3 grid((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        batch_measure<<<grid, block, 0, stream>>>(
            batch->states, d_probs, batch->puzzle_offsets,
            batch->batch_size, batch->n_qubits);
        
        cudaMemcpyAsync(h_probs, d_probs, prob_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  // Only sync this stream
        
        cudaFree(d_probs);
        return 1;
    }
    
    // Stream management
    cudaStream_t cuda_create_stream() {
        cudaStream_t stream;
        return (cudaStreamCreate(&stream) == cudaSuccess) ? stream : NULL;
    }
    
    void cuda_destroy_stream(cudaStream_t stream) {
        if (stream) cudaStreamDestroy(stream);
    }
    
    int cuda_stream_sync(cudaStream_t stream) {
        return (cudaStreamSynchronize(stream) == cudaSuccess) ? 1 : 0;
    }
} 