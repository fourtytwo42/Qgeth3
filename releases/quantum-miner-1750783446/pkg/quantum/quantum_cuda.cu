#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel for quantum state vector simulation
// Applies quantum gates on state vectors using GPU parallel processing

#define MAX_QUBITS 16
#define BLOCK_SIZE 256

// Complex number operations for quantum amplitudes
__device__ cuDoubleComplex complex_mult(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ cuDoubleComplex complex_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

// Apply Hadamard gate to qubit 'target' in state vector
__global__ void apply_hadamard_gate(cuDoubleComplex* state, int n_qubits, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = 1 << n_qubits;
    
    if (idx < total_states / 2) {
        int bit_mask = 1 << target;
        int i0 = idx & ~bit_mask;  // state with target bit = 0
        int i1 = idx | bit_mask;   // state with target bit = 1
        
        if (i0 < i1) {  // Process each pair only once
            cuDoubleComplex amp0 = state[i0];
            cuDoubleComplex amp1 = state[i1];
            
            // Hadamard matrix: (1/√2) * [[1, 1], [1, -1]]
            double inv_sqrt2 = 1.0 / sqrt(2.0);
            
            state[i0] = make_cuDoubleComplex(
                inv_sqrt2 * (amp0.x + amp1.x),
                inv_sqrt2 * (amp0.y + amp1.y)
            );
            state[i1] = make_cuDoubleComplex(
                inv_sqrt2 * (amp0.x - amp1.x),
                inv_sqrt2 * (amp0.y - amp1.y)
            );
        }
    }
}

// Apply T gate (π/4 phase rotation) to qubit 'target'
__global__ void apply_t_gate(cuDoubleComplex* state, int n_qubits, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = 1 << n_qubits;
    
    if (idx < total_states) {
        int bit_mask = 1 << target;
        if (idx & bit_mask) {  // If target qubit is |1⟩
            // Apply T gate: multiply by e^(iπ/4) = (1+i)/√2
            cuDoubleComplex t_phase = make_cuDoubleComplex(
                1.0 / sqrt(2.0), 1.0 / sqrt(2.0)
            );
            state[idx] = complex_mult(state[idx], t_phase);
        }
    }
}

// Apply CNOT gate with control and target qubits
__global__ void apply_cnot_gate(cuDoubleComplex* state, int n_qubits, int control, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = 1 << n_qubits;
    
    if (idx < total_states / 2) {
        int control_mask = 1 << control;
        int target_mask = 1 << target;
        
        // Only process states where control qubit is |1⟩
        int base_idx = (idx & ~target_mask) | control_mask;
        
        if (base_idx < total_states) {
            int i0 = base_idx & ~target_mask;  // target = 0
            int i1 = base_idx | target_mask;   // target = 1
            
            if (i0 != i1 && i0 < total_states && i1 < total_states) {
                // Swap amplitudes when control is |1⟩
                cuDoubleComplex temp = state[i0];
                state[i0] = state[i1];
                state[i1] = temp;
            }
        }
    }
}

// Measure quantum state and get outcome probabilities
__global__ void measure_quantum_state(cuDoubleComplex* state, double* probabilities, int n_qubits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = 1 << n_qubits;
    
    if (idx < total_states) {
        cuDoubleComplex amp = state[idx];
        probabilities[idx] = amp.x * amp.x + amp.y * amp.y;
    }
}

// Initialize quantum state to |0⟩^n
__global__ void initialize_quantum_state(cuDoubleComplex* state, int n_qubits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = 1 << n_qubits;
    
    if (idx < total_states) {
        if (idx == 0) {
            state[idx] = make_cuDoubleComplex(1.0, 0.0);  // |00...0⟩ = 1
        } else {
            state[idx] = make_cuDoubleComplex(0.0, 0.0);  // All other states = 0
        }
    }
}

// C interface functions for Go integration
extern "C" {
    // Allocate GPU memory for quantum state vector
    cuDoubleComplex* cuda_alloc_quantum_state(int n_qubits) {
        int total_states = 1 << n_qubits;
        cuDoubleComplex* d_state;
        
        cudaError_t error = cudaMalloc(&d_state, total_states * sizeof(cuDoubleComplex));
        if (error != cudaSuccess) {
            printf("CUDA malloc failed: %s\n", cudaGetErrorString(error));
            return NULL;
        }
        
        return d_state;
    }
    
    // Free GPU memory
    void cuda_free_quantum_state(cuDoubleComplex* d_state) {
        if (d_state) {
            cudaFree(d_state);
        }
    }
    
    // Initialize quantum state on GPU
    int cuda_init_quantum_state(cuDoubleComplex* d_state, int n_qubits) {
        int total_states = 1 << n_qubits;
        dim3 grid((total_states + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        initialize_quantum_state<<<grid, block>>>(d_state, n_qubits);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
            return 0;
        }
        
        cudaDeviceSynchronize();
        return 1;
    }
    
    // Apply Hadamard gate on GPU
    int cuda_apply_hadamard(cuDoubleComplex* d_state, int n_qubits, int target) {
        int total_states = 1 << n_qubits;
        dim3 grid((total_states / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        apply_hadamard_gate<<<grid, block>>>(d_state, n_qubits, target);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Hadamard kernel failed: %s\n", cudaGetErrorString(error));
            return 0;
        }
        
        cudaDeviceSynchronize();
        return 1;
    }
    
    // Apply T gate on GPU
    int cuda_apply_t_gate(cuDoubleComplex* d_state, int n_qubits, int target) {
        int total_states = 1 << n_qubits;
        dim3 grid((total_states + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        apply_t_gate<<<grid, block>>>(d_state, n_qubits, target);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA T gate kernel failed: %s\n", cudaGetErrorString(error));
            return 0;
        }
        
        cudaDeviceSynchronize();
        return 1;
    }
    
    // Apply CNOT gate on GPU
    int cuda_apply_cnot(cuDoubleComplex* d_state, int n_qubits, int control, int target) {
        int total_states = 1 << n_qubits;
        dim3 grid((total_states / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        apply_cnot_gate<<<grid, block>>>(d_state, n_qubits, control, target);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA CNOT kernel failed: %s\n", cudaGetErrorString(error));
            return 0;
        }
        
        cudaDeviceSynchronize();
        return 1;
    }
    
    // Measure quantum state and get probabilities
    int cuda_measure_state(cuDoubleComplex* d_state, double* h_probabilities, int n_qubits) {
        int total_states = 1 << n_qubits;
        
        // Allocate GPU memory for probabilities
        double* d_probabilities;
        cudaError_t error = cudaMalloc(&d_probabilities, total_states * sizeof(double));
        if (error != cudaSuccess) {
            printf("CUDA malloc for probabilities failed: %s\n", cudaGetErrorString(error));
            return 0;
        }
        
        // Launch measurement kernel
        dim3 grid((total_states + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        
        measure_quantum_state<<<grid, block>>>(d_state, d_probabilities, n_qubits);
        
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA measurement kernel failed: %s\n", cudaGetErrorString(error));
            cudaFree(d_probabilities);
            return 0;
        }
        
        // Copy results back to host
        error = cudaMemcpy(h_probabilities, d_probabilities, 
                          total_states * sizeof(double), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            printf("CUDA memcpy failed: %s\n", cudaGetErrorString(error));
            cudaFree(d_probabilities);
            return 0;
        }
        
        cudaFree(d_probabilities);
        cudaDeviceSynchronize();
        return 1;
    }
    
    // Get CUDA device properties
    int cuda_get_device_info(int device_id, char* name, int* major, int* minor, size_t* memory) {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
        
        if (error != cudaSuccess) {
            printf("Failed to get device properties: %s\n", cudaGetErrorString(error));
            return 0;
        }
        
        strncpy(name, prop.name, 256);
        *major = prop.major;
        *minor = prop.minor;
        *memory = prop.totalGlobalMem;
        
        return 1;
    }
    
    // Set CUDA device
    int cuda_set_device(int device_id) {
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            printf("Failed to set CUDA device %d: %s\n", device_id, cudaGetErrorString(error));
            return 0;
        }
        return 1;
    }
} 