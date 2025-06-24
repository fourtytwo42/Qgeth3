#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import our actual implementation
from pkg.quantum.qiskit_gpu import QiskitGPUBackend

def debug_benchmark():
    """Debug the exact benchmark that's failing"""
    
    print("=== Debug Benchmark ===")
    
    try:
        # Create backend exactly like the failing code
        backend = QiskitGPUBackend()
        print(f"Backend created, GPU available: {backend.gpu_available}")
        
        # Try the exact benchmark call that's failing
        const_qubits = 8
        const_gates = 50
        
        print(f"Attempting simulation with {const_qubits} qubits, {const_gates} gates")
        
        # This is the exact call from benchmark_performance
        outcome, sim_time = backend.simulate_quantum_puzzle(
            0, "benchmark_hash", 0, const_qubits, const_gates
        )
        
        print(f"✅ SUCCESS: {outcome} in {sim_time:.4f}s")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_benchmark() 