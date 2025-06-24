#!/usr/bin/env python3

import sys
import numpy as np
import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def debug_simulate_quantum_puzzle(puzzle_index: int, work_hash: str, qnonce: int,
                               n_qubits: int, n_gates: int):
    """Debug version of simulate_quantum_puzzle"""
    
    print(f"Input: puzzle_index={puzzle_index}, work_hash={work_hash}, qnonce={qnonce}")
    print(f"       n_qubits={n_qubits}, n_gates={n_gates}")
    
    # Generate deterministic circuit based on puzzle parameters
    seed = puzzle_index ^ qnonce ^ hash(work_hash) & 0xFFFFFFFF
    print(f"Seed: {seed}")
    
    np.random.seed(seed)
    
    # For large circuits (mining), use simplified simulation
    if n_gates > 1000:
        print("Using simplified simulation (n_gates > 1000)")
        start_time = time.time()
        
        # Simulate quantum randomness with classical computation
        outcome_int = seed % (2 ** min(n_qubits, 16))  # Limit to 16 qubits for safety
        print(f"Outcome int: {outcome_int}")
        print(f"Limited qubits: {min(n_qubits, 16)}")
        
        n_bytes = (min(n_qubits, 16) + 7) // 8
        print(f"Number of bytes: {n_bytes}")
        
        outcome_bytes = outcome_int.to_bytes(n_bytes, byteorder='little')
        print(f"Outcome bytes: {outcome_bytes}")
        
        simulation_time = time.time() - start_time
        return outcome_bytes, simulation_time
    
    print("Using full quantum simulation")
    
    # For smaller circuits, use full quantum simulation
    backend = AerSimulator(method='statevector')
    
    # Create a simple circuit for testing
    circuit = QuantumCircuit(n_qubits)
    
    # Add some gates (simplified)
    for gate_idx in range(min(n_gates, 10)):  # Limit gates for debugging
        gate_type = (seed + gate_idx) % 3
        target = (seed + gate_idx) % n_qubits
        
        print(f"Gate {gate_idx}: type={gate_type}, target={target}")
        
        if gate_type == 0:  # Hadamard
            circuit.h(target)
        elif gate_type == 1:  # T gate
            circuit.t(target)
        elif gate_type == 2 and n_qubits > 1:  # CNOT
            control = (target + 1) % n_qubits
            circuit.cx(control, target)
            print(f"           control={control}")
    
    # Add measurements
    circuit.measure_all()
    print(f"Circuit created with {len(circuit.data)} operations")
    
    # Transpile with minimal optimization for speed
    start_time = time.time()
    transpiled = transpile(circuit, backend, optimization_level=0)
    print(f"Circuit transpiled")
    
    # Run simulation with fewer shots for speed
    job = backend.run(transpiled, shots=1)  # Single shot for speed
    result = job.result()
    print(f"Simulation completed")
    
    simulation_time = time.time() - start_time
    
    # Get measurement counts
    counts = result.get_counts(0)
    print(f"Counts: {counts}")
    
    # Get the single outcome
    chosen_outcome = list(counts.keys())[0]
    print(f"Raw outcome: {repr(chosen_outcome)}")
    
    # Convert binary string to bytes (remove spaces and handle binary format)
    clean_outcome = chosen_outcome.replace(' ', '')  # Remove any spaces
    print(f"Clean outcome: {repr(clean_outcome)} (length: {len(clean_outcome)})")
    
    if len(clean_outcome) > 64:
        print(f"❌ Outcome too long: {len(clean_outcome)} bits > 64 bits")
        raise ValueError(f"Outcome too long: {len(clean_outcome)} bits")
    
    outcome_int = int(clean_outcome, 2)
    print(f"Outcome int: {outcome_int}")
    
    # Ensure safe byte conversion
    n_bytes = (min(n_qubits, 16) + 7) // 8
    print(f"Number of bytes: {n_bytes}")
    
    if outcome_int.bit_length() > n_bytes * 8:
        print(f"❌ Int too big for {n_bytes} bytes: {outcome_int.bit_length()} bits")
        raise ValueError(f"Int too big for {n_bytes} bytes")
    
    outcome_bytes = outcome_int.to_bytes(n_bytes, byteorder='little')
    print(f"✅ Outcome bytes: {outcome_bytes}")
    
    return outcome_bytes, simulation_time

if __name__ == "__main__":
    print("=== Debug Quantum Simulation ===")
    
    try:
        # Test case that's failing
        outcome, sim_time = debug_simulate_quantum_puzzle(
            0, "benchmark_hash", 0, 8, 50  # Parameters from benchmark
        )
        print(f"✅ SUCCESS: {outcome} in {sim_time:.4f}s")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc() 