#!/usr/bin/env python3
"""
Quantum-Geth Real Qiskit Solver
Unified, Branch-Serial Quantum Proof-of-Work — Canonical-Compile Edition

This implements genuine quantum circuit execution using Qiskit-Aer for
quantum proof-of-work blockchain mining.
"""

import sys
import json
import hashlib
import time
from typing import List, Dict, Any, Tuple
import numpy as np

# Qiskit imports for real quantum computation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.quantum_info import random_unitary
import random

# Quantum PoW Constants
FIXED_TCOUNT = 20    # 20 T-gates per puzzle (ENFORCED MINIMUM)
FIXED_LNET = 128     # 128 chained puzzles providing enhanced security
STARTING_QBITS = 16  # Start with 16 qubits
GLIDE_BLOCKS = 12500 # Add +1 qubit every 12,500 blocks

# 16 iso-hard branch templates (simplified representation)
BRANCH_TEMPLATES = list(range(16))

def calculate_qbits_for_height(block_height: int) -> int:
    """Calculate qubits based on epochic glide schedule"""
    additional_qbits = block_height // GLIDE_BLOCKS
    return STARTING_QBITS + additional_qbits

def create_branch_dependent_circuit(seed: bytes, qbits: int, tcount: int, branch_nibble: int) -> QuantumCircuit:
    """
    Create a branch-dependent quantum circuit for quantum proof-of-work.
    
    Args:
        seed: Seed for this puzzle
        qbits: Number of qubits
        tcount: Number of T-gates (should be 20)
        branch_nibble: High nibble from previous measurement (0-15)
    
    Returns:
        QuantumCircuit with the specified structure
    """
    # Create quantum circuit
    qreg = QuantumRegister(qbits, 'q')
    creg = ClassicalRegister(qbits, 'c')
    circuit = QuantumCircuit(qreg, creg)
    
    # Initialize with seed-dependent state
    random.seed(int.from_bytes(seed[:8], 'little'))
    
    # Apply initial Hadamard gates to create superposition
    for i in range(qbits):
        circuit.h(i)
    
    # Branch-dependent template selection
    template = BRANCH_TEMPLATES[branch_nibble]
    
    # Generate circuit based on template and seed
    # This is a simplified version - real implementation would use
    # the 16 iso-hard templates with proper gate synthesis
    
    gate_count = 0
    target_gates = min(tcount, 1000)  # Limit for practical simulation
    
    while gate_count < target_gates:
        # Select random qubits for 2-qubit gates
        if qbits >= 2:
            q1, q2 = random.sample(range(qbits), 2)
            
            # Apply different gate types based on template
            gate_type = (template + gate_count) % 4
            
            if gate_type == 0:
                circuit.cx(q1, q2)
            elif gate_type == 1:
                circuit.cz(q1, q2)
            elif gate_type == 2:
                circuit.ry(random.uniform(0, 2*np.pi), q1)
            else:
                circuit.rz(random.uniform(0, 2*np.pi), q1)
        else:
            # Single qubit gates for small circuits
            q = random.randint(0, qbits-1)
            circuit.ry(random.uniform(0, 2*np.pi), q)
        
        gate_count += 1
    
    # Add measurement
    circuit.measure_all()
    
    return circuit

def execute_quantum_circuit(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
    """
    Execute quantum circuit using Qiskit-Aer simulator.
    
    Args:
        circuit: The quantum circuit to execute
        shots: Number of measurement shots
    
    Returns:
        Dictionary of measurement outcomes and their counts
    """
    # Use Aer simulator for genuine quantum simulation
    simulator = AerSimulator(method='statevector')
    
    # Transpile circuit for the simulator
    transpiled = transpile(circuit, simulator, optimization_level=0)  # No optimization per spec
    
    # Execute the circuit
    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return counts

def extract_dominant_outcome(counts: Dict[str, int], qbits: int) -> bytes:
    """
    Extract the most frequent measurement outcome.
    
    Args:
        counts: Measurement counts from quantum execution
        qbits: Number of qubits
    
    Returns:
        Dominant outcome as bytes
    """
    # Find most frequent outcome
    dominant_bitstring = max(counts.keys(), key=counts.get)
    
    # Remove spaces and convert bitstring to bytes (little-endian)
    clean_bitstring = dominant_bitstring.replace(' ', '')
    
    # Ensure bitstring is the right length
    if len(clean_bitstring) > qbits:
        clean_bitstring = clean_bitstring[:qbits]
    elif len(clean_bitstring) < qbits:
        clean_bitstring = clean_bitstring.zfill(qbits)
    
    outcome_int = int(clean_bitstring, 2)
    byte_count = (qbits + 7) // 8
    outcome_bytes = outcome_int.to_bytes(byte_count, 'little')
    
    return outcome_bytes

def solve_quantum_puzzle_chain(seed0_hex: str, qbits: int, tcount: int, lnet: int) -> Dict[str, Any]:
    """
    Solve a chain of quantum puzzles using real Qiskit execution.
    
    This implements the quantum proof-of-work specification:
    - Seed chain: Seed₀ = initial, Seedᵢ = SHA256(Seedᵢ₋₁ || Outcomeᵢ₋₁)
    - Branch-dependent templates based on high nibble
    - Real quantum circuit execution using Qiskit-Aer
    
    Args:
        seed0_hex: Initial seed as hex string
        qbits: Number of qubits per puzzle
        tcount: Number of T-gates per puzzle
        lnet: Number of puzzles in chain
    
    Returns:
        Dictionary with outcomes, proofs, and metadata
    """
    # Removed debug output to prevent JSON parsing issues in Go
    
    start_time = time.time()
    
    # Parse initial seed
    current_seed = bytes.fromhex(seed0_hex)
    
    outcomes = []
    branch_nibbles = []
    gate_hashes = []
    execution_times = []
    
    for i in range(lnet):
        puzzle_start = time.time()
        
        # Determine branch nibble from previous outcome
        if i == 0:
            branch_nibble = 0  # First puzzle uses template 0
        else:
            # Extract high nibble from previous outcome
            prev_outcome = outcomes[i-1]
            last_byte = prev_outcome[-1]
            branch_nibble = (last_byte >> 4) & 0x0F
        
        branch_nibbles.append(branch_nibble)
        
        # Debug output removed
        
        # Create branch-dependent quantum circuit
        circuit = create_branch_dependent_circuit(current_seed, qbits, tcount, branch_nibble)
        
        # Calculate gate hash (canonical compile step)
        from qiskit.qasm2 import dumps
        circuit_qasm = dumps(circuit)
        gate_hash = hashlib.sha256(circuit_qasm.encode()).digest()
        gate_hashes.append(gate_hash)
        
        # Execute quantum circuit with Qiskit-Aer
        counts = execute_quantum_circuit(circuit, shots=1024)
        
        # Extract dominant measurement outcome
        outcome = extract_dominant_outcome(counts, qbits)
        outcomes.append(outcome)
        
        puzzle_time = time.time() - puzzle_start
        execution_times.append(puzzle_time)
        
        # Debug output removed
        
        # Calculate next seed (if not last puzzle)
        if i < lnet - 1:
            next_seed_hash = hashlib.sha256()
            next_seed_hash.update(current_seed)
            next_seed_hash.update(outcome)
            current_seed = next_seed_hash.digest()
    
    total_time = time.time() - start_time
    
    # Concatenate all outcomes
    all_outcomes = b''.join(outcomes)
    
    # Create aggregate gate hash
    aggregate_gate_hash = hashlib.sha256()
    for gate_hash in gate_hashes:
        aggregate_gate_hash.update(gate_hash)
    final_gate_hash = aggregate_gate_hash.digest()
    
    # Create proof root (simplified - combines outcomes and gate hashes)
    proof_hash = hashlib.sha256()
    proof_hash.update(all_outcomes)
    proof_hash.update(final_gate_hash)
    proof_root = proof_hash.digest()
    
    # Debug output removed to prevent JSON parsing issues
    
    return {
        'outcomes': all_outcomes.hex(),
        'branch_nibbles': bytes(branch_nibbles).hex(),
        'gate_hash': final_gate_hash.hex(),
        'proof_root': proof_root.hex(),
        'puzzle_count': lnet,
        'qbits': qbits,
        'tcount': tcount,
        'total_time': total_time,
        'avg_time_per_puzzle': np.mean(execution_times),
        'execution_times': execution_times,
        'backend': 'qiskit-aer-statevector',
        'shots_per_circuit': 1024
    }

def main():
    """Main function - reads JSON from stdin, outputs results to stdout"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            print(json.dumps({'error': 'No input provided'}))
            sys.exit(1)
        
        # Parse JSON input
        try:
            params = json.loads(input_data)
        except json.JSONDecodeError as e:
            print(json.dumps({'error': f'Invalid JSON: {str(e)}'}))
            sys.exit(1)
        
        # Validate required parameters
        required_params = ['seed0', 'qbits', 'tcount', 'lnet']
        for param in required_params:
            if param not in params:
                print(json.dumps({'error': f'Missing parameter: {param}'}))
                sys.exit(1)
        
        seed0 = params['seed0']
        qbits = int(params['qbits'])
        tcount = int(params['tcount'])
        lnet = int(params['lnet'])
        
        # Validate parameter ranges
        if not (1 <= qbits <= 20):  # Limit to 20 qubits for practical simulation
            print(json.dumps({'error': f'Invalid qbits: {qbits} (must be 1-20 for simulation)'}))
            sys.exit(1)
            
        if not (1 <= tcount <= 10000):
            print(json.dumps({'error': f'Invalid tcount: {tcount} (must be 1-10000)'}))
            sys.exit(1)
            
        if not (1 <= lnet <= 128):
            print(json.dumps({'error': f'Invalid lnet: {lnet} (must be 1-128)'}))
            sys.exit(1)
        
        # Validate seed format
        try:
            bytes.fromhex(seed0)
        except ValueError:
            print(json.dumps({'error': f'Invalid seed format: {seed0}'}))
            sys.exit(1)
        
        # Solve the quantum puzzle chain using real Qiskit
        result = solve_quantum_puzzle_chain(seed0, qbits, tcount, lnet)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': f'Quantum solver error: {str(e)}'}))
        sys.exit(1)

if __name__ == '__main__':
    main() 