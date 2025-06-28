#!/usr/bin/env python3
"""
Quantum Micro-Puzzle Solver for QMPoW
Simplified implementation for development and testing
"""

import json
import sys
import hashlib
import os
from typing import List, Dict, Any

def sha256_hash(data: bytes) -> bytes:
    """Compute SHA256 hash of data"""
    return hashlib.sha256(data).digest()

def simulate_quantum_circuit(seed: bytes, qbits: int, tcount: int) -> Dict[str, Any]:
    """
    Simulate a quantum circuit with the given parameters
    
    In a real implementation, this would run actual quantum circuits
    using Qiskit Aer or other quantum simulators.
    For development, we use deterministic simulation.
    """
    # Create a deterministic outcome based on seed and circuit parameters
    circuit_hash = hashlib.sha256()
    circuit_hash.update(seed)
    circuit_hash.update(qbits.to_bytes(1, 'little'))
    circuit_hash.update(tcount.to_bytes(2, 'little'))
    circuit_hash.update(b'quantum_outcome')
    
    digest = circuit_hash.digest()
    
    # Extract outcome bits (convert to bytes)
    outcome_bytes = (qbits + 7) // 8
    outcome = digest[:outcome_bytes]
    
    # Mask the last byte if qbits is not a multiple of 8
    if qbits % 8 != 0:
        last_byte_mask = (1 << (qbits % 8)) - 1
        outcome = outcome[:-1] + bytes([outcome[-1] & last_byte_mask])
    
    # Generate witness/proof
    witness_hash = hashlib.sha256()
    witness_hash.update(seed)
    witness_hash.update(outcome)
    witness_hash.update(qbits.to_bytes(1, 'little'))
    witness_hash.update(tcount.to_bytes(2, 'little'))
    witness_hash.update(b'mahadev_witness')
    
    witness = witness_hash.digest()
    
    return {
        'outcome': outcome.hex(),
        'witness': witness.hex()
    }

def solve_puzzle_chain(seed0: str, qbits: int, tcount: int, L: int) -> Dict[str, Any]:
    """
    Solve a chain of L quantum micro-puzzles
    
    Args:
        seed0: Initial seed (hex string)
        qbits: Number of qubits per puzzle
        tcount: Number of T-gates per puzzle  
        L: Number of puzzles in the chain
        
    Returns:
        Dictionary with outcomes and aggregated proof
    """
    seed_bytes = bytes.fromhex(seed0)
    current_seed = seed_bytes
    
    outcomes = []
    witnesses = []
    
    for i in range(L):
        # Solve quantum circuit with current seed
        result = simulate_quantum_circuit(current_seed, qbits, tcount)
        
        outcomes.append(result['outcome'])
        witnesses.append(result['witness'])
        
        # Generate seed for next puzzle (if not the last)
        if i < L - 1:
            next_seed_hash = hashlib.sha256()
            next_seed_hash.update(current_seed)
            next_seed_hash.update(bytes.fromhex(result['outcome']))
            current_seed = next_seed_hash.digest()
    
    # Concatenate all outcomes
    all_outcomes = ''.join(outcomes)
    
    # Create aggregate proof (simplified - just concatenate witnesses)
    aggregate_proof = ''.join(witnesses)
    
    return {
        'outcomes': all_outcomes,
        'proof': aggregate_proof,
        'puzzle_count': L,
        'qbits': qbits,
        'tcount': tcount
    }

def main():
    """Main solver function - reads from stdin, writes to stdout"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            print(json.dumps({'error': 'No input provided'}), file=sys.stderr)
            sys.exit(1)
        
        # Parse JSON input
        try:
            params = json.loads(input_data)
        except json.JSONDecodeError as e:
            print(json.dumps({'error': f'Invalid JSON: {str(e)}'}), file=sys.stderr)
            sys.exit(1)
        
        # Validate required parameters
        required_params = ['seed0', 'qbits', 'tcount', 'L']
        for param in required_params:
            if param not in params:
                print(json.dumps({'error': f'Missing parameter: {param}'}), file=sys.stderr)
                sys.exit(1)
        
        seed0 = params['seed0']
        qbits = int(params['qbits'])
        tcount = int(params['tcount'])
        L = int(params['L'])
        
        # Validate parameter ranges
        if not (1 <= qbits <= 64):
            print(json.dumps({'error': f'Invalid qbits: {qbits} (must be 1-64)'}), file=sys.stderr)
            sys.exit(1)
            
        if not (1 <= tcount <= 1000):
            print(json.dumps({'error': f'Invalid tcount: {tcount} (must be 1-1000)'}), file=sys.stderr)
            sys.exit(1)
            
        if not (1 <= L <= 256):
            print(json.dumps({'error': f'Invalid L: {L} (must be 1-256)'}), file=sys.stderr)
            sys.exit(1)
        
        # Validate seed format
        try:
            bytes.fromhex(seed0)
        except ValueError:
            print(json.dumps({'error': f'Invalid seed format: {seed0}'}), file=sys.stderr)
            sys.exit(1)
        
        # Solve the puzzle chain
        result = solve_puzzle_chain(seed0, qbits, tcount, L)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': f'Solver error: {str(e)}'}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 