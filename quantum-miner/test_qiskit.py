#!/usr/bin/env python3

import sys
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Test 1: Basic circuit and conversion
print("=== Test 1: Basic Circuit ===")
try:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.measure_all()
    
    backend = AerSimulator(method='statevector')
    result = backend.run(circuit, shots=1).result()
    counts = result.get_counts(0)
    
    print(f"Counts: {counts}")
    
    outcome = list(counts.keys())[0] 
    print(f"Raw outcome: {repr(outcome)}")
    
    clean = outcome.replace(' ', '')
    print(f"Clean outcome: {repr(clean)}")
    print(f"Length: {len(clean)}")
    
    if len(clean) <= 64:  # Limit to 64 bits for safety
        val = int(clean, 2)
        print(f"Value: {val}")
        
        # Convert to bytes
        n_bytes = (4 + 7) // 8  # 4 qubits = 1 byte
        byte_val = val.to_bytes(n_bytes, byteorder='little')
        print(f"Bytes: {byte_val}")
        print("✅ Test 1 PASSED")
    else:
        print(f"❌ Binary string too long: {len(clean)} bits")
        
except Exception as e:
    print(f"❌ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: GPU availability
print("=== Test 2: GPU Support ===")
try:
    gpu_backend = AerSimulator(device='GPU', method='statevector')
    print(f"✅ GPU backend created: {gpu_backend}")
except Exception as e:
    print(f"⚠️  GPU not available: {e}")

print()

# Test 3: Larger circuit with safety limits
print("=== Test 3: Larger Circuit ===")
try:
    circuit = QuantumCircuit(8)  # 8 qubits
    for i in range(4):
        circuit.h(i)
    circuit.measure_all()
    
    backend = AerSimulator(method='statevector')
    result = backend.run(circuit, shots=1).result()
    counts = result.get_counts(0)
    
    outcome = list(counts.keys())[0]
    clean = outcome.replace(' ', '')
    
    print(f"8-qubit outcome: {clean} (length: {len(clean)})")
    
    if len(clean) <= 16:  # Limit to 16 bits max
        val = int(clean, 2)
        n_bytes = (min(8, 16) + 7) // 8
        byte_val = val.to_bytes(n_bytes, byteorder='little')
        print(f"✅ Test 3 PASSED: {byte_val}")
    else:
        print(f"❌ Binary string too long for safe conversion")
        
except Exception as e:
    print(f"❌ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc() 