#!/usr/bin/env python3
"""
Direct CuPy GPU test to diagnose hanging issue
"""

import sys
import time
import json

def test_cupy_basic():
    """Test basic CuPy GPU functionality"""
    print("Testing CuPy GPU basic functionality...")
    
    try:
        import cupy as cp
        print(f"✅ CuPy imported successfully")
        
        # Test GPU device
        device = cp.cuda.Device(0)
        print(f"✅ GPU device 0 available: {device}")
        
        # Test basic GPU array operation
        print("Creating GPU array...")
        x = cp.array([1, 2, 3, 4, 5])
        print(f"✅ GPU array created: {x}")
        
        # Test computation
        print("Testing GPU computation...")
        y = x * 2
        result = cp.asnumpy(y)
        print(f"✅ GPU computation successful: {result}")
        
        # Test larger array (similar to quantum state vector)
        print("Testing quantum-sized array (16 qubits = 65536 elements)...")
        start_time = time.time()
        
        # 16 qubits = 2^16 = 65536 complex numbers
        state_size = 2 ** 16
        state = cp.ones(state_size, dtype=cp.complex64) / cp.sqrt(state_size)
        
        # Simple operation
        result_state = cp.abs(state) ** 2
        probabilities = cp.asnumpy(result_state[:10])  # First 10 elements
        
        elapsed = time.time() - start_time
        print(f"✅ Quantum-sized computation successful in {elapsed:.3f}s")
        print(f"   First 10 probabilities: {probabilities}")
        
        return True
        
    except ImportError as e:
        print(f"❌ CuPy import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False

def test_batch_simulation():
    """Test the actual batch simulation that's hanging"""
    print("\nTesting batch quantum simulation...")
    
    try:
        # Import the actual CuPy module
        sys.path.append('quantum-miner/pkg/quantum')
        from cupy_gpu import batch_simulate_quantum_puzzles_gpu
        
        # Create test puzzles
        puzzles = []
        for i in range(48):  # 48 puzzles like in mining
            puzzles.append({
                'num_qubits': 16,
                'target_state': 'entangled',
                'measurement_basis': 'computational',
                'work_hash': 'test_hash',
                'qnonce': 12345,
                'puzzle_id': i
            })
        
        print(f"Testing batch simulation with {len(puzzles)} puzzles...")
        start_time = time.time()
        
        results = batch_simulate_quantum_puzzles_gpu(puzzles)
        
        elapsed = time.time() - start_time
        print(f"✅ Batch simulation completed in {elapsed:.3f}s")
        print(f"   Results: {len(results)} puzzles processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 CuPy GPU Direct Test")
    print("=" * 50)
    
    # Test 1: Basic CuPy functionality
    basic_ok = test_cupy_basic()
    
    # Test 2: Batch simulation
    if basic_ok:
        batch_ok = test_batch_simulation()
    else:
        print("❌ Skipping batch test due to basic CuPy failure")
        batch_ok = False
    
    print("\n" + "=" * 50)
    if basic_ok and batch_ok:
        print("🎉 All tests passed - CuPy GPU should work for mining")
    elif basic_ok:
        print("⚠️  Basic CuPy works but batch simulation has issues")
    else:
        print("❌ CuPy GPU not working - fallback to CPU mining recommended") 