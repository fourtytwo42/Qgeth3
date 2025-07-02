#!/usr/bin/env python3
"""
Test script for deterministic quantum solver
Verifies that the solver produces consistent results
"""

import json
import subprocess
import sys
import os

def test_deterministic_solver():
    """Test the deterministic quantum solver"""
    print("Testing Deterministic Quantum Solver...")
    
    # Test input
    test_input = {
        "seed0": "deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcdef",
        "qbits": 4,
        "tcount": 20,
        "lnet": 2
    }
    
    solver_path = "quantum-geth/tools/solver/qiskit_solver.py"
    
    if not os.path.exists(solver_path):
        print(f"❌ Solver not found at {solver_path}")
        return False
    
    try:
        # Run solver twice to test determinism
        print("Running solver (first time)...")
        result1 = subprocess.run(
            [sys.executable, solver_path],
            input=json.dumps(test_input),
            capture_output=True,
            text=True
        )
        
        if result1.returncode != 0:
            print(f"❌ First run failed: {result1.stderr}")
            return False
        
        print("Running solver (second time)...")
        result2 = subprocess.run(
            [sys.executable, solver_path],
            input=json.dumps(test_input),
            capture_output=True,
            text=True
        )
        
        if result2.returncode != 0:
            print(f"❌ Second run failed: {result2.stderr}")
            return False
        
        # Parse results
        try:
            output1 = json.loads(result1.stdout)
            output2 = json.loads(result2.stdout)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON output: {e}")
            print(f"Output 1: {result1.stdout}")
            print(f"Output 2: {result2.stdout}")
            return False
        
        # Check determinism
        if output1.get('outcomes') != output2.get('outcomes'):
            print("❌ DETERMINISM FAILURE: Different outcomes!")
            print(f"Run 1: {output1.get('outcomes')}")
            print(f"Run 2: {output2.get('outcomes')}")
            return False
        
        if output1.get('proof_root') != output2.get('proof_root'):
            print("❌ DETERMINISM FAILURE: Different proof roots!")
            print(f"Run 1: {output1.get('proof_root')}")
            print(f"Run 2: {output2.get('proof_root')}")
            return False
        
        # Check consensus safety indicators
        if not output1.get('deterministic', False):
            print("❌ Missing deterministic flag")
            return False
        
        if not output1.get('consensus_safe', False):
            print("❌ Missing consensus_safe flag")
            return False
        
        print("✅ DETERMINISM TEST PASSED!")
        print(f"✅ Backend: {output1.get('backend')}")
        print(f"✅ Outcomes: {output1.get('outcomes')}")
        print(f"✅ Proof Root: {output1.get('proof_root')}")
        print(f"✅ Puzzle Count: {output1.get('puzzle_count')}")
        print(f"✅ Total Time: {output1.get('total_time'):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_deterministic_solver()
    sys.exit(0 if success else 1) 