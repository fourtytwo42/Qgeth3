#!/usr/bin/env python3

import sys
import numpy as np

def test_pycuda():
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # This initializes CUDA automatically
        from pycuda.compiler import SourceModule
        
        print("✅ PyCUDA import successful")
        
        # Get device count
        device_count = cuda.Device.count()
        print(f"🎯 CUDA devices available: {device_count}")
        
        if device_count == 0:
            print("❌ No CUDA devices found")
            return False
        
        # List all devices
        for i in range(device_count):
            device = cuda.Device(i)
            attrs = device.get_attributes()
            print(f"🔧 Device {i}: {device.name()}")
            print(f"   - Compute Capability: {device.compute_capability()}")
            print(f"   - Total Memory: {device.total_memory() // (1024**2)} MB")
            print(f"   - Max Threads per Block: {attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}")
        
        # Test simple GPU computation
        print("\n🧪 Testing simple GPU computation...")
        
        # Create test data
        a = np.random.randn(1000).astype(np.float32)
        b = np.random.randn(1000).astype(np.float32)
        
        # Allocate GPU memory
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(a.nbytes)
        
        # Copy data to GPU
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)
        
        # Simple CUDA kernel for vector addition
        mod = SourceModule("""
        __global__ void vector_add(float *a, float *b, float *c, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """)
        
        vector_add = mod.get_function("vector_add")
        
        # Execute kernel
        block_size = 256
        grid_size = (1000 + block_size - 1) // block_size
        vector_add(a_gpu, b_gpu, c_gpu, np.int32(1000), 
                  block=(block_size, 1, 1), grid=(grid_size, 1))
        
        # Copy result back
        c = np.empty_like(a)
        cuda.memcpy_dtoh(c, c_gpu)
        
        # Verify result
        expected = a + b
        if np.allclose(c, expected):
            print("✅ GPU computation test PASSED!")
            return True
        else:
            print("❌ GPU computation test FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ PyCUDA test failed: {e}")
        return False

def test_cupy():
    try:
        import cupy as cp
        print("✅ CuPy import successful")
        
        # Test CuPy array operations
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([6, 7, 8, 9, 10])
        c = a + b
        print(f"🧪 CuPy test: {a} + {b} = {c}")
        
        # Test GPU memory info
        print(f"🎯 GPU Memory: {cp.cuda.Device().mem_info}")
        return True
        
    except ImportError:
        print("⚠️  CuPy not available")
        return False
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing GPU Acceleration Capabilities")
    print("=" * 50)
    
    pycuda_ok = test_pycuda()
    print()
    cupy_ok = test_cupy()
    
    print("\n📊 Summary:")
    print(f"   PyCUDA: {'✅ Working' if pycuda_ok else '❌ Failed'}")
    print(f"   CuPy: {'✅ Working' if cupy_ok else '❌ Not Available'}")
    
    if pycuda_ok:
        print("\n🎉 GPU acceleration is available!")
        print("   We can implement custom CUDA kernels for quantum simulation.")
    else:
        print("\n⚠️  GPU acceleration not working properly.")
        print("   Falling back to CPU simulation.") 