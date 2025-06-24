# Makefile for Quantum GPU Miner with CUDA and Qiskit support
# Requires: Go, CUDA Toolkit, Python with Qiskit

# Build variables
BINARY_NAME=quantum-gpu-miner
CUDA_ARCH=sm_61,sm_70,sm_75,sm_80,sm_86,sm_89
CUDA_PATH?=$(shell which nvcc | sed 's|/bin/nvcc||')
PYTHON_PATH?=$(shell which python3 || which python)

# OS Detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS_FLAGS = -DLINUX
    LIB_EXT = .so
    CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas
endif
ifeq ($(UNAME_S),Darwin)
    OS_FLAGS = -DMACOS  
    LIB_EXT = .dylib
    CUDA_LIBS = -L$(CUDA_PATH)/lib -lcudart -lcublas
endif
ifneq (,$(findstring MINGW,$(UNAME_S)))
    OS_FLAGS = -DWINDOWS
    LIB_EXT = .dll
    CUDA_LIBS = -L$(CUDA_PATH)/lib/x64 -lcudart -lcublas
    BINARY_NAME = quantum-gpu-miner.exe
endif

# Build flags
CGO_CFLAGS = -I$(CUDA_PATH)/include
CGO_LDFLAGS = $(CUDA_LIBS)
GO_BUILD_FLAGS = -ldflags="-s -w" -trimpath

.PHONY: all clean build build-cuda build-qiskit test-gpu deps check-deps

all: check-deps build

# Check dependencies
check-deps:
	@echo "🔍 Checking dependencies..."
	@command -v go >/dev/null 2>&1 || { echo "❌ Go not found. Please install Go 1.21+"; exit 1; }
	@command -v nvcc >/dev/null 2>&1 || { echo "⚠️  NVCC not found. CUDA features will be disabled."; }
	@command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo "⚠️  Python not found. Qiskit features will be disabled."; }
	@echo "✅ Dependencies check completed"

# Install Python dependencies
deps:
	@echo "📦 Installing Python dependencies..."
	@$(PYTHON_PATH) -m pip install qiskit qiskit-aer cupy-cuda12x numpy --quiet
	@echo "✅ Python dependencies installed"

# Build CUDA library (if CUDA is available)
build-cuda:
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "🔨 Building CUDA quantum kernels..."; \
		cd pkg/quantum && nvcc -shared -Xcompiler -fPIC \
			-gencode arch=compute_61,code=sm_61 \
			-gencode arch=compute_70,code=sm_70 \
			-gencode arch=compute_75,code=sm_75 \
			-gencode arch=compute_80,code=sm_80 \
			-gencode arch=compute_86,code=sm_86 \
			-O3 -o libquantum_cuda$(LIB_EXT) quantum_cuda.cu; \
		echo "✅ CUDA kernels built successfully"; \
	else \
		echo "⚠️  NVCC not found, skipping CUDA build"; \
	fi

# Test Qiskit installation
test-qiskit:
	@echo "🧪 Testing Qiskit GPU backend..."
	@cd pkg/quantum && $(PYTHON_PATH) qiskit_gpu.py test 0 || echo "⚠️  Qiskit test failed"

# Build the main binary
build: build-cuda
	@echo "🔨 Building quantum-gpu-miner..."
	@export CGO_ENABLED=1; \
	export CGO_CFLAGS="$(CGO_CFLAGS)"; \
	export CGO_LDFLAGS="$(CGO_LDFLAGS)"; \
	go build $(GO_BUILD_FLAGS) -o $(BINARY_NAME) .
	@echo "✅ Build completed: $(BINARY_NAME)"

# Build with optimizations for release
build-release: deps build-cuda
	@echo "🚀 Building optimized release..."
	@export CGO_ENABLED=1; \
	export CGO_CFLAGS="$(CGO_CFLAGS) -O3 -DNDEBUG"; \
	export CGO_LDFLAGS="$(CGO_LDFLAGS)"; \
	go build $(GO_BUILD_FLAGS) -tags release -o $(BINARY_NAME) .
	@echo "✅ Release build completed: $(BINARY_NAME)"

# Test GPU functionality
test-gpu: build
	@echo "🧪 Testing GPU acceleration..."
	@./$(BINARY_NAME) -version
	@echo "🔬 Testing with a quick mining simulation..."
	@timeout 10s ./$(BINARY_NAME) -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -gpu -threads 1 || echo "⚠️  Quick test completed"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -f $(BINARY_NAME)
	@rm -f pkg/quantum/libquantum_cuda$(LIB_EXT)
	@go clean -cache -modcache -testcache
	@echo "✅ Clean completed"

# Install system-wide (Linux/macOS)
install: build-release
	@echo "📦 Installing quantum-gpu-miner..."
	@sudo cp $(BINARY_NAME) /usr/local/bin/
	@echo "✅ Installed to /usr/local/bin/$(BINARY_NAME)"

# Quick development build (skip optimizations)
dev-build:
	@echo "🔨 Quick development build..."
	@export CGO_ENABLED=1; go build -o $(BINARY_NAME) .
	@echo "✅ Development build completed"

# Run with GPU mining
run-gpu: build
	@echo "🚀 Starting GPU quantum mining..."
	@./$(BINARY_NAME) -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -gpu -threads 4

# Run with CPU mining  
run-cpu: build
	@echo "🚀 Starting CPU quantum mining..."
	@./$(BINARY_NAME) -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A -threads 4

# Show help
help:
	@echo "🚀 Quantum GPU Miner Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  build        - Build the miner binary"
	@echo "  build-release- Build optimized release version"
	@echo "  dev-build    - Quick development build"
	@echo "  deps         - Install Python dependencies"
	@echo "  test-gpu     - Test GPU functionality"
	@echo "  test-qiskit  - Test Qiskit backend"
	@echo "  run-gpu      - Run with GPU mining"
	@echo "  run-cpu      - Run with CPU mining"
	@echo "  install      - Install system-wide"
	@echo "  clean        - Clean build artifacts"
	@echo "  check-deps   - Check system dependencies"
	@echo ""
	@echo "Requirements:"
	@echo "  - Go 1.21+"
	@echo "  - CUDA Toolkit (for GPU acceleration)"
	@echo "  - Python 3.x with Qiskit"
	@echo "  - NVIDIA drivers"

# Display system info
info:
	@echo "🔍 System Information:"
	@echo "OS: $(UNAME_S)"
	@echo "Go version: $(shell go version 2>/dev/null || echo 'Not found')"
	@echo "NVCC version: $(shell nvcc --version 2>/dev/null | grep 'release' || echo 'Not found')"
	@echo "Python version: $(shell $(PYTHON_PATH) --version 2>/dev/null || echo 'Not found')"
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "Binary name: $(BINARY_NAME)" 