package quantum

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// CupyGPUSimulator wraps the CuPy-based GPU quantum simulator
type CupyGPUSimulator struct {
	pythonScript string
	available    bool
	deviceInfo   map[string]interface{}
}

// NewCupyGPUSimulator creates a new CuPy GPU simulator instance
func NewCupyGPUSimulator() *CupyGPUSimulator {
	simulator := &CupyGPUSimulator{
		pythonScript: findCupyScript(),
		available:    false,
	}

	// Test if CuPy GPU is available
	simulator.testAvailability()

	return simulator
}

// testAvailability checks if CuPy GPU simulation is available
func (c *CupyGPUSimulator) testAvailability() {
	start := time.Now()

	fmt.Println("🧪 Testing CuPy GPU availability...")

	// Test CuPy GPU availability by running test mode
	cmd := exec.Command("python", c.pythonScript)
	output, err := cmd.Output()

	if err != nil {
		fmt.Printf("❌ CuPy GPU test failed: %v\n", err)
		c.available = false
		return
	}

	// Parse test output to check for GPU availability
	outputStr := string(output)
	if contains(outputStr, "GPU Acceleration Available") || contains(outputStr, "Backend: cupy_gpu") {
		c.available = true
		fmt.Printf("✅ CuPy GPU simulator available (test took %v)\n", time.Since(start))

		// Extract device info if possible
		c.deviceInfo = map[string]interface{}{
			"backend": "cupy_gpu",
			"tested":  true,
		}
	} else {
		fmt.Println("⚠️  CuPy falling back to CPU")
		c.available = false
	}
}

// IsAvailable returns whether GPU acceleration is available
func (c *CupyGPUSimulator) IsAvailable() bool {
	return c.available
}

// SimulateQuantumPuzzle simulates a single quantum puzzle using CuPy GPU
func (c *CupyGPUSimulator) SimulateQuantumPuzzle(puzzleConfig map[string]interface{}) (map[string]interface{}, error) {
	if !c.available {
		return nil, fmt.Errorf("CuPy GPU simulator not available")
	}

	// Prepare JSON input
	input := map[string]interface{}{
		"command": "single_simulate",
		"puzzle":  puzzleConfig,
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %v", err)
	}

	// Execute Python script
	cmd := exec.Command("python", c.pythonScript, string(inputJSON))
	output, err := cmd.Output()

	if err != nil {
		return nil, fmt.Errorf("CuPy simulation failed: %v", err)
	}

	// Parse result
	var result map[string]interface{}
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse result: %v", err)
	}

	// Check for errors
	if status, ok := result["status"].(string); ok && status != "success" {
		if message, ok := result["message"].(string); ok {
			return nil, fmt.Errorf("simulation error: %s", message)
		}
		return nil, fmt.Errorf("simulation failed with status: %s", status)
	}

	// Extract simulation result
	if simResult, ok := result["result"].(map[string]interface{}); ok {
		return simResult, nil
	}

	return nil, fmt.Errorf("invalid result format")
}

// BatchSimulateQuantumPuzzles simulates multiple puzzles using CuPy GPU
func (c *CupyGPUSimulator) BatchSimulateQuantumPuzzles(puzzles []map[string]interface{}) ([]map[string]interface{}, error) {
	if !c.available {
		return nil, fmt.Errorf("CuPy GPU simulator not available")
	}

	// Prepare JSON input
	input := map[string]interface{}{
		"command": "batch_simulate",
		"puzzles": puzzles,
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %v", err)
	}

	// Create context with timeout to prevent hanging
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Execute Python script with timeout and interrupt handling
	cmd := exec.CommandContext(ctx, "python", c.pythonScript, string(inputJSON))
	output, err := cmd.Output()

	if err != nil {
		// Handle timeout
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("CuPy GPU simulation timeout (30s) - possible GPU hang")
		}
		// Handle Windows interrupt signals gracefully
		if exitError, ok := err.(*exec.ExitError); ok {
			if exitError.ExitCode() == -1073741510 { // 0xc000013a = Control+C
				return nil, fmt.Errorf("simulation interrupted")
			}
		}
		return nil, fmt.Errorf("CuPy batch simulation failed: %v", err)
	}

	// Parse result
	var result map[string]interface{}
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse result: %v", err)
	}

	// Check for errors
	if status, ok := result["status"].(string); ok && status != "success" {
		if message, ok := result["message"].(string); ok {
			return nil, fmt.Errorf("batch simulation error: %s", message)
		}
		return nil, fmt.Errorf("batch simulation failed with status: %s", status)
	}

	// Extract simulation results
	if simResults, ok := result["results"].([]interface{}); ok {
		results := make([]map[string]interface{}, len(simResults))
		for i, r := range simResults {
			if resultMap, ok := r.(map[string]interface{}); ok {
				results[i] = resultMap
			} else {
				return nil, fmt.Errorf("invalid result format at index %d", i)
			}
		}
		return results, nil
	}

	return nil, fmt.Errorf("invalid batch result format")
}

// GetDeviceInfo returns information about the GPU device
func (c *CupyGPUSimulator) GetDeviceInfo() map[string]interface{} {
	return c.deviceInfo
}

// BenchmarkGPU runs a performance benchmark on the GPU
func (c *CupyGPUSimulator) BenchmarkGPU(numPuzzles int, numQubits int) (map[string]interface{}, error) {
	if !c.available {
		return nil, fmt.Errorf("CuPy GPU simulator not available")
	}

	fmt.Printf("🚀 Running GPU benchmark: %d puzzles with %d qubits each\n", numPuzzles, numQubits)

	// Create benchmark puzzles
	puzzles := make([]map[string]interface{}, numPuzzles)
	for i := 0; i < numPuzzles; i++ {
		puzzles[i] = map[string]interface{}{
			"num_qubits":        numQubits,
			"target_state":      "entangled",
			"measurement_basis": "computational",
		}
	}

	start := time.Now()

	results, err := c.BatchSimulateQuantumPuzzles(puzzles)
	if err != nil {
		return nil, fmt.Errorf("benchmark failed: %v", err)
	}

	totalTime := time.Since(start)
	avgTime := totalTime / time.Duration(numPuzzles)

	// Calculate performance metrics
	var totalSimTime float64
	successCount := 0

	for _, result := range results {
		if simTime, ok := result["simulation_time"].(float64); ok {
			totalSimTime += simTime
		}
		if success, ok := result["success"].(bool); ok && success {
			successCount++
		}
	}

	benchmarkResult := map[string]interface{}{
		"total_puzzles":       numPuzzles,
		"num_qubits":          numQubits,
		"success_count":       successCount,
		"total_wall_time_sec": totalTime.Seconds(),
		"avg_wall_time_sec":   avgTime.Seconds(),
		"total_sim_time_sec":  totalSimTime,
		"avg_sim_time_sec":    totalSimTime / float64(numPuzzles),
		"puzzles_per_second":  float64(numPuzzles) / totalTime.Seconds(),
		"backend":             "cupy_gpu",
		"device_info":         c.deviceInfo,
	}

	fmt.Printf("✅ Benchmark completed: %.2f puzzles/sec (%.3fs avg)\n",
		benchmarkResult["puzzles_per_second"].(float64),
		benchmarkResult["avg_wall_time_sec"].(float64))

	return benchmarkResult, nil
}

// contains checks if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) && (s[0:len(substr)] == substr ||
			s[len(s)-len(substr):] == substr ||
			containsHelper(s, substr))))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// findCupyScript locates the CuPy GPU Python script
func findCupyScript() string {
	// Try different possible locations
	possiblePaths := []string{
		filepath.Join("quantum-miner", "pkg", "quantum", "cupy_gpu.py"),
		filepath.Join("pkg", "quantum", "cupy_gpu.py"),
		filepath.Join(".", "cupy_gpu.py"),
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			if absPath, err := filepath.Abs(path); err == nil {
				return absPath
			}
			return path
		}
	}

	// Default fallback
	return filepath.Join("pkg", "quantum", "cupy_gpu.py")
}
