package quantum

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
	"bytes"
	"strings"
)

// CupyGPUSimulator wraps the CuPy-based GPU quantum simulator
type CupyGPUSimulator struct {
	pythonScript string
	available    bool
	deviceInfo   map[string]interface{}
	pythonPath   string // Store the Python executable path
}

// NewCupyGPUSimulator creates a new CuPy GPU simulator instance
func NewCupyGPUSimulator() *CupyGPUSimulator {
	simulator := &CupyGPUSimulator{
		pythonScript: findCupyScript(),
		available:    false,
	}

	// Find Python executable (embedded first, then system)
	simulator.pythonPath = findPythonExecutable()

	// Check if WSL2 mode is enabled - skip CuPy testing in WSL2 mode
	if os.Getenv("WSL2_MODE") == "true" {
		fmt.Println("🐧 WSL2 Mode: Skipping CuPy testing, using Qiskit native GPU backend")
		simulator.available = false
		return simulator
	}

	// Test if CuPy GPU is available (only in non-WSL2 mode)
	simulator.testAvailability()

	return simulator
}

// findPythonExecutable finds the best Python executable to use
func findPythonExecutable() string {
	fmt.Println("🔍 Searching for Python executable...")
	
	// Check for WSL2 mode first
	if os.Getenv("WSL2_MODE") == "true" {
		if pythonExec := os.Getenv("PYTHON_EXEC"); pythonExec != "" {
			fmt.Printf("🐧 WSL2 Mode: Using Linux Python: %s\n", pythonExec)
			if fileExists(pythonExec) {
				return pythonExec
			} else {
				fmt.Printf("⚠️  WSL2 Python not found at: %s\n", pythonExec)
			}
		}
		
		// Try WSL2 Python fallback locations
		wsl2Paths := []string{
			"./go-wsl2/python-linux.sh",
			"../go-wsl2/python-linux.sh",
			"go-wsl2/python-linux.sh",
		}
		
		for _, path := range wsl2Paths {
			if fileExists(path) {
				fmt.Printf("🐧 Found WSL2 Python: %s\n", path)
				return path
			}
		}
		
		fmt.Println("⚠️  WSL2 mode enabled but Linux Python not found, trying system...")
	}
	
	// Get executable directory for embedded Python check
	exePath, err := os.Executable()
	if err != nil {
		fmt.Printf("⚠️  Could not get executable path: %v\n", err)
	} else {
		exeDir := filepath.Dir(exePath)
		
		// Check for embedded python.bat in same directory as executable
		embeddedPython := filepath.Join(exeDir, "python.bat")
		if fileExists(embeddedPython) {
			fmt.Printf("✅ Found embedded Python: %s\n", embeddedPython)
			return embeddedPython
		}
		
		// Check for python.exe in embedded directory
		embeddedPythonExe := filepath.Join(exeDir, "python.exe")
		if fileExists(embeddedPythonExe) {
			fmt.Printf("✅ Found embedded Python executable: %s\n", embeddedPythonExe)
			return embeddedPythonExe
		}
		
		fmt.Printf("ℹ️  No embedded Python found in: %s\n", exeDir)
	}
	
	// Try system Python
	fmt.Println("🔍 Checking system Python...")
	systemPythons := []string{"python", "python3", "py"}
	
	if runtime.GOOS == "windows" {
		// On Windows, also try specific paths
		username := os.Getenv("USERNAME")
		systemPythons = append(systemPythons, 
			"C:\\Users\\"+username+"\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
			"C:\\Users\\"+username+"\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
			"C:\\Python311\\python.exe",
			"C:\\Python310\\python.exe",
		)
	}
	
	for _, cmd := range systemPythons {
		if path, err := exec.LookPath(cmd); err == nil {
			// Test if it's a real Python (not Windows Store stub)
			if runtime.GOOS == "windows" && contains(path, "WindowsApps") {
				fmt.Printf("⚠️  Skipping Windows Store Python stub: %s\n", path)
				continue
			}
			fmt.Printf("✅ Found system Python: %s\n", path)
			return cmd
		}
	}
	
	fmt.Println("❌ No Python executable found!")
	return "python" // Fallback
}

// testAvailability checks if CuPy GPU simulation is available
func (c *CupyGPUSimulator) testAvailability() {
	start := time.Now()

	fmt.Printf("🧪 Testing CuPy GPU availability with: %s\n", c.pythonPath)
	fmt.Printf("📄 Script path: %s\n", c.pythonScript)

	// Test CuPy GPU availability by running test mode
	cmd := exec.Command(c.pythonPath, c.pythonScript)
	
	// Capture both stdout and stderr separately for better debugging
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	err := cmd.Run()

	// Get output and error output
	outputStr := stdout.String()
	errorStr := stderr.String()

	// Always log stderr if present for debugging
	if errorStr != "" {
		fmt.Printf("🔍 GPU Test: Python stderr output:\n%s\n", errorStr)
	}

	if err != nil {
		fmt.Printf("❌ CuPy GPU test failed: %v\n", err)
		
		if outputStr != "" {
			fmt.Printf("📝 Stdout: %s\n", outputStr)
		}
		if errorStr != "" {
			fmt.Printf("📝 Stderr: %s\n", errorStr)
		}
		
		fmt.Printf("💡 Diagnosis:\n")
		
		// Check both stdout and stderr for diagnostic information
		fullOutput := outputStr + " " + errorStr
		
		if contains(fullOutput, "ModuleNotFoundError") || contains(fullOutput, "No module named") {
			fmt.Printf("   • Missing Python packages (cupy, numpy, etc.)\n")
			fmt.Printf("   • Install with: pip install cupy-cuda12x numpy\n")
		} else if contains(fullOutput, "CUDA") || contains(fullOutput, "cuda") {
			fmt.Printf("   • CUDA driver/runtime issue\n")
			fmt.Printf("   • Check NVIDIA drivers and CUDA installation\n")
		} else if contains(err.Error(), "cannot run executable") || contains(err.Error(), "executable file not found") {
			fmt.Printf("   • Python executable not found or not accessible\n")
			fmt.Printf("   • Current Python path: %s\n", c.pythonPath)
		} else if contains(err.Error(), "exit status 1") {
			fmt.Printf("   • Python script execution error (see stderr output above)\n")
			fmt.Printf("   • Check Python dependencies and GPU drivers\n")
		} else {
			fmt.Printf("   • Unknown GPU initialization error\n")
			fmt.Printf("   • Check Python installation and GPU drivers\n")
		}
		
		c.available = false
		return
	}

	// Parse test output to check for GPU availability
	fmt.Printf("📊 GPU Test Output: %s\n", outputStr)
	
	if contains(outputStr, "GPU Acceleration Available") || contains(outputStr, "Backend: cupy_gpu") {
		c.available = true
		fmt.Printf("✅ CuPy GPU simulator available (test took %v)\n", time.Since(start))

		// Extract device info if possible
		c.deviceInfo = map[string]interface{}{
			"backend": "cupy_gpu",
			"tested":  true,
		}
	} else {
		fmt.Printf("⚠️  CuPy falling back to CPU (no GPU acceleration detected)\n")
		fmt.Printf("💡 This usually means CuPy is working but no compatible GPU was found\n")
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

	// FIXED: Use stdin instead of command line arguments to avoid potential length issues
	cmd := exec.Command(c.pythonPath, c.pythonScript, "--stdin")
	
	// Set up stdin pipe to send JSON data
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %v", err)
	}
	
	// Capture both stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Start the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start command: %v", err)
	}
	
	// Send JSON data via stdin
	go func() {
		defer stdin.Close()
		stdin.Write(inputJSON)
	}()
	
	// Wait for completion
	err = cmd.Wait()

	// Get output and error output
	outputStr := stdout.String()
	errorStr := stderr.String()

	// Only log stderr if there are actual errors (not INFO messages)
	if errorStr != "" && !contains(errorStr, "INFO:") {
		cleanErrorStr := strings.ReplaceAll(errorStr, "\x00", "")
		cleanErrorStr = strings.TrimSpace(cleanErrorStr)
		if cleanErrorStr != "" {
			fmt.Printf("🔍 GPU: Python stderr output:\n%s\n", cleanErrorStr)
		}
	}

	if err != nil {
		// Provide detailed error information
		errorMsg := fmt.Sprintf("CuPy simulation failed: %v", err)
		if errorStr != "" {
			errorMsg += fmt.Sprintf("\nStderr: %s", errorStr)
		}
		if outputStr != "" {
			errorMsg += fmt.Sprintf("\nStdout: %s", outputStr)
		}
		return nil, fmt.Errorf(errorMsg)
	}

	// Parse result from stdout
	var result map[string]interface{}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		errorMsg := fmt.Sprintf("failed to parse result: %v", err)
		if outputStr != "" {
			errorMsg += fmt.Sprintf("\nRaw output: %s", outputStr)
		}
		if errorStr != "" {
			errorMsg += fmt.Sprintf("\nError output: %s", errorStr)
		}
		return nil, fmt.Errorf(errorMsg)
	}

	// Check for errors or interruptions
	if status, ok := result["status"].(string); ok {
		if status == "interrupted" {
			return nil, fmt.Errorf("simulation interrupted by user")
		}
		if status != "success" {
			if message, ok := result["message"].(string); ok {
				return nil, fmt.Errorf("simulation error: %s", message)
			}
			return nil, fmt.Errorf("simulation failed with status: %s", status)
		}
	}

	// Create a simple result summary
	simResult := map[string]interface{}{
		"backend": result["backend"],
		"success": result["success"],
	}

	return simResult, nil
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

	// FIXED: Use stdin instead of command line arguments to avoid "command line too long" error
	cmd := exec.CommandContext(ctx, c.pythonPath, c.pythonScript, "--stdin")
	
	// Set up stdin pipe to send JSON data
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %v", err)
	}
	
	// Capture both stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Start the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start command: %v", err)
	}
	
	// Send JSON data via stdin (this avoids command line length limits)
	go func() {
		defer stdin.Close()
		stdin.Write(inputJSON)
	}()
	
	// Wait for completion
	err = cmd.Wait()

	// Get output and error output
	outputStr := stdout.String()
	errorStr := stderr.String()

	// Only log stderr if there are actual errors (not INFO messages)
	if errorStr != "" && !contains(errorStr, "INFO:") {
		cleanErrorStr := strings.ReplaceAll(errorStr, "\x00", "")
		cleanErrorStr = strings.TrimSpace(cleanErrorStr)
		if cleanErrorStr != "" {
			fmt.Printf("🔍 GPU: Python stderr output:\n%s\n", cleanErrorStr)
		}
	}

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
		
		// Provide detailed error information
		errorMsg := fmt.Sprintf("CuPy batch simulation failed: %v", err)
		if errorStr != "" {
			errorMsg += fmt.Sprintf("\nStderr: %s", errorStr)
		}
		return nil, fmt.Errorf(errorMsg)
	}

	// Parse result from stdout
	var result map[string]interface{}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		errorMsg := fmt.Sprintf("failed to parse result: %v", err)
		if outputStr != "" {
			errorMsg += fmt.Sprintf("\nRaw output: %s", outputStr)
		}
		if errorStr != "" {
			errorMsg += fmt.Sprintf("\nError output: %s", errorStr)
		}
		return nil, fmt.Errorf(errorMsg)
	}

	// Check for errors or interruptions
	if status, ok := result["status"].(string); ok {
		if status == "interrupted" {
			return nil, fmt.Errorf("simulation interrupted by user")
		}
		if status != "success" {
			if message, ok := result["message"].(string); ok {
				return nil, fmt.Errorf("batch simulation error: %s", message)
			}
			return nil, fmt.Errorf("batch simulation failed with status: %s", status)
		}
	}

	// Create dummy results based on the puzzle count
	var dummyResults []map[string]interface{}
	if puzzleCount, ok := result["puzzle_count"].(float64); ok {
		count := int(puzzleCount)
		dummyResults = make([]map[string]interface{}, count)
		
		backend := "unknown"
		if b, ok := result["backend"].(string); ok {
			backend = b
		}
		
		for i := 0; i < count; i++ {
			dummyResults[i] = map[string]interface{}{
				"puzzle_id": i,
				"backend":   backend,
				"success":   true,
			}
		}
	}

	// Only show a simple success message
	fmt.Printf("✅ GPU: Batch completed (%d puzzles)\n", len(dummyResults))
	return dummyResults, nil
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


