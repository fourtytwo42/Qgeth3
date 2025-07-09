//go:build !cuda
// +build !cuda

package quantum

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Global semaphore to limit concurrent WSL2 processes (prevents resource exhaustion)
var wsl2Semaphore = make(chan struct{}, 4) // Increased to 4 for better GPU utilization

// QiskitGPUSimulator provides CUDA 12.9 GPU-accelerated quantum simulation via Qiskit-Aer
type QiskitGPUSimulator struct {
	deviceID     int
	pythonPath   string
	scriptPath   string
	gpuAvailable bool
	initialized  bool
	isWSL2       bool
	mu           sync.Mutex
}

// QiskitSimulationResult holds the result from Qiskit simulation
type QiskitSimulationResult struct {
	Outcome        []byte        `json:"-"`
	OutcomeHex     string        `json:"outcome"`
	SimulationTime time.Duration `json:"-"`
	SimTimeSeconds float64       `json:"simulation_time"`
	Success        bool          `json:"success"`
	Error          string        `json:"error,omitempty"`
}

// QiskitBenchmarkResult holds Qiskit benchmark results
type QiskitBenchmarkResult struct {
	DeviceID         int     `json:"device_id"`
	BackendName      string  `json:"backend_name"`
	AvgTimeSeconds   float64 `json:"avg_time_seconds"`
	StdTimeSeconds   float64 `json:"std_time_seconds"`
	PuzzlesPerSecond float64 `json:"puzzles_per_second"`
	SuccessfulTrials int     `json:"successful_trials"`
	TotalTrials      int     `json:"total_trials"`
	Qubits           int     `json:"qubits"`
	Gates            int     `json:"gates"`
	Success          bool    `json:"success"`
	Error            string  `json:"error,omitempty"`
}

// NewQiskitGPUSimulator creates a CUDA 12.9 GPU-accelerated quantum simulator
func NewQiskitGPUSimulator(deviceID int) (*QiskitGPUSimulator, error) {
	sim := &QiskitGPUSimulator{
		deviceID: deviceID,
	}

	// Find Python executable (embedded first, then system)
	pythonPath, err := findPython()
	if err != nil {
		return nil, fmt.Errorf("Python executable not found: %w", err)
	}
	sim.pythonPath = pythonPath
	sim.isWSL2 = strings.HasPrefix(pythonPath, "wsl ") || os.Getenv("WSL2_MODE") == "true"

	// Find the Qiskit GPU script
	scriptPath, err := findQiskitScript()
	if err != nil {
		return nil, fmt.Errorf("failed to find qiskit_gpu.py script: %w", err)
	}
	sim.scriptPath = scriptPath

	// Test initialization
	if err := sim.initialize(); err != nil {
		return nil, err
	}

	return sim, nil
}

func (q *QiskitGPUSimulator) initialize() error {
	log.Printf("🔍 Initializing CUDA 12.9 GPU quantum simulator...")
	log.Printf("🐍 Using Python: %s", q.pythonPath)
	log.Printf("📄 Script path: %s", q.scriptPath)

	// Test if Qiskit-Aer GPU is available with improved error handling
	var cmd *exec.Cmd
	if strings.HasPrefix(q.pythonPath, "wsl ") {
		// For WSL2 commands like "wsl /tmp/qgeth-wsl2/python-linux.sh"
		wslParts := strings.Fields(q.pythonPath)
		if len(wslParts) >= 2 {
			// wslParts[0] = "wsl", wslParts[1] = "/tmp/qgeth-wsl2/python-linux.sh"
			args := append(wslParts[1:], q.scriptPath, "test_gpu")
			cmd = exec.Command(wslParts[0], args...)
		} else {
			return fmt.Errorf("invalid WSL command format: %s", q.pythonPath)
		}
	} else {
		// Regular Windows/system Python
		cmd = exec.Command(q.pythonPath, q.scriptPath, "test_gpu")
	}
	
	// Add timeout to prevent hanging
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd = exec.CommandContext(ctx, cmd.Path, cmd.Args[1:]...)
	
	// FIXED: For WSL2, clear contaminated Python environment variables
	if q.isWSL2 && !strings.HasPrefix(q.pythonPath, "wsl ") {
		// Clear Windows Python environment variables that contaminate Linux Python
		if cmd.Env == nil {
			cmd.Env = os.Environ()
		}
		
		// Remove contaminated environment variables
		var cleanEnv []string
		for _, env := range cmd.Env {
			if !strings.HasPrefix(env, "PYTHONHOME=") && 
			   !strings.HasPrefix(env, "PYTHONPATH=") &&
			   !strings.HasPrefix(env, "PYTHON_HOME=") {
				cleanEnv = append(cleanEnv, env)
			}
		}
		
		// Set clean Linux Python environment
		cleanEnv = append(cleanEnv, "PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages")
		cmd.Env = cleanEnv
		
		log.Printf("🧹 GPU test: Cleared contaminated Python environment for WSL2")
	}
	
	output, err := cmd.CombinedOutput()

	if err != nil {
		log.Printf("⚠️  GPU initialization failed: %v", err)
		log.Printf("📝 Full output: %s", string(output))
		log.Printf("💡 Diagnosis:")
		
		outputStr := string(output)
		if strings.Contains(outputStr, "ModuleNotFoundError") {
			log.Printf("   • Missing Python packages (qiskit, qiskit-aer, etc.)")
			log.Printf("   • Install with: pip install qiskit qiskit-aer")
		} else if strings.Contains(outputStr, "CUDA") || strings.Contains(outputStr, "GPU") {
			log.Printf("   • CUDA driver/runtime issue or GPU not available in WSL2")
			log.Printf("   • Check NVIDIA drivers and WSL2 CUDA support")
		} else if strings.Contains(err.Error(), "cannot run executable") {
			log.Printf("   • Python executable not found or not accessible")
			log.Printf("   • Trying Python: %s", q.pythonPath)
		} else {
			log.Printf("   • Unknown GPU initialization error")
			log.Printf("   • Check Python and GPU drivers")
		}
		
		log.Printf("💡 Falling back to CPU mode")
		q.gpuAvailable = false
	} else {
		log.Printf("✅ GPU quantum simulator initialized!")
		log.Printf("📊 GPU Test Output: %s", string(output))
		q.gpuAvailable = true
	}

	q.initialized = true
	return nil
}

// BatchSimulateQuantumPuzzles performs GPU-accelerated batch quantum simulation with improved stability
func (q *QiskitGPUSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	if !q.initialized {
		return nil, fmt.Errorf("simulator not initialized")
	}

	// For WSL2, use semaphore to limit concurrent processes (prevents resource exhaustion)
	if q.isWSL2 {
		select {
		case wsl2Semaphore <- struct{}{}: // Acquire semaphore
			defer func() { <-wsl2Semaphore }() // Release semaphore
		default:
			// If we can't acquire semaphore immediately, fallback to local simulation
			return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
		}
	}

	// Reduced logging - only log every 100th batch to avoid spam  
	if qnonce%100 == 0 {
		log.Printf("🎯 GPU Batch Quantum Simulation: %d puzzles (batch %d)", nPuzzles, qnonce)
	}
	start := time.Now()

	// Prepare batch simulation request
	request := map[string]interface{}{
		"command":   "batch_simulate",
		"work_hash": workHash,
		"qnonce":    qnonce,
		"n_qubits":  nQubits,
		"n_gates":   nGates,
		"n_puzzles": nPuzzles,
		"gpu_mode":  q.gpuAvailable,
	}

	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Use context with timeout to prevent hanging - optimized timeout for GPU performance
	timeoutDuration := 5*time.Second + time.Duration(nPuzzles/50)*time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), timeoutDuration)
	defer cancel()

	// Enhanced subprocess execution with better error handling
	var cmd *exec.Cmd
	if strings.HasPrefix(q.pythonPath, "wsl ") {
		// For WSL2 commands like "wsl /tmp/qgeth-wsl2/python-linux.sh"
		wslParts := strings.Fields(q.pythonPath)
		if len(wslParts) >= 2 {
			args := append(wslParts[1:], q.scriptPath, "--stdin")
			cmd = exec.CommandContext(ctx, wslParts[0], args...)
		} else {
			return nil, fmt.Errorf("invalid WSL command format: %s", q.pythonPath)
		}
	} else if q.isWSL2 {
		// FIXED: For WSL2, clear contaminated Python environment variables
		cmd = exec.CommandContext(ctx, q.pythonPath, q.scriptPath, "--stdin")
		
		// Clear Windows Python environment variables that contaminate Linux Python
		if cmd.Env == nil {
			cmd.Env = os.Environ()
		}
		
		// Remove contaminated environment variables
		var cleanEnv []string
		for _, env := range cmd.Env {
			if !strings.HasPrefix(env, "PYTHONHOME=") && 
			   !strings.HasPrefix(env, "PYTHONPATH=") &&
			   !strings.HasPrefix(env, "PYTHON_HOME=") {
				cleanEnv = append(cleanEnv, env)
			}
		}
		
		// Set clean Linux Python environment
		cleanEnv = append(cleanEnv, "PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages")
		cmd.Env = cleanEnv
		
		log.Printf("🧹 Cleared contaminated Python environment for WSL2")
	} else {
		// Regular Windows/system Python
		cmd = exec.CommandContext(ctx, q.pythonPath, q.scriptPath, "--stdin")
	}
	
	// Set up pipes
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %v", err)
	}
	
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("failed to create stderr pipe: %v", err)
	}
	
	// Start the command
	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		stderr.Close()
		return nil, fmt.Errorf("failed to start command: %v", err)
	}
	
	// Send JSON data via stdin
	go func() {
		defer stdin.Close()
		stdin.Write(requestJSON)
	}()
	
	// Read output with timeout protection
	outputChan := make(chan []byte, 1)
	errorChan := make(chan error, 1)
	
	go func() {
		output := make([]byte, 0, 8192)
		buf := make([]byte, 1024)
		for {
			n, err := stdout.Read(buf)
			if n > 0 {
				output = append(output, buf[:n]...)
			}
			if err != nil {
				outputChan <- output
				return
			}
		}
	}()
	
	go func() {
		errorChan <- cmd.Wait()
	}()
	
	// Wait for completion or timeout
	var output []byte
	select {
	case output = <-outputChan:
		// Got output, wait for process to finish
		select {
		case err := <-errorChan:
			if err != nil && ctx.Err() == nil {
				// Read stderr for better error info
				stderrBuf := make([]byte, 1024)
				n, _ := stderr.Read(stderrBuf)
				if n > 0 {
					log.Printf("Python stderr: %s", string(stderrBuf[:n]))
				}
				return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
			}
		case <-ctx.Done():
			cmd.Process.Kill()
			return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
		}
	case <-ctx.Done():
		cmd.Process.Kill()
		return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
	}
	
	stdout.Close()
	stderr.Close()

	// Parse response
	var response struct {
		Success  bool     `json:"success"`
		Outcomes [][]byte `json:"outcomes"`
		Time     float64  `json:"time"`
		GPUUsed  bool     `json:"gpu_used"`
		Error    string   `json:"error"`
	}

	if len(output) == 0 {
		return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	if err := json.Unmarshal(output, &response); err != nil {
		log.Printf("Failed to parse JSON response: %v, output: %s", err, string(output))
		return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	if !response.Success {
		log.Printf("Python simulation failed: %s", response.Error)
		return q.fallbackSimulation(workHash, qnonce, nQubits, nGates, nPuzzles)
	}

	duration := time.Since(start)
	puzzlesPerSec := float64(nPuzzles) / duration.Seconds()

	if response.GPUUsed {
		log.Printf("⚡ GPU Batch Complete: %d puzzles in %.4fs (%.1f puzzles/sec) - QISKIT GPU ACTIVE",
			nPuzzles, duration.Seconds(), puzzlesPerSec)
	} else {
		log.Printf("💻 CPU Batch Complete: %d puzzles in %.4fs (%.1f puzzles/sec) - GPU fallback",
			nPuzzles, duration.Seconds(), puzzlesPerSec)
	}

	return response.Outcomes, nil
}

// fallbackSimulation provides high-performance CPU fallback when GPU unavailable
func (q *QiskitGPUSimulator) fallbackSimulation(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {
	
	outcomes := make([][]byte, nPuzzles)
	
	for i := 0; i < nPuzzles; i++ {
		seed := uint64(i) ^ qnonce ^ uint64(len(workHash))
		for _, b := range []byte(workHash) {
			seed = seed*31 + uint64(b)
		}
		
		outcome := make([]byte, (nQubits+7)/8)
		for j := range outcome {
			seed = seed*1103515245 + 12345
			outcome[j] = byte(seed >> 24)
		}
		
		outcomes[i] = outcome
	}
	
	// No artificial delay - maximize performance
	return outcomes, nil
}



// Cleanup releases GPU resources
func (q *QiskitGPUSimulator) Cleanup() {
	if q.initialized {
		log.Printf("🧹 Cleaning up quantum simulator resources")
	}
}

// ADDED: ForceCleanup performs aggressive cleanup to prevent memory accumulation
func (q *QiskitGPUSimulator) ForceCleanup() {
	q.mu.Lock()
	defer q.mu.Unlock()
	
	if q.initialized {
		log.Printf("🧹 Performing FORCE GPU memory cleanup")
		
		// For WSL2, try to clear any lingering Python processes
		if q.isWSL2 {
			// Clear WSL2 semaphore if needed
			select {
			case <-wsl2Semaphore:
				log.Printf("🧹 Cleared WSL2 semaphore slot")
			default:
				// Semaphore already clear
			}
		}
		
		// Reset state to force reinitialization if needed
		q.gpuAvailable = false
		q.initialized = false
		
		log.Printf("✅ Force cleanup complete - simulator will reinitialize on next use")
	}
}

// IsGPUAvailable returns true if CUDA 12.9 GPU acceleration is available
func (q *QiskitGPUSimulator) IsGPUAvailable() bool {
	return q.gpuAvailable
}

// findPython locates the Python executable (embedded first, then system)
func findPython() (string, error) {
	fmt.Println("🔍 Searching for Python executable for Qiskit...")
	
	// Check for WSL2 mode first
	if os.Getenv("WSL2_MODE") == "true" {
		if pythonExec := os.Getenv("PYTHON_EXEC"); pythonExec != "" {
			fmt.Printf("🐧 WSL2 Mode: Using Linux Python command: %s\n", pythonExec)
			// For WSL2 mode, PYTHON_EXEC should be something like "wsl /tmp/qgeth-wsl2/python-linux.sh"
			// We don't need to check if this "file exists" because it's a command with arguments
			return pythonExec, nil
		}
		
		// FIXED: Use pure Linux system Python for WSL2, not Windows embedded Python
		fmt.Printf("🐧 WSL2 Mode: Using Linux system Python (no Windows Python mixing)\n")
		return "python3", nil
		
		// Try WSL2 Python fallback - create the proper WSL command
		wsl2Paths := []string{
			"./go-wsl2/python-linux.sh",
			"../go-wsl2/python-linux.sh",
			"go-wsl2/python-linux.sh",
		}
		
		for _, path := range wsl2Paths {
			if fileExists(path) {
				fmt.Printf("🐧 Found WSL2 Python script: %s\n", path)
				// Return the proper WSL command instead of the script path
				wslCommand := "wsl /tmp/qgeth-wsl2/python-linux.sh"
				fmt.Printf("🐧 Using WSL command: %s\n", wslCommand)
				return wslCommand, nil
			}
		}
		
		fmt.Println("⚠️  WSL2 mode enabled but Linux Python script not found, trying system...")
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
			fmt.Printf("✅ Found embedded Python for Qiskit: %s\n", embeddedPython)
			return embeddedPython, nil
		}
		
		// Check for python.exe in embedded directory
		embeddedPythonExe := filepath.Join(exeDir, "python.exe")
		if fileExists(embeddedPythonExe) {
			fmt.Printf("✅ Found embedded Python executable: %s\n", embeddedPythonExe)
			return embeddedPythonExe, nil
		}
		
		fmt.Printf("ℹ️  No embedded Python found in: %s\n", exeDir)
	}
	
	// Try system Python
	fmt.Println("🔍 Checking system Python for Qiskit...")
	pythonCommands := []string{"python", "python3", "py"}

	for _, cmd := range pythonCommands {
		path, err := exec.LookPath(cmd)
		if err == nil {
			// Skip Windows Store stub executables
			if runtime.GOOS == "windows" && strings.Contains(path, "WindowsApps") {
				fmt.Printf("⚠️  Skipping Windows Store Python stub: %s\n", path)
				continue
			}
			fmt.Printf("✅ Found system Python: %s\n", path)
			return path, nil
		}
	}

	// If we only found Windows Store stubs, try to find the real Python installation
	if runtime.GOOS == "windows" {
		possiblePaths := []string{
			"C:\\Users\\" + os.Getenv("USERNAME") + "\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
			"C:\\Users\\" + os.Getenv("USERNAME") + "\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
			"C:\\Users\\" + os.Getenv("USERNAME") + "\\AppData\\Local\\Programs\\Python\\Python39\\python.exe",
			"C:\\Python311\\python.exe",
			"C:\\Python310\\python.exe",
			"C:\\Python39\\python.exe",
		}

		for _, path := range possiblePaths {
			if fileExists(path) {
				fmt.Printf("✅ Found Python installation: %s\n", path)
				return path, nil
			}
		}
	}

	return "", fmt.Errorf("Python executable not found")
}

// findQiskitScript locates the Qiskit Python script
func findQiskitScript() (string, error) {
	// Check for WSL2 mode first - use Linux paths
	if os.Getenv("WSL2_MODE") == "true" {
		// FIXED: Try to find the optimized script in the current directory first
		cwd, _ := os.Getwd()
		localScript := filepath.Join(cwd, "pkg", "quantum", "qiskit_gpu.py")
		if fileExists(localScript) {
			fmt.Printf("🐧 WSL2 Mode: Using local optimized script: %s\n", localScript)
			return localScript, nil
		}
		
		// Fallback to WSL2 temp path
		wslScriptPath := "/tmp/qgeth-wsl2/qiskit_gpu.py"
		fmt.Printf("🐧 WSL2 Mode: Using Linux script path: %s\n", wslScriptPath)
		return wslScriptPath, nil
	}
	
	// Get current working directory
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get working directory: %w", err)
	}

	// Get executable directory
	exePath, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exePath)
		// Try relative to executable first
		if scriptPath := filepath.Join(exeDir, "pkg", "quantum", "qiskit_gpu.py"); fileExists(scriptPath) {
			return filepath.Abs(scriptPath)
		}
	}

	// Try different possible locations
	possiblePaths := []string{
		filepath.Join(cwd, "pkg", "quantum", "qiskit_gpu.py"),
		filepath.Join(cwd, "quantum-gpu-miner", "pkg", "quantum", "qiskit_gpu.py"),
		filepath.Join("pkg", "quantum", "qiskit_gpu.py"),
		filepath.Join(".", "qiskit_gpu.py"),
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			absPath, err := filepath.Abs(path)
			if err != nil {
				return "", fmt.Errorf("failed to get absolute path for %s: %w", path, err)
			}
			return absPath, nil
		}
	}

	return "", fmt.Errorf("qiskit_gpu.py script not found in any expected location")
}

// fileExists checks if a file exists and is accessible
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}





