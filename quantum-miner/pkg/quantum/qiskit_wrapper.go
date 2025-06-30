//go:build !cuda
// +build !cuda

package quantum

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// QiskitGPUSimulator provides CUDA 12.9 GPU-accelerated quantum simulation via Qiskit-Aer
type QiskitGPUSimulator struct {
	deviceID     int
	pythonPath   string
	scriptPath   string
	gpuAvailable bool
	initialized  bool
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

	// Test if Qiskit-Aer GPU is available
	cmd := exec.Command(q.pythonPath, q.scriptPath, "test_gpu")
	output, err := cmd.CombinedOutput() // Use CombinedOutput to get both stdout and stderr

	if err != nil {
		log.Printf("⚠️  GPU initialization failed: %v", err)
		log.Printf("📝 Full output: %s", string(output))
		log.Printf("💡 Diagnosis:")
		
		outputStr := string(output)
		if strings.Contains(outputStr, "ModuleNotFoundError") {
			log.Printf("   • Missing Python packages (qiskit, qiskit-aer, etc.)")
			log.Printf("   • Install with: pip install qiskit qiskit-aer")
		} else if strings.Contains(outputStr, "CUDA") {
			log.Printf("   • CUDA driver/runtime issue")
			log.Printf("   • Check NVIDIA drivers and CUDA installation")
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

// BatchSimulateQuantumPuzzles performs GPU-accelerated batch quantum simulation
func (q *QiskitGPUSimulator) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	if !q.initialized {
		return nil, fmt.Errorf("simulator not initialized")
	}

	log.Printf("🎯 GPU Batch Quantum Simulation: %d puzzles", nPuzzles)
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

	// Execute batch simulation
	cmd := exec.Command(q.pythonPath, q.scriptPath, string(requestJSON))
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("batch simulation failed: %w", err)
	}

	// Parse response
	var response struct {
		Success  bool     `json:"success"`
		Outcomes [][]byte `json:"outcomes"`
		Time     float64  `json:"time"`
		GPUUsed  bool     `json:"gpu_used"`
		Error    string   `json:"error"`
	}

	if err := json.Unmarshal(output, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if !response.Success {
		return nil, fmt.Errorf("simulation failed: %s", response.Error)
	}

	duration := time.Since(start)
	puzzlesPerSec := float64(nPuzzles) / duration.Seconds()

	if response.GPUUsed {
		log.Printf("⚡ GPU Batch Complete: %d puzzles in %.4fs (%.1f puzzles/sec) - CUDA 12.9 ACTIVE",
			nPuzzles, duration.Seconds(), puzzlesPerSec)
	} else {
		log.Printf("💻 CPU Batch Complete: %d puzzles in %.4fs (%.1f puzzles/sec) - GPU fallback",
			nPuzzles, duration.Seconds(), puzzlesPerSec)
	}

	return response.Outcomes, nil
}

// Cleanup releases GPU resources
func (q *QiskitGPUSimulator) Cleanup() {
	if q.initialized {
		log.Printf("🧹 Cleaning up quantum simulator resources")
	}
}

// IsGPUAvailable returns true if CUDA 12.9 GPU acceleration is available
func (q *QiskitGPUSimulator) IsGPUAvailable() bool {
	return q.gpuAvailable
}

// findPython locates the Python executable (embedded first, then system)
func findPython() (string, error) {
	fmt.Println("🔍 Searching for Python executable for Qiskit...")
	
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


