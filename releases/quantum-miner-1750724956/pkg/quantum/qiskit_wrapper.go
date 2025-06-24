package quantum

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// QiskitBackend provides access to Qiskit GPU acceleration
type QiskitBackend struct {
	deviceID   int
	pythonPath string
	scriptPath string
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

// NewQiskitBackend creates a new Qiskit backend instance
func NewQiskitBackend(deviceID int) (*QiskitBackend, error) {
	// Find Python executable
	pythonPath, err := findPython()
	if err != nil {
		return nil, fmt.Errorf("Python not found: %w", err)
	}

	// Find Qiskit script path
	scriptPath, err := findQiskitScript()
	if err != nil {
		return nil, fmt.Errorf("Qiskit script not found: %w", err)
	}

	backend := &QiskitBackend{
		deviceID:   deviceID,
		pythonPath: pythonPath,
		scriptPath: scriptPath,
	}

	// Test the backend
	if err := backend.Test(); err != nil {
		return nil, fmt.Errorf("Qiskit backend test failed: %w", err)
	}

	return backend, nil
}

// Test tests the Qiskit backend functionality
func (q *QiskitBackend) Test() error {
	cmd := exec.Command(q.pythonPath, q.scriptPath, "test", strconv.Itoa(q.deviceID))
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("Qiskit test command failed: %w", err)
	}

	var result QiskitSimulationResult
	if err := json.Unmarshal(output, &result); err != nil {
		return fmt.Errorf("failed to parse Qiskit test result: %w", err)
	}

	if !result.Success {
		return fmt.Errorf("Qiskit test failed: %s", result.Error)
	}

	return nil
}

// SimulateQuantumPuzzle simulates a quantum puzzle using Qiskit GPU backend
func (q *QiskitBackend) SimulateQuantumPuzzle(puzzleIndex int, workHash string, qnonce uint64,
	nQubits, nGates int) ([]byte, error) {

	// Run Qiskit Python script
	cmd := exec.Command(
		q.pythonPath,
		q.scriptPath,
		"simulate",
		strconv.Itoa(puzzleIndex),
		workHash,
		strconv.FormatUint(qnonce, 10),
		strconv.Itoa(nQubits),
		strconv.Itoa(nGates),
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("Qiskit simulation command failed: %w", err)
	}

	var result QiskitSimulationResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse Qiskit simulation result: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("Qiskit simulation failed: %s", result.Error)
	}

	// Decode hex outcome to bytes
	outcome, err := hex.DecodeString(result.OutcomeHex)
	if err != nil {
		return nil, fmt.Errorf("failed to decode Qiskit outcome: %w", err)
	}

	return outcome, nil
}

// Benchmark runs a performance benchmark of the Qiskit backend
func (q *QiskitBackend) Benchmark(nTrials int) (*QiskitBenchmarkResult, error) {
	cmd := exec.Command(
		q.pythonPath,
		q.scriptPath,
		"benchmark",
		strconv.Itoa(q.deviceID),
		strconv.Itoa(nTrials),
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("Qiskit benchmark command failed: %w", err)
	}

	var result QiskitBenchmarkResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse Qiskit benchmark result: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("Qiskit benchmark failed: %s", result.Error)
	}

	return &result, nil
}

// BatchSimulateQuantumPuzzles simulates multiple quantum puzzles using Qiskit GPU backend
func (q *QiskitBackend) BatchSimulateQuantumPuzzles(workHash string, qnonce uint64,
	nQubits, nGates, nPuzzles int) ([][]byte, error) {

	// Run Qiskit Python script for batch simulation
	cmd := exec.Command(
		q.pythonPath,
		q.scriptPath,
		"batch_simulate",
		workHash,
		strconv.FormatUint(qnonce, 10),
		strconv.Itoa(nQubits),
		strconv.Itoa(nGates),
		strconv.Itoa(nPuzzles),
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("Qiskit batch simulation command failed: %w", err)
	}

	var result struct {
		Outcomes []string `json:"outcomes"`
		Success  bool     `json:"success"`
		Error    string   `json:"error,omitempty"`
	}

	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse Qiskit batch simulation result: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("Qiskit batch simulation failed: %s", result.Error)
	}

	// Decode hex outcomes to bytes
	outcomes := make([][]byte, len(result.Outcomes))
	for i, outcomeHex := range result.Outcomes {
		outcome, err := hex.DecodeString(outcomeHex)
		if err != nil {
			return nil, fmt.Errorf("failed to decode Qiskit outcome %d: %w", i, err)
		}
		outcomes[i] = outcome
	}

	return outcomes, nil
}

// findPython locates the Python executable
func findPython() (string, error) {
	// Try common Python executable names, prioritizing actual installations
	pythonCommands := []string{"python", "python3", "py"}

	for _, cmd := range pythonCommands {
		path, err := exec.LookPath(cmd)
		if err == nil {
			// Skip Windows Store stub executables
			if !strings.Contains(path, "WindowsApps") {
				return path, nil
			}
		}
	}

	// If we only found Windows Store stubs, try to find the real Python installation
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
			return path, nil
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

// fileExists checks if a file exists
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
