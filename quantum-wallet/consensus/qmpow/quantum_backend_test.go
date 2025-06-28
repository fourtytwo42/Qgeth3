// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"strings"
	"testing"
)

func TestSimulatorBackend_Basic(t *testing.T) {
	backend := NewSimulatorBackend()

	// Test basic properties
	if backend.GetName() != "QuantumSimulator_v1.0" {
		t.Errorf("Expected name 'QuantumSimulator_v1.0', got '%s'", backend.GetName())
	}

	if !backend.IsAvailable() {
		t.Error("Simulator backend should be available")
	}

	// Test capabilities
	caps := backend.GetCapabilities()
	if caps.MaxQubits != 32 {
		t.Errorf("Expected MaxQubits 32, got %d", caps.MaxQubits)
	}
	if !caps.IsSimulator {
		t.Error("Backend should be marked as simulator")
	}
	if !caps.SupportsNoise {
		t.Error("Backend should support noise models")
	}
}

func TestSimulatorBackend_Execute(t *testing.T) {
	backend := NewSimulatorBackend()

	qasm := `qreg q[16];
creg c[16];
H q[0];
T q[1];
CX q[0],q[1];
measure q -> c;`

	seed := []byte("test_seed_12345678901234567890")

	outcome1, err := backend.Execute(qasm, seed)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Test determinism - same inputs should give same outputs
	outcome2, err := backend.Execute(qasm, seed)
	if err != nil {
		t.Fatalf("Second execute failed: %v", err)
	}

	if outcome1 != outcome2 {
		t.Errorf("Expected deterministic outcomes, got %d and %d", outcome1, outcome2)
	}

	// Test different seed gives different outcome
	differentSeed := []byte("different_seed_123456789012345")
	outcome3, err := backend.Execute(qasm, differentSeed)
	if err != nil {
		t.Fatalf("Third execute failed: %v", err)
	}

	if outcome1 == outcome3 {
		t.Error("Different seeds should produce different outcomes")
	}

	// Check stats were updated
	stats := backend.GetStats()
	if stats.TotalExecutions != 3 {
		t.Errorf("Expected 3 total executions, got %d", stats.TotalExecutions)
	}
	if stats.SuccessfulRuns != 3 {
		t.Errorf("Expected 3 successful runs, got %d", stats.SuccessfulRuns)
	}
}

func TestSimulatorBackend_NoiseModel(t *testing.T) {
	backend := NewSimulatorBackend()

	// Configure with noise
	noiseModel := CreateNoiseModel("light")
	config := BackendConfig{
		NoiseModel:   noiseModel,
		Shots:        1024,
		Optimization: 0,
		Timeout:      5000,
		DebugMode:    true,
	}

	err := backend.Configure(config)
	if err != nil {
		t.Fatalf("Configure failed: %v", err)
	}

	qasm := `qreg q[16];
creg c[16];
H q[0];
T q[1];
T q[2];
T q[3];
CX q[0],q[1];
CX q[1],q[2];
measure q -> c;`

	seed := []byte("noise_test_seed_1234567890123456")

	// Execute multiple times to see if noise introduces variation
	outcomes := make(map[uint16]int)
	for i := 0; i < 10; i++ {
		outcome, err := backend.Execute(qasm, seed)
		if err != nil {
			t.Fatalf("Noisy execute failed: %v", err)
		}
		outcomes[outcome]++
	}

	// With light noise, we might see some variation
	t.Logf("Noise test outcomes: %v", outcomes)
}

func TestBackendConfiguration(t *testing.T) {
	backend := NewSimulatorBackend()

	// Test invalid configurations
	invalidConfigs := []BackendConfig{
		{Shots: 0},         // Invalid shots
		{Shots: 200000},    // Too many shots
		{Optimization: -1}, // Invalid optimization
		{Optimization: 4},  // Too high optimization
		{Timeout: 50},      // Too short timeout
		{Timeout: 400000},  // Too long timeout
	}

	for i, config := range invalidConfigs {
		config.NoiseModel = CreateNoiseModel("ideal")
		err := backend.Configure(config)
		if err == nil {
			t.Errorf("Expected error for invalid config %d, but got none", i)
		}
	}

	// Test valid configuration
	validConfig := BackendConfig{
		NoiseModel:   CreateNoiseModel("realistic"),
		Shots:        2048,
		Optimization: 2,
		Timeout:      10000,
		DebugMode:    true,
	}

	err := backend.Configure(validConfig)
	if err != nil {
		t.Errorf("Valid configuration should not fail: %v", err)
	}
}

func TestNoiseModels(t *testing.T) {
	testCases := []struct {
		modelType string
		enabled   bool
		name      string
	}{
		{"ideal", false, "Ideal (No Noise)"},
		{"light", true, "Light Noise"},
		{"realistic", true, "Realistic Hardware"},
		{"heavy", true, "Heavy Noise"},
		{"unknown", false, "Ideal (No Noise)"}, // Falls back to ideal
	}

	for _, tc := range testCases {
		model := CreateNoiseModel(tc.modelType)
		if model.Enabled != tc.enabled {
			t.Errorf("Model %s: expected enabled=%v, got %v", tc.modelType, tc.enabled, model.Enabled)
		}
		if model.Name != tc.name {
			t.Errorf("Model %s: expected name='%s', got '%s'", tc.modelType, tc.name, model.Name)
		}
	}
}

func TestBackendManager(t *testing.T) {
	manager := NewBackendManager()

	// Test initial state
	if manager.GetDefaultBackend() == nil {
		t.Error("Manager should have a default backend")
	}

	available := manager.GetAvailableBackends()
	if len(available) == 0 {
		t.Error("Manager should have at least one available backend")
	}

	// Test backend lookup
	simulator := manager.GetBackendByName("QuantumSimulator_v1.0")
	if simulator == nil {
		t.Error("Should find simulator backend by name")
	}

	nonExistent := manager.GetBackendByName("NonExistentBackend")
	if nonExistent != nil {
		t.Error("Should return nil for non-existent backend")
	}

	// Test backend switching
	originalDefault := manager.GetDefaultBackend()
	manager.SetDefaultBackend(simulator)
	newDefault := manager.GetDefaultBackend()

	if originalDefault != newDefault && simulator.GetName() != newDefault.GetName() {
		t.Error("Backend switching failed")
	}

	// Test stats
	stats := manager.GetManagerStats()
	if stats.TotalBackends == 0 {
		t.Error("Manager should track total backends")
	}
}

func TestBackendSelection(t *testing.T) {
	manager := NewBackendManager()

	// Test best backend selection
	best := manager.GetBestBackend(16, 1000, true) // Prefer simulator
	if best == nil {
		t.Error("Should find a suitable backend")
	}

	if !best.GetCapabilities().IsSimulator {
		t.Error("Should prefer simulator when requested")
	}

	// Test with requirements that exceed capabilities
	impossible := manager.GetBestBackend(1000, 100000, false)
	if impossible != nil {
		t.Error("Should return nil when no backend meets requirements")
	}
}

func TestBackendCompatibility(t *testing.T) {
	backend := NewSimulatorBackend()

	// Test valid compatibility
	err := ValidateBackendCompatibility(backend, 16, 8192)
	if err != nil {
		t.Errorf("Valid parameters should be compatible: %v", err)
	}

	// Test invalid compatibility - too many qubits
	err = ValidateBackendCompatibility(backend, 100, 1000)
	if err == nil {
		t.Error("Should reject too many qubits")
	}

	// Test invalid compatibility - too high T-count (estimated depth)
	err = ValidateBackendCompatibility(backend, 16, 1000000)
	if err == nil {
		t.Error("Should reject circuits that are too deep")
	}
}

func TestCircuitAnalysis(t *testing.T) {
	backend := NewSimulatorBackend()

	qasm := `qreg q[8];
creg c[8];
H q[0];
H q[1];
T q[0];
T q[1];
T q[2];
CX q[0],q[1];
CX q[1],q[2];
measure q -> c;`

	// Execute to trigger circuit analysis
	seed := []byte("analysis_test_seed_123456789012")
	_, err := backend.Execute(qasm, seed)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	stats := backend.GetStats()
	circuitStats := stats.CircuitStats

	if circuitStats.AverageTGates != 3.0 {
		t.Errorf("Expected 3 T-gates, got %f", circuitStats.AverageTGates)
	}

	if circuitStats.MaxTGatesSeen != 3 {
		t.Errorf("Expected max T-gates 3, got %d", circuitStats.MaxTGatesSeen)
	}

	// Execute another circuit to test averaging
	qasm2 := `qreg q[4];
creg c[4];
T q[0];
measure q -> c;`

	_, err = backend.Execute(qasm2, seed)
	if err != nil {
		t.Fatalf("Second execute failed: %v", err)
	}

	stats2 := backend.GetStats()
	circuitStats2 := stats2.CircuitStats

	// Average should be (3 + 1) / 2 = 2.0
	if circuitStats2.AverageTGates != 2.0 {
		t.Errorf("Expected average T-gates 2.0, got %f", circuitStats2.AverageTGates)
	}
}

func TestExecutionStats(t *testing.T) {
	backend := NewSimulatorBackend()

	qasm := `qreg q[2];
creg c[2];
H q[0];
measure q -> c;`

	seed := []byte("stats_test_seed_1234567890123456")

	// Execute multiple times
	for i := 0; i < 5; i++ {
		_, err := backend.Execute(qasm, seed)
		if err != nil {
			t.Fatalf("Execute %d failed: %v", i, err)
		}
	}

	stats := backend.GetStats()

	if stats.TotalExecutions != 5 {
		t.Errorf("Expected 5 total executions, got %d", stats.TotalExecutions)
	}

	if stats.SuccessfulRuns != 5 {
		t.Errorf("Expected 5 successful runs, got %d", stats.SuccessfulRuns)
	}

	if stats.ErrorRate != 0.0 {
		t.Errorf("Expected 0%% error rate, got %f", stats.ErrorRate)
	}

	if stats.AverageTimeMS < 0 {
		t.Errorf("Expected non-negative average time, got %f", stats.AverageTimeMS)
	}

	if stats.LastExecutionTime.IsZero() {
		t.Error("Last execution time should be set")
	}
}

func TestQiskitAerBackend(t *testing.T) {
	backend := NewQiskitAerBackend("statevector")

	if backend.GetName() != "QiskitAer_statevector" {
		t.Errorf("Expected name 'QiskitAer_statevector', got '%s'", backend.GetName())
	}

	if backend.IsAvailable() {
		t.Error("Qiskit Aer backend should not be available (not implemented)")
	}

	caps := backend.GetCapabilities()
	if caps.MaxQubits != 20 {
		t.Errorf("Expected MaxQubits 20 for statevector, got %d", caps.MaxQubits)
	}

	// Test tensor backend
	tensorBackend := NewQiskitAerBackend("tensor")
	tensorCaps := tensorBackend.GetCapabilities()
	if tensorCaps.MaxQubits != 32 {
		t.Errorf("Expected MaxQubits 32 for tensor, got %d", tensorCaps.MaxQubits)
	}

	// Test execution (should fail since not implemented)
	_, err := backend.Execute("H q[0];", []byte("test"))
	if err == nil {
		t.Error("Qiskit Aer execution should fail (not implemented)")
	}
}

func TestIBMQuantumBackend(t *testing.T) {
	eagle := NewIBMQuantumBackend("eagle")
	heron := NewIBMQuantumBackend("heron")

	// Test Eagle backend
	if eagle.GetName() != "IBM_eagle" {
		t.Errorf("Expected name 'IBM_eagle', got '%s'", eagle.GetName())
	}

	if eagle.IsAvailable() {
		t.Error("IBM Quantum backend should not be available (not implemented)")
	}

	eagleCaps := eagle.GetCapabilities()
	if eagleCaps.MaxQubits != 127 {
		t.Errorf("Expected MaxQubits 127 for eagle, got %d", eagleCaps.MaxQubits)
	}

	if eagleCaps.IsSimulator {
		t.Error("IBM hardware should not be marked as simulator")
	}

	// Test Heron backend
	heronCaps := heron.GetCapabilities()
	if heronCaps.MaxQubits != 133 {
		t.Errorf("Expected MaxQubits 133 for heron, got %d", heronCaps.MaxQubits)
	}

	if heronCaps.ErrorRate >= eagleCaps.ErrorRate {
		t.Error("Heron should have better error rate than Eagle")
	}

	// Test execution (should fail since not implemented)
	_, err := eagle.Execute("H q[0];", []byte("test"))
	if err == nil {
		t.Error("IBM Quantum execution should fail (not implemented)")
	}
}

func TestCircuitValidation(t *testing.T) {
	backend := NewSimulatorBackend()

	// Test circuit that exceeds qubit limit
	largeQasm := `qreg q[50];
creg c[50];
H q[0];
measure q -> c;`

	seed := []byte("validation_test_seed_123456789")
	_, err := backend.Execute(largeQasm, seed)
	if err == nil {
		t.Error("Should reject circuit with too many qubits")
	}

	if !strings.Contains(err.Error(), "qubits") {
		t.Errorf("Error should mention qubits, got: %v", err)
	}
}

func TestBackendManagerStats(t *testing.T) {
	manager := NewBackendManager()

	// Get initial stats
	stats := manager.GetManagerStats()
	if stats.TotalBackends == 0 {
		t.Error("Should have some backends")
	}

	initialSwitches := stats.BackendSwitches

	// Switch backends
	simulator := manager.GetBackendByName("QuantumSimulator_v1.0")
	if simulator != nil {
		manager.SetDefaultBackend(simulator)
	}

	// Check stats updated
	newStats := manager.GetManagerStats()
	if newStats.BackendSwitches != initialSwitches+1 {
		t.Error("Backend switch count should increment")
	}

	if newStats.LastSwitchTime.IsZero() {
		t.Error("Last switch time should be set")
	}
}

func TestHelperFunctions(t *testing.T) {
	// Test countLines
	text := "line1\nline2\nline3"
	if countLines(text) != 3 {
		t.Errorf("Expected 3 lines, got %d", countLines(text))
	}

	// Test countSingleQubitGates
	qasm := "H q[0];\nT q[1];\nS q[2];\nX q[3];"
	if countSingleQubitGates(qasm) != 4 {
		t.Errorf("Expected 4 single-qubit gates, got %d", countSingleQubitGates(qasm))
	}

	// Test countTwoQubitGates
	qasm2 := "CX q[0],q[1];\nCX q[1],q[2];"
	if countTwoQubitGates(qasm2) != 2 {
		t.Errorf("Expected 2 two-qubit gates, got %d", countTwoQubitGates(qasm2))
	}

	// Test parseInt
	if parseInt("42") != 42 {
		t.Errorf("Expected 42, got %d", parseInt("42"))
	}
}

func TestBackendIntegrationWithPuzzleOrchestrator(t *testing.T) {
	// Test that backend integrates properly with puzzle orchestrator
	backend := NewSimulatorBackend()

	// Configure for debug mode
	config := BackendConfig{
		NoiseModel:   CreateNoiseModel("ideal"),
		Shots:        1024,
		Optimization: 0,
		Timeout:      5000,
		DebugMode:    false, // Keep false for cleaner test output
	}

	err := backend.Configure(config)
	if err != nil {
		t.Fatalf("Backend configuration failed: %v", err)
	}

	// Create a simple QASM circuit similar to what puzzle orchestrator would generate
	qasm := `qreg q[16];
creg c[16];
// Template-based quantum circuit
H q[0];
T q[0];
T q[1];
T q[2];
CX q[0],q[1];
CX q[1],q[2];
// Pauli-Z mask would be applied here
Z q[3];
Z q[5];
measure q -> c;`

	seed := []byte("integration_test_seed_1234567890")

	// Execute multiple times to ensure consistency
	outcomes := make([]uint16, 5)
	for i := 0; i < 5; i++ {
		outcome, err := backend.Execute(qasm, seed)
		if err != nil {
			t.Fatalf("Integration execute %d failed: %v", i, err)
		}
		outcomes[i] = outcome
	}

	// All outcomes should be identical (deterministic)
	for i := 1; i < len(outcomes); i++ {
		if outcomes[i] != outcomes[0] {
			t.Errorf("Expected deterministic outcomes, but got %d != %d", outcomes[i], outcomes[0])
		}
	}

	// Check final stats
	stats := backend.GetStats()
	if stats.TotalExecutions < 5 {
		t.Errorf("Expected at least 5 executions, got %d", stats.TotalExecutions)
	}

	if stats.ErrorRate > 0 {
		t.Errorf("Expected 0%% error rate for ideal backend, got %f", stats.ErrorRate)
	}

	t.Logf("✅ Backend integration test passed - %d executions, outcome: 0x%04x",
		stats.TotalExecutions, outcomes[0])
}

// Benchmark tests
func BenchmarkSimulatorBackend_Execute(b *testing.B) {
	backend := NewSimulatorBackend()

	qasm := `qreg q[16];
creg c[16];
H q[0];
T q[1];
T q[2];
CX q[0],q[1];
measure q -> c;`

	seed := []byte("benchmark_seed_1234567890123456")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := backend.Execute(qasm, seed)
		if err != nil {
			b.Fatalf("Benchmark execute failed: %v", err)
		}
	}
}

func BenchmarkSimulatorBackend_ExecuteWithNoise(b *testing.B) {
	backend := NewSimulatorBackend()

	// Configure with light noise
	config := BackendConfig{
		NoiseModel:   CreateNoiseModel("light"),
		Shots:        1024,
		Optimization: 0,
		Timeout:      5000,
		DebugMode:    false,
	}

	backend.Configure(config)

	qasm := `qreg q[16];
creg c[16];
H q[0];
T q[1];
T q[2];
T q[3];
CX q[0],q[1];
CX q[1],q[2];
measure q -> c;`

	seed := []byte("benchmark_noise_seed_123456789")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := backend.Execute(qasm, seed)
		if err != nil {
			b.Fatalf("Benchmark noisy execute failed: %v", err)
		}
	}
}
