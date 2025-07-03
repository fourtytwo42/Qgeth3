// quantum_simulation_validator_test.go
// Tests for Phase 3.2 Quantum Simulation Validation System

package qmpow

import (
	"context"
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/core/types"
)

func TestNewQuantumSimulationValidator(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	
	if validator == nil {
		t.Fatal("NewQuantumSimulationValidator returned nil")
	}
	
	if validator.config == nil {
		t.Fatal("Configuration not initialized")
	}
	
	if validator.stats == nil {
		t.Fatal("Statistics not initialized")
	}
	
	// Check configuration defaults
	if validator.config.MinQubits != 16 {
		t.Errorf("Expected MinQubits=16, got %d", validator.config.MinQubits)
	}
	
	if validator.config.MinTGates != 20 {
		t.Errorf("Expected MinTGates=20, got %d", validator.config.MinTGates)
	}
	
	if validator.config.MinEntanglementDepth != 128 {
		t.Errorf("Expected MinEntanglementDepth=128, got %d", validator.config.MinEntanglementDepth)
	}
	
	t.Log("✅ QuantumSimulationValidator initialized correctly")
}

func TestExtractQuantumMiningData(t *testing.T) {
	// Create test header with quantum parameters
	qbits := uint16(16)
	tcount := uint32(20)
	lnet := uint16(128)
	
	header := &types.Header{
		QBits:  &qbits,
		TCount: &tcount,
		LNet:   &lnet,
	}
	
	data, err := ExtractQuantumMiningData(header)
	if err != nil {
		t.Fatalf("ExtractQuantumMiningData failed: %v", err)
	}
	
	if data.QBits != 16 {
		t.Errorf("Expected QBits=16, got %d", data.QBits)
	}
	
	if data.TCount != 20 {
		t.Errorf("Expected TCount=20, got %d", data.TCount)
	}
	
	if data.LNet != 128 {
		t.Errorf("Expected LNet=128, got %d", data.LNet)
	}
	
	t.Log("✅ Quantum mining data extraction working correctly")
}

func TestExtractQuantumMiningDataMissingParams(t *testing.T) {
	// Test with missing parameters
	header := &types.Header{}
	
	_, err := ExtractQuantumMiningData(header)
	if err == nil {
		t.Fatal("Expected error for missing quantum parameters")
	}
	
	t.Log("✅ Missing parameter validation working correctly")
}

func TestValidateCircuitComplexity(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	ctx := context.Background()
	
	// Test with valid parameters
	qdata := &QuantumMiningData{
		QBits:  16,
		TCount: 20,
		LNet:   128,
	}
	
	result, err := validator.validateCircuitComplexity(ctx, qdata)
	if err != nil {
		t.Fatalf("Circuit complexity validation failed: %v", err)
	}
	
	if !result.Valid {
		t.Error("Circuit complexity validation should pass for valid parameters")
	}
	
	if result.QubitsCount != 16 {
		t.Errorf("Expected QubitsCount=16, got %d", result.QubitsCount)
	}
	
	if result.TGatesCount != 20 {
		t.Errorf("Expected TGatesCount=20, got %d", result.TGatesCount)
	}
	
	t.Log("✅ Circuit complexity validation working correctly")
}

func TestValidateCircuitComplexityInsufficientParams(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	ctx := context.Background()
	
	// Test with insufficient parameters
	qdata := &QuantumMiningData{
		QBits:  10, // Below minimum of 16
		TCount: 15, // Below minimum of 20
		LNet:   64, // Below minimum of 128
	}
	
	_, err := validator.validateCircuitComplexity(ctx, qdata)
	if err == nil {
		t.Fatal("Expected error for insufficient circuit complexity")
	}
	
	t.Log("✅ Circuit complexity validation correctly rejects insufficient parameters")
}

func TestValidateQuantumStateProperties(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	ctx := context.Background()
	
	// Test with valid parameters
	qdata := &QuantumMiningData{
		QBits:  16,
		TCount: 20,
		LNet:   128,
	}
	
	result, err := validator.validateQuantumStateProperties(ctx, qdata)
	if err != nil {
		t.Fatalf("State properties validation failed: %v", err)
	}
	
	if !result.Valid {
		t.Error("State properties validation should pass for valid parameters")
	}
	
	if !result.HasSuperposition {
		t.Error("Expected superposition to be detected")
	}
	
	if !result.HasEntanglement {
		t.Error("Expected entanglement to be detected")
	}
	
	if result.CoherenceTime <= 0 {
		t.Error("Expected positive coherence time")
	}
	
	if result.StateComplexity <= 0 {
		t.Error("Expected positive state complexity")
	}
	
	if result.QuantumVolume <= 0 {
		t.Error("Expected positive quantum volume")
	}
	
	t.Logf("✅ State properties validation working correctly")
	t.Logf("   - Superposition: %v", result.HasSuperposition)
	t.Logf("   - Entanglement: %v", result.HasEntanglement)
	t.Logf("   - Coherence time: %.2fms", result.CoherenceTime)
	t.Logf("   - State complexity: %.2f", result.StateComplexity)
	t.Logf("   - Quantum volume: %.2f", result.QuantumVolume)
}

func TestValidateInterferencePatterns(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	ctx := context.Background()
	
	// Test with valid parameters
	qdata := &QuantumMiningData{
		QBits:  16,
		TCount: 20,
		LNet:   128,
	}
	
	result, err := validator.validateInterferencePatterns(ctx, qdata)
	if err != nil {
		t.Fatalf("Interference pattern validation failed: %v", err)
	}
	
	if !result.Valid {
		t.Error("Interference pattern validation should pass for valid parameters")
	}
	
	if result.Visibility < 0 || result.Visibility > 1 {
		t.Errorf("Visibility should be between 0 and 1, got %.3f", result.Visibility)
	}
	
	if result.PhaseCoherence < 0 {
		t.Errorf("Phase coherence should be non-negative, got %.6f", result.PhaseCoherence)
	}
	
	if result.PatternScore < 0 || result.PatternScore > 1 {
		t.Errorf("Pattern score should be between 0 and 1, got %.3f", result.PatternScore)
	}
	
	if !result.IsQuantum {
		t.Error("Expected quantum interference to be detected")
	}
	
	t.Logf("✅ Interference pattern validation working correctly")
	t.Logf("   - Visibility: %.3f", result.Visibility)
	t.Logf("   - Phase coherence: %.6f", result.PhaseCoherence)
	t.Logf("   - Pattern score: %.3f", result.PatternScore)
	t.Logf("   - Is quantum: %v", result.IsQuantum)
}

func TestValidateEntanglementProperties(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	ctx := context.Background()
	
	// Test with valid parameters
	qdata := &QuantumMiningData{
		QBits:  16,
		TCount: 20,
		LNet:   128,
	}
	
	result, err := validator.validateEntanglementProperties(ctx, qdata)
	if err != nil {
		t.Fatalf("Entanglement properties validation failed: %v", err)
	}
	
	if !result.Valid {
		t.Error("Entanglement properties validation should pass for valid parameters")
	}
	
	if result.EntanglementEntropy < 0 {
		t.Errorf("Entanglement entropy should be non-negative, got %.6f", result.EntanglementEntropy)
	}
	
	if result.WitnessValue < 0 || result.WitnessValue > 1 {
		t.Errorf("Witness value should be between 0 and 1, got %.6f", result.WitnessValue)
	}
	
	if result.BellParameter < 0 {
		t.Errorf("Bell parameter should be non-negative, got %.6f", result.BellParameter)
	}
	
	if !result.IsGenuine {
		t.Error("Expected genuine entanglement to be detected")
	}
	
	t.Logf("✅ Entanglement properties validation working correctly")
	t.Logf("   - Entanglement entropy: %.6f", result.EntanglementEntropy)
	t.Logf("   - Witness value: %.6f", result.WitnessValue)
	t.Logf("   - Bell parameter: %.6f", result.BellParameter)
	t.Logf("   - Is genuine: %v", result.IsGenuine)
}

func TestQuantumSimulationValidation(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	
	// Create test header with quantum parameters
	qbits := uint16(16)
	tcount := uint32(20)
	lnet := uint16(128)
	
	header := &types.Header{
		Number: big.NewInt(1000),
		QBits:  &qbits,
		TCount: &tcount,
		LNet:   &lnet,
	}
	
	qdata, err := ExtractQuantumMiningData(header)
	if err != nil {
		t.Fatalf("Failed to extract quantum mining data: %v", err)
	}
	
	result, err := validator.ValidateQuantumSimulation(header, qdata)
	if err != nil {
		t.Fatalf("Quantum simulation validation failed: %v", err)
	}
	
	if !result.Valid {
		t.Errorf("Quantum simulation validation should pass for valid parameters: %s", result.ErrorMessage)
	}
	
	if result.ValidationTime <= 0 {
		t.Error("Expected positive validation time")
	}
	
	if result.Confidence <= 0 {
		t.Error("Expected positive confidence score")
	}
	
	t.Logf("✅ Complete quantum simulation validation working correctly")
	t.Logf("   - Valid: %v", result.Valid)
	t.Logf("   - Confidence: %.2f", result.Confidence)
	t.Logf("   - Validation time: %v", result.ValidationTime)
}

func TestGetValidationStats(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	
	stats := validator.GetValidationStats()
	if stats == nil {
		t.Fatal("GetValidationStats returned nil")
	}
	
	if stats.TotalValidations != 0 {
		t.Error("Expected zero total validations for new validator")
	}
	
	t.Log("✅ Validation statistics working correctly")
}

func TestDebugQuantumStateGeneration(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	
	// Test with valid parameters
	qdata := &QuantumMiningData{
		QBits:  16,
		TCount: 20,
		LNet:   128,
	}
	
	// Test state reconstruction
	state, err := validator.reconstructQuantumState(qdata)
	if err != nil {
		t.Fatalf("State reconstruction failed: %v", err)
	}
	
	t.Logf("State size: %d", len(state))
	
	// Analyze state properties
	nonZeroCount := 0
	maxAmp := 0.0
	totalProb := 0.0
	sumProbSquared := 0.0 // For correct effective states calculation
	
	for i, amp := range state {
		absAmp := real(amp)*real(amp) + imag(amp)*imag(amp)
		if absAmp > 1e-8 {
			nonZeroCount++
			if absAmp > maxAmp {
				maxAmp = absAmp
			}
			sumProbSquared += absAmp * absAmp // Sum of probability squares
		}
		totalProb += absAmp
		
		// Log first few amplitudes
		if i < 10 {
			t.Logf("Amplitude[%d]: %v (|amp|²=%.6f)", i, amp, absAmp)
		}
	}
	
	t.Logf("Non-zero amplitudes: %d", nonZeroCount)
	t.Logf("Max amplitude probability: %.6f", maxAmp)
	t.Logf("Total probability: %.6f", totalProb)
	t.Logf("Dominance: %.6f", maxAmp)
	t.Logf("Sum of probability squares: %.6f", sumProbSquared)
	t.Logf("Effective states (correct): %.6f", 1.0/sumProbSquared)
	
	// Test superposition validation
	hasSuperposition, err := validator.validateSuperposition(state)
	if err != nil {
		t.Fatalf("Superposition validation error: %v", err)
	}
	
	t.Logf("Has superposition: %v", hasSuperposition)
}

func TestDebugInterferencePatterns(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	
	// Test with smaller system for debugging
	qdata := &QuantumMiningData{
		QBits:  4, // Only 4 qubits = 16 states (much faster)
		TCount: 20,
		LNet:   128,
	}
	
	// Reconstruct state for analysis
	state, err := validator.reconstructQuantumState(qdata)
	if err != nil {
		t.Fatalf("State reconstruction failed: %v", err)
	}
	
	t.Logf("State size: %d", len(state))
	
	// Test visibility calculation
	visibility, err := validator.calculateInterferenceVisibility(state, qdata)
	if err != nil {
		t.Fatalf("Visibility calculation failed: %v", err)
	}
	
	// Test phase coherence
	phaseCoherence, err := validator.analyzePhaseCoherence(state, qdata)
	if err != nil {
		t.Fatalf("Phase coherence analysis failed: %v", err)
	}
	
	// Test pattern score
	patternScore := validator.calculateInterferencePatternScore(state, visibility, phaseCoherence)
	
	// Test quantum detection
	isQuantum := validator.isQuantumInterference(visibility, phaseCoherence, patternScore)
	
	t.Logf("Visibility: %.6f", visibility)
	t.Logf("Phase coherence: %.6f", phaseCoherence)
	t.Logf("Pattern score: %.6f", patternScore)
	t.Logf("Is quantum: %v", isQuantum)
	
	// Show thresholds
	t.Logf("Min visibility threshold: 0.5")
	t.Logf("Min coherence threshold: 0.01")
	t.Logf("Min pattern score threshold: 0.4")
}

func TestDebugInterferencePatterns16Qubit(t *testing.T) {
	validator := NewQuantumSimulationValidator()
	
	// Test with 16-qubit system 
	qdata := &QuantumMiningData{
		QBits:  16,
		TCount: 20,
		LNet:   128,
	}
	
	// Reconstruct state for analysis
	state, err := validator.reconstructQuantumState(qdata)
	if err != nil {
		t.Fatalf("State reconstruction failed: %v", err)
	}
	
	t.Logf("State size: %d", len(state))
	
	// Test visibility calculation
	visibility, err := validator.calculateInterferenceVisibility(state, qdata)
	if err != nil {
		t.Fatalf("Visibility calculation failed: %v", err)
	}
	
	// Test phase coherence
	phaseCoherence, err := validator.analyzePhaseCoherence(state, qdata)
	if err != nil {
		t.Fatalf("Phase coherence analysis failed: %v", err)
	}
	
	// Test pattern score
	patternScore := validator.calculateInterferencePatternScore(state, visibility, phaseCoherence)
	
	// Calculate adaptive threshold (same logic as in isQuantumInterference)
	minCoherence := 0.01  // Base threshold
	if patternScore > 0.5 {
		minCoherence = 0.0001  // Much more lenient for medium systems
	}
	if patternScore > 0.6 {
		minCoherence = 0.00001  // Very lenient for large systems
	}
	if patternScore > 0.8 {
		minCoherence = 0.000001 // Extremely lenient for very large systems
	}
	
	// Test quantum detection
	isQuantum := validator.isQuantumInterference(visibility, phaseCoherence, patternScore)
	
	t.Logf("16-Qubit Results:")
	t.Logf("   Visibility: %.6f (threshold: 0.5)", visibility)
	t.Logf("   Phase coherence: %.6f (adaptive threshold: %.6f)", phaseCoherence, minCoherence)
	t.Logf("   Pattern score: %.6f (threshold: 0.4)", patternScore)
	t.Logf("   Is quantum: %v", isQuantum)
	
	// Individual threshold checks
	t.Logf("Threshold Analysis:")
	t.Logf("   Visibility >= 0.5: %v", visibility >= 0.5)
	t.Logf("   Phase coherence >= %.6f: %v", minCoherence, phaseCoherence >= minCoherence)
	t.Logf("   Pattern score >= 0.4: %v", patternScore >= 0.4)
} 