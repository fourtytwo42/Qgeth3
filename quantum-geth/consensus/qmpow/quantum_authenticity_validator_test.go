package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"math/big"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
)

// TestQuantumAuthenticityValidator_ValidateQuantumAuthenticity tests the main validation function
func TestQuantumAuthenticityValidator_ValidateQuantumAuthenticity(t *testing.T) {
	validator := NewQuantumAuthenticityValidator()
	
	tests := []struct {
		name          string
		header        *types.Header
		expectValid   bool
		expectError   string
		description   string
	}{
		{
			name:          "Valid quantum computation",
			header:        createValidQuantumHeader(t),
			expectValid:   true,
			expectError:   "",
			description:   "Genuine quantum computation should pass all validation checks",
		},
		{
			name:          "Insufficient qubits",
			header:        createHeaderWithQBits(t, 8), // Below minimum of 16
			expectValid:   false,
			expectError:   "insufficient qubits",
			description:   "Classical computers can simulate small quantum circuits",
		},
		{
			name:          "Insufficient T-gates",
			header:        createHeaderWithTCount(t, 5), // Below minimum of 20
			expectValid:   false,
			expectError:   "insufficient T-gates",
			description:   "Non-universal quantum computation can be simulated classically",
		},
		{
			name:          "Low entanglement depth",
			header:        createHeaderWithLNet(t, 50), // Below minimum of 128
			expectValid:   false,
			expectError:   "insufficient entanglement depth",
			description:   "Separable quantum states can be simulated classically",
		},
		{
			name:          "Classical simulation attempt",
			header:        createClassicalSimulationHeader(t),
			expectValid:   false,
			expectError:   "classical computation attempt detected",
			description:   "Classical computers attempting to fake quantum computation",
		},
		{
			name:          "Low complexity circuit",
			header:        createLowComplexityHeader(t),
			expectValid:   false,
			expectError:   "quantum circuit complexity too low",
			description:   "Quantum circuits that can be efficiently simulated classically",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid, err := validator.ValidateQuantumAuthenticity(tt.header)
			
			if tt.expectValid {
				if !valid {
					t.Errorf("Expected validation to pass but got: valid=%v, err=%v", valid, err)
				}
				if err != nil {
					t.Errorf("Expected no error but got: %v", err)
				}
			} else {
				if valid {
					t.Errorf("Expected validation to fail but got: valid=%v", valid)
				}
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				if tt.expectError != "" && err != nil {
					if !contains(err.Error(), tt.expectError) {
						t.Errorf("Expected error containing '%s' but got: %v", tt.expectError, err)
					}
				}
			}
			
			t.Logf("Test case: %s - %s", tt.name, tt.description)
			t.Logf("Result: valid=%v, error=%v", valid, err)
		})
	}
}

// TestQuantumAuthenticityValidator_ComplexityValidation tests quantum complexity validation
func TestQuantumAuthenticityValidator_ComplexityValidation(t *testing.T) {
	validator := NewQuantumAuthenticityValidator()

	tests := []struct {
		name        string
		qbits       uint8
		tcount      uint16
		lnet        uint16
		expectValid bool
		description string
	}{
		{
			name:        "Quantum advantage parameters",
			qbits:       16,
			tcount:      20,
			lnet:        128,
			expectValid: true,
			description: "Parameters that provide genuine quantum advantage",
		},
		{
			name:        "Classically simulable",
			qbits:       8,
			tcount:      10,
			lnet:        50,
			expectValid: false,
			description: "Parameters that can be efficiently simulated classically",
		},
		{
			name:        "High quantum complexity",
			qbits:       20,
			tcount:      50,
			lnet:        256,
			expectValid: true,
			description: "High complexity quantum computation beyond classical reach",
		},
		{
			name:        "Edge case minimum",
			qbits:       16,
			tcount:      20,
			lnet:        128,
			expectValid: true,
			description: "Minimum parameters for quantum advantage",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := createHeaderWithParams(t, tt.qbits, tt.tcount, tt.lnet)
			valid, err := validator.validateQuantumComplexity(header)
			
			if tt.expectValid && (!valid || err != nil) {
				t.Errorf("Expected complexity validation to pass but got: valid=%v, err=%v", valid, err)
			}
			if !tt.expectValid && (valid || err == nil) {
				t.Errorf("Expected complexity validation to fail but got: valid=%v, err=%v", valid, err)
			}
			
			complexity := validator.calculateComplexityScore(header)
			t.Logf("Complexity score: %d for qbits=%d, tcount=%d, lnet=%d", 
				complexity, tt.qbits, tt.tcount, tt.lnet)
		})
	}
}

// TestQuantumAuthenticityValidator_EntanglementValidation tests entanglement authenticity validation
func TestQuantumAuthenticityValidator_EntanglementValidation(t *testing.T) {
	validator := NewQuantumAuthenticityValidator()

	tests := []struct {
		name         string
		headerData   []byte
		expectValid  bool
		description  string
	}{
		{
			name:         "High entanglement entropy",
			headerData:   createHighEntropyData(t),
			expectValid:  true,
			description:  "Quantum states with high entanglement entropy",
		},
		{
			name:         "Low entanglement entropy",
			headerData:   createLowEntropyData(t),
			expectValid:  false,
			description:  "Separable states with low entanglement entropy",
		},
		{
			name:         "Classical separable state",
			headerData:   createSeparableStateData(t),
			expectValid:  false,
			description:  "Classical systems produce separable quantum states",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := createHeaderWithCustomData(t, tt.headerData)
			valid, err := validator.validateEntanglementAuthenticity(header)
			
			if tt.expectValid && (!valid || err != nil) {
				t.Errorf("Expected entanglement validation to pass but got: valid=%v, err=%v", valid, err)
			}
			if !tt.expectValid && (valid || err == nil) {
				t.Errorf("Expected entanglement validation to fail but got: valid=%v, err=%v", valid, err)
			}
			
			entanglementData := validator.extractEntanglementData(header)
			entropy := validator.calculateEntanglementEntropy(entanglementData)
			t.Logf("Entanglement entropy: %.2f", entropy)
		})
	}
}

// TestQuantumAuthenticityValidator_BellCorrelationValidation tests Bell inequality validation
func TestQuantumAuthenticityValidator_BellCorrelationValidation(t *testing.T) {
	validator := NewQuantumAuthenticityValidator()

	tests := []struct {
		name           string
		correlationSeed uint64
		expectValid    bool
		description    string
	}{
		{
			name:           "Quantum Bell violation",
			correlationSeed: 2400, // Will produce Bell parameter > 2.0
			expectValid:    true,
			description:    "Quantum correlations violate Bell inequality",
		},
		{
			name:           "Classical hidden variables",
			correlationSeed: 500, // Will produce Bell parameter ≤ 2.0
			expectValid:    false,
			description:    "Classical systems obey Bell inequality",
		},
		{
			name:           "Maximum quantum violation",
			correlationSeed: 2828, // Will produce Bell parameter ≈ 2√2
			expectValid:    true,
			description:    "Maximum quantum Bell inequality violation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := createHeaderWithBellData(t, tt.correlationSeed)
			valid, err := validator.validateBellCorrelations(header)
			
			if tt.expectValid && (!valid || err != nil) {
				t.Errorf("Expected Bell validation to pass but got: valid=%v, err=%v", valid, err)
			}
			if !tt.expectValid && (valid || err == nil) {
				t.Errorf("Expected Bell validation to fail but got: valid=%v, err=%v", valid, err)
			}
			
			correlationData := validator.extractBellCorrelationData(header)
			bellParameter := validator.calculateBellParameter(correlationData)
			t.Logf("Bell parameter: %.3f (classical bound: 2.0, quantum bound: %.3f)", 
				bellParameter, 2.0, 2.0*1.414)
		})
	}
}

// TestQuantumAuthenticityValidator_ClassicalDetection tests classical computation detection
func TestQuantumAuthenticityValidator_ClassicalDetection(t *testing.T) {
	validator := NewQuantumAuthenticityValidator()

	tests := []struct {
		name         string
		headerType   string
		expectValid  bool
		description  string
	}{
		{
			name:         "Genuine quantum computation",
			headerType:   "quantum",
			expectValid:  true,
			description:  "Real quantum computation with genuine randomness",
		},
		{
			name:         "PRNG-based simulation",
			headerType:   "prng",
			expectValid:  false,
			description:  "Classical simulation using pseudorandom number generators",
		},
		{
			name:         "Deterministic patterns",
			headerType:   "deterministic",
			expectValid:  false,
			description:  "Deterministic computation masquerading as quantum",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := createHeaderByType(t, tt.headerType)
			isClassical, err := validator.detectClassicalComputation(header)
			
			// Note: detectClassicalComputation returns true for classical, false for quantum
			expectedClassical := !tt.expectValid
			
			if isClassical != expectedClassical {
				t.Errorf("Expected classical detection=%v but got: %v, err=%v", 
					expectedClassical, isClassical, err)
			}
			
			t.Logf("Classical detection result: %v for %s", isClassical, tt.headerType)
		})
	}
}

// TestQuantumAuthenticityValidator_Statistics tests validation statistics tracking
func TestQuantumAuthenticityValidator_Statistics(t *testing.T) {
	validator := NewQuantumAuthenticityValidator()
	
	// Reset statistics
	validator.ResetQuantumAuthenticityStats()
	
	// Test valid quantum computation
	validHeader := createValidQuantumHeader(t)
	valid, err := validator.ValidateQuantumAuthenticity(validHeader)
	if !valid || err != nil {
		t.Errorf("Valid header failed validation: valid=%v, err=%v", valid, err)
	}
	
	// Test classical simulation
	classicalHeader := createClassicalSimulationHeader(t)
	valid, err = validator.ValidateQuantumAuthenticity(classicalHeader)
	if valid || err == nil {
		t.Errorf("Classical header passed validation: valid=%v, err=%v", valid, err)
	}
	
	// Check statistics
	stats := validator.GetQuantumAuthenticityStats()
	if stats.TotalValidations != 2 {
		t.Errorf("Expected 2 total validations, got: %d", stats.TotalValidations)
	}
	if stats.AuthenticQuantum != 1 {
		t.Errorf("Expected 1 authentic quantum validation, got: %d", stats.AuthenticQuantum)
	}
	if stats.ClassicalDetected != 1 {
		t.Errorf("Expected 1 classical detection, got: %d", stats.ClassicalDetected)
	}
	
	t.Logf("Validation statistics: %+v", stats)
}

// Helper functions for creating test headers

func createValidQuantumHeader(t *testing.T) *types.Header {
	return createHeaderWithParams(t, 16, 20, 128)
}

func createHeaderWithQBits(t *testing.T, qbits uint8) *types.Header {
	return createHeaderWithParams(t, qbits, 20, 128)
}

func createHeaderWithTCount(t *testing.T, tcount uint16) *types.Header {
	return createHeaderWithParams(t, 16, tcount, 128)
}

func createHeaderWithLNet(t *testing.T, lnet uint16) *types.Header {
	return createHeaderWithParams(t, 16, 20, lnet)
}

func createHeaderWithParams(t *testing.T, qbits uint8, tcount uint16, lnet uint16) *types.Header {
	header := &types.Header{
		Number:     big.NewInt(100),
		Time:       uint64(time.Now().Unix()),
		Difficulty: big.NewInt(1000),
	}
	
	// Set quantum parameters
	header.QBits = &qbits
	header.TCount = &tcount
	header.LNet = &lnet
	
	// Set quantum fields with valid-looking data
	header.OutcomeRoot = &common.Hash{}
	copy(header.OutcomeRoot[:], sha256.Sum256([]byte("outcome"))[:])
	
	header.GateHash = &common.Hash{}
	copy(header.GateHash[:], sha256.Sum256([]byte("gates"))[:])
	
	header.ProofRoot = &common.Hash{}
	copy(header.ProofRoot[:], sha256.Sum256([]byte("proof"))[:])
	
	// Set other required fields
	header.BranchNibbles = make([]byte, 32)
	for i := range header.BranchNibbles {
		header.BranchNibbles[i] = byte(i)
	}
	
	header.ExtraNonce32 = make([]byte, 32)
	for i := range header.ExtraNonce32 {
		header.ExtraNonce32[i] = byte(i + 100)
	}
	
	return header
}

func createClassicalSimulationHeader(t *testing.T) *types.Header {
	header := createValidQuantumHeader(t)
	
	// Create patterns that indicate classical simulation
	// Use predictable PRNG-like patterns in quantum fields
	
	// Linear congruential generator pattern
	seed := uint32(12345)
	lcg := func() uint32 {
		seed = (seed*1103515245 + 12345) % 0x100000000
		return seed
	}
	
	// Fill fields with PRNG patterns that will be detected
	for i := 0; i < len(header.ExtraNonce32); i += 4 {
		value := lcg()
		binary.LittleEndian.PutUint32(header.ExtraNonce32[i:i+4], value)
	}
	
	return header
}

func createLowComplexityHeader(t *testing.T) *types.Header {
	// Create header with parameters that result in low complexity score
	return createHeaderWithParams(t, 8, 5, 10) // Very low complexity
}

func createHeaderWithCustomData(t *testing.T, data []byte) *types.Header {
	header := createValidQuantumHeader(t)
	
	// Embed custom data in quantum fields for testing
	if len(data) >= 32 {
		copy(header.OutcomeRoot[:], data[:32])
	}
	if len(data) >= 64 {
		copy(header.GateHash[:], data[32:64])
	}
	
	return header
}

func createHighEntropyData(t *testing.T) []byte {
	// Create data that will result in high entanglement entropy
	data := make([]byte, 64)
	entropy := uint64(15000) // High entropy value
	binary.LittleEndian.PutUint64(data[:8], entropy)
	
	// Fill rest with varied data
	for i := 8; i < len(data); i++ {
		data[i] = byte(i * 37) // Varied pattern
	}
	
	return data
}

func createLowEntropyData(t *testing.T) []byte {
	// Create data that will result in low entanglement entropy
	data := make([]byte, 64)
	entropy := uint64(1000) // Low entropy value
	binary.LittleEndian.PutUint64(data[:8], entropy)
	
	return data
}

func createSeparableStateData(t *testing.T) []byte {
	// Create data indicating separable quantum state
	data := make([]byte, 64)
	entropy := uint64(500) // Very low entropy indicating separability
	binary.LittleEndian.PutUint64(data[:8], entropy)
	
	return data
}

func createHeaderWithBellData(t *testing.T, correlationSeed uint64) *types.Header {
	header := createValidQuantumHeader(t)
	
	// Embed correlation data that will produce specific Bell parameter
	correlationData := make([]byte, 32)
	binary.LittleEndian.PutUint64(correlationData[:8], correlationSeed)
	copy(header.ExtraNonce32, correlationData)
	
	return header
}

func createHeaderByType(t *testing.T, headerType string) *types.Header {
	header := createValidQuantumHeader(t)
	
	switch headerType {
	case "quantum":
		// Already created as valid quantum header
		return header
		
	case "prng":
		// Create PRNG patterns
		seed := uint32(12345)
		for i := 0; i < len(header.ExtraNonce32); i += 4 {
			seed = (seed*1103515245 + 12345) % 0x100000000
			binary.LittleEndian.PutUint32(header.ExtraNonce32[i:i+4], seed)
		}
		
	case "deterministic":
		// Create deterministic patterns
		pattern := []byte{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22}
		for i := 0; i < len(header.ExtraNonce32); i++ {
			header.ExtraNonce32[i] = pattern[i%len(pattern)]
		}
	}
	
	return header
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || 
		    (len(s) > len(substr) && 
		     (s[:len(substr)] == substr || 
		      s[len(s)-len(substr):] == substr || 
		      containsInMiddle(s, substr))))
}

func containsInMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
} 