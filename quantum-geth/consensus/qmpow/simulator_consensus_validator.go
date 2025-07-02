// Package qmpow implements the Quantum Modified Proof-of-Work consensus algorithm
package qmpow

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/log"
)

// SimulatorConsensusValidator ensures quantum simulators maintain consensus
type SimulatorConsensusValidator struct {
	mu                    sync.RWMutex
	referenceTestSuite    *ReferenceQuantumTestSuite
	registeredSimulators  map[string]*SimulatorFingerprint
	consensusResults      *ConsensusResultsCache
	stats                *SimulatorConsensusStats
	config               *SimulatorConsensusConfig
}

// ReferenceQuantumTestSuite contains standard quantum computation test cases
type ReferenceQuantumTestSuite struct {
	TestCases             []*QuantumTestCase
	TestCasesByComplexity map[string][]*QuantumTestCase
	ValidationThresholds  *ValidationThresholds
	CreationTime          time.Time
}

// QuantumTestCase represents a single quantum computation test case
type QuantumTestCase struct {
	ID                   string                 `json:"id"`
	Name                 string                 `json:"name"`
	Description          string                 `json:"description"`
	ComplexityLevel      string                 `json:"complexity_level"` // "basic", "intermediate", "advanced", "expert"
	Parameters           *QuantumTestParameters `json:"parameters"`
	ExpectedOutcome      *QuantumTestOutcome    `json:"expected_outcome"`
	ToleranceThresholds  *ToleranceThresholds   `json:"tolerance_thresholds"`
	ConsensusRequired    bool                   `json:"consensus_required"`
	CreationTime         time.Time              `json:"creation_time"`
}

// QuantumTestParameters contains test case parameters
type QuantumTestParameters struct {
	Seed0        string `json:"seed0"`
	QBits        int    `json:"qbits"`
	TCount       int    `json:"tcount"`
	LNet         int    `json:"lnet"`
	MaxTime      int    `json:"max_time_ms"`
	Description  string `json:"description"`
}

// QuantumTestOutcome contains expected test results
type QuantumTestOutcome struct {
	OutcomesHex     string  `json:"outcomes_hex"`
	GateHashHex     string  `json:"gate_hash_hex"`
	ProofRootHex    string  `json:"proof_root_hex"`
	PuzzleCount     int     `json:"puzzle_count"`
	TotalTime       float64 `json:"total_time"`
	AvgTimePerPuzzle float64 `json:"avg_time_per_puzzle"`
	Backend         string  `json:"backend"`
	Deterministic   bool    `json:"deterministic"`
	ConsensusSafe   bool    `json:"consensus_safe"`
}

// ToleranceThresholds defines acceptable variance for non-deterministic measurements
type ToleranceThresholds struct {
	TimeTolerancePercent     float64 `json:"time_tolerance_percent"`
	OutcomeMatchRequired     bool    `json:"outcome_match_required"`
	GateHashMatchRequired    bool    `json:"gate_hash_match_required"`
	ProofRootMatchRequired   bool    `json:"proof_root_match_required"`
	DeterministicRequired    bool    `json:"deterministic_required"`
	ConsensusSafeRequired    bool    `json:"consensus_safe_required"`
}

// ValidationThresholds contains validation parameters
type ValidationThresholds struct {
	MinConsensusNodes        int     `json:"min_consensus_nodes"`
	ConsensusAgreementPercent float64 `json:"consensus_agreement_percent"`
	MaxTimeDivergencePercent  float64 `json:"max_time_divergence_percent"`
	MaxRetryAttempts         int     `json:"max_retry_attempts"`
	ConsensusTimeoutSeconds  int     `json:"consensus_timeout_seconds"`
}

// SimulatorFingerprint uniquely identifies a simulator implementation
type SimulatorFingerprint struct {
	Name                 string                    `json:"name"`
	Version              string                    `json:"version"`
	Backend              string                    `json:"backend"`
	Deterministic        bool                      `json:"deterministic"`
	ConsensusSafe        bool                      `json:"consensus_safe"`
	Capabilities         *SimulatorCapabilities    `json:"capabilities"`
	TestResults          map[string]*TestResult    `json:"test_results"`
	ValidationTimestamp  time.Time                 `json:"validation_timestamp"`
	FingerprintHash      string                    `json:"fingerprint_hash"`
}

// SimulatorCapabilities describes what a simulator can do
type SimulatorCapabilities struct {
	MaxQBits              int      `json:"max_qbits"`
	MaxTCount             int      `json:"max_tcount"`
	MaxLNet               int      `json:"max_lnet"`
	SupportedBackends     []string `json:"supported_backends"`
	DeterministicExecution bool     `json:"deterministic_execution"`
	ConsensusCompatible   bool     `json:"consensus_compatible"`
}

// TestResult contains results from running a test case
type TestResult struct {
	TestCaseID         string    `json:"test_case_id"`
	Success            bool      `json:"success"`
	ActualOutcome      *QuantumTestOutcome `json:"actual_outcome"`
	ComparisonResult   *ComparisonResult   `json:"comparison_result"`
	ExecutionTime      time.Duration       `json:"execution_time"`
	ErrorMessage       string              `json:"error_message,omitempty"`
	ExecutionTimestamp time.Time           `json:"execution_timestamp"`
}

// ComparisonResult contains detailed comparison with expected outcome
type ComparisonResult struct {
	OutcomeMatch     bool    `json:"outcome_match"`
	GateHashMatch    bool    `json:"gate_hash_match"`
	ProofRootMatch   bool    `json:"proof_root_match"`
	TimeWithinTolerance bool `json:"time_within_tolerance"`
	TimeDifferencePercent float64 `json:"time_difference_percent"`
	OverallMatch     bool    `json:"overall_match"`
	Details          string  `json:"details"`
}

// ConsensusResult represents results from multiple simulators
type ConsensusResult struct {
	TestCaseID           string                          `json:"test_case_id"`
	SimulatorResults     map[string]*TestResult          `json:"simulator_results"`
	ConsensusAchieved    bool                           `json:"consensus_achieved"`
	ConsensusNodes       []string                       `json:"consensus_nodes"`
	DissenterNodes       []string                       `json:"dissenter_nodes"`
	MajorityOutcome      *QuantumTestOutcome            `json:"majority_outcome"`
	ConsensusPercentage  float64                        `json:"consensus_percentage"`
	ValidationTimestamp  time.Time                      `json:"validation_timestamp"`
}

// ConsensusValidationResult represents the result of consensus validation
type ConsensusValidationResult struct {
	Success           bool                            `json:"success"`
	Message           string                          `json:"message"`
	FailedTestCases   []string                        `json:"failed_test_cases,omitempty"`
	ValidationResults map[string]*ConsensusResult     `json:"validation_results,omitempty"`
	ExecutionTime     time.Duration                   `json:"execution_time,omitempty"`
}

// NewSimulatorConsensusValidator creates a new consensus validator
func NewSimulatorConsensusValidator() *SimulatorConsensusValidator {
	validator := &SimulatorConsensusValidator{
		referenceTestSuite:   createReferenceTestSuite(),
		registeredSimulators: make(map[string]*SimulatorFingerprint),
		consensusResults:     newConsensusResultsCache(1000, 1*time.Hour),
		stats: &SimulatorConsensusStats{
			ConsensusAgreementAverage: 0.0,
		},
		config: &SimulatorConsensusConfig{
			EnableConsensusValidation: true,
			DefaultTimeoutSeconds:     30,
			MaxConcurrentValidations:  5,
			CacheMaxEntries:          1000,
			CacheEntryTTL:            1 * time.Hour,
			AutoRecoveryEnabled:      true,
			DebugMode:                false,
		},
	}
	
	// Register default deterministic Qiskit simulator
	validator.registerDefaultSimulators()
	
	return validator
}

// createReferenceTestSuite creates the standard quantum computation test suite
func createReferenceTestSuite() *ReferenceQuantumTestSuite {
	suite := &ReferenceQuantumTestSuite{
		TestCases:             make([]*QuantumTestCase, 0),
		TestCasesByComplexity: make(map[string][]*QuantumTestCase),
		ValidationThresholds: &ValidationThresholds{
			MinConsensusNodes:        2,
			ConsensusAgreementPercent: 95.0,
			MaxTimeDivergencePercent:  20.0,
			MaxRetryAttempts:         3,
			ConsensusTimeoutSeconds:  30,
		},
		CreationTime: time.Now(),
	}
	
	// Create comprehensive test cases for consensus validation
	testCases := []*QuantumTestCase{
		{
			ID:               "basic_2q_4t_1p",
			Name:             "Basic 2-Qubit 4-TGate Single Puzzle",
			Description:      "Simple quantum circuit for basic consensus validation",
			ComplexityLevel:  "basic",
			Parameters: &QuantumTestParameters{
				Seed0:       "0000000000000000000000000000000000000000000000000000000000000001",
				QBits:       2,
				TCount:      4,
				LNet:        1,
				MaxTime:     1000,
				Description: "Basic test case for 2 qubits, 4 T-gates, 1 puzzle",
			},
			ToleranceThresholds: &ToleranceThresholds{
				TimeTolerancePercent:     25.0,
				OutcomeMatchRequired:     true,
				GateHashMatchRequired:    true,
				ProofRootMatchRequired:   true,
				DeterministicRequired:    true,
				ConsensusSafeRequired:    true,
			},
			ConsensusRequired: true,
			CreationTime:     time.Now(),
		},
		{
			ID:               "intermediate_8q_16t_4p", 
			Name:             "Intermediate 8-Qubit 16-TGate Four Puzzles",
			Description:      "Intermediate complexity for comprehensive consensus validation",
			ComplexityLevel:  "intermediate",
			Parameters: &QuantumTestParameters{
				Seed0:       "0000000000000000000000000000000000000000000000000000000000000004",
				QBits:       8,
				TCount:      16,
				LNet:        4,
				MaxTime:     5000,
				Description: "Intermediate test case for 8 qubits, 16 T-gates, 4 puzzles",
			},
			ToleranceThresholds: &ToleranceThresholds{
				TimeTolerancePercent:     35.0,
				OutcomeMatchRequired:     true,
				GateHashMatchRequired:    true,
				ProofRootMatchRequired:   true,
				DeterministicRequired:    true,
				ConsensusSafeRequired:    true,
			},
			ConsensusRequired: true,
			CreationTime:     time.Now(),
		},
	}
	
	suite.TestCases = testCases
	suite.TestCasesByComplexity["basic"] = []*QuantumTestCase{testCases[0]}
	suite.TestCasesByComplexity["intermediate"] = []*QuantumTestCase{testCases[1]}
	
	return suite
}

// registerDefaultSimulators registers the built-in simulators
func (scv *SimulatorConsensusValidator) registerDefaultSimulators() {
	// Register the deterministic Qiskit solver
	qiskitFingerprint := &SimulatorFingerprint{
		Name:      "qiskit-deterministic",
		Version:   "1.0",
		Backend:   "qiskit-aer-statevector-deterministic",
		Deterministic: true,
		ConsensusSafe: true,
		Capabilities: &SimulatorCapabilities{
			MaxQBits:              20,
			MaxTCount:             10000,
			MaxLNet:               128,
			SupportedBackends:     []string{"qiskit-aer-statevector-deterministic"},
			DeterministicExecution: true,
			ConsensusCompatible:   true,
		},
		TestResults:         make(map[string]*TestResult),
		ValidationTimestamp: time.Now(),
	}
	
	// Calculate fingerprint hash
	qiskitFingerprint.FingerprintHash = scv.calculateSimulatorFingerprint(qiskitFingerprint)
	
	scv.mu.Lock()
	scv.registeredSimulators["qiskit-deterministic"] = qiskitFingerprint
	scv.stats.RegisteredSimulators++
	scv.mu.Unlock()
	
	log.Info("🔐 Registered default quantum simulator",
		"name", qiskitFingerprint.Name,
		"version", qiskitFingerprint.Version,
		"backend", qiskitFingerprint.Backend,
		"fingerprint", qiskitFingerprint.FingerprintHash[:16])
}

// ValidateSimulatorConsensus validates that simulators maintain consensus
func (scv *SimulatorConsensusValidator) ValidateSimulatorConsensus(testCaseIDs []string) (*ConsensusValidationResult, error) {
	if !scv.config.EnableConsensusValidation {
		return &ConsensusValidationResult{
			Success: true,
			Message: "Consensus validation is disabled",
		}, nil
	}
	
	scv.mu.Lock()
	scv.stats.TotalValidations++
	scv.mu.Unlock()
	
	startTime := time.Now()
	
	if scv.config.DebugMode {
		log.Debug("🧪 Starting simulator consensus validation",
			"test_cases", len(testCaseIDs),
			"registered_simulators", len(scv.registeredSimulators))
	}
	
	// If no specific test cases provided, use basic test cases
	if len(testCaseIDs) == 0 {
		testCaseIDs = []string{"basic_2q_4t_1p", "intermediate_8q_16t_4p"}
	}
	
	validationResults := make(map[string]*ConsensusResult)
	
	// Validate each test case across all simulators
	for _, testCaseID := range testCaseIDs {
		result, err := scv.validateTestCaseConsensus(testCaseID)
		if err != nil {
			scv.mu.Lock()
			scv.stats.ConsensusFailures++
			scv.mu.Unlock()
			
			log.Error("❌ Test case consensus validation failed",
				"test_case", testCaseID,
				"error", err)
			
			return &ConsensusValidationResult{
				Success:      false,
				Message:      fmt.Sprintf("Consensus validation failed for test case %s: %v", testCaseID, err),
				FailedTestCases: []string{testCaseID},
			}, err
		}
		
		validationResults[testCaseID] = result
		
		if !result.ConsensusAchieved {
			scv.mu.Lock()
			scv.stats.ConsensusFailures++
			scv.mu.Unlock()
			
			log.Warn("⚠️ Consensus not achieved for test case",
				"test_case", testCaseID,
				"consensus_percentage", result.ConsensusPercentage,
				"dissenter_nodes", result.DissenterNodes)
			
			return &ConsensusValidationResult{
				Success:           false,
				Message:           fmt.Sprintf("Consensus not achieved for test case %s: %.1f%% agreement", testCaseID, result.ConsensusPercentage),
				FailedTestCases:   []string{testCaseID},
				ValidationResults: validationResults,
			}, nil
		}
	}
	
	scv.mu.Lock()
	scv.stats.SuccessfulConsensus++
	executionTime := time.Since(startTime)
	scv.updateAverageConsensusTime(executionTime)
	scv.stats.LastValidationTime = time.Now()
	scv.mu.Unlock()
	
	if scv.config.DebugMode {
		log.Debug("✅ Simulator consensus validation completed",
			"test_cases", len(testCaseIDs),
			"execution_time", executionTime,
			"all_consensus_achieved", true)
	}
	
	return &ConsensusValidationResult{
		Success:           true,
		Message:           fmt.Sprintf("Consensus validation successful for %d test cases", len(testCaseIDs)),
		ValidationResults: validationResults,
		ExecutionTime:     executionTime,
	}, nil
}

// validateTestCaseConsensus validates consensus for a single test case
func (scv *SimulatorConsensusValidator) validateTestCaseConsensus(testCaseID string) (*ConsensusResult, error) {
	// Check cache first
	if cachedResult := scv.consensusResults.get(testCaseID); cachedResult != nil {
		if scv.config.DebugMode {
			log.Debug("📋 Using cached consensus result", "test_case", testCaseID)
		}
		return cachedResult, nil
	}
	
	// Find test case
	testCase := scv.findTestCase(testCaseID)
	if testCase == nil {
		return nil, fmt.Errorf("test case not found: %s", testCaseID)
	}
	
	if scv.config.DebugMode {
		log.Debug("🔍 Validating test case consensus",
			"test_case", testCaseID,
			"complexity", testCase.ComplexityLevel,
			"parameters", fmt.Sprintf("Q%d-T%d-L%d", testCase.Parameters.QBits, testCase.Parameters.TCount, testCase.Parameters.LNet))
	}
	
	// Execute test case on all registered simulators
	simulatorResults := make(map[string]*TestResult)
	
	scv.mu.RLock()
	simulators := make(map[string]*SimulatorFingerprint)
	for name, sim := range scv.registeredSimulators {
		simulators[name] = sim
	}
	scv.mu.RUnlock()
	
	for simulatorName, simulator := range simulators {
		result, err := scv.executeTestCase(testCase, simulator)
		if err != nil {
			scv.mu.Lock()
			scv.stats.SimulatorFailures++
			scv.mu.Unlock()
			
			log.Warn("⚠️ Simulator execution failed",
				"simulator", simulatorName,
				"test_case", testCaseID,
				"error", err)
				
			// Create failed result
			result = &TestResult{
				TestCaseID:         testCaseID,
				Success:            false,
				ErrorMessage:       err.Error(),
				ExecutionTimestamp: time.Now(),
			}
		}
		
		simulatorResults[simulatorName] = result
	}
	
	// Analyze consensus
	consensusResult := scv.analyzeConsensus(testCase, simulatorResults)
	
	// Cache result
	scv.consensusResults.set(testCaseID, consensusResult)
	
	return consensusResult, nil
}

// executeTestCase executes a test case on a specific simulator
func (scv *SimulatorConsensusValidator) executeTestCase(testCase *QuantumTestCase, simulator *SimulatorFingerprint) (*TestResult, error) {
	startTime := time.Now()
	
	if scv.config.DebugMode {
		log.Debug("⚡ Executing test case on simulator",
			"test_case", testCase.ID,
			"simulator", simulator.Name,
			"backend", simulator.Backend)
	}
	
	// Check if simulator supports this test case
	if !scv.simulatorSupportsTestCase(simulator, testCase) {
		return nil, fmt.Errorf("simulator %s does not support test case %s", simulator.Name, testCase.ID)
	}
	
	// Execute quantum computation using the deterministic Qiskit solver
	outcome, err := scv.executeQuantumComputation(testCase.Parameters)
	if err != nil {
		return nil, fmt.Errorf("quantum execution failed: %v", err)
	}
	
	executionTime := time.Since(startTime)
	
	// Create test result
	result := &TestResult{
		TestCaseID:    testCase.ID,
		Success:       true,
		ActualOutcome: outcome,
		ExecutionTime: executionTime,
		ExecutionTimestamp: time.Now(),
	}
	
	// Compare with expected outcome if available
	if testCase.ExpectedOutcome != nil {
		result.ComparisonResult = scv.compareOutcomes(testCase.ExpectedOutcome, outcome, testCase.ToleranceThresholds)
	} else {
		// First run - set this as expected outcome
		testCase.ExpectedOutcome = outcome
		result.ComparisonResult = &ComparisonResult{
			OutcomeMatch:          true,
			GateHashMatch:         true,
			ProofRootMatch:        true,
			TimeWithinTolerance:   true,
			TimeDifferencePercent: 0.0,
			OverallMatch:          true,
			Details:               "First execution - set as reference",
		}
	}
	
	if scv.config.DebugMode {
		log.Debug("✅ Test case execution completed",
			"test_case", testCase.ID,
			"simulator", simulator.Name,
			"success", result.Success,
			"execution_time", executionTime,
			"overall_match", result.ComparisonResult.OverallMatch)
	}
	
	return result, nil
}

// executeQuantumComputation executes quantum computation using the Qiskit solver
func (scv *SimulatorConsensusValidator) executeQuantumComputation(params *QuantumTestParameters) (*QuantumTestOutcome, error) {
	// Create JSON input for Qiskit solver
	input := map[string]interface{}{
		"seed0":  params.Seed0,
		"qbits":  params.QBits,
		"tcount": params.TCount,
		"lnet":   params.LNet,
	}
	
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %v", err)
	}
	
	// Execute Qiskit solver
	cmd := exec.Command("python3", "quantum-geth/tools/solver/qiskit_solver.py")
	cmd.Stdin = strings.NewReader(string(inputJSON))
	
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	err = cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("qiskit solver execution failed: %v, stderr: %s", err, stderr.String())
	}
	
	// Parse output
	var result map[string]interface{}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("failed to parse solver output: %v", err)
	}
	
	// Check for errors in result
	if errMsg, exists := result["error"]; exists {
		return nil, fmt.Errorf("solver returned error: %v", errMsg)
	}
	
	// Convert to QuantumTestOutcome
	outcome := &QuantumTestOutcome{
		OutcomesHex:      result["outcomes"].(string),
		GateHashHex:      result["gate_hash"].(string),
		ProofRootHex:     result["proof_root"].(string),
		PuzzleCount:      int(result["puzzle_count"].(float64)),
		TotalTime:        result["total_time"].(float64),
		AvgTimePerPuzzle: result["avg_time_per_puzzle"].(float64),
		Backend:          result["backend"].(string),
		Deterministic:    result["deterministic"].(bool),
		ConsensusSafe:    result["consensus_safe"].(bool),
	}
	
	return outcome, nil
}

// Helper functions continue in next part...
func (scv *SimulatorConsensusValidator) simulatorSupportsTestCase(simulator *SimulatorFingerprint, testCase *QuantumTestCase) bool {
	caps := simulator.Capabilities
	params := testCase.Parameters
	
	return params.QBits <= caps.MaxQBits &&
		   params.TCount <= caps.MaxTCount &&
		   params.LNet <= caps.MaxLNet &&
		   caps.ConsensusCompatible
}

func (scv *SimulatorConsensusValidator) compareOutcomes(expected, actual *QuantumTestOutcome, thresholds *ToleranceThresholds) *ComparisonResult {
	result := &ComparisonResult{}
	
	// Compare outcomes
	result.OutcomeMatch = expected.OutcomesHex == actual.OutcomesHex
	result.GateHashMatch = expected.GateHashHex == actual.GateHashHex
	result.ProofRootMatch = expected.ProofRootHex == actual.ProofRootHex
	
	// Compare timing with tolerance
	if expected.TotalTime > 0 {
		result.TimeDifferencePercent = math.Abs(actual.TotalTime-expected.TotalTime) / expected.TotalTime * 100.0
		result.TimeWithinTolerance = result.TimeDifferencePercent <= thresholds.TimeTolerancePercent
	} else {
		result.TimeWithinTolerance = true
	}
	
	// Overall match assessment
	result.OverallMatch = true
	details := []string{}
	
	if thresholds.OutcomeMatchRequired && !result.OutcomeMatch {
		result.OverallMatch = false
		details = append(details, "outcome mismatch")
	}
	
	if thresholds.GateHashMatchRequired && !result.GateHashMatch {
		result.OverallMatch = false
		details = append(details, "gate hash mismatch")
	}
	
	if thresholds.ProofRootMatchRequired && !result.ProofRootMatch {
		result.OverallMatch = false
		details = append(details, "proof root mismatch")
	}
	
	if !result.TimeWithinTolerance {
		result.OverallMatch = false
		details = append(details, fmt.Sprintf("time difference %.1f%% > %.1f%%", result.TimeDifferencePercent, thresholds.TimeTolerancePercent))
	}
	
	if thresholds.DeterministicRequired && !actual.Deterministic {
		result.OverallMatch = false
		details = append(details, "not deterministic")
	}
	
	if thresholds.ConsensusSafeRequired && !actual.ConsensusSafe {
		result.OverallMatch = false
		details = append(details, "not consensus safe")
	}
	
	if len(details) > 0 {
		result.Details = strings.Join(details, ", ")
	} else {
		result.Details = "all checks passed"
	}
	
	return result
}

func (scv *SimulatorConsensusValidator) analyzeConsensus(testCase *QuantumTestCase, results map[string]*TestResult) *ConsensusResult {
	consensusResult := &ConsensusResult{
		TestCaseID:          testCase.ID,
		SimulatorResults:    results,
		ConsensusAchieved:   false,
		ConsensusNodes:      make([]string, 0),
		DissenterNodes:      make([]string, 0),
		ValidationTimestamp: time.Now(),
	}
	
	// Count successful results
	successfulResults := make(map[string]*TestResult)
	for simName, result := range results {
		if result.Success && result.ComparisonResult != nil && result.ComparisonResult.OverallMatch {
			successfulResults[simName] = result
		} else {
			consensusResult.DissenterNodes = append(consensusResult.DissenterNodes, simName)
		}
	}
	
	// Calculate consensus percentage
	totalSimulators := len(results)
	consensusNodes := len(successfulResults)
	consensusResult.ConsensusPercentage = float64(consensusNodes) / float64(totalSimulators) * 100.0
	
	// Check if consensus is achieved
	thresholds := scv.referenceTestSuite.ValidationThresholds
	consensusResult.ConsensusAchieved = consensusResult.ConsensusPercentage >= thresholds.ConsensusAgreementPercent &&
										consensusNodes >= thresholds.MinConsensusNodes
	
	// Set consensus nodes
	for simName := range successfulResults {
		consensusResult.ConsensusNodes = append(consensusResult.ConsensusNodes, simName)
	}
	
	// Set majority outcome (use first successful result as reference)
	if len(successfulResults) > 0 {
		for _, result := range successfulResults {
			consensusResult.MajorityOutcome = result.ActualOutcome
			break
		}
	}
	
	return consensusResult
}

func (scv *SimulatorConsensusValidator) updateAverageConsensusTime(newTime time.Duration) {
	if scv.stats.SuccessfulConsensus == 1 {
		scv.stats.AverageConsensusTime = newTime
	} else {
		// Running average
		oldAvg := scv.stats.AverageConsensusTime
		count := scv.stats.SuccessfulConsensus
		scv.stats.AverageConsensusTime = time.Duration(
			(float64(oldAvg)*float64(count-1) + float64(newTime)) / float64(count),
		)
	}
}

func (scv *SimulatorConsensusValidator) findTestCase(testCaseID string) *QuantumTestCase {
	for _, testCase := range scv.referenceTestSuite.TestCases {
		if testCase.ID == testCaseID {
			return testCase
		}
	}
	return nil
}

func (scv *SimulatorConsensusValidator) calculateSimulatorFingerprint(simulator *SimulatorFingerprint) string {
	data := fmt.Sprintf("%s-%s-%s-%v-%v", 
		simulator.Name, 
		simulator.Version, 
		simulator.Backend,
		simulator.Deterministic,
		simulator.ConsensusSafe)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// ConsensusResultsCache and other structs continue...
type ConsensusResultsCache struct {
	mu            sync.RWMutex
	results       map[string]*ConsensusResult
	maxEntries    int
	entryTTL      time.Duration
	cleanupTicker *time.Ticker
}

type SimulatorConsensusStats struct {
	TotalValidations          int64     `json:"total_validations"`
	SuccessfulConsensus       int64     `json:"successful_consensus"`
	ConsensusFailures         int64     `json:"consensus_failures"`
	SimulatorFailures         int64     `json:"simulator_failures"`
	AverageConsensusTime      time.Duration `json:"average_consensus_time"`
	RegisteredSimulators      int       `json:"registered_simulators"`
	ActiveTestCases           int       `json:"active_test_cases"`
	LastValidationTime        time.Time `json:"last_validation_time"`
	ConsensusAgreementAverage float64   `json:"consensus_agreement_average"`
}

type SimulatorConsensusConfig struct {
	EnableConsensusValidation  bool          `json:"enable_consensus_validation"`
	DefaultTimeoutSeconds      int           `json:"default_timeout_seconds"`
	MaxConcurrentValidations   int           `json:"max_concurrent_validations"`
	CacheMaxEntries           int           `json:"cache_max_entries"`
	CacheEntryTTL             time.Duration `json:"cache_entry_ttl"`
	AutoRecoveryEnabled       bool          `json:"auto_recovery_enabled"`
	DebugMode                 bool          `json:"debug_mode"`
}

func newConsensusResultsCache(maxEntries int, entryTTL time.Duration) *ConsensusResultsCache {
	cache := &ConsensusResultsCache{
		results:    make(map[string]*ConsensusResult),
		maxEntries: maxEntries,
		entryTTL:   entryTTL,
	}
	
	// Start cleanup goroutine
	cache.cleanupTicker = time.NewTicker(entryTTL / 2)
	go cache.cleanupLoop()
	
	return cache
}

func (crc *ConsensusResultsCache) get(key string) *ConsensusResult {
	crc.mu.RLock()
	defer crc.mu.RUnlock()
	
	result, exists := crc.results[key]
	if !exists {
		return nil
	}
	
	// Check TTL
	if time.Since(result.ValidationTimestamp) > crc.entryTTL {
		return nil
	}
	
	return result
}

func (crc *ConsensusResultsCache) set(key string, result *ConsensusResult) {
	crc.mu.Lock()
	defer crc.mu.Unlock()
	
	// Check if we need to evict entries
	if len(crc.results) >= crc.maxEntries {
		// Remove oldest entry
		var oldestKey string
		var oldestTime time.Time
		for k, v := range crc.results {
			if oldestKey == "" || v.ValidationTimestamp.Before(oldestTime) {
				oldestKey = k
				oldestTime = v.ValidationTimestamp
			}
		}
		delete(crc.results, oldestKey)
	}
	
	crc.results[key] = result
}

func (crc *ConsensusResultsCache) cleanupLoop() {
	for range crc.cleanupTicker.C {
		crc.cleanup()
	}
}

func (crc *ConsensusResultsCache) cleanup() {
	crc.mu.Lock()
	defer crc.mu.Unlock()
	
	now := time.Now()
	for key, result := range crc.results {
		if now.Sub(result.ValidationTimestamp) > crc.entryTTL {
			delete(crc.results, key)
		}
	}
}

// Public interface methods
func (scv *SimulatorConsensusValidator) GetConsensusStats() *SimulatorConsensusStats {
	scv.mu.RLock()
	defer scv.mu.RUnlock()
	
	// Create copy of stats
	stats := *scv.stats
	stats.ActiveTestCases = len(scv.referenceTestSuite.TestCases)
	
	return &stats
}

func (scv *SimulatorConsensusValidator) GetRegisteredSimulators() map[string]*SimulatorFingerprint {
	scv.mu.RLock()
	defer scv.mu.RUnlock()
	
	// Create copy of simulators map
	simulators := make(map[string]*SimulatorFingerprint)
	for name, sim := range scv.registeredSimulators {
		simulators[name] = sim
	}
	
	return simulators
}

func (scv *SimulatorConsensusValidator) GetReferenceTestSuite() *ReferenceQuantumTestSuite {
	return scv.referenceTestSuite
} 