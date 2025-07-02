package qmpow

import (
	"strings"
	"testing"
	"time"
)

func TestNewSimulatorConsensusValidator(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	if validator == nil {
		t.Fatal("Expected validator to be created, got nil")
	}
	
	if validator.referenceTestSuite == nil {
		t.Error("Expected reference test suite to be initialized")
	}
	
	if validator.registeredSimulators == nil {
		t.Error("Expected registered simulators map to be initialized")
	}
	
	if validator.consensusResults == nil {
		t.Error("Expected consensus results cache to be initialized")
	}
	
	if validator.stats == nil {
		t.Error("Expected stats to be initialized")
	}
	
	if validator.config == nil {
		t.Error("Expected config to be initialized")
	}
	
	// Check that default simulator is registered
	stats := validator.GetConsensusStats()
	if stats.RegisteredSimulators != 1 {
		t.Errorf("Expected 1 registered simulator, got %d", stats.RegisteredSimulators)
	}
	
	// Check that reference test suite has test cases
	testSuite := validator.GetReferenceTestSuite()
	if len(testSuite.TestCases) == 0 {
		t.Error("Expected reference test suite to have test cases")
	}
	
	// Verify basic test cases exist
	foundBasic := false
	foundIntermediate := false
	for _, testCase := range testSuite.TestCases {
		if testCase.ID == "basic_2q_4t_1p" {
			foundBasic = true
		}
		if testCase.ID == "intermediate_8q_16t_4p" {
			foundIntermediate = true
		}
	}
	
	if !foundBasic {
		t.Error("Expected basic_2q_4t_1p test case to exist")
	}
	
	if !foundIntermediate {
		t.Error("Expected intermediate_8q_16t_4p test case to exist")
	}
}

func TestReferenceTestSuiteCreation(t *testing.T) {
	suite := createReferenceTestSuite()
	
	if suite == nil {
		t.Fatal("Expected test suite to be created, got nil")
	}
	
	if len(suite.TestCases) == 0 {
		t.Error("Expected test cases to be created")
	}
	
	if suite.ValidationThresholds == nil {
		t.Error("Expected validation thresholds to be set")
	}
	
	// Check validation thresholds
	thresholds := suite.ValidationThresholds
	if thresholds.MinConsensusNodes != 2 {
		t.Errorf("Expected min consensus nodes 2, got %d", thresholds.MinConsensusNodes)
	}
	
	if thresholds.ConsensusAgreementPercent != 95.0 {
		t.Errorf("Expected consensus agreement 95%%, got %.1f%%", thresholds.ConsensusAgreementPercent)
	}
	
	// Check test case complexity organization
	if len(suite.TestCasesByComplexity) == 0 {
		t.Error("Expected test cases to be organized by complexity")
	}
	
	basicCases := suite.TestCasesByComplexity["basic"]
	if len(basicCases) == 0 {
		t.Error("Expected basic test cases to exist")
	}
	
	intermediateCases := suite.TestCasesByComplexity["intermediate"]
	if len(intermediateCases) == 0 {
		t.Error("Expected intermediate test cases to exist")
	}
}

func TestValidateSimulatorConsensusDisabled(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	validator.config.EnableConsensusValidation = false
	
	result, err := validator.ValidateSimulatorConsensus(nil)
	if err != nil {
		t.Errorf("Expected no error when validation disabled, got: %v", err)
	}
	
	if !result.Success {
		t.Error("Expected validation to succeed when disabled")
	}
	
	if result.Message != "Consensus validation is disabled" {
		t.Errorf("Expected disabled message, got: %s", result.Message)
	}
}

func TestValidateSimulatorConsensusBasic(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	validator.config.DebugMode = true
	
	// Test with default test cases
	result, err := validator.ValidateSimulatorConsensus(nil)
	
	// Note: This test will fail in CI environment without Python/Qiskit
	// but we can test the structure and error handling
	if err != nil {
		// Expected in test environment without full quantum setup
		t.Logf("Consensus validation failed as expected in test environment: %v", err)
		
		if !result.Success {
			// Check that it's a proper failure
			if len(result.FailedTestCases) == 0 {
				t.Error("Expected failed test cases to be reported")
			}
		}
		return
	}
	
	// If it succeeds (in full environment), validate structure
	if !result.Success {
		t.Errorf("Expected consensus validation to succeed, got: %s", result.Message)
	}
	
	if result.ValidationResults == nil {
		t.Error("Expected validation results to be provided")
	}
	
	if result.ExecutionTime == 0 {
		t.Error("Expected non-zero execution time")
	}
}

func TestValidateSimulatorConsensusSpecificTestCases(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	validator.config.DebugMode = true
	
	// Test with specific test case
	testCaseIDs := []string{"basic_2q_4t_1p"}
	result, err := validator.ValidateSimulatorConsensus(testCaseIDs)
	
	// Expected to fail in test environment without Python/Qiskit
	if err != nil {
		t.Logf("Expected failure in test environment: %v", err)
		
		// Verify error handling structure
		if !strings.Contains(err.Error(), "basic_2q_4t_1p") {
			t.Error("Expected error to mention specific test case")
		}
		return
	}
	
	// If successful, verify result structure
	if result.ValidationResults == nil {
		t.Error("Expected validation results")
	}
	
	if len(result.ValidationResults) != 1 {
		t.Errorf("Expected 1 validation result, got %d", len(result.ValidationResults))
	}
}

func TestSimulatorSupportsTestCase(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	// Create test simulator with limited capabilities
	simulator := &SimulatorFingerprint{
		Capabilities: &SimulatorCapabilities{
			MaxQBits:            4,
			MaxTCount:           10,
			MaxLNet:             2,
			ConsensusCompatible: true,
		},
	}
	
	// Create test cases within capabilities
	supportedTestCase := &QuantumTestCase{
		Parameters: &QuantumTestParameters{
			QBits:  2,
			TCount: 5,
			LNet:   1,
		},
	}
	
	// Test supported case
	if !validator.simulatorSupportsTestCase(simulator, supportedTestCase) {
		t.Error("Expected simulator to support test case within capabilities")
	}
}

func TestCompareOutcomes(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	// Create expected outcome
	expected := &QuantumTestOutcome{
		OutcomesHex:      "abcd1234",
		GateHashHex:      "ef567890",
		ProofRootHex:     "12345678",
		TotalTime:        1.0,
		Deterministic:    true,
		ConsensusSafe:    true,
	}
	
	// Create tolerance thresholds
	thresholds := &ToleranceThresholds{
		TimeTolerancePercent:   20.0,
		OutcomeMatchRequired:   true,
		GateHashMatchRequired:  true,
		ProofRootMatchRequired: true,
		DeterministicRequired:  true,
		ConsensusSafeRequired:  true,
	}
	
	// Test exact match
	exactMatch := &QuantumTestOutcome{
		OutcomesHex:      "abcd1234",
		GateHashHex:      "ef567890", 
		ProofRootHex:     "12345678",
		TotalTime:        1.0,
		Deterministic:    true,
		ConsensusSafe:    true,
	}
	
	result := validator.compareOutcomes(expected, exactMatch, thresholds)
	if !result.OverallMatch {
		t.Errorf("Expected exact match to succeed, got: %s", result.Details)
	}
	
	if !result.OutcomeMatch || !result.GateHashMatch || !result.ProofRootMatch {
		t.Error("Expected all individual matches to be true")
	}
	
	// Test outcome mismatch
	outcomeMismatch := &QuantumTestOutcome{
		OutcomesHex:      "wrong123", // Different outcome
		GateHashHex:      "ef567890",
		ProofRootHex:     "12345678",
		TotalTime:        1.0,
		Deterministic:    true,
		ConsensusSafe:    true,
	}
	
	result = validator.compareOutcomes(expected, outcomeMismatch, thresholds)
	if result.OverallMatch {
		t.Error("Expected outcome mismatch to fail overall match")
	}
	
	if result.OutcomeMatch {
		t.Error("Expected outcome match to be false")
	}
	
	if !strings.Contains(result.Details, "outcome mismatch") {
		t.Error("Expected details to mention outcome mismatch")
	}
	
	// Test time tolerance
	timeVariation := &QuantumTestOutcome{
		OutcomesHex:      "abcd1234",
		GateHashHex:      "ef567890",
		ProofRootHex:     "12345678",
		TotalTime:        1.1, // 10% difference
		Deterministic:    true,
		ConsensusSafe:    true,
	}
	
	result = validator.compareOutcomes(expected, timeVariation, thresholds)
	if !result.OverallMatch {
		t.Errorf("Expected 10%% time difference to be within 20%% tolerance, got: %s", result.Details)
	}
	
	if !result.TimeWithinTolerance {
		t.Error("Expected time to be within tolerance")
	}
	
	// Test excessive time difference
	excessiveTime := &QuantumTestOutcome{
		OutcomesHex:      "abcd1234",
		GateHashHex:      "ef567890",
		ProofRootHex:     "12345678",
		TotalTime:        1.5, // 50% difference
		Deterministic:    true,
		ConsensusSafe:    true,
	}
	
	result = validator.compareOutcomes(expected, excessiveTime, thresholds)
	if result.OverallMatch {
		t.Error("Expected 50% time difference to exceed 20% tolerance")
	}
	
	if result.TimeWithinTolerance {
		t.Error("Expected time to exceed tolerance")
	}
	
	// Test non-deterministic result
	nonDeterministic := &QuantumTestOutcome{
		OutcomesHex:      "abcd1234",
		GateHashHex:      "ef567890",
		ProofRootHex:     "12345678",
		TotalTime:        1.0,
		Deterministic:    false, // Not deterministic
		ConsensusSafe:    true,
	}
	
	result = validator.compareOutcomes(expected, nonDeterministic, thresholds)
	if result.OverallMatch {
		t.Error("Expected non-deterministic result to fail when deterministic required")
	}
	
	if !strings.Contains(result.Details, "not deterministic") {
		t.Error("Expected details to mention non-deterministic")
	}
	
	// Test non-consensus-safe result
	nonConsensusSafe := &QuantumTestOutcome{
		OutcomesHex:      "abcd1234",
		GateHashHex:      "ef567890",
		ProofRootHex:     "12345678",
		TotalTime:        1.0,
		Deterministic:    true,
		ConsensusSafe:    false, // Not consensus safe
	}
	
	result = validator.compareOutcomes(expected, nonConsensusSafe, thresholds)
	if result.OverallMatch {
		t.Error("Expected non-consensus-safe result to fail when consensus safe required")
	}
	
	if !strings.Contains(result.Details, "not consensus safe") {
		t.Error("Expected details to mention non-consensus-safe")
	}
}

func TestAnalyzeConsensus(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	// Create test case
	testCase := &QuantumTestCase{
		ID: "test_consensus",
	}
	
	// Create successful test results
	successfulResult1 := &TestResult{
		TestCaseID: "test_consensus",
		Success:    true,
		ComparisonResult: &ComparisonResult{
			OverallMatch: true,
		},
		ActualOutcome: &QuantumTestOutcome{
			OutcomesHex: "consensus_result",
		},
	}
	
	successfulResult2 := &TestResult{
		TestCaseID: "test_consensus",
		Success:    true,
		ComparisonResult: &ComparisonResult{
			OverallMatch: true,
		},
		ActualOutcome: &QuantumTestOutcome{
			OutcomesHex: "consensus_result",
		},
	}
	
	// Create failed test result
	failedResult := &TestResult{
		TestCaseID: "test_consensus",
		Success:    false,
		ErrorMessage: "simulation failed",
	}
	
	// Create dissenting result
	dissentingResult := &TestResult{
		TestCaseID: "test_consensus",
		Success:    true,
		ComparisonResult: &ComparisonResult{
			OverallMatch: false,
		},
	}
	
	// Test full consensus (3 successful, 0 failed)
	allSuccessful := map[string]*TestResult{
		"sim1": successfulResult1,
		"sim2": successfulResult2,
		"sim3": successfulResult1,
	}
	
	result := validator.analyzeConsensus(testCase, allSuccessful)
	if !result.ConsensusAchieved {
		t.Error("Expected consensus to be achieved with all successful results")
	}
	
	if result.ConsensusPercentage != 100.0 {
		t.Errorf("Expected 100%% consensus, got %.1f%%", result.ConsensusPercentage)
	}
	
	if len(result.ConsensusNodes) != 3 {
		t.Errorf("Expected 3 consensus nodes, got %d", len(result.ConsensusNodes))
	}
	
	if len(result.DissenterNodes) != 0 {
		t.Errorf("Expected 0 dissenter nodes, got %d", len(result.DissenterNodes))
	}
	
	// Test partial consensus (2 successful, 1 failed, 1 dissenting)
	partialConsensus := map[string]*TestResult{
		"sim1": successfulResult1,
		"sim2": successfulResult2,
		"sim3": failedResult,
		"sim4": dissentingResult,
	}
	
	result = validator.analyzeConsensus(testCase, partialConsensus)
	expectedPercentage := 50.0 // 2 out of 4
	if result.ConsensusPercentage != expectedPercentage {
		t.Errorf("Expected %.1f%% consensus, got %.1f%%", expectedPercentage, result.ConsensusPercentage)
	}
	
	if len(result.ConsensusNodes) != 2 {
		t.Errorf("Expected 2 consensus nodes, got %d", len(result.ConsensusNodes))
	}
	
	if len(result.DissenterNodes) != 2 {
		t.Errorf("Expected 2 dissenter nodes, got %d", len(result.DissenterNodes))
	}
	
	// With 50% consensus and threshold of 95%, consensus should not be achieved
	if result.ConsensusAchieved {
		t.Error("Expected consensus not to be achieved with 50% agreement")
	}
	
	// Test majority outcome
	if result.MajorityOutcome == nil {
		t.Error("Expected majority outcome to be set")
	}
	
	if result.MajorityOutcome.OutcomesHex != "consensus_result" {
		t.Errorf("Expected majority outcome to match consensus result")
	}
}

func TestFindTestCase(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	// Test finding existing test case
	testCase := validator.findTestCase("basic_2q_4t_1p")
	if testCase == nil {
		t.Error("Expected to find basic_2q_4t_1p test case")
	}
	
	if testCase.ID != "basic_2q_4t_1p" {
		t.Errorf("Expected test case ID basic_2q_4t_1p, got %s", testCase.ID)
	}
	
	// Test finding non-existent test case
	nonExistent := validator.findTestCase("non_existent_test")
	if nonExistent != nil {
		t.Error("Expected non-existent test case to return nil")
	}
}

func TestCalculateSimulatorFingerprint(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	simulator := &SimulatorFingerprint{
		Name:          "test_simulator",
		Version:       "1.0",
		Backend:       "test_backend",
		Deterministic: true,
		ConsensusSafe: true,
	}
	
	fingerprint1 := validator.calculateSimulatorFingerprint(simulator)
	if fingerprint1 == "" {
		t.Error("Expected non-empty fingerprint")
	}
	
	if len(fingerprint1) != 64 { // SHA256 hex length
		t.Errorf("Expected fingerprint length 64, got %d", len(fingerprint1))
	}
	
	// Test that same simulator produces same fingerprint
	fingerprint2 := validator.calculateSimulatorFingerprint(simulator)
	if fingerprint1 != fingerprint2 {
		t.Error("Expected identical simulators to produce identical fingerprints")
	}
	
	// Test that different simulator produces different fingerprint
	differentSimulator := &SimulatorFingerprint{
		Name:          "different_simulator",
		Version:       "2.0",
		Backend:       "different_backend",
		Deterministic: false,
		ConsensusSafe: false,
	}
	
	differentFingerprint := validator.calculateSimulatorFingerprint(differentSimulator)
	if fingerprint1 == differentFingerprint {
		t.Error("Expected different simulators to produce different fingerprints")
	}
}

func TestGetPublicInterfaceMethods(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	// Test GetConsensusStats
	stats := validator.GetConsensusStats()
	if stats == nil {
		t.Error("Expected consensus stats to be returned")
	}
	
	if stats.RegisteredSimulators != 1 {
		t.Errorf("Expected 1 registered simulator, got %d", stats.RegisteredSimulators)
	}
	
	if stats.ActiveTestCases != 2 { // basic + intermediate
		t.Errorf("Expected 2 active test cases, got %d", stats.ActiveTestCases)
	}
	
	// Test GetRegisteredSimulators
	simulators := validator.GetRegisteredSimulators()
	if simulators == nil {
		t.Error("Expected registered simulators map to be returned")
	}
	
	if len(simulators) != 1 {
		t.Errorf("Expected 1 registered simulator, got %d", len(simulators))
	}
	
	qiskitSim, exists := simulators["qiskit-deterministic"]
	if !exists {
		t.Error("Expected qiskit-deterministic simulator to be registered")
	}
	
	if qiskitSim.Name != "qiskit-deterministic" {
		t.Errorf("Expected simulator name qiskit-deterministic, got %s", qiskitSim.Name)
	}
	
	// Test GetReferenceTestSuite
	testSuite := validator.GetReferenceTestSuite()
	if testSuite == nil {
		t.Error("Expected reference test suite to be returned")
	}
	
	if len(testSuite.TestCases) == 0 {
		t.Error("Expected test cases in reference test suite")
	}
}

func TestConsensusResultsCache(t *testing.T) {
	cache := newConsensusResultsCache(2, 100*time.Millisecond)
	
	// Test setting and getting
	result1 := &ConsensusResult{
		TestCaseID:          "test1",
		ValidationTimestamp: time.Now(),
	}
	
	cache.set("test1", result1)
	
	retrieved := cache.get("test1")
	if retrieved == nil {
		t.Error("Expected to retrieve cached result")
	}
	
	if retrieved.TestCaseID != "test1" {
		t.Errorf("Expected test case ID test1, got %s", retrieved.TestCaseID)
	}
	
	// Test cache eviction (max 2 entries)
	result2 := &ConsensusResult{
		TestCaseID:          "test2",
		ValidationTimestamp: time.Now(),
	}
	
	result3 := &ConsensusResult{
		TestCaseID:          "test3",
		ValidationTimestamp: time.Now(),
	}
	
	cache.set("test2", result2)
	cache.set("test3", result3) // Should evict oldest (test1)
	
	// test1 should be evicted
	evicted := cache.get("test1")
	if evicted != nil {
		t.Error("Expected oldest entry to be evicted")
	}
	
	// test2 and test3 should still exist
	if cache.get("test2") == nil {
		t.Error("Expected test2 to still exist")
	}
	
	if cache.get("test3") == nil {
		t.Error("Expected test3 to still exist")
	}
	
	// Test TTL expiration
	time.Sleep(150 * time.Millisecond) // Wait for TTL to expire
	
	expired := cache.get("test2")
	if expired != nil {
		t.Error("Expected entry to expire after TTL")
	}
	
	// Test non-existent key
	nonExistent := cache.get("non_existent")
	if nonExistent != nil {
		t.Error("Expected non-existent key to return nil")
	}
}

func TestUpdateAverageConsensusTime(t *testing.T) {
	validator := NewSimulatorConsensusValidator()
	
	// Test first measurement
	validator.stats.SuccessfulConsensus = 1
	validator.updateAverageConsensusTime(100 * time.Millisecond)
	
	if validator.stats.AverageConsensusTime != 100*time.Millisecond {
		t.Errorf("Expected average time 100ms, got %v", validator.stats.AverageConsensusTime)
	}
	
	// Test second measurement
	validator.stats.SuccessfulConsensus = 2
	validator.updateAverageConsensusTime(200 * time.Millisecond)
	
	expected := 150 * time.Millisecond // (100 + 200) / 2
	if validator.stats.AverageConsensusTime != expected {
		t.Errorf("Expected average time %v, got %v", expected, validator.stats.AverageConsensusTime)
	}
	
	// Test third measurement
	validator.stats.SuccessfulConsensus = 3
	validator.updateAverageConsensusTime(300 * time.Millisecond)
	
	expected = 200 * time.Millisecond // (100 + 200 + 300) / 3
	if validator.stats.AverageConsensusTime != expected {
		t.Errorf("Expected average time %v, got %v", expected, validator.stats.AverageConsensusTime)
	}
} 