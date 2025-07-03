// security_integration_testing_test.go
// Tests for Phase 5.2 Security Integration Testing

package qmpow

import (
	"testing"
	"time"
)

func TestSecurityIntegrationTester_Creation(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	tester := NewSecurityIntegrationTester(config)
	
	if tester == nil {
		t.Fatal("Failed to create security integration tester")
	}
	
	if tester.config != config {
		t.Error("Config not properly set")
	}
	
	if tester.testResults == nil {
		t.Error("Test results map not initialized")
	}
	
	if tester.testMetrics == nil {
		t.Error("Test metrics not initialized")
	}
}

func TestDefaultIntegrationTestConfig(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	
	if config.TestTimeout != 60*time.Minute {
		t.Errorf("Expected TestTimeout=60m, got %v", config.TestTimeout)
	}
	
	if config.BasicIntegrationTests != 20 {
		t.Errorf("Expected BasicIntegrationTests=20, got %d", config.BasicIntegrationTests)
	}
	
	if config.AdvancedIntegrationTests != 15 {
		t.Errorf("Expected AdvancedIntegrationTests=15, got %d", config.AdvancedIntegrationTests)
	}
	
	if config.RequiredDetectionRate != 0.95 {
		t.Errorf("Expected RequiredDetectionRate=0.95, got %f", config.RequiredDetectionRate)
	}
	
	if !config.TestAntiClassicalIntegration {
		t.Error("Expected TestAntiClassicalIntegration=true")
	}
}

func TestIntegrationTestTypes(t *testing.T) {
	testTypes := []IntegrationTestType{
		IntegrationTypeBasic,
		IntegrationTypeAdvanced,
		IntegrationTypeStress,
		IntegrationTypeFullPipeline,
		IntegrationTypeSecurity,
		IntegrationTypePerformance,
	}
	
	// Verify all test types are distinct
	typeMap := make(map[IntegrationTestType]bool)
	for _, testType := range testTypes {
		if typeMap[testType] {
			t.Errorf("Duplicate test type: %v", testType)
		}
		typeMap[testType] = true
	}
	
	if len(typeMap) != 6 {
		t.Errorf("Expected 6 distinct test types, got %d", len(typeMap))
	}
}

func TestSecurityIntegrationTester_ValidateComponentInitialization(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	tester := NewSecurityIntegrationTester(config)
	
	// Test with missing components (should fail)
	err := tester.validateComponentInitialization()
	if err == nil {
		t.Error("Expected validation error with missing components")
	}
	
	// Test with security suite only
	tester.securitySuite = &SecurityTestingSuite{}
	err = tester.validateComponentInitialization()
	if err == nil {
		t.Error("Expected validation error with missing block validator")
	}
}

func TestSecurityIntegrationTester_CreateTestBlock(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	tester := NewSecurityIntegrationTester(config)
	
	testBlock := tester.createTestBlock()
	if testBlock == nil {
		t.Error("Failed to create test block")
	}
	
	if testBlock.Header() == nil {
		t.Error("Test block has no header")
	}
	
	if testBlock.Header().GasLimit != 8000000 {
		t.Errorf("Expected GasLimit=8000000, got %d", testBlock.Header().GasLimit)
	}
}

func TestSecurityIntegrationTester_CalculateIntegrationScores(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	tester := NewSecurityIntegrationTester(config)
	
	// Initialize metrics
	tester.testMetrics = &IntegrationTestMetrics{
		TotalIntegrationTests:           10,
		PassedIntegrationTests:          8,
		FailedIntegrationTests:          2,
		AntiClassicalIntegrationSuccess: true,
		CacheIntegrationSuccess:        true,
		ParallelIntegrationSuccess:     false,
		FullPipelineIntegrationSuccess: true,
	}
	
	tester.calculateIntegrationScores()
	
	// Check component compatibility score (3/4 = 0.75)
	expectedCompatibility := 0.75
	if tester.testMetrics.ComponentCompatibilityScore != expectedCompatibility {
		t.Errorf("Expected ComponentCompatibilityScore=%.2f, got %.2f", 
			expectedCompatibility, tester.testMetrics.ComponentCompatibilityScore)
	}
	
	// Check integration health score (average of pass rate and compatibility)
	passRate := 8.0 / 10.0 // 0.8
	expectedHealth := (passRate + expectedCompatibility) / 2.0 // 0.775
	if tester.testMetrics.IntegrationHealthScore != expectedHealth {
		t.Errorf("Expected IntegrationHealthScore=%.3f, got %.3f", 
			expectedHealth, tester.testMetrics.IntegrationHealthScore)
	}
	
	// Check system stability score (should be 1.0 with no performance impact)
	if tester.testMetrics.SystemStabilityScore != 1.0 {
		t.Errorf("Expected SystemStabilityScore=1.0, got %.2f", 
			tester.testMetrics.SystemStabilityScore)
	}
}

func TestSecurityIntegrationTester_GetIntegrationTestResults(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	tester := NewSecurityIntegrationTester(config)
	
	// Add test result
	testResult := &IntegrationTestResult{
		TestName:  "test_integration",
		TestType:  IntegrationTypeBasic,
		StartTime: time.Now(),
		Duration:  time.Second,
		Success:   true,
		ComponentsIntegrated: []string{"SecurityTestingSuite"},
	}
	
	tester.testResults["test_integration"] = testResult
	
	results := tester.GetIntegrationTestResults()
	if len(results) != 1 {
		t.Errorf("Expected 1 test result, got %d", len(results))
	}
	
	if results["test_integration"] == nil {
		t.Error("Test result not found")
	}
	
	if results["test_integration"].TestName != "test_integration" {
		t.Errorf("Expected TestName=test_integration, got %s", 
			results["test_integration"].TestName)
	}
}

func TestSecurityIntegrationTester_GetIntegrationTestMetrics(t *testing.T) {
	config := DefaultIntegrationTestConfig()
	tester := NewSecurityIntegrationTester(config)
	
	// Initialize metrics
	tester.testMetrics = &IntegrationTestMetrics{
		TotalIntegrationTests:           5,
		PassedIntegrationTests:          4,
		FailedIntegrationTests:          1,
		IntegratedDetectionRate:         0.95,
		IntegratedSecurityScore:         0.90,
	}
	
	metrics := tester.GetIntegrationTestMetrics()
	if metrics == nil {
		t.Error("Failed to get integration test metrics")
	}
	
	if metrics.TotalIntegrationTests != 5 {
		t.Errorf("Expected TotalIntegrationTests=5, got %d", metrics.TotalIntegrationTests)
	}
	
	if metrics.PassedIntegrationTests != 4 {
		t.Errorf("Expected PassedIntegrationTests=4, got %d", metrics.PassedIntegrationTests)
	}
	
	if metrics.IntegratedDetectionRate != 0.95 {
		t.Errorf("Expected IntegratedDetectionRate=0.95, got %.2f", 
			metrics.IntegratedDetectionRate)
	}
}

func TestSecurityIntegrationTester_ConfigValidation(t *testing.T) {
	// Test with nil config (should use default)
	tester := NewSecurityIntegrationTester(nil)
	if tester.config == nil {
		t.Error("Expected default config when nil provided")
	}
	
	// Test with custom config
	customConfig := &IntegrationTestConfig{
		TestTimeout:                   30 * time.Minute,
		BasicIntegrationTests:         10,
		TestAntiClassicalIntegration: false,
		RequiredDetectionRate:        0.90,
	}
	
	tester = NewSecurityIntegrationTester(customConfig)
	if tester.config.TestTimeout != 30*time.Minute {
		t.Errorf("Expected TestTimeout=30m, got %v", tester.config.TestTimeout)
	}
	
	if tester.config.BasicIntegrationTests != 10 {
		t.Errorf("Expected BasicIntegrationTests=10, got %d", 
			tester.config.BasicIntegrationTests)
	}
	
	if tester.config.RequiredDetectionRate != 0.90 {
		t.Errorf("Expected RequiredDetectionRate=0.90, got %.2f", 
			tester.config.RequiredDetectionRate)
	}
} 