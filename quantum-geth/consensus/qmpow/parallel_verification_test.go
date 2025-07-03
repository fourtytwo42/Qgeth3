// parallel_verification_test.go
// Tests for Phase 4.1 Parallel Verification Implementation

package qmpow

import (
	"context"
	"runtime"
	"testing"
	"time"
)

func TestDefaultParallelVerificationConfig(t *testing.T) {
	config := DefaultParallelVerificationConfig()
	
	if config.CAPSSWorkers != runtime.NumCPU() {
		t.Errorf("Expected CAPSSWorkers=%d, got %d", runtime.NumCPU(), config.CAPSSWorkers)
	}
	
	if config.NovaWorkers != runtime.NumCPU()/2 {
		t.Errorf("Expected NovaWorkers=%d, got %d", runtime.NumCPU()/2, config.NovaWorkers)
	}
	
	if config.WorkerTimeout != 30*time.Second {
		t.Errorf("Expected WorkerTimeout=30s, got %v", config.WorkerTimeout)
	}
	
	if !config.EnableLoadBalancing {
		t.Error("Expected EnableLoadBalancing=true")
	}
	
	if !config.EnablePriorityQueue {
		t.Error("Expected EnablePriorityQueue=true")
	}
	
	if !config.EnableMemoryPooling {
		t.Error("Expected EnableMemoryPooling=true")
	}
	
	if !config.EnableMetrics {
		t.Error("Expected EnableMetrics=true")
	}
	
	t.Log("✅ Default parallel verification config validated")
}

func TestNewParallelVerificationEngine(t *testing.T) {
	config := DefaultParallelVerificationConfig()
	engine := NewParallelVerificationEngine(config)
	
	if engine == nil {
		t.Fatal("NewParallelVerificationEngine returned nil")
	}
	
	if engine.config.CAPSSWorkers != config.CAPSSWorkers {
		t.Errorf("Expected CAPSSWorkers=%d, got %d", config.CAPSSWorkers, engine.config.CAPSSWorkers)
	}
	
	if engine.capssWorkerPool == nil {
		t.Error("CAPSS worker pool not initialized")
	}
	
	if engine.novaWorkerPool == nil {
		t.Error("Nova worker pool not initialized")
	}
	
	if engine.taskQueue == nil {
		t.Error("Task queue not initialized")
	}
	
	if engine.memoryPool == nil {
		t.Error("Memory pool not initialized")
	}
	
	if engine.metrics == nil {
		t.Error("Metrics not initialized")
	}
	
	if engine.loadBalancer == nil {
		t.Error("Load balancer not initialized (should be enabled by default)")
	}
	
	if engine.running {
		t.Error("Engine should not be running initially")
	}
	
	t.Log("✅ Parallel verification engine created successfully")
}

func TestVerificationTaskType(t *testing.T) {
	// Test task type constants
	if TaskTypeCAPSSProof != 0 {
		t.Errorf("Expected TaskTypeCAPSSProof=0, got %d", TaskTypeCAPSSProof)
	}
	
	if TaskTypeNovaProof != 1 {
		t.Errorf("Expected TaskTypeNovaProof=1, got %d", TaskTypeNovaProof)
	}
	
	if TaskTypeBlockValidation != 2 {
		t.Errorf("Expected TaskTypeBlockValidation=2, got %d", TaskTypeBlockValidation)
	}
	
	if TaskTypeProofChain != 3 {
		t.Errorf("Expected TaskTypeProofChain=3, got %d", TaskTypeProofChain)
	}
	
	if TaskTypeQuantumAuthenticity != 4 {
		t.Errorf("Expected TaskTypeQuantumAuthenticity=4, got %d", TaskTypeQuantumAuthenticity)
	}
	
	t.Log("✅ Verification task types validated")
}

func TestVerificationPriority(t *testing.T) {
	// Test priority constants
	if PriorityLow != 0 {
		t.Errorf("Expected PriorityLow=0, got %d", PriorityLow)
	}
	
	if PriorityNormal != 1 {
		t.Errorf("Expected PriorityNormal=1, got %d", PriorityNormal)
	}
	
	if PriorityHigh != 2 {
		t.Errorf("Expected PriorityHigh=2, got %d", PriorityHigh)
	}
	
	if PriorityCritical != 3 {
		t.Errorf("Expected PriorityCritical=3, got %d", PriorityCritical)
	}
	
	t.Log("✅ Verification priorities validated")
}

func TestVerificationTask(t *testing.T) {
	task := &VerificationTask{
		ID:       "test_task_1",
		Type:     TaskTypeCAPSSProof,
		Priority: PriorityHigh,
		Data:     "test_data",
		Context:  context.Background(),
		ResultChan: make(chan VerificationTaskResult, 1),
		SubmitTime: time.Now(),
	}
	
	if task.ID != "test_task_1" {
		t.Errorf("Expected ID='test_task_1', got '%s'", task.ID)
	}
	
	if task.Type != TaskTypeCAPSSProof {
		t.Errorf("Expected Type=TaskTypeCAPSSProof, got %d", task.Type)
	}
	
	if task.Priority != PriorityHigh {
		t.Errorf("Expected Priority=PriorityHigh, got %d", task.Priority)
	}
	
	if task.Data != "test_data" {
		t.Errorf("Expected Data='test_data', got '%v'", task.Data)
	}
	
	if task.ResultChan == nil {
		t.Error("ResultChan should not be nil")
	}
	
	if task.SubmitTime.IsZero() {
		t.Error("SubmitTime should not be zero")
	}
	
	t.Log("✅ Verification task structure validated")
}

func TestVerificationTaskResult(t *testing.T) {
	result := VerificationTaskResult{
		TaskID:         "test_task_1",
		Success:        true,
		Error:          nil,
		ProcessingTime: 100 * time.Millisecond,
		WorkerID:       1,
		Result:         "test_result",
	}
	
	if result.TaskID != "test_task_1" {
		t.Errorf("Expected TaskID='test_task_1', got '%s'", result.TaskID)
	}
	
	if !result.Success {
		t.Error("Expected Success=true")
	}
	
	if result.Error != nil {
		t.Errorf("Expected Error=nil, got %v", result.Error)
	}
	
	if result.ProcessingTime != 100*time.Millisecond {
		t.Errorf("Expected ProcessingTime=100ms, got %v", result.ProcessingTime)
	}
	
	if result.WorkerID != 1 {
		t.Errorf("Expected WorkerID=1, got %d", result.WorkerID)
	}
	
	if result.Result != "test_result" {
		t.Errorf("Expected Result='test_result', got '%v'", result.Result)
	}
	
	t.Log("✅ Verification task result structure validated")
}

func TestParallelVerificationEngineLifecycle(t *testing.T) {
	config := DefaultParallelVerificationConfig()
	// Reduce workers for testing
	config.CAPSSWorkers = 2
	config.NovaWorkers = 1
	config.MetricsInterval = 100 * time.Millisecond
	
	engine := NewParallelVerificationEngine(config)
	
	// Test initial state
	if engine.isRunning() {
		t.Error("Engine should not be running initially")
	}
	
	// Test start
	err := engine.Start()
	if err != nil {
		t.Fatalf("Failed to start engine: %v", err)
	}
	
	if !engine.isRunning() {
		t.Error("Engine should be running after start")
	}
	
	// Test starting already running engine
	err = engine.Start()
	if err == nil {
		t.Error("Expected error when starting already running engine")
	}
	
	// Let it run for a short time
	time.Sleep(200 * time.Millisecond)
	
	// Test stop
	engine.Stop()
	
	if engine.isRunning() {
		t.Error("Engine should not be running after stop")
	}
	
	// Test stopping already stopped engine (should not panic)
	engine.Stop()
	
	t.Log("✅ Parallel verification engine lifecycle working correctly")
}

func TestParallelVerificationEngineNotRunning(t *testing.T) {
	config := DefaultParallelVerificationConfig()
	engine := NewParallelVerificationEngine(config)
	
	// Create a mock CAPSS proof
	proof := &CAPSSProof{
		TraceID: 12345,
		Proof:   []byte("test_proof_data"),
	}
	
	// Test submitting to non-running engine
	_, err := engine.SubmitCAPSSVerification(proof, PriorityNormal)
	if err == nil {
		t.Error("Expected error when submitting to non-running engine")
	}
	
	// Test submitting Nova verification to non-running engine
	novaProof := &NovaLiteProof{
		ProofID:   67890,
		ProofData: []byte("test_nova_proof_data"),
	}
	
	_, err = engine.SubmitNovaVerification(novaProof, PriorityNormal)
	if err == nil {
		t.Error("Expected error when submitting Nova verification to non-running engine")
	}
	
	// Test parallel verification on non-running engine
	_, _, err = engine.VerifyProofsParallel([]*CAPSSProof{proof}, []*NovaLiteProof{novaProof}, PriorityNormal)
	if err == nil {
		t.Error("Expected error when calling VerifyProofsParallel on non-running engine")
	}
	
	t.Log("✅ Non-running engine properly rejects verification requests")
}

func TestParallelVerificationMetrics(t *testing.T) {
	config := DefaultParallelVerificationConfig()
	engine := NewParallelVerificationEngine(config)
	
	// Get initial metrics
	metrics := engine.GetMetrics()
	
	if metrics.TotalTasks != 0 {
		t.Errorf("Expected initial TotalTasks=0, got %d", metrics.TotalTasks)
	}
	
	if metrics.CompletedTasks != 0 {
		t.Errorf("Expected initial CompletedTasks=0, got %d", metrics.CompletedTasks)
	}
	
	if metrics.FailedTasks != 0 {
		t.Errorf("Expected initial FailedTasks=0, got %d", metrics.FailedTasks)
	}
	
	if metrics.QueuedTasks != 0 {
		t.Errorf("Expected initial QueuedTasks=0, got %d", metrics.QueuedTasks)
	}
	
	if metrics.LastUpdateTime.IsZero() {
		t.Error("LastUpdateTime should not be zero")
	}
	
	t.Log("✅ Parallel verification metrics initialized correctly")
}

func TestWorkerMetrics(t *testing.T) {
	metrics := WorkerMetrics{
		TasksProcessed:        10,
		TotalProcessingTime:   1 * time.Second,
		AverageProcessingTime: 100 * time.Millisecond,
		ErrorCount:            1,
		LastTaskTime:          time.Now(),
		IsActive:              true,
	}
	
	if metrics.TasksProcessed != 10 {
		t.Errorf("Expected TasksProcessed=10, got %d", metrics.TasksProcessed)
	}
	
	if metrics.TotalProcessingTime != 1*time.Second {
		t.Errorf("Expected TotalProcessingTime=1s, got %v", metrics.TotalProcessingTime)
	}
	
	if metrics.AverageProcessingTime != 100*time.Millisecond {
		t.Errorf("Expected AverageProcessingTime=100ms, got %v", metrics.AverageProcessingTime)
	}
	
	if metrics.ErrorCount != 1 {
		t.Errorf("Expected ErrorCount=1, got %d", metrics.ErrorCount)
	}
	
	if !metrics.IsActive {
		t.Error("Expected IsActive=true")
	}
	
	if metrics.LastTaskTime.IsZero() {
		t.Error("LastTaskTime should not be zero")
	}
	
	t.Log("✅ Worker metrics structure validated")
}

func TestLoadBalancerMetrics(t *testing.T) {
	metrics := LoadBalancerMetrics{
		RebalanceOperations: 5,
		LastRebalanceTime:   time.Now(),
		WorkerUtilization:   []float64{0.8, 0.9, 0.7},
		AverageUtilization:  0.8,
	}
	
	if metrics.RebalanceOperations != 5 {
		t.Errorf("Expected RebalanceOperations=5, got %d", metrics.RebalanceOperations)
	}
	
	if metrics.LastRebalanceTime.IsZero() {
		t.Error("LastRebalanceTime should not be zero")
	}
	
	if len(metrics.WorkerUtilization) != 3 {
		t.Errorf("Expected WorkerUtilization length=3, got %d", len(metrics.WorkerUtilization))
	}
	
	if metrics.AverageUtilization != 0.8 {
		t.Errorf("Expected AverageUtilization=0.8, got %f", metrics.AverageUtilization)
	}
	
	t.Log("✅ Load balancer metrics structure validated")
}

func TestParallelVerificationConfigCustomization(t *testing.T) {
	config := ParallelVerificationConfig{
		CAPSSWorkers:          4,
		NovaWorkers:           2,
		MaxConcurrentBlocks:   8,
		WorkerTimeout:         45 * time.Second,
		EnableLoadBalancing:   false,
		LoadBalanceInterval:   10 * time.Second,
		EnablePriorityQueue:   false,
		HighPriorityTimeout:   5 * time.Second,
		NormalPriorityTimeout: 20 * time.Second,
		EnableMemoryPooling:   false,
		MaxPoolSize:           500,
		EnableMetrics:         false,
		MetricsInterval:       30 * time.Second,
	}
	
	engine := NewParallelVerificationEngine(config)
	
	if engine.config.CAPSSWorkers != 4 {
		t.Errorf("Expected CAPSSWorkers=4, got %d", engine.config.CAPSSWorkers)
	}
	
	if engine.config.NovaWorkers != 2 {
		t.Errorf("Expected NovaWorkers=2, got %d", engine.config.NovaWorkers)
	}
	
	if engine.config.EnableLoadBalancing {
		t.Error("Expected EnableLoadBalancing=false")
	}
	
	if engine.config.EnablePriorityQueue {
		t.Error("Expected EnablePriorityQueue=false")
	}
	
	if engine.config.EnableMemoryPooling {
		t.Error("Expected EnableMemoryPooling=false")
	}
	
	if engine.config.EnableMetrics {
		t.Error("Expected EnableMetrics=false")
	}
	
	// Load balancer should be nil when disabled
	if engine.loadBalancer != nil {
		t.Error("Load balancer should be nil when disabled")
	}
	
	t.Log("✅ Custom parallel verification config applied correctly")
}

func TestConcurrentEngineAccess(t *testing.T) {
	config := DefaultParallelVerificationConfig()
	engine := NewParallelVerificationEngine(config)
	
	// Test concurrent access to isRunning
	done := make(chan bool, 10)
	
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				engine.isRunning()
				engine.GetMetrics()
			}
			done <- true
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
	
	t.Log("✅ Concurrent engine access working correctly")
} 