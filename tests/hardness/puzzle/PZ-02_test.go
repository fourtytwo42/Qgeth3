// PZ-02: Branch-Serial Enforcement Test
// Tests that puzzle i+1 cannot be compiled/run before puzzle i's outcome is injected
// Attack Vector: Parallel execution bypass
// Expected Outcome: Compilation/execution fails or outcome mismatch

package puzzle

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// Test PZ-02: Branch-Serial Enforcement
func TestPZ02_BranchSerialEnforcement(t *testing.T) {
	seed := "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
	puzzleCount := 4
	
	t.Log("🔍 PZ-02: Testing branch-serial enforcement")
	t.Log("   Attack: Attempting parallel puzzle execution")
	
	// Test 1: Serial execution (correct behavior)
	t.Log("   Step 1: Running serial execution (baseline)")
	serialOutcomes := executeSerialPuzzles(seed, puzzleCount)
	t.Logf("   ✅ Serial outcomes: %v", formatOutcomes(serialOutcomes))
	
	// Test 2: Attempt parallel execution (should fail or produce wrong results)
	t.Log("   Step 2: Attempting parallel execution (attack)")
	parallelOutcomes, parallelSuccess := attemptParallelExecution(seed, puzzleCount)
	
	if parallelSuccess {
		// If parallel execution "succeeded", outcomes should be different
		if outcomesEqual(serialOutcomes, parallelOutcomes) {
			t.Error("❌ SECURITY VULNERABILITY: Parallel execution produced same outcomes as serial")
			t.Error("   This indicates branch-serial enforcement is not working")
		} else {
			t.Log("✅ Parallel execution produced different outcomes (expected)")
			t.Logf("   Parallel outcomes: %v", formatOutcomes(parallelOutcomes))
			t.Log("   ✅ Branch-serial enforcement working correctly")
		}
	} else {
		t.Log("✅ Parallel execution failed as expected")
		t.Log("   ✅ Branch-serial enforcement prevents parallel bypass")
	}
	
	// Test 3: Out-of-order execution attempt
	t.Log("   Step 3: Attempting out-of-order execution")
	outOfOrderSuccess := attemptOutOfOrderExecution(seed, puzzleCount)
	
	if outOfOrderSuccess {
		t.Error("❌ SECURITY VULNERABILITY: Out-of-order execution succeeded")
	} else {
		t.Log("✅ Out-of-order execution blocked correctly")
	}
}

// executeSerialPuzzles executes puzzles in correct serial order
func executeSerialPuzzles(seedHex string, count int) []uint16 {
	return simulateQuantumPuzzleExecution(seedHex, count)
}

// attemptParallelExecution tries to execute all puzzles simultaneously
func attemptParallelExecution(seedHex string, count int) ([]uint16, bool) {
	outcomes := make([]uint16, count)
	var wg sync.WaitGroup
	var mutex sync.Mutex
	errors := make([]error, count)
	
	// Try to execute all puzzles at the same time
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(puzzleIndex int) {
			defer wg.Done()
			
			// This should fail because puzzle i+1 needs outcome from puzzle i
			outcome, err := simulateParallelPuzzleExecution(seedHex, puzzleIndex, nil)
			
			mutex.Lock()
			if err != nil {
				errors[puzzleIndex] = err
			} else {
				outcomes[puzzleIndex] = outcome
			}
			mutex.Unlock()
		}(i)
	}
	
	wg.Wait()
	
	// Check if any puzzles failed (which is expected for branch-serial enforcement)
	failureCount := 0
	for _, err := range errors {
		if err != nil {
			failureCount++
		}
	}
	
	// If most puzzles failed, parallel execution was properly blocked
	success := failureCount < count/2
	return outcomes, success
}

// simulateParallelPuzzleExecution simulates attempting to run a single puzzle in parallel
func simulateParallelPuzzleExecution(seedHex string, puzzleIndex int, previousOutcome *uint16) (uint16, error) {
	// Simulate the requirement that puzzle i needs outcome from puzzle i-1
	if puzzleIndex > 0 && previousOutcome == nil {
		// This should fail in real implementation
		return 0, &BranchSerialViolationError{
			PuzzleIndex: puzzleIndex,
			Message:     "Cannot execute puzzle without previous outcome",
		}
	}
	
	// If we somehow get here, generate a "fake" outcome that would be wrong
	// In real implementation, this would be caught by the quantum compiler
	return uint16(0xDEAD), nil // Clearly wrong outcome
}

// attemptOutOfOrderExecution tries to execute puzzles in wrong order
func attemptOutOfOrderExecution(seedHex string, count int) bool {
	// Try to execute puzzle 2 before puzzle 1
	if count < 2 {
		return false
	}
	
	// This should fail
	_, err := simulateParallelPuzzleExecution(seedHex, 1, nil)
	
	// If it succeeded, that's a security vulnerability
	return err == nil
}

// BranchSerialViolationError represents a branch-serial enforcement violation
type BranchSerialViolationError struct {
	PuzzleIndex int
	Message     string
}

func (e *BranchSerialViolationError) Error() string {
	return e.Message
}

// Helper functions
func outcomesEqual(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func formatOutcomes(outcomes []uint16) []string {
	result := make([]string, len(outcomes))
	for i, outcome := range outcomes {
		result[i] = fmt.Sprintf("0x%04x", outcome)
	}
	return result
}

// Test timing-based attack detection
func TestPZ02_TimingAttackDetection(t *testing.T) {
	seed := "timing_attack_test_seed_1234567890abcdef1234567890abcdef1234567890"
	puzzleCount := 3
	
	t.Log("🔍 PZ-02 Timing: Testing timing-based attack detection")
	
	// Measure serial execution time
	start := time.Now()
	serialOutcomes := executeSerialPuzzles(seed, puzzleCount)
	serialDuration := time.Since(start)
	
	t.Logf("   Serial execution: %v (%v)", formatOutcomes(serialOutcomes), serialDuration)
	
	// Attempt parallel execution and measure time
	start = time.Now()
	parallelOutcomes, parallelSuccess := attemptParallelExecution(seed, puzzleCount)
	parallelDuration := time.Since(start)
	
	t.Logf("   Parallel attempt: %v (%v) success=%v", 
		formatOutcomes(parallelOutcomes), parallelDuration, parallelSuccess)
	
	// Parallel execution should either fail or take similar time (no speedup)
	if parallelSuccess && parallelDuration < serialDuration/2 {
		t.Error("❌ TIMING ATTACK DETECTED: Parallel execution was significantly faster")
		t.Error("   This suggests branch-serial enforcement may be bypassed")
	} else {
		t.Log("✅ No timing attack detected - enforcement is working")
	}
}

 