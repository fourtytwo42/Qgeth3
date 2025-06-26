// Copyright 2025 QPoW Hardness Test Suite
// Common test framework for security and hardness testing

package common

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus/qmpow"
)

// TestResult represents the outcome of a hardness test
type TestResult struct {
	TestID     string        `json:"test_id"`
	Name       string        `json:"name"`
	Category   string        `json:"category"`
	Status     string        `json:"status"` // PASS, FAIL, SKIP, ERROR
	Duration   time.Duration `json:"duration"`
	AttackType string        `json:"attack_type"`
	Expected   string        `json:"expected"`
	Actual     string        `json:"actual"`
	Message    string        `json:"message"`
	Timestamp  time.Time     `json:"timestamp"`
}

// HardnessTest interface for all security tests
type HardnessTest interface {
	GetID() string
	GetName() string
	GetCategory() string
	GetAttackVector() string
	GetExpectedOutcome() string
	Run(t *testing.T) TestResult
}

// BaseHardnessTest provides common functionality
type BaseHardnessTest struct {
	ID              string
	Name            string
	Category        string
	AttackVector    string
	ExpectedOutcome string
}

func (b *BaseHardnessTest) GetID() string              { return b.ID }
func (b *BaseHardnessTest) GetName() string            { return b.Name }
func (b *BaseHardnessTest) GetCategory() string        { return b.Category }
func (b *BaseHardnessTest) GetAttackVector() string    { return b.AttackVector }
func (b *BaseHardnessTest) GetExpectedOutcome() string { return b.ExpectedOutcome }

// Golden reference values for testing
var GoldenValues = struct {
	Seed        string
	Outcomes    []uint16
	GateHash    string
	OutcomeRoot string
	ProofRoot   string
	QNonce      uint64
	Difficulty  uint64
}{
	Seed:        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	Outcomes:    []uint16{0x1234, 0x5678, 0x9abc, 0xdef0, 0x2468, 0xace1, 0x3579, 0xbdf0},
	GateHash:    "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
	OutcomeRoot: "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
	ProofRoot:   "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
	QNonce:      0x123456789abcdef0,
	Difficulty:  1000,
}

// Test helper functions
func AssertEqual(t *testing.T, expected, actual interface{}, message string) bool {
	if expected != actual {
		t.Errorf("%s: expected %v, got %v", message, expected, actual)
		return false
	}
	return true
}

func AssertNotEqual(t *testing.T, unexpected, actual interface{}, message string) bool {
	if unexpected == actual {
		t.Errorf("%s: expected %v to be different from %v", message, actual, unexpected)
		return false
	}
	return true
}

func AssertTrue(t *testing.T, condition bool, message string) bool {
	if !condition {
		t.Errorf("%s: expected true, got false", message)
		return false
	}
	return true
}

func AssertFalse(t *testing.T, condition bool, message string) bool {
	if condition {
		t.Errorf("%s: expected false, got true", message)
		return false
	}
	return true
}

func AssertNoError(t *testing.T, err error, message string) bool {
	if err != nil {
		t.Errorf("%s: unexpected error: %v", message, err)
		return false
	}
	return true
}

func AssertError(t *testing.T, err error, message string) bool {
	if err == nil {
		t.Errorf("%s: expected error, got nil", message)
		return false
	}
	return true
}

func AssertBytesEqual(t *testing.T, expected, actual []byte, message string) bool {
	if len(expected) != len(actual) {
		t.Errorf("%s: length mismatch - expected %d bytes, got %d bytes", message, len(expected), len(actual))
		return false
	}

	for i := range expected {
		if expected[i] != actual[i] {
			t.Errorf("%s: byte mismatch at index %d - expected 0x%02x, got 0x%02x", message, i, expected[i], actual[i])
			return false
		}
	}
	return true
}

func AssertHashEqual(t *testing.T, expectedHex string, actual common.Hash, message string) bool {
	expected := common.HexToHash(expectedHex)
	if expected != actual {
		t.Errorf("%s: hash mismatch - expected %s, got %s", message, expected.Hex(), actual.Hex())
		return false
	}
	return true
}

// CreateTestSeed creates a deterministic seed for testing
func CreateTestSeed(suffix string) []byte {
	data := GoldenValues.Seed + suffix
	hash := sha256.Sum256([]byte(data))
	return hash[:]
}

// CreateTestOutcomes creates deterministic outcomes for testing
func CreateTestOutcomes(count int, seed []byte) []uint16 {
	outcomes := make([]uint16, count)
	hasher := sha256.New()
	hasher.Write(seed)

	for i := 0; i < count; i++ {
		hasher.Write([]byte(fmt.Sprintf("outcome_%d", i)))
		hash := hasher.Sum(nil)
		outcomes[i] = uint16(hash[0])<<8 | uint16(hash[1])
	}

	return outcomes
}

// CreateMaliciousInput creates intentionally malicious input for attack testing
func CreateMaliciousInput(inputType string) []byte {
	switch inputType {
	case "all_zeros":
		return make([]byte, 32)
	case "all_ones":
		data := make([]byte, 32)
		for i := range data {
			data[i] = 0xFF
		}
		return data
	case "alternating":
		data := make([]byte, 32)
		for i := range data {
			if i%2 == 0 {
				data[i] = 0xAA
			} else {
				data[i] = 0x55
			}
		}
		return data
	case "single_bit":
		data := make([]byte, 32)
		data[0] = 0x01
		return data
	default:
		return CreateTestSeed("malicious_" + inputType)
	}
}

// RunHardnessTest executes a hardness test and returns the result
func RunHardnessTest(t *testing.T, test HardnessTest) TestResult {
	startTime := time.Now()

	t.Logf("🔍 Running %s: %s", test.GetID(), test.GetName())
	t.Logf("   Attack Vector: %s", test.GetAttackVector())
	t.Logf("   Expected: %s", test.GetExpectedOutcome())

	result := test.Run(t)
	result.Duration = time.Since(startTime)
	result.Timestamp = time.Now()

	switch result.Status {
	case "PASS":
		t.Logf("✅ %s PASSED in %v", test.GetID(), result.Duration)
	case "FAIL":
		t.Errorf("❌ %s FAILED: %s", test.GetID(), result.Message)
	case "SKIP":
		t.Logf("⚠️ %s SKIPPED: %s", test.GetID(), result.Message)
	case "ERROR":
		t.Errorf("💥 %s ERROR: %s", test.GetID(), result.Message)
	}

	return result
}

// ValidateQuantumHeader validates quantum header fields
func ValidateQuantumHeader(t *testing.T, header *qmpow.QuantumHeader) bool {
	valid := true

	if header.QBits == nil {
		t.Error("QBits field is nil")
		valid = false
	} else if *header.QBits != 16 {
		t.Errorf("Invalid QBits: expected 16, got %d", *header.QBits)
		valid = false
	}

	if header.TCount == nil {
		t.Error("TCount field is nil")
		valid = false
	} else if *header.TCount != 20 {
		t.Errorf("Invalid TCount: expected 20, got %d", *header.TCount)
		valid = false
	}

	if header.LNet == nil {
		t.Error("LNet field is nil")
		valid = false
	} else if *header.LNet != 128 {
		t.Errorf("Invalid LNet: expected 128, got %d", *header.LNet)
		valid = false
	}

	if header.OutcomeRoot == nil {
		t.Error("OutcomeRoot field is nil")
		valid = false
	}

	if header.GateHash == nil {
		t.Error("GateHash field is nil")
		valid = false
	}

	if header.ProofRoot == nil {
		t.Error("ProofRoot field is nil")
		valid = false
	}

	if len(header.BranchNibbles) != 64 {
		t.Errorf("Invalid BranchNibbles length: expected 64, got %d", len(header.BranchNibbles))
		valid = false
	}

	if len(header.ExtraNonce32) != 32 {
		t.Errorf("Invalid ExtraNonce32 length: expected 32, got %d", len(header.ExtraNonce32))
		valid = false
	}

	return valid
}

// HexToBytes converts hex string to bytes with error handling
func HexToBytes(hexStr string) ([]byte, error) {
	if len(hexStr)%2 != 0 {
		hexStr = "0" + hexStr
	}
	return hex.DecodeString(hexStr)
}

// BytesToHex converts bytes to hex string
func BytesToHex(data []byte) string {
	return hex.EncodeToString(data)
}

// MeasureExecutionTime measures the execution time of a function
func MeasureExecutionTime(fn func()) time.Duration {
	start := time.Now()
	fn()
	return time.Since(start)
}

// CreateTestMiningInput creates a test mining input
func CreateTestMiningInput() *qmpow.MiningInput {
	parentHash := common.HexToHash(GoldenValues.OutcomeRoot)
	txRoot := common.HexToHash(GoldenValues.GateHash)
	extraNonce, _ := HexToBytes("1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")

	return &qmpow.MiningInput{
		ParentHash:   parentHash,
		TxRoot:       txRoot,
		ExtraNonce32: extraNonce,
		QNonce64:     GoldenValues.QNonce,
		BlockHeight:  100,
		QBits:        16,
		TCount:       20,
		LNet:         128,
	}
}

// SimulateAttack simulates various attack scenarios
func SimulateAttack(attackType string, input interface{}) (interface{}, error) {
	switch attackType {
	case "bit_flip":
		// Flip a single bit in input
		if data, ok := input.([]byte); ok {
			modified := make([]byte, len(data))
			copy(modified, data)
			if len(modified) > 0 {
				modified[0] ^= 0x01 // Flip LSB
			}
			return modified, nil
		}
		return nil, fmt.Errorf("bit_flip attack requires []byte input")

	case "truncate":
		// Truncate input by removing last byte
		if data, ok := input.([]byte); ok {
			if len(data) > 1 {
				return data[:len(data)-1], nil
			}
			return []byte{}, nil
		}
		return nil, fmt.Errorf("truncate attack requires []byte input")

	case "extend":
		// Extend input with extra bytes
		if data, ok := input.([]byte); ok {
			extended := append(data, 0xFF, 0xFF)
			return extended, nil
		}
		return nil, fmt.Errorf("extend attack requires []byte input")

	case "zero_out":
		// Zero out the input
		if data, ok := input.([]byte); ok {
			return make([]byte, len(data)), nil
		}
		return nil, fmt.Errorf("zero_out attack requires []byte input")

	case "overflow":
		// Attempt integer overflow
		if val, ok := input.(uint64); ok {
			return val + 1, nil
		}
		return nil, fmt.Errorf("overflow attack requires uint64 input")

	default:
		return nil, fmt.Errorf("unknown attack type: %s", attackType)
	}
}
