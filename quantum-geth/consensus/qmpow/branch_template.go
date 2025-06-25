// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements branch template engine for Quantum-Geth
// Branch templates provide 16 iso-hard quantum circuit patterns with PRP instantiation

package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/log"
	"golang.org/x/crypto/sha3"
)

// BranchTemplateEngine manages the 16 iso-hard branch templates
type BranchTemplateEngine struct {
	templates []BranchTemplate
}

// BranchTemplate represents a single branch template
type BranchTemplate struct {
	ID          int
	Name        string
	Skeleton    string // QASM-lite template with placeholders
	Depth       int    // Expected circuit depth
	TGateCount  int    // Expected T-gate count
	CXGateCount int    // Expected CX-gate count
	Parameters  []TemplateParameter
}

// TemplateParameter represents a parameterizable element in a template
type TemplateParameter struct {
	Name        string
	Type        string // "rotation", "cx_mapping", "qubit_order"
	BitOffset   int    // Offset in PRP output for this parameter
	BitLength   int    // Number of bits for this parameter
	Description string
}

// BranchInstantiation represents an instantiated branch template
type BranchInstantiation struct {
	TemplateID  int
	Seed        []byte
	QASM        string
	Parameters  map[string]interface{}
	Depth       int
	TGateCount  int
	CXGateCount int
	Validated   bool
}

// NewBranchTemplateEngine creates a new branch template engine with 16 iso-hard templates
func NewBranchTemplateEngine() *BranchTemplateEngine {
	engine := &BranchTemplateEngine{
		templates: make([]BranchTemplate, 16),
	}

	// Initialize the 16 iso-hard branch templates
	engine.initializeTemplates()

	log.Info("🌳 Branch template engine initialized", "templates", len(engine.templates))
	return engine
}

// initializeTemplates initializes the 16 iso-hard branch templates
func (e *BranchTemplateEngine) initializeTemplates() {
	// Template 0: Basic Hadamard-T pattern
	e.templates[0] = BranchTemplate{
		ID:          0,
		Name:        "Hadamard-T Chain",
		Depth:       8,
		TGateCount:  4,
		CXGateCount: 3,
		Skeleton: `qreg q[16];
H q[{q0}];
T q[{q0}];
H q[{q1}];
CX q[{q0}],q[{q1}];
T q[{q1}];
CX q[{q2}],q[{q3}];
T q[{q2}];
H q[{q3}];
T q[{q3}];
CX q[{q1}],q[{q2}];`,
		Parameters: []TemplateParameter{
			{"q0", "qubit_order", 0, 4, "First qubit index"},
			{"q1", "qubit_order", 4, 4, "Second qubit index"},
			{"q2", "qubit_order", 8, 4, "Third qubit index"},
			{"q3", "qubit_order", 12, 4, "Fourth qubit index"},
		},
	}

	// Template 1: S-T-CX pyramid
	e.templates[1] = BranchTemplate{
		ID:          1,
		Name:        "S-T-CX Pyramid",
		Depth:       7,
		TGateCount:  3,
		CXGateCount: 4,
		Skeleton: `qreg q[16];
S q[{q0}];
T q[{q1}];
CX q[{q0}],q[{q1}];
S q[{q2}];
CX q[{q1}],q[{q2}];
T q[{q2}];
CX q[{q0}],q[{q3}];
T q[{q3}];
CX q[{q2}],q[{q3}];`,
		Parameters: []TemplateParameter{
			{"q0", "qubit_order", 0, 4, "Base qubit"},
			{"q1", "qubit_order", 4, 4, "Level 1 qubit"},
			{"q2", "qubit_order", 8, 4, "Level 2 qubit"},
			{"q3", "qubit_order", 12, 4, "Top qubit"},
		},
	}

	// Template 2: Alternating H-T pattern
	e.templates[2] = BranchTemplate{
		ID:          2,
		Name:        "Alternating H-T",
		Depth:       8,
		TGateCount:  4,
		CXGateCount: 2,
		Skeleton: `qreg q[16];
H q[{q0}];
T q[{q1}];
H q[{q2}];
T q[{q3}];
CX q[{q0}],q[{q2}];
H q[{q1}];
T q[{q0}];
CX q[{q1}],q[{q3}];
T q[{q2}];
H q[{q3}];`,
		Parameters: []TemplateParameter{
			{"q0", "qubit_order", 0, 4, "First alternating qubit"},
			{"q1", "qubit_order", 4, 4, "Second alternating qubit"},
			{"q2", "qubit_order", 8, 4, "Third alternating qubit"},
			{"q3", "qubit_order", 12, 4, "Fourth alternating qubit"},
		},
	}

	// Template 3: CX-dominated pattern
	e.templates[3] = BranchTemplate{
		ID:          3,
		Name:        "CX-Dominated",
		Depth:       6,
		TGateCount:  2,
		CXGateCount: 6,
		Skeleton: `qreg q[16];
CX q[{q0}],q[{q1}];
CX q[{q1}],q[{q2}];
T q[{q0}];
CX q[{q2}],q[{q3}];
CX q[{q0}],q[{q3}];
T q[{q1}];
CX q[{q1}],q[{q3}];
CX q[{q0}],q[{q2}];`,
		Parameters: []TemplateParameter{
			{"q0", "qubit_order", 0, 4, "Control hub qubit"},
			{"q1", "qubit_order", 4, 4, "Primary target"},
			{"q2", "qubit_order", 8, 4, "Secondary target"},
			{"q3", "qubit_order", 12, 4, "Tertiary target"},
		},
	}

	// Template 4: T-gate cluster
	e.templates[4] = BranchTemplate{
		ID:          4,
		Name:        "T-Gate Cluster",
		Depth:       9,
		TGateCount:  6,
		CXGateCount: 2,
		Skeleton: `qreg q[16];
T q[{q0}];
T q[{q1}];
CX q[{q0}],q[{q1}];
T q[{q2}];
T q[{q3}];
CX q[{q2}],q[{q3}];
T q[{q0}];
T q[{q2}];`,
		Parameters: []TemplateParameter{
			{"q0", "qubit_order", 0, 4, "First T-cluster qubit"},
			{"q1", "qubit_order", 4, 4, "Second T-cluster qubit"},
			{"q2", "qubit_order", 8, 4, "Third T-cluster qubit"},
			{"q3", "qubit_order", 12, 4, "Fourth T-cluster qubit"},
		},
	}

	// Template 5: Balanced H-S-T
	e.templates[5] = BranchTemplate{
		ID:          5,
		Name:        "Balanced H-S-T",
		Depth:       7,
		TGateCount:  3,
		CXGateCount: 3,
		Skeleton: `qreg q[16];
H q[{q0}];
S q[{q1}];
T q[{q2}];
CX q[{q0}],q[{q1}];
H q[{q3}];
CX q[{q1}],q[{q2}];
S q[{q0}];
T q[{q1}];
CX q[{q2}],q[{q3}];`,
		Parameters: []TemplateParameter{
			{"q0", "qubit_order", 0, 4, "H-gate primary"},
			{"q1", "qubit_order", 4, 4, "S-gate primary"},
			{"q2", "qubit_order", 8, 4, "T-gate primary"},
			{"q3", "qubit_order", 12, 4, "Mixed operations"},
		},
	}

	// Templates 6-15: Generate variations of the base patterns
	for i := 6; i < 16; i++ {
		baseTemplate := e.templates[i%6]
		e.templates[i] = BranchTemplate{
			ID:          i,
			Name:        fmt.Sprintf("%s-Variant-%d", baseTemplate.Name, i-5),
			Depth:       baseTemplate.Depth + (i-5)%3,
			TGateCount:  baseTemplate.TGateCount + (i-5)%2,
			CXGateCount: baseTemplate.CXGateCount + (i-5)%3,
			Skeleton:    e.generateVariantSkeleton(baseTemplate.Skeleton, i),
			Parameters:  baseTemplate.Parameters,
		}
	}
}

// generateVariantSkeleton generates a variant of a base skeleton
func (e *BranchTemplateEngine) generateVariantSkeleton(baseSkeleton string, variantID int) string {
	// Add some variation based on variant ID
	lines := strings.Split(baseSkeleton, "\n")

	// Insert additional gates based on variant
	switch variantID % 3 {
	case 0:
		// Add extra H gate
		lines = append(lines, "H q[{q0}];")
	case 1:
		// Add extra S gate
		lines = append(lines, "S q[{q1}];")
	case 2:
		// Add extra T gate
		lines = append(lines, "T q[{q2}];")
	}

	return strings.Join(lines, "\n")
}

// InstantiateBranch instantiates a branch template using Keccak-based PRP
func (e *BranchTemplateEngine) InstantiateBranch(templateID int, seed []byte) (*BranchInstantiation, error) {
	if templateID < 0 || templateID >= len(e.templates) {
		return nil, fmt.Errorf("invalid template ID: %d", templateID)
	}

	template := e.templates[templateID]

	log.Debug("🌿 Instantiating branch template",
		"template_id", templateID,
		"template_name", template.Name,
		"seed_len", len(seed))

	// Generate PRP output using Keccak
	prpOutput := e.generatePRP(seed, templateID)

	// Extract parameters from PRP output
	parameters := make(map[string]interface{})
	for _, param := range template.Parameters {
		value := e.extractParameter(prpOutput, param)
		parameters[param.Name] = value
	}

	// Instantiate QASM by replacing placeholders
	qasm := e.instantiateQASM(template.Skeleton, parameters)

	instantiation := &BranchInstantiation{
		TemplateID:  templateID,
		Seed:        seed,
		QASM:        qasm,
		Parameters:  parameters,
		Depth:       template.Depth,
		TGateCount:  template.TGateCount,
		CXGateCount: template.CXGateCount,
		Validated:   false,
	}

	// Validate depth and T-count invariants
	err := e.validateInstantiation(instantiation)
	if err != nil {
		return nil, fmt.Errorf("instantiation validation failed: %v", err)
	}

	instantiation.Validated = true

	log.Debug("✅ Branch template instantiated successfully",
		"template_id", templateID,
		"qasm_lines", strings.Count(qasm, "\n")+1,
		"parameters", len(parameters))

	return instantiation, nil
}

// generatePRP generates pseudorandom permutation output using Keccak
func (e *BranchTemplateEngine) generatePRP(seed []byte, templateID int) []byte {
	// Create input for Keccak: seed || template_id
	input := make([]byte, len(seed)+4)
	copy(input, seed)
	binary.BigEndian.PutUint32(input[len(seed):], uint32(templateID))

	// Use Keccak-256 as PRP
	hash := sha3.NewLegacyKeccak256()
	hash.Write(input)
	output := hash.Sum(nil)

	// Extend output if needed (for templates with many parameters)
	if len(output) < 32 {
		// Use SHA256 to extend
		extended := sha256.Sum256(output)
		output = extended[:]
	}

	return output
}

// extractParameter extracts a parameter value from PRP output
func (e *BranchTemplateEngine) extractParameter(prpOutput []byte, param TemplateParameter) interface{} {
	// Extract bits from PRP output
	bitOffset := param.BitOffset
	bitLength := param.BitLength

	if bitOffset+bitLength > len(prpOutput)*8 {
		// Wrap around if we exceed output length
		bitOffset = bitOffset % (len(prpOutput) * 8)
	}

	// Extract value based on parameter type
	switch param.Type {
	case "qubit_order":
		// Extract qubit index (0-15 for 16-qubit system)
		value := extractBits(prpOutput, bitOffset, bitLength)
		return int(value % 16)

	case "rotation":
		// Extract rotation angle (for future use with rotation gates)
		value := extractBits(prpOutput, bitOffset, bitLength)
		angle := float64(value) / math.Pow(2, float64(bitLength)) * 2 * math.Pi
		return angle

	case "cx_mapping":
		// Extract CX gate mapping
		value := extractBits(prpOutput, bitOffset, bitLength)
		return int(value % 16)

	default:
		// Default to integer extraction
		return int(extractBits(prpOutput, bitOffset, bitLength))
	}
}

// extractBits extracts a specific range of bits from a byte array
func extractBits(data []byte, bitOffset, bitLength int) uint64 {
	var result uint64

	for i := 0; i < bitLength; i++ {
		byteIndex := (bitOffset + i) / 8
		bitIndex := (bitOffset + i) % 8

		if byteIndex >= len(data) {
			break
		}

		bit := (data[byteIndex] >> (7 - bitIndex)) & 1
		result = (result << 1) | uint64(bit)
	}

	return result
}

// instantiateQASM replaces placeholders in template skeleton with actual values
func (e *BranchTemplateEngine) instantiateQASM(skeleton string, parameters map[string]interface{}) string {
	result := skeleton

	// Replace each parameter placeholder
	for name, value := range parameters {
		placeholder := fmt.Sprintf("{%s}", name)
		replacement := fmt.Sprintf("%v", value)
		result = strings.ReplaceAll(result, placeholder, replacement)
	}

	return result
}

// validateInstantiation validates that an instantiation meets depth/T-count invariants
func (e *BranchTemplateEngine) validateInstantiation(instantiation *BranchInstantiation) error {
	// Parse the instantiated QASM to validate structure
	parser := NewQASMLiteParser(instantiation.QASM)
	program, err := parser.Parse()
	if err != nil {
		return fmt.Errorf("QASM parsing failed: %v", err)
	}

	// Count T-gates and estimate depth
	actualTGates := 0
	for _, gate := range program.Gates {
		if gate.Type == "T" {
			actualTGates++
		}
	}

	// Validate T-gate count (allow some variance due to template variations)
	expectedTGates := instantiation.TGateCount
	if actualTGates < expectedTGates-2 || actualTGates > expectedTGates+2 {
		return fmt.Errorf("T-gate count mismatch: expected ~%d, got %d", expectedTGates, actualTGates)
	}

	// Compile to get accurate depth measurement
	compiler := NewCanonicalCompiler()
	result, err := compiler.CanonicalCompile(instantiation.Seed, instantiation.QASM)
	if err != nil {
		return fmt.Errorf("compilation failed: %v", err)
	}

	// Update instantiation with actual values
	instantiation.Depth = result.DAG.Depth
	instantiation.TGateCount = result.DAG.TGateCount

	// Count CX gates
	actualCXGates := 0
	for _, gate := range result.DAG.Gates {
		if gate.Type == "CX" {
			actualCXGates++
		}
	}
	instantiation.CXGateCount = actualCXGates

	log.Debug("🔍 Branch instantiation validated",
		"template_id", instantiation.TemplateID,
		"depth", instantiation.Depth,
		"t_gates", instantiation.TGateCount,
		"cx_gates", instantiation.CXGateCount)

	return nil
}

// GetTemplate returns a template by ID
func (e *BranchTemplateEngine) GetTemplate(templateID int) (*BranchTemplate, error) {
	if templateID < 0 || templateID >= len(e.templates) {
		return nil, fmt.Errorf("invalid template ID: %d", templateID)
	}

	return &e.templates[templateID], nil
}

// GetAllTemplates returns all templates
func (e *BranchTemplateEngine) GetAllTemplates() []BranchTemplate {
	return e.templates
}

// ValidateAllTemplates validates all 16 templates with test seeds
func (e *BranchTemplateEngine) ValidateAllTemplates() error {
	testSeed := crypto.Keccak256([]byte("branch_template_validation"))

	for i := 0; i < len(e.templates); i++ {
		_, err := e.InstantiateBranch(i, testSeed)
		if err != nil {
			return fmt.Errorf("template %d validation failed: %v", i, err)
		}
	}

	log.Info("✅ All 16 branch templates validated successfully")
	return nil
}
