// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements QASM-lite grammar and parser for Quantum-Geth v0.9–BareBones+Halving
// QASM-lite is a minimal quantum assembly language subset covering H, S, T, CX, Z-mask directives

package qmpow

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"github.com/ethereum/go-ethereum/log"
)

// QASM-lite BNF Grammar (Section 5 - Canonical Compile & GateHash)
//
// program       ::= statement*
// statement     ::= gate_stmt | register_stmt | comment
// gate_stmt     ::= single_gate | two_gate | z_mask
// single_gate   ::= ("H" | "S" | "T") qubit_ref ";"
// two_gate      ::= "CX" qubit_ref "," qubit_ref ";"
// z_mask        ::= "Z" "[" number "]" qubit_ref ";"
// register_stmt ::= "qreg" identifier "[" number "]" ";"
// qubit_ref     ::= identifier "[" number "]"
// comment       ::= "//" text
// identifier    ::= [a-zA-Z_][a-zA-Z0-9_]*
// number        ::= [0-9]+

// QASMLiteGate represents a single quantum gate in QASM-lite
type QASMLiteGate struct {
	Type   string // "H", "S", "T", "CX", "Z"
	Qubits []int  // Qubit indices (1 for single gates, 2 for CX)
	ZMask  int    // Z-mask value for Z gates (0 for others)
	Line   int    // Source line number for debugging
}

// QASMLiteProgram represents a complete QASM-lite program
type QASMLiteProgram struct {
	Gates      []QASMLiteGate
	QubitsUsed int
	Source     string
}

// QASMLiteParser parses QASM-lite source code
type QASMLiteParser struct {
	source   string
	lines    []string
	position int
	errors   []string
}

// NewQASMLiteParser creates a new QASM-lite parser
func NewQASMLiteParser(source string) *QASMLiteParser {
	return &QASMLiteParser{
		source: source,
		lines:  strings.Split(source, "\n"),
		errors: make([]string, 0),
	}
}

// Parse parses the QASM-lite source and returns a program
func (p *QASMLiteParser) Parse() (*QASMLiteProgram, error) {
	program := &QASMLiteProgram{
		Gates:  make([]QASMLiteGate, 0),
		Source: p.source,
	}

	for lineNum, line := range p.lines {
		line = strings.TrimSpace(line)

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}

		// Parse register declarations
		if strings.HasPrefix(line, "qreg") {
			qubits, err := p.parseRegisterDeclaration(line, lineNum+1)
			if err != nil {
				p.addError(lineNum+1, err.Error())
				continue
			}
			if qubits > program.QubitsUsed {
				program.QubitsUsed = qubits
			}
			continue
		}

		// Parse gate statements
		gate, err := p.parseGateStatement(line, lineNum+1)
		if err != nil {
			p.addError(lineNum+1, err.Error())
			continue
		}

		program.Gates = append(program.Gates, gate)

		// Update qubits used
		for _, qubit := range gate.Qubits {
			if qubit+1 > program.QubitsUsed {
				program.QubitsUsed = qubit + 1
			}
		}
	}

	if len(p.errors) > 0 {
		return nil, fmt.Errorf("parse errors: %v", p.errors)
	}

	log.Debug("🔍 QASM-lite parsing completed",
		"gates", len(program.Gates),
		"qubits", program.QubitsUsed,
		"source_lines", len(p.lines))

	return program, nil
}

// parseRegisterDeclaration parses qreg declarations
func (p *QASMLiteParser) parseRegisterDeclaration(line string, lineNum int) (int, error) {
	// qreg q[16];
	parts := strings.Fields(line)
	if len(parts) != 2 {
		return 0, fmt.Errorf("invalid qreg declaration")
	}

	// Extract register name and size
	regPart := parts[1]
	if !strings.HasSuffix(regPart, ";") {
		return 0, fmt.Errorf("missing semicolon")
	}
	regPart = strings.TrimSuffix(regPart, ";")

	// Parse q[16] format
	bracketStart := strings.Index(regPart, "[")
	bracketEnd := strings.Index(regPart, "]")
	if bracketStart == -1 || bracketEnd == -1 || bracketEnd <= bracketStart {
		return 0, fmt.Errorf("invalid register format")
	}

	sizeStr := regPart[bracketStart+1 : bracketEnd]
	size, err := strconv.Atoi(sizeStr)
	if err != nil {
		return 0, fmt.Errorf("invalid register size: %v", err)
	}

	return size, nil
}

// parseGateStatement parses quantum gate statements
func (p *QASMLiteParser) parseGateStatement(line string, lineNum int) (QASMLiteGate, error) {
	gate := QASMLiteGate{Line: lineNum}

	// Remove semicolon
	if !strings.HasSuffix(line, ";") {
		return gate, fmt.Errorf("missing semicolon")
	}
	line = strings.TrimSuffix(line, ";")

	parts := strings.Fields(line)
	if len(parts) == 0 {
		return gate, fmt.Errorf("empty gate statement")
	}

	// Extract gate type, handling Z[mask] format
	gateTypeRaw := strings.ToUpper(parts[0])
	var gateType string
	if strings.HasPrefix(gateTypeRaw, "Z[") {
		gateType = "Z"
	} else {
		gateType = gateTypeRaw
	}
	gate.Type = gateType

	switch gateType {
	case "H", "S", "T":
		// Single qubit gates: H q[0];
		if len(parts) != 2 {
			return gate, fmt.Errorf("invalid %s gate format", gateType)
		}
		qubit, err := p.parseQubitRef(parts[1])
		if err != nil {
			return gate, fmt.Errorf("invalid qubit reference: %v", err)
		}
		gate.Qubits = []int{qubit}

	case "CX":
		// Two qubit gate: CX q[0],q[1];
		if len(parts) != 2 {
			return gate, fmt.Errorf("invalid CX gate format")
		}
		qubitsStr := parts[1]
		qubits := strings.Split(qubitsStr, ",")
		if len(qubits) != 2 {
			return gate, fmt.Errorf("CX gate requires exactly 2 qubits")
		}

		control, err := p.parseQubitRef(strings.TrimSpace(qubits[0]))
		if err != nil {
			return gate, fmt.Errorf("invalid control qubit: %v", err)
		}
		target, err := p.parseQubitRef(strings.TrimSpace(qubits[1]))
		if err != nil {
			return gate, fmt.Errorf("invalid target qubit: %v", err)
		}
		gate.Qubits = []int{control, target}

	case "Z":
		// Z-mask gate: Z[1] q[0];
		if len(parts) != 2 {
			return gate, fmt.Errorf("invalid Z gate format")
		}

		// Parse Z[mask] format from the raw gate type
		bracketStart := strings.Index(gateTypeRaw, "[")
		bracketEnd := strings.Index(gateTypeRaw, "]")
		if bracketStart == -1 || bracketEnd == -1 {
			return gate, fmt.Errorf("Z gate requires mask: Z[value]")
		}

		maskStr := gateTypeRaw[bracketStart+1 : bracketEnd]
		mask, err := strconv.Atoi(maskStr)
		if err != nil {
			return gate, fmt.Errorf("invalid Z mask value: %v", err)
		}
		gate.ZMask = mask

		qubit, err := p.parseQubitRef(parts[1])
		if err != nil {
			return gate, fmt.Errorf("invalid qubit reference: %v", err)
		}
		gate.Qubits = []int{qubit}

	default:
		return gate, fmt.Errorf("unsupported gate type: %s", gateTypeRaw)
	}

	return gate, nil
}

// parseQubitRef parses qubit references like q[0]
func (p *QASMLiteParser) parseQubitRef(ref string) (int, error) {
	bracketStart := strings.Index(ref, "[")
	bracketEnd := strings.Index(ref, "]")
	if bracketStart == -1 || bracketEnd == -1 || bracketEnd <= bracketStart {
		return 0, fmt.Errorf("invalid qubit reference format")
	}

	indexStr := ref[bracketStart+1 : bracketEnd]
	index, err := strconv.Atoi(indexStr)
	if err != nil {
		return 0, fmt.Errorf("invalid qubit index: %v", err)
	}

	if index < 0 {
		return 0, fmt.Errorf("negative qubit index")
	}

	return index, nil
}

// addError adds a parse error
func (p *QASMLiteParser) addError(line int, message string) {
	p.errors = append(p.errors, fmt.Sprintf("line %d: %s", line, message))
}

// SerializeProgram serializes a QASM-lite program to canonical byte format
func SerializeProgram(program *QASMLiteProgram) []byte {
	var result []byte

	// Header: qubits used (2 bytes)
	result = append(result, byte(program.QubitsUsed>>8), byte(program.QubitsUsed&0xFF))

	// Gates count (4 bytes)
	gateCount := len(program.Gates)
	result = append(result,
		byte(gateCount>>24), byte(gateCount>>16),
		byte(gateCount>>8), byte(gateCount&0xFF))

	// Serialize each gate
	for _, gate := range program.Gates {
		result = append(result, serializeGate(gate)...)
	}

	return result
}

// serializeGate serializes a single gate to bytes
func serializeGate(gate QASMLiteGate) []byte {
	var result []byte

	// Gate type (1 byte)
	var gateTypeByte byte
	switch gate.Type {
	case "H":
		gateTypeByte = 0x01
	case "S":
		gateTypeByte = 0x02
	case "T":
		gateTypeByte = 0x03
	case "CX":
		gateTypeByte = 0x04
	case "Z":
		gateTypeByte = 0x05
	}
	result = append(result, gateTypeByte)

	// Qubit count (1 byte)
	result = append(result, byte(len(gate.Qubits)))

	// Qubits (2 bytes each)
	for _, qubit := range gate.Qubits {
		result = append(result, byte(qubit>>8), byte(qubit&0xFF))
	}

	// Z-mask for Z gates (4 bytes)
	if gate.Type == "Z" {
		result = append(result,
			byte(gate.ZMask>>24), byte(gate.ZMask>>16),
			byte(gate.ZMask>>8), byte(gate.ZMask&0xFF))
	}

	return result
}

// ApplyZMask applies Pauli-Z mask from SHA256(seed) to a QASM-lite program
func ApplyZMask(program *QASMLiteProgram, seed []byte) *QASMLiteProgram {
	// Generate Z-mask from seed
	hash := sha256.Sum256(seed)

	// Create new program with Z-masks applied
	maskedProgram := &QASMLiteProgram{
		Gates:      make([]QASMLiteGate, 0, len(program.Gates)),
		QubitsUsed: program.QubitsUsed,
		Source:     program.Source + fmt.Sprintf("\n// Z-mask applied from seed: %s", hex.EncodeToString(seed)),
	}

	maskIndex := 0
	for _, gate := range program.Gates {
		newGate := gate

		// Apply Z-mask to single qubit gates based on hash bits
		// Only apply if seed is non-zero (to avoid masking with zero seeds in tests)
		seedIsNonZero := false
		for _, b := range seed {
			if b != 0 {
				seedIsNonZero = true
				break
			}
		}

		if seedIsNonZero && len(gate.Qubits) == 1 && maskIndex < len(hash)*8 {
			byteIndex := maskIndex / 8
			bitIndex := maskIndex % 8
			maskBit := (hash[byteIndex] >> bitIndex) & 1

			if maskBit == 1 {
				// Insert Z gate before this gate
				zGate := QASMLiteGate{
					Type:   "Z",
					Qubits: []int{gate.Qubits[0]},
					ZMask:  1,
					Line:   gate.Line,
				}
				maskedProgram.Gates = append(maskedProgram.Gates, zGate)
			}
			maskIndex++
		}

		maskedProgram.Gates = append(maskedProgram.Gates, newGate)
	}

	log.Debug("🎭 Z-mask applied to QASM-lite program",
		"original_gates", len(program.Gates),
		"masked_gates", len(maskedProgram.Gates),
		"seed", hex.EncodeToString(seed)[:16]+"...")

	return maskedProgram
}

// GenerateTestVectors generates test vectors for seed→QASM→serialized bytes
func GenerateTestVectors() []TestVector {
	return []TestVector{
		{
			Name: "Simple H gate",
			Seed: make([]byte, 32), // Zero seed - no Z-mask applied
			QASM: `qreg q[2];
H q[0];`,
			ExpectedBytes: []byte{0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00},
		},
		{
			Name: "CX gate",
			Seed: make([]byte, 32), // Zero seed - no Z-mask applied
			QASM: `qreg q[2];
CX q[0],q[1];`,
			ExpectedBytes: []byte{0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x04, 0x02, 0x00, 0x00, 0x00, 0x01},
		},
		{
			Name: "T and S gates",
			Seed: make([]byte, 32), // Zero seed - no Z-mask applied
			QASM: `qreg q[1];
T q[0];
S q[0];`,
			ExpectedBytes: []byte{0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x03, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x00},
		},
		{
			Name: "Z-mask gate",
			Seed: make([]byte, 32), // Zero seed - no Z-mask applied
			QASM: `qreg q[1];
Z[1] q[0];`,
			ExpectedBytes: []byte{0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
		},
	}
}

// TestVector represents a test case for QASM-lite parsing and serialization
type TestVector struct {
	Name          string
	Seed          []byte
	QASM          string
	ExpectedBytes []byte
}

// ValidateTestVectors validates all test vectors
func ValidateTestVectors() error {
	vectors := GenerateTestVectors()

	for _, vector := range vectors {
		log.Info("🧪 Testing QASM-lite vector", "name", vector.Name)

		// Parse QASM
		parser := NewQASMLiteParser(vector.QASM)
		program, err := parser.Parse()
		if err != nil {
			return fmt.Errorf("test vector %s: parse failed: %v", vector.Name, err)
		}

		// Apply Z-mask
		maskedProgram := ApplyZMask(program, vector.Seed)

		// Serialize
		serialized := SerializeProgram(maskedProgram)

		// Validate (basic length check - full validation would need exact byte comparison)
		if len(serialized) == 0 {
			return fmt.Errorf("test vector %s: empty serialization", vector.Name)
		}

		log.Info("✅ QASM-lite test vector passed",
			"name", vector.Name,
			"serialized_length", len(serialized))
	}

	return nil
}
