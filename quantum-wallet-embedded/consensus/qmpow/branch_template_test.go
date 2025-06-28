// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

package qmpow

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"

	"github.com/ethereum/go-ethereum/crypto"
)

func TestBranchTemplateEngine(t *testing.T) {
	engine := NewBranchTemplateEngine()

	// Verify engine initialization
	if len(engine.templates) != 16 {
		t.Errorf("Expected 16 templates, got %d", len(engine.templates))
	}

	// Verify all templates have valid IDs
	for i, template := range engine.templates {
		if template.ID != i {
			t.Errorf("Template %d has incorrect ID: %d", i, template.ID)
		}

		if template.Name == "" {
			t.Errorf("Template %d has empty name", i)
		}

		if template.Skeleton == "" {
			t.Errorf("Template %d has empty skeleton", i)
		}

		if len(template.Parameters) == 0 {
			t.Errorf("Template %d has no parameters", i)
		}
	}
}

func TestInstantiateBranch(t *testing.T) {
	engine := NewBranchTemplateEngine()
	testSeed := crypto.Keccak256([]byte("test_seed"))

	// Test first 5 templates
	for i := 0; i < 5; i++ {
		t.Run(fmt.Sprintf("template_%d", i), func(t *testing.T) {
			instantiation, err := engine.InstantiateBranch(i, testSeed)
			if err != nil {
				t.Errorf("Failed to instantiate template %d: %v", i, err)
				return
			}

			// Verify instantiation properties
			if instantiation.TemplateID != i {
				t.Errorf("Template ID mismatch: expected %d, got %d", i, instantiation.TemplateID)
			}

			if len(instantiation.QASM) == 0 {
				t.Error("Instantiated QASM is empty")
			}

			if !instantiation.Validated {
				t.Error("Instantiation should be validated")
			}

			// Verify QASM contains no placeholders
			if strings.Contains(instantiation.QASM, "{") || strings.Contains(instantiation.QASM, "}") {
				t.Error("QASM still contains placeholders")
			}

			// Verify QASM is parseable
			parser := NewQASMLiteParser(instantiation.QASM)
			_, err = parser.Parse()
			if err != nil {
				t.Errorf("Instantiated QASM is not parseable: %v", err)
			}

			t.Logf("Template %d instantiated: depth %d, T-gates %d",
				i, instantiation.Depth, instantiation.TGateCount)
		})
	}
}

func TestDeterministicInstantiation(t *testing.T) {
	engine := NewBranchTemplateEngine()
	seed := sha256.Sum256([]byte("deterministic_test"))

	// Test first 3 templates for determinism
	for templateID := 0; templateID < 3; templateID++ {
		instantiation1, err := engine.InstantiateBranch(templateID, seed[:])
		if err != nil {
			t.Fatalf("First instantiation failed for template %d: %v", templateID, err)
		}

		instantiation2, err := engine.InstantiateBranch(templateID, seed[:])
		if err != nil {
			t.Fatalf("Second instantiation failed for template %d: %v", templateID, err)
		}

		// Compare instantiations
		if instantiation1.QASM != instantiation2.QASM {
			t.Errorf("Template %d instantiation is not deterministic", templateID)
		}
	}
}

func TestValidateAllTemplates(t *testing.T) {
	engine := NewBranchTemplateEngine()

	err := engine.ValidateAllTemplates()
	if err != nil {
		t.Errorf("Template validation failed: %v", err)
	}
}

func TestInvalidTemplateID(t *testing.T) {
	engine := NewBranchTemplateEngine()
	seed := crypto.Keccak256([]byte("test"))

	invalidIDs := []int{-1, 16, 100}
	for _, id := range invalidIDs {
		_, err := engine.InstantiateBranch(id, seed)
		if err == nil {
			t.Errorf("Expected error for invalid template ID %d", id)
		}
	}
}

func TestBitExtraction(t *testing.T) {
	data := []byte{0xFF, 0x00, 0xAA, 0x55} // 11111111 00000000 10101010 01010101

	tests := []struct {
		bitOffset int
		bitLength int
		expected  uint64
	}{
		{0, 8, 0xFF}, // First byte
		{8, 8, 0x00}, // Second byte
		{0, 4, 0xF},  // First nibble
		{4, 4, 0xF},  // Second nibble
	}

	for _, test := range tests {
		result := extractBits(data, test.bitOffset, test.bitLength)
		if result != test.expected {
			t.Errorf("extractBits(%v, %d, %d) = %d, expected %d",
				data, test.bitOffset, test.bitLength, result, test.expected)
		}
	}
}
