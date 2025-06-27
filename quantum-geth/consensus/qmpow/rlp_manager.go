// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Centralized Quantum RLP Management
// Ensures consistent encoding/decoding of quantum fields across all network paths

package qmpow

import (
	"fmt"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rlp"
)

// QuantumRLPManager provides centralized quantum field RLP operations
// This prevents the inconsistencies that caused issues like uncle hash mismatches
type QuantumRLPManager struct {
	// Future: add configuration options if needed
}

// NewQuantumRLPManager creates a new centralized quantum RLP manager
func NewQuantumRLPManager() *QuantumRLPManager {
	return &QuantumRLPManager{}
}

// EncodeHeaderForTransmission ensures consistent header encoding for network transmission
// This function MUST be used whenever sending headers over the network
func (qrm *QuantumRLPManager) EncodeHeaderForTransmission(header *types.Header) ([]byte, error) {
	// Create a copy to avoid modifying the original
	headerCopy := types.CopyHeader(header)
	
	// CRITICAL FIX: Ensure quantum consensus fields are properly set
	// Quantum blockchains don't use post-merge Ethereum features
	headerCopy.WithdrawalsHash = nil
	
	// Always marshal quantum fields before network transmission
	headerCopy.MarshalQuantumBlob()
	
	log.Debug("🔗 Quantum RLP: Encoding header for transmission",
		"blockNumber", headerCopy.Number.Uint64(),
		"qblobSize", len(headerCopy.QBlob),
		"withdrawalsHash", headerCopy.WithdrawalsHash)
	
	// Encode the header with marshaled quantum fields
	encoded, err := rlp.EncodeToBytes(headerCopy)
	if err != nil {
		return nil, fmt.Errorf("header RLP encoding failed: %v", err)
	}
	
	// Validate encoding size is reasonable for quantum headers
	if len(encoded) < 500 {
		return nil, fmt.Errorf("encoded header too small: %d bytes (expected >500)", len(encoded))
	}
	if len(encoded) > 2048 {
		return nil, fmt.Errorf("encoded header too large: %d bytes (expected <2048)", len(encoded))
	}
	
	return encoded, nil
}

// DecodeHeaderFromNetwork ensures consistent header decoding from network
// This function MUST be used whenever receiving headers over the network
func (qrm *QuantumRLPManager) DecodeHeaderFromNetwork(data []byte) (*types.Header, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty header data")
	}
	
	// Decode the header
	header := new(types.Header)
	if err := rlp.DecodeBytes(data, header); err != nil {
		return nil, fmt.Errorf("header RLP decoding failed: %v", err)
	}
	
	// Always unmarshal quantum fields after network reception
	if err := header.UnmarshalQuantumBlob(); err != nil {
		return nil, fmt.Errorf("quantum blob unmarshaling failed: %v", err)
	}
	
	log.Debug("🔗 Quantum RLP: Decoded header from network",
		"blockNumber", header.Number.Uint64(),
		"qblobSize", len(header.QBlob))
	
	return header, nil
}

// EncodeBlockForTransmission ensures consistent block encoding for network transmission
// This function MUST be used whenever sending blocks over the network
func (qrm *QuantumRLPManager) EncodeBlockForTransmission(block *types.Block) ([]byte, error) {
	// Create a copy with properly marshaled quantum fields
	header := block.Header()
	header.MarshalQuantumBlob()
	
	// Create block copy with marshaled header
	encodedBlock := block.WithSeal(header)
	
	log.Debug("🔗 Quantum RLP: Encoding block for transmission",
		"blockNumber", header.Number.Uint64(),
		"blockHash", encodedBlock.Hash().Hex()[:10],
		"qblobSize", len(header.QBlob))
	
	// Encode the complete block
	encoded, err := rlp.EncodeToBytes(encodedBlock)
	if err != nil {
		return nil, fmt.Errorf("block RLP encoding failed: %v", err)
	}
	
	// Validate encoding size is reasonable
	if len(encoded) < 600 {
		return nil, fmt.Errorf("encoded block too small: %d bytes (expected >600)", len(encoded))
	}
	if len(encoded) > 1048576 { // 1MB max
		return nil, fmt.Errorf("encoded block too large: %d bytes (expected <1MB)", len(encoded))
	}
	
	return encoded, nil
}

// DecodeBlockFromNetwork ensures consistent block decoding from network
// This function MUST be used whenever receiving blocks over the network
func (qrm *QuantumRLPManager) DecodeBlockFromNetwork(data []byte) (*types.Block, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty block data")
	}
	
	// Decode the block
	var block types.Block
	if err := rlp.DecodeBytes(data, &block); err != nil {
		return nil, fmt.Errorf("block RLP decoding failed: %v", err)
	}
	
	// Always unmarshal quantum fields after network reception
	if err := block.Header().UnmarshalQuantumBlob(); err != nil {
		return nil, fmt.Errorf("quantum blob unmarshaling failed: %v", err)
	}
	
	log.Debug("🔗 Quantum RLP: Decoded block from network",
		"blockNumber", block.Number().Uint64(),
		"blockHash", block.Hash().Hex()[:10],
		"qblobSize", len(block.Header().QBlob))
	
	return &block, nil
}

// ValidateRLPConsistency performs roundtrip validation to ensure encoding/decoding consistency
// This prevents issues like hash mismatches between work templates and submissions
func (qrm *QuantumRLPManager) ValidateRLPConsistency(block *types.Block) error {
	originalHash := block.Hash()
	
	log.Debug("🔍 Quantum RLP: Starting consistency validation",
		"blockNumber", block.Number().Uint64(),
		"originalHash", originalHash.Hex()[:10])
	
	// Test block encoding roundtrip
	encoded, err := qrm.EncodeBlockForTransmission(block)
	if err != nil {
		return fmt.Errorf("encoding failed: %v", err)
	}
	
	// Test block decoding roundtrip
	decoded, err := qrm.DecodeBlockFromNetwork(encoded)
	if err != nil {
		return fmt.Errorf("decoding failed: %v", err)
	}
	
	// Verify hash consistency (critical for consensus)
	decodedHash := decoded.Hash()
	if originalHash != decodedHash {
		return fmt.Errorf("RLP roundtrip hash mismatch: original=%s, decoded=%s",
			originalHash.Hex(), decodedHash.Hex())
	}
	
	// Verify quantum field consistency
	if err := qrm.validateQuantumFieldConsistency(block.Header(), decoded.Header()); err != nil {
		return fmt.Errorf("quantum field consistency failed: %v", err)
	}
	
	// Verify transaction count consistency
	if len(block.Transactions()) != len(decoded.Transactions()) {
		return fmt.Errorf("transaction count mismatch: original=%d, decoded=%d",
			len(block.Transactions()), len(decoded.Transactions()))
	}
	
	// Verify uncle count consistency (should always be 0 for quantum consensus)
	if len(block.Uncles()) != len(decoded.Uncles()) {
		return fmt.Errorf("uncle count mismatch: original=%d, decoded=%d",
			len(block.Uncles()), len(decoded.Uncles()))
	}
	
	log.Debug("✅ Quantum RLP: Consistency validation passed",
		"blockNumber", block.Number().Uint64(),
		"hash", originalHash.Hex()[:10])
	
	return nil
}

// ValidateHeaderRLPConsistency performs validation for headers suitable for quantum consensus
func (qrm *QuantumRLPManager) ValidateHeaderRLPConsistency(header *types.Header) error {
	// Create a copy to avoid modifying the original header during validation
	headerCopy := types.CopyHeader(header)
	
	log.Debug("🔍 Quantum RLP: Starting header consistency validation",
		"blockNumber", headerCopy.Number.Uint64(),
		"withdrawalsHash", headerCopy.WithdrawalsHash)
	
	// CRITICAL FIX: Validate quantum consensus requirements
	// Quantum blockchains don't use post-merge Ethereum features
	if headerCopy.WithdrawalsHash != nil {
		return fmt.Errorf("quantum consensus does not support withdrawals: WithdrawalsHash must be nil")
	}
	
	// Test header encoding (this also normalizes the header)
	encoded, err := qrm.EncodeHeaderForTransmission(headerCopy)
	if err != nil {
		return fmt.Errorf("header encoding failed: %v", err)
	}
	
	// Test header decoding
	decoded, err := qrm.DecodeHeaderFromNetwork(encoded)
	if err != nil {
		return fmt.Errorf("header decoding failed: %v", err)
	}
	
	// Verify essential fields consistency (skip hash comparison for optional fields)
	if err := qrm.validateEssentialFieldConsistency(headerCopy, decoded); err != nil {
		return fmt.Errorf("essential field consistency failed: %v", err)
	}
	
	// Verify quantum field consistency
	if err := qrm.validateQuantumFieldConsistency(headerCopy, decoded); err != nil {
		return fmt.Errorf("quantum field consistency failed: %v", err)
	}
	
	log.Debug("✅ Quantum RLP: Header consistency validation passed",
		"blockNumber", headerCopy.Number.Uint64())
	
	return nil
}

// validateEssentialFieldConsistency checks that critical blockchain fields are consistent
func (qrm *QuantumRLPManager) validateEssentialFieldConsistency(original, decoded *types.Header) error {
	// Check block number
	if original.Number == nil || decoded.Number == nil || original.Number.Cmp(decoded.Number) != 0 {
		return fmt.Errorf("block number mismatch")
	}
	
	// Check parent hash
	if original.ParentHash != decoded.ParentHash {
		return fmt.Errorf("parent hash mismatch")
	}
	
	// Check difficulty
	if original.Difficulty == nil || decoded.Difficulty == nil || original.Difficulty.Cmp(decoded.Difficulty) != 0 {
		return fmt.Errorf("difficulty mismatch")
	}
	
	// Check state root
	if original.Root != decoded.Root {
		return fmt.Errorf("state root mismatch")
	}
	
	// Check transaction root
	if original.TxHash != decoded.TxHash {
		return fmt.Errorf("transaction root mismatch")
	}
	
	// Check receipt root
	if original.ReceiptHash != decoded.ReceiptHash {
		return fmt.Errorf("receipt root mismatch")
	}
	
	// Check gas limit and gas used
	if original.GasLimit != decoded.GasLimit {
		return fmt.Errorf("gas limit mismatch")
	}
	if original.GasUsed != decoded.GasUsed {
		return fmt.Errorf("gas used mismatch")
	}
	
	// Check timestamp
	if original.Time != decoded.Time {
		return fmt.Errorf("timestamp mismatch")
	}
	
	return nil
}

// validateQuantumFieldConsistency checks that quantum fields survived the roundtrip
func (qrm *QuantumRLPManager) validateQuantumFieldConsistency(original, decoded *types.Header) error {
	// Check QNonce64
	if (original.QNonce64 == nil) != (decoded.QNonce64 == nil) {
		return fmt.Errorf("QNonce64 nil mismatch")
	}
	if original.QNonce64 != nil && *original.QNonce64 != *decoded.QNonce64 {
		return fmt.Errorf("QNonce64 value mismatch: original=%d, decoded=%d",
			*original.QNonce64, *decoded.QNonce64)
	}
	
	// Check OutcomeRoot
	if (original.OutcomeRoot == nil) != (decoded.OutcomeRoot == nil) {
		return fmt.Errorf("OutcomeRoot nil mismatch")
	}
	if original.OutcomeRoot != nil && *original.OutcomeRoot != *decoded.OutcomeRoot {
		return fmt.Errorf("OutcomeRoot mismatch: original=%s, decoded=%s",
			original.OutcomeRoot.Hex(), decoded.OutcomeRoot.Hex())
	}
	
	// Check GateHash
	if (original.GateHash == nil) != (decoded.GateHash == nil) {
		return fmt.Errorf("GateHash nil mismatch")
	}
	if original.GateHash != nil && *original.GateHash != *decoded.GateHash {
		return fmt.Errorf("GateHash mismatch: original=%s, decoded=%s",
			original.GateHash.Hex(), decoded.GateHash.Hex())
	}
	
	// Check ProofRoot
	if (original.ProofRoot == nil) != (decoded.ProofRoot == nil) {
		return fmt.Errorf("ProofRoot nil mismatch")
	}
	if original.ProofRoot != nil && *original.ProofRoot != *decoded.ProofRoot {
		return fmt.Errorf("ProofRoot mismatch: original=%s, decoded=%s",
			original.ProofRoot.Hex(), decoded.ProofRoot.Hex())
	}
	
	// Check ExtraNonce32 size
	if len(original.ExtraNonce32) != len(decoded.ExtraNonce32) {
		return fmt.Errorf("ExtraNonce32 length mismatch: original=%d, decoded=%d",
			len(original.ExtraNonce32), len(decoded.ExtraNonce32))
	}
	
	// Check BranchNibbles size
	if len(original.BranchNibbles) != len(decoded.BranchNibbles) {
		return fmt.Errorf("BranchNibbles length mismatch: original=%d, decoded=%d",
			len(original.BranchNibbles), len(decoded.BranchNibbles))
	}
	
	return nil
}

// PrepareBlockForExternalMiner ensures blocks sent to external miners have consistent RLP structure
// This prevents the hash mismatches that caused the uncle root issue
func (qrm *QuantumRLPManager) PrepareBlockForExternalMiner(block *types.Block) (*types.Block, error) {
	// Validate RLP consistency before sending to external miners
	if err := qrm.ValidateRLPConsistency(block); err != nil {
		return nil, fmt.Errorf("block failed RLP consistency check: %v", err)
	}
	
	// Ensure quantum fields are properly marshaled
	header := block.Header()
	header.MarshalQuantumBlob()
	
	preparedBlock := block.WithSeal(header)
	
	log.Debug("🔗 Quantum RLP: Prepared block for external miner",
		"blockNumber", header.Number.Uint64(),
		"blockHash", preparedBlock.Hash().Hex()[:10],
		"qblobSize", len(header.QBlob))
	
	return preparedBlock, nil
}

// ValidateExternalMinerSubmission ensures submissions from external miners have consistent RLP structure
func (qrm *QuantumRLPManager) ValidateExternalMinerSubmission(block *types.Block) error {
	log.Debug("🔍 Quantum RLP: Validating external miner submission",
		"blockNumber", block.Number().Uint64(),
		"blockHash", block.Hash().Hex()[:10])
	
	// Perform comprehensive RLP consistency validation
	if err := qrm.ValidateRLPConsistency(block); err != nil {
		return fmt.Errorf("external miner submission failed RLP validation: %v", err)
	}
	
	log.Debug("✅ Quantum RLP: External miner submission validated",
		"blockNumber", block.Number().Uint64(),
		"blockHash", block.Hash().Hex()[:10])
	
	return nil
} 