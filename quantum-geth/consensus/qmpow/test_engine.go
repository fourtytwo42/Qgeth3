// Copyright 2024 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package qmpow

import (
	"math/big"
	"math/rand"
	"runtime"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// NewFaker creates a QMPoW consensus engine with fake quantum proof verification
// for testing purposes. This replaces qmpow.NewFaker() in quantum blockchain tests.
func NewFaker() *QMPoW {
	log.Info("🧪 Creating QMPoW faker for testing")
	return &QMPoW{
		config: Config{
			PowMode:  ModeFake,
			TestMode: true,
		},
		// Initialize with test quantum parameters
		qbits:    16,  // 16 qubits per puzzle
		tcount:   20,  // 20 T-gates per puzzle 
		lnet:     128, // 128 puzzles per block
		epochLen: 100, // Short epochs for testing
		
		asertQ: NewASERTQDifficulty(),
		rand:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// NewFakeFailer creates a QMPoW consensus engine that fails after a specific block number
// for testing blockchain error handling. This replaces qmpow.NewFakeFailer().
func NewFakeFailer(fail uint64) *QMPoW {
	log.Info("🧪 Creating QMPoW fake failer for testing", "failAtBlock", fail)
	qmpow := NewFaker()
	qmpow.config.TestMode = true
	qmpow.fakeFailure = fail
	return qmpow
}

// NewFullFaker creates a QMPoW consensus engine with full quantum proof simulation
// for comprehensive testing. This replaces ethash.NewFullFaker().
func NewFullFaker() *QMPoW {
	log.Info("🧪 Creating QMPoW full faker for testing")
	qmpow := NewFaker()
	qmpow.config.PowMode = ModeTest // Use test mode with simulated proofs
	return qmpow
}

// NewTestEngine creates a QMPoW consensus engine specifically configured for unit tests
func NewTestEngine() *QMPoW {
	log.Info("🧪 Creating QMPoW test engine")
	return &QMPoW{
		config: Config{
			PowMode:  ModeTest,
			TestMode: true,
		},
		qbits:    16,
		tcount:   20,
		lnet:     128,
		epochLen: 100,
		
		asertQ: NewASERTQDifficulty(),
		rand:   rand.New(rand.NewSource(1)), // Fixed seed for deterministic tests
	}
}

// NewSharedFaker creates a shared QMPoW test engine instance
func NewSharedFaker() *QMPoW {
	log.Info("🧪 Creating shared QMPoW faker for testing")
	return NewFaker()
}

// For ModeFake, we bypass all quantum verification and just mine instantly
func (q *QMPoW) fakeVerifySeal(header *types.Header) error {
	if q.config.PowMode != ModeFake {
		return q.verifySeal(header, true)
	}
	
	// In fake mode, always accept the seal
	log.Debug("🧪 Fake quantum seal verification passed", "number", header.Number)
	return nil
}

// For ModeFake, generate fake quantum fields
func (q *QMPoW) fakeGenerateQuantumFields(header *types.Header) error {
	if q.config.PowMode != ModeFake {
		return q.generateQuantumFields(header)
	}
	
	// Generate fake quantum blob with correct structure
	qnonce := rand.Uint64()
	puzzleCount := uint32(q.lnet)
	
	// Create minimal fake quantum proof structure
	fakeProof := &QuantumProofSubmission{
		QNonce:      qnonce,
		PuzzleCount: puzzleCount,
		Solutions:   make([]QuantumSolution, puzzleCount),
	}
	
	// Fill with fake solutions
	for i := uint32(0); i < puzzleCount; i++ {
		fakeProof.Solutions[i] = QuantumSolution{
			PuzzleIndex: i,
			QBits:       uint32(q.qbits),
			TCount:      uint32(q.tcount),
			ProofData:   make([]byte, 32), // Fake 32-byte proof
		}
		// Fill with random data
		rand.Read(fakeProof.Solutions[i].ProofData)
	}
	
	// Marshal fake quantum blob
	qblob, err := MarshalQuantumBlob(fakeProof)
	if err != nil {
		log.Error("Failed to marshal fake quantum blob", "err", err)
		return err
	}
	
	header.QBlob = qblob
	header.QNonce = qnonce
	header.QPuzzleCount = puzzleCount
	
	log.Debug("🧪 Generated fake quantum fields", 
		"number", header.Number,
		"qnonce", qnonce,
		"puzzles", puzzleCount,
		"blobSize", len(qblob))
	
	return nil
}

// Fake seal implementation for testing
func (q *QMPoW) fakeSeal(block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	if q.config.PowMode != ModeFake {
		return q.Seal(nil, block, results, stop)
	}
	
	header := types.CopyHeader(block.Header())
	
	// Generate fake quantum fields
	if err := q.fakeGenerateQuantumFields(header); err != nil {
		log.Error("Failed to generate fake quantum fields", "err", err)
		return err
	}
	
	// Check if we should fail at this block (for testing error handling)
	if q.fakeFailure != 0 && header.Number.Uint64() == q.fakeFailure {
		log.Debug("🧪 Fake failure triggered", "number", header.Number)
		return consensus.ErrInvalidPoW
	}
	
	// Simulate some mining delay for realistic testing
	delay := time.Duration(rand.Intn(100)) * time.Millisecond
	select {
	case <-time.After(delay):
	case <-stop:
		return nil
	}
	
	// Create new block with quantum fields
	newBlock := types.NewBlockWithHeader(header).WithBody(block.Transactions(), block.Uncles()).WithWithdrawals(block.Withdrawals())
	
	log.Debug("🧪 Fake quantum block sealed", 
		"number", header.Number,
		"hash", newBlock.Hash(),
		"qnonce", header.QNonce)
	
	select {
	case results <- newBlock:
	case <-stop:
	}
	
	return nil
}

// Override Seal method for fake mode
func (q *QMPoW) Seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	if q.config.PowMode == ModeFake {
		return q.fakeSeal(block, results, stop)
	}
	
	// For non-fake modes, use the regular sealing process
	return q.seal(chain, block, results, stop)
}

// Override VerifySeal for fake mode
func (q *QMPoW) VerifySeal(chain consensus.ChainHeaderReader, header *types.Header) error {
	if q.config.PowMode == ModeFake {
		return q.fakeVerifySeal(header)
	}
	
	return q.verifySeal(header, true)
}

// Note: fakeFailure field added to main QMPoW struct in qmpow.go 