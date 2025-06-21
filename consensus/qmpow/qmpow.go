// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the quantum micro-puzzle proof-of-work consensus engine.
package qmpow

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/consensus/qmpow/proof"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/holiman/uint256"
)

var (
	// ErrInvalidQuantumProof is returned when quantum proof verification fails
	ErrInvalidQuantumProof = errors.New("invalid quantum proof")

	// ErrMissingQuantumFields is returned when quantum fields are missing from header
	ErrMissingQuantumFields = errors.New("missing quantum fields in header")

	// ErrInvalidDifficulty is returned when difficulty parameters are invalid
	ErrInvalidDifficulty = errors.New("invalid difficulty parameters")

	// ErrInvalidOutcomeLength is returned when quantum outcome length is wrong
	ErrInvalidOutcomeLength = errors.New("invalid quantum outcome length")
)

// QMPoW is the quantum micro-puzzle proof-of-work consensus engine
type QMPoW struct {
	config   Config
	threads  int
	update   chan struct{}
	hashrate uint64

	lock sync.RWMutex
}

// Config represents the configuration of the quantum proof of work engine
type Config struct {
	PowMode       Mode
	SolverPath    string        // path to quantum solver executable
	SolverTimeout time.Duration // timeout for solver operations
	TestMode      bool          // simplified verification for testing
}

// Mode represents the consensus operating mode
type Mode uint

const (
	ModeNormal Mode = iota // normal quantum proof verification
	ModeTest               // simplified testing mode
	ModeFake               // fake mode for development
)

// New creates a quantum proof of work consensus engine
func New(config Config) *QMPoW {
	if config.SolverTimeout == 0 {
		config.SolverTimeout = 30 * time.Second
	}

	qmpow := &QMPoW{
		config:  config,
		threads: runtime.NumCPU(),
		update:  make(chan struct{}),
	}

	return qmpow
}

// Author implements consensus.Engine, returning the header's coinbase as the
// block author.
func (q *QMPoW) Author(header *types.Header) (common.Address, error) {
	return header.Coinbase, nil
}

// VerifyHeader checks whether a header conforms to the consensus rules
func (q *QMPoW) VerifyHeader(chain consensus.ChainHeaderReader, header *types.Header, seal bool) error {
	// Check that the header has quantum fields
	if header.QBits == nil || header.TCount == nil || header.LUsed == nil {
		return ErrMissingQuantumFields
	}

	// Get expected parameters for this height
	params := q.ParamsForHeight(header.Number.Uint64())

	// Verify quantum parameters match expected values
	if *header.QBits != params.QBits {
		return fmt.Errorf("invalid QBits: got %d, expected %d", *header.QBits, params.QBits)
	}

	if *header.TCount != params.TCount {
		return fmt.Errorf("invalid TCount: got %d, expected %d", *header.TCount, params.TCount)
	}

	// Verify outcome length
	expectedOutcomeLen := int(*header.LUsed) * int((*header.QBits+7)/8)
	if len(header.QOutcome) != expectedOutcomeLen {
		return ErrInvalidOutcomeLength
	}

	// If we're not verifying the seal, skip proof verification
	if !seal {
		return nil
	}

	// Verify the quantum proof
	return q.verifyQuantumProof(chain, header)
}

// VerifyHeaders is similar to VerifyHeader, but verifies a batch of headers
func (q *QMPoW) VerifyHeaders(chain consensus.ChainHeaderReader, headers []*types.Header, seals []bool) (chan<- struct{}, <-chan error) {
	abort := make(chan struct{})
	results := make(chan error, len(headers))

	go func() {
		defer close(results)

		for i, header := range headers {
			select {
			case <-abort:
				return
			default:
				err := q.VerifyHeader(chain, header, seals[i])
				results <- err
			}
		}
	}()

	return abort, results
}

// VerifyUncles verifies that the given block's uncles conform to the consensus rules
func (q *QMPoW) VerifyUncles(chain consensus.ChainReader, block *types.Block) error {
	// Quantum PoW uses the same uncle verification as Ethash
	if len(block.Uncles()) > 2 {
		return errors.New("too many uncles")
	}

	// Verify each uncle
	for _, uncle := range block.Uncles() {
		if err := q.VerifyHeader(chain, uncle, true); err != nil {
			return err
		}
	}

	return nil
}

// Prepare initializes the consensus fields of a block header
func (q *QMPoW) Prepare(chain consensus.ChainHeaderReader, header *types.Header) error {
	params := q.ParamsForHeight(header.Number.Uint64())

	// Set quantum parameters
	header.QBits = &params.QBits
	header.TCount = &params.TCount
	header.LUsed = &params.LNet

	// Clear quantum fields that will be filled during sealing
	header.QOutcome = nil
	header.QProof = nil

	return nil
}

// Finalize runs any post-transaction state modifications
func (q *QMPoW) Finalize(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB,
	txs []*types.Transaction, uncles []*types.Header, withdrawals []*types.Withdrawal) {

	// Apply block rewards (same as Ethash for now)
	blockReward := uint256.NewInt(2e18) // 2 ETH per block
	state.AddBalance(header.Coinbase, blockReward)
}

// FinalizeAndAssemble runs any post-transaction state modifications and assembles the final block
func (q *QMPoW) FinalizeAndAssemble(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB,
	txs []*types.Transaction, uncles []*types.Header, receipts []*types.Receipt,
	withdrawals []*types.Withdrawal) (*types.Block, error) {

	// Finalize the header
	q.Finalize(chain, header, state, txs, uncles, withdrawals)

	// Assemble and return the final block
	return types.NewBlockWithWithdrawals(header, txs, uncles, receipts, withdrawals, nil), nil
}

// Seal generates a new sealing request for the given input block
func (q *QMPoW) Seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	// Start sealing in a separate goroutine
	go q.seal(chain, block, results, stop)
	return nil
}

// seal is the actual sealing function that runs the quantum proof of work
func (q *QMPoW) seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) {
	header := types.CopyHeader(block.Header())

	// Generate the seed for quantum puzzles
	seed := q.SealHash(header)

	log.Info("🔬 Starting quantum proof of work",
		"number", header.Number.Uint64(),
		"qbits", *header.QBits,
		"tcount", *header.TCount,
		"lused", *header.LUsed,
		"seed", fmt.Sprintf("%x", seed.Bytes()[:8]))

	start := time.Now()

	// Generate quantum proofs
	log.Info("🔬 Calling solveQuantumPuzzles...")
	outcomes, aggregateProof, err := q.solveQuantumPuzzles(seed.Bytes(), *header.QBits, *header.TCount, *header.LUsed)
	if err != nil {
		log.Error("❌ Failed to solve quantum puzzles", "err", err)
		return
	}
	log.Info("✅ Quantum puzzles solved successfully", "outcomes_len", len(outcomes), "proof_len", len(aggregateProof))

	// Check if we were stopped
	select {
	case <-stop:
		log.Info("🛑 Sealing stopped")
		return
	default:
	}

	// Fill quantum fields in header
	header.QOutcome = outcomes
	header.QProof = aggregateProof

	sealTime := time.Since(start)

	// Update hashrate (puzzles per second)
	puzzlesPerSecond := float64(*header.LUsed) / sealTime.Seconds()
	q.lock.Lock()
	q.hashrate = uint64(puzzlesPerSecond)
	q.lock.Unlock()

	log.Info("🎯 Quantum proof of work completed",
		"number", header.Number.Uint64(),
		"puzzles", *header.LUsed,
		"time", sealTime,
		"hashrate", fmt.Sprintf("%.2f puzzles/sec", puzzlesPerSecond),
		"proof_size", len(aggregateProof))

	// Create and send the sealed block
	sealedBlock := block.WithSeal(header)

	log.Info("📦 Sending sealed block to results channel")
	select {
	case results <- sealedBlock:
		log.Info("✅ Sealed block sent successfully")
	case <-stop:
		log.Info("🛑 Sealing stopped while sending block")
	}
}

// SealHash returns the hash of a block prior to it being sealed
func (q *QMPoW) SealHash(header *types.Header) common.Hash {
	// Create a copy of the header without quantum fields
	headerCopy := types.CopyHeader(header)
	headerCopy.QOutcome = nil
	headerCopy.QProof = nil

	return rlpHash(headerCopy)
}

// CalcDifficulty is the difficulty adjustment algorithm
func (q *QMPoW) CalcDifficulty(chain consensus.ChainHeaderReader, time uint64, parent *types.Header) *big.Int {
	// For quantum PoW, difficulty is represented as a big integer equivalent of L_net
	// This maintains compatibility with existing difficulty-based code

	if parent.LUsed == nil {
		return big.NewInt(int64(DefaultLNet))
	}

	nextLNet := q.EstimateNextDifficulty(chain, &types.Header{
		ParentHash: parent.Hash(),
		Number:     new(big.Int).Add(parent.Number, big.NewInt(1)),
		Time:       time,
	})

	return big.NewInt(int64(nextLNet))
}

// APIs returns the RPC APIs this consensus engine provides
func (q *QMPoW) APIs(chain consensus.ChainHeaderReader) []rpc.API {
	return []rpc.API{
		{
			Namespace: "qmpow",
			Version:   "1.0",
			Service:   &API{q},
			Public:    true,
		},
	}
}

// Close terminates any background threads maintained by the consensus engine
func (q *QMPoW) Close() error {
	close(q.update)
	return nil
}

// Hashrate returns the current mining hashrate (quantum puzzles per second)
func (q *QMPoW) Hashrate() float64 {
	q.lock.RLock()
	defer q.lock.RUnlock()
	return float64(q.hashrate)
}

// SetThreads sets the number of mining threads
func (q *QMPoW) SetThreads(threads int) {
	q.lock.Lock()
	defer q.lock.Unlock()
	q.threads = threads
}

// verifyQuantumProof verifies the quantum proof in a block header
func (q *QMPoW) verifyQuantumProof(chain consensus.ChainHeaderReader, header *types.Header) error {
	if q.config.TestMode {
		// In test mode, just verify the structure
		return q.verifyQuantumProofStructure(header)
	}

	// Generate the seed
	seed := q.SealHash(header)

	// Deserialize the aggregate proof
	aggregateProof, err := proof.DeserializeAggregateProof(header.QProof)
	if err != nil {
		return fmt.Errorf("failed to deserialize quantum proof: %v", err)
	}

	// Verify the aggregate proof
	if err := proof.VerifyAggregate(aggregateProof, seed.Bytes(), header.ParentHash); err != nil {
		return fmt.Errorf("quantum proof verification failed: %v", err)
	}

	// Verify outcome consistency
	if !bytesEqual(header.QOutcome, aggregateProof.Outcomes) {
		return fmt.Errorf("quantum outcome mismatch")
	}

	return nil
}

// verifyQuantumProofStructure verifies just the structure of quantum fields (for testing)
func (q *QMPoW) verifyQuantumProofStructure(header *types.Header) error {
	// Verify outcome length
	expectedOutcomeLen := int(*header.LUsed) * int((*header.QBits+7)/8)
	if len(header.QOutcome) != expectedOutcomeLen {
		return ErrInvalidOutcomeLength
	}

	// Verify proof is not empty
	if len(header.QProof) == 0 {
		return ErrInvalidQuantumProof
	}

	return nil
}

// solveQuantumPuzzles generates quantum puzzle solutions (simplified for development)
func (q *QMPoW) solveQuantumPuzzles(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	if q.config.PowMode == ModeFake {
		return q.generateFakeQuantumSolution(seed, qbits, tcount, lnet)
	}

	// For development, use deterministic simulation
	return q.simulateQuantumSolver(seed, qbits, tcount, lnet)
}

// simulateQuantumSolver simulates quantum puzzle solving for development
func (q *QMPoW) simulateQuantumSolver(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	proofs := make([]proof.Proof, lnet)
	currentSeed := seed

	outcomeLen := int(qbits+7) / 8
	allOutcomes := make([]byte, 0, int(lnet)*outcomeLen)

	for i := uint16(0); i < lnet; i++ {
		// Generate Mahadev proof for this puzzle
		mahadevProof, err := proof.GenerateMahadevProof(currentSeed, qbits, tcount)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to generate proof for puzzle %d: %v", i, err)
		}

		proofs[i] = proof.Proof{
			PuzzleIndex: i,
			Outcome:     mahadevProof.Outcome,
			Witness:     mahadevProof.Witness,
		}

		allOutcomes = append(allOutcomes, mahadevProof.Outcome...)

		// Generate seed for next puzzle
		if i < lnet-1 {
			h := sha256.New()
			h.Write(currentSeed)
			h.Write(mahadevProof.Outcome)
			currentSeed = h.Sum(nil)
		}
	}

	// Create aggregate proof
	aggregateProof, err := proof.CreateAggregate(proofs, qbits, tcount)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create aggregate proof: %v", err)
	}

	return allOutcomes, aggregateProof.Serialize(), nil
}

// generateFakeQuantumSolution generates fake quantum solution for testing
func (q *QMPoW) generateFakeQuantumSolution(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	log.Info("🧪 Generating fake quantum solution", "qbits", qbits, "tcount", tcount, "lnet", lnet)

	// Generate deterministic fake outcomes
	h := sha256.New()
	h.Write(seed)
	h.Write([]byte("fake_quantum"))
	digest := h.Sum(nil)

	outcomeLen := int(qbits+7) / 8
	totalOutcomeLen := int(lnet) * outcomeLen
	outcomes := make([]byte, totalOutcomeLen)

	for i := 0; i < totalOutcomeLen; i++ {
		outcomes[i] = digest[i%len(digest)]
	}

	// Generate fake proof
	fakeProof := make([]byte, 64) // Fixed size fake proof
	copy(fakeProof, digest[:])

	log.Info("🧪 Fake quantum solution generated", "outcomes_len", len(outcomes), "proof_len", len(fakeProof))
	return outcomes, fakeProof, nil
}

// rlpHash computes the RLP hash of the given data
func rlpHash(x interface{}) common.Hash {
	h := sha256.New()
	rlp.Encode(h, x)
	return common.BytesToHash(h.Sum(nil))
}

// bytesEqual compares two byte slices
func bytesEqual(a, b []byte) bool {
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

// API exposes quantum proof of work related methods for RPC
type API struct {
	qmpow *QMPoW
}

// GetQuantumParams returns the current quantum parameters
func (api *API) GetQuantumParams(blockNr rpc.BlockNumber) (map[string]interface{}, error) {
	params := api.qmpow.ParamsForHeight(uint64(blockNr))

	return map[string]interface{}{
		"qbits":    params.QBits,
		"tcount":   params.TCount,
		"lnet":     params.LNet,
		"epochLen": params.EpochLen,
	}, nil
}

// GetHashrate returns the current quantum hashrate
func (api *API) GetHashrate() float64 {
	return api.qmpow.Hashrate()
}

// GetThreads returns the number of mining threads
func (api *API) GetThreads() int {
	api.qmpow.lock.RLock()
	defer api.qmpow.lock.RUnlock()
	return api.qmpow.threads
}

// GetDifficulty returns the current quantum difficulty
func (api *API) GetDifficulty() map[string]interface{} {
	params := api.qmpow.ParamsForHeight(0) // Get current params
	return map[string]interface{}{
		"puzzlesPerBlock": params.LNet,
		"qubitsPerPuzzle": params.QBits,
		"tgatesPerPuzzle": params.TCount,
		"totalComplexity": uint64(params.LNet) * uint64(params.QBits) * uint64(params.TCount),
	}
}

// GetMiningStats returns comprehensive mining statistics
func (api *API) GetMiningStats() map[string]interface{} {
	api.qmpow.lock.RLock()
	defer api.qmpow.lock.RUnlock()

	params := api.qmpow.ParamsForHeight(0)
	hashrate := float64(api.qmpow.hashrate)

	return map[string]interface{}{
		"hashrate":     hashrate,
		"hashrateUnit": "puzzles/second",
		"threads":      api.qmpow.threads,
		"difficulty": map[string]interface{}{
			"puzzlesPerBlock": params.LNet,
			"qubitsPerPuzzle": params.QBits,
			"tgatesPerPuzzle": params.TCount,
			"totalComplexity": uint64(params.LNet) * uint64(params.QBits) * uint64(params.TCount),
		},
		"retargetPeriod":  params.EpochLen,
		"targetBlockTime": TargetBlockTime,
	}
}

// GetNetworkStats returns network-wide statistics
func (api *API) GetNetworkStats() map[string]interface{} {
	params := api.qmpow.ParamsForHeight(0)

	// Calculate theoretical network hashrate based on difficulty and target time
	theoreticalHashrate := float64(params.LNet) / float64(TargetBlockTime)

	return map[string]interface{}{
		"currentDifficulty":   params.LNet,
		"theoreticalHashrate": theoreticalHashrate,
		"hashrateUnit":        "puzzles/second",
		"targetBlockTime":     TargetBlockTime,
		"retargetPeriod":      params.EpochLen,
		"quantumComplexity": map[string]interface{}{
			"qubits":     params.QBits,
			"tgates":     params.TCount,
			"stateSpace": uint64(1) << params.QBits, // 2^qubits
		},
	}
}
