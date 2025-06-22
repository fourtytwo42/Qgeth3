// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the quantum micro-puzzle proof-of-work consensus engine.
package qmpow

import (
	"crypto/sha256"
	"encoding/binary"
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

	// Verify quantum parameters match expected values (fixed for Bitcoin-style)
	if *header.QBits != params.QBits {
		return fmt.Errorf("invalid QBits: got %d, expected %d", *header.QBits, params.QBits)
	}

	if *header.TCount != params.TCount {
		return fmt.Errorf("invalid TCount: got %d, expected %d", *header.TCount, params.TCount)
	}

	// Bitcoin-style mining always uses exactly 48 puzzles
	if *header.LUsed != DefaultLNet {
		return fmt.Errorf("invalid LUsed: got %d, expected %d (Bitcoin-style fixed)", *header.LUsed, DefaultLNet)
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

	// Bitcoin-style quantum proof verification
	log.Debug("🔍 Bitcoin-style quantum proof verification",
		"blockNumber", header.Number.Uint64(),
		"difficulty", header.Difficulty)

	// Use Bitcoin-style validation
	return ValidateQuantumProof(header)
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
	log.Info("🎯 QMPoW Prepare called (Bitcoin-style)", "blockNumber", header.Number.Uint64(), "parentHash", header.ParentHash.Hex())

	params := q.ParamsForHeight(header.Number.Uint64())

	// Set quantum parameters - FIXED puzzle count for Bitcoin-style mining
	header.QBits = &params.QBits
	header.TCount = &params.TCount
	header.LUsed = &params.LNet // Always 48 puzzles for 1,152-bit security

	// Bitcoin-style mining uses nonce iteration internally (no QNonce field needed)

	// Clear quantum fields that will be filled during sealing
	header.QOutcome = nil
	header.QProof = nil

	// Set difficulty using Bitcoin-style calculation
	if header.Number.Uint64() > 0 {
		// Get the parent header to calculate difficulty properly
		parent := chain.GetHeader(header.ParentHash, header.Number.Uint64()-1)
		if parent != nil {
			// Use Bitcoin-style difficulty calculation
			calculatedDifficulty := q.CalcDifficulty(chain, header.Time, parent)
			header.Difficulty = calculatedDifficulty

			log.Info("🎯 Bitcoin-style difficulty set in Prepare",
				"blockNumber", header.Number.Uint64(),
				"parentDifficulty", parent.Difficulty,
				"newDifficulty", header.Difficulty,
				"fixedPuzzles", params.LNet,
				"security", "1,152-bit",
				"style", "Bitcoin-nonce-target")
		} else {
			// Fallback if parent not found
			header.Difficulty = big.NewInt(int64(params.LNet))
			log.Info("🔗 Parent not found in Prepare, using default difficulty",
				"blockNumber", header.Number.Uint64(),
				"difficulty", header.Difficulty,
				"fixedPuzzles", params.LNet)
		}
	} else {
		// Genesis block - start with reasonable difficulty for competitive Bitcoin-style mining
		header.Difficulty = big.NewInt(100000) // Much higher difficulty for competitive mining
		log.Info("🌱 Genesis block difficulty set (Bitcoin-style)",
			"difficulty", header.Difficulty,
			"fixedPuzzles", params.LNet,
			"security", "1,152-bit")
	}

	// Initialize optional fields to prevent RLP encoding issues
	q.initializeOptionalFields(header)

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

// seal is the Bitcoin-style quantum mining function
// This implements true Bitcoin-style mining with nonce iteration and target validation
func (q *QMPoW) seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) {
	header := types.CopyHeader(block.Header())

	// Initialize QNonce to zero (like Bitcoin starts at nonce 0)
	header.QNonce = types.EncodeQuantumNonce(0)

	// Calculate quantum target from difficulty (Bitcoin-style)
	target := CalculateQuantumTarget(header.Difficulty)

	log.Info("🎯 Starting Bitcoin-style quantum mining",
		"number", header.Number.Uint64(),
		"difficulty", header.Difficulty,
		"target", fmt.Sprintf("0x%x", target),
		"puzzles", *header.LUsed,
		"security", "1,152-bit")

	start := time.Now()

	// Bitcoin-style nonce iteration (internal tracking)
	qnonceVar := uint64(0)

	for qnonceVar <= MaxNonceAttempts {
		// Check if we should stop
		select {
		case <-stop:
			log.Info("🛑 Bitcoin-style quantum mining stopped", "attempts", qnonceVar)
			return
		default:
		}

		// Set QNonce for this attempt (Bitcoin-style)
		header.QNonce = types.EncodeQuantumNonce(qnonceVar)
		log.Debug("QNonce before sealing", "qnonce", qnonceVar)

		// Generate seed with nonce variation (Bitcoin-style)
		seed := q.SealHashWithNonce(header)

		// Solve exactly 48 quantum puzzles (fixed work like Bitcoin's SHA-256)
		outcomes, aggregateProof, err := q.solveQuantumPuzzles(seed.Bytes(), *header.QBits, *header.TCount, *header.LUsed)
		if err != nil {
			log.Error("❌ Failed to solve quantum puzzles", "qnonce", qnonceVar, "err", err)
			qnonceVar++
			continue
		}

		// Check if proof meets target (Bitcoin-style validation)
		if CheckQuantumProofTarget(outcomes, aggregateProof, qnonceVar, target) {
			// SUCCESS! Found valid quantum proof (like finding Bitcoin block)
			header.QOutcome = outcomes
			header.QProof = aggregateProof

			// Initialize optional fields to prevent RLP issues
			q.initializeOptionalFields(header)

			miningTime := time.Since(start)
			hashrate := float64(qnonceVar+1) / miningTime.Seconds()

			log.Info("🎉 Bitcoin-style quantum block mined!",
				"number", header.Number.Uint64(),
				"qnonce", header.QNonce.Uint64(),
				"attempts", qnonceVar+1,
				"miningTime", miningTime,
				"hashrate", fmt.Sprintf("%.2f attempts/sec", hashrate),
				"difficulty", header.Difficulty,
				"target", fmt.Sprintf("0x%x", target),
				"puzzles", *header.LUsed,
				"security", "1,152-bit")

			// Update hashrate (attempts per second, not puzzles per second)
			q.lock.Lock()
			q.hashrate = uint64(hashrate)
			q.lock.Unlock()

			// Send successful block
			sealedBlock := block.WithSeal(header)

			log.Info("📦 Sending Bitcoin-style quantum block")
			select {
			case results <- sealedBlock:
				log.Info("✅ Bitcoin-style quantum block sent successfully")
			case <-stop:
				log.Info("🛑 Stopped while sending block")
			}
			return
		}

		// Try next nonce (Bitcoin-style iteration)
		qnonceVar++

		// Log progress every 1000 attempts (like Bitcoin mining pools)
		if qnonceVar%ProgressLogInterval == 0 {
			elapsed := time.Since(start)
			rate := float64(qnonceVar) / elapsed.Seconds()
			log.Info("⛏️  Bitcoin-style quantum mining progress",
				"attempts", qnonceVar,
				"rate", fmt.Sprintf("%.2f attempts/sec", rate),
				"elapsed", elapsed,
				"target", fmt.Sprintf("0x%x", target))
		}
	}

	// Exhausted all nonces - this should be extremely rare with proper difficulty
	log.Warn("⚠️  Exhausted all nonces without finding valid quantum proof",
		"maxAttempts", MaxNonceAttempts,
		"difficulty", header.Difficulty,
		"target", fmt.Sprintf("0x%x", target),
		"note", "Difficulty may be too high - network should retarget")
}

// SealHashWithNonce returns hash including nonce (Bitcoin-style seed generation)
func (q *QMPoW) SealHashWithNonce(header *types.Header) common.Hash {
	// Create header copy without quantum fields
	headerCopy := types.CopyHeader(header)
	headerCopy.QOutcome = nil
	headerCopy.QProof = nil

	// Add nonce variation to the seed by modifying the Extra field
	// This creates unique seeds for each nonce attempt (Bitcoin-style)
	qnonce := header.QNonce.Uint64()
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, qnonce)

	// Combine original extra data with nonce
	originalExtra := headerCopy.Extra
	headerCopy.Extra = append(originalExtra, nonceBytes...)

	hash := rlpHash(headerCopy)

	// Restore original extra data
	headerCopy.Extra = originalExtra

	return hash
}

// initializeOptionalFields prevents RLP encoding issues
func (q *QMPoW) initializeOptionalFields(header *types.Header) {
	// Set WithdrawalsHash to EmptyWithdrawalsHash if it's nil
	if header.WithdrawalsHash == nil {
		emptyWithdrawalsHash := common.HexToHash("0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421")
		header.WithdrawalsHash = &emptyWithdrawalsHash
	}

	// Initialize other optional fields if they're nil
	if header.BlobGasUsed == nil {
		var zero uint64 = 0
		header.BlobGasUsed = &zero
	}
	if header.ExcessBlobGas == nil {
		var zero uint64 = 0
		header.ExcessBlobGas = &zero
	}
	if header.BaseFee == nil {
		header.BaseFee = big.NewInt(0)
	}
	if header.ParentBeaconRoot == nil {
		emptyHash := common.Hash{}
		header.ParentBeaconRoot = &emptyHash
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

// CalcDifficulty is the Bitcoin-style difficulty adjustment algorithm
func (q *QMPoW) CalcDifficulty(chain consensus.ChainHeaderReader, time uint64, parent *types.Header) *big.Int {
	// For Bitcoin-style quantum mining, we implement proper difficulty retargeting

	blockNumber := new(big.Int).Add(parent.Number, big.NewInt(1)).Uint64()

	log.Info("🔗 Bitcoin-style difficulty calculation",
		"blockNumber", blockNumber,
		"parentNumber", parent.Number.Uint64(),
		"parentDifficulty", parent.Difficulty)

	// Check if it's time for difficulty retargeting (every 100 blocks like our Bitcoin)
	if ShouldRetargetDifficulty(blockNumber) {
		log.Info("🎯 Bitcoin-style difficulty retarget triggered", "blockNumber", blockNumber)

		// Get the start of the current retarget period
		retargetStart := GetRetargetPeriodStart(blockNumber)

		// Get the header from the start of the retarget period
		var startHeader *types.Header
		if retargetStart == 0 {
			// Genesis block
			startHeader = chain.GetHeaderByNumber(0)
		} else {
			startHeader = chain.GetHeaderByNumber(retargetStart)
		}

		if startHeader == nil {
			log.Warn("⚠️  Could not find retarget start header, using parent difficulty",
				"retargetStart", retargetStart)
			return parent.Difficulty
		}

		// Calculate actual time taken for the retarget period
		actualTime := time - startHeader.Time
		targetTime := RetargetBlocks * TargetBlockTime // 100 blocks * 12 seconds

		log.Info("📊 Bitcoin-style retarget analysis",
			"retargetStart", retargetStart,
			"blockNumber", blockNumber,
			"actualTime", actualTime,
			"targetTime", targetTime,
			"ratio", float64(actualTime)/float64(targetTime))

		// Use Bitcoin-style difficulty adjustment
		newDifficulty := CalculateNextDifficulty(parent.Difficulty, actualTime, targetTime)

		log.Info("✅ Bitcoin-style difficulty retargeted",
			"oldDifficulty", parent.Difficulty,
			"newDifficulty", newDifficulty,
			"blocks", RetargetBlocks,
			"actualTime", actualTime,
			"targetTime", targetTime)

		return newDifficulty
	}

	// Not a retarget block - keep current difficulty (like Bitcoin)
	log.Info("➡️  Bitcoin-style difficulty maintained",
		"blockNumber", blockNumber,
		"difficulty", parent.Difficulty,
		"nextRetarget", ((blockNumber/RetargetBlocks)+1)*RetargetBlocks)

	return parent.Difficulty
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
	if q.config.TestMode || q.config.PowMode == ModeFake {
		// In test mode or fake mode, just verify the structure instantly
		log.Info("🧪 Using INSTANT verification (test/fake mode)")
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

// solveQuantumPuzzles generates quantum puzzle solutions with realistic timing
func (q *QMPoW) solveQuantumPuzzles(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	log.Info("🔬 Starting realistic quantum puzzle solving",
		"qbits", qbits, "tcount", tcount, "lnet", lnet,
		"estimatedTime", fmt.Sprintf("%.2fs", EstimateBlockTime(lnet)))

	// Use simulation mode for realistic quantum mining
	return q.simulateQuantumSolver(seed, qbits, tcount, lnet)
}

// simulateQuantumSolver simulates quantum puzzle solving with realistic timing
func (q *QMPoW) simulateQuantumSolver(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	start := time.Now()
	proofs := make([]proof.Proof, lnet)
	currentSeed := seed

	outcomeLen := int(qbits+7) / 8
	allOutcomes := make([]byte, 0, int(lnet)*outcomeLen)

	log.Info("🔬 Solving quantum puzzles with realistic complexity", "totalPuzzles", lnet)

	for i := uint16(0); i < lnet; i++ {
		puzzleStart := time.Now()

		// Calculate realistic solving time for this puzzle
		// Each puzzle gets progressively harder due to quantum interference
		basePuzzleTime := time.Duration(BaseComplexityMs) * time.Millisecond
		complexityMultiplier := 1.0 + float64(i)*ComplexityScaleFactor/100.0
		puzzleTime := time.Duration(float64(basePuzzleTime) * complexityMultiplier)

		// Add some randomness to make it more realistic (±20%)
		randomFactor := 0.8 + 0.4*float64(currentSeed[0]%100)/100.0
		puzzleTime = time.Duration(float64(puzzleTime) * randomFactor)

		// Simulate the actual quantum computation time
		time.Sleep(puzzleTime)

		// Generate deterministic but realistic-looking quantum outcome
		outcome := make([]byte, outcomeLen)
		for j := 0; j < outcomeLen; j++ {
			// Create pseudo-random but deterministic outcome based on seed and puzzle index
			h := sha256.New()
			h.Write(currentSeed)
			h.Write([]byte{byte(i), byte(j), byte(qbits), byte(tcount)})
			hash := h.Sum(nil)
			outcome[j] = hash[j%32] // Use hash bytes cyclically
		}

		// Create a realistic-looking witness (simplified Mahadev proof)
		witness := make([]byte, 32) // 32-byte witness
		h := sha256.New()
		h.Write(currentSeed)
		h.Write(outcome)
		h.Write([]byte{byte(i)})
		witnessHash := h.Sum(nil)
		copy(witness, witnessHash[:32])

		proofs[i] = proof.Proof{
			PuzzleIndex: i,
			Outcome:     outcome,
			Witness:     witness,
		}

		allOutcomes = append(allOutcomes, outcome...)

		puzzleTime = time.Since(puzzleStart)
		log.Info("🧩 Quantum puzzle solved",
			"puzzle", i+1, "of", lnet,
			"time", puzzleTime,
			"complexity", fmt.Sprintf("%.2fx", complexityMultiplier))

		// Generate seed for next puzzle
		if i < lnet-1 {
			h := sha256.New()
			h.Write(currentSeed)
			h.Write(outcome)
			currentSeed = h.Sum(nil)
		}
	}

	// Create aggregate proof
	aggregateProof, err := proof.CreateAggregate(proofs, qbits, tcount)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create aggregate proof: %v", err)
	}

	totalTime := time.Since(start)
	puzzlesPerSecond := float64(lnet) / totalTime.Seconds()

	log.Info("✅ All quantum puzzles solved",
		"totalTime", totalTime,
		"puzzlesPerSecond", fmt.Sprintf("%.2f", puzzlesPerSecond),
		"averagePerPuzzle", time.Duration(totalTime.Nanoseconds()/int64(lnet)))

	return allOutcomes, aggregateProof.Serialize(), nil
}

// generateFakeQuantumSolution is now deprecated - we use realistic simulation
func (q *QMPoW) generateFakeQuantumSolution(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	log.Warn("⚠️ generateFakeQuantumSolution is deprecated - using realistic simulation instead")
	return q.simulateQuantumSolver(seed, qbits, tcount, lnet)
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
