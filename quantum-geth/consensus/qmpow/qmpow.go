// Copyright 2025 Quantum-Geth Authors
// This file is part of the quantum-geth library.

// Package qmpow implements the Quantum-Geth v0.9-rc3-hw0 quantum proof-of-work consensus engine.
// Unified, Branch-Serial Quantum Proof-of-Work — Canonical-Compile Edition
package qmpow

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math/big"
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
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/holiman/uint256"
)

// Quantum-Geth v0.9–BareBones+Halving Constants
const (
	// Halving epoch parameters (Section 11)
	EpochBlocks   = HalvingEpochLength // Epoch = ⌊Height / 600,000⌋
	StartingQBits = 16                 // Start n = 16 at epoch 0 (126-bit security)

	// Dynamic circuit parameters (follow glide schedule from params.go)
	// These are now calculated dynamically based on height

	// Proof system sizes
	OutcomeRootSize   = 32 // Merkle root of outcomes
	BranchNibblesSize = 48 // One high-nibble per puzzle
	GateHashSize      = 32 // SHA-256 of gate streams
	ProofRootSize     = 32 // Merkle root of Nova proofs
	ExtraNonce32Size  = 32 // Entropy field size

	// Attestation modes
	AttestModeDilithium = 0x00 // Deterministic Dilithium attestation
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

	// ErrInvalidEpoch is returned when epoch calculation is wrong
	ErrInvalidEpoch = errors.New("invalid epoch value")

	// ErrInvalidQBits is returned when qubits don't match glide schedule
	ErrInvalidQBits = errors.New("invalid qubits for height")

	// ErrInvalidGateHash is returned when gate hash verification fails
	ErrInvalidGateHash = errors.New("invalid gate hash")

	// ErrInvalidBranchNibbles is returned when branch nibbles are invalid
	ErrInvalidBranchNibbles = errors.New("invalid branch nibbles")

	// errQMPoWStopped is returned when the consensus engine is stopped
	errQMPoWStopped = errors.New("qmpow stopped")

	// errNoMiningWork is returned when no mining work is available
	errNoMiningWork = errors.New("no mining work available")

	// errInvalidSealResult is returned when the submitted seal is invalid
	errInvalidSealResult = errors.New("invalid or stale proof of work")
)

// Remote sealer related types
type sealWork struct {
	errc chan error
	res  chan [5]string
}

type quantumMineResult struct {
	qnonce       uint64
	blockHash    common.Hash
	quantumProof QuantumProofSubmission
	errc         chan error
}

type hashrate struct {
	done chan struct{}
	rate uint64
	id   common.Hash
	ping time.Time
}

// QuantumProofSubmission represents a quantum proof submission from external miners
type QuantumProofSubmission struct {
	OutcomeRoot   common.Hash `json:"outcome_root"`
	GateHash      common.Hash `json:"gate_hash"`
	ProofRoot     common.Hash `json:"proof_root"`
	BranchNibbles []byte      `json:"branch_nibbles"`
	ExtraNonce32  []byte      `json:"extra_nonce32"`
}

// remoteSealer wraps the QMPoW to allow remote quantum mining
type remoteSealer struct {
	qmpow        *QMPoW
	works        map[common.Hash]*types.Block
	rates        map[common.Hash]hashrate
	currentWork  [5]string
	currentBlock *types.Block
	results      chan<- *types.Block
	fetchWorkCh  chan *sealWork
	submitWorkCh chan *quantumMineResult
	submitRateCh chan *hashrate
	requestExit  chan struct{}
	exitCh       chan struct{}
	
	// For automatic work preparation
	chain       consensus.ChainHeaderReader
	workReadyCh chan struct{}
	mu          sync.RWMutex
}

// QMPoW is the v0.9-rc3-hw0 quantum mining engine
type QMPoW struct {
	config   Config
	threads  int
	update   chan struct{}
	hashrate uint64

	// Block assembly integration
	blockAssembler *BlockAssembler
	lastPublicKey  []byte
	lastSignature  []byte

	// Halving & fee model
	halvingModel *HalvingFeeModel

	// Remote mining support
	remote *remoteSealer

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
		threads: 1,
		update:  make(chan struct{}),
	}

	// Initialize remote mining support
	qmpow.remote = &remoteSealer{
		qmpow:        qmpow,
		works:        make(map[common.Hash]*types.Block),
		rates:        make(map[common.Hash]hashrate),
		fetchWorkCh:  make(chan *sealWork),
		submitWorkCh: make(chan *quantumMineResult),
		submitRateCh: make(chan *hashrate),
		requestExit:  make(chan struct{}),
		exitCh:       make(chan struct{}),
		workReadyCh:  make(chan struct{}, 1),
	}

	// Start remote sealer
	go qmpow.remote.loop()

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
	if header.Epoch == nil || header.QBits == nil || header.TCount == nil || header.LNet == nil {
		return ErrMissingQuantumFields
	}

	// Verify epoch calculation: Epoch = ⌊Height / 50,000⌋
	expectedEpoch := uint32(header.Number.Uint64() / EpochBlocks)
	if *header.Epoch != expectedEpoch {
		return fmt.Errorf("%w: got %d, expected %d", ErrInvalidEpoch, *header.Epoch, expectedEpoch)
	}

	// Verify QBits according to glide schedule
	expectedQBits, _, _ := CalculateQuantumParamsForHeight(header.Number.Uint64())
	if *header.QBits != expectedQBits {
		return fmt.Errorf("%w: got %d, expected %d", ErrInvalidQBits, *header.QBits, expectedQBits)
	}

	// Verify parameters according to glide schedule
	_, expectedTCount, expectedLNet := CalculateQuantumParamsForHeight(header.Number.Uint64())
	if *header.TCount != expectedTCount {
		return fmt.Errorf("invalid TCount: got %d, expected %d", *header.TCount, expectedTCount)
	}

	if *header.LNet != expectedLNet {
		return fmt.Errorf("invalid LNet: got %d, expected %d", *header.LNet, expectedLNet)
	}

	// Verify field sizes
	if len(header.ExtraNonce32) != ExtraNonce32Size {
		return fmt.Errorf("invalid ExtraNonce32 size: got %d, expected %d",
			len(header.ExtraNonce32), ExtraNonce32Size)
	}

	if len(header.BranchNibbles) != BranchNibblesSize {
		return fmt.Errorf("invalid BranchNibbles size: got %d, expected %d",
			len(header.BranchNibbles), BranchNibblesSize)
	}

	// If we're not verifying the seal, skip proof verification
	if !seal {
		return nil
	}

	// Verify quantum proof according to v0.9-rc3-hw0 specification
	log.Debug("🔍 Quantum-Geth v0.9-rc3-hw0 proof verification",
		"blockNumber", header.Number.Uint64(),
		"epoch", *header.Epoch,
		"qbits", *header.QBits,
		"difficulty", header.Difficulty)

	return q.verifyQuantumProofMain(header)
}

// verifyQuantumProofMain verifies quantum proof according to specification
func (q *QMPoW) verifyQuantumProofMain(header *types.Header) error {
	// TODO: Implement full quantum verification
	// For now, use simplified verification for development
	if q.config.TestMode || q.config.PowMode == ModeFake {
		log.Info("🧪 Using simplified verification (test mode)")
		return q.verifyQuantumProofStructureMain(header)
	}

	// Full verification would include:
	// 1. Seed chain validation
	// 2. Branch-dependent template selection
	// 3. Canonical compiler verification
	// 4. Tier-A/B/C proof stack validation
	// 5. Dilithium attestation verification

	log.Warn("⚠️ Full quantum verification not yet implemented - using simplified mode")
	return q.verifyQuantumProofStructureMain(header)
}

// verifyQuantumProofStructureMain verifies structure according to quantum specification
func (q *QMPoW) verifyQuantumProofStructureMain(header *types.Header) error {
	// Verify required fields are present
	if header.OutcomeRoot == nil {
		return fmt.Errorf("missing OutcomeRoot")
	}
	if header.GateHash == nil {
		return fmt.Errorf("missing GateHash")
	}
	if header.ProofRoot == nil {
		return fmt.Errorf("missing ProofRoot")
	}

	log.Debug("✅ Quantum structure verification passed",
		"blockNumber", header.Number.Uint64(),
		"epoch", *header.Epoch,
		"qbits", *header.QBits)

	return nil
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
	log.Info("🎯 QMPoW Prepare called (v0.9-rc3-hw0)", "blockNumber", header.Number.Uint64(), "parentHash", header.ParentHash.Hex())

	params := q.ParamsForHeight(header.Number.Uint64())

	// Set quantum parameters according to v0.9-rc3-hw0 specification
	header.Epoch = &params.Epoch
	header.QBits = &params.QBits
	header.TCount = &params.TCount
	header.LNet = &params.LNet // Always 48 puzzles for 1,152-bit security

	// Initialize quantum nonce
	var qnonce64 uint64 = 0
	header.QNonce64 = &qnonce64

	// Initialize entropy field
	header.ExtraNonce32 = make([]byte, ExtraNonce32Size)

	// Initialize branch nibbles
	header.BranchNibbles = make([]byte, BranchNibblesSize)

	// Set attestation mode
	var attestMode uint8 = AttestModeDilithium
	header.AttestMode = &attestMode

	// Clear quantum fields that will be filled during sealing
	header.OutcomeRoot = nil
	header.GateHash = nil
	header.ProofRoot = nil

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
		header.Difficulty = big.NewInt(1000) // Match genesis.json difficulty
		log.Info("🌱 Genesis block difficulty set (Bitcoin-style)",
			"difficulty", header.Difficulty,
			"fixedPuzzles", params.LNet,
			"security", "1,152-bit")
	}

	// Initialize optional fields to prevent RLP encoding issues
	q.initializeOptionalFields(header)

	// Marshal quantum fields into QBlob for proper RLP encoding
	header.MarshalQuantumBlob()

	return nil
}

// Finalize runs any post-transaction state modifications
func (q *QMPoW) Finalize(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB,
	txs []*types.Transaction, uncles []*types.Header, withdrawals []*types.Withdrawal) {

	// Create halving fee model if not exists
	if q.halvingModel == nil {
		q.halvingModel = NewHalvingFeeModel(chain)
	}

	// For now, use simplified reward calculation without receipts
	// In production, this would get receipts from the block processing
	blockNumber := header.Number.Uint64()
	epoch := uint32(blockNumber / HalvingEpochSize)
	subsidyQGC := CalculateBlockSubsidy(epoch)
	// Convert QGC to wei (1 QGC = 10^18 wei)
	subsidyBig := new(big.Float).SetFloat64(subsidyQGC)
	weiBig := new(big.Float).Mul(subsidyBig, big.NewFloat(1e18))
	subsidyWei, _ := weiBig.Int(nil)

	// Calculate transaction fees (simplified without receipts)
	totalFees := big.NewInt(0)
	for _, tx := range txs {
		gasPrice := tx.GasPrice()
		if gasPrice != nil {
			// Estimate fee = gasPrice * gasLimit (simplified)
			txFee := new(big.Int).Mul(gasPrice, big.NewInt(int64(tx.Gas())))
			totalFees.Add(totalFees, txFee)
		}
	}

	// Total reward = subsidy + fees
	totalReward := new(big.Int).Add(subsidyWei, totalFees)

	// Award total reward to coinbase
	state.AddBalance(header.Coinbase, uint256.MustFromBig(totalReward))

	// Handle uncle rewards (if any)
	for _, uncle := range uncles {
		// Uncle reward is 1/32 of block subsidy (following Ethereum tradition)
		uncleReward := new(big.Int).Div(subsidyWei, big.NewInt(32))
		state.AddBalance(uncle.Coinbase, uint256.MustFromBig(uncleReward))
	}

	// Log halving events
	if blockNumber > 0 && blockNumber%HalvingEpochSize == 0 {
		prevSubsidy := CalculateBlockSubsidy(epoch - 1)
		log.Warn("🎉 HALVING EVENT!",
			"blockNumber", blockNumber,
			"epoch", epoch,
			"previousSubsidy", prevSubsidy,
			"newSubsidy", subsidyQGC,
			"reductionFactor", "2x")
	}

	log.Info("💰 Block reward applied",
		"blockNumber", blockNumber,
		"epoch", epoch,
		"subsidyQGC", subsidyQGC,
		"subsidyWei", subsidyWei,
		"transactionFees", totalFees,
		"totalReward", totalReward,
		"coinbase", header.Coinbase.Hex())
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
	// If remote mining is enabled, set up the work for external miners
	if q.remote != nil {
		q.remote.submitWork(block, results)
	}
	
	// Also start local mining in a separate goroutine
	go q.seal(chain, block, results, stop)
	return nil
}

// seal is the v0.9-rc3-hw0 quantum mining function
// This implements the unified, branch-serial quantum proof-of-work
func (q *QMPoW) seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) {
	header := types.CopyHeader(block.Header())

	// Initialize quantum fields
	q.initializeQuantumFields(header)

	log.Info("🔬 Starting quantum mining",
		"number", header.Number.Uint64(),
		"epoch", *header.Epoch,
		"qbits", *header.QBits,
		"puzzles", *header.LNet,
		"difficulty", header.Difficulty)

	start := time.Now()

	// Bitcoin-style nonce iteration
	for qnonce := uint64(0); qnonce <= MaxNonceAttempts; qnonce++ {
		// Check if we should stop
		select {
		case <-stop:
			log.Info("🛑 Quantum mining stopped", "attempts", qnonce)
			return
		default:
		}

		// Set QNonce64 for this attempt
		*header.QNonce64 = qnonce

		// Solve quantum puzzles
		err := q.SolveQuantumPuzzles(header)
		if err != nil {
			log.Error("❌ Failed to solve quantum puzzles", "qnonce", qnonce, "err", err)
			continue
		}

		// Check if proof meets target
		if q.checkQuantumTarget(header) {
			// SUCCESS! Found valid quantum proof
			miningTime := time.Since(start)
			hashrate := float64(qnonce+1) / miningTime.Seconds()

			log.Info("🎉 Quantum block mined!",
				"number", header.Number.Uint64(),
				"epoch", *header.Epoch,
				"qnonce", qnonce,
				"attempts", qnonce+1,
				"miningTime", miningTime,
				"hashrate", fmt.Sprintf("%.2f attempts/sec", hashrate),
				"qbits", *header.QBits,
				"puzzles", *header.LNet)

			// Update hashrate
			q.lock.Lock()
			q.hashrate = uint64(hashrate)
			q.lock.Unlock()

			// Send successful block
			sealedBlock := block.WithSeal(header)
			select {
			case results <- sealedBlock:
				log.Info("✅ Quantum block sent successfully")
			case <-stop:
				log.Info("🛑 Stopped while sending block")
			}
			return
		}

		// Log progress every 1000 attempts
		if qnonce%1000 == 0 && qnonce > 0 {
			elapsed := time.Since(start)
			rate := float64(qnonce) / elapsed.Seconds()
			log.Info("⛏️  Quantum mining progress",
				"attempts", qnonce,
				"rate", fmt.Sprintf("%.2f attempts/sec", rate),
				"elapsed", elapsed)
		}
	}

	// Exhausted all nonces
	log.Warn("⚠️  Exhausted all nonces without finding valid quantum proof",
		"maxAttempts", MaxNonceAttempts,
		"difficulty", header.Difficulty)
}

// SealHashWithNonce returns hash including nonce (v0.9-rc3-hw0 seed generation)
func (q *QMPoW) SealHashWithNonce(header *types.Header) common.Hash {
	// Create header copy for seed generation
	headerCopy := types.CopyHeader(header)

	// Clear quantum proof fields that will be calculated
	headerCopy.OutcomeRoot = nil
	headerCopy.GateHash = nil
	headerCopy.ProofRoot = nil

	// Include nonce in the seed calculation
	hash := rlpHash(headerCopy)
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
	// Create a copy of the header without quantum proof fields
	headerCopy := types.CopyHeader(header)
	headerCopy.OutcomeRoot = nil
	headerCopy.GateHash = nil
	headerCopy.ProofRoot = nil

	return rlpHash(headerCopy)
}

// CalcDifficulty is the Bitcoin-style difficulty adjustment algorithm
func (q *QMPoW) CalcDifficulty(chain consensus.ChainHeaderReader, time uint64, parent *types.Header) *big.Int {
	// Simplified Bitcoin-style difficulty retargeting every 100 blocks
	blockNumber := new(big.Int).Add(parent.Number, big.NewInt(1)).Uint64()

	// Use parent difficulty directly (no complex conversions)
	parentDifficulty := parent.Difficulty

	log.Info("🔗 Quantum difficulty calculation",
		"blockNumber", blockNumber,
		"parentDifficulty", FormatDifficulty(parentDifficulty))

	// Check if it's time for difficulty retargeting (every 100 blocks)
	if blockNumber > 0 && blockNumber%100 == 0 {
		log.Info("🎯 Difficulty retarget triggered", "blockNumber", blockNumber)

		// Get the header from 100 blocks ago
		retargetStart := blockNumber - 100
		var startHeader *types.Header
		if retargetStart == 0 {
			startHeader = chain.GetHeaderByNumber(0) // Genesis
		} else {
			startHeader = chain.GetHeaderByNumber(retargetStart)
		}

		if startHeader == nil {
			log.Warn("⚠️  Could not find retarget start header, maintaining difficulty")
			return parentDifficulty
		}

		// Calculate actual time for the last 100 blocks
		actualTime := time - startHeader.Time
		targetTime := uint64(100 * 12) // 100 blocks * 12 seconds = 1200 seconds

		log.Info("📊 Retarget analysis",
			"retargetStart", retargetStart,
			"actualTime", actualTime,
			"targetTime", targetTime,
			"ratio", float64(actualTime)/float64(targetTime))

		// Simple difficulty adjustment: newDiff = oldDiff * targetTime / actualTime
		if actualTime == 0 {
			actualTime = 1 // Prevent division by zero
		}

		newDifficulty := new(big.Int).Set(parentDifficulty)
		newDifficulty.Mul(newDifficulty, big.NewInt(int64(targetTime)))
		newDifficulty.Div(newDifficulty, big.NewInt(int64(actualTime)))

		// Clamp adjustment to 4x max change (like Bitcoin)
		maxIncrease := new(big.Int).Set(parentDifficulty)
		maxIncrease.Mul(maxIncrease, big.NewInt(4))

		maxDecrease := new(big.Int).Set(parentDifficulty)
		maxDecrease.Div(maxDecrease, big.NewInt(4))

		if newDifficulty.Cmp(maxIncrease) > 0 {
			newDifficulty.Set(maxIncrease)
			log.Info("🔒 Difficulty increase clamped to 4x")
		} else if newDifficulty.Cmp(maxDecrease) < 0 {
			newDifficulty.Set(maxDecrease)
			log.Info("🔒 Difficulty decrease clamped to 4x")
		}

		// Ensure minimum difficulty
		minDiff := big.NewInt(500) // 0.0005 in fixed-point
		if newDifficulty.Cmp(minDiff) < 0 {
			newDifficulty.Set(minDiff)
		}

		log.Info("✅ Difficulty retargeted",
			"oldDifficulty", FormatDifficulty(parentDifficulty),
			"newDifficulty", FormatDifficulty(newDifficulty),
			"actualMinutes", float64(actualTime)/60,
			"targetMinutes", float64(targetTime)/60)

		return newDifficulty
	}

	// Not a retarget block - keep current difficulty
	nextRetarget := ((blockNumber / 100) + 1) * 100
	log.Info("➡️  Difficulty maintained",
		"blockNumber", blockNumber,
		"difficulty", FormatDifficulty(parentDifficulty),
		"nextRetarget", nextRetarget)

	return parentDifficulty
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
		{
			Namespace: "eth",
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
// verifyQuantumProof is deprecated - use verifyQuantumProofMain instead
func (q *QMPoW) verifyQuantumProof(chain consensus.ChainHeaderReader, header *types.Header) error {
	// Redirect to main quantum verification
	return q.verifyQuantumProofMain(header)
}

// verifyQuantumProofStructure is deprecated - use ValidateQuantumHeader instead
func (q *QMPoW) verifyQuantumProofStructure(header *types.Header) error {
	// Redirect to quantum validation
	return ValidateQuantumHeader(header)
}

// solveQuantumPuzzles is deprecated - use SolveQuantumPuzzles instead
func (q *QMPoW) solveQuantumPuzzles(seed []byte, qbits uint8, tcount uint16, lnet uint16) ([]byte, []byte, error) {
	log.Warn("⚠️ solveQuantumPuzzles is deprecated - use SolveQuantumPuzzles instead")
	return nil, nil, fmt.Errorf("deprecated function - use quantum implementation")
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
		"qbits":        params.QBits,
		"tcount":       params.TCount,
		"lnet":         params.LNet,
		"epoch":        params.Epoch,
		"blockSubsidy": params.BlockSubsidy,
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
		"halvingEpoch":    params.Epoch,
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
		"halvingEpoch":        params.Epoch,
		"quantumComplexity": map[string]interface{}{
			"qubits":     params.QBits,
			"tgates":     params.TCount,
			"stateSpace": uint64(1) << params.QBits, // 2^qubits
		},
	}
}

// GetQuantumStats returns comprehensive quantum mining statistics for monitoring
func (api *API) GetQuantumStats() map[string]interface{} {
	api.qmpow.lock.RLock()
	defer api.qmpow.lock.RUnlock()

	params := api.qmpow.ParamsForHeight(0)
	hashrate := float64(api.qmpow.hashrate)

	return map[string]interface{}{
		"mining": map[string]interface{}{
			"hashrate":     hashrate,
			"hashrateUnit": "puzzles/second",
			"threads":      api.qmpow.threads,
			"isActive":     hashrate > 0,
		},
		"difficulty": map[string]interface{}{
			"puzzlesPerBlock": params.LNet,
			"qubitsPerPuzzle": params.QBits,
			"tgatesPerPuzzle": params.TCount,
			"totalComplexity": uint64(params.LNet) * uint64(params.QBits) * uint64(params.TCount),
		},
		"network": map[string]interface{}{
			"targetBlockTime":     TargetBlockTime,
			"currentEpoch":        params.Epoch,
			"nextHalvingAt":       (params.Epoch + 1) * EpochBlocks,
			"currentSubsidy":      params.BlockSubsidy,
			"theoreticalHashrate": float64(params.LNet) / float64(TargetBlockTime),
		},
		"quantum": map[string]interface{}{
			"effectiveSecurityBits": CalculateEffectiveSecurityBits(params.QBits, params.LNet),
			"stateSpaceSize":        uint64(1) << params.QBits,
			"quantumAdvantage":      "sqrt(2^n) speedup over classical",
			"proofSystem":           "Mahadev→CAPSS→Nova-Lite",
		},
	}
}

// GetWork returns a work package for external quantum miners.
//
// The work package consists of 5 strings:
//	result[0] - 32 bytes hex encoded current block header hash (without quantum fields)
//	result[1] - 32 bytes hex encoded block number
//	result[2] - 32 bytes hex encoded difficulty target
//	result[3] - hex encoded quantum parameters (qbits, tcount, lnet)
//	result[4] - hex encoded coinbase address
func (api *API) GetWork() ([5]string, error) {
	if api.qmpow.remote == nil {
		return [5]string{}, errors.New("remote mining not supported - start geth with --mine")
	}

	var (
		workCh = make(chan [5]string, 1)
		errc   = make(chan error, 1)
	)

	select {
	case api.qmpow.remote.fetchWorkCh <- &sealWork{errc: errc, res: workCh}:
	case <-api.qmpow.remote.exitCh:
		return [5]string{}, errQMPoWStopped
	}

	select {
	case work := <-workCh:
		return work, nil
	case err := <-errc:
		return [5]string{}, err
	}
}

// SubmitWork can be used by external quantum miners to submit their solution.
// It returns an indication if the work was accepted.
func (api *API) SubmitWork(qnonce uint64, blockHash common.Hash, quantumProof QuantumProofSubmission) bool {
	if api.qmpow.remote == nil {
		return false
	}

	var errc = make(chan error, 1)
	solution := &quantumMineResult{
		qnonce:      qnonce,
		blockHash:   blockHash,
		quantumProof: quantumProof,
		errc:        errc,
	}

	select {
	case api.qmpow.remote.submitWorkCh <- solution:
	case <-api.qmpow.remote.exitCh:
		return false
	}

	err := <-errc
	return err == nil
}

// SubmitHashrate can be used for remote quantum miners to submit their hash rate.
func (api *API) SubmitHashrate(rate uint64, id common.Hash) bool {
	if api.qmpow.remote == nil {
		return false
	}

	var done = make(chan struct{}, 1)
	select {
	case api.qmpow.remote.submitRateCh <- &hashrate{done: done, rate: rate, id: id}:
	case <-api.qmpow.remote.exitCh:
		return false
	}

	<-done
	return true
}

// Remote sealer implementation

// loop handles requests from external quantum miners
func (s *remoteSealer) loop() {
	defer close(s.exitCh)

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	workTicker := time.NewTicker(15 * time.Second) // Prepare work every 15 seconds
	defer workTicker.Stop()

	for {
		select {
		case work := <-s.fetchWorkCh:
			// Return current mining work to external quantum miner
			s.mu.RLock()
			currentBlock := s.currentBlock
			currentWork := s.currentWork
			s.mu.RUnlock()
			
			if currentBlock == nil {
				// Try to prepare work automatically
				s.tryPrepareWork()
				s.mu.RLock()
				if s.currentBlock == nil {
					s.mu.RUnlock()
					work.errc <- errNoMiningWork
				} else {
					work.res <- s.currentWork
					s.mu.RUnlock()
				}
			} else {
				work.res <- currentWork
			}

		case result := <-s.submitWorkCh:
			// Verify submitted quantum proof
			if s.submitQuantumWork(result.qnonce, result.blockHash, result.quantumProof) {
				result.errc <- nil
			} else {
				result.errc <- errInvalidSealResult
			}

		case result := <-s.submitRateCh:
			// Track hash rate from external quantum miners
			s.rates[result.id] = hashrate{rate: result.rate, ping: time.Now()}
			close(result.done)

		case <-ticker.C:
			// Clean up stale data
			for id, rate := range s.rates {
				if time.Since(rate.ping) > 10*time.Second {
					delete(s.rates, id)
				}
			}

		case <-workTicker.C:
			// Automatically prepare work for external miners
			s.tryPrepareWork()

		case <-s.requestExit:
			return
		}
	}
}

// submitWork prepares work for external quantum miners
func (s *remoteSealer) submitWork(block *types.Block, results chan<- *types.Block) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.currentBlock = block
	s.results = results
	s.makeQuantumWorkUnsafe(block)
}

// makeQuantumWork creates a quantum work package for external miners (thread-safe)
func (s *remoteSealer) makeQuantumWork(block *types.Block) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.makeQuantumWorkUnsafe(block)
}

// makeQuantumWorkUnsafe creates a quantum work package for external miners (requires lock)
func (s *remoteSealer) makeQuantumWorkUnsafe(block *types.Block) {
	header := block.Header()
	
	// Initialize quantum fields
	s.qmpow.initializeQuantumFields(header)
	
	// Calculate work hash (header without quantum proof fields)
	workHash := s.qmpow.SealHash(header)
	
	// Calculate target from difficulty
	target := DifficultyToTarget(header.Difficulty)
	
	// Prepare quantum parameters
	params := fmt.Sprintf("qbits:%d,tcount:%d,lnet:%d", 
		*header.QBits, *header.TCount, *header.LNet)
	
	s.currentWork[0] = workHash.Hex()                           // Work hash
	s.currentWork[1] = hexutil.EncodeBig(header.Number)         // Block number
	s.currentWork[2] = target.Text(16)                          // Difficulty target
	s.currentWork[3] = hexutil.Encode([]byte(params))           // Quantum parameters
	s.currentWork[4] = header.Coinbase.Hex()                    // Coinbase address
	
	// Store work for submission verification
	s.works[workHash] = block
	
	log.Info("🔗 Quantum work prepared for external miners",
		"number", header.Number.Uint64(),
		"difficulty", FormatDifficulty(header.Difficulty),
		"qbits", *header.QBits,
		"puzzles", *header.LNet)
}

// submitQuantumWork verifies and processes quantum proof submission
func (s *remoteSealer) submitQuantumWork(qnonce uint64, blockHash common.Hash, quantumProof QuantumProofSubmission) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.currentBlock == nil {
		log.Error("No current work for quantum submission", "hash", blockHash.Hex())
		return false
	}
	
	// Find the work
	block := s.works[blockHash]
	if block == nil {
		log.Warn("Quantum work not found", "hash", blockHash.Hex())
		return false
	}
	
	// Create header with quantum proof
	header := types.CopyHeader(block.Header())
	header.QNonce64 = &qnonce
	header.OutcomeRoot = &quantumProof.OutcomeRoot
	header.GateHash = &quantumProof.GateHash
	header.ProofRoot = &quantumProof.ProofRoot
	
	if len(quantumProof.BranchNibbles) >= BranchNibblesSize {
		copy(header.BranchNibbles, quantumProof.BranchNibbles[:BranchNibblesSize])
	}
	
	if len(quantumProof.ExtraNonce32) >= ExtraNonce32Size {
		copy(header.ExtraNonce32, quantumProof.ExtraNonce32[:ExtraNonce32Size])
	}
	
	// Verify quantum proof meets target
	if !s.qmpow.checkQuantumTarget(header) {
		log.Debug("Quantum proof does not meet target", "qnonce", qnonce)
		return false
	}
	
	// Create sealed block
	sealedBlock := block.WithSeal(header)
	
	// Submit successful block if we have a result channel
	if s.results != nil {
		select {
		case s.results <- sealedBlock:
			log.Info("✅ Quantum block submitted by external miner",
				"number", header.Number.Uint64(),
				"qnonce", qnonce,
				"miner", "external")
			return true
		default:
			log.Warn("Could not submit quantum block - channel full")
			return false
		}
	} else {
		// This was template work - we need to create a proper result channel
		// and try to submit via the miner subsystem
		log.Info("✅ Quantum block found by external miner (template work)",
			"number", header.Number.Uint64(),
			"qnonce", qnonce,
			"miner", "external")
		
		// For template blocks, we don't have a direct submission path
		// The external miner found a valid solution but we need the miner
		// subsystem to be running to handle block submission
		// Return true to indicate the proof is valid
		return true
	}
}

// setChain sets the blockchain reference for remote work preparation
func (s *remoteSealer) setChain(chain consensus.ChainHeaderReader) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chain = chain
	
	// Prepare initial work
	go s.tryPrepareWork()
}

// tryPrepareWork attempts to prepare work for external miners when none is available
func (s *remoteSealer) tryPrepareWork() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	// Skip if we already have recent work
	if s.currentBlock != nil {
		return
	}
	
	// Skip if no chain reference available
	if s.chain == nil {
		return
	}
	
	// Get current head block
	parent := s.chain.CurrentHeader()
	if parent == nil {
		log.Warn("No current header available for work preparation")
		return
	}
	
	// Create a template block for external miners
	header := &types.Header{
		ParentHash: parent.Hash(),
		Number:     new(big.Int).Add(parent.Number, big.NewInt(1)),
		Time:       uint64(time.Now().Unix()),
		GasLimit:   parent.GasLimit,
		Difficulty: s.qmpow.CalcDifficulty(s.chain, uint64(time.Now().Unix()), parent),
		Coinbase:   common.Address{}, // Will be set by external miner
	}
	
	// Prepare the header using QMPoW
	if err := s.qmpow.Prepare(s.chain, header); err != nil {
		log.Error("Failed to prepare header for remote mining", "err", err)
		return
	}
	
	// Create template block
	block := types.NewBlock(header, nil, nil, nil, nil)
	
	// Prepare work for external miners
	s.currentBlock = block
	s.results = nil // No result channel for template work
	s.makeQuantumWorkUnsafe(block)
	
	log.Info("🔗 Quantum work automatically prepared for external miners",
		"number", header.Number.Uint64(),
		"difficulty", FormatDifficulty(header.Difficulty),
		"parentHash", header.ParentHash.Hex()[:10]+"...")
}

// SetChain sets the blockchain reference for remote mining work preparation
func (q *QMPoW) SetChain(chain consensus.ChainHeaderReader) {
	if q.remote != nil {
		q.remote.setChain(chain)
	}
}
