// Copyright 2014 The go-ethereum Authors
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

// Package types contains data types related to Ethereum consensus.
package types

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math/big"
	"reflect"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rlp"
)

// A BlockNonce is a 64-bit hash which proves (combined with the
// mix-hash) that a sufficient amount of computation has been carried
// out on a block.
type BlockNonce [8]byte

// EncodeNonce converts the given integer to a block nonce.
func EncodeNonce(i uint64) BlockNonce {
	var n BlockNonce
	binary.BigEndian.PutUint64(n[:], i)
	return n
}

// Uint64 returns the integer value of a block nonce.
func (n BlockNonce) Uint64() uint64 {
	return binary.BigEndian.Uint64(n[:])
}

// MarshalText encodes n as a hex string with 0x prefix.
func (n BlockNonce) MarshalText() ([]byte, error) {
	return hexutil.Bytes(n[:]).MarshalText()
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (n *BlockNonce) UnmarshalText(input []byte) error {
	return hexutil.UnmarshalFixedText("BlockNonce", input, n[:])
}

// A QuantumNonce is a 64-bit nonce which proves (combined with the
// quantum proof) that a sufficient amount of quantum computation has been carried
// out on a block. This works exactly like Bitcoin's nonce field.
type QuantumNonce [8]byte

// EncodeQuantumNonce converts the given integer to a quantum nonce.
func EncodeQuantumNonce(i uint64) QuantumNonce {
	var n QuantumNonce
	binary.BigEndian.PutUint64(n[:], i)
	return n
}

// Uint64 returns the integer value of a quantum nonce.
func (n QuantumNonce) Uint64() uint64 {
	return binary.BigEndian.Uint64(n[:])
}

// MarshalText encodes n as a hex string with 0x prefix.
func (n QuantumNonce) MarshalText() ([]byte, error) {
	return hexutil.Bytes(n[:]).MarshalText()
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (n *QuantumNonce) UnmarshalText(input []byte) error {
	return hexutil.UnmarshalFixedText("QuantumNonce", input, n[:])
}

// EncodeRLP implements rlp.Encoder.
func (n QuantumNonce) EncodeRLP(w io.Writer) error {
	return rlp.Encode(w, n[:])
}

// DecodeRLP implements rlp.Decoder.
func (n *QuantumNonce) DecodeRLP(s *rlp.Stream) error {
	var b []byte
	if err := s.Decode(&b); err != nil {
		return err
	}
	if len(b) != 8 {
		return fmt.Errorf("invalid QuantumNonce length: got %d, want 8", len(b))
	}
	copy(n[:], b)
	return nil
}

//go:generate go run github.com/fjl/gencodec -type Header -field-override headerMarshaling -out gen_header_json.go

// Quantum-Geth Header Structure
// Implements the unified, branch-serial quantum proof-of-work specification

// Header represents a block header in the Ethereum blockchain.
type Header struct {
	ParentHash  common.Hash    `json:"parentHash"       gencodec:"required"`
	UncleHash   common.Hash    `json:"sha3Uncles"       gencodec:"required"`
	Coinbase    common.Address `json:"miner"`
	Root        common.Hash    `json:"stateRoot"        gencodec:"required"`
	TxHash      common.Hash    `json:"transactionsRoot" gencodec:"required"`
	ReceiptHash common.Hash    `json:"receiptsRoot"     gencodec:"required"`
	Bloom       Bloom          `json:"logsBloom"        gencodec:"required"`
	Difficulty  *big.Int       `json:"difficulty"       gencodec:"required"`
	Number      *big.Int       `json:"number"           gencodec:"required"`
	GasLimit    uint64         `json:"gasLimit"         gencodec:"required"`
	GasUsed     uint64         `json:"gasUsed"          gencodec:"required"`
	Time        uint64         `json:"timestamp"        gencodec:"required"`
	Extra       []byte         `json:"extraData"        gencodec:"required"`
	MixDigest   common.Hash    `json:"mixHash"`
	Nonce       BlockNonce     `json:"nonce"`

	// BaseFee was added by EIP-1559 and is ignored in legacy headers.
	BaseFee *big.Int `json:"baseFeePerGas" rlp:"optional"`

	// WithdrawalsHash was added by EIP-4895 and is ignored in legacy headers.
	WithdrawalsHash *common.Hash `json:"withdrawalsRoot" rlp:"optional"`

	// BlobGasUsed was added by EIP-4844 and is ignored in legacy headers.
	BlobGasUsed *uint64 `json:"blobGasUsed" rlp:"optional"`

	// ExcessBlobGas was added by EIP-4844 and is ignored in legacy headers.
	ExcessBlobGas *uint64 `json:"excessBlobGas" rlp:"optional"`

	// ParentBeaconRoot was added by EIP-4788 and is ignored in legacy headers.
	ParentBeaconRoot *common.Hash `json:"parentBeaconBlockRoot" rlp:"optional"`

	// Quantum-Geth "Quantum Blob" (Tier-2 fields)
	// Single opaque byte slice for all quantum fields - maintains backward compatibility
	QBlob []byte `json:"qBlob" rlp:"optional"`

	// Virtual fields for accessing quantum data (not serialized)
	// These are populated by UnmarshalQuantumBlob() and used by consensus
	Epoch         *uint32      `json:"epoch" rlp:"-"`         // ⌊Height / 50,000⌋
	QBits         *uint16      `json:"qBits" rlp:"-"`         // qubits per puzzle
	TCount        *uint32      `json:"tCount" rlp:"-"`        // T-gates per puzzle (const 20)
	LNet          *uint16      `json:"lNet" rlp:"-"`          // chained puzzle count (const 128)
	QNonce64      *uint64      `json:"qNonce64" rlp:"-"`      // primary nonce (Bitcoin-style)
	ExtraNonce32  []byte       `json:"extraNonce32" rlp:"-"`  // 32-byte entropy
	OutcomeRoot   *common.Hash `json:"outcomeRoot" rlp:"-"`   // Merkle root of outcomes
	BranchNibbles []byte       `json:"branchNibbles" rlp:"-"` // 128 nibbles for template selection (64 bytes)
	GateHash      *common.Hash `json:"gateHash" rlp:"-"`      // canonical compiler gate hash
	ProofRoot     *common.Hash `json:"proofRoot" rlp:"-"`     // Nova proof Merkle root
	AttestMode    *uint8       `json:"attestMode" rlp:"-"`    // attestation mode

	// Note: Dilithium signature & proofs are stored in block body, not header
}

// EncodeRLP implements rlp.Encoder for Header to properly handle all fields.
// RuntimeAssertHeaderSize adds a paranoid check to ensure header encoding is correct
func (h *Header) RuntimeAssertHeaderSize() {
	// Marshal quantum blob before encoding
	h.MarshalQuantumBlob()

	// Encode header to check size
	enc, err := rlp.EncodeToBytes(h)
	if err != nil {
		log.Crit("Header RLP encoding failed", "err", err)
	}

	// Minimum size check for quantum headers (~580 bytes as per spec)
	if IsQuantumActive(h.Number) && len(enc) < 500 {
		log.Crit("Quantum header too small, struct mismatch", "got", len(enc), "expected", ">500")
	}
}

// field type overrides for gencodec
type headerMarshaling struct {
	Difficulty    *hexutil.Big
	Number        *hexutil.Big
	GasLimit      hexutil.Uint64
	GasUsed       hexutil.Uint64
	Time          hexutil.Uint64
	Extra         hexutil.Bytes
	BaseFee       *hexutil.Big
	Hash          common.Hash `json:"hash"` // adds call to Hash() in MarshalJSON
	BlobGasUsed   *hexutil.Uint64
	ExcessBlobGas *hexutil.Uint64

	// Quantum-Geth field marshaling
	// Types must exactly match header field types for gencodec
	Epoch         hexutil.Uint64 // *uint32 marshaled as hexutil.Uint64
	QBits         hexutil.Uint64 // *uint16 marshaled as hexutil.Uint64
	TCount        hexutil.Uint64 // *uint32 marshaled as hexutil.Uint64
	LNet          hexutil.Uint64 // *uint16 marshaled as hexutil.Uint64
	QNonce64      hexutil.Uint64 // *uint64 marshaled as hexutil.Uint64
	ExtraNonce32  hexutil.Bytes  // []byte marshaled as hexutil.Bytes
	BranchNibbles hexutil.Bytes  // []byte marshaled as hexutil.Bytes
	AttestMode    hexutil.Uint64 // *uint8 marshaled as hexutil.Uint64
}

// Hash returns the block hash of the header, which is simply the keccak256 hash of its
// RLP encoding.
func (h *Header) Hash() common.Hash {
	return rlpHash(h)
}

var headerSize = common.StorageSize(reflect.TypeOf(Header{}).Size())

// Size returns the approximate memory used by all internal contents. It is used
// to approximate and limit the memory consumption of various caches.
func (h *Header) Size() common.StorageSize {
	var baseFeeBits int
	if h.BaseFee != nil {
		baseFeeBits = h.BaseFee.BitLen()
	}
	return headerSize + common.StorageSize(len(h.Extra)+(h.Difficulty.BitLen()+h.Number.BitLen()+baseFeeBits)/8)
}

// SanityCheck checks a few basic things -- these checks are way beyond what
// any 'sane' production values should hold, and can mainly be used to prevent
// that the unbounded fields are stuffed with junk data to add processing
// overhead
func (h *Header) SanityCheck() error {
	if h.Number != nil && !h.Number.IsUint64() {
		return fmt.Errorf("too large block number: bitlen %d", h.Number.BitLen())
	}
	if h.Difficulty != nil {
		if diffLen := h.Difficulty.BitLen(); diffLen > 80 {
			return fmt.Errorf("too large block difficulty: bitlen %d", diffLen)
		}
	}
	if eLen := len(h.Extra); eLen > 100*1024 {
		return fmt.Errorf("too large block extradata: size %d", eLen)
	}
	if h.BaseFee != nil {
		if bfLen := h.BaseFee.BitLen(); bfLen > 256 {
			return fmt.Errorf("too large base fee: bitlen %d", bfLen)
		}
	}
	return nil
}

// EmptyBody returns true if there is no additional 'body' to complete the header
// that is: no transactions, no uncles and no withdrawals.
func (h *Header) EmptyBody() bool {
	if h.WithdrawalsHash != nil {
		return h.TxHash == EmptyTxsHash && *h.WithdrawalsHash == EmptyWithdrawalsHash
	}
	return h.TxHash == EmptyTxsHash && h.UncleHash == EmptyUncleHash
}

// EmptyReceipts returns true if there are no receipts for this header/block.
func (h *Header) EmptyReceipts() bool {
	return h.ReceiptHash == EmptyReceiptsHash
}

// Body is a simple (mutable, non-safe) data container for storing and moving
// a block's data contents (transactions and uncles) together.
type Body struct {
	Transactions []*Transaction
	Uncles       []*Header
	Withdrawals  []*Withdrawal `rlp:"optional"`
}

// Block represents an Ethereum block.
//
// Note the Block type tries to be 'immutable', and contains certain caches that rely
// on that. The rules around block immutability are as follows:
//
//   - We copy all data when the block is constructed. This makes references held inside
//     the block independent of whatever value was passed in.
//
//   - We copy all header data on access. This is because any change to the header would mess
//     up the cached hash and size values in the block. Calling code is expected to take
//     advantage of this to avoid over-allocating!
//
//   - When new body data is attached to the block, a shallow copy of the block is returned.
//     This ensures block modifications are race-free.
//
//   - We do not copy body data on access because it does not affect the caches, and also
//     because it would be too expensive.
type Block struct {
	header       *Header
	uncles       []*Header
	transactions Transactions
	withdrawals  Withdrawals

	// caches
	hash atomic.Value
	size atomic.Value

	// These fields are used by package eth to track
	// inter-peer block relay.
	ReceivedAt   time.Time
	ReceivedFrom interface{}
}

// "external" block encoding. used for eth protocol, etc.
type extblock struct {
	Header      *Header
	Txs         []*Transaction
	Uncles      []*Header
	Withdrawals []*Withdrawal `rlp:"optional"`
}

// NewBlock creates a new block. The input data is copied, changes to header and to the
// field values will not affect the block.
//
// The values of TxHash, ParentUncles, ReceiptHash and Bloom in header
// are ignored and set to values derived from the given txs, uncles
// and receipts.
func NewBlock(header *Header, txs []*Transaction, uncles []*Header, receipts []*Receipt, hasher TrieHasher) *Block {
	b := &Block{header: CopyHeader(header)}

	// TODO: panic if len(txs) != len(receipts)
	if len(txs) == 0 {
		b.header.TxHash = EmptyTxsHash
	} else {
		b.header.TxHash = DeriveSha(Transactions(txs), hasher)
		b.transactions = make(Transactions, len(txs))
		copy(b.transactions, txs)
	}

	if len(receipts) == 0 {
		b.header.ReceiptHash = EmptyReceiptsHash
	} else {
		b.header.ReceiptHash = DeriveSha(Receipts(receipts), hasher)
		b.header.Bloom = CreateBloom(receipts)
	}

	if len(uncles) == 0 {
		b.header.UncleHash = EmptyUncleHash
	} else {
		b.header.UncleHash = CalcUncleHash(uncles)
		b.uncles = make([]*Header, len(uncles))
		for i := range uncles {
			b.uncles[i] = CopyHeader(uncles[i])
		}
	}

	return b
}

// NewBlockWithWithdrawals creates a new block with withdrawals. The input data is copied,
// changes to header and to the field values will not affect the block.
//
// The values of TxHash, UncleHash, ReceiptHash and Bloom in header are ignored and set to
// values derived from the given txs, uncles and receipts.
func NewBlockWithWithdrawals(header *Header, txs []*Transaction, uncles []*Header, receipts []*Receipt, withdrawals []*Withdrawal, hasher TrieHasher) *Block {
	b := NewBlock(header, txs, uncles, receipts, hasher)

	if withdrawals == nil {
		b.header.WithdrawalsHash = nil
	} else if len(withdrawals) == 0 {
		b.header.WithdrawalsHash = &EmptyWithdrawalsHash
	} else {
		h := DeriveSha(Withdrawals(withdrawals), hasher)
		b.header.WithdrawalsHash = &h
	}

	return b.WithWithdrawals(withdrawals)
}

// CopyHeader creates a deep copy of a block header.
func CopyHeader(h *Header) *Header {
	cpy := *h
	if cpy.Difficulty = new(big.Int); h.Difficulty != nil {
		cpy.Difficulty.Set(h.Difficulty)
	}
	if cpy.Number = new(big.Int); h.Number != nil {
		cpy.Number.Set(h.Number)
	}
	if h.BaseFee != nil {
		cpy.BaseFee = new(big.Int).Set(h.BaseFee)
	}
	if len(h.Extra) > 0 {
		cpy.Extra = make([]byte, len(h.Extra))
		copy(cpy.Extra, h.Extra)
	}
	if h.WithdrawalsHash != nil {
		cpy.WithdrawalsHash = new(common.Hash)
		*cpy.WithdrawalsHash = *h.WithdrawalsHash
	}
	if h.ExcessBlobGas != nil {
		cpy.ExcessBlobGas = new(uint64)
		*cpy.ExcessBlobGas = *h.ExcessBlobGas
	}
	if h.BlobGasUsed != nil {
		cpy.BlobGasUsed = new(uint64)
		*cpy.BlobGasUsed = *h.BlobGasUsed
	}
	if h.ParentBeaconRoot != nil {
		cpy.ParentBeaconRoot = new(common.Hash)
		*cpy.ParentBeaconRoot = *h.ParentBeaconRoot
	}
	// Deep copy quantum fields
	if h.Epoch != nil {
		cpy.Epoch = new(uint32)
		*cpy.Epoch = *h.Epoch
	}
	if h.QBits != nil {
		cpy.QBits = new(uint16)
		*cpy.QBits = *h.QBits
	}
	if h.TCount != nil {
		cpy.TCount = new(uint32)
		*cpy.TCount = *h.TCount
	}
	if h.LNet != nil {
		cpy.LNet = new(uint16)
		*cpy.LNet = *h.LNet
	}
	if h.QNonce64 != nil {
		cpy.QNonce64 = new(uint64)
		*cpy.QNonce64 = *h.QNonce64
	}
	if h.ExtraNonce32 != nil {
		cpy.ExtraNonce32 = make([]byte, len(h.ExtraNonce32))
		copy(cpy.ExtraNonce32, h.ExtraNonce32)
	}
	if h.OutcomeRoot != nil {
		cpy.OutcomeRoot = new(common.Hash)
		*cpy.OutcomeRoot = *h.OutcomeRoot
	}
	if h.BranchNibbles != nil {
		cpy.BranchNibbles = make([]byte, len(h.BranchNibbles))
		copy(cpy.BranchNibbles, h.BranchNibbles)
	}
	if h.GateHash != nil {
		cpy.GateHash = new(common.Hash)
		*cpy.GateHash = *h.GateHash
	}
	if h.ProofRoot != nil {
		cpy.ProofRoot = new(common.Hash)
		*cpy.ProofRoot = *h.ProofRoot
	}
	if h.AttestMode != nil {
		cpy.AttestMode = new(uint8)
		*cpy.AttestMode = *h.AttestMode
	}
	return &cpy
}

// DecodeRLP decodes a block from RLP.
func (b *Block) DecodeRLP(s *rlp.Stream) error {
	var eb extblock
	_, size, _ := s.Kind()
	if err := s.Decode(&eb); err != nil {
		return err
	}
	b.header, b.uncles, b.transactions, b.withdrawals = eb.Header, eb.Uncles, eb.Txs, eb.Withdrawals
	b.size.Store(rlp.ListSize(size))

	// Unmarshal quantum blob to populate virtual quantum fields
	if err := b.header.UnmarshalQuantumBlob(); err != nil {
		return err
	}

	return nil
}

// EncodeRLP serializes a block as RLP.
func (b *Block) EncodeRLP(w io.Writer) error {
	// Marshal quantum fields into QBlob before encoding
	b.header.MarshalQuantumBlob()

	return rlp.Encode(w, &extblock{
		Header:      b.header,
		Txs:         b.transactions,
		Uncles:      b.uncles,
		Withdrawals: b.withdrawals,
	})
}

// Body returns the non-header content of the block.
// Note the returned data is not an independent copy.
func (b *Block) Body() *Body {
	return &Body{b.transactions, b.uncles, b.withdrawals}
}

// Accessors for body data. These do not return a copy because the content
// of the body slices does not affect the cached hash/size in block.

func (b *Block) Uncles() []*Header          { return b.uncles }
func (b *Block) Transactions() Transactions { return b.transactions }
func (b *Block) Withdrawals() Withdrawals   { return b.withdrawals }

func (b *Block) Transaction(hash common.Hash) *Transaction {
	for _, transaction := range b.transactions {
		if transaction.Hash() == hash {
			return transaction
		}
	}
	return nil
}

// Header returns the block header (as a copy).
func (b *Block) Header() *Header {
	return CopyHeader(b.header)
}

// Header value accessors. These do copy!

func (b *Block) Number() *big.Int     { return new(big.Int).Set(b.header.Number) }
func (b *Block) GasLimit() uint64     { return b.header.GasLimit }
func (b *Block) GasUsed() uint64      { return b.header.GasUsed }
func (b *Block) Difficulty() *big.Int { return new(big.Int).Set(b.header.Difficulty) }
func (b *Block) Time() uint64         { return b.header.Time }

func (b *Block) NumberU64() uint64        { return b.header.Number.Uint64() }
func (b *Block) MixDigest() common.Hash   { return b.header.MixDigest }
func (b *Block) Nonce() uint64            { return binary.BigEndian.Uint64(b.header.Nonce[:]) }
func (b *Block) QNonce64() uint64         { return *b.header.QNonce64 }
func (b *Block) Bloom() Bloom             { return b.header.Bloom }
func (b *Block) Coinbase() common.Address { return b.header.Coinbase }
func (b *Block) Root() common.Hash        { return b.header.Root }
func (b *Block) ParentHash() common.Hash  { return b.header.ParentHash }
func (b *Block) TxHash() common.Hash      { return b.header.TxHash }
func (b *Block) ReceiptHash() common.Hash { return b.header.ReceiptHash }
func (b *Block) UncleHash() common.Hash   { return b.header.UncleHash }
func (b *Block) Extra() []byte            { return common.CopyBytes(b.header.Extra) }

func (b *Block) BaseFee() *big.Int {
	if b.header.BaseFee == nil {
		return nil
	}
	return new(big.Int).Set(b.header.BaseFee)
}

func (b *Block) BeaconRoot() *common.Hash { return b.header.ParentBeaconRoot }

func (b *Block) ExcessBlobGas() *uint64 {
	var excessBlobGas *uint64
	if b.header.ExcessBlobGas != nil {
		excessBlobGas = new(uint64)
		*excessBlobGas = *b.header.ExcessBlobGas
	}
	return excessBlobGas
}

func (b *Block) BlobGasUsed() *uint64 {
	var blobGasUsed *uint64
	if b.header.BlobGasUsed != nil {
		blobGasUsed = new(uint64)
		*blobGasUsed = *b.header.BlobGasUsed
	}
	return blobGasUsed
}

// Size returns the true RLP encoded storage size of the block, either by encoding
// and returning it, or returning a previously cached value.
func (b *Block) Size() uint64 {
	if size := b.size.Load(); size != nil {
		return size.(uint64)
	}
	c := writeCounter(0)
	rlp.Encode(&c, b)
	b.size.Store(uint64(c))
	return uint64(c)
}

// SanityCheck can be used to prevent that unbounded fields are
// stuffed with junk data to add processing overhead
func (b *Block) SanityCheck() error {
	return b.header.SanityCheck()
}

type writeCounter uint64

func (c *writeCounter) Write(b []byte) (int, error) {
	*c += writeCounter(len(b))
	return len(b), nil
}

func CalcUncleHash(uncles []*Header) common.Hash {
	if len(uncles) == 0 {
		return EmptyUncleHash
	}
	return rlpHash(uncles)
}

// NewBlockWithHeader creates a block with the given header data. The
// header data is copied, changes to header and to the field values
// will not affect the block.
func NewBlockWithHeader(header *Header) *Block {
	return &Block{header: CopyHeader(header)}
}

// WithSeal returns a new block with the data from b but the header replaced with
// the sealed one.
func (b *Block) WithSeal(header *Header) *Block {
	return &Block{
		header:       CopyHeader(header),
		transactions: b.transactions,
		uncles:       b.uncles,
		withdrawals:  b.withdrawals,
	}
}

// WithBody returns a copy of the block with the given transaction and uncle contents.
func (b *Block) WithBody(transactions []*Transaction, uncles []*Header) *Block {
	block := &Block{
		header:       b.header,
		transactions: make([]*Transaction, len(transactions)),
		uncles:       make([]*Header, len(uncles)),
		withdrawals:  b.withdrawals,
	}
	copy(block.transactions, transactions)
	for i := range uncles {
		block.uncles[i] = CopyHeader(uncles[i])
	}
	return block
}

// WithWithdrawals returns a copy of the block containing the given withdrawals.
func (b *Block) WithWithdrawals(withdrawals []*Withdrawal) *Block {
	block := &Block{
		header:       b.header,
		transactions: b.transactions,
		uncles:       b.uncles,
	}
	if withdrawals != nil {
		block.withdrawals = make([]*Withdrawal, len(withdrawals))
		copy(block.withdrawals, withdrawals)
	}
	return block
}

// Hash returns the keccak256 hash of b's header.
// The hash is computed on the first call and cached thereafter.
func (b *Block) Hash() common.Hash {
	if hash := b.hash.Load(); hash != nil {
		return hash.(common.Hash)
	}
	v := b.header.Hash()
	b.hash.Store(v)
	return v
}

type Blocks []*Block

// HeaderParentHashFromRLP returns the parentHash of an RLP-encoded
// header. If 'header' is invalid, the zero hash is returned.
func HeaderParentHashFromRLP(header []byte) common.Hash {
	// parentHash is the first list element.
	listContent, _, err := rlp.SplitList(header)
	if err != nil {
		return common.Hash{}
	}
	parentHash, _, err := rlp.SplitString(listContent)
	if err != nil {
		return common.Hash{}
	}
	if len(parentHash) != 32 {
		return common.Hash{}
	}
	return common.BytesToHash(parentHash)
}

// IsQuantumActive returns true if quantum consensus is active at the given block number
func IsQuantumActive(num *big.Int) bool {
	// For development, quantum is always active (fork block = 0)
	// In production, this would check against params.QuantumForkBlock
	return true
}

// MarshalQuantumBlob encodes quantum fields into the QBlob byte slice
func (h *Header) MarshalQuantumBlob() {
	if !IsQuantumActive(h.Number) {
		h.QBlob = nil
		return
	}

	// Only marshal if we have quantum fields
	if h.Epoch == nil && h.QBits == nil && h.TCount == nil && h.LNet == nil &&
		h.QNonce64 == nil && len(h.ExtraNonce32) == 0 && h.OutcomeRoot == nil &&
		len(h.BranchNibbles) == 0 && h.GateHash == nil && h.ProofRoot == nil && h.AttestMode == nil {
		h.QBlob = nil
		return
	}

	var buf bytes.Buffer

	// Helper function to ensure 32-byte padding for hashes
	pad32 := func(hash *common.Hash) {
		if hash != nil {
			buf.Write(hash[:])
		} else {
			buf.Write(make([]byte, 32)) // zero hash
		}
	}

	// Encode fields in fixed order (following quantum spec)
	// 16.1 Epoch (4 bytes)
	if h.Epoch != nil {
		binary.Write(&buf, binary.LittleEndian, *h.Epoch)
	} else {
		binary.Write(&buf, binary.LittleEndian, uint32(0))
	}

	// 16.2 QBits (2 bytes)
	if h.QBits != nil {
		binary.Write(&buf, binary.LittleEndian, *h.QBits)
	} else {
		binary.Write(&buf, binary.LittleEndian, uint16(0))
	}

	// 16.3 TCount (4 bytes)
	if h.TCount != nil {
		binary.Write(&buf, binary.LittleEndian, *h.TCount)
	} else {
		binary.Write(&buf, binary.LittleEndian, uint32(0))
	}

	// 16.4 LNet (2 bytes)
	if h.LNet != nil {
		binary.Write(&buf, binary.LittleEndian, *h.LNet)
	} else {
		binary.Write(&buf, binary.LittleEndian, uint16(0))
	}

	// 16.5 QNonce64 (8 bytes)
	if h.QNonce64 != nil {
		binary.Write(&buf, binary.LittleEndian, *h.QNonce64)
	} else {
		binary.Write(&buf, binary.LittleEndian, uint64(0))
	}

	// 16.6 ExtraNonce32 (32 bytes)
	if len(h.ExtraNonce32) >= 32 {
		buf.Write(h.ExtraNonce32[:32])
	} else {
		buf.Write(h.ExtraNonce32)
		buf.Write(make([]byte, 32-len(h.ExtraNonce32))) // zero pad
	}

	// 16.7 OutcomeRoot (32 bytes)
	pad32(h.OutcomeRoot)

	// 16.8 BranchNibbles (64 bytes)
	if len(h.BranchNibbles) >= 64 {
		buf.Write(h.BranchNibbles[:64])
	} else {
		buf.Write(h.BranchNibbles)
		buf.Write(make([]byte, 64-len(h.BranchNibbles))) // zero pad
	}

	// 16.9 GateHash (32 bytes)
	pad32(h.GateHash)

	// 16.10 ProofRoot (32 bytes)
	pad32(h.ProofRoot)

	// 16.11 AttestMode (1 byte)
	if h.AttestMode != nil {
		buf.WriteByte(*h.AttestMode)
	} else {
		buf.WriteByte(0)
	}

	h.QBlob = buf.Bytes()
}

// UnmarshalQuantumBlob decodes quantum fields from the QBlob byte slice
func (h *Header) UnmarshalQuantumBlob() error {
	if !IsQuantumActive(h.Number) || len(h.QBlob) == 0 {
		// Clear all quantum fields for non-quantum blocks
		h.Epoch = nil
		h.QBits = nil
		h.TCount = nil
		h.LNet = nil
		h.QNonce64 = nil
		h.ExtraNonce32 = nil
		h.OutcomeRoot = nil
		h.BranchNibbles = nil
		h.GateHash = nil
		h.ProofRoot = nil
		h.AttestMode = nil
		return nil
	}

	// Minimum size check (should be exactly 213 bytes for quantum headers)
	expectedSize := 4 + 2 + 4 + 2 + 8 + 32 + 32 + 64 + 32 + 32 + 1 // 213 bytes
	if len(h.QBlob) < expectedSize {
		return fmt.Errorf("quantum blob too short: got %d bytes, expected %d", len(h.QBlob), expectedSize)
	}

	buf := bytes.NewReader(h.QBlob)

	// Helper function to read 32-byte hashes
	readHash := func() (*common.Hash, error) {
		hashBytes := make([]byte, 32)
		if _, err := buf.Read(hashBytes); err != nil {
			return nil, err
		}
		// Convert zero hash to nil pointer
		hash := common.BytesToHash(hashBytes)
		if hash == (common.Hash{}) {
			return nil, nil
		}
		return &hash, nil
	}

	// Decode fields in the same order as encoding
	// 16.1 Epoch (4 bytes)
	var epoch uint32
	if err := binary.Read(buf, binary.LittleEndian, &epoch); err != nil {
		return err
	}
	h.Epoch = &epoch

	// 16.2 QBits (2 bytes)
	var qbits uint16
	if err := binary.Read(buf, binary.LittleEndian, &qbits); err != nil {
		return err
	}
	h.QBits = &qbits

	// 16.3 TCount (4 bytes)
	var tcount uint32
	if err := binary.Read(buf, binary.LittleEndian, &tcount); err != nil {
		return err
	}
	h.TCount = &tcount

	// 16.4 LNet (2 bytes)
	var lnet uint16
	if err := binary.Read(buf, binary.LittleEndian, &lnet); err != nil {
		return err
	}
	h.LNet = &lnet

	// 16.5 QNonce64 (8 bytes)
	var qnonce uint64
	if err := binary.Read(buf, binary.LittleEndian, &qnonce); err != nil {
		return err
	}
	h.QNonce64 = &qnonce

	// 16.6 ExtraNonce32 (32 bytes)
	h.ExtraNonce32 = make([]byte, 32)
	if _, err := buf.Read(h.ExtraNonce32); err != nil {
		return err
	}

	// 16.7 OutcomeRoot (32 bytes)
	var err error
	h.OutcomeRoot, err = readHash()
	if err != nil {
		return err
	}

	// 16.8 BranchNibbles (64 bytes)
	h.BranchNibbles = make([]byte, 64)
	if _, err := buf.Read(h.BranchNibbles); err != nil {
		return err
	}

	// 16.9 GateHash (32 bytes)
	h.GateHash, err = readHash()
	if err != nil {
		return err
	}

	// 16.10 ProofRoot (32 bytes)
	h.ProofRoot, err = readHash()
	if err != nil {
		return err
	}

	// 16.11 AttestMode (1 byte)
	var attestMode uint8
	if err := binary.Read(buf, binary.LittleEndian, &attestMode); err != nil {
		return err
	}
	h.AttestMode = &attestMode

	return nil
}
