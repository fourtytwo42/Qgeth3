// Copyright 2021 The go-ethereum Authors
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

package eth

import (
	"encoding/json"
	"fmt"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/ethereum/go-ethereum/trie"
)

func handleGetBlockHeaders(backend Backend, msg Decoder, peer *Peer) error {
	// Decode the complex header query
	var query GetBlockHeadersPacket
	if err := msg.Decode(&query); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	response := ServiceGetBlockHeadersQuery(backend, query.GetBlockHeadersRequest, peer)
	return peer.ReplyBlockHeadersRLP(query.RequestId, response)
}

// ServiceGetBlockHeadersQuery assembles the response to a header query. It is
// exposed to allow external packages to test protocol behavior.
func ServiceGetBlockHeadersQuery(backend Backend, query *GetBlockHeadersRequest, peer *Peer) []rlp.RawValue {
	// COMPREHENSIVE DEBUG LOGGING for header serving at verbosity 4
	log.Debug("🔍 HEADER REQUEST: Received header query", 
		"peer", peer.ID()[:8],
		"hash", query.Origin.Hash.Hex()[:8], 
		"number", query.Origin.Number,
		"amount", query.Amount,
		"skip", query.Skip,
		"reverse", query.Reverse,
		"hashMode", query.Origin.Hash != (common.Hash{}))
	
	var result []rlp.RawValue
	if query.Skip == 0 {
		log.Debug("🚀 HEADER REQUEST: Using contiguous query path", "peer", peer.ID()[:8])
		// The fast path: when the request is for a contiguous segment of headers.
		result = serviceContiguousBlockHeaderQuery(backend, query)
	} else {
		log.Debug("🚀 HEADER REQUEST: Using non-contiguous query path", "peer", peer.ID()[:8])
		result = serviceNonContiguousBlockHeaderQuery(backend, query, peer)
	}
	
	log.Debug("✅ HEADER REQUEST: Returning headers", 
		"peer", peer.ID()[:8],
		"requested", query.Amount,
		"returned", len(result),
		"success", len(result) > 0)
	
	if len(result) == 0 {
		log.Warn("❌ HEADER REQUEST: ZERO HEADERS RETURNED", 
			"peer", peer.ID()[:8],
			"hash", query.Origin.Hash.Hex()[:8], 
			"number", query.Origin.Number,
			"amount", query.Amount,
			"skip", query.Skip,
			"reverse", query.Reverse)
	}
	
	return result
}

func serviceNonContiguousBlockHeaderQuery(backend Backend, query *GetBlockHeadersRequest, peer *Peer) []rlp.RawValue {
	chain := backend.Chain()
	hashMode := query.Origin.Hash != (common.Hash{})
	first := true
	maxNonCanonical := uint64(100)

	log.Debug("🔍 NON-CONTIGUOUS: Starting query", 
		"peer", peer.ID()[:8],
		"hashMode", hashMode, 
		"originHash", query.Origin.Hash.Hex()[:8],
		"originNumber", query.Origin.Number,
		"amount", query.Amount,
		"skip", query.Skip,
		"reverse", query.Reverse)

	// Gather headers until the fetch or network limits is reached
	var (
		bytes   common.StorageSize
		headers []rlp.RawValue
		unknown bool
		lookups int
	)
	for !unknown && len(headers) < int(query.Amount) && bytes < softResponseLimit &&
		len(headers) < maxHeadersServe && lookups < 2*maxHeadersServe {
		lookups++
		log.Debug("🔍 NON-CONTIGUOUS: Lookup iteration", 
			"peer", peer.ID()[:8],
			"lookup", lookups, 
			"headersSoFar", len(headers),
			"originHash", query.Origin.Hash.Hex()[:8],
			"originNumber", query.Origin.Number)
		
		// Retrieve the next header satisfying the query
		var origin *types.Header
		if hashMode {
			if first {
				first = false
				log.Debug("🔍 NON-CONTIGUOUS: First hash lookup", 
					"peer", peer.ID()[:8],
					"hash", query.Origin.Hash.Hex()[:8])
				origin = chain.GetHeaderByHash(query.Origin.Hash)
				if origin != nil {
					query.Origin.Number = origin.Number.Uint64()
					log.Debug("✅ NON-CONTIGUOUS: Found origin header", 
						"peer", peer.ID()[:8],
						"hash", query.Origin.Hash.Hex()[:8],
						"number", origin.Number.Uint64())
				} else {
					log.Warn("❌ NON-CONTIGUOUS: Origin header not found", 
						"peer", peer.ID()[:8],
						"hash", query.Origin.Hash.Hex()[:8])
				}
			} else {
				log.Debug("🔍 NON-CONTIGUOUS: Subsequent hash lookup", 
					"peer", peer.ID()[:8],
					"hash", query.Origin.Hash.Hex()[:8],
					"number", query.Origin.Number)
				origin = chain.GetHeader(query.Origin.Hash, query.Origin.Number)
				if origin == nil {
					log.Debug("❌ NON-CONTIGUOUS: Header not found", 
						"peer", peer.ID()[:8],
						"hash", query.Origin.Hash.Hex()[:8],
						"number", query.Origin.Number)
				}
			}
		} else {
			log.Debug("🔍 NON-CONTIGUOUS: Number lookup", 
				"peer", peer.ID()[:8],
				"number", query.Origin.Number)
			origin = chain.GetHeaderByNumber(query.Origin.Number)
			if origin == nil {
				log.Debug("❌ NON-CONTIGUOUS: Header not found by number", 
					"peer", peer.ID()[:8],
					"number", query.Origin.Number)
			}
		}
		if origin == nil {
			log.Debug("❌ NON-CONTIGUOUS: Breaking due to nil header", 
				"peer", peer.ID()[:8],
				"lookup", lookups)
			break
		}
		
		// CRITICAL FIX: Ensure quantum fields are marshaled before RLP encoding
		// This ensures consistent header transmission during sync
		origin.MarshalQuantumBlob()
		
		if rlpData, err := rlp.EncodeToBytes(origin); err != nil {
			log.Error("Unable to encode header for transmission", "err", err)
			break
		} else {
			headers = append(headers, rlp.RawValue(rlpData))
			bytes += common.StorageSize(len(rlpData))
		}
		// Advance to the next header of the query
		switch {
		case hashMode && query.Reverse:
			// Hash based traversal towards the genesis block
			ancestor := query.Skip + 1
			log.Debug("🔍 NON-CONTIGUOUS: Hash reverse traversal", 
				"peer", peer.ID()[:8],
				"currentHash", query.Origin.Hash.Hex()[:8],
				"currentNumber", query.Origin.Number,
				"ancestor", ancestor)
			if ancestor == 0 {
				log.Debug("❌ NON-CONTIGUOUS: Ancestor == 0, marking unknown", 
					"peer", peer.ID()[:8])
				unknown = true
			} else {
				// CRITICAL FIX: Check if we're trying to go beyond genesis
				if query.Origin.Number < ancestor {
					log.Debug("❌ NON-CONTIGUOUS: Cannot go beyond genesis", 
						"peer", peer.ID()[:8],
						"currentNumber", query.Origin.Number,
						"ancestor", ancestor,
						"reason", "would result in negative block number")
					unknown = true
				} else {
					oldHash, oldNumber := query.Origin.Hash, query.Origin.Number
					query.Origin.Hash, query.Origin.Number = chain.GetAncestor(query.Origin.Hash, query.Origin.Number, ancestor, &maxNonCanonical)
					unknown = (query.Origin.Hash == common.Hash{})
					log.Debug("🔍 NON-CONTIGUOUS: Ancestor lookup result", 
						"peer", peer.ID()[:8],
						"oldHash", oldHash.Hex()[:8],
						"oldNumber", oldNumber,
						"newHash", query.Origin.Hash.Hex()[:8],
						"newNumber", query.Origin.Number,
						"unknown", unknown)
				}
			}
		case hashMode && !query.Reverse:
			// Hash based traversal towards the leaf block
			var (
				current = origin.Number.Uint64()
				next    = current + query.Skip + 1
			)
			if next <= current {
				infos, _ := json.MarshalIndent(peer.Peer.Info(), "", "  ")
				peer.Log().Warn("GetBlockHeaders skip overflow attack", "current", current, "skip", query.Skip, "next", next, "attacker", infos)
				unknown = true
			} else {
				if header := chain.GetHeaderByNumber(next); header != nil {
					nextHash := header.Hash()
					expOldHash, _ := chain.GetAncestor(nextHash, next, query.Skip+1, &maxNonCanonical)
					if expOldHash == query.Origin.Hash {
						query.Origin.Hash, query.Origin.Number = nextHash, next
					} else {
						unknown = true
					}
				} else {
					unknown = true
				}
			}
		case query.Reverse:
			// Number based traversal towards the genesis block
			skipAmount := query.Skip + 1
			log.Debug("🔍 NON-CONTIGUOUS: Number reverse traversal", 
				"peer", peer.ID()[:8],
				"currentNumber", query.Origin.Number,
				"skipAmount", skipAmount)
			if query.Origin.Number >= skipAmount {
				oldNumber := query.Origin.Number
				query.Origin.Number -= skipAmount
				log.Debug("✅ NON-CONTIGUOUS: Number traversal success", 
					"peer", peer.ID()[:8],
					"oldNumber", oldNumber,
					"newNumber", query.Origin.Number)
			} else {
				log.Debug("❌ NON-CONTIGUOUS: Cannot traverse beyond genesis", 
					"peer", peer.ID()[:8],
					"currentNumber", query.Origin.Number,
					"skipAmount", skipAmount,
					"reason", "would result in negative block number")
				unknown = true
			}

		case !query.Reverse:
			// Number based traversal towards the leaf block
			query.Origin.Number += query.Skip + 1
		}
	}
	
	log.Debug("✅ NON-CONTIGUOUS: Query complete", 
		"peer", peer.ID()[:8],
		"requested", query.Amount,
		"returned", len(headers),
		"totalLookups", lookups,
		"unknown", unknown,
		"bytes", bytes)
	
	return headers
}

func serviceContiguousBlockHeaderQuery(backend Backend, query *GetBlockHeadersRequest) []rlp.RawValue {
	chain := backend.Chain()
	count := query.Amount
	if count > maxHeadersServe {
		count = maxHeadersServe
	}
	
	log.Debug("🔍 CONTIGUOUS: Starting query", 
		"hashMode", query.Origin.Hash != (common.Hash{}),
		"originHash", query.Origin.Hash.Hex()[:8],
		"originNumber", query.Origin.Number,
		"amount", query.Amount,
		"count", count,
		"reverse", query.Reverse)
	
	if query.Origin.Hash == (common.Hash{}) {
		// Number mode, just return the canon chain segment. The backend
		// delivers in [N, N-1, N-2..] descending order, so we need to
		// accommodate for that.
		from := query.Origin.Number
		if !query.Reverse {
			from = from + count - 1
		}
		log.Debug("🔍 CONTIGUOUS: Number mode lookup", 
			"originNumber", query.Origin.Number,
			"from", from,
			"count", count,
			"reverse", query.Reverse)
		headers := chain.GetHeadersFrom(from, count)
		if !query.Reverse {
			for i, j := 0, len(headers)-1; i < j; i, j = i+1, j-1 {
				headers[i], headers[j] = headers[j], headers[i]
			}
		}
		log.Debug("✅ CONTIGUOUS: Number mode result", 
			"requested", count,
			"returned", len(headers))
		return headers
	}
	// Hash mode.
	var (
		headers []rlp.RawValue
		hash    = query.Origin.Hash
		header  = chain.GetHeaderByHash(hash)
	)
	
	log.Debug("🔍 CONTIGUOUS: Hash mode lookup", 
		"hash", hash.Hex()[:8])
	
	if header != nil {
		log.Debug("✅ CONTIGUOUS: Found origin header", 
			"hash", hash.Hex()[:8],
			"number", header.Number.Uint64())
		
		// CRITICAL FIX: Ensure quantum fields are marshaled before RLP encoding
		header.MarshalQuantumBlob()
		
		if rlpData, err := rlp.EncodeToBytes(header); err != nil {
			log.Error("Unable to encode header in contiguous query", "err", err)
		} else {
			headers = append(headers, rlpData)
			log.Debug("✅ CONTIGUOUS: Encoded origin header", 
				"hash", hash.Hex()[:8],
				"size", len(rlpData))
		}
	} else {
		log.Warn("❌ CONTIGUOUS: Origin header not found", 
			"hash", hash.Hex()[:8])
		// We don't even have the origin header
		return headers
	}
	num := header.Number.Uint64()
	if !query.Reverse {
		// Theoretically, we are tasked to deliver header by hash H, and onwards.
		// However, if H is not canon, we will be unable to deliver any descendants of
		// H.
		if canonHash := chain.GetCanonicalHash(num); canonHash != hash {
			// Not canon, we can't deliver descendants
			return headers
		}
		descendants := chain.GetHeadersFrom(num+count-1, count-1)
		for i, j := 0, len(descendants)-1; i < j; i, j = i+1, j-1 {
			descendants[i], descendants[j] = descendants[j], descendants[i]
		}
		headers = append(headers, descendants...)
		return headers
	}
	{ // Last mode: deliver ancestors of H
		for i := uint64(1); header != nil && i < count; i++ {
			header = chain.GetHeaderByHash(header.ParentHash)
			if header == nil {
				break
			}
			
			// CRITICAL FIX: Ensure quantum fields are marshaled before RLP encoding
			header.MarshalQuantumBlob()
			
			if rlpData, err := rlp.EncodeToBytes(header); err != nil {
				log.Error("Unable to encode header for ancestor", "err", err)
				break
			} else {
				headers = append(headers, rlpData)
			}
		}
		return headers
	}
}

func handleGetBlockBodies(backend Backend, msg Decoder, peer *Peer) error {
	// Decode the block body retrieval message
	var query GetBlockBodiesPacket
	if err := msg.Decode(&query); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	response := ServiceGetBlockBodiesQuery(backend.Chain(), query.GetBlockBodiesRequest)
	return peer.ReplyBlockBodiesRLP(query.RequestId, response)
}

// ServiceGetBlockBodiesQuery assembles the response to a body query. It is
// exposed to allow external packages to test protocol behavior.
func ServiceGetBlockBodiesQuery(chain *core.BlockChain, query GetBlockBodiesRequest) []rlp.RawValue {
	// Gather blocks until the fetch or network limits is reached
	var (
		bytes  int
		bodies []rlp.RawValue
	)
	for lookups, hash := range query {
		if bytes >= softResponseLimit || len(bodies) >= maxBodiesServe ||
			lookups >= 2*maxBodiesServe {
			break
		}
		if data := chain.GetBodyRLP(hash); len(data) != 0 {
			bodies = append(bodies, data)
			bytes += len(data)
		}
	}
	return bodies
}

func handleGetReceipts(backend Backend, msg Decoder, peer *Peer) error {
	// Decode the block receipts retrieval message
	var query GetReceiptsPacket
	if err := msg.Decode(&query); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	response := ServiceGetReceiptsQuery(backend.Chain(), query.GetReceiptsRequest)
	return peer.ReplyReceiptsRLP(query.RequestId, response)
}

// ServiceGetReceiptsQuery assembles the response to a receipt query. It is
// exposed to allow external packages to test protocol behavior.
func ServiceGetReceiptsQuery(chain *core.BlockChain, query GetReceiptsRequest) []rlp.RawValue {
	// Gather state data until the fetch or network limits is reached
	var (
		bytes    int
		receipts []rlp.RawValue
	)
	for lookups, hash := range query {
		if bytes >= softResponseLimit || len(receipts) >= maxReceiptsServe ||
			lookups >= 2*maxReceiptsServe {
			break
		}
		// Retrieve the requested block's receipts
		results := chain.GetReceiptsByHash(hash)
		if results == nil {
			if header := chain.GetHeaderByHash(hash); header == nil || header.ReceiptHash != types.EmptyRootHash {
				continue
			}
		}
		// If known, encode and queue for response packet
		if encoded, err := rlp.EncodeToBytes(results); err != nil {
			log.Error("Failed to encode receipt", "err", err)
		} else {
			receipts = append(receipts, encoded)
			bytes += len(encoded)
		}
	}
	return receipts
}

func handleNewBlockhashes(backend Backend, msg Decoder, peer *Peer) error {
	// A batch of new block announcements just arrived
	ann := new(NewBlockHashesPacket)
	if err := msg.Decode(ann); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	// Mark the hashes as present at the remote node
	for _, block := range *ann {
		peer.markBlock(block.Hash)
	}
	// Deliver them all to the backend for queuing
	return backend.Handle(peer, ann)
}

func handleNewBlock(backend Backend, msg Decoder, peer *Peer) error {
	// Retrieve and decode the propagated block
	ann := new(NewBlockPacket)
	if err := msg.Decode(ann); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	if err := ann.sanityCheck(); err != nil {
		return err
	}
	if hash := types.CalcUncleHash(ann.Block.Uncles()); hash != ann.Block.UncleHash() {
		log.Warn("Propagated block has invalid uncles", "have", hash, "exp", ann.Block.UncleHash())
		return nil // TODO(karalabe): return error eventually, but wait a few releases
	}
	if hash := types.DeriveSha(ann.Block.Transactions(), trie.NewStackTrie(nil)); hash != ann.Block.TxHash() {
		log.Warn("Propagated block has invalid body", "have", hash, "exp", ann.Block.TxHash())
		return nil // TODO(karalabe): return error eventually, but wait a few releases
	}
	ann.Block.ReceivedAt = msg.Time()
	ann.Block.ReceivedFrom = peer

	// Mark the peer as owning the block
	peer.markBlock(ann.Block.Hash())

	return backend.Handle(peer, ann)
}

func handleBlockHeaders(backend Backend, msg Decoder, peer *Peer) error {
	// A batch of headers arrived to one of our previous requests
	res := new(BlockHeadersPacket)
	if err := msg.Decode(res); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	
	// CRITICAL FIX: Unmarshal quantum blobs for all received headers
	// This ensures quantum fields are properly populated from peer data
	for _, header := range res.BlockHeadersRequest {
		if err := header.UnmarshalQuantumBlob(); err != nil {
			log.Warn("Failed to unmarshal quantum blob from peer header", 
				"hash", header.Hash().Hex()[:8], 
				"number", header.Number.Uint64(), 
				"peer", peer.ID(), 
				"err", err)
			// Don't reject the entire batch for one bad header,
			// let the consensus engine handle validation
		}
	}
	
	metadata := func() interface{} {
		hashes := make([]common.Hash, len(res.BlockHeadersRequest))
		for i, header := range res.BlockHeadersRequest {
			hashes[i] = header.Hash()
		}
		return hashes
	}
	return peer.dispatchResponse(&Response{
		id:   res.RequestId,
		code: BlockHeadersMsg,
		Res:  &res.BlockHeadersRequest,
	}, metadata)
}

func handleBlockBodies(backend Backend, msg Decoder, peer *Peer) error {
	// A batch of block bodies arrived to one of our previous requests
	res := new(BlockBodiesPacket)
	if err := msg.Decode(res); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	metadata := func() interface{} {
		var (
			txsHashes        = make([]common.Hash, len(res.BlockBodiesResponse))
			uncleHashes      = make([]common.Hash, len(res.BlockBodiesResponse))
			withdrawalHashes = make([]common.Hash, len(res.BlockBodiesResponse))
		)
		hasher := trie.NewStackTrie(nil)
		for i, body := range res.BlockBodiesResponse {
			txsHashes[i] = types.DeriveSha(types.Transactions(body.Transactions), hasher)
			uncleHashes[i] = types.CalcUncleHash(body.Uncles)
			if body.Withdrawals != nil {
				withdrawalHashes[i] = types.DeriveSha(types.Withdrawals(body.Withdrawals), hasher)
			}
		}
		return [][]common.Hash{txsHashes, uncleHashes, withdrawalHashes}
	}
	return peer.dispatchResponse(&Response{
		id:   res.RequestId,
		code: BlockBodiesMsg,
		Res:  &res.BlockBodiesResponse,
	}, metadata)
}

func handleReceipts(backend Backend, msg Decoder, peer *Peer) error {
	// A batch of receipts arrived to one of our previous requests
	res := new(ReceiptsPacket)
	if err := msg.Decode(res); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	metadata := func() interface{} {
		hasher := trie.NewStackTrie(nil)
		hashes := make([]common.Hash, len(res.ReceiptsResponse))
		for i, receipt := range res.ReceiptsResponse {
			hashes[i] = types.DeriveSha(types.Receipts(receipt), hasher)
		}
		return hashes
	}
	return peer.dispatchResponse(&Response{
		id:   res.RequestId,
		code: ReceiptsMsg,
		Res:  &res.ReceiptsResponse,
	}, metadata)
}

func handleNewPooledTransactionHashes(backend Backend, msg Decoder, peer *Peer) error {
	// New transaction announcement arrived, make sure we have
	// a valid and fresh chain to handle them
	if !backend.AcceptTxs() {
		return nil
	}
	ann := new(NewPooledTransactionHashesPacket)
	if err := msg.Decode(ann); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	if len(ann.Hashes) != len(ann.Types) || len(ann.Hashes) != len(ann.Sizes) {
		return fmt.Errorf("%w: message %v: invalid len of fields: %v %v %v", errDecode, msg, len(ann.Hashes), len(ann.Types), len(ann.Sizes))
	}
	// Schedule all the unknown hashes for retrieval
	for _, hash := range ann.Hashes {
		peer.markTransaction(hash)
	}
	return backend.Handle(peer, ann)
}

func handleGetPooledTransactions(backend Backend, msg Decoder, peer *Peer) error {
	// Decode the pooled transactions retrieval message
	var query GetPooledTransactionsPacket
	if err := msg.Decode(&query); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	hashes, txs := answerGetPooledTransactions(backend, query.GetPooledTransactionsRequest)
	return peer.ReplyPooledTransactionsRLP(query.RequestId, hashes, txs)
}

func answerGetPooledTransactions(backend Backend, query GetPooledTransactionsRequest) ([]common.Hash, []rlp.RawValue) {
	// Gather transactions until the fetch or network limits is reached
	var (
		bytes  int
		hashes []common.Hash
		txs    []rlp.RawValue
	)
	for _, hash := range query {
		if bytes >= softResponseLimit {
			break
		}
		// Retrieve the requested transaction, skipping if unknown to us
		tx := backend.TxPool().Get(hash)
		if tx == nil {
			continue
		}
		// If known, encode and queue for response packet
		if encoded, err := rlp.EncodeToBytes(tx); err != nil {
			log.Error("Failed to encode transaction", "err", err)
		} else {
			hashes = append(hashes, hash)
			txs = append(txs, encoded)
			bytes += len(encoded)
		}
	}
	return hashes, txs
}

func handleTransactions(backend Backend, msg Decoder, peer *Peer) error {
	// Transactions arrived, make sure we have a valid and fresh chain to handle them
	if !backend.AcceptTxs() {
		return nil
	}
	// Transactions can be processed, parse all of them and deliver to the pool
	var txs TransactionsPacket
	if err := msg.Decode(&txs); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	for i, tx := range txs {
		// Validate and mark the remote transaction
		if tx == nil {
			return fmt.Errorf("%w: transaction %d is nil", errDecode, i)
		}
		peer.markTransaction(tx.Hash())
	}
	return backend.Handle(peer, &txs)
}

func handlePooledTransactions(backend Backend, msg Decoder, peer *Peer) error {
	// Transactions arrived, make sure we have a valid and fresh chain to handle them
	if !backend.AcceptTxs() {
		return nil
	}
	// Transactions can be processed, parse all of them and deliver to the pool
	var txs PooledTransactionsPacket
	if err := msg.Decode(&txs); err != nil {
		return fmt.Errorf("%w: message %v: %v", errDecode, msg, err)
	}
	for i, tx := range txs.PooledTransactionsResponse {
		// Validate and mark the remote transaction
		if tx == nil {
			return fmt.Errorf("%w: transaction %d is nil", errDecode, i)
		}
		peer.markTransaction(tx.Hash())
	}
	requestTracker.Fulfil(peer.id, peer.version, PooledTransactionsMsg, txs.RequestId)

	return backend.Handle(peer, &txs.PooledTransactionsResponse)
}
