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

package core

import (
	crand "crypto/rand"
	"errors"
	"fmt"
	"math/big"
	mrand "math/rand"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/math"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// ForkChoice is the fork chooser based on the highest total difficulty of the
// chain(the fork choice used in the eth1) and the external fork choice (the fork
// choice used in the eth2). This main goal of this ForkChoice is not only for
// offering fork choice during the eth1/2 merge phase, but also keep the compatibility
// for all other proof-of-work networks.
type ForkChoice struct {
	chain consensus.ChainHeaderReader
	rand  *mrand.Rand

	// preserve is a helper function used in td fork choice.
	// Miners will prefer to choose the local mined block if the
	// local td is equal to the extern one. It can be nil for light
	// client
	preserve func(header *types.Header) bool
}

func NewForkChoice(chainReader consensus.ChainHeaderReader, preserve func(header *types.Header) bool) *ForkChoice {
	// Seed a fast but crypto originating random generator
	seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
	if err != nil {
		log.Crit("Failed to initialize random seed", "err", err)
	}
	return &ForkChoice{
		chain:    chainReader,
		rand:     mrand.New(mrand.NewSource(seed.Int64())),
		preserve: preserve,
	}
}

func (f *ForkChoice) CommonAncestor(current *types.Header, header *types.Header) (*types.Header, error) {
	oldH, newH := types.CopyHeader(current), types.CopyHeader(header)
	var commonAncestor *types.Header

	// Reduce the longer chain to the same number as the shorter one.
	if oldH.Number.Uint64() > newH.Number.Uint64() {
		for ; oldH != nil && oldH.Number.Uint64() != newH.Number.Uint64(); oldH = f.chain.GetHeader(oldH.ParentHash, oldH.Number.Uint64()-1) {
			// noop (txes and logs aggregation not handled here)
		}
	} else {
		for ; newH != nil && newH.Number.Uint64() != oldH.Number.Uint64(); newH = f.chain.GetHeader(newH.ParentHash, newH.Number.Uint64()-1) {
			// noop
		}
	}

	// Both sides of the reorg are at the same number, reduce both until the
	// common ancestor is found.
	for {
		if oldH.Hash() == newH.Hash() {
			commonAncestor = oldH
			break
		}
		oldH = f.chain.GetHeader(oldH.ParentHash, oldH.Number.Uint64()-1)
		if oldH == nil {
			return nil, fmt.Errorf("invalid oldH chain")
		}

		newH = f.chain.GetHeader(newH.ParentHash, newH.Number.Uint64()-1)
		if newH == nil {
			return nil, fmt.Errorf("invalid newH chain")
		}
	}
	return commonAncestor, nil
}

// ReorgNeeded returns whether the reorg should be applied
// based on the given external header and local canonical chain.
// In the td mode, the new head is chosen if the corresponding
// total difficulty is higher. In the extern mode, the trusted
// header is always selected as the head.
func (f *ForkChoice) ReorgNeeded(current *types.Header, extern *types.Header) (bool, error) {
	var (
		localTD  = f.chain.GetTd(current.Hash(), current.Number.Uint64())
		externTd = f.chain.GetTd(extern.Hash(), extern.Number.Uint64())
	)
	log.Debug("🔗 ForkChoice.ReorgNeeded", "current.number", current.Number.Uint64(), "extern.number", extern.Number.Uint64(), "localTD", localTD, "externTd", externTd)
	if localTD == nil || externTd == nil {
		log.Error("❌ ForkChoice: Missing TD", "localTD", localTD, "externTd", externTd, "current.hash", current.Hash().Hex()[:10], "extern.hash", extern.Hash().Hex()[:10])
		return false, errors.New("missing td")
	}
	// Accept the new header as the chain head if the transition
	// is already triggered. We assume all the headers after the
	// transition come from the trusted consensus layer.
	if ttd := f.chain.Config().GetEthashTerminalTotalDifficulty(); ttd != nil && ttd.Cmp(externTd) <= 0 {
		return true, nil
	}

	// Primary rule: Compare total difficulty (standard PoW consensus)
	if externTd.Cmp(localTD) > 0 {
		log.Debug("🔗 Accepting chain with higher total difficulty", 
			"localTD", localTD, "externTd", externTd,
			"tdDiff", new(big.Int).Sub(externTd, localTD))
		return true, nil
	}

	// QUANTUM FIX: Only accept higher block numbers if total difficulties are very close
	// This handles cases where quantum difficulty calculation might have minor variations
	// but prevents dangerous reorgs based solely on block numbers
	if extern.Number.Uint64() > current.Number.Uint64() {
		// Calculate TD difference as percentage
		tdDiff := new(big.Int).Sub(localTD, externTd)
		tdDiff.Abs(tdDiff) // Get absolute difference
		
		// Only accept if TD difference is less than 1% of local TD
		onePercent := new(big.Int).Div(localTD, big.NewInt(100))
		
		if tdDiff.Cmp(onePercent) <= 0 {
			log.Info("🔗 QUANTUM: Accepting higher block number with similar total difficulty",
				"current.number", current.Number.Uint64(),
				"extern.number", extern.Number.Uint64(),
				"localTD", localTD,
				"externTd", externTd,
				"tdDiff", tdDiff,
				"tdThreshold", onePercent)
			return true, nil
		} else {
			log.Warn("🔗 QUANTUM: Rejecting higher block number with significantly lower total difficulty",
				"current.number", current.Number.Uint64(),
				"extern.number", extern.Number.Uint64(),
				"localTD", localTD,
				"externTd", externTd,
				"tdDiff", tdDiff,
				"tdThreshold", onePercent)
		}
	}

	// Local and external difficulty is identical or very close.
	// Second clause in the if statement reduces the vulnerability to selfish mining.
	// Please refer to http://www.cs.cornell.edu/~ie53/publications/btcProcFC.pdf
	reorg := externTd.Cmp(localTD) > 0
	tie := externTd.Cmp(localTD) == 0
	if tie {
		externNum, localNum := extern.Number.Uint64(), current.Number.Uint64()
		if externNum < localNum {
			reorg = true
		} else if externNum == localNum {
			var currentPreserve, externPreserve bool
			if f.preserve != nil {
				currentPreserve, externPreserve = f.preserve(current), f.preserve(extern)
			}
			reorg = !currentPreserve && (externPreserve || f.rand.Float64() < 0.5)
		}
	}

	// If reorg is not needed (false), then we can just return.
	// The following logic adds a condition only in the case where a reorg would
	// otherwise be indicated.
	if !reorg {
		return reorg, nil
	}

	if bc, ok := f.chain.(*BlockChain); ok {
		// Short circuit if not configured for Artificial Finality.
		if !bc.IsArtificialFinalityEnabled() {
			return reorg, nil
		}
	}
	if !f.chain.Config().IsEnabled(f.chain.Config().GetECBP1100Transition, current.Number) {
		return reorg, nil
	}

	commonHeader, err := f.CommonAncestor(current, extern)
	if err != nil {
		return reorg, err
	}

	if err := ecbp1100(commonHeader, current, extern, f.chain.GetTd); err != nil {
		reorg = false
		log.Warn("Reorg disallowed", "error", err)
	} else if current.Number.Uint64()-commonHeader.Number.Uint64() > 2 {
		// Reorg is allowed, only log the MESS line if old chain is longer than normal.
		log.Info("ECBP1100-MESS 🔓",
			"status", "accepted",
			"age", common.PrettyAge(time.Unix(int64(commonHeader.Time), 0)),
			"current.span", common.PrettyDuration(time.Duration(current.Time-commonHeader.Time)*time.Second),
			"proposed.span", common.PrettyDuration(time.Duration(extern.Time-commonHeader.Time)*time.Second),
			"common.bno", commonHeader.Number.Uint64(), "common.hash", commonHeader.Hash(),
			"current.bno", current.Number.Uint64(), "current.hash", current.Hash(),
			"proposed.bno", extern.Number.Uint64(), "proposed.hash", extern.Hash(),
		)
	}

	return reorg, nil
}
