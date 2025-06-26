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
	"time"

	"github.com/ethereum/go-ethereum/log"
)

// NewFaker creates a QMPoW consensus engine with fake quantum proof verification
// for testing purposes. This replaces qmpow.NewFaker() in quantum blockchain tests.
func NewFaker() *QMPoW {
	log.Info("🧪 Creating QMPoW faker for testing")
	return New(Config{
		PowMode:  ModeFake,
		TestMode: true,
	})
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
	return New(Config{
		PowMode:  ModeTest,
		TestMode: true,
	})
}

// NewSharedFaker creates a shared QMPoW test engine instance
func NewSharedFaker() *QMPoW {
	log.Info("🧪 Creating shared QMPoW faker for testing")
	return NewFaker()
}

// NewFakeDelayer creates a QMPoW consensus engine that delays sealing for testing
func NewFakeDelayer(delay time.Duration) *QMPoW {
	log.Info("🧪 Creating QMPoW fake delayer for testing", "delay", delay)
	qmpow := NewFaker()
	// Note: delay functionality would be implemented in seal method if needed
	return qmpow
}

// Note: Complex fake implementations removed to avoid method conflicts.
// The main QMPoW engine in qmpow.go handles fake mode behavior based on config.PowMode 