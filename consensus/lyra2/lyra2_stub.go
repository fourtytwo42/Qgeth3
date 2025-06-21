//go:build !cgo
// +build !cgo

package lyra2

import (
	"math/big"
	"math/rand"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/rpc"
)

type Lyra2 struct {
	fakeMode  bool
	fakeFail  uint64
	fakeDelay time.Duration

	log  log.Logger
	lock sync.Mutex

	rand     *rand.Rand
	hashrate metrics.Meter
	update   chan struct{}
	threads  int
	remote   *remoteSealer
}

type Config struct {
	FakeMode  bool
	FakeFail  uint64
	FakeDelay time.Duration

	Log  log.Logger
	Rand *rand.Rand
}

func New(config *Config, notify []string, noverify bool) *Lyra2 {
	if config == nil {
		config = &Config{}
	}
	lyra2 := &Lyra2{
		fakeMode:  true, // Force fake mode when CGO disabled
		fakeFail:  config.FakeFail,
		fakeDelay: config.FakeDelay,
		log:       log.Root(),
		hashrate:  metrics.NewMeter(),
		update:    make(chan struct{}),
	}
	if config.Log != nil {
		lyra2.log = config.Log
	}
	if config.Rand != nil {
		lyra2.rand = config.Rand
	}
	lyra2.remote = startRemoteSealer(lyra2, notify, noverify)
	return lyra2
}

func NewTester(notify []string, noverify bool) *Lyra2 {
	lyra2 := &Lyra2{
		fakeMode:  true, // Force fake mode when CGO disabled
		fakeFail:  0,
		fakeDelay: 0,
		log:       log.Root(),
		hashrate:  metrics.NewMeter(),
		update:    make(chan struct{}),
	}
	lyra2.remote = startRemoteSealer(lyra2, notify, noverify)
	return lyra2
}

func (lyra2 *Lyra2) calcHash(headerBytes []byte, nonce uint64, tcost int) *big.Int {
	// Stub implementation - just return a hash based on header + nonce
	hash := common.BytesToHash(headerBytes)
	return hash.Big()
}

func (lyra2 *Lyra2) compute(ctx interface{}, blockBytes []byte, nonce uint64, tcost int) common.Hash {
	// Stub implementation - simple hash without Lyra2
	return common.BytesToHash(blockBytes)
}

func (lyra2 *Lyra2) Close() error {
	return nil
}

// APIs implements consensus.Engine, returning the user facing RPC APIs.
func (lyra2 *Lyra2) APIs(chain consensus.ChainHeaderReader) []rpc.API {
	return []rpc.API{
		{
			Namespace: "eth",
			Version:   "1.0",
			Service:   &API{lyra2},
			Public:    true,
		},
	}
}

func (lyra2 *Lyra2) Hashrate() float64 {
	return lyra2.hashrate.Snapshot().Rate1()
}

// Threads returns the number of mining threads currently enabled. This doesn't
// necessarily mean that mining is running!
func (lyra2 *Lyra2) Threads() int {
	lyra2.lock.Lock()
	defer lyra2.lock.Unlock()

	return lyra2.threads
}

// SetThreads updates the number of mining threads currently enabled. Calling
// this method does not start mining, only sets the thread count. If zero is
// specified, the miner will use all cores of the machine. Setting a thread
// count below zero is allowed and will cause the miner to idle, without any
// work being done.
func (lyra2 *Lyra2) SetThreads(threads int) {
	lyra2.lock.Lock()
	defer lyra2.lock.Unlock()

	// Update the threads and ping any running seal to pull in any changes
	lyra2.threads = threads
	select {
	case lyra2.update <- struct{}{}:
	default:
	}
}
