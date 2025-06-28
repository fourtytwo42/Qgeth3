// EVMC stub for Windows builds

package vm

import (
	"sync"

	"github.com/ethereum/go-ethereum/log"
)

// Stub implementations for EVMC when CGO is disabled

var evmcMux sync.Mutex

func InitEVMCEVM(config string) {
	log.Info("EVMC EVM disabled (CGO disabled)")
}

func InitEVMCEwasm(config string) {
	log.Info("EVMC EWASM disabled (CGO disabled)")
}
