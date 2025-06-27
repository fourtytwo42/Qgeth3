package main

import (
	"fmt"
	"math/big"
	
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/trie/triedb"
	"github.com/ethereum/go-ethereum/params"
)

func main() {
	fmt.Println("🧪 Testing quantum genesis block hash consistency...")
	
	// Create two separate genesis blocks using different paths
	genesis := params.QCoinTestnetGenesisBlock()
	
	// Path 1: Direct genesis block creation
	db1 := rawdb.NewMemoryDatabase()
	block1 := core.GenesisToBlock(genesis, db1)
	hash1 := block1.Hash()
	
	// Path 2: Genesis through CommitGenesis
	db2 := rawdb.NewMemoryDatabase()
	triedb2 := triedb.NewDatabase(db2, nil)
	block2, _ := core.CommitGenesis(genesis, db2, triedb2)
	hash2 := block2.Hash()
	
	// Display results
	fmt.Printf("📊 Genesis Block Hash Consistency Test:\n")
	fmt.Printf("   Path 1 (GenesisToBlock): %s\n", hash1.Hex())
	fmt.Printf("   Path 2 (CommitGenesis):  %s\n", hash2.Hex())
	fmt.Printf("   Hashes Match: %v\n", hash1 == hash2)
	
	// Check quantum field consistency
	header1 := block1.Header()
	header2 := block2.Header()
	
	fmt.Printf("🔬 Quantum Field Analysis:\n")
	fmt.Printf("   Block 1 WithdrawalsHash: %v\n", header1.WithdrawalsHash)
	fmt.Printf("   Block 2 WithdrawalsHash: %v\n", header2.WithdrawalsHash)
	fmt.Printf("   Block 1 ParentBeaconRoot: %v\n", header1.ParentBeaconRoot)
	fmt.Printf("   Block 2 ParentBeaconRoot: %v\n", header2.ParentBeaconRoot)
	
	if header1.WithdrawalsHash == nil && header2.WithdrawalsHash == nil {
		fmt.Println("✅ WithdrawalsHash correctly set to nil for both blocks")
	} else {
		fmt.Println("❌ WithdrawalsHash inconsistency detected!")
		fmt.Printf("   Block 1: %v, Block 2: %v\n", header1.WithdrawalsHash, header2.WithdrawalsHash)
	}
	
	if header1.ParentBeaconRoot == nil && header2.ParentBeaconRoot == nil {
		fmt.Println("✅ ParentBeaconRoot correctly set to nil for both blocks")
	} else {
		fmt.Println("❌ ParentBeaconRoot inconsistency detected!")
		fmt.Printf("   Block 1: %v, Block 2: %v\n", header1.ParentBeaconRoot, header2.ParentBeaconRoot)
	}
	
	if hash1 == hash2 {
		fmt.Println("🎉 SUCCESS: Genesis blocks now generate consistent hashes!")
		fmt.Println("🔧 The WithdrawalsHash consistency fix is working correctly.")
	} else {
		fmt.Println("⚠️  FAILURE: Genesis blocks still generate different hashes.")
		fmt.Println("   Additional investigation needed.")
	}
} 