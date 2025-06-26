package params

// QCoinMainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin mainnet network.
var QCoinMainnetBootnodes = []string{
	"enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@143.110.231.183:30303", // Q Coin VPS mainnet bootnode
}

// QCoinTestnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin testnet network.
var QCoinTestnetBootnodes = []string{
	// TODO: Replace with dedicated testnet nodes when available
	// Using dev network nodes temporarily (different ports for testnet)
	"enode://445846afd49a7a8a70b9cc5beb5de86a283fd4b77063ab79f5673bd898a1c561e690ecf52b3b9802943de56c1e844a2b5ad9e4d6a4009569d2016f20fd9bb3a0@64.23.179.84:30303",     // Q Coin Testnet VPS 1 (port 30303)
	"enode://a147d83e40644ba880b5378254881e37e66eb821a5617cf36412d088b5ad76698b0b4f260ffda118c2a1b3319e005a3d8d849c0df59d7c58c5284ee0c7cd8375@143.110.231.183:30303", // Q Coin Testnet VPS 2 (port 30303)
}

// QCoinDevBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin development network.
var QCoinDevBootnodes = []string{
	"enode://fd85117c596739432dcd5030ba9173b8cfcdda6f53767526a9cbe8b6c7e83afa69e9048e5a7629ddd1a718fd4c00ff3e7fc1584cc8ed7b1963394f43d87fd1b0@64.23.179.84:30305",  // Q Coin Dev VPS 1
	"enode://6cd6f9185e7cc14bf68a502b272d3624d5778334ddfb1704dc9a7dd54c94512f28822df551a29b5d4c39eca6d7fa11b7dfc4149f88fc806e9463307f7ff240dd@143.110.231.183:30305", // Q Coin Dev VPS 2
} 