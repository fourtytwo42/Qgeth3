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
	"enode://2dce2bce7256fd8b8da9bf3698288e93b1e80a7865d4261be60389ef3f4d531c90473ff50a7d150638f951e41268f0ba2e05c7d8c76ebcdcafe6f1e1f1f7020d@64.23.179.84:30305",    // Q Coin Dev VPS 1 (latest)
	"enode://53e65f1627335d2df75918c455e8a1b59efbea574b0c90006b0413c3eeeec5bfcbe84ea774a5cc47e5281161c60fecf60c7a45547bcdde1a9f50f5b100f46fdf@143.110.231.183:30305", // Q Coin Dev VPS 2 (latest)
} 