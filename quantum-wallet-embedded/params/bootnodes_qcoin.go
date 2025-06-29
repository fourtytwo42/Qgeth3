package params

// QCoinMainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin mainnet network.
var QCoinMainnetBootnodes = []string{
	"enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@143.110.231.183:30303", // Q Coin VPS mainnet bootnode
}

// QCoinTestnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin testnet network.
var QCoinTestnetBootnodes = []string{
	"enode://dde79dc6c2894dd543f52b2490bfc33d330c01bb9433376d7d8860c26e3081b71f66135430402863460414bef945bdb9bc7536fa727af0d641e39f09c28ac25b@134.199.202.42:30303", // Q Coin Testnet VPS 1 (updated)
	"enode://fe17dd35ad0b6a4acb7074bd3a3d0c619f2f2d286c13eae6492b90382af02fa0eae9ee9b11ae639fd11ac54770e1d1de949f52160170498c7aaeda3dfbcc5089@128.199.6.197:30303",   // Q Coin Testnet VPS 2 (updated)
}

// QCoinDevBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin development network.
var QCoinDevBootnodes = []string{
	"enode://fb63f743979b4a72eb87ad779e0444b122569b0bda9e009d6d10cad389f5bfcc346786b6c5de82c57b24582797a40deb227a54ed56f40b9c96cca375d09c9eb8@64.23.179.84:30305",    // Q Coin Dev VPS 1 (newest)
	"enode://53e65f1627335d2df75918c455e8a1b59efbea574b0c90006b0413c3eeeec5bfcbe84ea774a5cc47e5281161c60fecf60c7a45547bcdde1a9f50f5b100f46fdf@143.110.231.183:30305", // Q Coin Dev VPS 2 (newest)
} 