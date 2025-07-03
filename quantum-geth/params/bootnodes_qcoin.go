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
	"enode://09fc6379b4e820367db08cf0b8ec823d25ee7b399f4c10f6347ea61e58a4ffd29fab198466bce49c4d223ad079817e049e7acc8d28f1aa3399fc25384fc9761b@139.59.120.94:30303",  // Q Coin Testnet VPS 3 (new)
	"enode://b9f4a40c421eaef3e0ca3d89f15641e53696e02dad4c66930a5a8406d49a7e56cd64c2a9c21f18693922e2a5c6d00d87419d0e2fe1a2d7e55a44a8c7f534ea01@209.97.131.210:30303", // Q Coin Testnet VPS 4 (new)
}

// QCoinDevBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin development network.
var QCoinDevBootnodes = []string{
	"enode://fb63f743979b4a72eb87ad779e0444b122569b0bda9e009d6d10cad389f5bfcc346786b6c5de82c57b24582797a40deb227a54ed56f40b9c96cca375d09c9eb8@64.23.179.84:30305",    // Q Coin Dev VPS 1 (newest)
	"enode://53e65f1627335d2df75918c455e8a1b59efbea574b0c90006b0413c3eeeec5bfcbe84ea774a5cc47e5281161c60fecf60c7a45547bcdde1a9f50f5b100f46fdf@143.110.231.183:30305", // Q Coin Dev VPS 2 (newest)
}

// QCoinPlanckBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin Planck testnet network.
var QCoinPlanckBootnodes = []string{
	"enode://d95877e097ab3bc6a8f3c5aee4a508d9a950810f149a53d50b11b0fbbf8b7c2f139b01361e514d30054c080f46f3eb00f5a4f2e665ea1a8de2cd7ee37079f0b5@128.199.6.197:30307",   // Q Coin Planck VPS 1
	"enode://587f338c3168ea2d273dca416f923581b95f1c3c4056e897aa4b91e4688e1bc1400decadb0b97a2c52b6eab48cb6d77192c88e97b72f7d8ef1f46d0e46f92900@209.97.131.210:30307",  // Q Coin Planck VPS 2  
	"enode://f11a05bbfcb32db89077c0bcee94de13fc1ad5cbe40caee78141cc21bb5f7de62cd6bd23aad846548a66fecb040663bef9cf2a2ba874c9451401c455bfe700da@134.199.202.42:30307",  // Q Coin Planck VPS 3
	"enode://f7352a2475bb12ba7b444b2eeb0b041f991f164b3f460d31f7fd2aeb139035619d5a43ca3e6214fdf3a97fc816fd416e452b64cee738932a8c204f6480a7f749@139.59.120.94:30307",   // Q Coin Planck VPS 4
} 