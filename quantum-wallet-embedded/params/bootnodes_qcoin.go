package params

// QCoinMainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin mainnet network.
var QCoinMainnetBootnodes = []string{
	"enode://89df9647d6f5b901c63e8a7ad977900b5ce2386b916ed6d204d24069435740c7e2c188c9d3493bfc98c056d9d87c6213df057e9518fb43f12759ba55dff31b4c@143.110.231.183:30303", // Q Coin VPS mainnet bootnode
}

// QCoinTestnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin testnet network.
var QCoinTestnetBootnodes = []string{
	"enode://6300bb04bd88484b7c30ea47036c2bf3b2b544a3ef106817432a83cc99e6f806592aff0f49cc33b99440e837aea3b02dba7b051f89c1b7bf9a85d55c9906b079@134.199.202.42:30303", // Q Coin Testnet VPS 1 (updated)
	"enode://42e18390fa52923947e7ad009a84a35ef8ff71a166171af3b72ee6fb201601c461066f2c17944430e2b8ab6d043b60aa499d8d51766f4fb7563183e44df3c12a@128.199.6.197:30303",   // Q Coin Testnet VPS 2 (updated)
}

// QCoinDevBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the Q Coin development network.
var QCoinDevBootnodes = []string{
	"enode://fb63f743979b4a72eb87ad779e0444b122569b0bda9e009d6d10cad389f5bfcc346786b6c5de82c57b24582797a40deb227a54ed56f40b9c96cca375d09c9eb8@64.23.179.84:30305",    // Q Coin Dev VPS 1 (newest)
	"enode://53e65f1627335d2df75918c455e8a1b59efbea574b0c90006b0413c3eeeec5bfcbe84ea774a5cc47e5281161c60fecf60c7a45547bcdde1a9f50f5b100f46fdf@143.110.231.183:30305", // Q Coin Dev VPS 2 (newest)
} 