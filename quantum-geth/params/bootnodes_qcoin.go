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
	"enode://ebdd335675512b75c925c87776c6e2ada956dd1436782940dc457158da86ca77a9e6c638f01990ba14eab45f01a0e8d9dbe963569bddb16832cb9bbc27f5eb8c@143.110.231.183:30305", // Q Coin Dev VPS 1 (updated)
	"enode://a4e05766ef3b8ba5cc3445cb0500a9c37f8b713e43528d9a9089ce478c85d8472a1b6d088b079631cf2f4aa6ad4265f776eba062194aa83f49f086c27d380e17@64.23.179.84:30305",    // Q Coin Dev VPS 2 (updated)
} 