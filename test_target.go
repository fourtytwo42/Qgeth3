package main

import (
	"fmt"
	"math/big"
)

func main() {
	// Simulate the DifficultyToTarget calculation
	maxTarget := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	scalingFactor := big.NewInt(100000000) // 100M

	fmt.Printf("MaxTarget: %s\n", maxTarget.String())
	fmt.Printf("Scaling Factor: %s\n", scalingFactor.String())

	// Difficulty 199
	diff199 := big.NewInt(199)
	target199 := new(big.Int).Set(maxTarget)
	target199.Div(target199, diff199)
	target199.Mul(target199, scalingFactor)

	// Difficulty 200
	diff200 := big.NewInt(200)
	target200 := new(big.Int).Set(maxTarget)
	target200.Div(target200, diff200)
	target200.Mul(target200, scalingFactor)

	fmt.Printf("Difficulty 199 target: %s\n", target199.String())
	fmt.Printf("Difficulty 200 target: %s\n", target200.String())

	// Calculate ratio
	ratio := new(big.Float).Quo(new(big.Float).SetInt(target199), new(big.Float).SetInt(target200))
	fmt.Printf("Ratio (199/200): %s\n", ratio.String())

	// Calculate the difference
	diff := new(big.Int).Sub(target199, target200)
	fmt.Printf("Difference: %s\n", diff.String())

	// Calculate percentage change
	percentChange := new(big.Float).Quo(new(big.Float).SetInt(diff), new(big.Float).SetInt(target200))
	percentChange.Mul(percentChange, big.NewFloat(100))
	fmt.Printf("Percentage change: %s%%\n", percentChange.String())
}
