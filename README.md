# Quantum-Geth v0.9–BareBones+Halving

A quantum-anchored proof-of-work blockchain implementation with Bitcoin-style halving economics.

## Overview

Quantum-Geth v0.9 implements a novel consensus mechanism that combines quantum computing with traditional blockchain technology. The system uses quantum puzzle solving for proof-of-work, with cryptographic proofs generated using Mahadev's verification protocol and Nova-Lite recursive aggregation.

## Architecture

### Core Components

1. **QASM-lite Grammar & Parser** (`qasm_lite.go`)
   - Quantum Assembly Language parser for quantum circuits
   - Supports quantum gates: H, CNOT, T, X, Y, Z
   - Circuit validation and compilation

2. **Canonical-Compile Module** (`canonical_compile.go`)
   - Converts QASM-lite to canonical gate representation
   - Implements T-gate optimization and depth calculation
   - Generates GateHash for proof verification

3. **Branch-Template Engine** (`branch_template.go`)
   - 16 pre-computed quantum circuit templates
   - Branch selection based on mining input
   - Template instantiation and validation

4. **Puzzle Orchestrator** (`puzzle_orchestrator.go`)
   - Manages 48-puzzle execution pipeline
   - Seed chain generation and puzzle distribution
   - Result aggregation and verification

5. **Quantum Backend Abstraction** (`quantum_backend.go`)
   - Pluggable quantum simulator interface
   - Hardware compatibility layer
   - Performance optimization

6. **Mahadev Trace & CAPSS Integration** (`mahadev_trace.go`)
   - Interactive transcript generation
   - CAPSS proof generation (~2.2 kB proofs)
   - Cryptographic verification

7. **Nova-Lite Recursive Aggregation** (`nova_lite.go`)
   - Recursive proof compression
   - 6.29x compression ratio
   - Efficient verification

8. **ASERT-Q Difficulty Algorithm** (`asert_q.go`)
   - Quantum-aware difficulty adjustment
   - 12-second block target
   - Genesis protection and clamping

9. **Dilithium-2 Self-Attestation** (`dilithium_attest.go`)
   - Post-quantum digital signatures
   - 1312B public keys, 2420B signatures
   - CBD sampling for key generation

10. **Halving & Fee Model** (`halving_fee_model.go`)
    - Bitcoin-style reward halving
    - 50 QGC initial subsidy
    - 600,000 block halving intervals

### Mining Process

```
1. Generate Seed₀ from mining input
2. Select branch template based on Seed₀
3. Execute 48 quantum puzzles in parallel
4. Generate CAPSS proofs for each puzzle
5. Aggregate proofs using Nova-Lite
6. Create Dilithium attestation
7. Assemble quantum block
```

### Block Structure

Quantum-Geth extends Ethereum's block header with quantum-specific fields:

```go
type Header struct {
    // Standard Ethereum fields...
    
    // Quantum extensions
    QNonce64     *uint64      // Quantum nonce
    QBits        *uint8       // Quantum bits
    TCount       *uint16      // T-gate count
    LNet         *uint16      // L-network parameter
    Epoch        *uint32      // Mining epoch
    OutcomeRoot  *common.Hash // Quantum outcome root
    GateHash     *common.Hash // Gate hash
    ProofRoot    *common.Hash // Aggregated proof root
}
```

## API Reference

### Mining RPCs

```go
// Get current block substrate for mining
GetBlockSubstrate() (*BlockSubstrate, error)

// Get mining progress information
GetMiningProgress() (*MiningProgress, error)

// Generate block template for external miners
GenerateBlockTemplate(minerAddress common.Address) (*BlockTemplate, error)

// Submit mined block for validation
SubmitBlock(block *types.Block, quantumBlob []byte, 
           attestationKey []byte, signature []byte) (*SubmissionResult, error)
```

### Validation RPCs

```go
// Get audit guard rail status
GetAuditStatus() (map[string]interface{}, error)

// Force audit verification
ForceAuditVerification() (map[string]interface{}, error)

// Get RPC service statistics
GetRPCStats() (*RPCStats, error)
```

## Configuration

### Genesis Parameters

```go
const (
    InitialSubsidyQGC = 50.0      // Initial block subsidy in QGCoins
    HalvingEpochSize  = 600000    // Blocks per halving epoch
    TargetBlockTime   = 12        // Target block time in seconds
    QGCToWeiConstant  = 1e18      // QGCoin to wei conversion
)
```

### Quantum Parameters

```go
const (
    DefaultQBits  = 5     // Default quantum bits
    DefaultTCount = 10    // Default T-gate count
    DefaultLNet   = 48    // Default L-network size
    MaxPuzzles    = 48    // Maximum puzzles per block
)
```

## Mining Guide

### Setup

1. Initialize QMPoW consensus engine:
```go
config := Config{
    PowMode:       ModeNormal,
    SolverPath:    "/path/to/quantum/solver",
    SolverTimeout: 30 * time.Second,
    TestMode:      false,
}
qmpow := New(config)
```

2. Create block assembler:
```go
chainIDHash := common.HexToHash("0x1234567890abcdef")
assembler := NewBlockAssembler(chainIDHash)
```

### Mining Loop

```go
for {
    // Get block template
    template, err := rpc.GenerateBlockTemplate(ctx, minerAddress)
    if err != nil {
        log.Error("Failed to get template", "err", err)
        continue
    }
    
    // Mine block
    block, quantumBlob, attestation, signature, err := mineBlock(template)
    if err != nil {
        log.Error("Mining failed", "err", err)
        continue
    }
    
    // Submit block
    result, err := rpc.SubmitBlock(ctx, block, quantumBlob, attestation, signature)
    if err != nil {
        log.Error("Submission failed", "err", err)
        continue
    }
    
    if result.Accepted {
        log.Info("Block accepted", "number", result.BlockNumber)
    }
}
```

## Testing

### Unit Tests

```bash
# Test individual components
go test ./dilithium_attest_test.go ./dilithium_attest.go
go test ./audit_guard_test.go ./audit_guard.go
go test ./nova_lite_test.go ./nova_lite.go
```

### Integration Tests

```bash
# Test full pipeline
go test -run TestNewBlockAssembler -v
go test -run TestQuantumBlockValidation -v
```

### Benchmarks

```bash
# Performance benchmarks
go test -bench=BenchmarkKeyGeneration
go test -bench=BenchmarkSigning
go test -bench=BenchmarkVerification
```

## Monitoring

### Metrics

The system exposes Prometheus-compatible metrics:

- `quantum_puzzles_solved_total`
- `quantum_proofs_generated_total`
- `dilithium_signatures_total`
- `block_assembly_duration_seconds`
- `mining_hashrate_attempts_per_second`

### Performance

Typical performance characteristics:

- **Key Generation**: ~23.5 μs
- **Signing**: ~11.2 μs  
- **Verification**: ~1.4 μs
- **Proof Generation**: ~250 ms
- **Block Assembly**: ~300 ms

## Security Considerations

### Quantum Security

- Post-quantum cryptography via Dilithium-2
- Quantum-resistant hash functions
- Future-proof against quantum attacks

### Audit Trail

- Embedded cryptographic roots verification
- Template consistency checking
- Proof system integrity validation

### Network Security

- P2P proof chunk transmission
- CID-based content addressing
- Timeout-based DoS protection

## Troubleshooting

### Common Issues

1. **Build Errors**
   ```bash
   # Ensure Go 1.19+ is installed
   go version
   
   # Clean module cache
   go clean -modcache
   go mod tidy
   ```

2. **Mining Issues**
   ```bash
   # Check quantum solver path
   ls -la /path/to/quantum/solver
   
   # Verify permissions
   chmod +x /path/to/quantum/solver
   ```

3. **Validation Failures**
   ```bash
   # Check audit status
   curl -X POST -H "Content-Type: application/json" \
        --data '{"method":"qmpow_getAuditStatus","params":[],"id":1}' \
        http://localhost:8545
   ```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the LGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ethereum Foundation for the base protocol
- NIST for post-quantum cryptography standards
- Research community for quantum verification protocols

---

**Quantum-Geth v0.9** - Bridging the quantum-classical divide in blockchain technology. 