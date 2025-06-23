Below is a **step-by-step implementation roadmap** for Quantum-Geth v0.9–BareBones+Halving. Each item builds on the previous and, once complete, yields a fully functional blockchain exactly matching the spec. Tick each box as you finish:

---

## Implementation Roadmap

* [x] **Specification Freeze & Constant Embedding**
  Define and lock all genesis constants and parameter schedules in code:

  * `ProofSystemHash`, `TemplateAuditRoot_v2`, `GlideTableHash`, `CanonicompSHA`, `ChainIDHash`
  * Halving epoch length (600 000 blocks), initial subsidy (50 QGC), subsidy‐to‐zero schedule
  * Immutable glide schedule for QBits/TCount/LNet bumps

* [x] **QASM-lite Grammar & Parser**
  Publish the formal BNF of QASM-lite and implement a reference parser/serializer:

  * Cover H, S, T, CX, Z-mask directives
  * Include exhaustive test vectors: seed→QASM→serialized bytes

* [x] **Canonical-Compile Module**
  Build the deterministic compile pipeline:

  1. QASM-lite → DAG (Qiskit opt\_level=0)
  2. Apply Pauli-Z mask from `SHA256(seed)`
  3. Serialize gate list byte-exactly
  4. Unit-test against known seed/QASM→stream pairs

* [x] **Branch-Template Engine**
  Encode the 16 iso-hard branch templates and PRP instantiation:

  * Store template skeletons in QASM-lite form
  * Implement Keccak-based PRP to fill rotation angles/CX mapping from low‐bits
  * Validate depth/T-count invariants for all templates

* [x] **Puzzle Orchestrator & Seed-Chain**
  Implement mining loop that for each `QNonce64` / `ExtraNonce32` does:

  * Compute `Seed0`; for i in \[0…47]: compile branch\_i, execute, record `Outcome_i`, chain to `Seed_{i+1}`
  * Build `OutcomeRoot` (Merkle of outcomes) and `BranchNibbles`
  * Compute `GateHash` across compiled streams

* [x] **Quantum Backend Abstraction**
  Define an interface allowing interchangeable backends:

  * Qiskit AerSimulator (state-vector / tensor)
  * Qiskit Runtime (IBM Eagle/Heron)
  * Vendor shim API (QASM-lite in → bitstring out)
  * Inject noise-model toggles for future testing

* [x] **Mahadev Trace & CAPSS Integration**
  Wire in Urmila Mahadev's witnessing code to produce 48 interactive traces;
  wrap each into a CAPSS SNARK:

  * Automate transcript capture
  * Generate 2.2 kB proofs per puzzle
  * Ensure end-to-end trace→proof correctness

* [x] **Nova-Lite Recursive Aggregation**
Batch the 48 CAPSS proofs into 3 Nova-Lite proofs (≤ 6 kB each):

  * Construct Merkle tree over CAPSS proofs, compute `ProofRoot`
  * Implement tier-B proof generation & streaming API
  * Unit-test recursive verification logic

* [x] **Dilithium-2 Self-Attestation Module**
  Deterministic key derivation & signing:

  * `Seed_att = SHA256("ATTEST"‖Seed₀‖OutcomeRoot‖ChainIDHash‖BlockNumber)`
  * CBD sampler → `(sk,pk)` per NIST spec
  * Sign `(Seed₀‖OutcomeRoot‖GateHash)`; enforce public-key norm guard
  * Verify round-trip keygen/sign/verify

* [x] **RLP Header Extension & Tail Handling**
  Modify Geth's `Header` struct to append a single `[]byte` RLP-tail:

  * Ensure first 15 fields unchanged
  * Encode quantum blob fields in one contiguous slice
  * Add runtime sanity check on header‐encode length

* [x] **Proof & Attestation Integration in Block Assembly**
  In miner code, integrate:

  * `QBits, TCount, LNet` fields from config
  * `OutcomeRoot, BranchNibbles, GateHash, ProofRoot`
  * `pk+Sig` appended to block body
  * RLP‐encode and seal

* [x] **ASERT-Q Difficulty Filter**
  Embed Bitcoin-style exponential ASERT with λ=0.12:

  * `newTarget = oldTarget × 2^((t_actual–12 s)/150 s)`
  * ±10 % per‐block clamp
  * Integrate into block‐validation logic

* [x] **Halving & Fee Model Implementation**
  Implement reward schedule:

  * Start subsidy = 50 QGCoins; at each epoch (600 000 blocks), subsidy := subsidy/2
  * Track current subsidy via block height
  * Sum all transaction fees in block; award `Subsidy + Fees` to coinbase

* [x] **Block Validation Pipeline**
  Extend full‐node verify path to:

  1. RLP‐decode classical header + quantum blob
  2. Canonical‐compile & GateHash check
  3. Nova proof verify (Tier-B)
  4. Dilithium signature verify
  5. PoW target test via ASERT-Q
  6. EVM execution & state transition

* [x] **Audit-Guard Rail Enforcement**
  On startup, verify embedded `ProofSystemHash` and `TemplateAuditRoot_v2`:

  * Fetch audit artifacts, validate Merkle roots
  * Refuse to operate if any root mismatches

* [x] **P2P & Sync Adjustments**
  Ensure peers propagate full quantum blob + proofs:

  * Update block gossip protocols to include tail and proofs
  * Implement length/index/CID framing for proof chunks with timeout

* [x] **CLI & RPC Exposure**
  Expose status endpoints for miners:

  * Current block substrate (QBits, TCount, difficulty, subsidy)
  * Mining progress: puzzles/sec, nonce, proof sizes
  * RPC methods for generating block templates and submitting blocks

* [x] **Metrics & Instrumentation**
  Integrate Prometheus metrics:

  * Puzzle‐generation latency, proof‐generation time
  * Dilithium sign/verify duration
  * ASERT adjustment deviations
  * Subsidy and fee‐distribution stats

* [x] **Comprehensive End-to-End Validation**
  Execute a full chain run in a single process to:

  * Mine and validate >100 blocks in sequence
  * Verify halving events occur precisely at epoch boundaries
  * Stress‐test under concurrent mining agents

* [x] **Documentation & Reference Release**
  Publish:

  * Developer guide for each module (canonical compile, proof, attestation)
  * API reference for mining RPCs and backend adapters
  * On-chain RLP tail schema specification

---

*When all boxes are checked, you will have a fully working, governance-free quantum PoW blockchain with Bitcoin-style halving and full fee incentives, matching the v0.9–BareBones+Halving specification.*
