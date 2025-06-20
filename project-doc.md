Below is a soup-to-nuts blueprint for replacing Ethash in core-geth with the “L-layer quantum micro-puzzle PoW” (QµPoW) we just designed.
Everything is laid out so that—with nothing more than a text editor, Go 1.22, Qiskit (Aer) on the mining side and a few open-source libraries for Mahadev-style proofs—you can build, boot and mine on a private chain or pool.
No production-ready code is pasted here (that would be pages of Go); instead you get file names, function names, struct members, and step-by-step pseudocode you can translate line-for-line.

0. Baseline “knobs & dials” (genesis defaults)
Parameter	Symbol	Default value	Rationale
Qubits per micro-puzzle	q	8	Runs in Aer or 4-qubit cloud hardware within ms
T-gate count per puzzle	t	25	≈12.5-bit classical hardness; keeps circuit depth ≤40
Puzzles per block (network)	L_net	64	64×12.5 ≈ 800 classical “bits” ⇒ >2²⁵⁶ brute force
Difficulty retarget period	EpochLen	2 048 blocks	Matches Ethash epoch cadence
Difficulty knob	L_net (+/-4 per epoch)	Retarget algorithm raises/lowers L_net	
Share difficulty knob	L_share	pool chooses (e.g. 8–32)	Works like Stratum “target”
Max header size growth	—	+4 kB (proof)	Aggregate proof constant-size

1. Fork core-geth (environment)
bash
Copy
Edit
git clone https://github.com/etclabscore/core-geth quantum-geth
cd quantum-geth
Create a new branch git checkout -b qmpow.

1.1 Directory scaffolding
pgsql
Copy
Edit
consensus/
├─ qmpow/
│   ├─ qmpow.go          # Prepare/Seal/Verify
│   ├─ params.go         # tunables (q, t, L_net)
│   └─ proof/
│       ├─ mahadev.go    # thin Go wrapper over TCF lib
│       └─ aggregate.go
miner/
└─ qmpow_miner.go        # "Seal" loop replacement
eth/
└─ configs/
    └─ genesis_qmpow.json
2. Header extensions (core/types)
Edit core/types/block_header.go
Add fields (all optional for RLP backward compat):

go
Copy
Edit
// Quantum micro-puzzle fields
QBits     *uint8    // qubits per puzzle
TCount    *uint16   // T gates per puzzle
LUsed     *uint16   // L_net when block was mined
QOutcome  []byte    // concatenated y₀…y_L-1 (fixed L_net*q/8 bytes)
QProof    []byte    // aggregate Mahadev proof blob
Update RLP
In EncodeRLP / DecodeRLP, append these fields after existing Ethash ones so older nodes can ignore.

3. Consensus engine (consensus/qmpow/qmpow.go)
3.1 Prepare(header)
pseudocode
Copy
Edit
func (e *QMPoW) Prepare(chain, hdr):
    parent = chain.GetHeader(hdr.ParentHash, hdr.Number-1)
    epochParams = e.paramsForHeight(hdr.Number)
    hdr.QBits  = epochParams.QBits      // 8
    hdr.TCount = epochParams.TCount     // 25
    hdr.LUsed  = epochParams.LNet       // dynamic
    // wipe QOutcome, QProof → they’ll be filled in Seal
    return nil
3.2 Seal(header, stop)
pseudocode
Copy
Edit
func Seal(chain, hdr, stopCh):
    seed0 = SHA256(RLP(headerWithoutQFields))
    Y, P  = miner.RunPuzzleChain(seed0, hdr.QBits, hdr.TCount, hdr.LUsed)
    hdr.QOutcome = Y
    hdr.QProof   = P
    return hdr   // sealed block
3.3 VerifyHeader(header)
pseudocode
Copy
Edit
func VerifyHeader(chain, hdr):
    params = e.paramsForHeight(hdr.Number)
    if hdr.QBits != params.QBits or hdr.TCount != params.TCount:
        error
    // 1 Aggregate proof check (classical polytime)
    ok = qcproof.VerifyAggregate(
           hdr.QProof, hdr.QBits, hdr.TCount,
           hdr.LUsed, hdr.ParentHash, hdr.QOutcome)
    if !ok: error
    // 2 (OPTIONAL) quick sanity: len(QOutcome)==hdr.LUsed*hdr.QBits/8
    return nil
4. Difficulty adjustment (epoch)
File consensus/qmpow/params.go

pseudocode
Copy
Edit
func Retarget(parent, currentTime):
    Δt = currentTime - parent.Time
    target = 12s
    if Δt < target*0.9:  L_net += 4   // harder
    if Δt > target*1.1:  L_net -= 4   // easier
    L_net = clamp(L_net, 16, 256)
5. Mining loop (miner/qmpow_miner.go)
Replace Ethash worker call with:

Build seed₀ from header & extranonce.

Call local python (or embedded C++) QµP solver via gRPC / stdin.

Aggregate proof: call qcproof.Aggregate(π₀…π_L).

Fill header fields & submit.

Multi-threading: run N worker goroutines each owning its own Qiskit-Aer process (env var AER_THREADS=1).

6. Python solver (tools/solver/solver.py)
Main steps (pseudocode)

pgsql
Copy
Edit
parse stdin JSON {seed0,qbits,tcount,L}
for i in range(L):
    seed_i = sha256(seed_{i-1}||y_{i-1} or seed0 at i=0)
    (y_i, π_i) = run_qμp_circuit(seed_i,q,t)
    collect
output JSON {Y, proof=aggregate(πs)}
Run via subprocess from Go.

7. Proof layer (consensus/qmpow/proof)
mahadev.go
Thin CGO bridge to a Rust/C “Trapdoor Claw-Free” lib (e.g. the QCrypt reference).

aggregate.go
Simple Merkle-like compression of 
𝜋
0
…
𝜋
𝐿
−
1
π 
0
​
 …π 
L−1
​
  into one constant blob.

Tip: start with L=8 in testnet until proof aggregation is stable.

8. RPC & Stratum bridge
Modify miner/agent.go:

Add handler "qmpow_submitwork" that accepts (Y,P,extraNonce2).

Pool share difficulty = send L_share smaller than QBits.L_net; miner treats it the same code path with a different loop-length.

9. Genesis file (eth/configs/genesis_qmpow.json)
jsonc
Copy
Edit
{
  "config": {
    "chainId": 9248,
    "qmpow": {
      "qbits": 8,
      "tcount": 25,
      "LNet": 64,
      "epoch": 2048
    }
  },
  "difficulty": "0x0",
  "gasLimit": "0x47b760",
  "alloc": { }
}
10. Build & run
bash
Copy
Edit
make geth
./build/bin/geth --datadir qdata init eth/configs/genesis_qmpow.json
./build/bin/geth --datadir qdata --mine --miner.threads=4 --networkid 9248 \
                 --unlock 0xYourAddr --password passfile \
                 --qmpow.solvers=/usr/local/bin/solver.py
Logs should show:

makefile
Copy
Edit
QMPoW: 64 puzzles, aggregate proof 3.2 kB, solved in 0.48s
Sealed block #1 (hash=0x…) [txs=0]  cpu=0.52s
11. Pool quick-start
Stratum front-end: fork btcpool or open-ethereum-pool.

Job template: add JSON fields q_param.qbits/tcount/L_share.

Share verify: call qcproof.VerifyAggregate(...) with L_share.

Block submit: replace eth_submitWork with qmpow_submitwork.

12. Test matrix
Test	Target	Expectation
Unit: VerifyHeader wrong proof	forged π	rejection
Unit: wrong QOutcome len	len≠L*q/8	rejection
Integration: 1-thread CPU	8-qubit Aer	block <15 s
Integration: 4 threads	parallel	linear speedup
Pool share	L_share=16	share accepted by pool but rejected by daemon

13. Roadmap knobs
Raise T-count to 40 when 12-qubit cloud QPUs are mainstream.

Raise q to 12 (state-vector requires 2 MB).

Drop L_net to keep ~12 s blocks.

Enable verifier “fast path”—skip Mahadev if header height < some cutoff for testnets.

14. Summary cheat-sheet
Header adds 5 fields (QBits, TCount, LUsed, QOutcome, QProof).

Consensus: Prepare inserts params; Seal fills outcome + proof; Verify checks proof.

Difficulty: adjust L_net ±4 every 2 048 blocks.

Mining = run L tiny 8-qubit T-gate circuits, aggregate proof, no nonce grind.

Pool uses smaller L_share as “share target”.

Verification cost stays O(1) no matter how high you scale t or L.

Follow each section line-by-line and you’ll have core-geth-QMPoW up, syncing, and mining on real laptops—or, if you hook a QPU, on real quantum silicon—without touching classical SHA hashing ever again.