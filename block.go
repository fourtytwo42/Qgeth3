// Quantum micro-puzzle fields for QMPoW consensus
QBits    *uint8       `json:"qBits" rlp:"optional"`    // qubits per puzzle
TCount   *uint16      `json:"tCount" rlp:"optional"`   // T gates per puzzle
LUsed    *uint16      `json:"lUsed" rlp:"optional"`    // L_net when block was mined (fixed at 48)
QNonce   QuantumNonce `json:"qNonce" rlp:"optional"`   // quantum nonce for Bitcoin-style mining (fixed-size like Bitcoin)
QOutcome []byte       `json:"qOutcome" rlp:"optional"` // concatenated y₀…y_L-1
QProof   []byte       `json:"qProof" rlp:"optional"`   // aggregate Mahadev proof blob 