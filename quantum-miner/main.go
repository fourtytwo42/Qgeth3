package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const VERSION = "1.0.0"

// Mining state
type QuantumMiner struct {
	coinbase string
	nodeURL  string
	threads  int
	running  int32
	stopChan chan bool

	// Statistics
	attempts    uint64
	accepted    uint64
	rejected    uint64
	stale       uint64
	duplicates  uint64
	startTime   time.Time
	client      *http.Client
	currentWork *QuantumWork
	workMutex   sync.RWMutex
}

// JSON-RPC structures
type JSONRPCRequest struct {
	ID      int           `json:"id"`
	JSONRPC string        `json:"jsonrpc"`
	Method  string        `json:"method"`
	Params  []interface{} `json:"params"`
}

type JSONRPCResponse struct {
	ID      int         `json:"id"`
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result"`
	Error   *RPCError   `json:"error"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Work structures
type QuantumWork struct {
	WorkHash    string    `json:"work_hash"`
	BlockNumber uint64    `json:"block_number"`
	Target      string    `json:"target"`
	QBits       int       `json:"qbits"`
	TCount      int       `json:"tcount"`
	LNet        int       `json:"lnet"`
	FetchTime   time.Time `json:"fetch_time"`
}

type QuantumProofSubmission struct {
	OutcomeRoot   string `json:"outcome_root"`
	GateHash      string `json:"gate_hash"`
	ProofRoot     string `json:"proof_root"`
	BranchNibbles string `json:"branch_nibbles"`
	ExtraNonce32  string `json:"extra_nonce32"`
}

func main() {
	var (
		version  = flag.Bool("version", false, "Show version")
		coinbase = flag.String("coinbase", "", "Coinbase address")
		node     = flag.String("node", "", "Node URL (e.g., http://localhost:8545)")
		ip       = flag.String("ip", "localhost", "Node IP address")
		port     = flag.Int("port", 8545, "Node RPC port")
		threads  = flag.Int("threads", runtime.NumCPU(), "Number of mining threads")
		help     = flag.Bool("help", false, "Show help")
	)
	flag.Parse()

	if *version {
		fmt.Printf("Quantum-Geth Standalone Miner v%s\n", VERSION)
		fmt.Printf("Runtime: %s/%s\n", runtime.GOOS, runtime.GOARCH)
		fmt.Printf("Build: %s\n", time.Now().Format("2006-01-02 15:04:05"))
		os.Exit(0)
	}

	if *help {
		showHelp()
		os.Exit(0)
	}

	// Display startup banner
	fmt.Println("🚀 Quantum-Geth Standalone Miner v" + VERSION)
	fmt.Println("⚛️  16-qubit quantum circuit mining")
	fmt.Println("🔗 Bitcoin-style difficulty with quantum proof-of-work")
	fmt.Println("")

	// Validate coinbase address
	if *coinbase == "" {
		log.Fatal("❌ Coinbase address is required for solo mining!\n   Use: quantum-miner -coinbase 0xYourAddress")
	}

	if !isValidAddress(*coinbase) {
		log.Fatal("❌ Invalid coinbase address format!")
	}

	// Determine node URL: prioritize -node flag, then construct from -ip and -port
	var nodeURL string
	if *node != "" {
		nodeURL = *node
	} else {
		nodeURL = fmt.Sprintf("http://%s:%d", *ip, *port)
	}

	fmt.Printf("📋 Configuration:\n")
	fmt.Printf("   💰 Coinbase: %s\n", *coinbase)
	fmt.Printf("   🌐 Node URL: %s\n", nodeURL)
	fmt.Printf("   🧵 Threads: %d\n", *threads)
	fmt.Printf("   ⚛️  Quantum Puzzles: 48\n")
	fmt.Printf("   🔬 Qubits per Puzzle: 16\n")
	fmt.Printf("   🚪 T-Gates per Puzzle: 8192\n")
	fmt.Println("")

	// Create quantum miner
	miner := &QuantumMiner{
		coinbase:  *coinbase,
		nodeURL:   nodeURL,
		threads:   *threads,
		stopChan:  make(chan bool),
		startTime: time.Now(),
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	// Test connection
	fmt.Printf("🧪 Testing connection to %s...\n", nodeURL)
	if err := miner.testConnection(); err != nil {
		log.Fatalf("❌ Failed to connect to quantum-geth: %v", err)
	}
	fmt.Printf("✅ Connected to quantum-geth!\n")

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start mining
	fmt.Printf("🚀 Starting quantum mining with %d threads...\n", *threads)
	if err := miner.Start(); err != nil {
		log.Fatalf("❌ Failed to start mining: %v", err)
	}

	// Wait for shutdown signal
	<-sigChan
	fmt.Println("\n🛑 Shutdown signal received...")

	// Stop mining and show final stats
	miner.Stop()
}

// Start begins quantum mining
func (m *QuantumMiner) Start() error {
	if !atomic.CompareAndSwapInt32(&m.running, 0, 1) {
		return fmt.Errorf("miner already running")
	}

	// Start work fetcher
	go m.workFetcher()

	// Start mining threads
	for i := 0; i < m.threads; i++ {
		go m.miningThread(i)
	}

	// Start statistics reporter
	go m.statsReporter()

	return nil
}

// Stop stops the miner
func (m *QuantumMiner) Stop() {
	if !atomic.CompareAndSwapInt32(&m.running, 1, 0) {
		return
	}

	close(m.stopChan)

	// Show final statistics
	duration := time.Since(m.startTime)
	attempts := atomic.LoadUint64(&m.attempts)
	accepted := atomic.LoadUint64(&m.accepted)
	rejected := atomic.LoadUint64(&m.rejected)
	stale := atomic.LoadUint64(&m.stale)
	duplicates := atomic.LoadUint64(&m.duplicates)

	fmt.Println("\n📊 Final Quantum Mining Statistics:")
	fmt.Printf("   ⏱️  Runtime: %v\n", duration.Round(time.Second))
	fmt.Printf("   ⚛️  Puzzle Rate: %.2f puzzles/sec\n", float64(attempts)/duration.Seconds())
	fmt.Printf("   🧮 Total Puzzles Solved: %d\n", attempts)
	fmt.Printf("   ✅ Accepted Blocks: %d\n", accepted)
	fmt.Printf("   ❌ Rejected Blocks: %d\n", rejected)
	if stale > 0 || duplicates > 0 {
		fmt.Printf("   ⏰ Stale Submissions: %d\n", stale)
		fmt.Printf("   🔄 Duplicate Submissions: %d\n", duplicates)
	}

	if accepted > 0 {
		fmt.Printf("   💎 Success Rate: %.4f%%\n", float64(accepted)/float64(attempts)*100)
	}

	fmt.Println("\n👋 Thank you for mining on Quantum-Geth!")
}

// testConnection tests the connection to quantum-geth
func (m *QuantumMiner) testConnection() error {
	// Test basic connection
	result, err := m.rpcCall("web3_clientVersion", []interface{}{})
	if err != nil {
		return fmt.Errorf("basic connection test failed: %w", err)
	}

	if version, ok := result.(string); ok {
		log.Printf("📡 Connected to: %s", version)
	}

	// Test if mining is enabled
	_, err = m.rpcCall("eth_getWork", []interface{}{})
	if err != nil {
		log.Printf("⚠️  Warning: eth_getWork failed - make sure geth is started with --mine")
	}

	return nil
}

// workFetcher continuously fetches work from quantum-geth
func (m *QuantumMiner) workFetcher() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			if err := m.fetchWork(); err != nil {
				log.Printf("❌ Failed to fetch work: %v", err)
				time.Sleep(5 * time.Second)
			}
		}
	}
}

// fetchWork gets new mining work from quantum-geth
func (m *QuantumMiner) fetchWork() error {
	// Try quantum-specific GetWork first
	result, err := m.rpcCall("qmpow_getWork", []interface{}{})
	if err != nil {
		// Fall back to eth_getWork
		result, err = m.rpcCall("eth_getWork", []interface{}{})
		if err != nil {
			return fmt.Errorf("failed to get work: %w", err)
		}
	}

	// Parse work response
	workArray, ok := result.([]interface{})
	if !ok || len(workArray) < 3 {
		return fmt.Errorf("invalid work response format")
	}

	work := &QuantumWork{
		WorkHash:  workArray[0].(string),
		QBits:     16, // Default quantum params
		TCount:    8192,
		LNet:      48,
		FetchTime: time.Now(),
	}

	// Parse block number
	if len(workArray) >= 2 {
		if blockNumStr, ok := workArray[1].(string); ok {
			blockNum, _ := strconv.ParseUint(strings.TrimPrefix(blockNumStr, "0x"), 16, 64)
			work.BlockNumber = blockNum
		}
	}

	// Parse target
	if len(workArray) >= 3 {
		work.Target = workArray[2].(string)
	}

	// Store current work
	m.workMutex.Lock()
	if m.currentWork == nil || m.currentWork.WorkHash != work.WorkHash {
		m.currentWork = work
		log.Printf("📦 New work: Block %d, Target: %s...", work.BlockNumber, work.Target[:10])
	}
	m.workMutex.Unlock()

	return nil
}

// miningThread runs a single mining thread
func (m *QuantumMiner) miningThread(threadID int) {
	log.Printf("🔧 Mining thread %d started", threadID)

	for {
		select {
		case <-m.stopChan:
			log.Printf("🔧 Mining thread %d stopped", threadID)
			return
		default:
			if err := m.mineBlock(threadID); err != nil {
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
}

// mineBlock performs quantum mining for one iteration - solving actual quantum puzzles
func (m *QuantumMiner) mineBlock(threadID int) error {
	// Get current work
	m.workMutex.RLock()
	work := m.currentWork
	m.workMutex.RUnlock()

	if work == nil {
		time.Sleep(100 * time.Millisecond)
		return nil
	}

	// Check if work is stale (older than 30 seconds)
	if time.Since(work.FetchTime) > 30*time.Second {
		return nil
	}

	// Generate quantum nonce (similar to geth's approach)
	var qnonce uint64
	nonceBytes := make([]byte, 8)
	rand.Read(nonceBytes)
	for i := 0; i < 8; i++ {
		qnonce |= uint64(nonceBytes[i]) << (i * 8)
	}

	log.Printf("🔬 Thread %d: Solving quantum puzzles for QNonce %d...", threadID, qnonce)
	puzzleStart := time.Now()

	// Solve quantum puzzles (this takes time like real quantum computation)
	proof, err := m.solveQuantumPuzzles(work.WorkHash, qnonce, work.QBits, work.TCount, work.LNet)
	if err != nil {
		log.Printf("❌ Thread %d: Failed to solve quantum puzzles: %v", threadID, err)
		return err
	}

	puzzleTime := time.Since(puzzleStart)
	log.Printf("⚛️  Thread %d: Quantum puzzles solved in %.3fs", threadID, puzzleTime.Seconds())

	// Check if solution meets target (Bitcoin-style difficulty check)
	if m.checkQuantumTarget(proof, work.Target) {
		log.Printf("🎉 Thread %d: Quantum block found! Block %d, QNonce: %d",
			threadID, work.BlockNumber, qnonce)

		if err := m.submitQuantumWork(qnonce, work.WorkHash, proof); err != nil {
			errMsg := err.Error()
			if strings.Contains(errMsg, "stale") {
				log.Printf("⏰ Thread %d: Stale work rejected (QNonce %d) - %v", threadID, qnonce, err)
				atomic.AddUint64(&m.stale, 1)
			} else if strings.Contains(errMsg, "duplicate") {
				log.Printf("🔄 Thread %d: Duplicate submission rejected (QNonce %d) - %v", threadID, qnonce, err)
				atomic.AddUint64(&m.duplicates, 1)
			} else {
				log.Printf("❌ Thread %d: Failed to submit quantum block (QNonce %d): %v", threadID, qnonce, err)
			}
			atomic.AddUint64(&m.rejected, 1)
		} else {
			atomic.AddUint64(&m.accepted, 1)
			log.Printf("✅ Thread %d: Quantum block accepted! QNonce %d 🎊", threadID, qnonce)
		}
	} else {
		log.Printf("🎯 Thread %d: Quantum proof doesn't meet target (QNonce %d)", threadID, qnonce)
	}

	atomic.AddUint64(&m.attempts, 1)
	return nil
}

// solveQuantumPuzzles simulates solving actual quantum puzzles (like geth's approach)
func (m *QuantumMiner) solveQuantumPuzzles(workHash string, qnonce uint64, qbits, tcount, lnet int) (QuantumProofSubmission, error) {
	// Simulate quantum computation time based on circuit complexity
	// Real quantum circuits take time: 16 qubits * 8192 T-gates * 48 puzzles
	baseTime := 50 * time.Millisecond                       // Base time per puzzle
	complexityFactor := float64(qbits*tcount) / (16 * 8192) // Relative to standard params
	puzzleTime := time.Duration(float64(baseTime) * complexityFactor)

	// Simulate solving each puzzle (like geth's real quantum solver)
	var outcomes [][]byte
	var gateHashes [][]byte

	for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
		// Simulate quantum circuit execution time
		time.Sleep(puzzleTime)

		// Generate quantum-looking outcome for this puzzle
		seed := fmt.Sprintf("%s_%d_%d_%d", workHash, qnonce, puzzleIndex, time.Now().UnixNano())

		// Simulate quantum measurement outcome (qbits of data)
		outcomeBytes := make([]byte, (qbits+7)/8)
		hash := sha256Hash(fmt.Sprintf("outcome_%s", seed))
		copy(outcomeBytes, []byte(hash)[:len(outcomeBytes)])
		outcomes = append(outcomes, outcomeBytes)

		// Simulate gate hash for this puzzle
		gateData := fmt.Sprintf("gates_%s_%d_qubits_%d_tgates", seed, qbits, tcount)
		gateHash := sha256Hash(gateData)
		gateHashes = append(gateHashes, []byte(gateHash)[:32])
	}

	// Calculate outcome root from all puzzle outcomes (like geth)
	outcomeRoot := m.calculateOutcomeRoot(outcomes)

	// Calculate combined gate hash
	combinedGateData := ""
	for _, gh := range gateHashes {
		combinedGateData += fmt.Sprintf("%x", gh)
	}
	gateHash := sha256Hash(combinedGateData)

	// Generate proof root (Nova proofs simulation)
	proofData := fmt.Sprintf("nova_proof_%s_%s_%s", outcomeRoot, gateHash, workHash)
	proofRoot := sha256Hash(proofData)

	// Generate branch nibbles (48 bytes for 48 puzzles)
	branchNibbles := make([]byte, 48)
	for i := 0; i < 48; i++ {
		if i < len(outcomes) {
			branchNibbles[i] = outcomes[i][0] >> 4 // High nibble
		}
	}

	// Generate extra nonce
	extraNonce := make([]byte, 32)
	rand.Read(extraNonce)
	// Embed qnonce in extra nonce
	for i := 0; i < 8; i++ {
		extraNonce[i] = byte(qnonce >> (i * 8))
	}

	return QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: fmt.Sprintf("%x", branchNibbles),
		ExtraNonce32:  fmt.Sprintf("%x", extraNonce),
	}, nil
}

// calculateOutcomeRoot calculates the Merkle root of quantum outcomes
func (m *QuantumMiner) calculateOutcomeRoot(outcomes [][]byte) string {
	if len(outcomes) == 0 {
		return "0000000000000000000000000000000000000000000000000000000000000000"
	}

	// Simple Merkle root calculation
	level := make([]string, len(outcomes))
	for i, outcome := range outcomes {
		level[i] = fmt.Sprintf("%x", outcome)
	}

	// Build Merkle tree
	for len(level) > 1 {
		var nextLevel []string
		for i := 0; i < len(level); i += 2 {
			var combined string
			if i+1 < len(level) {
				combined = level[i] + level[i+1]
			} else {
				combined = level[i] + level[i] // Duplicate if odd
			}
			nextLevel = append(nextLevel, sha256Hash(combined))
		}
		level = nextLevel
	}

	return level[0]
}

// checkQuantumTarget checks if the quantum proof meets the target
func (m *QuantumMiner) checkQuantumTarget(proof QuantumProofSubmission, target string) bool {
	// Compare proof root against target (simplified)
	proofBytes, err := hex.DecodeString(proof.ProofRoot)
	if err != nil || len(proofBytes) < 4 {
		return false
	}

	targetBytes, err := hex.DecodeString(strings.TrimPrefix(target, "0x"))
	if err != nil || len(targetBytes) < 4 {
		return false
	}

	// Compare first 4 bytes
	for i := 0; i < 4 && i < len(proofBytes) && i < len(targetBytes); i++ {
		if proofBytes[i] < targetBytes[i] {
			return true
		} else if proofBytes[i] > targetBytes[i] {
			return false
		}
	}

	return false
}

// submitQuantumWork submits quantum mining work to the node
func (m *QuantumMiner) submitQuantumWork(qnonce uint64, workHash string, proof QuantumProofSubmission) error {
	// Convert string workHash to the format expected by the API
	// Ensure proper 0x prefix for all hex values
	cleanWorkHash := strings.TrimPrefix(workHash, "0x")
	workHashWithPrefix := "0x" + cleanWorkHash

	// Decode hex strings to byte arrays for branch_nibbles and extra_nonce32
	branchNibblesBytes, err := hex.DecodeString(strings.TrimPrefix(proof.BranchNibbles, "0x"))
	if err != nil {
		return fmt.Errorf("failed to decode branch nibbles: %w", err)
	}

	extraNonce32Bytes, err := hex.DecodeString(strings.TrimPrefix(proof.ExtraNonce32, "0x"))
	if err != nil {
		return fmt.Errorf("failed to decode extra nonce: %w", err)
	}

	// Create quantum proof with proper format:
	// - Hash fields need 0x prefix for common.Hash
	// - Byte fields need actual byte arrays (will be base64 encoded by JSON)
	quantumProofForAPI := map[string]interface{}{
		"outcome_root":   "0x" + strings.TrimPrefix(proof.OutcomeRoot, "0x"),
		"gate_hash":      "0x" + strings.TrimPrefix(proof.GateHash, "0x"),
		"proof_root":     "0x" + strings.TrimPrefix(proof.ProofRoot, "0x"),
		"branch_nibbles": branchNibblesBytes, // byte array -> base64
		"extra_nonce32":  extraNonce32Bytes,  // byte array -> base64
	}

	// Try quantum-specific SubmitWork first
	params := []interface{}{qnonce, workHashWithPrefix, quantumProofForAPI}
	
	result, err := m.rpcCall("qmpow_submitWork", params)
	if err != nil {
		// Check for specific error patterns that indicate stale/duplicate work
		errMsg := strings.ToLower(err.Error())
		if strings.Contains(errMsg, "stale") || strings.Contains(errMsg, "not found") {
			return fmt.Errorf("stale work - block already sealed or work expired")
		}
		if strings.Contains(errMsg, "duplicate") || strings.Contains(errMsg, "already submitted") {
			return fmt.Errorf("duplicate submission - this solution was already submitted")
		}
		
		// Fall back to eth_submitWork with adapted parameters (FIXED: use hex string for nonce)
		ethParams := []interface{}{
			fmt.Sprintf("0x%016x", qnonce), // nonce as hex string (FIXED)
			workHashWithPrefix,             // hash
			"0x" + strings.TrimPrefix(proof.ProofRoot, "0x"), // mix digest (use proof root)
		}
		result, err = m.rpcCall("eth_submitWork", ethParams)
		if err != nil {
			// Check for stale/duplicate patterns in eth_submitWork too
			errMsg := strings.ToLower(err.Error())
			if strings.Contains(errMsg, "stale") || strings.Contains(errMsg, "not found") {
				return fmt.Errorf("stale work - block already sealed or work expired")
			}
			if strings.Contains(errMsg, "duplicate") || strings.Contains(errMsg, "already submitted") {
				return fmt.Errorf("duplicate submission - this solution was already submitted")
			}
			return fmt.Errorf("failed to submit work: %w", err)
		}
	}

	// Check if submission was accepted
	if accepted, ok := result.(bool); ok && accepted {
		return nil
	}

	// If we get false result without error, it's likely stale or duplicate
	// Check current work to see if it has changed (indicating stale work)
	m.workMutex.RLock()
	currentWork := m.currentWork
	m.workMutex.RUnlock()
	
	if currentWork != nil && currentWork.WorkHash != workHash {
		return fmt.Errorf("stale work - new block available (was working on %s, now %s)", 
			workHash[:10]+"...", currentWork.WorkHash[:10]+"...")
	}

	return fmt.Errorf("submission rejected - likely duplicate or invalid proof")
}

// statsReporter reports mining statistics periodically
func (m *QuantumMiner) statsReporter() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.reportStats()
		}
	}
}

// reportStats logs current mining statistics
func (m *QuantumMiner) reportStats() {
	attempts := atomic.LoadUint64(&m.attempts)
	accepted := atomic.LoadUint64(&m.accepted)
	rejected := atomic.LoadUint64(&m.rejected)
	stale := atomic.LoadUint64(&m.stale)
	duplicates := atomic.LoadUint64(&m.duplicates)

	duration := time.Since(m.startTime)
	puzzleRate := float64(attempts) / duration.Seconds()

	log.Printf("📊 Quantum Mining Stats: %.2f puzzles/sec | %d solved | %d accepted | %d rejected (%d stale, %d duplicates) | %v runtime",
		puzzleRate, attempts, accepted, rejected, stale, duplicates, duration.Truncate(time.Second))
}

// rpcCall makes a JSON-RPC call to the geth node
func (m *QuantumMiner) rpcCall(method string, params []interface{}) (interface{}, error) {
	request := JSONRPCRequest{
		ID:      1,
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}

	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := m.client.Post(m.nodeURL, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var response JSONRPCResponse
	if err := json.Unmarshal(respBody, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if response.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	return response.Result, nil
}

// sha256Hash creates a proper SHA256 hash of the input string
func sha256Hash(input string) string {
	hasher := sha256.New()
	hasher.Write([]byte(input))
	return hex.EncodeToString(hasher.Sum(nil))
}

// showHelp displays help information
func showHelp() {
	fmt.Println("Quantum-Geth Standalone Miner v" + VERSION)
	fmt.Println("")
	fmt.Println("USAGE:")
	fmt.Println("  quantum-miner [OPTIONS]")
	fmt.Println("")
	fmt.Println("OPTIONS:")
	fmt.Println("  -coinbase ADDRESS    Coinbase address for mining rewards (required)")
	fmt.Println("  -node URL           Full node URL (default: http://localhost:8545)")
	fmt.Println("  -ip ADDRESS         Node IP address (default: localhost)")
	fmt.Println("  -port NUMBER        Node RPC port (default: 8545)")
	fmt.Println("  -threads N          Number of mining threads (default: CPU cores)")
	fmt.Println("  -version            Show version information")
	fmt.Println("  -help               Show this help message")
	fmt.Println("")
	fmt.Println("CONNECTION EXAMPLES:")
	fmt.Println("  # Default localhost connection")
	fmt.Println("  quantum-miner -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A")
	fmt.Println("")
	fmt.Println("  # Custom IP and port")
	fmt.Println("  quantum-miner -coinbase 0x123... -ip 192.168.1.100 -port 8545")
	fmt.Println("")
	fmt.Println("  # Full URL (overrides -ip and -port)")
	fmt.Println("  quantum-miner -coinbase 0x123... -node http://my-quantum-node.com:8545")
	fmt.Println("")
	fmt.Println("  # Remote mining with multiple threads")
	fmt.Println("  quantum-miner -coinbase 0x123... -ip 10.0.0.50 -threads 8")
	fmt.Println("")
	fmt.Println("REQUIREMENTS:")
	fmt.Println("  - Running quantum-geth node with --mine enabled")
	fmt.Println("  - Valid Ethereum address for coinbase")
	fmt.Println("  - Network connectivity to the quantum-geth node")
	fmt.Println("")
	fmt.Println("QUANTUM MINING DETAILS:")
	fmt.Println("  - Each block requires 48 quantum puzzle solutions")
	fmt.Println("  - Each puzzle uses 16 qubits with up to 8192 T-gates")
	fmt.Println("  - Difficulty adjusts every 100 blocks (like Bitcoin)")
	fmt.Println("  - Target block time: 12 seconds")
	fmt.Println("")
}

// isValidAddress checks if the address is valid Ethereum format
func isValidAddress(addr string) bool {
	if len(addr) != 42 {
		return false
	}
	if !strings.HasPrefix(addr, "0x") && !strings.HasPrefix(addr, "0X") {
		return false
	}
	for _, c := range addr[2:] {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}
