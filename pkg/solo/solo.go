package solo

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"quantum-miner/pkg/config"
	"quantum-miner/pkg/miner"
	"quantum-miner/pkg/quantum"
)

// SoloMiner implements solo mining against a quantum-geth node
type SoloMiner struct {
	config   *config.Config
	quantum  quantum.Backend
	stats    *miner.Stats
	running  int32
	stopChan chan bool

	// Mining state
	currentWork  *QuantumWork
	workMutex    sync.RWMutex
	minerThreads []chan bool
	hashCounter  uint64
	startTime    time.Time

	// Node connection
	nodeURL string
	client  *http.Client
}

// JSONRPCRequest represents a JSON-RPC request
type JSONRPCRequest struct {
	ID      int           `json:"id"`
	JSONRPC string        `json:"jsonrpc"`
	Method  string        `json:"method"`
	Params  []interface{} `json:"params"`
}

// JSONRPCResponse represents a JSON-RPC response
type JSONRPCResponse struct {
	ID      int         `json:"id"`
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result"`
	Error   *RPCError   `json:"error"`
}

// RPCError represents a JSON-RPC error
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// QuantumWork represents quantum mining work from geth
type QuantumWork struct {
	WorkHash        string    `json:"work_hash"`
	BlockNumber     uint64    `json:"block_number"`
	Target          string    `json:"target"`
	QuantumParams   string    `json:"quantum_params"`
	CoinbaseAddress string    `json:"coinbase_address"`
	QBits           int       `json:"qbits"`
	TCount          int       `json:"tcount"`
	LNet            int       `json:"lnet"`
	StartTime       time.Time `json:"start_time"`
	Deadline        time.Time `json:"deadline"`
}

// QuantumProofSubmission represents a quantum proof submission
type QuantumProofSubmission struct {
	OutcomeRoot   string `json:"outcome_root"`
	GateHash      string `json:"gate_hash"`
	ProofRoot     string `json:"proof_root"`
	BranchNibbles string `json:"branch_nibbles"`
	ExtraNonce32  string `json:"extra_nonce32"`
}

// NewMiner creates a new solo miner
func NewMiner(config *config.Config, quantum quantum.Backend) (*SoloMiner, error) {
	if config.Solo.NodeURL == "" {
		return nil, fmt.Errorf("node URL is required for solo mining")
	}

	if config.Solo.Coinbase == "" {
		return nil, fmt.Errorf("coinbase address is required for solo mining")
	}

	miner := &SoloMiner{
		config:   config,
		quantum:  quantum,
		nodeURL:  config.Solo.NodeURL,
		stopChan: make(chan bool),
		client: &http.Client{
			Timeout: time.Duration(config.Solo.Timeout) * time.Second,
		},
		stats: &miner.Stats{
			StartTime: time.Now(),
		},
	}

	return miner, nil
}

// Start begins quantum mining
func (s *SoloMiner) Start() error {
	if !atomic.CompareAndSwapInt32(&s.running, 0, 1) {
		return fmt.Errorf("miner already running")
	}

	s.startTime = time.Now()
	s.stats.StartTime = s.startTime

	log.Printf("🚀 Starting Quantum-Geth Solo Miner")
	log.Printf("📡 Node URL: %s", s.nodeURL)
	log.Printf("💰 Coinbase: %s", s.config.Solo.Coinbase)
	log.Printf("🧵 Threads: %d", s.config.Mining.Threads)

	// Test connection to node
	if err := s.testConnection(); err != nil {
		atomic.StoreInt32(&s.running, 0)
		return fmt.Errorf("failed to connect to node: %w", err)
	}

	// Start mining threads
	s.minerThreads = make([]chan bool, s.config.Mining.Threads)
	for i := 0; i < s.config.Mining.Threads; i++ {
		s.minerThreads[i] = make(chan bool)
		go s.miningThread(i)
	}

	// Start work fetcher
	go s.workFetcher()

	// Start statistics updater
	go s.statsUpdater()

	log.Printf("✅ Quantum miner started with %d threads", s.config.Mining.Threads)
	return nil
}

// Stop stops the miner
func (s *SoloMiner) Stop() error {
	if !atomic.CompareAndSwapInt32(&s.running, 1, 0) {
		return fmt.Errorf("miner not running")
	}

	log.Printf("🛑 Stopping quantum miner...")

	// Stop all threads
	for _, stopCh := range s.minerThreads {
		close(stopCh)
	}

	close(s.stopChan)
	log.Printf("✅ Quantum miner stopped")
	return nil
}

// IsRunning returns true if the miner is running
func (s *SoloMiner) IsRunning() bool {
	return atomic.LoadInt32(&s.running) == 1
}

// GetStats returns current mining statistics
func (s *SoloMiner) GetStats() *miner.Stats {
	s.stats.Duration = time.Since(s.startTime)
	s.stats.HashRate = float64(atomic.LoadUint64(&s.hashCounter)) / s.stats.Duration.Seconds()
	return s.stats
}

// testConnection tests the connection to the quantum-geth node
func (s *SoloMiner) testConnection() error {
	log.Printf("🔍 Testing connection to quantum-geth node...")

	// Test basic connection
	result, err := s.rpcCall("web3_clientVersion", []interface{}{})
	if err != nil {
		return fmt.Errorf("basic connection test failed: %w", err)
	}

	if version, ok := result.(string); ok {
		log.Printf("📡 Connected to: %s", version)
	}

	// Test quantum mining support
	_, err = s.rpcCall("qmpow_getQuantumParams", []interface{}{"latest"})
	if err != nil {
		log.Printf("⚠️  Quantum RPC methods not available, trying eth namespace...")
	} else {
		log.Printf("✅ Quantum-geth RPC detected")
	}

	return nil
}

// workFetcher continuously fetches work from the node
func (s *SoloMiner) workFetcher() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.stopChan:
			return
		case <-ticker.C:
			if err := s.fetchWork(); err != nil {
				log.Printf("❌ Failed to fetch work: %v", err)
				time.Sleep(5 * time.Second)
			}
		}
	}
}

// fetchWork gets new mining work from quantum-geth
func (s *SoloMiner) fetchWork() error {
	// Try quantum-specific GetWork first
	result, err := s.rpcCall("qmpow_getWork", []interface{}{})
	if err != nil {
		// Fall back to eth_getWork if quantum RPC not available
		result, err = s.rpcCall("eth_getWork", []interface{}{})
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
		WorkHash:      workArray[0].(string),
		StartTime:     time.Now(),
		Deadline:      time.Now().Add(60 * time.Second),
		QBits:         16,  // Default quantum params
		TCount:        8192,
		LNet:          48,
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

	// Parse quantum parameters if available (5-element array)
	if len(workArray) >= 5 {
		if paramsStr, ok := workArray[3].(string); ok {
			work.QuantumParams = paramsStr
			s.parseQuantumParams(work, paramsStr)
		}
		if coinbaseStr, ok := workArray[4].(string); ok {
			work.CoinbaseAddress = coinbaseStr
		}
	}

	s.workMutex.Lock()
	s.currentWork = work
	s.stats.BlockHeight = work.BlockNumber
	s.workMutex.Unlock()

	log.Printf("📦 New work: Block %d, Target: %s", work.BlockNumber, work.Target[:10]+"...")
	return nil
}

// parseQuantumParams parses quantum parameters from work
func (s *SoloMiner) parseQuantumParams(work *QuantumWork, params string) {
	// Parse "qbits:16,tcount:8192,lnet:48" format
	parts := strings.Split(params, ",")
	for _, part := range parts {
		kv := strings.Split(part, ":")
		if len(kv) != 2 {
			continue
		}
		
		switch kv[0] {
		case "qbits":
			if val, err := strconv.Atoi(kv[1]); err == nil {
				work.QBits = val
			}
		case "tcount":
			if val, err := strconv.Atoi(kv[1]); err == nil {
				work.TCount = val
			}
		case "lnet":
			if val, err := strconv.Atoi(kv[1]); err == nil {
				work.LNet = val
			}
		}
	}
}

// miningThread runs the mining loop for a single thread
func (s *SoloMiner) miningThread(threadID int) {
	log.Printf("🔧 Mining thread %d started", threadID)
	
	for {
		select {
		case <-s.minerThreads[threadID]:
			log.Printf("🔧 Mining thread %d stopped", threadID)
			return
		default:
			if err := s.mineBlock(threadID); err != nil {
				log.Printf("❌ Thread %d mining error: %v", threadID, err)
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
}

// mineBlock performs quantum mining for one iteration
func (s *SoloMiner) mineBlock(threadID int) error {
	// Get current work
	s.workMutex.RLock()
	work := s.currentWork
	s.workMutex.RUnlock()

	if work == nil {
		time.Sleep(100 * time.Millisecond)
		return nil
	}

	// Check if work is still valid
	if time.Now().After(work.Deadline) {
		return nil
	}

	// Generate quantum nonce
	var qnonce uint64
	rand.Read((*[8]byte)(unsafe.Pointer(&qnonce))[:])

	// Create quantum seed from work hash and nonce
	seed := s.createQuantumSeed(work.WorkHash, qnonce)

	// Solve quantum puzzles
	quantumResult, err := s.quantum.SolveQuantumPuzzles(seed, work.LNet)
	if err != nil {
		atomic.AddUint64(&s.stats.QuantumErrors, 1)
		return err
	}

	// Create quantum proof submission
	proof := QuantumProofSubmission{
		OutcomeRoot:   hex.EncodeToString(quantumResult.OutcomeRoot),
		GateHash:      hex.EncodeToString(quantumResult.GateHash),
		ProofRoot:     hex.EncodeToString(quantumResult.ProofRoot),
		BranchNibbles: hex.EncodeToString(quantumResult.Measurements),
		ExtraNonce32:  s.generateExtraNonce32(qnonce),
	}

	// Check if solution meets target (simplified check)
	if s.checkQuantumTarget(proof, work.Target) {
		log.Printf("🎉 Quantum block found by thread %d! Block %d, QNonce: %d", 
			threadID, work.BlockNumber, qnonce)
		
		if err := s.submitQuantumWork(qnonce, work.WorkHash, proof); err != nil {
			log.Printf("❌ Failed to submit quantum block: %v", err)
			atomic.AddUint64(&s.stats.Rejected, 1)
		} else {
			atomic.AddUint64(&s.stats.Accepted, 1)
			s.stats.LastBlock = time.Now()
			log.Printf("✅ Quantum block accepted! 🎊")
		}
	}

	atomic.AddUint64(&s.hashCounter, 1)
	return nil
}

// createQuantumSeed creates a quantum seed from work hash and nonce
func (s *SoloMiner) createQuantumSeed(workHash string, qnonce uint64) []byte {
	// Combine work hash with quantum nonce for seed
	hashBytes, _ := hex.DecodeString(strings.TrimPrefix(workHash, "0x"))
	nonceBytes := make([]byte, 8)
	for i := 0; i < 8; i++ {
		nonceBytes[i] = byte(qnonce >> (8 * (7 - i)))
	}
	
	seed := append(hashBytes, nonceBytes...)
	if len(seed) > 32 {
		seed = seed[:32]
	}
	return seed
}

// generateExtraNonce32 generates extra nonce for quantum proof
func (s *SoloMiner) generateExtraNonce32(qnonce uint64) string {
	extraNonce := make([]byte, 32)
	rand.Read(extraNonce)
	
	// Include qnonce in extra nonce for uniqueness
	nonceBytes := make([]byte, 8)
	for i := 0; i < 8; i++ {
		nonceBytes[i] = byte(qnonce >> (8 * (7 - i)))
	}
	copy(extraNonce[:8], nonceBytes)
	
	return hex.EncodeToString(extraNonce)
}

// checkQuantumTarget checks if the quantum proof meets the target
func (s *SoloMiner) checkQuantumTarget(proof QuantumProofSubmission, target string) bool {
	// Simple target check - compare proof root against target
	// In production, this would use the full quantum proof verification
	proofBytes, err := hex.DecodeString(proof.ProofRoot)
	if err != nil || len(proofBytes) < 4 {
		return false
	}
	
	targetBytes, err := hex.DecodeString(strings.TrimPrefix(target, "0x"))
	if err != nil || len(targetBytes) < 4 {
		return false
	}
	
	// Compare first 4 bytes (simplified)
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
func (s *SoloMiner) submitQuantumWork(qnonce uint64, workHash string, proof QuantumProofSubmission) error {
	// Try quantum-specific SubmitWork first
	params := []interface{}{qnonce, workHash, proof}
	
	result, err := s.rpcCall("qmpow_submitWork", params)
	if err != nil {
		// Fall back to eth_submitWork with adapted parameters
		ethParams := []interface{}{
			fmt.Sprintf("0x%016x", qnonce), // nonce
			workHash,                       // hash
			proof.ProofRoot,               // mix digest (use proof root)
		}
		result, err = s.rpcCall("eth_submitWork", ethParams)
		if err != nil {
			return fmt.Errorf("failed to submit work: %w", err)
		}
	}

	// Check if submission was accepted
	if accepted, ok := result.(bool); ok && accepted {
		return nil
	}
	
	return fmt.Errorf("work submission rejected")
}

// statsUpdater updates mining statistics
func (s *SoloMiner) statsUpdater() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.stopChan:
			return
		case <-ticker.C:
			s.updateQuantumStats()
			s.logStats()
		}
	}
}

// updateQuantumStats updates quantum-specific statistics
func (s *SoloMiner) updateQuantumStats() {
	if backendStats := s.quantum.GetStats(); backendStats != nil {
		s.stats.QuantumRate = float64(backendStats.TotalComputations) / time.Since(s.startTime).Seconds()
		s.stats.CircuitDepth = 16  // Fixed for quantum-geth
		s.stats.GateCount = 8192   // Fixed for quantum-geth
	}
}

// logStats logs current mining statistics
func (s *SoloMiner) logStats() {
	stats := s.GetStats()
	
	log.Printf("📊 Mining Stats:")
	log.Printf("   ⛏️  Hash Rate: %.2f attempts/sec", stats.HashRate)
	log.Printf("   🧮 Total Attempts: %d", atomic.LoadUint64(&s.hashCounter))
	log.Printf("   ✅ Accepted: %d", stats.Accepted)
	log.Printf("   ❌ Rejected: %d", stats.Rejected)
	log.Printf("   ⚛️  Quantum Rate: %.2f puzzles/sec", stats.QuantumRate)
	log.Printf("   📈 Uptime: %v", stats.Duration.Round(time.Second))
	
	if stats.Accepted > 0 {
		log.Printf("   🏆 Last Block: %v ago", time.Since(stats.LastBlock).Round(time.Second))
	}
}

// rpcCall makes a JSON-RPC call to the geth node
func (s *SoloMiner) rpcCall(method string, params []interface{}) (interface{}, error) {
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

	resp, err := s.client.Post(s.nodeURL, "application/json", bytes.NewReader(reqBody))
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

// Helper functions
func parseHexUint64(hex string) uint64 {
	if len(hex) < 2 || hex[:2] != "0x" {
		return 0
	}
	val, _ := strconv.ParseUint(hex[2:], 16, 64)
	return val
}

func getStringField(data map[string]interface{}, field string) string {
	if val, ok := data[field].(string); ok {
		return val
	}
	return ""
}
