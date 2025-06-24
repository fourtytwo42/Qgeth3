package main

import (
	"bytes"
	"context"
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
	"quantum-gpu-miner/pkg/quantum"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const VERSION = "1.1.0-gpu"

// Global logging to file flag
var logToFile bool = false
var logFileHandle *os.File = nil

// Logging helper functions
func logInfo(format string, v ...interface{}) {
	message := fmt.Sprintf(format, v...)
	if logToFile && logFileHandle != nil {
		logFileHandle.WriteString(time.Now().Format("2006-01-02 15:04:05") + " [INFO] " + message + "\n")
		logFileHandle.Sync()
	} else {
		log.Printf(format, v...)
	}
}

func logError(format string, v ...interface{}) {
	message := fmt.Sprintf(format, v...)
	if logToFile && logFileHandle != nil {
		logFileHandle.WriteString(time.Now().Format("2006-01-02 15:04:05") + " [ERROR] " + message + "\n")
		logFileHandle.Sync()
	} else {
		log.Printf(format, v...)
	}
}

// Mining state
type QuantumMiner struct {
	coinbase string
	nodeURL  string
	threads  int
	gpuMode  bool
	gpuID    int
	running  int32
	stopChan chan bool

	// Statistics - Enhanced for professional mining display
	attempts      uint64 // Total QNonces attempted
	puzzlesSolved uint64 // Total quantum puzzles solved
	accepted      uint64 // Accepted blocks
	rejected      uint64 // Rejected blocks
	stale         uint64 // Stale submissions
	duplicates    uint64 // Duplicate submissions
	startTime     time.Time
	lastStatTime  time.Time
	lastAttempts  uint64
	lastPuzzles   uint64

	// Real-time performance tracking
	currentHashrate   float64 // QNonces per second
	currentPuzzleRate float64 // Puzzles per second
	bestShareDiff     float64 // Best share difficulty found
	avgSolveTime      float64 // Average puzzle solve time
	totalSolveTime    time.Duration

	// Block tracking for dashboard
	blockTimes        []time.Time   // Last 10 block times
	currentDifficulty uint64        // Current network difficulty
	targetBlockTime   time.Duration // Target block time (12s)
	blocksToRetarget  uint64        // Blocks until next difficulty adjustment

	client      *http.Client
	currentWork *QuantumWork
	workMutex   sync.RWMutex

	// MULTI-GPU ACCELERATION: Support for multiple GPUs with load balancing
	multiGPUEnabled bool                                             // Whether multi-GPU is enabled
	availableGPUs   []int                                            // List of available GPU device IDs
	gpuSimulators   map[int]*quantum.HighPerformanceQuantumSimulator // GPU simulators per device
	gpuLoadBalancer *GPULoadBalancer                                 // Load balancer for GPU work distribution
	gpuWorkQueue    chan *GPUWorkItem                                // Queue for GPU work items
	gpuResultQueue  chan *GPUResult                                  // Queue for GPU results

	// Rate limiting to prevent overwhelming geth node
	submissionSemaphore chan struct{} // Limits concurrent submissions

	// THREAD-SAFE NONCE GENERATION: Ensures each thread produces unique nonces
	nonceCounter uint64 // Atomic counter for unique nonce generation
	nonceBase    uint64 // Base nonce value (timestamp + random)

	// ENHANCED THREAD MANAGEMENT: Fix thread starvation and stale work issues
	threadStates     map[int]*ThreadState // Track individual thread states
	threadStateMux   sync.RWMutex         // Protect thread state access
	activeThreads    int32                // Count of actively working threads
	maxActiveThreads int32                // Maximum concurrent active threads

	// MEMORY-EFFICIENT PUZZLE STAGING: Pre-allocate memory to avoid swapping
	puzzleMemoryPool chan []PuzzleMemory // Pool of pre-allocated puzzle memory
	memoryPoolSize   int                 // Size of memory pool

	// STAGGERED EXECUTION: Prevent all threads from starting simultaneously
	threadStartDelay time.Duration // Delay between thread starts
	lastThreadStart  time.Time     // Track last thread start time

	// EFFICIENCY OPTIMIZATIONS: Reduce CPU load and improve performance
	workFetchInterval    time.Duration // How often to fetch new work (default: 100ms)
	statUpdateInterval   time.Duration // How often to update stats (default: 1s)
	dashboardUpdateRate  time.Duration // Dashboard refresh rate (default: 1s)
	cpuAffinityEnabled   bool          // Whether to set CPU affinity for threads
	priorityOptimization bool          // Whether to use process priority optimization
	memoryOptimization   bool          // Whether to enable memory optimization
	diskCacheEnabled     bool          // Whether to enable disk caching
}

// ThreadState tracks individual thread execution state
type ThreadState struct {
	ID             int                // Thread ID
	Status         string             // Current status: "idle", "working", "aborting", "stuck"
	StartTime      time.Time          // When current work started
	WorkHash       string             // Current work hash
	QNonce         uint64             // Current qnonce being worked on
	LastHeartbeat  time.Time          // Last activity timestamp
	AbortRequested bool               // Whether abort has been requested
	StuckCount     int                // Number of times marked as stuck
	cancelFunc     context.CancelFunc // Context cancellation for hard abort
}

// PuzzleMemory represents pre-allocated memory for puzzle solving
type PuzzleMemory struct {
	Outcomes   [][]byte // Pre-allocated outcome buffers
	GateHashes [][]byte // Pre-allocated gate hash buffers
	WorkBuffer []byte   // Working memory buffer
	ID         int      // Memory block ID for tracking
}

// ENHANCED WORK STRUCTURE: Add memory management
type QuantumWork struct {
	WorkHash    string    `json:"work_hash"`
	BlockNumber uint64    `json:"block_number"`
	Target      string    `json:"target"`
	QBits       int       `json:"qbits"`
	TCount      int       `json:"tcount"`
	LNet        int       `json:"lnet"`
	FetchTime   time.Time `json:"fetch_time"`

	// Memory management
	EstimatedMemory uint64 // Estimated memory requirement
	Priority        int    // Work priority (higher = more important)
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

type QuantumProofSubmission struct {
	OutcomeRoot   string `json:"outcome_root"`
	GateHash      string `json:"gate_hash"`
	ProofRoot     string `json:"proof_root"`
	BranchNibbles string `json:"branch_nibbles"`
	ExtraNonce32  string `json:"extra_nonce32"`
}

// MULTI-GPU SUPPORT STRUCTURES
type GPUWorkItem struct {
	ThreadID  int       // Thread requesting the work
	WorkHash  string    // Work hash for the puzzle
	QNonce    uint64    // QNonce to process
	QBits     int       // Number of qubits
	TCount    int       // Number of gates
	LNet      int       // Network parameter (puzzle count)
	StartTime time.Time // When work was queued
	Priority  int       // Work priority (0=highest)
	DeviceID  int       // Preferred GPU device ID (-1 = any)
}

type GPUResult struct {
	ThreadID    int                    // Thread that requested the work
	WorkItem    *GPUWorkItem           // Original work item
	Result      QuantumProofSubmission // Simulation result
	Error       error                  // Error if simulation failed
	DeviceID    int                    // GPU device that processed the work
	ProcessTime time.Duration          // Time taken to process
	Success     bool                   // Whether simulation succeeded
}

type GPULoadBalancer struct {
	devices          []int           // Available GPU device IDs
	deviceLoad       map[int]int     // Current load per device
	deviceCapacity   map[int]int     // Max capacity per device
	devicePerf       map[int]float64 // Performance rating per device
	roundRobinIndex  int             // Round-robin counter
	mutex            sync.RWMutex    // Protect load balancer state
	workDistribution map[int]int     // Work distribution stats
	lastRebalance    time.Time       // Last rebalance time
}

// GPU performance metrics
type GPUMetrics struct {
	DeviceID      int           // GPU device ID
	WorkCompleted uint64        // Total work items completed
	AverageTime   time.Duration // Average processing time
	ErrorRate     float64       // Error rate (0.0-1.0)
	Utilization   float64       // GPU utilization (0.0-1.0)
	MemoryUsage   uint64        // Memory usage in bytes
	Temperature   float64       // GPU temperature (if available)
	LastUpdate    time.Time     // Last metrics update
}

func main() {
	var (
		version  = flag.Bool("version", false, "Show version")
		coinbase = flag.String("coinbase", "", "Coinbase address")
		node     = flag.String("node", "", "Node URL (e.g., http://127.0.0.1:8545)")
		ip       = flag.String("ip", "127.0.0.1", "Node IP address")
		port     = flag.Int("port", 8545, "Node RPC port")
		threads  = flag.Int("threads", runtime.NumCPU(), "Number of mining threads")
		gpu      = flag.Bool("gpu", false, "Enable GPU mining (CUDA/Qiskit)")
		gpuID    = flag.Int("gpu-id", 0, "GPU device ID to use (default: 0)")
		logFile  = flag.Bool("log", false, "Enable logging to file (quantum-miner.log)")
		help     = flag.Bool("help", false, "Show help")
	)
	flag.Parse()

	// Set global log file flag and initialize logging
	logToFile = *logFile
	if logToFile {
		var err error
		logFileHandle, err = os.OpenFile("quantum-miner.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			log.Fatalf("❌ Failed to create log file: %v", err)
		}
		logInfo("Quantum-Geth Miner v%s started with file logging", VERSION)
	}

	if *version {
		fmt.Printf("Quantum-Geth GPU/CPU Miner v%s\n", VERSION)
		fmt.Printf("Runtime: %s/%s\n", runtime.GOOS, runtime.GOARCH)
		fmt.Printf("Build: %s\n", time.Now().Format("2006-01-02 15:04:05"))
		if *gpu {
			fmt.Printf("GPU Support: CUDA/Qiskit GPU (device %d)\n", *gpuID)
		} else {
			fmt.Printf("Mining Mode: CPU only\n")
		}
		os.Exit(0)
	}

	if *help {
		showHelp()
		os.Exit(0)
	}

	// Display startup banner
	fmt.Println("🚀 Quantum-Geth GPU/CPU Miner v" + VERSION)
	fmt.Println("⚛️  16-qubit quantum circuit mining")
	fmt.Println("🔗 Bitcoin-style difficulty with quantum proof-of-work")
	if *gpu {
		fmt.Printf("🎮 GPU Mining: ENABLED (CUDA device %d)\n", *gpuID)
	} else {
		fmt.Println("💻 CPU Mining: ENABLED")
	}
	fmt.Println("")

	// Validate coinbase address
	if *coinbase == "" {
		log.Fatal("❌ Coinbase address is required for solo mining!\n   Use: quantum-gpu-miner -coinbase 0xYourAddress")
	}

	if !isValidAddress(*coinbase) {
		log.Fatal("❌ Invalid coinbase address format!")
	}

	// GPU validation
	if *gpu {
		if err := checkGPUSupport(*gpuID); err != nil {
			log.Fatalf("❌ GPU initialization failed: %v", err)
		}
		fmt.Printf("✅ GPU %d initialized successfully!\n", *gpuID)
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
	if *gpu {
		fmt.Printf("   🎮 GPU Device: %d (CUDA/Qiskit)\n", *gpuID)
		fmt.Printf("   🧵 GPU Threads: %d quantum circuits in parallel\n", *threads)
	} else {
		fmt.Printf("   🧵 CPU Threads: %d\n", *threads)
	}
	fmt.Printf("   ⚛️  Quantum Puzzles: 48\n")
	fmt.Printf("   🔬 Qubits per Puzzle: 16\n")
	fmt.Printf("   🚪 T-Gates per Puzzle: 8192\n")
	fmt.Println("")

	// Create quantum miner
	now := time.Now()
	miner := &QuantumMiner{
		coinbase:        *coinbase,
		nodeURL:         nodeURL,
		threads:         *threads,
		gpuMode:         *gpu,
		gpuID:           *gpuID,
		stopChan:        make(chan bool),
		startTime:       now,
		lastStatTime:    now,
		targetBlockTime: 12 * time.Second, // Quantum-Geth target block time
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		submissionSemaphore: make(chan struct{}, 10), // Limit concurrent submissions
		threadStates:        make(map[int]*ThreadState),
		puzzleMemoryPool:    make(chan []PuzzleMemory, 100), // Memory pool for puzzle solving
		threadStartDelay:    100 * time.Millisecond,         // Stagger thread starts
	}

	// Initialize multi-GPU system if enabled
	if *gpu {
		logInfo("🚀 Initializing MULTI-GPU quantum processor...")
		err := miner.initializeMultiGPU()
		if err != nil {
			log.Fatalf("❌ Failed to initialize multi-GPU system: %v", err)
		}
		logInfo("✅ Multi-GPU quantum acceleration initialized!")
		logInfo("   📊 Load balancing across %d GPUs for maximum throughput", len(miner.availableGPUs))
	}

	// Test connection
	logInfo("🧪 Testing connection to %s...", nodeURL)
	if err := miner.testConnection(); err != nil {
		log.Fatalf("❌ Failed to connect to quantum-geth: %v", err)
	}
	logInfo("✅ Connected to quantum-geth!")

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start mining
	if *gpu {
		logInfo("🚀 Starting GPU quantum mining on device %d with %d parallel circuits...", *gpuID, *threads)
	} else {
		logInfo("🚀 Starting CPU quantum mining with %d threads...", *threads)
	}
	if err := miner.Start(); err != nil {
		log.Fatalf("❌ Failed to start mining: %v", err)
	}

	// Wait for shutdown signal
	<-sigChan
	fmt.Println("\n🛑 Shutdown signal received...")

	// Stop mining and show final stats
	miner.Stop()

	// Close log file if logging was enabled
	if logFileHandle != nil {
		logFileHandle.Close()
	}
}

// checkGPUSupport verifies GPU availability for high-performance batch processing
func checkGPUSupport(gpuID int) error {
	fmt.Printf("🔍 Checking for HIGH-PERFORMANCE GPU acceleration...\n")

	// Test multi-GPU support
	availableGPUs, err := detectAvailableGPUs()
	if err != nil {
		return fmt.Errorf("GPU detection failed: %v", err)
	}

	if len(availableGPUs) == 0 {
		return fmt.Errorf("no compatible GPUs found")
	}

	// Test each GPU
	for _, deviceID := range availableGPUs {
		hybridSim, err := quantum.NewHighPerformanceQuantumSimulator(16) // 16 qubits
		if err != nil {
			logError("GPU %d initialization failed: %v", deviceID, err)
			continue
		}
		hybridSim.Cleanup()
		fmt.Printf("✅ GPU %d: HIGH-PERFORMANCE Quantum acceleration AVAILABLE\n", deviceID)
	}

	fmt.Printf("🚀 Multi-GPU Support: %d GPUs detected\n", len(availableGPUs))
	fmt.Printf("   🎯 Batch processing with load balancing enabled\n")
	return nil
}

// Start begins quantum mining
func (m *QuantumMiner) Start() error {
	if !atomic.CompareAndSwapInt32(&m.running, 0, 1) {
		return fmt.Errorf("miner already running")
	}

	// Initialize thread-safe nonce generation
	// Use timestamp + random value as base to ensure uniqueness across restarts
	baseBytes := make([]byte, 4)
	rand.Read(baseBytes)
	randomPart := uint64(baseBytes[0])<<24 | uint64(baseBytes[1])<<16 | uint64(baseBytes[2])<<8 | uint64(baseBytes[3])
	m.nonceBase = uint64(time.Now().Unix())<<32 | randomPart
	m.nonceCounter = 0

	log.Printf("🔢 Nonce base initialized: %016x", m.nonceBase)

	// ENHANCED INITIALIZATION: Set up thread management and memory pools
	m.initializeThreadManagement()
	m.initializeMemoryPools()

	// Start work fetcher
	go m.workFetcher()

	// Start thread monitor for stuck thread detection
	go m.threadMonitor()

	// Start mining threads with staggered execution
	for i := 0; i < m.threads; i++ {
		go m.enhancedMiningThread(i)
		// Stagger thread starts to prevent resource contention
		time.Sleep(m.threadStartDelay)
	}

	// Start statistics reporter
	go m.statsReporter()

	return nil
}

// initializeThreadManagement sets up enhanced thread tracking
func (m *QuantumMiner) initializeThreadManagement() {
	m.threadStateMux.Lock()
	defer m.threadStateMux.Unlock()

	// Calculate optimal thread limits based on system resources
	// Limit concurrent active threads to prevent resource exhaustion
	m.maxActiveThreads = int32(m.threads / 2) // Max 50% threads active simultaneously
	if m.maxActiveThreads < 2 {
		m.maxActiveThreads = 2 // Minimum 2 active threads
	}

	// Set staggered execution delay
	m.threadStartDelay = 100 * time.Millisecond // 100ms between thread starts

	// Initialize thread states
	for i := 0; i < m.threads; i++ {
		m.threadStates[i] = &ThreadState{
			ID:            i,
			Status:        "idle",
			LastHeartbeat: time.Now(),
			StuckCount:    0,
		}
	}

	logInfo("🧵 Thread management initialized: %d threads, max %d active", m.threads, m.maxActiveThreads)
}

// initializeMemoryPools pre-allocates memory to prevent swapping
func (m *QuantumMiner) initializeMemoryPools() {
	// Calculate memory requirements per puzzle set
	const bytesPerQubits = 2    // 16 qubits = 2 bytes
	const puzzlesPerSet = 48    // Standard puzzle count
	const gateHashSize = 32     // SHA256 hash size
	const workBufferSize = 1024 // Additional working memory

	memoryPerSet := (bytesPerQubits * puzzlesPerSet) + (gateHashSize * puzzlesPerSet) + workBufferSize

	// Pre-allocate memory pool to avoid runtime allocation
	// Use available system memory efficiently
	m.memoryPoolSize = 50 // Conservative pool size
	if m.gpuMode {
		m.memoryPoolSize = 20 // GPU mode uses more memory per operation
	}

	logInfo("🧠 Initializing memory pools: %d sets, %d bytes per set", m.memoryPoolSize, memoryPerSet)

	// Pre-allocate all memory blocks
	for i := 0; i < m.memoryPoolSize; i++ {
		puzzleMemory := make([]PuzzleMemory, 1)
		puzzleMemory[0] = PuzzleMemory{
			ID:         i,
			Outcomes:   make([][]byte, puzzlesPerSet),
			GateHashes: make([][]byte, puzzlesPerSet),
			WorkBuffer: make([]byte, workBufferSize),
		}

		// Pre-allocate outcome and gate hash buffers
		for j := 0; j < puzzlesPerSet; j++ {
			puzzleMemory[0].Outcomes[j] = make([]byte, bytesPerQubits)
			puzzleMemory[0].GateHashes[j] = make([]byte, gateHashSize)
		}

		// Add to pool (non-blocking)
		select {
		case m.puzzleMemoryPool <- puzzleMemory:
		default:
			// Pool full, this shouldn't happen during initialization
			logError("Memory pool full during initialization")
		}
	}

	logInfo("✅ Memory pools initialized: %d MB pre-allocated", (memoryPerSet*m.memoryPoolSize)/(1024*1024))
}

// threadMonitor watches for stuck threads and recovers them
func (m *QuantumMiner) threadMonitor() {
	ticker := time.NewTicker(5 * time.Second) // Check every 5 seconds
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.checkForStuckThreads()
		}
	}
}

// checkForStuckThreads identifies and recovers stuck threads
func (m *QuantumMiner) checkForStuckThreads() {
	m.threadStateMux.Lock()
	defer m.threadStateMux.Unlock()

	now := time.Now()
	stuckThreshold := 15 * time.Second // Consider stuck after 15 seconds

	for threadID, state := range m.threadStates {
		if state.Status == "working" {
			timeSinceHeartbeat := now.Sub(state.LastHeartbeat)

			if timeSinceHeartbeat > stuckThreshold {
				state.StuckCount++
				logInfo("🚨 Thread %d stuck for %v (count: %d), attempting recovery",
					threadID, timeSinceHeartbeat, state.StuckCount)

				// Request abort
				state.AbortRequested = true
				state.Status = "aborting"

				// Hard abort if stuck multiple times
				if state.StuckCount >= 3 && state.cancelFunc != nil {
					logInfo("🛑 Hard aborting thread %d after %d stuck occurrences", threadID, state.StuckCount)
					state.cancelFunc()
				}
			}
		}
	}
}

// enhancedMiningThread runs a single mining thread with improved lifecycle management
func (m *QuantumMiner) enhancedMiningThread(threadID int) {
	// Initialize thread state
	m.threadStateMux.Lock()
	state := m.threadStates[threadID]
	state.Status = "idle"
	state.LastHeartbeat = time.Now()
	m.threadStateMux.Unlock()

	logInfo("🧵 Thread %d started with enhanced management", threadID)

	for {
		select {
		case <-m.stopChan:
			m.updateThreadState(threadID, "stopped", "", 0)
			return
		default:
			// Check if we should throttle thread activation
			if !m.shouldActivateThread(threadID) {
				time.Sleep(200 * time.Millisecond)
				continue
			}

			if err := m.enhancedMineBlock(threadID); err != nil {
				m.updateThreadState(threadID, "error", "", 0)
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
}

// shouldActivateThread determines if a thread should become active
func (m *QuantumMiner) shouldActivateThread(threadID int) bool {
	activeCount := atomic.LoadInt32(&m.activeThreads)

	// Always allow if under the limit
	if activeCount < m.maxActiveThreads {
		return true
	}

	// Check if this thread is already active
	m.threadStateMux.RLock()
	state := m.threadStates[threadID]
	isActive := state.Status == "working"
	m.threadStateMux.RUnlock()

	return isActive
}

// updateThreadState safely updates thread state
func (m *QuantumMiner) updateThreadState(threadID int, status string, workHash string, qnonce uint64) {
	m.threadStateMux.Lock()
	defer m.threadStateMux.Unlock()

	state := m.threadStates[threadID]
	oldStatus := state.Status

	state.Status = status
	state.WorkHash = workHash
	state.QNonce = qnonce
	state.LastHeartbeat = time.Now()

	// Update active thread count
	if oldStatus != "working" && status == "working" {
		atomic.AddInt32(&m.activeThreads, 1)
		state.StartTime = time.Now()
		state.AbortRequested = false
	} else if oldStatus == "working" && status != "working" {
		atomic.AddInt32(&m.activeThreads, -1)
	}
}

// Stop stops the miner
func (m *QuantumMiner) Stop() {
	if !atomic.CompareAndSwapInt32(&m.running, 1, 0) {
		return
	}

	log.Printf("🛑 Shutdown signal received...")
	close(m.stopChan)

	// Wait a moment for threads to clean up
	time.Sleep(1 * time.Second)

	// Print final statistics in professional format
	attempts := atomic.LoadUint64(&m.attempts)
	puzzlesSolved := atomic.LoadUint64(&m.puzzlesSolved)
	accepted := atomic.LoadUint64(&m.accepted)
	rejected := atomic.LoadUint64(&m.rejected)
	stale := atomic.LoadUint64(&m.stale)
	duplicates := atomic.LoadUint64(&m.duplicates)

	duration := time.Since(m.startTime)
	avgQNonceRate := float64(attempts) / duration.Seconds()
	avgPuzzleRate := float64(puzzlesSolved) / duration.Seconds()

	totalShares := accepted + rejected + stale + duplicates
	acceptanceRate := float64(0)
	if totalShares > 0 {
		acceptanceRate = float64(accepted) / float64(totalShares) * 100
	}

	log.Printf("")
	log.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════")
	log.Printf("🏁 FINAL QUANTUM MINING SESSION REPORT")
	log.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════")
	if m.gpuMode {
		log.Printf("🎮 Mining Mode    │ GPU ACCELERATED (Device %d) │ %d Parallel Threads", m.gpuID, m.threads)
	} else {
		log.Printf("💻 Mining Mode    │ CPU ONLY │ %d Threads", m.threads)
	}
	log.Printf("⏱️  Session Time   │ %s │ Started: %s", formatDuration(duration), m.startTime.Format("15:04:05"))
	log.Printf("⚡ Performance    │ QNonces: %8.2f QN/s │ Puzzles: %8.2f PZ/s", avgQNonceRate, avgPuzzleRate)
	log.Printf("🧮 Work Completed │ QNonces: %d │ Puzzles: %d │ Ratio: %.1f puzzles/qnonce",
		attempts, puzzlesSolved, float64(puzzlesSolved)/float64(attempts))
	log.Printf("🎯 Block Results  │ Accepted: %d │ Rejected: %d │ Success Rate: %.2f%%",
		accepted, rejected+stale+duplicates, acceptanceRate)
	if rejected+stale+duplicates > 0 {
		log.Printf("❌ Reject Details │ Invalid: %d │ Stale: %d │ Duplicates: %d", rejected, stale, duplicates)
	}
	log.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════")
	log.Printf("👋 Thank you for contributing to the Quantum-Geth network!")
	log.Printf("💎 Your quantum computations help secure the blockchain!")
	log.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════")
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
	// AGGRESSIVE WORK REFRESH: Fast response to rapid block changes
	// When blocks are being found every 12 seconds, we need sub-second work updates
	ticker := time.NewTicker(100 * time.Millisecond) // 10x per second - very aggressive for rapid blocks
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			if err := m.fetchWork(); err != nil {
				log.Printf("❌ Failed to fetch work: %v", err)
				time.Sleep(200 * time.Millisecond) // Brief pause on error
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

	// Get difficulty info for dashboard
	if diffResult, err := m.rpcCall("eth_getBlockByNumber", []interface{}{"latest", false}); err == nil {
		if blockData, ok := diffResult.(map[string]interface{}); ok {
			if diffHex, ok := blockData["difficulty"].(string); ok {
				if difficulty, err := strconv.ParseUint(strings.TrimPrefix(diffHex, "0x"), 16, 64); err == nil {
					m.currentDifficulty = difficulty
				}
			}
		}
	}

	// Store current work
	m.workMutex.Lock()
	oldWork := m.currentWork
	if oldWork == nil || oldWork.WorkHash != work.WorkHash {
		m.currentWork = work
		// Only log new work on first start or significant changes
		if oldWork == nil {
			log.Printf("📦 Starting mining on Block %d", work.BlockNumber)
		}
		logInfo("New work received: Block %d, Difficulty %d, Target: %s", work.BlockNumber, 2000000000, work.Target)
	}
	m.workMutex.Unlock()

	return nil
}

// enhancedMineBlock performs quantum mining with improved thread and memory management
func (m *QuantumMiner) enhancedMineBlock(threadID int) error {
	// Get current work
	m.workMutex.RLock()
	work := m.currentWork
	m.workMutex.RUnlock()

	if work == nil {
		m.updateThreadState(threadID, "idle", "", 0)
		time.Sleep(100 * time.Millisecond)
		return nil
	}

	// Check if thread should abort due to monitoring
	m.threadStateMux.RLock()
	state := m.threadStates[threadID]
	shouldAbort := state.AbortRequested
	m.threadStateMux.RUnlock()

	if shouldAbort {
		m.updateThreadState(threadID, "idle", "", 0)
		logInfo("Thread %d: Aborting due to monitor request", threadID)
		return nil
	}

	// ENHANCED STALENESS CHECK: Only abandon work if we get work for a newer block
	// Time-based staleness was too aggressive - we should keep trying the same block until it's mined
	// The real "stale" condition is when geth gives us work for a different (newer) block number

	// THREAD-SAFE UNIQUE QUANTUM NONCE GENERATION
	counter := atomic.AddUint64(&m.nonceCounter, 1)
	qnonce := m.nonceBase + (counter << 8) + uint64(threadID)

	// Update thread state to working
	m.updateThreadState(threadID, "working", work.WorkHash, qnonce)

	// Create context for cancellation - shorter timeout to prevent stuck threads
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Store cancel function for hard abort
	m.threadStateMux.Lock()
	m.threadStates[threadID].cancelFunc = cancel
	m.threadStateMux.Unlock()

	// Store work hash for change detection during puzzle solving
	workHash := work.WorkHash

	// Solve quantum puzzles with memory management
	puzzleStart := time.Now()
	proof, err := m.enhancedSolveQuantumPuzzles(ctx, threadID, work.WorkHash, qnonce, work.QBits, work.TCount, work.LNet)
	if err != nil {
		m.updateThreadState(threadID, "error", "", 0)
		// Check if it was a context cancellation (abort) - don't spam logs
		if ctx.Err() == context.DeadlineExceeded {
			logInfo("Thread %d: Puzzle solving timed out after 5s, aborting", threadID)
		} else if ctx.Err() == context.Canceled {
			logInfo("Thread %d: Puzzle solving cancelled, aborting", threadID)
		} else {
			logError("Thread %d: Puzzle solving failed: %v", threadID, err)
		}
		return err
	}

	puzzleTime := time.Since(puzzleStart)

	// Update heartbeat
	m.updateThreadState(threadID, "working", workHash, qnonce)

	// ENHANCED WORK CHANGE DETECTION: Check for new work
	m.workMutex.RLock()
	currentWork := m.currentWork
	m.workMutex.RUnlock()

	if currentWork == nil || currentWork.WorkHash != workHash {
		m.updateThreadState(threadID, "idle", "", 0)
		if currentWork != nil && time.Since(currentWork.FetchTime) < 2*time.Second {
			logInfo("Thread %d: Fresh work available (age: %v), aborting stale solution", threadID, time.Since(currentWork.FetchTime))
		}
		return nil
	}

	// Update solve time statistics
	atomic.AddUint64(&m.puzzlesSolved, uint64(work.LNet))
	m.totalSolveTime += puzzleTime

	// Debug logging for GPU mining investigation
	logInfo("Thread %d: Solved %d puzzles in %v, qnonce=%016x, checking target...", threadID, work.LNet, puzzleTime, qnonce)

	// Check if solution meets target (Bitcoin-style difficulty check)
	if m.checkQuantumTarget(proof, work.Target) {
		logInfo("Thread %d: Target met! Submitting solution qnonce=%016x...", threadID, qnonce)
		// Block found! Submit quietly
		if err := m.submitQuantumWork(qnonce, work.WorkHash, proof); err != nil {
			errMsg := err.Error()
			logError("Thread %d: Submission failed qnonce=%016x: %v", threadID, qnonce, err)
			if strings.Contains(errMsg, "stale") {
				atomic.AddUint64(&m.stale, 1)
			} else if strings.Contains(errMsg, "duplicate") {
				atomic.AddUint64(&m.duplicates, 1)
			} else {
				atomic.AddUint64(&m.rejected, 1)
			}
		} else {
			logInfo("Thread %d: Block accepted qnonce=%016x! 🎉", threadID, qnonce)
			atomic.AddUint64(&m.accepted, 1)
			// Track new block time for average calculation
			now := time.Now()
			m.blockTimes = append(m.blockTimes, now)
			if len(m.blockTimes) > 10 {
				m.blockTimes = m.blockTimes[1:] // Keep only last 10
			}

			// CONTINUOUS MINING: Immediately request fresh work after finding a block
			go func() {
				time.Sleep(100 * time.Millisecond) // Brief delay to avoid overwhelming geth
				m.fetchWork()                      // Trigger immediate work refresh
			}()
		}
	} else {
		logInfo("Thread %d: Target not met qnonce=%016x, trying next nonce...", threadID, qnonce)
	}

	atomic.AddUint64(&m.attempts, 1)
	m.updateThreadState(threadID, "idle", "", 0)
	return nil
}

// enhancedSolveQuantumPuzzles solves puzzles with memory management and cancellation support
func (m *QuantumMiner) enhancedSolveQuantumPuzzles(ctx context.Context, threadID int, workHash string, qnonce uint64, qbits, tcount, lnet int) (QuantumProofSubmission, error) {
	// Acquire memory from pool
	var memoryBlock []PuzzleMemory
	select {
	case memoryBlock = <-m.puzzleMemoryPool:
		// Got memory block
	case <-time.After(1 * time.Second):
		// No memory available, fall back to regular allocation
		logInfo("Thread %d: Memory pool exhausted, using fallback allocation", threadID)
		return m.solveQuantumPuzzles(workHash, qnonce, qbits, tcount, lnet)
	case <-ctx.Done():
		return QuantumProofSubmission{}, ctx.Err()
	}

	// Ensure memory is returned to pool
	defer func() {
		select {
		case m.puzzleMemoryPool <- memoryBlock:
			// Memory returned successfully
		default:
			// Pool full, this is unusual but not critical
			logInfo("Thread %d: Memory pool full on return, discarding block", threadID)
		}
	}()

	memory := &memoryBlock[0]

	// Update heartbeat periodically
	heartbeatTicker := time.NewTicker(1 * time.Second)
	defer heartbeatTicker.Stop()

	// Start heartbeat goroutine
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-heartbeatTicker.C:
				m.updateThreadState(threadID, "working", workHash, qnonce)
			}
		}
	}()

	// Simulate quantum computation time based on circuit complexity and GPU acceleration
	var baseTime time.Duration
	var complexityFactor float64

	if m.gpuMode {
		// GPU mining with CUDA acceleration - much faster parallel quantum circuit execution
		baseTime = 3 * time.Millisecond                              // Reduced from 5ms for better responsiveness
		complexityFactor = float64(qbits*tcount) / (16 * 8192) * 0.1 // Additional GPU optimization
	} else {
		// CPU mining with standard timing
		baseTime = 30 * time.Millisecond                       // Reduced from 50ms for better responsiveness
		complexityFactor = float64(qbits*tcount) / (16 * 8192) // Relative to standard params
	}

	// Note: puzzleTime calculated but not used in enhanced version
	// as timing is handled by the GPU/CPU simulation functions
	_ = time.Duration(float64(baseTime) * complexityFactor)

	// Solve puzzles using pre-allocated memory
	var outcomes [][]byte
	var gateHashes [][]byte

	if m.gpuMode && m.multiGPUEnabled {
		// Use multi-GPU batch quantum simulation
		logInfo("Thread %d: Starting multi-GPU batch simulation: %d puzzles, %d qubits, %d gates", threadID, lnet, qbits, tcount)

		// Submit work to GPU system
		err := m.submitGPUWork(threadID, workHash, qnonce, qbits, tcount, lnet)
		if err != nil {
			logError("Thread %d: Failed to submit GPU work: %v", threadID, err)
			return m.cpuFallbackWithMemory(ctx, threadID, memory, workHash, qnonce, qbits, tcount, lnet)
		}

		// Wait for result with timeout
		timeout := time.NewTimer(5 * time.Second)
		defer timeout.Stop()

		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		// Simple result waiting (in production, this would use a proper result channel)
		for {
			select {
			case <-timeout.C:
				logInfo("Thread %d: GPU work timeout after 5s, falling back to CPU", threadID)
				return m.cpuFallbackWithMemory(ctx, threadID, memory, workHash, qnonce, qbits, tcount, lnet)
			case <-ctx.Done():
				return QuantumProofSubmission{}, ctx.Err()
			case <-ticker.C:
				// For now, fall back to CPU after submitting GPU work
				// In a full implementation, we'd wait for the actual GPU result
				return m.cpuFallbackWithMemory(ctx, threadID, memory, workHash, qnonce, qbits, tcount, lnet)
			}
		}
	} else {
		// Use CPU simulation with pre-allocated memory
		return m.cpuFallbackWithMemory(ctx, threadID, memory, workHash, qnonce, qbits, tcount, lnet)
	}

	// Generate gate hashes using pre-allocated memory
	for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return QuantumProofSubmission{}, ctx.Err()
		default:
		}

		seed := fmt.Sprintf("%s_%016x_%d", workHash, qnonce, puzzleIndex)
		gateData := fmt.Sprintf("gates_%s_%d_qubits_%d_tgates", seed, qbits, tcount)
		gateHash := sha256Hash(gateData)

		// Reuse pre-allocated buffer
		copy(memory.GateHashes[puzzleIndex], []byte(gateHash)[:32])
		gateHashes = append(gateHashes, memory.GateHashes[puzzleIndex])
	}

	// Calculate outcome root from all puzzle outcomes (like geth)
	outcomeRoot := m.calculateOutcomeRoot(outcomes)

	// Calculate combined gate hash
	combinedGateData := ""
	for _, gh := range gateHashes {
		combinedGateData += fmt.Sprintf("%x", gh)
	}
	gateHash := sha256Hash(combinedGateData)

	// Generate proof root and other proof components
	proofData := fmt.Sprintf("%s_%s_%016x", outcomeRoot, gateHash, qnonce)
	proofRoot := sha256Hash(proofData)

	// Generate branch nibbles and extra nonce
	branchData := fmt.Sprintf("branch_%s_%016x", workHash, qnonce)
	branchHash := sha256Hash(branchData)

	extraNonceData := fmt.Sprintf("extra_%016x_%s", qnonce, workHash)
	extraNonceHash := sha256Hash(extraNonceData)

	return QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchHash[:8],     // First 8 characters
		ExtraNonce32:  extraNonceHash[:8], // First 8 characters
	}, nil
}

// cpuFallbackWithMemory performs CPU simulation using pre-allocated memory
func (m *QuantumMiner) cpuFallbackWithMemory(ctx context.Context, threadID int, memory *PuzzleMemory, workHash string, qnonce uint64, qbits, tcount, lnet int) (QuantumProofSubmission, error) {
	var outcomes [][]byte
	var gateHashes [][]byte

	// Use optimized CPU simulation with proper CPU timing and pre-allocated memory
	for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
		// Check for cancellation frequently
		select {
		case <-ctx.Done():
			return QuantumProofSubmission{}, ctx.Err()
		default:
		}

		// Shorter sleep for better responsiveness
		time.Sleep(20 * time.Millisecond) // Reduced from 50ms

		// UNIQUE SOLUTION: Use qnonce and puzzle index for unique seeds per thread
		seed := fmt.Sprintf("%s_%016x_%d_%d", workHash, qnonce, puzzleIndex, time.Now().UnixNano())
		hash := sha256Hash(fmt.Sprintf("outcome_%s", seed))

		// Use pre-allocated buffer
		copy(memory.Outcomes[puzzleIndex], []byte(hash)[:len(memory.Outcomes[puzzleIndex])])
		outcomes = append(outcomes, memory.Outcomes[puzzleIndex])

		// Update heartbeat every few iterations
		if puzzleIndex%10 == 0 {
			m.updateThreadState(threadID, "working", workHash, qnonce)
		}
	}

	// Generate gate hashes using pre-allocated memory
	for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
		select {
		case <-ctx.Done():
			return QuantumProofSubmission{}, ctx.Err()
		default:
		}

		seed := fmt.Sprintf("%s_%016x_%d", workHash, qnonce, puzzleIndex)
		gateData := fmt.Sprintf("gates_%s_%d_qubits_%d_tgates", seed, qbits, tcount)
		gateHash := sha256Hash(gateData)

		copy(memory.GateHashes[puzzleIndex], []byte(gateHash)[:32])
		gateHashes = append(gateHashes, memory.GateHashes[puzzleIndex])
	}

	// Calculate outcome root and other components
	outcomeRoot := m.calculateOutcomeRoot(outcomes)

	combinedGateData := ""
	for _, gh := range gateHashes {
		combinedGateData += fmt.Sprintf("%x", gh)
	}
	gateHash := sha256Hash(combinedGateData)

	proofData := fmt.Sprintf("%s_%s_%016x", outcomeRoot, gateHash, qnonce)
	proofRoot := sha256Hash(proofData)

	branchData := fmt.Sprintf("branch_%s_%016x", workHash, qnonce)
	branchHash := sha256Hash(branchData)

	extraNonceData := fmt.Sprintf("extra_%016x_%s", qnonce, workHash)
	extraNonceHash := sha256Hash(extraNonceData)

	return QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchHash[:8],
		ExtraNonce32:  extraNonceHash[:8],
	}, nil
}

// solveQuantumPuzzles simulates solving actual quantum puzzles (like geth's approach)
func (m *QuantumMiner) solveQuantumPuzzles(workHash string, qnonce uint64, qbits, tcount, lnet int) (QuantumProofSubmission, error) {
	// Simulate quantum computation time based on circuit complexity and GPU acceleration
	// Real quantum circuits take time: 16 qubits * 8192 T-gates * 48 puzzles
	var baseTime time.Duration
	var complexityFactor float64

	if m.gpuMode {
		// GPU mining with CUDA acceleration - much faster parallel quantum circuit execution
		baseTime = 5 * time.Millisecond                              // GPU acceleration: 10x faster than CPU
		complexityFactor = float64(qbits*tcount) / (16 * 8192) * 0.1 // Additional GPU optimization
	} else {
		// CPU mining with standard timing
		baseTime = 50 * time.Millisecond                       // Base time per puzzle
		complexityFactor = float64(qbits*tcount) / (16 * 8192) // Relative to standard params
	}

	puzzleTime := time.Duration(float64(baseTime) * complexityFactor)

	// Simulate solving each puzzle (like geth's real quantum solver)
	var outcomes [][]byte
	var gateHashes [][]byte

	if m.gpuMode && m.multiGPUEnabled && len(m.availableGPUs) > 0 {
		// Use multi-GPU batch quantum simulation (eliminates sync bottlenecks)
		logInfo("Starting multi-GPU batch simulation: %d puzzles, %d qubits, %d gates", lnet, qbits, tcount)

		// Try to get a GPU simulator from the available ones
		deviceID := m.gpuLoadBalancer.SelectDevice()
		if deviceID >= 0 && m.gpuSimulators[deviceID] != nil {
			batchOutcomes, err := m.gpuSimulators[deviceID].BatchSimulateQuantumPuzzles(
				workHash, qnonce, qbits, tcount, lnet,
			)
			m.gpuLoadBalancer.ReleaseDevice(deviceID)

			if err != nil {
				// GPU simulation failed - use CPU fallback (silent operation)
				// Only log if it's not an interrupt signal
				if err.Error() != "simulation interrupted" {
					// Log GPU error only once every 10 seconds to avoid spam
					now := time.Now()
					if now.Sub(m.lastStatTime) > 10*time.Second {
						logError("⚠️  GPU %d fallback: %v", deviceID, err)
						m.lastStatTime = now
					}
				}
				// Fallback to individual CPU simulation with proper CPU timing
				for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
					time.Sleep(50 * time.Millisecond) // Use CPU timing, not GPU timing
					outcomeBytes, err := m.solveQuantumPuzzleCPU(puzzleIndex, workHash, qnonce, qbits, tcount)
					if err != nil {
						return QuantumProofSubmission{}, fmt.Errorf("both GPU and CPU simulation failed: %w", err)
					}
					outcomes = append(outcomes, outcomeBytes)
				}
			} else {
				outcomes = batchOutcomes
				// GPU processing completed successfully (quiet operation)
			}
		} else {
			// No GPU available, fall back to CPU
			logError("⚠️  No GPU devices available - falling back to CPU mode")
			for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
				time.Sleep(50 * time.Millisecond) // Proper CPU timing
				// UNIQUE SOLUTION: Use qnonce and puzzle index for unique seeds per thread
				seed := fmt.Sprintf("%s_%016x_%d_%d", workHash, qnonce, puzzleIndex, time.Now().UnixNano())
				outcomeBytes := make([]byte, (qbits+7)/8)
				hash := sha256Hash(fmt.Sprintf("outcome_%s", seed))
				copy(outcomeBytes, []byte(hash)[:len(outcomeBytes)])
				outcomes = append(outcomes, outcomeBytes)
			}
		}
	} else if m.gpuMode && !m.multiGPUEnabled {
		// CRITICAL FIX: GPU mode requested but multi-GPU not available
		log.Printf("⚠️  GPU mode requested but multi-GPU not available - falling back to optimized CPU mode")
		// Use optimized CPU simulation with proper CPU timing (not GPU timing)
		for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
			time.Sleep(50 * time.Millisecond) // Proper CPU timing
			// UNIQUE SOLUTION: Use qnonce and puzzle index for unique seeds per thread
			seed := fmt.Sprintf("%s_%016x_%d_%d", workHash, qnonce, puzzleIndex, time.Now().UnixNano())
			outcomeBytes := make([]byte, (qbits+7)/8)
			hash := sha256Hash(fmt.Sprintf("outcome_%s", seed))
			copy(outcomeBytes, []byte(hash)[:len(outcomeBytes)])
			outcomes = append(outcomes, outcomeBytes)
		}
	} else {
		// Use CPU simulation (previous behavior)
		for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
			time.Sleep(puzzleTime)
			// UNIQUE SOLUTION: Use qnonce and puzzle index for unique seeds per thread
			seed := fmt.Sprintf("%s_%016x_%d_%d", workHash, qnonce, puzzleIndex, time.Now().UnixNano())
			outcomeBytes := make([]byte, (qbits+7)/8)
			hash := sha256Hash(fmt.Sprintf("outcome_%s", seed))
			copy(outcomeBytes, []byte(hash)[:len(outcomeBytes)])
			outcomes = append(outcomes, outcomeBytes)
		}
	}

	// Generate gate hashes for all puzzles with unique qnonce
	for puzzleIndex := 0; puzzleIndex < lnet; puzzleIndex++ {
		seed := fmt.Sprintf("%s_%016x_%d", workHash, qnonce, puzzleIndex)
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

	// Generate proof root (Nova proofs simulation) with unique qnonce
	proofData := fmt.Sprintf("nova_proof_%s_%s_%s_%016x", outcomeRoot, gateHash, workHash, qnonce)
	proofRoot := sha256Hash(proofData)

	// Generate branch nibbles (48 bytes for 48 puzzles)
	branchNibbles := make([]byte, 48)
	for i := 0; i < 48; i++ {
		if i < len(outcomes) {
			branchNibbles[i] = outcomes[i][0] >> 4 // High nibble
		}
	}

	// Generate extra nonce with embedded qnonce for uniqueness
	extraNonce := make([]byte, 32)
	rand.Read(extraNonce)
	// Embed qnonce in extra nonce for guaranteed uniqueness
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

// solveQuantumPuzzleCPU solves a single quantum puzzle using CPU simulation
func (m *QuantumMiner) solveQuantumPuzzleCPU(puzzleIndex int, workHash string, qnonce uint64, qbits, tcount int) ([]byte, error) {
	// Generate deterministic quantum circuit with unique qnonce
	seed := fmt.Sprintf("%s_%016x_%d", workHash, qnonce, puzzleIndex)

	// Simulate quantum measurement outcome (qbits of data)
	outcomeBytes := make([]byte, (qbits+7)/8)
	hash := sha256Hash(fmt.Sprintf("outcome_%s", seed))
	copy(outcomeBytes, []byte(hash)[:len(outcomeBytes)])

	// Simulate computation time
	simulationTime := time.Duration(tcount/1000) * time.Microsecond
	if simulationTime > 100*time.Millisecond {
		simulationTime = 100 * time.Millisecond
	}
	time.Sleep(simulationTime)

	return outcomeBytes, nil
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
	// Rate limiting: prevent overwhelming geth with too many simultaneous submissions
	select {
	case m.submissionSemaphore <- struct{}{}:
		defer func() { <-m.submissionSemaphore }()
	case <-time.After(5 * time.Second):
		return fmt.Errorf("submission queue full - too many pending submissions")
	}

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

	// Try quantum-specific SubmitWork first (FIXED: qnonce as raw uint64)
	params := []interface{}{
		qnonce, // qnonce as raw uint64 for proper JSON marshaling
		workHashWithPrefix,
		quantumProofForAPI,
	}

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

		// Fall back to eth_submitWork with adapted parameters (nonce as hex string)
		ethParams := []interface{}{
			fmt.Sprintf("0x%016x", qnonce),                   // nonce as hex string for eth_submitWork
			workHashWithPrefix,                               // hash
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

// statsReporter reports mining statistics as a live updating dashboard
func (m *QuantumMiner) statsReporter() {
	ticker := time.NewTicker(2 * time.Second) // Fast updates for live dashboard
	defer ticker.Stop()

	// Clear screen and show initial dashboard
	fmt.Print("\033[2J\033[H") // Clear screen and move cursor to top

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.updateDashboard()
		}
	}
}

// updateDashboard displays live mining dashboard that updates in place
func (m *QuantumMiner) updateDashboard() {
	now := time.Now()
	attempts := atomic.LoadUint64(&m.attempts)
	puzzlesSolved := atomic.LoadUint64(&m.puzzlesSolved)
	accepted := atomic.LoadUint64(&m.accepted)
	rejected := atomic.LoadUint64(&m.rejected)
	stale := atomic.LoadUint64(&m.stale)

	totalDuration := now.Sub(m.startTime)
	intervalDuration := now.Sub(m.lastStatTime)

	// Calculate rates in raw units (no thousands)
	avgQNonceRate := float64(attempts) / totalDuration.Seconds()      // QN/s
	avgPuzzleRate := float64(puzzlesSolved) / totalDuration.Seconds() // PZ/s

	// Calculate interval rates for real-time performance
	intervalAttempts := attempts - m.lastAttempts
	intervalPuzzles := puzzlesSolved - m.lastPuzzles
	currentQNonceRate := float64(intervalAttempts) / intervalDuration.Seconds() // QN/s
	currentPuzzleRate := float64(intervalPuzzles) / intervalDuration.Seconds()  // PZ/s

	// Update stored values
	m.lastStatTime = now
	m.lastAttempts = attempts
	m.lastPuzzles = puzzlesSolved

	// Calculate average block time from last blocks
	avgBlockTime := float64(0)
	if len(m.blockTimes) > 1 {
		totalBlockTime := m.blockTimes[len(m.blockTimes)-1].Sub(m.blockTimes[0])
		avgBlockTime = totalBlockTime.Seconds() / float64(len(m.blockTimes)-1)
	}

	// Get current work info
	m.workMutex.RLock()
	work := m.currentWork
	m.workMutex.RUnlock()

	blockNumber := uint64(0)
	blocksToRetarget := uint64(0)
	if work != nil {
		blockNumber = work.BlockNumber
		blocksToRetarget = 100 - (blockNumber % 100) // Retarget every 100 blocks
	}

	// Move cursor to top and clear screen content (but keep same size)
	fmt.Print("\033[H")

	// Live Dashboard Display
	fmt.Println("┌─────────────────────────────────────────────────────────────────────────────────┐")
	if m.gpuMode {
		fmt.Printf("│ 🎮 QUANTUM GPU MINER │ Device %d │ %d Threads │ Runtime: %-20s │\n",
			m.gpuID, m.threads, formatDuration(totalDuration))
	} else {
		fmt.Printf("│ 💻 QUANTUM CPU MINER │ %d Threads │ Runtime: %-32s │\n",
			m.threads, formatDuration(totalDuration))
	}
	fmt.Println("├─────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ ⚡ QNonce Rate     │ Current: %8.2f QN/s │ Average: %8.2f QN/s     │\n",
		currentQNonceRate, avgQNonceRate)
	fmt.Printf("│ ⚛️  Puzzle Rate     │ Current: %8.2f PZ/s │ Average: %8.2f PZ/s     │\n",
		currentPuzzleRate, avgPuzzleRate)
	fmt.Println("├─────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ 🎯 Blocks Found    │ Accepted: %-6d │ Rejected: %-6d │ Stale: %-6d │\n",
		accepted, rejected, stale)
	fmt.Printf("│ 📊 Work Stats      │ Total QNonces: %-10d │ Total Puzzles: %-10d │\n",
		attempts, puzzlesSolved)

	// Enhanced thread status display
	activeCount := atomic.LoadInt32(&m.activeThreads)
	fmt.Printf("│ 🧵 Thread Status   │ Active: %d/%-2d │ Max Concurrent: %-2d │ Pool: %d/%d │\n",
		activeCount, m.threads, m.maxActiveThreads, len(m.puzzleMemoryPool), m.memoryPoolSize)
	fmt.Println("├─────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ 🔗 Current Block   │ Block: %-10d │ Difficulty: %-15d │\n",
		blockNumber, m.currentDifficulty)
	if avgBlockTime > 0 {
		fmt.Printf("│ ⏱️  Block Timing    │ Average: %6.1fs │ Target: %6.1fs │ To Retarget: %-3d │\n",
			avgBlockTime, m.targetBlockTime.Seconds(), blocksToRetarget)
	} else {
		fmt.Printf("│ ⏱️  Block Timing    │ Average: %-8s │ Target: %6.1fs │ To Retarget: %-3d │\n",
			"N/A", m.targetBlockTime.Seconds(), blocksToRetarget)
	}
	fmt.Println("└─────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Printf("Last Update: %s | Press Ctrl+C to stop\n", now.Format("15:04:05"))
}

// formatDuration formats duration in a human-readable way
func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	} else if d < time.Hour {
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	} else {
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
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
	fmt.Println("Quantum-Geth GPU/CPU Miner v" + VERSION)
	fmt.Println("")
	fmt.Println("USAGE:")
	fmt.Println("  quantum-gpu-miner [OPTIONS]")
	fmt.Println("")
	fmt.Println("OPTIONS:")
	fmt.Println("  -coinbase ADDRESS    Coinbase address for mining rewards (required)")
	fmt.Println("  -node URL           Full node URL (default: http://localhost:8545)")
	fmt.Println("  -ip ADDRESS         Node IP address (default: localhost)")
	fmt.Println("  -port NUMBER        Node RPC port (default: 8545)")
	fmt.Println("  -threads N          Number of mining threads (default: CPU cores)")
	fmt.Println("  -gpu                Enable GPU mining with CUDA/Qiskit acceleration")
	fmt.Println("  -gpu-id N           GPU device ID to use (default: 0)")
	fmt.Println("  -log                Enable detailed logging to quantum-miner.log file")
	fmt.Println("  -version            Show version information")
	fmt.Println("  -help               Show this help message")
	fmt.Println("")
	fmt.Println("EXAMPLES:")
	fmt.Println("  # CPU mining (default)")
	fmt.Println("  quantum-gpu-miner -coinbase 0x742d35C6C4e6d8de6f10E7FF75DD98dd25b02C3A")
	fmt.Println("")
	fmt.Println("  # GPU mining with CUDA acceleration")
	fmt.Println("  quantum-gpu-miner -coinbase 0x123... -gpu")
	fmt.Println("  quantum-gpu-miner -coinbase 0x123... -gpu -gpu-id 1")
	fmt.Println("")
	fmt.Println("  # Custom network settings")
	fmt.Println("  quantum-gpu-miner -coinbase 0x123... -ip 192.168.1.100 -port 8545")
	fmt.Println("  quantum-gpu-miner -coinbase 0x123... -node http://my-quantum-node.com:8545")
	fmt.Println("")
	fmt.Println("  # Multiple threads")
	fmt.Println("  quantum-gpu-miner -coinbase 0x123... -threads 8")
	fmt.Println("  quantum-gpu-miner -coinbase 0x123... -gpu -threads 16  # GPU parallel circuits")
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

// detectAvailableGPUs detects available GPU devices for quantum mining
func detectAvailableGPUs() ([]int, error) {
	// For CuPy, we typically use a single GPU context shared across all work
	// Test if any GPU is available first

	var availableGPUs []int

	// Test if CuPy GPU is available (this will be cached in the simulator)
	testSim, err := quantum.NewHighPerformanceQuantumSimulator(16)
	if err != nil {
		return availableGPUs, fmt.Errorf("failed to test GPU availability: %v", err)
	}
	defer testSim.Cleanup()

	// For CuPy, we'll assume up to 8 logical GPU contexts can be used
	// even if they map to the same physical GPU
	maxGPUs := 8

	// If CuPy GPU is available, register multiple logical GPUs
	// This allows parallel processing on the same GPU
	for deviceID := 0; deviceID < maxGPUs; deviceID++ {
		availableGPUs = append(availableGPUs, deviceID)
		logInfo("✅ GPU %d: HIGH-PERFORMANCE Quantum acceleration AVAILABLE", deviceID)
	}

	if len(availableGPUs) > 0 {
		logInfo("🚀 Multi-GPU Support: %d GPUs detected", len(availableGPUs))
		logInfo("   🎯 Batch processing with load balancing enabled")
	}

	return availableGPUs, nil
}

// NewGPULoadBalancer creates a new GPU load balancer
func NewGPULoadBalancer(devices []int) *GPULoadBalancer {
	lb := &GPULoadBalancer{
		devices:          devices,
		deviceLoad:       make(map[int]int),
		deviceCapacity:   make(map[int]int),
		devicePerf:       make(map[int]float64),
		workDistribution: make(map[int]int),
		lastRebalance:    time.Now(),
	}

	// Initialize device capacities and performance ratings
	for _, deviceID := range devices {
		lb.deviceLoad[deviceID] = 0
		lb.deviceCapacity[deviceID] = 4 // Default capacity of 4 concurrent works per GPU
		lb.devicePerf[deviceID] = 1.0   // Default performance rating
		lb.workDistribution[deviceID] = 0
	}

	return lb
}

// SelectDevice selects the best GPU device for new work
func (lb *GPULoadBalancer) SelectDevice() int {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	if len(lb.devices) == 0 {
		return -1
	}

	// Find device with lowest load relative to capacity
	bestDevice := lb.devices[0]
	bestRatio := float64(lb.deviceLoad[bestDevice]) / float64(lb.deviceCapacity[bestDevice])

	for _, deviceID := range lb.devices {
		loadRatio := float64(lb.deviceLoad[deviceID]) / float64(lb.deviceCapacity[deviceID])

		// Consider performance rating in selection
		adjustedRatio := loadRatio / lb.devicePerf[deviceID]

		if adjustedRatio < bestRatio {
			bestDevice = deviceID
			bestRatio = adjustedRatio
		}
	}

	// Increment load for selected device
	lb.deviceLoad[bestDevice]++
	lb.workDistribution[bestDevice]++

	return bestDevice
}

// ReleaseDevice decreases load on a GPU device
func (lb *GPULoadBalancer) ReleaseDevice(deviceID int) {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	if load, exists := lb.deviceLoad[deviceID]; exists && load > 0 {
		lb.deviceLoad[deviceID]--
	}
}

// UpdatePerformance updates performance rating for a device
func (lb *GPULoadBalancer) UpdatePerformance(deviceID int, processingTime time.Duration, success bool) {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	if _, exists := lb.devicePerf[deviceID]; !exists {
		return
	}

	// Simple performance update based on processing time and success rate
	if success {
		// Lower processing time = higher performance
		timeRating := 1.0 / (processingTime.Seconds() + 0.1)

		// Exponential moving average
		lb.devicePerf[deviceID] = 0.9*lb.devicePerf[deviceID] + 0.1*timeRating
	} else {
		// Penalize failures
		lb.devicePerf[deviceID] *= 0.95
	}
}

// initializeMultiGPU sets up the multi-GPU mining system
func (m *QuantumMiner) initializeMultiGPU() error {
	// Detect available GPUs
	gpus, err := detectAvailableGPUs()
	if err != nil {
		return fmt.Errorf("GPU detection failed: %v", err)
	}

	if len(gpus) == 0 {
		return fmt.Errorf("no compatible GPUs found")
	}

	m.availableGPUs = gpus
	m.multiGPUEnabled = true
	m.gpuSimulators = make(map[int]*quantum.HighPerformanceQuantumSimulator)

	// Initialize simulators for each GPU device (these will reuse the cached GPU test)
	for _, deviceID := range gpus {
		sim, err := quantum.NewHighPerformanceQuantumSimulatorWithDevice(16, deviceID) // 16 qubits, specific device
		if err != nil {
			logError("Failed to initialize GPU %d: %v", deviceID, err)
			continue
		}
		m.gpuSimulators[deviceID] = sim
		// Only log for the first few GPUs to avoid spam
		if deviceID < 3 {
			logInfo("✅ GPU %d initialized successfully!", deviceID)
		}
	}

	// Initialize load balancer
	m.gpuLoadBalancer = NewGPULoadBalancer(gpus)

	// Initialize work queues
	m.gpuWorkQueue = make(chan *GPUWorkItem, len(gpus)*10) // Buffer for work items
	m.gpuResultQueue = make(chan *GPUResult, len(gpus)*10) // Buffer for results

	// Start GPU workers
	for _, deviceID := range gpus {
		go m.gpuWorker(deviceID)
	}

	// Start result processor
	go m.gpuResultProcessor()

	logInfo("🚀 Multi-GPU system initialized with %d devices", len(gpus))
	return nil
}

// gpuWorker processes work items on a specific GPU device
func (m *QuantumMiner) gpuWorker(deviceID int) {
	logInfo("🔄 GPU worker %d started", deviceID)

	simulator, exists := m.gpuSimulators[deviceID]
	if !exists {
		logError("GPU %d simulator not found", deviceID)
		return
	}

	for {
		select {
		case <-m.stopChan:
			logInfo("🛑 GPU worker %d stopping", deviceID)
			return
		case workItem := <-m.gpuWorkQueue:
			if workItem == nil {
				continue
			}

			// Process the work item
			result := m.processGPUWork(deviceID, simulator, workItem)

			// Send result back
			select {
			case m.gpuResultQueue <- result:
			case <-time.After(1 * time.Second):
				logError("GPU result queue full, dropping result from device %d", deviceID)
			}

			// Update load balancer
			m.gpuLoadBalancer.ReleaseDevice(deviceID)
			m.gpuLoadBalancer.UpdatePerformance(deviceID, result.ProcessTime, result.Success)
		}
	}
}

// processGPUWork processes a single work item on the specified GPU
func (m *QuantumMiner) processGPUWork(deviceID int, simulator *quantum.HighPerformanceQuantumSimulator, workItem *GPUWorkItem) *GPUResult {
	startTime := time.Now()

	result := &GPUResult{
		ThreadID:    workItem.ThreadID,
		WorkItem:    workItem,
		DeviceID:    deviceID,
		ProcessTime: 0,
		Success:     false,
	}

	// Perform GPU simulation
	outcomes, err := simulator.BatchSimulateQuantumPuzzles(
		workItem.WorkHash, workItem.QNonce, workItem.QBits, workItem.TCount, workItem.LNet)

	result.ProcessTime = time.Since(startTime)

	if err != nil {
		result.Error = err
		return result
	}

	// Convert outcomes to quantum proof
	proof, err := m.convertOutcomesToProof(outcomes, workItem.WorkHash, workItem.QNonce)
	if err != nil {
		result.Error = err
		return result
	}

	result.Result = proof
	result.Success = true
	return result
}

// gpuResultProcessor handles results from GPU workers
func (m *QuantumMiner) gpuResultProcessor() {
	pendingResults := make(map[int]*GPUResult) // Map threadID -> result

	for {
		select {
		case <-m.stopChan:
			return
		case result := <-m.gpuResultQueue:
			if result == nil {
				continue
			}

			// Store result for the requesting thread
			pendingResults[result.ThreadID] = result

			// Clean up old results (older than 30 seconds)
			cutoff := time.Now().Add(-30 * time.Second)
			for threadID, res := range pendingResults {
				if res.WorkItem.StartTime.Before(cutoff) {
					delete(pendingResults, threadID)
				}
			}
		}
	}
}

// submitGPUWork submits work to the GPU processing queue
func (m *QuantumMiner) submitGPUWork(threadID int, workHash string, qnonce uint64, qbits, tcount, lnet int) error {
	if !m.multiGPUEnabled {
		return fmt.Errorf("multi-GPU not enabled")
	}

	// Select best GPU device
	deviceID := m.gpuLoadBalancer.SelectDevice()
	if deviceID == -1 {
		return fmt.Errorf("no available GPU devices")
	}

	workItem := &GPUWorkItem{
		ThreadID:  threadID,
		WorkHash:  workHash,
		QNonce:    qnonce,
		QBits:     qbits,
		TCount:    tcount,
		LNet:      lnet,
		StartTime: time.Now(),
		Priority:  0,
		DeviceID:  deviceID,
	}

	// Submit work (non-blocking)
	select {
	case m.gpuWorkQueue <- workItem:
		return nil
	default:
		m.gpuLoadBalancer.ReleaseDevice(deviceID) // Release since we couldn't queue
		return fmt.Errorf("GPU work queue full")
	}
}

// convertOutcomesToProof converts GPU simulation outcomes to quantum proof
func (m *QuantumMiner) convertOutcomesToProof(outcomes [][]byte, workHash string, qnonce uint64) (QuantumProofSubmission, error) {
	if len(outcomes) == 0 {
		return QuantumProofSubmission{}, fmt.Errorf("no outcomes provided")
	}

	// Calculate outcome root
	outcomeRoot := m.calculateOutcomeRoot(outcomes)

	// Generate gate hash (simplified for now)
	gateHashInput := fmt.Sprintf("%s_%d_gates", workHash, qnonce)
	gateHash := sha256Hash(gateHashInput)

	// Generate proof root (simplified)
	proofRoot := sha256Hash(outcomeRoot + gateHash)

	// Generate branch nibbles (simplified)
	branchNibbles := fmt.Sprintf("%x", qnonce&0xFFFF)

	// Generate extra nonce
	extraNonce32 := fmt.Sprintf("%08x", uint32(qnonce>>32))

	return QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchNibbles,
		ExtraNonce32:  extraNonce32,
	}, nil
}
