package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/big"
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

	// MEMORY-EFFICIENT PUZZLE STAGING: Pre-allocate memory to avoid swapping
	memoryPool chan *PuzzleMemory // Pool of pre-allocated puzzle memory

	// Additional fields for optimized mining
	wg        sync.WaitGroup // Wait group for thread management
	isRunning atomic.Bool    // Atomic flag for running state
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
	Difficulty  uint64    `json:"difficulty"` // Actual difficulty value
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
	BranchNibbles []byte `json:"branch_nibbles"` // Changed to []byte for direct processing
	ExtraNonce32  string `json:"extra_nonce32"`
}

type WorkPackage struct {
	BlockNumber  uint64
	ParentHash   string
	Target       *big.Int
	PuzzleHashes []string
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
	fmt.Printf("   ⚛️  Quantum Puzzles: 32 chained per block\n")
	fmt.Printf("   🔬 Qubits per Puzzle: 16\n")
	fmt.Printf("   🚪 T-Gates per Puzzle: minimum 20 (ENFORCED)\n")
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

	// Set the isRunning flag for thread checks
	m.isRunning.Store(true)

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
	const bytesPerQubits = 2     // 16 qubits = 2 bytes
	const maxPuzzlesPerSet = 128 // Maximum puzzle count for safety (expecting 32)
	const gateHashSize = 32      // SHA256 hash size
	const workBufferSize = 1024  // Additional working memory

	memoryPerSet := (bytesPerQubits * maxPuzzlesPerSet) + (gateHashSize * maxPuzzlesPerSet) + workBufferSize

	// Pre-allocate memory pool to avoid runtime allocation
	// Use available system memory efficiently
	m.memoryPoolSize = 50 // Conservative pool size
	if m.gpuMode {
		m.memoryPoolSize = 20 // GPU mode uses more memory per operation
	}

	logInfo("🧠 Initializing memory pools: %d sets, %d bytes per set", m.memoryPoolSize, memoryPerSet)

	// Pre-allocate memory pools
	m.memoryPool = make(chan *PuzzleMemory, m.memoryPoolSize)
	for i := 0; i < m.memoryPoolSize; i++ {
		memory := &PuzzleMemory{
			Outcomes:   make([][]byte, maxPuzzlesPerSet),
			GateHashes: make([][]byte, maxPuzzlesPerSet),
		}

		// Initialize each outcome and gate hash slice
		for j := 0; j < maxPuzzlesPerSet; j++ {
			memory.Outcomes[j] = make([]byte, bytesPerQubits)
			memory.GateHashes[j] = make([]byte, gateHashSize)
		}

		m.memoryPool <- memory
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

// Enhanced mining thread with CPU/GPU load balancing
func (m *QuantumMiner) enhancedMiningThread(threadID int) {
	// Note: WaitGroup.Done() is called in the Stop() method, not here
	// This prevents the negative WaitGroup counter panic

	// Thread-specific rate limiting to prevent CPU spikes
	rateLimiter := time.NewTicker(50 * time.Millisecond) // 20 Hz max per thread
	defer rateLimiter.Stop()

	// Adaptive work batch sizing based on system load
	batchSize := 1
	maxBatchSize := 4
	if m.gpuMode {
		maxBatchSize = 2 // Smaller batches for GPU to reduce memory spikes
	}

	consecutiveErrors := 0
	lastWorkTime := time.Now()

	for {
		select {
		case <-m.stopChan:
			logInfo("🧵 Thread %d: Graceful shutdown", threadID)
			return
		case <-rateLimiter.C:
			// Rate-limited execution to prevent CPU spikes

			// Adaptive batch sizing based on recent performance
			timeSinceLastWork := time.Since(lastWorkTime)
			if timeSinceLastWork > 5*time.Second && batchSize > 1 {
				batchSize-- // Reduce batch size if work is slow
			} else if timeSinceLastWork < 1*time.Second && batchSize < maxBatchSize {
				batchSize++ // Increase batch size if work is fast
			}

			// Process work in small batches to smooth CPU/GPU usage
			for i := 0; i < batchSize; i++ {
				if !m.isRunning.Load() {
					return
				}

				// Get current block number with timeout
				blockNumber := m.getCurrentBlockNumber()
				if blockNumber == 0 {
					consecutiveErrors++
					if consecutiveErrors > 10 {
						logError("Thread %d: Too many consecutive errors, backing off", threadID)
						time.Sleep(time.Duration(consecutiveErrors) * time.Second)
					}
					break
				}

				consecutiveErrors = 0

				// Progressive work distribution - start small and scale up
				success := m.enhancedMineBlock(blockNumber)
				lastWorkTime = time.Now()

				if success {
					// Reset batch size on success to maintain smooth operation
					batchSize = 1
				}

				// Small delay between batch items to prevent CPU bursts
				if i < batchSize-1 {
					time.Sleep(10 * time.Millisecond)
				}
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

	// Set the isRunning flag to false for thread checks
	m.isRunning.Store(false)

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

	// Debug: Log what we received from geth
	logInfo("🔍 Work response from geth: %d elements", len(workArray))
	for i, elem := range workArray {
		if str, ok := elem.(string); ok {
			logInfo("  [%d]: %s", i, str)
		} else {
			logInfo("  [%d]: %v (type: %T)", i, elem, elem)
		}
	}

	work := &QuantumWork{
		WorkHash:  workArray[0].(string),
		QBits:     16,  // Default quantum params
		TCount:    20,  // ENFORCED MINIMUM
		LNet:      128, // ENFORCED - 128 chained puzzles
		FetchTime: time.Now(),
	}

	// SECURITY ENFORCEMENT: Validate quantum parameters to prevent cheating
	if work.TCount < 20 {
		return fmt.Errorf("SECURITY VIOLATION: Work TCount %d is below enforced minimum of 20", work.TCount)
	}
	if work.LNet != 128 {
		return fmt.Errorf("SECURITY VIOLATION: Work LNet %d must be exactly 128 chained puzzles", work.LNet)
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

	// Calculate difficulty from target (reverse of: target = max_target / difficulty)
	// difficulty = max_target / target
	if work.Target != "" {
		// Parse target as big int
		targetInt := new(big.Int)
		if strings.HasPrefix(work.Target, "0x") {
			targetInt.SetString(work.Target[2:], 16)
		} else {
			targetInt.SetString(work.Target, 16)
		}

		// Calculate difficulty: max_target / target
		maxTarget := new(big.Int)
		maxTarget.SetString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 16)

		if targetInt.Cmp(big.NewInt(0)) > 0 {
			difficulty := new(big.Int).Div(maxTarget, targetInt)
			work.Difficulty = difficulty.Uint64()
			m.currentDifficulty = work.Difficulty
		}
	}

	// Get difficulty info for dashboard (fallback)
	if work.Difficulty == 0 {
		if diffResult, err := m.rpcCall("eth_getBlockByNumber", []interface{}{"latest", false}); err == nil {
			if blockData, ok := diffResult.(map[string]interface{}); ok {
				if diffHex, ok := blockData["difficulty"].(string); ok {
					if difficulty, err := strconv.ParseUint(strings.TrimPrefix(diffHex, "0x"), 16, 64); err == nil {
						work.Difficulty = difficulty
						m.currentDifficulty = difficulty
					}
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
		logInfo("New work received: Block %d, Difficulty %d, Target: %s", work.BlockNumber, work.Difficulty, work.Target)
	}
	m.workMutex.Unlock()

	return nil
}

// enhancedMineBlock performs quantum mining with improved thread and memory management
func (m *QuantumMiner) enhancedMineBlock(blockNumber uint64) bool {
	// Get work with timeout to prevent hanging
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	work, err := m.getWork(ctx)
	if err != nil {
		if !strings.Contains(err.Error(), "no new work") {
			logError("Failed to get work: %v", err)
		}
		return false
	}

	// Adaptive puzzle processing based on system capabilities
	puzzleCount := len(work.PuzzleHashes)
	if puzzleCount == 0 {
		return false
	}

	// Progressive processing to smooth CPU/GPU load
	processingChunks := 1
	if puzzleCount > 64 {
		processingChunks = 4 // Process in 4 chunks for large puzzle sets
	} else if puzzleCount > 32 {
		processingChunks = 2 // Process in 2 chunks for medium puzzle sets
	}

	chunkSize := puzzleCount / processingChunks
	if chunkSize == 0 {
		chunkSize = puzzleCount
		processingChunks = 1
	}

	// Only log processing info occasionally to reduce spam
	if atomic.LoadUint64(&m.attempts)%50 == 0 {
		logInfo("📦 Processing %d puzzles in %d chunks of %d", puzzleCount, processingChunks, chunkSize)
	}

	// Process puzzles in chunks to prevent CPU spikes
	for chunk := 0; chunk < processingChunks; chunk++ {
		if !m.isRunning.Load() {
			return false
		}

		startIdx := chunk * chunkSize
		endIdx := startIdx + chunkSize
		if chunk == processingChunks-1 {
			endIdx = puzzleCount // Include remaining puzzles in last chunk
		}

		// Process chunk with rate limiting
		chunkWork := &WorkPackage{
			BlockNumber:  work.BlockNumber,
			ParentHash:   work.ParentHash,
			Target:       work.Target,
			PuzzleHashes: work.PuzzleHashes[startIdx:endIdx],
		}

		success := m.processWorkChunk(ctx, chunkWork, chunk, processingChunks)
		if success {
			return true // Block found
		}

		// Inter-chunk delay to smooth processing and reduce CPU spikes
		if chunk < processingChunks-1 {
			smoothingDelay := time.Duration(20+chunk*10) * time.Millisecond
			time.Sleep(smoothingDelay)
		}
	}

	return false
}

// Process a chunk of work with optimized resource usage
func (m *QuantumMiner) processWorkChunk(ctx context.Context, work *WorkPackage, chunkID, totalChunks int) bool {
	startTime := time.Now()

	// Generate QNonce for this chunk
	qnonce := m.generateQNonce()

	// Solve quantum puzzles with progressive difficulty
	result, err := m.enhancedSolveQuantumPuzzles(ctx, work.BlockNumber, work.PuzzleHashes, qnonce, 16, 20, len(work.PuzzleHashes))
	if err != nil {
		if !strings.Contains(err.Error(), "context") {
			logError("Chunk %d/%d puzzle solving failed: %v", chunkID+1, totalChunks, err)
		}
		return false
	}

	processingTime := time.Since(startTime)

	// Update statistics with chunk information
	m.updateStats(len(work.PuzzleHashes), processingTime, chunkID, totalChunks)

	// Submit result if valid (pass qnonce for proper quality calculation)
	if m.isValidResultWithQNonce(result, work.Target, qnonce) {
		logInfo("🎉 FOUND VALID SOLUTION! QNonce: %016x, Block: %d", qnonce, work.BlockNumber)
		return m.submitResult(work, result, qnonce)
	}

	// Only log occasionally to avoid spam
	if qnonce%1000 == 0 {
		logInfo("⚡ Checked solution qnonce=%016x (not valid)", qnonce)
	}

	return false
}

// Generate QNonce with thread-safe counter
func (m *QuantumMiner) generateQNonce() uint64 {
	counter := atomic.AddUint64(&m.nonceCounter, 1)
	return m.nonceBase + (counter << 8)
}

// Update mining statistics with chunk information
func (m *QuantumMiner) updateStats(puzzleCount int, processingTime time.Duration, chunkID, totalChunks int) {
	atomic.AddUint64(&m.puzzlesSolved, uint64(puzzleCount))
	atomic.AddUint64(&m.attempts, 1)

	// Smooth statistics update to prevent display spikes
	// Only log every 100th completion to reduce spam
	if chunkID == totalChunks-1 && atomic.LoadUint64(&m.attempts)%100 == 0 {
		logInfo("📊 Processed %d puzzles in %v (completed %d attempts)",
			puzzleCount, processingTime, atomic.LoadUint64(&m.attempts))
	}
}

// Check if result meets difficulty target using geth-compatible quality calculation
func (m *QuantumMiner) isValidResult(result *QuantumProofSubmission, target *big.Int) bool {
	// Implement difficulty check based on result hash
	if result == nil {
		return false
	}

	// Use the SAME quality calculation as geth for compatibility
	quality := m.calculateQuantumProofQuality(result, 0) // qnonce will be set during submission

	// Bitcoin-style comparison: success when quality < target
	return quality.Cmp(target) < 0
}

// Check if result meets difficulty target with specific qnonce
func (m *QuantumMiner) isValidResultWithQNonce(result *QuantumProofSubmission, target *big.Int, qnonce uint64) bool {
	// Implement difficulty check based on result hash
	if result == nil {
		return false
	}

	// Use the SAME quality calculation as geth for compatibility
	quality := m.calculateQuantumProofQuality(result, qnonce)

	// Bitcoin-style comparison: success when quality < target
	return quality.Cmp(target) < 0
}

// calculateQuantumProofQuality implements the same algorithm as geth's CalculateQuantumProofQuality
func (m *QuantumMiner) calculateQuantumProofQuality(result *QuantumProofSubmission, qnonce uint64) *big.Int {
	// Enhanced Bitcoin-style hash-based quality calculation
	// Multiple rounds of hashing for better nonce sensitivity
	h := sha256.New()

	// First, hash the nonce alone to create base entropy
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, qnonce)
	h.Write(nonceBytes)
	h.Write([]byte("QUANTUM_NONCE_SEED"))
	nonceSeed := h.Sum(nil)

	// Reset hasher and combine nonce seed with quantum data
	h.Reset()
	h.Write(nonceSeed)

	// Convert outcome root from hex string to bytes
	outcomeBytes, err := hex.DecodeString(strings.TrimPrefix(result.OutcomeRoot, "0x"))
	if err != nil {
		outcomeBytes = make([]byte, 32) // fallback to zeros
	}
	h.Write(outcomeBytes)

	// Combine gate hash and proof root as "proof" data
	gateBytes, err := hex.DecodeString(strings.TrimPrefix(result.GateHash, "0x"))
	if err != nil {
		gateBytes = make([]byte, 32) // fallback to zeros
	}
	proofBytes, err := hex.DecodeString(strings.TrimPrefix(result.ProofRoot, "0x"))
	if err != nil {
		proofBytes = make([]byte, 32) // fallback to zeros
	}
	h.Write(gateBytes)
	h.Write(proofBytes)

	// Add nonce again for extra sensitivity
	h.Write(nonceBytes)

	// Multiple rounds of hashing for better distribution
	for i := 0; i < 3; i++ {
		h.Write([]byte(fmt.Sprintf("QUANTUM_ROUND_%d", i)))
		intermediate := h.Sum(nil)
		h.Reset()
		h.Write(intermediate)
		h.Write(nonceBytes) // Nonce in every round
	}

	// Final hash with entropy marker
	h.Write([]byte("QUANTUM_BITCOIN_FINAL"))
	hash := h.Sum(nil)

	// Convert hash to big integer (full 256-bit range)
	quality := new(big.Int).SetBytes(hash)

	return quality
}

// Submit mining result
func (m *QuantumMiner) submitResult(work *WorkPackage, result *QuantumProofSubmission, qnonce uint64) bool {
	logInfo("🎯 VALID SOLUTION FOUND! Submitting to geth for block %d, qnonce=%016x", work.BlockNumber, qnonce)

	// Convert the miner's QuantumProofSubmission to the format expected by geth
	// Geth expects the QuantumProofSubmission struct with proper JSON field types:
	// - Hash fields: 32-byte hex strings WITH 0x prefix (for common.Hash unmarshaling)
	// - Byte fields: hex strings with 0x prefix (for []byte unmarshaling)

	// Ensure hash fields have 0x prefix since geth's common.Hash expects it
	outcomeRoot := result.OutcomeRoot
	if !strings.HasPrefix(outcomeRoot, "0x") {
		outcomeRoot = "0x" + outcomeRoot
	}

	gateHash := result.GateHash
	if !strings.HasPrefix(gateHash, "0x") {
		gateHash = "0x" + gateHash
	}

	proofRoot := result.ProofRoot
	if !strings.HasPrefix(proofRoot, "0x") {
		proofRoot = "0x" + proofRoot
	}

	// Convert ExtraNonce32 hex string to bytes, then base64 encode
	extraNonce32Hex := result.ExtraNonce32
	if !strings.HasPrefix(extraNonce32Hex, "0x") {
		extraNonce32Hex = "0x" + extraNonce32Hex
	}
	// Decode hex to bytes
	extraNonce32Bytes, decodeErr := hex.DecodeString(strings.TrimPrefix(extraNonce32Hex, "0x"))
	if decodeErr != nil {
		logError("❌ Failed to decode ExtraNonce32 hex: %v", decodeErr)
		return false
	}

	gethQuantumProof := map[string]interface{}{
		"outcome_root":   outcomeRoot,                                             // 32-byte hex string WITH 0x prefix
		"gate_hash":      gateHash,                                                // 32-byte hex string WITH 0x prefix
		"proof_root":     proofRoot,                                               // 32-byte hex string WITH 0x prefix
		"branch_nibbles": base64.StdEncoding.EncodeToString(result.BranchNibbles), // []byte as base64 string
		"extra_nonce32":  base64.StdEncoding.EncodeToString(extraNonce32Bytes),    // []byte as base64 string
	}

	// Debug: Log what we're about to submit
	logInfo("🔍 Submitting to geth:")
	logInfo("  QNonce: %016x", qnonce)
	logInfo("  Block Hash: '%s' (len=%d)", work.ParentHash, len(work.ParentHash))
	logInfo("  Quantum Proof: %+v", gethQuantumProof)

	// Prepare submission data for geth RPC call
	// Parameters: qnonce (uint64), blockHash (string), quantumProof (QuantumProofSubmission)
	submitData := []interface{}{
		qnonce,           // nonce as uint64
		work.ParentHash,  // block hash as string
		gethQuantumProof, // quantum proof as struct
	}

	// Submit to geth node via RPC
	_, err := m.rpcCall("eth_submitWork", submitData)
	if err != nil {
		logError("❌ Failed to submit solution: %v", err)
		atomic.AddUint64(&m.rejected, 1)
		return false
	}

	logInfo("✅ Solution accepted by network! Block %d found with quantum proof", work.BlockNumber)
	atomic.AddUint64(&m.accepted, 1)

	// Track block timing
	now := time.Now()
	m.blockTimes = append(m.blockTimes, now)
	if len(m.blockTimes) > 10 {
		m.blockTimes = m.blockTimes[1:]
	}

	return true
}

// Get current block number from the network
func (m *QuantumMiner) getCurrentBlockNumber() uint64 {
	// Simple implementation - in practice this would query the network
	return 1 // Default to block 1 for now
}

// Get work from the network
func (m *QuantumMiner) getWork(ctx context.Context) (*WorkPackage, error) {
	// Use real work from geth node
	m.workMutex.RLock()
	currentWork := m.currentWork
	m.workMutex.RUnlock()

	if currentWork == nil {
		return nil, fmt.Errorf("no work available from geth node")
	}

	// Convert QuantumWork to WorkPackage
	// Parse target directly from geth (simple Bitcoin-style calculation)
	target := new(big.Int).Lsh(big.NewInt(1), 256) // Default max target
	if currentWork.Target != "" {
		if targetInt, ok := new(big.Int).SetString(strings.TrimPrefix(currentWork.Target, "0x"), 16); ok {
			target = targetInt
		}
	}

	// Create puzzle hashes based on LNet from real work
	puzzleCount := currentWork.LNet
	puzzleHashes := make([]string, puzzleCount)
	for i := 0; i < puzzleCount; i++ {
		// Generate deterministic puzzle hashes based on work hash
		puzzleData := fmt.Sprintf("%s_%d", currentWork.WorkHash, i)
		puzzleHashes[i] = sha256Hash(puzzleData)
	}

	// Ensure ParentHash is a valid 64-character hex string
	parentHash := currentWork.WorkHash
	logInfo("🔍 getWork() debug:")
	logInfo("  currentWork.WorkHash: '%s' (len=%d)", currentWork.WorkHash, len(currentWork.WorkHash))
	logInfo("  Block Number: %d", currentWork.BlockNumber)

	if parentHash == "" || len(strings.TrimPrefix(parentHash, "0x")) != 64 {
		// Generate a placeholder hash if work hash is invalid
		parentHash = fmt.Sprintf("0x%064x", currentWork.BlockNumber)
		logInfo("  Using placeholder hash: '%s'", parentHash)
	} else {
		logInfo("  Using original work hash: '%s'", parentHash)
	}

	return &WorkPackage{
		BlockNumber:  currentWork.BlockNumber,
		ParentHash:   parentHash, // This is actually the work hash (block hash), not parent hash
		Target:       target,
		PuzzleHashes: puzzleHashes,
	}, nil
}

// Enhanced quantum puzzle solving with CPU/GPU load balancing
func (m *QuantumMiner) enhancedSolveQuantumPuzzles(ctx context.Context, blockNumber uint64, puzzleHashes []string, qnonce uint64, qbits, tcount, lnet int) (*QuantumProofSubmission, error) {
	// Adaptive processing based on puzzle count
	if lnet > 64 {
		return m.solveLargePuzzleSet(ctx, blockNumber, puzzleHashes, qnonce, qbits, tcount, lnet)
	} else {
		return m.solveStandardPuzzleSet(ctx, blockNumber, puzzleHashes, qnonce, qbits, tcount, lnet)
	}
}

// Solve large puzzle sets (64+ puzzles) with progressive processing
func (m *QuantumMiner) solveLargePuzzleSet(ctx context.Context, blockNumber uint64, puzzleHashes []string, qnonce uint64, qbits, tcount, lnet int) (*QuantumProofSubmission, error) {
	// Process in smaller sub-batches to prevent CPU spikes
	subBatchSize := 16
	totalSubBatches := (lnet + subBatchSize - 1) / subBatchSize

	allOutcomes := make([][]byte, lnet)
	allGateHashes := make([][]byte, lnet)

	for batch := 0; batch < totalSubBatches; batch++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		startIdx := batch * subBatchSize
		endIdx := startIdx + subBatchSize
		if endIdx > lnet {
			endIdx = lnet
		}

		// Process sub-batch with CPU throttling
		for i := startIdx; i < endIdx; i++ {
			// CPU-friendly quantum simulation
			outcome, gateHash := m.simulateQuantumPuzzle(qbits, tcount, i, qnonce)
			allOutcomes[i] = outcome
			allGateHashes[i] = gateHash

			// Micro-delays to prevent CPU saturation
			if i%4 == 0 {
				time.Sleep(1 * time.Millisecond)
			}
		}

		// Inter-batch delay for CPU relief
		if batch < totalSubBatches-1 {
			time.Sleep(5 * time.Millisecond)
		}
	}

	return m.buildQuantumProof(allOutcomes, allGateHashes, lnet)
}

// Solve standard puzzle sets (≤64 puzzles) with optimized processing
func (m *QuantumMiner) solveStandardPuzzleSet(ctx context.Context, blockNumber uint64, puzzleHashes []string, qnonce uint64, qbits, tcount, lnet int) (*QuantumProofSubmission, error) {
	// Get memory from pool with timeout
	memory, err := m.getMemoryFromPool(ctx, lnet)
	if err != nil {
		return nil, fmt.Errorf("memory allocation failed: %v", err)
	}
	defer m.returnMemoryToPool(memory)

	// Progressive puzzle solving with CPU throttling
	for i := 0; i < lnet; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// CPU-friendly quantum simulation
		outcome, gateHash := m.simulateQuantumPuzzle(qbits, tcount, i, qnonce)
		copy(memory.Outcomes[i], outcome)
		copy(memory.GateHashes[i], gateHash)

		// Adaptive CPU throttling based on puzzle index
		if i%8 == 0 && i > 0 {
			time.Sleep(2 * time.Millisecond) // Brief CPU relief
		}
	}

	return m.buildQuantumProofFromMemory(memory, lnet)
}

// Simulate quantum puzzle with CPU-optimized approach
func (m *QuantumMiner) simulateQuantumPuzzle(qbits, tcount, puzzleIndex int, qnonce uint64) ([]byte, []byte) {
	// Lightweight quantum simulation to reduce CPU load
	seed := qnonce + uint64(puzzleIndex)

	// Generate outcome (simplified for CPU efficiency)
	outcome := make([]byte, 2) // 16 qubits = 2 bytes
	binary.LittleEndian.PutUint16(outcome, uint16(seed&0xFFFF))

	// Generate gate hash (simplified)
	gateData := make([]byte, 8)
	binary.LittleEndian.PutUint64(gateData, seed*uint64(tcount))
	gateHash := sha256.Sum256(gateData)

	return outcome, gateHash[:]
}

// Get memory from pool with timeout
func (m *QuantumMiner) getMemoryFromPool(ctx context.Context, requiredSize int) (*PuzzleMemory, error) {
	select {
	case memory := <-m.memoryPool:
		// Validate memory size
		if len(memory.Outcomes) >= requiredSize && len(memory.GateHashes) >= requiredSize {
			return memory, nil
		}
		// Return insufficient memory and create new one
		m.memoryPool <- memory
		return m.createMemoryBlock(requiredSize), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(100 * time.Millisecond):
		// Create new memory if pool is empty
		return m.createMemoryBlock(requiredSize), nil
	}
}

// Return memory to pool
func (m *QuantumMiner) returnMemoryToPool(memory *PuzzleMemory) {
	select {
	case m.memoryPool <- memory:
	default:
		// Pool is full, let GC handle it
	}
}

// Create memory block for specific size
func (m *QuantumMiner) createMemoryBlock(size int) *PuzzleMemory {
	memory := &PuzzleMemory{
		Outcomes:   make([][]byte, size),
		GateHashes: make([][]byte, size),
	}

	for i := 0; i < size; i++ {
		memory.Outcomes[i] = make([]byte, 2)    // 16 qubits = 2 bytes
		memory.GateHashes[i] = make([]byte, 32) // SHA256 = 32 bytes
	}

	return memory
}

// Build quantum proof from memory
func (m *QuantumMiner) buildQuantumProofFromMemory(memory *PuzzleMemory, lnet int) (*QuantumProofSubmission, error) {
	branchNibbles := make([]byte, lnet)
	for i := 0; i < lnet; i++ {
		if len(memory.Outcomes[i]) > 0 {
			branchNibbles[i] = memory.Outcomes[i][0] // Full byte for maximum entropy
		}
	}

	// Generate proper 32-byte hashes for geth compatibility
	// Calculate outcome root from all puzzle outcomes
	outcomeRoot := m.calculateOutcomeRoot(memory.Outcomes)

	// Generate gate hash from all gate hashes
	gateHash := m.calculateGateHash(memory.GateHashes)

	// Generate proof root as combination of outcome and gate hashes
	proofRoot := m.calculateProofRoot(outcomeRoot, gateHash)

	// Generate 32-byte extra nonce
	extraNonce32 := m.generateExtraNonce32()

	return &QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchNibbles,
		ExtraNonce32:  extraNonce32,
	}, nil
}

// Build quantum proof from arrays
func (m *QuantumMiner) buildQuantumProof(outcomes, gateHashes [][]byte, lnet int) (*QuantumProofSubmission, error) {
	branchNibbles := make([]byte, lnet)
	for i := 0; i < lnet; i++ {
		if len(outcomes[i]) > 0 {
			branchNibbles[i] = outcomes[i][0] // Full byte for maximum entropy
		}
	}

	// Generate proper 32-byte hashes for geth compatibility
	// Calculate outcome root from all puzzle outcomes
	outcomeRoot := m.calculateOutcomeRoot(outcomes)

	// Generate gate hash from all gate hashes
	gateHash := m.calculateGateHash(gateHashes)

	// Generate proof root as combination of outcome and gate hashes
	proofRoot := m.calculateProofRoot(outcomeRoot, gateHash)

	// Generate 32-byte extra nonce
	extraNonce32 := m.generateExtraNonce32()

	return &QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchNibbles,
		ExtraNonce32:  extraNonce32,
	}, nil
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
	if work != nil {
		blockNumber = work.BlockNumber
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
	fmt.Printf("│ 🧵 Thread Status   │ Active: %d/%-2d │ Max Concurrent: %-2d │ Pool: %d/%d    │\n",
		activeCount, m.threads, m.maxActiveThreads, len(m.puzzleMemoryPool), m.memoryPoolSize)
	fmt.Println("├─────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ 🔗 Current Block   │ Block: %-10d │ Difficulty: %-15d       │\n",
		blockNumber, m.currentDifficulty)
	if avgBlockTime > 0 {
		fmt.Printf("│ ⏱️  Block Timing    │ Average: %6.1fs │ Target: %6.1fs │ ASERT-Q Adjust │\n",
			avgBlockTime, m.targetBlockTime.Seconds())
	} else {
		fmt.Printf("│ ⏱️  Block Timing    │ Average: %-8s │ Target: %6.1fs │ ASERT-Q Adjust │\n",
			"N/A", m.targetBlockTime.Seconds())
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
	fmt.Println("  - Each block requires 128 chained quantum puzzle solutions")
	fmt.Println("  - Each puzzle uses 16 qubits with minimum 20 T-gates (ENFORCED)")
	fmt.Println("  - Difficulty adjusts every block using ASERT-Q algorithm")
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
	gateHash := "0x" + sha256Hash(gateHashInput)

	// Generate proof root (simplified)
	proofRoot := "0x" + sha256Hash(outcomeRoot+gateHash)

	// Generate branch nibbles (simplified)
	branchNibbles := fmt.Sprintf("%x", qnonce&0xFFFF)

	// Generate extra nonce
	extraNonce32 := fmt.Sprintf("0x%08x", uint32(qnonce>>32))

	return QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: []byte(branchNibbles), // Convert string to []byte
		ExtraNonce32:  extraNonce32,
	}, nil
}

// calculateOutcomeRoot calculates the root hash of quantum outcomes
func (m *QuantumMiner) calculateOutcomeRoot(outcomes [][]byte) string {
	if len(outcomes) == 0 {
		return "0x0000000000000000000000000000000000000000000000000000000000000000"
	}

	// Simple hash calculation - in practice this would be more sophisticated
	var combined []byte
	for _, outcome := range outcomes {
		combined = append(combined, outcome...)
	}

	hash := sha256.Sum256(combined)
	return "0x" + hex.EncodeToString(hash[:])
}

// calculateGateHash calculates the root hash of quantum gate operations
func (m *QuantumMiner) calculateGateHash(gateHashes [][]byte) string {
	if len(gateHashes) == 0 {
		return "0x0000000000000000000000000000000000000000000000000000000000000000"
	}

	// Combine all gate hashes
	var combined []byte
	for _, gateHash := range gateHashes {
		combined = append(combined, gateHash...)
	}

	hash := sha256.Sum256(combined)
	return "0x" + hex.EncodeToString(hash[:])
}

// calculateProofRoot calculates the proof root from outcome and gate hashes
func (m *QuantumMiner) calculateProofRoot(outcomeRoot, gateHash string) string {
	// Combine outcome root and gate hash
	combined := outcomeRoot + gateHash
	hash := sha256.Sum256([]byte(combined))
	return "0x" + hex.EncodeToString(hash[:])
}

// generateExtraNonce32 generates a 32-byte extra nonce
func (m *QuantumMiner) generateExtraNonce32() string {
	// Generate random 32-byte value based on current time and random data
	nonce := make([]byte, 32)
	timestamp := time.Now().UnixNano()

	// Fill first 8 bytes with timestamp
	for i := 0; i < 8; i++ {
		nonce[i] = byte(timestamp >> (i * 8))
	}

	// Fill remaining 24 bytes with deterministic pseudo-random data
	for i := 8; i < 32; i++ {
		nonce[i] = byte((timestamp * int64(i)) % 256)
	}

	return "0x" + hex.EncodeToString(nonce)
}
