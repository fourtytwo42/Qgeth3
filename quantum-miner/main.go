package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/big"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	// RESTORED: Import quantum simulation packages
	"quantum-gpu-miner/pkg/quantum"
)

const VERSION = "1.2.0-wsl2-fixed"

// Global logging to file flag
var logToFile bool = false
var logFileHandle *os.File = nil
var logBuffer chan string = make(chan string, 1000) // Buffered logging
var logShutdown chan bool = make(chan bool)

// Logging helper functions with buffered I/O
func logInfo(format string, v ...interface{}) {
	message := fmt.Sprintf(format, v...)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logMessage := timestamp + " [INFO] " + message + "\n"
	
	if logToFile && logFileHandle != nil {
		// Non-blocking write to buffer
		select {
		case logBuffer <- logMessage:
		default:
			// Buffer full, skip this log to avoid blocking
		}
	} else {
		fmt.Print(logMessage) // Use fmt.Print instead of log.Printf for speed
	}
}

func logError(format string, v ...interface{}) {
	message := fmt.Sprintf(format, v...)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logMessage := timestamp + " [ERROR] " + message + "\n"
	
	if logToFile && logFileHandle != nil {
		// Non-blocking write to buffer
		select {
		case logBuffer <- logMessage:
		default:
			// Buffer full, skip this log to avoid blocking
		}
	} else {
		fmt.Print(logMessage) // Use fmt.Print instead of log.Printf for speed
	}
}

// Background log writer to reduce I/O blocking
func startLogWriter() {
	go func() {
		ticker := time.NewTicker(100 * time.Millisecond) // Batch writes every 100ms
		defer ticker.Stop()
		
		var pendingLogs []string
		
		for {
			select {
			case logMsg := <-logBuffer:
				pendingLogs = append(pendingLogs, logMsg)
				
			case <-ticker.C:
				if len(pendingLogs) > 0 && logFileHandle != nil {
					// Write all pending logs at once
					for _, msg := range pendingLogs {
						logFileHandle.WriteString(msg)
					}
					logFileHandle.Sync() // Sync less frequently
					pendingLogs = pendingLogs[:0] // Clear slice
				}
				
			case <-logShutdown:
				// Flush remaining logs before shutdown
				if len(pendingLogs) > 0 && logFileHandle != nil {
					for _, msg := range pendingLogs {
						logFileHandle.WriteString(msg)
					}
					logFileHandle.Sync()
				}
				return
			}
		}
	}()
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

	// Statistics
	attempts      uint64
	puzzlesSolved uint64
	accepted      uint64
	rejected      uint64
	stale         uint64
	startTime     time.Time
	lastStatTime  time.Time
	lastAttempts  uint64
	lastPuzzles   uint64
	solutions     uint64

	// Performance tracking
	currentHashrate   float64
	currentPuzzleRate float64
	currentDifficulty uint64
	targetBlockTime   time.Duration

	client      *http.Client
	currentWork *QuantumWork
	workMutex   sync.RWMutex

	// Thread management
	threadStates     map[int]*ThreadState
	threadStateMux   sync.RWMutex
	activeThreads    int32
	maxActiveThreads int32

	// ENHANCED: Memory management with cleanup
	memoryPool chan *PuzzleMemory
	memoryCleanupTicker *time.Ticker
	lastGCTime  time.Time
	gcInterval  time.Duration

	// RESTORED: GPU simulation support
	gpuSimulator     *quantum.QiskitGPUSimulator
	gpuInitialized   bool
	gpuInitMutex     sync.RWMutex
	gpuCleanupTicker *time.Ticker

	// Additional fields for optimized mining
	wg        sync.WaitGroup
	isRunning atomic.Bool

	// ADDED: Resource management
	workPackagePool sync.Pool
	puzzleHashPool  sync.Pool
	circuitPool     sync.Pool
	
	// CRITICAL: Byte array pools to eliminate allocations
	outcomePool  sync.Pool
	gateDataPool sync.Pool
	
	// ULTRA-OPTIMIZED: Additional pools for zero-allocation mining
	bigIntPool     sync.Pool    // Pool big.Int objects
	bufferPool     sync.Pool    // Pool bytes.Buffer objects
	stringBuilder  sync.Pool    // Pool strings.Builder objects
	hexBufferPool  sync.Pool    // Pool hex decode buffers
	branchPool     sync.Pool    // Pool branch nibbles
	noncePool      sync.Pool    // Pool nonce byte arrays
	mapPool        sync.Pool    // Pool submission maps
	
	// PERFORMANCE: Cache frequently used values
	maxTarget      *big.Int     // Cached max target
	nonceRandom    *rand.Rand   // Fast random source
	nonceMutex     sync.Mutex   // Protect random source

	// ADDED: Channel for immediate work refresh requests
	forceWorkRefresh chan struct{}
	
	// ADDED: Global solution submission rate limiting
	solutionSubmissionMutex sync.Mutex
	lastSolutionTime        time.Time
	minSolutionInterval     time.Duration
	
	// ADDED: Quantum-geth state monitoring
	consecutiveWorkFailures int32
	lastWorkFailureTime     time.Time
	nodeRecoveryMode        bool
}

// ThreadState tracks individual thread execution state
type ThreadState struct {
	ID             int                
	Status         string             
	StartTime      time.Time          
	WorkHash       string             
	QNonce         uint64             
	LastHeartbeat  time.Time          
	AbortRequested bool               
	cancelFunc     context.CancelFunc 
}

// PuzzleMemory represents pre-allocated memory for puzzle solving
type PuzzleMemory struct {
	Outcomes   [][]byte 
	GateHashes [][]byte 
	WorkBuffer []byte   
	ID         int      
}

// ENHANCED WORK STRUCTURE: Add memory management
type QuantumWork struct {
	WorkHash    string    `json:"work_hash"`
	BlockNumber uint64    `json:"block_number"`
	Target      string    `json:"target"`
	Difficulty  uint64    `json:"difficulty"`
	QBits       int       `json:"qbits"`
	TCount      int       `json:"tcount"`
	LNet        int       `json:"lnet"`
	FetchTime   time.Time `json:"fetch_time"`
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

// FIXED: Use proper QuantumProofSubmission structure matching quantum-geth
type QuantumProofSubmission struct {
	OutcomeRoot   string `json:"outcome_root"`
	GateHash      string `json:"gate_hash"`
	ProofRoot     string `json:"proof_root"`
	BranchNibbles []byte `json:"branch_nibbles"`
	ExtraNonce32  []byte `json:"extra_nonce32"`
}

type WorkPackage struct {
	BlockNumber  uint64
	ParentHash   string
	Target       *big.Int
	PuzzleHashes []string
}

func main() {
	var coinbase = flag.String("coinbase", "0x0000000000000000000000000000000000000001", "Coinbase address for mining rewards")
	var nodeURL = flag.String("node", "http://localhost:8545", "quantum-geth node URL")
	var threads = flag.Int("threads", runtime.NumCPU(), "Number of mining threads")
	var gpuMode = flag.Bool("gpu", false, "Enable GPU acceleration")
	var gpuID = flag.Int("gpu-id", 0, "GPU device ID (for multi-GPU systems)")
	var enableLogging = flag.Bool("log", true, "Enable logging to file (default: true)")
	var help = flag.Bool("help", false, "Show help information")
	flag.Parse()

	if *help {
		showHelp()
		return
	}

	// ENABLE FILE LOGGING BY DEFAULT for performance debugging
	if *enableLogging {
		logFileName := fmt.Sprintf("quantum-miner-%d.log", time.Now().Unix())
		var err error
		logFileHandle, err = os.Create(logFileName)
		if err != nil {
			log.Printf("⚠️  Failed to create log file %s: %v", logFileName, err)
			log.Printf("   Continuing with console logging only")
		} else {
			logToFile = true
			// Start buffered log writer for performance
			startLogWriter()
			log.Printf("📝 Performance logging enabled: %s", logFileName)
			logInfo("🚀 [PERF] Quantum CPU Miner started with optimized performance logging")
			logInfo("⚙️ [PERF] Configuration: Threads=%d, GPU=%v, Node=%s", *threads, *gpuMode, *nodeURL)
		}
	}

	fmt.Printf("🚀 Quantum-Geth GPU/CPU Miner v%s (FIXED)\n", VERSION)
	fmt.Printf("⚛️  16-qubit quantum circuit mining\n")
	fmt.Printf("🔗 Bitcoin-style difficulty with quantum proof-of-work\n")
	
	if *gpuMode {
		fmt.Printf("🎮 GPU Mining: ENABLED (Device: %d)\n", *gpuID)
	} else {
		fmt.Printf("💻 CPU Mining: ENABLED\n")
	}

	fmt.Printf("\n📋 Configuration:\n")
	fmt.Printf("   💰 Coinbase: %s\n", *coinbase)
	fmt.Printf("   🌐 Node URL: %s\n", *nodeURL)
	fmt.Printf("   🧵 Threads: %d\n", *threads)
	fmt.Printf("   ⚛️  Quantum Puzzles: 128 chained per block\n")
	fmt.Printf("   🔬 Qubits per Puzzle: 16\n")
	fmt.Printf("   🚪 T-Gates per Puzzle: minimum 20\n")
	if *enableLogging {
		fmt.Printf("   📝 Performance Logging: ENABLED\n")
	}

	// Validate coinbase address
	if !isValidAddress(*coinbase) {
		log.Fatal("❌ Invalid coinbase address format")
	}

	// Create the miner instance with performance monitoring
	miner := &QuantumMiner{
		coinbase:  *coinbase,
		nodeURL:   *nodeURL,
		threads:   *threads,
		gpuMode:   *gpuMode,
		gpuID:     *gpuID,
		stopChan:  make(chan bool),
		client:    &http.Client{Timeout: 30 * time.Second},
		memoryPool: make(chan *PuzzleMemory, 20), // Pre-allocate memory pool
		threadStates: make(map[int]*ThreadState),
		
		// PERFORMANCE: Initialize caching
		maxTarget: new(big.Int).Lsh(big.NewInt(1), 256), // 2^256
		nonceRandom: rand.New(rand.NewSource(time.Now().UnixNano())),
		forceWorkRefresh: make(chan struct{}, 1),
		
		// ADDED: Initialize solution rate limiting (500ms minimum between submissions)
		minSolutionInterval: 500 * time.Millisecond,
	}

	logInfo("🔧 [PERF] Miner instance created with performance monitoring")

	// Initialize GPU if requested
	if *gpuMode {
		logInfo("🎮 [PERF] Initializing GPU mode")
		if err := miner.initializeGPU(); err != nil {
			logError("⚠️  [PERF] GPU initialization failed: %v", err)
			logError("   Falling back to CPU mining")
			miner.gpuMode = false
		}
	}

	logInfo("🔗 [PERF] Testing connection to quantum-geth node")
	// Test connection
	if err := miner.testConnection(); err != nil {
		log.Fatalf("❌ Failed to connect to quantum-geth: %v", err)
	}
	logInfo("✅ [PERF] Connection test successful")

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	logInfo("🚀 [PERF] Starting mining operation")
	// Start mining
	if err := miner.Start(); err != nil {
		log.Fatalf("❌ Failed to start mining: %v", err)
	}

	// Wait for shutdown signal
	<-sigChan
	fmt.Println("\n🛑 Shutdown signal received...")
	logInfo("🛑 [PERF] Shutdown signal received")

	// Stop mining and show final stats
	miner.Stop()

	// Close log file if logging was enabled
	if logFileHandle != nil {
		logInfo("📝 [PERF] Closing performance log file")
		logFileHandle.Close()
	}
}

// createMemoryBlock creates a new memory block for puzzle solving
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

// RESTORED: Initialize GPU simulator for WSL2 or native GPU mining
func (m *QuantumMiner) initializeGPU() error {
	m.gpuInitMutex.Lock()
	defer m.gpuInitMutex.Unlock()

	logInfo("🔍 Initializing GPU quantum simulator (Device ID: %d)", m.gpuID)
	
	// Check if WSL2 mode is enabled
	if os.Getenv("WSL2_MODE") == "true" {
		logInfo("🐧 WSL2 mode detected - using Linux GPU acceleration")
	} else {
		logInfo("🖥️  Native Windows GPU mode")
	}

	// Initialize Qiskit GPU simulator
	simulator, err := quantum.NewQiskitGPUSimulator(m.gpuID)
	if err != nil {
		return fmt.Errorf("failed to create GPU simulator: %v", err)
	}

	m.gpuSimulator = simulator
	m.gpuInitialized = true

	if m.gpuSimulator.IsGPUAvailable() {
		logInfo("✅ GPU quantum simulator initialized successfully")
	} else {
		logInfo("⚠️  GPU not available, using CPU fallback")
	}

	return nil
}

// Basic mining functions
func (m *QuantumMiner) Start() error {
	m.startTime = time.Now()
	m.lastGCTime = m.startTime
	m.gcInterval = 5 * time.Second // AGGRESSIVE: Force GC every 5 seconds for CPU mining
	atomic.StoreInt32(&m.running, 1)
	
	// Test connection first
	if err := m.testConnection(); err != nil {
		return fmt.Errorf("failed to connect to node: %v", err)
	}
	
	// Initialize memory pool
	m.initializeMemoryPools()
	
	// ADDED: Initialize resource pools to prevent memory leaks
	m.initializeResourcePools()
	
	// ADDED: Start memory cleanup routines
	m.startMemoryCleanup()
	
	// Start work fetcher
	go m.workFetcher(context.Background())
	
	// Start mining threads
	for i := 0; i < m.threads; i++ {
		m.wg.Add(1)
		go m.miningThread(i)
	}
	
	// Start stats reporter
	go m.statsReporter()
	
	fmt.Printf("✅ Mining started with %d threads\n", m.threads)
	return nil
}

func (m *QuantumMiner) Stop() {
	atomic.StoreInt32(&m.running, 0)
	close(m.stopChan)
	m.wg.Wait()
	
	// ADDED: Stop cleanup timers
	if m.memoryCleanupTicker != nil {
		m.memoryCleanupTicker.Stop()
	}
	if m.gpuCleanupTicker != nil {
		m.gpuCleanupTicker.Stop()
	}
	
	// Final memory cleanup
	logInfo("🧹 Performing final memory cleanup...")
	runtime.GC()
	debug.FreeOSMemory()
	
	// Clean up GPU resources
	if m.gpuInitialized && m.gpuSimulator != nil {
		logInfo("🧹 Cleaning up GPU resources...")
		m.gpuSimulator.Cleanup()
		m.gpuSimulator.ForceCleanup()
	}
	
	// ADDED: Shutdown log writer properly
	if logToFile && logFileHandle != nil {
		logInfo("📝 [PERF] Shutting down log writer")
		// Signal log writer to flush and shutdown
		select {
		case logShutdown <- true:
		case <-time.After(1 * time.Second):
			// Timeout if log writer is stuck
		}
		// Give log writer time to flush
		time.Sleep(200 * time.Millisecond)
	}
	
	// Show final report
	m.showFinalReport()
}

func (m *QuantumMiner) testConnection() error {
	_, err := m.rpcCall("eth_blockNumber", []interface{}{})
	return err
}

func (m *QuantumMiner) initializeMemoryPools() {
	// Pre-allocate memory pools for efficient puzzle solving
	for i := 0; i < 10; i++ {
		memory := m.createMemoryBlock(128) // 128 puzzles per block
		select {
		case m.memoryPool <- memory:
		default:
			break
		}
	}
}

// ADDED: Initialize resource pools to prevent memory allocation overhead
func (m *QuantumMiner) initializeResourcePools() {
	// WorkPackage pool
	m.workPackagePool = sync.Pool{
		New: func() interface{} {
			return &WorkPackage{
				PuzzleHashes: make([]string, 128), // Pre-allocate for 128 puzzles
			}
		},
	}
	
	// Puzzle hash slice pool
	m.puzzleHashPool = sync.Pool{
		New: func() interface{} {
			return make([]string, 128) // Pre-allocate for 128 puzzles
		},
	}
	
	// Circuit pool for reusing quantum circuits
	m.circuitPool = sync.Pool{
		New: func() interface{} {
			return make(map[string]interface{}) // Circuit cache
		},
	}
	
	// CRITICAL: Byte array pools to eliminate CPU puzzle allocations
	m.outcomePool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 2) // Pre-allocate outcome bytes
		},
	}
	
	m.gateDataPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 8) // Pre-allocate gate data bytes
		},
	}
	
	// ULTRA-OPTIMIZED: Additional pools for zero-allocation mining
	m.bigIntPool = sync.Pool{
		New: func() interface{} {
			return new(big.Int)
		},
	}
	m.bufferPool = sync.Pool{
		New: func() interface{} {
			return new(bytes.Buffer)
		},
	}
	m.stringBuilder = sync.Pool{
		New: func() interface{} {
			return new(strings.Builder)
		},
	}
	m.hexBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 0, 64) // 32 bytes hex string
		},
	}
	m.branchPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 0, 128) // 64 nibbles
		},
	}
	m.noncePool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 32) // 32-byte nonce
		},
	}
	m.mapPool = sync.Pool{
		New: func() interface{} {
			return make(map[string]interface{})
		},
	}
	
	// PERFORMANCE: Cache frequently used values
	m.maxTarget = m.bigIntPool.Get().(*big.Int)
	m.maxTarget.SetString("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16)
	m.nonceRandom = rand.New(rand.NewSource(time.Now().UnixNano()))
	m.nonceMutex.Lock()
	m.nonceRandom.Seed(time.Now().UnixNano())
	m.nonceMutex.Unlock()
}

// ADDED: Start memory cleanup routines to prevent performance degradation
func (m *QuantumMiner) startMemoryCleanup() {
	logInfo("🔧 [PERF] Starting memory cleanup system")
	
	// Check if cleanup timer is already running (prevent stacking)
	if m.memoryCleanupTicker != nil {
		logError("⚠️ [PERF] Memory cleanup timer already running! Stopping old timer.")
		m.memoryCleanupTicker.Stop()
	}
	
	// Create new cleanup timer
	m.memoryCleanupTicker = time.NewTicker(m.gcInterval)
	
	// Check if GPU cleanup timer is already running (prevent stacking)
	if m.gpuCleanupTicker != nil && m.gpuMode {
		logError("⚠️ [PERF] GPU cleanup timer already running! Stopping old timer.")
		m.gpuCleanupTicker.Stop()
	}
	
	// Start memory cleanup routine
	go func() {
		cleanupCount := 0
		startTime := time.Now()
		
		logInfo("🧹 [PERF] Memory cleanup routine started")
		
		for {
			select {
			case <-m.stopChan:
				logInfo("🛑 [PERF] Memory cleanup routine stopped after %d cleanups in %.1fs", 
					cleanupCount, time.Since(startTime).Seconds())
				return
			case <-m.memoryCleanupTicker.C:
				cleanupCount++
				cleanupStart := time.Now()
				
				logInfo("🧹 [PERF] Starting cleanup #%d at runtime %.1fs", 
					cleanupCount, time.Since(m.startTime).Seconds())
				
				m.performMemoryCleanup()
				
				cleanupDuration := time.Since(cleanupStart)
				if cleanupDuration > 100*time.Millisecond {
					logError("⚠️ [PERF] Slow memory cleanup #%d took %v", cleanupCount, cleanupDuration)
				}
				
				// Log cleanup stats every 10 cleanups
				if cleanupCount%10 == 0 {
					avgCleanupTime := time.Since(startTime).Seconds() / float64(cleanupCount)
					logInfo("📊 [PERF] Cleanup stats: %d cleanups, %.2fs avg, Runtime: %.1fs", 
						cleanupCount, avgCleanupTime, time.Since(startTime).Seconds())
				}
			}
		}
	}()
	
	// Start GPU cleanup if GPU mode is enabled
	if m.gpuMode {
		m.gpuCleanupTicker = time.NewTicker(60 * time.Second) // Every minute for GPU
		go func() {
			gpuCleanupCount := 0
			
			logInfo("🎮 [PERF] GPU cleanup routine started")
			
			for {
				select {
				case <-m.stopChan:
					logInfo("🛑 [PERF] GPU cleanup routine stopped after %d cleanups", gpuCleanupCount)
					return
				case <-m.gpuCleanupTicker.C:
					gpuCleanupCount++
					gpuCleanupStart := time.Now()
					
					logInfo("🎮 [PERF] Starting GPU cleanup #%d", gpuCleanupCount)
					
					m.performGPUCleanup()
					
					gpuCleanupDuration := time.Since(gpuCleanupStart)
					logInfo("🎮 [PERF] GPU cleanup #%d completed in %v", gpuCleanupCount, gpuCleanupDuration)
				}
			}
		}()
	}
	
	logInfo("✅ [PERF] Memory cleanup system initialized with %v interval", m.gcInterval)
}

// ENHANCED: Memory management with cleanup
func (m *QuantumMiner) performMemoryCleanup() {
	now := time.Now()
	logInfo("🧹 [PERF] Memory cleanup started - Runtime: %.1fs", now.Sub(m.startTime).Seconds())
	
	// Force garbage collection (AGGRESSIVE for CPU mining)
	runtime.GC()
	runtime.GC() // Double GC for CPU mining
	
	// Log memory stats before cleanup
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	logInfo("📊 [PERF] Pre-cleanup: Alloc=%dMB, TotalAlloc=%dMB, Sys=%dMB, NumGC=%d", 
		memStats.Alloc/1024/1024, memStats.TotalAlloc/1024/1024, memStats.Sys/1024/1024, memStats.NumGC)
	
	// Force OS memory return
	debug.FreeOSMemory()
	
	// Log memory stats after cleanup
	runtime.ReadMemStats(&memStats)
	logInfo("🧹 [PERF] Post-cleanup: Alloc=%dMB, TotalAlloc=%dMB, Sys=%dMB, NumGC=%d", 
		memStats.Alloc/1024/1024, memStats.TotalAlloc/1024/1024, memStats.Sys/1024/1024, memStats.NumGC)
	
	// Update last GC time
	m.lastGCTime = now
	
	// Log pool statistics
	logInfo("🔧 [PERF] Memory pool status checked")
	
	cleanupDuration := time.Since(now)
	logInfo("✅ [PERF] Memory cleanup completed in %v", cleanupDuration)
	
	// Check for goroutine leaks
	goroutines := runtime.NumGoroutine()
	logInfo("🧵 [PERF] Active goroutines: %d", goroutines)
	if goroutines > 50 {
		logError("⚠️ [PERF] High goroutine count detected: %d", goroutines)
	}
}

// ADDED: Perform GPU cleanup to prevent memory accumulation
func (m *QuantumMiner) performGPUCleanup() {
	if m.gpuInitialized && m.gpuSimulator != nil {
		logInfo("🎮 Performing GPU memory cleanup")
		// Force cleanup of GPU simulator
		m.gpuSimulator.ForceCleanup()
	}
}

func (m *QuantumMiner) workFetcher(ctx context.Context) {
	// ENHANCED: Work fetcher with immediate refresh capability
	go func() {
		defer func() {
			if r := recover(); r != nil {
				logError("Work fetcher panic: %v", r)
			}
		}()
		
		logInfo("🔄 [PERF] Work fetcher started (200ms interval)")
		
		fetchCount := 0
		startTime := time.Now()
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()
		
		for {
			select {
			case <-ctx.Done():
				logInfo("🛑 [PERF] Work fetcher stopped after %d fetches in %.1fs", fetchCount, time.Since(startTime).Seconds())
				return
				
			case <-ticker.C:
				// Regular scheduled fetch
				m.fetchWork(fetchCount)
				fetchCount++
				
			case <-m.forceWorkRefresh:
				// Immediate refresh requested (e.g., after solution submission)
				logInfo("⚡ [PERF] Force work refresh requested - fetching immediately")
				m.fetchWork(fetchCount)
				fetchCount++
				
				// Also restart the ticker to avoid double-fetching
				ticker.Reset(200 * time.Millisecond)
			}
		}
	}()
}

// ENHANCED: Centralized work fetching with performance monitoring
func (m *QuantumMiner) fetchWork(fetchCount int) {
	startTime := time.Now()
	fetchStart := time.Now()
	
	// Fetch work from quantum-geth
	workResult, err := m.rpcCall("qmpow_getWork", []interface{}{})
	if err != nil {
		// ADDED: Detect "no mining work available yet" failures
		if strings.Contains(err.Error(), "no mining work available yet") {
			failures := atomic.AddInt32(&m.consecutiveWorkFailures, 1)
			m.lastWorkFailureTime = time.Now()
			
			if failures >= 10 && !m.nodeRecoveryMode {
				logError("🚨 [RECOVERY] Quantum-geth node appears stuck after %d consecutive work failures - entering recovery mode", failures)
				m.nodeRecoveryMode = true
				
				// Try recovery by requesting block information to "wake up" the node
				go m.attemptNodeRecovery()
			} else if failures <= 5 {
				logError("⚠️ [PERF] Work fetch #%d failed (attempt %d/10): %v", fetchCount, failures, err)
			}
		} else {
			// Reset failure count for other types of errors
			atomic.StoreInt32(&m.consecutiveWorkFailures, 0)
			logError("❌ [PERF] Work fetch #%d failed: %v", fetchCount, err)
		}
		return
	}
	
	// ADDED: Reset failure count on successful work fetch
	if atomic.LoadInt32(&m.consecutiveWorkFailures) > 0 {
		failures := atomic.SwapInt32(&m.consecutiveWorkFailures, 0)
		if m.nodeRecoveryMode {
			logInfo("✅ [RECOVERY] Quantum-geth node recovered after %d failures - exiting recovery mode", failures)
			m.nodeRecoveryMode = false
		} else {
			logInfo("✅ [PERF] Work fetch recovered after %d failures", failures)
		}
	}

	rpcDuration := time.Since(fetchStart)
	if rpcDuration > 100*time.Millisecond { // Reduced threshold for faster polling
		logError("⚠️ [PERF] Slow RPC call took %v on fetch #%d", rpcDuration, fetchCount)
	}

	// DEBUG: Log the actual format we're receiving (only first few times)
	if fetchCount <= 3 {
		logInfo("🔍 [PERF] Work fetch #%d raw result type: %T", fetchCount, workResult)
	}
	
	// Parse work data - handle different possible formats
	parseStart := time.Now()
	var workHash, blockNumberHex, target string
	var blockNumber, difficulty uint64
	var qbits, tcount, lnet int
	
	// Try to parse as array first (expected format)
	if workArray, ok := workResult.([]interface{}); ok {
		if fetchCount <= 3 {
			logInfo("📋 [PERF] Work fetch #%d: Array format with %d elements", fetchCount, len(workArray))
		}
		
		if len(workArray) >= 3 {
			// Basic work data
			workHash, _ = workArray[0].(string)
			blockNumberHex, _ = workArray[1].(string)
			target, _ = workArray[2].(string)
			
			// Parse block number from hex
			if blockNum, err := strconv.ParseUint(strings.TrimPrefix(blockNumberHex, "0x"), 16, 64); err == nil {
				blockNumber = blockNum
			} else {
				blockNumber = 1 // Fallback
			}
			
			// Get quantum parameters from array if available
			qbits, tcount, lnet = 16, 20, 128 // Defaults
			if len(workArray) > 4 {
				if qb, ok := workArray[4].(float64); ok {
					qbits = int(qb)
				}
			}
			if len(workArray) > 5 {
				if tc, ok := workArray[5].(float64); ok {
					tcount = int(tc)
				}
			}
			if len(workArray) > 6 {
				if ln, ok := workArray[6].(float64); ok {
					lnet = int(ln)
				}
			}
		} else {
			logError("❌ [PERF] Work fetch #%d: Array too short (%d elements)", fetchCount, len(workArray))
			return
		}
	} else {
		// Try to parse as object/map (alternative format)
		if workMap, ok := workResult.(map[string]interface{}); ok {
			if fetchCount <= 3 {
				logInfo("📋 [PERF] Work fetch #%d: Object format with keys: %v", fetchCount, getMapKeys(workMap))
			}
			
			workHash, _ = workMap["work_hash"].(string)
			if workHash == "" {
				workHash, _ = workMap["hash"].(string)
			}
			if workHash == "" {
				workHash, _ = workMap["headerHash"].(string)
			}
			
			blockNumberHex, _ = workMap["block_number"].(string)
			if blockNumberHex == "" {
				blockNumberHex, _ = workMap["number"].(string)
			}
			
			target, _ = workMap["target"].(string)
			if target == "" {
				target, _ = workMap["boundary"].(string)
			}
			
			// Parse block number
			if blockNum, err := strconv.ParseUint(strings.TrimPrefix(blockNumberHex, "0x"), 16, 64); err == nil {
				blockNumber = blockNum
			} else {
				blockNumber = 1 // Fallback
			}
			
			// Set defaults for quantum parameters
			qbits, tcount, lnet = 16, 20, 128
		} else {
			logError("❌ [PERF] Work fetch #%d: Unknown format type %T", fetchCount, workResult)
			return
		}
	}
	
	// Get actual difficulty using eth_getBlockByNumber
	difficulty = 200 // Default fallback
	if blockNumber > 0 {
		difficultyResult, err := m.rpcCall("eth_getBlockByNumber", []interface{}{blockNumberHex, false})
		if err == nil {
			if blockData, ok := difficultyResult.(map[string]interface{}); ok {
				if diffHex, ok := blockData["difficulty"].(string); ok {
					if parsedDiff, err := strconv.ParseUint(strings.TrimPrefix(diffHex, "0x"), 16, 64); err == nil {
						difficulty = parsedDiff
					}
				}
			}
		}
	}

	parseDuration := time.Since(parseStart)
	if parseDuration > 10*time.Millisecond {
		logError("⚠️ [PERF] Slow work parsing took %v on fetch #%d", parseDuration, fetchCount)
	}

	// Validate we have minimum required data
	if workHash == "" || target == "" {
		logError("❌ [PERF] Work fetch #%d: Missing required fields (hash: %s, target: %s)", 
			fetchCount, workHash, target)
		return
	}

	// Update current work with atomic write
	updateStart := time.Now()
	
	newWork := &QuantumWork{
		WorkHash:    workHash,
		BlockNumber: blockNumber,
		Target:      target,
		Difficulty:  difficulty,
		QBits:       qbits,
		TCount:      tcount,
		LNet:        lnet,
		FetchTime:   time.Now(),
	}

	m.workMutex.Lock()
	oldWork := m.currentWork
	m.currentWork = newWork
	m.workMutex.Unlock()
	
	updateDuration := time.Since(updateStart)
	totalFetchDuration := time.Since(fetchStart)
	
	// CRITICAL: Log work updates to detect work changes
	if oldWork == nil || oldWork.WorkHash != newWork.WorkHash {
		oldWorkHash := ""
		if oldWork != nil {
			oldWorkHash = oldWork.WorkHash
		}
		logInfo("🆕 [PERF] NEW WORK DETECTED #%d: %s... (Block: %d, Diff: %d) - Previous: %s...", 
			fetchCount, safeStringTruncate(newWork.WorkHash, 16), newWork.BlockNumber, newWork.Difficulty, safeStringTruncate(oldWorkHash, 16))
		logInfo("🔄 [PERF] Mining threads should immediately switch to new work!")
	}
	
	// Log performance warnings
	if totalFetchDuration > 500*time.Millisecond { // Adjusted for faster polling
		logError("⚠️ [PERF] Very slow work fetch #%d took %v (RPC: %v, Parse: %v, Update: %v)", 
			fetchCount, totalFetchDuration, rpcDuration, parseDuration, updateDuration)
	}
	
	// Log periodic work fetcher stats (every 50 fetches instead of 10)
	if fetchCount%50 == 0 {
		avgFetchTime := time.Since(startTime).Seconds() / float64(fetchCount)
		logInfo("📊 [PERF] Work fetcher stats: %d fetches, %.2fs avg, Runtime: %.1fs", 
			fetchCount, avgFetchTime, time.Since(startTime).Seconds())
	}
}

func (m *QuantumMiner) miningThread(threadID int) {
	defer m.wg.Done()
	
	startTime := time.Now()
	lastLogTime := startTime
	iterationsCount := 0
	workChanges := 0
	
	logInfo("🧵 [PERF] Mining thread %d started", threadID)
	
	for atomic.LoadInt32(&m.running) == 1 {
		iterationsCount++
		iterationStart := time.Now()
		
		// Get current work ONCE per iteration (not per qnonce!)
		m.workMutex.RLock()
		work := m.currentWork
		m.workMutex.RUnlock()
		
		if work == nil {
			time.Sleep(100 * time.Millisecond) // Shorter sleep
			continue
		}
		
		// CRITICAL FIX: Get work package ONCE per iteration
		workPackage, err := m.getWork(context.Background())
		if err != nil {
			continue
		}
		
		// Cache work hash for early exit detection
		currentWorkHash := work.WorkHash
		
		// ADDED: Check for work change immediately before starting iteration
		m.workMutex.RLock()
		latestWork := m.currentWork
		m.workMutex.RUnlock()
		if latestWork != nil && latestWork.WorkHash != currentWorkHash {
			// Work changed while we were getting work package - use latest
			logInfo("🔄 [PERF] Thread %d: Work changed before iteration start! Using latest: %s...", 
				threadID, safeStringTruncate(latestWork.WorkHash, 16))
			currentWorkHash = latestWork.WorkHash
		}
		
		// REDUCED: Smaller iteration batches for faster work switching (100K instead of 1M)
		qnonceStart := uint64(threadID * 100000)
		qnonceEnd := uint64((threadID + 1) * 100000)
		workChangeDetected := false
		
		for qnonce := qnonceStart; qnonce < qnonceEnd && atomic.LoadInt32(&m.running) == 1; qnonce++ {
			atomic.AddUint64(&m.attempts, 1)
			
			// Solve quantum puzzles using cached work package
			result, err := m.enhancedSolveQuantumPuzzles(context.Background(), workPackage.BlockNumber, workPackage.PuzzleHashes, qnonce, work.QBits, work.TCount, work.LNet)
			if err != nil {
				continue
			}
			
			// Check if solution meets target
			if m.isValidResultWithQNonce(result, workPackage.Target, qnonce) {
				// ADDED: Solution rate limiting to prevent quantum-geth work preparation failures
				m.solutionSubmissionMutex.Lock()
				timeSinceLastSolution := time.Since(m.lastSolutionTime)
				if timeSinceLastSolution < m.minSolutionInterval {
					waitTime := m.minSolutionInterval - timeSinceLastSolution
					logInfo("⏳ [RATE] Thread %d: Rate limiting solution submission - waiting %dms", threadID, waitTime.Milliseconds())
					m.solutionSubmissionMutex.Unlock()
					time.Sleep(waitTime)
					m.solutionSubmissionMutex.Lock()
				}
				m.lastSolutionTime = time.Now()
				logInfo("🚀 [RATE] Thread %d: Solution submission allowed - last solution was %dms ago", threadID, timeSinceLastSolution.Milliseconds())
				m.solutionSubmissionMutex.Unlock()
				
				// Submit solution
				if m.submitResult(workPackage, result, qnonce) {
					logInfo("🚀 [PERF] Thread %d: Solution submitted for QNonce %d", threadID, qnonce)
					
					// INTELLIGENT POST-SOLUTION WORK SWITCHING
					// Wait for quantum-geth to process the solution and prepare new work
					logInfo("⏳ [PERF] Thread %d: Waiting for new work after solution submission...", threadID)
					
					currentBlockNumber := work.BlockNumber
					newWorkFound := false
					
					// Try to get new work up to 10 times with exponential backoff
					for attempt := 1; attempt <= 10; attempt++ {
						// Wait progressively longer each attempt (1ms, 2ms, 4ms, 8ms, ...)
						waitTime := time.Duration(1<<uint(attempt-1)) * time.Millisecond
						if waitTime > 50*time.Millisecond {
							waitTime = 50 * time.Millisecond // Cap at 50ms
						}
						time.Sleep(waitTime)
						
						// Check if new work is available
						m.workMutex.RLock()
						latestWork := m.currentWork
						m.workMutex.RUnlock()
						
						if latestWork != nil && latestWork.BlockNumber > currentBlockNumber {
							// New block work available!
							logInfo("✅ [PERF] Thread %d: New work found for block %d (attempt %d, waited %dms)", 
								threadID, latestWork.BlockNumber, attempt, waitTime.Milliseconds())
							newWorkFound = true
							break
						} else if latestWork != nil && latestWork.BlockNumber == currentBlockNumber {
							// Still same block - solution might be rejected or processing
							logInfo("⏳ [PERF] Thread %d: Still block %d (attempt %d/%d, waited %dms)", 
								threadID, currentBlockNumber, attempt, 10, waitTime.Milliseconds())
						}
					}
					
					if newWorkFound {
						logInfo("🎯 [PERF] Thread %d: Successfully switched to new work after solution!", threadID)
					} else {
						logInfo("⚠️ [PERF] Thread %d: No new work after 10 attempts - assuming solution rejected, continuing current block", threadID)
					}
					
					// Clean up and exit iteration to get fresh work
					m.cleanupQuantumProof(result)
					m.returnWorkPackage(workPackage)
					
					// TRIGGER: Request immediate work refresh after solution submission
					select {
					case m.forceWorkRefresh <- struct{}{}:
						logInfo("⚡ [PERF] Thread %d: Triggered immediate work refresh after solution", threadID)
					default:
						// Channel full, refresh already pending
					}
					
					return // Get completely fresh work package
				}
				
				// ADDED: Check for work change after submission attempt (new work often comes after solutions)
				m.workMutex.RLock()
				postSubmitWork := m.currentWork
				m.workMutex.RUnlock()
				if postSubmitWork != nil && postSubmitWork.WorkHash != currentWorkHash {
					workChanges++
					workChangeDetected = true
					logInfo("🔄 [PERF] Thread %d: Work change detected after submission at QNonce %d!", threadID, qnonce)
					m.cleanupQuantumProof(result)
					break // New work available, exit qnonce loop immediately
				}
			}
			
			// Clean up proof (but NOT work package - reuse it!)
			m.cleanupQuantumProof(result)
			
			// CRITICAL: Check for new work EXTREMELY frequently (every 50 attempts)
			if qnonce%50 == 0 {
				m.workMutex.RLock()
				newWork := m.currentWork
				m.workMutex.RUnlock()
				if newWork != nil && newWork.WorkHash != currentWorkHash {
					workChanges++
					workChangeDetected = true
					logInfo("🔄 [PERF] Thread %d: Work change detected at QNonce %d! Old: %s... New: %s...", 
						threadID, qnonce, safeStringTruncate(currentWorkHash, 16), safeStringTruncate(newWork.WorkHash, 16))
					logInfo("🚫 [PERF] Thread %d: Abandoning %d remaining qnonces (%.1f%% of iteration)", 
						threadID, qnonceEnd-qnonce, float64(qnonceEnd-qnonce)/float64(qnonceEnd-qnonceStart)*100)
					break // New work available, exit qnonce loop immediately
				}
				
				// Log debug info every 10,000 qnonces to see what's happening
				if qnonce%10000 == 0 {
					logInfo("🔍 [PERF] Thread %d: QNonce %d, Current work: %s..., Cached work: %s...", 
						threadID, qnonce, safeStringTruncate(newWork.WorkHash, 16), safeStringTruncate(currentWorkHash, 16))
				}
			}
		}
		
		// Return work package ONCE per iteration
		m.returnWorkPackage(workPackage)
		
		// Reduced logging frequency - only log every 100 iterations instead of 1000
		if iterationsCount%100 == 0 {
			elapsed := time.Since(lastLogTime)
			iterationsPerSec := float64(100) / elapsed.Seconds()
			logInfo("🔍 [PERF] Thread %d: %d iterations, %.1f iter/sec, %d work changes, Runtime: %.1fs", 
				threadID, iterationsCount, iterationsPerSec, workChanges, time.Since(startTime).Seconds())
			lastLogTime = time.Now()
		}
		
		// Track iteration time with higher threshold (less noise)
		iterationDuration := time.Since(iterationStart)
		if iterationDuration > 5*time.Second { // Only log if REALLY slow
			logError("⚠️ [PERF] Thread %d: Very slow iteration took %v", threadID, iterationDuration)
		}
		
		// Log work change detection
		if workChangeDetected {
			logInfo("✅ [PERF] Thread %d: Successfully switched to new work", threadID)
		}
	}
	
	totalRuntime := time.Since(startTime)
	logInfo("🏁 [PERF] Mining thread %d finished: %d iterations, %d work changes in %.1fs (%.1f iter/sec)", 
		threadID, iterationsCount, workChanges, totalRuntime.Seconds(), float64(iterationsCount)/totalRuntime.Seconds())
}

// ULTRA-OPTIMIZED: Get work with ZERO allocations
func (m *QuantumMiner) getWork(ctx context.Context) (*WorkPackage, error) {
	// Use real work from geth node
	m.workMutex.RLock()
	currentWork := m.currentWork
	m.workMutex.RUnlock()

	if currentWork == nil {
		return nil, fmt.Errorf("no work available from geth node")
	}

	// ZERO-ALLOCATION: Reuse WorkPackage from pool
	workPackage := m.workPackagePool.Get().(*WorkPackage)
	
	// ZERO-ALLOCATION: Reuse big.Int from pool instead of new()
	target := m.bigIntPool.Get().(*big.Int)
	target.Set(m.maxTarget) // Use cached max target
	if currentWork.Target != "" {
		if _, ok := target.SetString(strings.TrimPrefix(currentWork.Target, "0x"), 16); !ok {
			// Parse failed, use max target
			target.Set(m.maxTarget)
		}
		// If ok is true, target is already set to the parsed value
	}

	// ZERO-ALLOCATION: Reuse puzzle hash slice
	puzzleCount := currentWork.LNet
	if cap(workPackage.PuzzleHashes) < puzzleCount {
		workPackage.PuzzleHashes = make([]string, puzzleCount)
	} else {
		workPackage.PuzzleHashes = workPackage.PuzzleHashes[:puzzleCount]
	}
	
	// ZERO-ALLOCATION: Use string builder instead of concatenation
	sb := m.stringBuilder.Get().(*strings.Builder)
	defer func() {
		sb.Reset()
		m.stringBuilder.Put(sb)
	}()
	
	baseHash := currentWork.WorkHash
	for i := 0; i < puzzleCount; i++ {
		// ZERO-ALLOCATION: Use string builder for puzzle data
		sb.Reset()
		sb.WriteString(baseHash)
		sb.WriteString("_")
		sb.WriteString(strconv.Itoa(i))
		workPackage.PuzzleHashes[i] = sha256Hash(sb.String())
	}

	// ZERO-ALLOCATION: Use string builder for parent hash
	parentHash := currentWork.WorkHash
	if parentHash == "" || len(strings.TrimPrefix(parentHash, "0x")) != 64 {
		sb.Reset()
		sb.WriteString("0x")
		// ZERO-ALLOCATION: Convert uint64 to hex manually
		blockNumHex := strconv.FormatUint(currentWork.BlockNumber, 16)
		// Pad to 64 characters
		for len(blockNumHex) < 64 {
			sb.WriteString("0")
		}
		sb.WriteString(blockNumHex)
		parentHash = sb.String()
	}

	// Update WorkPackage fields
	workPackage.BlockNumber = currentWork.BlockNumber
	workPackage.ParentHash = parentHash
	workPackage.Target = target

	return workPackage, nil
}

// ULTRA-OPTIMIZED: Return WorkPackage to pool with proper cleanup
func (m *QuantumMiner) returnWorkPackage(wp *WorkPackage) {
	// Clear sensitive data before returning to pool
	wp.BlockNumber = 0
	wp.ParentHash = ""
	
	// CRITICAL: Return big.Int to pool before clearing reference
	if wp.Target != nil {
		m.bigIntPool.Put(wp.Target)
		wp.Target = nil
	}
	
	// Keep PuzzleHashes slice for reuse but clear contents
	for i := range wp.PuzzleHashes {
		wp.PuzzleHashes[i] = ""
	}
	
	m.workPackagePool.Put(wp)
}

// Enhanced quantum puzzle solving (based on working old miner)
func (m *QuantumMiner) enhancedSolveQuantumPuzzles(ctx context.Context, blockNumber uint64, puzzleHashes []string, qnonce uint64, qbits, tcount, lnet int) (*QuantumProofSubmission, error) {
	// Use GPU acceleration if available
	if m.gpuMode && m.gpuInitialized {
		return m.solveQuantumPuzzlesGPU(ctx, blockNumber, puzzleHashes, qnonce, qbits, tcount, lnet)
	}

	// Fallback to CPU mode
	return m.solveQuantumPuzzlesCPU(ctx, blockNumber, puzzleHashes, qnonce, qbits, tcount, lnet)
}

// RESTORED: GPU-accelerated quantum puzzle solving
func (m *QuantumMiner) solveQuantumPuzzlesGPU(ctx context.Context, blockNumber uint64, puzzleHashes []string, qnonce uint64, qbits, tcount, lnet int) (*QuantumProofSubmission, error) {
	m.gpuInitMutex.RLock()
	simulator := m.gpuSimulator
	m.gpuInitMutex.RUnlock()

	if simulator == nil {
		return nil, fmt.Errorf("GPU simulator not initialized")
	}

	// ZERO-ALLOCATION: Create work hash using string builder
	sb := m.stringBuilder.Get().(*strings.Builder)
	defer func() {
		sb.Reset()
		m.stringBuilder.Put(sb)
	}()
	
	sb.WriteString(strconv.FormatUint(blockNumber, 10))
	sb.WriteString("_")
	sb.WriteString(strconv.FormatUint(qnonce, 10))
	workHash := sb.String()

	// GPU batch simulation
	outcomes, err := simulator.BatchSimulateQuantumPuzzles(workHash, qnonce, qbits, tcount, lnet)
	if err != nil {
		logError("❌ GPU simulation failed: %v", err)
		// Fallback to CPU
		return m.solveQuantumPuzzlesCPU(ctx, blockNumber, puzzleHashes, qnonce, qbits, tcount, lnet)
	}

	// ZERO-ALLOCATION: Generate gate hashes for each puzzle
	gateHashes := make([][]byte, lnet)
	for i := 0; i < lnet; i++ {
		// ZERO-ALLOCATION: Create gate data using string builder
		sb.Reset()
		sb.WriteString(workHash)
		sb.WriteString("_")
		sb.WriteString(strconv.Itoa(i))
		sb.WriteString("_")
		sb.WriteString(strconv.Itoa(tcount))
		
		hash := sha256.Sum256([]byte(sb.String()))
		gateHashes[i] = hash[:]
	}

	atomic.AddUint64(&m.puzzlesSolved, uint64(lnet))

	// Build quantum proof from GPU results
	return m.buildQuantumProof(outcomes, gateHashes, lnet)
}

// CPU fallback quantum puzzle solving
func (m *QuantumMiner) solveQuantumPuzzlesCPU(ctx context.Context, blockNumber uint64, puzzleHashes []string, qnonce uint64, qbits, tcount, lnet int) (*QuantumProofSubmission, error) {
	// Get memory from pool with timeout
	memory, err := m.getMemoryFromPool(ctx, lnet)
	if err != nil {
		return nil, fmt.Errorf("memory allocation failed: %v", err)
	}
	defer m.returnMemoryToPool(memory)

	// Progressive puzzle solving
	for i := 0; i < lnet; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// OPTIMIZED: CPU-friendly quantum simulation with pool cleanup
		outcome, gateHash := m.simulateQuantumPuzzle(qbits, tcount, i, qnonce)
		copy(memory.Outcomes[i], outcome)
		copy(memory.GateHashes[i], gateHash)
		
		// CRITICAL: Return outcome to pool after copying
		m.outcomePool.Put(outcome)

		// Brief CPU relief
		if i%8 == 0 && i > 0 {
			time.Sleep(1 * time.Millisecond)
		}
	}

	atomic.AddUint64(&m.puzzlesSolved, uint64(lnet))
	return m.buildQuantumProofFromMemory(memory, lnet)
}

// OPTIMIZED: Simulate quantum puzzle with ZERO allocations
func (m *QuantumMiner) simulateQuantumPuzzle(qbits, tcount, puzzleIndex int, qnonce uint64) ([]byte, []byte) {
	seed := qnonce + uint64(puzzleIndex)

	// CRITICAL: Get reusable byte arrays from pools (ZERO allocations!)
	outcome := m.outcomePool.Get().([]byte)
	gateData := m.gateDataPool.Get().([]byte)
	
	// Generate outcome (simplified for CPU efficiency)
	binary.LittleEndian.PutUint16(outcome, uint16(seed&0xFFFF))

	// Generate gate hash (simplified)
	binary.LittleEndian.PutUint64(gateData, seed*uint64(tcount))
	gateHash := sha256.Sum256(gateData)

	// CRITICAL: Return gateData to pool immediately
	m.gateDataPool.Put(gateData)

	// NOTE: outcome will be returned to pool by caller
	return outcome, gateHash[:]
}

// Get memory from pool with timeout
func (m *QuantumMiner) getMemoryFromPool(ctx context.Context, requiredSize int) (*PuzzleMemory, error) {
	select {
	case memory := <-m.memoryPool:
		if len(memory.Outcomes) >= requiredSize && len(memory.GateHashes) >= requiredSize {
			return memory, nil
		}
		m.memoryPool <- memory
		return m.createMemoryBlock(requiredSize), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(100 * time.Millisecond):
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

// ULTRA-OPTIMIZED: Build quantum proof from memory with ZERO allocations
func (m *QuantumMiner) buildQuantumProofFromMemory(memory *PuzzleMemory, lnet int) (*QuantumProofSubmission, error) {
	// ZERO-ALLOCATION: Get branch nibbles from pool
	branchNibbles := m.branchPool.Get().([]byte)
	if cap(branchNibbles) < lnet {
		branchNibbles = make([]byte, lnet)
	} else {
		branchNibbles = branchNibbles[:lnet]
	}
	
	for i := 0; i < lnet; i++ {
		if len(memory.Outcomes[i]) > 0 {
			branchNibbles[i] = memory.Outcomes[i][0]
		} else {
			branchNibbles[i] = 0
		}
	}

	// Generate proper 32-byte hashes for geth compatibility
	outcomeRoot := m.calculateOutcomeRoot(memory.Outcomes)
	gateHash := m.calculateGateHash(memory.GateHashes)
	proofRoot := m.calculateProofRoot(outcomeRoot, gateHash)
	extraNonce32 := m.generateExtraNonce32()

	return &QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchNibbles,
		ExtraNonce32:  extraNonce32,
	}, nil
}

// ULTRA-OPTIMIZED: Build quantum proof from GPU outcomes with ZERO allocations
func (m *QuantumMiner) buildQuantumProof(outcomes, gateHashes [][]byte, lnet int) (*QuantumProofSubmission, error) {
	// ZERO-ALLOCATION: Get branch nibbles from pool
	branchNibbles := m.branchPool.Get().([]byte)
	if cap(branchNibbles) < lnet {
		branchNibbles = make([]byte, lnet)
	} else {
		branchNibbles = branchNibbles[:lnet]
	}
	
	for i := 0; i < lnet; i++ {
		if len(outcomes[i]) > 0 {
			branchNibbles[i] = outcomes[i][0]
		} else {
			branchNibbles[i] = 0
		}
	}

	// Generate proper 32-byte hashes for geth compatibility
	outcomeRoot := m.calculateOutcomeRoot(outcomes)
	gateHash := m.calculateGateHash(gateHashes)
	proofRoot := m.calculateProofRoot(outcomeRoot, gateHash)
	extraNonce32 := m.generateExtraNonce32()

	return &QuantumProofSubmission{
		OutcomeRoot:   outcomeRoot,
		GateHash:      gateHash,
		ProofRoot:     proofRoot,
		BranchNibbles: branchNibbles,
		ExtraNonce32:  extraNonce32,
	}, nil
}

// ULTRA-OPTIMIZED: Calculate outcome root with ZERO allocations
func (m *QuantumMiner) calculateOutcomeRoot(outcomes [][]byte) string {
	buffer := m.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		buffer.Reset()
		m.bufferPool.Put(buffer)
	}()
	
	for _, outcome := range outcomes {
		buffer.Write(outcome)
	}
	hash := sha256.Sum256(buffer.Bytes())
	
	// ZERO-ALLOCATION: Use pooled string builder for hex encoding
	sb := m.stringBuilder.Get().(*strings.Builder)
	defer func() {
		sb.Reset()
		m.stringBuilder.Put(sb)
	}()
	
	sb.WriteString("0x")
	sb.WriteString(hex.EncodeToString(hash[:]))
	return sb.String()
}

// ULTRA-OPTIMIZED: Calculate gate hash with ZERO allocations
func (m *QuantumMiner) calculateGateHash(gateHashes [][]byte) string {
	buffer := m.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		buffer.Reset()
		m.bufferPool.Put(buffer)
	}()
	
	for _, gateHash := range gateHashes {
		buffer.Write(gateHash)
	}
	hash := sha256.Sum256(buffer.Bytes())
	
	// ZERO-ALLOCATION: Use pooled string builder for hex encoding
	sb := m.stringBuilder.Get().(*strings.Builder)
	defer func() {
		sb.Reset()
		m.stringBuilder.Put(sb)
	}()
	
	sb.WriteString("0x")
	sb.WriteString(hex.EncodeToString(hash[:]))
	return sb.String()
}

// ULTRA-OPTIMIZED: Calculate proof root with ZERO allocations
func (m *QuantumMiner) calculateProofRoot(outcomeRoot, gateHash string) string {
	// ZERO-ALLOCATION: Use string builder instead of concatenation
	sb := m.stringBuilder.Get().(*strings.Builder)
	defer func() {
		sb.Reset()
		m.stringBuilder.Put(sb)
	}()
	
	sb.WriteString(outcomeRoot)
	sb.WriteString(gateHash)
	hash := sha256.Sum256([]byte(sb.String()))
	
	// Reuse string builder for result
	sb.Reset()
	sb.WriteString("0x")
	sb.WriteString(hex.EncodeToString(hash[:]))
	return sb.String()
}

// ULTRA-OPTIMIZED: Generate extra nonce with ZERO allocations
func (m *QuantumMiner) generateExtraNonce32() []byte {
	// ZERO-ALLOCATION: Get nonce from pool
	nonce := m.noncePool.Get().([]byte)
	
	// Fast random generation using local random source
	m.nonceMutex.Lock()
	for i := 0; i < 32; i += 8 {
		val := m.nonceRandom.Uint64()
		nonce[i] = byte(val)
		nonce[i+1] = byte(val >> 8)
		nonce[i+2] = byte(val >> 16)
		nonce[i+3] = byte(val >> 24)
		nonce[i+4] = byte(val >> 32)
		nonce[i+5] = byte(val >> 40)
		nonce[i+6] = byte(val >> 48)
		nonce[i+7] = byte(val >> 56)
	}
	m.nonceMutex.Unlock()
	
	return nonce
}

// ULTRA-OPTIMIZED: Check if result is valid with proper cleanup
func (m *QuantumMiner) isValidResultWithQNonce(result *QuantumProofSubmission, target *big.Int, qnonce uint64) bool {
	proofQuality := m.calculateQuantumProofQuality(result, qnonce)
	defer m.bigIntPool.Put(proofQuality) // Return big.Int to pool
	return proofQuality.Cmp(target) <= 0
}

// ULTRA-OPTIMIZED: Calculate quantum proof quality with ZERO allocations
func (m *QuantumMiner) calculateQuantumProofQuality(result *QuantumProofSubmission, qnonce uint64) *big.Int {
	// CRITICAL: This must match CalculateQuantumProofQuality in quantum-geth/consensus/qmpow/bitcoin_style.go exactly!
	
	// ZERO-ALLOCATION: Get hex decode buffers from pool
	outcomeBuffer := m.hexBufferPool.Get().([]byte)
	gateBuffer := m.hexBufferPool.Get().([]byte)
	proofBuffer := m.hexBufferPool.Get().([]byte)
	defer func() {
		m.hexBufferPool.Put(outcomeBuffer[:0])
		m.hexBufferPool.Put(gateBuffer[:0])
		m.hexBufferPool.Put(proofBuffer[:0])
	}()
	
	// ZERO-ALLOCATION: Decode hex strings using pooled buffers
	outcomeHex := strings.TrimPrefix(result.OutcomeRoot, "0x")
	gateHex := strings.TrimPrefix(result.GateHash, "0x")
	proofHex := strings.TrimPrefix(result.ProofRoot, "0x")
	
	// Ensure buffers have enough capacity
	if cap(outcomeBuffer) < len(outcomeHex)/2 {
		outcomeBuffer = make([]byte, len(outcomeHex)/2)
	} else {
		outcomeBuffer = outcomeBuffer[:len(outcomeHex)/2]
	}
	if cap(gateBuffer) < len(gateHex)/2 {
		gateBuffer = make([]byte, len(gateHex)/2)
	} else {
		gateBuffer = gateBuffer[:len(gateHex)/2]
	}
	if cap(proofBuffer) < len(proofHex)/2 {
		proofBuffer = make([]byte, len(proofHex)/2)
	} else {
		proofBuffer = proofBuffer[:len(proofHex)/2]
	}
	
	// ZERO-ALLOCATION: Decode hex manually to avoid allocations
	hex.Decode(outcomeBuffer, []byte(outcomeHex))
	hex.Decode(gateBuffer, []byte(gateHex))
	hex.Decode(proofBuffer, []byte(proofHex))
	
	// ZERO-ALLOCATION: Get buffer from pool for combined data
	combinedBuffer := m.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		combinedBuffer.Reset()
		m.bufferPool.Put(combinedBuffer)
	}()
	
	combinedBuffer.Reset()
	combinedBuffer.Write(gateBuffer)
	combinedBuffer.Write(proofBuffer)
	
	// Use quantum-geth's exact algorithm
	h := sha256.New()

	// ZERO-ALLOCATION: Get nonce bytes from pool
	nonceBytes := m.noncePool.Get().([]byte)
	nonceBytes = nonceBytes[:8] // Only need 8 bytes for uint64
	defer m.noncePool.Put(nonceBytes[:32]) // Return full capacity
	
	// First, hash the nonce alone to create base entropy (BIG ENDIAN, not little!)
	binary.BigEndian.PutUint64(nonceBytes, qnonce)
	h.Write(nonceBytes)
	h.Write([]byte("QUANTUM_NONCE_SEED"))
	
	// ZERO-ALLOCATION: Get intermediate hash buffer
	intermBuffer := m.hexBufferPool.Get().([]byte)
	if cap(intermBuffer) < 32 {
		intermBuffer = make([]byte, 32)
	} else {
		intermBuffer = intermBuffer[:32]
	}
	defer m.hexBufferPool.Put(intermBuffer[:0])
	
	h.Sum(intermBuffer[:0])
	nonceSeed := intermBuffer

	// Reset hasher and combine nonce seed with quantum data
	h.Reset()
	h.Write(nonceSeed)
	h.Write(outcomeBuffer)
	h.Write(combinedBuffer.Bytes())

	// Add nonce again for extra sensitivity
	h.Write(nonceBytes)

	// ZERO-ALLOCATION: Round markers as byte arrays
	roundMarkers := [3][]byte{
		[]byte("QUANTUM_ROUND_0"),
		[]byte("QUANTUM_ROUND_1"),
		[]byte("QUANTUM_ROUND_2"),
	}
	
	// Multiple rounds of hashing for better distribution (EXACT MATCH)
	for i := 0; i < 3; i++ {
		h.Write(roundMarkers[i])
		h.Sum(intermBuffer[:0])
		h.Reset()
		h.Write(intermBuffer)
		h.Write(nonceBytes) // Nonce in every round
	}

	// Final hash with entropy marker (EXACT MATCH)
	h.Write([]byte("QUANTUM_BITCOIN_FINAL"))
	
	// ZERO-ALLOCATION: Get final hash buffer
	finalHash := m.hexBufferPool.Get().([]byte)
	if cap(finalHash) < 32 {
		finalHash = make([]byte, 32)
	} else {
		finalHash = finalHash[:32]
	}
	defer m.hexBufferPool.Put(finalHash[:0])
	
	h.Sum(finalHash[:0])

	// ZERO-ALLOCATION: Get big.Int from pool
	quality := m.bigIntPool.Get().(*big.Int)
	quality.SetBytes(finalHash)
	
	return quality
}

// ULTRA-OPTIMIZED: Clean up quantum proof submission and return pooled resources
func (m *QuantumMiner) cleanupQuantumProof(proof *QuantumProofSubmission) {
	// Return branch nibbles to pool
	if proof.BranchNibbles != nil {
		m.branchPool.Put(proof.BranchNibbles[:0])
		proof.BranchNibbles = nil
	}
	
	// Return extra nonce to pool  
	if proof.ExtraNonce32 != nil {
		m.noncePool.Put(proof.ExtraNonce32)
		proof.ExtraNonce32 = nil
	}
}

// ULTRA-OPTIMIZED: Submit mining result with ZERO allocations
func (m *QuantumMiner) submitResult(work *WorkPackage, result *QuantumProofSubmission, qnonce uint64) bool {
	// ZERO-ALLOCATION: Get map from pool
	gethQuantumProof := m.mapPool.Get().(map[string]interface{})
	defer func() {
		// Clear map before returning to pool
		for k := range gethQuantumProof {
			delete(gethQuantumProof, k)
		}
		m.mapPool.Put(gethQuantumProof)
	}()
	
	// Create quantum proof structure with correct data types for quantum-geth
	gethQuantumProof["outcome_root"] = result.OutcomeRoot
	gethQuantumProof["gate_hash"] = result.GateHash
	gethQuantumProof["proof_root"] = result.ProofRoot
	gethQuantumProof["branch_nibbles"] = result.BranchNibbles
	gethQuantumProof["extra_nonce32"] = result.ExtraNonce32

	// Ensure work hash has 0x prefix (required by quantum-geth)
	workHash := work.ParentHash
	if !strings.HasPrefix(workHash, "0x") {
		workHash = "0x" + workHash
	}

	// Submit to quantum-geth using correct qmpow RPC method
	submitData := []interface{}{
		qnonce,           // nonce as uint64
		workHash,         // block hash as hex string (with 0x prefix)
		gethQuantumProof, // quantum proof as struct
	}

	// FIXED: Use qmpow_submitWork method for quantum-geth
	result_rpc, err := m.rpcCall("qmpow_submitWork", submitData)
	if err != nil {
		// CRITICAL: Detect stale work errors
		errorMsg := err.Error()
		if strings.Contains(errorMsg, "Quantum work not found") || 
		   strings.Contains(errorMsg, "stale") || 
		   strings.Contains(errorMsg, "work not found") {
			atomic.AddUint64(&m.stale, 1)
			
			// Log stale work detection (but limit frequency to avoid spam)
			staleCount := atomic.LoadUint64(&m.stale)
			if staleCount%10 == 1 {
				logError("🚫 [PERF] Stale work detected: %s... (showing 1 in 10, total: %d)", 
					safeStringTruncate(workHash, 16), staleCount)
			}
			
			// Log every 50th stale work for monitoring
			if staleCount%50 == 0 {
				logError("📊 [PERF] Stale work milestone: %d total stale submissions", staleCount)
			}
			
			return false
		}
		
		// Regular error - not stale work
		logError("❌ Failed to submit solution: %v", err)
		atomic.AddUint64(&m.rejected, 1)
		return false
	}

	// Check if submission was accepted
	if accepted, ok := result_rpc.(bool); ok && accepted {
		atomic.AddUint64(&m.accepted, 1)
		logInfo("✅ Solution accepted by quantum-geth!")
		return true
	} else {
		atomic.AddUint64(&m.rejected, 1)
		// Only log rejection errors occasionally to reduce spam
		if atomic.LoadUint64(&m.rejected)%10 == 1 {
			logError("❌ Solution rejected by quantum-geth (showing 1 in 10)")
		}
		return false
	}
}

func (m *QuantumMiner) rpcCall(method string, params []interface{}) (interface{}, error) {
	// ZERO-ALLOCATION: Get buffer from pool for JSON marshaling (if pools are initialized)
	var buffer *bytes.Buffer
	
	// Check if pools are initialized (they're set up in Start() method)
	if m.bufferPool.New != nil {
		buffer = m.bufferPool.Get().(*bytes.Buffer)
		defer func() {
			buffer.Reset()
			m.bufferPool.Put(buffer)
		}()
	} else {
		// Fallback for early calls (like testConnection) before pools are initialized
		buffer = bytes.NewBuffer(nil)
	}

	req := JSONRPCRequest{
		ID:      1,
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	
	// Use buffer for JSON marshaling
	encoder := json.NewEncoder(buffer)
	if err := encoder.Encode(&req); err != nil {
		return nil, err
	}
	
	resp, err := m.client.Post(m.nodeURL, "application/json", buffer)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var jsonResp JSONRPCResponse
	if err := json.NewDecoder(resp.Body).Decode(&jsonResp); err != nil {
		return nil, err
	}
	
	if jsonResp.Error != nil {
		return nil, fmt.Errorf("RPC error: %s", jsonResp.Error.Message)
	}
	
	return jsonResp.Result, nil
}

func (m *QuantumMiner) statsReporter() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.updateDashboard()
		}
	}
}

func (m *QuantumMiner) updateDashboard() {
	now := time.Now()
	attempts := atomic.LoadUint64(&m.attempts)
	puzzles := atomic.LoadUint64(&m.puzzlesSolved)
	accepted := atomic.LoadUint64(&m.accepted)
	rejected := atomic.LoadUint64(&m.rejected)
	stale := atomic.LoadUint64(&m.stale)
	
	elapsed := now.Sub(m.startTime)
	
	// Calculate rates
	if elapsed.Seconds() > 0 {
		m.currentHashrate = float64(attempts) / elapsed.Seconds()
		m.currentPuzzleRate = float64(puzzles) / elapsed.Seconds()
	}
	
	// Clear screen and update dashboard
	fmt.Printf("\033[H\033[2J") // Clear screen
	fmt.Printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n")
	if m.gpuMode {
		fmt.Printf("│ 🎮 QUANTUM GPU MINER │ %d Threads │ Runtime: %.0fs                     │\n", m.threads, elapsed.Seconds())
	} else {
		fmt.Printf("│ 💻 QUANTUM CPU MINER │ %d Threads │ Runtime: %.0fs                     │\n", m.threads, elapsed.Seconds())
	}
	fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("│ ⚡ QNonce Rate     │ Current: %8.2f QN/s │ Average: %8.2f QN/s     │\n", m.currentHashrate, m.currentHashrate)
	fmt.Printf("│ ⚛️  Puzzle Rate     │ Current: %8.2f PZ/s │ Average: %8.2f PZ/s     │\n", m.currentPuzzleRate, m.currentPuzzleRate)
	fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("│ 🎯 Blocks Found    │ Accepted: %-8d │ Rejected: %-8d │ Stale: %-6d │\n", accepted, rejected, stale)
	fmt.Printf("│ 📊 Work Stats      │ Total QNonces: %-12d │ Total Puzzles: %-12d │\n", attempts, puzzles)
	fmt.Printf("│ 🧵 Thread Status   │ Active: %d/%d  │ All threads mining    │ Pool: ∞    │\n", m.threads, m.threads)
	fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────┤\n")
	// Get current block number and difficulty from work
	var currentBlock uint64 = 1
	var currentDifficulty uint64 = 200
	m.workMutex.RLock()
	if m.currentWork != nil {
		currentBlock = m.currentWork.BlockNumber
		currentDifficulty = m.currentWork.Difficulty
	}
	m.workMutex.RUnlock()
	
	fmt.Printf("│ 🔗 Current Block   │ Block: %-12d │ Difficulty: %-18d │\n", currentBlock, currentDifficulty)
	fmt.Printf("│ ⏱️  Block Timing    │ Average: %6.1fs │ Target: %6.1fs │ ASERT-Q Adjust │\n", 12.0, 12.0)
	fmt.Printf("└─────────────────────────────────────────────────────────────────────────────────┘\n")
	fmt.Printf("Last Update: %s | Press Ctrl+C to stop\n", now.Format("15:04:05"))
}

func (m *QuantumMiner) showFinalReport() {
	elapsed := time.Since(m.startTime)
	attempts := atomic.LoadUint64(&m.attempts)
	puzzles := atomic.LoadUint64(&m.puzzlesSolved)
	accepted := atomic.LoadUint64(&m.accepted)
	rejected := atomic.LoadUint64(&m.rejected)
	
	fmt.Printf("\n📊 ═══════════════════════════════════════════════════════════════════════════════\n")
	fmt.Printf("🏁 FINAL QUANTUM MINING SESSION REPORT\n")
	fmt.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════\n")
	if m.gpuMode {
		fmt.Printf("🎮 Mining Mode    │ GPU ACCELERATED │ %d Parallel Threads\n", m.threads)
	} else {
		fmt.Printf("🎮 Mining Mode    │ CPU PROCESSING │ %d Parallel Threads\n", m.threads)
	}
	fmt.Printf("⏱️  Session Time   │ %.0fs │ Started: %s\n", elapsed.Seconds(), m.startTime.Format("15:04:05"))
	fmt.Printf("⚡ Performance    │ QNonces: %8.2f QN/s │ Puzzles: %8.2f PZ/s\n", 
		float64(attempts)/elapsed.Seconds(), float64(puzzles)/elapsed.Seconds())
	fmt.Printf("🧮 Work Completed │ QNonces: %d │ Puzzles: %d │ Ratio: %.1f puzzles/qnonce\n", 
		attempts, puzzles, float64(puzzles)/float64(max(attempts, 1)))
	fmt.Printf("🎯 Block Results  │ Accepted: %d │ Rejected: %d │ Success Rate: %.2f%%\n", 
		accepted, rejected, float64(accepted)/float64(max(accepted+rejected, 1))*100)
	fmt.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════\n")
	fmt.Printf("👋 Thank you for contributing to the Quantum-Geth network!\n")
	fmt.Printf("💎 Your quantum computations help secure the blockchain!\n")
	fmt.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════\n")
}

func max(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

func sha256Hash(input string) string {
	hash := sha256.Sum256([]byte(input))
	return hex.EncodeToString(hash[:])
}

func showHelp() {
	fmt.Println("🚀 Quantum-Geth GPU/CPU Miner v" + VERSION + " (FIXED)")
	fmt.Println("⚛️  Advanced quantum proof-of-work mining")
	fmt.Println("")
	fmt.Println("📖 USAGE:")
	fmt.Println("  quantum-miner [OPTIONS]")
	fmt.Println("")
	fmt.Println("🔧 REQUIRED OPTIONS:")
	fmt.Println("  -coinbase ADDRESS    Your wallet address for block rewards")
	fmt.Println("")
	fmt.Println("🌐 CONNECTION OPTIONS:")
	fmt.Println("  -node URL           Node URL (default: http://127.0.0.1:8545)")
	fmt.Println("  -ip ADDRESS         Node IP address (default: 127.0.0.1)")
	fmt.Println("  -port NUMBER        Node RPC port (default: 8545)")
	fmt.Println("")
	fmt.Println("⚡ MINING OPTIONS:")
	fmt.Println("  -threads NUMBER     Mining threads (default: CPU cores)")
	fmt.Println("  -gpu               Enable GPU mining")
	fmt.Println("  -cpu               Force CPU-only mining")
	fmt.Println("")
	fmt.Println("📝 OTHER OPTIONS:")
	fmt.Println("  -log               Enable file logging")
	fmt.Println("  -version           Show version")
	fmt.Println("  -help              Show this help")
	fmt.Println("")
	fmt.Println("💡 EXAMPLES:")
	fmt.Println("  # GPU mining with 8 threads")
	fmt.Println("  quantum-miner -coinbase 0xYourAddress -threads 8")
	fmt.Println("")
	fmt.Println("  # CPU-only mining")
	fmt.Println("  quantum-miner -coinbase 0xYourAddress -cpu -threads 4")
	fmt.Println("")
	fmt.Println("  # Connect to remote node")
	fmt.Println("  quantum-miner -coinbase 0xYourAddress -node http://192.168.1.100:8545")
}

func isValidAddress(addr string) bool {
	if len(addr) != 42 {
		return false
	}
	if !strings.HasPrefix(addr, "0x") {
		return false
	}
	for _, char := range addr[2:] {
		if !((char >= '0' && char <= '9') || (char >= 'a' && char <= 'f') || (char >= 'A' && char <= 'F')) {
			return false
		}
	}
	return true
}

// Helper function to safely truncate strings for logging
func safeStringTruncate(s string, maxLen int) string {
	if len(s) == 0 {
		return "empty"
	}
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// Helper function to get map keys for debugging
func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// ADDED: Attempt to recover quantum-geth node from "no mining work available yet" state
func (m *QuantumMiner) attemptNodeRecovery() {
	logInfo("🔧 [RECOVERY] Starting quantum-geth node recovery procedure...")
	
	// Recovery strategy: Make various RPC calls to "wake up" the node
	recoveryMethods := []struct {
		name string
		fn   func() error
	}{
		{"Block Number", func() error {
			_, err := m.rpcCall("eth_blockNumber", []interface{}{})
			return err
		}},
		{"Latest Block", func() error {
			_, err := m.rpcCall("eth_getBlockByNumber", []interface{}{"latest", false})
			return err
		}},
		{"Pending Block", func() error {
			_, err := m.rpcCall("eth_getBlockByNumber", []interface{}{"pending", false})
			return err
		}},
		{"Chain ID", func() error {
			_, err := m.rpcCall("eth_chainId", []interface{}{})
			return err
		}},
	}
	
	for _, method := range recoveryMethods {
		logInfo("🔄 [RECOVERY] Trying recovery method: %s", method.name)
		err := method.fn()
		if err != nil {
			logError("❌ [RECOVERY] Recovery method '%s' failed: %v", method.name, err)
		} else {
			logInfo("✅ [RECOVERY] Recovery method '%s' succeeded", method.name)
		}
		time.Sleep(100 * time.Millisecond) // Small delay between recovery attempts
	}
	
	// Wait a bit and then try to force work refresh
	time.Sleep(1 * time.Second)
	logInfo("⚡ [RECOVERY] Forcing work refresh after recovery attempts...")
	
	select {
	case m.forceWorkRefresh <- struct{}{}:
		logInfo("✅ [RECOVERY] Force work refresh triggered")
	default:
		logInfo("⚠️ [RECOVERY] Force work refresh channel was full")
	}
	
	logInfo("🏁 [RECOVERY] Node recovery procedure completed")
}

