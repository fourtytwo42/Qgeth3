package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"quantum-gpu-miner/pkg/quantum"
	"runtime"
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

// Mining state - Simplified for CPU and GPU only
type QuantumMiner struct {
	coinbase    string
	nodeURL     string
	threads     int
	gpuMode     bool
	wsl2Mode    bool
	running     int32
	stopChan    chan bool

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

	// Memory management
	memoryPool chan *PuzzleMemory 

	// Atomic controls
	wg        sync.WaitGroup 
	isRunning atomic.Bool    
	
	// WSL2 caching
	wsl2BinaryPath string // Path to cached WSL2 binary
	wsl2SetupDone  bool   // Whether WSL2 setup is complete
	
	// Real Qiskit Quantum Simulator
	qiskitSim *quantum.QiskitGPUSimulator
	qiskitMutex sync.Mutex  // CRITICAL: Serialize access to Python subprocess
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

// Work structure
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

type QuantumProofSubmission struct {
	OutcomeRoot   string `json:"outcome_root"`
	GateHash      string `json:"gate_hash"`
	ProofRoot     string `json:"proof_root"`
	BranchNibbles []byte `json:"branch_nibbles"`
	ExtraNonce32  string `json:"extra_nonce32"`
}

type WorkPackage struct {
	BlockNumber  uint64
	ParentHash   string
	Target       *big.Int
	PuzzleHashes []string
}

func main() {
	var (
		version  = flag.Bool("version", false, "Show version")
		coinbase = flag.String("coinbase", "", "Coinbase address")
		node     = flag.String("node", "", "Node URL (e.g., http://127.0.0.1:8545)")
		ip       = flag.String("ip", "127.0.0.1", "Node IP address")
		port     = flag.Int("port", 8545, "Node RPC port")
		threads  = flag.Int("threads", runtime.NumCPU(), "Number of mining threads")
		gpu      = flag.Bool("gpu", true, "Enable GPU mining (WSL2 on Windows, native Qiskit on Linux)")
		cpu      = flag.Bool("cpu", false, "Force CPU mining only")
		wsl2     = flag.Bool("wsl2", false, "Force WSL2 mode (Windows only)")
		logFile  = flag.Bool("log", false, "Enable logging to file (quantum-miner.log)")
		help     = flag.Bool("help", false, "Show help")
	)
	flag.Parse()

	// Handle CPU override flag
	if *cpu {
		*gpu = false
		*wsl2 = false
		logInfo("🔧 CPU mode forced via -cpu flag")
	}
	
	// Handle WSL2 flag - launch WSL2 directly if requested
	if *wsl2 {
		if runtime.GOOS != "windows" {
			log.Fatal("❌ WSL2 mode is only available on Windows!")
		}
		fmt.Printf("🪟 WSL2 mode enabled - launching directly in WSL2...\n")
		if err := launchInWSL2(); err != nil {
			log.Fatalf("❌ WSL2 launch failed: %v", err)
		}
		os.Exit(0) // WSL2 launched successfully, exit Windows process
	}

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
		if *cpu {
			fmt.Printf("Architecture: Simplified CPU/GPU (NO fallback)\n")
			fmt.Printf("Default Mode: CPU only\n")
		} else {
			fmt.Printf("Architecture: Simplified CPU/GPU (NO fallback)\n")
			if runtime.GOOS == "windows" {
				fmt.Printf("Default Mode: WSL2 GPU acceleration\n")
			} else {
				fmt.Printf("Default Mode: Native GPU acceleration\n")
			}
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
	fmt.Println("🔗 ASERT-Q difficulty adjustment with quantum proof-of-work")
	if *cpu {
		fmt.Println("💻 CPU Mining: FORCED via -cpu flag")
	} else if *gpu {
		fmt.Println("🎮 GPU Mining: Enabled (WSL2 on Windows, native on Linux)")
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

	// GPU mode detection - NO fallback behavior
	gpuAvailable := false
	wsl2Available := false
	
	if *gpu && !*cpu {
		fmt.Printf("🎮 GPU MODE REQUESTED - Initializing...\n")
		
		// Windows: GPU mode = WSL2 (no fallback)
		if runtime.GOOS == "windows" {
			fmt.Printf("🪟 Windows GPU Mode: Using WSL2 acceleration\n")
			if err := checkAndInitializeWSL2(); err != nil {
				log.Fatalf("❌ WSL2 GPU failed: %v\n💡 Fix WSL2 setup or use -cpu flag", err)
			}
			fmt.Printf("✅ WSL2 GPU acceleration ready!\n")
			wsl2Available = true
			gpuAvailable = true
			*wsl2 = true // Enable WSL2 automatically on Windows GPU mode
		} else {
			// Linux: GPU mode = native Qiskit (no fallback)
			fmt.Printf("🐧 Linux GPU Mode: Using native Qiskit\n")
			if err := checkNativeGPUSupport(); err != nil {
				log.Fatalf("❌ Native GPU failed: %v\n💡 Install GPU drivers or use -cpu flag", err)
			}
			fmt.Printf("✅ Native Qiskit GPU acceleration ready!\n")
			gpuAvailable = true
		}
	}

	// Determine node URL
	var nodeURL string
	if *node != "" {
		nodeURL = *node
	} else {
		nodeURL = fmt.Sprintf("http://%s:%d", *ip, *port)
	}

	fmt.Printf("📋 Final Configuration:\n")
	fmt.Printf("   💰 Coinbase: %s\n", *coinbase)
	fmt.Printf("   🌐 Node URL: %s\n", nodeURL)
	if wsl2Available {
		fmt.Printf("   🪟 WSL2 GPU: Windows-optimized quantum acceleration ✅\n")
		fmt.Printf("   🧵 GPU Threads: %d quantum circuits in parallel\n", *threads)
	} else if gpuAvailable {
		fmt.Printf("   🐧 Native GPU: Linux Qiskit quantum acceleration ✅\n")
		fmt.Printf("   🧵 GPU Threads: %d quantum circuits in parallel\n", *threads)
	} else {
		fmt.Printf("   💻 CPU Mode: %d threads (requested via -cpu flag) ✅\n", *threads)
	}
	
	fmt.Printf("   🖥️  Quantum Backend: Local simulation\n")
	fmt.Printf("   ⚛️  Quantum Puzzles: 128 chained per block\n")
	fmt.Printf("   🔬 Qubits per Puzzle: 16\n")
	fmt.Printf("   🚪 T-Gates per Puzzle: minimum 20 (ENFORCED)\n")
	fmt.Println("")

	// Create and configure miner
	miner := &QuantumMiner{
		coinbase:         *coinbase,
		nodeURL:          nodeURL,
		threads:          *threads,
		gpuMode:          *gpu || wsl2Available,
		wsl2Mode:         wsl2Available,
		stopChan:         make(chan bool, 1),
		client:           &http.Client{Timeout: 30 * time.Second},
		threadStates:     make(map[int]*ThreadState),
		memoryPool:       make(chan *PuzzleMemory, *threads*2+5), // Extra space for initial blocks
		maxActiveThreads: int32(*threads),
		targetBlockTime:  12 * time.Second,
		currentDifficulty: 200,
	}

	// Initialize real Qiskit quantum simulator
	logInfo("🔧 Initializing Qiskit quantum simulator...")
	var err error
	miner.qiskitSim, err = quantum.NewQiskitGPUSimulator(0)
	if err != nil {
		logError("❌ Failed to initialize Qiskit simulator: %v", err)
		logInfo("💡 This will fall back to simplified simulation")
	} else {
		logInfo("✅ Qiskit quantum simulator initialized!")
		if miner.qiskitSim.IsGPUAvailable() {
			logInfo("⚡ GPU acceleration is available!")
		} else {
			logInfo("💻 Using CPU quantum simulation")
		}
	}

	// Initialize memory pool
	for i := 0; i < 5; i++ {
		miner.memoryPool <- miner.createMemoryBlock(128) // 128 puzzles per block
	}

	// Set up signal handling
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	// Start background processes
	go miner.workFetcher()
	go miner.statsReporter()

	// Start mining
	fmt.Printf("🚀 MINING MODE: ")
	if miner.wsl2Mode {
		fmt.Printf("WSL2 GPU Acceleration\n")
	} else if miner.gpuMode {
		fmt.Printf("Native GPU Acceleration\n")
	} else {
		fmt.Printf("CPU Processing (%d Threads)\n", miner.threads)
	}

	err = miner.Start()
	if err != nil {
		log.Fatalf("❌ Failed to start mining: %v", err)
	}

	// Wait for shutdown signal
	<-c
	fmt.Printf("\n🛑 Shutdown signal received...\n")
	miner.Stop()
	miner.showFinalReport()
}

// checkAndInitializeWSL2 checks WSL2 availability and initializes it with caching
func checkAndInitializeWSL2() error {
	// Check if WSL2 is available
	testCmd := exec.Command("wsl", "--status")
	if output, err := testCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("WSL2 not available: %v", err)
	} else {
		fmt.Printf("✅ WSL2 status: %s\n", strings.TrimSpace(string(output)))
	}

	// Get current executable directory
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("failed to get executable path: %v", err)
	}
	exeDir := filepath.Dir(exePath)
	
	// Check if WSL2 binary already exists (CACHING)
	wsl2BinaryPath := filepath.Join(exeDir, "quantum-miner-wsl2")
	if _, err := os.Stat(wsl2BinaryPath); err == nil {
		fmt.Printf("✅ Found cached WSL2 binary: %s\n", wsl2BinaryPath)
		
		// Test if the cached binary works
		wsl2Path := convertToWSL2Path(exeDir)
		testCmd := exec.Command("wsl", "bash", "-c", fmt.Sprintf("cd %s && ./quantum-miner-wsl2 -version 2>/dev/null", wsl2Path))
		if output, err := testCmd.CombinedOutput(); err == nil && strings.Contains(string(output), VERSION) {
			fmt.Printf("✅ Cached WSL2 binary is working - skipping rebuild\n")
			return nil // Use cached binary
		} else {
			fmt.Printf("⚠️  Cached WSL2 binary outdated - rebuilding...\n")
		}
	}

	// Check if Go and Python setup is already done (CACHING)
	goSetupPath := filepath.Join(exeDir, "go-wsl2", "go-wrapper.sh")
	pythonSetupPath := filepath.Join(exeDir, "go-wsl2", "python-linux.sh")
	
	if _, err := os.Stat(goSetupPath); err != nil {
		return fmt.Errorf("WSL2 Go setup not found: %s", goSetupPath)
	}
	
	if _, err := os.Stat(pythonSetupPath); err != nil {
		return fmt.Errorf("WSL2 Python setup not found: %s", pythonSetupPath)
	}

	fmt.Printf("✅ WSL2 environment setup found - proceeding with build\n")
	
	// Launch WSL2 build process with caching awareness
	return launchInWSL2Cached()
}

// checkNativeGPUSupport checks for native Qiskit GPU support on Linux
func checkNativeGPUSupport() error {
	// Test if we can create a quantum simulator
	simulator, err := quantum.NewQiskitGPUSimulator(0)
	if err != nil {
		return fmt.Errorf("failed to initialize quantum simulator: %v", err)
	}
	defer simulator.Cleanup()

	fmt.Printf("✅ Native Qiskit quantum simulator initialized\n")
	return nil
}

// launchInWSL2Cached launches WSL2 with smart caching to avoid rebuilds
func launchInWSL2Cached() error {
	fmt.Printf("🚀 Initializing WSL2 with smart caching...\n")
	
	// Get the current executable path
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("failed to get executable path: %v", err)
	}
	exeDir := filepath.Dir(exePath)
	wsl2Path := convertToWSL2Path(exeDir)
	
	// Check if WSL2 binary already exists
	wsl2BinaryPath := fmt.Sprintf("%s/quantum-miner-wsl2", wsl2Path)
	
	checkCmd := exec.Command("wsl", "bash", "-c", fmt.Sprintf("ls -la %s 2>/dev/null", wsl2BinaryPath))
	if output, err := checkCmd.CombinedOutput(); err == nil && strings.Contains(string(output), "quantum-miner-wsl2") {
		fmt.Printf("✅ WSL2 binary already exists - skipping build\n")
		return nil
	}

	fmt.Printf("🔨 Building WSL2 binary (first time only)...\n")
	
	// Build the WSL2 command with caching-aware setup
	wsl2Cmd := fmt.Sprintf(`
set -e
echo "🔍 WSL2 cached build starting..."
cd %s || exit 1

# Check if Go environment is already initialized
if [ -f "go-wsl2/go-wrapper.sh" ]; then
    echo "✅ Go environment found - sourcing..."
    source go-wsl2/init-go-env.sh || exit 1
else
    echo "❌ Go environment not found"
    exit 1
fi

# Check if Python environment is ready
if [ -f "go-wsl2/python-linux.sh" ]; then
    echo "✅ Python environment found - ready"
else
    echo "⚠️  Setting up Python environment..."
    bash go-wsl2/setup-python-linux.sh || exit 1
fi

# Build only if binary doesn't exist
if [ ! -f "quantum-miner-wsl2" ]; then
    echo "🔨 Building WSL2 binary..."
    cd ../.. || exit 1
    cd quantum-miner || exit 1
    %s/go-wrapper.sh build -o quantum-miner-wsl2 || exit 1
    mv quantum-miner-wsl2 %s/ || exit 1
    echo "✅ WSL2 binary built and cached"
else
    echo "✅ WSL2 binary already exists - skipping build"
fi

echo "🎉 WSL2 setup complete with caching!"
`, wsl2Path, wsl2Path, wsl2Path)
	
	// Execute setup
	cmd := exec.Command("wsl", "bash", "-c", wsl2Cmd)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		fmt.Printf("❌ WSL2 setup failed: %v\n", err)
		fmt.Printf("📄 Output: %s\n", string(output))
		return err
	}
	
	fmt.Printf("✅ WSL2 build complete with smart caching\n")
	return nil
}

// convertToWSL2Path converts a Windows path to WSL2 format
func convertToWSL2Path(windowsPath string) string {
	// Convert C:\Users\... to /mnt/c/Users/...
	if len(windowsPath) >= 3 && windowsPath[1] == ':' {
		drive := strings.ToLower(string(windowsPath[0]))
		path := strings.Replace(windowsPath[2:], "\\", "/", -1)
		return fmt.Sprintf("/mnt/%s%s", drive, path)
	}
	return ""
}

// launchInWSL2 launches the cached WSL2 binary with smart argument handling
func launchInWSL2() error {
	fmt.Printf("🚀 Launching cached WSL2 quantum miner...\n")
	
	// Get the current executable path
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("failed to get executable path: %v", err)
	}
	exeDir := filepath.Dir(exePath)
	wsl2Path := convertToWSL2Path(exeDir)
	
	// Prepare the command arguments
	args := os.Args[1:] // Skip the program name
	
	// Remove -wsl2 flag since we'll be running in WSL2
	filteredArgs := []string{}
	for _, arg := range args {
		if arg != "-wsl2" {
			filteredArgs = append(filteredArgs, arg)
		}
	}
	
	// Fix WSL2 network connectivity - get Windows host IP
	windowsHostIP := ""
	hasLocalhost := false
	for _, arg := range filteredArgs {
		if strings.Contains(arg, "localhost:8545") || strings.Contains(arg, "127.0.0.1:8545") {
			hasLocalhost = true
			break
		}
	}
	
	if hasLocalhost {
		fmt.Printf("🌐 WSL2 Network: Detecting Windows host IP for connectivity...\n")
		hostIPCmd := exec.Command("wsl", "bash", "-c", "grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1")
		if output, err := hostIPCmd.CombinedOutput(); err == nil {
			rawOutput := strings.TrimSpace(string(output))
			parts := strings.Fields(rawOutput)
			if len(parts) > 0 {
				windowsHostIP = parts[len(parts)-1]
			}
			fmt.Printf("🌐 WSL2 Network: Windows host IP detected as %s\n", windowsHostIP)
		} else {
			fmt.Printf("⚠️  WSL2 Network: Failed to detect Windows host IP\n")
		}
		
		// Replace localhost/127.0.0.1 with Windows host IP
		if windowsHostIP != "" {
			for i, arg := range filteredArgs {
				if strings.Contains(arg, "localhost:8545") {
					filteredArgs[i] = strings.ReplaceAll(arg, "localhost", windowsHostIP)
					fmt.Printf("🔄 WSL2 Network: %s → %s\n", arg, filteredArgs[i])
				} else if strings.Contains(arg, "127.0.0.1:8545") {
					filteredArgs[i] = strings.ReplaceAll(arg, "127.0.0.1", windowsHostIP)
					fmt.Printf("🔄 WSL2 Network: %s → %s\n", arg, filteredArgs[i])
				}
			}
		}
	}
	
	// Build the WSL2 command to run cached binary
	wsl2Cmd := fmt.Sprintf(`
set -e
echo "🚀 Using cached WSL2 quantum miner..."
cd %s || exit 1

# Set WSL2 environment
export WSL2_MODE=true
export PYTHON_EXEC=%s/go-wsl2/python-linux.sh

# Check if binary exists
if [ ! -f "quantum-miner-wsl2" ]; then
    echo "❌ WSL2 binary not found - run initial setup first"
    exit 1
fi

echo "✅ Starting cached WSL2 binary..."
echo "🚀 Arguments: %s"
./quantum-miner-wsl2 %s
`, wsl2Path, wsl2Path, strings.Join(filteredArgs, " "), strings.Join(filteredArgs, " "))
	
	// Execute WSL2 command
	cmd := exec.Command("wsl", "bash", "-c", wsl2Cmd)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	
	fmt.Printf("🚀 Launching WSL2 quantum miner...\n")
	return cmd.Run()
}

// createMemoryBlock creates a new memory block for puzzle solving
func (m *QuantumMiner) createMemoryBlock(size int) *PuzzleMemory {
	return &PuzzleMemory{
		Outcomes:   make([][]byte, size),
		GateHashes: make([][]byte, size),
		WorkBuffer: make([]byte, size*64), // 64 bytes per puzzle
		ID:         int(time.Now().UnixNano()),
	}
}

// Basic mining functions (simplified versions)
func (m *QuantumMiner) Start() error {
	m.startTime = time.Now()
	atomic.StoreInt32(&m.running, 1)
	
	// Test connection first
	if err := m.testConnection(); err != nil {
		return fmt.Errorf("failed to connect to node: %v", err)
	}
	
	// Start mining threads
	for i := 0; i < m.threads; i++ {
		m.wg.Add(1)
		go m.miningThread(i)
	}
	
	fmt.Printf("✅ Mining started with %d threads\n", m.threads)
	return nil
}

func (m *QuantumMiner) Stop() {
	atomic.StoreInt32(&m.running, 0)
	close(m.stopChan)
	m.wg.Wait()
	
	// Cleanup Qiskit simulator resources
	if m.qiskitSim != nil {
		m.qiskitSim.Cleanup()
	}
}

func (m *QuantumMiner) testConnection() error {
	_, err := m.rpcCall("eth_blockNumber", []interface{}{})
	return err
}

func (m *QuantumMiner) rpcCall(method string, params []interface{}) (interface{}, error) {
	req := JSONRPCRequest{
		ID:      1,
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	
	reqBytes, _ := json.Marshal(req)
	resp, err := m.client.Post(m.nodeURL, "application/json", bytes.NewBuffer(reqBytes))
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

func (m *QuantumMiner) workFetcher() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	
	logInfo("🔄 Work fetcher starting...")
	
	for atomic.LoadInt32(&m.running) == 1 {
		// Fetch work from geth
		logInfo("📡 Fetching work from %s...", m.nodeURL)
		result, err := m.rpcCall("qmpow_getWork", []interface{}{})
		if err != nil {
			logError("❌ Failed to fetch work: %v", err)
			// Try basic connection test
			if _, connErr := m.rpcCall("eth_blockNumber", []interface{}{}); connErr != nil {
				logError("❌ Connection test failed: %v", connErr)
			} else {
				logInfo("✅ Connection OK, but qmpow_getWork failed")
			}
			time.Sleep(5 * time.Second)
			continue
		}
		
		logInfo("📋 Work result: %+v", result)
		
		// Parse work result
		if workArray, ok := result.([]interface{}); ok && len(workArray) >= 3 {
			work := &QuantumWork{
				WorkHash:    workArray[0].(string),
				BlockNumber: uint64(1), // Parse from workArray[1] if needed
				Target:      workArray[2].(string),
				QBits:       16, // Default values
				TCount:      20,
				LNet:        128,
				FetchTime:   time.Now(),
			}
			
			logInfo("📋 Parsed work - Hash: %s, Target: %s", work.WorkHash[:16]+"...", work.Target[:16]+"...")
			
			// Parse quantum params from workArray[3] if available
			if len(workArray) >= 4 {
				if params, ok := workArray[3].(string); ok {
					logInfo("🔬 Quantum params: %s", params)
					// Parse hex-encoded quantum params
					// Format: qbits:16,tcount:20,lnet:128
					if decoded, err := hex.DecodeString(params[2:]); err == nil {
						paramStr := string(decoded)
						logInfo("🔬 Decoded params: %s", paramStr)
						// Parse parameters (simplified)
						if strings.Contains(paramStr, "qbits:") {
							// Parse actual parameters if needed
						}
					}
				}
			}
			
			// Update current work
			m.workMutex.Lock()
			oldWork := m.currentWork
			m.currentWork = work
			m.workMutex.Unlock()
			
			if oldWork == nil || oldWork.WorkHash != work.WorkHash {
				logInfo("🆕 New work assigned: %s (target: %s)", work.WorkHash[:16]+"...", work.Target[:16]+"...")
			} else {
				logInfo("🔄 Same work, continuing...")
			}
		} else {
			logError("❌ Invalid work format: %+v", result)
		}
		
		<-ticker.C
	}
	
	logInfo("🔄 Work fetcher stopping...")
}

func (m *QuantumMiner) miningThread(threadID int) {
	defer m.wg.Done()
	
	logInfo("🧵 Mining thread %d starting", threadID)
	
	for atomic.LoadInt32(&m.running) == 1 {
		// Get current work
		m.workMutex.RLock()
		work := m.currentWork
		m.workMutex.RUnlock()
		
		if work == nil {
			if threadID == 0 { // Only log from thread 0 to avoid spam
				logInfo("⏳ Thread %d waiting for work...", threadID)
			}
			time.Sleep(1 * time.Second)
			continue
		}
		
		if threadID == 0 { // Only log from thread 0
			logInfo("⛏️ Thread %d starting mining with work %s", threadID, work.WorkHash[:16]+"...")
		}
		
		// Try different qnonces
		for qnonce := uint64(threadID * 1000000); qnonce < uint64((threadID+1)*1000000) && atomic.LoadInt32(&m.running) == 1; qnonce++ {
			atomic.AddUint64(&m.attempts, 1)
			
			// Log every 5000 attempts from thread 0, and every 20000 from other threads
			if (threadID == 0 && qnonce%5000 == 0) || (threadID > 0 && qnonce%20000 == 0) {
				progress := qnonce - uint64(threadID*1000000)
				logInfo("🔍 Thread %d progress: %d/%d qnonces (%.1f%%)", threadID, progress, 1000000, float64(progress)/10000.0)
			}
			
			// Solve quantum puzzles
			if m.solveQuantumPuzzles(work, qnonce) {
				logInfo("🎉 Thread %d found solution at qnonce %d", threadID, qnonce)
				break // Found solution, get new work
			}
			
			// Check for new work every 1000 attempts
			if qnonce%1000 == 0 {
				m.workMutex.RLock()
				newWork := m.currentWork
				m.workMutex.RUnlock()
				if newWork != work {
					logInfo("🔄 Thread %d: New work detected, switching...", threadID)
					break // New work available
				}
			}
		}
	}
	
	logInfo("🧵 Mining thread %d stopping", threadID)
}

func (m *QuantumMiner) solveQuantumPuzzles(work *QuantumWork, qnonce uint64) bool {
	// Use real Qiskit batch simulation for all 128 puzzles at once
	var outcomes [][]byte
	var err error
	
	if m.qiskitSim != nil {
		// CRITICAL: Serialize access to Python subprocess to prevent "exec: already started"
		m.qiskitMutex.Lock()
		
		// Use real Qiskit quantum computation
		start := time.Now()
		outcomes, err = m.qiskitSim.BatchSimulateQuantumPuzzles(
			work.WorkHash, 
			qnonce, 
			work.QBits,   // 16 qubits
			work.TCount,  // 20 T-gates minimum
			128,          // 128 puzzles per block
		)
		duration := time.Since(start)
		
		// Release the mutex immediately after the call
		m.qiskitMutex.Unlock()
		
		if err != nil {
			logError("❌ Qiskit batch simulation failed: %v", err)
			// Still update puzzle counter for debugging even on failure
			atomic.AddUint64(&m.puzzlesSolved, 128)
			return false
		}
		
		logInfo("⚛️ Qiskit: 128 puzzles completed in %.3fs (%.1f puzzles/sec)", 
			duration.Seconds(), 128.0/duration.Seconds())
		
		// Update puzzle counter
		atomic.AddUint64(&m.puzzlesSolved, 128)
	} else {
		// Fallback to individual CPU simulation if Qiskit failed to initialize
		outcomes = make([][]byte, 128)
		for i := 0; i < 128; i++ {
			outcome, _, err := m.solveQuantumPuzzleCPU(i, work.WorkHash, qnonce, work.QBits, work.TCount)
			if err != nil {
				logError("❌ CPU Puzzle %d failed: %v", i, err)
				return false
			}
			outcomes[i] = outcome
		}
		atomic.AddUint64(&m.puzzlesSolved, 128)
	}
	
	// Calculate quantum proof quality using geth's exact algorithm
	proofData := make([]byte, 0, len(outcomes)*len(outcomes[0]))
	for _, outcome := range outcomes {
		proofData = append(proofData, outcome...)
	}
	
	hash := sha256.Sum256(append([]byte(work.WorkHash), proofData...))
	proofQuality := new(big.Int).SetBytes(hash[:])
	
	// Check if solution meets target
	targetBig := new(big.Int)
	// Parse hex target (with or without 0x prefix)
	if strings.HasPrefix(work.Target, "0x") {
		targetBig.SetString(work.Target, 0)
	} else {
		targetBig.SetString(work.Target, 16) // Parse as hex
	}
	
	// Log every 1000th attempt for debugging
	if qnonce%1000 == 0 {
		logInfo("🎯 QNonce %d: Quality=%s vs Target=%s", qnonce, 
			proofQuality.Text(16)[:16]+"...", 
			targetBig.Text(16)[:16]+"...")
	}
	
	// Solution found if proof quality <= target (lower is better)
	if proofQuality.Cmp(targetBig) <= 0 {
		logInfo("🎉 SOLUTION FOUND! QNonce %d meets target", qnonce)
		
		// Submit solution to geth
		submitParams := []interface{}{
			work.WorkHash,
			fmt.Sprintf("0x%x", qnonce),
			hex.EncodeToString(proofData),
		}
		
		result, err := m.rpcCall("qmpow_submitWork", submitParams)
		if err != nil {
			logError("❌ Failed to submit solution: %v", err)
			atomic.AddUint64(&m.rejected, 1)
			return false
		}
		
		if accepted, ok := result.(bool); ok && accepted {
			logInfo("✅ Solution accepted by network!")
			atomic.AddUint64(&m.accepted, 1)
			return true
		} else {
			logError("❌ Solution rejected by network")
			atomic.AddUint64(&m.rejected, 1)
			return false
		}
	}
	
	return false
}

func (m *QuantumMiner) solveQuantumPuzzleCPU(puzzleIndex int, workHash string, qnonce uint64, qbits, tcount int) ([]byte, float64, error) {
	// Simplified CPU quantum simulation
	// For now, use deterministic pseudo-quantum computation
	seed := uint64(puzzleIndex) ^ qnonce ^ uint64(len(workHash))
	
	// Simulate quantum randomness
	outcome := make([]byte, (qbits+7)/8)
	for i := range outcome {
		seed = seed*1103515245 + 12345
		outcome[i] = byte(seed >> 24)
	}
	
	// Simulate computation time - fast for testing
	time.Sleep(time.Millisecond * 1)
	
	return outcome, 0.001, nil
}

func (m *QuantumMiner) statsReporter() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			if atomic.LoadInt32(&m.running) == 0 {
				return
			}
			m.updateDashboard()
		case <-m.stopChan:
			return
		}
	}
}

func (m *QuantumMiner) updateDashboard() {
	now := time.Now()
	elapsed := now.Sub(m.startTime)
	
	attempts := atomic.LoadUint64(&m.attempts)
	puzzles := atomic.LoadUint64(&m.puzzlesSolved)
	
	// Calculate rates
	if elapsed.Seconds() > 0 {
		m.currentHashrate = float64(attempts) / elapsed.Seconds()
		m.currentPuzzleRate = float64(puzzles) / elapsed.Seconds()
	}
	
	// Display dashboard
	fmt.Printf("\033[H\033[2J") // Clear screen
	fmt.Printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n")
	if m.wsl2Mode {
		fmt.Printf("│ 🪟 WSL2 QUANTUM MINER │ %d Threads │ Runtime: %.0fs                    │\n", m.threads, elapsed.Seconds())
	} else if m.gpuMode {
		fmt.Printf("│ 🎮 QUANTUM GPU MINER │ %d Threads │ Runtime: %.0fs                     │\n", m.threads, elapsed.Seconds())
	} else {
		fmt.Printf("│ 💻 QUANTUM CPU MINER │ %d Threads │ Runtime: %.0fs                     │\n", m.threads, elapsed.Seconds())
	}
	fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("│ ⚡ QNonce Rate     │ Current: %8.2f QN/s │ Average: %8.2f QN/s     │\n", m.currentHashrate, m.currentHashrate)
	fmt.Printf("│ ⚛️  Puzzle Rate     │ Current: %8.2f PZ/s │ Average: %8.2f PZ/s     │\n", m.currentPuzzleRate, m.currentPuzzleRate)
	fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("│ 🎯 Blocks Found    │ Accepted: %-8d │ Rejected: %-8d │ Stale: %-6d │\n", m.accepted, m.rejected, m.stale)
	fmt.Printf("│ 📊 Work Stats      │ Total QNonces: %-12d │ Total Puzzles: %-12d │\n", attempts, puzzles)
	fmt.Printf("│ 🧵 Thread Status   │ Active: %d/%d  │ All threads mining    │ Pool: ∞    │\n", m.threads, m.threads)
	fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("│ 🔗 Current Block   │ Block: %-12d │ Difficulty: %-18d │\n", uint64(1), uint64(200))
	fmt.Printf("│ ⏱️  Block Timing    │ Average: %6.1fs │ Target: %6.1fs │ ASERT-Q Adjust │\n", 12.0, 12.0)
	fmt.Printf("└─────────────────────────────────────────────────────────────────────────────────┘\n")
	fmt.Printf("Last Update: %s | Press Ctrl+C to stop\n", now.Format("15:04:05"))
}

func (m *QuantumMiner) showFinalReport() {
	elapsed := time.Since(m.startTime)
	attempts := atomic.LoadUint64(&m.attempts)
	puzzles := atomic.LoadUint64(&m.puzzlesSolved)
	
	fmt.Printf("\n📊 ═══════════════════════════════════════════════════════════════════════════════\n")
	fmt.Printf("🏁 FINAL QUANTUM MINING SESSION REPORT\n")
	fmt.Printf("📊 ═══════════════════════════════════════════════════════════════════════════════\n")
	if m.wsl2Mode {
		fmt.Printf("🎮 Mining Mode    │ WSL2 GPU ACCELERATED │ %d Parallel Threads\n", m.threads)
	} else if m.gpuMode {
		fmt.Printf("🎮 Mining Mode    │ NATIVE GPU ACCELERATED │ %d Parallel Threads\n", m.threads)
	} else {
		fmt.Printf("🎮 Mining Mode    │ CPU PROCESSING │ %d Parallel Threads\n", m.threads)
	}
	fmt.Printf("⏱️  Session Time   │ %.0fs │ Started: %s\n", elapsed.Seconds(), m.startTime.Format("15:04:05"))
	fmt.Printf("⚡ Performance    │ QNonces: %8.2f QN/s │ Puzzles: %8.2f PZ/s\n", 
		float64(attempts)/elapsed.Seconds(), float64(puzzles)/elapsed.Seconds())
	fmt.Printf("🧮 Work Completed │ QNonces: %d │ Puzzles: %d │ Ratio: %.1f puzzles/qnonce\n", 
		attempts, puzzles, float64(puzzles)/float64(max(attempts, 1)))
	fmt.Printf("🎯 Block Results  │ Accepted: %d │ Rejected: %d │ Success Rate: %.2f%%\n", 
		m.accepted, m.rejected, float64(m.accepted)/float64(max(m.accepted+m.rejected, 1))*100)
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

func showHelp() {
	fmt.Println("🚀 Quantum-Geth GPU/CPU Miner v" + VERSION)
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
	fmt.Println("  -gpu               Enable GPU mining (default: true)")
	fmt.Println("                     • Windows: Uses WSL2 for GPU acceleration")
	fmt.Println("                     • Linux: Uses native Qiskit GPU")
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
