# Contributing to Q Coin

Thank you for your interest in contributing to Q Coin! This guide will help you get started with development and contributions.

## 🚀 Quick Start for Contributors

### Development Environment Setup
```bash
# Clone the repository
git clone https://github.com/fourtytwo42/Qgeth3.git
cd Qgeth3

# Install dependencies
sudo apt install golang-go python3 python3-pip git build-essential  # Linux
# Or follow the Installation Guide for Windows

# Build development version
./quick-start.sh build

# Test your setup
./quick-start.sh start
./quick-start.sh start-mining
```

### Quantum Wallet Development
For developing the quantum wallet UI:
```bash
# IMPORTANT: Must run from quantum-wallet directory
cd quantum-wallet
wails dev

# NOT from root directory (will fail):
# wails dev  # ❌ ERROR: wails.json not found
```

**PowerShell (Windows):**
```powershell
# Use semicolon for command chaining in PowerShell
cd quantum-wallet; wails dev

# NOT && (doesn't work in PowerShell):
# cd quantum-wallet && wails dev  # ❌ ERROR
```

### Testing Your Changes
```bash
# Run unit tests
cd quantum-geth && go test ./...
cd quantum-miner && go test ./...

# Test build system
./scripts/linux/build-linux.sh both --clean

# Test on different platforms
./scripts/windows/build-release.ps1  # Windows
```

## 🛠️ Development Guidelines

### Code Style

#### Go Code Style
Follow standard Go conventions:
```go
// Good: Clear function names and documentation
// ValidateQuantumProof validates a quantum proof-of-work solution
func ValidateQuantumProof(proof *QuantumProof, target *big.Int) bool {
    // Implementation
}

// Good: Proper error handling
result, err := processQuantumCircuit(circuit)
if err != nil {
    return nil, fmt.Errorf("failed to process quantum circuit: %w", err)
}

// Good: Consistent naming
type QuantumBlock struct {
    Header       *types.Header
    QBlob        []byte
    Puzzles      []QuantumPuzzle
    Transactions types.Transactions
}
```

#### Shell Script Style
```bash
#!/bin/bash
# Use proper error handling
set -e

# Clear function documentation
# build_geth builds the quantum-geth binary
build_geth() {
    local target="$1"
    echo "🏗️ Building quantum-geth..."
    
    if ! command -v go >/dev/null; then
        echo "❌ Go compiler not found"
        return 1
    fi
    
    # Implementation
}

# Use consistent variable naming
GETH_BINARY="./geth.bin"
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%d_%H:%M:%S")
```

#### Documentation Style
- Use clear, descriptive headings
- Include code examples for all features
- Test all code examples before committing
- Link related documentation sections
- Include troubleshooting for common issues

### Git Workflow

#### Branching Strategy
```bash
# Create feature branch
git checkout -b feature/quantum-difficulty-adjustment

# Create bugfix branch
git checkout -b bugfix/mining-connection-timeout

# Create documentation branch
git checkout -b docs/update-vps-guide
```

#### Commit Messages
```bash
# Good commit messages
git commit -m "consensus: implement ASERT-Q difficulty adjustment algorithm

- Add quantum-specific difficulty calculation
- Include time-based adjustments for quantum block validation
- Maintain compatibility with existing consensus interface"

git commit -m "scripts: fix Go temp directory creation in build-linux.sh

- Ensure GOCACHE, GOTMPDIR, and TMPDIR exist before build
- Prevents 'no such file or directory' errors on VPS
- Fixes issue #123"

git commit -m "docs: add GPU mining troubleshooting section

- Cover CUDA installation issues
- Add Python dependency troubleshooting
- Include performance optimization tips"
```

## 🏗️ Architecture Guidelines

### Adding New Features

#### Quantum Consensus Features
When modifying consensus logic:
```go
// quantum-geth/consensus/qmpow/
├── consensus.go          // Main consensus interface
├── difficulty.go         // ASERT-Q difficulty adjustment
├── quantum_validation.go // Quantum proof validation
└── rewards.go           // Block reward calculation

// Always maintain interface compatibility
type Engine interface {
    Author(header *types.Header) (common.Address, error)
    VerifyHeader(chain ChainHeaderReader, header *types.Header) error
    Prepare(chain ChainHeaderReader, header *types.Header) error
    Finalize(chain ChainHeaderReader, header *types.Header, state *state.StateDB, txs []*types.Transaction) error
    // Add new methods carefully
}
```

#### Mining Features
When adding mining functionality:
```go
// quantum-miner/internal/miner/
├── miner.go             // Main mining logic
├── quantum.go           // Quantum circuit handling
├── gpu.go              // GPU acceleration
└── worker.go           // Mining worker management

// Use consistent error handling
type MiningError struct {
    Type    string
    Message string
    Cause   error
}

func (e *MiningError) Error() string {
    if e.Cause != nil {
        return fmt.Sprintf("%s: %s (caused by: %v)", e.Type, e.Message, e.Cause)
    }
    return fmt.Sprintf("%s: %s", e.Type, e.Message)
}
```

### Script Development

#### Adding New Scripts
```bash
# Follow naming convention
scripts/
├── linux/
│   └── your-new-script.sh      # Linux-specific
├── windows/
│   └── your-new-script.ps1     # Windows-specific
└── deployment/
    └── your-deployment-tool.sh  # Cross-platform deployment

# Include proper headers
#!/bin/bash
# your-new-script.sh - Brief description of what this script does
# Usage: ./your-new-script.sh [options]
# Author: Your Name <your.email@example.com>
```

#### Script Best Practices
```bash
# Use functions for reusable code
check_dependencies() {
    local missing_deps=()
    
    for cmd in go git python3; do
        if ! command -v "$cmd" >/dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "❌ Missing dependencies: ${missing_deps[*]}"
        return 1
    fi
}

# Provide helpful output
print_step() {
    echo "🔄 $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1" >&2
}
```

## 🧪 Testing Guidelines

### Unit Testing
```go
// quantum-geth/consensus/qmpow/difficulty_test.go
func TestQuantumDifficultyAdjustment(t *testing.T) {
    tests := []struct {
        name           string
        parentTime     uint64
        currentTime    uint64
        parentDiff     *big.Int
        expectedDiff   *big.Int
    }{
        {
            name:         "normal adjustment",
            parentTime:   1000,
            currentTime:  1012, // 12 second block time
            parentDiff:   big.NewInt(1000),
            expectedDiff: big.NewInt(1000), // Should remain stable
        },
        // Add more test cases
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test implementation
        })
    }
}
```

### Integration Testing
```bash
# scripts/test-integration.sh
#!/bin/bash
set -e

echo "🧪 Running integration tests..."

# Test build system
./quick-start.sh build

# Test node startup
timeout 30s ./scripts/linux/start-geth.sh testnet &
sleep 10

# Test RPC API
if curl -s -X POST -H "Content-Type: application/json" \
   --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
   http://localhost:8545 | grep -q "Geth"; then
    echo "✅ RPC API responding"
else
    echo "❌ RPC API not responding"
    exit 1
fi

echo "✅ Integration tests passed"
```

### Performance Testing
```bash
# Test mining performance
echo "⚡ Testing mining performance..."
timeout 60s ./scripts/linux/start-miner.sh --testnet --verbose | grep "puzzles/sec" | tail -5

# Test resource usage
echo "📊 Resource usage during mining:"
ps aux | grep -E "(geth|quantum-miner)" | awk '{print $3, $4, $11}'
```

## 📚 Documentation Standards

### Code Documentation
```go
// Package qmpow implements the Quantum Modified Proof-of-Work consensus algorithm.
//
// QMPoW combines traditional proof-of-work with quantum circuit validation,
// providing quantum-resistant security while maintaining Bitcoin-style mining.
package qmpow

// QuantumProof represents a solution to a quantum proof-of-work puzzle.
type QuantumProof struct {
    // CircuitResult contains the quantum circuit execution result
    CircuitResult []byte `json:"circuitResult"`
    
    // Nonce is the traditional proof-of-work nonce
    Nonce uint64 `json:"nonce"`
    
    // QuantumNonce provides additional entropy for quantum validation
    QuantumNonce []byte `json:"quantumNonce"`
}

// ValidateProof validates a quantum proof against the given target difficulty.
//
// The validation process:
//  1. Verifies the circuit result format and structure
//  2. Reconstructs the quantum circuit from the proof
//  3. Validates the circuit execution matches the provided result
//  4. Checks that the combined hash meets the target difficulty
//
// Returns true if the proof is valid, false otherwise.
func ValidateProof(proof *QuantumProof, target *big.Int) bool {
    // Implementation
}
```

### User Documentation
- Always include working examples
- Test examples on clean systems
- Include common error scenarios
- Link to related documentation
- Keep instructions up to date

## 🔧 Build System Contributions

### Adding Platform Support
```bash
# scripts/linux/build-your-platform.sh
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux" ;;
        Darwin*)    PLATFORM="macos" ;;
        CYGWIN*)    PLATFORM="windows" ;;
        *)          PLATFORM="unknown" ;;
    esac
    echo "Platform detected: $PLATFORM"
}

# Add GPU detection for new platforms
detect_gpu_capabilities() {
    case "$PLATFORM" in
        "linux")   detect_linux_gpu ;;
        "macos")   detect_macos_gpu ;;
        "windows") detect_windows_gpu ;;
        *)         echo "GPU detection not implemented for $PLATFORM" ;;
    esac
}
```

### Build System Testing
```bash
# Test all build targets
for target in geth miner both; do
    echo "Testing build target: $target"
    ./scripts/linux/build-linux.sh "$target" --clean
    
    # Verify binaries were created
    case "$target" in
        "geth") test -f ./geth.bin ;;
        "miner") test -f ./quantum-miner ;;
        "both") test -f ./geth.bin && test -f ./quantum-miner ;;
    esac
done
```

## 🐛 Bug Reports and Issues

### Bug Report Template
```markdown
## Bug Description
Brief description of the issue

## Environment
- OS: Linux Ubuntu 22.04
- Go Version: 1.21.5
- GPU: NVIDIA RTX 3080
- Build Target: geth

## Steps to Reproduce
1. Clone repository
2. Run `./quick-start.sh build`
3. Run `./scripts/linux/start-geth.sh testnet`
4. Observe error

## Expected Behavior
Node should start and begin syncing

## Actual Behavior
Node crashes with "quantum validation failed"

## Logs
```
[logs here]
```

## Additional Context
Any other relevant information
```

### Feature Request Template
```markdown
## Feature Description
Brief description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
High-level approach to implementing this feature

## Alternatives Considered
Other ways to solve this problem

## Additional Context
Any other relevant information
```

## 🚀 Release Process

### Version Numbering
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes to consensus or API
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, documentation updates

### Release Checklist
```bash
# 1. Update version numbers
# 2. Update CHANGELOG.md
# 3. Test on all platforms
./scripts/test-all-platforms.sh

# 4. Create release binaries
./scripts/linux/build-linux.sh both --release
./scripts/windows/build-release.ps1

# 5. Test release binaries
./test-release-binaries.sh

# 6. Tag release
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3

# 7. Create GitHub release with binaries
# 8. Update documentation
```

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and development discussion
- **Discord**: Real-time chat (link in main README)
- **Email**: Direct contact for security issues

### Code Review Process
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request
5. Address review feedback
6. Maintainer approval and merge

### Coding Questions
- Check existing documentation first
- Search GitHub issues for similar questions
- Include relevant code examples in questions
- Provide context about what you're trying to achieve

## ✅ Contributor Checklist

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Documentation updated for new features
- [ ] Changes tested on target platforms
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch

### Pull Request
- [ ] Clear description of changes
- [ ] Links to related issues
- [ ] Screenshots for UI changes
- [ ] Breaking changes documented
- [ ] Performance impact considered

### After Submission
- [ ] Respond to review feedback promptly
- [ ] Keep branch up to date with main
- [ ] Test final merged result
- [ ] Celebrate your contribution! 🎉

Thank you for contributing to Q Coin! Your contributions help make quantum blockchain technology accessible to everyone. 