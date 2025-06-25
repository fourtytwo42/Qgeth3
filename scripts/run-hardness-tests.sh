#!/bin/bash
# QPoW Hardness and Security Test Suite Runner
# Comprehensive testing of all attack vectors and security assumptions

# Default parameters
TEST_CATEGORY="all"
VERBOSE=false
FAIL_FAST=false
OUTPUT_FORMAT="console"

# Test configuration
TEST_ROOT="tests/hardness"
START_TIME=$(date +%s)
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
ERROR_TESTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;37m'
DARK_GRAY='\033[1;30m'
NC='\033[0m' # No Color

# Function to show help
show_help() {
    echo "QPoW Hardness and Security Test Suite"
    echo ""
    echo "Usage:"
    echo "  $0 [options]"
    echo ""
    echo "Options:"
    echo "  --category <cat>       Test category: all, puzzle, gatehash, merkle, proofs, quality (default: all)"
    echo "  --verbose              Show detailed test output"
    echo "  --fail-fast            Stop on first test failure"
    echo "  --output <format>      Output format: console, json (default: console)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Test Categories:"
    echo "  puzzle     - Quantum Puzzle Execution (deterministic execution, branch-serial)"
    echo "  gatehash   - Canonical-Compile and GateHash (QASM compilation, collision resistance)"
    echo "  merkle     - OutcomeRoot and BranchNibbles (Merkle consistency, nibble extraction)"
    echo "  proofs     - Proof Generation and Verification (Mahadev, CAPSS, Nova)"
    echo "  quality    - ProofQuality and Target Test (quality computation, difficulty validation)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests"
    echo "  $0 --category puzzle         # Run only puzzle tests"
    echo "  $0 --verbose --fail-fast     # Verbose output, stop on first failure"
    echo ""
}

# Function to write test header
write_test_header() {
    local category="$1"
    local name="$2"
    local description="$3"
    
    echo ""
    echo -e "${DARK_GRAY}============================================================================${NC}"
    echo -e "${CYAN}$name${NC}"
    echo -e "${GRAY}$description${NC}"
    echo -e "${DARK_GRAY}============================================================================${NC}"
}

# Function to run a single test
run_test() {
    local test_id="$1"
    local test_name="$2"
    local category="$3"
    
    local test_start=$(date +%s.%3N)
    
    # Check if test script exists
    local test_script="$TEST_ROOT/$category/${test_id}.go"
    
    if [ -f "$test_script" ]; then
        echo -n -e "Testing $test_id : $test_name"
        
        # Run Go test
        local result
        result=$(go test -run "$test_id" "$TEST_ROOT/$category" -v 2>&1)
        local exit_code=$?
        
        local test_end=$(date +%s.%3N)
        local duration=$(echo "$test_end - $test_start" | bc -l)
        local duration_ms=$(echo "$duration * 1000" | bc -l | cut -d. -f1)
        
        if [ $exit_code -eq 0 ]; then
            echo -e " ${GREEN}PASS${NC}"
            echo -e "   ${GRAY}Duration: ${duration_ms}ms${NC}"
            
            ((PASSED_TESTS++))
            
            if [ "$VERBOSE" = true ]; then
                echo -e "   ${GRAY}Output: $result${NC}"
            fi
            
            return 0
        else
            echo -e " ${RED}FAIL${NC}"
            echo -e "   ${GRAY}Duration: ${duration_ms}ms${NC}"
            
            ((FAILED_TESTS++))
            
            if [ "$VERBOSE" = true ]; then
                echo -e "   ${RED}Error: $result${NC}"
            fi
            
            if [ "$FAIL_FAST" = true ]; then
                echo "Test $test_id failed, stopping due to --fail-fast"
                exit 1
            fi
            
            return 1
        fi
    else
        echo -n -e "Testing $test_id : $test_name"
        echo -e " ${YELLOW}SKIP (Not implemented)${NC}"
        
        ((SKIPPED_TESTS++))
        
        return 0
    fi
    
    ((TOTAL_TESTS++))
}

# Function to show test summary
show_test_summary() {
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    local total_tests=$((PASSED_TESTS + FAILED_TESTS + SKIPPED_TESTS + ERROR_TESTS))
    
    echo ""
    echo -e "${DARK_GRAY}============================================================================${NC}"
    echo -e "${CYAN}QPoW Hardness and Security Test Results${NC}"
    echo -e "${DARK_GRAY}============================================================================${NC}"
    
    echo -e "Total Tests:   $total_tests"
    echo -e "${GREEN}Passed:        $PASSED_TESTS${NC}"
    echo -e "${RED}Failed:        $FAILED_TESTS${NC}"
    echo -e "${YELLOW}Skipped:       $SKIPPED_TESTS${NC}"
    echo -e "${MAGENTA}Errors:        $ERROR_TESTS${NC}"
    echo -e "${GRAY}Total Time:    ${total_duration}s${NC}"
    
    local success_rate=0
    if [ $total_tests -gt 0 ]; then
        success_rate=$(echo "scale=1; ($PASSED_TESTS * 100) / $total_tests" | bc -l)
    fi
    
    if (( $(echo "$success_rate == 100" | bc -l) )); then
        echo -e "${GREEN}Success Rate:  ${success_rate}%${NC}"
    elif (( $(echo "$success_rate >= 80" | bc -l) )); then
        echo -e "${YELLOW}Success Rate:  ${success_rate}%${NC}"
    else
        echo -e "${RED}Success Rate:  ${success_rate}%${NC}"
    fi
    
    echo ""
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -gt 0 ] || [ $ERROR_TESTS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --output)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}QPoW Hardness and Security Test Suite${NC}"
echo -e "${YELLOW}Testing quantum proof-of-work security assumptions${NC}"
echo -e "${GREEN}Test Category: $TEST_CATEGORY${NC}"
echo ""

# Check if Go is available
if ! command -v go &> /dev/null; then
    echo -e "${RED}ERROR: Go is not installed or not in PATH${NC}"
    echo "Please install Go 1.19 or later"
    exit 1
fi

# Check if bc is available for floating point calculations
if ! command -v bc &> /dev/null; then
    echo -e "${RED}ERROR: bc is not installed${NC}"
    echo "Please install bc for floating point calculations"
    exit 1
fi

# Check if test directory exists
if [ ! -d "$TEST_ROOT" ]; then
    echo -e "${RED}ERROR: Test directory $TEST_ROOT not found${NC}"
    exit 1
fi

# Define test categories and their tests
declare -A TEST_CATEGORIES

# Run tests based on category
case "$TEST_CATEGORY" in
    "puzzle")
        write_test_header "puzzle" "Quantum Puzzle Execution" "Deterministic execution, branch-serial enforcement, nonce wraparound"
        run_test "PZ-01" "Deterministic Puzzle Execution" "puzzle"
        run_test "PZ-02" "Branch-Serial Enforcement" "puzzle"
        run_test "PZ-03" "Nonce Wraparound Handling" "puzzle"
        ;;
    "gatehash")
        write_test_header "gatehash" "Canonical-Compile and GateHash" "QASM compilation, Z-mask application, collision resistance"
        run_test "GH-01" "QASM Compilation Consistency" "gatehash"
        run_test "GH-02" "Z-Mask Application" "gatehash"
        run_test "GH-03" "GateHash Collision Resistance" "gatehash"
        ;;
    "merkle")
        write_test_header "merkle" "OutcomeRoot and BranchNibbles" "Merkle tree consistency, nibble extraction, proof verification"
        run_test "MR-01" "Merkle Tree Consistency" "merkle"
        run_test "MR-02" "BranchNibbles Extraction" "merkle"
        run_test "MR-03" "Merkle Proof Verification" "merkle"
        ;;
    "proofs")
        write_test_header "proofs" "Proof Generation and Verification" "Mahadev traces, CAPSS SNARKs, Nova aggregation, ProofRoot binding"
        run_test "PR-01" "Mahadev Trace Generation" "proofs"
        run_test "PR-02" "CAPSS SNARK Proofs" "proofs"
        run_test "PR-03" "Nova Proof Aggregation" "proofs"
        run_test "PR-04" "ProofRoot Binding" "proofs"
        ;;
    "quality")
        write_test_header "quality" "ProofQuality and Target Test" "Quality computation, boundary conditions, difficulty validation"
        run_test "PQ-01" "Quality Computation" "quality"
        run_test "PQ-02" "Boundary Conditions" "quality"
        run_test "PQ-03" "Difficulty Validation" "quality"
        run_test "PQ-04" "Target Comparison" "quality"
        ;;
    "all")
        # Run all test categories
        write_test_header "puzzle" "Quantum Puzzle Execution" "Deterministic execution, branch-serial enforcement, nonce wraparound"
        run_test "PZ-01" "Deterministic Puzzle Execution" "puzzle"
        run_test "PZ-02" "Branch-Serial Enforcement" "puzzle"
        run_test "PZ-03" "Nonce Wraparound Handling" "puzzle"
        
        write_test_header "gatehash" "Canonical-Compile and GateHash" "QASM compilation, Z-mask application, collision resistance"
        run_test "GH-01" "QASM Compilation Consistency" "gatehash"
        run_test "GH-02" "Z-Mask Application" "gatehash"
        run_test "GH-03" "GateHash Collision Resistance" "gatehash"
        
        write_test_header "merkle" "OutcomeRoot and BranchNibbles" "Merkle tree consistency, nibble extraction, proof verification"
        run_test "MR-01" "Merkle Tree Consistency" "merkle"
        run_test "MR-02" "BranchNibbles Extraction" "merkle"
        run_test "MR-03" "Merkle Proof Verification" "merkle"
        
        write_test_header "proofs" "Proof Generation and Verification" "Mahadev traces, CAPSS SNARKs, Nova aggregation, ProofRoot binding"
        run_test "PR-01" "Mahadev Trace Generation" "proofs"
        run_test "PR-02" "CAPSS SNARK Proofs" "proofs"
        run_test "PR-03" "Nova Proof Aggregation" "proofs"
        run_test "PR-04" "ProofRoot Binding" "proofs"
        
        write_test_header "quality" "ProofQuality and Target Test" "Quality computation, boundary conditions, difficulty validation"
        run_test "PQ-01" "Quality Computation" "quality"
        run_test "PQ-02" "Boundary Conditions" "quality"
        run_test "PQ-03" "Difficulty Validation" "quality"
        run_test "PQ-04" "Target Comparison" "quality"
        ;;
    *)
        echo -e "${RED}ERROR: Invalid test category '$TEST_CATEGORY'${NC}"
        echo "Valid categories: all, puzzle, gatehash, merkle, proofs, quality"
        exit 1
        ;;
esac

show_test_summary 