#!/bin/bash
set -e

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check if shellcheck is installed
if ! command -v shellcheck &>/dev/null; then
    print_error "shellcheck is not installed. Please install it first."
    echo "  brew install shellcheck (macOS)"
    echo "  apt-get install shellcheck (Ubuntu/Debian)"
    echo "  or visit https://github.com/koalaman/shellcheck#installing"
    exit 1
fi

# Check if bats is installed (optional)
BATS_AVAILABLE=false
if command -v bats &>/dev/null; then
    BATS_AVAILABLE=true
    print_header "bats is available, will use it for testing"
else
    print_warning "bats is not installed. Will use basic testing only."
    echo "For more thorough testing, consider installing bats:"
    echo "  brew install bats-core (macOS)"
    echo "  or visit https://github.com/bats-core/bats-core#installation"
fi

# Find all bash scripts
print_header "Finding bash scripts"
SCRIPT_DIR="scripts"
BASH_SCRIPTS=$(find "${SCRIPT_DIR}" -name "*.sh" -type f)

if [ -z "${BASH_SCRIPTS}" ]; then
    print_warning "No bash scripts found in ${SCRIPT_DIR}"
    exit 0
fi

# Run shellcheck on all bash scripts
print_header "Running shellcheck on bash scripts"
for script in ${BASH_SCRIPTS}; do
    echo "Checking ${script}..."
    shellcheck -x "${script}" || print_error "shellcheck failed for ${script}"
done

# Run basic tests on all bash scripts
print_header "Running basic tests on bash scripts"
for script in ${BASH_SCRIPTS}; do
    echo "Testing ${script}..."

    # Check if script is executable
    if [ ! -x "${script}" ]; then
        print_warning "${script} is not executable. Setting executable permission."
        chmod +x "${script}"
    fi

    # Check if script has a shebang
    if ! head -n 1 "${script}" | grep -q "^#!/bin/.*sh"; then
        print_error "${script} does not have a proper shebang. It should start with #!/bin/bash or #!/bin/sh"
    fi

    # Check if script has help/usage information
    if ! grep -q "\-\-help" "${script}" && ! grep -q "Usage:" "${script}"; then
        print_warning "${script} does not appear to have help/usage information. Consider adding it."
    fi
done

# Run bats tests if available
if [ "${BATS_AVAILABLE}" = true ] && [ -d "tests/bats" ]; then
    print_header "Running bats tests"
    bats tests/bats/*.bats
else
    if [ "${BATS_AVAILABLE}" = true ]; then
        print_warning "No bats tests found in tests/bats directory."
        echo "Consider adding bats tests for your bash scripts."
    fi
fi

print_header "All bash script tests completed"
