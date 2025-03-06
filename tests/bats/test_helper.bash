#!/usr/bin/env bash

# Set the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Load bats-support and bats-assert if available
if [ -d "${PROJECT_ROOT}/tests/bats/libs/bats-support" ]; then
    load "${PROJECT_ROOT}/tests/bats/libs/bats-support/load.bash"
fi

if [ -d "${PROJECT_ROOT}/tests/bats/libs/bats-assert" ]; then
    load "${PROJECT_ROOT}/tests/bats/libs/bats-assert/load.bash"
fi

# Custom assertions

# Assert that a command succeeds
assert_success() {
    [ "$status" -eq 0 ]
}

# Assert that a command fails
assert_failure() {
    [ "$status" -ne 0 ]
}

# Assert that the output contains a string
assert_output() {
    local expected
    if [ $# -eq 0 ]; then
        expected="$(cat -)"
    else
        expected="$1"
    fi
    [ "$output" = "$expected" ]
}

# Assert that the output contains a partial string
assert_output_contains() {
    local expected="$1"
    [[ $output == *"$expected"* ]]
}

# Assert that a file exists
assert_file_exists() {
    [ -f "$1" ]
}

# Assert that a directory exists
assert_dir_exists() {
    [ -d "$1" ]
}

# Assert that a file is executable
assert_file_executable() {
    [ -x "$1" ]
}
