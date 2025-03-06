#!/usr/bin/env bats

# Load test helpers
load test_helper

# Setup function runs before each test
setup() {
    # Create a temporary directory for test artifacts
    TEST_TEMP_DIR="$(mktemp -d)"

    # Export variables for use in tests
    export TEST_TEMP_DIR
}

# Teardown function runs after each test
teardown() {
    # Clean up temporary directory
    rm -rf "$TEST_TEMP_DIR"
}

# Test that test_docker.sh exists and is executable
@test "test_docker.sh exists and is executable" {
    assert_file_executable "scripts/test_docker.sh"
}

# Test that test_kubernetes.sh exists and is executable
@test "test_kubernetes.sh exists and is executable" {
    assert_file_executable "scripts/test_kubernetes.sh"
}

# Test that test_deployment.sh exists and is executable
@test "test_deployment.sh exists and is executable" {
    assert_file_executable "scripts/test_deployment.sh"
}

# Test that test_docker.sh has proper shebang
@test "test_docker.sh has proper shebang" {
    run head -n 1 "scripts/test_docker.sh"
    assert_output "#!/bin/bash"
}

# Test that test_kubernetes.sh has proper shebang
@test "test_kubernetes.sh has proper shebang" {
    run head -n 1 "scripts/test_kubernetes.sh"
    assert_output "#!/bin/bash"
}

# Test that test_deployment.sh has proper shebang
@test "test_deployment.sh has proper shebang" {
    run head -n 1 "scripts/test_deployment.sh"
    assert_output "#!/bin/bash"
}

# Test that test_deployment.sh --help works
@test "test_deployment.sh --help works" {
    run scripts/test_deployment.sh --help
    assert_success
    assert_output --partial "Usage:"
}

# Test that test_deployment.sh with invalid argument fails
@test "test_deployment.sh with invalid argument fails" {
    run scripts/test_deployment.sh --invalid-arg
    assert_failure
    assert_output --partial "Unknown argument"
}

# Helper functions
assert_file_executable() {
    [ -x "$1" ]
}
