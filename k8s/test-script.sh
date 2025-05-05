#!/bin/bash
set -e

# Configurable test parameters
SERVICE_NAME="llm-rag"
HEALTH_ENDPOINT="http://${SERVICE_NAME}/health"
QUERY_ENDPOINT="http://${SERVICE_NAME}/query"
INITIAL_WAIT=5
MAX_RETRIES=60
RETRY_INTERVAL=1

# Function for better logging with timestamps
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check service health
check_health() {
    curl --output /dev/null --silent --fail "${HEALTH_ENDPOINT}"
    return $?
}

# Wait for initial startup
log "Waiting ${INITIAL_WAIT}s for initial service startup..."
sleep ${INITIAL_WAIT}

# Wait for the service to be ready with better retry logic
log "Testing connectivity to ${SERVICE_NAME} service..."
counter=0

until check_health; do
    counter=$((counter + 1))
    if [ $counter -eq $MAX_RETRIES ]; then
        log "❌ ERROR: Timed out waiting for service to be ready after ${MAX_RETRIES} attempts"
        # Get service details for diagnostics
        log "--- Service details ---"
        kubectl get service ${SERVICE_NAME} -o yaml 2>/dev/null || log "Service not found"
        log "--- Endpoints ---"
        kubectl get endpoints ${SERVICE_NAME} -o yaml 2>/dev/null || log "Endpoints not found"
        log "--- Pod status ---"
        kubectl get pods -l app=${SERVICE_NAME} -o wide 2>/dev/null || log "No pods found"
        exit 1
    fi
    log "Waiting for service to be ready... ($counter/$MAX_RETRIES)"
    sleep ${RETRY_INTERVAL}
done

log "✅ Service is reachable!"

# Test health endpoint with proper validation
log "Testing health endpoint..."
health_response=$(curl -s ${HEALTH_ENDPOINT})
log "Health endpoint response: ${health_response}"

if ! echo "${health_response}" | grep -q -i "healthy\|ok\|ready\|live"; then
    log "❌ Health check response doesn't indicate service is healthy"
    exit 1
fi

# Test the API with more robust validation
log "Testing API with query: 'What is RAG?'"
query_response=$(curl -s -X POST ${QUERY_ENDPOINT} \
    -H "Content-Type: application/json" \
    -d '{"query": "What is RAG?", "top_k": 1}')

# Check if curl command failed
if [ $? -ne 0 ]; then
    log "❌ Failed to connect to query endpoint"
    exit 1
fi

log "Query response received (truncated): ${query_response:0:100}..."

# More robust JSON validation
if echo "${query_response}" | grep -q "response"; then
    log "✅ TEST PASSED: API returned valid response"

    # Extract response length for verification
    response_length=$(echo "${query_response}" | grep -o '"response"' | wc -l)
    log "Response contains ${response_length} response field(s)"

    # Print summary stats from response
    log "Response size: $(echo -n "${query_response}" | wc -c) bytes"

    exit 0
else
    log "❌ TEST FAILED: API response missing expected 'response' field"
    log "Full response: ${query_response}"
    exit 1
fi
