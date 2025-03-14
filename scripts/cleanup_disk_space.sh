#!/bin/bash
# Script to free up disk space on GitHub Actions runners

# Display usage information
usage() {
    echo "Usage: $(basename "$0")"
    echo
    echo "Free up disk space on GitHub Actions runners by removing unnecessary tools and packages."
    echo "This script is particularly useful for CI environments with limited disk space."
    echo
    echo "The script performs the following actions:"
    echo "  - Removes .NET, Android SDK, and other large toolsets"
    echo "  - Prunes Docker images"
    echo "  - Cleans apt cache"
    echo
    echo "No options or arguments are required."
}

# Parse command line arguments
if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
    exit 0
fi

echo "Freeing up disk space before Docker build..."
df -h

# Remove unnecessary tools and packages
sudo rm -rf /usr/share/dotnet
sudo rm -rf /usr/local/lib/android
sudo rm -rf /opt/ghc
sudo rm -rf /opt/hostedtoolcache/CodeQL

# Clean Docker resources
sudo docker image prune -af

# Clean apt cache
sudo apt-get clean

echo "Disk space after cleanup:"
df -h
