#!/bin/bash
# Script to free up disk space on GitHub Actions runners

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
