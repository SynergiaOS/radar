#!/bin/bash
# WIG30 Frontend Startup Script

set -e

echo "🎯 WIG30 Frontend Startup Script v2.0"
echo "========================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js nie jest zainstalowany"
    echo "📦 Proszę zainstalować Node.js: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm nie jest zainstalowany"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Project directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/gpw-smart-analyzer"

echo "📁 Project root: $PROJECT_ROOT"
echo "🎨 Frontend directory: $FRONTEND_DIR"

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Nie znaleziono katalogu frontend: $FRONTEND_DIR"
    exit 1
fi

# Change to frontend directory
cd "$FRONTEND_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Instalowanie zależności..."
    npm install
    echo "✅ Zależności zainstalowane"
else
    echo "✅ Zależności już zainstalowane"
fi

echo ""
echo "🚀 Uruchamiam WIG30 Frontend (Next.js)..."
echo "🌐 Frontend będzie dostępny na: http://localhost:3001"
echo "========================================"

# Start Next.js development server
npm run dev