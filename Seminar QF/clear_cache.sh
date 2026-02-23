#!/bin/bash
# Clear all Python cache files to ensure fresh imports

echo "🧹 Clearing Python cache..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Clear __pycache__ directories
echo "  Removing __pycache__ directories..."
find "$SCRIPT_DIR" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null

# Clear .pyc files
echo "  Removing .pyc files..."
find "$SCRIPT_DIR" -name '*.pyc' -delete 2>/dev/null

# Clear .pyo files
echo "  Removing .pyo files..."
find "$SCRIPT_DIR" -name '*.pyo' -delete 2>/dev/null

echo "✅ Cache cleared successfully!"
echo ""
echo "Now:"
echo "  1. Restart your Jupyter kernel"
echo "  2. Clear all outputs"
echo "  3. Run all cells from the top"
