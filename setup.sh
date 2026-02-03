#!/bin/bash
#
# SpeechLift - Setup Script
# Usage: source setup.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Symbols
CHECK="[${GREEN}✓${NC}]"
CROSS="[${RED}✗${NC}]"
WARN="[${YELLOW}!${NC}]"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}   ${BOLD}SpeechLift - Setup${NC}              ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo -e "${CROSS} Python 3 not found"
    echo -e "    Please install Python 3.10 or higher"
    return 1 2>/dev/null || exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${CHECK} Python ${PYTHON_VERSION} detected"

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo -ne "    Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo -e "\r${CROSS} Failed to create virtual environment"
        return 1 2>/dev/null || exit 1
    fi
    echo -e "\r${CHECK} Virtual environment created            "
else
    echo -e "${CHECK} Virtual environment found"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${CROSS} Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi

# Upgrade pip quietly
echo -ne "    Upgrading pip..."
pip install --upgrade pip -q 2>/dev/null
echo -e "\r${CHECK} pip upgraded                           "

# Install dependencies
echo -ne "    Installing dependencies (this may take a moment)..."
pip install -r "$SCRIPT_DIR/requirements.txt" -q 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "\r${CROSS} Failed to install dependencies                    "
    echo "    Run: pip install -r requirements.txt"
    return 1 2>/dev/null || exit 1
fi
echo -e "\r${CHECK} Dependencies installed                              "

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo -e "${CHECK} FFmpeg found"
else
    echo -e "${WARN} FFmpeg not installed"
    echo -e "    Install with: ${CYAN}brew install ffmpeg${NC}"
fi

# Success message
echo ""
echo -e "${CYAN}──────────────────────────────────────────────────${NC}"
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo -e "${CYAN}──────────────────────────────────────────────────${NC}"
echo ""
echo -e "Run the transcriber:  ${CYAN}python main.py${NC}"
echo ""
echo -e "Next session:         ${CYAN}source venv/bin/activate${NC}"
echo ""
