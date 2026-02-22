#!/bin/bash
# GhostReply Installer
# One-liner: curl -sL hrampell.github.io/ghostreply/install.sh | bash
set -e

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║     GhostReply Installer v1.0    ║"
echo "  ║   iMessage Auto-Reply on Mac     ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: GhostReply only works on macOS (needs iMessage)."
    exit 1
fi

# --- Switch Terminal to dark profile ---
# Default Terminal.app is white — GhostReply looks best on dark background
if [[ "$TERM_PROGRAM" == "Apple_Terminal" ]]; then
    current_profile=$(defaults read com.apple.Terminal "Default Window Settings" 2>/dev/null || echo "")
    if [[ "$current_profile" != "Pro" && "$current_profile" != "Homebrew" && "$current_profile" != "Novel" ]]; then
        echo "  Switching Terminal to dark mode for best experience..."
        osascript -e 'tell application "Terminal" to set current settings of front window to settings set "Pro"' 2>/dev/null || true
        echo ""
    fi
fi

# --- Check Python 3 (without triggering Xcode dev tools popup) ---
PYTHON=""

# 1. Check Homebrew python first
if command -v /usr/local/bin/python3 &>/dev/null; then
    PYTHON="/usr/local/bin/python3"
elif command -v /opt/homebrew/bin/python3 &>/dev/null; then
    PYTHON="/opt/homebrew/bin/python3"
fi

# 2. Check if system python3 actually works (not just the Xcode shim)
if [[ -z "$PYTHON" ]]; then
    if python3 --version &>/dev/null 2>&1; then
        PYTHON="python3"
    fi
fi

# 3. No working python3 found
if [[ -z "$PYTHON" ]]; then
    echo "  Python 3 is required but not installed."
    echo ""
    echo "  Easiest way to install:"
    echo "    1. Go to python.org/downloads"
    echo "    2. Download the macOS installer"
    echo "    3. Run it (takes 1 minute)"
    echo "    4. Come back and run this install command again"
    echo ""
    echo "  Or if you have Homebrew: brew install python3"
    echo ""

    # Offer to open python.org
    read -p "  Open python.org now? (y/n): " open_python </dev/tty
    if [[ "$open_python" == "y" || "$open_python" == "Y" ]]; then
        open "https://www.python.org/downloads/"
    fi
    exit 1
fi

# Find pip for the same python
PIP="${PYTHON} -m pip"

echo "[1/4] Creating ~/.ghostreply directory..."
mkdir -p ~/.ghostreply

echo "[2/4] Downloading GhostReply..."
curl -sL https://raw.githubusercontent.com/Hrampell/ghostreply/main/client/ghostreply.py -o ~/.ghostreply/ghostreply.py

echo "[3/4] Installing dependencies..."
$PIP install --break-system-packages -q openai 2>/dev/null || $PIP install -q openai 2>/dev/null || echo "  (pip install failed — will retry on first run)"

echo "[4/4] Setting up alias..."

# Determine shell config file
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.zshrc"
fi

# Add alias if not already present
if ! grep -q 'alias ghostreply=' "$SHELL_RC" 2>/dev/null; then
    echo '' >> "$SHELL_RC"
    echo '# GhostReply - iMessage Auto-Reply' >> "$SHELL_RC"
    echo "alias ghostreply=\"$PYTHON ~/.ghostreply/ghostreply.py\"" >> "$SHELL_RC"
fi

echo ""
echo "  ✓ GhostReply installed successfully!"
echo ""
echo "  Starting GhostReply..."
echo ""

$PYTHON ~/.ghostreply/ghostreply.py </dev/tty
