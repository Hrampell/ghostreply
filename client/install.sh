#!/bin/bash
# GhostReply Installer
# One-liner: curl -sL ghostreply.lol/install.sh | bash
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

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required. Install it from python.org or via 'brew install python3'."
    exit 1
fi

echo "[1/4] Creating ~/.ghostreply directory..."
mkdir -p ~/.ghostreply

echo "[2/4] Downloading GhostReply..."
curl -sL https://raw.githubusercontent.com/Hrampell/ghostreply/main/client/ghostreply.py -o ~/.ghostreply/ghostreply.py

echo "[3/4] Installing dependencies..."
pip3 install --break-system-packages -q openai 2>/dev/null || pip3 install -q openai

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
    echo 'alias ghostreply="python3 ~/.ghostreply/ghostreply.py"' >> "$SHELL_RC"
fi

echo ""
echo "  ✓ GhostReply installed successfully!"
echo ""
echo "  Starting GhostReply..."
echo ""

python3 ~/.ghostreply/ghostreply.py </dev/tty
