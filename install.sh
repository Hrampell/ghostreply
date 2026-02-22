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
if [[ "$TERM_PROGRAM" == "Apple_Terminal" ]]; then
    current_profile=$(defaults read com.apple.Terminal "Default Window Settings" 2>/dev/null || echo "")
    if [[ "$current_profile" != "Pro" && "$current_profile" != "Homebrew" && "$current_profile" != "Novel" ]]; then
        echo "  Switching Terminal to dark mode for best experience..."
        osascript -e 'tell application "Terminal" to set current settings of front window to settings set "Pro"' 2>/dev/null || true
        echo ""
    fi
fi

# --- Find or install Python 3 ---
find_python() {
    # 1. Homebrew python (Apple Silicon)
    if /opt/homebrew/bin/python3 --version &>/dev/null 2>&1; then
        echo "/opt/homebrew/bin/python3"; return
    fi
    # 2. Homebrew python (Intel)
    if /usr/local/bin/python3 --version &>/dev/null 2>&1; then
        echo "/usr/local/bin/python3"; return
    fi
    # 3. Python.org install location
    if /usr/local/bin/python3 --version &>/dev/null 2>&1; then
        echo "/usr/local/bin/python3"; return
    fi
    if /Library/Frameworks/Python.framework/Versions/Current/bin/python3 --version &>/dev/null 2>&1; then
        echo "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"; return
    fi
    # 4. System python3 — only if it actually works (not the Xcode shim)
    # The shim exits non-zero and shows a dialog, so --version will fail
    if python3 --version &>/dev/null 2>&1; then
        echo "python3"; return
    fi
    echo ""
}

PYTHON=$(find_python)

if [[ -z "$PYTHON" ]]; then
    echo "  Python 3 is needed (don't worry, it's quick)."
    echo ""
    echo "  Downloading Python installer (~40MB)..."
    echo ""

    # Detect architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        PKG_URL="https://www.python.org/ftp/python/3.12.8/python-3.12.8-macos11.pkg"
    else
        PKG_URL="https://www.python.org/ftp/python/3.12.8/python-3.12.8-macos11.pkg"
    fi

    PKG_PATH="/tmp/python-installer.pkg"
    curl -L -# "$PKG_URL" -o "$PKG_PATH"

    echo ""
    echo "  Installing Python (you may need to enter your password)..."
    echo ""
    sudo installer -pkg "$PKG_PATH" -target / 2>/dev/null

    # Clean up
    rm -f "$PKG_PATH"

    # Find python again after install
    PYTHON=$(find_python)

    if [[ -z "$PYTHON" ]]; then
        echo ""
        echo "  ERROR: Python install didn't work. Try installing manually:"
        echo "    https://www.python.org/downloads/"
        echo ""
        echo "  Then run this install command again."
        exit 1
    fi

    echo "  ✓ Python installed!"
    echo ""
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
