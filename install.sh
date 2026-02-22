#!/bin/bash
# GhostReply Installer
# One-liner: curl -sL hrampell.github.io/ghostreply/install.sh | bash

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║     GhostReply Installer v1.0    ║"
echo "  ║   iMessage Auto-Reply on Mac     ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "  ERROR: GhostReply only works on macOS (needs iMessage)."
    exit 1
fi

# Check macOS version (need 11+ for the Python installer)
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
if [[ "$MACOS_MAJOR" -lt 11 ]]; then
    echo "  ERROR: macOS 11 (Big Sur) or later required."
    echo "  You're on macOS $(sw_vers -productVersion)."
    exit 1
fi

# --- Find Python 3 (without triggering Xcode dev tools popup) ---
find_python() {
    # Check specific known paths — never call bare "python3" which triggers the Xcode shim
    local paths=(
        "/opt/homebrew/bin/python3"
        "/usr/local/bin/python3"
        "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    )
    for p in "${paths[@]}"; do
        if [[ -x "$p" ]] && "$p" --version &>/dev/null; then
            echo "$p"
            return
        fi
    done

    # Last resort: check if /usr/bin/python3 is real (not the Xcode shim)
    # The shim is tiny (~165KB). Real python3 is 30MB+.
    if [[ -x "/usr/bin/python3" ]]; then
        local size
        size=$(wc -c < /usr/bin/python3 2>/dev/null | tr -d ' ')
        if [[ -n "$size" && "$size" -gt 1000000 ]]; then
            echo "/usr/bin/python3"
            return
        fi
    fi

    echo ""
}

PYTHON=$(find_python)

if [[ -z "$PYTHON" ]]; then
    echo "  Python 3 is needed (don't worry, it's quick)."
    echo ""
    echo "  Downloading Python installer (~40MB)..."
    echo ""

    PKG_URL="https://www.python.org/ftp/python/3.12.8/python-3.12.8-macos11.pkg"
    PKG_PATH="/tmp/ghostreply-python-installer.pkg"

    # Download with progress bar
    if ! curl -fL -# "$PKG_URL" -o "$PKG_PATH"; then
        echo ""
        echo "  ERROR: Download failed. Check your internet connection."
        echo "  Or install Python manually: https://www.python.org/downloads/"
        rm -f "$PKG_PATH"
        exit 1
    fi

    # Verify the download isn't empty/corrupt
    PKG_SIZE=$(wc -c < "$PKG_PATH" | tr -d ' ')
    if [[ "$PKG_SIZE" -lt 1000000 ]]; then
        echo ""
        echo "  ERROR: Download appears incomplete. Try again."
        rm -f "$PKG_PATH"
        exit 1
    fi

    echo ""
    echo "  Installing Python (you may need to enter your password)..."
    echo ""

    # < /dev/tty needed because stdin is the curl pipe when run via curl | bash
    if ! sudo installer -pkg "$PKG_PATH" -target / </dev/tty; then
        echo ""
        echo "  ERROR: Python install failed."
        echo "  Try installing manually: https://www.python.org/downloads/"
        rm -f "$PKG_PATH"
        exit 1
    fi

    rm -f "$PKG_PATH"

    # Find python again after install
    PYTHON=$(find_python)

    if [[ -z "$PYTHON" ]]; then
        echo ""
        echo "  ERROR: Python installed but can't find it."
        echo "  Try closing Terminal, reopening, and running this again."
        exit 1
    fi

    echo "  ✓ Python installed!"
    echo ""
fi

echo "  Using: $PYTHON"
echo ""

echo "[1/4] Creating ~/.ghostreply directory..."
mkdir -p ~/.ghostreply

echo "[2/4] Downloading GhostReply..."
if ! curl -sfL https://raw.githubusercontent.com/Hrampell/ghostreply/main/client/ghostreply.py -o ~/.ghostreply/ghostreply.py; then
    echo "  ERROR: Failed to download GhostReply. Check your internet connection."
    exit 1
fi

# Verify the download is actually Python (not a 404 page)
if ! head -1 ~/.ghostreply/ghostreply.py | grep -q "python"; then
    echo "  ERROR: Downloaded file appears corrupted. Try again."
    rm -f ~/.ghostreply/ghostreply.py
    exit 1
fi

echo "[3/4] Installing dependencies..."
"$PYTHON" -m pip install --break-system-packages -q openai 2>/dev/null \
    || "$PYTHON" -m pip install -q openai 2>/dev/null \
    || { echo "  WARNING: pip install failed. Run this manually:"; echo "    $PYTHON -m pip install openai"; }

echo "[4/4] Setting up alias..."

# Determine shell config file
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.zshrc"
fi

# Create shell config if it doesn't exist
touch "$SHELL_RC"

# Remove old alias if present, then add fresh one
sed -i '' '/# GhostReply - iMessage Auto-Reply/d' "$SHELL_RC" 2>/dev/null
sed -i '' '/alias ghostreply=/d' "$SHELL_RC" 2>/dev/null
echo '# GhostReply - iMessage Auto-Reply' >> "$SHELL_RC"
echo "alias ghostreply=\"$PYTHON ~/.ghostreply/ghostreply.py\"" >> "$SHELL_RC"

echo ""
echo "  ✓ GhostReply installed successfully!"
echo ""
echo "  Starting GhostReply..."
echo ""

"$PYTHON" ~/.ghostreply/ghostreply.py </dev/tty
