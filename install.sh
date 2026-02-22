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

    # Last resort: check if system python3 exists AND is real (not the Xcode shim)
    # The shim lives at /usr/bin/python3 and is a tiny file (~130KB)
    # Real python3 is much larger. Also check if xcode-select has a path set.
    if [[ -x "/usr/bin/python3" ]]; then
        if xcode-select -p &>/dev/null; then
            # Dev tools are installed, so /usr/bin/python3 is real
            echo "/usr/bin/python3"
            return
        fi
        # Dev tools not installed — /usr/bin/python3 is the shim. Skip it.
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
    if ! curl -L -# "$PKG_URL" -o "$PKG_PATH"; then
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

    if ! sudo installer -pkg "$PKG_PATH" -target /; then
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
curl -sL https://raw.githubusercontent.com/Hrampell/ghostreply/main/client/ghostreply.py -o ~/.ghostreply/ghostreply.py

echo "[3/4] Installing dependencies..."
"$PYTHON" -m pip install --break-system-packages -q openai 2>/dev/null \
    || "$PYTHON" -m pip install -q openai 2>/dev/null \
    || echo "  (pip install failed — will retry on first run)"

echo "[4/4] Setting up alias..."

# Determine shell config file
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.zshrc"
fi

# Add alias if not already present (remove old one first if python path changed)
if grep -q 'alias ghostreply=' "$SHELL_RC" 2>/dev/null; then
    # Update existing alias to use correct python path
    sed -i '' '/alias ghostreply=/d' "$SHELL_RC"
    sed -i '' '/# GhostReply - iMessage Auto-Reply/d' "$SHELL_RC"
fi
echo '' >> "$SHELL_RC"
echo '# GhostReply - iMessage Auto-Reply' >> "$SHELL_RC"
echo "alias ghostreply=\"$PYTHON ~/.ghostreply/ghostreply.py\"" >> "$SHELL_RC"

echo ""
echo "  ✓ GhostReply installed successfully!"
echo ""
echo "  Starting GhostReply..."
echo ""

"$PYTHON" ~/.ghostreply/ghostreply.py </dev/tty
