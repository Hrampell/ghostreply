# GhostReply — Project Instructions

## Version Bumping (REQUIRED on every code change)

Whenever you commit changes to `client/ghostreply.py` or `index.html`, you MUST bump the version:

### Steps:
1. **Determine bump type:**
   - Bug fixes, wrapping, small tweaks → **patch** (1.2.0 → 1.2.1)
   - New features, significant changes → **minor** (1.2.0 → 1.3.0)
   - Breaking changes → **major** (1.2.0 → 2.0.0)

2. **Update version string** in `client/ghostreply.py`:
   ```
   VERSION = "X.Y.Z"
   ```

3. **Update version references** in `index.html` (search for the old version like `v1.2.0`):
   - Top bar version display
   - Terminal installer demo text (2 occurrences)

4. **Regenerate the SHA256 hash** (auto-updater uses this):
   ```bash
   shasum -a 256 client/ghostreply.py | awk '{print $1}' > client/ghostreply.py.sha256
   ```

5. **After pushing**, update the git tag:
   ```bash
   git tag -d vX.Y.Z 2>/dev/null; git tag vX.Y.Z; git push origin --tags --force
   ```

### Important files:
- `client/ghostreply.py` — the main client (Python, single file)
- `client/ghostreply.py.sha256` — integrity hash for auto-updater
- `client/install.sh` — installer script (doesn't hardcode version)
- `index.html` — landing page (root, single file, vanilla HTML/CSS/JS + GSAP CDN)
- `assets/` — screenshot PNGs and their source HTML templates

## Project Structure
- Single-file Python CLI app (`client/ghostreply.py`)
- Single-file landing page (`index.html`)
- macOS only (iMessage via AppleScript)
- AI via Groq API (free, BYOK)
- License via LemonSqueezy API
- Trial: 10 free replies (counter-based, stored in encrypted config)

## Terminal Color Theme
- Green (`#5fff87` / `\033[92m`) — success, "You:" labels, checkmarks
- Cyan/Blue (`#5fffff` / `\033[96m`) — "Bot:" labels, links, highlights
- Gray (`\033[90m`) — secondary text, descriptions
- White (`\033[97m`) — primary text, user input
- Red (`\033[91m`) — errors
- Yellow (`\033[93m`) — warnings
- Do NOT use sandy/yellow for chat labels (was removed in v1.2.0)
