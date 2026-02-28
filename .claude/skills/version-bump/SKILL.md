---
name: version-bump
description: Bump the GhostReply version across all files, regenerate the SHA256 hash, commit, tag, and push. Use when code changes are made to ghostreply.py or index.html.
---

# Version Bump

When code changes are committed to `client/ghostreply.py` or `index.html`, bump the version.

## Steps

1. **Determine bump type** from the changes:
   - Bug fixes, wrapping, small tweaks → **patch** (1.2.0 → 1.2.1)
   - New features, significant changes → **minor** (1.2.0 → 1.3.0)
   - Breaking changes → **major** (1.2.0 → 2.0.0)

2. **Update `VERSION`** in `client/ghostreply.py`:
   ```
   VERSION = "X.Y.Z"
   ```

3. **Update version refs** in `index.html` (search for the old `vX.Y.Z` — there are 3 occurrences):
   - Top bar version display
   - Terminal installer demo text (2 occurrences)

4. **Regenerate SHA256 hash** (the auto-updater uses this):
   ```bash
   shasum -a 256 client/ghostreply.py | awk '{print $1}' > client/ghostreply.py.sha256
   ```

5. **Commit** with a descriptive message.

6. **Tag and push**:
   ```bash
   git tag -d vX.Y.Z 2>/dev/null; git tag vX.Y.Z; git push origin main --tags --force
   ```

## Files
- `client/ghostreply.py` — main client (VERSION = line ~100)
- `client/ghostreply.py.sha256` — integrity hash for auto-updater
- `index.html` — landing page (3 version string occurrences)
