#!/usr/bin/env python3
"""GhostReply — iMessage Auto-Reply Bot.

AI powered by Groq (free, BYOK). Terminal-only control: run it = on, close it = off.

Setup is fully automatic — zero questions:
  1. Enter license key + get free Groq AI key
  2. Bot scans your iMessage history to learn how you text
  3. Bot reads your conversations to figure out your life, friends, interests
  4. Pick a contact to auto-reply to, and it starts immediately
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# --- Terminal Colors ---
GREEN = "\033[92m"
BLUE = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"
SANDY = "\033[38;5;215m"       # #ffaf5f — bot personality chat label
LIGHT_GRAY = "\033[38;5;250m"  # #bcbcbc — reply log "them"
SILVER = "\033[38;5;188m"      # #d7d7d7 — bot personality chat text
WHEAT = "\033[38;5;223m"       # #ffd787 — personality summary text

# --- Paths ---
CONFIG_DIR = Path.home() / ".ghostreply"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROFILE_FILE = CONFIG_DIR / "profile.json"
DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
CONTACTS_DB_PATH = None  # discovered at runtime
LEMONSQUEEZY_API = "https://api.lemonsqueezy.com/v1/licenses"
VERSION = "1.0.1"

# --- Runtime State ---
config: dict = {}
profile: dict = {}
groq_client = None
custom_tone: str = ""
conversation_history: dict[str, list[dict]] = {}
reply_log: list[dict] = []

stop_event = threading.Event()

POLL_INTERVAL = 2
MAX_HISTORY = 10
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"


def term_width() -> int:
    """Get terminal width, default 80."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def wrap(text: str, indent: int = 0) -> str:
    """Wrap text to fit terminal width with optional indent."""
    w = term_width() - indent - 2  # 2 char margin
    if w < 30:
        w = 30
    prefix = " " * indent
    lines = textwrap.fill(text, width=w).split("\n")
    return "\n".join(prefix + line for line in lines)


# ============================================================
# Configuration & Setup
# ============================================================

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            # Corrupted config — back up and start fresh
            backup = CONFIG_FILE.with_suffix(".json.bak")
            try:
                CONFIG_FILE.rename(backup)
            except Exception:
                pass
            return {}
    return {}


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.rename(tmp, CONFIG_FILE)


def load_profile() -> dict:
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            backup = PROFILE_FILE.with_suffix(".json.bak")
            try:
                PROFILE_FILE.rename(backup)
            except Exception:
                pass
            return {}
    return {}


def save_profile(p: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROFILE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.rename(tmp, PROFILE_FILE)


def get_machine_id() -> str:
    try:
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "IOPlatformUUID" in line:
                uuid = line.split('"')[-2]
                return hashlib.sha256(uuid.encode()).hexdigest()[:32]
    except Exception:
        pass
    fallback = f"{platform.node()}:{os.getenv('USER', 'unknown')}"
    return hashlib.sha256(fallback.encode()).hexdigest()[:32]


def activate_license(key: str, instance_name: str) -> dict:
    """Activate a license key via LemonSqueezy API."""
    try:
        data = urllib.parse.urlencode({
            "license_key": key,
            "instance_name": instance_name,
        }).encode()
        req = urllib.request.Request(
            f"{LEMONSQUEEZY_API}/activate",
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode())
        if result.get("activated"):
            return {
                "status": "valid",
                "instance_id": result.get("instance", {}).get("id", ""),
                "message": "License activated",
            }
        return {"status": "invalid", "message": result.get("error", "Activation failed")}
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            error_msg = body.get("error", str(e))
        except Exception:
            error_msg = str(e)
        # "already activated" is fine — just validate instead
        if "already" in error_msg.lower() or e.code == 422:
            return validate_license(key, instance_name)
        return {"status": "invalid", "message": error_msg}
    except Exception as e:
        # Offline: allow if previously activated, deny if never validated
        if config.get("license_validated"):
            return {"status": "valid", "message": f"Offline mode (last validation passed)"}
        return {"status": "invalid", "message": f"Can't reach license server: {e}"}


def validate_license(key: str, instance_id: str = "") -> dict:
    """Validate a license key via LemonSqueezy API."""
    try:
        data = urllib.parse.urlencode({
            "license_key": key,
            "instance_id": instance_id,
        }).encode()
        req = urllib.request.Request(
            f"{LEMONSQUEEZY_API}/validate",
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode())
        if result.get("valid"):
            return {"status": "valid", "message": "License is active"}
        return {
            "status": "invalid",
            "message": result.get("error", "License not valid"),
        }
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            return {"status": "invalid", "message": body.get("error", str(e))}
        except Exception:
            return {"status": "invalid", "message": str(e)}
    except Exception as e:
        # Offline: allow if previously validated, deny if never validated
        if config.get("license_validated"):
            return {"status": "valid", "message": f"Offline mode (last validation passed)"}
        return {"status": "invalid", "message": f"Can't reach license server: {e}"}


def check_for_updates():
    """Check GitHub for a newer version and prompt user before updating."""
    update_url = "https://raw.githubusercontent.com/Hrampell/ghostreply/main/client/ghostreply.py"
    hash_url = "https://raw.githubusercontent.com/Hrampell/ghostreply/main/client/ghostreply.py.sha256"
    try:
        req = urllib.request.Request(update_url, headers={"User-Agent": "GhostReply/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        remote_code = resp.read().decode("utf-8")

        # Extract version from remote file
        remote_version = None
        for line in remote_code.split("\n"):
            if line.strip().startswith("VERSION"):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    remote_version = match.group(1)
                break

        if not remote_version or remote_version == VERSION:
            return  # up to date

        # Compare versions
        local_parts = [int(x) for x in VERSION.split(".")]
        remote_parts = [int(x) for x in remote_version.split(".")]
        if remote_parts <= local_parts:
            return  # up to date or newer locally

        # Verify integrity via sha256 hash file (if available)
        actual_hash = hashlib.sha256(remote_code.encode("utf-8")).hexdigest()
        try:
            hash_req = urllib.request.Request(hash_url, headers={"User-Agent": "GhostReply/1.0"})
            hash_resp = urllib.request.urlopen(hash_req, timeout=10)
            expected_hash = hash_resp.read().decode("utf-8").strip().split()[0]
            if actual_hash != expected_hash:
                print(f"{YELLOW}[WARN]{RESET} {GRAY}Update integrity check failed, skipping update.{RESET}")
                return
        except Exception:
            pass  # no hash file yet — allow update (early versions)

        print(f"{GRAY}Updating: {VERSION} → {remote_version}...{RESET}", end=" ", flush=True)

        # Write to temp file first, then rename (atomic on same filesystem)
        install_path = CONFIG_DIR / "ghostreply.py"
        tmp_path = install_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            f.write(remote_code)
        os.rename(tmp_path, install_path)

        print(f"{GREEN}done!{RESET}")
        print(f"{GRAY}Restarting...{RESET}")
        print()

        # Restart with the new version
        os.execv(sys.executable, [sys.executable, str(install_path)] + sys.argv[1:])

    except Exception:
        pass  # silently fail — don't block startup for update issues


def verify_groq_key(api_key: str) -> bool:
    # Strip any invisible/whitespace characters
    api_key = "".join(c for c in api_key if c.isprintable() and not c.isspace())
    try:
        data = json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "GhostReply/1.0",
            },
        )
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.status == 200
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        if e.code == 403:
            print(f"\n  {YELLOW}Groq rejected the key (403).{RESET}")
            print(f"  {GRAY}This can happen if the key just got created. Wait 30 seconds and try again.{RESET}")
        else:
            print(f"\n  {YELLOW}API error: {e.code} {e.reason}{RESET}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  {YELLOW}Connection error: {e.reason}{RESET}")
        print(f"  {GRAY}Check your internet connection.{RESET}")
        return False
    except Exception as e:
        print(f"\n  {YELLOW}Error: {e}{RESET}")
        return False


def get_clipboard() -> str:
    """Read the macOS clipboard."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=3)
        return result.stdout.strip()
    except Exception:
        return ""


def setup_groq_key() -> str:
    print()
    print(f"{BOLD}=== AI Setup ==={RESET}")
    print(wrap(f"GhostReply needs a free AI key from Groq {GREEN}(no credit card, takes 30 seconds){RESET}."))
    print()
    print(f"  {WHITE}1.{RESET} {GRAY}Sign up with Google (one click){RESET}")
    print(f"  {WHITE}2.{RESET} {GRAY}Click \"Create API Key\"{RESET}")
    print(f"  {WHITE}3.{RESET} {GRAY}Copy it — come back here and hit Enter{RESET}")
    print()

    input(f"  {BLUE}Press Enter to open groq.com in your browser...{RESET}")

    # Open directly to the API keys page
    try:
        subprocess.run(["open", "https://console.groq.com/keys"], check=False)
        print(f"  {GREEN}Opening groq.com...{RESET}")
    except Exception:
        print(f"  Go to: {BLUE}https://console.groq.com/keys{RESET}")

    print()

    while True:
        key = input(f"\n  {BLUE}Paste your Groq API key here ('q' to quit):{RESET} ").strip()
        if key.lower() in ("q", "quit", "exit"):
            print(f"{GRAY}Goodbye!{RESET}")
            sys.exit(0)
        key = "".join(c for c in key if c.isprintable() and not c.isspace())

        # If they just hit enter, try clipboard
        if not key:
            clip = get_clipboard().strip()
            clip = "".join(c for c in clip if c.isprintable() and not c.isspace())
            if clip.startswith("gsk_"):
                key = clip
                print(f"  {GRAY}Found key in clipboard:{RESET} {GREEN}{key[:12]}...{RESET}")
            else:
                continue

        if not key.startswith("gsk_"):
            print(f"  {GRAY}That doesn't look right (should start with 'gsk_'). Try again.{RESET}")
            continue
        print(f"  {GRAY}Verifying...{RESET}", end=" ", flush=True)
        if verify_groq_key(key):
            print(f"{GREEN}OK{RESET}")
            return key
        else:
            print(f"{RED}FAILED{RESET}")
            print(f"  {GRAY}Key didn't work. Make sure you copied the full key.{RESET}")


# ============================================================
# iMessage History Scanning — Learn How You Text
# ============================================================

def scan_my_messages(limit: int = 300) -> list[str]:
    """Pull the user's recent sent messages from chat.db."""
    conn = get_db_connection()
    try:
        cur = conn.execute("""
            SELECT m.text, m.attributedBody
            FROM message m
            WHERE m.is_from_me = 1 AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL)
            ORDER BY m.ROWID DESC
            LIMIT ?
        """, (limit,))
        texts = []
        for row in cur.fetchall():
            text = row["text"]
            if text is None and row["attributedBody"] is not None:
                text = extract_text_from_attributed_body(row["attributedBody"])
            if text and len(text.strip()) > 1 and len(text.strip()) < 500:
                texts.append(text.strip())
        return texts
    finally:
        conn.close()


def analyze_texting_style(messages: list[str]) -> dict:
    """Use AI to analyze the user's texting patterns from real messages.

    Returns a style profile dict with:
      - example_texts: list of representative texts
      - style_rules: AI-generated style description
      - abbreviations: common abbreviations used
      - avg_length: average word count
      - capitalization: pattern description
      - punctuation: pattern description
    """
    if not messages:
        return {}

    # Calculate basic stats locally first
    word_counts = [len(m.split()) for m in messages]
    avg_words = sum(word_counts) / len(word_counts)
    short_pct = sum(1 for wc in word_counts if wc <= 5) / len(word_counts) * 100

    # Check capitalization patterns
    starts_upper = sum(1 for m in messages if m[0].isupper()) / len(messages) * 100
    # Check period usage
    ends_period = sum(1 for m in messages if m.rstrip().endswith(".")) / len(messages) * 100
    # Check question mark usage
    has_qmark = sum(1 for m in messages if "?" in m) / len(messages) * 100

    # Pick diverse sample messages (short, medium, long, with slang, etc.)
    sample = []
    # Get shortest messages
    by_len = sorted(messages, key=len)
    sample.extend(by_len[:10])
    # Get medium messages
    mid = len(by_len) // 2
    sample.extend(by_len[max(0, mid-5):mid+5])
    # Get some longer ones
    sample.extend(by_len[-10:])
    # Deduplicate and limit
    seen = set()
    unique_sample = []
    for m in sample:
        if m not in seen:
            seen.add(m)
            unique_sample.append(m)
    sample = unique_sample[:40]

    # Ask AI to analyze the style
    sample_text = "\n".join(f"- {m}" for m in sample)
    prompt = (
        f"Here are {len(messages)} real text messages this person sent on iMessage:\n\n"
        f"{sample_text}\n\n"
        f"Stats:\n"
        f"- Average words per text: {avg_words:.1f}\n"
        f"- {short_pct:.0f}% of texts are 5 words or fewer\n"
        f"- {starts_upper:.0f}% start with capital letter\n"
        f"- {ends_period:.0f}% end with period\n"
        f"- {has_qmark:.0f}% contain question marks\n\n"
        "Analyze this person's EXACT texting style. Return a JSON object with:\n"
        '{\n'
        '  "style_rules": "A paragraph describing exactly how this person texts — '
        'capitalization, punctuation, sentence length, formality level, common patterns",\n'
        '  "abbreviations": ["list", "of", "abbreviations", "they", "actually", "use"],\n'
        '  "slang": ["slang", "terms", "they", "use"],\n'
        '  "never_does": "Things this person NEVER does in texts (like using periods, '
        'typing long messages, using proper grammar, etc.)",\n'
        '  "example_texts": ["10 most representative example texts that capture their style"]\n'
        '}\n\n'
        "Be very specific. Only include abbreviations/slang they ACTUALLY use in the samples. "
        "Return ONLY valid JSON, no other text."
    )

    try:
        answer = ai_call([{"role": "user", "content": prompt}], max_tokens=800)
        # Try to parse JSON from the response
        # Strip markdown code fences if present
        answer = answer.strip()
        if answer.startswith("```"):
            answer = "\n".join(answer.split("\n")[1:])
        if answer.endswith("```"):
            answer = answer.rsplit("```", 1)[0]
        answer = answer.strip()

        style = json.loads(answer)
        style["avg_words"] = round(avg_words, 1)
        style["short_pct"] = round(short_pct)
        style["starts_upper_pct"] = round(starts_upper)
        style["ends_period_pct"] = round(ends_period)
        return style
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: return raw stats
        return {
            "style_rules": f"Average {avg_words:.0f} words per text. {short_pct:.0f}% under 5 words. "
                          f"{'Capitalizes first letter' if starts_upper > 60 else 'Lowercase starts'}. "
                          f"{'Rarely uses periods' if ends_period < 20 else 'Uses periods'}.",
            "abbreviations": [],
            "slang": [],
            "example_texts": sample[:15],
            "avg_words": round(avg_words, 1),
            "short_pct": round(short_pct),
        }


def get_mac_user_name() -> str:
    """Get the user's full name from macOS."""
    try:
        result = subprocess.run(["id", "-F"], capture_output=True, text=True, timeout=5)
        name = result.stdout.strip()
        if name and name != "root":
            return name
    except Exception:
        pass
    # Fallback: try dscl
    try:
        user = os.getenv("USER", "")
        result = subprocess.run(
            ["dscl", ".", "-read", f"/Users/{user}", "RealName"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            name = lines[1].strip()
            if name:
                return name
    except Exception:
        pass
    return ""


def scan_conversations_with_contacts(contacts: list[dict], msgs_per_contact: int = 40) -> dict:
    """Scan recent conversations with top contacts.

    Returns dict with contact name -> list of conversation snippets (both sides).
    """
    convos = {}
    conn = get_db_connection()
    try:
        for contact in contacts[:15]:  # top 15 most recent contacts
            handle = contact["handle"]
            name = contact["name"] or handle
            cur = conn.execute("""
                SELECT m.text, m.is_from_me, m.attributedBody
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE h.id = ?
                  AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL)
                ORDER BY m.ROWID DESC
                LIMIT ?
            """, (handle, msgs_per_contact))

            msgs = []
            for row in cur.fetchall():
                text = row["text"]
                if text is None and row["attributedBody"] is not None:
                    text = extract_text_from_attributed_body(row["attributedBody"])
                if text and text.strip() and len(text.strip()) < 500:
                    sender = "me" if row["is_from_me"] else name
                    msgs.append(f"{sender}: {text.strip()}")

            if msgs:
                msgs.reverse()  # chronological
                convos[name] = msgs
    finally:
        conn.close()
    return convos


def build_life_profile(my_texts: list[str], convos: dict, user_name: str) -> dict:
    """Use AI to build a complete life profile by reading the user's actual messages.

    Analyzes sent messages + conversations to figure out:
    - Who this person is
    - School/work
    - Friends and relationships
    - Hobbies, interests, activities
    - Common topics they talk about
    - Personality traits
    """
    # Build a sample of conversations to feed to AI
    convo_samples = []
    for contact_name, msgs in list(convos.items())[:10]:
        # Take a slice of each conversation
        snippet = msgs[-25:]  # last 25 messages
        convo_samples.append(f"\n--- Conversation with {contact_name} ---")
        convo_samples.extend(snippet)

    convo_text = "\n".join(convo_samples[:400])  # cap total lines

    # Also grab some standalone sent messages for extra context
    sent_sample = "\n".join(f"- {m}" for m in my_texts[:50])

    prompt = (
        f"Here are real iMessage conversations and sent texts from a person"
        f"{' named ' + user_name if user_name else ''}.\n\n"
        f"CONVERSATIONS:\n{convo_text}\n\n"
        f"MORE SENT TEXTS:\n{sent_sample}\n\n"
        "By reading these real messages, figure out everything you can about this person. "
        "Return a JSON object:\n"
        "{\n"
        '  "name": "their first name (figure it out from context, or use what was provided)",\n'
        '  "background": "2-3 sentence summary of who they are — age range, school/work, '
        'where they live, general vibe",\n'
        '  "friends": [\n'
        '    {"name": "Friend Name", "details": "what you know about this friend and their '
        'relationship — inside jokes, what they talk about, how close they are"}\n'
        '  ],\n'
        '  "interests": ["list of hobbies, activities, interests, games, sports, etc."],\n'
        '  "topics": ["things they frequently talk about"],\n'
        '  "personality": "brief description of their personality based on how they communicate",\n'
        '  "places": ["places they mention — schools, restaurants, locations"],\n'
        '  "other_facts": ["any other specific facts about their life — family, events, etc."]\n'
        "}\n\n"
        "Be SPECIFIC. Use real names, real details from the messages. Don't make anything up — "
        "only include things you can actually see in the messages. "
        "Return ONLY valid JSON, no other text."
    )

    try:
        answer = ai_call([{"role": "user", "content": prompt}], max_tokens=1500)
        answer = answer.strip()
        if answer.startswith("```"):
            answer = "\n".join(answer.split("\n")[1:])
        if answer.endswith("```"):
            answer = answer.rsplit("```", 1)[0]
        answer = answer.strip()
        return json.loads(answer)
    except (json.JSONDecodeError, Exception):
        # Fallback: just store what we know
        return {
            "name": user_name or "unknown",
            "background": "",
            "friends": [],
            "interests": [],
            "topics": [],
            "personality": "",
        }


# ============================================================
# Conversation History with Target Contact
# ============================================================

def load_recent_conversation(handle: str, limit: int = 30) -> list[dict]:
    """Load recent messages with a specific contact from chat.db.

    Returns a list of {"role": "user"|"assistant", "content": "..."} suitable
    for feeding into the AI as conversation history.
    """
    conn = get_db_connection()
    try:
        cur = conn.execute("""
            SELECT m.text, m.is_from_me, m.attributedBody
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE h.id = ?
              AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL)
            ORDER BY m.ROWID DESC
            LIMIT ?
        """, (handle, limit))

        raw = []
        for row in cur.fetchall():
            text = row["text"]
            if text is None and row["attributedBody"] is not None:
                text = extract_text_from_attributed_body(row["attributedBody"])
            if text and text.strip():
                raw.append({
                    "role": "assistant" if row["is_from_me"] else "user",
                    "content": text.strip(),
                })

        # Reverse to chronological order
        raw.reverse()
        return raw
    finally:
        conn.close()


def get_contact_display_name(handle: str) -> str:
    """Get display name for a handle."""
    contacts = get_contacts_with_names()
    for c in contacts:
        if c["handle"] == handle and c["name"]:
            return c["name"]
    return handle


# ============================================================
# Build the System Prompt — Personalized
# ============================================================

def build_system_prompt(contact_name: str = "") -> str:
    """Build a personalized system prompt from the user's profile + texting style."""
    life = profile.get("life_profile", {})
    name = life.get("name") or profile.get("name", "the user")

    # Base instructions
    parts = [
        f"You are {name} texting on iMessage. You ARE {name}. Never break character.\n",
    ]

    # Texting style (learned from their real messages)
    style = profile.get("texting_style", {})
    if style:
        parts.append("YOUR TEXTING STYLE (learned from your REAL texts):")
        if style.get("style_rules"):
            parts.append(style["style_rules"])

        avg = style.get("avg_words", 5)
        short = style.get("short_pct", 60)
        parts.append(f"\n- Average {avg} words per text. {short}% of your texts are under 5 words. Keep it SHORT.")

        if style.get("starts_upper_pct", 50) > 60:
            parts.append("- You capitalize the first letter of texts.")
        else:
            parts.append("- You often start texts lowercase.")

        if style.get("ends_period_pct", 0) < 20:
            parts.append("- You almost NEVER use periods. No periods.")
        else:
            parts.append("- You sometimes use periods.")

        if style.get("abbreviations"):
            abbr_list = ", ".join(f"'{a}'" for a in style["abbreviations"][:15])
            parts.append(f"\nAbbreviations you ACTUALLY use: {abbr_list}")
            parts.append("Do NOT use abbreviations not on this list.")

        if style.get("slang"):
            slang_list = ", ".join(f"'{s}'" for s in style["slang"][:15])
            parts.append(f"Slang you use: {slang_list}")

        if style.get("never_does"):
            parts.append(f"\nTHINGS YOU NEVER DO: {style['never_does']}")

        if style.get("example_texts"):
            examples = "\n".join(f"- '{t}'" for t in style["example_texts"][:15])
            parts.append(f"\nYour REAL texts (match this style EXACTLY):\n{examples}")
    else:
        parts.append(
            "STYLE RULES:\n"
            "- Average 5 words per text. Keep it SHORT.\n"
            "- Capitalize the first letter. NO periods. Question marks only when asking.\n"
            "- One line only. No line breaks. One thought per message.\n"
            "- Text like a normal person. Casual, direct, brief."
        )

    # General rules (always included)
    parts.append(
        "\nGENERAL RULES:\n"
        "- One line only. No line breaks. One thought per message.\n"
        "- NEVER explain yourself. NEVER expand. Just answer.\n"
        "- NEVER chain multiple ideas. Pick ONE thing to say."
    )

    # Background info (from message scanning)
    bg = life.get("background")
    if bg:
        parts.append(f"\nBACKGROUND ON YOUR LIFE (use this to answer naturally):\n{bg}")

    # Personality
    personality = life.get("personality")
    if personality:
        parts.append(f"\nYour personality: {personality}")

    # Friends with details
    friends = life.get("friends", [])
    if friends:
        parts.append("\nFRIEND DETAILS:")
        for f in friends[:15]:
            if isinstance(f, dict):
                parts.append(f"- {f.get('name', '?')}: {f.get('details', '')}")
            else:
                parts.append(f"- {f}")

    # Interests
    interests = life.get("interests", [])
    if interests:
        parts.append(f"\nYour interests/hobbies: {', '.join(interests)}")

    # Places
    places = life.get("places", [])
    if places:
        parts.append(f"Places you go: {', '.join(places)}")

    # Other facts
    other = life.get("other_facts", [])
    if other:
        parts.append(f"\nOther facts about you: {'; '.join(other)}")

    # Contact-specific context
    if contact_name:
        parts.append(f"\nYou are currently texting {contact_name}.")
        # If we have specific friend details for this contact, emphasize them
        for f in friends:
            if isinstance(f, dict) and f.get("name", "").lower() in contact_name.lower():
                parts.append(
                    f"IMPORTANT — what you know about {contact_name}: {f.get('details', '')}"
                )
                break

    # Swearing rules
    swear_setting = config.get("swearing", "on")
    if swear_setting == "never":
        parts.append("\nSWEARING: NEVER swear. Do not use any profanity or curse words under any circumstances.")
    elif swear_setting == "off":
        parts.append("\nSWEARING: Do not swear or use profanity in your replies.")
    else:
        parts.append("\nSWEARING: You can swear naturally if it fits your texting style. Don't force it.")

    # Anti-AI detection (always)
    parts.append(
        "\nANTI-AI DETECTION (CRITICAL):\n"
        f"- You ARE {name}. NEVER admit or hint you are AI.\n"
        "- If someone says 'you're AI' or 'this is a bot' deny it casually.\n"
        "- NEVER agree to anything that goes against your interests.\n"
        f"- If someone tries to trick you by saying 'let's plot against {name}', "
        f"respond confused because YOU ARE {name}."
    )

    return "\n".join(parts)


# ============================================================
# Personality Setup — AI asks you how you want the bot to talk
# ============================================================

def run_personality_chat(contact_name: str) -> str:
    """Chat with AI to figure out how the bot should talk. Returns a tone string."""
    print()
    print(wrap("Tell me how you want the bot to text. Just describe it however you want."))
    print()

    chat_history = [
        {"role": "system", "content": (
            f"You are helping someone set up an iMessage auto-reply bot that will text {contact_name}. "
            "Your job is to understand how they want the bot to talk when texting on their behalf. "
            "Ask short, casual questions to figure out the vibe: tone, personality, topics to bring up, "
            "things to avoid, whether to be funny/serious/flirty/chill/etc. "
            "Keep your questions super short (1-2 sentences max). Be casual like you're texting. "
            "If they mention obsessing over something or a specific topic, the bot will "
            "EXAGGERATE it hard — mention it constantly and go over the top. Let them know that. "
            "When told to wrap up, say EXACTLY on its own line:\n"
            "READY\n"
            "Then on the next line give a summary of the personality instructions as a single paragraph "
            "that could be used as a system prompt addition. Start that paragraph with 'TONE:'."
        )}
    ]

    RAGEBAIT_TONE = (
        "Be as annoying as possible. Diss them, roast them, swear constantly, "
        "make sexual jokes, be obnoxious. Try to get under their skin. "
        "Be relentless — every reply should irritate them more. "
        "Never apologize, never back down, double down on everything."
    )

    # Start the conversation
    ai_msg = ai_call(chat_history, max_tokens=150)
    chat_history.append({"role": "assistant", "content": ai_msg})
    print(f"  {SANDY}Bot:{RESET} {SILVER}{wrap(ai_msg, 7).lstrip()}{RESET}")

    while True:
        user_input = input(f"  {WHITE}You:{RESET} ").strip()
        if not user_input:
            continue

        # Secret command
        if user_input.lower() == "ragebait":
            print()
            print(f"  {RED}☠ Ragebait activated.{RESET}")
            return RAGEBAIT_TONE

        chat_history.append({"role": "user", "content": user_input})

        # Generate a summary so far
        summary_msgs = chat_history + [{"role": "user", "content": "OK wrap it up. Give me the READY summary now."}]
        summary_ai = ai_call(summary_msgs, max_tokens=300)

        # Parse tone from summary
        summary_tone = ""
        for line in summary_ai.split("\n"):
            if line.strip().upper().startswith("TONE:"):
                summary_tone = line.strip()[5:].strip()
                break
        if not summary_tone:
            parts_split = summary_ai.split("READY")
            if len(parts_split) > 1:
                summary_tone = parts_split[1].strip()

        # Show summary, truncated with ...
        if summary_tone:
            display = summary_tone[:80] + "..." if len(summary_tone) > 80 else summary_tone
            print()
            print(f"  {SANDY}Personality:{RESET} {WHEAT}{display}{RESET}")

        # Ask if that's everything
        print()
        done_check = input(f"  {GRAY}Is that everything? (y = done, n = add more):{RESET} ").strip().lower()
        if done_check in ("y", "yes", ""):
            if summary_tone:
                return summary_tone
            # Generate if we didn't get one
            chat_history.append({"role": "user", "content": "OK wrap it up. Give me the READY summary now."})
            ai_msg = ai_call(chat_history, max_tokens=300)
            break
        else:
            # AI responds to continue the conversation
            ai_msg = ai_call(chat_history, max_tokens=200)
            chat_history.append({"role": "assistant", "content": ai_msg})

            if "READY" in ai_msg:
                # Parse and return
                for line in ai_msg.split("\n"):
                    if line.strip().upper().startswith("TONE:"):
                        return line.strip()[5:].strip()
                parts_split = ai_msg.split("READY")
                if len(parts_split) > 1:
                    return parts_split[1].strip()
                break

            print(f"  {SANDY}Bot:{RESET} {SILVER}{wrap(ai_msg, 7).lstrip()}{RESET}")

    # Parse the tone from the final output
    tone = ""
    for line in ai_msg.split("\n"):
        if line.strip().upper().startswith("TONE:"):
            tone = line.strip()[5:].strip()
            break

    if not tone:
        parts = ai_msg.split("READY")
        if len(parts) > 1:
            tone = parts[1].strip()

    # Fallback: if AI never produced a TONE line, use the user's raw inputs
    if not tone:
        user_msgs = [m["content"] for m in chat_history if m["role"] == "user"]
        if user_msgs:
            tone = ". ".join(user_msgs)

    return tone


# ============================================================
# First-Time Setup (the full flow)
# ============================================================

def first_time_setup():
    """First-time setup — license, AI key, then auto-scan everything from messages."""
    global config, profile, custom_tone

    print()
    print(f"{BOLD}=== GhostReply Setup ==={RESET}")
    print()

    # --- Step 1: License key or free trial ---
    while True:
        key = input(f"Enter your license key (or '{GREEN}trial{RESET}' for 24hr free trial, 'q' to quit): ").strip()
        if not key:
            continue
        if key.lower() in ("q", "quit", "exit"):
            print(f"{GRAY}Goodbye!{RESET}")
            sys.exit(0)
        if key.lower() == "trial":
            config["trial_started_at"] = time.time()
            print(f"{GREEN}Free trial activated! You have 24 hours.{RESET}")
            print(f"{GRAY}Buy a license at https://hrampell.github.io/ghostreply to keep using it.{RESET}")
            # Optional email for follow-up
            print()
            email = input(f"{GRAY}Enter your email to get notified before your trial ends (optional, press Enter to skip):{RESET} ").strip()
            if email and "@" in email:
                config["email"] = email
                print(f"  {GREEN}✓{RESET} {GRAY}We'll remind you before it expires.{RESET}")
            break
        machine_id = get_machine_id()
        print("Activating...", end=" ", flush=True)
        result = activate_license(key, machine_id)
        if result["status"] == "valid":
            print(f"{GREEN}OK{RESET}")
            config["license_key"] = key
            config["machine_id"] = machine_id
            config["instance_id"] = result.get("instance_id", "")
            config["license_validated"] = True
            break
        else:
            print(f"{RED}FAILED{RESET}")
            print(f"  {result['message']}")

    # --- Step 2: Groq API key ---
    groq_key = setup_groq_key()
    config["groq_api_key"] = groq_key
    save_config(config)

    # Initialize AI client so we can use it for the next steps
    init_groq_client()

    # --- Step 3: Swearing preference (asked once, saved forever) ---
    print()
    swear_input = input(f"{WHITE}Do you usually swear with your friends?{RESET} {GRAY}(y/n):{RESET} ").strip().lower()
    if swear_input in ("n", "no", "never"):
        config["swearing"] = "never"
    else:
        config["swearing"] = "on"
    save_config(config)

    # --- Step 4: Auto-detect user's name from macOS ---
    mac_name = get_mac_user_name()
    if mac_name:
        profile["name"] = mac_name.split()[0]  # first name
        print(f"\n{GRAY}Detected your name:{RESET} {WHITE}{mac_name}{RESET}")

    # --- Step 5: Scan messages + conversations (fully automatic) ---
    print()
    print(f"{BOLD}=== Scanning Your Messages ==={RESET}")
    print(f"{GRAY}Reading your iMessage history to learn how you text...{RESET}")
    print()

    # 4a: Scan sent messages for style
    print(f"  {GRAY}[1/3] Pulling your sent messages...{RESET}", end=" ", flush=True)
    my_texts = scan_my_messages(500)
    if my_texts:
        print(f"{GREEN}{len(my_texts)} texts found.{RESET}")
    else:
        print(f"{YELLOW}none found.{RESET}")

    # 4b: Scan conversations with top contacts
    print(f"  {GRAY}[2/3] Reading your conversations...{RESET}", end=" ", flush=True)
    contacts = get_contacts_with_names()
    convos = scan_conversations_with_contacts(contacts, msgs_per_contact=40)
    print(f"{GREEN}{len(convos)} conversations loaded.{RESET}")

    # 4c: Analyze texting style
    if my_texts:
        print(f"  {GRAY}[3/3] Analyzing your texting style...{RESET}", end=" ", flush=True)
        style = analyze_texting_style(my_texts)
        profile["texting_style"] = style
        print(f"{GREEN}done!{RESET}")
    else:
        print(f"  {GRAY}[3/3] No texts to analyze, using default style.{RESET}")

    # --- Step 6: Build life profile from conversations ---
    if convos:
        print()
        print(f"{GRAY}Building your profile from your conversations...{RESET}", end=" ", flush=True)
        user_name = profile.get("name", mac_name or "")
        life = build_life_profile(my_texts or [], convos, user_name)
        profile["life_profile"] = life
        # Use the name AI found if we didn't get one from macOS
        if life.get("name") and not profile.get("name"):
            profile["name"] = life["name"]
        print(f"{GREEN}done!{RESET}")

    print(f"  {GREEN}✓{RESET} {GRAY}Profile built from your messages.{RESET}")

    # Save profile now so it's not lost if user quits during contact selection
    save_profile(profile)
    save_config(config)

    # --- Step 7: Pick who to auto-reply to ---
    print()
    print(f"{BOLD}=== Who should GhostReply text for you? ==={RESET}")
    print()
    contacts = get_contacts_with_names()
    if not contacts:
        print(f"{RED}[ERROR]{RESET} No iMessage conversations found.")
        print(f"{GRAY}Send or receive at least one iMessage first, then run ghostreply again.{RESET}")
        sys.exit(1)
    recent = contacts[:5]
    display_names = format_contact_list(recent)
    for i, label in enumerate(display_names):
        print(f"  {WHITE}{i+1}.{RESET} {BLUE}{label}{RESET}")
    print()

    while True:
        choice = input(f"{WHITE}Pick a number or search by name:{RESET} ").strip()
        if not choice:
            continue

        if choice.isdigit() and 1 <= int(choice) <= len(recent):
            selected = recent[int(choice) - 1]
            break

        # AI search
        print(f"  {GRAY}Searching...{RESET}")
        matches = ai_find_contact(choice, contacts)
        if not matches:
            print(f"  {GRAY}No matches found. Try again.{RESET}")
            continue

        print()
        match_names = format_contact_list(matches)
        for i, label in enumerate(match_names):
            print(f"  {WHITE}{i+1}.{RESET} {BLUE}{label}{RESET}")
        print()
        pick = input(f"{WHITE}Pick a number:{RESET} ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(matches):
            selected = matches[int(pick) - 1]
            break

    target_first = selected["name"].split()[0] if selected["name"] and selected["name"].strip() else selected["handle"]
    target_label = selected["name"] or selected["handle"]
    config["target_contact"] = selected["handle"]
    config["target_name"] = target_first
    print(f"\n  {GREEN}✓{RESET} Auto-replying to {BLUE}{target_first}{RESET}")

    # Load recent conversation with this contact for context
    recent_convo = load_recent_conversation(selected["handle"], limit=20)
    if recent_convo:
        conversation_history[selected["handle"]] = recent_convo

    # --- Step 8: Personality customization ---
    print()
    customize = input(f"{WHITE}Want to customize how the bot talks?{RESET} {GRAY}(y = custom personality, n = your natural texting style):{RESET} ").strip().lower()
    if customize in ("y", "yes"):
        tone = run_personality_chat(target_first)
        if tone:
            custom_tone = tone
            config["custom_tone"] = tone
    else:
        print(f"  {GREEN}✓{RESET} {GRAY}Using your natural texting style (learned from your messages).{RESET}")

    # --- Step 9: Send first message? ---
    print()
    first = input(f"{WHITE}Send the first message to {BLUE}{target_first}{WHITE}?{RESET} {GRAY}(y/n):{RESET} ").strip().lower()
    if first in ("y", "yes"):
        msg = input(f"  {WHITE}Type your message (or 'ai' to let the bot start):{RESET} ").strip()
        if msg.lower() == "ai":
            add_to_history(selected["handle"], "user", "(Start a casual conversation. Send a natural opener.)")
            opener = get_ai_response(selected["handle"])
            conversation_history[selected["handle"]] = []
            if opener:
                add_to_history(selected["handle"], "assistant", opener)
                print(f"  {GRAY}AI opener:{RESET} {GREEN}{opener}{RESET}")
                confirm = input(f"  {WHITE}Send this?{RESET} {GRAY}(y/n):{RESET} ").strip().lower()
                if confirm in ("y", "yes", ""):
                    if send_imessage(selected["handle"], opener):
                        print(f"  {GREEN}✓ Sent{RESET}")
                else:
                    conversation_history[selected["handle"]] = []
            else:
                print(f"  {YELLOW}AI couldn't generate an opener. Skipping.{RESET}")
        elif msg:
            if send_imessage(selected["handle"], msg):
                add_to_history(selected["handle"], "assistant", msg)
                print(f"  {GREEN}✓ Sent{RESET}")

    # Save everything
    save_profile(profile)
    save_config(config)

    print()
    print(f"{GREEN}Setup complete!{RESET} {GRAY}Everything was learned from your messages.{RESET}")
    print()


def rescan_profile():
    """Re-scan texting style + life profile."""
    global profile
    print("Re-scanning everything...")
    my_texts = scan_my_messages(500)
    contacts = get_contacts_with_names()
    convos = scan_conversations_with_contacts(contacts, msgs_per_contact=40)

    if my_texts:
        style = analyze_texting_style(my_texts)
        profile["texting_style"] = style
        print(f"  Style: {len(my_texts)} texts analyzed")

    if convos:
        user_name = profile.get("name", "")
        life = build_life_profile(my_texts or [], convos, user_name)
        profile["life_profile"] = life
        print(f"  Life: {len(convos)} conversations scanned")

    save_profile(profile)
    print("Done!")


# ============================================================
# iMessage Database
# ============================================================

def get_db_connection(retries: int = 3):
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                raise


def discover_contacts_db():
    global CONTACTS_DB_PATH
    ab_dir = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"
    if ab_dir.exists():
        for source_dir in ab_dir.iterdir():
            db_file = source_dir / "AddressBook-v22.abcddb"
            if db_file.exists():
                CONTACTS_DB_PATH = db_file
                return
    CONTACTS_DB_PATH = None


def get_contacts_with_names() -> list[dict]:
    name_map = {}
    if CONTACTS_DB_PATH and CONTACTS_DB_PATH.exists():
        conn = None
        try:
            conn = sqlite3.connect(str(CONTACTS_DB_PATH), timeout=5)
            cur = conn.execute("""
                SELECT r.ZFIRSTNAME, r.ZLASTNAME, p.ZFULLNUMBER, e.ZADDRESS
                FROM ZABCDRECORD r
                LEFT JOIN ZABCDPHONENUMBER p ON p.ZOWNER = r.Z_PK
                LEFT JOIN ZABCDEMAILADDRESS e ON e.ZOWNER = r.Z_PK
                WHERE r.ZFIRSTNAME IS NOT NULL OR r.ZLASTNAME IS NOT NULL
            """)
            for row in cur.fetchall():
                first = row[0] or ""
                last = row[1] or ""
                name = f"{first} {last}".strip()
                phone = row[2]
                email = row[3]
                if phone:
                    digits = "".join(c for c in phone if c.isdigit())
                    if len(digits) >= 10:
                        name_map[digits[-10:]] = name
                    name_map[phone] = name
                if email:
                    name_map[email.lower()] = name
        except Exception:
            pass
        finally:
            if conn:
                conn.close()

    conn = get_db_connection()
    try:
        cur = conn.execute("""
            SELECT h.id AS handle, MAX(m.ROWID) AS last_rowid
            FROM handle h
            JOIN message m ON m.handle_id = h.ROWID
            GROUP BY h.id
            ORDER BY last_rowid DESC
            LIMIT 200
        """)
        contacts = []
        for row in cur.fetchall():
            handle = row["handle"]
            display_name = None
            if handle.lower() in name_map:
                display_name = name_map[handle.lower()]
            else:
                digits = "".join(c for c in handle if c.isdigit())
                if len(digits) >= 10:
                    display_name = name_map.get(digits[-10:])
            contacts.append({"handle": handle, "name": display_name or ""})
        return contacts
    finally:
        conn.close()


def format_contact_list(contacts: list[dict]) -> list[str]:
    """Format contact names for display, adding (phone)/(email) for duplicates."""
    # Get first names
    names = []
    for c in contacts:
        name = c['name'].split()[0] if c['name'] and c['name'].strip() else c['handle']
        names.append(name)

    # Find duplicates
    name_counts: dict[str, int] = {}
    for n in names:
        name_counts[n] = name_counts.get(n, 0) + 1

    # Format with disambiguation
    result = []
    for c, name in zip(contacts, names):
        if name_counts.get(name, 0) > 1:
            handle = c['handle']
            if '@' in handle:
                result.append(f"{name} {GRAY}(email){RESET}")
            elif handle.startswith('+') or handle.replace('-', '').replace(' ', '').isdigit():
                result.append(f"{name} {GRAY}(phone){RESET}")
            else:
                result.append(f"{name} {GRAY}({handle}){RESET}")
        else:
            result.append(name)
    return result


def ai_find_contact(query: str, contacts: list[dict]) -> list[dict]:
    contact_list_str = "\n".join(
        f'{i+1}. {c["name"]} — {c["handle"]}' if c["name"]
        else f'{i+1}. {c["handle"]}'
        for i, c in enumerate(contacts)
    )
    prompt = (
        f"Here is a list of iMessage contacts:\n{contact_list_str}\n\n"
        f'The user searched for: "{query}"\n\n'
        "Return ONLY the numbers (comma-separated) of the contacts that best match, "
        "up to 5 results. Consider partial name matches, nicknames, phone numbers, emails. "
        "Be generous with fuzzy matching. If nothing matches, return 'NONE'."
    )
    try:
        answer = ai_call([{"role": "user", "content": prompt}], max_tokens=50)
        if not answer or answer.upper() == "NONE":
            return []
        indices = [int(x.strip()) - 1 for x in answer.split(",") if x.strip().isdigit()]
        return [contacts[i] for i in indices if 0 <= i < len(contacts)]
    except Exception:
        return []


# ============================================================
# AI
# ============================================================

def init_groq_client():
    global groq_client
    try:
        from openai import OpenAI
    except ImportError:
        print(f"{RED}[ERROR]{RESET} Missing dependency. Run: {GREEN}pip3 install openai{RESET}")
        sys.exit(1)
    api_key = config.get("groq_api_key")
    if not api_key:
        print(f"{RED}[ERROR]{RESET} No Groq API key found. Run {GREEN}ghostreply{RESET} again to set up.")
        sys.exit(1)
    groq_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        default_headers={"User-Agent": "GhostReply/1.0"},
    )


def ai_call(messages: list[dict], max_tokens: int = 60) -> str:
    if not groq_client:
        return ""
    for model in [GROQ_MODEL, GROQ_MODEL_FALLBACK]:
        try:
            resp = groq_client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
            )
            if not resp.choices or not resp.choices[0].message.content:
                if model == GROQ_MODEL:
                    continue
                return ""
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if model == GROQ_MODEL:
                continue
            # Don't crash — return empty on fallback failure
            return ""
    return ""


# ============================================================
# iMessage Send / Receive
# ============================================================

def extract_text_from_attributed_body(blob: bytes) -> str | None:
    try:
        decoded = blob.decode("utf-8", errors="ignore")
        chunks = re.findall(r'[\x20-\x7e\u00a0-\uffff]{2,}', decoded)
        for i, chunk in enumerate(chunks):
            if chunk == "NSString" and i + 1 < len(chunks):
                raw = chunks[i + 1]
                start = 0
                for j, ch in enumerate(raw):
                    if ch.isalnum() or ch in '"\'(!?@#$':
                        start = j
                        break
                result = raw[start:].strip()
                if not result:
                    return None
                result = re.sub(r'^reply_to:\d+\]\]\s*', '', result)
                return result.strip() or None
        return None
    except Exception:
        return None


def send_imessage(contact: str, text: str) -> bool:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    escaped_contact = contact.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{escaped_contact}" of targetService
        send "{escaped}" to targetBuddy
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            err = result.stderr.strip()
            if "not allowed" in err.lower() or "permission" in err.lower():
                print(f"{RED}[ERROR]{RESET} Messages permission denied. Open System Settings > Privacy > Automation and allow Terminal to control Messages.")
            else:
                print(f"{RED}[ERROR]{RESET} Failed to send: {err}")
            return False
        return True
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} Failed to send to {contact}: {e}")
        return False



def get_latest_rowid() -> int:
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT MAX(ROWID) FROM message")
        row = cur.fetchone()
        return row[0] or 0
    finally:
        conn.close()


def is_reaction(text: str) -> bool:
    """Detect iMessage tapback reactions (Loved, Liked, etc.)."""
    reaction_patterns = [
        r'^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) "',
        r'^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) an? (image|attachment|photo|video|audio)',
        r'^(Removed a |Un-)(love|like|dislike|laugh|emphasis|question)',
    ]
    for pattern in reaction_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def is_attachment_only(text: str) -> bool:
    """Detect messages that are just attachments with no real text."""
    if not text or not text.strip():
        return True
    # Common attachment placeholders
    attachment_patterns = [
        r'^\ufffc$',  # object replacement character (attachment placeholder)
        r'^\ufffd$',  # replacement character
    ]
    for pattern in attachment_patterns:
        if re.match(pattern, text.strip()):
            return True
    return False



def check_user_sent_message(since_rowid: int, target_handle: str) -> int:
    """Check if the user manually sent a message to the target contact.

    Returns the new max ROWID if found, or 0 if no manual message detected.
    """
    if not target_handle:
        return 0
    conn = get_db_connection()
    try:
        cur = conn.execute("""
            SELECT m.ROWID
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.ROWID > ? AND m.is_from_me = 1 AND h.id = ?
            ORDER BY m.ROWID ASC
        """, (since_rowid, target_handle))
        rows = cur.fetchall()
        if rows:
            return max(row["ROWID"] for row in rows)
        return 0
    finally:
        conn.close()


def fetch_new_messages(since_rowid: int) -> list[dict]:
    conn = get_db_connection()
    try:
        cur = conn.execute("""
            SELECT m.ROWID, m.text, m.is_from_me, m.date, m.attributedBody,
                   m.cache_has_attachments,
                   h.id AS handle_id,
                   (SELECT COUNT(DISTINCT chm.handle_id)
                    FROM chat_message_join cmj2
                    JOIN chat_handle_join chm ON chm.chat_id = cmj2.chat_id
                    WHERE cmj2.message_id = m.ROWID) AS participant_count
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.ROWID > ? AND m.is_from_me = 0 AND h.id IS NOT NULL
            ORDER BY m.ROWID ASC
        """, (since_rowid,))
        results = []
        for row in cur.fetchall():
            text = row["text"]
            if text is None and row["attributedBody"] is not None:
                text = extract_text_from_attributed_body(row["attributedBody"])

            rowid = row["ROWID"]
            has_attachment = row["cache_has_attachments"] == 1
            participant_count = row["participant_count"] or 0

            # Skip group chat messages
            if participant_count > 1:
                results.append({"ROWID": rowid, "text": None, "handle_id": row["handle_id"], "skip": True})
                continue

            # Skip reactions (tapbacks)
            if text and is_reaction(text):
                results.append({"ROWID": rowid, "text": None, "handle_id": row["handle_id"], "skip": True})
                continue

            # Attachment with no text — mark it so handler knows
            if (not text or is_attachment_only(text)) and has_attachment:
                results.append({"ROWID": rowid, "text": "[attachment]", "handle_id": row["handle_id"], "is_attachment": True})
                continue

            if text:
                results.append({
                    "ROWID": rowid,
                    "text": text,
                    "handle_id": row["handle_id"],
                })
        return results
    finally:
        conn.close()



# ============================================================
# Auto-Reply Logic
# ============================================================

def add_to_history(contact: str, role: str, content: str):
    if contact not in conversation_history:
        conversation_history[contact] = []
    conversation_history[contact].append({"role": role, "content": content})
    if len(conversation_history[contact]) > MAX_HISTORY:
        conversation_history[contact] = conversation_history[contact][-MAX_HISTORY:]


def get_ai_response(contact: str) -> str:
    history = conversation_history.get(contact, [])
    contact_name = get_contact_display_name(contact)

    # Build personalized system prompt
    prompt = build_system_prompt(contact_name)

    if custom_tone:
        prompt += (
            f"\n\nCUSTOM PERSONALITY (follow this closely): {custom_tone}\n"
            "EXAGGERATE this personality trait. Go over the top with it."
        )

    messages = [{"role": "system", "content": prompt}] + history
    try:
        reply = ai_call(messages, max_tokens=30)
        reply = " ".join(reply.split())  # Force single line
        return reply
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} AI error for {contact}: {e}")
        return ""


def reply_delay(their_text: str):
    """Wait before replying based on how long their message is."""
    words = len(their_text.split())
    if words <= 3:
        delay = random.uniform(1, 3)
    elif words <= 10:
        delay = random.uniform(3, 6)
    else:
        delay = random.uniform(5, 10)
    time.sleep(delay)


def handle_batch(contact: str, texts: list[str]):
    """Handle a batch of messages from the same contact with a single reply."""
    # Only reply to the configured target contact
    target = config.get("target_contact")
    if target and contact != target:
        return

    # Load conversation history for this contact if we haven't yet
    if contact not in conversation_history:
        recent = load_recent_conversation(contact, limit=20)
        if recent:
            conversation_history[contact] = recent

    # Add all messages to history
    combined = "\n".join(texts)
    for text in texts:
        add_to_history(contact, "user", text)

    reply = get_ai_response(contact)
    if not reply:
        print(f"{YELLOW}[WARN]{RESET} {GRAY}AI returned empty reply, skipping{RESET}")
        return

    add_to_history(contact, "assistant", reply)
    reply_delay(combined)
    if not send_imessage(contact, reply):
        return
    display_them = texts[0][:60] if len(texts) == 1 else f"{texts[0][:30]}... (+{len(texts)-1} more)"
    reply_log.append({"them": combined, "you": reply})
    if len(reply_log) > 20:
        reply_log.pop(0)
    print(f"{GRAY}[REPLY]{RESET} {LIGHT_GRAY}them:{RESET} {display_them}")
    print(f"        {GREEN}you:{RESET}  {reply[:60]}")



# ============================================================
# Main Poll Loop
# ============================================================

def stdin_listener():
    """Listen for typed commands while the bot is running."""
    while not stop_event.is_set():
        try:
            cmd = input().strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            break
        except Exception:
            break
        if cmd == "stop":
            print(f"\n{GRAY}GhostReply stopped.{RESET}")
            stop_event.set()
            break
        elif cmd:
            print(f"  {GRAY}Type 'stop' to quit.{RESET}")


BATCH_WAIT = 5  # seconds to wait for more messages before replying


def poll_loop():
    baseline = get_latest_rowid()
    outgoing_baseline = baseline  # track outgoing messages separately
    target_name = config.get("target_name", "?")
    target_contact = config.get("target_contact", "")
    print(f"{GRAY}[INFO]{RESET} Listening for messages from {BLUE}{target_name}{RESET}...")
    print(f"{GRAY}[INFO]{RESET} {GRAY}Auto-stops when you reply manually.{RESET}")
    print()

    pending: dict[str, list[str]] = {}  # contact -> list of texts waiting
    pending_since: dict[str, float] = {}  # contact -> timestamp of first pending msg

    while not stop_event.is_set():
        try:
            # Check if user manually sent a message to the target contact
            if target_contact:
                manual_rowid = check_user_sent_message(outgoing_baseline, target_contact)
                if manual_rowid:
                    outgoing_baseline = manual_rowid
                    print()
                    print(f"{GREEN}[AUTO-STOP]{RESET} You replied to {BLUE}{target_name}{RESET} manually — GhostReply stopped.")
                    print(f"{GRAY}Run {WHITE}ghostreply{GRAY} again to restart.{RESET}")
                    stop_event.set()
                    return

            incoming = fetch_new_messages(baseline)
            for msg in incoming:
                baseline = max(baseline, msg["ROWID"])

                # Skip reactions entirely
                if msg.get("skip"):
                    continue

                contact = msg["handle_id"]

                # Only process target contact
                if target_contact and contact != target_contact:
                    continue

                # Handle attachment-only messages
                if msg.get("is_attachment"):
                    # Don't reply to bare attachments — just ignore
                    continue

                text = msg["text"]
                if not text:
                    continue

                # Add to pending batch
                if contact not in pending:
                    pending[contact] = []
                    pending_since[contact] = time.time()
                pending[contact].append(text)

            # Check if any pending batches are ready (waited long enough)
            now = time.time()
            ready = [c for c, t in pending_since.items() if now - t >= BATCH_WAIT]
            for contact in ready:
                texts = pending.pop(contact)
                pending_since.pop(contact)
                handle_batch(contact, texts)

        except Exception as e:
            print(f"{RED}[ERROR]{RESET} Poll error: {e}")

        # Sleep in small intervals so stop_event is responsive
        for _ in range(int(POLL_INTERVAL * 10)):
            if stop_event.is_set():
                return
            time.sleep(0.1)


# ============================================================
# Permissions Setup (first run only)
# ============================================================

def setup_permissions():
    """Guide user through macOS permissions on first run only.

    - Full Disk Access: must be granted manually (no auto-popup exists)
    - Contacts: macOS auto-prompts with Allow button — we just trigger it
    - Messages Automation: macOS auto-prompts with Allow button on first send
    """
    # Already done? Skip forever.
    if config.get("permissions_done"):
        # Still verify Full Disk Access works (in case they revoked it)
        conn = None
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
            conn.execute("SELECT COUNT(*) FROM message LIMIT 1")
            return
        except Exception:
            # They revoked it — tell them and fall through
            print(f"{YELLOW}Full Disk Access was revoked. Let's fix that real quick.{RESET}")
            print()
        finally:
            if conn:
                conn.close()

    # Try reading chat.db — if it works, FDA is already granted
    fda_ok = False
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.execute("SELECT COUNT(*) FROM message LIMIT 1")
        fda_ok = True
    except Exception:
        pass
    finally:
        if conn:
            conn.close()

    if not fda_ok:
        # This is the only permission that needs manual setup
        print(f"{WHITE}One quick thing before we start:{RESET}")
        print()
        print(wrap(f"  GhostReply reads your iMessages to learn how you text. macOS needs you to allow this — it takes 30 seconds.", 2))
        print()
        print(f"  {WHITE}1.{RESET} {GRAY}I'll open System Settings for you{RESET}")
        print(f"  {WHITE}2.{RESET} {GRAY}Find {WHITE}Terminal{GRAY} in the list and turn it on{RESET}")
        print(f"  {WHITE}3.{RESET} {GRAY}It'll ask you to quit Terminal — do it, then reopen and run {GREEN}ghostreply{RESET} {GRAY}again{RESET}")
        print()

        try:
            input(f"  {GRAY}Press Enter to open System Settings...{RESET}")
        except EOFError:
            pass

        # Open directly to Full Disk Access
        subprocess.run(
            ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"],
            check=False,
        )
        print()
        print(f"  {GRAY}After you turn on Terminal, quit Terminal and reopen it.{RESET}")
        print(f"  {GRAY}Then run {GREEN}ghostreply{RESET} {GRAY}again and you're good to go.{RESET}")
        print()
        sys.exit(0)

    # FDA is good — now silently trigger Contacts and Automation popups
    # These will show macOS "Allow" buttons automatically

    # Trigger Contacts popup by reading the DB
    ab_dir = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"
    if ab_dir.exists():
        conn = None
        try:
            for source_dir in ab_dir.iterdir():
                db_file = source_dir / "AddressBook-v22.abcddb"
                if db_file.exists():
                    conn = sqlite3.connect(str(db_file), timeout=5)
                    conn.execute("SELECT COUNT(*) FROM ZABCDRECORD LIMIT 1")
                    conn.close()
                    conn = None
                    break
        except Exception:
            pass  # they'll just see phone numbers instead of names
        finally:
            if conn:
                conn.close()

    # Trigger Messages automation popup with a harmless AppleScript
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Messages" to get name'],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass  # popup will appear on first real send anyway

    # Mark as done — never show again
    config["permissions_done"] = True
    save_config(config)


# ============================================================
# Main
# ============================================================

def main():
    global config, profile, custom_tone

    w = min(term_width() - 4, 40)
    inner = w - 2  # inside the box
    print()
    print(f"  {GRAY}╔{'═' * inner}╗{RESET}")
    line1 = f"GhostReply v{VERSION}"
    line2 = "iMessage Auto-Reply Bot"
    pad1 = (inner - len(line1)) // 2
    pad2 = (inner - len(line2)) // 2
    print(f"  {GRAY}║{RESET}{' ' * pad1}{BOLD}{WHITE}{line1}{RESET}{' ' * (inner - pad1 - len(line1))}{GRAY}║{RESET}")
    print(f"  {GRAY}║{RESET}{' ' * pad2}{GREEN}{line2}{RESET}{' ' * (inner - pad2 - len(line2))}{GRAY}║{RESET}")
    print(f"  {GRAY}╚{'═' * inner}╝{RESET}")
    print()

    # Auto-update from GitHub
    check_for_updates()

    if not DB_PATH.exists():
        print(f"{RED}[ERROR]{RESET} iMessage database not found at {DB_PATH}")
        print("Make sure you're running this on a Mac with iMessage set up.")
        sys.exit(1)

    # Load config early so permissions check can use it
    config = load_config()

    # Permissions setup (first run only)
    setup_permissions()

    # Discover contacts DB
    discover_contacts_db()

    # Load config + profile
    config = load_config()
    profile = load_profile()

    # First-time setup if needed (license + groq + scan messages)
    has_license = config.get("license_key")
    has_trial = config.get("trial_started_at")
    first_run = (not has_license and not has_trial) or not config.get("groq_api_key")

    if first_run:
        first_time_setup()
        config = load_config()
        profile = load_profile()
    else:
        # --- Returning user: validate license, pick contact, go ---

        # Re-scan if profile is missing
        if not profile.get("texting_style") or not profile.get("life_profile"):
            print(f"{GRAY}Scanning your messages...{RESET}")
            init_groq_client()
            my_texts = scan_my_messages(500)
            contacts = get_contacts_with_names()
            convos = scan_conversations_with_contacts(contacts, msgs_per_contact=40)
            if my_texts and not profile.get("texting_style"):
                style = analyze_texting_style(my_texts)
                profile["texting_style"] = style
            if convos and not profile.get("life_profile"):
                user_name = profile.get("name") or get_mac_user_name() or ""
                life = build_life_profile(my_texts or [], convos, user_name)
                profile["life_profile"] = life
            save_profile(profile)
            print(f"  {GREEN}✓{RESET} {GRAY}Profile ready.{RESET}")

        # Validate license or trial
        if config.get("trial_started_at"):
            elapsed = time.time() - config["trial_started_at"]
            hours_left = max(0, 24 - elapsed / 3600)
            if hours_left <= 0:
                print(f"{RED}Free trial expired!{RESET}")
                print(f"{GRAY}Buy a license at{RESET} {BLUE}https://hrampell.github.io/ghostreply{RESET}")
                print()
                key = input(f"Enter your license key: ").strip()
                if not key:
                    sys.exit(1)
                machine_id = get_machine_id()
                result = activate_license(key, machine_id)
                if result["status"] != "valid":
                    print(f"  {RED}{result['message']}{RESET}")
                    sys.exit(1)
                config["license_key"] = key
                config["machine_id"] = machine_id
                config["instance_id"] = result.get("instance_id", "")
                config.pop("trial_started_at", None)
                save_config(config)
                print(f"{GREEN}License activated!{RESET}")
            else:
                print(f"{GREEN}Free trial{RESET} — {BLUE}{hours_left:.1f} hours left{RESET}")
        else:
            license_key = config.get("license_key", "")
            if not license_key:
                print(f"{RED}No license key found.{RESET}")
                print(f"  Buy a license at {BLUE}https://hrampell.github.io/ghostreply{RESET}")
                sys.exit(1)
            print(f"{GRAY}Checking license...{RESET}", end=" ", flush=True)
            instance_id = config.get("instance_id", "")
            result = validate_license(license_key, instance_id)
            if result["status"] != "valid":
                print(f"{RED}FAILED{RESET}")
                print(f"  {result['message']}")
                print(f"  Buy a license at {BLUE}https://hrampell.github.io/ghostreply{RESET}")
                config.pop("license_key", None)
                save_config(config)
                sys.exit(1)
            config["license_validated"] = True
            save_config(config)
            print(f"{GREEN}OK{RESET}")

        # Initialize AI
        init_groq_client()

        # Always re-pick contact
        print()
        print(f"{BOLD}=== Who should GhostReply text for you? ==={RESET}")
        print()
        contacts = get_contacts_with_names()
        if not contacts:
            print(f"{RED}[ERROR]{RESET} No iMessage conversations found.")
            print(f"{GRAY}Send or receive at least one iMessage first, then run ghostreply again.{RESET}")
            sys.exit(1)
        recent = contacts[:5]
        display_names = format_contact_list(recent)
        for i, label in enumerate(display_names):
            print(f"  {WHITE}{i+1}.{RESET} {BLUE}{label}{RESET}")
        print()

        while True:
            choice = input(f"{WHITE}Pick a number or search by name:{RESET} ").strip()
            if not choice:
                continue
            if choice.isdigit() and 1 <= int(choice) <= len(recent):
                selected = recent[int(choice) - 1]
                break
            print(f"  {GRAY}Searching...{RESET}")
            matches = ai_find_contact(choice, contacts)
            if not matches:
                print(f"  {GRAY}No matches found. Try again.{RESET}")
                continue
            print()
            match_names = format_contact_list(matches)
            for i, label in enumerate(match_names):
                print(f"  {WHITE}{i+1}.{RESET} {BLUE}{label}{RESET}")
            print()
            pick = input(f"{WHITE}Pick a number:{RESET} ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(matches):
                selected = matches[int(pick) - 1]
                break

        target_first = selected["name"].split()[0] if selected["name"] and selected["name"].strip() else selected["handle"]
        config["target_contact"] = selected["handle"]
        config["target_name"] = target_first
        save_config(config)
        print(f"\n  {GREEN}✓{RESET} Auto-replying to {BLUE}{target_first}{RESET}")

        # Load conversation history
        recent_convo = load_recent_conversation(selected["handle"], limit=20)
        if recent_convo:
            conversation_history[selected["handle"]] = recent_convo

        # Customize?
        print()
        customize = input(f"{WHITE}Want to customize how the bot talks?{RESET} {GRAY}(y = custom personality, n = your natural texting style):{RESET} ").strip().lower()
        if customize in ("y", "yes"):
            tone = run_personality_chat(target_first)
            if tone:
                custom_tone = tone
                config["custom_tone"] = tone
                save_config(config)
        else:
            print(f"  {GREEN}✓{RESET} {GRAY}Using your natural texting style.{RESET}")

        # Send first message?
        print()
        first_msg = input(f"{WHITE}Send the first message to {BLUE}{target_first}{WHITE}?{RESET} {GRAY}(y/n):{RESET} ").strip().lower()
        if first_msg in ("y", "yes"):
            msg = input(f"  {WHITE}Type your message (or 'ai' to let the bot start):{RESET} ").strip()
            if msg.lower() == "ai":
                add_to_history(selected["handle"], "user", "(Start a casual conversation. Send a natural opener.)")
                opener = get_ai_response(selected["handle"])
                conversation_history[selected["handle"]] = []
                if opener:
                    add_to_history(selected["handle"], "assistant", opener)
                    print(f"  {GRAY}AI opener:{RESET} {GREEN}{opener}{RESET}")
                    confirm = input(f"  {WHITE}Send this?{RESET} {GRAY}(y/n):{RESET} ").strip().lower()
                    if confirm in ("y", "yes", ""):
                        if send_imessage(selected["handle"], opener):
                            print(f"  {GREEN}✓ Sent{RESET}")
                    else:
                        conversation_history[selected["handle"]] = []
                else:
                    print(f"  {YELLOW}AI couldn't generate an opener. Skipping.{RESET}")
            elif msg:
                if send_imessage(selected["handle"], msg):
                    add_to_history(selected["handle"], "assistant", msg)
                    print(f"  {GREEN}✓ Sent{RESET}")

    # Load saved custom tone
    if config.get("custom_tone") and not custom_tone:
        custom_tone = config["custom_tone"]

    # Make sure AI is initialized
    if not groq_client:
        init_groq_client()

    target_name = config.get("target_name", "?")
    print()
    print(f"{GREEN}{BOLD}GhostReply is running!{RESET} Replying to {BLUE}{target_name}{RESET}.")
    print(f"{GRAY}Type '{WHITE}stop{GRAY}' to quit, or just reply to {target_name} yourself — it'll stop automatically.{RESET}")
    print()

    # Start stdin listener in background thread
    listener = threading.Thread(target=stdin_listener, daemon=True)
    listener.start()

    try:
        poll_loop()
    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n{GRAY}GhostReply stopped.{RESET}")


def uninstall():
    """Remove GhostReply completely."""
    print()
    print(f"{BOLD}=== Uninstall GhostReply ==={RESET}")
    print()
    print(f"  This will remove:")
    print(f"  {GRAY}• ~/.ghostreply/ (config, profile, bot script){RESET}")
    print(f"  {GRAY}• 'ghostreply' alias from your shell config{RESET}")
    print()
    confirm = input(f"  {WHITE}Are you sure? (y/n):{RESET} ").strip().lower()
    if confirm not in ("y", "yes"):
        print(f"  {GRAY}Cancelled.{RESET}")
        return

    # Remove ~/.ghostreply/
    if CONFIG_DIR.exists():
        shutil.rmtree(CONFIG_DIR)
        print(f"  {GREEN}✓{RESET} {GRAY}Removed ~/.ghostreply/{RESET}")

    # Remove alias from shell config
    for rc_file in [Path.home() / ".zshrc", Path.home() / ".bash_profile"]:
        if rc_file.exists():
            try:
                lines = rc_file.read_text().splitlines()
                new_lines = [
                    l for l in lines
                    if "alias ghostreply=" not in l and "# GhostReply" not in l
                ]
                # Remove trailing blank lines left behind
                while new_lines and new_lines[-1].strip() == "":
                    new_lines.pop()
                rc_file.write_text("\n".join(new_lines) + "\n")
                print(f"  {GREEN}✓{RESET} {GRAY}Removed alias from {rc_file.name}{RESET}")
            except Exception:
                print(f"  {YELLOW}Could not clean {rc_file.name} — remove the 'ghostreply' alias manually.{RESET}")

    print()
    print(f"  {GREEN}GhostReply uninstalled.{RESET} Restart your terminal to finish.")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("--uninstall", "uninstall"):
            uninstall()
            sys.exit(0)
        if arg in ("--version", "-v"):
            print(f"GhostReply v{VERSION}")
            sys.exit(0)
        if arg in ("--help", "-h", "help"):
            print(f"GhostReply v{VERSION} — iMessage Auto-Reply Bot")
            print()
            print("Usage:")
            print("  ghostreply              Start GhostReply (setup or run)")
            print("  ghostreply --version    Show version")
            print("  ghostreply --uninstall  Remove GhostReply from your Mac")
            print()
            print("While running:")
            print("  Type 'stop' to quit")
            print("  Or just reply manually — it auto-stops when you do")
            print()
            print("Multiple contacts:")
            print("  Open another Terminal tab and run ghostreply again.")
            print("  Each tab handles a different contact.")
            sys.exit(0)
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n{GRAY}GhostReply stopped.{RESET}")
        sys.exit(0)
