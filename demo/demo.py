#!/usr/bin/env python3
"""GhostReply Demo Script — Simulates the full experience for screen recording.

Run this on ANY Mac. No iMessage, no config, no real data needed.
It reproduces the exact same terminal output as the real ghostreply.py.

Usage:
  python3 demo.py              # Fake mode (no real messages sent)
  python3 demo.py --live       # Live mode (sends real SMS + auto-detects replies)
"""

import subprocess
import sqlite3
import sys
import time
import shutil
import random
import re
from pathlib import Path

# --- Terminal Colors (EXACT same as ghostreply.py) ---
GREEN = "\033[92m"
BLUE = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"
SANDY = "\033[38;5;215m"
LIGHT_GRAY = "\033[38;5;250m"
SILVER = "\033[38;5;188m"
WHEAT = "\033[38;5;223m"

VERSION = "1.0.9"

# --- Config ---
FAKE_NAME = "Harrison"
LIVE_NUMBER = "+15105161057"
DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
FAKE_CONTACTS = [
    {"name": "Andrew", "handle": "+12125551234"},
    {"name": "Nikhil", "handle": "+15105559876"},
    {"name": "Kieran", "handle": "+14085553456"},
    {"name": "Jimothy", "handle": LIVE_NUMBER},
]

LIVE_MODE = "--live" in sys.argv


def term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def ensure_dark_terminal():
    """Same as _ensure_dark_terminal() in ghostreply.py."""
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and "Dark" in result.stdout:
            return
    except Exception:
        pass
    sys.stdout.write("\033]11;#1c1c1e\007")
    sys.stdout.write("\033]10;#ffffff\007")
    sys.stdout.flush()


def slow_print(text, delay=0.04, end="\n"):
    """Print text character by character for realistic typing effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def fake_input(prompt, response, delay_before=0.8, typing_speed=0.07):
    """Simulate user input: show prompt, then 'type' the response."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    time.sleep(delay_before)
    slow_print(response, delay=typing_speed)
    return response


def pause(seconds=1.0):
    time.sleep(seconds)


def send_sms(phone, text):
    """Send a real SMS via AppleScript (works with Google Voice numbers)."""
    # Escape quotes for AppleScript
    escaped = text.replace('\\', '\\\\').replace('"', '\\"')
    script = f'''
    tell application "Messages"
        set smsService to 1st service whose service type = SMS
        set targetBuddy to buddy "{phone}" of smsService
        send "{escaped}" to targetBuddy
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            # Fallback: try iMessage service in case SMS doesn't work
            script2 = f'''
            tell application "Messages"
                set targetService to 1st account whose service type = iMessage
                set targetBuddy to participant "{phone}" of targetService
                send "{escaped}" to targetBuddy
            end tell
            '''
            subprocess.run(["osascript", "-e", script2], capture_output=True, timeout=10)
    except Exception as e:
        print(f"  {RED}[SEND ERROR]{RESET} {e}")


def get_latest_rowid():
    """Get the latest message ROWID from chat.db."""
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        cur = conn.execute("SELECT MAX(ROWID) FROM message")
        row = cur.fetchone()
        conn.close()
        return row[0] or 0
    except Exception:
        return 0


def extract_text_from_attributed_body(blob):
    """Extract plain text from attributedBody blob."""
    try:
        if isinstance(blob, bytes):
            text = blob.decode("utf-8", errors="ignore")
            # Find the text between NSString and NSDictionary markers
            match = re.search(r'NSString\x01.(.+?)\x86', text)
            if match:
                return match.group(1)
            # Simpler fallback
            parts = text.split("NSString")
            if len(parts) > 1:
                chunk = parts[1]
                # Strip control chars
                clean = "".join(c for c in chunk if c.isprintable())
                if clean:
                    return clean[:500]
    except Exception:
        pass
    return None


def wait_for_incoming(baseline_rowid, phone):
    """Poll chat.db until a new message arrives from phone. Returns the message text."""
    # Normalize phone for matching — strip to last 10 digits
    phone_digits = re.sub(r'\D', '', phone)
    if len(phone_digits) > 10:
        phone_digits = phone_digits[-10:]

    while True:
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT m.ROWID, m.text, m.is_from_me, m.attributedBody, h.id as handle
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.ROWID > ?
                  AND m.is_from_me = 0
                ORDER BY m.ROWID ASC
            """, (baseline_rowid,))

            for row in cur.fetchall():
                handle = row["handle"] or ""
                handle_digits = re.sub(r'\D', '', handle)
                if len(handle_digits) > 10:
                    handle_digits = handle_digits[-10:]

                if handle_digits == phone_digits:
                    text = row["text"]
                    if text is None and row["attributedBody"] is not None:
                        text = extract_text_from_attributed_body(row["attributedBody"])
                    conn.close()
                    return row["ROWID"], (text or "").strip()

            conn.close()
        except Exception:
            pass

        time.sleep(1.0)


def reply_line(them_text, you_text, baseline_rowid=0):
    """Print a [REPLY] pair and optionally send a real SMS."""
    new_baseline = baseline_rowid

    if LIVE_MODE:
        # Wait for their message to actually arrive in chat.db
        new_baseline, actual_text = wait_for_incoming(baseline_rowid, LIVE_NUMBER)

    print(f"{GRAY}[REPLY]{RESET} {LIGHT_GRAY}them:{RESET} {them_text}")

    if LIVE_MODE:
        # Small delay to simulate "thinking"
        pause(random.uniform(1.5, 3.0))
        send_sms(LIVE_NUMBER, you_text)

    print(f"        {GREEN}you:{RESET}  {you_text}")

    return new_baseline


def main():
    ensure_dark_terminal()

    if LIVE_MODE:
        print(f"\n  {RED}{BOLD}LIVE MODE{RESET} — Real messages will be sent to {BLUE}{LIVE_NUMBER}{RESET}")
        print(f"  {GRAY}Script will auto-detect incoming messages and reply automatically.{RESET}")
        print(f"  {GRAY}Send replies from your Google Voice on your phone.{RESET}")
        print(f"  {GRAY}Press Enter to start...{RESET}")
        input()

    # =========================================
    # Header box (exact same as main())
    # =========================================
    w = min(term_width() - 4, 40)
    inner = w - 2
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

    pause(2.0)

    # =========================================
    # Step 1: License key / free trial
    # =========================================
    print(f"{BOLD}=== GhostReply Setup ==={RESET}")
    print()
    pause(0.8)

    fake_input(
        f"Enter your license key (or '{GREEN}trial{RESET}' for 24hr free trial, 'q' to quit): ",
        "trial",
        delay_before=1.5,
        typing_speed=0.10,
    )

    pause(0.5)
    print(f"{GREEN}Free trial activated! You have 24 hours.{RESET}")
    print(f"{GRAY}Buy a license at https://ghostreply.lol to keep using it.{RESET}")
    print()

    # Optional email
    fake_input(
        f"{GRAY}Enter your email to get notified before your trial ends (optional, press Enter to skip):{RESET} ",
        "",
        delay_before=1.2,
        typing_speed=0.05,
    )

    pause(0.8)

    # =========================================
    # Step 2: Groq API key
    # =========================================
    print()
    print(f"{BOLD}=== AI Setup ==={RESET}")
    print(f"GhostReply needs a free AI key from Groq {GREEN}(no credit card, takes 30 seconds){RESET}.")
    print()
    print(f"  {WHITE}1.{RESET} {GRAY}Sign up with Google (one click){RESET}")
    print(f"  {WHITE}2.{RESET} {GRAY}Click \"Create API Key\"{RESET}")
    print(f"  {WHITE}3.{RESET} {GRAY}Copy it — come back here and hit Enter{RESET}")
    print()

    pause(1.0)

    fake_input(
        f"  {BLUE}Press Enter to open groq.com in your browser...{RESET}",
        "",
        delay_before=2.0,
        typing_speed=0.05,
    )
    print(f"  {GREEN}Opening groq.com...{RESET}")
    print()

    pause(3.0)

    # Paste API key
    fake_input(
        f"\n  {BLUE}Paste your Groq API key here ('q' to quit):{RESET} ",
        "gsk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890ABCDEF",
        delay_before=2.0,
        typing_speed=0.012,  # fast paste
    )

    sys.stdout.write(f"  {GRAY}Verifying...{RESET} ")
    sys.stdout.flush()
    pause(2.0)
    print(f"{GREEN}OK{RESET}")

    pause(0.8)

    # =========================================
    # Step 3: Auto-detect name
    # =========================================
    print(f"\n{GRAY}Detected your name:{RESET} {WHITE}{FAKE_NAME} Rampell{RESET}")

    pause(1.2)

    # =========================================
    # Step 4: Scan messages
    # =========================================
    print()
    print(f"{BOLD}=== Scanning Your Messages ==={RESET}")
    print(f"{GRAY}Reading your iMessage history to learn how you text...{RESET}")
    print()

    # 4a
    sys.stdout.write(f"  {GRAY}[1/3] Pulling your sent messages...{RESET} ")
    sys.stdout.flush()
    pause(2.0)
    print(f"{GREEN}347 texts found.{RESET}")

    # 4b
    sys.stdout.write(f"  {GRAY}[2/3] Reading your conversations...{RESET} ")
    sys.stdout.flush()
    pause(2.5)
    print(f"{GREEN}12 conversations loaded.{RESET}")

    # 4c
    sys.stdout.write(f"  {GRAY}[3/3] Analyzing your texting style...{RESET} ")
    sys.stdout.flush()
    pause(3.5)
    print(f"{GREEN}done!{RESET}")

    pause(0.8)

    # Life profile
    print()
    sys.stdout.write(f"{GRAY}Building your profile from your conversations...{RESET} ")
    sys.stdout.flush()
    pause(4.0)
    print(f"{GREEN}done!{RESET}")

    print(f"  {GREEN}✓{RESET} {GRAY}Profile built from your messages.{RESET}")

    pause(0.8)

    # =========================================
    # Step 6: Swearing detection
    # =========================================
    print(f"\n  {GREEN}✓{RESET} {GRAY}Detected you swear in your texts — GhostReply will match that.{RESET}")
    print()

    fake_input(
        f"{WHITE}Are there contacts GhostReply should never say anything inappropriate to? (y/n, hit enter):{RESET} ",
        "n",
        delay_before=2.0,
        typing_speed=0.12,
    )

    pause(1.2)

    # =========================================
    # Step 7: Pick contact
    # =========================================
    print()
    print(f"{BOLD}=== Who should GhostReply text for you? ==={RESET}")
    print()

    for i, c in enumerate(FAKE_CONTACTS):
        print(f"  {WHITE}{i+1}.{RESET} {BLUE}{c['name']}{RESET}")
    print()

    fake_input(
        f"{WHITE}Type a number and hit Enter (or search by name):{RESET} ",
        "4",
        delay_before=2.5,
        typing_speed=0.15,
    )

    print(f"\n  {GREEN}✓{RESET} Auto-replying to {BLUE}Jimothy{RESET}")

    pause(0.8)

    # =========================================
    # Step 8: Personality — ragebait
    # =========================================
    print()
    fake_input(
        f"{WHITE}Want to customize how the bot talks?{RESET} {GRAY}(y = custom personality, n = your natural texting style):{RESET} ",
        "ragebait",
        delay_before=2.0,
        typing_speed=0.08,
    )
    print(f"  {RED}☠ Ragebait activated.{RESET}")

    pause(0.8)

    # Send first message — ragebait auto-sends
    print()
    sys.stdout.write(f"  {GRAY}Generating opener...{RESET} ")
    sys.stdout.flush()
    pause(2.0)
    if LIVE_MODE:
        send_sms(LIVE_NUMBER, "wtf do u want")
    print(f"{GREEN}Sent:{RESET} wtf do u want")

    pause(0.5)

    # =========================================
    # Setup complete
    # =========================================
    print()
    print(f"{GREEN}Setup complete!{RESET} {GRAY}Everything was learned from your messages.{RESET}")
    print()

    pause(1.5)

    # =========================================
    # Running! — Ragebait mode
    # =========================================
    print(f"{GREEN}{BOLD}GhostReply is running!{RESET} Replying to {BLUE}Jimothy{RESET}.")
    print()
    print(f"  {GRAY}To stop: type {WHITE}stop{GRAY} and hit Enter, or press {WHITE}Ctrl+C{RESET}")
    print(f"  {GRAY}Or just reply to Jimothy yourself — it'll stop automatically.{RESET}")
    print()

    print(f"{GRAY}[INFO]{RESET} Listening for messages from {BLUE}Jimothy{RESET}...")
    print(f"{GRAY}[INFO]{RESET} {GRAY}Auto-stops when you reply manually.{RESET}")
    print()

    # Get baseline ROWID so we only detect NEW messages
    baseline = get_latest_rowid() if LIVE_MODE else 0

    if not LIVE_MODE:
        pause(5.0)

    # Ragebait conversation
    baseline = reply_line("u wanna know harrisons lore?", "no idgaf", baseline)

    if not LIVE_MODE:
        pause(random.uniform(7, 10))

    baseline = reply_line("Wb my lore", "kys", baseline)

    if not LIVE_MODE:
        pause(random.uniform(6, 9))

    baseline = reply_line("🍆", "u wish u had one that big", baseline)

    if not LIVE_MODE:
        pause(6.0)
    else:
        pause(3.0)

    # Auto-stop
    print()
    print(f"{GREEN}[AUTO-STOP]{RESET} You replied to {BLUE}Jimothy{RESET} manually — GhostReply stopped.")
    print(f"{GRAY}Run {WHITE}ghostreply{GRAY} again to restart.{RESET}")
    print()


if __name__ == "__main__":
    main()
