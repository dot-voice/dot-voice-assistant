"""Save the spoken passphrase for the phone line (twilio_auth = "phrase") in the macOS Keychain.

    python3 set_phrase.py        # asks twice with hidden input, prints DONE and exits at once

The phrase is stored normalized (lowercase, no punctuation) in the Keychain item "dot-voice-phrase". It is never printed.
"""
import getpass
import subprocess
import sys

import twilio_bridge as tb


def save(phrase, run=subprocess.run):
    norm = tb.normalize_phrase(phrase)
    if len(norm.split()) < 3:
        raise ValueError("use at least 3 words")
    run(["security", "add-generic-password", "-U", "-s", tb.PHRASE_ITEM, "-a", "dot", "-w", norm], check=True,
        capture_output=True, timeout=10)
    return len(norm.split())


def ask(prompt):
    """Hidden input: a macOS dialog (works when run through clog without a terminal), else getpass."""
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    script = 'text returned of (display dialog "%s" default answer "" with hidden answer with title "Dot")' % prompt
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SystemExit("ERROR: dialog cancelled, nothing saved")
    return r.stdout.rstrip("\n")


def main():
    a = ask("Dot passphrase (hidden, 3+ words):")
    if a != ask("Same passphrase again:"):
        raise SystemExit("ERROR: the two entries differ, nothing saved")
    try:
        n = save(a)
    except (ValueError, subprocess.SubprocessError, OSError) as e:
        raise SystemExit("ERROR: %s" % type(e).__name__ if not isinstance(e, ValueError) else "ERROR: %s" % e)
    print("DONE: phrase saved (%d words)" % n)


if __name__ == "__main__":
    main()
