"""Replies from the Code sessions, spoken to the owner in the call.

A session that finished something the owner asked for by voice runs:
    python3 <this folder>/updates.py <ROLE> "<one or two short spoken sentences, English>"
The update waits in updates/ until a call is open and Dot is quiet, then Dot says it. Files are never deleted:
a spoken one gets "spoken" set.
"""
import json
import os
import re
import sys
import time

import config

HERE = os.path.dirname(os.path.abspath(__file__))
# Kept outside this folder so every session's sandbox may write there; this folder belongs to the voice session only.
FOLDER = os.environ.get("VOICE_UPDATES_DIR", config.path("updates_dir"))
MAX_CHARS = 400
MAX_AGE_S = 2 * 24 * 3600   # older than two days: stale news, not spoken


def post(role, text, folder=FOLDER):
    role = re.sub(r"[^A-Z0-9-]", "", str(role).upper())[:20] or "SESSION"
    text = " ".join(str(text).split())[:MAX_CHARS]
    if not text:
        raise ValueError("empty update")
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(folder, "%s-%s.json" % (stamp, role))
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, "%s-%s-%d.json" % (stamp, role, n))
    with open(path, "w") as f:
        json.dump({"from": role, "text": text, "created": time.time(), "spoken": False}, f, indent=2)
    return path


def pending(folder=FOLDER, now=None):
    """Unspoken, recent updates, oldest first: [(path, record)]."""
    now = now or time.time()
    out = []
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
    except OSError:
        return out
    for n in names:
        p = os.path.join(folder, n)
        try:
            rec = json.load(open(p))
        except (OSError, ValueError):
            continue
        if not rec.get("spoken") and now - float(rec.get("created", 0)) < MAX_AGE_S:
            out.append((p, rec))
    return out


def mark_spoken(path):
    try:
        rec = json.load(open(path))
        rec["spoken"], rec["spoken_at"] = True, time.time()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, path)
    except (OSError, ValueError):
        pass


def recent_lines(folder=FOLDER, now=None, limit=6, max_chars=1200):
    """The newest updates (spoken or not), newest first, for the front desk: what sessions really did, in their words."""
    now = now or time.time()
    rows = []
    try:
        names = sorted((n for n in os.listdir(folder) if n.endswith(".json")), reverse=True)
    except OSError:
        return ""
    for n in names:
        try:
            rec = json.load(open(os.path.join(folder, n)))
        except (OSError, ValueError):
            continue
        if rec.get("from") == "TEST" or now - float(rec.get("created", 0)) > MAX_AGE_S:
            continue
        rows.append("Update from %s at %s: %s" % (rec.get("from"), time.strftime("%H:%M", time.localtime(rec["created"])), rec.get("text", "")))
        if len(rows) >= limit:
            break
    return "\n".join(rows)[:max_chars]


def last_update(role, folder=FOLDER):
    """The newest update a session wrote for the owner (spoken or not), or None."""
    try:
        names = sorted((n for n in os.listdir(folder) if n.endswith(".json")), reverse=True)
    except OSError:
        return None
    for n in names:
        try:
            rec = json.load(open(os.path.join(folder, n)))
        except (OSError, ValueError):
            continue
        if rec.get("from") == role and time.time() - float(rec.get("created", 0)) < MAX_AGE_S:
            return rec
    return None


def spoken_line(rec):
    who = {"VOICE": "Voice", "ORG": "ORG"}.get(rec.get("from"), str(rec.get("from", "")).title())
    return "Update from %s: %s" % (who, rec.get("text", ""))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: updates.py <ROLE> "<short spoken update, English>"')
    print("QUEUED for the call: " + post(sys.argv[1], sys.argv[2]))
