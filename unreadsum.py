"""R48: "anything new from clients?" while driving. A read-only digest of recent messages that other modules (email, iMessage,
WeChat) write as JSON files into config digest_dir (default "" = off). Each file is a list of
{"ts": ISO time or epoch, "channel", "from_name", "one_line_summary", "needs_reply": bool}. Dot only reads and formats."""
import datetime as dt
import glob
import json
import os
import time

import config

NONE_TEXT = "There is no message digest yet."
MAX_ITEMS = 8
MAX_CHARS = 1000


def _dir():
    try:
        d = config.get("digest_dir")
    except KeyError:
        return ""
    return os.path.expanduser(d) if isinstance(d, str) and d.strip() else ""


def _ts(v):
    """Epoch seconds from an ISO string or a number, or None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v / 1000.0 if v > 1e11 else v)
    try:
        d = dt.datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (d if d.tzinfo else d.astimezone()).timestamp()


def _one(s, n):
    return " ".join(str(s or "").split())[:n]


def load_items(folder, since_hours=12, now=None):
    """All valid items newer than since_hours, from every *.json in folder. Bad files and bad items are skipped."""
    now = time.time() if now is None else now
    cutoff = now - since_hours * 3600
    items = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        for it in data:
            if not isinstance(it, dict):
                continue
            t = _ts(it.get("ts"))
            if t is None or t < cutoff or t > now + 3600:
                continue
            items.append({"t": t, "channel": _one(it.get("channel"), 20) or "other", "who": _one(it.get("from_name"), 40) or "someone",
                          "text": _one(it.get("one_line_summary"), 120), "reply": it.get("needs_reply") is True})
    return items


def format_digest(items, since_hours=12):
    items = sorted(items, key=lambda i: (not i["reply"], -i["t"]))
    if not items:
        return "Nothing new in the last %d hours." % since_hours
    counts = {}
    for i in items:
        counts[i["channel"]] = counts.get(i["channel"], 0) + 1
    need = sum(1 for i in items if i["reply"])
    head = "%d new message%s in the last %d hours (%s)" % (
        len(items), "" if len(items) == 1 else "s", since_hours, ", ".join("%d %s" % (c, k) for k, c in sorted(counts.items())))
    head += "; %d need%s a reply." % (need, "s" if need == 1 else "") if need else "; none need a reply."
    lines = []
    size = len(head)
    for i in items[:MAX_ITEMS]:
        line = "%s%s on %s: %s" % ("Needs reply, " if i["reply"] else "", i["who"], i["channel"], i["text"])
        if size + len(line) + 1 > MAX_CHARS:
            break
        lines.append(line)
        size += len(line) + 1
    text = head + " " + " | ".join(lines)
    if len(lines) < len(items):
        text += " | %d more not read." % (len(items) - len(lines))
    return text[:MAX_CHARS + 40]


def unread_digest(since_hours=12, now=None):
    """(text for Dot to say, number of items or -1 on failure/off/no digest). Never raises."""
    try:
        folder = _dir()
        if not folder or not os.path.isdir(folder) or not glob.glob(os.path.join(folder, "*.json")):
            return NONE_TEXT, -1
        try:
            since_hours = max(1, min(72, int(since_hours)))
        except (TypeError, ValueError):
            since_hours = 12
        items = load_items(folder, since_hours, now)
        return format_digest(items, since_hours), len(items)
    except Exception:
        return NONE_TEXT, -1
