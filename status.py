"""A short, read-only status of the owner's agent sessions (one per module), so the call can answer 'what is going on'.

The old job queue (<team_jobs_dir>) is retired. Each agent session keeps a handover in
<carryover_dir> (index: TASKS.json); this reads each one's "Next step". team_status() is kept for old tests."""
import glob
import json
import os
import re
import time

import config

CARRYOVER = os.environ.get("VOICE_CARRYOVER", config.path("carryover_dir"))
# Personal sessions are never read out on the call (the reply text also goes to Qwen for the voice).
PRIVATE_ROLES = set(os.environ["VOICE_PRIVATE_ROLES"].split(",") if "VOICE_PRIVATE_ROLES" in os.environ else config.get("private_roles"))
TEAM_JOBS = os.environ.get("VOICE_TEAM_JOBS", config.path("team_jobs_dir"))
ACTIVE = ("RUNNING", "QUEUED", "NEEDS_YOU", "NEEDS_COWORK")
WORDS = {"RUNNING": "running", "QUEUED": "waiting to start", "NEEDS_YOU": "waiting for " + config.get("owner"), "NEEDS_COWORK": "waiting for an agent session",
         "DONE": "finished"}
STATUS_HINTS = ("inbox", "session", "org ", "what is going on", "going on", "\u6536\u4ef6\u7bb1", "task", "job", "status", "running", "finished", "progress", "board", "update", "what's happening", "whats happening",
                "\u4efb\u52a1", "\u8fdb\u5ea6", "\u72b6\u6001", "\u5b8c\u6210", "\u5728\u8dd1", "\u770b\u677f", "\u7ed3\u679c")


# Only names that are not everyday words (an everyday phrase must not become a status question); "the <name> session" still works.
SESSION_NAMES = tuple(config.get("session_names"))


def wants_status(text: str) -> bool:
    t = text.lower()
    if any(h in t for h in STATUS_HINTS):
        return True
    words = set(re.findall(r"[a-z][a-z-]*", t))   # a session name as a whole word: "what is CRM working on"
    return any((n in words) if " " not in n else (n in t) for n in SESSION_NAMES)


def _first_lines(path, n=2, width=140):
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip(" #*-\t") for ln in f if ln.strip()]
    except OSError:
        return ""
    return " / ".join(lines[:n])[:width]


def team_status(jobs_dir=None, now=None, max_chars=1600, recent_hours=12) -> str:
    jobs_dir = jobs_dir or TEAM_JOBS
    now = now or time.time()
    active, done = [], []
    for f in glob.glob(os.path.join(jobs_dir, "*", "state.json")):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        state = d.get("state")
        if state not in ACTIVE and state != "DONE":
            continue
        mtime = os.path.getmtime(f)
        if state == "DONE" and now - mtime > recent_hours * 3600:
            continue
        line = "#%s %s (%s)" % (d.get("num") or d.get("id", "?")[:4].lstrip("0"), str(d.get("title", ""))[:70], WORDS[state])
        if state == "DONE":
            line += ": " + _first_lines(os.path.join(os.path.dirname(f), "RESULT.md"))
        (active if state in ACTIVE else done).append((mtime, line))
    active.sort(reverse=True)
    done.sort(reverse=True)
    out = ["TEAM STATUS (read just now from the team folder):"]
    out += ["- " + ln for _, ln in active[:8]] or ["- nothing is running or waiting"]
    if done:
        out.append("Finished in the last %d hours:" % recent_hours)
        out += ["- " + ln for _, ln in done[:5]]
    return "\n".join(out)[:max_chars]


def _section(text, name):
    out, on = [], False
    for ln in text.splitlines():
        if ln.startswith("## "):
            if on:
                break
            on = ln[3:].strip().lower() == name
            continue
        if on and ln.strip():
            out.append(ln.strip(" -*\t"))
    return " ".join(out)


ROLES_JSON = os.environ.get("VOICE_ROLES_JSON", config.path("roles_json"))


def session_titles(path=None) -> dict:
    """Task key -> the session name the owner sees (from roles_json), without emoji."""
    try:
        roles = json.load(open(path or ROLES_JSON))
    except (OSError, ValueError):
        return {}
    out = {}
    for d in roles.values():
        if not isinstance(d, dict):
            continue
        titles = d.get("task_titles") or {}
        for task in d.get("tasks", []):
            t = titles.get(task) or (d.get("title") if len(d.get("tasks", [])) == 1 else "")
            name = re.sub(r"^[^A-Za-z]+", "", t or "").strip()
            if name:
                out[task] = name
    return out


def sessions_status(folder=None, now=None, max_chars=1800, recent_hours=48, width=150, titles=None) -> str:
    folder = folder or CARRYOVER
    now = now or time.time()
    titles = session_titles() if titles is None else titles
    try:
        index = json.load(open(os.path.join(folder, "TASKS.json")))
    except (OSError, ValueError):
        return "SESSIONS STATUS: the sessions' handovers cannot be read right now."
    rows = []
    for role, d in index.items():
        if not isinstance(d, dict) or not d.get("file") or not role.isupper() or role in PRIVATE_ROLES:
            continue
        name = titles.get(role, role)
        if name != role:
            name = "%s (key %s)" % (name, role)
        path = os.path.join(folder, d["file"])
        try:
            mtime = os.path.getmtime(path)
            text = open(path, encoding="utf-8").read()
        except OSError:   # registered but never saved: still say it exists, so the caller is not told about the wrong session
            rows.append((0, "%s: no handover saved yet, so its status is unknown" % name))
            continue
        if now - mtime > recent_hours * 3600:
            continue
        nxt = _section(text, "next step")[:width]
        when = time.strftime("%a %H:%M", time.localtime(mtime))
        rows.append((mtime, "%s (saved %s): next step: %s" % (name, when, nxt or "not written")))
    rows.sort(reverse=True)
    out = ["SESSIONS STATUS (the owner's agent sessions, read just now from their handovers; last %d hours):" % recent_hours]
    out += ["- " + ln for _, ln in rows[:12]] or ["- no session saved a handover recently"]
    return "\n".join(out)[:max_chars]


# ----------------------------------------------------------------------------- status read word for word (no model)
# Precision rule: never claim work happened without a tool result. A question about ONE session is answered by code
# from that session's own saved words, read verbatim; the chat model never retells it.
ALIASES = [tuple(a) for a in config.get("session_aliases")]
ASKING = re.compile(r"\b(what|how|status|doing|done|progress|where|finished|working on|up to)\b", re.I)
ORDERING = re.compile(r"\b(tell|ask|send|forward|have|make|let|please fix|fix|change|move|add|remove|delete)\b", re.I)


GENERAL = re.compile(r"going on|what's happening|whats happening|status|all (the )?sessions|everyone|every session|"
                     r"what are (they|the sessions) doing|any (news|updates)", re.I)


def overview(folder, now, titles, n=3):
    """The three most recently saved sessions, each in its own words."""
    try:
        index = json.load(open(os.path.join(folder, "TASKS.json")))
    except (OSError, ValueError):
        return "I cannot read the sessions' notes right now."
    rows = []
    for role, d in index.items():
        if not isinstance(d, dict) or not d.get("file") or not role.isupper() or role in PRIVATE_ROLES:
            continue
        try:
            path = os.path.join(folder, d["file"])
            mtime = os.path.getmtime(path)
            nxt = _section(open(path, encoding="utf-8").read(), "next step")
        except OSError:
            continue
        if nxt:
            rows.append((mtime, "%s, %s: %s" % (titles.get(role, role), _ago(now - mtime), _first_sentence(nxt, 120))))
    rows.sort(reverse=True)
    if not rows:
        return "No session saved notes recently."
    return "Latest from your sessions, in their own words. " + " ".join(r for _, r in rows[:n])


def named_session(text):
    """The one session a question names, or None (zero or several named)."""
    t = " " + str(text or "").lower() + " "
    found = []
    for role, pat in ALIASES:
        if re.search(r"\b(%s)\b" % pat, t):
            found.append(role)
            t = re.sub(r"\b(%s)\b" % pat, " ", t)   # a longer name must not also count as its shorter parts
    return found[0] if len(found) == 1 else None


def _first_sentence(s, width=170):
    s = " ".join(str(s).split())
    m = re.search(r"^(.{20,%d}?[.!?])(\s|$)" % width, s)
    if m:
        return m.group(1).strip()
    if len(s) <= width:
        return s
    cut = s[:width]
    cut = cut[:cut.rfind(";")] if cut.rfind(";") > width // 2 else cut[:cut.rfind(" ")]   # never stop mid-word
    return cut.rstrip(" ,;") + "."


def _ago(seconds):
    m = int(seconds // 60)
    if m < 2:
        return "just now"
    if m < 90:
        return "%d minutes ago" % m
    h = round(m / 60)
    return "%d hours ago" % h if h < 36 else "%d days ago" % round(h / 24)


def direct_status(text, folder=None, now=None, updates_fn=None, titles=None):
    """A spoken answer built only from the session's own words, or None when this is not a one-session status question."""
    if not ASKING.search(text or "") or ORDERING.search(text or ""):
        return None
    role = named_session(text)
    folder = folder or CARRYOVER
    now = now or time.time()
    titles = session_titles() if titles is None else titles
    if not role and GENERAL.search(text or ""):
        return overview(folder, now, titles)
    if not role or role in PRIVATE_ROLES:
        return None
    name = titles.get(role, role)
    parts = []
    try:
        d = json.load(open(os.path.join(folder, "TASKS.json"))).get(role) or {}
        path = os.path.join(folder, d["file"])
        mtime = os.path.getmtime(path)
        nxt = _section(open(path, encoding="utf-8").read(), "next step")
        parts.append("%s saved its notes at %s, %s. Its next step, in its own words: %s" % (
            name, time.strftime("%H:%M", time.localtime(mtime)), _ago(now - mtime), _first_sentence(nxt) if nxt else "not written"))
    except (OSError, ValueError, KeyError):
        parts.append("%s has no saved notes I can read" % name)
    try:
        if updates_fn is None:
            from updates import last_update
            updates_fn = last_update
        up = updates_fn(role)
        if up:
            parts.append("Its last update to you, at %s: %s" % (time.strftime("%H:%M", time.localtime(up["created"])), up["text"]))
    except Exception:
        pass
    return ". ".join(p.rstrip(".") for p in parts) + "."
