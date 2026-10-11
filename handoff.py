"""Hand real work to the right Code session and keep a local record of every hand-off.

Each hand-off runs the configured carry script as `carry.py tell <ROLE> "<task> | caller words" --task <KEY>`:
the session the task names (route()), ORG when unclear. Tasks about this call stay in handoff/ for the VOICE session."""
import json
import os
import re
import subprocess
import time

import config

HERE = os.path.dirname(os.path.abspath(__file__))
CARRY = config.path("carry_script")
HANDOFF_ROLE = os.environ.get("VOICE_HANDOFF_ROLE", "AUTO")   # AUTO = route() below; a role name forces one session


def tell_role(role, text):
    """Leave a short message for another Code session. carry.py caps text at 600 chars."""
    p = subprocess.run(["/usr/bin/python3", CARRY, "tell", role, text[:590], "--task", "VOICE"],
                       capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError("carry.py tell failed: " + (p.stderr or p.stdout).strip()[-200:])
    return p.stdout.strip()[-200:]


class Handoff:
    def create(self, task: str, spoken: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


# Route by what the task names: the call itself -> VOICE (kept here, this session reads handoff/), a named module -> that
# session, anything unclear -> ORG. First match wins, so the specific patterns come first.
ROUTES = [tuple(r) for r in config.get("handoff_routes")]


# The menu on the call page lets the caller pick which session gets the request. The most used sessions come first. Keys are carry.py roles.
TARGETS = [tuple(t) for t in config.get("handoff_targets")]
TARGET_KEYS = {k for k, _ in TARGETS}


def route(task, spoken):
    text = (str(task or "") + " " + str(spoken or "")).lower()
    for role, pattern in ROUTES:
        if re.search(pattern, text):
            return role
    return "ORG"


class FileHandoff(Handoff):
    """Always writes handoff/<timestamp>.json. Also tells the routed session (post=True)."""

    def __init__(self, folder=None, post=True, teller=None, role=None):
        self.folder = folder or os.path.join(HERE, "handoff")
        self.post = post
        self.teller = teller or tell_role
        self.role = role or HANDOFF_ROLE

    def create(self, task, spoken, role=None):
        os.makedirs(self.folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rec = {"id": stamp, "task": task, "spoken": spoken, "source": "voice_call", "created": stamp, "posted": False}
        forced = role if role in TARGET_KEYS and role != "AUTO" else None   # the caller's pick in the call's menu wins
        target = forced or (route(task, spoken) if self.role == "AUTO" else self.role)
        rec["to"] = target
        if self.post and target == "VOICE":
            rec["posted"] = True   # about the call itself: stays here in handoff/, the VOICE session works it
        elif self.post:
            # The exact words only, never Dot's summary. Dot's task line stays in the local json for reference.
            words = " ".join(str(spoken or "").split())
            where = os.path.join(HERE, "handoff", stamp + ".json")
            if re.search(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]", words):   # carry.py refuses CJK text
                words = "(the caller spoke Chinese; their exact words are in %s)" % where
            elif len(words) > 480:
                words = words[:480] + "... (rest of their exact words in %s)" % where
            msg = "%s said by voice (exact words): \"%s\"" % (config.get("owner"), words)
            try:
                rec["told"] = self.teller(target, msg)
                rec["posted"] = True
            except Exception as e:  # unknown role or carry.py down: ORG gets it, then the local file is the last resort
                rec["post_error"] = str(e)[:200]
                if target != "ORG":
                    try:
                        rec["told"] = self.teller("ORG", msg + " (meant for %s)" % target)
                        rec["posted"], rec["to"] = True, "ORG"
                    except Exception as e2:
                        rec["post_error"] += " | " + str(e2)[:200]
        path = os.path.join(self.folder, stamp + ".json")
        n = 1
        while os.path.exists(path):
            n += 1
            path = os.path.join(self.folder, "%s-%d.json" % (stamp, n))
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        rec["path"] = path
        return rec


class FakeHandoff(Handoff):
    def __init__(self):
        self.items = []

    def create(self, task, spoken, role=None):
        self.items.append((task, spoken))
        self.roles = getattr(self, "roles", []) + [role]
        return {"id": str(len(self.items)), "task": task, "posted": False}
