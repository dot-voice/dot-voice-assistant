"""R38 call recap and R39 per-call cost/latency line. Both are off unless the config flags are on.

One CallRecap object follows one Live call (page Live call or Twilio bridge call) from "live on" to "live off". At the end
the server asks it for a recap file (logs/recaps/<date>-<time>.md), an optional Telegram copy, and one call_summary line.
No client phone numbers or emails are ever written to the recap (scrubbed here)."""
import math
import os
import re
import subprocess
import time
import urllib.request
import json

import config

TWILIO_USD_PER_MIN = 0.0125
KEEP_DAYS = 14
# Keychain item of the Telegram bot token (config: telegram_keychain_service, telegram_keychain_account)
_PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def scrub(text):
    return _PHONE.sub("[number]", _EMAIL.sub("[email]", str(text or "")))


def short(text, n=120):
    t = " ".join(scrub(text).split())
    return t if len(t) <= n else t[:n - 3].rstrip() + "..."


def percentile(values, pct):
    vals = sorted(values)
    if not vals:
        return 0
    return vals[min(len(vals) - 1, int(math.ceil(pct / 100.0 * len(vals))) - 1)]


def twilio_usd(secs):
    return round(math.ceil(max(0, secs) / 60.0) * TWILIO_USD_PER_MIN, 4) if secs > 0 else 0.0


class CallRecap:
    def __init__(self, bridge_call=False, clock=time.time):
        self.clock = clock
        self.started = clock()
        self.bridge_call = bridge_call
        self.turns = 0
        self.first_audio_ms = []
        self.lookups = []
        self.talk_used = False

    def note_turn(self):
        self.turns += 1

    def note_first_audio(self, ms):
        self.first_audio_ms.append(int(ms))

    def note_lookup(self, words):
        self.lookups.append(short(words, 80))

    def note_talk(self):
        self.talk_used = True

    def secs(self):
        return int(max(0, self.clock() - self.started))

    def summary_line(self):
        s = self.secs()
        line = "call_summary: secs=%d turns=%d first_audio_p50_ms=%d p95_ms=%d" % (
            s, self.turns, percentile(self.first_audio_ms, 50), percentile(self.first_audio_ms, 95))
        if self.bridge_call:
            line += " twilio_usd~=%s" % twilio_usd(s)
        return line

    def build(self, history, rows, label_of=lambda k: k):
        """The recap text. history = [{"role","text"}], rows = the call's hand-off rows."""
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started))
        secs = self.secs()
        out = ["# Call recap", "", "- Started: %s" % started,
               "- Duration: %d min %d s%s" % (secs // 60, secs % 60, " (phone call)" if self.bridge_call else " (call page)")]
        asked = [short(h["text"]) for h in history if h.get("role") == "user" and not h.get("talk") and str(h.get("text", "")).strip()]
        out += ["", "## What she asked"] + (["- " + a for a in asked[-12:]] or ["- (nothing recorded)"])
        did = []
        for r in rows:
            note = {"replied": "replied", "sent": "sent, no reply yet", "waiting": "not sent yet",
                    "cancelled": "cancelled"}.get(r.get("status"), "did not go through")
            did.append("- Hand-off to %s: %s (%s)" % (label_of(r.get("key", "AUTO")), short(r.get("snippet", ""), 60), note))
        did += ["- Lookup: %s" % w for w in self.lookups[-10:]]
        if self.talk_used:
            did.append("- talk mode used")
        out += ["", "## What Dot did"] + (did or ["- Nothing handed off or looked up."])
        open_items = ["- Waiting on %s: %s" % (label_of(r.get("key", "AUTO")), short(r.get("snippet", ""), 60))
                      for r in rows if r.get("status") in ("sent", "waiting")]
        open_items += ["- Failed, ask again: %s (%s)" % (short(r.get("snippet", ""), 60), label_of(r.get("key", "AUTO")))
                       for r in rows if r.get("status") not in ("sent", "waiting", "replied", "cancelled")]
        out += ["", "## Open items"] + (open_items or ["- None."])
        return "\n".join(out) + "\n"


def write_recap(log_dir, text, now=None):
    """Write logs/recaps/<date>-<time>.md and move files older than KEEP_DAYS into recaps/old/ (never delete)."""
    now = now or time.time()
    d = os.path.join(log_dir, "recaps")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, time.strftime("%Y-%m-%d-%H%M%S", time.localtime(now)) + ".md")
    n = 1
    while os.path.exists(path):   # two calls in the same second never overwrite each other
        n += 1
        path = os.path.join(d, time.strftime("%Y-%m-%d-%H%M%S", time.localtime(now)) + "-%d.md" % n)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    prune(d, now)
    return path


def prune(recap_dir, now=None):
    now = now or time.time()
    old = os.path.join(recap_dir, "old")
    for name in os.listdir(recap_dir):
        p = os.path.join(recap_dir, name)
        if name.endswith(".md") and os.path.isfile(p) and now - os.path.getmtime(p) > KEEP_DAYS * 86400:
            os.makedirs(old, exist_ok=True)
            os.replace(p, os.path.join(old, name))


def telegram_token():
    try:
        r = subprocess.run(["security", "find-generic-password", "-a", config.get("telegram_keychain_account"), "-s", config.get("telegram_keychain_service"), "-w"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def send_telegram(text, chat_id, token_fn=telegram_token, urlopen=urllib.request.urlopen):
    """Same Bot API call as wechat-assistant/sender/tg_wechat_bot.py (sendMessage). True when sent; never raises."""
    chat_id = chat_id or os.environ.get("VHBOT_CHAT_ID", "")
    token = token_fn()
    if not chat_id or not token:
        return False
    req = urllib.request.Request("https://api.telegram.org/bot%s/sendMessage" % token,
                                 data=json.dumps({"chat_id": chat_id, "text": text[:4000]}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=15) as r:
            return bool(json.load(r).get("ok"))
    except Exception:
        return False
