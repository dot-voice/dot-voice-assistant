#!/usr/bin/env python3
"""Hands-free voice call to the assistant. One port serves the page and the WebSocket.

Listens on 127.0.0.1 by default; `tailscale serve` puts HTTPS (a real certificate) in front of it, for the tailnet only.
Every connection needs the PIN. Speech to text is local. Only the text of a turn goes to the front-desk model.

Run:   python3 server.py --run
Check: python3 server.py --check-config      Self test: python3 server.py --selftest
"""
import argparse
import asyncio
import base64
import hmac
import json
import os
import queue
import re
import secrets
import shutil
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
from audio import split_sentences  # noqa: E402
from diag import log_exception, log_line  # noqa: E402
from auth import PinGuard, host_allowed, load_or_create_pin, login_ok, origin_ok  # noqa: E402
from frontdesk import ClaudeFrontDesk, FakeFrontDesk, Reply  # noqa: E402
from status import direct_status  # noqa: E402
from version import BUILT, VERSION  # noqa: E402
from handoff import TARGET_KEYS, TARGETS, FakeHandoff, FileHandoff  # noqa: E402
from handoff import route as handoff_route  # noqa: E402
from stt import FakeSpeechToText, WhisperSpeechToText, available_engine  # noqa: E402
import tts as tts_mod  # noqa: E402
import updates  # noqa: E402
import askmodel  # noqa: E402
import handoff as handoff_mod  # noqa: E402
import propfacts  # noqa: E402
import lastcall  # noqa: E402
import nextevents  # noqa: E402
import unreadsum  # noqa: E402
import callrecap  # noqa: E402
import callback  # noqa: E402
from tts import FakeTextToSpeech, default_tts, find_and_store_qwen_key, QwenTextToSpeech  # noqa: E402
import realtime  # noqa: E402
import transfer  # noqa: E402

PORT = 8770
AUTH_TIMEOUT_S = 20
BUG_TIMEOUT_S = 60
RESUME_TTL_S = 12 * 3600   # a reconnect token lasts one working day
RESUME_MAX = 10
CALL_MEMORY_S = 30 * 60    # a dropped call's memory waits this long for the phone to come back
HISTORY_MAX = 40           # lines kept per call
SEED_LINES, SEED_CHARS = 20, 3000
HANDOFF_HOLD_S = 6.0       # Live: a hand-off waits this long so "no, I meant CRM" can still correct it   # what a new Live session is told about the call so far


class CallState:
    """What one phone call remembers across reconnects (her ask 2026-10-12: after a network reconnect Dot must
    not forget the call). Memory only: never logged, never written to disk, gone when the service restarts."""

    def __init__(self):
        self.history = []          # {"role", "text"} plus "turn" / "reply" tags from Live
        self.handoff_count = 0
        self.target = "AUTO"        # R28: the call's channel (menu pick or spoken transfer); a reconnect keeps it
        self.target_set = False    # True once the server holds the truth (a spoken transfer or a menu pick)
        self.stack = []             # R29: channels she left, newest last ("previous" pops one; "back to Dot" clears it)
        self.came_from = "AUTO"     # R29: the channel she was on before the last move
        self.via_transfer = False   # R29: the current channel was reached by a spoken transfer (hand-offs carry context)
        self.suggest_asked = set()  # R29: channels Dot already offered a transfer to in this call (once per topic)
        self.pending_suggestion = None   # R29 normal mode: {"key", "text", "reply", "target"} waiting for her yes / no
        self.live_suggestion = None      # R29 Live: {"key", "words", "agent"} - the hand-off held back while Dot asks
        self.posted_handoffs = []
        self.pending_handoff = None   # Live: the hand-off still in its correction window
        self.notices = []             # lines Dot must say when she is quiet (a held hand-off that failed)
        self.channel_rows = []        # R27: one row per hand-off of this call, shown on the page (max CHANNEL_ROWS_MAX)
        self.tag_base = 0             # each Live session numbers its lines from here (no clash across sessions)
        self.ask_tasks = set()     # R30: second-opinion calls still running
        self.tokens = set()        # every reconnect token issued in this call
        self.call = None           # the Call using it now
        self.recap = None          # R38/R39: callrecap.CallRecap of the Live call in progress
        self.last_used = time.time()

CHANNEL_ROWS_MAX = 10
STACK_MAX = 10
SUGGEST_TRANSFERS = True   # R29 switch (tests that predate it turn it off)
# carry.py aliases: a session that answers under the other name still answers that hand-off (R27)
ROLE_ALIASES = {"CRM": "VAULT", "VAULT": "CRM", "VIDEO": "MW", "MW": "VIDEO"}


def _short(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def channel_row_add(st, words, agent, status):
    """R27: add a row to the call's Channels list (newest last, at most CHANNEL_ROWS_MAX)."""
    key = agent if agent else "AUTO"
    row = {"id": secrets.token_hex(4), "key": key, "label": dict(TARGETS).get(key, key),
           "snippet": _short(words, 40), "status": status, "reply": "", "sent_at": 0.0}
    st.channel_rows.append(row)
    del st.channel_rows[:-CHANNEL_ROWS_MAX]
    return row


def channel_rows_public(st):
    return [{k: r[k] for k in ("id", "label", "snippet", "status", "reply")} for r in st.channel_rows]


async def channels_push(st):
    """R27: send the list to the phone's current line (after a reconnect that is state.call). Never raises."""
    now = st.call
    if now is None:
        return
    try:
        await now.send({"type": "channels", "rows": channel_rows_public(st)})
    except Exception:
        pass


def channel_replies(st, items):
    """R27: a session update (updates.pending items) posted after a hand-off was sent marks its row 'replied'.
    One update answers one row (the oldest sent one for that role or its alias). Returns True if a row changed."""
    changed = False
    for path, rec in items:
        if any(r.get("upd") == path for r in st.channel_rows):
            continue
        role = str(rec.get("from", "")).upper()
        keys = {role, ROLE_ALIASES.get(role, role)}
        created = float(rec.get("created", 0))
        for r in st.channel_rows:
            if r["status"] == "sent" and r["key"] in keys and created >= r["sent_at"]:
                r["status"], r["reply"], r["upd"] = "replied", _short(rec.get("text", ""), 60), path
                changed = True
                break
    return changed


PIN_FILE = os.path.join(HERE, "pin.txt")
TURN_LOG = os.path.join(HERE, "logs", "turns.log")
CALL_LOG_DIR = os.path.join(HERE, "logs", "calls")
INDEX = os.path.join(HERE, "static", "index.html")


def _cap_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return {}
    snap = dict(snapshot)
    if "lines" in snap and isinstance(snap["lines"], list):
        capped_lines = []
        for item in snap["lines"][-12:]:
            if isinstance(item, dict):
                d = dict(item)
                if "text" in d and d["text"] is not None:
                    d["text"] = str(d["text"])[:400]
                capped_lines.append(d)
            else:
                capped_lines.append(item)
        snap["lines"] = capped_lines
    if "events" in snap and isinstance(snap["events"], list):
        snap["events"] = list(snap["events"][-30:])
    raw = json.dumps(snap, ensure_ascii=False)
    if len(raw) > 20000:
        while len(raw) > 20000 and snap.get("events"):
            snap["events"] = snap["events"][1:]
            raw = json.dumps(snap, ensure_ascii=False)
        while len(raw) > 20000 and snap.get("lines"):
            snap["lines"] = snap["lines"][1:]
            raw = json.dumps(snap, ensure_ascii=False)
        if len(raw) > 20000:
            for k, v in list(snap.items()):
                if isinstance(v, str) and len(v) > 200:
                    snap[k] = v[:200]
            raw = json.dumps(snap, ensure_ascii=False)
        if len(raw) > 20000:
            minimal = {k: snap[k] for k in ("version", "mode", "state") if k in snap}
            snap = minimal
    return snap


def _last_turn_lines(path, n=40):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\r\n") for ln in lines[-n:]]
    except OSError:
        return []


def _noise_clip(seconds=1.0):
    import array
    import random
    rnd = random.Random(7)
    return array.array("h", (rnd.randint(-3000, 3000) for _ in range(int(16000 * seconds)))).tobytes()


KEEPALIVE_CLIP = _noise_clip()


def prefetch_parts(tts, text):
    """Run tts.speak_stream(text) in a background thread; returns a queue of WAV parts ending with None."""
    q = queue.Queue()

    def work():
        try:
            for wav in tts.speak_stream(text):
                q.put(wav)
        except Exception as e:
            log_line("tts failed: %s" % type(e).__name__)
        finally:
            q.put(None)
    threading.Thread(target=work, daemon=True).start()
    return q

class Services:
    """Everything a call needs; tests pass fakes."""

    def __init__(self, stt, frontdesk, tts, handoff, guard, allowed_login="", log_calls=False, turn_log=TURN_LOG, passkeys=None):
        self.stt = self.speech = stt
        self.frontdesk, self.tts, self.handoff = frontdesk, tts, handoff
        self.guard, self.allowed_login, self.log_calls, self.turn_log = guard, allowed_login, log_calls, turn_log
        self.passkeys = passkeys          # Face ID sign-in (passkey.py); None = PIN only
        self.updates_folder = updates.FOLDER
        self.live_connect = None          # tests/harness: replaces the Qwen realtime websocket connect
        self.resume = {}                  # reconnect tokens: token -> expiry. Memory only, gone when the service restarts.
        self.call_states = []             # CallState per recent call (memory only)

    def call_state(self, token):
        """The memory of the call this reconnect token belongs to (any token issued in it), or None."""
        now = time.time()
        self.call_states = [st for st in self.call_states
                            if st.call is not None or now - st.last_used < CALL_MEMORY_S][-RESUME_MAX:]
        for st in self.call_states:
            if any(hmac.compare_digest(t.encode(), token.encode()) for t in st.tokens):
                return st
        return None

    def issue_resume(self):
        """After a good sign-in the page gets a token for reconnects in the same call, so a dropped line does not ask
        for Face ID again. Kept only in the page's memory, never stored."""
        now = time.time()
        self.resume = {t: e for t, e in self.resume.items() if e > now}
        while len(self.resume) >= RESUME_MAX:
            self.resume.pop(min(self.resume, key=self.resume.get))
        token = secrets.token_urlsafe(24)
        self.resume[token] = now + RESUME_TTL_S
        return token

    def resume_ok(self, token):
        if not isinstance(token, str) or not token:
            return False
        exp = next((e for t, e in self.resume.items() if hmac.compare_digest(t.encode(), token.encode())), 0)
        return exp > time.time()


def log_turn(path, **kw):
    """Timing line only. Never writes what was said."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join("%s=%s" % kv for kv in kw.items()) + "\n")
    except OSError:
        pass


class LiveTools:
    def __init__(self, call):
        self.call = call
        self._summary_asked = ""

    def _say_sent(self, agent, words=""):
        """R36: the exact line Dot says after a hand-off. Plain "Sent." when she is already in that channel (never "It went to
        the Vault" in the Vault channel, never a role code), else "Sent to <label>."."""
        zh = self._zh(words)
        key = agent if agent and agent != "AUTO" else None
        ch = getattr(self.call, "target", "AUTO")
        if not key or key == ch or ROLE_ALIASES.get(key) == ch:
            return "\u5df2\u53d1\u9001\u3002" if zh else "Sent."
        name = transfer.spoken_label(key, TARGETS)
        return ("\u5df2\u53d1\u7ed9 %s\u3002" % name) if zh else "Sent to %s." % name

    def reminder_agent(self, words, agent):
        """R35: reminders and alarms are never a property session's job: they go to the personal / organizer role."""
        low = " ".join(str(words or "").lower().split())
        if not any(w in low for w in config.get("reminder_words")):
            return agent
        role = handoff_mod.reminder_role(TARGETS)
        return role if role and agent != role else agent

    async def hand_off(self, words, agent, replaces_waiting=False, summary=None):
        """Live: held HANDOFF_HOLD_S so she can correct it (her gap G4, 2026-10-12). A hand_off marked as her
        correction replaces the waiting one (her correction is added to the words, the new session wins); any other
        hand_off sends the waiting one first; cancel_hand_off drops it. Whatever still waits is sent at once when
        the call or Live ends."""
        if getattr(self.call, "bug", None):
            return "not sent: this is a bug report, say only: noted"
        st = self.call.state
        moved = await self.guard_transfer(words)   # R29: a transfer phrase is never posted as a task
        if moved:
            return moved
        if not replaces_waiting and propfacts.is_property_question(words):   # R35: a property fact is answered here, in seconds
            facts = await propfacts.lookup(words)
            if facts:
                return ("answered from the Vault lookup, nothing was handed off. Say this to the caller in one or two short "
                        "sentences: %s" % facts)
        agent = self.reminder_agent(words, agent)
        if transfer.has_cjk(words) and not summary and self._summary_asked != words:   # R34: carry.py cannot take Chinese
            self._summary_asked = words
            return ("not sent yet: call hand_off again with summary = one short English sentence saying what she asked "
                    "(same agent), then say the line its result gives")
        sug = await self.suggestion_step(words, agent, replaces_waiting)
        if isinstance(sug, str):
            return sug
        words, agent = sug
        p = st.pending_handoff
        if p is not None and not replaces_waiting:
            await self.post_pending(p)   # a new request, not a correction: both go out
            p = None
        if p is not None:
            p["task"].cancel()
            if words and words not in p["words"]:
                p["words"] = (p["words"] + " " + words).strip()
            p["agent"] = (agent if agent and agent != "AUTO" else None) or p["agent"]
            p["task"] = asyncio.create_task(self._post_later(p))
            row = p.get("row")
            if row:
                row.update(key=p["agent"] or "AUTO", label=dict(TARGETS).get(p["agent"] or "AUTO", p["agent"] or "AUTO"),
                           snippet=_short(p["words"], 40))
                await channels_push(st)
            return "updated: the waiting hand-off now goes to %s. Say exactly this and nothing else: \"%s\"" % (
                p["agent"] or "the right session", self._say_sent(p["agent"], words))
        if getattr(self.call, "handoff_count", 0) >= 6:
            return "limit reached, tell the caller to use the menu"
        if not agent or agent == "AUTO":   # R28: on a channel, work goes there unless the caller names another
            ch = getattr(self.call, "target", "AUTO")
            agent = ch if ch not in ("AUTO", "VOICE") else agent
        p = {"words": words, "agent": agent, "summary": summary}
        p["row"] = channel_row_add(st, words, agent, "waiting")
        p["task"] = asyncio.create_task(self._post_later(p))
        st.pending_handoff = p
        await channels_push(st)
        return "queued for %s, sent in a few seconds unless the caller corrects it. Say exactly this and nothing else: \"%s\"" % (
            agent or "the right session", self._say_sent(agent, words))

    def _zh(self, words=""):
        return transfer.has_cjk(words) if words else getattr(self.call, "live_lang", "") == "zh-CN"

    async def transfer(self, channel, words=""):
        """R28: transfer_call tool. The tool result tells the model the exact line to say (and, R29, who it is now)."""
        zh = self._zh(words)
        if transfer.is_prev_name(transfer.resolve_words_polite(channel)):
            res = ("prev",)
        else:
            aliases, titles = self.call.transfer_names()
            keys = transfer.resolve_words_all(channel, TARGETS, aliases, titles)
            res = ("to", keys[0]) if len(keys) == 1 else (("ask", keys[:3]) if keys else ("unknown", channel))
        return await self._run(res, zh)

    async def _run(self, res, zh):
        call = self.call
        ls, call.state.live_suggestion = call.state.live_suggestion, None

        sent = []

        async def forward(key):
            if not ls or ls["key"] != key:
                return False
            sent.append(key)
            return await self.send_kept(ls, key)

        kind, say, key = await call.run_plan(res, zh, forward)
        if ls and not sent:   # she went elsewhere (or nowhere): her kept request still goes out, to where it belonged
            await self.send_kept(ls)
        if kind != "moved":
            return "%s, target unchanged. Say exactly this and nothing else: \"%s\"" % (
                "no previous channel" if res[0] == "prev" else ("channel name unclear" if res[0] == "ask" else "unknown channel"), say)
        if key == "AUTO":
            say = say if say != "Dot here." else "Dot here. Go ahead."   # R36: a two-word line was sometimes not spoken at all
            note = "The caller is back with you (Dot): speak the line aloud, then handle her requests as before."
        else:
            name = transfer.spoken_label(key, TARGETS)
            note = ("You are now the %s line: the caller is on the %s channel; real work goes to %s (hand_off with no agent "
                    "goes there). Her sessions' replies are spoken when ready." % (name, name, key))
        return "transferred. Say exactly this and nothing else: \"%s\" %s" % (say, note)

    async def send_kept(self, ls, agent=None):
        """R29: post the request held back by a suggestion (her exact words, R15). Honours the per-call cap; a failure
        marks the Channels row failed and queues a spoken notice. True when posted."""
        call, st = self.call, self.call.state
        agent = agent or ls.get("agent") or ls["key"]
        row = channel_row_add(st, ls["words"], agent, "waiting")
        if getattr(call, "handoff_count", 0) >= 6:
            out = "not sent: limit reached"
        else:
            out = await self._post(ls["words"], agent, row)
        if out.startswith("posted"):
            return True
        row["status"] = "failed"
        await channels_push(st)
        st.notices.append(transfer.sorry_say(transfer.has_cjk(ls["words"])))
        return False

    async def guard_transfer(self, words):
        """R29: hand_off was called with words that are a transfer request (the model picked the wrong tool): do the
        transfer, post nothing. None when the words are not a transfer."""
        aliases, titles = self.call.transfer_names()
        res = transfer.plan(words, TARGETS, aliases, titles)
        if res is None or res[0] == "unknown":   # "connect with Alex tomorrow about the offer" is still a task
            return None
        log_line("live: hand_off with a transfer phrase became a transfer", self.call.svc.turn_log)
        return await self._run(res, transfer.has_cjk(words))

    async def suggestion_step(self, words, agent, replaces_waiting):
        """R29: on the front desk, work that belongs to one channel is offered a transfer first (once per channel).
        Returns a result string (not sent yet: ask), or (words, agent) to carry on - the kept request when she
        answered no or anything else."""
        call, st = self.call, self.call.state
        ls = st.live_suggestion
        if replaces_waiting and not ls:
            return words, agent
        if ls:
            if transfer.is_yes(words):   # she said yes but the model called hand_off: still a transfer (_run posts the kept request)
                return await self._run(("to", ls["key"]), self._zh(words))
            st.live_suggestion = None
            norm = lambda w: " ".join(str(w).lower().split()).strip(" .,!?")
            if transfer.is_no(words) or norm(words) == norm(ls["words"]):
                return ls["words"], agent or ls["agent"]   # her request as she said it (R15), sent as before
            await self.send_kept(ls)   # anything else is a new request: the kept one still goes out first
            if replaces_waiting:
                return words, agent
        key = call.suggest_key(words, agent)
        if not key:
            return words, agent
        st.suggest_asked.add(key)
        st.live_suggestion = {"key": key, "words": words, "agent": agent}
        q = transfer.suggest_say(key, TARGETS, self._zh(words))
        return ("not sent yet. Ask the caller exactly this and nothing else: \"%s\" If she says yes, call transfer_call with "
                "channel %s. If she says no or anything else, call hand_off again (her request is kept)." % (q, key))

    async def cancel_hand_off(self):
        st = self.call.state
        p, st.pending_handoff = st.pending_handoff, None
        if p is None:
            last = st.channel_rows[-1] if st.channel_rows else None
            if last is not None and last["status"] in ("sent", "replied"):
                return ("nothing to cancel: the last hand-off was already sent to %s. Say exactly this and nothing else: "
                        "\"It was already sent.\"" % last["label"])
            return "nothing to cancel: nothing was sent. Say exactly this and nothing else: \"Nothing was sent.\""
        p["task"].cancel()
        if p.get("row"):
            p["row"]["status"] = "cancelled"
            await channels_push(self.call.state)
        log_line("live: hand-off cancelled before sending", self.call.svc.turn_log)
        return "cancelled, nothing was sent"

    async def _post_later(self, p):
        await asyncio.sleep(HANDOFF_HOLD_S)
        await self.post_pending(p)

    async def post_pending(self, p=None):
        """Send the waiting hand-off now (its window ended, or the call / Live is ending)."""
        st = self.call.state
        p = p or st.pending_handoff
        if p is None or st.pending_handoff is not p:
            return
        st.pending_handoff = None
        out = await self._post(p["words"], p["agent"], p.get("row"), p.get("summary"))
        if p.get("row") and not out.startswith("posted"):
            p["row"]["status"] = "failed"
            await channels_push(st)
        if not out.startswith("posted"):
            log_line("live: held hand-off failed", self.call.svc.turn_log)
            st.notices.append("Sorry, that hand-off did not go through. Please say it again.")   # said when she is quiet

    async def _post(self, words, agent, row=None, summary=None):
        try:
            ctx = self.call.warm_context(words)
            extra = {"context": ctx} if ctx else {}
            if summary:
                extra["summary"] = summary
            rec = await asyncio.to_thread(self.call.svc.handoff.create, words, words, role=agent, **extra)
        except Exception as e:
            return "not sent: %s" % (str(e) or type(e).__name__)[:120]
        if rec and rec.get("posted"):
            self.call.handoff_count = getattr(self.call, "handoff_count", 0) + 1
            self.call.posted_handoffs.append({"to": rec.get("to", ""), "words": words})
            if row is not None:
                to = rec.get("to") or row["key"]
                row.update(status="sent", sent_at=time.time(), key=to, label=dict(TARGETS).get(to, to))
                await channels_push(self.call.state)
            now = getattr(self.call.state, "call", None) or self.call   # after a reconnect: the phone's current line
            try:
                await now.send({"type": "handoff", "id": rec.get("id", ""), "text": words, "to": rec.get("to", "")})
            except Exception:
                pass
            return "posted to %s" % rec.get("to", agent or "")
        reason = (rec.get("post_error") if rec else "") or "failed to post"
        return "not sent: %s" % reason

    FOLLOW_UP = re.compile(r"\b(it|that|this|found|find|get|got|done|finished|reply|replied|answer|result|heard|back|yet|update)\b|"
                           "\u627e\u5230|\u56de\u590d|\u5b8c\u4e86|\u7ed3\u679c", re.I)

    def handoff_record(self, question, session=None):
        """R37: "did you find it?" is about a hand-off of THIS call. Answers from the call's Channels rows (waiting, sent and
        how long ago, replied with the reply, cancelled, failed). None when this call made no hand-off to ask about: then the
        session's own saved note answers as before."""
        rows = self.call.state.channel_rows
        if not rows:
            return None
        row = None
        if session and session != "AUTO":
            keys = {session, ROLE_ALIASES.get(session, session)}
            mine = [r for r in rows if r["key"] in keys]
            row = mine[-1] if mine else None
        elif self.FOLLOW_UP.search(str(question or "")):
            row = rows[-1]
        if row is None:
            return None
        who = transfer.spoken_label(row["key"], TARGETS)
        zh = transfer.has_cjk(row["snippet"])
        st = row["status"]
        if st == "replied":
            return "%s replied: %s" % (who, row["reply"] or "it answered")
        if st == "sent":
            mins = int(max(0, time.time() - (row.get("sent_at") or time.time())) // 60)
            ago = "just now" if mins < 1 else "%d minute%s ago" % (mins, "" if mins == 1 else "s")
            return "Sent to %s %s, no reply yet." % (who, ago)
        if st == "waiting":
            return "Your request to %s is about to be sent." % who
        if st == "cancelled":
            return "That request was cancelled, nothing was sent."
        return "That request to %s did not go through." % who

    @staticmethod
    def brief(text, sentences=2, limit=240):
        """R37: a status answer is spoken in at most two short sentences, never read out in full."""
        t = " ".join(str(text or "").split())
        parts = re.split(r"(?<=[.!?\u3002])\s+", t)
        out = " ".join(parts[:sentences])
        return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "."

    async def status(self, question, session=None):
        """The model decides what is asked (no phrase rules): a named session gets its own status, else all sessions."""
        import status as status_mod
        rec = self.handoff_record(question, session)
        if rec:
            return rec
        res = None
        try:
            role = session if session in TARGET_KEYS and session != "AUTO" else None
            if not role:   # the model left session out but a name was said ("how is the CRM going")
                role = status_mod.named_session(question)
            if role:
                res = await asyncio.to_thread(status_mod.session_status, role)
            if not res:
                res = await asyncio.to_thread(status_mod.overview, status_mod.CARRYOVER, time.time(),
                                              status_mod.session_titles())
        except Exception:
            res = None
        if res:
            return self.brief(res)
        return "No saved status for that."

    async def lookup(self, words):
        if self.call.state.recap:
            self.call.state.recap.note_lookup(words)
        if propfacts.is_property_question(words):   # R35: property facts first, from the Vault lookup, in seconds
            facts = await propfacts.lookup(words)
            if facts:
                return facts
            miss = (" No Vault record was found by the quick lookup: say so in one short sentence and call hand_off so the "
                    "Vault session looks into it.")
        else:
            miss = ""
        try:
            import frontdesk
            notes = await asyncio.to_thread(frontdesk.gather_notes, words, frontdesk.brain_notes)
        except Exception:
            return "Nothing found."
        notes = (notes or "").strip()
        return (notes if notes else "Nothing found.") + miss

    async def ask_model(self, model, question):
        """R30: a second opinion from Opus / Sonnet / Haiku. Returns at once; the answer joins state.notices, which the
        updates loop speaks when Dot is quiet (same path as a failed hand-off notice). Question text is never logged."""
        key, err = askmodel.pick_model(model)
        if err:
            return err
        if not (question or "").strip():
            return "No question heard, ask the caller to say it again."
        st = self.call.state
        task = asyncio.create_task(self._ask_model_bg(key, question))
        st.ask_tasks.add(task)
        task.add_done_callback(st.ask_tasks.discard)
        return "Asking %s now, the answer will be spoken in a few seconds." % key.capitalize()

    SEND_MAX_CHARS = 1000
    SEND_MAX_PER_CALL = 10

    @staticmethod
    def build_phone_message(kind, text):
        """R44: the text of the Telegram message, or None for a bad kind / empty text."""
        text = " ".join(str(text or "").split())
        if not text or kind not in ("directions", "note"):
            return None
        if kind == "directions":
            return "Directions to %s\nhttps://www.google.com/maps/dir/?api=1&destination=%s" % (text, urllib.parse.quote(text, safe=""))
        return text

    async def send_to_phone(self, kind, text):
        """R44 (P26): one Telegram message to her own chat. Never raises; the result text is what Dot says. The message text
        is never logged, only kind and ok."""
        st = self.call.state
        kind = str(kind or "").strip().lower()
        ok = False
        try:
            if len(str(text or "")) > self.SEND_MAX_CHARS:
                out = "Too long to send (max %d characters), ask the caller for a shorter version." % self.SEND_MAX_CHARS
            elif getattr(st, "phone_sends", 0) >= self.SEND_MAX_PER_CALL:
                out = "Limit reached: %d messages already sent on this call." % self.SEND_MAX_PER_CALL
            else:
                msg = self.build_phone_message(kind, text)
                if msg is None:
                    out = "Nothing to send: need kind directions or note and some text."
                else:
                    ok = bool(await asyncio.to_thread(callrecap.send_telegram, msg, config.get("call_recap_telegram_chat_id")))
                    if ok:
                        st.phone_sends = getattr(st, "phone_sends", 0) + 1
                    out = "Sent to your Telegram." if ok else "Could not send to Telegram, tell the caller it failed."
        except Exception:
            out = "Could not send to Telegram, tell the caller it failed."
        log_line("live_tool=send_to_phone kind=%s ok=%s" % (kind or "-", "yes" if ok else "no"), self.call.svc.turn_log)
        return out

    async def next_events(self, hours=12):
        """R45 (P27): her next calendar events, read-only, from config calendar_list_cmd. Never raises; titles are never logged."""
        t0 = time.monotonic()
        n = -1
        try:
            out, n = await nextevents.next_events(hours if hours else 12)
        except Exception:
            out = "I could not read your calendar just now."
        log_line("live_tool=next_events n=%d ms=%d" % (n, int((time.monotonic() - t0) * 1000)), self.call.svc.turn_log)
        return out

    async def last_call(self, n=1):
        """R46 (P28): the newest call recap(s), read-only. Never raises; recap content is never logged."""
        t0 = time.monotonic()
        cnt = -1
        try:
            rc = getattr(self.call.state, "recap", None)
            skip = getattr(rc, "started", None)
            log_dir = os.path.dirname(self.call.svc.turn_log)
            out, cnt = await asyncio.to_thread(lastcall.last_recaps, log_dir, n if n else 1, skip)
        except Exception:
            out = "I could not read the last call recap just now."
        log_line("live_tool=last_call n=%d ms=%d" % (cnt, int((time.monotonic() - t0) * 1000)), self.call.svc.turn_log)
        return out

    async def unread_digest(self, since_hours=12):
        """R48 (P29): digest of new client messages, read-only. Never raises; names and text are never logged."""
        t0 = time.monotonic()
        n = -1
        try:
            out, n = await asyncio.to_thread(unreadsum.unread_digest, since_hours if since_hours else 12)
        except Exception:
            out = unreadsum.NONE_TEXT
        log_line("live_tool=unread_digest n=%d ms=%d" % (n, int((time.monotonic() - t0) * 1000)), self.call.svc.turn_log)
        return out

    async def _ask_model_bg(self, key, question):
        st = self.call.state
        status, text, ms = await askmodel.ask(key, question, list(st.history))
        log_line("ask_model: %s ms=%d %s" % (key, ms, status), self.call.svc.turn_log)
        if status == "ok":
            st.notices.append("%s says: %s" % (key.capitalize(), text))
        else:
            st.notices.append("Sorry, %s did not answer%s." % (key.capitalize(), " in time" if status == "timeout" else ""))


class Call:
    def __init__(self, ws, svc):
        self.ws, self.svc = ws, svc
        self.state = CallState()
        self.history = self.state.history   # the same list: kept across reconnects
        self.replaced = False
        self.phone_voice = False
        self.ears = "mac"
        self.task = None
        self.live = None
        self.live_pending = False
        self.live_lang = "en-US"
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.bug = None
        self.bug_timer = None
        self.client = ""
        self.svc.last_call = self   # tests reach the live Call through this

    @property
    def target(self):
        return self.state.target

    @target.setter
    def target(self, key):
        self.state.target = key

    @property
    def handoff_count(self):
        return self.state.handoff_count

    @handoff_count.setter
    def handoff_count(self, n):
        self.state.handoff_count = n

    @property
    def posted_handoffs(self):
        return self.state.posted_handoffs

    @posted_handoffs.setter
    def posted_handoffs(self, items):
        self.state.posted_handoffs = items

    def _use_state(self, st):
        self.state = st
        self.history = st.history

    async def take_over(self):
        """The phone came back on a new connection: this older Call must stop talking and stop writing to the
        shared memory at once (its dead socket can take many seconds to close)."""
        self.replaced = True   # its bridge's late items and failures are ignored from now on
        for t in (getattr(self, "keep", None), getattr(self, "upd", None), getattr(self, "bug_timer", None)):
            if t:
                t.cancel()
        live, self.live = self.live, None
        if live:
            try:
                await live.close()
            except Exception:
                pass
        await self.cancel_turn()
        # a waiting hand-off keeps its timer; it reports on the new line (state.call)
        self._closing = asyncio.ensure_future(self._close_quietly())   # a half-open socket can take 10 s: never wait

    async def _close_quietly(self):
        try:
            await self.ws.close()
        except Exception:
            pass

    def remember(self, role, text, **tags):
        """One line of the call into its memory. In Live a line carries its turn / reply number, so a second
        version of the same line (Whisper words, a cut reply) replaces it instead of adding one."""
        text = (text or "").strip()
        if not text:
            return
        key = "turn" if role == "user" else "reply"
        tag = tags.get(key)
        if tag is not None:
            for line in reversed(self.history[-6:]):
                if line.get("role") == role and line.get(key) == tag:
                    line["text"] = text
                    return
        line = {"role": role, "text": text}
        line.update({k: v for k, v in tags.items() if v is not None})
        if role == "user" and self.state.recap:
            self.state.recap.note_turn()
        self.history.append(line)
        del self.history[:-HISTORY_MAX]
        self.state.last_used = time.time()

    def recap_start(self):
        """R38/R39: a Live call begins (flags off = nothing tracked)."""
        if (config.get("call_recap") or config.get("call_summary_line")) and not self.state.recap:
            self.state.recap = callrecap.CallRecap(bridge_call=self.client == "twilio")

    async def recap_finish(self):
        """R38/R39: the Live call ended: summary line, recap file, optional Telegram copy. Never raises into the call."""
        rc, self.state.recap = self.state.recap, None
        if rc is None:
            return
        try:
            log_dir = os.path.dirname(self.svc.turn_log)
            if config.get("call_summary_line"):
                log_line(rc.summary_line(), self.svc.turn_log)
            if config.get("call_recap"):
                text = rc.build(list(self.history), list(self.state.channel_rows), lambda k: dict(TARGETS).get(k, k))
                path = await asyncio.to_thread(callrecap.write_recap, log_dir, text)
                log_line("call_recap: written %s" % os.path.basename(path), self.svc.turn_log)
                if config.get("call_recap_telegram"):
                    ok = await asyncio.to_thread(callrecap.send_telegram, text, config.get("call_recap_telegram_chat_id"))
                    log_line("call_recap: telegram %s" % ("sent" if ok else "not sent"), self.svc.turn_log)
        except Exception as e:
            log_line("call_recap: failed (%s)" % (str(e) or type(e).__name__)[:80], self.svc.turn_log)

    def seed_lines(self):
        """The last lines of this call for a new Live session (reconnect, Live off-on, provider switch)."""
        out, n = [], 0
        for line in reversed(self.history[-SEED_LINES:]):
            n += len(line["text"])
            if n > SEED_CHARS:
                break
            out.append({"role": line["role"], "text": line["text"]})
        return out[::-1]

    def _on_live_closed(self, bridge=None):
        if bridge is None or self.live is bridge:   # a replaced bridge closing late never ends its successor
            self.live = None

    async def _start_live(self, classes):
        """Start the first Live provider that comes up (6 s each); raise the last failure."""
        transcribe_fn = getattr(getattr(self.svc, "speech", None) or getattr(self.svc, "stt", None), "transcribe", None)
        extra = {"connect": self.svc.live_connect} if getattr(self.svc, "live_connect", None) else {}
        for i, cls in enumerate(classes):
            holder = []
            bridge = cls(
                self._live_out,
                LiveTools(self),
                realtime.build_instructions(config.get("owner"), self.target, TARGETS),
                transcribe=transcribe_fn,
                on_closed=lambda h=holder: self._on_live_closed(h[0] if h else None),
                turn_log=self.svc.turn_log,
                **extra,
            )
            holder.append(bridge)
            bridge.seed = self.seed_lines()   # what this call said so far: a new session must not forget it
            self.state.tag_base += 100000     # its turn / reply numbers never match an older session's lines
            bridge._turn_no = bridge._reply_no = self.state.tag_base
            bridge.voice_channel = getattr(self, "source", "web")   # R42 (log-only voiceprint)
            bridge.signed_in = True   # a Live call exists only after PIN / passkey / phone sign-in
            bridge.language = self.live_lang   # the page's ZH/EN choice: a hint for the provider's ears
            try:
                await asyncio.wait_for(bridge.start(), 6)   # never block the call on a silent provider
                if bridge.closed:   # died right after connecting (its live_error already went out)
                    raise RuntimeError("closed at start")
                return bridge
            except BaseException as e:
                try:
                    await bridge.close()
                except Exception:
                    pass
                if i == len(classes) - 1 or not isinstance(e, Exception):
                    raise
                log_line("live: %s failed (%s), next provider" % (cls.__name__, (str(e) or type(e).__name__)[:80]), self.svc.turn_log)

    async def _live_to_qwen(self, failed):
        """The first-choice Live provider died mid-call (quota, network): carry on in Qwen Live instead of dropping
        her to classic mode."""
        if self.replaced or not getattr(failed, "falls_back_to_qwen", False) or self.live_pending:
            return False
        self.live_pending = True
        try:
            self.live = await self._start_live([realtime.LiveBridge])
        except Exception as e:
            log_line("live: Qwen fallback failed (%s)" % (str(e) or type(e).__name__)[:80], self.svc.turn_log)
            self.live = None
            return False
        finally:
            self.live_pending = False
        log_line("live: switched to Qwen mid-call", self.svc.turn_log)
        await self.send({"type": "live_ok"})
        try:   # her last words went to the dead provider
            await self.live.say_exact("Sorry, I switched lines. Please say that again.")
        except Exception:
            pass
        return True

    async def _live_out(self, item):
        if self.replaced:   # the phone is on a newer connection: this old session's words are not the call's
            return
        if isinstance(item, bytes):
            stopped_at = getattr(self.live, "stopped_at", None) if self.live else None
            if stopped_at is not None:
                ms = int((time.monotonic() - stopped_at) * 1000)
                log_line("live first_audio_ms=%d" % ms, self.svc.turn_log)
                if self.state.recap:
                    self.state.recap.note_first_audio(ms)
                if self.live:
                    self.live.stopped_at = None
            b64 = base64.b64encode(item).decode("ascii")
            await self.send({"type": "live_audio", "b64": b64})
        elif isinstance(item, dict):
            if item.get("type") == "live_error":
                log_line("live: error %s" % str(item.get("text", ""))[:120], self.svc.turn_log)
                failed, self.live = self.live, None
                if await self._live_to_qwen(failed):
                    return
            elif item.get("type") == "live_heard":
                words = str(item.get("text", "")).strip()
                if self.bug and words:
                    if self.live:
                        await self.live.cancel()
                    await self._record_bug_words(words)
                    await self.send(item)
                    return
                self.remember("user", words, turn=item.get("turn"))
            elif item.get("type") == "live_said":
                self.remember("assistant", str(item.get("text", "")), reply=item.get("reply"))
            elif item.get("type") == "live_cut":   # she interrupted: she heard only this much of the reply
                self.remember("assistant", str(item.get("text", "")), reply=item.get("reply"))
                return   # memory only, the phone shows nothing
            await self.send({k: v for k, v in item.items() if k not in ("turn", "reply")})

    async def send(self, obj):
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def authenticate(self) -> bool:
        headers = self.ws.request.headers
        if not origin_ok(headers) or not login_ok(headers, self.svc.allowed_login):
            await self.send({"type": "auth_fail", "locked_s": 0})
            return False
        try:
            raw = await asyncio.wait_for(self.ws.recv(), AUTH_TIMEOUT_S)
        except (asyncio.TimeoutError, Exception):
            return False
        try:
            msg = json.loads(raw) if isinstance(raw, str) else {}
        except ValueError:
            msg = {}
        # Order: reconnect token, then the PIN, then Face ID. A stale token (service restarted) is not a failed try:
        # the page also sends the PIN it still holds, so a PIN call reconnects after a restart.
        ok, locked, retry = False, 0, False
        pin = str(msg.get("pin", "")) if msg.get("type") == "auth" else ""
        if msg.get("type") != "auth":
            _, locked = self.svc.guard.record_failure()
        elif msg.get("resume") and self.svc.resume_ok(msg.get("resume")):
            ok = True
            st = self.svc.call_state(msg["resume"])
            if st is not None:   # the same call, back after a drop: keep its memory
                old = st.call
                if old is not None and old is not self:
                    await old.take_over()
                self._use_state(st)
        elif pin:
            ok, locked = self.svc.guard.check(pin)
        elif isinstance(msg.get("passkey"), dict) and self.svc.passkeys:
            locked = self.svc.guard.locked_for()
            if not locked:
                res = await asyncio.to_thread(self.svc.passkeys.verify_login, msg["passkey"])
                ok, retry = res is True, res is None
                if ok:
                    self.svc.guard.reset()
                elif not retry:
                    _, locked = self.svc.guard.record_failure()
            log_line("call: Face ID sign-in %s" % ("ok" if ok else "stale, retry" if retry else "refused"), self.svc.turn_log)
        else:
            locked = self.svc.guard.locked_for()   # nothing to check (e.g. stale token only): refuse, do not count
        self.phone_voice = msg.get("voice") == "phone"
        self.client = str(msg.get("client") or "")[:20]
        self.source = "phone" if (msg.get("source") == "phone" or self.client == "twilio") else "web"   # R42: which voiceprint profile
        self.ears = "phone" if msg.get("ears") == "phone" else "mac"
        if not self.state.target_set:   # R28: after a spoken transfer the server's channel wins over the page's old pick
            self.target = msg.get("target") if msg.get("target") in TARGET_KEYS else "AUTO"
        if not ok:
            await self.send({"type": "auth_retry"} if retry else {"type": "auth_fail", "locked_s": locked,
                                                                  "face": bool(msg.get("passkey")) and not pin})
            return False
        log_line("call: connected, %s, phone_voice=%s, ears=%s, home_screen_app=%s" % (VERSION, self.phone_voice, self.ears,
                 bool(msg.get("standalone"))), self.svc.turn_log)
        self.signed_with_pin = bool(msg.get("pin"))
        token = self.svc.issue_resume()
        self.state.tokens.add(token)
        self.state.last_used = time.time()
        if self.state not in self.svc.call_states:
            self.svc.call_states.append(self.state)
        await self.send({"type": "auth_ok", "version": VERSION, "resume": token, "targets": TARGETS,
                         "passkeys": self.svc.passkeys.count() if self.svc.passkeys else -1})
        self.state.call = self
        if self.state.channel_rows:
            await channels_push(self.state)   # R27: the list survives a reconnect
        if self.state.target_set:
            await self.send_target()          # R28: the menu and the call agree after a reconnect
        self.keep = asyncio.create_task(self._keepalive())
        self.upd = asyncio.create_task(self._updates_loop())
        await self.send({"type": "status", "state": "listening"})
        return True

    # ---- R28: call-center transfers ----
    def transfer_names(self):
        """(aliases, titles) from roles_json at run time; both empty when the file is missing."""
        aliases, titles = [], {}
        try:
            import status as status_mod
            titles = status_mod.session_titles()
            aliases = json.load(open(status_mod.ROLES_JSON)).get("_aliases") or []
        except Exception:
            pass
        return aliases, titles

    def resolve_channel(self, name):
        aliases, titles = self.transfer_names()
        return transfer.resolve_words(name, TARGETS, aliases, titles)

    async def send_target(self, chime=False):
        key = self.target
        now = self.state.call or self
        msg = {"type": "target", "role": key, "label": dict(TARGETS).get(key, key)}
        if chime:
            msg["chime"] = True   # R29: the page plays a soft chime; a reconnect resync never does
        try:
            await now.send(msg)
        except Exception:
            pass

    def _move(self, key, how):
        """R29: bookkeeping for a channel change. how: "transfer" | "previous" | "menu"."""
        st = self.state
        old = st.target
        if key != old:
            if how == "previous":
                pass
            elif key == "AUTO":
                st.stack.clear()          # "back to Dot" empties the stack
            else:
                st.stack.append(old)
                del st.stack[:-STACK_MAX]
            st.came_from = old
        st.via_transfer = how in ("transfer", "previous") and key not in ("AUTO", "VOICE")
        st.target = key
        st.target_set = True

    def pop_previous(self):
        st = self.state
        while st.stack:
            key = st.stack.pop()
            if key in TARGET_KEYS and key != st.target:
                return key
        return None

    async def set_channel(self, key, how="transfer"):
        """Same effect as the menu: the call's channel is KEY (hand-offs go there); the page is told."""
        self._move(key, how)
        self.state.pending_suggestion = None
        log_line("call: transferred to %s" % key, self.svc.turn_log)
        await self.send_target(chime=how in ("transfer", "previous"))

    async def greet(self, key, zh, passed=False):
        """R29: "Vault here." + where replies are spoken (+ "I passed on your request" after a yes to a suggestion)."""
        return transfer.greeting(key, TARGETS, zh, passed)

    async def run_plan(self, res, zh, forward=None):
        """transfer.plan() result -> (kind, line to say, key). kind: "moved" | "unknown" (a prev with nothing to pop is "unknown")."""
        if res[0] == "unknown":
            return "unknown", transfer.unknown_say(TARGETS, zh), None
        if res[0] == "ask":   # R41: the name fits two roles: ask, move nowhere
            return "unknown", transfer.ask_say(res[1], TARGETS, zh), None
        if res[0] == "prev":
            key = self.pop_previous()
            if key is None:
                return "unknown", transfer.no_previous_say(zh), None
            how = "previous"
        else:
            key, how = res[1], "transfer"
        await self.set_channel(key, how)
        passed = bool(forward and await forward(key))   # a yes to a suggestion also posts her kept request there
        return "moved", await self.greet(key, zh, passed), key

    async def try_transfer(self, text, zh=None):
        """Normal mode: her words ask for a transfer -> the line to say; None when they do not."""
        aliases, titles = self.transfer_names()
        res = transfer.plan(text, TARGETS, aliases, titles)
        if res is None:
            return None
        zh = transfer.has_cjk(text) if zh is None else zh
        return (await self.run_plan(res, zh))[1]

    def suggest_key(self, words, agent=None, task=None):
        """R29: the channel to offer a transfer to, or None. Only on the front desk (AUTO), only for real work whose
        route() is one specific channel, once per channel per call, never for status questions (R14) or when she
        already named the channel ("tell the website ...")."""
        st = self.state
        if not SUGGEST_TRANSFERS or self.target != "AUTO" or getattr(self, "bug", None):
            return None
        role = agent if agent and agent != "AUTO" else handoff_route(task or words, words)
        key = role if role in TARGET_KEYS else ROLE_ALIASES.get(role)
        if key not in TARGET_KEYS or key in ("ORG", "VOICE", "AUTO") or key in st.suggest_asked:
            return None
        import status as status_mod
        if status_mod.wants_status(words) or transfer.explicit_addressee(words, key, TARGETS):
            return None
        return key

    def warm_context(self, words):
        """R29: one labelled line for a hand-off posted on a channel reached by a transfer: the last few lines of the call,
        <= 120 ASCII chars, no phone numbers or e-mails, never Chinese text. None when it does not apply."""
        st = self.state
        if not st.via_transfer or self.target in ("AUTO", "VOICE"):
            return None
        mine = " ".join(str(words or "").split())
        lines = [l for l in self.history if l["role"] == "user" and " ".join(l["text"].split()) != mine
                 and not transfer.is_suggestion_chatter(l)][-3:]   # R36: her lines only, never Dot's own filler replies
        if not lines:
            return None
        frm = "front desk" if st.came_from in ("AUTO", "VOICE") else transfer.spoken_label(st.came_from, TARGETS)
        if any(transfer.has_cjk(l["text"]) for l in lines):
            body = "(context in Chinese, see voice call)"
        else:
            parts = []
            for l in lines:
                t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", l["text"])
                t = re.sub(r"\+?\d[\d\-\s().]{6,}\d", "[number]", t)
                t = " ".join(t.encode("ascii", "ignore").decode().split())
                if t:
                    parts.append("Caller: " + _short(t, 40))
            body = _short(" / ".join(parts), 120)
        return ("Context (Dot, from %s): %s" % (frm, body)) if body else None

    async def turn(self, pcm, text=None):
        t0 = time.monotonic()
        await self.send({"type": "status", "state": "thinking"})
        if text is None:
            text = await asyncio.to_thread(self.svc.stt.transcribe, pcm)
            log_line("stt done: %d chars, %.1fs of audio" % (len(text), len(pcm) / 32000.0), self.svc.turn_log)
        else:
            log_line("text from the phone: %d chars" % len(text), self.svc.turn_log)
        t_stt = time.monotonic()
        if not text:
            await self.send({"type": "status", "state": "listening"})
            return
        if self.bug:
            await self.send({"type": "transcript", "role": "user", "text": text})
            await self._record_bug_words(text)
            return
        if self.history and self.history[-1]["role"] == "user" and text.startswith(self.history[-1]["text"]):
            self.history.pop()   # phone ears joined the interrupted first half with the rest: keep one line
        self.history.append({"role": "user", "text": text})
        await self.send({"type": "transcript", "role": "user", "text": text})
        pend, self.state.pending_suggestion = self.state.pending_suggestion, None
        if pend is not None:   # R29: her answer to "Sounds like Vault - want me to transfer you?"
            if transfer.is_yes(text):
                await self._say_line(text, await self._yes_transfer(pend, text))
                return
            now = time.monotonic()   # no, or anything else: her request goes as before
            await self._deliver(pend["text"], pend["reply"], pend["target"], now, now, now)
            if transfer.is_no(text):
                return
        moved = await self.try_transfer(text)   # R28: "transfer me to vault" is never a hand-off
        if moved is not None:
            await self._say_line(text, moved)
            return
        asked = text
        target = getattr(self, "target", "AUTO")
        if target not in ("AUTO", "VOICE"):   # the caller picked a session in the menu: the front desk knows who real work goes to
            asked = "(%s picked the %s session in the call menu; hand real work to it.) %s" % (config.get("owner"), target, text)
        direct = await asyncio.to_thread(direct_status, text)   # one-session status: its own words, no model retelling
        if direct:
            log_line("status read word for word", self.svc.turn_log)
            reply = Reply(direct)
        else:
            reply = await asyncio.to_thread(self.svc.frontdesk.reply, list(self.history[:-1]), asked)
        t_fd = time.monotonic()
        key = None if direct or not reply.handoff else self.suggest_key(text, task=reply.handoff)
        if key:   # R29: real work for one specific channel: offer the transfer once; the request waits for her answer
            self.state.suggest_asked.add(key)
            self.state.pending_suggestion = {"key": key, "text": text, "reply": reply, "target": target}
            await self._say_line(text, transfer.suggest_say(key, TARGETS, transfer.has_cjk(text)))
            return
        await self._deliver(text, reply, target, t0, t_stt, t_fd)

    async def release_suggestions(self):
        """R29: a question that was never answered must not outlive its mode: post the held request to where it belonged."""
        st = self.state
        pend, st.pending_suggestion = st.pending_suggestion, None
        if pend is not None:
            try:   # post only, never speak: Live may be starting and Dot never talks over it (R8)
                if pend["reply"].handoff:
                    await self._post_reply(pend["text"], pend["reply"], pend["target"])
            except Exception:
                pass
        ls = st.live_suggestion
        if ls is not None:
            st.live_suggestion = None
            try:
                await LiveTools(self).send_kept(ls)
            except Exception:
                pass

    async def _yes_transfer(self, pend, text):
        """She said yes: transfer, then post the kept request (her exact words, R15) to the new channel."""
        await self.set_channel(pend["key"])
        ok = False
        if pend["reply"].handoff:
            if self.handoff_count < 6:
                try:
                    ok = await self._post_reply(pend["text"], pend["reply"], pend["key"])
                except Exception:
                    ok = False
            if not ok:
                return transfer.sorry_say(transfer.has_cjk(text))
        return await self.greet(pend["key"], transfer.has_cjk(text), ok)

    async def _say_line(self, text, line):
        log_line("reply: " + line[:200], self.svc.turn_log)
        self.history.append({"role": "assistant", "text": line})
        await self.send({"type": "transcript", "role": "assistant", "text": line})
        await self.speak(line)
        if self.svc.log_calls:
            self._append_call_log(text, line)

    async def _post_reply(self, text, reply, target):
        """Post a front-desk hand-off (her exact words) and update the Channels list; returns True when it was posted."""
        ctx = self.warm_context(text)
        extra = {"context": ctx} if ctx else {}
        rec = await asyncio.to_thread(self.svc.handoff.create, reply.handoff, text, target, **extra)
        ok = bool(rec and rec.get("posted"))
        if ok:
            self.posted_handoffs.append({"to": rec.get("to", target), "words": text})
        row = channel_row_add(self.state, text, (rec or {}).get("to") or target, "sent" if ok else "failed")
        row["sent_at"] = time.time()
        await self.send({"type": "handoff", "id": rec.get("id", ""), "text": reply.handoff, "to": rec.get("to", "")})
        await channels_push(self.state)
        return ok

    async def _deliver(self, text, reply, target, t0, t_stt, t_fd):
        say = reply.say
        if reply.handoff:
            await self.send({"type": "status", "state": "working"})
            try:
                await self._post_reply(text, reply, target)
            except Exception:
                say = "Sorry, I could not hand that over. Please try again."
        log_line("reply: " + say[:200].replace("\n", " "), self.svc.turn_log)   # what the assistant said (not what the caller said), to debug refusals
        self.history.append({"role": "assistant", "text": say})
        await self.send({"type": "transcript", "role": "assistant", "text": say})
        t_first = await self.speak(say)
        log_turn(self.svc.turn_log, stt_ms=int((t_stt - t0) * 1000), frontdesk_ms=int((t_fd - t_stt) * 1000),
                 first_audio_ms=int(((t_first or time.monotonic()) - t0) * 1000), reply_chars=len(say),
                 **{k: v for k, v in getattr(self.svc.frontdesk, "last_timing", {}).items()})
        if self.svc.log_calls:
            self._append_call_log(text, say)

    async def speak(self, say):
        """Speak one reply in Qwen's voice; returns when the first sound went out (monotonic) or None."""
        await self.send({"type": "status", "state": "speaking"})
        t_first = None
        if self.phone_voice:
            await self.send({"type": "say", "text": say})   # the phone speaks it: no synthesis wait on the Mac
            t_first = time.monotonic()
        else:
            # The next sentence is synthesized while this one plays (one ahead): a fresh Qwen request takes 1-2 s, and that
            # gap between sentences sounded choppy.
            pieces = split_sentences(say)
            fb0 = getattr(self.svc.tts, "fallbacks", 0)
            ahead = None
            for i, piece in enumerate(pieces):
                q = ahead or prefetch_parts(self.svc.tts, piece)
                ahead = prefetch_parts(self.svc.tts, pieces[i + 1]) if i + 1 < len(pieces) else None
                while True:
                    wav = await asyncio.to_thread(q.get)
                    if wav is None:
                        break
                    await self.ws.send(wav)
                    if t_first is None:
                        t_first = time.monotonic()
        if not self.phone_voice and getattr(self.svc.tts, "fallbacks", 0) > fb0 and \
                isinstance(getattr(self.svc.tts, "fallback", None), tts_mod.MacSayTextToSpeech):
            log_line("voice: Mac voice spoke (Qwen %s)" % getattr(self.svc.tts, "last_error", ""), self.svc.turn_log)
            await self.send({"type": "voice_used", "engine": "mac"})
        await self.send({"type": "audio_end"})
        await self.send({"type": "status", "state": "listening"})
        return t_first

    def _append_call_log(self, user_text, say):
        try:
            os.makedirs(CALL_LOG_DIR, exist_ok=True)
            with open(os.path.join(CALL_LOG_DIR, time.strftime("%Y%m%d") + ".log"), "a") as f:
                f.write("%s | %s: %s | Assistant: %s\n" % (time.strftime("%H:%M:%S"), config.get("owner"), user_text, say))
        except OSError:
            pass

    async def _updates_loop(self):
        """Sessions' replies (updates.py) are spoken when Dot is quiet. A new sentence from the caller interrupts them like any reply."""
        await asyncio.sleep(2)
        while True:
            try:
                busy = (self.task is not None and not self.task.done()) or (self.live is not None and getattr(self.live, "busy", getattr(self.live, "response_active", False)))
                open_rows = any(r["status"] == "sent" for r in self.state.channel_rows)
                items = await asyncio.to_thread(updates.pending, self.svc.updates_folder) \
                    if (open_rows or (not busy and not self.state.notices)) else []
                if open_rows and channel_replies(self.state, items):
                    await channels_push(self.state)   # R27: shown even while Dot is busy; speaking is unchanged (R8, R16)
                if not busy and self.state.notices:
                    self.task = asyncio.create_task(self._say_update(self.state.notices.pop(0)))
                elif not busy:
                    if items:
                        path, rec = items[0]
                        self.task = asyncio.create_task(self._say_update(updates.spoken_line(rec), path, rec))
            except asyncio.CancelledError:
                raise
            except Exception:
                log_exception("updates loop")
            await asyncio.sleep(3)

    async def _say_update(self, line, path=None, rec=None):
        if self.live and getattr(self.live, "busy", False):
            return   # still pending: the loop tries again when Dot is quiet (never lost)
        if path:
            await asyncio.to_thread(updates.mark_spoken, path)
            log_line("update spoken from %s" % (rec or {}).get("from"), self.svc.turn_log)
        if self.live:   # Live shows Dot's words itself (live_said, also its memory): one bubble, not two
            await self.live.say_exact(line)
            return
        self.history.append({"role": "assistant", "text": line})
        await self.send({"type": "transcript", "role": "assistant", "text": line})
        await self.speak(line)

    async def _keepalive(self):
        """Keep the speech model and the front-desk process resident in memory while a call is open (idle processes get swapped out on this Mac)."""
        try:
            t0 = time.monotonic()
            await asyncio.gather(asyncio.to_thread(self._touch_stt), asyncio.to_thread(self._prepare_desk), return_exceptions=True)
            log_line("call: warm-up took %d ms" % int((time.monotonic() - t0) * 1000), self.svc.turn_log)
            n = 0
            while True:
                await asyncio.sleep(25)
                n += 1
                if self.task is None or self.task.done():
                    await asyncio.to_thread(self._touch_stt)
                    if n % 5 == 0:   # about every two minutes
                        await asyncio.to_thread(self._ping_desk)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_line("keepalive stopped: %s" % type(e).__name__)

    def _touch_stt(self):
        if self.ears == "mac":   # when the phone listens, leave the Mac speech model alone (it only costs memory)
            self.svc.stt.transcribe(KEEPALIVE_CLIP)

    def _ping_desk(self):
        ping = getattr(self.svc.frontdesk, "ping", None)
        if ping:
            ping()

    def _prepare_desk(self):
        prep = getattr(self.svc.frontdesk, "prepare", None)
        if prep:
            prep()

    async def cancel_turn(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        self.task = None

    async def run(self):
        if not await self.authenticate():
            return
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    if self.live_pending:
                        continue
                    if self.live and not getattr(self.live, "closed", False):
                        await self.live.feed_audio(msg)
                    else:
                        self.live = None
                        if len(msg) < 16000:
                            continue
                        log_line("call: utterance received, %.1fs" % (len(msg) / 32000.0), self.svc.turn_log)
                        await self.cancel_turn()  # a new utterance replaces whatever was in progress
                        self.task = asyncio.create_task(self._guarded_turn(msg))
                else:
                    try:
                        data = json.loads(msg)
                    except ValueError:
                        continue
                    kind = data.get("type")
                    if kind == "live":
                        await self.release_suggestions()
                        if data.get("lang") in ("zh-CN", "en-US"):
                            self.live_lang = data["lang"]
                        if data.get("on"):
                            self.live_pending = True
                            try:
                                if self.live:
                                    await self.live.close()
                                    self.live = None
                                bridge = await self._start_live(realtime.bridge_classes())
                                self.live = bridge
                                self.live_pending = False
                                await self.send({"type": "live_ok"})
                                log_line("live: on (%s)" % self.live_lang, self.svc.turn_log)
                                self.recap_start()
                            except Exception as e:
                                self.live = None
                                self.live_pending = False
                                reason = (str(e) or type(e).__name__)[:120]
                                await self.send({"type": "live_error", "text": reason})
                                log_line("live: error %s" % reason, self.svc.turn_log)
                        else:
                            self.live_pending = False
                            await LiveTools(self).post_pending()
                            if self.live:
                                await self.live.close()
                                self.live = None
                                log_line("live: off", self.svc.turn_log)
                            await self.recap_finish()
                    elif kind == "barge":   # Live: the phone heard her over Dot and stopped playing
                        if self.live and hasattr(self.live, "barge"):
                            await self.live.barge()
                    elif kind == "interrupt":
                        await self.cancel_turn()
                        if self.live:
                            await self.live.cancel()
                        await self.send({"type": "status", "state": "listening"})
                    elif kind == "text":
                        if self.live:
                            if self.bug:
                                said = str(data.get("text", "")).strip()[:600]
                                if said:
                                    await self._record_bug_words(said)
                        else:
                            said = str(data.get("text", "")).strip()[:600]
                            min_len = 1 if self.bug else 2
                            if len(said) >= min_len:
                                await self.cancel_turn()
                                self.task = asyncio.create_task(self._guarded_turn(None, said))
                    elif kind == "bug":
                        await self.cancel_turn()
                        self.task = asyncio.create_task(self._on_bug(data.get("snapshot") or {}))
                    elif kind == "ears":
                        self.ears = "phone" if data.get("mode") == "phone" else "mac"
                    elif kind == "target":
                        self._move(data.get("role") if data.get("role") in TARGET_KEYS else "AUTO", "menu")
                    elif kind == "voice":
                        self.phone_voice = data.get("engine") == "phone"
                    elif kind == "passkey_register_options" and self.svc.passkeys and getattr(self, "signed_with_pin", False):
                        # enrolling Face ID needs a call that was opened with the PIN
                        opts, self.reg_challenge = await asyncio.to_thread(self.svc.passkeys.register_options)
                        await self.send({"type": "passkey_options", "options": opts})
                    elif kind == "passkey_register" and self.svc.passkeys and getattr(self, "reg_challenge", None):
                        challenge, self.reg_challenge = self.reg_challenge, None
                        cred = data.get("credential") if isinstance(data.get("credential"), dict) else {}
                        ok = await asyncio.to_thread(self.svc.passkeys.verify_register, cred, challenge, "iPhone")
                        log_line("call: Face ID enrolled" if ok else "call: Face ID enrolment refused", self.svc.turn_log)
                        await self.send({"type": "passkey_ok" if ok else "passkey_fail"})
                    elif kind == "ping":
                        await self.send({"type": "pong"})
                    elif kind == "metric":   # phone: end of the caller's speech -> first reply sound. A number only, never text.
                        try:
                            e2e = int(data.get("e2e_ms"))
                        except (TypeError, ValueError):
                            continue
                        if 0 <= e2e <= 120000:
                            log_turn(self.svc.turn_log, phone_e2e_ms=e2e)
        except Exception as e:
            if type(e).__name__ not in ("ConnectionClosedError", "ConnectionClosedOK", "ConnectionClosed"):
                raise
        finally:
            for t in (getattr(self, "keep", None), getattr(self, "upd", None), getattr(self, "bug_timer", None)):
                if t:
                    t.cancel()
            await self.cancel_turn()
            if self.live:
                try:
                    await self.live.close()
                except Exception:
                    pass
                self.live = None
            if not self.replaced:
                await self.recap_finish()
                log_line("live: off", self.svc.turn_log)
            if not self.replaced:
                await LiveTools(self).post_pending()
                await self.release_suggestions()
            if self.state.call is self:
                self.state.call = None
                self.state.last_used = time.time()
            self._write_call_summary()

    async def _on_bug(self, snapshot):
        try:
            await self._on_bug_inner(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception:
            log_exception("bug report")

    async def _on_bug_inner(self, snapshot):
        if self.live and not getattr(self.live, "closed", False):
            if getattr(self.live, "cancel", None):
                await self.live.cancel()
        if getattr(self, "bug_timer", None) and not self.bug_timer.done():
            self.bug_timer.cancel()
            self.bug_timer = None

        stamp = time.strftime("%Y%m%d-%H%M%S")
        server_lines = _last_turn_lines(getattr(self.svc, "turn_log", None), 40)
        self.bug = {
            "snapshot": _cap_snapshot(snapshot),
            "server": server_lines,
            "version": VERSION,
            "created": stamp,
        }
        log_line("call: bug recorded", self.svc.turn_log)
        self.bug_timer = asyncio.create_task(self._bug_timeout())   # before speaking: the prompt may be cancelled
        await self.send({"type": "status_note", "text": "Bug recorded. What went wrong?"})
        if self.live and not getattr(self.live, "closed", False):
            await self.live.say_exact("Bug recorded. What went wrong?")
        else:
            await self.speak("Bug recorded. What went wrong?")

    async def _bug_timeout(self):
        try:
            await asyncio.sleep(BUG_TIMEOUT_S)
            if self.bug:
                await self._record_bug_words("", speak_reply=False)
        except asyncio.CancelledError:
            pass
        except Exception:
            log_exception("bug_timeout")

    async def _record_bug_words(self, words, speak_reply=True):
        if getattr(self, "bug_timer", None) and not self.bug_timer.done():
            self.bug_timer.cancel()
            self.bug_timer = None
        if not self.bug:
            return
        words = str(words or "").strip()
        self.bug["words"] = words
        stamp = self.bug.get("created") or time.strftime("%Y%m%d-%H%M%S")
        doc = {
            "to": "VOICE",
            "task": "Bug report from the call",
            "spoken": words,
            "source": "voice_call_bug",
            "bug": self.bug,
        }
        handoff_obj = getattr(self.svc, "handoff", None)
        folder = getattr(handoff_obj, "folder", None) or os.path.join(HERE, "handoff")
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "%s-bug.json" % stamp)
            n = 1
            while os.path.exists(path):
                n += 1
                path = os.path.join(folder, "%s-bug-%d.json" % (stamp, n))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
        except OSError:
            log_exception("bug_report_write")
            saved = False
        else:
            saved = True

        self.posted_handoffs.append({"to": "VOICE", "words": words})

        self.bug = None   # cleared before speaking: a second utterance meanwhile goes to the model, not a 2nd report
        reply = "Thanks, sent to Voice." if saved else "Sorry, I could not save the bug report."
        if speak_reply:
            if self.live and not getattr(self.live, "closed", False):
                await self.live.say_exact(reply)
            else:
                await self.send({"type": "transcript", "role": "assistant", "text": reply})
                await self.speak(reply)

    def _write_call_summary(self):
        try:
            from status import PRIVATE_ROLES
            valid = [
                {"to": h["to"], "words": h["words"]}
                for h in self.posted_handoffs
                if h.get("to") and h.get("to") not in PRIVATE_ROLES and not h.get("summarized")
            ]
            for h in self.posted_handoffs:   # a reconnect never writes the same hand-off twice
                h["summarized"] = True
            if not valid:
                return
            stamp = time.strftime("%Y%m%d-%H%M%S")
            ended = time.strftime("%Y-%m-%d %H:%M:%S")
            summary = {
                "source": "voice_call_summary",
                "handoffs": valid,
                "started": getattr(self, "started_at", ended),
                "ended": ended,
            }
            handoff_dir = getattr(getattr(self, "svc", None), "handoff", None)
            folder = getattr(handoff_dir, "folder", None) or os.path.join(HERE, "handoff")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "%s-call-summary.json" % stamp)
            n = 1
            while os.path.exists(path):
                n += 1
                path = os.path.join(folder, "%s-call-summary-%d.json" % (stamp, n))
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

    async def _guarded_turn(self, pcm, text=None):
        try:
            await self.turn(pcm, text)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep the call alive; say what failed in plain words
            log_exception("turn")
            try:
                await self.send({"type": "error", "message": "Sorry, I missed that. Please say it again."})
                await self.send({"type": "status", "state": "listening"})
            except Exception:
                pass


def make_process_request(svc=None):
    def process_request(connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        if request.path in ("/", "/index.html"):
            try:
                body = open(INDEX, encoding="utf-8").read()
                body = config.fill(body.replace("@@VERSION@@", VERSION).replace("@@BUILT@@", BUILT))
            except OSError:
                return connection.respond(HTTPStatus.NOT_FOUND, "page missing\n")
            resp = connection.respond(HTTPStatus.OK, body)
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            resp.headers["Cache-Control"] = "no-store"
            return resp
        m = re.fullmatch(r"/dot/([a-z0-9-]{1,40}\.jpg)", request.path)   # Dot's face (static/dot/), names checked: no paths
        if m:
            try:
                data = open(os.path.join(HERE, "static", "dot", m.group(1)), "rb").read()
            except OSError:
                return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
            resp = connection.respond(HTTPStatus.OK, "")
            resp.body = data
            for k, v in (("Content-Type", "image/jpeg"), ("Content-Length", str(len(data))), ("Cache-Control", "max-age=86400")):
                if k in resp.headers:
                    del resp.headers[k]   # respond() already set text ones; Headers appends, so replace
                resp.headers[k] = v
            return resp
        if request.path == "/passkey/login-options":   # a fresh single-use Face ID challenge; no secret in it
            pk = svc.passkeys if svc else None
            if not pk or not login_ok(request.headers, svc.allowed_login):
                return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
            opts = pk.login_options()
            if not opts:
                return connection.respond(HTTPStatus.NOT_FOUND, "no passkey\n")
            resp = connection.respond(HTTPStatus.OK, json.dumps(opts))
            del resp.headers["Content-Type"]
            resp.headers["Content-Type"] = "application/json"
            resp.headers["Cache-Control"] = "no-store"
            return resp
        if request.path.split("?")[0] == "/board":   # read-only Sessions Board; only for a signed-in call page (its token)
            q = urllib.parse.parse_qs(request.path.partition("?")[2])
            if not svc or not svc.resume_ok((q.get("t") or [""])[0]) or not login_ok(request.headers, svc.allowed_login):
                return connection.respond(HTTPStatus.FORBIDDEN, "Open the board from the call menu after signing in.\n")
            try:
                body = open(config.path("board_file"), encoding="utf-8").read()
            except OSError:
                return connection.respond(HTTPStatus.NOT_FOUND, "board not built yet\n")
            resp = connection.respond(HTTPStatus.OK, body)
            for k, v in (("Content-Type", "text/html; charset=utf-8"), ("Cache-Control", "no-store"),
                         ("Referrer-Policy", "no-referrer")):
                if k in resp.headers:
                    del resp.headers[k]
                resp.headers[k] = v
            return resp
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "ok\n")
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
    return process_request


async def serve(svc, host, port):
    from websockets.asyncio.server import serve as ws_serve

    async def handler(ws):
        await Call(ws, svc).run()

    return await ws_serve(handler, host, port, process_request=make_process_request(svc), max_size=8 * 1024 * 1024,
                          ping_interval=30, ping_timeout=120)   # the Mac is short of memory: be patient before calling a call dead


def real_services(args):
    pin = load_or_create_pin(PIN_FILE)
    speech = WhisperSpeechToText()
    speech.warm()   # loads the speech model in its own process while the service starts
    desk = ClaudeFrontDesk()
    try:   # brain.txt says which brain answers: "claude" (default) or "qwen" (fast, but the caller's words go to Alibaba Cloud)
        brain = open(os.path.join(HERE, "brain.txt")).read().strip().lower()
    except OSError:
        brain = "claude"
    if brain == "qwen":
        from frontdesk import ChainFrontDesk, QwenFrontDesk
        q = QwenFrontDesk()
        if q.key:
            from frontdesk import QwenAgainFrontDesk
            desk = ChainFrontDesk(q, QwenAgainFrontDesk(q))   # Qwen-only mode: no Haiku answers
    threading.Thread(target=desk.warm, daemon=True).start()   # persistent front-desk process, ready before the first call
    return Services(speech, desk, default_tts(), FileHandoff(post=not args.no_post),
                    PinGuard(pin), allowed_login=os.environ.get("VOICE_ALLOWED_LOGIN", ""),
                    log_calls=os.environ.get("VOICE_CALL_LOGS", "0") == "1", passkeys=load_passkeys())


def load_passkeys():
    """Face ID sign-in. Without py_webauthn the PIN still works."""
    try:
        from passkey import Passkeys
        return Passkeys()
    except Exception as e:
        log_line("passkeys off: %s" % type(e).__name__)
        return None


class SlowFakeFrontDesk(FakeFrontDesk):
    """Harness only: the fake front desk with a think delay, so words can arrive while it thinks."""

    def __init__(self, delay_s):
        super().__init__()
        self.delay_s = delay_s

    def reply(self, history, text):
        time.sleep(self.delay_s)
        return super().reply(history, text)


def fake_services(args):
    """--fake: no models, no cloud, no messages to other sessions. For the browser harness on a staging port."""
    pin = load_or_create_pin(PIN_FILE)
    delay = float(os.environ.get("VOICE_FAKE_DELAY_S", "1.5"))
    svc = Services(FakeSpeechToText("words from mic"), SlowFakeFrontDesk(delay), FakeTextToSpeech(), FakeHandoff(),
                   PinGuard(pin))

    import fake_live   # a stand-in Live session (no network); Qwen Live is always refused, so fallbacks are real
    svc.live_connect = fake_live.fake_connect
    return svc


async def callback_loop(svc, every_s=5):
    """R47: look for session replies to hand-offs while no call is open; callback.tick decides (off unless callback_on_reply)."""
    while True:
        try:
            if config.get("callback_on_reply"):
                live = any(st.call is not None for st in svc.call_states)
                items = await asyncio.to_thread(updates.pending, svc.updates_folder)
                await asyncio.to_thread(callback.tick, live, items=items, aliases=ROLE_ALIASES, log_path=svc.turn_log)
        except asyncio.CancelledError:
            raise
        except Exception:
            log_exception("callback loop")
        await asyncio.sleep(every_s)


async def run_forever(args):
    svc = fake_services(args) if args.fake else real_services(args)
    server = await serve(svc, args.host, args.port)
    svc.callback_task = asyncio.create_task(callback_loop(svc))
    print("voice-call listening on %s:%d (PIN in %s). Put HTTPS in front with: tailscale serve --bg %d"
          % (args.host, args.port, PIN_FILE, args.port), flush=True)
    await server.serve_forever()


# ----------------------------------------------------------------------------- self test and config check
def fake_clip(seconds=1.0, rate=16000):
    import array
    import math
    n = int(seconds * rate)
    a = array.array("h", (int(8000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n)))
    return a.tobytes()


async def selftest_async():
    import tempfile
    from websockets.asyncio.client import connect
    tmp = tempfile.mkdtemp(prefix="voice-selftest-")
    svc = Services(FakeSpeechToText("please research the weekly report"), FakeFrontDesk(), FakeTextToSpeech(),
                   FakeHandoff(), PinGuard("123456"), turn_log=os.path.join(tmp, "turns.log"))
    server = await serve(svc, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with connect("ws://127.0.0.1:%d" % port) as ws:
        await ws.send(json.dumps({"type": "auth", "pin": "123456"}))
        assert json.loads(await ws.recv())["type"] == "auth_ok"
        await ws.recv()  # status listening
        await ws.send(b"\x00\x00" * 16000)  # one second of silence: must be ignored
        await ws.send(fake_clip())
        got = {"user": False, "assistant": False, "audio": False, "handoff": False}
        for _ in range(20):
            m = await asyncio.wait_for(ws.recv(), 10)
            if isinstance(m, bytes):
                got["audio"] = True
                continue
            d = json.loads(m)
            if d["type"] == "transcript":
                got[d["role"]] = True
            if d["type"] == "handoff":
                got["handoff"] = True
            if d["type"] == "audio_end":
                break
        assert all(got.values()), got
    server.close()
    await server.wait_closed()
    assert len(svc.handoff.items) == 1
    return True


def selftest():
    asyncio.run(selftest_async())
    print("SELFTEST OK")


def check_config():
    problems = []
    try:
        import websockets  # noqa: F401
    except ImportError:
        problems.append("websockets is not installed (pip install -r requirements.txt)")
    if not os.path.exists(INDEX):
        problems.append("static/index.html is missing")
    if not os.path.exists(PIN_FILE):
        problems.append("pin.txt does not exist yet (it is created with a random 6-digit PIN on the first --run)")
    if not available_engine():
        problems.append("no local speech model package (pip install mlx-whisper, or faster-whisper)")
    from tts import qwen_key
    if not qwen_key():
        problems.append("no Qwen key found (replies use the Mac voice); run: server.py --qwen-key-setup")
    for tool, why in (("claude", "the front desk"), ("say", "spoken replies"), ("afconvert", "audio conversion"),
                      ("tailscale", "HTTPS for the iPhone (tailscale serve)")):
        if not shutil.which(tool) and not (tool == "claude" and os.path.exists(os.path.expanduser("~/.local/bin/claude"))):
            problems.append("%s not found (needed for %s)" % (tool, why))
    from frontdesk import BRAIN_TOOL
    if not os.path.exists(BRAIN_TOOL):
        problems.append("second brain tool not found at %s (answers will have no notes)" % BRAIN_TOOL)
    for p in problems:
        print("MISSING: " + p)
    print("CONFIG OK" if not any("websockets" in p or "index.html" in p for p in problems) else "CONFIG INCOMPLETE")


def bench():
    from frontdesk import ClaudeFrontDesk, brain_notes
    t = time.monotonic()
    notes = brain_notes("what is my rule about weekends")
    print("brain search: %d ms (%d chars)" % ((time.monotonic() - t) * 1000, len(notes)), flush=True)
    t = time.monotonic()
    fd = ClaudeFrontDesk(notes_fn=lambda q: "")
    r = fd.reply([], "Say hello in five words.")
    print("front desk: %d ms -> %s" % ((time.monotonic() - t) * 1000, r.say[:60]), flush=True)
    print("front desk detail: %s" % getattr(fd, "last_timing", {}), flush=True)
    import audio as audio_mod
    speech = WhisperSpeechToText()
    speech.warm()
    clip = fake_clip(5.0)
    for n in (1, 2):
        t = time.monotonic()
        try:
            speech.transcribe(clip)
            print("speech to text, 5 s clip, call %d: %d ms" % (n, (time.monotonic() - t) * 1000), flush=True)
        except Exception as e:
            print("speech to text FAILED: %s" % e, flush=True)
    speech._stop()
    for name, engine in (("mac say", tts_mod.MacSayTextToSpeech()), ("qwen", tts_mod.QwenTextToSpeech())):
        if name == "qwen" and not engine.available():
            print("qwen voice: no key")
            continue
        t = time.monotonic()
        try:
            n = len(engine.speak("Hello %s, this is a speed test." % config.get("owner")))
            print("%s voice: %d ms (%d bytes)" % (name, (time.monotonic() - t) * 1000, n), flush=True)
        except Exception as e:
            print("%s voice FAILED: %s" % (name, e), flush=True)


def install_launchagent():
    label = config.get("launchagent_label")
    path = os.path.expanduser("~/Library/LaunchAgents/%s.plist" % label)
    plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>%s</string>
<key>ProgramArguments</key><array><string>%s</string><string>%s</string><string>--run</string></array>
<key>WorkingDirectory</key><string>%s</string>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>%s/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
<key>KeepAlive</key><true/>
<key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>%s/logs/server.out.log</string>
<key>StandardErrorPath</key><string>%s/logs/server.err.log</string>
</dict></plist>
""" % (label, sys.executable, os.path.join(HERE, "server.py"), HERE, os.path.expanduser("~"), HERE, HERE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(plist)
    print("Wrote %s (not loaded). To start it: launchctl load %s" % (path, path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run in the foreground")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-config", action="store_true")
    ap.add_argument("--install-launchagent", action="store_true")
    ap.add_argument("--qwen-key-setup", action="store_true", help="copy the saved Qwen key into the Keychain item this service reads")
    ap.add_argument("--qwen-test", action="store_true", help="speak one short sentence with Qwen and report the result")
    ap.add_argument("--bench", action="store_true", help="time the brain search, the front desk and the voice once")
    ap.add_argument("--warm", action="store_true", help="download and load the speech model now (first run only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-post", action="store_true", help="staging: hand-offs stay in handoff/, no messages to sessions")
    ap.add_argument("--fake", action="store_true", help="fake brain/voice/hand-offs (browser harness, staging port)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.check_config:
        return check_config()
    if args.install_launchagent:
        return install_launchagent()
    if args.qwen_key_setup:
        print(find_and_store_qwen_key())
        return
    if args.qwen_test:
        q = QwenTextToSpeech()
        if not q.available():
            print("QWEN TEST FAILED: no key. Run --qwen-key-setup first.")
            return
        try:
            wav = q.speak("Hello %s, this is the Qwen voice." % config.get("owner"))
            print("QWEN TEST OK: %d bytes of audio" % len(wav))
        except Exception as e:
            print("QWEN TEST FAILED: %s" % e)
        return
    if args.bench:
        return bench()
    if args.warm:
        from stt import WhisperEngine
        WhisperEngine().warm()
        print("SPEECH MODEL READY")
        return
    if not host_allowed(args.host):
        sys.exit("Refusing to listen on %r. Use 127.0.0.1 (with tailscale serve) or the Tailscale address." % args.host)
    if args.run:
        try:
            asyncio.run(run_forever(args))
        except KeyboardInterrupt:
            pass
        return
    ap.print_help()


if __name__ == "__main__":
    main()
