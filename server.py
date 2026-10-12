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
from stt import FakeSpeechToText, WhisperSpeechToText, available_engine  # noqa: E402
import tts as tts_mod  # noqa: E402
import updates  # noqa: E402
from tts import FakeTextToSpeech, default_tts, find_and_store_qwen_key, QwenTextToSpeech  # noqa: E402
import realtime  # noqa: E402

PORT = 8770
AUTH_TIMEOUT_S = 20
BUG_TIMEOUT_S = 60
RESUME_TTL_S = 12 * 3600   # a reconnect token lasts one working day
RESUME_MAX = 10
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

    async def hand_off(self, words, agent):
        if getattr(self.call, "bug", None):
            return "not sent: this is a bug report, say only: noted"
        if getattr(self.call, "handoff_count", 0) >= 6:
            return "limit reached, tell the caller to use the menu"
        try:
            rec = await asyncio.to_thread(self.call.svc.handoff.create, words, words, role=agent)
        except Exception as e:
            return "not sent: %s" % (str(e) or type(e).__name__)[:120]
        if rec and rec.get("posted"):
            self.call.handoff_count = getattr(self.call, "handoff_count", 0) + 1
            self.call.posted_handoffs.append({"to": rec.get("to", ""), "words": words})
            try:
                await self.call.send({"type": "handoff", "id": rec.get("id", ""), "text": words, "to": rec.get("to", "")})
            except Exception:
                pass
            return "posted to %s" % rec.get("to", agent or "")
        reason = (rec.get("post_error") if rec else "") or "failed to post"
        return "not sent: %s" % reason

    async def status(self, question):
        try:
            res = await asyncio.to_thread(direct_status, question)
        except Exception:
            res = None
        if res:
            return str(res).strip()
        return "No saved status for that."

    async def lookup(self, words):
        try:
            import frontdesk
            notes = await asyncio.to_thread(frontdesk.gather_notes, words, frontdesk.brain_notes)
        except Exception:
            return "Nothing found."
        notes = (notes or "").strip()
        return notes if notes else "Nothing found."


class Call:
    def __init__(self, ws, svc):
        self.ws, self.svc = ws, svc
        self.history = []
        self.phone_voice = False
        self.ears = "mac"
        self.task = None
        self.live = None
        self.live_pending = False
        self.posted_handoffs = []
        self.handoff_count = 0
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.bug = None
        self.bug_timer = None
        self.svc.last_call = self   # tests reach the live Call through this

    def _on_live_closed(self):
        self.live = None

    async def _live_out(self, item):
        if isinstance(item, bytes):
            stopped_at = getattr(self.live, "stopped_at", None) if self.live else None
            if stopped_at is not None:
                ms = int((time.monotonic() - stopped_at) * 1000)
                log_line("live first_audio_ms=%d" % ms, self.svc.turn_log)
                if self.live:
                    self.live.stopped_at = None
            b64 = base64.b64encode(item).decode("ascii")
            await self.send({"type": "live_audio", "b64": b64})
        elif isinstance(item, dict):
            if item.get("type") == "live_error":
                log_line("live: error %s" % str(item.get("text", ""))[:120], self.svc.turn_log)
                self.live = None
            elif item.get("type") == "live_heard":
                words = str(item.get("text", "")).strip()
                if self.bug and words:
                    if self.live:
                        await self.live.cancel()
                    await self._record_bug_words(words)
                    await self.send(item)
                    return
            await self.send(item)

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
        self.ears = "phone" if msg.get("ears") == "phone" else "mac"
        self.target = msg.get("target") if msg.get("target") in TARGET_KEYS else "AUTO"
        if not ok:
            await self.send({"type": "auth_retry"} if retry else {"type": "auth_fail", "locked_s": locked,
                                                                  "face": bool(msg.get("passkey")) and not pin})
            return False
        log_line("call: connected, %s, phone_voice=%s, ears=%s, home_screen_app=%s" % (VERSION, self.phone_voice, self.ears,
                 bool(msg.get("standalone"))), self.svc.turn_log)
        self.signed_with_pin = bool(msg.get("pin"))
        await self.send({"type": "auth_ok", "version": VERSION, "resume": self.svc.issue_resume(), "targets": TARGETS,
                         "passkeys": self.svc.passkeys.count() if self.svc.passkeys else -1})
        self.keep = asyncio.create_task(self._keepalive())
        self.upd = asyncio.create_task(self._updates_loop())
        await self.send({"type": "status", "state": "listening"})
        return True

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
        say = reply.say
        if reply.handoff:
            await self.send({"type": "status", "state": "working"})
            try:
                rec = await asyncio.to_thread(self.svc.handoff.create, reply.handoff, text, target)
                if rec and rec.get("posted"):
                    self.posted_handoffs.append({"to": rec.get("to", target), "words": text})
                await self.send({"type": "handoff", "id": rec.get("id", ""), "text": reply.handoff, "to": rec.get("to", "")})
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
                if not busy:
                    items = await asyncio.to_thread(updates.pending, self.svc.updates_folder)
                    if items:
                        path, rec = items[0]
                        await asyncio.to_thread(updates.mark_spoken, path)
                        line = updates.spoken_line(rec)
                        log_line("update spoken from %s" % rec.get("from"), self.svc.turn_log)
                        self.task = asyncio.create_task(self._say_update(line))
            except asyncio.CancelledError:
                raise
            except Exception:
                log_exception("updates loop")
            await asyncio.sleep(3)

    async def _say_update(self, line):
        if self.live and getattr(self.live, "busy", False):
            return
        self.history.append({"role": "assistant", "text": line})
        await self.send({"type": "transcript", "role": "assistant", "text": line})
        if self.live:
            await self.live.say_exact(line)
        else:
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
                        if data.get("on"):
                            self.live_pending = True
                            try:
                                if self.live:
                                    await self.live.close()
                                    self.live = None
                                transcribe_fn = getattr(getattr(self.svc, "speech", None) or getattr(self.svc, "stt", None), "transcribe", None)
                                extra = {"connect": self.svc.live_connect} if getattr(self.svc, "live_connect", None) else {}
                                bridge = realtime.LiveBridge(
                                    self._live_out,
                                    LiveTools(self),
                                    realtime.build_instructions(config.get("owner")),
                                    transcribe=transcribe_fn,
                                    on_closed=self._on_live_closed,
                                    turn_log=self.svc.turn_log,
                                    **extra,
                                )
                                try:
                                    await asyncio.wait_for(bridge.start(), 6)   # never block the call on a silent provider
                                except BaseException:
                                    try:
                                        await bridge.close()
                                    except Exception:
                                        pass
                                    raise
                                self.live = bridge
                                self.live_pending = False
                                await self.send({"type": "live_ok"})
                                log_line("live: on", self.svc.turn_log)
                            except Exception as e:
                                self.live = None
                                self.live_pending = False
                                reason = (str(e) or type(e).__name__)[:120]
                                await self.send({"type": "live_error", "text": reason})
                                log_line("live: error %s" % reason, self.svc.turn_log)
                        else:
                            self.live_pending = False
                            if self.live:
                                await self.live.close()
                                self.live = None
                                log_line("live: off", self.svc.turn_log)
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
                        self.target = data.get("role") if data.get("role") in TARGET_KEYS else "AUTO"
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
                log_line("live: off", self.svc.turn_log)
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
                if h.get("to") and h.get("to") not in PRIVATE_ROLES
            ]
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

    async def no_live(url, **kw):   # never reach Qwen from the harness: Live fails and the page must fall back
        raise ConnectionRefusedError("fake mode: no Live provider")
    svc.live_connect = no_live
    return svc


async def run_forever(args):
    svc = fake_services(args) if args.fake else real_services(args)
    server = await serve(svc, args.host, args.port)
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
