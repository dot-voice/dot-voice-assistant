"""Phone bridge: call Dot from a real phone number (Twilio Media Streams). An adapter only: it speaks to the running Dot
service exactly like the call page does (ws://127.0.0.1:8770/ws, Live mode) and converts the audio both ways.

Run:  python3 twilio_bridge.py --run      (127.0.0.1:8772; expose only /tw/<secret> with `tailscale funnel`, see docs/PHONE.md)

Security layers: secret path token, caller-ID allowlist, keypad PIN or spoken passphrase (twilio_auth, 3 tries), one-time per-call stream token (60 s),
and the Twilio request signature when an auth token is stored in the Keychain (KEYCHAIN_ITEM below).

Twilio is set to HTTP GET for the voice and PIN webhooks: the websockets server cannot read a POST body, and GET keeps this
to one small port and one library. Audio: Twilio is 8 kHz mu-law 20 ms frames; Dot takes 16 kHz mono 16-bit PCM
(960-sample chunks) and answers with 24 kHz mono 16-bit PCM in "live_audio" messages; "live_stop" means stop playing.
"""
import asyncio
import base64
import binascii
import difflib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from hashlib import sha1
from http import HTTPStatus
from urllib.parse import parse_qsl, urlsplit
from xml.sax.saxutils import escape, quoteattr

import numpy as np

import config
from auth import PinGuard

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_FILE = os.path.join(HERE, "pin.txt")
DOT_URL = "ws://127.0.0.1:8770/ws"
MAX_PIN_TRIES = 3
TOKEN_TTL_S = 60
MIN_SECRET_LEN = 24
DOT_CHUNK = 960          # samples of 16 kHz PCM per message to Dot (the page sends the same)
TW_FRAME = 160           # 20 ms of 8 kHz mu-law
KEYCHAIN_ITEM = "voice-call-twilio-auth"
PHRASE_ITEM = "dot-voice-phrase"   # Keychain item holding the normalized spoken passphrase
PHRASE_MIN_RATIO = 0.85
PHRASE_LONG_WORD = 4               # every phrase word at least this long must be heard
_STT_VARIANTS = {"okay": "ok", "hay": "hey", "a": "", "the": ""}   # tiny, applied to both sides


# ------------------------------------------------------------------ audio
def _build_mulaw_tables():
    u = (~np.arange(256, dtype=np.int32)) & 0xFF
    mag = ((((u & 0x0F) << 3) + 0x84) << ((u >> 4) & 7)) - 0x84
    dec = np.where(u & 0x80, -mag, mag).astype(np.int16)
    s = np.arange(-32768, 32768, dtype=np.int32)
    sign = np.where(s < 0, 0x80, 0)
    m = np.minimum(np.abs(s), 32635) + 0x84
    exp = np.floor(np.log2(m)).astype(np.int32) - 7
    enc = ((~(sign | (exp << 4) | ((m >> (exp + 3)) & 0x0F))) & 0xFF).astype(np.uint8)
    return dec, enc


_MU_DEC, _MU_ENC = _build_mulaw_tables()


def mulaw_decode(data: bytes) -> np.ndarray:
    return _MU_DEC[np.frombuffer(data, dtype=np.uint8)]


def mulaw_encode(pcm: np.ndarray) -> bytes:
    return _MU_ENC[pcm.astype(np.int32) + 32768].tobytes()


# Band-limited sample-rate conversion (v54). Before v54 the phone path used linear interpolation up (8k -> 16k) and a plain
# average of 3 samples down (24k -> 8k): the average is a very weak low-pass (only ~6 dB down at 5 kHz), so high sounds folded
# back into the voice band as a metallic hiss, and the interpolation added images. Now both directions use a long
# linear-phase FIR (Kaiser window) in polyphase form, with the filter tail carried across chunks so frame edges are seamless.
def _design_lowpass(cutoff_hz, rate_hz, taps):
    from scipy.signal import firwin
    return firwin(taps, cutoff_hz, window=("kaiser", 7.0), fs=rate_hz)


_DOWN_TAPS = _design_lowpass(3900.0, 24000.0, 129)   # flat to 3.4 kHz, > 60 dB down from 4.4 kHz, group delay 2.7 ms
_UP_TAPS = _design_lowpass(3900.0, 16000.0, 97)      # removes the 4-8 kHz images of the 2x zero-stuffing, delay 3 ms
SOFT_KNEE = 28000.0


def soft_limit(x: np.ndarray) -> np.ndarray:
    """float samples -> int16 with a smooth knee above 28000 (filter overshoot never wraps or hard-clips)."""
    a = np.abs(x)
    room = 32767.0 - SOFT_KNEE
    y = np.where(a > SOFT_KNEE, SOFT_KNEE + room * np.tanh((a - SOFT_KNEE) / room), a)
    return np.rint(np.copysign(y, x)).astype(np.int16)


class Upsampler8to16:
    """8 kHz int16 -> 16 kHz int16: polyphase 2x interpolation with a low-pass; the filter tail is kept between frames."""

    def __init__(self):
        self.h0 = _UP_TAPS[0::2] * 2.0
        self.h1 = _UP_TAPS[1::2] * 2.0
        self.hist = np.zeros(len(self.h0) - 1)

    def feed(self, pcm8: np.ndarray) -> np.ndarray:
        if not len(pcm8):
            return np.zeros(0, dtype=np.int16)
        buf = np.concatenate((self.hist, pcm8.astype(np.float64)))
        n = len(pcm8)
        k = len(self.h0)
        out = np.empty(2 * n)
        out[0::2] = np.convolve(buf, self.h0, "valid")[:n]
        out[1::2] = np.convolve(buf, self.h1[:k] if len(self.h1) == k else np.append(self.h1, 0.0), "valid")[:n]
        self.hist = buf[len(buf) - (k - 1):]
        return soft_limit(out)


class Downsampler24to8:
    """24 kHz 16-bit PCM bytes -> 8 kHz int16: low-pass FIR evaluated only at every third sample (polyphase); the filter
    tail and the leftover odd bytes / samples are kept between chunks, so no sample is dropped or repeated."""

    def __init__(self):
        self.rest = b""
        self.hist = np.zeros(len(_DOWN_TAPS) - 1)

    def reset(self):
        self.rest = b""
        self.hist = np.zeros(len(_DOWN_TAPS) - 1)

    def feed(self, data: bytes) -> np.ndarray:
        data = self.rest + data
        n = len(data) // 6 * 6
        self.rest = data[n:]
        if not n:
            return np.zeros(0, dtype=np.int16)
        x = np.frombuffer(data[:n], dtype="<i2").astype(np.float64)
        buf = np.concatenate((self.hist, x))
        taps = len(_DOWN_TAPS)
        win = np.lib.stride_tricks.sliding_window_view(buf, taps)[::3]   # one row per output sample
        y = win @ _DOWN_TAPS[::-1]
        self.hist = buf[len(buf) - (taps - 1):]
        return soft_limit(y)


# ------------------------------------------------------------------ security helpers
def twilio_signature(auth_token: str, url: str, params=()) -> str:
    """Twilio's X-Twilio-Signature: base64(HMAC-SHA1(token, url + sorted key/value pairs))."""
    data = url + "".join(k + v for k, v in sorted(params))
    return base64.b64encode(hmac.new(auth_token.encode(), data.encode(), sha1).digest()).decode()


def keychain_auth_token():
    """The Twilio auth token from the Keychain, or "" when the item does not exist (signature check then skipped)."""
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_ITEM, "-w"], capture_output=True,
                           text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def normalize_phrase(text) -> str:
    """Lowercase, drop punctuation, collapse spaces, apply a few common speech-to-text variants."""
    t = re.sub(r"[^a-z0-9\s]", "", str(text or "").lower().replace("'", ""))
    words = [_STT_VARIANTS.get(w, w) for w in t.split()]
    return " ".join(w for w in words if w)


def phrase_score(spoken, stored):
    """(ok, ratio): fuzzy ratio >= PHRASE_MIN_RATIO and every long word of the stored phrase was heard."""
    a, b = normalize_phrase(spoken), normalize_phrase(stored)
    if not a or not b:
        return False, 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    heard = set(a.split())
    long_ok = all(w in heard for w in b.split() if len(w) >= PHRASE_LONG_WORD)
    return (ratio >= PHRASE_MIN_RATIO and long_ok), ratio


def keychain_phrase():
    """The stored passphrase from the Keychain, or "" when none is set."""
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", PHRASE_ITEM, "-w"], capture_output=True,
                           text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


class StreamTokens:
    """One-time stream tokens minted after the PIN: single use, TTL seconds."""

    def __init__(self, ttl=TOKEN_TTL_S, clock=time.monotonic):
        self.ttl, self.clock, self._t = ttl, clock, {}

    def mint(self, call_sid: str) -> str:
        now = self.clock()
        self._t = {k: v for k, v in self._t.items() if v[1] > now}
        tok = secrets.token_urlsafe(24)
        self._t[tok] = (call_sid, now + self.ttl)
        return tok

    def consume(self, token) -> bool:
        item = self._t.pop(token, None) if isinstance(token, str) else None
        return bool(item) and item[1] > self.clock()


# R57: Deepgram streams (was Google V1 phone_call + deprecated enhanced: 6.4-7.8 s sign-ins, her words 2026-10-30
# "the verification is still very slow"). Valid per twilio.com/docs/voice/twiml/gather.
SPEECH_MODEL = "deepgram_nova-3"


def twiml(*inner) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(inner) + "</Response>"


def log(msg):
    sys.stderr.write("%s twilio_bridge: %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


# ------------------------------------------------------------------ the bridge
class Bridge:
    def __init__(self, pin, secret, allowed_callers, public_host, dot_url=DOT_URL, auth_token_fn=keychain_auth_token,
                 allowed_login="", guard=None, tokens=None, fallback_voice_url="", stale_s=0,
                 auth_mode="pin", phrase_fn=keychain_phrase, greeting_name="", dot_number="", machine_fn=None):
        if len(secret or "") < MIN_SECRET_LEN:
            raise ValueError("twilio_path_secret must be at least %d characters" % MIN_SECRET_LEN)
        self.pin, self.secret = pin, secret
        self.allowed = {str(c) for c in allowed_callers}
        self.public_host, self.dot_url = public_host, dot_url
        self.auth_token_fn, self.allowed_login = auth_token_fn, allowed_login
        self.guard = guard or PinGuard(pin)
        self.tokens = tokens or StreamTokens()
        self.stale_s = stale_s   # R39: a call with no media from Twilio for this long is closed (0 = off)
        self.last_media = time.monotonic()
        self.tries = {}   # CallSid -> wrong PINs so far
        self.signin_start = {}   # CallSid -> time.time() of first sign-in prompt
        self.phrase_tries = {}   # CallSid -> spoken tries count
        self.base = "/tw/" + secret
        self.auth_mode, self.phrase_fn, self.greeting_name = auth_mode, phrase_fn, greeting_name
        self.dot_number = str(dot_number or "")   # R47: our own number; an outbound call-back has From = this, To = an allowed caller
        self.machine_fn = machine_fn              # R47: called (in a thread) when a machine answered a call-back
        self.fallback_voice_url = fallback_voice_url or ""   # other callers are sent here (her old call-forward service)

    # --- HTTP (TwiML) ---
    def _xml(self, connection, body, status=HTTPStatus.OK):
        resp = connection.respond(status, body)
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = "text/xml; charset=utf-8"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _signature_ok(self, request) -> bool:
        token = self.auth_token_fn()
        if not token:
            return True   # optional layer: no auth token stored, skip
        url = "https://" + self.public_host + request.path
        sig = request.headers.get("X-Twilio-Signature", "")
        return hmac.compare_digest(twilio_signature(token, url).encode(), sig.encode())

    def process_request(self, connection, request):
        parts = urlsplit(request.path)
        path = parts.path
        is_ws = request.headers.get("Upgrade", "").lower() == "websocket"
        if is_ws and hmac.compare_digest(path.encode(), (self.base + "/media").encode()):
            return None
        if is_ws or not path.startswith("/tw/"):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        route = None
        for name in ("voice", "pin", "phrase"):
            if hmac.compare_digest(path.encode(), ("%s/%s" % (self.base, name)).encode()):
                route = name
        if route is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        q = dict(parse_qsl(parts.query))
        if not self._signature_ok(request):
            log("refused: bad Twilio signature")
            return connection.respond(HTTPStatus.FORBIDDEN, "forbidden\n")
        if q.get("From", "") not in self.allowed and not self._own_callback(q):   # an empty allowlist rejects everyone
            log("refused: caller not on the allowlist")
            if route == "voice" and self.fallback_voice_url:
                return self._xml(connection, twiml("<Redirect method=\"POST\">%s</Redirect>" % escape(self.fallback_voice_url)))
            return self._xml(connection, twiml("<Reject/>"))
        if route == "voice" and self._own_callback(q) and (q.get("AnsweredBy", "").startswith("machine") or q.get("AnsweredBy") == "fax"):
            log("call-back answered by a machine: hanging up")
            if self.machine_fn:
                threading.Thread(target=self.machine_fn, daemon=True).start()
            return self._xml(connection, twiml("<Hangup/>"))
        body = {"voice": self.voice_twiml, "pin": self.pin_twiml, "phrase": self.phrase_twiml}[route](q)
        return self._xml(connection, body)

    def _own_callback(self, q) -> bool:
        """R47: a call Dot placed itself (Twilio REST): From is our number, To is an allowed caller. The secret path and the
        Twilio signature still apply before this is reached."""
        return bool(self.dot_number) and q.get("From", "") == self.dot_number and q.get("To", "") in self.allowed \
            and str(q.get("Direction", "")).startswith("outbound")

    def _gather(self, prompt):
        action = "https://%s%s/pin" % (self.public_host, self.base)
        return ('<Gather input="dtmf" finishOnKey="#" timeout="10" method="GET" action=%s><Say>%s</Say></Gather>'
                '<Hangup/>' % (quoteattr(action), escape(prompt)))

    def _phrase_mode(self) -> bool:
        if self.auth_mode != "phrase":
            return False
        if not self.phrase_fn():
            log("twilio_auth=phrase but no passphrase is stored: falling back to the PIN")
            return False
        return True

    def _gather_phrase(self, prompt):
        action = "https://%s%s/phrase" % (self.public_host, self.base)
        return ('<Gather input="speech dtmf" speechTimeout="1" speechModel="%s" timeout="6" '
                'language="en-US" finishOnKey="#" method="GET" action=%s><Say>%s</Say></Gather><Hangup/>'
                % (SPEECH_MODEL, quoteattr(action), escape(prompt)))

    def voice_twiml(self, q) -> str:
        if self._phrase_mode():
            sid = q.get("CallSid", "")
            if sid not in self.signin_start:
                self.signin_start[sid] = time.time()
            if len(self.signin_start) > 200:
                self.signin_start.pop(next(iter(self.signin_start)))
            who = (" " + self.greeting_name) if self.greeting_name else ""
            return twiml(self._gather_phrase("Hi%s." % who))
        return twiml(self._gather("Dot here. Enter your PIN, then press pound."))

    def phrase_twiml(self, q) -> str:
        """Speech goes to the phrase check, keypad digits to the PIN check; both share the PinGuard lockout."""
        if q.get("Digits"):
            return self.pin_twiml(q)
        sid = q.get("CallSid", "")
        tries = self.phrase_tries.get(sid, 0) + 1
        self.phrase_tries[sid] = tries
        if len(self.phrase_tries) > 200:
            self.phrase_tries.pop(next(iter(self.phrase_tries)))
        stored = self.phrase_fn() if self.auth_mode == "phrase" else ""
        left = self.guard.locked_for()
        ok, score = (False, 0.0) if (left or not stored) else phrase_score(q.get("SpeechResult", ""), stored)
        if ok:
            log("phrase ok after_s=%.1f tries=%d" % (time.time() - self.signin_start.get(sid, time.time()), tries))
            self.signin_start.pop(sid, None)
            self.phrase_tries.pop(sid, None)
            self.guard.reset()
            return self._connect(sid)
        locked = bool(left)
        if not left and q.get("SpeechResult"):
            _, locked_s = self.guard.record_failure()
            locked = bool(locked_s)
        log("phrase no match score=%.2f%s" % (score, ", locked" if locked else ""))
        n = self.tries.get(sid, 0) + 1
        self.tries[sid] = n
        if len(self.tries) > 200:
            self.tries.pop(next(iter(self.tries)))
        if n >= MAX_PIN_TRIES or locked:
            self.signin_start.pop(sid, None)
            self.phrase_tries.pop(sid, None)
            return twiml("<Say>Goodbye.</Say><Hangup/>")
        return twiml(self._gather_phrase("Sorry, say it again."))

    def _connect(self, sid):
        self.tries.pop(sid, None)
        tok = self.tokens.mint(sid)
        url = "wss://%s%s/media" % (self.public_host, self.base)
        return twiml("<Connect><Stream url=%s><Parameter name=\"token\" value=%s/></Stream></Connect><Hangup/>"
                     % (quoteattr(url), quoteattr(tok)))

    def pin_twiml(self, q) -> str:
        sid = q.get("CallSid", "")
        ok, locked = self.guard.check(q.get("Digits", ""))
        if ok:
            return self._connect(sid)
        n = self.tries.get(sid, 0) + 1
        self.tries[sid] = n
        if len(self.tries) > 200:
            self.tries.pop(next(iter(self.tries)))
        log("wrong PIN (try %d)%s" % (n, ", locked" if locked else ""))
        if n >= MAX_PIN_TRIES or locked:
            return twiml("<Say>Goodbye.</Say><Hangup/>")
        if self._phrase_mode():
            return twiml(self._gather_phrase("That PIN was not right. Try again."))
        return twiml(self._gather("That PIN was not right. Try again."))

    # --- Media Streams ---
    async def handler(self, ws):
        try:
            await self.media_call(ws)
        except Exception as e:   # one bad call never takes the bridge down
            if type(e).__name__ not in ("ConnectionClosedError", "ConnectionClosedOK", "ConnectionClosed"):
                log("call error: %s" % (str(e) or type(e).__name__)[:120])

    async def media_call(self, ws):
        from websockets.asyncio.client import connect
        stream_sid = None
        deadline = time.monotonic() + 10
        while stream_sid is None:   # connected, then start (with the one-time token)
            raw = await asyncio.wait_for(ws.recv(), max(0.1, deadline - time.monotonic()))
            ev = json.loads(raw) if isinstance(raw, str) else {}
            if ev.get("event") == "start":
                token = (ev.get("start", {}).get("customParameters") or {}).get("token")
                if not self.tokens.consume(token):
                    log("refused: bad or expired stream token")
                    await ws.close()
                    return
                stream_sid = ev.get("streamSid") or ev["start"].get("streamSid", "")
        headers = {"Tailscale-User-Login": self.allowed_login} if self.allowed_login else None
        async with connect(self.dot_url, additional_headers=headers, max_size=8 * 1024 * 1024) as dot:
            await dot.send(json.dumps({"type": "auth", "pin": self.pin, "voice": "phone", "ears": "phone",
                                       "client": "twilio", "source": "phone"}))
            if json.loads(await asyncio.wait_for(dot.recv(), 10)).get("type") != "auth_ok":
                log("Dot refused the bridge sign-in")
                await ws.close()
                return
            await dot.send(json.dumps({"type": "live", "on": True, "lang": "en-US"}))
            log("call connected")
            tasks = [asyncio.create_task(self._from_twilio(ws, dot)),
                     asyncio.create_task(self._from_dot(ws, dot, stream_sid))]
            self.last_media = time.monotonic()
            if self.stale_s:
                tasks.append(asyncio.create_task(self._reaper(self.stale_s)))
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks:
                    t.cancel()
                try:
                    await dot.send(json.dumps({"type": "live", "on": False}))
                except Exception:
                    pass
        await ws.close()
        log("call ended")

    async def _reaper(self, stale_s):
        """R39: Twilio sends media all the time on a live call; none for stale_s seconds means a dead call. Returning ends it."""
        while True:
            await asyncio.sleep(max(0.05, min(5.0, stale_s / 4.0)))
            quiet = time.monotonic() - self.last_media
            if quiet >= stale_s:
                log("stale call closed: no media for %d s" % int(quiet))
                return

    async def _from_twilio(self, ws, dot):
        up, acc = Upsampler8to16(), np.zeros(0, dtype=np.int16)
        async for raw in ws:
            ev = json.loads(raw) if isinstance(raw, str) else {}
            kind = ev.get("event")
            if kind == "stop":
                return
            if kind != "media" or ev.get("media", {}).get("track", "inbound") != "inbound":
                continue
            self.last_media = time.monotonic()
            try:
                frame = base64.b64decode(ev["media"]["payload"], validate=True)
            except (KeyError, binascii.Error):
                continue
            acc = np.concatenate((acc, up.feed(mulaw_decode(frame))))
            while len(acc) >= DOT_CHUNK:
                await dot.send(acc[:DOT_CHUNK].astype("<i2").tobytes())
                acc = acc[DOT_CHUNK:]

    async def _from_dot(self, ws, dot, stream_sid):
        down, out, live_up = Downsampler24to8(), b"", False
        async for raw in dot:
            if isinstance(raw, bytes):
                continue   # classic-mode WAV replies are not used in a phone call (Live only)
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            kind = m.get("type")
            if kind == "live_ok":
                live_up = True
            elif kind == "live_error" and not live_up:
                log("Live did not start: %s" % str(m.get("text", ""))[:80])
                return
            elif kind == "live_audio":
                try:
                    pcm = base64.b64decode(m.get("b64", ""))
                except (binascii.Error, ValueError):
                    continue
                out += mulaw_encode(down.feed(pcm))
                while len(out) >= TW_FRAME:
                    frame, out = out[:TW_FRAME], out[TW_FRAME:]
                    await ws.send(json.dumps({"event": "media", "streamSid": stream_sid,
                                              "media": {"payload": base64.b64encode(frame).decode()}}))
            elif kind == "live_stop":   # barge-in: flush what Twilio still holds
                out = b""
                down.reset()
                await ws.send(json.dumps({"event": "clear", "streamSid": stream_sid}))
            # everything else is page UI (status, transcript, ...): ignored


async def serve(bridge, host="127.0.0.1", port=8772):
    from websockets.asyncio.server import serve as ws_serve
    return await ws_serve(bridge.handler, host, port, process_request=bridge.process_request,
                          max_size=1024 * 1024, ping_interval=20, ping_timeout=60)


def _machine_fn():
    if not config.get("callback_on_reply"):
        return None
    import callback
    return callback.on_machine


def from_config():
    try:
        pin = open(PIN_FILE).read().strip()
    except OSError:
        raise SystemExit("pin.txt is missing: start the Dot service once first")
    return Bridge(pin, config.get("twilio_path_secret"), config.get("twilio_allowed_callers"),
                  config.get("twilio_public_host"), allowed_login=os.environ.get("VOICE_ALLOWED_LOGIN", ""),
                  fallback_voice_url=config.get("twilio_fallback_voice_url") or "",
                  stale_s=int(config.get("twilio_stale_s")) if config.get("twilio_reaper") else 0,
                  auth_mode=config.get("twilio_auth"), greeting_name=str(config.get("owner")).split()[0] if config.get("owner") else "",
                  dot_number=config.get("twilio_number") or "", machine_fn=_machine_fn())


async def run_forever():
    bridge = from_config()
    server = await serve(bridge, "127.0.0.1", int(config.get("twilio_port")))
    log("listening on 127.0.0.1:%s" % config.get("twilio_port"))
    await server.serve_forever()


if __name__ == "__main__":
    if "--run" not in sys.argv:
        raise SystemExit("usage: python3 twilio_bridge.py --run")
    try:
        asyncio.run(run_forever())
    except ValueError as e:
        raise SystemExit(str(e))
