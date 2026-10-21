"""R49 voicemail pickup (P21 phase 2): when the Mac was offline, the relay answered Dot's Twilio number with Say + Record and
Twilio kept the recording. This module collects those recordings once the Mac is back: it lists new Twilio Recordings for calls
to twilio_number, downloads each WAV under logs/voicemail/, transcribes it locally, sends the owner a Telegram text and queues
a short spoken update so Dot mentions it on the next call.

Read-only toward Twilio (GET only, nothing is ever deleted there). Off unless config voicemail_pickup is true. Nothing raises;
the log line carries counts and milliseconds only: no text, no phone numbers, no credentials."""
import array
import base64
import email.utils
import json
import os
import time
import urllib.parse
import urllib.request
import wave

import callback
import callrecap
import config
import updates

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "logs", "voicemail_state.json")
AUDIO_DIR = os.path.join(HERE, "logs", "voicemail")
API = "https://api.twilio.com"
FIRST_LOOKBACK_S = 24 * 3600
KEEP_DONE = 300
MAX_TRANSCRIPT = 500
MAX_PER_TICK = 10
NO_TEXT = "could not transcribe"


def _log(message, log_path=None):
    try:
        from diag import log_line
        log_line("voicemail: " + message, log_path)
    except Exception:
        pass


# ------------------------------------------------------------------ state
def load_state(path=STATE_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        if isinstance(st, dict):
            done = st.get("done") if isinstance(st.get("done"), dict) else {}
            clean = {str(k): v for k, v in done.items() if isinstance(v, dict)}
            since = st.get("since") if isinstance(st.get("since"), (int, float)) else None
            return {"since": since, "done": clean}
    except (OSError, ValueError):
        pass
    return {"since": None, "done": {}}


def save_state(st, path=STATE_FILE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = st.get("done", {})
        if len(d) > KEEP_DONE:
            for k in sorted(d, key=lambda k: d[k].get("at", 0))[:len(d) - KEEP_DONE]:
                del d[k]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        return False


# ------------------------------------------------------------------ Twilio (GET only)
def http_get(url, auth, timeout=30, urlopen=urllib.request.urlopen):
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", auth)
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _auth(creds):
    return "Basic " + base64.b64encode(("%s:%s" % (creds["key_sid"], creds["key_secret"])).encode()).decode()


def _epoch(rfc2822):
    try:
        return email.utils.parsedate_to_datetime(str(rfc2822)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def list_recordings(creds, since, get=http_get):
    """Recording dicts created on or after the day of `since` (Twilio filters by date only), newest first."""
    day = time.strftime("%Y-%m-%d", time.gmtime(since))
    q = urllib.parse.urlencode({"DateCreated>": day, "PageSize": 100})
    url = "%s/2010-04-01/Accounts/%s/Recordings.json?%s" % (API, creds["account_sid"], q)
    body = json.loads(get(url, _auth(creds)).decode("utf-8", "replace"))
    return [r for r in (body.get("recordings") or []) if isinstance(r, dict)]


def call_parties(creds, call_sid, get=http_get):
    """(from, to) of the call a recording belongs to."""
    url = "%s/2010-04-01/Accounts/%s/Calls/%s.json" % (API, creds["account_sid"], urllib.parse.quote(str(call_sid)))
    body = json.loads(get(url, _auth(creds)).decode("utf-8", "replace"))
    return str(body.get("from") or ""), str(body.get("to") or "")


def download_wav(creds, rec_sid, folder=AUDIO_DIR, get=http_get):
    """Save the recording WAV under folder and return its path."""
    url = "%s/2010-04-01/Accounts/%s/Recordings/%s.wav" % (API, creds["account_sid"], urllib.parse.quote(str(rec_sid)))
    data = get(url, _auth(creds), 120)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s.wav" % "".join(c for c in str(rec_sid) if c.isalnum()))
    with open(path, "wb") as f:
        f.write(data)
    return path


# ------------------------------------------------------------------ audio and text
def wav_to_pcm16k(path):
    """PCM 16-bit mono 16 kHz bytes from a PCM WAV (first channel, linear resample). Raises on anything else."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("not 16-bit PCM")
        ch, rate, raw = w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
    a = array.array("h")
    a.frombytes(raw[:len(raw) - len(raw) % 2])
    if ch > 1:
        a = a[::ch]
    if rate != 16000 and rate > 0 and len(a) > 1:
        n = int(len(a) * 16000 / rate)
        out = array.array("h")
        step = rate / 16000.0
        for i in range(n):
            pos = i * step
            j = int(pos)
            frac = pos - j
            nxt = a[j + 1] if j + 1 < len(a) else a[j]
            out.append(int(a[j] * (1 - frac) + nxt * frac))
        a = out
    return a.tobytes()


def transcribe(path, stt_fn):
    """Transcript text, or NO_TEXT. Never raises."""
    try:
        text = " ".join(str(stt_fn(wav_to_pcm16k(path)) or "").split())
        return text[:MAX_TRANSCRIPT] if text else NO_TEXT
    except Exception:
        return NO_TEXT


def who(from_number, allowed):
    """'you' for an allowed caller, else only the last 4 digits."""
    digits = "".join(c for c in str(from_number) if c.isdigit())
    for n in allowed or []:
        if digits and "".join(c for c in str(n) if c.isdigit()) == digits:
            return "you"
    return ("number ending %s" % digits[-4:]) if len(digits) >= 4 else "an unknown number"


def telegram_text(secs, caller, transcript):
    return "Voicemail while Dot was offline, %s s, from %s: %s" % (secs, caller, transcript[:MAX_TRANSCRIPT])


def spoken_text(secs, caller, transcript):
    body = "" if transcript == NO_TEXT else " It says: " + transcript[:160]
    return "A voicemail came in while I was offline, %s seconds, from %s.%s" % (secs, caller, body)


# ------------------------------------------------------------------ one pass
def tick(stt_fn, now=None, log_path=None, cfg=config.get, state_path=STATE_FILE, audio_dir=AUDIO_DIR, creds_fn=None, get=http_get,
         telegram=callrecap.send_telegram, post=updates.post, updates_folder=None):
    """Collect new voicemails once. Returns the number handled, or None when off. Never raises."""
    t0 = time.monotonic()
    n = 0
    try:
        if not cfg("voicemail_pickup"):
            return None
        number = str(cfg("twilio_number") or "")
        if not number:
            return None
        now = time.time() if now is None else now
        st = load_state(state_path)
        since = st["since"] if st["since"] is not None else now - FIRST_LOOKBACK_S
        creds = (creds_fn or callback.twilio_credentials)()
        allowed = list(cfg("twilio_allowed_callers") or [])
        recs = list_recordings(creds, since - 3600, get)   # an hour of slack: Twilio filters by day only
        fresh = [r for r in recs if str(r.get("sid", "")) not in st["done"] and _epoch(r.get("date_created")) >= since]
        fresh.sort(key=lambda r: _epoch(r.get("date_created")))
        newest = since
        for r in fresh[:MAX_PER_TICK]:
            sid = str(r.get("sid", ""))
            frm, to = call_parties(creds, r.get("call_sid", ""), get)
            created = _epoch(r.get("date_created"))
            if "".join(c for c in to if c.isdigit()) != "".join(c for c in number if c.isdigit()):
                st["done"][sid] = {"at": now, "skip": True}
                newest = max(newest, created)
                continue
            flags = {"at": now}
            path = download_wav(creds, sid, audio_dir, get)
            text = transcribe(path, stt_fn)
            try:
                secs = int(float(r.get("duration") or 0))
            except (TypeError, ValueError):
                secs = 0
            caller = who(frm, allowed)
            flags["tg"] = bool(telegram(telegram_text(secs, caller, text), cfg("call_recap_telegram_chat_id")))
            try:
                kw = {"folder": updates_folder} if updates_folder else {}
                post("VOICEMAIL", spoken_text(secs, caller, text), **kw)
                flags["q"] = True
            except Exception:
                flags["q"] = False
            st["done"][sid] = flags   # recorded even if Telegram failed: the spoken update still carries it, no repeat texts
            newest = max(newest, created)
            n += 1
        st["since"] = newest if (fresh or st["since"] is not None) else now - 3600
        save_state(st, state_path)
    except Exception as e:
        _log("error %s" % type(e).__name__, log_path)
        return n
    _log("n=%d ms=%d" % (n, int((time.monotonic() - t0) * 1000)), log_path)
    return n
