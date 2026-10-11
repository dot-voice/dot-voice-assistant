"""Text to speech: Qwen streaming voice by default (a second Qwen request if the first is slow); the macOS 'say' voice
is available as a local fallback with VOICE_TTS_ENGINE=qwen-mac or when no Qwen key is set."""
import base64
import json
import os
import queue
import threading
import time
import struct
import subprocess
import sys
import tempfile
import urllib.request

from audio import has_chinese


class TextToSpeech:
    def speak(self, text: str) -> bytes:  # pragma: no cover - interface
        """Return a complete WAV file (bytes) for one short piece of text."""
        raise NotImplementedError

    def speak_stream(self, text):
        """Yield WAV files that play back to back. Engines that can stream yield the first part early."""
        yield self.speak(text)


def silent_wav(seconds=0.2, rate=16000) -> bytes:
    return pcm_wav(b"\x00\x00" * int(seconds * rate), rate)


def pcm_wav(data: bytes, rate=24000) -> bytes:
    """16-bit mono PCM -> a complete WAV the phone can decode on its own."""
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(data)) + data


class FakeTextToSpeech(TextToSpeech):
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)
        return silent_wav()


class MacSayTextToSpeech(TextToSpeech):
    def __init__(self, zh_voice=None, en_voice=None, rate_wpm=None):
        self.zh_voice = zh_voice or os.environ.get("VOICE_TTS_ZH", "Tingting")
        self.en_voice = en_voice or os.environ.get("VOICE_TTS_EN", "Samantha")
        self.rate_wpm = rate_wpm or int(os.environ.get("VOICE_TTS_RATE", "190"))

    def speak(self, text):
        voice = self.zh_voice if has_chinese(text) else self.en_voice
        with tempfile.TemporaryDirectory(prefix="voice-tts-") as d:
            aiff, wav = os.path.join(d, "o.aiff"), os.path.join(d, "o.wav")
            for v in (voice, None):  # fall back to the system voice if the named one is missing
                cmd = ["say", "-r", str(self.rate_wpm), "-o", aiff]
                if v:
                    cmd += ["-v", v]
                p = subprocess.run(cmd + ["--", text], capture_output=True, timeout=30)
                if p.returncode == 0 and os.path.exists(aiff):
                    break
            else:
                raise RuntimeError("say failed")
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff, wav], check=True, capture_output=True, timeout=30)
            with open(wav, "rb") as f:
                return f.read()


# ----------------------------------------------------------------------------- Qwen voice
# Only the TEXT OF THE REPLY goes to Qwen, to be turned into speech. Speech recognition stays local.
QWEN_BASES = ["https://dashscope-intl.aliyuncs.com/api/v1", "https://dashscope.aliyuncs.com/api/v1"]
QWEN_KEYCHAIN = "voice-call-qwen"
QWEN_ENV_NAMES = ("QWEN_API_KEY", "DASHSCOPE_API_KEY")
QWEN_OTHER_KEYCHAIN = ("dashscope", "DASHSCOPE_API_KEY", "qwen-api-key", "QWEN_API_KEY", "dashscope-api-key")


def _keychain(service: str) -> str:
    if sys.platform != "darwin" or not os.path.exists("/usr/bin/security"):
        return ""
    try:
        p = subprocess.run(["/usr/bin/security", "find-generic-password", "-s", service, "-w"],
                           capture_output=True, text=True, timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return (p.stdout or "").strip() if p.returncode == 0 else ""


def qwen_key() -> str:
    """The key the service uses: the voice-call Keychain item first, then environment variables."""
    return _keychain(QWEN_KEYCHAIN) or next((os.environ[n] for n in QWEN_ENV_NAMES if os.environ.get(n)), "")


def find_and_store_qwen_key():
    """Look for the saved key (environment of this shell, then common Keychain names) and copy it into the Keychain item
    this service reads. Returns a short human-readable result. The key itself is never printed."""
    where, key = "", ""
    for n in QWEN_ENV_NAMES:
        if os.environ.get(n):
            where, key = "environment variable " + n, os.environ[n]
            break
    if not key:
        for svc in QWEN_OTHER_KEYCHAIN:
            k = _keychain(svc)
            if k:
                where, key = "Keychain item " + svc, k
                break
    if not key:
        return "NOT FOUND: no Qwen key in this shell's environment or the usual Keychain names."
    subprocess.run(["/usr/bin/security", "add-generic-password", "-U", "-s", QWEN_KEYCHAIN, "-a",
                    os.environ.get("USER", "voice"), "-w", key], capture_output=True, timeout=10)
    ok = _keychain(QWEN_KEYCHAIN) == key
    return ("FOUND in %s; saved to Keychain item %s" % (where, QWEN_KEYCHAIN)) if ok else "FOUND but could not save to Keychain"


def _http_post_json(url, body, key, timeout=20):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_get(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _http_sse_pcm(url, body, key, timeout=None):
    """Qwen streaming (SSE): yields raw 16-bit 24 kHz mono PCM pieces as they are generated."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "X-DashScope-SSE": "enable"})
    # Beijing sometimes stalls 20 s (measured): no sound within this many seconds -> the Mac voice speaks instead.
    timeout = timeout or float(os.environ.get("VOICE_QWEN_TTS_TIMEOUT", "7"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = json.loads(line[5:])
            if d.get("code"):
                raise RuntimeError("Qwen stream error: %s" % d.get("code"))
            data = ((d.get("output") or {}).get("audio") or {}).get("data")
            if data:
                yield strip_wav_header(base64.b64decode(data))


def strip_wav_header(b: bytes) -> bytes:
    """Qwen's first streamed piece starts with a WAV header. Played as sound it was a loud click at the start of every
    sentence (the 'first second jitter'). Keep only what follows the 'data' chunk header."""
    if b[:4] != b"RIFF":
        return b
    i = b.find(b"data", 12)
    return b[i + 8:] if i >= 0 else b[44:]


class QwenTextToSpeech(TextToSpeech):
    def __init__(self, key=None, voice=None, model=None, post=_http_post_json, get=_http_get, bases=None, sse=_http_sse_pcm):
        self.sse = sse
        self.key = key or qwen_key()
        self.voice = voice or os.environ.get("VOICE_QWEN_VOICE", "Cherry")
        self.model = model or os.environ.get("VOICE_QWEN_MODEL", "qwen3-tts-flash")
        self.post, self.get = post, get
        env_base = os.environ.get("VOICE_QWEN_BASE")
        self.bases = bases or ([env_base] if env_base else list(QWEN_BASES))
        self._good_base = None

    def available(self) -> bool:
        return bool(self.key)

    def speak(self, text):
        if not self.key:
            raise RuntimeError("no Qwen key")
        body = {"model": self.model,
                "input": {"text": text, "voice": self.voice, "language_type": "Chinese" if has_chinese(text) else "English"}}
        last = None
        for base in ([self._good_base] if self._good_base else self.bases):
            try:
                res = self.post(base.rstrip("/") + "/services/aigc/multimodal-generation/generation", body, self.key)
                out = res.get("output", res)
                url = (out.get("audio") or {}).get("url")
                if not url:
                    raise RuntimeError("Qwen reply had no audio url")
                self._good_base = base
                return self.get(url)
            except Exception as e:  # try the other region, then give up (the caller falls back to the Mac voice)
                last = e
        raise RuntimeError("Qwen speech failed: %s" % type(last).__name__)

    def _hedged(self, url, body, hedge_s=None, deadline_s=None):
        """Beijing often takes 4-9 s to start one request and ~2 s for the next (5 of 8 slow, measured),
        which switched the call to the Mac voice. If the first request has no
        sound after hedge_s, a second one starts; the first to make sound wins. Yields PCM pieces of the winner."""
        hedge_s = hedge_s if hedge_s is not None else float(os.environ.get("VOICE_QWEN_HEDGE_S", "2"))
        deadline_s = deadline_s if deadline_s is not None else float(os.environ.get("VOICE_QWEN_TTS_DEADLINE", "7"))
        firsts = queue.Queue()

        def attempt(n):
            q = queue.Queue()
            try:
                for i, pcm in enumerate(self.sse(url, body, self.key)):
                    if i == 0:
                        firsts.put((n, q))
                    q.put(pcm)
                q.put(None)
            except Exception as e:
                firsts.put((n, e))
                q.put(e)
        t0 = time.monotonic()
        threading.Thread(target=attempt, args=(1,), daemon=True).start()
        started, errors, winner = 1, [], None
        while winner is None:
            elapsed = time.monotonic() - t0
            left = deadline_s - elapsed
            if left <= 0 or (started == 2 and len(errors) >= 2):   # out of time, or both requests failed
                break
            wait = min(left, hedge_s - elapsed) if started == 1 else left
            try:
                n, got = firsts.get(timeout=max(0.05, wait))
            except queue.Empty:
                if started == 1:
                    started = 2
                    threading.Thread(target=attempt, args=(2,), daemon=True).start()
                continue
            if isinstance(got, Exception):
                errors.append(got)
                if started == 1:   # failed fast: try again at once
                    started = 2
                    threading.Thread(target=attempt, args=(2,), daemon=True).start()
                continue
            winner = got
        if winner is None:
            raise errors[-1] if errors else TimeoutError("Qwen voice had no sound in %.0f s" % deadline_s)
        while True:
            pcm = winner.get(timeout=deadline_s)
            if pcm is None:
                return
            if isinstance(pcm, Exception):
                raise pcm
            yield pcm

    def speak_stream(self, text, first_s=0.9, chunk_s=0.8, rate=24000):
        """Streaming: the first ~0.9 s of sound goes out as soon as Qwen makes it (about 2 s instead of ~5 s for
        generate + download from Beijing). Qwen sends 0.32 s pieces; a 0.3 s first part played alone and the next part
        arrived just as it ended, so the start sounded jittery. After ~1 s Qwen runs far ahead of playback."""
        if not self.key:
            raise RuntimeError("no Qwen key")
        body = {"model": self.model,
                "input": {"text": text, "voice": self.voice, "language_type": "Chinese" if has_chinese(text) else "English"}}
        last = None
        for base in ([self._good_base] if self._good_base else self.bases):
            buf, sent = b"", 0
            try:
                for pcm in self._hedged(base.rstrip("/") + "/services/aigc/multimodal-generation/generation", body):
                    buf += pcm
                    if len(buf) >= int((first_s if not sent else chunk_s) * rate) * 2:
                        cut = len(buf) - len(buf) % 2
                        yield pcm_wav(buf[:cut], rate)
                        buf, sent = buf[cut:], sent + 1
                        self._good_base = base
                if len(buf) >= 2:
                    yield pcm_wav(buf[:len(buf) - len(buf) % 2], rate)
                    sent += 1
                if sent:
                    self._good_base = base
                    return
                raise RuntimeError("Qwen stream had no audio")
            except Exception as e:
                if sent:   # part of the reply already played: do not repeat it in another voice
                    raise
                last = e
        raise RuntimeError("Qwen speech failed: %s" % type(last).__name__)


class ChainTextToSpeech(TextToSpeech):
    """Qwen first; if it fails for any reason the Mac voice speaks, so a call never goes silent."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.last_error = ""
        self.fallbacks = 0   # how many sentences the Mac voice spoke; the call shows it so the switch is never a mystery

    def speak(self, text):
        try:
            return self.primary.speak(text)
        except Exception as e:
            self.last_error = type(e).__name__
            return self.fallback.speak(text)

    def speak_stream(self, text):
        sent = False
        try:
            for wav in self.primary.speak_stream(text):
                sent = True
                yield wav
        except Exception as e:
            self.last_error = type(e).__name__
            if not sent:
                self.fallbacks += 1
                yield self.fallback.speak(text)


def default_tts():
    mac = MacSayTextToSpeech()
    if os.environ.get("VOICE_TTS_ENGINE", "qwen") == "mac":
        return mac
    q = QwenTextToSpeech()
    if not q.available():
        return mac
    # Qwen-only mode: a failed Qwen sentence gets a second Qwen try, never the
    # Mac voice. VOICE_TTS_ENGINE=qwen-mac brings the Mac voice back as the backup.
    return ChainTextToSpeech(q, mac if os.environ.get("VOICE_TTS_ENGINE") == "qwen-mac" else q)
