"""Local speech to text. No audio and no transcript ever leaves this Mac."""
import array
import os
import sys

import config
from audio import SAMPLE_RATE, is_hallucination, worth_transcribing

INITIAL_PROMPT = config.get("stt_vocabulary")


class SpeechToText:
    """Interface. transcribe() takes 16 kHz mono int16 little-endian PCM and returns text ('' for noise)."""

    def transcribe(self, pcm: bytes) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class FakeSpeechToText(SpeechToText):
    """For tests and --selftest: returns canned text, honours the same silence and noise filters."""

    def __init__(self, text="hello", check_audio=True):
        self.text = text
        self.check_audio = check_audio
        self.calls = 0

    def transcribe(self, pcm: bytes) -> str:
        self.calls += 1
        if self.check_audio and not worth_transcribing(pcm):
            return ""
        return "" if is_hallucination(self.text) else self.text


def _to_float(pcm: bytes):
    import numpy as np  # only imported when the real model runs
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return np.array(samples, dtype=np.float32) / 32768.0


def _simplify(text: str) -> str:
    try:
        from opencc import OpenCC  # optional
        return OpenCC("t2s").convert(text)
    except Exception:
        return text


class WhisperEngine(SpeechToText):
    """In-process engine; it runs only inside stt_worker.py, on that process's main thread. mlx-whisper first (fast on Apple silicon), faster-whisper as the fallback. The model loads on first use."""

    def __init__(self, model=None):
        self.model = model or os.environ.get("VOICE_STT_MODEL", "")
        self._engine = None

    def _load(self):
        if self._engine:
            return self._engine
        try:
            import mlx_whisper  # noqa: F401
            repo = self.model or "mlx-community/whisper-small-mlx"   # small = about 4x faster than large-v3-turbo; set VOICE_STT_MODEL to change
            self._engine = ("mlx", repo)
        except ImportError:
            from faster_whisper import WhisperModel
            name = self.model or "small"
            self._engine = ("faster", WhisperModel(name, device="cpu", compute_type="int8"))
        return self._engine

    def warm(self):
        """Load (and on the very first run download) the model, then run it once on a short noisy clip."""
        import array
        import random
        kind, handle = self._load()
        random.seed(1)
        pcm = array.array("h", (random.randint(-3000, 3000) for _ in range(SAMPLE_RATE))).tobytes()
        audio = _to_float(pcm)
        if kind == "mlx":
            import mlx_whisper
            mlx_whisper.transcribe(audio, path_or_hf_repo=handle, task="transcribe")
        else:
            list(handle.transcribe(audio, task="transcribe")[0])

    def transcribe(self, pcm: bytes) -> str:
        if not worth_transcribing(pcm):
            return ""
        kind, handle = self._load()
        audio = _to_float(pcm)
        if kind == "mlx":
            import mlx_whisper
            res = mlx_whisper.transcribe(audio, path_or_hf_repo=handle, task="transcribe",
                                         initial_prompt=INITIAL_PROMPT, condition_on_previous_text=False)
            text = (res.get("text") or "").strip()
        else:
            segments, _info = handle.transcribe(audio, task="transcribe", initial_prompt=INITIAL_PROMPT,
                                                vad_filter=True, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments).strip()
        text = _simplify(text)
        return "" if is_hallucination(text) else text


def available_engine() -> str:
    """Name of the engine that would load, or '' if none is installed. Does not load a model."""
    for mod, name in (("mlx_whisper", "mlx-whisper"), ("faster_whisper", "faster-whisper")):
        try:
            __import__(mod)
            return name
        except ImportError:
            continue
    return ""


class WhisperSpeechToText(SpeechToText):
    """Runs the speech model in its own process (stt_worker.py). MLX aborts the whole process when its model is touched from
    a non-main thread, and the server calls speech to text from worker threads; a separate process avoids that and also
    means a model crash cannot take the call service down. The model loads once, in the background, at start."""

    FIRST_TIMEOUT_S = 180
    TIMEOUT_S = 60

    def __init__(self, worker_cmd=None, env=None):
        import threading
        self._cmd = worker_cmd or [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stt_worker.py")]
        self._env = env
        self._proc = None
        self._lock = threading.Lock()
        self._ready = False

    def start(self):
        import subprocess
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=self._env)
        self._ready = False

    def _read_line(self, timeout):
        import select
        r, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not r:
            raise TimeoutError("speech worker did not answer in %ss" % timeout)
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("speech worker exited")
        return line.decode("utf-8", "replace").strip()

    def _stop(self):
        try:
            if self._proc:
                self._proc.kill()
        except OSError:
            pass
        self._proc = None

    def transcribe(self, pcm: bytes) -> str:
        if not worth_transcribing(pcm):
            return ""
        import json
        import struct
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self.start()
                if not self._ready:
                    line = self._read_line(self.FIRST_TIMEOUT_S)
                    if line != "READY":
                        raise RuntimeError("speech worker said %r instead of READY" % line[:80])
                    self._ready = True
                self._proc.stdin.write(struct.pack("<I", len(pcm)) + pcm)
                self._proc.stdin.flush()
                reply = json.loads(self._read_line(self.TIMEOUT_S))
            except Exception:
                self._stop()   # next call starts a fresh worker
                raise
        if reply.get("error"):
            raise RuntimeError("speech worker error: " + str(reply["error"])[:300])
        return str(reply.get("text") or "")

    def warm(self):
        """Start the worker now so the model loads in the background."""
        with self._lock:
            if self._proc is None:
                self.start()
