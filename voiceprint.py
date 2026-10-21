"""P25 phase 1 (R42): a local voiceprint, LOG-ONLY. It scores each turn against the owner's profile and writes one line to
turns.log. It never blocks a reply and never changes what Dot does (voiceprint_enforce is reserved for phase 2).

Model: sherpa-onnx speaker embedding (nemo_en_titanet_small.onnx, models/, see fetch_models.py), CPU only, no cloud.
Profiles: data/voiceprint/<channel>.npz, one per channel ("phone" = Twilio bridge audio, "web" = call page audio), because
8 kHz phone audio and page audio sound different. A profile holds the mean embedding and up to 20 utterance embeddings; no
audio and no words are ever stored.
"""
import os
import threading

import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "nemo_en_titanet_small.onnx")
DATA_DIR = os.path.join(HERE, "data", "voiceprint")
CHANNELS = ("phone", "web")
RATE = 16000
MIN_SCORE_S = 1.5      # a turn needs this much speech to be scored
MIN_ENROLL_S = 2.0     # and this much to be enrolled
ENROLL_TARGET_S = 60.0  # speech needed before scoring starts
MAX_UTTS = 20
FRAME = 480            # 30 ms at 16 kHz

_lock = threading.Lock()          # the extractor and the profile files are used by one thread at a time
_embedder = None                  # test hook: set_embedder(fn) replaces the real model
_extractor = None


def enabled():
    return bool(config.get("voiceprint_log"))


def set_embedder(fn):
    """Tests: fn(float32 samples at 16 kHz) -> vector. None restores the real model."""
    global _embedder
    _embedder = fn


def _real_embed(x):
    global _extractor
    if _extractor is None:
        import sherpa_onnx
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL_PATH, num_threads=1))
    st = _extractor.create_stream()
    st.accept_waveform(RATE, x)
    st.input_finished()
    return np.array(_extractor.compute(st), dtype=np.float32)


def embed(pcm16_16k):
    """pcm16 bytes (or int16 array) at 16 kHz -> unit-length embedding vector."""
    a = np.frombuffer(pcm16_16k, dtype="<i2") if isinstance(pcm16_16k, (bytes, bytearray)) else np.asarray(pcm16_16k)
    x = a.astype(np.float32) / 32768.0
    v = np.asarray((_embedder or _real_embed)(x), dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def speech_only(pcm):
    """Keep the voiced frames of a turn (adaptive energy detector: well above the quietest 10 % of frames, 150 ms hangover).
    Returns (pcm16 bytes of speech, seconds)."""
    a = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2")
    n = len(a) // FRAME
    if n == 0:
        return b"", 0.0
    fr = a[:n * FRAME].astype(np.float32).reshape(n, FRAME)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    floor, top = float(np.percentile(rms, 10)), float(np.percentile(rms, 95))
    voiced = rms > max(250.0, min(3.0 * floor, 0.3 * top))   # the cap keeps a turn that is all speech from reading as silence
    keep = voiced.copy()
    for i in np.flatnonzero(voiced):
        keep[i:i + 5] = True
    if not keep.any():
        return b"", 0.0
    out = fr[keep].astype("<i2")
    return out.tobytes(), len(out) * FRAME / float(RATE)


class Profile:
    def __init__(self, channel, directory=None):
        self.channel = channel
        self.path = os.path.join(directory or DATA_DIR, channel + ".npz")
        self.total = None      # sum of all enrolled embeddings
        self.n = 0
        self.utts = []
        self.secs = 0.0
        self.load()

    def load(self):
        try:
            with np.load(self.path) as z:
                self.total, self.n, self.secs = z["total"], int(z["n"]), float(z["secs"])
                self.utts = list(z["utts"])
        except (OSError, ValueError, KeyError):
            pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, total=self.total, n=self.n, secs=self.secs, utts=np.array(self.utts))
        os.replace(tmp, self.path)

    def mean(self):
        if self.total is None:
            return None
        norm = float(np.linalg.norm(self.total))
        return self.total / norm if norm > 0 else None

    def add(self, vec, speech_s):
        self.total = vec.copy() if self.total is None else self.total + vec
        self.n += 1
        self.secs += speech_s
        self.utts = (self.utts + [vec])[-MAX_UTTS:]
        self.save()


def process_turn(channel, pcm, signed_in, directory=None):
    """One turn: returns the turns.log line, or None when there is nothing to say. Blocking (model run): call it off the
    event loop. Never raises on bad audio."""
    if channel not in CHANNELS or not pcm:
        return None
    sp, secs = speech_only(pcm)
    if secs < MIN_SCORE_S:
        return None
    with _lock:
        prof = Profile(channel, directory)
        if prof.secs < ENROLL_TARGET_S:
            if not signed_in or secs < MIN_ENROLL_S:
                return None
            prof.add(embed(sp), secs)
            return "voiceprint: channel=%s enrolling speech_s=%.1f enrolled_s=%d" % (channel, secs, prof.secs)
        mean = prof.mean()
        if mean is None:
            return None
        score = float(np.dot(embed(sp), mean))
        return "voiceprint: channel=%s score=%.2f speech_s=%.1f enrolled_s=%d" % (channel, score, secs, prof.secs)
