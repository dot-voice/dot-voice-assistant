"""Small audio helpers. Pure Python, no third-party packages."""
import array
import math

SAMPLE_RATE = 16000
MIN_SECONDS = 0.45          # shorter than this is a cough or a click, not speech
MIN_RMS = 90.0              # int16 units; below this the clip is silence

# Phrases Whisper invents on silence or noise. Matched after lower-casing and stripping punctuation.
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "you", "bye", "the end",
    "subtitles by the amara.org community", "please subscribe",
    "\u8c22\u8c22\u89c2\u770b", "\u8bf7\u4e0d\u540d\u70b9\u8d5e \u8ba2\u9605 \u8f6c\u53d1 \u6253\u8d4f\u652f\u6301\u660e\u955c\u4e0e\u70b9\u70b9\u680f\u76ee",
    "\u5b57\u5e55\u7531 amara.org \u793e\u533a\u63d0\u4f9b",
}


def pcm_seconds(pcm: bytes, rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / 2.0 / rate


def pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = sum(s * s for s in samples)
    return math.sqrt(total / len(samples))


def worth_transcribing(pcm: bytes, rate: int = SAMPLE_RATE) -> bool:
    return pcm_seconds(pcm, rate) >= MIN_SECONDS and pcm_rms(pcm) >= MIN_RMS


def _norm(text: str) -> str:
    keep = []
    for ch in text.lower():
        keep.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(keep).split())


def is_hallucination(text: str) -> bool:
    n = _norm(text)
    if not n:
        return True
    return n in {_norm(h) for h in HALLUCINATIONS}


def split_sentences(text: str):
    """Split a reply into short speakable pieces so the first one can play early."""
    out, buf = [], ""
    for ch in text.strip():
        buf += ch
        if ch in "\u3002\uff01\uff1f.!?\n" and len(buf.strip()) > 1:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    merged = []
    for piece in out:
        if merged and len(piece) < 3:
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)
