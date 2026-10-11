#!/usr/bin/env python3
"""Speech-to-text worker process. Reads length-prefixed 16 kHz int16 PCM on stdin, answers one JSON line per clip on stdout.
Runs the model on this process's main thread (MLX requires that). Protocol lines go to the real stdout; anything the
libraries print is sent to stderr instead."""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

proto = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)


def say(text: str):
    proto.write((text + "\n").encode("utf-8"))


def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    if os.environ.get("VOICE_STT_FAKE"):
        from stt import FakeSpeechToText
        engine = FakeSpeechToText(os.environ["VOICE_STT_FAKE"])
    else:
        from stt import WhisperEngine
        engine = WhisperEngine()
        engine.warm()   # download and load the model before saying READY
    say("READY")
    while True:
        head = read_exact(4)
        if head is None:
            return
        pcm = read_exact(struct.unpack("<I", head)[0])
        if pcm is None:
            return
        try:
            say(json.dumps({"text": engine.transcribe(pcm)}))
        except Exception as e:
            say(json.dumps({"error": "%s: %s" % (type(e).__name__, e)}))


if __name__ == "__main__":
    main()
