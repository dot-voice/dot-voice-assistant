"""Download the speaker-embedding model used by voiceprint.py into models/ (gitignored) and check its sha256.

    python3 fetch_models.py

Source: official sherpa-onnx release "speaker-recongition-models" (the tag is spelled that way upstream).
Model: NVIDIA NeMo TitaNet small, CC-BY-4.0.
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
MODELS = {"nemo_en_titanet_small.onnx": "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, want, directory=os.path.join(HERE, "models")):
    os.makedirs(directory, exist_ok=True)
    dst = os.path.join(directory, name)
    if os.path.exists(dst) and sha256(dst) == want:
        return dst
    tmp = dst + ".part"
    urllib.request.urlretrieve(BASE + name, tmp)
    got = sha256(tmp)
    if got != want:
        os.remove(tmp)
        raise SystemExit("sha256 mismatch for %s: got %s, want %s" % (name, got, want))
    os.replace(tmp, dst)
    return dst


if __name__ == "__main__":
    for n, s in MODELS.items():
        print("ok", fetch(n, s))
    sys.exit(0)
