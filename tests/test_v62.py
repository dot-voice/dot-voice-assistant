"""v62: R49 voicemail pickup (P21 phase 2). Fake HTTP, temp dirs, neutral names, no network."""
import email.utils
import json
import os
import sys
import tempfile
import time
import unittest
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import config  # noqa: E402
import voicemail  # noqa: E402

CREDS = {"account_sid": "ACtest", "key_sid": "SKtest", "key_secret": "secret-test-value"}
OURS = "+15550001111"
ALLOWED = "+15550002222"
OTHER = "+15550003333"


def stamp(t):
    return email.utils.formatdate(t, usegmt=True)


def make_wav(path, rate=8000, secs=0.5):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x10\x00" * int(rate * secs))


class Fake:
    def __init__(self, tmp, recs, calls):
        self.tmp, self.recs, self.calls, self.urls, self.auths = tmp, recs, calls, [], []

    def get(self, url, auth, timeout=30):
        self.urls.append(url)
        self.auths.append(auth)
        if "/Recordings.json" in url:
            return json.dumps({"recordings": self.recs}).encode()
        if "/Calls/" in url:
            sid = url.split("/Calls/")[1].split(".json")[0]
            return json.dumps(self.calls[sid]).encode()
        if url.endswith(".wav"):
            p = os.path.join(self.tmp, "src.wav")
            make_wav(p)
            with open(p, "rb") as fh:
                return fh.read()
        raise AssertionError(url)


class TestVoicemail(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.now = time.time()
        self.cfg = {"voicemail_pickup": True, "twilio_number": OURS, "twilio_allowed_callers": [ALLOWED],
                    "call_recap_telegram_chat_id": "1"}.get
        self.sent, self.posted = [], []
        self.log = os.path.join(self.d, "turns.log")

    def tearDown(self):
        self.tmp.cleanup()

    def rec(self, sid, call, ago=60, dur="12"):
        return {"sid": sid, "call_sid": call, "duration": dur, "date_created": stamp(self.now - ago)}

    def run_tick(self, fake, stt=lambda pcm: "please call me back", cfg=None, tg=True):
        def telegram(text, chat):
            self.sent.append(text)
            return tg

        def post(role, text, folder=None):
            self.posted.append((role, text))
            return "x"
        return voicemail.tick(stt, now=self.now, log_path=self.log, cfg=cfg or self.cfg, state_path=os.path.join(self.d, "st.json"),
                              audio_dir=os.path.join(self.d, "vm"), creds_fn=lambda: CREDS, get=fake.get, telegram=telegram, post=post)

    def test_off_by_default_does_nothing(self):
        self.assertFalse(config.DEFAULTS["voicemail_pickup"])
        f = Fake(self.d, [self.rec("RE1", "CA1")], {})
        self.assertIsNone(self.run_tick(f, cfg=lambda k: False if k == "voicemail_pickup" else self.cfg(k)))
        self.assertEqual(f.urls, [])

    def test_new_voicemail_goes_to_telegram_and_the_update_queue_and_audio_is_kept(self):
        f = Fake(self.d, [self.rec("RE1", "CA1")], {"CA1": {"from": ALLOWED, "to": OURS}})
        self.assertEqual(self.run_tick(f), 1)
        self.assertEqual(self.sent, ["Voicemail while Dot was offline, 12 s, from you: please call me back"])
        self.assertEqual(self.posted[0][0], "VOICEMAIL")
        self.assertIn("12 seconds", self.posted[0][1])
        self.assertIn("please call me back", self.posted[0][1])
        self.assertTrue(os.path.exists(os.path.join(self.d, "vm", "RE1.wav")))

    def test_unknown_caller_shows_last_four_digits_only(self):
        f = Fake(self.d, [self.rec("RE1", "CA1")], {"CA1": {"from": OTHER, "to": OURS}})
        self.run_tick(f)
        self.assertIn("from number ending 3333:", self.sent[0])
        self.assertNotIn("+1555", self.sent[0] + self.posted[0][1])

    def test_only_calls_to_our_number_and_each_recording_once(self):
        f = Fake(self.d, [self.rec("RE1", "CA1"), self.rec("RE2", "CA2")],
                 {"CA1": {"from": OTHER, "to": "+15559998888"}, "CA2": {"from": OTHER, "to": OURS}})
        self.assertEqual(self.run_tick(f), 1)
        self.assertEqual(self.run_tick(f), 0)
        self.assertEqual(len(self.sent), 1)

    def test_old_recordings_before_the_last_check_are_ignored(self):
        f = Fake(self.d, [self.rec("RE1", "CA1", ago=3 * 86400)], {"CA1": {"from": OTHER, "to": OURS}})
        self.assertEqual(self.run_tick(f), 0)

    def test_no_transcript_says_could_not_transcribe(self):
        f = Fake(self.d, [self.rec("RE1", "CA1")], {"CA1": {"from": OTHER, "to": OURS}})

        def boom(pcm):
            raise RuntimeError("no engine")
        self.run_tick(f, stt=boom)
        self.assertIn("could not transcribe", self.sent[0])
        self.assertNotIn("It says", self.posted[0][1])

    def test_transcript_is_capped_at_500_characters(self):
        f = Fake(self.d, [self.rec("RE1", "CA1")], {"CA1": {"from": OTHER, "to": OURS}})
        self.run_tick(f, stt=lambda pcm: "word " * 400)
        self.assertLessEqual(len(self.sent[0].split(": ", 1)[1]), 500)

    def test_failed_telegram_still_queues_once_and_does_not_repeat(self):
        f = Fake(self.d, [self.rec("RE1", "CA1")], {"CA1": {"from": OTHER, "to": OURS}})
        self.run_tick(f, tg=False)
        self.run_tick(f)
        self.assertEqual(len(self.posted), 1)

    def test_get_only_basic_auth_and_no_secrets_or_numbers_in_the_log(self):
        f = Fake(self.d, [self.rec("RE1", "CA1")], {"CA1": {"from": OTHER, "to": OURS}})
        self.run_tick(f)
        self.assertTrue(all(a.startswith("Basic ") for a in f.auths))
        self.assertTrue(all("DateCreated%3E=" in u or "/Calls/" in u or u.endswith(".wav") for u in f.urls))
        log = open(self.log).read()
        self.assertRegex(log, r"voicemail: n=1 ms=\d+")
        for bad in ("secret-test-value", "SKtest", "5550", "call me"):
            self.assertNotIn(bad, log)

    def test_never_raises_on_bad_json_or_network_failure(self):
        class Bad:
            def get(self, *a, **k):
                raise OSError("down")
        self.assertEqual(self.run_tick(Bad()), 0)
        self.assertIn("voicemail: error OSError", open(self.log).read())

    def test_wav_8k_becomes_16k_pcm(self):
        p = os.path.join(self.d, "a.wav")
        make_wav(p, rate=8000, secs=0.5)
        self.assertEqual(len(voicemail.wav_to_pcm16k(p)), 2 * 8000)

    def test_declared_and_loop_started(self):
        self.assertEqual(config.DEFAULTS["voicemail_every_s"], 600)
        src = open(os.path.join(os.path.dirname(HERE), "server.py")).read()
        self.assertIn("asyncio.create_task(voicemail_loop(svc))", src)


if __name__ == "__main__":
    unittest.main()
