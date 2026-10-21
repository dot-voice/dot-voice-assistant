"""v55: R42 voiceprint phase 1 (log-only scores, auto-enroll from signed-in calls). The embedding model is stubbed."""
import asyncio
import os
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import config  # noqa: E402
import gemini_live  # noqa: E402
import voiceprint  # noqa: E402

VEC = {"v": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}


def speech(secs, amp=6000):
    """Loud tone (speech stand-in) with 0.5 s of quiet noise on each side, as pcm16 bytes."""
    rng = np.random.default_rng(1)
    n = int(16000 * secs)
    t = np.arange(n) / 16000.0
    loud = amp * np.sin(2 * np.pi * 220 * t)
    quiet = rng.normal(0, 20, 8000)
    return np.concatenate([quiet, loud, quiet]).astype("<i2").tobytes()


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.calls = 0

        def fake(x):
            self.calls += 1
            return VEC["v"]
        voiceprint.set_embedder(fake)
        VEC["v"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._saved = config._VALUES.get("voiceprint_log")
        config._VALUES["voiceprint_log"] = True

    def tearDown(self):
        voiceprint.set_embedder(None)
        config._VALUES["voiceprint_log"] = self._saved

    def enroll_to_target(self, channel="phone"):
        line = None
        while True:
            line = voiceprint.process_turn(channel, speech(5.0), True, self.dir)
            if "enrolling" not in line or voiceprint.Profile(channel, self.dir).secs >= voiceprint.ENROLL_TARGET_S:
                return line


class TestVoiceprint(Base):
    def test_speech_only_measures_voiced_seconds(self):
        _, secs = voiceprint.speech_only(speech(3.0))
        self.assertAlmostEqual(secs, 3.0, delta=0.4)
        self.assertEqual(voiceprint.speech_only(b"\x00\x00" * 16000)[1], 0.0)

    def test_enrolls_until_60_s_then_logs_scores(self):
        first = voiceprint.process_turn("phone", speech(5.0), True, self.dir)
        self.assertIn("voiceprint: channel=phone enrolling", first)
        for _ in range(14):
            voiceprint.process_turn("phone", speech(5.0), True, self.dir)
        prof = voiceprint.Profile("phone", self.dir)
        self.assertGreaterEqual(prof.secs, 60.0)
        line = voiceprint.process_turn("phone", speech(3.0), True, self.dir)
        self.assertRegex(line, r"^voiceprint: channel=phone score=1\.00 speech_s=\d\.\d enrolled_s=\d+$")
        self.assertEqual(voiceprint.Profile("phone", self.dir).n, prof.n)   # scoring does not enroll

    def test_other_voice_scores_low(self):
        self.enroll_to_target()
        VEC["v"] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.assertIn("score=0.00", voiceprint.process_turn("phone", speech(3.0), True, self.dir))

    def test_unsigned_or_short_turns_never_enroll(self):
        self.assertIsNone(voiceprint.process_turn("phone", speech(5.0), False, self.dir))
        self.assertIsNone(voiceprint.process_turn("phone", speech(1.0), True, self.dir))   # under 1.5 s: not scored
        self.assertIsNone(voiceprint.process_turn("phone", speech(1.7), True, self.dir))   # 1.5-2 s: not enrolled
        self.assertEqual(self.calls, 0)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "phone.npz")))

    def test_channels_are_separate_profiles(self):
        voiceprint.process_turn("phone", speech(5.0), True, self.dir)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "phone.npz")))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "web.npz")))
        self.assertEqual(voiceprint.Profile("web", self.dir).secs, 0.0)
        self.assertIsNone(voiceprint.process_turn("other", speech(5.0), True, self.dir))

    def test_profile_keeps_at_most_20_utterances_and_survives_reload(self):
        for _ in range(25):
            voiceprint.process_turn("web", speech(2.5), True, self.dir)
            if voiceprint.Profile("web", self.dir).secs >= 60:
                break
        prof = voiceprint.Profile("web", self.dir)
        self.assertLessEqual(len(prof.utts), 20)
        self.assertAlmostEqual(float(np.linalg.norm(prof.mean())), 1.0, places=5)

    def test_log_line_has_no_audio_or_words(self):
        line = voiceprint.process_turn("phone", speech(3.0), True, self.dir)
        self.assertRegex(line, r"^voiceprint: channel=phone enrolling speech_s=\d\.\d enrolled_s=\d+$")

    def test_flag_defaults_off(self):
        self.assertIs(config.DEFAULTS["voiceprint_log"], False)
        self.assertIs(config.DEFAULTS["voiceprint_enforce"], False)


class FakeBridge(gemini_live.GeminiBridge):
    def __init__(self, log):
        self.voice_channel, self.signed_in, self._vp_tasks = "phone", True, set()
        self.lines = log

    def _log_line(self, text):
        self.lines.append(text)


class TestHook(Base):
    def setUp(self):
        super().setUp()
        self._dir = voiceprint.DATA_DIR
        voiceprint.DATA_DIR = self.dir

    def tearDown(self):
        voiceprint.DATA_DIR = self._dir
        super().tearDown()

    def test_turn_scoring_never_delays_the_reply_path(self):
        slow = []

        def slow_embed(x):
            time.sleep(0.4)
            slow.append(1)
            return VEC["v"]
        voiceprint.set_embedder(slow_embed)

        async def go():
            b = FakeBridge([])
            t0 = time.monotonic()
            b._voiceprint_turn(speech(4.0))
            took = time.monotonic() - t0
            self.assertLess(took, 0.1)          # returned at once, the model is still running
            self.assertEqual(slow, [])
            await asyncio.gather(*b._vp_tasks)
            return b
        b = asyncio.run(go())
        self.assertEqual(slow, [1])
        self.assertTrue(b.lines[0].startswith("voiceprint: channel=phone enrolling"))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "phone.npz")))

    def test_model_error_is_logged_not_raised(self):
        def boom(x):
            raise RuntimeError("model missing")
        voiceprint.set_embedder(boom)

        async def go():
            b = FakeBridge([])
            b._voiceprint_turn(speech(4.0))
            await asyncio.gather(*b._vp_tasks)
            return b.lines
        self.assertIn("voiceprint: error model missing", asyncio.run(go())[0])

    def test_disabled_flag_does_no_work(self):
        config._VALUES["voiceprint_log"] = False

        async def go():
            b = FakeBridge([])
            b._voiceprint_turn(speech(4.0))
            return b
        b = asyncio.run(go())
        self.assertEqual(b._vp_tasks, set())
        self.assertEqual(self.calls, 0)
        self.assertEqual(b.lines, [])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "phone.npz")))

    def test_bridge_tells_dot_the_call_is_a_phone_call(self):
        with open(os.path.join(os.path.dirname(HERE), "twilio_bridge.py")) as f:
            src = f.read()
        self.assertIn('"source": "phone"', src)


class TestRealModel(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(voiceprint.MODEL_PATH), "model not fetched (python3 fetch_models.py)")
    def test_real_embedding_is_unit_length_and_fast(self):
        voiceprint.set_embedder(None)
        pcm = speech(3.0)
        voiceprint.embed(pcm)
        t = time.perf_counter()
        v = voiceprint.embed(pcm)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)
        self.assertLess(time.perf_counter() - t, 1.0)


if __name__ == "__main__":
    unittest.main()
