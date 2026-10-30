"""v64: R51 fast spoken sign-in (her words 2026-10-25: "the voice print verify ... It's taking way too long")."""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import twilio_bridge as tb  # noqa: E402

PHRASE = "blue harbor seven"


class TestFastSignin(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self._log = tb.log
        tb.log = self.lines.append
        self.b = tb.Bridge("246810", "secret-path-long-enough-0123456", ["+15550001111"], "dot.example.com", auth_token_fn=lambda: "",
                           auth_mode="phrase", phrase_fn=lambda: PHRASE)

    def tearDown(self):
        tb.log = self._log

    def test_phrase_gather_is_fast_and_private(self):
        x = self.b.voice_twiml({"CallSid": "CA1"})
        self.assertIn('speechTimeout="1"', x)
        self.assertIn('speechModel="deepgram_nova-3"', x)   # R57
        self.assertNotIn("enhanced", x)
        self.assertNotIn("hints", x)
        self.assertNotIn("harbor", x)

    def test_signin_seconds_logged(self):
        self.b.voice_twiml({"CallSid": "CA2"})
        self.b.phrase_twiml({"CallSid": "CA2", "SpeechResult": "Blue harbor, seven."})
        ok = [l for l in self.lines if l.startswith("phrase ok")]
        self.assertTrue(ok, self.lines)
        self.assertRegex(ok[-1], r"^phrase ok after_s=\d+\.\d tries=1$")
        self.assertFalse(any("harbo" in l for l in self.lines))

    def test_retry_counts_tries(self):
        self.b.voice_twiml({"CallSid": "CA3"})
        self.b.phrase_twiml({"CallSid": "CA3", "SpeechResult": "something else entirely"})
        self.b.phrase_twiml({"CallSid": "CA3", "SpeechResult": "blue harbor seven"})
        ok = [l for l in self.lines if l.startswith("phrase ok")]
        self.assertTrue(re.match(r"^phrase ok after_s=\d+\.\d tries=2$", ok[-1]), ok)


if __name__ == "__main__":
    unittest.main()
