"""v64: hand_off logs its slow steps (her call 2026-10-25 21:59: tool_ms=4542, 5 s of silence)."""
import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import callrecap  # noqa: E402
import propfacts  # noqa: E402
import server  # noqa: E402


def make(tmp):
    st = server.CallState()
    st.recap = callrecap.CallRecap()

    async def send(msg):
        return None
    svc = types.SimpleNamespace(turn_log=os.path.join(tmp, "turns.log"), handoff=None)
    call = types.SimpleNamespace(state=st, svc=svc, posted_handoffs=[], handoff_count=0, send=send,
                                 warm_context=lambda w: "", target="AUTO", suggest_key=lambda w, a: None,
                                 transfer_names=lambda: ([], []))
    return server.LiveTools(call), st


class TestHandoffTiming(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def log(self):
        p = os.path.join(self.tmp.name, "turns.log")
        return open(p).read() if os.path.exists(p) else ""

    async def test_slow_property_lookup_is_logged(self):
        tools, st = make(self.tmp.name)

        async def slow(words):
            await asyncio.sleep(1.1)
            return ""
        with mock.patch.object(propfacts, "is_property_question", lambda w: True), \
                mock.patch.object(propfacts, "lookup", slow):
            out = await tools.hand_off("secret words about 5 Main Street", "VAULT")
        if st.pending_handoff:
            st.pending_handoff["task"].cancel()
        line = [l for l in self.log().splitlines() if "hand_off steps:" in l]
        self.assertTrue(line, self.log())
        self.assertIn("prop=", line[-1])
        self.assertIn("total=", line[-1])
        self.assertNotIn("secret", self.log())

    async def test_fast_hand_off_logs_nothing(self):
        tools, st = make(self.tmp.name)
        with mock.patch.object(propfacts, "is_property_question", lambda w: False):
            await tools.hand_off("check the overlay", "VAULT")
        if st.pending_handoff:
            st.pending_handoff["task"].cancel()
        self.assertNotIn("hand_off steps:", self.log())


if __name__ == "__main__":
    unittest.main()
