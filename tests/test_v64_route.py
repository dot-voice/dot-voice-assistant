"""v64: R54 wiring - after a hand-off Dot checks the module: asleep -> headless job; awake -> passed on; said aloud."""
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
import dotapi  # noqa: E402
import server  # noqa: E402


class Posting:
    def __init__(self, to):
        self.to = to

    def create(self, task, spoken, role=None, context=None, summary=None):
        return {"id": "1", "posted": True, "to": self.to}


def make(tmp, to):
    st = server.CallState()
    st.recap = callrecap.CallRecap()

    async def send(msg):
        return None
    svc = types.SimpleNamespace(turn_log=os.path.join(tmp, "turns.log"), handoff=Posting(to))
    call = types.SimpleNamespace(state=st, svc=svc, posted_handoffs=[], handoff_count=0, send=send,
                                 warm_context=lambda w: "")
    return server.LiveTools(call), st


class TestRoute(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def post(self, to, state):
        tools, st = make(self.tmp.name, to)
        asked = []

        def ask(role, words, *a, **k):
            asked.append((role, words))
            return "%s was asleep, so I started a background job." % role.title()
        with mock.patch.object(dotapi, "session_state", lambda r: state), mock.patch.object(dotapi, "ask_module", ask):
            out = await tools._post("check the overlay", to)
            await asyncio.gather(*list(st.ask_tasks))
        return out, st, asked

    async def test_asleep_starts_headless_and_says_so(self):
        out, st, asked = await self.post("ZETA", "asleep")
        self.assertTrue(out.startswith("posted"))
        self.assertEqual(asked, [("ZETA", "check the overlay")])
        self.assertTrue(any("asleep" in n for n in st.notices), st.notices)
        log = open(os.path.join(self.tmp.name, "turns.log")).read()
        self.assertIn("dotapi: role=ZETA state=asleep", log)

    async def test_awake_only_passes_on(self):
        out, st, asked = await self.post("ZETA", "busy")
        self.assertEqual(asked, [])
        self.assertTrue(any("awake" in n for n in st.notices), st.notices)

    async def test_voice_and_org_are_not_routed(self):
        for to in ("VOICE", "ORG"):
            out, st, asked = await self.post(to, "asleep")
            self.assertEqual((asked, st.notices), ([], []))

    async def test_dot_tells_tick(self):
        svc = types.SimpleNamespace(turn_log=os.path.join(self.tmp.name, "turns.log"))
        with mock.patch.object(dotapi, "pump_tells", lambda *a, **k: 2):
            await server.dot_tells_tick(svc)
        self.assertIn("2 tell(s) for Dot queued", open(svc.turn_log).read())
        with mock.patch.object(dotapi, "pump_tells", mock.Mock(side_effect=RuntimeError("x"))):
            await server.dot_tells_tick(svc)


if __name__ == "__main__":
    unittest.main()
