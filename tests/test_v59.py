"""v59: R46 last_call (P28). Temp dir recaps, neutral names, no network."""
import os
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import lastcall  # noqa: E402
import realtime  # noqa: E402
import server  # noqa: E402


def make(log_dir, name, body, age_days=0, sub=""):
    d = os.path.join(log_dir, "recaps", sub)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(body)
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))
    return p


class TestLastCall(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_newest_first_and_n(self):
        make(self.d, "2026-10-19-100000.md", "# Recap\n## What she asked\n- alpha topic\n")
        make(self.d, "2026-10-20-100000.md", "# Recap\n## What she asked\n- beta topic\n")
        out, n = lastcall.last_recaps(self.d)
        self.assertEqual(n, 1)
        self.assertIn("beta topic", out)
        self.assertNotIn("alpha topic", out)
        out, n = lastcall.last_recaps(self.d, n=2)
        self.assertEqual(n, 2)
        self.assertLess(out.index("beta topic"), out.index("alpha topic"))

    def test_old_folder_ignored(self):
        make(self.d, "2026-09-01-100000.md", "ancient topic", age_days=40, sub="old")
        out, n = lastcall.last_recaps(self.d)
        self.assertEqual(n, 0)
        self.assertIn("no recent call recap", out)

    def test_skips_current_call(self):
        make(self.d, "2026-10-20-100000.md", "previous topic", age_days=1)
        make(self.d, "2026-10-21-100000.md", "current topic")
        out, n = lastcall.last_recaps(self.d, skip_after=time.time() - 60)
        self.assertEqual(n, 1)
        self.assertIn("previous topic", out)
        self.assertNotIn("current topic", out)

    def test_trimmed(self):
        make(self.d, "2026-10-20-100000.md", "\n".join("- line number %d of a long recap" % i for i in range(300)))
        out, n = lastcall.last_recaps(self.d)
        self.assertEqual(n, 1)
        self.assertLessEqual(len(out), 1300)
        self.assertIn("(trimmed)", out)

    def test_none_and_missing_dir(self):
        self.assertEqual(lastcall.last_recaps(self.d)[1], 0)
        self.assertEqual(lastcall.last_recaps(os.path.join(self.d, "nope"))[1], 0)

    def test_never_raises(self):
        out, n = lastcall.last_recaps(None)
        self.assertEqual(n, -1)
        self.assertIn("could not read", out)
        self.assertEqual(lastcall.last_recaps(self.d, n="x")[1], 0)

    async def test_log_line_has_no_content(self):
        make(self.d, "2026-10-20-100000.md", "secret zebra topic")
        log = os.path.join(self.d, "turns.log")
        tools = server.LiveTools(types.SimpleNamespace(state=types.SimpleNamespace(recap=None), svc=types.SimpleNamespace(turn_log=log)))
        out = await tools.last_call(1)
        text = open(log).read()
        self.assertIn("live_tool=last_call n=1 ms=", text)
        self.assertNotIn("zebra", text)
        self.assertIn("zebra", out)

    async def test_declared_and_gated_on_sign_in(self):
        tool = next(t for t in realtime.TOOLS if t["name"] == "last_call")
        for phrase in ("continue where we left off", "what did we talk about last time", "anything open"):
            self.assertIn(phrase, tool["description"])
        calls = []

        class Impl:
            async def last_call(self, n):
                calls.append(n)
                return "Call x."

        bridge = types.SimpleNamespace(tools_impl=Impl(), signed_in=False, _snap=None)
        out = await realtime.LiveBridge._exec_tool_inner(bridge, "last_call", {"n": 2})
        self.assertIn("Sign-in is needed", out)
        self.assertEqual(calls, [])
        bridge.signed_in = True
        out = await realtime.LiveBridge._exec_tool_inner(bridge, "last_call", {"n": 2})
        self.assertEqual((out, calls), ("Call x.", [2]))


if __name__ == "__main__":
    unittest.main()
