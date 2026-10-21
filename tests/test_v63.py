"""v63: R50 talk-it-through mode (P30). Neutral names, temp dirs, no network."""
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import callrecap  # noqa: E402
import realtime  # noqa: E402
import server  # noqa: E402

SECRET = "my zebra secret worry"


def make_tools(tmp):
    log = os.path.join(tmp, "turns.log")
    st = server.CallState()
    st.recap = callrecap.CallRecap()
    call = types.SimpleNamespace(state=st, svc=types.SimpleNamespace(turn_log=log))
    return server.LiveTools(call), st, log


class TestTalkMode(unittest.IsolatedAsyncioTestCase):
    def test_prompt_has_the_mode_safety_line_and_988(self):
        p = realtime.build_instructions("Test Owner")
        for part in ("TALK-IT-THROUGH MODE", "I just need to talk", "back to work", "988", "911", "pulling over",
                     "not a therapist", "eyes open", "Mandarin Chinese", "No lists"):
            self.assertIn(part, p)

    def test_tool_declared_and_needs_no_sign_in(self):
        tool = next(t for t in realtime.TOOLS if t["name"] == "talk_mode")
        self.assertIn("on", tool["parameters"]["properties"])
        calls = []

        class Impl:
            async def talk_mode(self, on):
                calls.append(on)
                return "ok"

        bridge = types.SimpleNamespace(tools_impl=Impl(), signed_in=False, _snap=None)
        import asyncio
        out = asyncio.run(realtime.LiveBridge._exec_tool_inner(bridge, "talk_mode", {"on": True}))
        self.assertEqual((out, calls), ("ok", [True]))

    async def test_recap_excludes_talk_turns_and_says_talk_mode_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools, st, log = make_tools(tmp)
            st.history.append({"role": "user", "text": "what is next today"})
            st.history.append({"role": "user", "text": "I just need to talk"})
            await tools.talk_mode(True)
            st.history.append({"role": "user", "text": SECRET, "talk": True})
            st.history.append({"role": "assistant", "text": "That sounds heavy.", "talk": True})
            await tools.talk_mode(False)
            text = st.recap.build(list(st.history), [])
            self.assertNotIn("zebra", text)
            self.assertNotIn("just need to talk", text)
            self.assertNotIn("heavy", text)
            self.assertIn("what is next today", text)
            self.assertIn("- talk mode used", text)

    async def test_no_talk_mode_no_line(self):
        rc = callrecap.CallRecap()
        self.assertNotIn("talk mode used", rc.build([{"role": "user", "text": "hello"}], []))

    async def test_mode_log_line_has_no_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools, st, log = make_tools(tmp)
            st.history.append({"role": "user", "text": SECRET})
            await tools.talk_mode(True)
            await tools.talk_mode(False)
            text = open(log).read()
            self.assertIn("mode: talk on", text)
            self.assertIn("mode: talk off", text)
            self.assertNotIn("zebra", text)
            self.assertFalse(st.talk)

    async def test_lines_spoken_in_talk_mode_are_tagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools, st, log = make_tools(tmp)
            call = server.Call.__new__(server.Call)
            call.state, call.history = st, st.history
            await tools.talk_mode(True)
            server.Call.remember(call, "user", SECRET, turn=1)
            self.assertTrue(st.history[-1].get("talk"))
            await tools.talk_mode(False)
            server.Call.remember(call, "user", "work again", turn=2)
            self.assertFalse(st.history[-1].get("talk"))


if __name__ == "__main__":
    unittest.main()
