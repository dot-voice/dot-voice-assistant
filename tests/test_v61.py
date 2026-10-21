"""v61: R48 unread_digest (P29). Temp dir digests, neutral names, no network."""
import json
import os
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import config  # noqa: E402
import realtime  # noqa: E402
import server  # noqa: E402
import unreadsum  # noqa: E402


def put(d, name, items):
    with open(os.path.join(d, name), "w") as f:
        f.write(items if isinstance(items, str) else json.dumps(items))


def item(ago_h, ch="email", who="Alex Example", text="asks about a showing", reply=False):
    return {"ts": time.time() - ago_h * 3600, "channel": ch, "from_name": who, "one_line_summary": text, "needs_reply": reply}


class TestDigest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self._old = config.get("digest_dir")
        config._VALUES["digest_dir"] = self.d

    def tearDown(self):
        self.tmp.cleanup()
        config._VALUES["digest_dir"] = self._old

    def test_merge_order_and_counts(self):
        put(self.d, "a.json", [item(1, "email", "Pat One", "older reply item", True), item(3, "email", "Sam Two", "plain item")])
        put(self.d, "b.json", [item(0.5, "wechat", "Lee Three", "newest plain item"), item(2, "imessage", "Kit Four", "newer reply item", True)])
        out, n = unreadsum.unread_digest(12)
        self.assertEqual(n, 4)
        self.assertIn("2 email", out)
        self.assertIn("1 wechat", out)
        self.assertIn("1 imessage", out)
        self.assertIn("2 need a reply", out)
        order = [out.index(x) for x in ("older reply item", "newer reply item", "newest plain item", "Sam Two")]
        self.assertEqual(order, sorted(order))

    def test_since_hours_filters_old_and_future(self):
        put(self.d, "a.json", [item(1, text="fresh item"), item(30, text="stale item"), item(-5, text="future item")])
        out, n = unreadsum.unread_digest(12)
        self.assertEqual(n, 1)
        self.assertIn("fresh item", out)
        self.assertNotIn("stale item", out)
        out, n = unreadsum.unread_digest(48)
        self.assertEqual(n, 2)
        put(self.d, "a.json", [item(30)])
        self.assertIn("Nothing new", unreadsum.unread_digest(12)[0])

    def test_max_items_and_chars(self):
        put(self.d, "a.json", [item(i / 10.0, text="x" * 300, who="Person %d" % i) for i in range(1, 21)])
        out, n = unreadsum.unread_digest(12)
        self.assertEqual(n, 20)
        self.assertLessEqual(len(out), 1100)
        self.assertLessEqual(out.count(" on email:"), 8)
        self.assertIn("more not read", out)

    def test_epoch_and_iso_timestamps(self):
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 600))
        put(self.d, "a.json", [{"ts": int(time.time() - 600), "channel": "sms", "from_name": "A", "one_line_summary": "epoch one"},
                               {"ts": iso, "channel": "sms", "from_name": "B", "one_line_summary": "iso one"},
                               {"ts": "garbage", "channel": "sms"}, "junk", {"ts": True}])
        out, n = unreadsum.unread_digest(1)
        self.assertEqual(n, 2)
        self.assertIn("none need a reply", out)

    def test_missing_dir_off_and_bad_json_friendly(self):
        config._VALUES["digest_dir"] = os.path.join(self.d, "nope")
        self.assertEqual(unreadsum.unread_digest(12), (unreadsum.NONE_TEXT, -1))
        config._VALUES["digest_dir"] = ""
        self.assertEqual(unreadsum.unread_digest(12)[1], -1)
        config._VALUES["digest_dir"] = self.d
        self.assertEqual(unreadsum.unread_digest(12)[1], -1)   # folder with no files
        put(self.d, "bad.json", "{not json")
        put(self.d, "obj.json", {"a": 1})
        out, n = unreadsum.unread_digest(12)
        self.assertEqual(n, 0)
        self.assertIn("Nothing new", out)

    def test_never_raises(self):
        put(self.d, "a.json", [item(1)])
        out, n = unreadsum.unread_digest("x")
        self.assertEqual(n, 1)
        orig = unreadsum.load_items
        unreadsum.load_items = lambda *a, **k: 1 / 0
        try:
            self.assertEqual(unreadsum.unread_digest(12), (unreadsum.NONE_TEXT, -1))
        finally:
            unreadsum.load_items = orig

    async def test_log_line_has_no_names_or_text(self):
        put(self.d, "a.json", [item(1, who="Zed Zebra", text="secret zebra topic")])
        log = os.path.join(self.d, "turns.log")
        tools = server.LiveTools(types.SimpleNamespace(state=None, svc=types.SimpleNamespace(turn_log=log)))
        out = await tools.unread_digest(12)
        text = open(log).read()
        self.assertIn("live_tool=unread_digest n=1 ms=", text)
        self.assertNotIn("zebra", text.lower())
        self.assertIn("zebra", out)

    async def test_declared_and_gated_on_sign_in(self):
        tool = next(t for t in realtime.TOOLS if t["name"] == "unread_digest")
        for phrase in ("any new messages", "anything from clients", "who do I need to reply to"):
            self.assertIn(phrase, tool["description"])
        calls = []

        class Impl:
            async def unread_digest(self, h):
                calls.append(h)
                return "Digest x."

        bridge = types.SimpleNamespace(tools_impl=Impl(), signed_in=False, _snap=None)
        out = await realtime.LiveBridge._exec_tool_inner(bridge, "unread_digest", {"since_hours": 6})
        self.assertIn("Sign-in is needed", out)
        self.assertEqual(calls, [])
        bridge.signed_in = True
        out = await realtime.LiveBridge._exec_tool_inner(bridge, "unread_digest", {"since_hours": 6})
        self.assertEqual((out, calls), ("Digest x.", [6]))


if __name__ == "__main__":
    unittest.main()
