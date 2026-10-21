"""v58: R45 next_events (P27). Fake command, no network, no calendar."""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import config  # noqa: E402
import nextevents  # noqa: E402
import realtime  # noqa: E402
import server  # noqa: E402

EVS = [
    {"start": "2026-10-21T14:00:00-04:00", "end": "2026-10-21T15:30:00-04:00", "title": "Showing Elm St", "location": "12 Elm St, Springfield"},
    {"start": "2026-10-21T15:00:00-04:00", "end": "2026-10-21T16:00:00-04:00", "title": "Call Alex", "location": ""},
    {"start": "2026-10-21T16:00:00-04:00", "end": "2026-10-21T17:00:00-04:00", "title": "Dinner", "location": ""},
]


def cfg(cmd):
    real = config.get
    return mock.patch.object(config, "get", lambda k: cmd if k == "calendar_list_cmd" else real(k))


class TestNextEvents(unittest.IsolatedAsyncioTestCase):
    def test_off_by_default(self):
        self.assertEqual(config.DEFAULTS["calendar_list_cmd"] if hasattr(config, "DEFAULTS") else [], [])

    def test_format_has_location_and_conflict(self):
        t = nextevents.format_events(EVS)
        self.assertIn("Showing Elm St at 12 Elm St, Springfield", t)
        self.assertIn("Showing Elm St overlaps Call Alex", t)
        self.assertNotIn("Dinner overlaps", t)   # touching end to start is not a conflict

    def test_empty(self):
        self.assertIn("Nothing on your calendar", nextevents.format_events([]))

    def test_all_day_event_has_no_conflict(self):
        t = nextevents.format_events([{"start": "2026-10-21", "end": "2026-10-22", "title": "Holiday"}] + EVS[:1])
        self.assertIn("all day Holiday", t)
        self.assertNotIn("Conflicts", t)

    async def test_off_when_not_configured(self):
        with cfg([]):
            out, n = await nextevents.next_events()
        self.assertEqual(n, -1)
        self.assertIn("not set up", out)

    async def test_fake_command_runs_with_hours_last(self):
        seen = []

        async def run(cmd):
            seen.append(cmd)
            return json.dumps(EVS)
        with cfg(["fakecal", "--x"]):
            out, n = await nextevents.next_events(6, run=run)
        self.assertEqual((n, seen), (3, [["fakecal", "--x", "6"]]))
        self.assertIn("Next 3 events", out)

    async def test_bad_output_and_crash_never_raise(self):
        for bad in ("not json", "{}", ""):
            async def run(cmd, bad=bad):
                return bad
            with cfg(["fakecal"]):
                out, n = await nextevents.next_events(run=run)
            self.assertEqual(n, -1)
            self.assertIn("could not read", out)

        async def boom(cmd):
            raise RuntimeError("x")
        with cfg(["fakecal"]):
            out, n = await nextevents.next_events(run=boom)
        self.assertEqual(n, -1)

    async def test_real_subprocess_timeout(self):
        with cfg(["sleep"]), mock.patch.object(config, "get", side_effect=lambda k: ["sleep"] if k == "calendar_list_cmd" else 0.2):
            out, n = await nextevents.next_events()
        self.assertEqual(n, -1)

    async def test_log_line_has_no_titles(self):
        tmp = tempfile.TemporaryDirectory()
        log = os.path.join(tmp.name, "turns.log")
        tools = server.LiveTools(types.SimpleNamespace(state=types.SimpleNamespace(), svc=types.SimpleNamespace(turn_log=log)))

        async def run(cmd):
            return json.dumps(EVS)
        real = nextevents.next_events
        with cfg(["fakecal"]), mock.patch.object(nextevents, "next_events", lambda h: real(h, run=run)):
            out = await tools.next_events(12)
        text = open(log).read()
        tmp.cleanup()
        self.assertIn("live_tool=next_events n=3 ms=", text)
        self.assertNotIn("Elm", text)
        self.assertIn("Showing Elm St", out)

    async def test_declared_and_gated_on_sign_in(self):
        tool = next(t for t in realtime.TOOLS if t["name"] == "next_events")
        for phrase in ("what's next", "what's on my calendar"):
            self.assertIn(phrase, tool["description"])
        calls = []

        class Impl:
            async def next_events(self, hours):
                calls.append(hours)
                return "Next 1 event: x."

        bridge = types.SimpleNamespace(tools_impl=Impl(), signed_in=False, _snap=None)
        out = await realtime.LiveBridge._exec_tool_inner(bridge, "next_events", {"hours": 6})
        self.assertIn("Sign-in is needed", out)
        self.assertEqual(calls, [])
        bridge.signed_in = True
        out = await realtime.LiveBridge._exec_tool_inner(bridge, "next_events", {"hours": 6})
        self.assertEqual((out, calls), ("Next 1 event: x.", [6]))


if __name__ == "__main__":
    unittest.main()
