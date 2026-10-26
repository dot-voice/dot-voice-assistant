"""v64: R52 fresh session status (newest first, with clock time) and R53 needs_her call-back flag."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, ROOT)

import callback  # noqa: E402
import status  # noqa: E402
import updates  # noqa: E402


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).astimezone().isoformat(timespec="seconds")


class TestFreshStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.now = time.time()
        with open(os.path.join(d, "TASKS.json"), "w") as f:
            json.dump({"ZETA": {"file": "Z.md"}}, f)
        self.md = os.path.join(d, "Z.md")
        with open(self.md, "w") as f:
            f.write("## Goal\nx\n## Next step\nShip the blue widget.\n")
        os.utime(self.md, (self.now - 3600, self.now - 3600))
        self.feed = os.path.join(d, "feed.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def write_feed(self, t, text):
        with open(self.feed, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"ts": iso(t), "kind": "tell", "from_role": "OTHER", "to": "ORG", "text": "other news"}) + "\n")
            f.write(json.dumps({"ts": iso(t), "kind": "tell", "from_role": "ZETA", "to": "ORG", "text": text}) + "\n")

    def run_status(self):
        return status.session_status("ZETA", self.tmp.name, self.now, updates_fn=lambda r: None,
                                     titles={"ZETA": "Zeta"}, feed_path=self.feed)

    def test_newer_feed_line_comes_first_with_clock_time(self):
        t = self.now - 120
        self.write_feed(t, "ZETA changed: overlay rebuilt and live")
        out = self.run_status()
        first = out.split(". ")[0]
        self.assertIn("overlay rebuilt", first)
        self.assertIn(time.strftime("%H:%M", time.localtime(t)), first)
        self.assertIn("Ship the blue widget", out)
        self.assertNotIn("other news", out)

    def test_older_feed_line_keeps_handover_first(self):
        self.write_feed(self.now - 7200, "ZETA changed: old thing")
        first = self.run_status().split(". ")[0]
        self.assertIn("saved its notes", first)
        self.assertNotIn("old thing", first)

    def test_newest_feed_missing_file(self):
        self.assertIsNone(status.newest_feed("ZETA", os.path.join(self.tmp.name, "none.jsonl")))

    def test_multi_task_role_keeps_session_names(self):
        p = os.path.join(self.tmp.name, "roles.json")
        with open(p, "w") as f:
            json.dump({"/x/vault": {"session": "VAULT", "tasks": ["VAULT", "VAULT-LI"], "title": "\U0001F510 VAULT"}}, f)
        folder, names = status._role_entry("VAULT", p)
        self.assertEqual(folder, "/x/vault")
        self.assertIn("vault", names)


class TestNeedsHer(unittest.TestCase):
    def test_post_and_cli_flag(self):
        with tempfile.TemporaryDirectory() as d:
            p = updates.post("VAULT", "Need your yes on the overlay", folder=d, needs_her=True)
            self.assertTrue(json.load(open(p))["needs_her"])
            p2 = updates.post("VAULT", "plain", folder=d)
            self.assertNotIn("needs_her", json.load(open(p2)))
            env = dict(os.environ, VOICE_UPDATES_DIR=d)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "updates.py"), "SITE", "Need you", "--needs-her"],
                               env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = [json.load(open(os.path.join(d, n))) for n in os.listdir(d) if "SITE" in n]
            self.assertTrue(got and got[0].get("needs_her"))

    def test_needs_her_is_candidate_without_handoff(self):
        now = time.time()
        items = [("/x/a.json", {"from": "VAULT", "text": "plain", "created": now}),
                 ("/x/b.json", {"from": "VAULT", "text": "need you", "created": now, "needs_her": True})]
        got = callback.candidate(now, items, [], {}, {})
        self.assertEqual(got[0], "b.json")
        self.assertIsNone(callback.candidate(now, items, [], {"b.json": now}, {}))


if __name__ == "__main__":
    unittest.main()
