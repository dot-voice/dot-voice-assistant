"""v64: R54 Dot front door (dotapi): replies read from the feed, session state, headless job in the module folder."""
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

import dotapi  # noqa: E402


def line(to, text, frm="ORG"):
    return json.dumps({"ts": "2026-10-25T21:00:00-04:00", "kind": "tell", "from_role": frm, "to": to, "text": text}) + "\n"


class TestFeedTells(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.feed = os.path.join(self.tmp.name, "feed.jsonl")
        self.cur = os.path.join(self.tmp.name, "cursor.json")
        with open(self.feed, "w") as f:
            f.write(line("VOICE", "DOT: old reply"))

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, s):
        with open(self.feed, "a") as f:
            f.write(s)

    def test_first_run_skips_history_then_reads_only_marked(self):
        self.assertEqual(dotapi.read_tells(self.feed, self.cur), [])
        self.add(line("VOICE", "Weekly loop work order"))
        self.add(line("VAULT", "DOT: not for voice"))
        self.add(line("VOICE", "DOT:  Overlay is fixed.", frm="vault"))
        self.add(line("VOICE", "NEEDS HER: approve the send", frm="SITE"))
        self.add('{"to": "VOICE", "text": "DOT: half')
        got = dotapi.read_tells(self.feed, self.cur)
        self.assertEqual(got, [{"from": "VAULT", "text": "Overlay is fixed.", "needs_her": False},
                               {"from": "SITE", "text": "approve the send", "needs_her": True}])
        self.assertEqual(dotapi.read_tells(self.feed, self.cur), [])
        self.add(' line"}\n')
        self.assertEqual(dotapi.read_tells(self.feed, self.cur)[0]["text"], "half line")

    def test_pump_posts_updates(self):
        dotapi.read_tells(self.feed, self.cur)
        self.add(line("VOICE", "DOT: done"))
        posted = []
        n = dotapi.pump_tells(self.feed, self.cur, post_fn=lambda r, t, needs_her=False: posted.append((r, t, needs_her)))
        self.assertEqual((n, posted), (1, [("ORG", "done", False)]))

    def test_missing_feed(self):
        self.assertEqual(dotapi.read_tells(os.path.join(self.tmp.name, "no.jsonl"), self.cur), [])


class TestSessionState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.mod = os.path.join(d, "mod.folder")
        os.makedirs(self.mod)
        self.roles = os.path.join(d, "roles.json")
        with open(self.roles, "w") as f:
            json.dump({self.mod: {"session": "ZETA", "tasks": ["ZETA", "ZETA-B"]},
                       "/old": {"session": "OLD", "tasks": ["OLD"], "retired_to": "ZETA"}}, f)
        self.projects = os.path.join(d, "projects")
        self.pdir = dotapi.project_dir(self.mod, self.projects)
        os.makedirs(self.pdir)
        self.jl = os.path.join(self.pdir, "a.jsonl")
        open(self.jl, "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def state(self, role, states=None):
        return dotapi.session_state(role, self.roles, self.projects, states or os.path.join(self.tmp.name, "none.json"))

    def test_folder_lookup(self):
        self.assertEqual(dotapi.module_folder("zeta-b", self.roles), self.mod)
        self.assertIsNone(dotapi.module_folder("OLD", self.roles))
        self.assertNotIn(".", os.path.basename(self.pdir))

    def test_recent_transcript_is_busy_old_is_asleep(self):
        self.assertEqual(self.state("ZETA"), "busy")
        old = time.time() - 3600
        os.utime(self.jl, (old, old))
        self.assertEqual(self.state("ZETA"), "asleep")
        self.assertEqual(self.state("NOPE"), "unknown")

    def test_states_file_wins_when_fresh(self):
        p = os.path.join(self.tmp.name, "states.json")
        with open(p, "w") as f:
            json.dump({"ZETA": {"state": "idle", "ts": time.time()}}, f)
        self.assertEqual(self.state("ZETA", p), "live")


class TestHeadless(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, cmd, **kw):
        self.calls.append((cmd, kw))
        out = "noise " + json.dumps({"result": "Done.\n Two  things.", "is_error": False, "total_cost_usd": 0.01,
                                     "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100,
                                               "cache_read_input_tokens": 50, "output_tokens": 8}})
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    def test_flags_folder_env_tokens(self):
        os.environ["CLAUDECODE"] = "1"
        try:
            r = dotapi.run_headless("ZETA", "check the overlay", folder=self.tmp.name, runner=self.runner,
                                    lock_dir=self.tmp.name)
        finally:
            os.environ.pop("CLAUDECODE", None)
        self.assertTrue(r["ok"])
        self.assertEqual((r["text"], r["tokens"]), ("Done. Two things.", 160))
        cmd, kw = self.calls[0]
        self.assertEqual(kw["cwd"], self.tmp.name)
        self.assertNotIn("CLAUDECODE", kw["env"])
        for flag in ("-p", "--strict-mcp-config", "--no-session-persistence", "--append-system-prompt"):
            self.assertIn(flag, cmd)
        self.assertNotIn("Bash", cmd[cmd.index("--allowedTools") + 1])
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_lock_blocks_second_writer(self):
        open(os.path.join(self.tmp.name, "headless_ZETA.lock"), "w").close()
        r = dotapi.run_headless("ZETA", "x", folder=self.tmp.name, runner=self.runner, lock_dir=self.tmp.name)
        self.assertFalse(r["ok"])
        self.assertEqual(self.calls, [])

    def test_ask_module_paths(self):
        told, posted = [], []
        tell = lambda r, w: told.append((r, w))
        post = lambda r, t, **k: posted.append((r, t))
        s = dotapi.ask_module("ZETA", "do it", tell, post, state_fn=lambda r: "busy")
        self.assertIn("awake", s)
        self.assertEqual(told, [("ZETA", "do it")])
        told.clear()
        s = dotapi.ask_module("ZETA", "do it", tell, post, state_fn=lambda r: "asleep", spawn=lambda f: f(),
                              headless_fn=lambda r, w: {"ok": True, "text": "All set.", "seconds": 9.0, "tokens": 5,
                                                        "model": "sonnet"})
        self.assertIn("asleep", s)
        self.assertEqual(posted, [("ZETA", "All set.")])
        self.assertEqual(told[0][0], "ZETA")
        s = dotapi.ask_module("NOPE", "do it", tell, post, state_fn=lambda r: "unknown")
        self.assertEqual(told[-1], ("ORG", "do it"))


if __name__ == "__main__":
    unittest.main()
