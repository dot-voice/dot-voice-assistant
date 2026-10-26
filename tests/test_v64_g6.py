"""v64: gap G6 - an open Desktop session of a module counts as awake, so Dot never starts a second writer."""
import json
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import dotapi  # noqa: E402


class TestG6(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.folder = os.path.join(t, "site")
        os.makedirs(os.path.join(self.folder, ".claude", "worktrees", "wt1"))
        self.roles = os.path.join(t, "roles.json")
        with open(self.roles, "w") as f:
            json.dump({self.folder: {"session": "SITE"}}, f)
        self.projects = os.path.join(t, "projects")
        pdir = dotapi.project_dir(self.folder, projects=self.projects)
        os.makedirs(pdir)
        old = os.path.join(pdir, "a.jsonl")
        open(old, "w").close()
        past = time.time() - 3600
        os.utime(old, (past, past))
        self.states = os.path.join(t, "none.json")

    def tearDown(self):
        self.tmp.cleanup()

    def state(self, paths):
        return dotapi.session_state("SITE", roles_path=self.roles, projects=self.projects, states_path=self.states,
                                    lister=lambda: paths)

    def test_open_desktop_session_is_live(self):
        self.assertEqual(self.state([self.folder]), "live")

    def test_worktree_inside_folder_is_live(self):
        self.assertEqual(self.state([os.path.join(self.folder, ".claude", "worktrees", "wt1")]), "live")

    def test_no_open_session_and_old_transcript_is_asleep(self):
        self.assertEqual(self.state(["/", os.path.join(self.tmp.name, "other")]), "asleep")

    def test_sibling_prefix_is_not_a_match(self):
        self.assertFalse(dotapi.desktop_open(self.folder, lister=lambda: [self.folder + "-old"]))

    def test_lister_error_means_not_open(self):
        def boom():
            raise OSError("no lsof")
        self.assertEqual(dotapi.desktop_folders(lister=boom), [])
        self.assertFalse(dotapi.desktop_open("", lister=lambda: ["/"]))


if __name__ == "__main__":
    unittest.main()
