"""Owner-specific settings. Neutral defaults live here; a local, gitignored file overrides them.

Lookup order for the file: $VOICE_CONFIG, then local_config.json next to this file. Keys missing from the file keep their
default. Environment variables that a module already reads (VOICE_RP_ID, ...) still win over this file.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.environ.get("VOICE_CONFIG") or os.path.join(HERE, "local_config.json")

DEFAULTS = {
    "owner": "Owner",
    "assistant": "Dot",
    "assistant_note": "",
    "owner_job": "a busy professional",
    "style_language": "Reply in the language they spoke.",
    "style_extra": "Keep it to one thing at a time.",
    "sessions_description": ("- Work may run in separate sessions, each owning one module. When asked \"what is going on\" or about a "
                             "session, use the SESSIONS STATUS block."),
    "lookup_names": [],
    "lookup_words": ["my ", "rule", "who ", "who's", "remember", "decid", "last time"],
    "claude_keychain_item": "claude-oauth",
    "claude_disabled_plugins": [],
    "stt_vocabulary": "",
    "places": {"london": [51.5072, -0.1276]},
    "default_place": "london",
    "timezone": "Europe/London",
    "time_label": "London",
    "carry_script": "~/.dot/carry.py",
    "handoff_targets": [["AUTO", "Auto"]],
    "handoff_routes": [],
    "session_aliases": [],
    "session_names": [],
    "private_roles": [],
    "carryover_dir": "~/.dot/carryover",
    "roles_json": "~/.dot/roles.json",
    "team_jobs_dir": "~/.dot/team/jobs",
    "rp_id": "localhost",
    "updates_dir": "~/.dot/updates",
    "brain_tool": "~/.dot/brain/brain.py",
    "launchagent_label": "com.dot.voicecall",
}


def _load(path=PATH):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_VALUES = dict(DEFAULTS)
_VALUES.update(_load())


def get(key):
    return _VALUES[key]


def path(key):
    """A path-valued setting with ~ expanded."""
    return os.path.expanduser(str(_VALUES[key]))


def user_name():
    """Lowercase owner name used as the sign-in user name (the page and the passkey)."""
    return "".join(c for c in str(_VALUES["owner"]).lower() if c.isalnum()) or "owner"


def fill(text):
    """Replace @@OWNER@@, @@ASSISTANT@@ and @@USER@@ in a text (the call page and the prompt)."""
    return (text.replace("@@OWNER@@", str(get("owner"))).replace("@@ASSISTANT@@", str(get("assistant")))
            .replace("@@USER@@", user_name()))
