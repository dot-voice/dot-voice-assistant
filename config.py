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
    "board_file": "~/.dot/board.html",
    "brain_tool": "~/.dot/brain/brain.py",
    "launchagent_label": "com.dot.voicecall",
    "property_lookup_cmd": [],       # R35: command whose last argument is an address or MLS number; stdout = instant property answer
    "property_lookup_timeout_s": 8,
    "calendar_list_cmd": [],         # R45: command (hours appended as last argument) printing a JSON list of {start,end,title,location}; empty = next_events off
    "calendar_list_timeout_s": 8,
    "reminder_words": ["remind me", "reminder", "set an alarm", "wake me"],
    "twilio_path_secret": "",        # phone bridge: long random URL token (>= 24 chars); empty = bridge refuses to start
    "twilio_allowed_callers": [],    # phone bridge: E.164 numbers allowed to call; empty = reject everyone
    "twilio_auth": "pin",            # phone sign-in: "pin" (keypad) or "phrase" (spoken passphrase in Keychain, PIN still works)
    "twilio_public_host": "",        # phone bridge: public host name of the Funnel, e.g. host.tailnet.ts.net
    "twilio_fallback_voice_url": "",  # phone bridge: callers not on the allowlist are redirected here (empty = reject)
    "call_recap": False,             # R38: write logs/recaps/<date>-<time>.md when a Live call ends
    "call_recap_telegram": False,    # R38: also send the recap to Telegram (token: Keychain item of the WeChat bot)
    "telegram_keychain_service": "dot-telegram-bot-token",  # R38: Keychain service holding the bot token
    "telegram_keychain_account": "",  # R38: Keychain account of that item
    "call_recap_telegram_chat_id": "",  # R38: Telegram chat id (empty = $VHBOT_CHAT_ID, else not sent)
    "call_summary_line": False,      # R39: one call_summary line in turns.log when a Live call ends
    "twilio_reaper": False,          # R39: close a bridge call with no media for twilio_stale_s seconds
    "twilio_stale_s": 120,
    "voiceprint_log": False,         # R42: score each turn against the owner voiceprint and log it (log-only, never blocks)
    "voiceprint_enforce": False,     # P25 phase 2: reserved, does nothing yet
    "twilio_port": 8772,             # phone bridge: local port (127.0.0.1 only)
    "twilio_number": "",             # R47: the Dot number (E.164) calls are placed from; empty = call-back off
    "twilio_creds_path": "",         # R47: folder added to sys.path to import the credentials module
    "twilio_creds_module": "appt_sms",  # R47: module whose get_credentials() returns account_sid, key_sid, key_secret
    "callback_on_reply": False,      # R47: phone her when a session answers a hand-off and no call is live
    "callback_quiet": "21:00-08:00",  # R47: quiet hours (local, wraps midnight): Telegram only, no call
    "callback_timezone": "America/Toronto",
    "callback_min_gap_s": 1200,      # R47: at most one call-back per 20 minutes
    "callback_max_per_day": 6,
    "callback_window_s": 3600,       # R47: only replies to hand-offs made in the last 60 minutes
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
