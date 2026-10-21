"""R47 call-back on hand-off reply (P19 phase 1): when a session answers a hand-off the owner made by voice and no call is
live, Dot places ONE outbound call to the owner so the answer is heard hands-free.

The call goes to the same public voice webhook an inbound call uses, so it passes the same phrase sign-in, and then the
normal updates loop speaks the pending update first. Twilio's machine detection runs before the webhook: when a machine
answered, the bridge hangs up and on_machine() sends the update text to the owner's Telegram instead.

Everything here is off unless config callback_on_reply is true. Nothing raises; log lines carry no message text, no phone
numbers and no secret path. The "To" number is always the first entry of twilio_allowed_callers, never another number."""
import base64
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import callrecap
import config
import updates

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "logs", "callback_state.json")
HANDOFF_DIR = os.path.join(HERE, "handoff")
RETRY_AFTER_FAIL_S = 300
KEEP_HANDLED = 200
_latched = False   # set when the state file cannot be written: no more calls until the service restarts
_logged = set()   # (update name, reason) already logged in this process, so a 5 s loop does not repeat itself


def _log(message, log_path=None):
    try:
        from diag import log_line
        log_line("callback: " + message, log_path)
    except Exception:
        pass


def _log_once(name, reason, message, log_path=None):
    if (name, reason) in _logged:
        return
    if len(_logged) > 500:
        _logged.clear()
    _logged.add((name, reason))
    _log(message, log_path)


# ------------------------------------------------------------------ state and rules
def load_state(path=STATE_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        if isinstance(st, dict):
            placed = []
            for t in (st.get("placed") if isinstance(st.get("placed"), list) else []):
                try:
                    placed.append(float(t))
                except (TypeError, ValueError):
                    pass
            handled = st.get("handled") if isinstance(st.get("handled"), dict) else {}
            out = {"placed": placed, "handled": {str(k): v for k, v in handled.items() if isinstance(v, (int, float))}}
            if isinstance(st.get("fail_at"), (int, float)):
                out["fail_at"] = st["fail_at"]
            return out
    except (OSError, ValueError):
        pass
    return {"placed": [], "handled": {}}


def save_state(st, path=STATE_FILE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        st["placed"] = [float(t) for t in st.get("placed", [])][-50:]
        h = st.get("handled", {})
        if len(h) > KEEP_HANDLED:
            for k in sorted(h, key=h.get)[:len(h) - KEEP_HANDLED]:
                del h[k]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def local_time(now, tz_name):
    """(minutes since local midnight, local date string) in the named time zone; falls back to the machine's zone."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        d = datetime.fromtimestamp(now, ZoneInfo(tz_name))
    except Exception:
        from datetime import datetime
        d = datetime.fromtimestamp(now)
    return d.hour * 60 + d.minute, d.strftime("%Y-%m-%d")


def in_quiet(now, spec, tz_name):
    """True inside quiet hours "HH:MM-HH:MM" (wraps midnight). A bad spec means never quiet."""
    try:
        a, b = str(spec).split("-")
        ah, am = a.strip().split(":")
        bh, bm = b.strip().split(":")
        start, end = int(ah) * 60 + int(am), int(bh) * 60 + int(bm)
    except (ValueError, AttributeError):
        return False
    if start == end:
        return False
    minute, _ = local_time(now, tz_name)
    return (start <= minute < end) if start < end else (minute >= start or minute < end)


def _parse_stamp(stamp):
    try:
        return time.mktime(time.strptime(str(stamp), "%Y%m%d-%H%M%S"))
    except (ValueError, OverflowError):
        return 0.0


def recent_handoffs(now, window_s, folder=HANDOFF_DIR):
    """[(created_epoch, role)] of hand-offs sent to a session within the window."""
    out = []
    try:
        names = os.listdir(folder)
    except OSError:
        return out
    for n in names:
        if not n.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, n), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        created = _parse_stamp(rec.get("created"))
        if rec.get("source") == "voice_call" and rec.get("posted") and created and 0 <= now - created <= window_s:
            out.append((created, str(rec.get("to", "")).upper()))
    return out


def candidate(now, items, handoffs, handled, aliases):
    """The first pending update that answers a recent hand-off and was not handled yet: (name, path, record) or None."""
    for path, rec in items:
        name = os.path.basename(path)
        if name in handled:
            continue
        role = str(rec.get("from", "")).upper()
        if role in ("", "TEST"):
            continue
        keys = {role, aliases.get(role, role)}
        created = float(rec.get("created", 0))
        if any(r in keys and c <= created for c, r in handoffs):
            return name, path, rec
    return None


# ------------------------------------------------------------------ Twilio
def twilio_credentials():
    """dict account_sid, key_sid, key_secret from the configured credentials module."""
    folder = os.path.expanduser(str(config.get("twilio_creds_path") or ""))
    if folder and folder not in sys.path:
        sys.path.insert(0, folder)
    return importlib.import_module(str(config.get("twilio_creds_module"))).get_credentials()


def place_call(to, from_, url, creds_fn=twilio_credentials, urlopen=urllib.request.urlopen):
    """Twilio Calls.create with machine detection. Returns (ok, reason); reason never holds a number, secret or credential."""
    try:
        c = creds_fn()
        sid, key, secret = c["account_sid"], c["key_sid"], c["key_secret"]
        data = urllib.parse.urlencode({"To": to, "From": from_, "Url": url, "Method": "GET",
                                       "MachineDetection": "Enable", "Timeout": "30"}).encode()
        req = urllib.request.Request("https://api.twilio.com/2010-04-01/Accounts/%s/Calls.json" % sid, data=data)
        req.add_header("Authorization", "Basic " + base64.b64encode(("%s:%s" % (key, secret)).encode()).decode())
    except Exception as e:
        return False, type(e).__name__   # nothing was sent
    try:
        with urlopen(req, timeout=15) as r:
            code = getattr(r, "status", 200)
        return (200 <= code < 300), ("ok" if 200 <= code < 300 else "http_%s" % code)
    except urllib.error.HTTPError as e:
        return False, "http_%s" % e.code
    except Exception as e:
        return False, "uncertain_" + type(e).__name__   # the request may have reached Twilio: the caller counts it as placed


def voice_url():
    return "https://%s/tw/%s/voice" % (config.get("twilio_public_host"), config.get("twilio_path_secret"))


def send_text(text, telegram=callrecap.send_telegram):
    try:
        return bool(telegram(text, config.get("call_recap_telegram_chat_id")))
    except Exception:
        return False


def _persist(st, state_path, log_path):
    """Save; when the file cannot be written, refuse every further call until restart (a lost record could repeat a call)."""
    global _latched
    if save_state(st, state_path):
        return True
    _latched = True
    _log("skipped reason=state_unwritable", log_path)
    return False


# ------------------------------------------------------------------ the decision
def tick(live, now=None, log_path=None, aliases=None, cfg=config.get, items=None, handoffs=None, state_path=STATE_FILE,
         place=place_call, telegram=callrecap.send_telegram, url_fn=voice_url):
    """One look at the updates. live = a call is open right now. Returns a short reason string, or None when there was
    nothing to decide. Never raises."""
    try:
        if not cfg("callback_on_reply"):
            return None
        if _latched:
            _log_once("-", "latch", "skipped reason=state_unwritable", log_path)
            return "state_unwritable"
        now = time.time() if now is None else now
        aliases = aliases or {}
        st = load_state(state_path)
        if items is None:
            items = updates.pending()
        if handoffs is None:
            handoffs = recent_handoffs(now, float(cfg("callback_window_s")))
        found = candidate(now, items, handoffs, st["handled"], aliases)
        if not found:
            return None
        name, _path, rec = found
        if live:
            _log_once(name, "live", "skipped reason=call_live", log_path)
            return "call_live"
        line = updates.spoken_line(rec)
        if in_quiet(now, cfg("callback_quiet"), cfg("callback_timezone")):
            ok = send_text(line, telegram)
            if ok:   # a failed send is tried again on the next look
                st["handled"][name] = now
                _persist(st, state_path, log_path)
            _log("skipped reason=quiet_hours telegram=%s" % ("sent" if ok else "failed"), log_path)
            return "quiet_hours"
        to_list = list(cfg("twilio_allowed_callers") or [])
        from_ = str(cfg("twilio_number") or "")
        if not to_list or not from_ or not cfg("twilio_public_host") or not cfg("twilio_path_secret"):
            _log_once(name, "cfg", "skipped reason=not_configured", log_path)
            return "not_configured"
        placed = [float(t) for t in st["placed"]]
        if placed and now - max(placed) < float(cfg("callback_min_gap_s")):
            _log_once(name, "gap", "skipped reason=gap", log_path)
            return "gap"
        _, today = local_time(now, cfg("callback_timezone"))
        if sum(1 for t in placed if local_time(t, cfg("callback_timezone"))[1] == today) >= int(cfg("callback_max_per_day")):
            _log_once(name, "cap", "skipped reason=daily_cap", log_path)
            return "daily_cap"
        if now - float(st.get("fail_at", 0)) < RETRY_AFTER_FAIL_S:
            return "retry_wait"
        ok, why = place(str(to_list[0]), from_, url_fn())
        if ok or str(why).startswith("uncertain_"):   # an unclear failure may have created the call: count it
            st["placed"] = placed + [now]
            st["handled"][name] = now
            st.pop("fail_at", None)
            _persist(st, state_path, log_path)
            _log("placed" if ok else "placed reason=uncertain_twilio_reply", log_path)
            return "placed"
        st["fail_at"] = now
        _persist(st, state_path, log_path)
        _log("skipped reason=twilio_error %s" % why, log_path)
        return "twilio_error"
    except Exception as e:
        _log("skipped reason=error %s" % type(e).__name__, log_path)
        return "error"


def on_machine(log_path=None, telegram=callrecap.send_telegram, items=None, mark=updates.mark_spoken):
    """The call was answered by a machine: send the pending update text to Telegram instead. Never raises."""
    try:
        got = updates.pending() if items is None else items
        if not got:
            _log("skipped reason=machine_no_update", log_path)
            return False
        text = "\n".join(updates.spoken_line(rec) for _, rec in got[:3])
        ok = send_text(text, telegram)
        if ok:
            for path, _ in got[:3]:
                mark(path)
        _log("skipped reason=machine telegram=%s" % ("sent" if ok else "failed"), log_path)
        return ok
    except Exception as e:
        _log("skipped reason=error %s" % type(e).__name__, log_path)
        return False
