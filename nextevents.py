"""R45: "what's next" while driving. A read-only list of her next calendar events from a configured command.
Config calendar_list_cmd = list of arguments; the number of hours is appended as the last argument; stdout is a JSON list of
{"start": ISO time with offset (or a date for all-day), "end": ..., "title": ..., "location": ...}. Empty = feature off.
Dot never writes to the calendar; it only runs this command and formats what it prints."""
import asyncio
import datetime as dt
import json

import config


def configured():
    try:
        cmd = config.get("calendar_list_cmd")
    except KeyError:
        return False
    return isinstance(cmd, list) and bool(cmd)


def _parse(s):
    """A timezone-aware datetime, or None for a missing, all-day (date only) or bad value."""
    s = str(s or "")
    if "T" not in s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.astimezone()


def conflicts(events):
    """Index pairs of timed events whose times overlap (touching end to start is not a conflict)."""
    spans = [(_parse(e.get("start")), _parse(e.get("end"))) for e in events]
    out = []
    for a in range(len(spans)):
        for b in range(a + 1, len(spans)):
            sa, ea = spans[a]
            sb, eb = spans[b]
            if sa and ea and sb and eb and sa < eb and sb < ea:
                out.append((a, b))
    return out


def _title(e, n):
    return " ".join(str(e.get("title") or "(no title)").split())[:n]


def format_events(events, hours=12):
    """The spoken-ready text for the events list."""
    evs = [e for e in events if isinstance(e, dict)]
    evs.sort(key=lambda e: str(e.get("start") or ""))
    if not evs:
        return "Nothing on your calendar in the next %d hours." % hours
    shown = evs[:8]
    lines = []
    for e in shown:
        s = _parse(e.get("start"))
        when = s.astimezone().strftime("%-I:%M %p") if s else "all day"
        loc = " ".join(str(e.get("location") or "").split())[:100]
        lines.append("%s %s%s" % (when, _title(e, 80), (" at " + loc) if loc else ""))
    text = "Next %d event%s: " % (len(shown), "" if len(shown) == 1 else "s") + "; ".join(lines) + "."
    pairs = conflicts(shown)
    if pairs:
        text += " Conflicts: " + "; ".join("%s overlaps %s" % (_title(shown[a], 40), _title(shown[b], 40)) for a, b in pairs) + "."
    if len(evs) > 8:
        text += " %d more later." % (len(evs) - 8)
    return text


async def next_events(hours=12, run=None):
    """(text for Dot to say, number of events or -1 on failure). Never raises."""
    if not configured():
        return "The calendar is not set up for Dot yet.", -1
    try:
        hours = max(1, min(48, int(hours)))
    except (TypeError, ValueError):
        hours = 12
    cmd = [str(x) for x in config.get("calendar_list_cmd")] + [str(hours)]
    timeout = float(config.get("calendar_list_timeout_s"))

    async def _run():
        p = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            p.kill()
            raise
        return out.decode("utf-8", "replace") if p.returncode == 0 else ""

    try:
        out = await (run(cmd) if run else _run())
        events = json.loads(out)
        if not isinstance(events, list):
            raise ValueError("not a list")
        return format_events(events, hours), len([e for e in events if isinstance(e, dict)])
    except Exception:
        return "I could not read your calendar just now.", -1
