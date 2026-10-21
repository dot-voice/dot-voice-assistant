"""R46: "continue where we left off". Returns the newest call recap(s) from logs/recaps, trimmed, so Dot can say what was
discussed and what is still open. Recaps in recaps/old (older than 14 days) are ignored. Read-only; never raises."""
import os
import re

MAX_CHARS = 1200
MAX_N = 3
_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}(-\d+)?\.md$")


def _trim(text, n):
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit("\n", 1)[0].rstrip() + "\n(trimmed)"


def last_recaps(log_dir, n=1, skip_after=None, max_chars=MAX_CHARS):
    """(text for Dot to say, number of recaps used or -1 on failure). skip_after = epoch seconds; a recap file written at or
    after it belongs to the current call and is skipped."""
    try:
        try:
            n = max(1, min(MAX_N, int(n)))
        except (TypeError, ValueError):
            n = 1
        d = os.path.join(log_dir, "recaps")
        names = sorted((x for x in os.listdir(d) if _NAME.match(x) and os.path.isfile(os.path.join(d, x))), reverse=True) \
            if os.path.isdir(d) else []
        if skip_after is not None:
            names = [x for x in names if os.path.getmtime(os.path.join(d, x)) < skip_after]
        names = names[:n]
        if not names:
            return "There is no recent call recap to continue from.", 0
        per = max(300, max_chars // len(names))
        parts = []
        for x in names:
            with open(os.path.join(d, x), encoding="utf-8", errors="replace") as f:
                parts.append("Call %s:\n%s" % (x[:-3], _trim(f.read(), per)))
        return _trim("\n\n".join(parts), max_chars + 40 * len(names)), len(names)
    except Exception:
        return "I could not read the last call recap just now.", -1
