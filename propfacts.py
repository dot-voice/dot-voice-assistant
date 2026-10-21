"""R35: instant property answers. A property fact question (owner, MLS number, listing agent, "information at <address>")
is answered by the configured property lookup command in seconds; only when it finds nothing is the request handed off.
The command is a config value (property_lookup_cmd: list of arguments, the address or MLS number is appended as the last
argument; stdout is the answer). Empty = no instant lookup, everything is handed off as before."""
import asyncio
import re

import config

FACT_WORDS = re.compile(
    r"\b(owner|owns|owned|who owns|mls|listing agent|listed by|list agent|last sold|sold for|sold price|"
    r"pin number|property information|information (?:at|on|for)|info (?:at|on|for)|details (?:at|on|for)|"
    r"lookup|look up|search for)\b", re.I)
ADDRESS = re.compile(r"\b(\d{1,6}[A-Za-z]?)\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2})", re.I)
MLS = re.compile(r"\b([A-Z]\d{7,9})\b")
STOP = {"in", "on", "at", "for", "and", "to", "the", "of", "is", "it", "a", "so", "if", "me", "my", "i", "can", "you"}
# words after the street name that are not part of the address
CUT = {"in", "and", "so", "then", "to", "tell", "if", "is", "it", "for", "please", "can", "on", "at", "with", "from"}


def query_from(text):
    """The address fragment or MLS number in her words, or ''."""
    t = str(text or "")
    m = MLS.search(t)
    if m:
        return m.group(1)
    m = ADDRESS.search(t)
    if not m:
        return ""
    words = []
    for w in m.group(2).split():
        if w.lower().strip(".,") in CUT:
            break
        words.append(w.strip(".,"))
    return (m.group(1) + " " + " ".join(words)).strip() if words else ""


def is_property_question(text):
    t = str(text or "")
    return bool(FACT_WORDS.search(t) and query_from(t))


def configured():
    try:
        cmd = config.get("property_lookup_cmd")
    except KeyError:
        return False
    return isinstance(cmd, list) and bool(cmd)


async def lookup(text, run=None):
    """The lookup command's answer for the property in her words, '' when there is no command, no answer or any problem."""
    q = query_from(text)
    if not q or not configured():
        return ""
    cmd = [str(x) for x in config.get("property_lookup_cmd")] + [q]
    timeout = float(config.get("property_lookup_timeout_s"))

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
    except Exception:
        return ""
    out = " ".join(str(out or "").split())
    if not out or out.lower().startswith("no match"):
        return ""
    return out[:700]
