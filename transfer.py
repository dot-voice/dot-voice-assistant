"""R28: the call is a call center. She says "transfer me to vault" (or "put me through to the website", "switch to
market watch", Chinese phrases, "back to Dot") and the call's channel (the menu's target) changes.

Pure functions, no I/O: parse() finds a transfer request in her words, resolve_words() maps a spoken name to a
TARGETS key (keys, labels, label words, aliases both ways, role titles, then a close spelling), greeting() is the
one short line the new channel says. The caller (server.py) supplies the live TARGETS, aliases and titles."""
import difflib
import re

CJK = re.compile(r"[\u4e00-\u9fff]")
# Built-in aliases (also read from roles.json "_aliases" at runtime). Undirected: either name finds the other.
BUILTIN_ALIASES = [("CRM", "VAULT"), ("VIDEO", "MW"), ("\u7f51\u7ad9", "WEBSITE"), ("\u7814\u7a76", "RESEARCH"),
                   ("\u89c6\u9891", "VIDEO")]
# R41: sound-alike names. Phone speech-to-text often hears "vault" as "fault", "volt" or "bolt". Per role: the TARGETS keys
# it can be (any one that exists) and the phrases (normalized, spaces and punctuation dropped) that mean it.
SOUND_ALIASES = {
    "VAULT": (("VAULT",), ("vault", "vaults", "fault", "faults", "volt", "bolt", "vote", "valt", "vaul", "walt", "wall",
                           "world", "thevault", "involved", "evolve", "vaultchannel", "bolts", "volts", "boat", "folt")),
    "ORG": (("ORG",), ("organization", "organisation", "org", "or", "orgchannel", "theorg", "organizations")),
    "CRM": (("CRM",), ("crm", "c r m", "cerem", "seearem", "seearm", "searem", "cr m", "crms")),
    "SITE": (("SITE", "WEBSITE"), ("site", "sight", "website", "cite", "web site", "thesite", "thewebsite", "sites")),
    "WX": (("WX", "WECHAT"), ("wx", "w x", "wechat", "we chat", "doubleyouex", "doubleux", "dubyouex", "weechat", "wechats")),
    "REFARM": (("REFARM",), ("refarm", "re farm", "reform", "refund", "refarms", "refom", "reform ")),
    "SOCIAL": (("SOCIAL",), ("social", "socials", "socialmedia", "sosial")),
}
# A lone filler word is only a role when it is the whole name and nothing else fits ("fall" -> VAULT).
LONE_ONLY = {"VAULT": ("fall", "fell", "vol")}
FUZZY_RATIO = 0.8
FUZZY_MIN_LEN = 4


def _sound_table():
    """normalized phrase -> role group name."""
    t = {}
    for role, (_keys, phrases) in SOUND_ALIASES.items():
        for p in phrases:
            t.setdefault(norm(p), role)
        t.setdefault(norm(role), role)
    return t


def _present_roles(targets):
    return [(role, _group_keys(role, targets)[0]) for role in SOUND_ALIASES if _group_keys(role, targets)]


def stt_hint(targets):
    """R41: the channel names for the Whisper vocabulary prompt ("Vault, Org, Site, ..."), only roles that exist."""
    return ", ".join(spoken_label(k, targets) for _r, k in _present_roles(targets))


def live_hint(targets):
    """R41: one sentence for the Gemini Live instructions: the channel names and what speech recognition may hear instead."""
    parts = []
    for role, key in _present_roles(targets):
        alt = [p for p in SOUND_ALIASES[role][1][:7] if norm(p) != norm(key) and len(norm(p)) >= 3 and " " not in p]
        parts.append("%s (may be heard as %s)" % (key, ", ".join(alt[:6])) if alt else key)
    if not parts:
        return ""
    return (" Channel names the caller may say: " + "; ".join(parts) + ". Speech recognition often mishears them, so for transfer_call "
            "pass the channel name you think she meant (for example \"fault\" or \"volt\" means VAULT), not the misheard word.")


def _group_keys(role, targets):
    have = {k for k, _ in targets}
    return [k for k in SOUND_ALIASES[role][0] if k in have]


BACK_WORDS = {"back", "dot", "auto", "frontdesk", "main", "mainline", "mainmenu", "\u524d\u53f0", "\u603b\u53f0"}
LABEL_STOP = {"this", "call", "dot", "auto", "the", "and"}
POLITE = {"please", "now", "channel", "session", "chat", "thanks", "thank", "you", "for", "me", "ok", "okay", "line"}
POLITE_ZH = ("\u9891\u9053", "\u4f1a\u8bdd", "\u90a3\u8fb9", "\u4e00\u4e0b", "\u8c22\u8c22", "\u5427", "\u554a")

_NAME_EN = r"(?P<n>[a-z0-9+&'\-]+(?:\s+[a-z0-9+&'\-]+){0,5})"
_NAME_EN_LOOSE = r"(?P<n>[^.!?,;]{1,50})"
_ME = r"(?:\s+(?:me|us|the\s+call|this\s+call|my\s+call))?"
STRONG_EN = re.compile(r"\b(?:transfer|connect|patch|put)\b" + _ME + r"(?:\s+(?:through|over))?\s+(?:to|with)\s+(?:the\s+)?" +
                       _NAME_EN_LOOSE)
WEAK_EN = re.compile(r"\b(?:switch|change|take|send|pass|bring|go)\b" + _ME + r"(?:\s+over)?\s+(?:back\s+)?to\s+(?:the\s+)?" +
                     _NAME_EN_LOOSE)
BACK_EN = re.compile(r"\b(?:(?:transfer|put|switch|take|send|bring)\b" + _ME + r"\s+back\b|switch\s+back\b|"
                     r"back\s+to\s+(?:dot|the\s+front\s+desk|the\s+main\s+(?:desk|line|menu))\b)")
_NAME_ZH = r"(?P<n>[A-Za-z0-9+&'\-][A-Za-z0-9+&'\- ]{0,30}|[\u4e00-\u9fff]{1,6})"
STRONG_ZH = re.compile("(?:\u8f6c\u7ed9|\u8f6c\u5230|\u8f6c\u63a5\u5230|\u8f6c\u63a5\u7ed9|\u8f6c\u63a5|\u63a5\u901a)(?:\u4e00\u4e0b)?\\s*" + _NAME_ZH)
WEAK_ZH = re.compile("(?:\u5207\u6362\u5230|\u5207\u5230|\u6362\u5230|\u56de\u5230|\u63a5)\\s*" + _NAME_ZH)
BACK_ZH = re.compile("\u8f6c\u56de|\u56de\u5230\\s*(?:dot|\u524d\u53f0|\u603b\u53f0)", re.I)
# R29: "previous" pops one level of the call's channel stack. Only a short utterance counts (never "the previous listing").
PREV_NAMES = {"previous", "previouschannel", "previousone", "previousline", "previoussession", "last", "lastone", "lastchannel"}
PREV_EN = re.compile(r"^(?:ok(?:ay)?\s+|please\s+|and\s+)?(?:(?:go|going)\s+back(?:\s+one)?|(?:take|put|transfer|switch|send|bring)\s+me\s+"
                     r"(?:back\s+)?(?:to\s+)?(?:the\s+)?(?:previous|last)(?:\s+(?:channel|one|line|session))?|(?:the\s+)?previous"
                     r"(?:\s+(?:channel|one|line|session))?|(?:the\s+)?last\s+(?:channel|one))(?:\s+(?:please|now))?$")
PREV_ZH = re.compile("^(?:\u8bf7|\u5e2e\u6211|\u56de\u5230|\u8f6c\u5230|\u5207\u5230|\u5207\u56de|\u56de)?\\s*(?:\u4e0a\u4e00\u4e2a|\u4e0a\u4e2a|\u524d\u4e00\u4e2a)"
                     "(?:\u9891\u9053|\u4f1a\u8bdd|\u7ebf\u8def)?\\s*(?:\u5427|\u554a|\u8c22\u8c22)?$")
YES_EN = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "please", "yes please", "sure thing", "go ahead",
          "of course", "absolutely", "please do", "yes do it", "ok do it", "okay do it", "sounds good", "transfer me"}
NO_EN = {"no", "nope", "nah", "no thanks", "no thank you", "not now", "dont", "do not", "no dont", "dont transfer",
         "stay", "stay here", "no stay here", "no need"}
YES_ZH = {"\u597d", "\u597d\u7684", "\u53ef\u4ee5", "\u8f6c\u5427", "\u884c", "\u8981", "\u662f", "\u662f\u7684", "\u55ef", "\u5bf9", "\u597d\u554a",
          "\u8f6c", "\u53ef\u4ee5\u7684", "\u6ca1\u95ee\u9898", "\u8f6c\u8fc7\u53bb\u5427"}
NO_ZH = {"\u4e0d", "\u4e0d\u7528", "\u4e0d\u8981", "\u4e0d\u4e86", "\u522b", "\u4e0d\u7528\u4e86", "\u4e0d\u8f6c", "\u4e0d\u884c", "\u7b97\u4e86", "\u4e0d\u597d"}


def has_cjk(text):
    return bool(CJK.search(str(text or "")))


def norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(s or "").lower())


def _pairs(aliases):
    out = list(BUILTIN_ALIASES)
    for p in aliases or ():
        try:
            a, b = p
            out.append((str(a), str(b)))
        except (TypeError, ValueError):
            continue
    return out


def candidates(targets, titles=None):
    """normalized spoken form -> TARGETS key. Keys first, then labels, label words, role titles (first one wins)."""
    cand = {}
    keys = [k for k, _ in targets]
    for k in keys:
        cand.setdefault(norm(k), k)
    for k, label in targets:
        label = re.sub(r"\(.*?\)", " ", str(label))
        cand.setdefault(norm(label), k)
        for w in re.split(r"[^A-Za-z0-9]+", label):
            if len(w) >= 2 and w.lower() not in LABEL_STOP:
                cand.setdefault(norm(w), k)
    for k, title in (titles or {}).items():
        if k in keys:
            cand.setdefault(norm(title), k)
    cand.pop("", None)
    cand.pop(norm("AUTO"), None)    # AUTO is only reached by "back"
    return cand


def resolve(name, targets, aliases=(), titles=None, fuzzy=True):
    """A spoken channel name -> TARGETS key, or None."""
    n = norm(name)
    if not n:
        return None
    if n in BACK_WORDS:
        return "AUTO" if any(k == "AUTO" for k, _ in targets) else None
    cand = candidates(targets, titles)
    if n in cand:
        return cand[n]
    pairs, seen, frontier = _pairs(aliases), {n}, [n]
    for _ in range(3):   # CRM -> VAULT, VIDEO -> MW, a Chinese word -> VIDEO -> MW
        nxt = []
        for x in frontier:
            for a, b in pairs:
                for src, dst in ((a, b), (b, a)):
                    if norm(src) == x and norm(dst) not in seen:
                        seen.add(norm(dst))
                        if norm(dst) in cand:
                            return cand[norm(dst)]
                        nxt.append(norm(dst))
        frontier = nxt
    if fuzzy and len(n) >= 3:
        pool = list(cand) + [norm(a) for a, _ in pairs] + [norm(b) for _, b in pairs]
        pool = [p for p in pool if p and not CJK.search(p)]
        hit = difflib.get_close_matches(n, pool, n=1, cutoff=0.75)
        if hit:
            return resolve(hit[0], targets, aliases, titles, fuzzy=False)
    return None


def resolve_all(name, targets, aliases=(), titles=None):
    """R41: every TARGETS key a spoken name can mean (usually 0 or 1). Order: exact key / label / alias, then the sound-alike
    table, then a close spelling (difflib ratio >= 0.8, or the same metaphone-like code). Two different roles -> both are
    returned, so the caller asks instead of guessing."""
    n = norm(name)
    if not n or CJK.search(n):
        return [k for k in [resolve(name, targets, aliases, titles)] if k]
    first = resolve(name, targets, aliases, titles, fuzzy=False)
    if first:
        return [first]
    table = _sound_table()
    role = table.get(n)
    if role:
        keys = _group_keys(role, targets)
        if keys:
            return keys[:1]
    for role, words in LONE_ONLY.items():
        if n in words and _group_keys(role, targets):
            return _group_keys(role, targets)[:1]
    # fuzzy fallback over the sound-alike words and the real keys / labels
    cand = candidates(targets, titles)
    scored = {}
    pool = [(w, r) for w, r in table.items() if len(w) >= FUZZY_MIN_LEN and _group_keys(r, targets)]
    pool += [(w, k) for w, k in cand.items() if len(w) >= FUZZY_MIN_LEN and not CJK.search(w)]
    code = _phonetic(n)
    for w, r in pool:
        key = _group_keys(r, targets)[0] if r in SOUND_ALIASES else r
        ratio = difflib.SequenceMatcher(None, n, w).ratio()
        if len(n) >= FUZZY_MIN_LEN and (ratio >= FUZZY_RATIO or (code and code == _phonetic(w) and len(code) >= 3)):
            scored[key] = max(scored.get(key, 0), ratio)
    if not scored:
        leg = resolve(name, targets, aliases, titles)   # the older fuzzy (cutoff 0.75) still applies as a last try
        return [leg] if leg else []
    top = max(scored.values())
    return [k for k, v in sorted(scored.items(), key=lambda kv: -kv[1]) if top - v < 0.1]


_PH_RULES = (("ph", "f"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("v", "f"), ("w", "f"), ("b", "p"), ("d", "t"),
             ("g", "k"), ("c", "k"))


def _phonetic(w):
    """A crude metaphone: lower-case letters, merge look-alike consonants, drop vowels after the first letter and repeats."""
    w = re.sub(r"[^a-z]", "", str(w).lower())
    for a, b in _PH_RULES:
        w = w.replace(a, b)
    if not w:
        return ""
    out = w[0]
    for ch in w[1:]:
        if ch in "aeiouyh" or ch == out[-1]:
            continue
        out += ch
    return out


def ask_say(keys, targets, zh=False):
    names = [spoken_label(k, targets) for k in keys]
    if zh:
        return "\u4f60\u662f\u6307 %s \u8fd8\u662f %s\uff1f" % (names[0], " \u8fd8\u662f ".join(names[1:]))
    return "Did you mean %s?" % (" or ".join(names[:-1]) + " or " + names[-1] if len(names) > 2 else " or ".join(names))


def _polite_trim(name):
    name = " ".join(str(name or "").split())
    for suffix in POLITE_ZH:
        while name.endswith(suffix) and len(name) > len(suffix):
            name = name[:-len(suffix)].strip()
    words = name.split()
    while words and words[-1].lower().strip("'") in POLITE:
        words.pop()
    if not words or len(words) > 4:
        return None
    return " ".join(words)


def resolve_words_all(name, targets, aliases=(), titles=None):
    full = " ".join(str(name or "").split())
    if full and len(full.split()) <= 4:   # "we chat" ends in a polite word ("chat"): try the words as they are first
        keys = resolve_all(full, targets, aliases, titles)
        if len(keys) == 1 and norm(full) in set(_sound_table()) | set(candidates(targets, titles)):
            return keys
    n = _polite_trim(name)
    return resolve_all(n, targets, aliases, titles) if n else []


def resolve_words(name, targets, aliases=(), titles=None):
    """Like resolve(), after dropping polite words around the name ("the website please", "vault channel")."""
    name = " ".join(str(name or "").split())
    for suffix in POLITE_ZH:
        while name.endswith(suffix) and len(name) > len(suffix):
            name = name[:-len(suffix)].strip()
    words = name.split()
    while words and words[-1].lower().strip("'") in POLITE:
        words.pop()
    if not words or len(words) > 4:
        return None
    keys = resolve_all(" ".join(words), targets, aliases, titles)
    return keys[0] if len(keys) == 1 else None


def _answer_key(text):
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff ]", "", str(text or "").lower().replace("'", ""))
    return " ".join(t.split())


def is_yes(text):
    """R29: her whole answer to "want me to transfer you?" is a yes (a short answer only; "yes but ..." is not)."""
    k = _answer_key(text)
    return k in YES_EN or k.replace(" ", "") in YES_ZH


def is_no(text):
    k = _answer_key(text)
    return k in NO_EN or k.replace(" ", "") in NO_ZH


def is_prev_name(name):
    return norm(name) in PREV_NAMES


def parse(text):
    """Her words -> None, {"kind": "back"} or {"kind": "to", "name": str, "strong": bool}. strong = clearly a transfer request ("transfer / put / connect / switch me ..." with a name of <= 3 words, or zh
    zhuanjie / jietong / zhuan gei wo / ba wo zhuan), so an unknown name is still answered; weak verbs
    (switch / take / go / zh jie) are ignored when the name is not a channel."""
    t = " ".join(str(text or "").split())
    if not t:
        return None
    low = t.lower()
    if BACK_EN.search(low) or BACK_ZH.search(t):
        return {"kind": "back"}
    if PREV_EN.match(low.strip(" .,!?")) or PREV_ZH.match(t.strip(" \u3002\uff0c\uff01\uff1f.,!?")):
        return {"kind": "previous"}
    for rx, strong, src in ((STRONG_EN, True, low), (WEAK_EN, False, low), (STRONG_ZH, True, t), (WEAK_ZH, False, t)):
        m = rx.search(src)
        if m:
            name = m.group("n").strip()
            if is_prev_name(resolve_words_polite(name)):
                return {"kind": "previous"}
            clear = bool(strong and (
                (len(name.split()) <= 3 and re.search(r"\b(?:transfer|put|connect|patch|switch)\s+me\b", low))
                or re.search("\u8f6c\u63a5|\u63a5\u901a|\u8f6c\u7ed9\u6211|\u628a\u6211\u8f6c", t)))
            return {"kind": "to", "name": name, "strong": clear}
    return None


def resolve_words_polite(name):
    """The name with polite words dropped ("the previous channel please" -> "previous channel")."""
    words = " ".join(str(name or "").split()).split()
    while words and words[-1].lower().strip("'") in POLITE - {"channel", "line", "session"}:
        words.pop()
    if words and words[0].lower() == "the":
        words.pop(0)
    return "".join(words)


def plan(text, targets, aliases=(), titles=None):
    """None (not a transfer), ("to", KEY), ("prev",), ("ask", [KEYS]) (R41: the name fits two roles) or ("unknown", name)."""
    p = parse(text)
    if not p:
        return None
    if p["kind"] == "previous":
        return ("prev",)
    if p["kind"] == "back":
        return ("to", "AUTO")
    keys = resolve_words_all(p["name"], targets, aliases, titles)
    if len(keys) == 1:
        return ("to", keys[0])
    if len(keys) > 1:
        return ("ask", keys[:3])
    return ("unknown", p["name"]) if p["strong"] else None


def spoken_label(key, targets):
    if key == "AUTO":
        return "Dot"
    label = dict(targets).get(key, key)
    label = re.sub(r"\(.*?\)", "", str(label)).split("+")[0].strip()
    return label or key


def greeting(key, targets, zh=False, passed=False):
    """AUTO: "Dot here." A channel: "Vault here. Go ahead - replies are spoken here when ready."; passed = her request was
    just posted there too (R29: she said yes to a suggestion)."""
    name = spoken_label(key, targets)
    if key == "AUTO":
        return "\u8fd9\u91cc\u662f Dot\u3002" if zh else "Dot here."
    if zh:
        tail = ("\u5df2\u7ecf\u628a\u4f60\u7684\u8bf7\u6c42\u8f6c\u8fc7\u53bb\u4e86\uff0c\u56de\u590d\u51c6\u5907\u597d\u540e\u4f1a\u5728\u8fd9\u91cc\u8bfb\u7ed9\u4f60\u542c\u3002" if passed else
                "\u8bf7\u8bf4\uff0c\u56de\u590d\u51c6\u5907\u597d\u540e\u4f1a\u5728\u8fd9\u91cc\u8bfb\u7ed9\u4f60\u542c\u3002")
        return "\u8fd9\u91cc\u662f %s\u3002%s" % (name, tail)
    if passed:
        return "%s here. I passed on your request; replies are spoken here when ready." % name
    return "%s here. Go ahead - replies are spoken here when ready." % name


def no_previous_say(zh=False):
    return "\u6ca1\u6709\u4e0a\u4e00\u4e2a\u9891\u9053\u3002" if zh else "There is no previous channel."


def suggest_say(key, targets, zh=False):
    name = spoken_label(key, targets)
    if zh:
        return "\u8fd9\u50cf\u662f %s \u7684\u4e8b\uff0c\u8981\u8f6c\u8fc7\u53bb\u5417\uff1f" % name
    return "Sounds like %s - want me to transfer you?" % name


SORRY_EN = "Sorry, that request did not go through, please say it again."
SORRY_ZH = "\u62b1\u6b49\uff0c\u90a3\u4e2a\u8bf7\u6c42\u6ca1\u6709\u53d1\u51fa\u53bb\uff0c\u8bf7\u518d\u8bf4\u4e00\u904d\u3002"


def sorry_say(zh=False):
    return SORRY_ZH if zh else SORRY_EN


def is_suggestion_chatter(line):
    """A history line that is the transfer question or her yes/no to it - not context for the next channel."""
    t = str(line.get("text", "")).strip()
    if line.get("role") == "assistant":
        return bool(re.match(r"Sounds like .+ want me to transfer you\?$", t)) or t.startswith("\u8fd9\u50cf\u662f")
    return is_yes(t) or is_no(t)


def explicit_addressee(words, key, targets):
    """True when she already named the channel as the addressee ("tell the website to ...") - no suggestion then."""
    names = {key.lower(), spoken_label(key, targets).lower()}
    names |= {w.lower() for w in re.split(r"[^A-Za-z0-9]+", dict(targets).get(key, "")) if len(w) >= 3 and w.lower() not in LABEL_STOP}
    for n in names:   # any whole-word mention counts as naming the channel
        if re.search(r"\b" + re.escape(n) + r"\b", str(words or "").lower()):
            return True
    return bool(re.search("(?:\u544a\u8bc9|\u95ee\u4e00\u4e0b|\u901a\u77e5|\u53d1\u7ed9|\u8ba9)\\s*" + "|".join(re.escape(n) for n in names), str(words or ""), re.I))


def examples(targets, n=4):
    return [spoken_label(k, targets) for k, _ in targets if k not in ("AUTO", "VOICE")][:n]


def unknown_say(targets, zh=False):
    names = examples(targets)
    if zh:
        return "\u6211\u4e0d\u8ba4\u8bc6\u90a3\u4e2a\u9891\u9053\u3002\u53ef\u4ee5\u8f6c\u7ed9 %s\u3002" % ", ".join(names)
    return "Sorry, I do not know that channel. I can put you through to %s." % ", ".join(names)
