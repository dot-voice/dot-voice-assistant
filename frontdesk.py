"""The 'front desk': a fast Claude (Haiku by default) assistant that answers in short spoken sentences, looks things up in the
second brain, and hands real work to the task list. Text only goes to Claude Haiku via `claude -p` (VOICE_MODEL overrides). No audio leaves the Mac."""
import json
import os
import re
import subprocess
import threading
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import config
from diag import log_line
from status import sessions_status, wants_status

HOME = os.path.expanduser("~")
BRAIN_TOOL = os.environ.get("VOICE_BRAIN_TOOL", config.path("brain_tool"))
CLAUDE_BIN_CANDIDATES = [os.path.join(HOME, ".local/bin/claude"), "/opt/homebrew/bin/claude", "/usr/local/bin/claude"]
MAX_HISTORY_TURNS = 10

PROMPT_TEMPLATE = """You are @@ASSISTANT@@, the front desk of @@OWNER@@'s phone-call assistant@@ASSISTANT_NOTE@@. @@OWNER@@ is @@JOB@@.
@@OWNER@@ is talking to you by voice, hands free, so everything you write is spoken aloud.

STYLE
- @@STYLE_LANGUAGE@@
- At most two short sentences. No lists, no markdown, no URLs read out, no emoji. Say numbers plainly.
- Warm, direct, calm. @@STYLE_HEALTH@@

FACTS
- Answer from the NOTES below when they contain the answer. If the question is about their rules, people or past decisions
  and the notes do not contain it, say you do not have a note on that. Never guess a price, date, name or address.
@@SESSIONS@@
- You cannot read email, check the calendar or browse the web yourself during the call. Do not say you cannot: if they ask for any of
  those (their email inbox, today's calendar, a web lookup), say you are handing it over and set "handoff" to a read-only task.
- DONE vs PLANNED (what @@ASSISTANT@@ says must be precise):
  - A SESSIONS STATUS line's "next step" is a PLAN, not work done. Say it as a plan: "CRM plans to ...". Never say a plan was done.
  - Work DONE comes only from RECENT UPDATES lines ("Update from X ... : text"), which the session wrote for them. Say those as done,
    close to their own words, with the time: "At 14:23 Voice said: ...".
  - Keep their words. Do not turn technical words into new meanings; if a phrase is unclear to you, leave it out rather than guess.
  - At most three sessions per turn, most recent first. If there is no block, you cannot see the sessions right now: say so.

GROUNDING (very important)
- Only mention a session, a count, a deadline, or the words "overdue" / "late" / "blocked" if it is written in the SESSIONS STATUS
  block or the NOTES for this turn. Old notes can mention old job numbers: never report them.
- For "status" or "what is going on", use ONLY the SESSIONS STATUS block. Never invent items, counts or dates.
- If you are not sure, say what you do see and say you are not sure.
- Never explain WHY something happened (a cause, "network", "it's normal") unless a line in the notes says it. Hand the question over instead.
- Never say you will check, look, test or fix anything yourself: you only answer and hand over. The session answers later as "Update from ...".
- About this call app (screen, buttons, voice), use only CALL APP FACTS.

CALL APP FACTS
- Bottom buttons: Mute (stops listening to them), Skip (stops you talking), Ears Phone or Mac (which device hears them), the language button
  (the language the phone LISTENS in; it does not change your reply language), End (ends the call).
- Top: your photo (tap to change your look) and a menu button that picks which session their requests go to (Auto = you decide).
- Your voice is always Qwen. Sign-in is the PIN or Face ID.

LIVE FACTS
- A LIVE FACTS block (weather, time) may appear in the notes. Use it for those questions and say the numbers plainly.

NEVER REFUSE
- Never say "I can't", "I cannot", "I don't have access", "I'm unable" or "I don't have live task status". You are the front desk of a set of sessions that CAN do
  almost anything; your job is to say what you will do. If you cannot answer from the NOTES or SESSIONS STATUS, hand the request to ORG and say so
  in one short sentence, for example "Okay, I'm handing that to ORG, it will pass it to the right session." Handing over is always allowed.

WORK
- If they ask for real work (research, a list, a file, code, a draft, anything that needs more than a quick answer),
  say in one short sentence that you handed it over, and set "handoff" to a clear English description of the task.
- When you hand over, say ONLY who it went to and that the session will answer them here, for example "Sent to Voice. It
  will tell you here when it's done." Never describe the fix, the cause or the result, and never promise what the session
  will do: you do not know yet, and guesses that differ from what the session really does confuse the caller.
  The session's real answer comes back later as "Update from ...".
- You never send WhatsApp, SMS or email, and never create, change or delete calendar, Gmail or Drive items. If they ask for
  one of those, say you will prepare a draft for them to approve, and hand it over as "Prepare a draft for approval: ...".
- Never ask for or repeat passwords, codes or 2FA.

OUTPUT FORMAT (strict): one JSON object and nothing else:
{"say": "<what to speak>", "handoff": null}
or
{"say": "<what to speak>", "handoff": "<task for ORG, English, self-contained>"}
"""


def build_system_prompt() -> str:
    """The prompt with this owner's settings filled in (config.py / local_config.json)."""
    return (config.fill(PROMPT_TEMPLATE)
            .replace("@@ASSISTANT_NOTE@@", config.get("assistant_note")).replace("@@JOB@@", config.get("owner_job"))
            .replace("@@STYLE_LANGUAGE@@", config.get("style_language")).replace("@@STYLE_HEALTH@@", config.get("style_extra"))
            .replace("@@SESSIONS@@", config.get("sessions_description")))


SYSTEM_PROMPT = build_system_prompt()


@dataclass
class Reply:
    say: str
    handoff: Optional[str] = None


class FrontDesk:
    def reply(self, history: List[dict], text: str) -> Reply:  # pragma: no cover - interface
        raise NotImplementedError


class FakeFrontDesk(FrontDesk):
    """For tests: echoes, and hands off when the text contains 'research'."""

    def __init__(self):
        self.seen = []

    def reply(self, history, text):
        self.seen.append((list(history), text))
        if "research" in text.lower():
            return Reply("OK, I handed that to ORG.", "Research: " + text)
        return Reply("You said: " + text)


def parse_reply(raw: str) -> Reply:
    """Turn the model output into a Reply. Tolerates code fences and chatter; falls back to plain text."""
    raw = (raw or "").strip()
    if not raw:
        return Reply("Sorry, I did not get that. Please say it again.")
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            say = str(obj.get("say") or "").strip()
            handoff = obj.get("handoff")
            handoff = str(handoff).strip() if handoff else None
            if say:
                return Reply(say, handoff or None)
        except (ValueError, AttributeError):
            pass
    plain = re.sub(r"```[a-z]*", "", raw).strip()
    return Reply(plain[:400])


def build_prompt(history: List[dict], text: str, notes: str, include_rules: bool = True) -> str:
    lines = [SYSTEM_PROMPT if include_rules else "", "NOTES (from the owner's second brain; may be empty):", notes.strip() or "(none)", "", "CALL SO FAR:"]
    for turn in history[-MAX_HISTORY_TURNS * 2:]:
        who = config.get("owner") if turn.get("role") == "user" else "You"
        lines.append("%s: %s" % (who, turn.get("text", "")))
    lines += ["", config.get("owner") + " now says: " + text, "", "Reply with the JSON object only."]
    return "\n".join(lines)


LOOKUP_HINTS = tuple(config.get("lookup_names")) + tuple(config.get("lookup_words"))


def needs_lookup(text: str) -> bool:
    """Searching the second brain costs seconds, so only do it when the question is about the owner's own world."""
    t = text.strip().lower()
    if len(t) < 6:
        return False
    return any(h in t for h in LOOKUP_HINTS)


def brain_notes(query: str, timeout: int = 20, limit: int = 1800) -> str:
    """Search the second brain (local). Returns a short text block, or '' on any problem."""
    if not os.path.exists(BRAIN_TOOL):
        return ""
    try:
        p = subprocess.run([sys.executable, BRAIN_TOOL, "search", query[:200]], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    out = (p.stdout or "").strip()
    return out[:limit]


def team_token() -> str:
    """A long-lived Claude login from the macOS Keychain (item name from config), so a
    LaunchAgent never stops on an expired session. Returns '' when it is not set up; then the normal login is used."""
    if sys.platform != "darwin" or not os.path.exists("/usr/bin/security"):
        return ""
    try:
        p = subprocess.run(["/usr/bin/security", "find-generic-password", "-s", config.get("claude_keychain_item"), "-a",
                            os.environ.get("USER", ""), "-w"], capture_output=True, text=True, timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    tok = (p.stdout or "").strip() if p.returncode == 0 else ""
    return tok if tok.startswith("sk-ant-") else ""


def child_env() -> dict:
    keep = ("HOME", "USER", "LOGNAME", "SHELL", "LANG", "TERM", "SSH_AUTH_SOCK", "__CF_USER_TEXT_ENCODING")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PATH"] = os.pathsep.join([os.path.join(HOME, ".local/bin"), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
                                   os.environ.get("PATH", "")])
    env.update(CLAUDE_CODE_DISABLE_BACKGROUND_TASKS="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    tok = team_token()
    if tok:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
    return env


def find_claude() -> str:
    from shutil import which
    return which("claude") or next((c for c in CLAUDE_BIN_CANDIDATES if os.path.exists(c)), "claude")


def gather_notes(text, notes_fn):
    """What the front desk may rely on this turn. Status questions use ONLY the live sessions status (old notes mention closed jobs and
    made the model invent 'overdue' items); other questions search the second brain; weather and time come from quickfacts."""
    if wants_status(text):
        notes = sessions_status()
        try:
            from updates import recent_lines
            done = recent_lines()
            if done:
                notes = "RECENT UPDATES (work done, the sessions' own words):\n" + done + "\n\n" + notes
        except Exception:
            pass
    elif needs_lookup(text):
        notes = notes_fn(text)
    else:
        notes = ""
    try:
        from quickfacts import quick_facts
        extra = quick_facts(text)
    except Exception:
        extra = ""
    return (notes + "\n\n" + extra).strip() if extra else notes


class StreamClaude:
    """One long-lived `claude -p` process fed over stream-json. It skips the 3-4 s startup of a fresh process on every turn
    (measured: one-shot 4.3 s, persistent 0.9 s per turn). Restarts itself after MAX_TURNS or when a new call begins."""
    MAX_TURNS = 10

    def __init__(self, command, env_fn, workdir, timeout=60):
        self.command, self.env_fn, self.workdir, self.timeout = command, env_fn, workdir, timeout
        self.proc = None
        self.turns = 0
        self.buf = b""
        self.lock = threading.Lock()
        self.err_path = os.path.join(workdir, "stream.err")
        self.last_lines = []

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _start(self):
        self.stop()
        self.buf, self.turns = b"", 0
        err = open(self.err_path, "ab")
        self.proc = subprocess.Popen(self.command(), cwd=self.workdir, env=self.env_fn(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=err, bufsize=0)
        err.close()

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()
            except OSError:
                pass
            try:
                p.terminate()
                p.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    p.kill()
                except OSError:
                    pass

    def needs_fresh(self, history) -> bool:
        return (not self.alive()) or self.turns >= self.MAX_TURNS

    def _readline(self, deadline):
        import select
        import time
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired("claude stream", self.timeout)
            r, _, _ = select.select([fd], [], [], min(left, 1.0))
            if not r:
                if not self.alive():
                    raise RuntimeError("claude stream process exited")
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError("claude stream process closed its output")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def ask(self, prompt: str, fresh: bool = False) -> str:
        import time
        with self.lock:
            if fresh or not self.alive():
                self._start()
            try:
                msg = json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n"
                self.proc.stdin.write(msg.encode("utf-8"))
                deadline = time.monotonic() + self.timeout
                while True:
                    line = self._readline(deadline).strip()
                    if not line:
                        continue
                    self.last_lines = (self.last_lines + [line[:200].decode("utf-8", "replace")])[-4:]
                    try:
                        obj = json.loads(line.decode("utf-8", "replace"))
                    except ValueError:
                        continue
                    if obj.get("type") == "result":
                        if obj.get("is_error") or not str(obj.get("result") or "").strip():
                            raise RuntimeError("claude stream error result: %s" % str(obj)[:300])
                        self.turns += 1
                        return str(obj.get("result"))
            except BaseException:
                rc = self.proc.poll() if self.proc is not None else None
                log_line("front desk stream died: exit=%s last output=%s" % (rc, " || ".join(self.last_lines)[-500:]))
                self.stop()   # a half-read stream cannot be trusted; the next turn starts a clean process
                try:
                    with open(self.err_path, "rb") as f:
                        f.seek(0, 2)
                        f.seek(max(0, f.tell() - 600))
                        log_line("front desk stream stderr tail: " + f.read().decode("utf-8", "replace").strip()[-500:])
                except OSError:
                    pass
                raise


class ClaudeFrontDesk(FrontDesk):
    def __init__(self, model=None, timeout=60, notes_fn=brain_notes, runner=None):
        self.model = model or os.environ.get("VOICE_MODEL", "haiku")   # the front desk only speaks 1-2 short sentences: the fastest model is enough
        self.timeout = timeout
        self.notes_fn = notes_fn
        self.runner = runner or self._run
        self.workdir = tempfile.mkdtemp(prefix="voice-call-")
        self._fast = None   # unknown until the first call works
        self._stream = None
        self.use_stream = runner is None and os.environ.get("VOICE_NO_STREAM") != "1"

    FAST_FLAGS = ["--system-prompt", SYSTEM_PROMPT, "--tools", "", "--setting-sources", "project", "--disable-slash-commands"]

    def _command(self, prompt: str, fast: bool) -> list:
        cmd = [find_claude(), "-p", prompt, "--output-format", "json", "--model", self.model,
               "--permission-mode", "dontAsk", "--max-turns", "1", "--strict-mcp-config",
               "--settings", json.dumps({"enabledPlugins": {k: False for k in config.get("claude_disabled_plugins")}})]
        return cmd + (self.FAST_FLAGS if fast else [])

    def _run(self, prompt: str) -> str:
        """One Claude call. The fast flags skip the user's CLAUDE.md, hooks, tools and the big default prompt (that startup
        cost tens of seconds). If this version of claude rejects a flag, fall back to the plain command once and remember."""
        env = child_env()
        last = None
        for fast in ([self._fast] if self._fast is not None else [True, False]):
            p = subprocess.run(self._command(prompt, fast), cwd=self.workdir, env=env, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=self.timeout)
            out = (p.stdout or "").strip()
            result, is_error = "", p.returncode != 0
            try:
                obj = json.loads(out.splitlines()[-1])
                result = str(obj.get("result") or "")
                is_error = is_error or bool(obj.get("is_error"))
            except (ValueError, IndexError, AttributeError):
                is_error = True
            if not is_error and result.strip():
                self._fast = fast
                return result
            last = "claude failed (exit %s, fast=%s): %s | %s" % (p.returncode, fast, (p.stderr or "")[:300].strip(), out[:300])
            log_line("front desk: " + last)
        raise RuntimeError(last)

    def _stream_command(self):
        return [find_claude(), "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--model", self.model, "--permission-mode", "dontAsk", "--strict-mcp-config",
                "--settings", json.dumps({"enabledPlugins": {k: False for k in config.get("claude_disabled_plugins")}})] + self.FAST_FLAGS

    def _ask_stream(self, history, text, notes):
        if self._stream is None:
            self._stream = StreamClaude(self._stream_command, child_env, self.workdir, timeout=self.timeout)
        st = self._stream
        fresh = st.needs_fresh(history)
        # a fresh process knows nothing, so it gets the recent turns; a warm one already has them
        prompt = build_prompt(history if fresh else [], text, notes, include_rules=False)
        if not history and not fresh and st.turns > 0:
            prompt = "NEW CALL: " + config.get("owner") + " just started a new call. Earlier calls no longer matter.\n" + prompt   # no restart: a restart is a cold start
        return st.ask(prompt, fresh=fresh)

    def prepare(self):
        """A call just connected. If the persistent process is alive, only touch it (so it is not swapped out); otherwise start one.
        Never restart a warm process: a cold start of claude takes 5-100 s on this Mac."""
        if not self.use_stream:
            return
        if self._stream is not None and self._stream.alive() and self._stream.turns < StreamClaude.MAX_TURNS:
            self.ping()
        else:
            self.warm()

    def ping(self):
        """One tiny turn that keeps the process in memory. It does not count as a conversation turn."""
        st = self._stream
        if st is None or not st.alive():
            return
        try:
            before = st.turns
            st.ask('Keep-alive. Reply with the JSON object {"say": "", "handoff": null} only.')
            st.turns = before
        except Exception as e:
            log_line("front desk ping failed: %s: %s" % (type(e).__name__, str(e)[:200]))

    def warm(self):
        """Start the persistent process and run one tiny turn so the first real question is fast."""
        if not self.use_stream:
            return
        try:
            if self._stream is None:
                self._stream = StreamClaude(self._stream_command, child_env, self.workdir, timeout=self.timeout)
            self._stream.ask('Warm-up. Reply with the JSON object {"say": "ok", "handoff": null} only.', fresh=True)
            self._stream.turns = 0   # the warm-up does not count: the first real call keeps this process
        except Exception as e:
            log_line("front desk warm-up failed: %s: %s" % (type(e).__name__, str(e)[:300]))

    def reply(self, history, text):
        import time
        t0 = time.monotonic()
        notes = gather_notes(text, self.notes_fn)
        t1 = time.monotonic()
        self.last_timing = {"notes_ms": int((t1 - t0) * 1000), "notes_chars": len(notes)}
        try:
            raw = None
            if self.use_stream:
                try:
                    raw = self._ask_stream(history, text, notes)
                except Exception as e:
                    log_line("front desk stream failed, using one-shot: %s: %s" % (type(e).__name__, str(e)[:300]))
            if raw is None:
                raw = self.runner(build_prompt(history, text, notes, include_rules=(getattr(self, '_fast', None) is not True)))
            self.last_timing["llm_ms"] = int((time.monotonic() - t1) * 1000)
        except subprocess.TimeoutExpired:
            log_line("front desk: claude timed out after %ss" % self.timeout)
            return Reply("Sorry, that took too long. Please ask again.")
        except Exception as e:
            log_line("front desk: %s: %s" % (type(e).__name__, str(e)[:600]))
            return Reply("Sorry, my brain is not reachable right now.")
        return parse_reply(raw)


class QwenFrontDesk(FrontDesk):
    """Front desk answered by a Qwen chat model (about 1 s per answer, measured). Off unless brain.txt says "qwen":
    what the caller says, the second-brain notes and the session status then go to Alibaba Cloud (the owner must opt in)."""
    BASES = ["https://dashscope.aliyuncs.com/compatible-mode/v1", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"]

    def __init__(self, key=None, model=None, post=None, notes_fn=brain_notes):
        from tts import qwen_key
        self.key = key or qwen_key()
        self.model = model or os.environ.get("VOICE_QWEN_CHAT", "qwen-plus")
        self.notes_fn = notes_fn
        self.post = post or self._post
        self._good = None
        # A far-away region -> Beijing usually answers in ~2 s but sometimes stalls 14-40 s (measured): give up fast, Claude answers.
        # The answer is not streamed, so this covers the whole answer: 5 s cut off longer answers to long questions
        # and every such turn fell back to Haiku at ~20 s. Normal answers take ~2 s.
        self.timeout = float(os.environ.get("VOICE_QWEN_TIMEOUT", "10"))
        self.last_timing = {}

    def _post(self, base, body):
        import urllib.request
        req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _hedged_post(self, base, body):
        """Beijing sometimes sits on a request for 10-40 s while a second identical one answers in ~2 s (measured): if
        there is no answer after VOICE_QWEN_HEDGE_CHAT_S, a second request starts and the first answer wins."""
        import queue
        import threading
        import time
        hedge = float(os.environ.get("VOICE_QWEN_HEDGE_CHAT_S", "3.5"))
        done, t0, errors, started = queue.Queue(), time.monotonic(), [], 1

        def attempt():
            try:
                done.put(self.post(base, body))
            except Exception as e:
                done.put(e)
        threading.Thread(target=attempt, daemon=True).start()
        while True:
            left = self.timeout - (time.monotonic() - t0)
            if left <= 0:
                raise TimeoutError("Qwen chat: no answer in %.0f s" % self.timeout)
            wait = min(left, hedge - (time.monotonic() - t0)) if started == 1 else left
            try:
                got = done.get(timeout=max(0.05, wait))
            except queue.Empty:
                if started == 1:
                    started = 2
                    threading.Thread(target=attempt, daemon=True).start()
                continue
            if not isinstance(got, Exception):
                return got
            errors.append(got)
            if started == 1:
                started = 2
                threading.Thread(target=attempt, daemon=True).start()
            elif len(errors) >= 2:
                raise errors[-1]

    def prepare(self):
        """One tiny request when a call opens: picks the working region and wakes the route before the first question."""
        body = {"model": self.model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        for base in ([self._good] if self._good else self.BASES):
            try:
                self.post(base, body)
                self._good = base
                return
            except Exception:
                continue

    def reply(self, history, text):
        import time
        t0 = time.monotonic()
        notes = gather_notes(text, self.notes_fn)
        t1 = time.monotonic()
        body = {"model": self.model, "max_tokens": 220, "temperature": 0.1,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": build_prompt(history, text, notes, include_rules=False)}]}
        last = None
        for base in ([self._good] if self._good else self.BASES):
            try:
                out = self._hedged_post(base, body)
                self._good = base
                self.last_timing = {"notes_ms": int((t1 - t0) * 1000), "notes_chars": len(notes), "llm_ms": int((time.monotonic() - t1) * 1000)}
                return parse_reply(out["choices"][0]["message"]["content"])
            except Exception as e:
                last = e
        raise RuntimeError("Qwen chat failed: %s" % type(last).__name__)


class QwenAgainFrontDesk(FrontDesk):
    """Qwen-only mode: when the first
    Qwen answer fails, Qwen gets one more full try; if that fails too the caller hears a short sorry, never another model."""

    def __init__(self, qwen):
        self.qwen = qwen
        self.last_timing = {}

    def reply(self, history, text):
        try:
            r = self.qwen.reply(history, text)
            self.last_timing = getattr(self.qwen, "last_timing", {})
            return r
        except Exception as e:
            log_line("Qwen chat failed twice (%s)" % type(e).__name__)
            return Reply("Sorry, Qwen did not answer. Please say that again.")

    def prepare(self):
        pass


class ChainFrontDesk(FrontDesk):
    """Try the fast front desk first; if it fails, the other one answers, so a question is never lost."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.last_timing = {}

    def reply(self, history, text):
        try:
            r = self.primary.reply(history, text)
            self.last_timing = getattr(self.primary, "last_timing", {})
            return r
        except Exception as e:
            log_line("front desk primary failed (%s), using the fallback" % type(e).__name__)
            r = self.fallback.reply(history, text)
            self.last_timing = getattr(self.fallback, "last_timing", {})
            return r

    def prepare(self):
        for desk in (self.primary, self.fallback):   # warm both: a cold first Qwen call took 12 s
            try:
                getattr(desk, "prepare", lambda: None)()
            except Exception as e:
                log_line("front desk prepare failed: %s" % type(e).__name__)

    def ping(self):
        getattr(self.fallback, "ping", lambda: None)()

    def warm(self):
        getattr(self.fallback, "warm", lambda: None)()
