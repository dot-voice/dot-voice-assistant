"""R30: a second opinion from a Claude model during a call ("ask Opus", "what does Sonnet think").
One headless `claude -p` run per question: no tools, no MCP, no settings, no session saved. Text only."""
import asyncio
import time

MODELS = {"opus": "opus", "sonnet": "sonnet", "haiku": "haiku"}
DEFAULT_MODEL = "opus"
TIMEOUT_S = 60.0
CONTEXT_TURNS = 6
SYSTEM_PROMPT = ("You are a concise advisor to a busy Toronto-area realtor who is driving. "
                 "Answer in 2-4 short spoken sentences, no lists, no markdown.")


def pick_model(name):
    """'opus' / 'Sonnet' / '' -> (key, None); an unknown name -> (None, message listing the ones available)."""
    key = (name or "").strip().lower()
    if not key:
        return DEFAULT_MODEL, None
    if key in MODELS:
        return key, None
    return None, "Unknown model. Available: Opus, Sonnet, Haiku."


def build_prompt(question, history):
    lines = []
    for h in (history or [])[-CONTEXT_TURNS:]:
        text = str(h.get("text") or "").strip()
        if text:
            lines.append(("Caller: " if h.get("role") == "user" else "Assistant: ") + text)
    ctx = ("Recent call context:\n" + "\n".join(lines) + "\n\n") if lines else ""
    return ctx + "Question: " + question.strip()


def command(key, prompt):
    import frontdesk
    return [frontdesk.find_claude(), "-p", prompt, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--disable-slash-commands", "--no-session-persistence", "--model", MODELS[key],
            "--system-prompt", SYSTEM_PROMPT]


async def ask(key, question, history, timeout=TIMEOUT_S):
    """Returns (status, text, ms); status is ok / timeout / error. Never raises (except cancellation)."""
    t0 = time.monotonic()
    proc = None
    try:
        import frontdesk
        proc = await asyncio.create_subprocess_exec(
            *command(key, build_prompt(question, history)), env=frontdesk.child_env(),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        text = (out or b"").decode("utf-8", "replace").strip()
        status = "ok" if proc.returncode == 0 and text else "error"
    except asyncio.TimeoutError:
        status, text = "timeout", ""
        _kill(proc)
    except asyncio.CancelledError:
        _kill(proc)
        raise
    except Exception:
        status, text = "error", ""
        _kill(proc)
    return status, text, int((time.monotonic() - t0) * 1000)


def _kill(proc):
    if proc is not None and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
