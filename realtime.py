"""Qwen-Omni realtime Live mode bridge (DashScope OpenAI-realtime-style WebSocket)."""
import asyncio
import base64
import json
import os
import time

import websockets

import config
import tts

URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model="
DEFAULT_MODEL = "qwen3.5-omni-flash-realtime"
DEFAULT_VOICE = "Tina"


def _cfg(key):
    """config.get that never raises on a missing key."""
    try:
        return config.get(key)
    except (KeyError, Exception):
        return None


def _agents():
    try:
        names = [str(t[0]) for t in (_cfg("handoff_targets") or [])]
    except Exception:
        names = []
    return [n for n in names if n != "AUTO"]


def _role_lines():
    """R35: one line per role, 'KEY = label', from the labels in handoff_targets, so the model can match a request to a role."""
    try:
        return "; ".join("%s = %s" % (t[0], t[1]) for t in (_cfg("handoff_targets") or []) if t[0] != "AUTO")
    except Exception:
        return ""


def build_tools():
    lines = _role_lines()
    agent = {"type": "string", "description": "Which session gets the work. Match the request to the role's label: " + lines
             if lines else "Which session gets the work."}
    names = _agents()
    if names:
        agent["enum"] = names
    return [
        {"type": "function", "name": "hand_off",
         "description": "Pass real work or a request to the right agent session. It waits a few seconds before it "
                        "is sent: if the caller corrects it (another session, more detail), call hand_off again and "
                        "it replaces the waiting one.",
         "parameters": {"type": "object", "properties": {"agent": agent, "replaces_waiting": {
             "type": "boolean", "description": "True only when the caller corrects the hand-off you just made."},
             "summary": {"type": "string", "description": "One short sentence in English saying what she asked. Always give it "
                         "when she spoke Chinese (the exact words cannot be posted); otherwise leave it out."}},
             "required": []}},
        {"type": "function", "name": "get_status",
         "description": "Status, tasks, progress or updates of the agent sessions. Pass session when one is named; "
                        "leave it out for all of them.",
         "parameters": {"type": "object", "properties": {"session": dict(agent, description="The one session asked about, if any.")}}},
        {"type": "function", "name": "cancel_hand_off",
         "description": "The caller says not to send the hand-off just made.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "transfer_call",
         "description": "The caller asks to be transferred, put through or switched to a channel or session (or back to "
                        "Dot, or to the previous channel). Pass the name she said. Never use hand_off for that.",
         "parameters": {"type": "object", "properties": {"channel": {
             "type": "string", "description": "The channel name as she said it, e.g. vault, website, market watch; \"Dot\" or \"back\" for the front desk; \"previous\" for the channel she was on before."}},
             "required": ["channel"]}},
        {"type": "function", "name": "lookup",
         "description": "Look up weather, time, notes, rules, people, properties, or past decisions.",
         "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "What to look up."}}, "required": ["question"]}},
        {"type": "function", "name": "ask_model",
         "description": "A second opinion from a stronger Claude model. Call it when the caller says ask Opus, ask "
                        "Sonnet, ask Haiku, what does Opus think, Sonnet's opinion, or asks for a second opinion. "
                        "It returns at once and the answer is spoken a few seconds later.",
         "parameters": {"type": "object", "properties": {
             "model": {"type": "string", "description": "Opus, Sonnet or Haiku, as the caller named it; Opus if none."},
             "question": {"type": "string", "description": "The question to put to that model, complete in itself."}},
             "required": ["question"]}},
    ]


TOOLS = build_tools()


def build_instructions(owner, channel=None):
    note = ""
    if channel and channel not in ("AUTO", "VOICE"):   # R28: she was transferred to a channel (a new Live session keeps it)
        note = (f" The caller is on the {channel} channel; real work goes to {channel} (hand_off with no agent goes there). "
                "To change channel call transfer_call.")
    return (
        f"You are the assistant of {owner} on a hands-free call. Reply in the language spoken, "
        "in one or two short sentences. For any real work call hand_off. Never claim work is done "
        "unless a tool output says so. If the caller interrupts you, she did not hear the rest of your reply. After hand_off say exactly the line its result gives, nothing more. For status questions "
        "call get_status and say its output in at most two short sentences, in your own words without losing a fact; if it has no answer say you have no data for that, never guess. For weather, time, the caller's rules, people, "
        "properties or past decisions call lookup and answer only from its output (a property's owner, MLS number, listing agent or address facts: call lookup FIRST and say its answer; only when it found nothing call hand_off, to the session whose label is about property records); reminders and personal tasks go to the personal or life session, never a property session; never say you have "
        "no real-time access. What the caller already said in this call counts: answer that from the conversation, "
        "without lookup. When the caller says ask Opus, ask Sonnet, what does Opus think, Sonnet's opinion or "
        "a second opinion, call ask_model (never answer it yourself), then say only that you are asking; "
        "the answer is spoken when it arrives. Never invent facts. If the caller asks to be transferred, put through or switched to a channel or "
        "session, back to Dot, or to the previous channel, call transfer_call (never hand_off) and say exactly the line its result gives. "
        "If hand_off returns \"not sent yet\" with a question, ask exactly that question and nothing else: on yes call transfer_call, "
        "on no or anything else call hand_off again. Never offer a transfer on your own, never for status questions, never when the caller "
        "is already on a channel." + note
    )


def history_note(seed):
    """The call so far, for a provider that cannot be seeded with turns (Qwen): appended to its instructions."""
    if not seed:
        return ""
    return "\n\nConversation so far in this call (continue it, do not repeat it):\n" + "\n".join(
        ("Caller: " if line["role"] == "user" else "You: ") + line["text"] for line in seed)


def cut_text(said, fraction):
    """The part of a reply she heard before she interrupted it."""
    said = (said or "").strip()
    if fraction >= 0.98 or not said:
        return said
    words = said.split()
    if len(words) > 1:
        heard = " ".join(words[:int(len(words) * max(0.0, fraction))])
    else:   # Chinese: no spaces
        heard = said[:int(len(said) * max(0.0, fraction))]
    return (heard + " ... (she interrupted)") if heard else "(she interrupted before hearing this reply)"


_DOWN_UNTIL = {}   # provider -> monotonic time it may be tried again (quota used up)


def mark_down(provider, seconds):
    _DOWN_UNTIL[provider] = time.monotonic() + seconds


def bridge_classes():
    """Live providers to try in order. live_provider "gemini": the second provider first, Qwen when it fails to start
    or its free quota is used up (then Qwen only, until the pause ends)."""
    if _cfg("live_provider") == "gemini" and _DOWN_UNTIL.get("gemini", 0) <= time.monotonic():
        import gemini_live
        return [gemini_live.GeminiBridge, LiveBridge]
    return [LiveBridge]


def bridge_class():
    return bridge_classes()[0]


class TurnSnap:
    """R32: her words of the turn that ended in a tool call, frozen when the call arrives. Dot's own next turn
    (turn_complete after the tool call) wipes the live turn state while the tool may still wait on Whisper; the tool reads
    this record instead. Only the record is frozen: what she says after that is a new turn and never merges into it."""

    def __init__(self, turn_no, pcm, transcript, words, whisper):
        self.turn_no = turn_no
        self.pcm = pcm
        self.transcript = transcript
        self.words = words
        self.whisper = whisper
        self.done = asyncio.Event()
        if transcript:
            self.done.set()


class LiveBridge:
    def __init__(self, send_to_phone, tools_impl, instructions, model=None, connect=websockets.connect,
                 transcribe=None, on_closed=None, turn_log=None):
        self.send_to_phone = send_to_phone
        self.tools_impl = tools_impl
        self.instructions = instructions
        self.model = model or _cfg("live_model") or DEFAULT_MODEL
        self.voice = _cfg("live_voice") or DEFAULT_VOICE
        self._connect = connect
        self.transcribe = transcribe
        self.on_closed = on_closed
        self.turn_log = turn_log
        self.ws = None
        self.reader = None
        self.tool_task = None
        self.last_heard = ""
        self.stopped_at = None
        self._transcript_done = asyncio.Event()
        self.response_active = False
        self._closing = False
        self.closed = False
        self._failed = False

        # Audio buffering for local Whisper
        self._preroll = bytearray()
        self._turn_buf = bytearray()
        self.turn_pcm = b""
        self._in_turn = False
        self._turn_transcript = None
        self._turn_words = None
        self._whisper_words = None
        self._snap = None           # R32: TurnSnap of the turn whose tool call is running
        self._last_snap = None      # R32: (TurnSnap, speech counter, monotonic) of the previous tool call
        self._speech_n = 0          # counts moments of her speech: a tool call with none since the last one is the same request
        self._last_handoff = None   # R33: (monotonic, turn_no, result) of the last hand-off this bridge returned

        # State tracking
        self._user_speaking = False
        self.current_response_id = None
        self.cancelled_response_ids = set()
        self.handoff_count = 0
        self.seed = []              # the call so far (server sets it before start): a new session must not forget it
        self.language = "en-US"
        self._turn_no = 0           # her turns, so a second version of her words replaces the first in memory
        self._reply_no = 0          # Dot's replies

    @property
    def user_speaking(self):
        if self._user_speaking:
            if self.stopped_at is not None and (time.monotonic() - self.stopped_at) >= 2.0:
                self._user_speaking = False
                return False
            return True
        return False

    @property
    def busy(self):
        tool_running = bool(self.tool_task is not None and not self.tool_task.done())
        return bool(self.user_speaking or self.response_active or tool_running)

    async def _send(self, event):
        if self.closed or self.ws is None:
            return
        try:
            await self.ws.send(json.dumps(event))
        except Exception as e:
            await self._fail(str(e) or type(e).__name__)

    async def _expect(self, etype):
        while True:
            ev = json.loads(await self.ws.recv())
            if ev.get("type") == etype:
                return ev
            if ev.get("type") == "error":
                raise RuntimeError(str((ev.get("error") or {}).get("message") or "server error")[:120])

    transcript_wait_s = 1.5   # how long a tool waits for Qwen's transcript before running Whisper

    async def start(self):
        key = tts._keychain(tts.QWEN_KEYCHAIN)
        self.ws = await self._connect(URL + self.model, additional_headers={"Authorization": "Bearer " + key})
        await self._expect("session.created")
        session_cfg = {
            "modalities": ["text", "audio"],
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm24",
            # semantic_vad waits for the end of the thought, not just silence (Alibaba recommends it for 3.5-omni)
            "turn_detection": {"type": _cfg("live_vad") or "semantic_vad", "threshold": 0.5,
                               "silence_duration_ms": int(_cfg("live_silence_ms") or 900)},
            "tools": TOOLS,
            "tool_choice": "auto",
            "instructions": self.instructions + history_note(self.seed),
        }
        m = _cfg("live_transcription_model")
        if m != "off":   # the documented (fixed) model for the omni realtime series; Whisper stays the fallback
            session_cfg["input_audio_transcription"] = {"model": "qwen3-asr-flash-realtime"}
        await self._send({"type": "session.update", "session": session_cfg})
        await self._expect("session.updated")
        self.reader = asyncio.ensure_future(self._read_loop())

    async def feed_audio(self, pcm):
        if self.closed or self.ws is None:
            return
        # Keep ~0.5 s (16000 bytes at 16 kHz 16-bit mono) as pre-roll
        self._preroll = (self._preroll + pcm)[-16000:]
        if self._in_turn:
            if len(self._turn_buf) < 960000:
                self._turn_buf.extend(pcm)
                if len(self._turn_buf) > 960000:
                    self._turn_buf = self._turn_buf[:960000]
        try:
            await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        except Exception as e:
            await self._fail(str(e) or type(e).__name__)

    async def say_exact(self, text):
        if self.closed or self.ws is None:
            return
        await self._send({
            "type": "response.create",
            "response": {
                "instructions": f"Say exactly this, word for word, nothing else: {text}",
                "modalities": ["text", "audio"],
            }
        })
        self.response_active = True

    async def cancel(self):
        if self.response_active and self.ws is not None:
            if self.current_response_id:
                self.cancelled_response_ids.add(self.current_response_id)
            await self._send({"type": "response.cancel"})
            self.response_active = False

    async def close(self):
        self._closing = True
        self.closed = True
        if self.reader and self.reader is not asyncio.current_task():
            self.reader.cancel()
        if getattr(self, "tool_task", None) and self.tool_task is not asyncio.current_task():
            self.tool_task.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _read_loop(self):
        try:
            async for raw in self.ws:
                await self._handle(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closing and not self.closed:
                await self._fail(str(e) or type(e).__name__)
            return
        if not self._closing and not self.closed:
            await self._fail("connection closed")

    async def _fail(self, reason):
        if self._failed:
            return
        self._failed = True
        self.closed = True
        try:
            await self.send_to_phone({"type": "live_error", "text": str(reason)[:120]})
        except Exception:
            pass
        if self.ws is not None:
            try:
                asyncio.ensure_future(self.ws.close())   # do not leak the provider socket
            except Exception:
                pass
        if self.on_closed:
            try:
                res = self.on_closed()
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception:
                pass

    async def _handle(self, ev):
        t = ev.get("type", "")
        if t == "response.created":
            resp_id = (ev.get("response") or {}).get("id") or ev.get("response_id")
            if resp_id:
                self.current_response_id = resp_id
        elif t == "response.audio.delta":
            resp_id = ev.get("response_id") or (ev.get("response") or {}).get("id")
            if resp_id:
                self.current_response_id = resp_id
            if resp_id and resp_id in self.cancelled_response_ids:
                return
            self.response_active = True
            await self.send_to_phone(base64.b64decode(ev.get("delta", "")))
        elif t == "input_audio_buffer.speech_started":
            self._turn_no += 1
            self._user_speaking = True
            self.stopped_at = None
            self._transcript_done.clear()
            self._turn_transcript = None
            self._turn_words = None
            self._whisper_words = None
            self.turn_pcm = b""
            self._turn_buf = bytearray(self._preroll)
            self._in_turn = True
            if self.response_active:
                if self.current_response_id:
                    self.cancelled_response_ids.add(self.current_response_id)
                await self._send({"type": "response.cancel"})
                self.response_active = False
            await self.send_to_phone({"type": "live_stop"})
        elif t == "input_audio_buffer.speech_stopped":
            self.stopped_at = time.monotonic()
            self._in_turn = False
            self.turn_pcm = bytes(self._turn_buf)
        elif t == "conversation.item.input_audio_transcription.completed":
            self._turn_transcript = ev.get("transcript", "")
            self.last_heard = self._turn_transcript
            self._turn_words = self._turn_transcript
            self._transcript_done.set()
            self._feed_snap(self._turn_transcript, replace=True)
            await self.send_to_phone({"type": "live_heard", "text": self.last_heard, "turn": self._turn_no})
        elif t == "response.audio_transcript.done":
            resp_id = ev.get("response_id") or (ev.get("response") or {}).get("id")
            if not (resp_id and resp_id in self.cancelled_response_ids):
                self._reply_no += 1
                await self.send_to_phone({"type": "live_said", "text": ev.get("transcript", ""), "reply": self._reply_no})
        elif t == "response.done":
            self.response_active = False
            self._user_speaking = False
        elif t == "response.function_call_arguments.done":
            self._snap = self._snapshot_turn()
            self.tool_task = asyncio.create_task(self._run_tool(ev))
        elif t == "error":
            err_data = ev.get("error") if isinstance(ev.get("error"), dict) else {}
            err_msg = str(err_data.get("message") or ev.get("message") or "server error")
            low = err_msg.lower()
            if any(ign in low for ign in ("cancel", "active response", "no response")):
                try:
                    from diag import log_line
                    log_line("live provider error (ignored): %s" % err_msg[:120])
                except Exception:
                    pass
            else:
                await self._fail(err_msg)

    def _snapshot_turn(self):
        pcm = self.turn_pcm or (bytes(self._turn_buf) if self._turn_buf else b"")
        return TurnSnap(self._turn_no, pcm, self._turn_transcript, self._turn_words, self._whisper_words)

    def _frozen_for_call(self):
        """The TurnSnap for a tool call. A second tool call of the same request (a lookup, then the hand-off: Gemini sends them as
        separate calls, and Dot's turn in between wiped the live state) reuses the first one's words when she has not spoken since."""
        snap = self._snapshot_turn()
        last = self._last_snap
        if (not snap.transcript and not snap.whisper and last is not None and last[1] == self._speech_n
                and time.monotonic() - last[2] < 60.0):
            snap = last[0]
        self._last_snap = (snap, self._speech_n, time.monotonic())
        return snap

    def _feed_snap(self, text, replace=False):
        """Words that reach the bridge after the tool call started (Gemini's transcript is often late) join the snapshot."""
        s = self._snap
        if s is None or not text:
            return
        s.transcript = text.strip() if replace else ((s.transcript or "") + " " + text).strip()
        s.done.set()

    async def _get_turn_words(self):
        """Her exact words for a hand-off. Local Whisper of the turn audio first: Qwen's separate ASR sometimes
        returns the wrong language. Qwen's
        transcript is the fallback when Whisper is missing or hears nothing. Works on the frozen TurnSnap of the turn
        that called the tool (R32), so Dot's next turn starting cannot wipe it."""
        snap = self._snap or self._snapshot_turn()
        if snap.whisper is not None:
            return snap.whisper
        if self.transcribe is not None:
            try:
                words = str(await asyncio.to_thread(self.transcribe, snap.pcm) or "").strip()
            except Exception:
                words = ""
            if words:
                snap.whisper = snap.words = words
                if snap.turn_no == self._turn_no:   # still her current turn: the live state agrees
                    self._whisper_words = self.last_heard = self._turn_words = words
                await self.send_to_phone({"type": "live_heard", "text": words, "turn": snap.turn_no})
                return words
        if not snap.done.is_set() and not self._transcript_done.is_set():
            try:
                await asyncio.wait_for(snap.done.wait(),
                                       timeout=2.5 if self.transcribe is None else self.transcript_wait_s)
            except (asyncio.TimeoutError, Exception):
                self._log_line("live: no words (Whisper empty, Qwen transcript late)")
        words = snap.transcript or (self._turn_transcript if snap.turn_no == self._turn_no else "") or ""
        if not words and snap.words:
            words = snap.words
        if words and snap.turn_no == self._turn_no:
            self.last_heard = self._turn_words = words
        return words or ""

    def _log_line(self, text):
        path = self.turn_log or getattr(getattr(getattr(self.tools_impl, "call", None), "svc", None), "turn_log", None)
        if path:
            try:
                with open(path, "a") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
            except OSError:
                pass

    def _log_tool(self, name, duration_ms):
        path = self.turn_log
        if not path and hasattr(self.tools_impl, "call"):
            path = getattr(getattr(self.tools_impl.call, "svc", None), "turn_log", None)
        if path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"live_tool={name} tool_ms={duration_ms}\n")
            except OSError:
                pass

    async def _exec_tool(self, name, args):
        owns = self._snap is None
        if owns:
            self._snap = self._snapshot_turn()
        try:
            return await self._exec_tool_inner(name, args)
        finally:
            if owns:
                self._snap = None

    DUP_HANDOFF_S = 10.0

    def _duplicate_handoff(self, args):
        """R33: Gemini sometimes calls hand_off twice for one request (one turn, or a second call right after the first's
        result with no new speech from her). The second one returns the first's result and sends nothing."""
        last = self._last_handoff
        if not last or args.get("replaces_waiting"):
            return None
        t, turn_no, out = last
        if time.monotonic() - t > self.DUP_HANDOFF_S:
            return None
        snap = self._snap
        same_turn = snap is not None and snap.turn_no == turn_no
        no_new_speech = (snap is None or not snap.transcript) and not str(getattr(self, "_heard", "") or "").strip()
        return out if (same_turn or no_new_speech) else None

    async def _exec_tool_inner(self, name, args):
        snap = self._snap
        if name == "hand_off":
            if self.handoff_count >= 6:
                return "limit reached, tell the caller to use the menu"
            dup = self._duplicate_handoff(args)
            if dup:
                return "already handled, do not repeat it. " + dup
            words = await self._get_turn_words()
            if not words or len(words) < 2:
                return "not sent: I did not catch the words, ask the caller to say it again"
            kw = {"replaces_waiting": True} if args.get("replaces_waiting") else {}
            if str(args.get("summary") or "").strip():
                kw["summary"] = str(args["summary"]).strip()[:300]
            out = await self.tools_impl.hand_off(words, args.get("agent"), **kw)
            if str(out).startswith(("posted", "queued")):   # a correction ("updated") is not a new hand-off
                self.handoff_count += 1
            if str(out).startswith(("posted", "queued", "updated")):
                self._last_handoff = (time.monotonic(), snap.turn_no, str(out))
            return out
        elif name == "transfer_call":
            fn = getattr(self.tools_impl, "transfer", None)
            if not fn:
                return "failed: transfers are not available"
            return await fn(str(args.get("channel") or ""), (snap.transcript or "").strip())
        elif name == "cancel_hand_off":
            cancel = getattr(self.tools_impl, "cancel_hand_off", None)
            return await cancel() if cancel else "nothing to cancel"
        elif name == "get_status":   # the model picks the session; her words only help pick it
            if not snap.transcript and not snap.done.is_set():
                try:   # her words of THIS turn may still be on their way (never the last turn's words)
                    await asyncio.wait_for(snap.done.wait(), timeout=1.5)
                except (asyncio.TimeoutError, Exception):
                    pass
            q = snap.transcript or ""
            return await self.tools_impl.status(q, session=args.get("session"))
        elif name == "lookup":   # the model's question (it understood the audio) plus the heard words (exact names)
            q = " ".join(x for x in ((args.get("question") or "").strip(), (snap.transcript or "").strip()) if x)
            if len(q) < 2:
                q = await self._get_turn_words()
            lookup_fn = getattr(self.tools_impl, "lookup", None)
            if lookup_fn:
                return await lookup_fn(q)
            return "Nothing found."
        elif name == "ask_model":   # R30: returns at once; the answer is spoken later (server.LiveTools.ask_model)
            q = (args.get("question") or "").strip()
            if len(q) < 2:
                q = await self._get_turn_words()
            ask_fn = getattr(self.tools_impl, "ask_model", None)
            if not ask_fn:
                return "Asking another model is not available."
            return await ask_fn(args.get("model"), q)
        else:
            return "failed: unknown tool"

    async def _run_tool(self, ev):
        snap = self._snap
        try:
            name, call_id = ev.get("name"), ev.get("call_id")
            try:
                args = json.loads(ev.get("arguments") or "{}")
            except ValueError:
                args = {}
            t0 = time.monotonic()
            try:
                out = await asyncio.wait_for(self._exec_tool(name, args), timeout=20.0)
            except asyncio.TimeoutError:
                out = "timed out"
            except Exception as e:
                out = "failed: " + (str(e) or type(e).__name__)[:100]
            dur_ms = int((time.monotonic() - t0) * 1000)
            self._log_tool(name, dur_ms)

            await self._send({"type": "conversation.item.create",
                              "item": {"type": "function_call_output", "call_id": call_id, "output": str(out)}})
            if self._in_turn:
                # She is talking while the tool finished: do not talk over her. The result is in the
                # conversation; server VAD answers when she stops, with the result in context.
                return
            await self._send({"type": "response.create"})
            self.response_active = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closing and not self.closed:
                await self._fail(str(e) or type(e).__name__)
        finally:
            if snap is not None and self._snap is snap:
                self._snap = None
