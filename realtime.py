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


def build_tools():
    agent = {"type": "string", "description": "Which session gets the work."}
    names = _agents()
    if names:
        agent["enum"] = names
    return [
        {"type": "function", "name": "hand_off",
         "description": "Pass real work or a request to the right agent session.",
         "parameters": {"type": "object", "properties": {"agent": agent}, "required": []}},
        {"type": "function", "name": "get_status",
         "description": "Get the current status of the agent sessions.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "lookup",
         "description": "Look up weather, time, notes, rules, people, properties, or past decisions.",
         "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "What to look up."}}, "required": ["question"]}},
    ]


TOOLS = build_tools()


def build_instructions(owner):
    return (
        f"You are the assistant of {owner} on a hands-free call. Reply in the language spoken, "
        "in one or two short sentences. For any real work call hand_off. Never claim work is done "
        "unless a tool output says so. After hand_off say only who it went to. For status questions "
        "call get_status and read its output word for word. For weather, time, the caller's rules, people, "
        "properties or past decisions call lookup and answer only from its output; never say you have "
        "no real-time access. Never invent facts."
    )


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

        # State tracking
        self._user_speaking = False
        self.current_response_id = None
        self.cancelled_response_ids = set()
        self.handoff_count = 0

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

    async def start(self):
        key = tts._keychain(tts.QWEN_KEYCHAIN)
        self.ws = await self._connect(URL + self.model, additional_headers={"Authorization": "Bearer " + key})
        await self._expect("session.created")
        session_cfg = {
            "modalities": ["text", "audio"],
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm24",
            "turn_detection": {"type": "server_vad"},
            "tools": TOOLS,
            "tool_choice": "auto",
            "instructions": self.instructions,
        }
        if _cfg("live_transcription_model"):
            m = _cfg("live_transcription_model")
            m_name = m if isinstance(m, str) and m not in ("1", "true", "True") else "gummy-realtime-v1"
            session_cfg["input_audio_transcription"] = {"model": m_name}
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
            self._user_speaking = True
            self.stopped_at = None
            self._transcript_done.clear()
            self._turn_transcript = None
            self._turn_words = None
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
            await self.send_to_phone({"type": "live_heard", "text": self.last_heard})
        elif t == "response.audio_transcript.done":
            resp_id = ev.get("response_id") or (ev.get("response") or {}).get("id")
            if not (resp_id and resp_id in self.cancelled_response_ids):
                await self.send_to_phone({"type": "live_said", "text": ev.get("transcript", "")})
        elif t == "response.done":
            self.response_active = False
            self._user_speaking = False
        elif t == "response.function_call_arguments.done":
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

    async def _get_turn_words(self):
        if self._turn_words is not None:
            return self._turn_words
        if not self._transcript_done.is_set() and self.transcribe is None:
            try:
                await asyncio.wait_for(self._transcript_done.wait(), timeout=2.5)
            except (asyncio.TimeoutError, Exception):
                pass
        words = self._turn_transcript
        pcm = self.turn_pcm or (bytes(self._turn_buf) if self._turn_buf else b"")
        if not words and self.transcribe is not None:
            try:
                words = await asyncio.to_thread(self.transcribe, pcm)
                words = str(words or "").strip()
                if words:
                    self.last_heard = words
                    self._turn_words = words
                    await self.send_to_phone({"type": "live_heard", "text": words})
            except Exception:
                words = ""
        elif words:
            self.last_heard = words
            self._turn_words = words
        return self._turn_words or ""

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
        if name == "hand_off":
            if self.handoff_count >= 6:
                return "limit reached, tell the caller to use the menu"
            words = await self._get_turn_words()
            if not words or len(words) < 2:
                return "not sent: I did not catch the words, ask the caller to say it again"
            out = await self.tools_impl.hand_off(words, args.get("agent"))
            if str(out).startswith("posted"):
                self.handoff_count += 1
            return out
        elif name == "get_status":
            words = await self._get_turn_words()
            return await self.tools_impl.status(words or self.last_heard)
        elif name == "lookup":
            words = await self._get_turn_words()
            q = words if (words and len(words) >= 2) else (args.get("question") or "")
            lookup_fn = getattr(self.tools_impl, "lookup", None)
            if lookup_fn:
                return await lookup_fn(q)
            return "Nothing found."
        else:
            return "failed: unknown tool"

    async def _run_tool(self, ev):
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
            await self._send({"type": "response.create"})
            self.response_active = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closing and not self.closed:
                await self._fail(str(e) or type(e).__name__)
