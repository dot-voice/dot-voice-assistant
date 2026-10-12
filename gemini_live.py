"""Gemini Live mode bridge: drop-in for realtime.LiveBridge (same phone messages, same tools and tool logic).

Selected by config live_provider == "gemini". Key: the Keychain entry named by KEYCHAIN. Uses the google-genai SDK.
Gemini has no turn ids, so a turn is tracked from the stream: her turn runs from the end of Dot's last reply
(turn_complete / interrupted) until Dot's first output (audio, words or a tool call).
"""
import asyncio
import time

import realtime
import tts

KEYCHAIN = "voice-call-gemini"
DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Iapetus"
TURN_CAP = 960000   # 30 s of 16 kHz pcm16
QUOTA_WORDS = ("429", "resource_exhausted", "resource exhausted", "quota", "rate limit")
QUOTA_PAUSE_S = 3600   # free quota used up: Qwen for an hour, then try Gemini again


def _note_quota(reason):
    low = str(reason).lower()
    if any(w in low for w in QUOTA_WORDS):
        realtime.mark_down("gemini", QUOTA_PAUSE_S)
        return True
    return False


def _types():
    from google.genai import types
    return types


def gemini_tools():
    """realtime.TOOLS (OpenAI-realtime shape) as Gemini function declarations: same names, text and parameters."""
    decls = [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
             for t in realtime.TOOLS]
    return [{"function_declarations": decls}]


class _SdkSession:
    """The SDK's connect is an async context manager; keep it so close() can exit it."""

    def __init__(self, cm, session):
        self._cm = cm
        self._s = session

    def __getattr__(self, name):
        return getattr(self._s, name)

    async def close(self):
        await self._cm.__aexit__(None, None, None)


async def sdk_connect(model, config=None, api_key=None):
    from google import genai
    cm = genai.Client(api_key=api_key).aio.live.connect(model=model, config=config)
    return _SdkSession(cm, await cm.__aenter__())


class GeminiBridge(realtime.LiveBridge):
    falls_back_to_qwen = True   # server.Call switches a failing call to Qwen Live

    def __init__(self, send_to_phone, tools_impl, instructions, model=None, connect=None,
                 transcribe=None, on_closed=None, turn_log=None):
        super().__init__(send_to_phone, tools_impl, instructions, model=model or realtime._cfg("live_model_gemini")
                         or DEFAULT_MODEL, connect=connect or sdk_connect, transcribe=transcribe,
                         on_closed=on_closed, turn_log=turn_log)
        self.voice = realtime._cfg("live_voice_gemini") or DEFAULT_VOICE
        self.session = None
        self._key = None
        self._resume_handle = None          # latest session resumption handle
        self._reconnecting = False
        self._drop_audio = False     # cancel(): drop the rest of this reply
        self._heard = ""             # her words so far in this turn (input transcription)
        self._heard_sent = False
        self._in_turn = True         # she has the floor until Dot answers
        self._turn_buf = bytearray()
        self._said = ""
        self._activity_at = 0.0      # last sign of her voice (VAD start or words)
        self._draining = 0.0         # cancel(): time the dropped reply started draining (until its turn_complete)
        self._exact_pending = False  # say_exact sent while an old reply drains
        self._barge = False          # she started talking while Dot answered
        self._barge_heard = ""       # her words from that moment (they belong to her next turn)
        self._pending_audio = []     # her audio while a resume is in progress
        self._resumes = []           # times of recent resumes (cap)
        self._resume_lock = asyncio.Lock()
        self._seeded = False
        self._play_until = 0.0       # when the phone finishes playing what was sent (Gemini sends faster than real time)
        self._turn_started = 0.0

    @property
    def busy(self):
        now = time.monotonic()
        draining = bool(self._draining and now - self._draining < 8.0)
        return bool(super().busy or draining or now < self._play_until)

    @property
    def user_speaking(self):
        # Without a speech-end event (no VAD signal, noise Gemini ignores) never stay "speaking" for ever.
        return bool(self._in_turn and self._user_speaking and time.monotonic() - self._activity_at < 3.0)

    # ---- connection ----

    def _config(self, handle=None):
        types = _types()
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))),
            system_instruction=self.instructions,
            tools=gemini_tools(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(handle=handle),
            # Free tier allows 65K tokens/min and each turn sends the whole context (measured ~1K tokens a turn,
            # growing): keep the context small so a long call never hits the limit.
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=8000, sliding_window=types.SlidingWindow(target_tokens=4000)),
        )

    async def start(self):
        self._key = tts._keychain(KEYCHAIN)
        if not self._key:
            raise RuntimeError("no Gemini key")
        try:
            self.session = await self._connect(self.model, config=self._config(), api_key=self._key)
        except Exception as e:
            _note_quota(e)
            raise
        self.ws = self.session   # LiveBridge._fail closes self.ws
        self.reader = asyncio.ensure_future(self._read_loop())

    async def _reconnect(self, old=None):
        """GoAway (about every 10 min) or a dropped socket: resume the same conversation on a new connection.
        Callers set _reconnecting first so no second resume starts meanwhile."""
        old = old or self.session
        async with self._resume_lock:
            if self.session is not old or self.closed:
                self._reconnecting = False
                return
            now = time.monotonic()
            self._resumes = [t for t in self._resumes if now - t < 30] + [now]
            if len(self._resumes) > 2:
                self._reconnecting = False
                raise RuntimeError("Gemini keeps dropping")
            self._reconnecting = True
            try:
                new = await asyncio.wait_for(
                    self._connect(self.model, config=self._config(self._resume_handle), api_key=self._key), 6)
            except BaseException:
                self._reconnecting = False
                raise
            if self.closed:   # the call ended while we were connecting
                self._reconnecting = False
                try:
                    await new.close()
                except Exception:
                    pass
                return
            self.session = self.ws = new
            pending, self._pending_audio = self._pending_audio, []
            self._reconnecting = False
            self._log_line("live: gemini resumed")
            for pcm in pending:   # what she said during the hand-over
                try:
                    await new.send_realtime_input(audio=_types().Blob(data=pcm, mime_type="audio/pcm;rate=16000"))
                except Exception:
                    break
            if not self._in_turn:   # the old socket's turn_complete may be lost: give her the floor
                self._start_turn()
        try:
            await old.close()
        except Exception:
            pass

    async def _fail(self, reason):
        if _note_quota(reason):
            self._log_line("live: gemini quota used up, Qwen for the next hour")
        await super()._fail(reason)

    async def close(self):
        if self.closed and self._closing:
            return
        self._closing = True
        self.closed = True
        if self.reader and self.reader is not asyncio.current_task():
            self.reader.cancel()
        if self.tool_task and self.tool_task is not asyncio.current_task():
            self.tool_task.cancel()
        if self.session is not None:
            try:
                await self.session.close()
            except Exception:
                pass

    # ---- phone -> Gemini ----

    async def feed_audio(self, pcm):
        if self.closed or self.session is None:
            return
        self._preroll = (self._preroll + pcm)[-16000:]
        if self._in_turn:   # rolling: the last 30 s, so a long silence before she speaks never crowds out her words
            self._turn_buf.extend(pcm)
            if len(self._turn_buf) > TURN_CAP:
                del self._turn_buf[:len(self._turn_buf) - TURN_CAP]
        if self._reconnecting:
            if sum(len(p) for p in self._pending_audio) < 160000:   # up to 5 s, sent after the resume
                self._pending_audio.append(pcm)
            return
        try:
            await self.session.send_realtime_input(audio=_types().Blob(data=pcm, mime_type="audio/pcm;rate=16000"))
        except Exception as e:
            if self._resume_handle and not _note_quota(e):
                self._pending_audio.append(pcm)   # the reader sees the dead socket and resumes
                return
            if not self._reconnecting:
                await self._fail(str(e) or type(e).__name__)

    async def say_exact(self, text):
        if self.closed or self.session is None:
            return
        types = _types()
        if self._draining:
            self._exact_pending = True   # the old reply's turn_complete must not end this one
        else:
            self._drop_audio = False
        try:
            await self.session.send_client_content(turns=types.Content(role="user", parts=[types.Part(
                text="Say exactly this, word for word, nothing else: " + text)]), turn_complete=True)
        except Exception as e:
            await self._fail(str(e) or type(e).__name__)
            return
        self.response_active = True

    async def cancel(self):
        # No cancel in the Live API: drop the rest of this reply (same effect for her as a barge-in).
        self._play_until = 0.0
        if self.response_active:
            self._drop_audio = True
            self._draining = time.monotonic()
            self.response_active = False

    # ---- Gemini -> phone ----

    async def _read_loop(self):
        empty = 0
        while not self.closed:
            got = False
            sess = self.session
            try:
                async for msg in sess.receive():
                    got = True
                    await self._on_message(msg)
                    if self.closed:
                        return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._closing or self.closed:
                    return
                if self.session is not sess or self._reconnecting:   # the old socket of a resumed session
                    while self._reconnecting:
                        await asyncio.sleep(0.05)
                    continue
                if self._resume_handle and not _note_quota(e):
                    try:
                        await self._reconnect(sess)
                        continue
                    except Exception as e2:
                        e = e2
                await self._fail(str(e) or type(e).__name__)
                return
            empty = 0 if got else empty + 1
            if empty >= 3:
                if not self._closing and not self.closed:
                    await self._fail("connection closed")
                return

    def _start_turn(self):
        """Dot's reply ended: the floor is hers again; collect her audio and words for the next turn."""
        self._in_turn = True
        self._user_speaking = False
        self.response_active = False
        self._drop_audio = False
        self._draining = 0.0
        self._heard = self._barge_heard   # words she said while Dot was still answering
        self._seeded = bool(self._heard)
        self._turn_started = time.monotonic()
        self._barge_heard = ""
        self._barge = False
        self._heard_sent = False
        self._said = ""
        self._activity_at = 0.0
        self._turn_transcript = None
        self._turn_words = None
        self._whisper_words = None
        self._transcript_done.clear()
        self.turn_pcm = b""
        self._turn_buf = bytearray(self._preroll)

    async def _end_turn(self):
        """Dot starts answering: her turn is over."""
        if not self._in_turn:
            return
        self._in_turn = False
        self._user_speaking = False
        if self.stopped_at is None:   # her last words (no VAD end event) or now
            self.stopped_at = self._activity_at if self._activity_at else time.monotonic()
        self.turn_pcm = bytes(self._turn_buf)
        self.response_active = True
        if self._heard.strip():
            self._turn_transcript = self.last_heard = self._turn_words = self._heard.strip()
            self._transcript_done.set()
            await self._send_heard()

    async def _send_heard(self):
        if not self._heard_sent and self._heard.strip():
            self._heard_sent = True
            await self.send_to_phone({"type": "live_heard", "text": self._heard.strip()})

    async def _on_message(self, msg):
        upd = getattr(msg, "session_resumption_update", None)
        if upd is not None and getattr(upd, "resumable", False) and getattr(upd, "new_handle", None):
            self._resume_handle = upd.new_handle
        va = getattr(msg, "voice_activity", None)
        if va is not None:
            kind = str(getattr(va.voice_activity_type, "value", va.voice_activity_type) or "")
            if kind.endswith("ACTIVITY_START"):
                await self._speech_started()
            elif kind.endswith("ACTIVITY_END"):
                self.stopped_at = time.monotonic()
                self._user_speaking = False
        sc = getattr(msg, "server_content", None)
        if sc is not None:
            await self._on_content(sc)
        tc = getattr(msg, "tool_call", None)
        if tc is not None and tc.function_calls:
            await self._end_turn()
            self.tool_task = asyncio.create_task(self._run_calls(list(tc.function_calls)))
        cancel = getattr(msg, "tool_call_cancellation", None)
        if cancel is not None and self.tool_task and not self.tool_task.done():
            self.tool_task.cancel()   # she barged in; Gemini dropped the call
        if getattr(msg, "go_away", None) is not None and not self._reconnecting:
            if self._resume_handle:
                self._reconnecting = True
                asyncio.ensure_future(self._go_away())
            else:
                self._log_line("live: gemini go_away without resume handle")

    async def _go_away(self):
        try:
            await self._reconnect(self.session)
        except Exception as e:
            await self._fail("resume failed: " + (str(e) or type(e).__name__)[:80])

    async def _speech_started(self):
        if not self._in_turn:
            # Dot is answering: Gemini decides if this is a barge-in (then it sends interrupted -> live_stop).
            # Noise it ignores must not cut Dot off on the phone.
            self._barge = True
            return
        self._user_speaking = True
        self._activity_at = time.monotonic()
        self.stopped_at = None
        if self._activity_at < self._play_until:
            # Gemini finished the reply but the phone is still playing its tail: stop it, she is talking.
            self._play_until = 0.0
            await self.send_to_phone({"type": "live_stop"})

    async def _on_content(self, sc):
        it = getattr(sc, "input_transcription", None)
        if it is not None and it.text:
            late = (self._in_turn and not self._seeded and not self._activity_at
                    and time.monotonic() - self._turn_started < 2.0)
            if late:   # the last turn's words arriving after its turn_complete: not part of this turn
                self.last_heard = ((self.last_heard or "") + " " + it.text).strip()
            elif self._in_turn:
                self._user_speaking = True
                self._activity_at = time.monotonic()
                self.stopped_at = None
                self._heard += it.text
            elif self._barge:   # she talks over Dot: these words start her next turn
                self._barge_heard += it.text
            else:   # late words for the turn that just ended
                self._heard += it.text
                self._turn_transcript = self.last_heard = self._heard.strip()
                if not self._whisper_words:
                    self._turn_words = self._turn_transcript
                self._transcript_done.set()
        if sc.interrupted:
            self._play_until = 0.0
            await self._send_heard()
            self._start_turn()
            self._user_speaking = True
            self._activity_at = time.monotonic()
            await self.send_to_phone({"type": "live_stop"})
            return
        if sc.model_turn and sc.model_turn.parts:
            for p in sc.model_turn.parts:
                data = getattr(getattr(p, "inline_data", None), "data", None)
                if data:
                    await self._end_turn()
                    if not self._drop_audio:
                        now = time.monotonic()
                        self._play_until = max(now, self._play_until) + len(data) / 48000.0
                        await self.send_to_phone(bytes(data))
        ot = getattr(sc, "output_transcription", None)
        if ot is not None and ot.text:
            await self._end_turn()
            self._said += ot.text
        if sc.turn_complete:
            await self._send_heard()
            said = self._said.strip()
            self._said = ""
            if said and not self._drop_audio:
                await self.send_to_phone({"type": "live_said", "text": said})
            if not self._in_turn or self._draining:
                exact = self._exact_pending
                self._exact_pending = False
                self._start_turn()
                if exact:   # say_exact was sent while the old reply drained: it is the reply now playing
                    self.response_active = True

    # ---- tools (logic shared with LiveBridge._exec_tool) ----

    async def _run_calls(self, calls):
        types = _types()
        try:
            responses = []
            for fc in calls:
                t0 = time.monotonic()
                try:
                    out = await asyncio.wait_for(self._exec_tool(fc.name, dict(fc.args or {})), timeout=20.0)
                except asyncio.TimeoutError:
                    out = "timed out"
                except Exception as e:
                    out = "failed: " + (str(e) or type(e).__name__)[:100]
                self._log_tool(fc.name, int((time.monotonic() - t0) * 1000))
                responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"output": str(out)}))
            if self.closed:
                return
            await self.session.send_tool_response(function_responses=responses)
            if not self._in_turn:   # she barged in meanwhile: Gemini answers after her, maybe never
                self.response_active = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closing and not self.closed:
                await self._fail(str(e) or type(e).__name__)
