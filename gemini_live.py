"""Gemini Live mode bridge: drop-in for realtime.LiveBridge (same phone messages, same tools and tool logic).

Selected by config live_provider == "gemini". Key: the Keychain entry named by KEYCHAIN. Uses the google-genai SDK.
Gemini has no turn ids, so a turn is tracked from the stream: her turn runs from the end of Dot's last reply
(turn_complete / interrupted) until Dot's first output (audio, words or a tool call).
"""
import array
import asyncio
import math
import time

import realtime
import tts

KEYCHAIN = "voice-call-gemini"
DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Iapetus"
TURN_CAP = 960000   # 30 s of 16 kHz pcm16
PIECE = 9600        # 0.2 s of 24 kHz pcm16 per phone message
AHEAD_S = 1.0       # keep at most this much audio queued on the phone
QUOTA_WORDS = ("429", "resource_exhausted", "resource exhausted", "quota", "rate limit")
QUOTA_PAUSE_S = 3600   # free quota used up: Qwen for an hour, then try Gemini again
SILENCE_MS = 800       # quiet time that ends her turn (a short pause mid-sentence must not; R3)
VA_CONFIRM_S = 0.25    # a voice-activity start counts as her barge-in only after this long AND with her voice level
VA_CONFIRM_MAX_S = 1.2  # ...seen within this long (speaker echo starts "activity" but is quiet in her mic)
ECHO_GRACE_S = 0.6     # Dot's first moments of sound: a voice start needs this long to count (speaker echo)
LATE_S = 2.0           # her words this soon after Dot started answering belong to the turn Dot is answering
RECENT_BYTES = 48000   # 1.5 s of her audio kept while Dot plays, so a barge-in keeps her first words
ONSET_PAD_S = 0.3      # audio kept from before her voice started
TAIL_KEEP = 9600       # 0.3 s: all that is kept of the mic while Dot plays and she is quiet (no echo in hand-offs)
VOICE_HANG_S = 0.3     # her voice stretch ends after this much quiet (one stretch spans gaps between words)
VOICE_MIN_S = 0.2      # loud time in a stretch before it counts as her voice
ECHO_RATIO = 2.5       # while Dot plays, her voice must be this much louder than the mic's level during Dot's sound
ECHO_GROWTH = 1.5      # the learned echo level may grow at most this much per reply (her voice can't teach it)
SPEECH_RMS = 700       # server-side voice level floor (pcm16 RMS), above 3x the room noise
CLEAR_VOICE_RMS = 5000  # her voice close to the phone; echo left after the phone's echo cancel stays below it


def _rms(pcm):
    a = array.array("h")
    a.frombytes(pcm[:len(pcm) - len(pcm) % 2])
    if not a:
        return 0.0
    return math.sqrt(sum(x * x for x in a) / len(a))


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
        self._reply_total = 0        # bytes of the current reply's audio (all of it, also what was dropped)
        self._reply_sent = 0         # bytes of it sent to the phone
        self._cut_played = None      # seconds of the reply she heard when she interrupted it
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
        self._barge_started = False  # a barge-in stopped Dot before this reply's turn_complete
        self._play_until = 0.0       # when the phone finishes playing what was sent
        # Gemini sends a whole reply in about a second; the phone got scratchy on such bursts, so audio
        # is paced out at speaking speed like Qwen's stream.
        self._out_q = asyncio.Queue()
        self._out_gen = 0
        self._out_task = None
        self._turn_started = 0.0
        self._turn_ended = 0.0       # when Dot started answering
        self._play_started = 0.0     # when the phone started playing the current stretch of audio
        self._va_token = 0           # bumps on every voice-activity start/end (a pending confirm checks it)
        self._recent_said = ""       # Dot's words of this and the last reply (to spot her mic hearing Dot)
        # One voice detector on her mic audio (server side), used by everything that asks "is she talking?"
        self._floor = 300.0          # room noise level while Dot is quiet
        self._play_floor = 0.0       # mic level while Dot plays (speaker echo left after the phone's echo cancel)
        self._play_cap = SPEECH_RMS  # this reply's ceiling for _play_floor
        self._voice_s = 0.0          # loud time in her current voice stretch
        self._quiet_s = 0.0          # quiet time since her last loud chunk
        self._voice_bytes = 0        # audio since her current stretch began (her onset, counted in audio)
        self._reply_levels = []      # mic levels while Dot played (logged per reply: measures the echo)
        self._was_playing = False
        self._va_start_at = 0.0      # when the last voice-activity start came
        self._recent = b""           # her last 1.5 s of audio
        self._confirms = set()       # pending voice-start confirmations

    @property
    def busy(self):
        now = time.monotonic()
        draining = bool(self._draining and now - self._draining < 8.0)
        return bool(super().busy or draining or now < self._play_until or not self._out_q.empty())

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
            # Measured on the real API 2026-10-11: the default ended her turn at a 0.7 s pause mid-sentence (Dot
            # answered half a sentence); low end sensitivity + 800 ms kept it (reply ~1.0 s after she stops).
            # Low START sensitivity made Gemini deaf (no reply at all): keep the default start.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=300,
                    silence_duration_ms=int(realtime._cfg("live_silence_ms_gemini") or SILENCE_MS))),
            # Her ZH/EN choice first; both always allowed (real API: Chinese transcribed with either order).
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=["cmn-Hans-CN", "en-US"] if self.language == "zh-CN" else ["en-US", "cmn-Hans-CN"]),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(handle=handle),
            # Free tier allows 65K tokens/min and each turn sends the whole context (measured ~1K tokens a turn,
            # growing 100-150): 16K keeps a long call under the limit; 8K forgot the start of a call too soon.
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=16000, sliding_window=types.SlidingWindow(target_tokens=8000)),
        )

    async def start(self):
        self._key = tts._keychain(KEYCHAIN)
        if not self._key and self._connect is sdk_connect:   # a test or --fake connect needs no key
            raise RuntimeError("no Gemini key")
        try:
            self.session = await self._connect(self.model, config=self._config(), api_key=self._key)
        except Exception as e:
            _note_quota(e)
            raise
        self.ws = self.session   # LiveBridge._fail closes self.ws
        await self._seed_session()
        self._out_task = asyncio.ensure_future(self._playout())
        self.reader = asyncio.ensure_future(self._read_loop())

    async def _seed_session(self):
        """A new session in the same call (phone reconnect, Live off-on, provider switch) gets the call so far as
        past turns, without a reply (real API 2026-10-12: turn_complete=False is remembered and not answered).
        Only at start: a GoAway resume keeps the session's own memory."""
        if not self.seed:
            return
        types = _types()
        turns = []
        seed = [dict(line) for line in self.seed]
        if seed[-1]["role"] == "user":   # her last words never got a reply (the line dropped): not a question now
            seed[-1]["text"] += " (not answered: the line dropped)"
        for line in seed:
            role = "user" if line["role"] == "user" else "model"
            if turns and turns[-1][0] == role:
                turns[-1][1].append(line["text"])
            else:
                turns.append((role, [line["text"]]))
        try:
            await self.session.send_client_content(turns=[types.Content(role=r, parts=[types.Part(text=" ".join(t))])
                                                          for r, t in turns], turn_complete=False)
            self._log_line("live: seeded %d lines" % len(self.seed))
        except Exception as e:   # never fail the call over its memory
            self._log_line("live: seed failed (%s)" % (str(e) or type(e).__name__)[:80])

    def _flush_audio(self):
        self._out_gen += 1   # queued pieces of the old reply are skipped
        self._play_until = 0.0

    async def _playout(self):
        while not self.closed:
            gen, piece = await self._out_q.get()
            if gen != self._out_gen or self._drop_audio:
                continue
            ahead = self._play_until - time.monotonic()
            if ahead > AHEAD_S:
                await asyncio.sleep(ahead - AHEAD_S)
                if gen != self._out_gen or self._drop_audio or self.closed:
                    continue
            now = time.monotonic()
            if self._play_until <= now:
                self._play_started = now
            self._play_until = max(now, self._play_until) + len(piece) / 48000.0
            try:
                await self.send_to_phone(piece)
                self._reply_sent += len(piece)
            except Exception:
                pass

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
        if self._out_task and self._out_task is not asyncio.current_task():
            self._out_task.cancel()
        await super()._fail(reason)

    async def close(self):
        if self.closed and self._closing:
            return
        self._closing = True
        self.closed = True
        if self.reader and self.reader is not asyncio.current_task():
            self.reader.cancel()
        if self._out_task and self._out_task is not asyncio.current_task():
            self._out_task.cancel()
        if self.tool_task and self.tool_task is not asyncio.current_task():
            self.tool_task.cancel()
        for t in list(self._confirms):
            t.cancel()
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
        self._recent = (self._recent + pcm)[-RECENT_BYTES:]
        playing = time.monotonic() < self._play_until
        self._local_vad(pcm, playing)
        if self._in_turn:
            if playing:   # Dot's tail plays: her audio from her voice onset if she talks, else only 0.3 s (no echo)
                self._turn_buf = bytearray(self._onset_audio() if self._her_voice() else self._recent[-TAIL_KEEP:])
            else:   # rolling: the last 30 s, so a long silence before she speaks never crowds out her words
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

    def _local_vad(self, pcm, playing):
        """Her voice in her own mic. Speaker echo is judged against the mic level measured while Dot plays, so it
        never counts as her voice; a stretch spans the gaps between her words (VOICE_HANG_S)."""
        level = _rms(pcm)
        dur = len(pcm) / 32000.0
        thr = max(SPEECH_RMS, 3 * self._floor)
        if playing:
            if not self._was_playing:   # a new stretch of Dot's sound: cap how far the echo level may rise
                self._play_cap = max(SPEECH_RMS, ECHO_GROWTH * self._play_floor)
            thr = max(thr, ECHO_RATIO * self._play_floor)
            self._reply_levels.append(int(level))
        elif self._was_playing:
            self._log_reply_levels()
        self._was_playing = playing
        if playing and time.monotonic() - self._play_started < ECHO_GRACE_S:
            thr = max(thr, CLEAR_VOICE_RMS)   # Dot's first moments: learn the echo; only a clear voice is hers
        loud = level > thr
        if not loud:   # learn the quiet levels (and the echo level while Dot plays) from non-voice audio
            if playing:   # capped: if she talks as Dot starts, her voice must not become "echo"
                self._play_floor = 0.9 * self._play_floor + 0.1 * min(level, self._play_cap)
            elif level < self._floor:
                self._floor = 0.5 * self._floor + 0.5 * max(level, 50.0)
            else:
                self._floor += (level - self._floor) * 0.002
        if loud:
            if not self._voice_s:
                self._voice_bytes = 0
            self._voice_s += dur
            self._quiet_s = 0.0
        elif self._voice_s:
            self._quiet_s += dur
            if self._quiet_s >= VOICE_HANG_S:
                self._voice_s = 0.0
        if self._voice_s:
            self._voice_bytes += len(pcm)
        if self._her_voice() and self._in_turn and not playing:
            self._user_speaking = True
            self._activity_at = time.monotonic()
            self.stopped_at = None

    def _her_voice(self):
        return self._voice_s >= VOICE_MIN_S

    def _log_reply_levels(self):
        lv = sorted(self._reply_levels)
        self._reply_levels = []
        if lv:   # G5: measures the echo on her phone (her voice over Dot shows as a high p95)
            self._log_line("live: mic during reply p50=%d p95=%d echo_floor=%d" % (
                lv[len(lv) // 2], lv[int(len(lv) * 0.95)], self._play_floor))

    async def barge(self):
        """The phone heard her voice over Dot's reply (it already stopped its own playback)."""
        if self._in_turn and not self._activity_at:
            self._user_speaking = True
            self._activity_at = time.monotonic()
        await self._barge_in("phone")

    async def say_exact(self, text):
        if self.closed or self.session is None:
            return
        types = _types()
        if self._draining and time.monotonic() - self._draining > 8.0:   # its turn_complete never came
            self._draining = 0.0
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
        self._flush_audio()
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
        self._turn_no += 1
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
        self._turn_buf = bytearray(self._onset_audio() if self._barge_started else self._preroll)
        self._barge_started = False

    async def _end_turn(self):
        """Dot starts answering: her turn is over."""
        if not self._in_turn:
            return
        self._in_turn = False
        self._reply_no += 1
        self._reply_total = self._reply_sent = 0
        self._cut_played = None
        self._user_speaking = False
        if self.stopped_at is None:   # her last words (no VAD end event) or now
            self.stopped_at = self._activity_at if self._activity_at else time.monotonic()
        self.turn_pcm = bytes(self._turn_buf)
        self.response_active = True
        self._turn_ended = time.monotonic()
        if self._heard.strip():
            self._turn_transcript = self.last_heard = self._turn_words = self._heard.strip()
            self._transcript_done.set()
            await self._send_heard()

    async def _send_heard(self):
        if not self._heard_sent and self._heard.strip():
            self._heard_sent = True
            await self.send_to_phone({"type": "live_heard", "text": self._heard.strip(), "turn": self._turn_no})

    async def _on_message(self, msg):
        upd = getattr(msg, "session_resumption_update", None)
        if upd is not None and getattr(upd, "resumable", False) and getattr(upd, "new_handle", None):
            self._resume_handle = upd.new_handle
        va = getattr(msg, "voice_activity", None)
        if va is not None:
            kind = str(getattr(va.voice_activity_type, "value", va.voice_activity_type) or "")
            self._log_line("live: va %s in_turn=%s playing=%.1fs" % (kind.rsplit("_", 1)[-1], self._in_turn,
                                                                      max(0.0, self._play_until - time.monotonic())))
            self._va_token += 1
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

    def _dot_sounding(self):
        return time.monotonic() < self._play_until or not self._out_q.empty()

    def _onset_audio(self):
        """Her recent audio from just before her voice started (her first words, not Dot's echo before them)."""
        if self._voice_s:   # her current voice stretch, counted in audio (not wall clock)
            back = self._voice_bytes / 32000.0
        elif self._va_start_at:
            back = time.monotonic() - self._va_start_at
        else:
            back = 1.0
        keep = min(RECENT_BYTES, int(max(0.3, back + ONSET_PAD_S) * 32000) & ~1)
        return self._recent[-keep:]

    async def _barge_in(self, why):
        """She talks over Dot: stop the phone NOW (her requirement 2026-10-11: "when I interrupt, it stop
        talking"). Gemini sends a reply far ahead of playback, so its own interrupted often never comes."""
        if not (self._dot_sounding() or (not self._in_turn and self.response_active)):
            return False
        self._log_line("live: she interrupted, Dot stopped (%s) voice=%.1fs echo_floor=%d" % (
            why, self._voice_s, self._play_floor))
        # what she heard of this reply: sent minus what the phone still had queued
        self._cut_played = max(0.0, self._reply_sent / 48000.0 - max(0.0, self._play_until - time.monotonic()))
        self._flush_audio()
        if self._in_turn:   # the tail of a finished reply: her turn audio starts at her voice
            self._turn_buf = bytearray(self._onset_audio())
            await self._send_cut(self._recent_said)
        else:
            self._barge = True               # her words from now on start her next turn
            self._barge_started = True
            self._drop_audio = True          # the rest of this reply, until its turn_complete
            self._draining = time.monotonic()
            self.response_active = False
        await self.send_to_phone({"type": "live_stop"})
        return True

    async def _confirm_voice(self, token, wait):
        """A voice start stops Dot only if it lasts and her mic is really loud (echo starts activity too, and with
        a long end-of-speech silence its end event can never come in time)."""
        try:
            await asyncio.sleep(wait)
            deadline = time.monotonic() + VA_CONFIRM_MAX_S
            while token == self._va_token and not self.closed and time.monotonic() < deadline:
                if self._her_voice():
                    await self._barge_in("voice")
                    return
                await asyncio.sleep(0.05)
        finally:
            self._confirms.discard(asyncio.current_task())

    async def _speech_started(self):
        now = time.monotonic()
        self._va_start_at = now
        if self._in_turn:
            self._user_speaking = True
            self._activity_at = now
            self.stopped_at = None
        if self._dot_sounding() or not self._in_turn:
            wait = VA_CONFIRM_S if now - self._play_started > ECHO_GRACE_S else ECHO_GRACE_S
            self._confirms.add(asyncio.ensure_future(self._confirm_voice(self._va_token, wait)))

    def _late(self):
        return time.monotonic() - self._turn_ended < LATE_S

    async def _on_content(self, sc):
        # Words never stop Dot (her own late words and Dot's echo arrive as words too); only her voice does.
        it = getattr(sc, "input_transcription", None)
        if it is not None and it.text:
            late = (self._in_turn and not self._seeded and not self._activity_at and self._late()
                    and time.monotonic() - self._turn_started < LATE_S)
            if late:   # the last turn's words arriving after its turn_complete: not part of this turn
                self.last_heard = ((self.last_heard or "") + " " + it.text).strip()
            elif self._in_turn:
                self._user_speaking = True
                self._activity_at = time.monotonic()
                self.stopped_at = None
                self._heard += it.text
                await self.send_to_phone({"type": "live_partial", "text": self._heard.strip()})
            elif self._barge:   # she stopped Dot: these words start her next turn
                self._barge_heard += it.text
                await self.send_to_phone({"type": "live_partial", "text": self._barge_heard.strip()})
            else:   # late words for the turn that just ended
                self._heard += it.text
                self._turn_transcript = self.last_heard = self._heard.strip()
                if not self._whisper_words:
                    self._turn_words = self._turn_transcript
                self._transcript_done.set()
        if sc.interrupted:
            self._log_line("live: gemini interrupted playing=%.1fs" % max(0.0, self._play_until - time.monotonic()))
            if self._cut_played is None:   # memory keeps only what she heard of this reply
                self._cut_played = max(0.0, self._reply_sent / 48000.0 - max(0.0, self._play_until - time.monotonic()))
            await self._send_cut(self._said.strip())   # also after a phone barge-in (its cut is still waiting)
            self._flush_audio()
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
                    self._reply_total += len(data)
                    if not self._drop_audio:
                        data = bytes(data)
                        for i in range(0, len(data), PIECE):
                            self._out_q.put_nowait((self._out_gen, data[i:i + PIECE]))
        ot = getattr(sc, "output_transcription", None)
        if ot is not None and ot.text:
            await self._end_turn()
            self._said += ot.text
        if sc.turn_complete:
            await self._send_heard()
            said = self._said.strip()
            self._said = ""
            self._recent_said = said
            if said and not self._drop_audio:
                await self.send_to_phone({"type": "live_said", "text": said, "reply": self._reply_no})
            elif said and self._cut_played is not None:   # stopped mid-reply: remember only what she heard
                await self._send_cut(said)
            if not self._in_turn or self._draining:
                exact = self._exact_pending
                self._exact_pending = False
                self._start_turn()
                if exact:   # say_exact was sent while the old reply drained: it is the reply now playing
                    self.response_active = True

    async def _send_cut(self, said):
        """She interrupted this reply: the call's memory keeps only the part she heard (server only, not shown)."""
        if self._cut_played is None or not said or not self._reply_total:
            return
        fraction = self._cut_played / (self._reply_total / 48000.0)
        self._cut_played = None
        await self.send_to_phone({"type": "live_cut", "text": realtime.cut_text(said, fraction), "reply": self._reply_no})

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
