import asyncio
import json
import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import gemini_live  # noqa: E402
import realtime  # noqa: E402
import server  # noqa: E402
from test_realtime import FakeCallWS, Tools  # noqa: E402


def content(**kw):
    base = dict(input_transcription=None, output_transcription=None, interrupted=None, model_turn=None,
                turn_complete=None)
    base.update(kw)
    return NS(server_content=NS(**base))


def audio(data):
    return content(model_turn=NS(parts=[NS(inline_data=NS(data=data))]))


def heard(text):
    return content(input_transcription=NS(text=text))


def said(text):
    return content(output_transcription=NS(text=text))


def done():
    return content(turn_complete=True)


def call(name, args, cid="c1"):
    return NS(tool_call=NS(function_calls=[NS(id=cid, name=name, args=args)]))


class FakeSession:
    """Like the SDK session: receive() yields until a turn_complete, raises when the socket is gone."""

    def __init__(self):
        self.q = asyncio.Queue()
        self.audio = []
        self.tool_responses = []
        self.client = []
        self.closed = False

    def push(self, msg):
        self.q.put_nowait(msg)

    async def receive(self):
        while True:
            msg = await self.q.get()
            if isinstance(msg, Exception):
                raise msg
            yield msg
            sc = getattr(msg, "server_content", None)
            if sc is not None and getattr(sc, "turn_complete", None):
                return

    async def send_realtime_input(self, audio=None):
        self.audio.append(audio)

    async def send_tool_response(self, function_responses=None):
        self.tool_responses.append(function_responses)

    async def send_client_content(self, turns=None, turn_complete=True):
        self.client.append(turns)
        self.client_complete = getattr(self, "client_complete", []) + [turn_complete]

    async def close(self):
        self.closed = True


class GeminiBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def make(self, transcribe=None, key="k", seed=None, language=None):
        self.sessions = []
        self.configs = []
        self.phone = []
        self.closed_calls = 0

        async def to_phone(x):
            self.phone.append(x)

        async def connect(model, config=None, api_key=None):
            self.model = model
            self.configs.append(config)
            s = FakeSession()
            self.sessions.append(s)
            return s

        def on_closed():
            self.closed_calls += 1

        self.tools = Tools()
        with mock.patch.object(gemini_live.tts, "_keychain", return_value=key):
            self.b = gemini_live.GeminiBridge(to_phone, self.tools, "be brief", connect=connect,
                                              transcribe=transcribe, on_closed=on_closed)
            if seed:
                self.b.seed = seed
            if language:
                self.b.language = language
            await self.b.start()
        self.s = self.sessions[0]

    async def talk(self, seconds):
        for _ in range(int(seconds / 0.1)):
            await self.b.feed_audio(b"\x00\x30" * 1600)   # 0.1 s of her loud voice

    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0)

    def dicts(self, kind):
        return [m for m in self.phone if isinstance(m, dict) and m.get("type") == kind]

    async def test_config_has_voice_tools_resumption(self):
        await self.make()
        cfg = self.configs[0]
        self.assertEqual(cfg.speech_config.voice_config.prebuilt_voice_config.voice_name, self.b.voice)
        names = [f.name for f in cfg.tools[0].function_declarations]
        self.assertEqual(names, ["hand_off", "get_status", "cancel_hand_off", "transfer_call", "lookup", "ask_model"])
        self.assertIsNotNone(cfg.session_resumption)
        self.assertIsNotNone(cfg.context_window_compression.sliding_window)
        self.assertEqual(cfg.system_instruction.parts[0].text if hasattr(cfg.system_instruction, "parts")
                         else cfg.system_instruction, "be brief")
        await self.b.close()

    async def test_missing_key_raises(self):
        with mock.patch.object(gemini_live.tts, "_keychain", return_value=None):
            b = gemini_live.GeminiBridge(None, Tools(), "be brief")
            with self.assertRaises(RuntimeError):
                await b.start()

    async def test_audio_to_phone_and_words(self):
        await self.make()
        await self.b.feed_audio(b"\x00\x01" * 100)
        self.assertEqual(self.s.audio[0].mime_type, "audio/pcm;rate=16000")
        self.s.push(heard("hello "))
        self.s.push(heard("there"))
        self.s.push(audio(b"\x01\x02"))
        self.s.push(said("Hi."))
        self.s.push(done())
        await self.settle()
        self.assertIn(b"\x01\x02", self.phone)
        self.assertEqual([m["text"] for m in self.dicts("live_heard")], ["hello there"])
        self.assertEqual([m["text"] for m in self.dicts("live_partial")], ["hello", "hello there"])   # words while she talks
        self.assertEqual([m["text"] for m in self.dicts("live_said")], ["Hi."])
        self.assertIsNotNone(self.b.stopped_at)
        self.assertFalse(self.b.busy)
        await self.b.close()

    async def test_reply_paced_to_phone(self):
        await self.make()
        self.s.push(audio(b"\x01" * 144000))   # 3 s of reply arriving at once
        await self.settle()
        sent = sum(len(m) for m in self.phone if isinstance(m, bytes))
        self.assertLessEqual(sent / 48000.0, gemini_live.AHEAD_S + 0.25)
        self.assertTrue(all(len(m) <= gemini_live.PIECE for m in self.phone if isinstance(m, bytes)))
        self.s.push(content(interrupted=True))
        await self.settle()
        await asyncio.sleep(0.3)
        self.assertEqual(sum(len(m) for m in self.phone if isinstance(m, bytes)), sent)   # the rest is dropped
        await self.b.close()

    async def test_interrupted_stops_phone(self):
        await self.make()
        self.s.push(audio(b"\x01"))
        self.s.push(content(interrupted=True))
        await self.settle()
        self.assertTrue(self.dicts("live_stop"))
        self.assertFalse(self.b.response_active)
        await self.b.close()

    async def test_cancel_drops_rest_of_reply(self):
        await self.make()
        self.s.push(audio(b"\x01"))
        await self.settle()
        await self.b.cancel()
        self.s.push(audio(b"\x02"))
        self.s.push(said("dropped"))
        self.s.push(done())
        await self.settle()
        self.assertIn(b"\x01", self.phone)
        self.assertNotIn(b"\x02", self.phone)
        self.assertFalse(self.dicts("live_said"))
        await self.b.close()

    async def test_hand_off_uses_whisper_words_and_responds(self):
        await self.make(transcribe=lambda pcm: "send the report to Alex")
        await self.b.feed_audio(b"\x00\x01" * 100)
        self.s.push(heard("send the report"))
        self.s.push(call("hand_off", {"agent": "CRM"}))
        await self.settle()
        await self.b.tool_task
        self.assertEqual(self.tools.calls, [("send the report to Alex", "CRM")])
        resp = self.s.tool_responses[0][0]
        self.assertEqual((resp.id, resp.name), ("c1", "hand_off"))
        self.assertTrue(resp.response["output"].startswith("posted"))
        self.assertGreater(len(self.b.turn_pcm), 0)
        await self.b.close()

    async def test_turn_audio_keeps_latest_after_long_silence(self):
        await self.make()
        for _ in range(40):
            await self.b.feed_audio(b"\x00" * 32000)   # 40 s of silence
        await self.b.feed_audio(b"\x07" * 3200)
        self.s.push(audio(b"\x01"))
        await self.settle()
        self.assertEqual(len(self.b.turn_pcm), gemini_live.TURN_CAP)
        self.assertTrue(self.b.turn_pcm.endswith(b"\x07" * 3200))
        await self.b.close()

    async def test_hand_off_cap(self):
        await self.make(transcribe=lambda pcm: "do the thing")
        self.b.handoff_count = 6
        self.s.push(call("hand_off", {}))
        await self.settle()
        await self.b.tool_task
        self.assertEqual(self.tools.calls, [])
        self.assertIn("limit", self.s.tool_responses[0][0].response["output"])
        await self.b.close()

    async def test_get_status_passes_session(self):
        await self.make()
        self.s.push(heard("how is CRM"))
        self.s.push(call("get_status", {"session": "CRM"}))
        await self.settle()
        await self.b.tool_task
        self.assertEqual(self.tools.status_sessions, ["CRM"])
        self.assertEqual(self.s.tool_responses[0][0].response["output"], "all quiet")
        await self.b.close()

    async def test_go_away_resumes_with_handle(self):
        await self.make()
        self.s.push(NS(session_resumption_update=NS(resumable=True, new_handle="h1")))
        self.s.push(NS(go_away=NS(time_left="5s")))
        await self.settle()
        self.assertEqual(len(self.sessions), 2)
        self.assertEqual(self.configs[1].session_resumption.handle, "h1")
        self.assertTrue(self.sessions[0].closed)
        self.s.push(Exception("socket gone"))   # the old socket closing must not end the call
        await self.settle()
        self.sessions[1].push(audio(b"\x09"))
        await self.settle()
        self.assertIn(b"\x09", self.phone)
        self.assertFalse(self.dicts("live_error"))
        self.assertFalse(self.b.closed)
        await self.b.close()

    async def test_fatal_error_once(self):
        await self.make()
        self.s.push(Exception("boom"))
        await self.settle()
        self.assertEqual(len(self.dicts("live_error")), 1)
        self.assertEqual(self.closed_calls, 1)
        self.assertTrue(self.b.closed)

    async def test_cancel_stays_busy_until_old_reply_ends(self):
        await self.make()
        self.s.push(audio(b"\x01"))
        await self.settle()
        await self.b.cancel()
        self.assertTrue(self.b.busy)
        await self.b.say_exact("Update.")
        self.s.push(audio(b"\x02"))   # rest of the cancelled reply: still dropped
        self.s.push(done())
        await self.settle()
        self.assertNotIn(b"\x02", self.phone)
        self.assertTrue(self.b.response_active)   # the update is the reply now
        self.s.push(audio(b"\x03"))
        await self.settle()
        self.assertIn(b"\x03", self.phone)
        await self.b.close()

    async def test_barge_in_on_reply_tail_stops_phone(self):
        await self.make()
        self.s.push(audio(b"\x01" * 96000))   # 2 s of reply, sent at once
        self.s.push(done())
        await self.settle()
        self.assertTrue(self.b.busy)   # the phone is still playing
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        await self.talk(0.3)   # her voice in the mic
        await asyncio.sleep(gemini_live.ECHO_GRACE_S + 0.1)   # a voice start counts once it lasts (not an echo)
        self.assertEqual(len(self.dicts("live_stop")), 1)
        self.assertEqual(self.b._play_until, 0.0)
        self.assertTrue(self.b.user_speaking)
        await self.b.close()

    async def test_idle_noise_after_tail_does_not_stop(self):
        await self.make()
        self.s.push(audio(b"\x01" * 2))
        self.s.push(done())
        await self.settle()
        await asyncio.sleep(0.01)
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        self.assertFalse(self.dicts("live_stop"))
        await self.b.close()

    async def test_she_interrupts_dot_stops(self):
        # Requirement R2: when she talks over Dot, Dot stops at once, even if Gemini never sends interrupted.
        await self.make()
        self.s.push(audio(b"\x01" * 144000))   # 3 s reply
        await self.settle()
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        await self.talk(0.3)
        await asyncio.sleep(gemini_live.ECHO_GRACE_S + 0.1)
        self.assertEqual(len(self.dicts("live_stop")), 1)
        sent = sum(len(m) for m in self.phone if isinstance(m, bytes))
        self.s.push(audio(b"\x02" * 9600))     # Gemini keeps generating the old reply
        await self.settle()
        await asyncio.sleep(0.3)
        self.assertEqual(sum(len(m) for m in self.phone if isinstance(m, bytes)), sent)
        self.assertNotIn(b"\x02" * 9600, self.phone)
        await self.b.close()

    async def test_echo_blip_does_not_stop_dot(self):
        # Speaker echo: a voice start that ends at once must not cut Dot off.
        await self.make()
        self.s.push(audio(b"\x01" * 144000))
        await self.settle()
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_END")))
        await self.settle()
        await asyncio.sleep(gemini_live.ECHO_GRACE_S + 0.1)
        self.assertFalse(self.dicts("live_stop"))
        await self.b.close()

    async def test_echo_voice_start_without_her_level_does_not_stop(self):
        # Echo makes the service report activity, but her mic is quiet: Dot keeps talking (R2 echo).
        await self.make()
        self.s.push(audio(b"\x01" * 192000))
        await self.settle()
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        await asyncio.sleep(gemini_live.ECHO_GRACE_S + gemini_live.VA_CONFIRM_MAX_S + 0.1)
        self.assertFalse(self.dicts("live_stop"))
        await self.b.close()

    async def test_late_words_do_not_stop_reply_tail(self):
        # Review blocker: her last words arrive after Gemini's turn_complete while the tail plays; not a barge-in.
        await self.make()
        self.s.push(heard("check my"))
        self.s.push(audio(b"\x01" * 96000))
        self.s.push(done())
        await self.settle()
        self.s.push(heard(" tasks please"))
        await self.settle()
        self.assertFalse(self.dicts("live_stop"))
        self.assertIn("tasks please", self.b.last_heard)
        await self.b.close()

    async def test_new_session_is_seeded_with_the_call(self):
        # R23: a new session (reconnect, Live off-on) gets the call so far as past turns, without a reply.
        seed = [{"role": "user", "text": "the blue house"}, {"role": "user", "text": "on Main Street"},
                {"role": "assistant", "text": "Noted."}]
        await self.make(seed=seed)
        turns = self.s.client[0]
        self.assertEqual([t.role for t in turns], ["user", "model"])
        self.assertEqual(turns[0].parts[0].text, "the blue house on Main Street")
        self.assertEqual(self.s.client_complete, [False])
        await self.b.close()
        await self.make(seed=seed[:2])   # her last words never got a reply: the seed says so (no answer now)
        self.assertIn("not answered", self.s.client[0][-1].parts[0].text)
        await self.b.close()

    async def test_barge_then_gemini_interrupt_remembers_what_she_heard(self):
        # R25 main path: she talks over Dot (phone stops it), then Gemini says interrupted, no turn_complete.
        await self.make()
        self.s.push(audio(b"\x01" * 480000))
        self.s.push(said("one two three four five six seven eight nine ten"))
        await self.settle()
        await asyncio.sleep(0.5)
        await self.b.barge()
        self.s.push(content(interrupted=True))
        await self.settle()
        cuts = self.dicts("live_cut")
        self.assertEqual(len(cuts), 1)
        self.assertNotIn("ten", cuts[0]["text"])
        await self.b.close()

    async def test_gemini_interrupt_remembers_what_she_heard(self):
        # R25 when Gemini itself says interrupted: the memory still gets the part she heard.
        await self.make()
        self.s.push(audio(b"\x01" * 480000))
        self.s.push(said("one two three four five six seven eight nine ten"))
        await self.settle()
        await asyncio.sleep(0.5)
        self.s.push(content(interrupted=True))
        await self.settle()
        cuts = self.dicts("live_cut")
        self.assertEqual(len(cuts), 1)
        self.assertNotIn("ten", cuts[0]["text"])
        await self.b.close()

    async def test_language_hint_follows_the_zh_button(self):
        # R24: the ZH/EN button picks the language Gemini listens for first.
        await self.make(language="zh-CN")
        self.assertEqual(self.configs[0].input_audio_transcription.language_codes[0], "cmn-Hans-CN")
        await self.b.close()
        await self.make()
        self.assertEqual(self.configs[0].input_audio_transcription.language_codes[0], "en-US")
        await self.b.close()

    async def test_barge_mid_reply_remembers_only_what_she_heard(self):
        # R25: she stops Dot early; the call's memory keeps only the start of the reply, not the unplayed rest.
        await self.make()
        self.s.push(audio(b"\x01" * 480000))   # a 10 s reply, all sent at once
        self.s.push(said("one two three four five six seven eight nine ten"))
        await self.settle()
        await asyncio.sleep(0.5)
        await self.b.barge()
        self.s.push(done())
        await self.settle()
        cuts = self.dicts("live_cut")
        self.assertEqual(len(cuts), 1)
        self.assertTrue(cuts[0]["text"].endswith("(she interrupted)") or "before hearing" in cuts[0]["text"])
        self.assertNotIn("ten", cuts[0]["text"])
        self.assertEqual(self.dicts("live_said"), [])
        await self.b.close()

    async def test_barge_keeps_her_first_words_for_hand_off(self):
        # R15: after a barge-in on the tail, the turn audio starts at her voice (not 0.3 s before the stop).
        await self.make()
        self.s.push(audio(b"\x01" * 192000))
        self.s.push(done())
        await self.settle()
        await asyncio.sleep(gemini_live.ECHO_GRACE_S)
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        await self.talk(0.8)          # she talks for 0.8 s (audio arrives faster than real time here)
        await asyncio.sleep(gemini_live.VA_CONFIRM_S + 0.1)
        self.assertEqual(len(self.dicts("live_stop")), 1)
        self.assertGreaterEqual(len(self.b._turn_buf), int(0.8 * 32000))
        await self.b.close()

    async def test_page_barge_stops_dot_without_voice_events(self):
        # R2 with no voice-activity event at all: the phone heard her and says so.
        await self.make()
        self.s.push(audio(b"\x01" * 240000))
        self.s.push(done())
        await self.settle()
        await self.b.barge()
        self.assertEqual(len(self.dicts("live_stop")), 1)
        self.assertEqual(self.b._play_until, 0.0)
        await self.b.close()

    async def test_echo_level_during_reply_is_not_her_voice(self):
        # G5/R2: steady speaker echo in her mic while Dot plays never counts as her; her real voice over it does.
        await self.make()
        self.s.push(audio(b"\x01" * 240000))   # 5 s reply
        await self.settle()
        echo = b"\x00\x06" * 1600               # 0.1 s at ~1500 RMS (above the 700 floor)
        for _ in range(10):
            await self.b.feed_audio(echo)
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        for _ in range(8):
            await self.b.feed_audio(echo)
        await asyncio.sleep(gemini_live.ECHO_GRACE_S + 0.1)
        self.assertFalse(self.dicts("live_stop"))
        await self.talk(0.3)                      # her voice, far above the echo
        await asyncio.sleep(0.15)
        self.assertEqual(len(self.dicts("live_stop")), 1)
        await self.b.close()

    async def test_talking_as_dot_starts_is_not_learned_as_echo(self):
        # Review round 3 blocker: she talks (normal level) from Dot's first moment; after the echo grace her voice
        # must still stop Dot.
        await self.make()
        self.s.push(audio(b"\x01" * 240000))
        await self.settle()
        voice = b"\x00\x0a" * 1600              # 0.1 s at ~2560 RMS: a normal voice, under the clear-voice level
        for _ in range(8):                        # during the grace: learned as echo, but capped
            await self.b.feed_audio(voice)
        await asyncio.sleep(gemini_live.ECHO_GRACE_S)
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        for _ in range(4):
            await self.b.feed_audio(voice)
        await asyncio.sleep(gemini_live.VA_CONFIRM_S + 0.15)
        self.assertEqual(len(self.dicts("live_stop")), 1)
        await self.b.close()

    async def test_status_gets_words_that_arrive_after_the_tool_call(self):
        # get_status uses THIS turn's words even when they arrive after Gemini's tool call (never "").
        await self.make()
        self.s.push(call("get_status", {}))
        await self.settle()
        self.s.push(heard("check my tasks"))
        await asyncio.sleep(0.1)
        await self.settle()
        if self.b.tool_task:
            await self.b.tool_task
        self.assertEqual(self.tools.status_queries[-1], "check my tasks")
        await self.b.close()

    async def test_echo_words_do_not_stop_dot(self):
        await self.make()
        self.s.push(said("Your next showing is at three."))
        self.s.push(audio(b"\x01" * 96000))
        self.s.push(done())
        await self.settle()
        self.s.push(heard("showing is at three"))   # her mic hears Dot
        await self.settle()
        self.assertFalse(self.dicts("live_stop"))
        await self.b.close()

    async def test_vad_config_keeps_pauses(self):
        # R3: low sensitivity and a long enough silence so a pause mid-sentence does not end her turn.
        await self.make()
        aad = self.configs[0].realtime_input_config.automatic_activity_detection
        self.assertGreaterEqual(aad.silence_duration_ms, 700)
        self.assertIn("LOW", str(aad.end_of_speech_sensitivity))
        self.assertNotIn("LOW", str(aad.start_of_speech_sensitivity))   # low start: Gemini heard nothing (real API)
        self.assertGreaterEqual(self.configs[0].context_window_compression.trigger_tokens, 16000)
        await self.b.close()

    async def test_hand_off_audio_has_no_dot_tail(self):
        # The turn audio for Whisper starts when Dot's playback ends, not at her turn_complete (no Dot echo).
        await self.make()
        self.s.push(audio(b"\x01" * 96000))
        self.s.push(done())
        await self.settle()
        await self.b.feed_audio(b"\x05\x00" * 16000)   # 1 s of mic while Dot's tail plays
        self.assertLessEqual(len(self.b._turn_buf), gemini_live.RECENT_BYTES)
        await self.b.close()

    async def test_her_voice_level_marks_speaking(self):
        # R8: even with no voice-activity event, her voice makes Dot busy (no session update over her).
        await self.make()
        loud = (b"\x00\x30" * 1600)   # 0.1 s of loud audio
        for _ in range(4):
            await self.b.feed_audio(loud)
        self.assertTrue(self.b.user_speaking)
        self.assertTrue(self.b.busy)
        await self.b.close()

    async def test_barge_in_words_start_next_turn(self):
        await self.make()
        self.s.push(heard("first question"))
        self.s.push(audio(b"\x01"))
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        self.s.push(content(interrupted=True))   # the service interrupts at her voice; her words come after
        self.s.push(heard("wait stop"))
        self.s.push(heard(" and call Alex"))
        self.s.push(audio(b"\x02"))
        await self.settle()
        texts = [m["text"] for m in self.dicts("live_heard")]
        self.assertEqual(texts, ["first question", "wait stop and call Alex"])
        await self.b.close()

    async def test_late_words_stay_out_of_next_turn(self):
        await self.make()
        self.s.push(heard("hello"))
        self.s.push(audio(b"\x01"))
        self.s.push(done())
        self.s.push(heard(" there"))   # late chunk of the finished turn
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        self.s.push(heard("new thing"))
        self.s.push(audio(b"\x02"))
        await self.settle()
        self.assertEqual([m["text"] for m in self.dicts("live_heard")], ["hello", "new thing"])
        await self.b.close()

    async def test_quota_close_fails_without_resume_loop(self):
        await self.make()
        self.s.push(NS(session_resumption_update=NS(resumable=True, new_handle="h1")))
        self.s.push(Exception("1011 RESOURCE_EXHAUSTED quota"))
        await self.settle()
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(len(self.dicts("live_error")), 1)
        realtime._DOWN_UNTIL.clear()

    async def test_send_error_with_handle_waits_for_resume(self):
        await self.make()
        self.s.push(NS(session_resumption_update=NS(resumable=True, new_handle="h1")))
        await self.settle()

        async def dead(audio=None):
            raise ConnectionError("socket gone")
        self.s.send_realtime_input = dead
        await self.b.feed_audio(b"\x05" * 10)
        self.assertFalse(self.b.closed)
        self.s.push(ConnectionError("socket gone"))
        await self.settle()
        self.assertEqual(len(self.sessions), 2)
        self.assertEqual(self.sessions[1].audio[0].data, b"\x05" * 10)   # held audio sent after the resume
        self.assertFalse(self.dicts("live_error"))
        await self.b.close()

    async def test_resume_gives_floor_back(self):
        await self.make()
        self.s.push(NS(session_resumption_update=NS(resumable=True, new_handle="h1")))
        self.s.push(audio(b"\x01"))
        await self.settle()
        self.assertTrue(self.b.response_active)
        self.s.push(NS(go_away=NS(time_left="1s")))
        await self.settle()
        self.assertEqual(len(self.sessions), 2)
        self.assertFalse(self.b.busy)
        await self.b.close()

    async def test_say_exact_sends_client_turn(self):
        await self.make()
        await self.b.say_exact("Bug recorded.")
        self.assertIn("Bug recorded.", self.s.client[0].parts[0].text)
        self.assertTrue(self.b.response_active)
        await self.b.close()


class QuotaFallbackTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        realtime._DOWN_UNTIL.clear()

    async def test_quota_error_pauses_gemini(self):
        with mock.patch.object(realtime, "_cfg", lambda k: "gemini" if k == "live_provider" else None):
            self.assertEqual(realtime.bridge_classes(), [gemini_live.GeminiBridge, realtime.LiveBridge])

            async def connect(model, config=None, api_key=None):
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
            with mock.patch.object(gemini_live.tts, "_keychain", return_value="k"):
                b = gemini_live.GeminiBridge(None, Tools(), "x", connect=connect)
                with self.assertRaises(RuntimeError):
                    await b.start()
            self.assertEqual(realtime.bridge_classes(), [realtime.LiveBridge])

    async def test_server_falls_back_to_qwen(self):
        from test_realtime import FakeLiveBridge
        svc = server.Services(server.FakeSpeechToText(""), server.FakeFrontDesk(), server.FakeTextToSpeech(),
                              server.FakeHandoff(), server.PinGuard("123456"), turn_log=os.devnull)
        ws = FakeCallWS([json.dumps({"type": "live", "on": True}), None])
        c = server.Call(ws, svc)
        c.authenticate = mock.AsyncMock(return_value=True)

        class Broken(FakeLiveBridge):
            async def start(self):
                raise RuntimeError("429 quota")

        class Qwen(FakeLiveBridge):
            pass

        with mock.patch.object(realtime, "bridge_classes", return_value=[Broken, Qwen]):
            await c.run()
        self.assertIsInstance(FakeLiveBridge.last_instance, Qwen)
        self.assertTrue(any(json.loads(m).get("type") == "live_ok" for m in ws.sent if isinstance(m, str)))


class MidCallFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_error_mid_call_switches_to_qwen(self):
        from test_realtime import FakeLiveBridge
        svc = server.Services(server.FakeSpeechToText(""), server.FakeFrontDesk(), server.FakeTextToSpeech(),
                              server.FakeHandoff(), server.PinGuard("123456"), turn_log=os.devnull)
        ws = FakeCallWS([])
        c = server.Call(ws, svc)

        class Gem(FakeLiveBridge):
            falls_back_to_qwen = True

        class Qwen(FakeLiveBridge):
            pass

        gem = Gem(c._live_out, None, "", on_closed=lambda: c._on_live_closed(gem))
        c.live = gem
        with mock.patch.object(realtime, "LiveBridge", Qwen):
            await c._live_out({"type": "live_error", "text": "429 quota"})
            gem.on_closed()   # the failed bridge reports closed after its error: must not end Qwen
        self.assertIsInstance(c.live, Qwen)
        self.assertTrue(c.live.started)
        types_sent = [json.loads(m).get("type") for m in ws.sent if isinstance(m, str)]
        self.assertIn("live_ok", types_sent)
        self.assertNotIn("live_error", types_sent)

    async def test_qwen_error_still_reaches_phone(self):
        from test_realtime import FakeLiveBridge
        svc = server.Services(server.FakeSpeechToText(""), server.FakeFrontDesk(), server.FakeTextToSpeech(),
                              server.FakeHandoff(), server.PinGuard("123456"), turn_log=os.devnull)
        ws = FakeCallWS([])
        c = server.Call(ws, svc)
        c.live = FakeLiveBridge(c._live_out, None, "")
        await c._live_out({"type": "live_error", "text": "boom"})
        self.assertIsNone(c.live)
        self.assertIn("live_error", [json.loads(m).get("type") for m in ws.sent if isinstance(m, str)])


class ServerProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_live_provider_gemini_builds_gemini_bridge(self):
        from test_realtime import FakeLiveBridge
        svc = server.Services(server.FakeSpeechToText(""), server.FakeFrontDesk(), server.FakeTextToSpeech(),
                              server.FakeHandoff(), server.PinGuard("123456"), turn_log=os.devnull)
        ws = FakeCallWS([json.dumps({"type": "live", "on": True}), None])
        c = server.Call(ws, svc)
        c.authenticate = mock.AsyncMock(return_value=True)

        class FakeGemini(FakeLiveBridge):
            pass

        real_cfg = realtime._cfg
        with mock.patch.object(realtime, "_cfg", lambda k: "gemini" if k == "live_provider" else real_cfg(k)), \
                mock.patch.object(gemini_live, "GeminiBridge", FakeGemini), \
                mock.patch.object(realtime, "LiveBridge", side_effect=AssertionError("Qwen used")):
            await c.run()
        self.assertIsInstance(FakeLiveBridge.last_instance, FakeGemini)
        self.assertTrue(FakeLiveBridge.last_instance.started)


if __name__ == "__main__":
    unittest.main()
