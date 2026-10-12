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

    async def close(self):
        self.closed = True


class GeminiBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def make(self, transcribe=None, key="k"):
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
            await self.b.start()
        self.s = self.sessions[0]

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
        self.assertEqual(names, ["hand_off", "get_status", "lookup"])
        self.assertIsNotNone(cfg.session_resumption)
        self.assertIsNotNone(cfg.context_window_compression.sliding_window)
        self.assertEqual(cfg.system_instruction.parts[0].text if hasattr(cfg.system_instruction, "parts")
                         else cfg.system_instruction, "be brief")
        await self.b.close()

    async def test_missing_key_raises(self):
        with self.assertRaises(RuntimeError):
            await self.make(key=None)

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

    async def test_noise_during_reply_does_not_cut_dot_off(self):
        await self.make()
        self.s.push(audio(b"\x01"))
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        await self.settle()
        self.assertFalse(self.dicts("live_stop"))
        await self.b.close()

    async def test_barge_in_words_start_next_turn(self):
        await self.make()
        self.s.push(heard("first question"))
        self.s.push(audio(b"\x01"))
        self.s.push(NS(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
        self.s.push(heard("wait stop"))
        self.s.push(content(interrupted=True))
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
