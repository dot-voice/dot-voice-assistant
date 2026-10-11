import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
# Tests never read the owner's local_config.json (set before config is imported).
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import realtime  # noqa: E402
import server  # noqa: E402


class FakeWS:
    def __init__(self, events):
        self.q = asyncio.Queue()
        for e in events:
            self.q.put_nowait(json.dumps(e))
        self.sent = []
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        return await self.q.get()

    def push(self, ev):
        self.q.put_nowait(json.dumps(ev))

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self.q.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def close(self):
        self.closed = True


class Tools:
    def __init__(self):
        self.calls = []
        self.lookups = []

    async def hand_off(self, words, agent):
        self.calls.append((words, agent))
        return "posted to CRM"

    async def status(self, q):
        return "all quiet"

    async def lookup(self, q):
        self.lookups.append(q)
        return "lookup result: " + str(q)


class LiveBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def make(self, transcribe=None):
        self.ws = FakeWS([{"type": "session.created"}, {"type": "session.updated"}])
        self.phone = []

        async def to_phone(x):
            self.phone.append(x)

        async def connect(url, additional_headers=None):
            self.url = url
            return self.ws

        self.tools = Tools()
        with mock.patch.object(realtime.tts, "_keychain", return_value="k"):
            self.b = realtime.LiveBridge(to_phone, self.tools, "be brief", connect=connect, transcribe=transcribe)
            await self.b.start()

    async def settle(self):
        for _ in range(10):
            await asyncio.sleep(0)

    async def test_session_update_has_tools_and_voice(self):
        await self.make()
        upd = self.ws.sent[0]
        self.assertEqual(upd["type"], "session.update")
        names = [t["name"] for t in upd["session"]["tools"]]
        self.assertEqual(names, ["hand_off", "get_status", "lookup"])
        self.assertTrue(upd["session"]["voice"])
        self.assertIn("model=", self.url)
        await self.b.close()

    async def test_audio_delta_to_phone(self):
        await self.make()
        self.ws.push({"type": "response.audio.delta", "delta": base64.b64encode(b"\x01\x02").decode()})
        await self.settle()
        self.assertIn(b"\x01\x02", self.phone)
        await self.b.close()

    async def test_barge_in(self):
        await self.make()
        self.ws.push({"type": "response.audio.delta", "delta": ""})
        self.ws.push({"type": "input_audio_buffer.speech_started"})
        await self.settle()
        self.assertIn({"type": "response.cancel"}, self.ws.sent)
        self.assertIn({"type": "live_stop"}, self.phone)
        await self.b.close()

    async def test_hand_off_uses_heard_words(self):
        await self.make()
        self.ws.push({"type": "input_audio_buffer.speech_started"})
        self.ws.push({"type": "input_audio_buffer.speech_stopped"})
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "tell crm hello"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c1",
                      "arguments": json.dumps({"agent": "CRM", "words": "WRONG"})})
        await self.settle()
        self.assertEqual(self.tools.calls, [("tell crm hello", "CRM")])
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["output"], "posted to CRM")
        self.assertEqual(out[0]["item"]["call_id"], "c1")
        self.assertIn({"type": "response.create"}, self.ws.sent)
        await self.b.close()

    async def test_close_sends_live_error(self):
        await self.make()
        self.ws.q.put_nowait(None)
        await self.settle()
        self.assertEqual(self.phone[-1]["type"], "live_error")
        await self.b.close()

    async def test_function_call_before_transcript_waits_for_it(self):
        await self.make()
        self.ws.push({"type": "input_audio_buffer.speech_started"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c_wait",
                      "arguments": json.dumps({"agent": "CRM"})})
        await asyncio.sleep(0.02)
        self.assertEqual(self.tools.calls, [])
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "delayed text"})
        await self.settle()
        self.assertEqual(self.tools.calls, [("delayed text", "CRM")])
        await self.b.close()

    async def test_cancel_sends_cancel_only_if_response_active(self):
        await self.make()
        self.b.response_active = False
        await self.b.cancel()
        self.assertNotIn({"type": "response.cancel"}, self.ws.sent)
        self.b.response_active = True
        await self.b.cancel()
        self.assertIn({"type": "response.cancel"}, self.ws.sent)
        self.assertFalse(self.b.response_active)
        await self.b.close()

    async def test_whisper_fallback_when_no_provider_transcript(self):
        fake_transcribe = mock.Mock(return_value="transcribed by whisper")
        await self.make(transcribe=fake_transcribe)
        self.ws.push({"type": "input_audio_buffer.speech_started"})
        self.ws.push({"type": "input_audio_buffer.speech_stopped"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c_wh",
                      "arguments": json.dumps({"agent": "CRM"})})
        await self.settle()
        self.assertEqual(self.tools.calls, [("transcribed by whisper", "CRM")])
        fake_transcribe.assert_called_once()
        self.assertTrue(any(isinstance(p, dict) and p.get("type") == "live_heard" and p.get("text") == "transcribed by whisper" for p in self.phone))
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["output"], "posted to CRM")
        await self.b.close()

    async def test_empty_words_refuses_hand_off(self):
        await self.make()
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": ""})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c_empty",
                      "arguments": json.dumps({"agent": "CRM"})})
        await self.settle()
        self.assertEqual(self.tools.calls, [])
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertTrue(out[0]["item"]["output"].startswith("not sent"))
        self.assertEqual(out[0]["item"]["call_id"], "c_empty")
        await self.b.close()

    async def test_lookup_uses_turn_words(self):
        await self.make()
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "what is the weather in London"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "lookup", "call_id": "c_look",
                      "arguments": json.dumps({"question": "IGNORE ARGUMENT"})})
        await self.settle()
        self.assertEqual(self.tools.lookups, ["what is the weather in London"])
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["output"], "lookup result: what is the weather in London")
        self.assertEqual(out[0]["item"]["call_id"], "c_look")
        await self.b.close()

    async def test_max_handoffs_per_call(self):
        await self.make()
        self.b.handoff_count = 6
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "pass to crm"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c_limit",
                      "arguments": json.dumps({"agent": "CRM"})})
        await self.settle()
        self.assertEqual(self.tools.calls, [])
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertIn("limit reached", out[0]["item"]["output"])
        await self.b.close()

    async def test_tool_done_while_she_talks_does_not_talk_over_her(self):
        await self.make()
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "any news"})
        await self.settle()
        self.b._in_turn = True   # she started a new sentence while the tool runs
        self.ws.push({"type": "response.function_call_arguments.done", "name": "get_status", "call_id": "c_bt",
                      "arguments": "{}"})
        await self.settle()
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["call_id"], "c_bt")
        self.assertNotIn({"type": "response.create"}, self.ws.sent)
        await self.b.close()

    async def test_tool_timeout_output_timed_out(self):
        await self.make()
        self.ws.push({"type": "response.function_call_arguments.done", "name": "get_status", "call_id": "c_to",
                      "arguments": "{}"})
        with mock.patch("realtime.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await self.settle()
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["output"], "timed out")
        self.assertEqual(out[0]["item"]["call_id"], "c_to")
        await self.b.close()

    async def test_non_fatal_provider_error_does_not_send_live_error(self):
        await self.make()
        self.ws.push({"type": "error", "error": {"message": "Active response in progress"}})
        self.ws.push({"type": "error", "error": {"message": "Cannot cancel response"}})
        await self.settle()
        self.assertFalse(any(isinstance(p, dict) and p.get("type") == "live_error" for p in self.phone))
        self.assertFalse(self.b.closed)
        self.assertFalse(self.b._failed)
        await self.b.close()

    async def test_say_exact_sends_only_response_create(self):
        await self.make()
        sent_before = len(self.ws.sent)
        await self.b.say_exact("Session update: CRM finished")
        sent_after = self.ws.sent[sent_before:]
        self.assertEqual(len(sent_after), 1)
        self.assertEqual(sent_after[0]["type"], "response.create")
        self.assertNotIn("conversation.item.create", [s.get("type") for s in sent_after])
        self.assertIn("Say exactly this", sent_after[0]["response"]["instructions"])
        self.assertTrue(self.b.response_active)
        await self.b.close()

    async def test_audio_deltas_of_cancelled_response_id_are_dropped(self):
        await self.make()
        self.b.response_active = True
        self.b.current_response_id = "resp_cancelled_1"
        await self.b.cancel()
        self.assertIn("resp_cancelled_1", self.b.cancelled_response_ids)

        delta = base64.b64encode(b"\xde\xad").decode()
        self.ws.push({"type": "response.audio.delta", "response_id": "resp_cancelled_1", "delta": delta})
        await self.settle()
        self.assertNotIn(b"\xde\xad", self.phone)
        await self.b.close()


class FakeCallWS:
    def __init__(self, incoming):
        self.q = asyncio.Queue()
        for item in incoming:
            self.q.put_nowait(item)
        self.sent = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return await self.q.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.q.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self):
        self.closed = True


class FakeLiveBridge:
    last_instance = None

    def __init__(self, send_to_phone, tools_impl, instructions, model=None, connect=None,
                 transcribe=None, on_closed=None, turn_log=None, **kwargs):
        FakeLiveBridge.last_instance = self
        self.send_to_phone = send_to_phone
        self.tools_impl = tools_impl
        self.instructions = instructions
        self.model = model
        self.connect = connect
        self.transcribe = transcribe
        self.on_closed = on_closed
        self.turn_log = turn_log
        self.started = False
        self.closed = False
        self.fed = []
        self.cancelled = False
        self.response_active = False
        self.stopped_at = None

    async def start(self):
        self.started = True

    async def feed_audio(self, pcm):
        self.fed.append(pcm)

    async def cancel(self):
        self.cancelled = True

    async def close(self):
        self.closed = True

    async def say_exact(self, text):
        pass


class ServerLiveTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeLiveBridge.last_instance = None
        self.svc = server.Services(
            server.FakeSpeechToText(""),
            server.FakeFrontDesk(),
            server.FakeTextToSpeech(),
            server.FakeHandoff(),
            server.PinGuard("123456"),
            turn_log="/tmp/test_turn.log",
        )

    async def test_live_on_start_called(self):
        ws = FakeCallWS([json.dumps({"type": "live", "on": True}), None])
        call = server.Call(ws, self.svc)
        call.authenticate = mock.AsyncMock(return_value=True)
        with mock.patch("realtime.LiveBridge", FakeLiveBridge):
            await call.run()
        self.assertIsNotNone(FakeLiveBridge.last_instance)
        self.assertTrue(FakeLiveBridge.last_instance.started)
        self.assertTrue(FakeLiveBridge.last_instance.closed)
        self.assertIsNone(call.live)
        self.assertFalse(call.live_pending)
        self.assertTrue(any(json.loads(m).get("type") == "live_ok" for m in ws.sent if isinstance(m, str)))

    async def test_binary_while_live_feeds_audio_no_turn(self):
        pcm = b"\x01\x02" * 160
        ws = FakeCallWS([pcm, None])
        call = server.Call(ws, self.svc)
        call.authenticate = mock.AsyncMock(return_value=True)
        fake_bridge = FakeLiveBridge(call._live_out, None, "")
        call.live = fake_bridge
        call._guarded_turn = mock.AsyncMock()
        await call.run()
        self.assertEqual(fake_bridge.fed, [pcm])
        call._guarded_turn.assert_not_called()
        self.assertIsNone(call.task)

    async def test_binary_frames_while_live_pending_do_not_start_old_mode_turn(self):
        pcm = b"\x01\x02" * 16000
        ws = FakeCallWS([pcm, None])
        call = server.Call(ws, self.svc)
        call.authenticate = mock.AsyncMock(return_value=True)
        call.live_pending = True
        call._guarded_turn = mock.AsyncMock()
        await call.run()
        call._guarded_turn.assert_not_called()
        self.assertIsNone(call.task)

    async def test_live_audio_json_is_base64(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        pcm = b"\x10\x20\x30\x40"
        await call._live_out(pcm)
        self.assertEqual(len(ws.sent), 1)
        msg = json.loads(ws.sent[0])
        self.assertEqual(msg.get("type"), "live_audio")
        self.assertEqual(msg.get("b64"), base64.b64encode(pcm).decode("ascii"))
        self.assertEqual(base64.b64decode(msg.get("b64")), pcm)

    async def test_hand_off_returns_posted_to_agent_only_when_posted(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        tools = server.LiveTools(call)

        # Record says posted: True
        self.svc.handoff.create = mock.Mock(return_value={"posted": True, "to": "CRM", "id": "1"})
        res = await tools.hand_off("research area", "CRM")
        self.assertEqual(res, "posted to CRM")

        # Record says posted: False
        self.svc.handoff.create = mock.Mock(return_value={"posted": False, "to": "CRM", "post_error": "carry down"})
        res = await tools.hand_off("research area", "CRM")
        self.assertEqual(res, "not sent: carry down")

    async def test_live_tools_max_handoffs_limit(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        call.handoff_count = 6
        tools = server.LiveTools(call)
        res = await tools.hand_off("research area", "CRM")
        self.assertEqual(res, "limit reached, tell the caller to use the menu")

    async def test_status_direct_status_and_fallback(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        tools = server.LiveTools(call)

        with mock.patch("server.direct_status", return_value="CRM is updating leads"):
            res = await tools.status("what is CRM doing")
            self.assertEqual(res, "CRM is updating leads")

        with mock.patch("server.direct_status", return_value=None):
            res = await tools.status("what is happening")
            self.assertEqual(res, "No saved status for that.")

    async def test_live_tools_lookup(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        tools = server.LiveTools(call)
        with mock.patch("frontdesk.gather_notes", return_value="some notes"):
            res = await tools.lookup("my rules")
            self.assertEqual(res, "some notes")

    async def test_interrupt_while_live_cancels_bridge(self):
        ws = FakeCallWS([json.dumps({"type": "interrupt"}), None])
        call = server.Call(ws, self.svc)
        call.authenticate = mock.AsyncMock(return_value=True)
        fake_bridge = FakeLiveBridge(call._live_out, None, "")
        call.live = fake_bridge
        await call.run()
        self.assertTrue(fake_bridge.cancelled)

    async def test_say_update_uses_say_exact_when_live(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        fake_bridge = FakeLiveBridge(call._live_out, None, "")
        fake_bridge.say_exact = mock.AsyncMock()
        call.live = fake_bridge
        call.speak = mock.AsyncMock()

        await call._say_update("Update from CRM: leads refreshed")
        fake_bridge.say_exact.assert_called_once_with("Update from CRM: leads refreshed")
        call.speak.assert_not_called()

    async def test_first_audio_logs_latency(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        fake_bridge = FakeLiveBridge(call._live_out, None, "")
        fake_bridge.stopped_at = time.monotonic() - 0.200
        call.live = fake_bridge

        with mock.patch("server.log_line") as mock_log:
            await call._live_out(b"\xaa\xbb")
            mock_log.assert_called()
            line = mock_log.call_args[0][0]
            self.assertTrue(line.startswith("live first_audio_ms="))
            self.assertIsNone(fake_bridge.stopped_at)

    async def test_call_summary_file_written_only_when_handoff_posted_without_private_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = FakeCallWS([])
            call = server.Call(ws, self.svc)
            call.svc.handoff.folder = tmpdir

            # 1. No handoffs posted -> no summary file
            call.posted_handoffs = []
            call._write_call_summary()
            self.assertEqual(os.listdir(tmpdir), [])

            # 2. Only private roles posted -> no summary file
            call.posted_handoffs = [{"to": "PRIVATE", "words": "secret message"}]
            call._write_call_summary()
            self.assertEqual(os.listdir(tmpdir), [])

            # 3. Valid handoff and private handoff -> summary written excluding private role
            call.posted_handoffs = [
                {"to": "CRM", "words": "follow up with client"},
                {"to": "PRIVATE", "words": "secret note"},
            ]
            call._write_call_summary()
            files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
            self.assertEqual(data["source"], "voice_call_summary")
            self.assertEqual(data["handoffs"], [{"to": "CRM", "words": "follow up with client"}])


class TestBoardRoute(unittest.TestCase):
    def test_board_needs_a_live_resume_token(self):
        import server
        svc = server.Services(server.FakeSpeechToText(""), server.FakeFrontDesk(), server.FakeTextToSpeech(),
                              server.FakeHandoff(), server.PinGuard("123456"))
        pr = server.make_process_request(svc)

        class Req:
            def __init__(self, path):
                self.path, self.headers = path, {}

        class Conn:
            def respond(self, status, body):
                class R:
                    pass
                r = R()
                r.status, r.body, r.headers = status, body, {}
                return r
        self.assertEqual(pr(Conn(), Req("/board")).status, server.HTTPStatus.FORBIDDEN)
        self.assertEqual(pr(Conn(), Req("/board?t=wrong")).status, server.HTTPStatus.FORBIDDEN)
        tok = svc.issue_resume()
        self.assertNotEqual(pr(Conn(), Req("/board?t=" + tok)).status, server.HTTPStatus.FORBIDDEN)


if __name__ == "__main__":
    unittest.main()
