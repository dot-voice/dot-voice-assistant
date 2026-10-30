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

    async def status(self, q, session=None):
        self.status_sessions = getattr(self, "status_sessions", []) + [session]
        self.status_queries = getattr(self, "status_queries", []) + [q]
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
        self.assertEqual(names, ["hand_off", "get_status", "cancel_hand_off", "transfer_call", "lookup", "ask_model", "send_to_phone", "next_events", "last_call", "unread_digest", "talk_mode"])
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

    async def test_session_uses_semantic_vad_and_qwen_asr(self):
        await self.make()
        upd = [e for e in self.ws.sent if e.get("type") == "session.update"][0]["session"]
        self.assertEqual(upd["turn_detection"]["type"], "semantic_vad")
        self.assertEqual(upd["input_audio_transcription"]["model"], "qwen3-asr-flash-realtime")
        await self.b.close()

    async def test_whisper_fallback_when_no_provider_transcript(self):
        fake_transcribe = mock.Mock(return_value="transcribed by whisper")
        await self.make(transcribe=fake_transcribe)
        self.b.transcript_wait_s = 0
        self.ws.push({"type": "input_audio_buffer.speech_started"})
        self.ws.push({"type": "input_audio_buffer.speech_stopped"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c_wh",
                      "arguments": json.dumps({"agent": "CRM"})})
        await self.settle()
        if self.b.tool_task:
            await self.b.tool_task   # Whisper runs in a thread; settle() alone raced under a busy suite
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
                      "arguments": json.dumps({"question": "weather London"})})
        await self.settle()
        self.assertEqual(self.tools.lookups, ["weather London what is the weather in London"])
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["output"], "lookup result: weather London what is the weather in London")
        self.assertEqual(out[0]["item"]["call_id"], "c_look")
        await self.b.close()

    async def test_live_calls_ask_model_with_the_model_and_question(self):
        await self.make()
        got = []

        async def ask_model(model, question):
            got.append((model, question))
            return "Asking Sonnet now"
        self.tools.ask_model = ask_model
        self.ws.push({"type": "response.function_call_arguments.done", "name": "ask_model", "call_id": "c_ask",
                      "arguments": json.dumps({"model": "Sonnet", "question": "Is the price fair?"})})
        await self.settle()
        self.assertEqual(got, [("Sonnet", "Is the price fair?")])
        out = [e for e in self.ws.sent if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(out[0]["item"]["output"], "Asking Sonnet now")
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

    async def test_hand_off_prefers_whisper_over_qwen_transcript(self):
        fake_transcribe = mock.Mock(return_value="is anything stuck")
        await self.make(transcribe=fake_transcribe)
        self.ws.push({"type": "input_audio_buffer.speech_started"})
        self.ws.push({"type": "input_audio_buffer.speech_stopped"})
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "wrong language"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "hand_off", "call_id": "c_w",
                      "arguments": json.dumps({"agent": "ORG"})})
        await self.settle()
        await asyncio.sleep(0.05)
        self.assertEqual(self.tools.calls, [("is anything stuck", "ORG")])
        await self.b.close()

    async def test_get_status_passes_the_models_session(self):
        await self.make()
        self.ws.push({"type": "conversation.item.input_audio_transcription.completed", "transcript": "how is crm"})
        self.ws.push({"type": "response.function_call_arguments.done", "name": "get_status", "call_id": "c_gs",
                      "arguments": json.dumps({"session": "CRM"})})
        await self.settle()
        self.assertEqual(self.tools.status_sessions, ["CRM"])
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
        p = mock.patch.object(server, "SUGGEST_TRANSFERS", False)   # R29 is tested in test_transfer.py; these predate it
        p.start()
        self.addCleanup(p.stop)
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

    async def test_live_on_greets_once_per_call(self):
        """R56: her words 2026-10-28 "then authentication still took a loooooong time" - 58 s of silence after sign-in."""
        on = json.dumps({"type": "live", "on": True})
        ws = FakeCallWS([on, on, None])
        call = server.Call(ws, self.svc)
        call.authenticate = mock.AsyncMock(return_value=True)
        with mock.patch("realtime.LiveBridge", FakeLiveBridge):
            await call.run()
        self.assertTrue(call.greeted)
        line = "%s here. Go ahead." % server.config.get("assistant")
        self.assertEqual(call.state.notices.count(line), 1)

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
        res = await tools._post("research area", "CRM")
        self.assertEqual(res, "posted to CRM")

        # Record says posted: False
        self.svc.handoff.create = mock.Mock(return_value={"posted": False, "to": "CRM", "post_error": "carry down"})
        res = await tools._post("research area", "CRM")
        self.assertEqual(res, "not sent: carry down")

    async def test_hand_off_can_be_corrected_before_it_is_sent(self):
        # R26 (her gap G4, 2026-10-12): "no, I meant CRM" right after a hand-off changes it; nothing goes out early.
        call = server.Call(FakeCallWS([]), self.svc)
        tools = server.LiveTools(call)
        self.svc.handoff.create = mock.Mock(return_value={"posted": True, "to": "CRM", "id": "1"})
        with mock.patch.object(server, "HANDOFF_HOLD_S", 0.2):
            res = await tools.hand_off("Call Alex about the offer.", "MW")
            self.assertTrue(res.startswith("queued"))
            self.svc.handoff.create.assert_not_called()
            res = await tools.hand_off("No, I meant CRM.", "CRM", replaces_waiting=True)
            self.assertTrue(res.startswith("updated"))
            await asyncio.sleep(0.35)
            self.svc.handoff.create.assert_called_once()
            args, kw = self.svc.handoff.create.call_args
            self.assertEqual(kw.get("role"), "CRM")
            self.assertEqual(args[0], "Call Alex about the offer. No, I meant CRM.")
            self.assertEqual(len(call.posted_handoffs), 1)
            # a new request (not a correction) never swallows the waiting one: both go out
            await tools.hand_off("Ask CRM about Sam.", "CRM")
            await tools.hand_off("Ask Vault about Main Street.", "VAULT")
            self.assertEqual(self.svc.handoff.create.call_count, 2)
            await asyncio.sleep(0.35)
        self.assertEqual([c.kwargs.get("role") for c in self.svc.handoff.create.call_args_list], ["CRM", "CRM", "VAULT"])

    async def test_failed_held_hand_off_is_said_when_she_is_quiet(self):
        # R8 + R26: a held hand-off that fails is told to her, but never over her voice (the updates loop says it).
        call = server.Call(FakeCallWS([]), self.svc)
        self.svc.handoff.create = mock.Mock(return_value={"posted": False, "post_error": "down"})
        tools = server.LiveTools(call)
        await tools.hand_off("Call Alex.", "CRM")
        await tools.post_pending()
        self.assertEqual(len(call.state.notices), 1)

    async def test_channels_rows_waiting_sent_replied(self):
        # R27: waiting -> sent -> replied; the list is sent to the phone on every change.
        import updates
        folder = tempfile.mkdtemp()
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        call.state.call = call
        tools = server.LiveTools(call)
        self.svc.handoff.create = mock.Mock(return_value={"posted": True, "to": "VAULT", "id": "1"})

        def last_rows():
            msgs = [json.loads(m) for m in ws.sent if isinstance(m, str)]
            return [m for m in msgs if m.get("type") == "channels"][-1]["rows"]

        with mock.patch.object(server, "HANDOFF_HOLD_S", 0.1):
            await tools.hand_off("Please check Main Street for Alex and tell me what the owner paid last time.", "CRM")
            rows = last_rows()
            self.assertEqual([r["status"] for r in rows], ["waiting"])
            self.assertLessEqual(len(rows[0]["snippet"]), 40)
            await asyncio.sleep(0.25)
        self.assertEqual([r["status"] for r in last_rows()], ["sent"])
        # an older update does not count; an update from the aliased role after the post is the reply
        updates.post("VAULT", "Old news.", folder=folder)
        path = os.path.join(folder, os.listdir(folder)[0])
        with open(path) as f:
            rec = json.load(f)
        rec["created"] = call.state.channel_rows[0]["sent_at"] - 100
        self.assertFalse(server.channel_replies(call.state, [(path, rec)]))
        rec["created"] = time.time() + 1
        rec["from"] = "VAULT"
        long_text = "The owner paid 1 dollar. " * 10
        rec["text"] = long_text
        self.assertTrue(server.channel_replies(call.state, [(path, rec)]))
        await server.channels_push(call.state)
        row = last_rows()[0]
        self.assertEqual(row["status"], "replied")
        self.assertLessEqual(len(row["reply"]), 60)
        # the same update never answers a second hand-off
        call.state.channel_rows.append(dict(call.state.channel_rows[0], id="x", status="sent", reply="", sent_at=0))
        call.state.channel_rows[1].pop("upd", None)
        self.assertFalse(server.channel_replies(call.state, [(path, rec)]))

    async def test_channels_rows_cancelled_and_failed(self):
        # R27
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        call.state.call = call
        tools = server.LiveTools(call)
        await tools.hand_off("Call Alex.", "CRM")
        await tools.cancel_hand_off()
        self.assertEqual([r["status"] for r in call.state.channel_rows], ["cancelled"])
        self.svc.handoff.create = mock.Mock(return_value={"posted": False, "post_error": "down"})
        await tools.hand_off("Call Sam.", "CRM")
        await tools.post_pending()
        self.assertEqual([r["status"] for r in call.state.channel_rows], ["cancelled", "failed"])
        # a correction changes the waiting row's label and words, not the row count
        await tools.hand_off("Ask about Main Street.", "MW")
        await tools.hand_off("No, I meant CRM.", "CRM", replaces_waiting=True)
        self.assertEqual(len(call.state.channel_rows), 3)
        self.assertEqual(call.state.channel_rows[2]["label"], "CRM")
        await tools.cancel_hand_off()
        for i in range(12):
            server.channel_row_add(call.state, "w%d" % i, "CRM", "sent")
        self.assertEqual(len(call.state.channel_rows), server.CHANNEL_ROWS_MAX)

    async def test_channels_rows_resent_after_reconnect(self):
        # R27: the rows live on the shared CallState; the new line gets them right after auth_ok.
        old = server.Call(FakeCallWS([]), self.svc)
        server.channel_row_add(old.state, "Call Alex.", "CRM", "sent")
        new_ws = FakeCallWS([])
        new = server.Call(new_ws, self.svc)
        new.state = old.state            # what the resume token does
        new.history = new.state.history
        new.state.call = new
        await server.channels_push(new.state)
        msgs = [json.loads(m) for m in new_ws.sent if isinstance(m, str)]
        self.assertEqual(msgs[-1]["type"], "channels")
        self.assertEqual(msgs[-1]["rows"][0]["status"], "sent")
        self.assertNotIn("sent_at", msgs[-1]["rows"][0])

    async def test_new_live_session_never_overwrites_older_lines(self):
        # R23: each Live session numbers its lines from a new base, so a restart (ZH/EN, reconnect) adds lines.
        call = server.Call(FakeCallWS([]), self.svc)
        for _ in range(2):
            base = call.state.tag_base = call.state.tag_base + 100000
            await call._live_out({"type": "live_heard", "text": "question %d" % base, "turn": base})
            await call._live_out({"type": "live_said", "text": "answer %d" % base, "reply": base + 1})
        self.assertEqual(len(call.history), 4)

    async def test_replaced_call_is_silent(self):
        # R23: after the phone came back on a new line, the old Call writes nothing to the shared memory.
        call = server.Call(FakeCallWS([]), self.svc)
        await call.take_over()
        await call._live_out({"type": "live_said", "text": "late words", "reply": 9})
        self.assertEqual(call.history, [])
        self.assertFalse(await call._live_to_qwen(mock.MagicMock(falls_back_to_qwen=True)))

    async def test_hand_off_cancel_and_flush(self):
        call = server.Call(FakeCallWS([]), self.svc)
        tools = server.LiveTools(call)
        self.svc.handoff.create = mock.Mock(return_value={"posted": True, "to": "CRM", "id": "1"})
        await tools.hand_off("Call Alex.", "CRM")
        self.assertEqual(await tools.cancel_hand_off(), "cancelled, nothing was sent")
        self.assertTrue((await tools.cancel_hand_off()).startswith("nothing to cancel"))
        await tools.hand_off("Call Sam.", "CRM")
        await tools.post_pending()   # the call / Live ends: a waiting hand-off goes out at once, never lost
        self.svc.handoff.create.assert_called_once()
        self.assertIsNone(call.state.pending_handoff)

    async def test_live_tools_max_handoffs_limit(self):
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        call.handoff_count = 6
        tools = server.LiveTools(call)
        res = await tools.hand_off("research area", "CRM")
        self.assertTrue(res.startswith("limit reached: 6 hand-offs"))

    async def test_status_named_session_and_overview(self):
        # 2026-10-11 her real call: "check my tasks" hit a phrase rule and got "No saved status". The model now
        # picks the session; no phrase rules in the Live path.
        ws = FakeCallWS([])
        call = server.Call(ws, self.svc)
        tools = server.LiveTools(call)
        with mock.patch("status.session_status", return_value="CRM saved its notes at 21:00") as one, \
                mock.patch("status.overview", return_value="Latest from your sessions: all") as allp:
            self.assertEqual(await tools.status("how is the client list going", session="CRM"),
                             "CRM saved its notes at 21:00")
            one.assert_called_once_with("CRM")
            for q in ("check my tasks", "any updates", "give me an update", "whats new", "is anything stuck"):
                self.assertEqual(await tools.status(q), "Latest from your sessions: all")
            self.assertEqual(await tools.status("x", session="AUTO"), "Latest from your sessions: all")
        with mock.patch("status.session_status", return_value=None), mock.patch("status.overview", return_value=None):
            self.assertEqual(await tools.status("check my tasks"), "No saved status for that.")

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

    async def test_live_words_become_call_memory_without_doubles(self):
        # R23: Live writes the call's memory; a second version of the same line replaces it (Whisper words, a
        # cut reply), her next question never replaces the last one, an update is stored once.
        call = server.Call(FakeCallWS([]), self.svc)
        await call._live_out({"type": "live_heard", "text": "is anything", "turn": 1})
        await call._live_out({"type": "live_heard", "text": "Is anything stuck?", "turn": 1})
        await call._live_out({"type": "live_heard", "text": "Call Alex.", "turn": 2})   # her reply was dropped
        await call._live_out({"type": "live_said", "text": "Sent to the list. Anything else?", "reply": 1})
        await call._live_out({"type": "live_cut", "text": "Sent to the ... (she interrupted)", "reply": 1})
        self.assertEqual([(l["role"], l["text"]) for l in call.history], [
            ("user", "Is anything stuck?"), ("user", "Call Alex."),
            ("assistant", "Sent to the ... (she interrupted)")])
        self.assertFalse(any("live_cut" in m for m in call.ws.sent))   # memory only
        self.assertFalse(any('"turn"' in m or '"reply"' in m for m in call.ws.sent))
        call.live = mock.MagicMock(busy=False, say_exact=mock.AsyncMock())
        await call._say_update("CRM says done.")
        self.assertEqual(len(call.history), 3)   # Live's own words (live_said) are the only writer

    async def test_seed_lines_capped(self):
        call = server.Call(FakeCallWS([]), self.svc)
        for i in range(60):
            call.remember("user" if i % 2 else "assistant", "line %d " % i + "x" * 200)
        self.assertEqual(len(call.history), server.HISTORY_MAX)
        seed = call.seed_lines()
        self.assertLessEqual(sum(len(l["text"]) for l in seed), server.SEED_CHARS)
        self.assertTrue(seed[-1]["text"].startswith("line 59"))

    def test_qwen_gets_the_call_in_its_instructions(self):
        note = realtime.history_note([{"role": "user", "text": "the blue house"}, {"role": "assistant", "text": "Noted."}])
        self.assertIn("Caller: the blue house", note)
        self.assertIn("You: Noted.", note)
        self.assertEqual(realtime.history_note([]), "")

    def test_cut_text_keeps_what_she_heard(self):
        self.assertEqual(realtime.cut_text("one two three four", 0.5), "one two ... (she interrupted)")
        self.assertEqual(realtime.cut_text("one two", 1.0), "one two")
        self.assertIn("before hearing", realtime.cut_text("one two", 0.1))
        self.assertTrue(realtime.cut_text("\u4f60\u597d\u4e16\u754c", 0.5).startswith("\u4f60\u597d ..."))

    async def test_summary_never_repeats_a_hand_off_after_reconnect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            call = server.Call(FakeCallWS([]), self.svc)
            call.svc.handoff.folder = tmpdir
            call.posted_handoffs.append({"to": "CRM", "words": "follow up"})
            call._write_call_summary()
            call2 = server.Call(FakeCallWS([]), self.svc)
            call2._use_state(call.state)
            call2._write_call_summary()
            self.assertEqual(len(os.listdir(tmpdir)), 1)
            call2.posted_handoffs.append({"to": "MW", "words": "new post"})
            call2._write_call_summary()
            self.assertEqual(len(os.listdir(tmpdir)), 2)

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
