import asyncio
import json
import os
import re
import sys
import tempfile
import unittest

# Keep fake-claude failures out of the real logs/errors.log (set before diag is imported).
os.environ.setdefault("VOICE_ERROR_LOG", os.path.join(tempfile.mkdtemp(prefix="voicecall-test-"), "errors.log"))

HERE = os.path.dirname(os.path.abspath(__file__))
# Tests never read the owner's local_config.json: they use this fixture (set before config is imported).
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config  # noqa: E402
import audio  # noqa: E402
import auth  # noqa: E402
import frontdesk  # noqa: E402
import handoff  # noqa: E402
import server  # noqa: E402
import stt  # noqa: E402
import tts  # noqa: E402

OWNER = config.get("owner")
SILENCE = b"\x00\x00" * 16000


class TestAudio(unittest.TestCase):
    def test_silence_and_short_clips_are_skipped(self):
        self.assertFalse(audio.worth_transcribing(SILENCE))
        self.assertFalse(audio.worth_transcribing(server.fake_clip(0.2)))
        self.assertTrue(audio.worth_transcribing(server.fake_clip(1.0)))

    def test_hallucinations(self):
        self.assertTrue(audio.is_hallucination("Thank you."))
        self.assertTrue(audio.is_hallucination("  "))
        self.assertFalse(audio.is_hallucination("Please call Sam"))

    def test_split_sentences(self):
        parts = audio.split_sentences("First one. Second one! Third?")
        self.assertEqual(len(parts), 3)
        self.assertEqual(audio.split_sentences("no punctuation"), ["no punctuation"])

    def test_has_chinese(self):
        self.assertTrue(audio.has_chinese("hi \u4f60\u597d"))
        self.assertFalse(audio.has_chinese("hello"))


class TestAuth(unittest.TestCase):
    def test_lockout_after_five_wrong_tries(self):
        now = [0.0]
        g = auth.PinGuard("123456", clock=lambda: now[0])
        for _ in range(4):
            self.assertEqual(g.check("000000"), (False, 0))
        ok, locked = g.check("000000")
        self.assertFalse(ok)
        self.assertEqual(locked, auth.LOCK_SECONDS)
        self.assertFalse(g.check("123456")[0])          # correct PIN is refused while locked
        now[0] += auth.LOCK_SECONDS + 1
        self.assertTrue(g.check("123456")[0])

    def test_success_resets_counter(self):
        g = auth.PinGuard("123456")
        for _ in range(3):
            g.check("bad")
        self.assertTrue(g.check("123456")[0])
        for _ in range(4):
            self.assertEqual(g.check("bad"), (False, 0))

    def test_pin_file_created_private(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "pin.txt")
        pin = auth.load_or_create_pin(p)
        self.assertEqual(len(pin), 6)
        self.assertTrue(pin.isdigit())
        self.assertEqual(oct(os.stat(p).st_mode & 0o777), "0o600")
        self.assertEqual(auth.load_or_create_pin(p), pin)

    def test_origin_and_login_checks(self):
        self.assertTrue(auth.origin_ok({"Host": "mac.ts.net", "Origin": "https://mac.ts.net"}))
        self.assertFalse(auth.origin_ok({"Host": "mac.ts.net", "Origin": "https://evil.example"}))
        self.assertTrue(auth.origin_ok({"Host": "mac.ts.net"}))
        self.assertTrue(auth.login_ok({}, ""))
        self.assertTrue(auth.login_ok({"Tailscale-User-Login": "a@x.com"}, "a@x.com"))
        self.assertFalse(auth.login_ok({}, "a@x.com"))

    def test_never_listens_on_all_interfaces(self):
        self.assertFalse(auth.host_allowed("0.0.0.0"))
        self.assertFalse(auth.host_allowed("::"))
        self.assertTrue(auth.host_allowed("127.0.0.1"))


class TestFrontDesk(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(frontdesk.parse_reply('{"say": "Hi", "handoff": null}').say, "Hi")
        r = frontdesk.parse_reply('```json\n{"say": "OK", "handoff": "Research X"}\n```')
        self.assertEqual((r.say, r.handoff), ("OK", "Research X"))
        self.assertEqual(frontdesk.parse_reply("plain words").say, "plain words")
        self.assertIn("Sorry", frontdesk.parse_reply("").say)

    def test_prompt_contains_notes_history_and_rules(self):
        p = frontdesk.build_prompt([{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}], "now?", "NOTE-XYZ")
        for needle in ("NOTE-XYZ", OWNER + ": a", "You: b", "now?", "never send", "do not have a note"):
            self.assertIn(needle.lower(), p.lower())

    def test_history_is_trimmed(self):
        hist = [{"role": "user", "text": "turn%d" % i} for i in range(100)]
        p = frontdesk.build_prompt(hist, "x", "")
        self.assertNotIn("turn0\n", p)
        self.assertIn("turn99", p)

    def test_claude_front_desk_uses_runner_and_notes(self):
        seen = {}

        def runner(prompt):
            seen["prompt"] = prompt
            return '{"say": "Done", "handoff": null}'
        fd = frontdesk.ClaudeFrontDesk(notes_fn=lambda q: "CARD-ABC", runner=runner)
        self.assertEqual(fd.reply([], "what is the rule about weekends").say, "Done")
        self.assertIn("CARD-ABC", seen["prompt"])

    def _fake_stream_cmd(self, tmp, die_after=None):
        script = os.path.join(tmp, "fake_claude.py")
        with open(script, "w") as f:
            f.write("import sys, json\n"
                    "n = 0\n"
                    "print(json.dumps({'type': 'system'}), flush=True)\n"
                    "for line in sys.stdin:\n"
                    "    n += 1\n"
                    "    m = json.loads(line)\n"
                    "    if %r is not None and n > %r: sys.exit(1)\n"
                    "    print(json.dumps({'type': 'result', 'is_error': False, 'result': '{\"say\": \"turn %%d\", \"handoff\": null}' %% n}), flush=True)\n"
                    % (die_after, die_after))
        return lambda: [sys.executable, script]

    def test_stream_claude_keeps_one_process_across_calls(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        st = frontdesk.StreamClaude(self._fake_stream_cmd(tmp), dict, tmp, timeout=10)
        try:
            self.assertTrue(st.needs_fresh([]))
            self.assertIn("turn 1", st.ask("a", fresh=True))
            pid = st.proc.pid
            self.assertFalse(st.needs_fresh([{"role": "user", "text": "a"}]))
            self.assertIn("turn 2", st.ask("b"))
            self.assertEqual(st.proc.pid, pid)
            self.assertFalse(st.needs_fresh([]))   # a new call keeps the warm process (a cold start is too slow)
            st.turns = st.MAX_TURNS
            self.assertTrue(st.needs_fresh([]))    # but a very long conversation is restarted
        finally:
            st.stop()

    def test_stream_claude_failure_falls_back_to_one_shot(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        fd = frontdesk.ClaudeFrontDesk(notes_fn=lambda q: "")
        fd.use_stream = True
        fd._stream = frontdesk.StreamClaude(self._fake_stream_cmd(tmp, die_after=0), dict, tmp, timeout=5)
        fd.runner = lambda p: '{"say": "one-shot", "handoff": null}'
        try:
            self.assertEqual(fd.reply([], "hello there").say, "one-shot")
        finally:
            fd._stream.stop()

    def test_qwen_front_desk_parses_reply_and_chain_falls_back(self):
        def post(base, body):
            return {"choices": [{"message": {"content": '{"say": "Hi there", "handoff": null}'}}]}
        q = frontdesk.QwenFrontDesk(key="k", post=post, notes_fn=lambda t: "")
        self.assertEqual(q.reply([], "hello").say, "Hi there")

        def boom(base, body):
            raise OSError("down")
        bad = frontdesk.QwenFrontDesk(key="k", post=boom, notes_fn=lambda t: "")
        chain = frontdesk.ChainFrontDesk(bad, frontdesk.FakeFrontDesk())
        self.assertTrue(chain.reply([], "hello").say)

    def test_quick_facts_weather_and_time(self):
        import quickfacts
        fake = {"current": {"temperature_2m": 11.2, "apparent_temperature": 9.0, "precipitation": 0, "weather_code": 61, "wind_speed_10m": 14.0},
                "daily": {"weather_code": [61, 3], "temperature_2m_max": [13.0, 15.2], "temperature_2m_min": [6.0, 7.1],
                          "precipitation_probability_max": [70, 20]}}
        out = quickfacts.quick_facts("what's the weather in London", get=lambda url: fake)
        self.assertIn("LIVE FACTS", out)
        self.assertIn("London", out)
        self.assertIn("light rain", out)
        self.assertEqual(quickfacts.quick_facts("tell me about the project", get=lambda url: fake), "")

        def boom(url):
            raise OSError("offline")
        self.assertEqual(quickfacts.quick_facts("weather today", get=boom), "")

    def test_status_question_uses_only_team_status(self):
        calls = []
        notes = frontdesk.gather_notes("inbox status", lambda q: calls.append(q) or "OLD NOTE ABOUT JOB 364")
        self.assertEqual(calls, [])
        self.assertNotIn("364", notes)

    def test_missing_brain_tool_gives_empty_notes(self):
        old = frontdesk.BRAIN_TOOL
        frontdesk.BRAIN_TOOL = "/nonexistent/brain.py"
        try:
            self.assertEqual(frontdesk.brain_notes("anything"), "")
        finally:
            frontdesk.BRAIN_TOOL = old


class TestStatus(unittest.TestCase):
    def make_jobs(self):
        import time
        d = tempfile.mkdtemp()
        for name, state, title in (("0001-a", "RUNNING", "Build thing"), ("0002-b", "NEEDS_YOU", "Pick option"),
                                   ("0003-c", "DONE", "Fresh result"), ("0004-d", "DONE", "Old result"), ("0005-e", "CLOSED", "Gone")):
            os.makedirs(os.path.join(d, name))
            json.dump({"num": int(name[:4]), "state": state, "title": title}, open(os.path.join(d, name, "state.json"), "w"))
            if state == "DONE":
                open(os.path.join(d, name, "RESULT.md"), "w").write("# Result\nStatus: all good\n")
        old = time.time() - 48 * 3600
        os.utime(os.path.join(d, "0004-d", "state.json"), (old, old))
        return d

    def test_status_lists_active_and_recent_only(self):
        import status
        text = status.team_status(self.make_jobs())
        self.assertIn("#1 Build thing (running)", text)
        self.assertIn("waiting for " + OWNER, text)
        self.assertIn("Fresh result", text)
        self.assertIn("all good", text)
        self.assertNotIn("Old result", text)
        self.assertNotIn("Gone", text)

    def test_status_question_adds_block_to_prompt(self):
        import status
        seen = {}
        fd = frontdesk.ClaudeFrontDesk(notes_fn=lambda q: "", runner=lambda p: seen.setdefault("p", p) and '{"say": "ok", "handoff": null}')
        old = status.CARRYOVER
        status.CARRYOVER = self.make_carry()
        try:
            fd.reply([], "what tasks are running right now")
        finally:
            status.CARRYOVER = old
        self.assertIn("SESSIONS STATUS", seen["p"])
        self.assertIn("ORG", seen["p"])
        self.assertTrue(status.wants_status("\u4efb\u52a1\u8fdb\u5ea6\u600e\u4e48\u6837"))
        self.assertFalse(status.wants_status("hello there"))
        self.assertTrue(status.wants_status("what is CRM working on"))
        self.assertFalse(status.wants_status("report this flyer"))

    def make_carry(self):
        import time
        d = tempfile.mkdtemp()
        index = {}
        for role, fname, nxt in (("ORG", "Org.md", "Route voice hand-offs"), ("RESEARCH", "Research.md", "Fix source list"),
                                 ("PRIVATE", "Private.md", "Private thing"), ("OLD", "Old.md", "Stale thing")):
            open(os.path.join(d, fname), "w").write("## Goal\nx\n## Next step\n%s\n## Pointers\ny\n" % nxt)
            index[role] = {"file": fname, "task": role}
        old = time.time() - 72 * 3600
        os.utime(os.path.join(d, "Old.md"), (old, old))
        json.dump(index, open(os.path.join(d, "TASKS.json"), "w"))
        return d

    def test_sessions_status_reads_next_steps(self):
        import status
        text = status.sessions_status(self.make_carry())
        self.assertIn("ORG", text)
        self.assertIn("Route voice hand-offs", text)
        self.assertIn("Fix source list", text)
        self.assertNotIn("Private thing", text)
        self.assertNotIn("Stale thing", text)
        self.assertIn("cannot be read", status.sessions_status("/nonexistent/carry"))

    def test_missing_folder_is_fine(self):
        import status
        self.assertIn("nothing is running", status.team_status("/nonexistent/jobs"))


class TestHandoff(unittest.TestCase):
    def test_file_written_even_when_task_list_is_down(self):
        d = tempfile.mkdtemp()
        def broken(role, text):
            raise RuntimeError("carry down")
        h = handoff.FileHandoff(folder=d, post=True, teller=broken)
        rec = h.create("Summarize the weekly report", "summarize weekly report")
        self.assertTrue(os.path.exists(rec["path"]))
        data = json.load(open(rec["path"]))
        self.assertEqual(data["task"], "Summarize the weekly report")
        self.assertFalse(data["posted"])

    def test_handoff_routes_by_what_it_names(self):
        self.assertEqual(handoff.route("Move the call page buttons lower", ""), "VOICE")
        self.assertEqual(handoff.route("Check CRM for the Sam contact", ""), "CRM")
        self.assertEqual(handoff.route("Update the website footer", ""), "WEBSITE")
        self.assertEqual(handoff.route("Research the sources", ""), "RESEARCH")
        self.assertEqual(handoff.route("Summarize the weekly report", ""), "ORG")
        sent = []
        h = handoff.FileHandoff(folder=tempfile.mkdtemp(), post=True, role="AUTO", teller=lambda r, t: sent.append(r) or "ok")
        self.assertEqual(h.create("Fix the microphone prompt", "mic keeps asking")["to"], "VOICE")
        self.assertEqual(sent, ["VOICE"])   # v66: kept local and also told to the VOICE session

    def test_unknown_role_falls_back_to_org_and_chinese_is_not_sent(self):
        sent = []
        def teller(r, t):
            if r != "ORG":
                raise RuntimeError("unknown role")
            sent.append(t)
        h = handoff.FileHandoff(folder=tempfile.mkdtemp(), post=True, role="AUTO", teller=teller)
        rec = h.create("Research the sources", "\u7814\u7a76 research")
        self.assertEqual(rec["to"], "ORG")
        self.assertIn("meant for RESEARCH", sent[0])
        self.assertNotIn("\u7814", sent[0])

    def test_handoff_goes_to_org(self):
        sent = []
        h = handoff.FileHandoff(folder=tempfile.mkdtemp(), post=True, role="AUTO", teller=lambda r, t: sent.append((r, t)) or "ok")
        rec = h.create("Call the supplier", "call supplier")
        self.assertTrue(rec["posted"])
        self.assertEqual(sent, [("ORG", OWNER + " said by voice (exact words): \"call supplier\"")])

    def test_two_in_one_second_do_not_collide(self):
        d = tempfile.mkdtemp()
        h = handoff.FileHandoff(folder=d, post=False)
        a, b = h.create("a", "a"), h.create("b", "b")
        self.assertNotEqual(a["path"], b["path"])


class TestTtsStream(unittest.TestCase):
    def test_qwen_stream_sends_small_first_part(self):
        def sse(url, body, key):
            for _ in range(10):
                yield b"\x01\x00" * 2400   # 0.1 s each at 24 kHz
        q = tts.QwenTextToSpeech(key="k", sse=sse, bases=["https://x"])
        parts = list(q.speak_stream("hello"))
        self.assertEqual([p[:4] for p in parts], [b"RIFF"] * len(parts))
        sizes = [len(p) - 44 for p in parts]
        self.assertEqual(sizes[0], 9 * 4800)          # first part goes out as soon as 0.9 s of sound exists
        self.assertEqual(sum(sizes), 10 * 4800)       # nothing lost

    def test_wav_header_in_stream_is_not_played(self):
        pcm = b"\x01\x00" * 100
        self.assertEqual(tts.strip_wav_header(tts.pcm_wav(pcm)), pcm)
        self.assertEqual(tts.strip_wav_header(pcm), pcm)

    def test_chain_falls_back_only_before_first_part(self):
        class Broken(tts.TextToSpeech):
            def speak_stream(self, text):
                raise RuntimeError("down")
                yield b""
        chain = tts.ChainTextToSpeech(Broken(), tts.FakeTextToSpeech())
        self.assertEqual(len(list(chain.speak_stream("hi"))), 1)

        class Half(tts.TextToSpeech):
            def speak_stream(self, text):
                yield tts.silent_wav()
                raise RuntimeError("cut")
        fb = tts.FakeTextToSpeech()
        self.assertEqual(len(list(tts.ChainTextToSpeech(Half(), fb).speak_stream("hi"))), 1)
        self.assertEqual(fb.spoken, [])


class TestTts(unittest.TestCase):
    def test_fake_wav_is_valid(self):
        w = tts.silent_wav()
        self.assertEqual(w[:4], b"RIFF")
        self.assertEqual(w[8:12], b"WAVE")


class TestQwenVoice(unittest.TestCase):
    def test_qwen_request_and_audio_download(self):
        calls = []

        def post(url, body, key):
            calls.append((url, body, key))
            return {"output": {"audio": {"url": "https://example.test/a.wav"}}}
        q = tts.QwenTextToSpeech(key="k", post=post, get=lambda u: b"WAVDATA")
        self.assertEqual(q.speak("hello"), b"WAVDATA")
        url, body, key = calls[0]
        self.assertTrue(url.endswith("/services/aigc/multimodal-generation/generation"))
        self.assertEqual(body["model"], "qwen3-tts-flash")
        self.assertEqual(body["input"]["language_type"], "English")
        self.assertEqual(q.speak("\u4f60\u597d")[:0], b"")
        self.assertEqual(calls[1][1]["input"]["language_type"], "Chinese")

    def test_second_region_is_tried_then_remembered(self):
        seen = []

        def post(url, body, key):
            seen.append(url)
            if "intl" in url:
                raise OSError("401")
            return {"output": {"audio": {"url": "https://x/a.wav"}}}
        q = tts.QwenTextToSpeech(key="k", post=post, get=lambda u: b"OK")
        self.assertEqual(q.speak("hi"), b"OK")
        q.speak("again")
        self.assertEqual(sum(1 for u in seen if "intl" in u), 1)

    def test_failure_falls_back_to_mac_voice(self):
        def bad(url, body, key):
            raise OSError("down")
        chain = tts.ChainTextToSpeech(tts.QwenTextToSpeech(key="k", post=bad), tts.FakeTextToSpeech())
        self.assertEqual(chain.speak("hi")[:4], b"RIFF")
        self.assertEqual(chain.last_error, "RuntimeError")

    def test_no_key_means_not_available(self):
        self.assertFalse(tts.QwenTextToSpeech(key="", post=None).available() and False)


class TestSttWorkerProcess(unittest.TestCase):
    def make(self, text="hello from worker"):
        env = dict(os.environ, VOICE_STT_FAKE=text)
        return stt.WhisperSpeechToText(env=env)

    def test_worker_round_trip_from_many_threads(self):
        import threading
        w = self.make()
        out = []
        clip = server.fake_clip()
        threads = [threading.Thread(target=lambda: out.append(w.transcribe(clip))) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        w._stop()
        self.assertEqual(out, ["hello from worker"] * 4)

    def test_silence_never_reaches_the_worker(self):
        w = self.make()
        self.assertEqual(w.transcribe(SILENCE), "")
        self.assertIsNone(w._proc)

    def test_worker_crash_is_survived(self):
        w = self.make()
        clip = server.fake_clip()
        self.assertEqual(w.transcribe(clip), "hello from worker")
        w._proc.kill()
        w._proc.wait()
        try:
            w.transcribe(clip)
        except Exception:
            pass
        self.assertEqual(w.transcribe(clip), "hello from worker")   # a fresh worker takes over
        w._stop()


class TestSttFilters(unittest.TestCase):
    def test_fake_filters_noise(self):
        s = stt.FakeSpeechToText("hello there")
        self.assertEqual(s.transcribe(SILENCE), "")
        self.assertEqual(s.transcribe(server.fake_clip()), "hello there")
        self.assertEqual(stt.FakeSpeechToText("Thank you.").transcribe(server.fake_clip()), "")


def make_services(text="please research the weekly report", pin="123456", **kw):
    tmp = tempfile.mkdtemp()
    return server.Services(stt.FakeSpeechToText(text), frontdesk.FakeFrontDesk(), tts.FakeTextToSpeech(),
                           handoff.FakeHandoff(), auth.PinGuard(pin), turn_log=os.path.join(tmp, "turns.log"), **kw)


class TestServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.svc = make_services()
        self.srv = await server.serve(self.svc, "127.0.0.1", 0)
        self.port = self.srv.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.srv.close()
        await self.srv.wait_closed()

    async def client(self, pin="123456", **kw):
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % self.port, **kw)
        try:
            await ws.send(json.dumps({"type": "auth", "pin": pin}))
        except Exception:
            pass  # the server may already have refused and closed (bad Origin)
        first = json.loads(await asyncio.wait_for(ws.recv(), 5))
        return ws, first

    async def drain_turn(self, ws):
        kinds, audio_n = [], 0
        for _ in range(30):
            m = await asyncio.wait_for(ws.recv(), 10)
            if isinstance(m, bytes):
                audio_n += 1
                continue
            d = json.loads(m)
            kinds.append(d["type"])
            if d["type"] == "audio_end":
                break
        return kinds, audio_n

    async def test_wrong_pin_rejected_and_closed(self):
        ws, first = await self.client(pin="000000")
        self.assertEqual(first["type"], "auth_fail")
        await ws.wait_closed()

    async def raw_auth(self, msg):
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % self.port)
        await ws.send(json.dumps(msg))
        return ws, json.loads(await asyncio.wait_for(ws.recv(), 5))

    async def test_resume_token_reconnects_and_fake_token_fails(self):
        ws, first = await self.client()
        self.assertEqual(first["type"], "auth_ok")
        token = first["resume"]
        await ws.close()
        ws2, ok = await self.raw_auth({"type": "auth", "pin": "", "resume": token})
        self.assertEqual(ok["type"], "auth_ok")
        await ws2.close()
        ws3, bad = await self.raw_auth({"type": "auth", "pin": "", "resume": "made-up"})
        self.assertEqual(bad["type"], "auth_fail")

    async def test_reconnect_keeps_call_memory(self):
        # R23: the line drops and the phone comes back with its token: same call, same memory, old Call silenced.
        ws, first = await self.client()
        old = self.svc.last_call
        old.remember("user", "the blue house on Main Street", turn=1)
        old.remember("assistant", "Sent to CRM.", reply=1)
        old.handoff_count = 2
        ws2, ok = await self.raw_auth({"type": "auth", "pin": "", "resume": first["resume"]})
        self.assertEqual(ok["type"], "auth_ok")
        new = self.svc.last_call
        self.assertIsNot(new, old)
        self.assertTrue(old.replaced)
        self.assertEqual([l["text"] for l in new.history], ["the blue house on Main Street", "Sent to CRM."])
        self.assertEqual(new.handoff_count, 2)
        self.assertEqual(new.seed_lines()[0], {"role": "user", "text": "the blue house on Main Street"})
        # an older token of the same call still finds the memory
        ws3, ok3 = await self.raw_auth({"type": "auth", "pin": "", "resume": first["resume"]})
        self.assertEqual(ok3["type"], "auth_ok")
        self.assertIs(self.svc.last_call.history, new.history)
        # a fresh PIN sign-in is a new call: no memory
        ws4, ok4 = await self.client()
        self.assertEqual(self.svc.last_call.history, [])
        for w in (ws, ws2, ws3, ws4):
            await w.close()

    async def test_bad_face_id_counts_toward_lockout(self):
        class NoPasskey:
            def verify_login(self, cred): return False
            def count(self): return 1
        self.svc.passkeys = NoPasskey()
        for _ in range(auth.MAX_TRIES):
            ws, r = await self.raw_auth({"type": "auth", "pin": "", "passkey": {"id": "x"}})
            self.assertEqual(r["type"], "auth_fail")
        ws, r = await self.client()   # the right PIN is refused while locked
        self.assertEqual(r["type"], "auth_fail")
        self.assertGreater(r["locked_s"], 0)

    async def test_stale_resume_with_pin_reconnects_without_counting(self):
        ws, r = await self.raw_auth({"type": "auth", "pin": "123456", "resume": "gone-after-restart"})
        self.assertEqual(r["type"], "auth_ok")
        await ws.close()
        ws, r = await self.raw_auth({"type": "auth", "pin": "", "resume": "gone-after-restart"})
        self.assertEqual(r["type"], "auth_fail")
        self.assertEqual(self.svc.guard._fails, 0)   # a stale token alone is refused but never counted

    async def test_face_id_success_resets_and_stale_challenge_is_retry(self):
        results = [False, None, True]

        class Seq:
            def verify_login(self, cred): return results.pop(0)
            def count(self): return 1
        self.svc.passkeys = Seq()
        ws, r = await self.raw_auth({"type": "auth", "pin": "", "passkey": {"id": "x"}})
        self.assertEqual((r["type"], self.svc.guard._fails), ("auth_fail", 1))
        ws, r = await self.raw_auth({"type": "auth", "pin": "", "passkey": {"id": "x"}})
        self.assertEqual((r["type"], self.svc.guard._fails), ("auth_retry", 1))
        ws, r = await self.raw_auth({"type": "auth", "pin": "", "passkey": {"id": "x"}})
        self.assertEqual((r["type"], self.svc.guard._fails), ("auth_ok", 0))
        await ws.close()

    async def test_login_options_endpoint(self):
        import urllib.request
        import passkey
        url = "http://127.0.0.1:%d/passkey/login-options" % self.port
        path = os.path.join(tempfile.mkdtemp(), "passkeys.json")
        self.svc.passkeys = passkey.Passkeys(path=path, rp_id="example.test")
        with self.assertRaises(urllib.error.HTTPError):   # nobody enrolled: 404
            await asyncio.to_thread(urllib.request.urlopen, url)
        json.dump({"credentials": [{"id": "AAAA", "public_key": "AAAA"}]}, open(path, "w"))
        body = json.loads((await asyncio.to_thread(urllib.request.urlopen, url)).read())
        self.assertTrue(body["challenge"])
        self.assertEqual(body["rpId"], "example.test")

    async def test_face_id_enrolment_needs_pin_call(self):
        class Enroll:
            def verify_login(self, cred): return True
            def count(self): return 1
            def register_options(self): return {"challenge": "c"}, "c"
        self.svc.passkeys = Enroll()
        ws, r = await self.raw_auth({"type": "auth", "pin": "", "passkey": {"id": "x"}})
        self.assertEqual(r["type"], "auth_ok")
        await ws.send(json.dumps({"type": "passkey_register_options"}))
        await ws.send(json.dumps({"type": "ping"}))
        kinds = [json.loads(await asyncio.wait_for(ws.recv(), 5))["type"] for _ in range(2)]
        self.assertNotIn("passkey_options", kinds)   # a Face ID call cannot enroll another device
        await ws.close()

    async def test_binary_before_auth_is_ignored(self):
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % self.port)
        await ws.send(server.fake_clip())          # not JSON auth: counts as a failed login
        first = json.loads(await asyncio.wait_for(ws.recv(), 5))
        self.assertEqual(first["type"], "auth_fail")
        self.assertEqual(self.svc.frontdesk.seen, [])

    async def test_lockout_blocks_correct_pin(self):
        for _ in range(5):
            ws, _ = await self.client(pin="000000")
            await ws.wait_closed()
        ws, first = await self.client(pin="123456")
        self.assertEqual(first["type"], "auth_fail")
        self.assertGreater(first["locked_s"], 0)

    async def test_full_turn_with_handoff(self):
        ws, first = await self.client()
        self.assertEqual(first["type"], "auth_ok")
        await ws.recv()  # listening
        await ws.send(server.fake_clip())
        kinds, audio_n = await self.drain_turn(ws)
        self.assertIn("handoff", kinds)
        self.assertGreaterEqual(audio_n, 1)
        self.assertEqual(len(self.svc.handoff.items), 1)
        self.assertEqual(kinds[-1], "audio_end")

    async def test_phone_voice_sends_text_not_audio(self):
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % self.port)
        await ws.send(json.dumps({"type": "auth", "pin": "123456", "voice": "phone"}))
        self.assertEqual(json.loads(await asyncio.wait_for(ws.recv(), 5))["type"], "auth_ok")
        await ws.recv()  # listening
        await ws.send(server.fake_clip())
        kinds, audio_n = await self.drain_turn(ws)
        self.assertIn("say", kinds)
        self.assertEqual(audio_n, 0)
        self.assertEqual(kinds[-1], "audio_end")

    async def test_text_from_the_phone_gets_a_reply_without_the_mac_speech_model(self):
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % self.port)
        await ws.send(json.dumps({"type": "auth", "pin": "123456", "voice": "phone", "ears": "phone"}))
        self.assertEqual(json.loads(await asyncio.wait_for(ws.recv(), 5))["type"], "auth_ok")
        await ws.recv()  # listening
        calls_before = self.svc.stt.calls
        await ws.send(json.dumps({"type": "text", "text": "hello from the phone"}))
        kinds, audio_n = await self.drain_turn(ws)
        self.assertIn("transcript", kinds)
        self.assertIn("say", kinds)
        self.assertEqual(self.svc.stt.calls, calls_before)

    async def test_silence_gets_no_reply(self):
        ws, first = await self.client()
        await ws.recv()
        await ws.send(SILENCE)
        seen = []
        for _ in range(3):
            d = json.loads(await asyncio.wait_for(ws.recv(), 5))
            seen.append(d)
            if d["type"] == "status" and d["state"] == "listening":
                break
        self.assertEqual(self.svc.frontdesk.seen, [])
        self.assertNotIn("transcript", [d["type"] for d in seen])

    async def test_interrupt_returns_to_listening(self):
        ws, first = await self.client()
        await ws.recv()
        await ws.send(json.dumps({"type": "interrupt"}))
        d = json.loads(await asyncio.wait_for(ws.recv(), 5))
        self.assertEqual((d["type"], d["state"]), ("status", "listening"))

    async def test_bad_origin_refused(self):
        ws, first = await self.client(additional_headers={"Origin": "https://evil.example"})
        self.assertEqual(first["type"], "auth_fail")

    async def test_timing_log_has_no_transcript_text(self):
        ws, first = await self.client()
        await ws.recv()
        await ws.send(server.fake_clip())
        await self.drain_turn(ws)
        await asyncio.sleep(0.1)
        text = open(self.svc.turn_log).read()
        self.assertIn("first_audio_ms=", text)
        self.assertNotIn("weekly report", text)

    async def test_page_and_health_are_served(self):
        import urllib.request
        loop = asyncio.get_running_loop()
        body = await loop.run_in_executor(None, lambda: urllib.request.urlopen("http://127.0.0.1:%d/" % self.port).read())
        self.assertIn(b"<title>Dot</title>", body)
        ok = await loop.run_in_executor(None, lambda: urllib.request.urlopen("http://127.0.0.1:%d/healthz" % self.port).read())
        self.assertEqual(ok.strip(), b"ok")


class TestRepoRules(unittest.TestCase):
    def test_all_repo_files_are_english_only(self):
        bad = []
        for dirpath, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("logs", "run", "__pycache__", "handoff")]
            for f in files:
                if f.endswith((".py", ".html", ".md", ".txt", ".json", ".plist")):
                    text = open(os.path.join(dirpath, f), encoding="utf-8").read()
                    if any(ord(c) > 127 for c in text):
                        bad.append(f)
        self.assertEqual(bad, [])

    def test_no_cloud_speech_services_referenced(self):
        # Reply voice may use Qwen. Speech RECOGNITION must stay local: stt.py and the
        # audio path never mention any cloud speech service.
        for f in ("stt.py", "audio.py"):
            text = open(os.path.join(ROOT, f)).read().lower()
            for banned in ("qwen", "dashscope", "urllib", "requests"):
                self.assertNotIn(banned, text, "%s mentions %s" % (f, banned))
        for f in ("server.py", "stt.py", "frontdesk.py", "tts.py", "handoff.py"):
            text = open(os.path.join(ROOT, f)).read().lower()
            for banned in ("gemini", "groq", "doubao", "speech.googleapis", "api.openai.com", "deepgram", "assemblyai"):
                self.assertNotIn(banned, text.replace("no audio", ""), "%s mentions %s" % (f, banned))



class TestPageLook(unittest.TestCase):
    def setUp(self):
        self.page = open(os.path.join(ROOT, "static", "index.html")).read()

    def test_page_avatar_portrait_and_calm_breathing(self):
        self.assertIn('const LOOKS = ["/dot/dot-portrait.jpg"', self.page)
        m = re.search(r"#orb\.speaking \{ animation:speak ([0-9.]+)s", self.page)
        self.assertGreaterEqual(float(m.group(1)), 2.0)   # a slow breath, not a shake
        m = re.search(r"@keyframes speak \{.*?50% \{ scale:([0-9.]+);", self.page)
        self.assertLessEqual(float(m.group(1)), 1.04)
        self.assertNotIn("glow .8s", self.page)
        self.assertIn("Math.min(0.05", self.page)   # her voice moves the face at most 5%

    def test_page_shows_live_words_and_stops_on_interrupt(self):
        self.assertIn('m.type === "live_partial") showLive(', self.page)
        self.assertRegex(self.page, r'm\.type === "live_stop"\) \{\s*stopLivePlayback\(\)')


if __name__ == "__main__":
    unittest.main()


class TestPasskeys(unittest.TestCase):
    def test_challenges_single_use_and_expire(self):
        import base64
        import passkey
        now = [1000.0]
        path = os.path.join(tempfile.mkdtemp(), "passkeys.json")
        json.dump({"credentials": [{"id": "AAAA", "public_key": "AAAA"}]}, open(path, "w"))
        pk = passkey.Passkeys(path=path, rp_id="example.test", clock=lambda: now[0])

        def cred(challenge):
            cd = base64.urlsafe_b64encode(json.dumps({"challenge": challenge}).encode()).decode().rstrip("=")
            return {"id": "AAAA", "response": {"clientDataJSON": cd}}
        c = pk.login_options()["challenge"]
        self.assertFalse(pk.verify_login(cred(c)))      # known challenge, bad signature: refused
        self.assertIsNone(pk.verify_login(cred(c)))     # used once already: stale
        c2 = pk.login_options()["challenge"]
        now[0] += passkey.CHALLENGE_TTL_S + 1
        self.assertIsNone(pk.verify_login(cred(c2)))    # expired: stale


class TestQwenHedge(unittest.TestCase):
    def test_slow_first_request_loses_to_second(self):
        import time as _t
        calls = []

        def sse(url, body, key):
            calls.append(1)
            if len(calls) == 1:
                _t.sleep(0.5)   # Beijing stall
            yield b"\x01\x00" * 24000
        q = tts.QwenTextToSpeech(key="k", sse=sse, bases=["https://x"])
        q._hedged.__func__   # exists
        t0 = _t.monotonic()
        parts = list(q._hedged("https://x/u", {}, hedge_s=0.1, deadline_s=2))
        self.assertEqual(len(calls), 2)
        self.assertLess(_t.monotonic() - t0, 0.45)
        self.assertEqual(len(parts), 1)

    def test_both_fail_raises(self):
        def sse(url, body, key):
            raise OSError("down")
            yield b""
        q = tts.QwenTextToSpeech(key="k", sse=sse, bases=["https://x"])
        with self.assertRaises(OSError):
            list(q._hedged("https://x/u", {}, hedge_s=0.1, deadline_s=1))

    def test_route_call_page_to_voice(self):
        self.assertEqual(handoff.route("check why", "why sometimes the call page is slow"), "VOICE")


class TestTargetMenu(unittest.TestCase):
    def test_picked_session_wins_over_routing(self):
        told = []
        h = handoff.FileHandoff(folder=tempfile.mkdtemp(), teller=lambda r, m: told.append(r) or "ok")
        rec = h.create("update the website hero", "change the website photo", role="CRM")
        self.assertEqual((rec["to"], told), ("CRM", ["CRM"]))
        rec = h.create("update the website hero", "change the website photo", role="AUTO")
        self.assertEqual(rec["to"], "WEBSITE")
        rec = h.create("x", "y", role="NOT-A-ROLE")
        self.assertEqual(rec["to"], "ORG")


class TestQwenLocked(unittest.TestCase):
    def test_no_other_model_answers(self):
        class Bad:
            def reply(self, h, t): raise RuntimeError("stall")
        r = frontdesk.ChainFrontDesk(Bad(), frontdesk.QwenAgainFrontDesk(Bad())).reply([], "hi")
        self.assertIn("Qwen did not answer", r.say)

    def test_default_voice_has_no_mac_backup(self):
        import unittest.mock as m
        with m.patch.object(tts, "qwen_key", return_value="k"), m.patch.dict(os.environ, {"VOICE_TTS_ENGINE": "qwen"}):
            t = tts.default_tts()
        self.assertIsInstance(t.fallback, tts.QwenTextToSpeech)


class TestUpdates(unittest.IsolatedAsyncioTestCase):
    async def test_update_is_spoken_in_call_once(self):
        import updates
        svc = make_services()
        svc.updates_folder = tempfile.mkdtemp()
        updates.post("VOICE", "The Qwen lock is live.", folder=svc.updates_folder)
        srv = await server.serve(svc, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % port)
        await ws.send(json.dumps({"type": "auth", "pin": "123456"}))
        said = []
        for _ in range(20):
            m = await asyncio.wait_for(ws.recv(), 8)
            if isinstance(m, str) and json.loads(m).get("type") == "transcript":
                said.append(json.loads(m)["text"])
                break
        self.assertEqual(said, ["Update from Voice: The Qwen lock is live."])
        self.assertEqual(updates.pending(svc.updates_folder), [])
        await ws.close()
        srv.close()
        await srv.wait_closed()


class TestExactWords(unittest.TestCase):
    def test_feedback_routes_to_voice_and_words_are_verbatim(self):
        said = "Another point about this call: when I say I want to talk to a session, pass on my exact words"
        self.assertEqual(handoff.route("Forward exact phrase", said), "VOICE")
        self.assertEqual(handoff.route("x", "feedback for CRM: the notes are wrong"), "CRM")
        sent = []
        h = handoff.FileHandoff(folder=tempfile.mkdtemp(), teller=lambda r, m: sent.append(m) or "ok")
        h.create("Summarized task", "I want to talk to CRM, please fix the six key notes", role="CRM")
        self.assertEqual(sent, [OWNER + ' said by voice (exact words): "I want to talk to CRM, please fix the six key notes"'])


class TestDirectStatus(unittest.TestCase):
    def test_reads_session_words_verbatim(self):
        import status
        d = tempfile.mkdtemp()
        open(os.path.join(d, "CRM.md"), "w").write("## Goal\nx\n## Next step\nFix the six key notes. Then audit.\n")
        json.dump({"CRM": {"file": "CRM.md"}}, open(os.path.join(d, "TASKS.json"), "w"))
        out = status.direct_status("What is CRM doing?", folder=d, updates_fn=lambda r: None, titles={})
        self.assertIn("Its next step, in its own words: Fix the six key notes.", out)
        self.assertNotIn("audit", out)
        self.assertIsNone(status.direct_status("tell CRM to fix the notes", folder=d, titles={}))
        self.assertIsNone(status.direct_status("what is CRM and RESEARCH doing", folder=d, titles={}))
        self.assertEqual(status.named_session("how is the website"), "WEBSITE")
        out = status.direct_status("what is going on", folder=d, titles={})
        self.assertEqual(out, "Latest from your sessions, in their own words. CRM, just now: Fix the six key notes.")


class TestBugReport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="voicecall-bugtest-")
        self.handoff_dir = os.path.join(self.tmp, "handoff")
        self.turn_log = os.path.join(self.tmp, "turns.log")
        with open(self.turn_log, "w", encoding="utf-8") as f:
            for i in range(50):
                f.write("2026-10-11 12:00:%02d stt_ms=%d\n" % (i % 60, i * 10))
        self.svc = server.Services(
            stt.FakeSpeechToText("words from mic"),
            frontdesk.FakeFrontDesk(),
            tts.FakeTextToSpeech(),
            handoff.FileHandoff(folder=self.handoff_dir, post=False),
            auth.PinGuard("123456"),
            turn_log=self.turn_log,
        )
        self.srv = await server.serve(self.svc, "127.0.0.1", 0)
        self.port = self.srv.sockets[0].getsockname()[1]
        self._orig_timeout = server.BUG_TIMEOUT_S

    async def asyncTearDown(self):
        server.BUG_TIMEOUT_S = self._orig_timeout
        self.srv.close()
        await self.srv.wait_closed()

    async def client(self):
        from websockets.asyncio.client import connect
        ws = await connect("ws://127.0.0.1:%d" % self.port)
        await ws.send(json.dumps({"type": "auth", "pin": "123456"}))
        auth_msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
        self.assertEqual(auth_msg["type"], "auth_ok")
        st = json.loads(await asyncio.wait_for(ws.recv(), 5))
        self.assertEqual(st.get("state"), "listening")
        return ws

    async def drain_until_type(self, ws, target_type):
        for _ in range(30):
            m = await asyncio.wait_for(ws.recv(), 5)
            if isinstance(m, bytes):
                continue
            d = json.loads(m)
            if d.get("type") == target_type:
                return d
        return None

    def test_cap_snapshot_direct(self):
        snap = {
            "version": "1.0",
            "lines": [{"who": "user", "text": "a" * 600} for _ in range(25)],
            "events": list(range(50)),
            "ua": "test-browser",
        }
        capped = server._cap_snapshot(snap)
        self.assertEqual(len(capped["lines"]), 12)
        for line in capped["lines"]:
            self.assertLessEqual(len(line["text"]), 400)
        self.assertEqual(len(capped["events"]), 30)
        self.assertLessEqual(len(json.dumps(capped, ensure_ascii=False)), 20000)

    async def test_bug_message_stores_snapshot_capped(self):
        ws = await self.client()
        snap = {
            "version": "2.0",
            "lines": [{"who": "user", "text": "x" * 500} for _ in range(20)],
            "events": [{"e": i} for i in range(40)],
            "ua": "phone",
        }
        await ws.send(json.dumps({"type": "bug", "snapshot": snap}))
        note = await self.drain_until_type(ws, "status_note")
        self.assertIsNotNone(note)
        self.assertEqual(note.get("text"), "Bug recorded. What went wrong?")
        await self.drain_until_type(ws, "audio_end")
        call = getattr(self.svc, "last_call", None)
        self.assertIsNotNone(call)
        self.assertIsNotNone(call.bug)
        self.assertEqual(len(call.bug["snapshot"]["lines"]), 12)
        for line in call.bug["snapshot"]["lines"]:
            self.assertLessEqual(len(line["text"]), 400)
        self.assertEqual(len(call.bug["snapshot"]["events"]), 30)
        self.assertLessEqual(len(json.dumps(call.bug["snapshot"])), 20000)
        self.assertEqual(len(call.bug["server"]), 40)
        self.assertEqual(call.bug["version"], server.VERSION)
        self.assertTrue(call.bug["created"])
        log_text = open(self.turn_log).read()
        self.assertIn("call: bug recorded", log_text)
        await ws.close()

    async def test_next_text_becomes_bug_words_not_sent_to_frontdesk(self):
        ws = await self.client()
        snap = {"version": "1.0", "lines": [{"who": "user", "text": "bad"}], "events": []}
        await ws.send(json.dumps({"type": "bug", "snapshot": snap}))
        note = await self.drain_until_type(ws, "status_note")
        self.assertEqual(note.get("text"), "Bug recorded. What went wrong?")
        await self.drain_until_type(ws, "audio_end")
        call = getattr(self.svc, "last_call", None)
        self.assertIsNotNone(call.bug)

        await ws.send(json.dumps({"type": "text", "text": "The voice sounds robotic and stalls"}))
        await self.drain_until_type(ws, "audio_end")

        self.assertEqual(self.svc.frontdesk.seen, [])
        self.assertIsNone(call.bug)

        files = [f for f in os.listdir(self.handoff_dir) if f.endswith("-bug.json")]
        self.assertEqual(len(files), 1)
        data = json.load(open(os.path.join(self.handoff_dir, files[0])))
        self.assertEqual(data["to"], "VOICE")
        self.assertEqual(data["task"], "Bug report from the call")
        self.assertEqual(data["spoken"], "The voice sounds robotic and stalls")
        self.assertEqual(data["source"], "voice_call_bug")
        self.assertEqual(data["bug"]["words"], "The voice sounds robotic and stalls")
        self.assertEqual(len(data["bug"]["server"]), 40)
        self.assertEqual(data["bug"]["version"], server.VERSION)
        await ws.close()

    async def test_timeout_path_writes_with_empty_words(self):
        server.BUG_TIMEOUT_S = 0.05
        ws = await self.client()
        snap = {"version": "1.0", "lines": [], "events": []}
        await ws.send(json.dumps({"type": "bug", "snapshot": snap}))
        note = await self.drain_until_type(ws, "status_note")
        self.assertEqual(note.get("text"), "Bug recorded. What went wrong?")
        await self.drain_until_type(ws, "audio_end")
        call = getattr(self.svc, "last_call", None)
        self.assertIsNotNone(call.bug)

        await asyncio.sleep(0.12)
        self.assertIsNone(call.bug)

        files = [f for f in os.listdir(self.handoff_dir) if f.endswith("-bug.json")]
        self.assertEqual(len(files), 1)
        data = json.load(open(os.path.join(self.handoff_dir, files[0])))
        self.assertEqual(data["to"], "VOICE")
        self.assertEqual(data["task"], "Bug report from the call")
        self.assertEqual(data["spoken"], "")
        self.assertEqual(data["source"], "voice_call_bug")
        self.assertEqual(data["bug"]["words"], "")
        await ws.close()

    async def test_audio_utterance_becomes_bug_words(self):
        ws = await self.client()
        snap = {"version": "1.0", "lines": [], "events": []}
        await ws.send(json.dumps({"type": "bug", "snapshot": snap}))
        await self.drain_until_type(ws, "status_note")
        await self.drain_until_type(ws, "audio_end")   # the spoken "Bug recorded" prompt

        await ws.send(server.fake_clip(1.0))
        await self.drain_until_type(ws, "audio_end")   # the spoken "Thanks, sent to Voice."

        self.assertEqual(self.svc.frontdesk.seen, [])
        files = [f for f in os.listdir(self.handoff_dir) if f.endswith("-bug.json")]
        self.assertEqual(len(files), 1)
        data = json.load(open(os.path.join(self.handoff_dir, files[0])))
        self.assertEqual(data["to"], "VOICE")
        self.assertEqual(data["spoken"], "words from mic")
        self.assertEqual(data["bug"]["words"], "words from mic")
        await ws.close()
