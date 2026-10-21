"""v60: R47 call-back on hand-off reply (P19). Fake HTTP, fake Telegram, temp dirs, neutral names, no network, no real call."""
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import callback  # noqa: E402
import twilio_bridge as tb  # noqa: E402

TO = "+15550001111"
FROM = "+15550002222"
SECRET = "s" * 30
HOST = "dot.example.net"
NOON_UTC = 1792584000.0   # 2026-10-21 12:00:00 UTC
ALIASES = {"CRM": "VAULT", "VAULT": "CRM"}


def cfg_for(**over):
    base = {"callback_on_reply": True, "callback_quiet": "21:00-08:00", "callback_timezone": "UTC",
            "callback_min_gap_s": 1200, "callback_max_per_day": 6, "callback_window_s": 3600,
            "twilio_allowed_callers": [TO, "+15550009999"], "twilio_number": FROM, "twilio_public_host": HOST,
            "twilio_path_secret": SECRET, "call_recap_telegram_chat_id": "chat"}
    base.update(over)
    return base.__getitem__


class Fakes:
    def __init__(self):
        self.calls, self.texts = [], []
        self.ok = True

    def place(self, to, from_, url):
        self.calls.append((to, from_, url))
        return (self.ok, "ok" if self.ok else "http_500")

    def telegram(self, text, chat):
        self.texts.append(text)
        return True


class TestTick(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.tmp.name, "state.json")
        self.log = os.path.join(self.tmp.name, "turns.log")
        self.f = Fakes()
        self.now = NOON_UTC

    def tearDown(self):
        self.tmp.cleanup()
        callback._logged.clear()
        callback._latched = False

    def item(self, role="CRM", age=60, name="u1.json", text="SECRET REPLY TEXT", now=None):
        now = now or self.now
        return [("/x/" + name, {"from": role, "text": text, "created": now - age})]

    def hand(self, role="CRM", age=600, now=None):
        return [((now or self.now) - age, role)]

    def run_tick(self, live=False, items=None, handoffs=None, now=None, **over):
        n = now or self.now
        return callback.tick(live, now=n, log_path=self.log, aliases=ALIASES, cfg=cfg_for(**over),
                             items=self.item(now=n) if items is None else items,
                             handoffs=self.hand(now=n) if handoffs is None else handoffs, state_path=self.state,
                             place=self.f.place, telegram=self.f.telegram, url_fn=lambda: "https://x/voice")

    def logtext(self):
        return open(self.log).read() if os.path.exists(self.log) else ""

    def test_off_by_default_does_nothing(self):
        self.assertIsNone(self.run_tick(callback_on_reply=False))
        self.assertEqual(self.f.calls, [])
        self.assertFalse(os.path.exists(self.log))

    def test_reply_to_recent_handoff_places_one_call_to_first_caller_only(self):
        self.assertEqual(self.run_tick(), "placed")
        self.assertEqual(self.f.calls, [(TO, FROM, "https://x/voice")])
        self.assertIsNone(self.run_tick(now=self.now + 5, items=self.item()))   # the same update never calls twice
        self.assertEqual(len(self.f.calls), 1)

    def test_alias_role_counts_as_the_same_session(self):
        self.assertEqual(self.run_tick(items=self.item(role="VAULT"), handoffs=self.hand(role="CRM")), "placed")

    def test_no_call_without_a_matching_recent_handoff(self):
        self.assertIsNone(self.run_tick(handoffs=[]))
        self.assertIsNone(self.run_tick(handoffs=self.hand(role="WEBSITE")))
        self.assertIsNone(self.run_tick(handoffs=[(self.now + 100, "CRM")]))   # hand-off made after the reply
        self.assertIsNone(self.run_tick(items=self.item(role="TEST")))
        self.assertEqual(self.f.calls, [])

    def test_recent_handoffs_reads_files_and_honours_the_60_minute_window(self):
        d = os.path.join(self.tmp.name, "handoff")
        os.makedirs(d)
        base = time.time()
        fresh = time.strftime("%Y%m%d-%H%M%S", time.localtime(base - 600))
        old = time.strftime("%Y%m%d-%H%M%S", time.localtime(base - 7200))
        for fname, stamp, posted in (("a.json", fresh, True), ("b.json", old, True), ("c.json", fresh, False)):
            with open(os.path.join(d, fname), "w") as f:
                json.dump({"created": stamp, "source": "voice_call", "posted": posted, "to": "crm"}, f)
        got = callback.recent_handoffs(base, 3600, d)
        self.assertEqual([r for _, r in got], ["CRM"])

    def test_never_while_a_call_is_live(self):
        self.assertEqual(self.run_tick(live=True), "call_live")
        self.assertEqual(self.f.calls, [])
        self.assertEqual(self.f.texts, [])
        self.assertEqual(self.run_tick(now=self.now + 30, items=self.item(now=self.now + 30)), "placed")   # still waiting after the call

    def test_quiet_hours_send_telegram_only_and_once(self):
        night = self.now + 10 * 3600   # 22:00 UTC
        self.assertEqual(self.run_tick(now=night, items=self.item(text="reply words", now=night)), "quiet_hours")
        self.assertEqual(self.f.calls, [])
        self.assertEqual(len(self.f.texts), 1)
        self.assertIn("reply words", self.f.texts[0])
        self.assertIsNone(self.run_tick(now=night + 5, items=self.item(text="reply words", now=night)))
        self.assertEqual(len(self.f.texts), 1)

    def test_quiet_window_wraps_midnight_and_bad_spec_is_never_quiet(self):
        def at(h):
            return self.now - 12 * 3600 + h * 3600
        self.assertTrue(callback.in_quiet(at(23), "21:00-08:00", "UTC"))
        self.assertTrue(callback.in_quiet(at(3), "21:00-08:00", "UTC"))
        self.assertFalse(callback.in_quiet(at(8), "21:00-08:00", "UTC"))
        self.assertFalse(callback.in_quiet(at(12), "21:00-08:00", "UTC"))
        self.assertFalse(callback.in_quiet(at(23), "garbage", "UTC"))

    def test_one_call_back_per_20_minutes(self):
        self.assertEqual(self.run_tick(), "placed")
        n1, n2 = self.now + 600, self.now + 1300
        self.assertEqual(self.run_tick(now=n1, items=self.item(name="u2.json", now=n1)), "gap")
        self.assertEqual(self.run_tick(now=n2, items=self.item(name="u2.json", now=n2)), "placed")
        self.assertEqual(len(self.f.calls), 2)

    def test_six_per_day(self):
        start = self.now - 4 * 3600   # 08:00 UTC, outside quiet hours
        for i in range(6):
            n = start + i * 1300
            self.assertEqual(self.run_tick(now=n, items=self.item(name="u%d.json" % i, now=n)), "placed")
        n = start + 6 * 1300
        self.assertEqual(self.run_tick(now=n, items=self.item(name="u9.json", now=n)), "daily_cap")
        self.assertEqual(len(self.f.calls), 6)

    def test_twilio_failure_never_raises_waits_then_retries(self):
        self.f.ok = False
        self.assertEqual(self.run_tick(), "twilio_error")
        self.assertEqual(self.run_tick(now=self.now + 60, items=self.item(now=self.now + 60)), "retry_wait")
        self.f.ok = True
        self.assertEqual(self.run_tick(now=self.now + 400, items=self.item(now=self.now + 400)), "placed")

    def test_not_configured_when_number_or_callers_missing(self):
        self.assertEqual(self.run_tick(twilio_number=""), "not_configured")
        self.assertEqual(self.run_tick(twilio_allowed_callers=[]), "not_configured")
        self.assertEqual(self.f.calls, [])

    def test_crash_inside_never_raises(self):
        def boom(*a):
            raise RuntimeError("x")
        r = callback.tick(False, now=self.now, log_path=self.log, aliases=ALIASES, cfg=cfg_for(), items=self.item(),
                          handoffs=self.hand(), state_path=self.state, place=boom, telegram=self.f.telegram, url_fn=lambda: "u")
        self.assertEqual(r, "error")

    def test_log_lines_have_no_text_numbers_or_secret(self):
        self.run_tick()
        self.run_tick(live=True, items=self.item(name="u2.json"))
        night = self.now + 10 * 3600
        self.run_tick(now=night, items=self.item(name="u3.json", now=night))
        text = self.logtext()
        self.assertIn("callback: placed", text)
        self.assertIn("callback: skipped reason=call_live", text)
        for bad in ("SECRET REPLY TEXT", "5550001111", "5550002222", SECRET, HOST):
            self.assertNotIn(bad, text)


class TestRepeatCallGuards(TestTick):
    def test_timeout_after_send_counts_as_placed(self):
        self.f.place = lambda to, from_, url: (False, "uncertain_TimeoutError")
        self.assertEqual(self.run_tick(), "placed")
        n = self.now + 600
        self.assertEqual(self.run_tick(now=n, items=self.item(name="u2.json", now=n)), "gap")

    def test_place_call_marks_a_urlopen_timeout_uncertain_but_a_creds_failure_not(self):
        def slow(req, timeout=0):
            raise TimeoutError()
        creds = lambda: {"account_sid": "AC", "key_sid": "SK", "key_secret": "x"}
        self.assertEqual(callback.place_call(TO, FROM, "u", creds, slow), (False, "uncertain_TimeoutError"))
        def nocreds():
            raise KeyError("x")
        self.assertEqual(callback.place_call(TO, FROM, "u", nocreds, slow), (False, "KeyError"))

    def test_unwritable_state_latches_and_refuses_further_calls(self):
        self.state = os.path.join(self.tmp.name, "nodir", "state.json")
        open(os.path.join(self.tmp.name, "nodir"), "w").close()   # a file where the folder should be
        self.assertEqual(self.run_tick(), "placed")
        self.assertEqual(len(self.f.calls), 1)
        n = self.now + 5000
        self.assertEqual(self.run_tick(now=n, items=self.item(name="u2.json", now=n)), "state_unwritable")
        self.assertEqual(len(self.f.calls), 1)
        self.assertIn("reason=state_unwritable", self.logtext())

    def test_wrong_typed_state_fields_are_reset_not_fatal(self):
        for bad in ({"placed": "x", "handled": [1]}, {"placed": [None, "y", 5.0], "handled": {"a": "z"}, "fail_at": "q"}, [1, 2]):
            with open(self.state, "w") as f:
                json.dump(bad, f)
            st = callback.load_state(self.state)
            self.assertIsInstance(st["placed"], list)
            self.assertIsInstance(st["handled"], dict)
            self.assertNotIn("a", st["handled"])
        with open(self.state, "w") as f:
            json.dump({"placed": "x", "handled": [1]}, f)
        self.assertEqual(self.run_tick(), "placed")

    def test_quiet_hours_failed_telegram_is_retried(self):
        night = self.now + 10 * 3600
        it = self.item(text="r", now=night)
        self.f.telegram = lambda text, chat: False
        self.assertEqual(self.run_tick(now=night, items=it), "quiet_hours")
        sent = []
        self.f.telegram = lambda text, chat: sent.append(text) or True
        self.assertEqual(self.run_tick(now=night + 5, items=it), "quiet_hours")
        self.assertEqual(len(sent), 1)
        self.assertIsNone(self.run_tick(now=night + 10, items=it))


class FakeResp:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestPlaceCall(unittest.TestCase):
    creds = staticmethod(lambda: {"account_sid": "ACtest", "key_sid": "SKtest", "key_secret": "shh"})

    def test_request_shape(self):
        seen = {}

        def fake(req, timeout=0):
            seen["url"], seen["data"], seen["auth"] = req.full_url, req.data.decode(), req.get_header("Authorization")
            return FakeResp()
        ok, why = callback.place_call(TO, FROM, "https://h/tw/x/voice", self.creds, fake)
        self.assertTrue(ok)
        self.assertTrue(seen["url"].endswith("/Accounts/ACtest/Calls.json"))
        self.assertIn("MachineDetection=Enable", seen["data"])
        self.assertIn("Method=GET", seen["data"])
        self.assertTrue(seen["auth"].startswith("Basic "))

    def test_http_error_and_bad_credentials_return_a_reason_without_secrets(self):
        def http_err(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)
        self.assertEqual(callback.place_call(TO, FROM, "u", self.creds, http_err), (False, "http_401"))

        def bad_creds():
            raise KeyError("shh")
        ok, why = callback.place_call(TO, FROM, "u", bad_creds, http_err)
        self.assertEqual((ok, why), (False, "KeyError"))


class TestMachine(unittest.TestCase):
    def test_machine_sends_update_text_to_telegram_and_marks_spoken(self):
        texts, marked = [], []
        items = [("/x/a.json", {"from": "CRM", "text": "first reply"}), ("/x/b.json", {"from": "WEBSITE", "text": "second reply"})]
        ok = callback.on_machine(log_path=os.devnull, telegram=lambda t, c: texts.append(t) or True, items=items, mark=marked.append)
        self.assertTrue(ok)
        self.assertIn("first reply", texts[0])
        self.assertIn("second reply", texts[0])
        self.assertEqual(marked, ["/x/a.json", "/x/b.json"])

    def test_failed_telegram_keeps_updates_pending(self):
        marked = []
        ok = callback.on_machine(log_path=os.devnull, telegram=lambda t, c: False,
                                 items=[("/x/a.json", {"from": "CRM", "text": "r"})], mark=marked.append)
        self.assertFalse(ok)
        self.assertEqual(marked, [])

    def test_machine_path_never_raises(self):
        def boom(t, c):
            raise RuntimeError("x")
        self.assertFalse(callback.on_machine(log_path=os.devnull, telegram=boom,
                                             items=[("/x/a.json", {"from": "CRM", "text": "r"})]))


class TestBridgeCallback(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.hits = []
        self.bridge = tb.Bridge("246810", SECRET, [TO], HOST, auth_token_fn=lambda: "", dot_number=FROM,
                                machine_fn=lambda: self.hits.append(1))
        self.server = await tb.serve(self.bridge, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    def get(self, query):
        def go():
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/tw/%s/voice?%s" % (self.port, SECRET, query), timeout=5) as r:
                    return r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()
        return asyncio.to_thread(go)

    def q(self, **kw):
        base = {"From": FROM, "To": TO, "Direction": "outbound-api", "CallSid": "CA1", "AnsweredBy": "human"}
        base.update(kw)
        return urllib.parse.urlencode(base)

    async def test_human_answer_goes_through_the_normal_sign_in(self):
        status, body = await self.get(self.q())
        self.assertEqual(status, 200)
        self.assertIn("<Gather", body)
        self.assertEqual(self.hits, [])

    async def test_machine_answer_hangs_up_and_triggers_the_text(self):
        status, body = await self.get(self.q(AnsweredBy="machine_start"))
        self.assertIn("<Hangup/>", body)
        self.assertNotIn("<Gather", body)
        await asyncio.sleep(0.2)
        self.assertEqual(self.hits, [1])

    async def test_our_number_is_rejected_unless_it_is_our_own_outbound_to_an_allowed_number(self):
        for kw in ({"Direction": "inbound"}, {"To": "+15550007777"}, {"From": "+15550008888"}):
            status, body = await self.get(self.q(**kw))
            self.assertIn("<Reject/>", body)

    async def test_machine_flag_on_an_inbound_call_is_ignored(self):
        status, body = await self.get(self.q(From=TO, To=FROM, Direction="inbound", AnsweredBy="machine_start"))
        self.assertIn("<Gather", body)
        self.assertEqual(self.hits, [])


class TestLoopWiring(unittest.TestCase):
    def test_defaults_are_off_and_the_loop_is_started(self):
        import config
        self.assertFalse(config.DEFAULTS["callback_on_reply"])
        self.assertEqual(config.DEFAULTS["twilio_number"], "")
        src = open(os.path.join(os.path.dirname(HERE), "server.py")).read()
        self.assertIn("asyncio.create_task(callback_loop(svc))", src)


if __name__ == "__main__":
    unittest.main()
