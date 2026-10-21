import asyncio
import base64
import json
import os
import re
import sys
import unittest
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VOICE_CONFIG"] = os.path.join(HERE, "test_config.json")
sys.path.insert(0, os.path.dirname(HERE))

import twilio_bridge as tb  # noqa: E402

SECRET = "s" * 30
CALLER = "+15550001111"
PIN = "246810"
HOST = "dot.example.net"


def make_bridge(callers=(CALLER,), token="", dot_url="ws://127.0.0.1:1/ws", **kw):
    return tb.Bridge(PIN, SECRET, list(callers), HOST, dot_url=dot_url, auth_token_fn=lambda: token, **kw)


async def http_get(port, path, headers=None):
    def go():
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    return await asyncio.to_thread(go)


class TestBridgeHttp(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = make_bridge()
        self.server = await tb.serve(self.bridge, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    def voice(self, caller=CALLER):
        return http_get(self.port, "/tw/%s/voice?From=%%2B%s&CallSid=CA1" % (SECRET, caller.lstrip("+")))

    def pin(self, digits, sid="CA1", caller=CALLER):
        return http_get(self.port, "/tw/%s/pin?From=%%2B%s&CallSid=%s&Digits=%s" % (SECRET, caller.lstrip("+"), sid, digits))

    async def test_allowed_caller_is_asked_for_the_pin(self):
        status, body = await self.voice()
        self.assertEqual(status, 200)
        self.assertIn("<Gather", body)
        self.assertIn('input="dtmf"', body)
        self.assertIn("https://%s/tw/%s/pin" % (HOST, SECRET), body)
        self.assertNotIn("<Stream", body)

    async def test_blocked_caller_is_rejected(self):
        status, body = await self.voice("+15559998888")
        self.assertIn("<Reject/>", body)
        self.assertNotIn("<Gather", body)

    async def test_blocked_caller_is_redirected_to_the_old_voice_url_when_set(self):
        old = "https://old.example/voice.php?client=V&greeting=a+b"
        self.bridge.fallback_voice_url = old
        status, body = await self.voice("+15559998888")
        self.assertIn('<Redirect method="POST">https://old.example/voice.php?client=V&amp;greeting=a+b</Redirect>', body)
        self.assertNotIn("<Gather", body)
        _, body = await self.voice()   # the allowed caller still gets the PIN prompt
        self.assertIn("<Gather", body)
        self.assertNotIn("<Redirect", body)

    async def test_empty_allowlist_rejects_everyone(self):
        self.bridge.allowed = set()
        status, body = await self.voice()
        self.assertIn("<Reject/>", body)

    async def test_wrong_secret_path_is_not_found(self):
        status, _ = await http_get(self.port, "/tw/%s/voice?From=%%2B15550001111" % ("x" * 30))
        self.assertEqual(status, 404)
        status, _ = await http_get(self.port, "/")
        self.assertEqual(status, 404)

    async def test_right_pin_connects_the_stream_with_a_token(self):
        status, body = await self.pin(PIN)
        self.assertIn("<Connect><Stream", body)
        self.assertIn('url="wss://%s/tw/%s/media"' % (HOST, SECRET), body)
        tok = re.search(r'<Parameter name="token" value="([^"]+)"', body).group(1)
        self.assertTrue(self.bridge.tokens.consume(tok))

    async def test_wrong_pin_asks_again_then_hangs_up_on_the_third(self):
        _, b1 = await self.pin("000000")
        _, b2 = await self.pin("000000")
        self.assertIn("<Gather", b1)
        self.assertIn("<Gather", b2)
        _, b3 = await self.pin("000000")
        self.assertIn("<Hangup/>", b3)
        self.assertNotIn("<Gather", b3)
        self.assertNotIn("<Stream", b3)

    async def test_pin_failures_lock_the_bridge(self):
        for i in range(5):
            await self.pin("000000", sid="CA%d" % i)
        _, body = await self.pin(PIN, sid="CA99")   # right PIN while locked is still refused
        self.assertNotIn("<Stream", body)
        self.assertIn("<Hangup/>", body)

    async def test_pin_step_also_checks_the_allowlist(self):
        _, body = await self.pin(PIN, caller="+15559998888")
        self.assertIn("<Reject/>", body)

    async def test_signature_checked_when_an_auth_token_exists(self):
        self.bridge.auth_token_fn = lambda: "tok123"
        path = "/tw/%s/voice?From=%%2B15550001111&CallSid=CA1" % SECRET
        status, _ = await http_get(self.port, path)
        self.assertEqual(status, 403)
        sig = tb.twilio_signature("tok123", "https://%s%s" % (HOST, path))
        status, body = await http_get(self.port, path, {"X-Twilio-Signature": sig})
        self.assertEqual(status, 200)
        self.assertIn("<Gather", body)

    async def test_signature_skipped_without_an_auth_token(self):
        status, _ = await self.voice()
        self.assertEqual(status, 200)

    def test_short_secret_is_refused(self):
        with self.assertRaises(ValueError):
            tb.Bridge(PIN, "short", [CALLER], HOST)


class TestStreamTokens(unittest.TestCase):
    def test_token_is_single_use(self):
        t = tb.StreamTokens()
        tok = t.mint("CA1")
        self.assertTrue(t.consume(tok))
        self.assertFalse(t.consume(tok))

    def test_token_expires_after_60_seconds(self):
        now = [100.0]
        t = tb.StreamTokens(clock=lambda: now[0])
        tok = t.mint("CA1")
        now[0] += 61
        self.assertFalse(t.consume(tok))

    def test_unknown_token_is_refused(self):
        t = tb.StreamTokens()
        self.assertFalse(t.consume("nope"))
        self.assertFalse(t.consume(None))


class TestAudio(unittest.TestCase):
    def test_mulaw_round_trip_stays_close(self):
        x = np.concatenate((np.arange(-32768, 32768, 37), np.array([0, 1, -1, 32767, -32768]))).astype(np.int16)
        y = tb.mulaw_decode(tb.mulaw_encode(x)).astype(np.int32)
        err = np.abs(y - x.astype(np.int32))
        # mu-law keeps ~2.5 percent relative error (plus the small bias near zero)
        self.assertTrue(np.all(err <= np.maximum(160, np.abs(x.astype(np.int32)) // 30)), int(err.max()))

    def test_mulaw_known_values(self):
        self.assertEqual(tb.mulaw_encode(np.array([0], dtype=np.int16)), b"\xff")
        self.assertEqual(tb.mulaw_encode(np.array([32767], dtype=np.int16)), b"\x80")
        self.assertEqual(tb.mulaw_encode(np.array([-32768], dtype=np.int16)), b"\x00")
        self.assertEqual(int(tb.mulaw_decode(b"\xff")[0]), 0)

    def test_upsample_doubles_length_across_frames(self):
        up = tb.Upsampler8to16()
        a = up.feed(np.arange(160, dtype=np.int16))
        b = up.feed(np.arange(160, dtype=np.int16))
        self.assertEqual((len(a), len(b)), (320, 320))

    def test_downsample_24k_to_8k_length_with_odd_chunks(self):
        down = tb.Downsampler24to8()
        pcm = np.zeros(2400, dtype="<i2").tobytes()   # 100 ms at 24 kHz
        total = 0
        for i in range(0, len(pcm), 1001):   # chunks that split samples
            total += len(down.feed(pcm[i:i + 1001]))
        self.assertEqual(total, 800)

    def test_downsample_keeps_a_tone(self):
        t = np.arange(2400) / 24000.0
        tone = (8000 * np.sin(2 * np.pi * 400 * t)).astype("<i2")
        out = tb.Downsampler24to8().feed(tone.tobytes())
        self.assertGreater(int(np.abs(out).max()), 7000)


class FakeDot:
    """Stands in for the Dot service: same handshake, collects what the bridge sends."""

    def __init__(self):
        self.msgs, self.audio = [], []
        self.reply = np.full(2400, 3000, dtype="<i2").tobytes()   # 100 ms of 24 kHz PCM
        self.live_off = asyncio.Event()
        self.heard = asyncio.Event()

    async def handler(self, ws):
        auth = json.loads(await ws.recv())
        self.msgs.append(auth)
        await ws.send(json.dumps({"type": "auth_ok", "version": "x"}))
        await ws.send(json.dumps({"type": "status", "state": "listening"}))
        async for m in ws:
            if isinstance(m, bytes):
                self.audio.append(m)
                if len(self.audio) == 2 and not self.heard.is_set():
                    self.heard.set()
                    await ws.send(json.dumps({"type": "live_audio", "b64": base64.b64encode(self.reply).decode()}))
                    await ws.send(json.dumps({"type": "transcript", "role": "assistant", "text": "ignored"}))
                continue
            d = json.loads(m)
            self.msgs.append(d)
            if d.get("type") == "live" and d.get("on"):
                await ws.send(json.dumps({"type": "live_ok"}))
            elif d.get("type") == "live":
                self.live_off.set()


class TestEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from websockets.asyncio.server import serve
        self.dot = FakeDot()
        self.dot_server = await serve(self.dot.handler, "127.0.0.1", 0)
        dport = self.dot_server.sockets[0].getsockname()[1]
        self.bridge = make_bridge(dot_url="ws://127.0.0.1:%d/ws" % dport)
        self.server = await tb.serve(self.bridge, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        for s in (self.server, self.dot_server):
            s.close()
            await s.wait_closed()

    def url(self):
        return "ws://127.0.0.1:%d/tw/%s/media" % (self.port, SECRET)

    async def test_call_exchanges_audio_both_ways(self):
        from websockets.asyncio.client import connect
        token = self.bridge.tokens.mint("CA1")
        async with connect(self.url()) as tw:
            await tw.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
            await tw.send(json.dumps({"event": "start", "streamSid": "MZ1",
                                      "start": {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": token}}}))
            frame = base64.b64encode(tb.mulaw_encode(np.full(160, 1000, dtype=np.int16))).decode()
            for _ in range(6):   # 6 x 20 ms = 120 ms = 1920 samples at 16 kHz = two 960-sample chunks
                await tw.send(json.dumps({"event": "media", "streamSid": "MZ1", "media": {"track": "inbound", "payload": frame}}))
            media = []
            while sum(len(base64.b64decode(m["media"]["payload"])) for m in media) < 800:
                m = json.loads(await asyncio.wait_for(tw.recv(), 5))
                self.assertEqual(m["event"], "media")
                self.assertEqual(m["streamSid"], "MZ1")
                media.append(m)
            self.assertTrue(all(len(base64.b64decode(m["media"]["payload"])) == 160 for m in media))
            self.assertEqual(len(media), 5)   # 100 ms of reply = 800 mu-law bytes = five 20 ms frames
            self.assertGreater(int(np.abs(tb.mulaw_decode(base64.b64decode(media[0]["media"]["payload"]))).mean()), 2000)
            await tw.send(json.dumps({"event": "stop", "streamSid": "MZ1"}))
            await asyncio.wait_for(self.dot.live_off.wait(), 5)
        auth = self.dot.msgs[0]
        self.assertEqual((auth["type"], auth["pin"], auth["voice"], auth["ears"]), ("auth", PIN, "phone", "phone"))
        self.assertTrue(any(m.get("type") == "live" and m.get("on") for m in self.dot.msgs))
        self.assertEqual(len(self.dot.audio), 2)
        self.assertTrue(all(len(a) == 1920 for a in self.dot.audio))   # 960 samples of 16-bit PCM

    async def test_bad_stream_token_is_dropped_before_dot_is_reached(self):
        from websockets.asyncio.client import connect
        async with connect(self.url()) as tw:
            await tw.send(json.dumps({"event": "start", "streamSid": "MZ1",
                                      "start": {"streamSid": "MZ1", "customParameters": {"token": "forged"}}}))
            with self.assertRaises(Exception):
                await asyncio.wait_for(tw.recv(), 5)
        self.assertEqual(self.dot.msgs, [])

    async def test_live_stop_becomes_a_twilio_clear(self):
        sent = []

        class Twilio:
            async def send(self, s):
                sent.append(json.loads(s))

        async def dot_stream():
            yield json.dumps({"type": "live_ok"})
            yield json.dumps({"type": "live_stop"})
        await self.bridge._from_dot(Twilio(), dot_stream(), "MZ2")
        self.assertEqual(sent, [{"event": "clear", "streamSid": "MZ2"}])


if __name__ == "__main__":
    unittest.main()
