"""--fake only: a stand-in for the Gemini Live session, so the browser harness can drive the real GeminiBridge and
the real page in Live mode. No network.

It behaves like the real service where it matters for her calls:
- server VAD from the audio level: speech start/end (voice_activity events unless the file "fake_no_va" exists in
  the working directory), with the silence length from the bridge's realtime_input_config;
- her words while she talks (input transcription "w1 w2 ...", WORD_LAG_S late like the real one, so her last words
  arrive after the reply already started; the real service sends no interim transcription);
- a reply sent much faster than real time (3 s of tone in bursts) followed at once by turn_complete, so a barge-in
  on the reply's tail gets no "interrupted" from the service - exactly like the real one;
- "interrupted" only while the reply is still being generated.
The file "fake_live_down" makes connect fail (provider outage); "fake_echo" makes the service hear Dot's own reply
(a voice start and Dot's words, while her mic stays quiet). VOICE_FAKE_LIVE_CONNECT_S delays connect.
"""
import array
import asyncio
import math
import os
from types import SimpleNamespace as NS

RATE_IN = 16000
SPEECH_RMS = 1500
REPLY_S = 3.0
WORD_LAG_S = 1.4
SAID = "This is the fake reply for you."


def _flag(name):
    return os.path.exists(os.path.join(os.getcwd(), name))


def _rms(pcm):
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if not a:
        return 0.0
    return math.sqrt(sum(x * x for x in a) / len(a))


def _tone(seconds, rate=24000):
    n = int(seconds * rate)
    return array.array("h", (int(6000 * math.sin(2 * math.pi * 330 * i / rate)) for i in range(n))).tobytes()


def _content(**kw):
    base = dict(input_transcription=None, interim_input_transcription=None, output_transcription=None,
                interrupted=None, model_turn=None, turn_complete=None)
    base.update(kw)
    return NS(server_content=NS(**base), voice_activity=None, tool_call=None, tool_call_cancellation=None,
              go_away=None, session_resumption_update=None)


def _msg(**kw):
    base = dict(server_content=None, voice_activity=None, tool_call=None, tool_call_cancellation=None, go_away=None,
                session_resumption_update=None)
    base.update(kw)
    return NS(**base)


class FakeLiveSession:
    def __init__(self, config):
        self.q = asyncio.Queue()
        aad = getattr(getattr(config, "realtime_input_config", None), "automatic_activity_detection", None)
        self.silence_s = (getattr(aad, "silence_duration_ms", None) or 500) / 1000.0
        self.speaking = False
        self.quiet_s = 0.0
        self.words = 0
        self.word_clock = 0.0
        self.generating = None   # task sending the reply
        self.closed = False

    async def send_realtime_input(self, audio=None, **kw):
        if self.closed:
            raise ConnectionError("closed")
        pcm = audio.data
        dur = len(pcm) / 2.0 / RATE_IN
        loud = _rms(pcm) > SPEECH_RMS
        if loud:
            self.quiet_s = 0.0
            if not self.speaking:
                self.speaking = True
                if not _flag("fake_no_va"):
                    self.q.put_nowait(_msg(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
                if self.generating and not self.generating.done():
                    self.generating.cancel()
                    self.q.put_nowait(_content(interrupted=True))
            self.word_clock += dur
            if self.word_clock >= 0.3:
                self.word_clock = 0.0
                self.words += 1
                w = NS(text=" w%d" % self.words, finished=False)
                asyncio.get_running_loop().call_later(WORD_LAG_S, self.q.put_nowait, _content(input_transcription=w))
        elif self.speaking:
            self.quiet_s += dur
            if self.quiet_s >= self.silence_s:
                self.speaking = False
                self.words = 0
                if not _flag("fake_no_va"):
                    self.q.put_nowait(_msg(voice_activity=NS(voice_activity_type="ACTIVITY_END")))
                self.generating = asyncio.ensure_future(self._reply())

    async def _reply(self):
        audio = _tone(REPLY_S)
        piece = 15360
        for i in range(0, len(audio), piece):   # far faster than real time, like the real service
            part = NS(inline_data=NS(data=audio[i:i + piece]))
            self.q.put_nowait(_content(model_turn=NS(parts=[part])))
            await asyncio.sleep(0.01)
        self.q.put_nowait(_content(output_transcription=NS(text=SAID)))
        self.q.put_nowait(_content(turn_complete=True))
        if _flag("fake_echo"):   # the phone plays the reply; its speaker leaks into her mic
            await asyncio.sleep(1.0)
            self.q.put_nowait(_msg(voice_activity=NS(voice_activity_type="ACTIVITY_START")))
            self.q.put_nowait(_content(input_transcription=NS(text=" the fake reply for", finished=False)))
            await asyncio.sleep(REPLY_S)
            self.q.put_nowait(_msg(voice_activity=NS(voice_activity_type="ACTIVITY_END")))

    async def receive(self):
        while True:
            msg = await self.q.get()
            if isinstance(msg, Exception):
                raise msg
            yield msg
            sc = msg.server_content
            if sc is not None and sc.turn_complete:
                return

    async def send_client_content(self, turns=None, turn_complete=True):
        if not turn_complete:   # past turns of the call (seed): remembered, never answered - like the real one
            self.seeded = list(turns or [])
            return
        self.generating = asyncio.ensure_future(self._reply())

    async def send_tool_response(self, function_responses=None):
        pass

    async def close(self):
        self.closed = True
        self.q.put_nowait(ConnectionError("closed"))


async def fake_connect(model, config=None, api_key=None, additional_headers=None, **kw):
    if additional_headers is not None:   # the Qwen bridge: never faked
        raise ConnectionRefusedError("fake mode: no Qwen Live")
    if _flag("fake_live_down"):
        raise ConnectionRefusedError("fake mode: Live provider down")
    await asyncio.sleep(float(os.environ.get("VOICE_FAKE_LIVE_CONNECT_S", "0.8")))
    return FakeLiveSession(config)
