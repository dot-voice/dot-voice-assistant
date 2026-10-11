# Dot - a hands-free voice front desk for your AI agents

Dot is a small, self-hosted voice assistant you talk to from your phone, hands free, like a phone call that is
always on. It answers short questions itself and hands real work to your coding or automation agents (for
example Claude Code sessions). When an agent finishes, Dot reads its reply back to you in the call.

It was built for a busy person who works on the road with earphones in, and who wanted to talk to several AI
agents without opening a laptop.

- Talk naturally; your words appear on screen while you speak.
- Dot answers in one or two short sentences, in the language you spoke.
- Real work is handed over with **your exact words**, never a summary.
- Status answers are read **word for word** from what the agents themselves saved. Dot does not guess or retell.
- A menu at the top picks which agent your requests go to (or "Auto": Dot routes by what you say).
- **Live mode** (optional): speech-to-speech with Qwen-Omni realtime, about 1.5 s to first sound, and you can talk
  over Dot to interrupt.
- Sign in once with a PIN, then with Face ID (passkeys).
- Runs on your own Mac. Nothing is exposed to the internet: the phone reaches it over Tailscale.

## How it works

```
iPhone (Safari / Home Screen web app)
  |  speech-to-text on the phone, or audio to the Mac (local Whisper via mlx-whisper)
  v
server.py  (WebSocket + small HTTP, 127.0.0.1:8770, published to your tailnet with `tailscale serve`)
  |- frontdesk.py   chat model answers short questions (Claude CLI by default, or Qwen qwen-plus)
  |- status.py      reads the agents' saved notes word for word for status questions
  |- handoff.py     sends real work to an agent (your exact words) and keeps a local record
  |- updates.py     agents drop short spoken replies here; Dot says them in your next quiet moment
  |- tts.py         Qwen streaming voice (qwen3-tts-flash), with a second request when the first is slow
  `- auth.py / passkey.py   PIN with lockout, Face ID (WebAuthn passkey), reconnect tokens
```

Measured on an Apple-silicon Mac with Qwen's Beijing endpoint: first sound 5-7 s after you stop talking in the
classic mode, about 1.5 s (2 s when a tool runs) in Live mode. Lower latency is the next goal.

## Requirements

- A Mac with Apple silicon (for local Whisper through MLX), macOS 14 or later, Python 3.10-3.12.
- [Tailscale](https://tailscale.com) on the Mac and the phone, with **HTTPS certificates** enabled in the tailnet
  settings (the phone needs HTTPS for the microphone and for passkeys).
- An Alibaba Cloud Model Studio (DashScope) API key for Qwen chat and voice. Keys from the China (Beijing) region
  use `dashscope.aliyuncs.com`; international keys use `dashscope-intl.aliyuncs.com`. Both are tried.
- The [Claude Code](https://claude.com/claude-code) CLI (the default chat model), or set Qwen in `brain.txt` (below), and any
  agent system that can read a hand-off message (see "Connecting your agents").

## Install

```bash
git clone https://github.com/<you>/dot-voice-assistant.git
cd dot-voice-assistant
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.json local_config.json      # then edit it (see below)
```

Store your Qwen key in the macOS Keychain (it is never written to a file):

```bash
security add-generic-password -U -s voice-call-qwen -a "$USER" -w
```

(`security` asks for the key; paste it. `QWEN_API_KEY` or `DASHSCOPE_API_KEY` in the environment also work.)

Check the setup, warm the speech model (first run downloads about 1.5 GB), and run:

```bash
.venv/bin/python server.py --check-config
.venv/bin/python server.py --warm
.venv/bin/python server.py --run
```

The first run creates `pin.txt` with a random PIN. Keep it private.

Publish the port to your tailnet (HTTPS, tailnet only, not the public internet):

```bash
tailscale serve --bg 8770
```

Open `https://<your-mac>.<your-tailnet>.ts.net/` on your phone, type the PIN, and start talking.
Tip: in Safari, Share -> Add to Home Screen gives you a full-screen app. In a plain Safari tab you can stop the
microphone prompt with aA -> Website Settings -> Microphone -> Allow.

To keep it running in the background, `server.py --install-launchagent` writes a LaunchAgent that starts the
service at login and restarts it if it stops.

## Configure (`local_config.json`)

`config.example.json` shows every key. The main ones:

| key | what it does |
|---|---|
| `owner`, `assistant` | how Dot greets you and what it calls itself |
| `owner_job`, `style_language`, `style_extra` | a few words about you and how replies should sound, used in the chat prompt |
| `sessions_description` | one paragraph telling the chat model which agents exist |
| `stt_vocabulary`, `lookup_names` | names and places the speech recognizer should expect; names that trigger a notes lookup |
| `places`, `default_place`, `timezone`, `time_label` | for the weather and time answers |
| `handoff_targets`, `handoff_routes` | your agents: the menu entries, and regex rules that route a request by what you say |
| `session_aliases`, `session_names`, `private_roles` | how agents are named in status answers; agents never mentioned |
| `carry_script` | the command that leaves a message for an agent |
| `carryover_dir`, `updates_dir` | where your agents' saved notes are read and where their spoken replies are queued |
| `rp_id` | your Mac's tailnet name, for Face ID passkeys (must match the address you open on the phone) |
| `launchagent_label` | the name of the background service (default `com.dot.voicecall`) |

Environment variables named `VOICE_*` override the config (for example `VOICE_QWEN_VOICE`, `VOICE_QWEN_CHAT`,
`VOICE_QWEN_TIMEOUT`). `VOICE_TTS_ENGINE=qwen-mac` makes the local macOS voice the backup when Qwen's
voice is slow. `local_config.json`, `pin.txt`, `passkeys.json`, `logs/` and `handoff/` are gitignored.

## Dot's face

Put two square pictures at `static/dot/dot-real.jpg` and `static/dot/dot-cartoon.jpg` (tap the face in the call to
switch between them). Without them the page shows a plain animated ring. Use pictures you have the rights to.

## Live mode

Tap **Live** in the call. The phone streams 16 kHz audio to the Mac, which relays it to Qwen-Omni realtime
(`qwen3.5-omni-flash-realtime` by default; `live_model` and `live_voice` in the config change it). Dot answers in
streamed speech and stops the moment you start talking.

- Your audio goes to Alibaba Cloud while Live is on (the classic mode sends text only). Live is off by default.
- Real work still goes through `hand_off` with **your exact words**: the Mac transcribes the turn's audio locally
  with Whisper, because the realtime model may not return a transcript on tool turns. No words, no hand-off.
- Status answers are read word for word (`get_status`); weather, time and notes come from `lookup`.
- Limits per call: 6 hand-offs, 20 s per tool. When a call ends, posted hand-offs are listed in
  `handoff/<timestamp>-call-summary.json`.
- If Live fails, the page says "Live off" and falls back to the classic mode without dropping the call.

## Choosing the chat model

By default Dot answers with the Claude Code CLI (you must be signed in to `claude`). To use Qwen instead (faster from
some regions; your question text then goes to Alibaba Cloud), write one word into `brain.txt`:

```bash
echo qwen > brain.txt
```

## Face ID sign-in

1. Sign in once with the PIN.
2. Tap **Use Face ID next time** in the call.
3. From then on, tap **Talk to Dot (Face ID)**; the PIN stays as the fallback.

Wrong PINs and failed passkeys share one lockout. Passkeys are tied to `rp_id`: if you rename the Mac in
Tailscale, enroll again.

## Connecting your agents

Dot is agent-agnostic. It needs three things, all plain files or one command:

1. **Hand-off** - `handoff.py` passes `(agent, message)` to a "teller". The default teller runs the script named by
   `carry_script` in your config as `python3 <carry_script> tell <AGENT> "<message>" --task VOICE` (not included:
   write your own small Python script that accepts those arguments), or replace `tell_role()` in `handoff.py` with a queue, an API call or a file drop.
   Until you do, hand-offs are still saved in `handoff/<timestamp>.json` with the task and your exact words.
2. **Status** - each agent keeps a short markdown note with a `## Next step` section, indexed in
   `<carryover_dir>/TASKS.json` (`{"CRM": {"file": "CRM.md"}}`). Dot reads the first sentence of it word for word.
3. **Spoken replies** - when an agent finishes something you asked for by voice, it runs:

   ```bash
   python3 updates.py CRM "Done: the six notes are fixed."
   ```

   Dot says "Update from CRM: ..." in your next quiet moment, once.

## Privacy

- Your speech is turned into text on the phone (or locally by Whisper on the Mac).
- The text of each question goes to the chat model you chose (Qwen or Claude), and the text of each reply goes to
  Qwen to be spoken. Nothing else leaves your Mac.
- `logs/turns.log` keeps timings and Dot's own reply text (to debug odd answers), never what you said. Full
  transcripts are written only if you turn on `VOICE_CALL_LOGS=1`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## License

MIT - see [LICENSE](LICENSE). Use it, fork it, change it. No warranty and no support promise: issues are off,
but pull requests and forks are welcome.
