# NAO Choregraphe scripts

Client-side Python for NAOqi 2.5 that talks to the HRI-EMO server (`POST /predict`).

## Architecture

```
NAO microphone
  → ALSoundDetection   (SoundDetected events via ALMemory)
  → ALAudioRecorder    (mono 16 kHz PCM WAV, stops after END_SILENCE_SEC quiet)
  → HTTP POST /predict (your laptop or server)
  → ALTextToSpeech     (speak transcript + predicted emotion)
```

This matches the [NAOqi Audio](http://doc.aldebaran.com/2-5/naoqi/audio/index.html) modules:
`ALSoundDetection`, `ALAudioRecorder`, and `ALTextToSpeech`.

## Is this meant for Choregraphe?

**Yes.** The primary entry point is `__main__(app)` in `emotion_client.py`. You paste the script into a **Script** box in Choregraphe and wire it into a behavior (e.g. after a touch or dialog trigger).

You do **not** need to copy the whole `server/` tree onto the robot — only this Python file (plus network access to the machine running `uvicorn`).

## Setup

### 1. Start the emotion server (on your PC)

From the repo root:

```bash
python -m server.main
```

Note your PC’s LAN IP (e.g. `192.168.1.100`). The robot must reach `http://<IP>:8000/predict`.

### 2. Configure the script

Edit the top of `emotion_client.py`:

| Variable | Meaning |
|----------|---------|
| `SERVER_HOST` | IP of the machine running the server |
| `SERVER_PORT` | Default `8000` |
| `SERVER_BENCHMARK` | `iemocap` or `mosei` |

### 3. Add to Choregraphe

1. Open Choregraphe and connect to the NAO.
2. Create or open a behavior.
3. Add a **Script** box (Python).
4. Copy the full contents of `emotion_client.py` into the box.
5. Connect **onStart** (or your trigger) → the script box.
6. Upload and run on the robot.

The box calls `__main__(app)` automatically; do not rename that function.

### 4. Network

- NAO and the server must be on the same network (or routable).
- Allow inbound TCP on port `8000` on the server host (firewall).
- Test from a browser on another device: `http://<SERVER_HOST>:8000/health`.

## Behavior notes

- **Stop when you finish speaking**: `SoundDetected` events **or** WAV file growth, then `END_SILENCE_SEC` quiet.
- **Fallback**: if Choregraphe never fires events, stops at `FALLBACK_RECORD_SEC` (5 s) and still uploads.
- Never blocks on “heard speech” alone — Whisper on the server decides if audio has content.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| “emotion recognition failed” | Server running? `SERVER_HOST` correct? `/health` reachable? |
| “Sorry, I did not hear you” | Should not happen on latest script; update and re-upload |
| Stops at 5 s every time | Normal if Log shows `fallback stop` — events not firing; audio still uploads |
| Never stops before 10 s | Check Log for `voice activity`; raise `SOUND_SENSITIVITY` or lower `END_SILENCE_SEC` |
| Empty or tiny WAV | Try `MIC_CHANNELS = [1, 1, 1, 1]` |
| HTTP 500 | Server logs; checkpoints in `server/config.py` (random weights still return JSON) |
