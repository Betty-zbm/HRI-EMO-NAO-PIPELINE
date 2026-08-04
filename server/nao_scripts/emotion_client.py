# -*- coding: utf-8 -*-
"""
HRI-EMO NAO client — record speech, send WAV to emotion server, speak result.

Designed for NAOqi 2.5 (Python 2.7) inside a Choregraphe **Python script** box.

Flow (matches project architecture):
  NAO microphone
    -> ALSoundDetection (auto start / stop on silence)
    -> ALAudioRecorder (mono 16 kHz WAV)
    -> HTTP POST /predict on HRI-EMO server
    -> ALTextToSpeech (speak predicted emotion)

Choregraphe entry point: ``__main__(app)`` (see README in this folder).
"""
from __future__ import print_function

import json
import os
import time
import uuid

# ---------------------------------------------------------------------------
# Configuration — edit before deploying to the robot
# ---------------------------------------------------------------------------
SERVER_HOST = "192.168.1.107"  # LAN IP of the machine running ``python -m server.main``
SERVER_PORT = 8000
# None = use the checkpoint + output settings configured on the platform
# (http://<SERVER_HOST>:<SERVER_PORT>/platform). Set to "iemocap" or "mosei"
# only to force the legacy fixed benchmarks.
SERVER_BENCHMARK = None

RECORD_DIR = "/home/nao/recordings"
RECORD_BASENAME = "hri_emo_capture.wav"

SAMPLE_RATE = 16000
# NAO has 4 mics. Use all mics for reliability; try [0, 0, 1, 0] for front only.
MIC_CHANNELS = [1, 1, 1, 1]

# ALSoundDetection + ALMemory events; byte-growth fallback when callbacks don't fire.
SOUND_SENSITIVITY = 0.65  # 0–1, higher = easier to trigger
SOUND_SILENCE_SEC = 0.4
MIN_RECORD_SEC = 0.8
END_SILENCE_SEC = 1.5     # stop after this much quiet once speech was heard
FALLBACK_RECORD_SEC = 5.0 # if events never fire, stop after this (still uploads)
MAX_RECORD_SEC = 10.0
POLL_INTERVAL_SEC = 0.05
BYTE_GROWTH_THRESHOLD = 400  # bytes of WAV growth treated as ongoing speech

TTS_LANGUAGE = "English"
TTS_VOLUME = 0.85


class EmotionClient(object):
    """Capture utterance on NAO and call the HRI-EMO ``/predict`` API."""

    def __init__(self, session):
        self.session = session
        self.tts = session.service("ALTextToSpeech")
        self.recorder = session.service("ALAudioRecorder")
        self.sound_detection = session.service("ALSoundDetection")
        self.memory = session.service("ALMemory")
        self.module_name = "HRIEmoEmotionClient"

        self._recording_path = os.path.join(RECORD_DIR, RECORD_BASENAME)
        self._heard_speech = False
        self._last_voice_time = 0.0
        self._last_byte_size = 0

    def _touch_voice(self, source=None):
        """Refresh last-voice timestamp (events or growing WAV file)."""
        self._heard_speech = True
        self._last_voice_time = time.time()
        if source:
            print("[HRI-EMO] voice activity (%s)" % source)

    # Public API
    def run_once(self):
        """Single interaction: listen, upload, speak emotion result."""
        self._ensure_record_dir()
        self.tts.setLanguage(TTS_LANGUAGE)
        self.tts.setVolume(TTS_VOLUME)

        try:
            self._setup_sound_detection()
            self.say("I am listening. Please speak now.")
            wav_path = self._record_until_silence()
            self.say("Analyzing your speech.")
            print("[HRI-EMO] uploading:", wav_path, os.path.getsize(wav_path), "bytes")
            result = self._upload_wav(wav_path)
            self.say(self._format_response(result))
        finally:
            self._teardown_sound_detection()

    # ALMemory callback — fired by ALSoundDetection when sound is present.
    def onSoundDetected(self, value, msg):
        try:
            level = float(value)
        except Exception:
            level = 0.0
        self._touch_voice("event level=%.3f" % level)

    # ------------------------------------------------------------------
    # NAO audio pipeline
    # ------------------------------------------------------------------

    def _ensure_record_dir(self):
        if not os.path.isdir(RECORD_DIR):
            os.makedirs(RECORD_DIR)

    def _setup_sound_detection(self):
        self.sound_detection.subscribe(
            self.module_name,
            SOUND_SENSITIVITY,
            SOUND_SILENCE_SEC,
        )
        self.memory.subscribeToEvent(
            "ALSoundDetection/SoundDetected",
            self.module_name,
            "onSoundDetected",
        )
        print("[HRI-EMO] ALSoundDetection subscribed")

    def _recording_byte_size(self):
        try:
            if os.path.isfile(self._recording_path):
                return os.path.getsize(self._recording_path)
        except Exception:
            pass
        return 0

    def _teardown_sound_detection(self):
        try:
            self.memory.unsubscribeToEvent(
                "ALSoundDetection/SoundDetected",
                self.module_name,
                "onSoundDetected",
            )
        except Exception:
            pass
        try:
            self.sound_detection.unsubscribe(self.module_name)
        except Exception:
            pass

    def _record_until_silence(self):
        if os.path.isfile(self._recording_path):
            try:
                os.remove(self._recording_path)
            except OSError:
                pass

        self._heard_speech = False
        self._last_voice_time = 0.0
        self._last_byte_size = 0

        print("[HRI-EMO] start recording ->", self._recording_path)
        self.recorder.startMicrophonesRecording(
            self._recording_path,
            "wav",
            SAMPLE_RATE,
            MIC_CHANNELS,
        )

        start = time.time()
        try:
            while True:
                now = time.time()
                elapsed = now - start

                byte_size = self._recording_byte_size()
                if byte_size > self._last_byte_size:
                    if byte_size >= self._last_byte_size + BYTE_GROWTH_THRESHOLD:
                        self._touch_voice("wav %d bytes" % byte_size)
                    else:
                        self._touch_voice()
                    self._last_byte_size = byte_size

                if elapsed >= MAX_RECORD_SEC:
                    print("[HRI-EMO] max record time reached")
                    break

                if self._heard_speech and elapsed >= MIN_RECORD_SEC:
                    if (now - self._last_voice_time) >= END_SILENCE_SEC:
                        print("[HRI-EMO] silence after speech, stopping")
                        break

                if not self._heard_speech and elapsed >= FALLBACK_RECORD_SEC:
                    print("[HRI-EMO] fallback stop at %.1fs (no events)" % FALLBACK_RECORD_SEC)
                    break

                time.sleep(POLL_INTERVAL_SEC)
        finally:
            self.recorder.stopMicrophonesRecording()
            self._wait_for_file(self._recording_path, timeout_sec=8.0)

        if not os.path.isfile(self._recording_path):
            raise RuntimeError("Recording file was not created: %s" % self._recording_path)

        size = os.path.getsize(self._recording_path)
        if size < 1024:
            raise RuntimeError(
                "Recording too short (%d bytes). Try MIC_CHANNELS = [1, 1, 1, 1]." % size
            )

        if not self._heard_speech:
            print("[HRI-EMO] warning: no voice activity signal; uploading anyway")

        return self._recording_path

    @staticmethod
    def _wait_for_file(path, timeout_sec=5.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for recorder output: %s" % path)

    # ------------------------------------------------------------------
    # HTTP upload (stdlib only — no ``requests`` on stock NAO)
    # ------------------------------------------------------------------

    def _upload_wav(self, wav_path):
        with open(wav_path, "rb") as handle:
            audio_bytes = handle.read()

        fields = []
        if SERVER_BENCHMARK:
            fields.append(("benchmark", SERVER_BENCHMARK))

        body, content_type = _encode_multipart(
            fields=fields,
            files=[("audio", os.path.basename(wav_path), audio_bytes, "audio/wav")],
        )

        if _use_httplib():
            import httplib

            conn = httplib.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=120)
            try:
                conn.request(
                    "POST",
                    "/predict",
                    body,
                    {
                        "Content-Type": content_type,
                        "Content-Length": str(len(body)),
                    },
                )
                resp = conn.getresponse()
                raw = resp.read()
                status = resp.status
            finally:
                conn.close()
        else:
            import http.client as httplib  # noqa: F401 — Py3 fallback for local lint

            conn = httplib.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=120)
            try:
                conn.request(
                    "POST",
                    "/predict",
                    body,
                    {
                        "Content-Type": content_type,
                        "Content-Length": str(len(body)),
                    },
                )
                resp = conn.getresponse()
                raw = resp.read()
                status = resp.status
            finally:
                conn.close()

        if status != 200:
            detail = raw
            try:
                payload = json.loads(raw)
                detail = payload.get("detail", raw)
                if isinstance(detail, list):
                    detail = "; ".join(str(item) for item in detail)
            except Exception:
                pass
            raise RuntimeError("Server returned HTTP %s: %s" % (status, detail))

        try:
            return json.loads(raw)
        except Exception as exc:
            raise RuntimeError("Invalid JSON from server: %s" % exc)

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_response(result):
        transcript = (result.get("transcript") or "").strip()
        predicted = result.get("predicted") or []

        if not predicted:
            if transcript and transcript != "<empty>":
                return "I heard you say: %s. I could not infer a clear emotion." % transcript
            return "I did not catch any speech."

        labels = ", ".join(predicted)
        if transcript and transcript != "<empty>":
            return "You said: %s. You sound %s." % (transcript, labels)
        return "You sound %s." % labels

    def say(self, text):
        self.tts.say(str(text))


def _use_httplib():
    try:
        import httplib  # noqa: F401

        return True
    except ImportError:
        return False


def _encode_multipart(fields, files):
    """
    Build ``multipart/form-data`` body for ``/predict``.

    ``fields``: list of (name, value)
    ``files``: list of (field_name, filename, bytes, content_type)
    """
    boundary = uuid.uuid4().hex
    lines = []

    for name, value in fields:
        lines.append("--%s\r\n" % boundary)
        lines.append('Content-Disposition: form-data; name="%s"\r\n\r\n' % name)
        lines.append("%s\r\n" % value)

    for field_name, filename, data, content_type in files:
        lines.append("--%s\r\n" % boundary)
        lines.append(
            'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
            % (field_name, filename)
        )
        lines.append("Content-Type: %s\r\n\r\n" % content_type)

    body = "".join(lines).encode("utf-8") + data + ("\r\n--%s--\r\n" % boundary).encode(
        "utf-8"
    )
    content_type = "multipart/form-data; boundary=%s" % boundary
    return body, content_type


# Choregraphe entry point

def _error_hint(exc):
    msg = str(exc).lower()
    if "no speech" in msg:
        return "Sorry, I did not hear you. Please speak louder, closer to my head."
    if "recording" in msg or "recorder" in msg:
        return "Sorry, microphone recording failed."
    if "http" in msg or "server returned" in msg:
        return "Sorry, the server could not analyze the audio."
    if "timed out" in msg:
        return "Sorry, the server took too long to respond."
    return "Sorry, emotion recognition failed."


def __main__(app):
    """
    Paste this file into a Choregraphe **Script** box (Python).

    Connect the box after a start signal (e.g. voice event or button).
    """
    client = EmotionClient(app.session)
    try:
        app.session.registerService(
            client.module_name,
            client,
            client.module_name,
        )
        print("[HRI-EMO] registerService ok")
    except Exception as exc:
        print("[HRI-EMO] registerService:", exc)

    try:
        client.run_once()
    except Exception as exc:
        print("[HRI-EMO] error:", exc)
        try:
            client.say(_error_hint(exc))
        except Exception:
            pass
        raise
