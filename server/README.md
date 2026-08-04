# HRI-EMO Server

FastAPI server that turns speech into an emotion prediction, plus the web
**configuration platform** where an experimenter picks which trained checkpoint and
which output format the pipeline uses. NAO and the browser both talk to the same
`/predict` endpoint.

This page is the full setup guide. Follow the steps in order.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 or newer | Developed and tested on 3.12 |
| ~3 GB free disk | WavLM, BERT and Whisper weights download on first use |
| A modern browser | For the platform page and its in-browser recorder |

`ffmpeg` is **not** required. Uploads are decoded with `soundfile` and `librosa`,
and the browser converts its recording to 16 kHz mono WAV before sending it.

Only if you plan to use the two **MOSEI** checkpoints, you additionally need
MATLAB (or Octave), the COVAREP repository, and the GloVe vectors. See Section 6.
The three other checkpoints work without any of that.

---

## 2. Install

From the repository root:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Which requirements file do I need?

| File | Use it when |
|---|---|
| `requirements.txt` | **Always.** This is the only file the server needs. It covers PyTorch, Whisper, WavLM/BERT, FastAPI and the GloVe reader. |
| `requirements-server.txt` | Never. Deprecated alias that just points to `requirements.txt`, kept so old commands do not break. |
| `requirements-mosei-offline.txt` | Only if you re-extract MOSEI features from the original CSD files for **training** (`scripts/mosei_feature_extraction_seq_level/`). It pulls in the CMU MultimodalSDK. **It is not needed to run the MOSEI checkpoints on the server**; the online MOSEI path uses COVAREP and GloVe instead (Section 6). |

### Do I need a GPU?

Not for running the server. It uses a GPU automatically if one is available, and
falls back to CPU otherwise; the latency figures reported for this project were
measured on CPU. A GPU matters if you retrain the checkpoints or evaluate a whole
corpus in batch. To install a CUDA build of PyTorch, do it before the install
above:

```bash
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 3. Put the checkpoints in place

Trained weights live in `runs/`, which is gitignored, so a fresh clone does not
have them. Copy them from the machine that trained them, or retrain with the
scripts in `scripts/fusion/`. The server expects exactly these paths, relative to
the repository root:

```
runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt
runs/meld_4cls_wavlm_bert/best_fusion_seq_decoder.pt
runs/meld_sentiment_wavlm_bert/best_fusion_seq_decoder.pt
runs/mosei_6cls_v3/best_mosei_6cls.pt
runs/mosei_sentiment/best_mosei_sentiment.pt
```

The paths are declared in `server/checkpoint_registry.py`. A checkpoint whose file
is missing still appears on the platform, but is marked unavailable with the
reason, so nothing crashes if you only have some of them.

---

## 4. Start the server

Run from the **repository root**, not from inside `server/`:

```bash
source venv/bin/activate        # Windows: venv\Scripts\activate
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Keep `--host 0.0.0.0` so that NAO can reach the server over the LAN.

If the virtual environment lives somewhere else, for example one shared between
several projects, activate it by its full path but still run uvicorn from this
repository root:

```bash
cd ~/projects/hri-emo-server        # this repository
source ~/envs/shared/bin/activate   # the environment, wherever it is
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

uvicorn resolves the `server` package from the current working directory, not from
the location of the environment. Starting it from a different folder loads whatever
`server/` package is there, which is a common cause of `ImportError` on names that
exist in this repository but not in the other copy.

Wait for this line before opening the browser:

```
Application startup complete.
```

Check that it is alive:

```bash
curl -s http://localhost:8000/health
```

Notes:

- **The first prediction is slow.** Models load on demand. The first WavLM+BERT
  request downloads and loads several hundred MB and takes a few seconds. The
  first MOSEI request also loads the GloVe table and starts MATLAB, which can take
  several minutes. Later requests are fast.
- For auto-reload during development use `--reload --reload-dir server`. Do not use
  a bare `--reload`, which watches the virtual environment and reload-loops.
- Stop with `Ctrl+C`. If you backgrounded it, use
  `pkill -f "uvicorn server.main"` on macOS or Linux, or find the process on the
  port with `netstat -ano | findstr :8000` and `taskkill /PID <pid> /F` on Windows.

---

## 5. Open the configuration platform

**http://localhost:8000/platform**

| Tab | What it does |
|---|---|
| **Model Settings** | Pick the active checkpoint and the output format. Each card shows the labels, the metrics, the expected latency, and guidance on when to use it. Saved settings go to `server/runtime_settings.json` and apply to the next request, including requests from NAO. No restart needed. |
| **Playground** | Record from your microphone in the browser, send it through the live pipeline, and see the transcript and the prediction. Use this to confirm the whole chain works before involving the robot. |
| **NAO Integration** | Enter this machine's LAN IP and generate the Choregraphe script with the address already filled in, plus copy-paste setup steps. |

### Quickest way to check the pipeline

This link loads a built-in example clip, sends it through the pipeline and shows the
result, with no microphone involved:

```
http://localhost:8000/platform?demo=1#playground
```

If a transcript and a prediction appear, then Whisper, the feature extraction, the
model and the output policy are all working. The clip is `server/demo_audio.wav`,
about three seconds of speech; if that file is missing the page reports it and the
rest of the platform still works.

The platform accepts a few query parameters, which are convenient for demos and for
taking screenshots:

| Parameter | Effect |
|---|---|
| `?demo=1` | Load the built-in clip and run it immediately |
| `?sel=<checkpoint_id>` | Preselect a checkpoint, for example `meld_4cls` |
| `?ip=<address>` | Prefill the NAO tab and generate the script |
| `#playground` or `#nao` | Open that tab directly |

They combine, so `?sel=meld_4cls&demo=1#playground` selects the MELD 4-class model,
opens the Playground and runs one prediction.

Two other pages are available: `http://localhost:8000/docs` for the interactive API
documentation, and `http://localhost:8000/localtestpage` for a minimal upload test
page.

---

## 6. Optional: enable the MOSEI checkpoints

The two MOSEI checkpoints use COVAREP audio features and GloVe word vectors, which
are not Python packages. Until these are configured, both appear as unavailable on
the platform and the other three checkpoints keep working normally.

You need three things:

1. The [COVAREP](https://github.com/covarep/covarep) repository, cloned locally.
2. MATLAB or Octave, to run COVAREP.
3. `glove.840B.300d.txt` from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/),
   about 2 GB uncompressed.

Then set the paths in `server/config.py`. That file has a Windows branch and a
macOS/Linux branch, so edit the one that matches your machine:

| Variable | Meaning |
|---|---|
| `COVAREP_ROOT` | Path to the cloned COVAREP repository |
| `COVAREP_RUNNER_BIN` | Path to the `matlab` executable, or `octave` |
| `GLOVE_MODEL_PATH` | Path to `glove.840B.300d.txt` |
| `COVAREP_TIMEOUT_SEC` | Raise this if the first MATLAB launch times out |

Restart the server and reload the platform. The MOSEI cards should now be
selectable. See [tools/covarep/README.md](../tools/covarep/README.md) for the
COVAREP setup details.

---

## 7. Connect NAO

Open the **NAO Integration** tab, enter this machine's LAN IP, and generate the
script. Paste it into a Python Script box in Choregraphe and wire the box to a
trigger. The robot and the server must be on the same network.

The generated script contains no model settings, so changing the checkpoint on the
platform does not require regenerating it. Full details in
[nao_scripts/README.md](nao_scripts/README.md).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'server'` | You started uvicorn from inside `server/`. Run it from the repository root. |
| A card says "Checkpoint file missing" | The `.pt` file is not at the path listed in Section 3. |
| MOSEI cards stay unavailable | A COVAREP, MATLAB or GloVe path is wrong in `server/config.py`. The card text names which one. |
| First MOSEI request times out | MATLAB's first launch is slow. Increase `COVAREP_TIMEOUT_SEC` in `server/config.py`. |
| NAO cannot reach the server | The server was started without `--host 0.0.0.0`, or the robot is on a different network. |
| The server reloads in a loop | You used a bare `--reload`. Add `--reload-dir server`. |
| `No module named '_lzma'` on macOS with pyenv | The Python build is missing xz. See the note at the top of `requirements.txt`. |

---

## Layout

```
server/
├── main.py                 # HTTP routes
├── emotion_service.py      # Orchestrates one prediction
├── checkpoint_registry.py  # Which checkpoints exist, and their metadata
├── runtime_settings.py     # Reads and validates the saved platform settings
├── config.py               # Paths, model sizes, COVAREP and GloVe settings
├── emotion_inference/      # Model loading, caching, output policy
├── feature_extraction/     # Whisper, WavLM, BERT, COVAREP, GloVe
├── platform.html           # The configuration platform page
├── localtestpage.html      # Minimal upload test page
└── nao_scripts/            # Choregraphe client for the robot
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Audio in, prediction out. Used by NAO and the Playground. |
| `GET /api/checkpoints` | All checkpoints with metadata and availability |
| `GET` / `PUT /api/config` | Read or save the active checkpoint and output format |
| `GET /api/nao-script?host=<ip>` | Generate the robot script |
| `GET /platform` | The configuration platform |
| `GET /health` | Loaded models and feature extractor availability |
