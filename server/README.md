# HRI-EMO Server

FastAPI server for NAO emotion recognition (`/predict`).

## Local self-test

Run from the **repo root** (parent of `server/`):

```bash
python -m server.main
```

Watch the terminal until you see:

```
Application startup complete.
```

Then open in your browser:

**http://localhost:8000/localtestpage**

Record audio, send it to `/predict`, and inspect the JSON response.

Other URLs:

- API docs: http://localhost:8000/docs

## Layout

```
server/
├── main.py              # HTTP routes + app entry
├── emotion_service.py   # Orchestrates predict flow
├── emotion_inference/   # Fusion decoder wrapper
├── feature_extraction/  # Whisper, WavLM, BERT, MOSEI features
├── config.py            # Checkpoints and model settings
├── localtestpage.html   # Browser self-test page
└── nao_scripts/         # Choregraphe Python client for NAO
```

NAO integration: see [nao_scripts/README.md](nao_scripts/README.md).

## MOSEI online features (COVAREP + GloVe)

Before using `benchmark=mosei`, configure `server/config.py`:

| Variable | Description |
|----------|-------------|
| `COVAREP_ROOT` | Path to cloned [covarep/covarep](https://github.com/covarep/covarep) repo |
| `COVAREP_RUNNER_BIN` | `matlab` or `octave` |
| `GLOVE_MODEL_PATH` | Path to **`glove.840B.300d.txt`** ([Stanford GloVe](https://nlp.stanford.edu/projects/glove/)) |
| `WHISPER_WORD_TIMESTAMPS` | `True` for MOSEI: order GloVe words by Whisper word timestamps |
| `MOSEI_CHECKPOINT` | `checkpoints/mosei/best_mosei_fusion_decoder.pt` (set in `server/config.py`) |

Validate extraction vs offline CSD (optional):

```bash
PYTHONPATH=. python scripts/validate_mosei_online_features.py --wav your_clip.wav
```

See [tools/covarep/README.md](../tools/covarep/README.md) for COVAREP setup.
