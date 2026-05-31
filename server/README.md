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
