# Online COVAREP extraction (MOSEI)

This folder contains the MATLAB/Octave bridge used by `server/feature_extraction/covarep_runner.py`.

## Setup

1. Clone [COVAREP](https://github.com/covarep/covarep) somewhere on your machine:

```bash
git clone https://github.com/covarep/covarep.git ~/tools/covarep-upstream
```

2. Install **MATLAB** with:

   - **Signal Processing Toolbox** — check: `which lpc -all`
   - **Statistics and Machine Learning Toolbox** — check: `which skewness -all`

3. Set paths in `server/config.py`:

```python
COVAREP_ROOT = "/absolute/path/to/covarep"
COVAREP_RUNNER_BIN = "matlab"  # or "octave"
# macOS MATLAB not on PATH:
# COVAREP_RUNNER_BIN = "/Applications/MATLAB_R2026a.app/bin/matlab"
```

## Manual test

From repo root:

```bash
# Octave
octave --no-gui tools/covarep/extract_covarep_segment_octave.m test.wav out.mat /path/to/covarep

# MATLAB
matlab -batch "addpath('tools/covarep'); extract_covarep_segment('test.wav','out.mat','/path/to/covarep')"
```

Output `out.mat` contains `features` with shape `[T, 74]` at ~100 Hz.

## Note on MOSEI alignment

CMU-MOSEI ships precomputed `CMU_MOSEI_COVAREP.csd`. Online extraction may differ slightly
(COVAREP version, polarity, resampling). Use `scripts/validate_mosei_online_features.py`
to quantify the gap before deploying to NAO.
