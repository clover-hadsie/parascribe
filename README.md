# parascribe

A thin, self-hostable HTTP server that exposes the OpenAI
`/v1/audio/transcriptions` API in front of an [`onnx-asr`](https://github.com/istupakov/onnx-asr)
Parakeet TDT model (NVIDIA Parakeet TDT 0.6B v3,
[`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)),
**on GPU, with correct word/segment timestamps**, so it can sit behind a LiteLLM
gateway like any other model.

It exists because the common alternatives each miss a requirement: Speaches
refuses `verbose_json` (no timestamps) for the Parakeet backend, and the Go
`achetronic/parakeet` server is CPU-only. parascribe gives you **timestamps AND
GPU** behind an OpenAI-compatible surface.

License: Apache-2.0.

## Features

- OpenAI-compatible `POST /v1/audio/transcriptions` (`json`, `text`,
  `verbose_json`, `srt`, `vtt`).
- Real word- and segment-level timestamps, correct against the original
  timeline even for multi-hour files (VAD chunking via onnx-asr; per-token
  offsetting + word grouping in `stitch.py`).
- Timestamps are emitted by default for `verbose_json` (does not depend on the
  `timestamp_granularities[]` param, which LiteLLM is known to drop).
- GPU-or-fail-loudly: refuses to start on CPU when configured for CUDA.
- Server-Sent Events streaming for long files (`stream=true`).
- Optional video input (audio track extracted via ffmpeg).
- Forensic-clean: tmpfs temp files deleted in a `finally`, content-free logs.
- Bearer-token auth (constant-time), `/health`, serialized single-GPU inference.

Optional speaker diarization fills the per-segment `speaker` field (opt-in; see
[Diarization](#diarization-optional)). Without it, `speaker` is `null`.

## Requirements

- Python 3.12
- `ffmpeg` and `ffprobe` on `PATH`
- For GPU: an NVIDIA card + CUDA runtime compatible with the pinned
  `onnxruntime-gpu` (see [Pascal / ONNX Runtime](#pascal--onnx-runtime-pin)).

## Install

parascribe uses a plain venv + pip (not uv). Pick the requirements file for your
target:

```bash
python3.12 -m venv .venv

# Deployment (NVIDIA GPU):
.venv/bin/pip install -r requirements-gpu.txt

# Development / CPU-only (e.g. Apple Silicon):
.venv/bin/pip install -r requirements-dev.txt   # includes CPU onnxruntime + test tooling
.venv/bin/pip install -e .
```

Confirm GPU engagement on the deployment host in one command:

```bash
PARASCRIBE_EXECUTION_PROVIDER=cuda .venv/bin/python scripts/check_gpu.py
```

It prints the active execution provider and exits non-zero if CUDA was requested
but is not actually engaged.

## Run

```bash
PARASCRIBE_EXECUTION_PROVIDER=cuda \
PARASCRIBE_API_KEY=your-secret \
.venv/bin/python -m uvicorn parascribe.main:app --host 127.0.0.1 --port 8000
```

On a CPU-only dev box, set `PARASCRIBE_EXECUTION_PROVIDER=cpu`.

Run it behind your network boundary (Tailscale / internal) like the rest of the
fleet; bind to localhost and reverse-proxy or front with LiteLLM.

## Configuration

All settings are environment variables prefixed `PARASCRIBE_` (see `.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `istupakov/parakeet-tdt-0.6b-v3-onnx` | onnx-asr model id (HF repo or builtin alias). |
| `EXECUTION_PROVIDER` | `cuda` | `cuda` \| `cpu` \| `coreml`. `cuda` fails loudly if not engaged. |
| `GPU_DEVICE_ID` | `0` | CUDA device index. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. |
| `API_KEY` / `API_KEY_FILE` | (none) | Bearer token (inline or file). If unset, auth is disabled with a loud warning. |
| `WORK_DIR` | `/run/parascribe` | tmpfs dir for temp uploads (deleted after each request). |
| `MAX_CHUNK_S` | `24` | Max VAD speech-segment length (onnx-asr `max_speech_duration_s`). |
| `CHUNK_OVERLAP_S` | `0` | VAD pad (onnx-asr `speech_pad_ms`). 0 = cut cleanly at silence. |
| `VAD_THRESHOLD` | `0.5` | Silero VAD threshold. |
| `MAX_UPLOAD_MB` | `2048` | Max upload size; larger returns 413. |
| `MAX_QUEUE` | `16` | Max admitted requests (1 in-flight + queued); beyond this returns 503. |
| `ENABLE_VIDEO` | `false` | Accept video input (extract audio track). |
| `DEFAULT_LANGUAGE` | (none) | ISO language hint. Accepted but IGNORED by Parakeet TDT (auto-detects); only Whisper/Canary use it. |
| `ENABLE_DIARIZATION` | `false` | Load the diarizer at startup (needs `requirements-diarization.txt`). See [Diarization](#diarization-optional). |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | pyannote pipeline (gated model). |
| `DIARIZATION_DEVICE` | (follow ASR) | `cuda`/`cpu`. Use `cpu` to avoid VRAM contention with ASR on small cards. |
| `HF_TOKEN` / `HF_TOKEN_FILE` | (none) | HuggingFace token for the one-time gated diarization-model download. |
| `LOG_LEVEL` | `INFO` | Operational verbosity (content-free): `DEBUG`/`INFO`/`WARNING`/`ERROR`. |
| `DEBUG_LOGGING` | `false` | Forces `DEBUG` and logs transcript content. WARNING: exposes content. |

## Logging

parascribe owns the `parascribe` logger (its own handler, does not depend on
uvicorn's logging config), so its lines always appear. Each request gets a short
`rid` that threads through every line for that request:

```
2026-06-24 11:40:01 INFO    parascribe.main rid=a1b2c3d4 | recv: format=verbose_json stream=False lang=- words=True
2026-06-24 11:40:03 INFO    parascribe.main rid=a1b2c3d4 | done: dur=11.0s decode=48ms infer=1820ms segments=4 words=22 format=verbose_json
```

- `LOG_LEVEL=INFO` (default) logs request receipt, a per-request timing summary
  (decode ms, inference ms, segment/word counts), 503s, and decode failures.
  **No transcript text, segment text, or filenames** appear at INFO.
- `LOG_LEVEL=DEBUG` adds decode timing detail and other diagnostics, still
  content-free.
- `DEBUG_LOGGING=true` forces `DEBUG` **and** additionally logs the transcript
  text (gated, with a startup warning) for deep debugging. Keep it off in
  production (invariant: content-free logs).

Under systemd these go to the journal (`journalctl -u parascribe -f`).

## API

### `POST /v1/audio/transcriptions` (multipart/form-data)

Standard OpenAI params: `file` (required), `model` (required, accepted for
compatibility), `response_format`, `timestamp_granularities[]`, `language`,
`stream`, `temperature`, `prompt`. `temperature`/`prompt` are accepted and
ignored (logged) since the backend does not use them. `language` is likewise
accepted but ignored by Parakeet TDT (it auto-detects); it would only take effect
with a Whisper/Canary backend.

`verbose_json` always includes `segments` with real `start`/`end`. `words` are
included when `timestamp_granularities[]` contains `word`. Whisper-only fields
(`seek`, `tokens`, `compression_ratio`, `no_speech_prob`) are omitted;
`avg_logprob` is provided per segment. `speaker` is `null` unless diarization is
requested (see [Diarization](#diarization-optional)).

```bash
# json (text only)
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@meeting.mp3 -F model=parascribe

# verbose_json with word timestamps
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@meeting.mp3 -F model=parascribe \
  -F response_format=verbose_json -F 'timestamp_granularities[]=word'

# streaming (SSE) for long files
curl -N http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@two-hour-recording.wav -F model=parascribe \
  -F response_format=verbose_json -F stream=true
```

Streaming emits OpenAI `transcript.text.delta` events as each segment finalizes,
then a terminal `transcript.text.done`. For `verbose_json` the done event also
carries the assembled `segments` (and `words` if requested) -- a parascribe
extension beyond the OpenAI streaming spec. Streaming is progressive *output*,
not realtime input (the file is decoded and VAD-segmented first). `stream=true`
with `srt`/`vtt`/`text` is ignored (logged) and returns the normal response.

### Diarization (optional)

Speaker diarization ("who said what") is opt-in per request and disabled by
default. It runs pyannote.audio, aligns the speaker turns onto the ASR word
timestamps, and fills the `speaker` field (otherwise `null`).

Setup:

1. `pip install -r requirements-diarization.txt` (heavy — pulls PyTorch).
2. Accept the licenses for `pyannote/speaker-diarization-3.1` and
   `pyannote/segmentation-3.0` on HuggingFace, then set `PARASCRIBE_HF_TOKEN`
   (or `HF_TOKEN_FILE`) for the first download. It runs offline from cache after.
3. Start with `PARASCRIBE_ENABLE_DIARIZATION=true`. On a small card shared with
   ASR, set `PARASCRIBE_DIARIZATION_DEVICE=cpu` to avoid VRAM contention.

Request it per call:

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret" \
  -F file=@meeting.wav -F model=parascribe \
  -F response_format=verbose_json -F diarization=true \
  -F 'timestamp_granularities[]=word'        # adds per-word speaker too
```

- `diarization=true` labels each segment's `speaker` (`SPEAKER_00`, ...); opaque
  labels only — no speaker naming/identification.
- `num_speakers=N` optionally fixes the speaker count; omitted = automatic.
- Speaker labels surface in `verbose_json`. Diarization is **incompatible with
  `stream=true`** (it needs the whole file) — streaming is ignored and the
  request returns non-streamed.
- If `diarization=true` but the server wasn't started with it enabled, the
  request returns **400** (never a silent no-speaker result).

### `GET /health`

```json
{"status": "ok", "model_id": "...", "device": "cuda:0", "provider_active": true}
```

## Pascal / ONNX Runtime pin

Target hardware includes a **GTX 1080 Ti (Pascal, sm_61)**. Some prebuilt
`onnxruntime-gpu` wheels drop old compute capabilities and would silently fall
back to CPU -- which parascribe refuses to do. `requirements-gpu.txt` pins
**`onnxruntime-gpu==1.24.4`**, the version verified running on the target card
(via the existing Speaches deployment); it is a CUDA-12 build and CUDA 12 still
includes sm_61. Do not float this dependency. The startup GPU check (and
`scripts/check_gpu.py`) is what catches a bad wheel before it reaches
production.

On hardened kernels that reject executable stacks, NVIDIA/onnxruntime `.so`
files may need `patchelf --clear-execstack <file>` (the same fix the Speaches
deployment applies to ctranslate2). If the model fails to load with an execstack
error, run `patchelf --clear-execstack` over the offending library in
`.venv/lib/python3.12/site-packages/onnxruntime*`.

## Development

```bash
.venv/bin/python -m pytest        # full suite
.venv/bin/ruff check src tests
.venv/bin/mypy
```

The offset/word-grouping logic (`stitch.py`) is the correctness-critical core and
is unit-tested in isolation (`tests/test_stitch.py`). GPU integration tests are
marked `gpu` and skipped where no CUDA device is present.

## Deployment

See `deploy/parascribe.service` for a hardened systemd unit (dedicated user,
tmpfs `RuntimeDirectory`, GPU device allow-list, writable HF cache).
