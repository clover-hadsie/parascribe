#!/usr/bin/env python3
"""Load the configured model and report the active ONNX Runtime provider.

Run on the deployment host to confirm GPU engagement in one command:

    .venv/bin/python scripts/check_gpu.py

Exits non-zero if execution_provider=cuda but CUDA is not actually engaged
(the same condition that makes the server refuse to start, invariant #2).
"""

from __future__ import annotations

import sys

import onnxruntime as ort

from parascribe.asr import GpuUnavailableError, Transcriber
from parascribe.config import Settings


def main() -> int:
    settings = Settings()
    print(f"configured execution_provider : {settings.execution_provider}")
    print(f"onnxruntime version           : {ort.__version__}")
    print(f"providers in this build       : {ort.get_available_providers()}")
    print(f"model_id                      : {settings.model_id}")
    print("loading model ...")
    try:
        transcriber = Transcriber(settings)
    except GpuUnavailableError as exc:
        print(f"\nGPU CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"active providers              : {sorted(transcriber.providers_active)}")
    print(f"device                        : {transcriber.device}")
    print(f"gpu_active                    : {transcriber.gpu_active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
