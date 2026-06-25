"""Speaker diarization via pyannote.audio.

pyannote + torch are heavy and optional (requirements-diarization.txt), so they
are imported lazily inside the Diarizer rather than at module import: this module
stays importable without them, and the API layer constructs a Diarizer only when
``enable_diarization`` is set, raising DiarizationUnavailableError when the deps or
gated models are missing (never a silent no-speaker fallback).

The model produces opaque speaker labels (SPEAKER_00, ...) and runs over the whole
file (clustering is global), so it is not streamable. The gated pyannote models
require a HuggingFace token and license acceptance for the first download.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from parascribe.align import SpeakerTurn
from parascribe.asr import SAMPLE_RATE

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from parascribe.config import Settings

logger = logging.getLogger(__name__)


class DiarizationUnavailableError(RuntimeError):
    """pyannote deps or models are not available (maps to a clear API error)."""


# Files at/above this duration log progress every 10%; shorter ones every 20%.
_PROGRESS_STEP_THRESHOLD_S = 600.0


class _ProgressLogger:
    """A pyannote hook that logs diarization progress.

    Fire-and-forget: announces each stage (segmentation/embeddings/clustering),
    and additionally logs % milestones for the slow **embeddings** stage so a long
    CPU run shows real progress there. Segmentation is fast and clustering is
    effectively one-shot, so those get only a start line. Nothing is returned.
    """

    def __init__(self, rid: str, step_pct: int) -> None:
        self._rid = rid
        self._step_pct = step_pct
        self._last: dict[str, int] = {}

    def __call__(
        self,
        step_name: str,
        step_artifact: object,
        file: object = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        if step_name not in self._last:
            self._last[step_name] = 0
            logger.info("diarization: %s started", step_name, extra={"rid": self._rid})
        # Percentages only for the slow embeddings stage; others are fast/one-shot.
        if "embed" not in step_name.lower() or not total or completed is None:
            return
        pct = int(completed / total * 100)
        bucket = pct - (pct % self._step_pct)
        if bucket >= self._step_pct and bucket > self._last[step_name]:
            self._last[step_name] = bucket
            logger.info("diarization: %s %d%%", step_name, bucket, extra={"rid": self._rid})


class Diarizer:
    """Loads the pyannote pipeline once and produces speaker turns for audio."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        device = settings.resolved_diarization_device
        try:
            import torch
            # We always pass in-memory waveforms, so pyannote never uses torchcodec
            # for file decoding; silence its noisy CUDA-lib load failure at import.
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io"
            )
            # Benign: pyannote's stats pooling computes std over 1-frame windows.
            warnings.filterwarnings(
                "ignore", message=r"std\(\): degrees of freedom is <= 0"
            )
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationUnavailableError(
                f"diarization deps failed to import: {exc}. Install "
                "requirements-diarization.txt with a CPU torch build "
                "(pip install torch torchaudio --index-url "
                "https://download.pytorch.org/whl/cpu)."
            ) from exc

        logger.info("loading diarization model %s on %s", settings.diarization_model, device)
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model, token=settings.resolved_hf_token()
        )
        if pipeline is None:
            # pyannote returns None when the model is gated and access/token is missing.
            raise DiarizationUnavailableError(
                f"could not load {settings.diarization_model}: accept its license on "
                "HuggingFace and set hf_token/hf_token_file (or pre-cache the model)"
            )
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        self._torch = torch
        logger.info("diarization model loaded")

    def diarize(
        self,
        audio: npt.NDArray[np.float32],
        *,
        num_speakers: int | None = None,
        rid: str = "-",
    ) -> list[SpeakerTurn]:
        """Return speaker turns for a 16 kHz mono float32 array (absolute times)."""
        # .copy(): the decoded PCM is a read-only np.frombuffer view; torch needs writable.
        waveform = self._torch.from_numpy(audio.copy()).unsqueeze(0)  # (1, num_samples)
        duration = audio.shape[0] / SAMPLE_RATE
        step_pct = 10 if duration >= _PROGRESS_STEP_THRESHOLD_S else 20
        params: dict[str, object] = {"hook": _ProgressLogger(rid, step_pct)}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        result = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE}, **params
        )
        # The pipeline result wraps the Annotation in `.speaker_diarization`;
        # fall back to the result itself if it is already an Annotation.
        annotation = getattr(result, "speaker_diarization", result)
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
