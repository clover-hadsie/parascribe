"""Build the OpenAI 'tokens' usage object that LiteLLM reads for spend tracking.

Parakeet is not token-billed, so the audio-input/transcription/diarization counts
are config-driven (see Settings usage fields) rather than a fixed rate. Pure module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parascribe.config import Settings
    from parascribe.stitch import Transcript

_UNIT_COUNTS: dict[str, Callable[[Transcript], float]] = {
    "token": lambda t: t.token_count,
    "word": lambda t: len(t.words),
    "segment": lambda t: len(t.segments),
    "char": lambda t: len(t.text),
    "file_duration": lambda t: t.duration,
}


def _component(transcript: Transcript, unit: str, multiplier: float) -> int:
    return round(_UNIT_COUNTS[unit](transcript) * multiplier)


def build_usage(
    transcript: Transcript, settings: Settings, *, diarized: bool
) -> dict[str, object]:
    """Build the OpenAI 'tokens' usage object for ``transcript`` under ``settings``."""
    input_tokens = 0
    output_tokens = 0

    audio = _component(
        transcript,
        settings.audio_input_usage_unit,
        settings.audio_input_usage_multiplier,
    )
    audio_to_input = settings.audio_input_usage_field == "input_tokens"
    if audio_to_input:
        input_tokens += audio
    else:
        output_tokens += audio

    output_tokens += _component(
        transcript,
        settings.transcription_usage_unit,
        settings.transcription_usage_multiplier,
    )
    if diarized:
        output_tokens += _component(
            transcript,
            settings.diarization_usage_unit,
            settings.diarization_usage_multiplier,
        )

    return {
        "type": "tokens",
        "input_tokens": input_tokens,
        "input_token_details": {
            "text_tokens": 0,
            "audio_tokens": audio if audio_to_input else 0,
        },
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
