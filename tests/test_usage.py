"""Tests for the configurable usage object (build_usage)."""

from __future__ import annotations

import pytest

from parascribe.config import Settings
from parascribe.stitch import Segment, Transcript, Word
from parascribe.usage import build_usage


@pytest.fixture
def transcript() -> Transcript:
    # 4 subword tokens, 2 words, 1 segment, 12 chars, 3.0s duration.
    return Transcript(
        text="Hello world.",
        language="en",
        duration=3.0,
        segments=[Segment(id=0, start=0.0, end=1.5, text="Hello world.", speaker=None)],
        words=[Word("Hello", 0.0, 0.5), Word("world.", 0.5, 1.5)],
        token_count=4,
    )


def settings(**overrides) -> Settings:
    return Settings(execution_provider="cpu", **overrides)


class TestDefaults:
    def test_audio_input_is_duration_times_ten(self, transcript):
        # Default audio input: file_duration * 10 -> input_tokens (OpenAI parity).
        u = build_usage(transcript, settings(), diarized=False)
        assert u["input_tokens"] == 30  # round(3.0 * 10)
        assert u["input_token_details"]["audio_tokens"] == 30

    def test_transcription_goes_to_output(self, transcript):
        u = build_usage(transcript, settings(), diarized=False)
        assert u["output_tokens"] == 4  # token_count * 1.0

    def test_total_is_input_plus_output(self, transcript):
        u = build_usage(transcript, settings(), diarized=False)
        assert u["total_tokens"] == 30 + 4

    def test_diarization_adds_to_output_only(self, transcript):
        u = build_usage(transcript, settings(), diarized=True)
        assert u["output_tokens"] == 4 + 4 * 5  # transcription + diarization (5x)
        assert u["input_tokens"] == 30  # audio input unaffected by diarization


class TestAudioInputConfig:
    def test_disabled_when_multiplier_zero(self, transcript):
        u = build_usage(transcript, settings(audio_input_usage_multiplier=0.0), diarized=False)
        assert u["input_tokens"] == 0
        assert u["input_token_details"]["audio_tokens"] == 0

    def test_rate_is_configurable(self, transcript):
        u = build_usage(transcript, settings(audio_input_usage_multiplier=25.0), diarized=False)
        assert u["input_tokens"] == 75  # round(3.0 * 25)

    def test_field_can_route_audio_to_output(self, transcript):
        # Fold audio into a single combined output count instead of input_tokens.
        u = build_usage(
            transcript,
            settings(audio_input_usage_field="output_tokens"),
            diarized=False,
        )
        assert u["input_tokens"] == 0
        assert u["input_token_details"]["audio_tokens"] == 0  # only set when in input
        assert u["output_tokens"] == 30 + 4  # audio + transcription

    def test_audio_unit_is_configurable(self, transcript):
        # Bill audio input by word count instead of duration.
        u = build_usage(
            transcript,
            settings(audio_input_usage_unit="word", audio_input_usage_multiplier=1.0),
            diarized=False,
        )
        assert u["input_tokens"] == 2


class TestUnitsAndMultipliers:
    def test_transcription_word_unit(self, transcript):
        u = build_usage(transcript, settings(transcription_usage_unit="word"), diarized=False)
        assert u["output_tokens"] == 2

    def test_transcription_char_unit(self, transcript):
        u = build_usage(transcript, settings(transcription_usage_unit="char"), diarized=False)
        assert u["output_tokens"] == len("Hello world.")

    def test_fractional_multiplier(self, transcript):
        u = build_usage(
            transcript, settings(transcription_usage_multiplier=0.5), diarized=False
        )
        assert u["output_tokens"] == 2  # round(4 * 0.5)

    def test_diarization_unit_independent_of_transcription(self, transcript):
        u = build_usage(
            transcript,
            settings(
                audio_input_usage_multiplier=0.0,  # isolate the two output components
                transcription_usage_unit="word",
                diarization_usage_unit="segment",
                diarization_usage_multiplier=3.0,
            ),
            diarized=True,
        )
        assert u["output_tokens"] == 2 + 1 * 3

    def test_type_is_tokens(self, transcript):
        assert build_usage(transcript, settings(), diarized=False)["type"] == "tokens"
