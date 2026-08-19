"""Tests for response formatting (json/text/verbose_json/srt/vtt)."""

from __future__ import annotations

import json

import pytest

from parascribe.formats import (
    asr_json_body,
    done_event,
    render,
    render_asr,
    to_srt,
    to_vtt,
    verbose_json_body,
)
from parascribe.stitch import Segment, Transcript, Word


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        text="Hello world.",
        language="en",
        duration=3.0,
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="Hello world.", speaker=None, avg_logprob=-0.1)
        ],
        words=[Word("Hello", 0.0, 0.5), Word("world.", 0.5, 1.5)],
        token_count=4,
    )


# A representative usage object; render/done_event treat it as an opaque dict, so
# its exact derivation (tested in test_usage.py) is irrelevant here.
_USAGE: dict[str, object] = {
    "type": "tokens",
    "input_tokens": 30,
    "output_tokens": 4,
    "total_tokens": 34,
}


class TestRenderJson:
    def test_json_returns_only_text(self, transcript):
        r = render(transcript, "json", include_words=False)
        assert json.loads(r.body) == {"text": "Hello world."}
        assert r.media_type == "application/json"

    def test_text_returns_plain_transcript(self, transcript):
        r = render(transcript, "text", include_words=False)
        assert r.body == "Hello world."
        assert r.media_type.startswith("text/plain")


class TestVerboseJson:
    def test_segments_present_even_without_word_granularity(self, transcript):
        # Invariant #1: verbose_json always carries segments with real times.
        body = verbose_json_body(transcript, include_words=False)
        assert body["segments"][0]["start"] == 0.0
        assert body["segments"][0]["end"] == 1.5
        assert "words" not in body

    def test_words_included_when_requested(self, transcript):
        body = verbose_json_body(transcript, include_words=True)
        assert [w["word"] for w in body["words"]] == ["Hello", "world."]

    def test_speaker_field_present_and_null(self, transcript):
        body = verbose_json_body(transcript, include_words=False)
        assert body["segments"][0]["speaker"] is None

    def test_top_level_shape(self, transcript):
        body = verbose_json_body(transcript, include_words=True)
        assert body["task"] == "transcribe"
        assert body["language"] == "en"
        assert body["duration"] == 3.0
        assert body["text"] == "Hello world."


class TestSubtitles:
    def test_srt_block_structure(self, transcript):
        out = to_srt(transcript)
        assert out.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello world.")

    def test_vtt_starts_with_header_and_dot_separator(self, transcript):
        out = to_vtt(transcript)
        assert out.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.500" in out


class TestUsageInRender:
    def test_json_carries_usage(self, transcript):
        r = render(transcript, "json", include_words=False, usage=_USAGE)
        assert json.loads(r.body)["usage"] == _USAGE

    def test_verbose_json_carries_usage(self, transcript):
        r = render(transcript, "verbose_json", include_words=False, usage=_USAGE)
        assert json.loads(r.body)["usage"] == _USAGE

    def test_json_omits_usage_when_none(self, transcript):
        r = render(transcript, "json", include_words=False)
        assert "usage" not in json.loads(r.body)

    def test_text_format_ignores_usage(self, transcript):
        # Plain text has nowhere to carry usage; it must not corrupt the body.
        r = render(transcript, "text", include_words=False, usage=_USAGE)
        assert r.body == "Hello world."

    def test_done_event_carries_usage(self, transcript):
        event = done_event(
            transcript, response_format="verbose_json", include_words=False, usage=_USAGE,
        )
        assert event["usage"] == _USAGE


class TestAsrOutput:
    """Contract with ASR-webservice clients: they read top-level text/segments/
    language and per-segment speaker/text/start/end; extra keys are ignored."""

    def test_json_golden_shape(self, transcript):
        body = asr_json_body(transcript)
        # Top-level keys clients read; nothing else is promised on this surface.
        assert set(body) == {"text", "segments", "language"}
        assert body["text"] == "Hello world."
        assert body["language"] == "en"
        seg = body["segments"][0]
        # Per-segment keys clients read (extras like id/avg_logprob are ignored).
        assert {"speaker", "text", "start", "end"} <= set(seg)
        assert (seg["start"], seg["end"], seg["text"]) == (0.0, 1.5, "Hello world.")

    def test_json_speaker_null_without_diarization(self, transcript):
        assert asr_json_body(transcript)["segments"][0]["speaker"] is None

    def test_json_carries_speaker_labels(self, transcript):
        diarized = Transcript(
            text=transcript.text, language=transcript.language,
            duration=transcript.duration, words=transcript.words,
            token_count=transcript.token_count,
            segments=[
                Segment(id=0, start=0.0, end=1.5, text="Hello world.",
                        speaker="SPEAKER_00", avg_logprob=-0.1)
            ],
        )
        assert asr_json_body(diarized)["segments"][0]["speaker"] == "SPEAKER_00"

    def test_render_asr_json(self, transcript):
        r = render_asr(transcript, "json")
        assert r.media_type == "application/json"
        assert json.loads(r.body)["text"] == "Hello world."

    def test_render_asr_txt_matches_text_format(self, transcript):
        r = render_asr(transcript, "txt")
        assert r.body == render(transcript, "text", include_words=False).body

    def test_render_asr_srt_and_vtt_reuse_renderers(self, transcript):
        assert render_asr(transcript, "srt").body == to_srt(transcript)
        assert render_asr(transcript, "vtt").body == to_vtt(transcript)


def test_unsupported_format_raises():
    t = Transcript(text="", language=None, duration=0.0)
    with pytest.raises(ValueError, match="unsupported response_format"):
        render(t, "flac", include_words=False)
