"""Tests for response formatting (json/text/verbose_json/srt/vtt)."""

from __future__ import annotations

import json

import pytest

from parascribe.formats import render, to_srt, to_vtt, verbose_json_body
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
    )


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


def test_unsupported_format_raises():
    t = Transcript(text="", language=None, duration=0.0)
    with pytest.raises(ValueError, match="unsupported response_format"):
        render(t, "flac", include_words=False)
