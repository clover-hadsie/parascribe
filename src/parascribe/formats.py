"""Render a Transcript into the OpenAI transcription response formats.

Framework-free: each renderer returns a ``Rendered(body, media_type)`` so the API
layer can wrap it in a Response. JSON formats are serialized here so the body is
always a string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from parascribe.stitch import Segment, Transcript, Word

ALLOWED_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
# Formats whose output can be streamed incrementally; srt/vtt/text fall back to a
# single non-streamed response.
STREAMABLE_FORMATS = ("json", "verbose_json")


@dataclass(frozen=True)
class Rendered:
    body: str
    media_type: str


def _segment_dict(seg: Segment) -> dict[str, object]:
    return {
        "id": seg.id,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "speaker": seg.speaker,  # opaque label, or null when diarization didn't run
        "avg_logprob": seg.avg_logprob,
    }


def _word_dict(word: Word) -> dict[str, object]:
    body: dict[str, object] = {"word": word.word, "start": word.start, "end": word.end}
    if word.speaker is not None:  # only present when diarization ran
        body["speaker"] = word.speaker
    return body


def verbose_json_body(transcript: Transcript, *, include_words: bool) -> dict[str, object]:
    """The verbose_json object. Segments are ALWAYS present."""
    body: dict[str, object] = {
        "task": "transcribe",
        "language": transcript.language,
        "duration": transcript.duration,
        "text": transcript.text,
        "segments": [_segment_dict(s) for s in transcript.segments],
    }
    if include_words:
        body["words"] = [_word_dict(w) for w in transcript.words]
    return body


def _format_timestamp(seconds: float, *, millis_sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    lines: list[str] = []
    for index, seg in enumerate(transcript.segments, start=1):
        start = _format_timestamp(seg.start, millis_sep=",")
        end = _format_timestamp(seg.end, millis_sep=",")
        lines.append(f"{index}\n{start} --> {end}\n{seg.text}\n")
    return "\n".join(lines)


def to_vtt(transcript: Transcript) -> str:
    blocks = ["WEBVTT\n"]
    for seg in transcript.segments:
        start = _format_timestamp(seg.start, millis_sep=".")
        end = _format_timestamp(seg.end, millis_sep=".")
        blocks.append(f"{start} --> {end}\n{seg.text}\n")
    return "\n".join(blocks)


def sse_event(payload: dict[str, object]) -> str:
    """Serialize one Server-Sent Event line for streaming."""
    return f"data: {json.dumps(payload)}\n\n"


def delta_event(text: str) -> dict[str, object]:
    """An incremental transcript.text.delta event (OpenAI streaming shape)."""
    return {"type": "transcript.text.delta", "delta": text}


def done_event(
    transcript: Transcript,
    *,
    response_format: str,
    include_words: bool,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    """The terminal transcript.text.done event carrying the full result.

    For verbose_json the final event also carries the assembled segments (and
    words if requested) - a parascribe extension beyond the OpenAI streaming spec.
    """
    payload: dict[str, object] = {"type": "transcript.text.done", "text": transcript.text}
    if response_format == "verbose_json":
        payload["segments"] = [_segment_dict(s) for s in transcript.segments]
        if include_words:
            payload["words"] = [_word_dict(w) for w in transcript.words]
    if usage is not None:
        payload["usage"] = usage
    return payload


def render(
    transcript: Transcript,
    response_format: str,
    *,
    include_words: bool,
    usage: dict[str, object] | None = None,
) -> Rendered:
    """Render ``transcript`` for the given (already-validated) response_format.

    ``usage`` is attached to the JSON bodies only; text/srt/vtt have nowhere to
    carry it.
    """
    if response_format == "text":
        return Rendered(transcript.text, "text/plain; charset=utf-8")
    if response_format == "json":
        body: dict[str, object] = {"text": transcript.text}
        if usage is not None:
            body["usage"] = usage
        return Rendered(json.dumps(body), "application/json")
    if response_format == "verbose_json":
        body = verbose_json_body(transcript, include_words=include_words)
        if usage is not None:
            body["usage"] = usage
        return Rendered(json.dumps(body), "application/json")
    if response_format == "srt":
        return Rendered(to_srt(transcript), "application/x-subrip; charset=utf-8")
    if response_format == "vtt":
        return Rendered(to_vtt(transcript), "text/vtt; charset=utf-8")
    raise ValueError(f"unsupported response_format: {response_format!r}")
