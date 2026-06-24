"""End-to-end integration against the real model (SPEC §11).

Opt-in and slow (downloads ~2GB on first run). Skipped unless
PARASCRIBE_RUN_MODEL_TESTS=1 is set. Runs on CPU for dev; mark `gpu` so a CUDA
runner picks it up. Synthesizes speech with `say` (macOS) or `espeak`/`espeak-ng`
(Linux); skips if no TTS tool is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.getenv("PARASCRIBE_RUN_MODEL_TESTS"),
        reason="set PARASCRIBE_RUN_MODEL_TESTS=1 to run the real-model integration tests",
    ),
]


def _synthesize(text: str, dest: Path) -> None:
    if shutil.which("say"):  # macOS
        aiff = dest.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1",
             str(dest), "-y"], check=True,
        )
        return
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak:
        subprocess.run(
            [espeak, "-w", str(dest), text], check=True
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(dest), "-ar", "16000", "-ac", "1",
             str(dest.with_name("norm.wav")), "-y"], check=True,
        )
        shutil.move(str(dest.with_name("norm.wav")), str(dest))
        return
    pytest.skip("no TTS tool (say/espeak) available to synthesize a fixture")


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory) -> tuple[Path, float]:
    """Three utterances separated by 1.5s silence -> multiple VAD segments."""
    d = tmp_path_factory.mktemp("clip")
    parts = []
    for i, text in enumerate(
        ["The first segment starts at the beginning.",
         "Here is the second segment after a pause.",
         "And the third segment is near the end."]
    ):
        p = d / f"p{i}.wav"
        _synthesize(text, p)
        parts.append(p)
    sil = d / "sil.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-t", "1.5",
         "-i", "anullsrc=r=16000:cl=mono", str(sil), "-y"], check=True,
    )
    listing = d / "list.txt"
    listing.write_text("".join(
        f"file '{parts[0]}'\nfile '{sil}'\nfile '{parts[1]}'\nfile '{sil}'\nfile '{parts[2]}'\n"
    ))
    out = d / "multi.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-ar", "16000", "-ac", "1", str(out), "-y"], check=True,
    )
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout.strip())
    return out, dur


@pytest.fixture(scope="module")
def transcriber():
    from parascribe.asr import Transcriber
    from parascribe.config import Settings

    provider = os.getenv("PARASCRIBE_EXECUTION_PROVIDER", "cpu")
    return Transcriber(Settings(execution_provider=provider))


class TestRealModel:
    def test_segments_present_and_start_near_zero(self, transcriber, long_clip):
        from parascribe.stitch import assemble

        clip, dur = long_clip
        result = assemble(
            transcriber.transcribe(str(clip), language="en"), language="en", duration=dur
        )
        assert len(result.segments) >= 1
        assert result.segments[0].start < 0.5

    def test_no_gaps_or_overlaps_at_seams_and_end_near_duration(self, transcriber, long_clip):
        from parascribe.stitch import assemble

        clip, dur = long_clip
        result = assemble(
            transcriber.transcribe(str(clip), language="en"), language="en", duration=dur
        )
        starts_ends = [(s.start, s.end) for s in result.segments]
        # Monotonic, non-overlapping segments.
        for (s0, e0), (s1, _e1) in zip(starts_ends, starts_ends[1:], strict=False):
            assert e0 <= s1
            assert s0 < e0
        assert result.segments[-1].end <= dur + 0.1

    def test_word_times_monotonic_and_global(self, transcriber, long_clip):
        from parascribe.stitch import assemble

        clip, dur = long_clip
        result = assemble(
            transcriber.transcribe(str(clip), language="en"), language="en", duration=dur
        )
        flat = [v for w in result.words for v in (w.start, w.end)]
        assert flat == sorted(flat)
        assert result.words[-1].end <= dur + 0.1
