"""Tests for ffmpeg decode (real ffmpeg, generated fixtures, no network)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from parascribe.asr import SAMPLE_RATE
from parascribe.media import DecodeError, decode_to_pcm, duration_seconds


@pytest.fixture
def wav_2s(tmp_path: Path) -> Path:
    """A 2-second 8 kHz stereo tone, to prove resample+downmix to 16k mono."""
    out = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2:sample_rate=8000",
            "-ac", "2", str(out), "-y",
        ],
        check=True,
    )
    return out


class TestDecodeToPcm:
    def test_resamples_and_downmixes_to_16k_mono(self, wav_2s: Path):
        audio = decode_to_pcm(wav_2s)
        assert audio.ndim == 1
        assert duration_seconds(audio) == pytest.approx(2.0, abs=0.05)

    def test_sample_count_matches_16k_rate(self, wav_2s: Path):
        audio = decode_to_pcm(wav_2s)
        assert audio.shape[0] == pytest.approx(2 * SAMPLE_RATE, abs=SAMPLE_RATE // 10)

    def test_non_media_file_raises_decode_error(self, tmp_path: Path):
        bogus = tmp_path / "not-audio.bin"
        bogus.write_bytes(b"this is definitely not media" * 10)
        with pytest.raises(DecodeError):
            decode_to_pcm(bogus)

    def test_missing_file_raises_decode_error(self, tmp_path: Path):
        with pytest.raises(DecodeError):
            decode_to_pcm(tmp_path / "does-not-exist.wav")
