"""Unit tests for model/VAD construction (CUDA library preload, VAD placement)."""

from __future__ import annotations

import logging

import pytest

from parascribe import asr
from parascribe.config import Settings


def settings(**overrides) -> Settings:
    base = {"execution_provider": "cuda", "model_id": "a"}
    return Settings(**{**base, **overrides})


@pytest.fixture
def calls(monkeypatch):
    """Record the order of preload_dlls and load_vad, stubbing both out."""
    order: list[str] = []
    monkeypatch.setattr(
        asr.ort, "preload_dlls", lambda *a, **k: order.append("preload"), raising=False
    )
    monkeypatch.setattr(
        asr.onnx_asr, "load_vad", lambda *a, **k: order.append("load_vad") or object()
    )
    monkeypatch.setattr(asr, "active_providers", lambda _: {"CUDAExecutionProvider"})
    return order


class TestBuildVad:
    def test_preloads_cuda_libs_before_creating_the_session(self, calls):
        asr.build_vad(settings())
        assert calls == ["preload", "load_vad"]

    def test_does_not_preload_when_not_on_cuda(self, calls):
        asr.build_vad(settings(execution_provider="cpu"))
        assert calls == ["load_vad"]

    def test_warns_when_cuda_requested_but_vad_lands_on_cpu(self, calls, monkeypatch, caplog):
        monkeypatch.setattr(asr, "active_providers", lambda _: {"CPUExecutionProvider"})
        with caplog.at_level(logging.WARNING, logger=asr.logger.name):
            asr.build_vad(settings())
        assert "falls back to CPU" in caplog.text

    def test_silent_when_vad_is_on_cuda(self, calls, caplog):
        with caplog.at_level(logging.WARNING, logger=asr.logger.name):
            asr.build_vad(settings())
        assert caplog.text == ""

    def test_no_cuda_warning_when_running_on_cpu_by_configuration(
        self, calls, monkeypatch, caplog
    ):
        monkeypatch.setattr(asr, "active_providers", lambda _: {"CPUExecutionProvider"})
        with caplog.at_level(logging.WARNING, logger=asr.logger.name):
            asr.build_vad(settings(execution_provider="cpu"))
        assert caplog.text == ""

    def test_no_warning_when_sessions_are_not_introspectable(self, calls, monkeypatch, caplog):
        monkeypatch.setattr(asr, "active_providers", lambda _: set())
        with caplog.at_level(logging.WARNING, logger=asr.logger.name):
            asr.build_vad(settings())
        assert caplog.text == ""
