"""Runtime configuration via environment variables (PARASCRIBE_* prefix)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ExecutionProvider = Literal["cuda", "cpu", "coreml"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARASCRIBE_",
        env_file=".env",
        extra="ignore",
        # 'model_id' would otherwise collide with pydantic's protected 'model_' namespace.
        protected_namespaces=(),
    )

    # Model / inference
    model_id: str = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    execution_provider: ExecutionProvider = "cuda"
    gpu_device_id: int = 0

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Auth
    api_key: str | None = None
    api_key_file: Path | None = None

    # Chunking / VAD (mapped to onnx-asr VadOptions)
    max_chunk_s: float = 24.0
    chunk_overlap_s: float = 0.0
    vad_threshold: float = 0.5

    # Limits / IO
    work_dir: Path = Path("/run/parascribe")
    max_upload_mb: int = 2048
    enable_video: bool = False
    # Max requests admitted at once (1 in-flight + the rest queued, SPEC §5.3).
    # Beyond this the server returns 503.
    max_queue: int = 16

    # Language / logging
    default_language: str | None = None
    # Operational log verbosity (content-free). debug_logging overrides this to
    # DEBUG and additionally permits transcript content into the logs.
    log_level: str = "INFO"
    debug_logging: bool = False

    def resolved_api_key(self) -> str | None:
        """The configured bearer token, from api_key or api_key_file (or None)."""
        if self.api_key:
            return self.api_key
        if self.api_key_file is not None and self.api_key_file.exists():
            token = self.api_key_file.read_text(encoding="utf-8").strip()
            return token or None
        return None
